"""Tests for MPODIME (MPO with a DIME diffusion actor, path-space M-step)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

NUM_OBS = 5
NUM_ACT = 2
DIFF_STEPS = 4


def _diffusion_cfg(**overrides):
    cfg = {
        "diff_steps": DIFF_STEPS,
        "init_std": 2.5,
        "friction": 1.0,
        "per_dim_friction": True,
        "learn_friction": False,
        "learn_dt": False,
        "per_step_dt": False,
        "learn_prior": False,
        "learn_forward": True,
        "learn_backward": False,
        "score_model": {"num_layers": 2, "num_hid": 32, "layer_norm": True},
    }
    cfg.update(overrides)
    return cfg


def _make_policy(num_obs: int = NUM_OBS, num_actions: int = NUM_ACT, **policy_overrides):
    from safe_rl.modules import MPODIMEActorCritic

    kwargs = dict(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        critic_type="standard",
        critic_kwargs={"hidden_dims": [32, 32]},
        diffusion=_diffusion_cfg(),
    )
    kwargs.update(policy_overrides)
    return MPODIMEActorCritic(**kwargs)


def _make_alg(policy=None, **overrides):
    from safe_rl.algorithms import MPODIME

    kwargs = dict(
        batch_size=16,
        num_updates_per_step=1,
        sample_action_num=8,
        mstep_iteration_num=2,
        device="cpu",
    )
    kwargs.update(overrides)
    return MPODIME(policy if policy is not None else _make_policy(), **kwargs)


def _fill_buffer(alg, n: int = 128) -> None:
    alg.init_storage(buffer_size=1000, num_envs=1, obs_shape=[NUM_OBS], act_shape=[NUM_ACT])
    for _ in range(n):
        obs = torch.randn(1, NUM_OBS)
        action = torch.rand(1, NUM_ACT) * 2 - 1
        reward = torch.randn(1)
        done = torch.zeros(1)
        next_obs = torch.randn(1, NUM_OBS)
        alg.store_transition(obs, action, reward, done, next_obs)


# ----------------------------------------------------------------------
# Construction / validation
# ----------------------------------------------------------------------


def test_builds_with_frozen_target_actor() -> None:
    alg = _make_alg()
    assert hasattr(alg, "actor_target")
    assert hasattr(alg.actor_target, "diffusion_model")
    assert all(not p.requires_grad for p in alg.actor_target.parameters())
    # SAC entropy is disabled — exploration comes from the E-step trust region.
    assert not alg.auto_entropy_tuning


def test_rejects_gaussian_policy() -> None:
    from safe_rl.algorithms import MPODIME
    from safe_rl.modules import SACActorCritic

    gaussian = SACActorCritic(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACT,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"hidden_dims": [32, 32]},
    )
    with pytest.raises(TypeError, match="diffusion"):
        MPODIME(gaussian, device="cpu")


def test_rejects_learn_prior() -> None:
    policy = _make_policy(diffusion=_diffusion_cfg(learn_prior=True))
    with pytest.raises(ValueError, match="learn_prior"):
        _make_alg(policy=policy)


def test_rejects_unknown_kl_form() -> None:
    with pytest.raises(ValueError, match="kl_form"):
        _make_alg(kl_form="marginal")


# ----------------------------------------------------------------------
# path_mle helpers
# ----------------------------------------------------------------------


def test_sde_rollout_with_traj_shapes_and_chain_consistency() -> None:
    from safe_rl.networks.dime.path_mle import sde_rollout_with_traj

    torch.manual_seed(0)
    policy = _make_policy()
    dm = policy.actor.diffusion_model
    m = 512
    obs = torch.randn(m, NUM_OBS)
    with torch.no_grad():
        traj, means, scales = sde_rollout_with_traj(dm, obs)

    assert traj.shape == (DIFF_STEPS + 1, m, NUM_ACT)
    assert means.shape == (DIFF_STEPS, m, NUM_ACT)
    assert scales.shape == (DIFF_STEPS, NUM_ACT)
    assert torch.isfinite(traj).all() and torch.isfinite(means).all()
    assert (scales > 0).all()

    # x_{k+1} = mean_k + scale_k * eps with eps ~ N(0, I): the standardized
    # residuals must be unit-scale white noise.
    residuals = (traj[1:] - means) / scales.unsqueeze(1)
    assert residuals.mean().abs().item() < 0.05
    assert 0.9 < residuals.std().item() < 1.1


def test_path_logprob_matches_manual_kernel_sum_and_self_kl_is_zero() -> None:
    # Evaluating the SAME model that generated the trajectories: the recomputed
    # per-step means must reproduce the rollout's, so the path log-prob equals the
    # sum of transition kernels at the stored parameters and both KL forms are 0.
    from safe_rl.networks.dime.path_mle import path_logprob_and_kl, sde_rollout_with_traj
    from safe_rl.networks.dime.utils import log_prob_kernel

    torch.manual_seed(0)
    policy = _make_policy()
    dm = policy.actor.diffusion_model
    m = 32
    obs = torch.randn(m, NUM_OBS)
    with torch.no_grad():
        traj, means, scales = sde_rollout_with_traj(dm, obs)
        manual = torch.zeros(m)
        for k in range(DIFF_STEPS):
            manual += log_prob_kernel(traj[k + 1], means[k], scales[k])
        for kl_form in ("simplified", "full"):
            log_prob, kl = path_logprob_and_kl(dm, obs, traj, means, scales, kl_form=kl_form)
            assert torch.allclose(log_prob, manual, atol=1e-5)
            assert torch.allclose(kl, torch.zeros(m), atol=1e-9), kl_form


def test_fresh_policy_path_kl_is_zero_vs_target() -> None:
    # actor_target is a deepcopy at init, so the KL against it must be exactly 0
    # before the first M-step gradient — the trust-region multiplier starts inactive.
    from safe_rl.networks.dime.path_mle import path_logprob_and_kl, sde_rollout_with_traj

    torch.manual_seed(0)
    alg = _make_alg()
    m = 16
    obs = torch.randn(m, NUM_OBS)
    with torch.no_grad():
        traj, means, scales = sde_rollout_with_traj(alg.actor_target.diffusion_model, obs)
        _, kl = path_logprob_and_kl(alg.policy.actor.diffusion_model, obs, traj, means, scales)
    assert torch.allclose(kl, torch.zeros(m), atol=1e-9)


# ----------------------------------------------------------------------
# Update step
# ----------------------------------------------------------------------


def test_update_runs_and_is_finite_with_diagnostics() -> None:
    alg = _make_alg(num_updates_per_step=2)
    _fill_buffer(alg, n=128)
    info = alg.update()
    for key in ("critic", "actor"):
        assert key in info
        assert torch.isfinite(torch.tensor(info[key])), key

    penalty = alg.get_penalty_info()
    assert penalty["eta"] > 0
    for key in (
        "kl_q",
        "kl_q_rel",
        "dual_residual_eta",
        "ess",
        "ess_min",
        "kl_path",
        "kl_path_rel",
        "alpha_path",
        "path_mle",
        "pretanh_absmax",
        "frac_saturated",
        "dime_friction",
        "dime_dt",
        "dime_noise_scale",
        "solver_status",
        "solver_iters",
    ):
        assert key in penalty, f"missing diagnostic {key}"
        assert torch.isfinite(torch.tensor(penalty[key])), key
    # ESS lives in [1, N] by construction.
    assert 1.0 <= penalty["ess_min"] <= alg.sample_action_num + 1e-6
    assert penalty["solver_status"] == 0.0, "SLSQP did not converge"


def test_update_moves_actor_params() -> None:
    alg = _make_alg(num_updates_per_step=2)
    _fill_buffer(alg, n=128)
    before = [p.detach().clone() for p in alg.policy.actor.diffusion_model.fwd_model.parameters()]
    alg.update()
    after = list(alg.policy.actor.diffusion_model.fwd_model.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_alpha_path_dual_ascent_and_cap() -> None:
    # beta = 0 with a huge ascent step: the first M-step iteration sees KL exactly 0
    # (actor == target), every later one sees KL > 0, so the multiplier must leave 0
    # and be clipped at its cap.
    alg = _make_alg(kl_path_constraint=0.0, alpha_path_scale=1e8, alpha_path_max=5.0, mstep_iteration_num=3)
    assert alg.alpha_path == 0.0
    obs = torch.randn(16, NUM_OBS)
    alg._update_actor_and_alpha(obs)
    assert alg.alpha_path == pytest.approx(5.0)


def test_learn_friction_with_simplified_kl_warns(capsys) -> None:
    policy = _make_policy(diffusion=_diffusion_cfg(learn_friction=True))
    _make_alg(policy=policy, kl_form="simplified")
    assert "noise-collapse" in capsys.readouterr().out


# ----------------------------------------------------------------------
# Module surface (SAC/runner contracts)
# ----------------------------------------------------------------------


def test_sample_with_log_prob_is_two_tuple() -> None:
    policy = _make_policy()
    obs = torch.randn(7, NUM_OBS)
    action, log_prob = policy.sample_with_log_prob(obs)
    assert action.shape == (7, NUM_ACT)
    assert log_prob.shape == (7, 1)
    assert (action.abs() <= policy.action_scale).all()
    assert torch.isfinite(log_prob).all()


def test_act_paths_and_random_action_bounds() -> None:
    policy = _make_policy()
    obs = torch.randn(4, NUM_OBS)
    assert policy.act_with_noise(obs).shape == (4, NUM_ACT)
    assert policy.act_inference(obs).shape == (4, NUM_ACT)
    rand = policy.sample_random_action(9)
    assert rand.shape == (9, NUM_ACT)
    assert (rand.abs() <= policy.action_scale).all()
    # action_std is the per-step SDE noise scale — positive, one entry per action dim.
    assert policy.action_std.shape == (NUM_ACT,)
    assert (policy.action_std > 0).all()


def test_as_onnx_raises() -> None:
    with pytest.raises(NotImplementedError, match="ONNX"):
        _make_policy().as_onnx()


def test_nstep_distributional_critic_runs() -> None:
    from safe_rl.algorithms import MPODIME

    policy = _make_policy(
        critic_type="distributional",
        critic_kwargs={
            "num_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "network_kwargs": {"hidden_dims": [32, 32]},
        },
    )
    alg = MPODIME(
        policy,
        batch_size=16,
        num_updates_per_step=2,
        sample_action_num=8,
        mstep_iteration_num=2,
        n_step=3,
        device="cpu",
    )
    assert alg.policy.is_distributional_critic
    _fill_buffer(alg, n=256)
    info = alg.update()
    for key in ("critic", "actor"):
        assert torch.isfinite(torch.tensor(info[key])), key
    assert alg.get_penalty_info()["eta"] > 0
