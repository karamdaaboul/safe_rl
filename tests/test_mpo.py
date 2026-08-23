"""Tests for MPO (Maximum a Posteriori Policy Optimization)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

NUM_OBS = 6
NUM_ACT = 2


def _make_policy(num_obs: int = NUM_OBS, num_actions: int = NUM_ACT):
    from safe_rl.modules import SACActorCritic

    return SACActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"hidden_dims": [32, 32]},
    )


def _make_mpo(**overrides):
    from safe_rl.algorithms import MPO

    kwargs = dict(
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=16,
        mstep_iteration_num=2,
        device="cpu",
    )
    kwargs.update(overrides)
    return MPO(_make_policy(), **kwargs)


def _fill_buffer(alg, n: int = 128) -> None:
    alg.init_storage(buffer_size=1000, num_envs=1, obs_shape=[NUM_OBS], act_shape=[NUM_ACT])
    for _ in range(n):
        obs = torch.randn(1, NUM_OBS)
        action = torch.rand(1, NUM_ACT) * 2 - 1
        reward = torch.randn(1)
        done = torch.zeros(1)
        next_obs = torch.randn(1, NUM_OBS)
        alg.store_transition(obs, action, reward, done, next_obs)


def test_mpo_builds_with_frozen_target_actor() -> None:
    alg = _make_mpo()
    # A dedicated frozen target actor must exist for the E-step / M-step KL.
    assert hasattr(alg, "actor_target")
    assert all(not p.requires_grad for p in alg.actor_target.parameters())
    # SAC entropy is disabled — exploration comes from the E-step KL trust region.
    assert not alg.auto_entropy_tuning


def test_mpo_eta_solver_returns_positive() -> None:
    import numpy as np

    alg = _make_mpo()
    rng = np.random.default_rng(0)
    q = rng.normal(size=(16, 32))
    eta = alg._solve_eta(q)
    assert eta > 0
    assert np.isfinite(eta)


def test_mpo_estep_weights_are_a_distribution() -> None:
    # The variational weights are a softmax over the N candidate actions per state,
    # so each state's column must sum to 1.
    alg = _make_mpo(sample_action_num=16)
    obs = torch.randn(32, NUM_OBS)
    with torch.no_grad():
        mean_old, log_std_old = alg.actor_target(obs)
        dist_old = torch.distributions.Normal(mean_old, log_std_old.exp())
        x = dist_old.sample((16,))
        actions = torch.tanh(x)
        cobs = obs.unsqueeze(0).expand(16, -1, -1).reshape(16 * 32, -1)
        q1, q2 = alg.policy.evaluate_q(cobs, actions.reshape(16 * 32, -1))
        q = torch.min(q1, q2).reshape(16, 32)
        eta = alg._solve_eta(q.numpy().astype("float64"))
        weights = torch.softmax(q / eta, dim=0)
    assert torch.allclose(weights.sum(dim=0), torch.ones(32), atol=1e-5)


def test_mpo_update_step_runs_and_is_finite() -> None:
    alg = _make_mpo(num_updates_per_step=2)
    _fill_buffer(alg, n=128)
    info = alg.update()
    for key in ("critic", "actor"):
        assert key in info
        assert torch.isfinite(torch.tensor(info[key]))
    penalty = alg.get_penalty_info()
    assert penalty["eta"] > 0
    for key in ("kl_mean", "kl_var", "alpha_mean", "alpha_var"):
        assert key in penalty
        assert torch.isfinite(torch.tensor(penalty[key]))


def test_mpo_update_moves_actor_params() -> None:
    alg = _make_mpo(num_updates_per_step=3)
    _fill_buffer(alg, n=128)
    before = [p.detach().clone() for p in alg.policy.actor.parameters()]
    alg.update()
    after = list(alg.policy.actor.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_mpo_dual_optimum_satisfies_kkt_on_eta() -> None:
    # dg/deta = eps - KL(q*||pi_old) exactly, so at the SLSQP optimum the actualized
    # non-parametric KL must equal the E-step trust region eps_dual.
    import numpy as np

    from safe_rl.algorithms.mpo import nonparametric_kl_from_weights

    alg = _make_mpo(dual_constraint=0.1)
    rng = np.random.default_rng(0)
    q_np = rng.normal(size=(64, 128))
    eta = alg._solve_eta(q_np)

    weights = torch.softmax(torch.from_numpy(q_np) / eta, dim=0)
    kl_q = nonparametric_kl_from_weights(weights).mean().item()
    assert abs(kl_q - alg.eps_dual) < 0.02, f"KKT residual too large: kl_q={kl_q}, eps={alg.eps_dual}"


def test_mpo_ess_bounds() -> None:
    from safe_rl.algorithms.mpo import effective_sample_size

    n, b = 16, 4
    # Uniform weights (the E-step did nothing) -> ESS is the full sample count.
    uniform = torch.full((n, b), 1.0 / n)
    assert torch.allclose(effective_sample_size(uniform), torch.full((b,), float(n)), atol=1e-4)

    # All mass on one action per state -> ESS collapses to 1.
    collapsed = torch.zeros(n, b)
    collapsed[0] = 1.0
    assert torch.allclose(effective_sample_size(collapsed), torch.ones(b), atol=1e-4)


def test_mpo_per_dim_constraining_matches_scalar_for_one_action_dim() -> None:
    # With a single action dimension, summing the per-dim KLs is a no-op, so the two
    # trust-region modes must produce identical updates.
    from copy import deepcopy

    from safe_rl.algorithms import MPO

    torch.manual_seed(0)
    policy = _make_policy(num_actions=1)
    common = dict(batch_size=8, num_updates_per_step=1, sample_action_num=16, mstep_iteration_num=3, device="cpu")
    alg_scalar = MPO(deepcopy(policy), per_dim_constraining=False, **common)
    alg_per_dim = MPO(deepcopy(policy), per_dim_constraining=True, **common)

    obs = torch.randn(8, NUM_OBS)
    torch.manual_seed(1)
    alg_scalar._update_actor_and_alpha(obs)
    torch.manual_seed(1)
    alg_per_dim._update_actor_and_alpha(obs)

    assert alg_scalar.alpha_mean.shape == (1,)
    assert alg_per_dim.alpha_mean.shape == (1,)
    for key in ("kl_mean", "kl_var", "alpha_mean", "alpha_var"):
        assert alg_scalar._last_actor_info[key] == pytest.approx(alg_per_dim._last_actor_info[key], rel=1e-5)
    for p, q in zip(alg_scalar.policy.actor.parameters(), alg_per_dim.policy.actor.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_mpo_per_dim_constraining_allocates_one_multiplier_per_action() -> None:
    alg = _make_mpo(per_dim_constraining=True)
    assert alg.alpha_mean.shape == (NUM_ACT,)
    assert alg.alpha_var.shape == (NUM_ACT,)
    _fill_buffer(alg, n=128)
    info = alg.update()
    assert torch.isfinite(torch.tensor(info["actor"]))


def test_mpo_decoupled_mstep_doubles_mle_at_the_identity_point() -> None:
    # Before the actor moves, online == target, so both halves of Acme's split cross-entropy
    # are the same distribution: the decoupled objective is exactly 2x the coupled one.
    # This is the same effective-scale doubling Acme gets from summing its two weight sets.
    from torch.distributions import Normal

    torch.manual_seed(0)
    mean = torch.randn(8, NUM_ACT)
    std = torch.rand(8, NUM_ACT) + 0.5
    x = Normal(mean, std).sample((16,))
    weights = torch.softmax(torch.randn(16, 8), dim=0)

    dist_mean = Normal(mean, std)  # Normal(mean, std_old) with std_old == std
    dist_var = Normal(mean, std)  # Normal(mean_old, std) with mean_old == mean
    coupled = (weights * Normal(mean, std).log_prob(x).sum(dim=-1)).sum(dim=0).mean()
    decoupled = (weights * dist_mean.log_prob(x).sum(dim=-1)).sum(dim=0).mean() + (
        weights * dist_var.log_prob(x).sum(dim=-1)
    ).sum(dim=0).mean()
    assert decoupled.item() == pytest.approx(2.0 * coupled.item(), rel=1e-5)


def test_mpo_estep_options_run() -> None:
    for kwargs in ({"estep_use_target_critic": True}, {"decoupled_mstep": True}):
        alg = _make_mpo(**kwargs)
        _fill_buffer(alg, n=128)
        info = alg.update()
        assert torch.isfinite(torch.tensor(info["actor"])), kwargs


def test_mpo_caps_default_high() -> None:
    # The old defaults (0.1 / 10) pinned the multipliers and un-enforced the M-step trust
    # region (measured on three envs). Guard the raised defaults against regression.
    alg = _make_mpo()
    assert alg.alpha_mean_max >= 10.0
    assert alg.alpha_var_max >= 1000.0


def test_mpo_hard_target_update_copies_on_period_only() -> None:
    alg = _make_mpo(target_actor_update="hard", target_actor_period=3)
    obs = torch.randn(32, NUM_OBS)
    target_before = [p.detach().clone() for p in alg.actor_target.parameters()]

    for step in range(1, 4):
        alg._update_actor_and_alpha(obs)
        changed = any(not torch.allclose(b, p) for b, p in zip(target_before, alg.actor_target.parameters()))
        if step < 3:
            assert not changed, f"target moved at update {step}, before the period"
        else:
            assert changed, "target was not copied at the period boundary"
    # At the copy point the target must equal the online actor exactly.
    for p, tp in zip(alg.policy.actor.parameters(), alg.actor_target.parameters()):
        assert torch.equal(p, tp)


def test_mpo_rejects_unknown_target_update() -> None:
    with pytest.raises(ValueError, match="target_actor_update"):
        _make_mpo(target_actor_update="soft")


def _make_distributional_policy(num_obs: int = NUM_OBS, num_actions: int = NUM_ACT):
    from safe_rl.modules import SACActorCritic

    return SACActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        critic_type="distributional",
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"num_atoms": 51, "v_min": -10.0, "v_max": 10.0, "network_kwargs": {"hidden_dims": [32, 32]}},
    )


def test_mpo_distributional_critic_with_nstep_runs() -> None:
    # DMPO-style stack: C51 critics + in-storage 3-step returns. The E-step consumes
    # scalar Q from the distributional heads; the critic update must dispatch to the
    # categorical projection with the per-sample gamma**n bootstrap discount.
    from safe_rl.algorithms import MPO

    alg = MPO(
        _make_distributional_policy(),
        batch_size=32,
        num_updates_per_step=2,
        sample_action_num=16,
        mstep_iteration_num=2,
        n_step=3,
        device="cpu",
    )
    assert alg.policy.is_distributional_critic
    assert alg.n_step == 3
    _fill_buffer(alg, n=256)
    info = alg.update()
    for key in ("critic", "actor"):
        assert torch.isfinite(torch.tensor(info[key])), key
    assert alg.get_penalty_info()["eta"] > 0


def test_distributional_project_uses_per_sample_nstep_discount() -> None:
    # With a deterministic next distribution at atom value z, reward 0 and no terminal,
    # the projected mean must be gamma**n * z per sample — not gamma * z for every sample.
    policy = _make_distributional_policy()
    critic = policy.critic_1_target
    n_atoms = critic.num_atoms
    batch = 2

    # All mass on the atom closest to z = 4.0.
    z_idx = int(torch.argmin((critic.q_support - 4.0).abs()))
    next_dist = torch.zeros(batch, n_atoms)
    next_dist[:, z_idx] = 1.0
    z = float(critic.q_support[z_idx])

    gamma = 0.9
    eff_n = torch.tensor([1.0, 3.0])
    proj = critic.project(
        next_dist=next_dist,
        rewards=torch.zeros(batch),
        bootstrap=torch.ones(batch),
        discount=gamma**eff_n,
    )
    means = (proj * critic.q_support).sum(dim=-1)
    assert means[0].item() == pytest.approx(gamma * z, abs=0.2)
    assert means[1].item() == pytest.approx(gamma**3 * z, abs=0.2)
    # The old scalar-gamma behaviour would give means[1] == gamma * z; make sure it doesn't.
    assert abs(means[1].item() - gamma * z) > 0.5


def test_mpo_reports_estep_diagnostics() -> None:
    alg = _make_mpo()
    _fill_buffer(alg, n=128)
    alg.update()
    info = alg.get_penalty_info()
    for key in (
        "kl_q",
        "kl_q_rel",
        "dual_residual_eta",
        "ess",
        "ess_min",
        "kl_mean_rel",
        "kl_var_rel",
        "pi_std_min",
        "pi_std_max",
        "pi_std_cond",
        "pretanh_mean_absmax",
        "frac_saturated",
        "solver_status",
        "solver_iters",
    ):
        assert key in info, f"missing diagnostic {key}"
        assert torch.isfinite(torch.tensor(info[key])), key
    # ESS lives in [1, N] by construction.
    assert 1.0 <= info["ess_min"] <= alg.sample_action_num + 1e-6
    assert info["solver_status"] == 0.0, "SLSQP did not converge"
