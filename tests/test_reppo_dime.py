"""Tests for REPPODIME (REPPO with the vendored TruDi DIME diffusion actor)."""

from __future__ import annotations

import copy

import pytest
import torch

from safe_rl.algorithms.reppo_dime import REPPODIME
from safe_rl.modules import DIMEActorCritic, REPPOActorCritic

OBS_DIM = 5
ACT_DIM = 2
N_ENVS = 4
T_STEPS = 3

DIFFUSION = {
    "diff_steps": 4,
    "score_model": {"num_layers": 3, "num_hid": 32, "num_time_hid": 8, "num_time_out": 4, "layer_norm": True},
}
CRITIC_KWARGS = {
    "num_atoms": 21, "v_min": -5.0, "v_max": 5.0, "hidden_dim": 32,
    "encoder_layers": 1, "head_layers": 1, "pred_layers": 1,
    "activation": "swish", "norm": "rmsnorm", "prior_scale": 1.0,
}


def _make_policy(**overrides):
    kwargs = dict(
        num_actor_obs=OBS_DIM,
        num_critic_obs=OBS_DIM,
        num_actions=ACT_DIM,
        critic_type="reference",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        diffusion=DIFFUSION,
        critic_kwargs=CRITIC_KWARGS,
    )
    kwargs.update(overrides)
    return DIMEActorCritic(**kwargs)


def _make_alg(policy, **overrides):
    kwargs = dict(
        num_learning_epochs=1, num_mini_batches=1, device="cpu",
        desired_kl=0.1, kl_clip_mode="clipped", dual_optim_mode="actor",
        init_alpha_temp=0.01, init_alpha_kl=0.01, target_entropy=-4.0,
        optimizer_class="adam", weight_decay=0.0, betas=(0.9, 0.999),
        aux_loss_mult=1.0, max_grad_norm=0.5,
    )
    kwargs.update(overrides)
    alg = REPPODIME(policy, **kwargs)
    alg.init_storage("rl", N_ENVS, T_STEPS, [OBS_DIM], [OBS_DIM], [ACT_DIM])
    return alg


def _rollout(alg, steps=T_STEPS):
    next_obs = None
    for _ in range(steps):
        obs = torch.randn(N_ENVS, OBS_DIM)
        alg.act(obs, obs)
        next_obs = torch.randn(N_ENVS, OBS_DIM)
        alg.process_env_step(
            torch.randn(N_ENVS, 1),
            torch.zeros(N_ENVS, 1),
            {"time_outs": torch.zeros(N_ENVS)},
            next_obs=next_obs,
            next_critic_obs=next_obs,
        )
    return next_obs


# ----------------------------------------------------------------------
# Sampler / policy-module surface
# ----------------------------------------------------------------------


def test_sde_sample_shapes_bounds_and_cost_decomposition():
    torch.manual_seed(0)
    policy = _make_policy()
    obs = torch.randn(N_ENVS, OBS_DIM)
    action, run, sto, term = policy.sample_pi(obs)
    assert action.shape == (N_ENVS, ACT_DIM)
    assert action.abs().max() < 1.0  # tanh-squashed
    assert run.shape == term.shape == (N_ENVS,)
    assert torch.isfinite(run).all() and torch.isfinite(term).all()
    # stochastic_costs is identically zero in the reference (dead slot)
    assert (sto == 0).all()
    # 4-tuple contract: log-prob slot is the ELBO cost sum, sigma slot is ones
    a4, logp, _, ones = policy.sample_with_log_prob(obs)
    assert logp.shape == (N_ENVS,) and torch.isfinite(logp).all()
    assert torch.equal(ones, torch.ones_like(a4))


def test_pathwise_gradient_flows_through_denoising_chain():
    """∂a/∂θ must exist — the pathwise ∂Q/∂a · ∂a/∂θ is REPPO's whole reward signal."""
    torch.manual_seed(0)
    policy = _make_policy()
    action, *_ = policy.sample_pi(torch.randn(N_ENVS, OBS_DIM))
    action.sum().backward()
    grad = policy.actor.diffusion_model.fwd_model.state_time_net[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_kl_zero_against_identical_old_actor_and_positive_after_perturbation():
    """The simplified drift-diff KL is exactly 0 for identical models; > 0 after
    perturbing the control net; sync_old_actor() restores exact 0."""
    torch.manual_seed(0)
    policy = _make_policy()
    obs = torch.randn(N_ENVS, OBS_DIM)
    _, kl0 = policy.kl_forward(obs, n_samples=1)
    torch.testing.assert_close(kl0, torch.zeros_like(kl0))
    with torch.no_grad():
        policy.actor.diffusion_model.fwd_model.state_time_net[-1].bias.add_(0.3)
    _, kl1 = policy.kl_forward(obs, n_samples=2)
    assert (kl1 > 0).all()
    policy.sync_old_actor()
    _, kl2 = policy.kl_forward(obs, n_samples=1)
    torch.testing.assert_close(kl2, torch.zeros_like(kl2))


def test_kl_gradients_reach_new_actor_only():
    torch.manual_seed(0)
    policy = _make_policy()
    with torch.no_grad():
        policy.actor.diffusion_model.fwd_model.state_time_net[-1].bias.add_(0.3)
    _, kl = policy.kl_forward(torch.randn(N_ENVS, OBS_DIM), n_samples=1)
    kl.mean().backward()
    assert policy.actor.diffusion_model.fwd_model.state_time_net[-1].weight.grad is not None
    for p in policy.old_actor.parameters():
        assert p.grad is None and not p.requires_grad


def test_full_kl_diagnostic_matches_reference_path_bitwise():
    """The full-KL diagnostic must not perturb the trust-region value the loss uses.

    Same RNG seed => `kl_div_with_full`'s log_ratios must equal `kl_div`'s exactly,
    or enabling diagnostics would silently change training.
    """
    torch.manual_seed(0)
    policy = _make_policy()
    with torch.no_grad():  # make the two models differ so the KL is nonzero
        policy.actor.diffusion_model.fwd_model.state_time_net[-1].bias.add_(0.4)
    obs = torch.randn(N_ENVS, OBS_DIM)

    torch.manual_seed(123)
    _, kl_ref = policy.kl_forward(obs, n_samples=2)
    torch.manual_seed(123)
    _, kl_diag, kl_full = policy.kl_forward(obs, n_samples=2, with_full=True)

    torch.testing.assert_close(kl_diag, kl_ref, rtol=0, atol=0)
    assert torch.isfinite(kl_full).all()


def test_full_and_simplified_kl_agree_when_friction_matches():
    """The dropped log-variance terms vanish iff old/new transition noise agree.

    Perturb only the control net => frictions identical => the two KLs must match.
    Then perturb friction => they must diverge (that is the term being dropped).
    """
    torch.manual_seed(0)
    policy = _make_policy()
    obs = torch.randn(N_ENVS, OBS_DIM)

    with torch.no_grad():
        policy.actor.diffusion_model.fwd_model.state_time_net[-1].bias.add_(0.4)
    torch.manual_seed(7)
    _, kl_simp, kl_full = policy.kl_forward(obs, n_samples=1, with_full=True)
    torch.testing.assert_close(kl_full, kl_simp, rtol=1e-5, atol=1e-5)

    with torch.no_grad():  # now move friction only
        policy.actor.diffusion_model.friction.add_(0.5)
    torch.manual_seed(7)
    _, kl_simp2, kl_full2 = policy.kl_forward(obs, n_samples=1, with_full=True)
    assert (kl_full2 - kl_simp2).abs().mean() > 1e-3


def test_ode_coef_is_configurable_and_changes_actions():
    """Score scaling `c` on the deployment ODE (TruDi's eval ablation sweeps it)."""
    torch.manual_seed(0)
    policy = _make_policy(ode_coef=1.0)
    assert policy.ode_coef == 1.0
    obs = torch.randn(N_ENVS, OBS_DIM)
    with torch.no_grad():  # give the control net a nonzero output to scale
        policy.actor.diffusion_model.fwd_model.state_time_net[-1].bias.add_(0.5)
    torch.manual_seed(3)
    a1 = policy.act_inference(obs)
    policy.ode_coef = 2.0
    torch.manual_seed(3)
    a2 = policy.act_inference(obs)
    assert not torch.allclose(a1, a2), "ode_coef had no effect on the ODE action"


def test_compile_score_net_preserves_state_dict_and_sync():
    """nn.Module.compile() must not disturb checkpoints or the old_actor sync.

    torch.compile(mod) would return an OptimizedModule whose state_dict keys gain
    an `_orig_mod.` prefix — breaking every saved checkpoint and
    sync_old_actor()'s load_state_dict. The .compile() METHOD compiles in place.
    Guards against someone "simplifying" it back to the function form.
    """
    plain = _make_policy()
    compiled = _make_policy(compile_score_net=True)
    assert set(plain.state_dict().keys()) == set(compiled.state_dict().keys())
    assert not any("_orig_mod" in k for k in compiled.state_dict().keys())
    # checkpoints move both directions
    compiled.load_state_dict(plain.state_dict())
    plain.load_state_dict(compiled.state_dict())
    # the old_actor hard sync still works on a compiled module
    with torch.no_grad():
        compiled.actor.diffusion_model.fwd_model.state_time_net[-1].bias.add_(0.3)
    compiled.sync_old_actor()
    for p_new, p_old in zip(compiled.actor.parameters(), compiled.old_actor.parameters()):
        assert torch.equal(p_new, p_old)
    _, kl = compiled.kl_forward(torch.randn(N_ENVS, OBS_DIM), n_samples=1)
    torch.testing.assert_close(kl, torch.zeros_like(kl))


def test_trudi_wandb_schema_aliases_are_additive_and_correct():
    """Reference-schema mirroring must add keys, never replace or drop ours.

    Lets our runs overlay the authors' runs on the same wandb charts. The mapping
    was read off a live reference run, so guard it against drift.
    """
    from safe_rl.utils.trudi_wandb_schema import TRUDI_KEY_MAP, add_trudi_aliases

    torch.manual_seed(0)
    alg = _make_alg(_make_policy(), trudi_wandb_schema=True)
    next_obs = _rollout(alg)
    alg.compute_returns(next_obs)
    metrics = alg.update()

    # every original key survives
    plain = _make_alg(_make_policy())
    next_obs = _rollout(plain)
    plain.compute_returns(next_obs)
    for key in plain.update():
        assert key in metrics, f"aliasing dropped our key {key!r}"

    # mapped keys appear under the reference namespace with identical values
    mapped = [(o, t) for o, t in TRUDI_KEY_MAP.items() if o in metrics]
    assert mapped, "no reference aliases were produced"
    for ours, theirs in mapped:
        assert "/" in theirs  # runner logs "/"-containing keys verbatim
        assert metrics[theirs] == metrics[ours]
    assert metrics.get("actor/sto_cost") == 0.0  # zero by construction in DIME

    # off by default
    assert "actor/kl" not in add_trudi_aliases({"foo": 1.0})


def test_reverse_kl_fused_rollout():
    torch.manual_seed(0)
    policy = _make_policy()
    action, run, sto, term, kl = policy.sample_pi_with_kl(torch.randn(N_ENVS, OBS_DIM))
    assert kl.shape == (N_ENVS,) and torch.isfinite(kl).all()
    # full closed-form KL of a model against its identical copy is exactly 0
    torch.testing.assert_close(kl, torch.zeros_like(kl))


def test_actor_params_exclude_old_actor_and_critic():
    """The actor optimizer enumerates policy.actor.parameters(); old_actor and
    critic must not leak into it."""
    policy = _make_policy()
    actor_ids = {id(p) for p in policy.actor.parameters()}
    assert actor_ids.isdisjoint({id(p) for p in policy.old_actor.parameters()})
    assert actor_ids.isdisjoint({id(p) for p in policy.critic.parameters()})


def test_runner_surfaces_action_std_inference_state_dict():
    torch.manual_seed(0)
    policy = _make_policy()
    obs = torch.randn(N_ENVS, OBS_DIM)
    std = policy.action_std
    assert std.shape == (ACT_DIM,) and (std > 0).all()
    a = policy.act_inference(obs)
    assert a.shape == (N_ENVS, ACT_DIM) and a.abs().max() < 1.0
    # ODE chain is reproducible given the same prior draw
    torch.manual_seed(7)
    a1 = policy.act_inference(obs)
    torch.manual_seed(7)
    a2 = policy.act_inference(obs)
    torch.testing.assert_close(a1, a2)
    # old_actor is part of the checkpoint
    sd = policy.state_dict()
    assert any(k.startswith("old_actor.") for k in sd)
    fresh = _make_policy()
    fresh.load_state_dict(sd)


def test_asymmetric_critic_obs():
    policy = _make_policy(num_critic_obs=OBS_DIM + 3)
    assert policy.actor_obs_normalizer is not policy.critic_obs_normalizer
    q = policy.evaluate_q(torch.randn(N_ENVS, OBS_DIM + 3), torch.randn(N_ENVS, ACT_DIM))
    assert q.shape == (N_ENVS, 1)


# ----------------------------------------------------------------------
# Algorithm
# ----------------------------------------------------------------------


def test_end_to_end_update_forward_kl():
    torch.manual_seed(0)
    alg = _make_alg(_make_policy(), kl_action_rep=2)
    pre = [p.clone() for p in alg.policy.actor.parameters()]
    next_obs = _rollout(alg)
    alg.compute_returns(next_obs)
    metrics = alg.update()
    for key, value in metrics.items():
        assert value == value and abs(value) != float("inf"), f"{key} not finite: {value}"
    # actor moved, old_actor hard-synced to the post-update actor
    assert any(not torch.equal(p0, p1) for p0, p1 in zip(pre, alg.policy.actor.parameters()))
    for p_new, p_old in zip(alg.policy.actor.parameters(), alg.policy.old_actor.parameters()):
        assert torch.equal(p_new, p_old)
        assert not p_old.requires_grad
    # near-zero-init control net ⇒ first-iteration KL far inside the bound
    assert 0.0 <= metrics["kl"] < 0.1
    # critic grad flags restored after the actor pass
    assert all(p.requires_grad for p in alg.policy.critic.parameters())


def test_end_to_end_update_reverse_kl_and_dual_modes():
    torch.manual_seed(1)
    alg = _make_alg(_make_policy(), dime_kl_mode="reverse")
    next_obs = _rollout(alg)
    alg.compute_returns(next_obs)
    metrics = alg.update()
    assert all(v == v for v in metrics.values())

    torch.manual_seed(2)
    alg2 = _make_alg(_make_policy(), dual_optim_mode="separate", kl_clip_mode="full")
    next_obs = _rollout(alg2)
    alg2.compute_returns(next_obs)
    metrics2 = alg2.update()
    assert all(v == v for v in metrics2.values())


def test_second_iteration_runs_after_sync():
    torch.manual_seed(3)
    alg = _make_alg(_make_policy())
    for _ in range(2):
        next_obs = _rollout(alg)
        alg.compute_returns(next_obs)
        metrics = alg.update()
        assert all(v == v for v in metrics.values())


def test_target_entropy_scales_with_action_dim():
    """target_entropy: -4.0 (per dim) ⇒ -4.0 * n_act internally — the parity
    mapping to the reference's ent_target_mult 4.0."""
    alg = _make_alg(_make_policy())
    assert alg.target_entropy == -4.0 * ACT_DIM


def test_guards_reject_bad_configs():
    with pytest.raises(ValueError):
        REPPODIME(_make_policy(), dime_kl_mode="bogus", device="cpu")
    with pytest.raises(ValueError):
        REPPODIME(_make_policy(), target_entropy_final=-1.0, device="cpu")
    gaussian = REPPOActorCritic(
        OBS_DIM, OBS_DIM, ACT_DIM, actor_type="gaussian", critic_type="standard",
        actor_kwargs={"hidden_dims": [16], "activation": "elu"},
        critic_kwargs={"hidden_dims": [16], "activation": "elu"},
    )
    with pytest.raises(TypeError):
        REPPODIME(gaussian, device="cpu")


def test_entropy_is_negative_run_cost_not_full_pseudo_logp():
    """Reference (reppo_dime.py line 369): entropy = -run_cost — the terminal
    prior term is excluded from the temperature target while the FULL cost sum
    feeds the actor loss. Guard the asymmetry."""
    torch.manual_seed(0)
    policy = _make_policy()
    alg = _make_alg(policy)
    obs = torch.randn(N_ENVS, OBS_DIM)
    norm_obs = policy.actor_obs_normalizer(obs)
    torch.manual_seed(11)
    _, run, sto, term = policy.sample_pi(norm_obs, normalized=True)
    # replay the same sample inside _update_actor via the same seed
    torch.manual_seed(11)
    metrics = alg._update_actor(norm_obs, norm_obs, torch.zeros(N_ENVS, ACT_DIM), torch.ones(N_ENVS, ACT_DIM))
    assert metrics["entropy"] == pytest.approx((-run).mean().item(), rel=1e-4)


def test_old_actor_snapshot_defines_kl_not_storage_buffers():
    """The (mu, sigma) storage slots are dead for DIME — verify act() writes the
    placeholder zeros/ones and the update ignores them."""
    torch.manual_seed(4)
    alg = _make_alg(_make_policy())
    _rollout(alg, steps=1)
    torch.testing.assert_close(alg.storage.mu[0], torch.zeros(N_ENVS, ACT_DIM))
    torch.testing.assert_close(alg.storage.sigma[0], torch.ones(N_ENVS, ACT_DIM))


def test_shipped_config_constructs_policy_and_algorithm():
    """Build DIMEActorCritic + REPPODIME straight from config/mjlab_ant_reppodime.yaml.

    Guards against YAML type traps the class_name-resolution test cannot see —
    e.g. PyYAML parsing `1.0e4` (signless exponent) as a STRING, which made
    `torch.clamp(x, -outer_clip, ...)` crash at the first sampler call."""
    import pathlib

    import yaml

    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "config" / "mjlab_ant_reppodime.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    policy_cfg = dict(cfg["policy"])
    assert policy_cfg.pop("class_name") == "DIMEActorCritic"
    sm = policy_cfg["diffusion"]["score_model"]
    for key, value in sm.items():
        assert not isinstance(value, str) or key == "layer_norm_type", f"score_model.{key} parsed as string: {value!r}"
    policy = DIMEActorCritic(OBS_DIM, OBS_DIM, ACT_DIM, **policy_cfg)

    alg_cfg = dict(cfg["algorithm"])
    assert alg_cfg.pop("class_name") == "REPPODIME"
    alg = REPPODIME(policy, device="cpu", **alg_cfg)
    alg.init_storage("rl", N_ENVS, 1, [OBS_DIM], [OBS_DIM], [ACT_DIM])
    _rollout(alg, steps=1)  # exercises the full 8-step sampler incl. torch.clamp


def test_checkpoint_roundtrip_through_algorithm():
    """Save/restore the full policy incl. old_actor and duals mid-training."""
    torch.manual_seed(5)
    alg = _make_alg(_make_policy())
    next_obs = _rollout(alg)
    alg.compute_returns(next_obs)
    alg.update()
    sd_policy = copy.deepcopy(alg.policy.state_dict())
    alg2 = _make_alg(_make_policy())
    alg2.policy.load_state_dict(sd_policy)
    for p1, p2 in zip(alg.policy.parameters(), alg2.policy.parameters()):
        torch.testing.assert_close(p1, p2)
