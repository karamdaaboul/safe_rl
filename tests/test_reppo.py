"""Tests for the REPPO algorithm (soft-Q λ-target, v3 reference-faithful parts)."""

from __future__ import annotations

import math

import pytest
import torch

from safe_rl.algorithms.reppo import REPPO
from safe_rl.modules import REPPOActorCritic

OBS_DIM = 5
ACT_DIM = 2
N_ENVS = 4


def _make_policy(**overrides):
    kwargs = dict(
        num_actor_obs=OBS_DIM,
        num_critic_obs=OBS_DIM,
        num_actions=ACT_DIM,
        actor_type="gaussian",
        critic_type="standard",
        actor_kwargs={"hidden_dims": [16], "activation": "elu", "init_noise_std": 1.0, "noise_std_type": "log"},
        critic_kwargs={"hidden_dims": [16], "activation": "elu"},
    )
    kwargs.update(overrides)
    return REPPOActorCritic(**kwargs)


def _make_alg(policy, **overrides):
    kwargs = dict(num_learning_epochs=1, num_mini_batches=1, device="cpu")
    kwargs.update(overrides)
    alg = REPPO(policy, **kwargs)
    alg.init_storage("rl", N_ENVS, 1, [OBS_DIM], [OBS_DIM], [ACT_DIM])
    return alg


def _rollout_one_step(alg, dones=None, time_outs=None):
    obs = torch.randn(N_ENVS, OBS_DIM)
    alg.act(obs, obs)
    next_obs = torch.randn(N_ENVS, OBS_DIM)
    rewards = torch.randn(N_ENVS, 1)
    dones = dones if dones is not None else torch.zeros(N_ENVS, 1)
    infos = {"time_outs": time_outs if time_outs is not None else torch.zeros(N_ENVS)}
    alg.process_env_step(rewards, dones, infos, next_obs=next_obs, next_critic_obs=next_obs)
    return next_obs, rewards


def test_one_step_soft_q_target_counts_entropy_once():
    """target = r + gamma * (Q_tgt(s',a') - alpha * logp')  — NO -alpha*logp_t term.

    Guards against the entropy double-count regression: the entropy bonus enters
    only through soft_V(s'), never also through the reward at step t.
    """
    torch.manual_seed(0)
    policy = _make_policy()
    alg = _make_alg(policy)
    next_obs, rewards = _rollout_one_step(alg)

    # Bootstrap quantities are computed at COLLECTION time now (reference
    # collect_fn semantics) — verify the recursion against the collected
    # buffers: target = (r + ent_bonus) + gamma * Q'(s', a'), with the entropy
    # bonus entering ONCE at full weight through the reward.
    soft_v = alg._collect_next_values[0]
    ent_bonus = alg._collect_ent_bonus[0]
    alg.compute_returns(next_obs)
    got = alg.storage.returns[0]
    expected = rewards + ent_bonus + alg.gamma * soft_v
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_no_target_networks_exist():
    """REPPO must carry no target networks -- the reference has none.

    Its bootstrap reads the live actor/critic; freezing `next_values` once per
    iteration at collection IS the target mechanism. Polyak targets were inherited
    SAC scaffolding and were never enabled in any shipped config.
    """
    policy = _make_policy()
    for attr in ("actor_target", "critic_target", "critic_targets", "critics"):
        assert not hasattr(policy, attr), f"{attr} should no longer exist"
    assert hasattr(policy, "critic")
    alg = _make_alg(policy)
    for attr in ("use_target_networks", "tau"):
        assert not hasattr(alg, attr), f"REPPO.{attr} should no longer exist"


def test_privileged_critic_obs_is_separate_from_actor_obs():
    """Asymmetric obs must survive the refactor: the critic keeps its own input
    dimension and its own normalizer, distinct from the actor's."""
    policy = REPPOActorCritic(
        num_actor_obs=OBS_DIM,
        num_critic_obs=OBS_DIM + 3,          # privileged obs is wider
        num_actions=ACT_DIM,
        actor_type="gaussian",
        critic_type="standard",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_kwargs={"hidden_dims": [16], "activation": "elu"},
        critic_kwargs={"hidden_dims": [16], "activation": "elu"},
    )
    assert policy.actor_obs_normalizer is not policy.critic_obs_normalizer
    assert policy.critic_obs_normalizer._mean.shape[-1] == OBS_DIM + 3
    assert policy.actor_obs_normalizer._mean.shape[-1] == OBS_DIM
    q = policy.evaluate_q(torch.randn(4, OBS_DIM + 3), torch.randn(4, ACT_DIM))
    assert q.shape == (4, 1)


def test_min_std_floor_and_state_dependent_sigma():
    torch.manual_seed(0)
    policy = _make_policy(
        actor_type="stochastic",
        min_std=0.05,
        actor_kwargs={"hidden_dims": [16], "activation": "elu", "log_std_squash": "tanh",
                      "log_std_min": -3.0, "log_std_max": 0.7},
    )
    obs = torch.randn(N_ENVS, OBS_DIM)
    dist = policy._build_distribution(obs)
    assert (dist.stddev >= 0.05).all()
    assert (dist.stddev <= math.exp(0.7) + 0.05 + 1e-6).all()
    # state-dependent: different obs -> different sigma (almost surely)
    assert not torch.allclose(dist.stddev[0], dist.stddev[1])

    # act() must store the PER-STATE sigma, not a batch average
    alg = _make_alg(policy)
    alg.act(obs, obs)
    torch.testing.assert_close(alg.transition.action_sigma, dist.stddev.detach())


def test_reward_scale_scales_targets_linearly():
    """reward_scale must multiply stored rewards (and hence targets) — runner logging stays raw."""
    torch.manual_seed(0)
    policy = _make_policy()
    alg = _make_alg(policy, reward_scale=10.0)
    obs = torch.randn(N_ENVS, OBS_DIM)
    alg.act(obs, obs)
    rewards = torch.ones(N_ENVS, 1) * 0.5
    alg.process_env_step(rewards, torch.zeros(N_ENVS, 1), {"time_outs": torch.zeros(N_ENVS)},
                         next_obs=obs, next_critic_obs=obs)
    torch.testing.assert_close(alg.storage.rewards[0], rewards * 10.0)


def test_sigmoid_std_bounded_below_one():
    """log_std_squash='sigmoid' (author's ActorQ): sigma = sigmoid(x)+1e-4 in (0,1)."""
    torch.manual_seed(0)
    policy = _make_policy(
        actor_type="stochastic",
        min_std=0.0,
        squash="tanh",
        actor_kwargs={"network_type": "simba", "network_kwargs": {"hidden_dim": 16, "num_blocks": 1},
                      "log_std_squash": "sigmoid", "init_noise_std": 0.5},
    )
    obs = torch.randn(64, OBS_DIM)
    dist = policy._build_distribution(obs)
    assert (dist.stddev > 0.0).all() and (dist.stddev < 1.0 + 1e-4).all()
    # init_noise_std=0.5 -> predictor std-bias = logit(0.5) = 0 -> sigma near 0.5 at init
    assert 0.2 < dist.stddev.mean().item() < 0.8


def test_target_entropy_scales_with_action_dim_for_stochastic_actor():
    """Regression: StochasticActor lacked .num_actions -> target scaled by 1 not n_act,
    turning the temperature dual into a sigma-pump pinning entropy at -0.5 (v10-v12)."""
    policy = _make_policy(
        actor_type="stochastic",
        actor_kwargs={"hidden_dims": [16], "activation": "elu"},
    )
    alg = _make_alg(policy, target_entropy=-0.5)
    assert alg.target_entropy == -0.5 * ACT_DIM


def test_kl_dual_decays_to_zero_under_slack():
    """With the constraint slack, the multiplier must fall toward 0 -- that is
    complementary slackness, not a failure.

    The old `alpha_kl_min` floor blocked this and quietly changed the objective from
    "constrained improvement" to "constrained improvement + a permanent KL
    regularizer". The v10/v11 collapse it patched was a dual *rate* problem (512
    Adam steps per iteration), not a flaw in the formulation.
    """
    torch.manual_seed(0)
    alg = _make_alg(_make_policy(), desired_kl=10.0,  # huge slack: KL will never bind
                    alpha_lr=0.5, num_learning_epochs=4, num_mini_batches=1)
    before = alg.alpha_kl.item()
    obs = torch.randn(N_ENVS, OBS_DIM)
    alg.act(obs, obs)
    alg.process_env_step(torch.randn(N_ENVS, 1), torch.zeros(N_ENVS, 1),
                         {"time_outs": torch.zeros(N_ENVS)}, next_obs=obs, next_critic_obs=obs)
    alg.compute_returns(obs)
    alg.update()
    assert alg.alpha_kl.item() < before, "slack constraint must push the multiplier down"
    assert not hasattr(alg, "alpha_kl_min")


def test_tanh_squash_bounded_actions_and_mc_kl():
    """squash='tanh': actions in (-1,1), finite log-probs, MC-KL ~ 0 for identical dists,
    and a full update runs through the MC-KL/MC-entropy branch."""
    torch.manual_seed(0)
    policy = _make_policy(
        actor_type="stochastic",
        min_std=0.1,
        squash="tanh",
        actor_kwargs={"hidden_dims": [16], "activation": "elu",
                      "log_std_squash": "clamp", "log_std_min": -5.0, "log_std_max": 2.0},
    )
    obs = torch.randn(N_ENVS, OBS_DIM)
    a = policy.act(obs)
    assert (a.abs() < 1.0).all()
    lp = policy.get_actions_log_prob(a)
    assert torch.isfinite(lp).all()
    a2, lp2, mu, sigma = policy.sample_with_log_prob(obs)
    assert (a2.abs() < 1.0).all() and torch.isfinite(lp2).all()

    # MC-KL between identical squashed dists is ~0
    td = policy.squashed(torch.distributions.Normal(mu, sigma))
    s = policy._clamp_squashed(td.sample((64,)))
    kl = (td.log_prob(s).sum(-1) - td.log_prob(s).sum(-1)).mean()
    assert abs(kl.item()) < 1e-6

    alg = _make_alg(policy, kl_clip_mode="clipped")
    _rollout_one_step(alg)
    alg.compute_returns(torch.randn(N_ENVS, OBS_DIM))
    metrics = alg.update()
    for key, value in metrics.items():
        assert value == value, f"NaN in metric {key}"


def test_kl_clip_mode_validation_and_gate():
    """'clipped' mode must gate per sample; invalid mode must raise."""
    with pytest.raises(ValueError):
        _make_alg(_make_policy(), kl_clip_mode="bogus")
    alg = _make_alg(_make_policy(), kl_clip_mode="clipped")
    assert alg.kl_clip_mode == "clipped"
    # smoke the gated actor-update path end-to-end
    obs = torch.randn(N_ENVS, OBS_DIM)
    alg.act(obs, obs)
    alg.process_env_step(torch.randn(N_ENVS, 1), torch.zeros(N_ENVS, 1),
                         {"time_outs": torch.zeros(N_ENVS)}, next_obs=obs, next_critic_obs=obs)
    alg.compute_returns(obs)
    metrics = alg.update()
    assert metrics["kl"] == metrics["kl"]  # not NaN


def test_reward_normalization_scales_adaptively():
    """With reward_normalization on, stored rewards are divided by the running-return denom."""
    torch.manual_seed(0)
    policy = _make_policy()
    alg = _make_alg(policy, reward_normalization=True, reward_norm_g_max=10.0)
    obs = torch.randn(N_ENVS, OBS_DIM)
    alg.act(obs, obs)
    rewards = torch.full((N_ENVS, 1), 4.0)
    alg.process_env_step(rewards, torch.zeros(N_ENVS, 1), {"time_outs": torch.zeros(N_ENVS)},
                         next_obs=obs, next_critic_obs=obs)
    stored = alg.storage.rewards[0]
    assert torch.isfinite(stored).all()
    assert not torch.allclose(stored, rewards)  # scaled, not raw
    # denom floor = max|G|/g_max = 4/10 -> normalized reward <= 10
    assert (stored.abs() <= 10.0 + 1e-5).all()


def test_zero_init_prior_starts_at_zero_value():
    """zero_init_prior must make the initial E[Q] ~ 0 despite an asymmetric support."""
    from safe_rl.modules.critic import DistributionalCritic

    torch.manual_seed(0)
    kwargs = dict(num_obs=OBS_DIM, num_actions=ACT_DIM, num_atoms=151, v_min=-20.0, v_max=100.0,
                  network_type="simba", network_kwargs={"hidden_dim": 16, "num_blocks": 1})
    plain = DistributionalCritic(**kwargs)
    primed = DistributionalCritic(**kwargs, zero_init_prior=True)
    obs, act = torch.randn(8, OBS_DIM), torch.randn(8, ACT_DIM)
    v_plain = plain.get_value(plain.get_dist(plain(obs, act)))
    v_primed = primed.get_value(primed.get_dist(primed(obs, act)))
    assert v_plain.abs().mean() > 10.0  # uniform-ish init -> near support mean (+40)
    assert v_primed.abs().mean() < 1.0  # prior pins the init to ~0


def test_stored_obs_are_normalized_so_kl_starts_at_zero():
    """Regression: the KL must measure the POLICY change, not observation-stat drift.

    The reference stores normalized obs in its rollout buffer. We stored raw obs and
    re-normalized at update time, so the stored (mu, sigma) — produced under
    mid-rollout statistics — no longer matched what the unchanged policy produces at
    update time, and the trust region was charged for the difference.
    """
    torch.manual_seed(0)
    policy = _make_policy(actor_obs_normalization=True, critic_obs_normalization=True)
    alg = _make_alg(policy, num_mini_batches=1)
    alg.init_storage("rl", N_ENVS, 3, [OBS_DIM], [OBS_DIM], [ACT_DIM])

    # Drifting observations: the empirical normalizer's statistics move every step.
    for step in range(3):
        obs = torch.randn(N_ENVS, OBS_DIM) * (step + 1) + 10.0 * step
        alg.act(obs, obs)
        alg.process_env_step(
            torch.randn(N_ENVS, 1), torch.zeros(N_ENVS, 1),
            {"time_outs": torch.zeros(N_ENVS)}, next_obs=obs, next_critic_obs=obs,
        )

    # The policy has NOT been updated, so re-evaluating it on the stored observations
    # must reproduce the stored distribution exactly — for every step, including the
    # first (whose statistics are the most stale).
    for step in range(3):
        dist = policy._build_distribution(alg.storage.observations[step])
        mu, sigma = dist.mean, dist.scale
        torch.testing.assert_close(mu, alg.storage.mu[step], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(sigma, alg.storage.sigma[step], rtol=1e-6, atol=1e-6)


def test_reppo_normalizer_matches_reference_eps_placement():
    """Reference: divide by sqrt(var + eps); legacy: divide by (sqrt(var) + eps).
    On a near-constant channel (var=1e-4) the legacy form amplifies ~5x harder —
    a real 9%-level performance difference vs the authors' code (Humanoid A/B)."""
    from safe_rl.modules.normalizer import EmpiricalNormalization

    torch.manual_seed(0)
    x = torch.randn(4096, 1) * 0.01 + 3.0  # near-constant channel, std 0.01
    ref = EmpiricalNormalization(1, eps_mode="add_var")
    legacy = EmpiricalNormalization(1, eps_mode="add_std")
    ref.update(x); legacy.update(x)
    ref.eval(); legacy.eval()
    probe = torch.tensor([[3.05]])  # +5 sigma excursion
    out_ref = ref(probe).item()
    out_legacy = legacy(probe).item()
    assert abs(out_ref) < 1.0  # reference caps the gain at 1/sqrt(eps) = 10
    assert abs(out_legacy) > 2.0 * abs(out_ref)  # legacy amplifies much harder

    policy = _make_policy(actor_obs_normalization=True)
    assert policy.actor_obs_normalizer.eps_mode == "add_var"


def test_aux_predictor_head_is_applied_only_on_the_online_side():
    """Reference aux loss is pred(f(s,a)) -> sg[f(s',a')]; without the head it degenerates
    into pulling the critic's own features toward their next-state value."""
    from safe_rl.modules.critic import DistributionalCritic

    torch.manual_seed(0)
    kwargs = dict(num_obs=OBS_DIM, num_actions=ACT_DIM, num_atoms=21, v_min=-5.0, v_max=5.0,
                  network_type="simba", network_kwargs={"hidden_dim": 16, "num_blocks": 1})
    plain = DistributionalCritic(**kwargs)
    with_head = DistributionalCritic(**kwargs, aux_predictor=True)
    obs, act = torch.randn(8, OBS_DIM), torch.randn(8, ACT_DIM)

    assert plain.aux_predictor is None
    torch.testing.assert_close(plain.predict_features(plain.features(obs, act)),
                               plain.features(obs, act))
    feats = with_head.features(obs, act)
    assert not torch.allclose(with_head.predict_features(feats), feats)

    # The head participates in the critic's parameters (hence the critic optimizer).
    assert any("aux_predictor" in name for name, _ in with_head.named_parameters())


def test_simba_actor_and_aux_loss_update():
    """Full v3-style config: simba actor+critic, single critic, aux loss, online bootstrap."""
    torch.manual_seed(0)
    policy = _make_policy(
        actor_type="stochastic",
        critic_type="distributional",
        min_std=0.05,
        actor_kwargs={"network_type": "simba", "network_kwargs": {"hidden_dim": 16, "num_blocks": 1},
                      "log_std_squash": "tanh", "log_std_min": -3.0, "log_std_max": 0.7},
        critic_kwargs={"num_atoms": 21, "v_min": -5.0, "v_max": 5.0,
                       "network_type": "simba", "network_kwargs": {"hidden_dim": 16, "num_blocks": 1}},
    )
    alg = _make_alg(policy, aux_loss_mult=1.0)
    next_obs, _ = _rollout_one_step(alg, time_outs=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                                    dones=torch.tensor([[1.0], [0.0], [0.0], [0.0]]))
    alg.compute_returns(next_obs)
    assert alg._aux_targets is not None and alg._aux_targets.shape == (N_ENVS, 16)
    metrics = alg.update()
    for key, value in metrics.items():
        assert value == value, f"NaN in metric {key}"


# ---------------------------------------------------------------------------
# Reference-parity switches (v28). Each defaults to our historical behaviour;
# the "reference" setting reproduces the TruDi torch trainer exactly.
# ---------------------------------------------------------------------------


def test_dual_optim_mode_placement_and_clipping():
    """`dual_optim_mode` decides which optimizer owns the duals AND whether they are clipped.

    Reference: `log_temp`/`log_lagrange` are nn.Parameters inside the Actor, so they ride
    the single actor optimizer and sit inside `clip_grad_norm_(actor.parameters(), ...)`.
    Ours historically gave them a separate optimizer and never clipped them, so under a
    binding clip our duals take a strictly larger step.
    """
    import copy

    torch.manual_seed(0)
    policy_sep = _make_policy()
    policy_act = copy.deepcopy(policy_sep)

    alg_sep = _make_alg(policy_sep, dual_optim_mode="separate", max_grad_norm=1e-6)
    alg_act = _make_alg(policy_act, dual_optim_mode="actor", max_grad_norm=1e-6)

    # Placement
    sep_params = {id(p) for group in alg_sep.optimizer.param_groups for p in group["params"]}
    act_params = {id(p) for group in alg_act.optimizer.param_groups for p in group["params"]}
    assert id(alg_sep.log_alpha_temp) not in sep_params
    assert alg_sep.alpha_optimizer is not None
    assert id(alg_act.log_alpha_temp) in act_params
    assert id(alg_act.log_alpha_kl) in act_params
    assert alg_act.alpha_optimizer is None

    # Clipping: identical policies + identical RNG stream => identical pre-clip dual
    # gradients, so any difference afterwards is the clip.
    for alg in (alg_sep, alg_act):
        torch.manual_seed(1234)
        next_obs, _ = _rollout_one_step(alg)
        alg.compute_returns(next_obs)
        alg.update()

    assert alg_sep.log_alpha_temp.grad is not None and alg_act.log_alpha_temp.grad is not None
    assert alg_act.log_alpha_temp.grad.abs().item() < alg_sep.log_alpha_temp.grad.abs().item()


def test_force_last_step_truncated_bootstraps_and_masks_last_step():
    """Reference `compute_gve` sets `truncated[-1] = 1.0` in place before the recursion."""
    for force in (False, True):
        torch.manual_seed(0)
        policy = _make_policy()
        alg = REPPO(
            policy,
            num_learning_epochs=1,
            num_mini_batches=1,
            device="cpu",
            force_last_step_truncated=force,
        )
        alg.init_storage("rl", N_ENVS, 2, [OBS_DIM], [OBS_DIM], [ACT_DIM])
        # Step 0 ordinary; step 1 (the last) terminates with no timeout flag.
        _rollout_one_step(alg)
        next_obs, _ = _rollout_one_step(alg, dones=torch.ones(N_ENVS, 1))
        alg.compute_returns(next_obs)

        soft_r = alg.storage.rewards[-1] + torch.stack(alg._collect_ent_bonus)[-1]
        next_v = torch.stack(alg._collect_next_values)[-1]
        if force:
            assert torch.allclose(alg.storage.truncated[-1], torch.ones_like(alg.storage.truncated[-1]))
            # truncated branch: pure one-step bootstrap, termination mask overridden
            expected = soft_r + alg.gamma * next_v
        else:
            assert torch.allclose(alg.storage.truncated[-1], torch.zeros_like(alg.storage.truncated[-1]))
            # terminal: no bootstrap at all
            expected = soft_r
        assert torch.allclose(alg.storage.returns[-1], expected, atol=1e-6)


def test_critic_loss_denominator_conventions():
    """`batch` (reference) keeps masked samples in the denominator; `mask` drops them."""
    import copy

    torch.manual_seed(0)
    policy_mask = _make_policy()
    policy_batch = copy.deepcopy(policy_mask)
    alg_mask = _make_alg(policy_mask, critic_loss_denominator="mask")
    alg_batch = _make_alg(policy_batch, critic_loss_denominator="batch")

    critic_obs = torch.randn(8, OBS_DIM)
    actions = torch.randn(8, ACT_DIM)
    returns = torch.randn(8, 1)
    truncated = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(-1, 1)
    surviving, total = 6.0, 8.0

    loss_mask = alg_mask._update_critic(critic_obs, actions, returns, truncated)
    loss_batch = alg_batch._update_critic(critic_obs, actions, returns, truncated)
    assert loss_batch == pytest.approx(loss_mask * surviving / total, rel=1e-5)


def test_action_scale_widens_squashed_range_and_shifts_entropy_target():
    """`action_scale` s makes the tanh policy span (-s, s) with an exact log-Jacobian.

    tanh caps |a| at 1, but mjlab locomotion applies no action clipping, so an
    unbounded-Gaussian policy (PPO) commands far more — measured up to 4.46 on Go2's
    calf joints. Widening the range must (a) actually widen every action path and
    (b) shift target_entropy by n*log(s), so `target_entropy: -0.5` keeps meaning the
    same thing rather than silently demanding a log(s)-per-dim sharper policy.
    """
    scale = 3.0
    torch.manual_seed(0)
    policy = _make_policy(actor_type="stochastic", squash="tanh", action_scale=scale,
                          actor_kwargs={"hidden_dims": [16], "activation": "elu"})
    alg = _make_alg(policy, target_entropy=-0.5)

    obs = torch.randn(128, OBS_DIM)
    sampled = policy.act(obs)
    deterministic = policy.act_inference(obs)
    reparam, _, _, _ = policy.sample_with_log_prob(obs)
    for name, actions in (("act", sampled), ("act_inference", deterministic), ("rsample", reparam)):
        assert actions.abs().max().item() <= scale, f"{name} exceeded the action scale"
        assert actions.abs().max().item() > 1.0, f"{name} never left the unscaled +-1 range"

    expected = -0.5 * ACT_DIM + ACT_DIM * math.log(scale)
    assert alg.target_entropy == pytest.approx(expected, rel=1e-6)

    # scale 1 must be bit-identical to the historical behaviour
    torch.manual_seed(0)
    plain = _make_policy(actor_type="stochastic", squash="tanh",
                         actor_kwargs={"hidden_dims": [16], "activation": "elu"})
    plain_alg = _make_alg(plain, target_entropy=-0.5)
    assert plain_alg.target_entropy == pytest.approx(-0.5 * ACT_DIM)
    assert plain.act(obs).abs().max().item() <= 1.0

    with pytest.raises(ValueError, match="action_scale"):
        _make_policy(actor_type="stochastic", squash="tanh", action_scale=0.0)


def test_reference_parity_switch_validation():
    policy = _make_policy()
    with pytest.raises(ValueError, match="dual_optim_mode"):
        REPPO(policy, device="cpu", dual_optim_mode="bogus")
    with pytest.raises(ValueError, match="critic_loss_denominator"):
        REPPO(policy, device="cpu", critic_loss_denominator="bogus")


def test_legacy_twin_critic_checkpoint_still_loads(capsys):
    """Checkpoints predating the twin-critic removal stored `critics.0.*`.

    They must remap onto the single critic rather than becoming unloadable, and a
    stored SECOND critic must be dropped loudly — silently keeping only critic 1
    would misreport a twin-min policy as reproduced.
    """
    torch.manual_seed(0)
    policy = _make_policy()
    sd = policy.state_dict()

    legacy = {}
    for key, value in sd.items():
        if key.startswith("critic."):
            legacy["critics.0." + key[len("critic."):]] = value
            legacy["critics.1." + key[len("critic."):]] = value.clone()
        elif key.startswith("critic_target."):
            legacy["critic_targets.0." + key[len("critic_target."):]] = value
        else:
            legacy[key] = value
    assert not any(k.startswith("critic.") for k in legacy)

    fresh = _make_policy()
    fresh.load_state_dict(legacy)
    assert "dropped" in capsys.readouterr().out
    torch.testing.assert_close(fresh.critic.state_dict()["network.0.weight"],
                               sd["critic.network.0.weight"])


# ---------------------------------------------------------------------------
# Reference (JAX) aux loss: embedding MSE + reward MSE, averaged over D+1 slots
# ---------------------------------------------------------------------------

def _make_reference_policy(predict_reward: bool):
    return _make_policy(
        critic_type="reference",
        critic_kwargs={
            "num_atoms": 11,
            "v_min": 0.0,
            "v_max": 10.0,
            "hidden_dim": 8,
            "encoder_layers": 2,
            "head_layers": 2,
            "pred_layers": 2,
            "predict_reward": predict_reward,
        },
    )


def test_reward_prediction_head_widens_by_one_and_splits_reference_order():
    """`pred[..., :1]` is the reward, `pred[..., 1:]` the next-state features."""
    torch.manual_seed(0)
    critic = _make_reference_policy(predict_reward=True).critic
    assert critic.pred_module[-1][-1].out_features == critic.hidden_dim + 1

    feats = torch.randn(3, critic.hidden_dim)
    raw = critic.pred_module(feats)
    pred_f, pred_r = critic.predict_features_reward(feats)
    torch.testing.assert_close(pred_r, raw[..., :1])
    torch.testing.assert_close(pred_f, raw[..., 1:])
    # predict_features stays the feature slice, so the non-reward path is unchanged
    torch.testing.assert_close(critic.predict_features(feats), raw[..., 1:])

    off = _make_reference_policy(predict_reward=False).critic
    assert off.pred_module[-1][-1].out_features == off.hidden_dim
    with pytest.raises(RuntimeError, match="predict_reward"):
        off.predict_features_reward(feats)


def test_aux_loss_matches_reference_concat_mean_and_done_mask():
    """mean over D+1 of (1-done)*concat[feature_err, reward_err] -- not a 50/50 split."""
    torch.manual_seed(0)
    policy = _make_reference_policy(predict_reward=True)
    alg = _make_alg(policy, aux_loss_mult=1.0, aux_reward_pred=True, critic_loss_denominator="batch")

    B, D = 6, policy.critic.hidden_dim
    critic_obs = torch.randn(B, OBS_DIM)
    actions = torch.randn(B, ACT_DIM)
    aux_target = torch.randn(B, D)
    aux_reward = torch.randn(B)
    aux_done = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    truncated = torch.zeros(B)

    with torch.no_grad():
        pred_f, pred_r = policy.evaluate_q_features_reward(critic_obs, actions, normalized=True)
        se = torch.cat([(pred_f - aux_target).pow(2), (pred_r - aux_reward.view(-1, 1)).pow(2)], dim=-1)
        expected = ((1.0 - aux_done).view(-1, 1) * se).mean(dim=-1).mean()

    # The reward slot carries weight 1/(D+1); a 50/50 split would differ materially.
    with torch.no_grad():
        naive = 0.5 * ((pred_f - aux_target).pow(2).mean(-1) + (pred_r.squeeze(-1) - aux_reward).pow(2))
        naive = ((1.0 - aux_done) * naive).mean()
    assert not torch.isclose(expected, naive, rtol=1e-3)

    # Snapshot BEFORE the update: _update_critic steps the optimizer, so the
    # aux-off comparison has to start from the same parameters, not the updated ones.
    import copy

    sd = copy.deepcopy(policy.state_dict())
    total = alg._update_critic(critic_obs, actions, torch.zeros(B), truncated,
                               aux_target, aux_reward, aux_done)
    # Isolate the aux contribution by re-running with the aux term switched off.
    policy2 = _make_reference_policy(predict_reward=True)
    policy2.load_state_dict(sd)
    alg2 = _make_alg(policy2, aux_loss_mult=0.0, critic_loss_denominator="batch")
    value_only = alg2._update_critic(critic_obs, actions, torch.zeros(B), truncated)
    torch.testing.assert_close(torch.tensor(total - value_only), expected, rtol=1e-4, atol=1e-6)


def test_aux_reward_pred_off_leaves_the_embedding_only_path_untouched():
    torch.manual_seed(0)
    policy = _make_reference_policy(predict_reward=False)
    alg = _make_alg(policy, aux_loss_mult=1.0, aux_reward_pred=False)
    B, D = 4, policy.critic.hidden_dim
    critic_obs, actions = torch.randn(B, OBS_DIM), torch.randn(B, ACT_DIM)
    loss = alg._update_critic(critic_obs, actions, torch.zeros(B), torch.zeros(B), torch.randn(B, D))
    assert math.isfinite(loss)
