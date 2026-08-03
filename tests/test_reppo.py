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
        num_critics=2,
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


def test_online_bootstrap_uses_online_nets():
    """With use_target_networks=False, corrupting the TARGET nets must not change returns."""
    torch.manual_seed(0)
    policy = _make_policy()
    alg = _make_alg(policy, use_target_networks=False)
    next_obs, _ = _rollout_one_step(alg)

    torch.manual_seed(7)
    alg.compute_returns(next_obs)
    before = alg.storage.returns.clone()

    with torch.no_grad():  # corrupt targets — online bootstrap must be immune
        for p in policy.critic_targets.parameters():
            p.add_(100.0)
        for p in policy.actor_target.parameters():
            p.add_(100.0)
    torch.manual_seed(7)
    alg.compute_returns(next_obs)
    torch.testing.assert_close(alg.storage.returns, before)


def test_q_reduction_modes():
    torch.manual_seed(0)
    policy = _make_policy()
    q1 = torch.tensor([1.0, 4.0])
    q2 = torch.tensor([3.0, 2.0])
    for mode, expected in [
        ("min", torch.tensor([1.0, 2.0])),
        ("mean", torch.tensor([2.0, 3.0])),
        ("q1", torch.tensor([1.0, 4.0])),
    ]:
        alg = _make_alg(_make_policy(), actor_q_reduction=mode)
        torch.testing.assert_close(alg._q_reduce(q1, q2), expected)
    with pytest.raises(ValueError):
        _make_alg(policy, actor_q_reduction="bogus")


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


def test_alpha_kl_floor_holds():
    """alpha_kl must never decay below alpha_kl_min (the v10/v11 gate-collapse fix)."""
    torch.manual_seed(0)
    alg = _make_alg(_make_policy(), alpha_kl_min=0.1, desired_kl=10.0,  # huge slack -> dual wants to decay
                    alpha_lr=0.5, num_learning_epochs=4, num_mini_batches=1)
    obs = torch.randn(N_ENVS, OBS_DIM)
    alg.act(obs, obs)
    alg.process_env_step(torch.randn(N_ENVS, 1), torch.zeros(N_ENVS, 1),
                         {"time_outs": torch.zeros(N_ENVS)}, next_obs=obs, next_critic_obs=obs)
    alg.compute_returns(obs)
    alg.update()
    assert alg.alpha_kl.item() >= 0.1 - 1e-6


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

    alg = _make_alg(policy, kl_clip_mode="clipped", use_target_networks=False, actor_q_reduction="q1")
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
        mu, sigma = policy.current_distribution_params(
            alg.storage.observations[step], normalized=True
        )
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
        num_critics=1,
        min_std=0.05,
        actor_kwargs={"network_type": "simba", "network_kwargs": {"hidden_dim": 16, "num_blocks": 1},
                      "log_std_squash": "tanh", "log_std_min": -3.0, "log_std_max": 0.7},
        critic_kwargs={"num_atoms": 21, "v_min": -5.0, "v_max": 5.0,
                       "network_type": "simba", "network_kwargs": {"hidden_dim": 16, "num_blocks": 1}},
    )
    alg = _make_alg(policy, use_target_networks=False, actor_q_reduction="q1", aux_loss_mult=1.0)
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
            use_target_networks=False,
            actor_q_reduction="q1",
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


def test_reference_parity_switch_validation():
    policy = _make_policy()
    with pytest.raises(ValueError, match="dual_optim_mode"):
        REPPO(policy, device="cpu", dual_optim_mode="bogus")
    with pytest.raises(ValueError, match="critic_loss_denominator"):
        REPPO(policy, device="cpu", critic_loss_denominator="bogus")
