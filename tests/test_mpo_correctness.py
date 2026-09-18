"""Regression checks for MPO's constrained updates and their training integration."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.distributions import Normal, kl_divergence

from safe_rl.algorithms import MPO, SAC
from safe_rl.algorithms.mpo import nonparametric_kl_from_weights
from safe_rl.modules import SACActorCritic


def _policy(**kwargs):
    return SACActorCritic(
        6,
        6,
        2,
        actor_kwargs={"hidden_dims": [16, 16], "head_init": "zero"},
        critic_kwargs={"hidden_dims": [16, 16]},
        **kwargs,
    )


def _mpo(**kwargs):
    return MPO(_policy(), sample_action_num=16, **kwargs)


@pytest.mark.parametrize("scale,offset", [(1e-8, 0), (1e-4, 0), (1, 1e8), (1e6, 0)])
def test_temperature_solve_is_invariant_to_q_units_and_state_baselines(scale, offset):
    alg = _mpo()
    base = torch.from_numpy(np.random.default_rng(0).normal(size=(64, 32)))
    weights = alg._estep_weights(base, None, None)
    # A state-dependent baseline must not change the action preferences in that state.
    shifted = base * scale + torch.linspace(-offset, offset, 32, dtype=torch.float64)
    actual = alg._estep_weights(shifted, None, None)
    assert torch.allclose(actual, weights, atol=1e-8, rtol=1e-7)
    assert nonparametric_kl_from_weights(actual).mean().item() == pytest.approx(alg.eps_dual, abs=1e-8)
    assert alg._solver_status == 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_estep_weights_are_detached_and_spend_the_kl_budget(dtype):
    alg = _mpo()
    q = (torch.from_numpy(np.random.default_rng(1).normal(size=(32, 8))) * 1e-5).to(dtype)
    q.requires_grad_()
    weights = alg._estep_weights(q, None, None)
    assert not weights.requires_grad
    assert weights.dtype == dtype
    assert torch.allclose(weights.sum(0), torch.ones(8, dtype=dtype))
    assert nonparametric_kl_from_weights(weights).mean().item() == pytest.approx(0.1, abs=2e-7)


def test_flat_q_and_one_sample_have_uniform_weights_without_forcing_equality():
    alg = _mpo()
    for q in (torch.tensor([[1.0, -4.0]]).expand(8, -1), torch.tensor([[1.0, 3.0]])):
        weights = alg._estep_weights(q, None, None)
        assert torch.equal(weights, torch.full_like(q, 1.0 / q.shape[0]))
        assert nonparametric_kl_from_weights(weights).mean() == 0
        assert np.isfinite(alg.eta) and alg.eta > 0


def test_inactive_constraint_shares_mass_among_tied_maxima():
    alg = _mpo(dual_constraint=1.0)
    q = torch.tensor([[2.0, 3.0], [2.0, 0.0], [0.0, 3.0], [0.0, 0.0]])
    weights = alg._estep_weights(q, None, None)
    assert torch.allclose(weights, torch.tensor([[0.5, 0.5], [0.5, 0.0], [0.0, 0.5], [0.0, 0.0]]))
    assert nonparametric_kl_from_weights(weights).mean() <= alg.eps_dual


@pytest.mark.parametrize("q", [np.array([[np.nan]]), np.array([[np.inf]]), np.zeros((0, 3)), np.zeros(3)])
def test_invalid_q_does_not_silently_reuse_an_old_temperature(q):
    with pytest.raises(ValueError, match="Q values"):
        _mpo()._solve_eta(q)


@pytest.mark.parametrize("scale", [0.0, 0.5, 2.0, float("nan"), float("inf")])
def test_proposals_without_importance_correction_are_rejected(scale):
    with pytest.raises(ValueError, match="estep_sample_std_scale"):
        _mpo(estep_sample_std_scale=scale)


@pytest.mark.parametrize("reduction,expected", [("min", 0.0), ("mean", 1.0)])
@pytest.mark.parametrize("target", [False, True])
def test_configured_critic_reduction_controls_estep_scores(monkeypatch, reduction, expected, target):
    alg = _mpo(estep_q_reduction=reduction, estep_use_target_critic=target)

    def evaluate(obs, actions):
        return torch.zeros(len(obs), 1), torch.full((len(obs), 1), 2.0)

    monkeypatch.setattr(alg.policy, "evaluate_q_target" if target else "evaluate_q", evaluate)
    with torch.no_grad():
        _, _, q, _, _ = alg._estep_sample(torch.zeros(8, 6), torch.zeros(8, 6))
    assert torch.equal(q, torch.full_like(q, expected))


def test_invalid_critic_reduction_is_rejected():
    with pytest.raises(ValueError, match="estep_q_reduction"):
        _mpo(estep_q_reduction="typo")


@pytest.mark.parametrize("critic_type", ["standard", "distributional", "quantile"])
def test_all_reward_backups_sample_the_reference_actor(monkeypatch, critic_type):
    from safe_rl.modules import SafeActorCritic

    critic_kwargs = (
        {"hidden_dims": [16, 16]}
        if critic_type == "standard"
        else {
            "network_kwargs": {"hidden_dims": [16, 16]},
            **({"n_quantiles": 8} if critic_type == "quantile" else {"num_atoms": 11}),
        }
    )
    policy = SafeActorCritic(
        6,
        6,
        2,
        critic_type=critic_type,
        critic_kwargs=critic_kwargs,
        actor_kwargs={"hidden_dims": [16, 16]},
        cost_critic_kwargs={"hidden_dims": [16, 16]},
        actor_obs_normalization=True,
    )
    alg = MPO(policy)
    policy.actor_obs_normalizer.eval()
    policy.actor_obs_normalizer._mean.fill_(2.0)
    policy.actor_obs_normalizer._std.fill_(3.0)
    obs = torch.ones(8, 6)
    calls = []

    def sample(normalized_obs):
        calls.append(normalized_obs.clone())
        assert not torch.is_grad_enabled()
        return torch.zeros(8, 2), torch.full((8, 1), 123.0)

    def unexpected(*args):
        pytest.fail("MPO reward backup used the online actor")

    monkeypatch.setattr(alg.actor_target, "sample", sample)
    monkeypatch.setattr(policy, "sample_with_log_prob", unexpected)
    loss = alg._update_critic(obs, obs, torch.zeros(8, 2), torch.ones(8, 1), torch.zeros(8, 1), obs, obs)
    assert np.isfinite(loss)
    assert len(calls) == 1
    assert torch.allclose(calls[0], torch.full_like(obs, -1.0 / 3.01))
    with torch.no_grad():
        _, log_prob = alg._sample_next_actions(obs)
    assert torch.equal(log_prob, torch.zeros(8, 1))  # No residual SAC entropy bonus.


def test_sac_backup_keeps_online_sampling_and_entropy(monkeypatch):
    alg = SAC(_policy())
    actions, log_prob = torch.ones(8, 2), torch.full((8, 1), -2.0)
    monkeypatch.setattr(alg.policy, "sample_with_log_prob", lambda obs: (actions, log_prob))
    actual_actions, actual_log_prob = alg._sample_next_actions(torch.zeros(8, 6))
    assert actual_actions is actions and actual_log_prob is log_prob


@pytest.mark.parametrize("per_dim", [False, True])
@pytest.mark.parametrize("iterations", [1, 3])
def test_mstep_diagnostics_describe_the_returned_actor(per_dim, iterations):
    torch.manual_seed(0)
    alg = _mpo(per_dim_constraining=per_dim, mstep_iteration_num=iterations)
    obs = torch.randn(8, 6)
    with torch.no_grad():
        old_mean, old_logstd = alg.actor_target(obs)
    alg._update_actor_and_alpha(obs)
    with torch.no_grad():
        mean, logstd = alg.policy.actor(obs)
        old = Normal(old_mean, old_logstd.exp())
        mean_dims = kl_divergence(old, Normal(mean, old_logstd.exp())).mean(0)
        var_dims = kl_divergence(old, Normal(old_mean, logstd.exp())).mean(0)
    info = alg.get_penalty_info()
    for name, values, budget in (("mean", mean_dims, alg.eps_kl_mean), ("var", var_dims, alg.eps_kl_var)):
        assert values.sum() > 0
        assert info[f"kl_{name}"] == pytest.approx(values.sum().item(), abs=1e-10)
        maximum = values.max() if per_dim else values.sum()
        assert info[f"kl_{name}_max_rel"] == pytest.approx(maximum.item() / budget, abs=1e-10)


def test_duals_can_grow_past_the_old_default_caps():
    alg = _mpo(mstep_iteration_num=1)
    alg.alpha_mean[:] = 10.0
    alg.alpha_var[:] = 1000.0
    with torch.no_grad():
        alg.policy.actor.mean_head.bias.fill_(1.0)
        alg.policy.actor.log_std_head.bias.fill_(1.0)
    alg._update_actor_and_alpha(torch.zeros(8, 6))
    assert (alg.alpha_mean > 10.0).all() and (alg.alpha_var > 1000.0).all()
    assert alg.get_penalty_info()["alpha_mean_at_cap"] == 0.0


class _Env:
    num_envs, num_actions, cfg = 1, 2, {}

    def get_observations(self):
        return torch.zeros(1, 6), {}

    def reset(self):
        return self.get_observations()


def _runner():
    from safe_rl.runners import OffPolicyRunner

    cfg = {
        "algorithm": {
            "class_name": "MPO",
            "sample_action_num": 16,
            "mstep_iteration_num": 1,
            "target_actor_update": "hard",
            "target_actor_period": 3,
        },
        "policy": {
            "class_name": "SACActorCritic",
            "actor_kwargs": {"hidden_dims": [16, 16]},
            "critic_kwargs": {"hidden_dims": [16, 16]},
        },
        "runner": {"max_size": 32, "reward_normalization": False},
    }
    return OffPolicyRunner(_Env(), cfg, log_dir=None, device="cpu")


def test_runner_checkpoint_restores_the_next_mpo_update(tmp_path):
    torch.manual_seed(1)
    src, dst = _runner(), _runner()
    obs = torch.randn(8, 6)
    for _ in range(2):
        src.alg._update_actor_and_alpha(obs)
    src.alg.update_step = 17
    src.current_learning_iteration, src.tot_timesteps = 23, 1234
    path = tmp_path / "mpo.pt"
    src.save(str(path))
    dst.load(str(path))
    assert dst.alg._actor_update_count == 2 and dst.alg.update_step == 17
    assert dst.current_learning_iteration == 23 and dst.tot_timesteps == 1234
    for key, value in src.alg.actor_target.state_dict().items():
        assert torch.equal(value, dst.alg.actor_target.state_dict()[key])
    assert dst.alg.eta == src.alg.eta
    assert np.array_equal(dst.alg.alpha_mean, src.alg.alpha_mean)
    assert np.array_equal(dst.alg.alpha_var, src.alg.alpha_var)
    # Restore RNG externally to isolate algorithm/optimizer state (replay and RNG are not saved).
    for alg in (src.alg, dst.alg):
        torch.manual_seed(12)
        alg._update_actor_and_alpha(obs)
    for left, right in zip(src.alg.policy.actor.parameters(), dst.alg.policy.actor.parameters()):
        assert torch.equal(left, right)
    assert np.array_equal(src.alg.alpha_mean, dst.alg.alpha_mean)
    for param, target in zip(dst.alg.policy.actor.parameters(), dst.alg.actor_target.parameters()):
        assert torch.equal(param, target)  # Restored counter triggers the hard copy on update three.


def test_legacy_checkpoint_initializes_reference_from_loaded_actor(tmp_path):
    src, dst = _runner(), _runner()
    with torch.no_grad():
        src.alg.policy.actor.mean_head.bias.fill_(2.0)
    path = tmp_path / "legacy.pt"
    src.save(str(path))
    saved = torch.load(path, weights_only=False)
    del saved["algorithm_state_dict"]
    torch.save(saved, path)
    dst.alg.alpha_mean[:] = 3.0
    dst.load(str(path), load_optimizer=False)
    for param, target in zip(dst.alg.policy.actor.parameters(), dst.alg.actor_target.parameters()):
        assert torch.equal(param, target)
        assert not target.requires_grad
    assert np.array_equal(dst.alg.alpha_mean, np.zeros(1))


def test_checkpoint_rejects_incompatible_kl_multiplier_shapes():
    state = deepcopy(_mpo(per_dim_constraining=True).get_checkpoint_state())
    with pytest.raises(ValueError, match="per_dim_constraining"):
        _mpo(per_dim_constraining=False).load_checkpoint_state(state)
