"""Tests for Regularized Online Adaptation (ActorCriticROA, PPO roa_cfg, ROAObsWrapper)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

NUM_PROP, NUM_PRIV, HIST_LEN, HIST_DIM, NUM_ACTIONS = 6, 4, 10, 5, 3
NUM_OBS = NUM_PROP + NUM_PRIV + HIST_LEN * HIST_DIM


def _make_policy(**overrides):
    from safe_rl.modules import ActorCriticROA

    kwargs = dict(
        num_prop=NUM_PROP,
        num_priv=NUM_PRIV,
        hist_len=HIST_LEN,
        hist_dim=HIST_DIM,
        priv_encoder_dims=[16, 8],
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"hidden_dims": [32, 32]},
    )
    kwargs.update(overrides)
    return ActorCriticROA(NUM_OBS, 7, NUM_ACTIONS, **kwargs)


def _grad_norm(module):
    grads = [p.grad.norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(grads).sum()) if grads else 0.0


@pytest.mark.parametrize("hist_len", [10, 20, 50])
def test_history_encoder_output_shape(hist_len):
    from safe_rl.modules import StateHistoryEncoder

    encoder = StateHistoryEncoder(HIST_DIM, hist_len, 8)
    assert encoder(torch.randn(4, hist_len, HIST_DIM)).shape == (4, 8)


def test_history_encoder_rejects_unsupported_length():
    from safe_rl.modules import StateHistoryEncoder

    with pytest.raises(ValueError):
        StateHistoryEncoder(HIST_DIM, 7, 8)


def test_rejects_mismatched_obs_width():
    from safe_rl.modules import ActorCriticROA

    with pytest.raises(ValueError):
        ActorCriticROA(NUM_OBS + 1, 7, NUM_ACTIONS, num_prop=NUM_PROP, num_priv=NUM_PRIV, hist_len=10, hist_dim=5)


def test_act_switches_latent_source():
    policy = _make_policy()
    obs = torch.randn(5, NUM_OBS)
    assert policy.act(obs).shape == (5, NUM_ACTIONS)
    teacher_mean = policy.action_mean.clone()
    policy.use_history_latent = True
    policy.act(obs)
    student_mean = policy.action_mean.clone()
    assert not torch.allclose(teacher_mean, student_mean)
    # Deployment path is the student, regardless of the rollout flag.
    policy.use_history_latent = False
    assert torch.allclose(policy.act_inference(obs), student_mean)
    assert torch.allclose(policy.act_inference_privileged(obs), teacher_mean)


def test_student_ignores_privileged_slice():
    policy = _make_policy()
    obs = torch.randn(5, NUM_OBS)
    corrupted = obs.clone()
    corrupted[:, NUM_PROP : NUM_PROP + NUM_PRIV] = 123.0
    assert torch.allclose(policy.act_inference(obs), policy.act_inference(corrupted))
    assert not torch.allclose(policy.act_inference_privileged(obs), policy.act_inference_privileged(corrupted))


def test_priv_reg_loss_trains_teacher_not_student():
    policy = _make_policy()
    policy.act(torch.randn(8, NUM_OBS))
    policy.priv_reg_loss().backward()
    assert _grad_norm(policy.priv_encoder) > 0
    assert _grad_norm(policy.history_encoder) == 0


def test_adaptation_loss_trains_student_only():
    policy = _make_policy()
    policy.adaptation_loss(torch.randn(8, NUM_OBS)).backward()
    assert _grad_norm(policy.history_encoder) > 0
    assert _grad_norm(policy.priv_encoder) == 0
    assert _grad_norm(policy.actor) == 0


def test_adaptation_loss_does_not_update_normalizer():
    policy = _make_policy(actor_obs_normalization=True)
    count = int(policy.actor_obs_normalizer.count)
    policy.adaptation_loss(torch.randn(8, NUM_OBS))
    assert int(policy.actor_obs_normalizer.count) == count
    policy.act(torch.randn(8, NUM_OBS))
    assert int(policy.actor_obs_normalizer.count) == count + 8


def test_priv_reg_loss_requires_teacher_act():
    policy = _make_policy()
    with pytest.raises(RuntimeError):
        policy.priv_reg_loss()


# -- PPO roa_cfg ------------------------------------------------------------------


def _make_ppo(policy=None, **roa_overrides):
    from safe_rl.algorithms import PPO

    roa_cfg = {"priv_reg_coef_schedule": [0.0, 0.1, 0, 0], "adaptation_interval": 2}
    roa_cfg.update(roa_overrides)
    ppo = PPO(policy or _make_policy(), num_learning_epochs=1, num_mini_batches=2, roa_cfg=roa_cfg)
    ppo.init_storage("rl", 4, 8, [NUM_OBS], [7], [NUM_ACTIONS])
    return ppo


def _fill_storage(ppo, adaptation=False):
    ppo.policy.use_history_latent = adaptation
    for _ in range(8):
        ppo.act(torch.randn(4, NUM_OBS), torch.randn(4, 7))
        ppo.process_env_step(torch.randn(4), torch.zeros(4, dtype=torch.bool), {})
    ppo.compute_returns(torch.randn(4, 7))
    ppo.policy.use_history_latent = False


def _snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def _changed(module, before):
    return any(not torch.equal(p, q) for p, q in zip(module.parameters(), before))


def test_ppo_keeps_history_encoder_out_of_main_optimizer():
    ppo = _make_ppo()
    main_ids = {id(p) for group in ppo.optimizer.param_groups for p in group["params"]}
    assert not main_ids & {id(p) for p in ppo.policy.history_encoder.parameters()}
    assert {id(p) for p in ppo.policy.priv_encoder.parameters()} <= main_ids
    assert {id(p) for group in ppo.roa_optimizer.param_groups for p in group["params"]} == {
        id(p) for p in ppo.policy.history_encoder.parameters()
    }


def test_ppo_update_regularizes_teacher_only():
    ppo = _make_ppo()
    _fill_storage(ppo)
    hist_before, priv_before = _snapshot(ppo.policy.history_encoder), _snapshot(ppo.policy.priv_encoder)
    loss_dict = ppo.update()
    assert loss_dict["roa_priv_reg"] > 0
    assert loss_dict["roa_priv_reg_coef"] == pytest.approx(0.1)
    assert not _changed(ppo.policy.history_encoder, hist_before)
    assert _changed(ppo.policy.priv_encoder, priv_before)


def test_adaptation_update_trains_history_encoder_only():
    ppo = _make_ppo()
    _fill_storage(ppo, adaptation=True)
    hist_before, rest_before = _snapshot(ppo.policy.history_encoder), _snapshot(ppo.policy.actor)
    priv_before = _snapshot(ppo.policy.priv_encoder)
    loss_dict = ppo.update_roa_adaptation()
    assert set(loss_dict) == {"roa_adaptation"} and loss_dict["roa_adaptation"] > 0
    assert _changed(ppo.policy.history_encoder, hist_before)
    assert not _changed(ppo.policy.actor, rest_before)
    assert not _changed(ppo.policy.priv_encoder, priv_before)


def test_adaptation_reduces_latent_gap():
    torch.manual_seed(0)
    ppo = _make_ppo(learning_rate=1e-2)
    obs = torch.randn(32, NUM_OBS)
    gap_before = float(ppo.policy.adaptation_loss(obs))
    for _ in range(30):
        _fill_storage(ppo, adaptation=True)
        ppo.update_roa_adaptation()
    assert float(ppo.policy.adaptation_loss(obs)) < gap_before


def test_priv_reg_coef_schedule_ramps_linearly():
    ppo = _make_ppo(priv_reg_coef_schedule=[0.0, 0.1, 10, 20])
    for count, expected in [(0, 0.0), (10, 0.0), (20, 0.05), (30, 0.1), (100, 0.1)]:
        ppo.roa_update_count = count
        assert ppo.roa_priv_reg_coef() == pytest.approx(expected)


def test_roa_cfg_validation():
    from safe_rl.algorithms import PPO
    from safe_rl.modules import ActorCritic

    with pytest.raises(ValueError, match="history encoder"):
        PPO(ActorCritic(NUM_OBS, 7, NUM_ACTIONS), roa_cfg={})
    with pytest.raises(ValueError, match="Unknown roa_cfg"):
        _make_ppo(typo=1)
    with pytest.raises(ValueError, match="priv_reg_coef_schedule"):
        _make_ppo(priv_reg_coef_schedule=[0.0, 0.1])


def test_roa_extra_state_round_trip():
    ppo = _make_ppo()
    _fill_storage(ppo, adaptation=True)
    ppo.update_roa_adaptation()
    state = ppo.extra_state_dict()
    fresh = _make_ppo()
    fresh.load_extra_state(state)
    assert fresh.roa_update_count == 1
    assert fresh.roa_optimizer.state_dict()["state"]


def test_plain_ppo_has_no_roa_state():
    from safe_rl.algorithms import PPO
    from safe_rl.modules import ActorCritic

    ppo = PPO(ActorCritic(NUM_OBS, 7, NUM_ACTIONS))
    assert ppo.roa_optimizer is None and ppo.roa_adaptation_interval == 0 and ppo.extra_state_dict() == {}


# -- Wrapper + runner ----------------------------------------------------------------


class _ROAStubEnv:
    """VecEnv publishing 'critic', 'priv' and 'actor_history' groups."""

    def __init__(self, num_envs=4, horizon=6, with_roa_groups=True):
        self.num_envs = num_envs
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = horizon
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.device = torch.device("cpu")
        self.cfg = {}
        self.with_roa_groups = with_roa_groups
        self.priv = torch.randn(num_envs, NUM_PRIV)
        self.history = torch.randn(num_envs, HIST_LEN, HIST_DIM)

    def _obs(self):
        groups = {"critic": torch.randn(self.num_envs, 7)}
        if self.with_roa_groups:
            groups.update(priv=self.priv, actor_history=self.history)
        return torch.randn(self.num_envs, NUM_PROP), {"observations": groups}

    def get_observations(self):
        return self._obs()

    def reset(self):
        return self._obs()

    def step(self, actions):
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        self.episode_length_buf[dones] = 0
        self.history = torch.cat([self.history[:, 1:], torch.randn(self.num_envs, 1, HIST_DIM)], dim=1)
        obs, extras = self._obs()
        extras["time_outs"] = dones.float()
        return obs, torch.randn(self.num_envs), dones, extras


def test_wrapper_composes_actor_observation():
    from safe_rl.envs.roa_obs_wrapper import ROAObsWrapper

    env = _ROAStubEnv()
    wrapped = ROAObsWrapper(env)
    assert wrapped.roa_obs_dims == {
        "num_prop": NUM_PROP,
        "num_priv": NUM_PRIV,
        "hist_len": HIST_LEN,
        "hist_dim": HIST_DIM,
    }
    obs, _, _, extras = wrapped.step(torch.zeros(4, NUM_ACTIONS))
    assert obs.shape == (4, NUM_OBS)
    assert torch.equal(obs[:, NUM_PROP : NUM_PROP + NUM_PRIV], env.priv)
    assert torch.equal(obs[:, NUM_PROP + NUM_PRIV :].reshape(4, HIST_LEN, HIST_DIM), env.history)
    wrapped.episode_length_buf = torch.full((4,), 3)
    assert torch.equal(env.episode_length_buf, torch.full((4,), 3))


def test_wrapper_requires_roa_groups():
    from safe_rl.envs.roa_obs_wrapper import ROAObsWrapper

    with pytest.raises(ValueError, match="priv"):
        ROAObsWrapper(_ROAStubEnv(with_roa_groups=False))


def _runner_cfg(roa_cfg):
    algorithm = {"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 2}
    if roa_cfg is not None:
        algorithm["roa_cfg"] = roa_cfg
    return {
        "algorithm": algorithm,
        "policy": {
            "class_name": "ActorCriticROA",
            "priv_encoder_dims": [16, 8],
            "actor_kwargs": {"hidden_dims": [16]},
            "critic_kwargs": {"hidden_dims": [16]},
        },
        "num_steps_per_env": 6,
        "save_interval": 100,
        "empirical_normalization": False,
        "logger": "tensorboard",
    }


def test_runner_alternates_adaptation_and_ppo_iterations(tmp_path):
    from safe_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(_ROAStubEnv(), _runner_cfg({"adaptation_interval": 2}), log_dir=str(tmp_path), device="cpu")
    policy = runner.alg.policy
    assert (policy.num_prop, policy.num_priv, policy.hist_len, policy.hist_dim) == (
        NUM_PROP,
        NUM_PRIV,
        HIST_LEN,
        HIST_DIM,
    )

    calls = []
    adapt, update = runner.alg.update_roa_adaptation, runner.alg.update
    runner.alg.update_roa_adaptation = lambda: calls.append(("adapt", policy.use_history_latent)) or adapt()
    runner.alg.update = lambda: calls.append(("ppo", policy.use_history_latent)) or update()
    runner.learn(4)
    assert calls == [("adapt", False), ("ppo", False), ("adapt", False), ("ppo", False)]
    assert runner.alg.roa_update_count == 4


def test_runner_requires_roa_cfg_for_roa_policy():
    from safe_rl.runners import OnPolicyRunner

    with pytest.raises(ValueError, match="roa_cfg"):
        OnPolicyRunner(_ROAStubEnv(), _runner_cfg(None), log_dir=None, device="cpu")
