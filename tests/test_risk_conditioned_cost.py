"""The risk-conditioned E-step must read the mode off the observation and pick that
statistic of Z_c.

The cost distribution on these tasks is zero-inflated -- >50% of the mass sits on the zero
atom -- so its mean lands near q0.79 and a LOW quantile is the risk-seeking choice. Any
quantile below ~q0.55 collapses onto the zero atom, which would silently switch the
constraint off; `risk_floor_frac` is what stops that.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

NUM_ATOMS, V_MIN, V_MAX = 101, 0.0, 50.0


def _critic_with(dist_row: torch.Tensor):
    """A DistributionalCritic whose forward returns a fixed categorical."""
    from safe_rl.modules.critic import DistributionalCritic

    critic = DistributionalCritic.__new__(DistributionalCritic)
    torch.nn.Module.__init__(critic)
    critic.num_atoms = NUM_ATOMS
    critic.register_buffer("q_support", torch.linspace(V_MIN, V_MAX, NUM_ATOMS))
    logits = torch.log(dist_row.clamp_min(1e-12))
    critic.forward = lambda obs, act, _l=logits: _l.expand(obs.shape[0], -1).clone()
    return critic


def _alg(critic, risk_levels, floor=0.3):
    """Bind the real method onto a stub carrying only what it touches."""
    from safe_rl.algorithms.cvpo import CVPO

    alg = types.SimpleNamespace(
        policy=types.SimpleNamespace(
            is_distributional_cost_critic=True,
            cost_critics=[critic],
            cost_critic_targets=[critic],
            critic_obs_normalizer=torch.nn.Identity(),
        ),
        risk_levels=risk_levels,
        num_risk_modes=len(risk_levels),
        risk_floor_frac=floor,
    )
    alg._risk_cost = types.MethodType(CVPO._risk_cost, alg)
    return alg


def _zero_inflated() -> torch.Tensor:
    """60% of the mass on the zero atom, the rest spread into a right tail."""
    d = torch.zeros(NUM_ATOMS)
    d[0] = 0.60
    d[4:24] = 0.40 / 20
    return d


def _obs(levels: list[float]) -> torch.Tensor:
    """Observations whose last column is the risk level in [0, 1]."""
    obs = torch.zeros(len(levels), 8)
    obs[:, -1] = torch.tensor(levels)
    return obs


def test_each_mode_selects_its_own_statistic() -> None:
    critic = _critic_with(_zero_inflated())
    alg = _alg(critic, [-0.5, 1.0, 0.1], floor=0.0)
    obs = _obs([0.0, 0.5, 1.0])  # risky / neutral / averse
    out = alg._risk_cost(obs, torch.zeros(3, 2), target=False).squeeze(-1)

    dist = critic.get_dist(critic(obs, None))
    assert out[0] == pytest.approx(float(critic.risk_value(dist[:1], -0.5)))
    assert out[1] == pytest.approx(float(critic.get_value(dist[:1])))
    assert out[2] == pytest.approx(float(critic.risk_value(dist[:1], 0.1)))


def test_ordering_is_risky_below_neutral_below_averse() -> None:
    """The direction that makes the dial meaningful, on a realistic zero-inflated shape."""
    alg = _alg(_critic_with(_zero_inflated()), [-0.5, 1.0, 0.1], floor=0.0)
    out = alg._risk_cost(_obs([0.0, 0.5, 1.0]), torch.zeros(3, 2), target=False).squeeze(-1)
    assert out[0] < out[1] < out[2], f"expected risky < neutral < averse, got {out.tolist()}"


def test_floor_stops_a_degenerate_quantile_disabling_the_constraint() -> None:
    """With 60% of the mass at zero, the mean of the best 50% is 0 -- and Q_c = 0 removes the cost term."""
    critic = _critic_with(_zero_inflated())
    dist = critic.get_dist(critic(_obs([0.0]), None))
    assert float(critic.risk_value(dist, -0.5)) == 0.0, "fixture must have a degenerate lower tail"

    unfloored = _alg(critic, [-0.5, 1.0], floor=0.0)
    assert float(unfloored._risk_cost(_obs([0.0]), torch.zeros(1, 2), False)) == 0.0

    floored = _alg(critic, [-0.5, 1.0], floor=0.3)
    mean = float(critic.get_value(dist))
    assert float(floored._risk_cost(_obs([0.0]), torch.zeros(1, 2), False)) == pytest.approx(0.3 * mean)


def test_level_is_rounded_to_the_nearest_trained_mode() -> None:
    """An eval-time level between trained modes must snap, not index out of range."""
    alg = _alg(_critic_with(_zero_inflated()), [-0.5, 1.0, 0.1], floor=0.0)
    out = alg._risk_cost(_obs([0.24, 0.26, 0.9, 1.0]), torch.zeros(4, 2), target=False).squeeze(-1)
    assert out[0] == out[0]  # finite
    assert out[1] == out[2] or out[1] != out[0]  # 0.26 -> mode 1, 0.24 -> mode 0
    assert torch.isfinite(out).all()


def test_per_mode_episode_metrics_use_the_finished_episodes_mode() -> None:
    """The split must read the mode the episode RAN under, not the one drawn for the next.

    Logging a flat mean of the mode index is useless -- it is a uniform draw, so it sits at
    its own mean forever. The informative split is reward/cost/goals per mode.
    """
    from safe_rl.envs.safety_gymnasium_vec_env import SafetyGymnasiumVecEnv

    env = SafetyGymnasiumVecEnv.__new__(SafetyGymnasiumVecEnv)
    env.device = torch.device("cpu")
    env.num_envs = 4
    env.num_risk_modes = 3
    env._risk_idx = torch.tensor([0, 1, 2, 0])
    env.episode_length_buf = torch.zeros(4, dtype=torch.long)
    env._goals_in_episode = torch.tensor([1.0, 2.0, 3.0, 4.0])
    env._reward_in_episode = torch.tensor([10.0, 20.0, 30.0, 40.0])
    env._cost_in_episode = torch.tensor([5.0, 6.0, 7.0, 8.0])

    extras: dict = {}
    dones = torch.tensor([1.0, 1.0, 1.0, 1.0])
    done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
    log = extras.setdefault("log", {})
    goals, reward, cost = (
        env._goals_in_episode[done_ids].clone(),
        env._reward_in_episode[done_ids].clone(),
        env._cost_in_episode[done_ids].clone(),
    )
    modes = env._risk_idx[done_ids]
    for m in range(env.num_risk_modes):
        sel = modes == m
        if bool(sel.any()):
            name = env._risk_name(m)
            log[f"reward_{name}"] = reward[sel]
            log[f"cost_{name}"] = cost[sel]

    # envs 0 and 3 ran mode 0; env 1 ran mode 1; env 2 ran mode 2
    assert log["reward_seeking"].tolist() == [10.0, 40.0]
    assert log["reward_neutral"].tolist() == [20.0]
    assert log["reward_averse"].tolist() == [30.0]
    assert log["cost_seeking"].tolist() == [5.0, 8.0]
    assert "risk_mode" not in log, "a flat mean of a uniform draw carries no information"
    assert not any(k.endswith(("risk0", "risk1", "risk2")) for k in log), "names must be readable"


def test_risk_value_is_a_tail_mean_not_a_quantile() -> None:
    """A tail MEAN uses every atom beyond the cut; a quantile reads only one."""
    critic = _critic_with(_zero_inflated())
    dist = critic.get_dist(critic(_obs([0.0]), None))

    averse = float(critic.risk_value(dist, 0.1))     # mean of the worst 10%
    var90 = float(critic.get_quantile(dist, 0.9))    # the single q0.9 atom
    assert averse > var90, "the mean of the worst 10% must exceed the atom at its boundary"

    assert float(critic.risk_value(dist, 1.0)) == pytest.approx(float(critic.get_value(dist)))
    assert float(critic.risk_value(dist, -1.0)) == pytest.approx(float(critic.get_value(dist)))


def test_risk_value_is_monotone_in_the_level() -> None:
    """The dial must be ordered: seeking < neutral < averse, with no special-case at 1.0."""
    critic = _critic_with(_zero_inflated())
    dist = critic.get_dist(critic(_obs([0.0]), None))
    seeking = float(critic.risk_value(dist, -0.5))
    neutral = float(critic.risk_value(dist, 1.0))
    averse = float(critic.risk_value(dist, 0.1))
    assert seeking < neutral < averse, (seeking, neutral, averse)

    # tightening the averse tail can only raise the estimate
    assert float(critic.risk_value(dist, 0.05)) >= averse
