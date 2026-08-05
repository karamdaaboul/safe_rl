"""Item 1b: wiring measured qc_scale into CVPO behind `use_measured_qc_scale` (default off).

The flag must be inert when off (the smoke oracle covers bit-exactness); when on it may
change ONLY qc_thres and whatever follows from it (lambda, delta), never the E-step, the
M-step, or the critics.

The threshold is estimated once from the first N completed episodes after warm-up and then
FROZEN. A continuously-updating threshold would make the item-3 lambda controller impossible
to reason about: delta would move because the target moved, not because the policy did.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from safe_rl.common.cost_scaling import analytic_qc_scale  # noqa: E402

NUM_OBS = 6
NUM_ACT = 2
GAMMA = 0.99
EP_LEN = 200  # short synthetic episodes keep the test fast


def _make_alg(**overrides):
    from safe_rl.algorithms import CVPO
    from safe_rl.modules import SafeSACActorCritic

    policy = SafeSACActorCritic(
        num_actor_obs=NUM_OBS, num_critic_obs=NUM_OBS, num_actions=NUM_ACT, num_costs=1,
        actor_kwargs={"hidden_dims": [16, 16]},
        critic_kwargs={"hidden_dims": [16, 16]},
        cost_critic_kwargs={"hidden_dims": [16, 16]},
    )
    kwargs = dict(cost_limits=[25.0], batch_size=16, num_updates_per_step=1,
                  sample_action_num=8, mstep_iteration_num=1, gamma=GAMMA,
                  cost_horizon=EP_LEN, device="cpu")
    kwargs.update(overrides)
    alg = CVPO(policy, **kwargs)
    alg.init_storage(buffer_size=4000, num_envs=2, obs_shape=[NUM_OBS], act_shape=[NUM_ACT])
    return alg


def _feed_episodes(alg, n_episodes: int, cost_per_step: float = 1.0, ep_len: int = EP_LEN):
    """Push n_episodes of synthetic transitions per env with a constant per-step cost."""
    n_envs = 2
    for _ in range(n_episodes):
        for t in range(ep_len):
            done = torch.ones(n_envs) if t == ep_len - 1 else torch.zeros(n_envs)
            alg.store_transition(
                torch.randn(n_envs, NUM_OBS), torch.randn(n_envs, NUM_ACT),
                torch.randn(n_envs), done, torch.randn(n_envs, NUM_OBS),
                cost=torch.full((n_envs, 1), float(cost_per_step)),
                bootstrap=torch.zeros(n_envs),
            )


def test_flag_off_leaves_threshold_untouched() -> None:
    alg = _make_alg()
    assert alg.use_measured_qc_scale is False
    before = alg.qc_thres
    _feed_episodes(alg, n_episodes=4)
    assert alg.qc_thres == before, "threshold must not move when the flag is off"
    assert alg._qc_scale_frozen is True


def test_flag_on_estimates_and_freezes() -> None:
    alg = _make_alg(use_measured_qc_scale=True, qc_scale_estimate_episodes=4)
    analytic = alg._qc_scale
    assert analytic == pytest.approx(analytic_qc_scale(GAMMA, EP_LEN), rel=1e-9)
    assert alg._qc_scale_frozen is False

    _feed_episodes(alg, n_episodes=2)          # 2 episodes x 2 envs = 4 completed
    assert alg._qc_scale_frozen is True, "should freeze once enough episodes are seen"

    # Uniform cost -> measured scale must recover the analytic formula.
    assert alg._qc_scale == pytest.approx(analytic_qc_scale(GAMMA, EP_LEN), abs=1e-6)
    assert alg.qc_thres == pytest.approx(25.0 * alg._qc_scale, rel=1e-9)

    frozen = alg.qc_thres
    _feed_episodes(alg, n_episodes=3, cost_per_step=5.0)
    assert alg.qc_thres == frozen, "threshold must stay frozen after estimation"


def test_late_weighted_cost_lowers_the_threshold() -> None:
    """The real effect: discounting weights early steps, which carry less cost -> scale < analytic."""
    alg = _make_alg(use_measured_qc_scale=True, qc_scale_estimate_episodes=2)
    n_envs = 2
    for t in range(EP_LEN):
        done = torch.ones(n_envs) if t == EP_LEN - 1 else torch.zeros(n_envs)
        cost = 1.0 if t >= EP_LEN // 2 else 0.0     # all cost in the second half
        alg.store_transition(
            torch.randn(n_envs, NUM_OBS), torch.randn(n_envs, NUM_ACT),
            torch.randn(n_envs), done, torch.randn(n_envs, NUM_OBS),
            cost=torch.full((n_envs, 1), cost), bootstrap=torch.zeros(n_envs),
        )
    assert alg._qc_scale_frozen is True
    assert alg._qc_scale < analytic_qc_scale(GAMMA, EP_LEN)
    assert alg.qc_thres < 25.0 * analytic_qc_scale(GAMMA, EP_LEN)


def test_zero_cost_episodes_do_not_break_estimation() -> None:
    """~10% of real episodes cost nothing; estimation must survive them."""
    alg = _make_alg(use_measured_qc_scale=True, qc_scale_estimate_episodes=4)
    _feed_episodes(alg, n_episodes=1, cost_per_step=0.0)   # 2 zero-cost episodes
    assert alg._qc_scale_frozen is False, "all-zero episodes carry no information yet"
    _feed_episodes(alg, n_episodes=1, cost_per_step=1.0)   # 2 informative episodes
    assert alg._qc_scale_frozen is True
    assert np.isfinite(alg._qc_scale) and alg._qc_scale > 0


def test_flag_on_does_not_touch_critics_or_actor() -> None:
    """Only qc_thres may move.

    Feeding transitions consumes RNG by construction (the synthetic tensors are random), so
    the estimator is isolated by replaying an IDENTICAL pre-generated sequence with the flag
    off and on and comparing the resulting RNG state and parameters. Any divergence is
    attributable to the estimator alone.
    """
    n_envs = 2
    torch.manual_seed(0)
    # Late-weighted cost, so the measured scale genuinely differs from the analytic one.
    # (Uniform cost would make them identical and the final assertion vacuous.)
    seq = [(torch.randn(n_envs, NUM_OBS), torch.randn(n_envs, NUM_ACT), torch.randn(n_envs),
            torch.ones(n_envs) if t == EP_LEN - 1 else torch.zeros(n_envs),
            torch.randn(n_envs, NUM_OBS),
            torch.full((n_envs, 1), 1.0 if t >= EP_LEN // 2 else 0.0))
           for _ in range(2) for t in range(EP_LEN)]

    def replay(**flags):
        torch.manual_seed(123)
        alg = _make_alg(**flags)
        for obs, act, rew, done, nobs, cost in seq:
            alg.store_transition(obs, act, rew, done, nobs, cost=cost, bootstrap=torch.zeros(n_envs))
        params = torch.cat([p.detach().flatten() for p in alg.policy.parameters()])
        return params, torch.get_rng_state(), alg

    p_off, rng_off, alg_off = replay()
    p_on, rng_on, alg_on = replay(use_measured_qc_scale=True, qc_scale_estimate_episodes=4)

    assert torch.equal(p_off, p_on), "estimator must not mutate any parameter"
    assert torch.equal(rng_off, rng_on), "estimator must not consume RNG"
    # ...and the one thing it IS allowed to change did change.
    assert alg_on.qc_thres != alg_off.qc_thres or alg_on._qc_scale != alg_off._qc_scale


def test_penalty_info_exposes_the_scale_and_source() -> None:
    alg = _make_alg(use_measured_qc_scale=True, qc_scale_estimate_episodes=4)
    _feed_episodes(alg, n_episodes=2)
    info = alg.get_penalty_info()
    assert info["qc_scale"] == pytest.approx(alg._qc_scale)
    assert info["qc_thres"] == pytest.approx(alg.qc_thres)
