"""FH-DCMPO: the wiring that makes the cost constraint mean what the config says.

Most of these tests guard *silent* failure paths. A stray ``qc_scale``, a discounted bootstrap, a
discounted n-step window, or an unsorted tail read all produce a run that trains happily and
enforces the wrong budget -- which is indistinguishable from "the method didn't help".
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from safe_rl.common.fh_cost import conservatism_statistic, quantile_cvar  # noqa: E402

NUM_OBS = 8
NUM_ACT = 2
N_Q = 16
SMALL_NET = {"hidden_dims": [32, 32], "activation": "relu"}


def _quantile_policy(cost_critic_type: str = "quantile"):
    from safe_rl.modules import SafeActorCritic

    if cost_critic_type == "quantile":
        cost_kwargs = {"n_quantiles": N_Q, "nonneg": True, "network_kwargs": SMALL_NET}
    else:
        cost_kwargs = {"num_atoms": 21, "v_min": 0.0, "v_max": 50.0, "network_kwargs": SMALL_NET}
    return SafeActorCritic(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACT,
        critic_type="quantile",
        cost_critic_type=cost_critic_type,
        num_costs=1,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"n_quantiles": N_Q, "nonneg": False, "network_kwargs": SMALL_NET},
        cost_critic_kwargs=cost_kwargs,
    )


def _make_fhdcmpo(policy=None, **overrides):
    from safe_rl.algorithms import FHDCMPO

    kwargs = dict(
        cost_limits=[25.0],
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=8,
        mstep_iteration_num=2,
        cost_horizon=1000,
        device="cpu",
    )
    kwargs.update(overrides)
    return FHDCMPO(policy or _quantile_policy(), **kwargs)


# -- Units: the threshold IS the cost limit ----------------------------------------------------


def test_threshold_equals_the_cost_limit_with_no_qc_scale() -> None:
    alg = _make_fhdcmpo()
    report = alg.cost_units_report()
    assert report["qc_thres"] == pytest.approx(25.0)
    assert report["cost_limit"] == pytest.approx(25.0)
    assert report["qc_scale"] == pytest.approx(1.0)


def test_both_cost_discounts_are_neutralised() -> None:
    """Two independent discounts. Missing either leaves the target discounted regardless."""
    alg = _make_fhdcmpo()
    report = alg.cost_units_report()
    assert report["cost_gamma"] == pytest.approx(1.0)  # the n-step window sum, in the buffer
    assert report["bootstrap_discount_1step"] == pytest.approx(1.0)  # the bootstrap
    assert report["bootstrap_discount_nstep_max"] == pytest.approx(1.0)


def test_cost_channel_does_not_bootstrap_across_the_horizon_boundary() -> None:
    """A time-limit truncation is real for a finite-horizon cost and artificial for the reward.

    Bootstrapping across ``T`` would add a whole extra episode of cost to every target beneath it,
    and would leave ``theta(s, u=0)`` unanchored -- it is only ever read as a bootstrap value, never
    regressed toward anything, so a softplus head could park it at a positive constant.
    """
    alg = _make_fhdcmpo()
    done = torch.ones(4, 1)
    timeout = torch.ones(4, 1)  # truncation: done=1, bootstrap=1
    assert float(alg._cost_bootstrap_mask(done, timeout).max()) == pytest.approx(0.0)
    # ...while the reward channel keeps the standard truncation-aware behaviour.
    assert float(alg._bootstrap_mask(done, timeout).max()) == pytest.approx(1.0)
    # Mid-episode both channels bootstrap normally.
    live = torch.zeros(4, 1)
    assert float(alg._cost_bootstrap_mask(live, live).min()) == pytest.approx(1.0)


def test_cost_units_report_records_the_truncation_treatment() -> None:
    report = _make_fhdcmpo().cost_units_report()
    assert report["truncation_mask"] == pytest.approx(0.0)
    assert report["truncation_mask_reward"] == pytest.approx(1.0)


def test_reward_channel_keeps_its_discount() -> None:
    """gamma_c = 1 must not leak into the reward critic."""
    alg = _make_fhdcmpo(gamma=0.99)
    assert alg.gamma == pytest.approx(0.99)
    assert float(alg._bootstrap_discount(None)) == pytest.approx(0.99)
    assert float(alg._bootstrap_discount(torch.tensor([10]))) == pytest.approx(0.99**10)
    # ...while the cost channel is flat at every horizon.
    assert float(alg._cost_bootstrap_discount(torch.tensor([10]))) == pytest.approx(1.0)


def test_rejects_a_qc_scale(caplog) -> None:
    with pytest.raises(ValueError, match="does not use qc_scale"):
        _make_fhdcmpo(qc_scale_measured=0.0764)


def test_rejects_measured_qc_scale_estimation() -> None:
    with pytest.raises(ValueError, match="meaningless"):
        _make_fhdcmpo(use_measured_qc_scale=True)


def test_rejects_a_categorical_cost_critic() -> None:
    """The categorical head clips at v_max; undiscounted episodic costs run past any fixed edge."""
    with pytest.raises(RuntimeError, match="quantile cost critic"):
        _make_fhdcmpo(policy=_quantile_policy(cost_critic_type="distributional"))


def test_recalibration_fails_loudly_rather_than_silently_doing_nothing() -> None:
    with pytest.raises(NotImplementedError, match="quantile head"):
        _make_fhdcmpo(recalibrate_cvar=True)


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"fh_risk_mode": "nope"}, "fh_risk_mode"),
        ({"fh_alpha": 1.0}, "fh_alpha"),
        ({"fh_alpha": -0.1}, "fh_alpha"),
        ({"fh_kappa": -1.0}, "fh_kappa"),
    ],
)
def test_rejects_bad_risk_configuration(bad: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _make_fhdcmpo(**bad)


# -- The conservatism statistic inside the E-step -----------------------------------------------


def _normalize(alg, obs):
    """Match the algorithm's own normalizer handling: it is `Identity` when normalization is off."""
    n = alg.policy.critic_obs_normalizer
    return n.normalize(obs) if hasattr(n, "normalize") else n(obs)


def _estep_cost_of(alg, obs, actions):
    # The real caller wraps the whole E-step in `no_grad` (mpo.py:358); the cost read is a value
    # lookup, never a gradient path -- that is the structural point of the weighted-M-step design.
    with torch.no_grad():
        return alg._estep_cost(obs, actions, target=False)[:, 0]


def test_estep_cost_at_kappa_zero_is_the_critic_mean() -> None:
    alg = _make_fhdcmpo(fh_risk_mode="mean")
    obs, actions = torch.randn(12, NUM_OBS), torch.rand(12, NUM_ACT) * 2 - 1
    got = _estep_cost_of(alg, obs, actions)
    obs_n = _normalize(alg, obs)
    expected = torch.stack([c(obs_n, actions).mean(dim=-1) for c in alg.policy.cost_critics]).mean(0)
    assert torch.allclose(got, expected, atol=1e-6)


def test_estep_cost_at_kappa_one_is_the_critic_cvar() -> None:
    alg = _make_fhdcmpo(fh_risk_mode="cvar", fh_alpha=0.9, fh_kappa=1.0, fh_kappa_warmup=0, fh_kappa_ramp=0)
    alg._fh_kappa = 1.0
    obs, actions = torch.randn(12, NUM_OBS), torch.rand(12, NUM_ACT) * 2 - 1
    got = _estep_cost_of(alg, obs, actions)
    obs_n = _normalize(alg, obs)
    expected = torch.stack([quantile_cvar(c(obs_n, actions), 0.9) for c in alg.policy.cost_critics]).mean(0)
    assert torch.allclose(got, expected, atol=1e-6)


def test_estep_cost_is_never_below_the_mean_once_kappa_is_positive() -> None:
    alg = _make_fhdcmpo(fh_risk_mode="cvar", fh_alpha=0.9)
    obs, actions = torch.randn(64, NUM_OBS), torch.rand(64, NUM_ACT) * 2 - 1
    obs_n = _normalize(alg, obs)
    mean = torch.stack([c(obs_n, actions).mean(dim=-1) for c in alg.policy.cost_critics]).mean(0)
    for kappa in (0.25, 0.5, 1.0):
        alg._fh_kappa = kappa
        assert bool((_estep_cost_of(alg, obs, actions) >= mean - 1e-6).all())


def test_estep_cost_matches_the_standalone_statistic() -> None:
    """The algorithm must not re-derive the statistic; it must call the tested one."""
    alg = _make_fhdcmpo(fh_risk_mode="cvar", fh_alpha=0.8)
    alg._fh_kappa = 0.6
    obs, actions = torch.randn(10, NUM_OBS), torch.rand(10, NUM_ACT) * 2 - 1
    obs_n = _normalize(alg, obs)
    expected = torch.stack([conservatism_statistic(c(obs_n, actions), 0.8, 0.6) for c in alg.policy.cost_critics]).mean(
        0
    )
    assert torch.allclose(_estep_cost_of(alg, obs, actions), expected, atol=1e-7)


# -- The kappa ramp, as driven by the E-step ----------------------------------------------------


def test_mean_mode_pins_kappa_at_zero_forever() -> None:
    alg = _make_fhdcmpo(fh_risk_mode="mean", fh_kappa=1.0, fh_kappa_warmup=0, fh_kappa_ramp=1)
    assert alg.fh_kappa_target == 0.0
    q = torch.randn(8, 4)
    actions = torch.rand(8, 4, NUM_ACT) * 2 - 1
    critic_obs = torch.randn(4, NUM_OBS)
    with torch.no_grad():
        for _ in range(5):
            alg._estep_weights(q, actions, critic_obs)
    assert alg.fh_kappa == 0.0


def test_kappa_advances_with_estep_updates_and_is_logged() -> None:
    alg = _make_fhdcmpo(fh_risk_mode="cvar", fh_kappa=1.0, fh_kappa_warmup=2, fh_kappa_ramp=4)
    q = torch.randn(8, 4)
    actions = torch.rand(8, 4, NUM_ACT) * 2 - 1
    critic_obs = torch.randn(4, NUM_OBS)
    seen = []
    with torch.no_grad():
        for _ in range(10):
            alg._estep_weights(q, actions, critic_obs)
            seen.append(alg._last_estep_info["fh_kappa"])
    assert seen[0] == 0.0 and seen[1] == 0.0  # warmup
    assert all(seen[i] <= seen[i + 1] + 1e-12 for i in range(len(seen) - 1))
    assert seen[-1] == pytest.approx(1.0)
    # The logged value is the one the weights were built with, not the next one.
    assert alg.get_penalty_info()["fh_kappa"] == pytest.approx(seen[-1])


def test_penalty_info_reports_the_predicted_violation_rate_only_at_full_kappa() -> None:
    alg = _make_fhdcmpo(fh_risk_mode="cvar", fh_alpha=0.9, fh_kappa=1.0, fh_kappa_warmup=0, fh_kappa_ramp=0)
    assert np.isnan(alg.get_penalty_info()["fh_predicted_violation_rate"])  # kappa still 0
    alg._fh_kappa = 1.0
    assert alg.get_penalty_info()["fh_predicted_violation_rate"] == pytest.approx(0.1)


# -- The dual stays a solved convex program ----------------------------------------------------


def _eq_cost(q, qc, eta, lam):
    """``E_q*[rho_c]`` under the E-step's own closed-form weights."""
    z = (q - lam * qc) / eta
    w = np.exp(z - z.max(axis=0, keepdims=True))
    w /= w.sum(axis=0, keepdims=True)
    return float((w * qc).sum(axis=0).mean())


def test_dual_kkt_residual_vanishes_when_the_constraint_is_active() -> None:
    """``dg/dlambda = d - E_q[rho_c]``, so an interior lambda must bind the constraint exactly.

    The costs are centred at 40 against a limit of 25 with enough spread (and enough KL budget)
    that reweighting the sampled actions can actually reach 25. That reachability is the whole
    precondition -- see the companion test below for what happens without it.
    """
    alg = _make_fhdcmpo(lambda_mode="dual", lambda_max=50.0, dual_constraint=1.0)
    rng = np.random.default_rng(0)
    q = rng.normal(size=(32, 16))
    qc = np.abs(rng.normal(loc=40.0, scale=15.0, size=(32, 16)))
    eta, lam = alg._solve_dual(q, qc)
    assert eta > 0 and 1e-6 < lam < 50.0, (eta, lam)
    residual = alg._effective_thres() - _eq_cost(q, qc, eta, lam)
    assert abs(residual) < 1e-2, residual


def test_dual_drives_lambda_to_the_floor_when_the_constraint_is_slack() -> None:
    """Complementary slackness: an inactive constraint must carry a zero multiplier."""
    alg = _make_fhdcmpo(lambda_mode="dual", lambda_max=50.0)
    rng = np.random.default_rng(1)
    q = rng.normal(size=(32, 16))
    qc = np.abs(rng.normal(loc=1.0, scale=0.2, size=(32, 16)))  # far under a limit of 25
    eta, lam = alg._solve_dual(q, qc)
    assert lam < 1e-3
    assert _eq_cost(q, qc, eta, lam) < alg._effective_thres()


def test_dual_pins_lambda_at_the_cap_when_the_sampled_support_cannot_reach_the_threshold() -> None:
    """The documented bang-bang failure, and the feasibility probe that is supposed to catch it.

    Costs centred at 40 with a *narrow* spread and a tight KL budget: no reweighting of these
    candidates reaches 25, so the dual is right to push lambda to its bound
    (codex/cvpo-negative-result.md). The point of asserting it is that the probe must *report* the
    infeasibility rather than leaving a pinned multiplier to look like a tuning problem.
    """
    alg = _make_fhdcmpo(lambda_mode="dual", lambda_max=50.0, dual_constraint=0.05)
    rng = np.random.default_rng(2)
    q = rng.normal(size=(32, 16))
    qc = np.abs(rng.normal(loc=40.0, scale=2.0, size=(32, 16)))
    eta, lam = alg._solve_dual(q, qc)
    assert lam == pytest.approx(50.0, rel=1e-3)
    alg._probe_feasibility(qc, alg._effective_thres())
    feas = alg._last_feasibility
    assert feas["estep_feasible"] == 0.0
    # The reachable floor is the honest statement of how far off the constraint is.
    assert feas["qc_reachable_min"] > alg._effective_thres()
    assert feas["estep_feasibility_margin"] < 0.0


# -- Undiscounted n-step cost aggregation in the buffer -----------------------------------------


def _nstep_cost_from_storage(cost_gamma, costs):
    from safe_rl.storage.replay_storage import ReplayStorage

    n = len(costs)
    store = ReplayStorage(
        num_envs=1,
        max_size=64,
        obs_shape=[NUM_OBS],
        action_shape=[NUM_ACT],
        n_step=n,
        gamma=0.99,
        cost_gamma=cost_gamma,
    )
    for c in costs:
        store.add(
            torch.zeros(1, NUM_OBS),
            torch.zeros(1, NUM_ACT),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1, NUM_OBS),
            costs=torch.full((1, 1), float(c)),
        )
    batch = store._gather_n_step(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    return float(batch["costs"].sum())


def test_cost_gamma_one_gives_the_plain_undiscounted_window_sum() -> None:
    costs = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert _nstep_cost_from_storage(1.0, costs) == pytest.approx(5.0)


def test_cost_gamma_defaults_to_gamma_so_existing_algorithms_are_unchanged() -> None:
    costs = [1.0, 1.0, 1.0, 1.0, 1.0]
    discounted = sum(0.99**k for k in range(5))
    assert _nstep_cost_from_storage(None, costs) == pytest.approx(discounted)
    assert _nstep_cost_from_storage(0.99, costs) == pytest.approx(discounted)


def test_storage_records_the_cost_discount_it_was_given() -> None:
    from safe_rl.storage.replay_storage import ReplayStorage

    plain = ReplayStorage(num_envs=1, max_size=64, obs_shape=[NUM_OBS], action_shape=[NUM_ACT], gamma=0.99)
    assert plain.cost_gamma == pytest.approx(0.99)
    fh = ReplayStorage(num_envs=1, max_size=64, obs_shape=[NUM_OBS], action_shape=[NUM_ACT], gamma=0.99, cost_gamma=1.0)
    assert fh.cost_gamma == pytest.approx(1.0)


# -- Horizon-augmented env ---------------------------------------------------------------------


def test_registry_routes_horizon_feature_to_the_augmented_env() -> None:
    pytest.importorskip("safety_gymnasium")
    from safe_rl.envs.horizon_augmented_vec_env import HorizonAugmentedVecEnv
    from safe_rl.envs.registry import make_env

    env = make_env("SafetyPointGoal1-v0", num_envs=2, horizon_feature=True, cost_limits=[25.0])
    try:
        assert isinstance(env, HorizonAugmentedVecEnv)
        assert env.risk_obs_dim == 1
        obs, _ = env.reset()
        # u == 1 at the start of an episode.
        assert obs[:, -1].allclose(torch.ones(2), atol=1e-6)
        base_dim = obs.shape[1] - 1
        obs2, _, _, _ = env.step(torch.zeros(2, env.num_actions))
        assert obs2.shape[1] == base_dim + 1
        # ...and strictly decreasing thereafter.
        assert bool((obs2[:, -1] < 1.0).all())
    finally:
        env.close()


def test_registry_leaves_the_plain_env_alone_when_the_flag_is_off() -> None:
    pytest.importorskip("safety_gymnasium")
    from safe_rl.envs.horizon_augmented_vec_env import HorizonAugmentedVecEnv
    from safe_rl.envs.registry import make_env

    env = make_env("SafetyPointGoal1-v0", num_envs=2, horizon_feature=False, cost_limits=[25.0])
    try:
        assert not isinstance(env, HorizonAugmentedVecEnv)
        assert env.risk_obs_dim == 0
    finally:
        env.close()


def test_budget_feature_requires_the_horizon_feature() -> None:
    pytest.importorskip("safety_gymnasium")
    from safe_rl.envs.horizon_augmented_vec_env import HorizonAugmentedVecEnv

    with pytest.raises(ValueError, match="budget_feature requires horizon_feature"):
        HorizonAugmentedVecEnv(
            env_id="SafetyPointGoal1-v0",
            num_envs=1,
            cost_limits=[25.0],
            horizon_feature=False,
            budget_feature=True,
        )
