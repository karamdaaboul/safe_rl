"""FHDCMPOPerState: the composition itself, which nothing else covers.

``FHDCMPOPerState`` contains almost no code -- it is an MRO and a five-line ``kappa`` ramp. That is
exactly why it needs a test. Every failure mode here is silent:

* reorder the bases and ``_estep_weights`` resolves to FH-DCMPO's, which runs the SHARED-lambda
  E-step. The arm then reports itself as per-state while running the thing it was built to replace;
* drop the ramp override and ``kappa`` never advances, so every ``fh_risk_mode: cvar`` arm quietly
  trains as a ``mean`` arm -- the constraint silently becomes the wrong one;
* let ``CVPOPerState.__init__`` stop cooperating with ``super()`` and FH-DCMPO's constructor is
  skipped: no ``gamma_c = 1``, no TD(lambda) window, no undiscounted threshold, and the run still
  starts.

None of these raise. All of them produce a plausible training curve that answers a different
question than the one the config asks. The assertions below are the cheapest available guard.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

NUM_OBS = 10
NUM_ACT = 2


def _policy():
    from safe_rl.modules import SafeActorCritic

    return SafeActorCritic(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACT,
        critic_type="quantile",
        cost_critic_type="quantile",
        num_costs=1,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"n_quantiles": 16, "nonneg": False, "network_kwargs": {"hidden_dims": [32, 32]}},
        cost_critic_kwargs={"n_quantiles": 16, "nonneg": True, "network_kwargs": {"hidden_dims": [32, 32]}},
    )


def _make(**overrides):
    from safe_rl.algorithms import FHDCMPOPerState

    kwargs = dict(
        cost_limits=[25.0],
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=16,
        mstep_iteration_num=2,
        cost_horizon=1000,
        cost_n_step=64,
        cost_td_lambda=0.995,
        n_step=10,
        lambda_max=1.8,
        device="cpu",
    )
    kwargs.update(overrides)
    return FHDCMPOPerState(_policy(), **kwargs)


def test_mro_is_pinned():
    """Base order decides which E-step runs. Pin it."""
    from safe_rl.algorithms import CVPO, CVPOPerState, FHDCMPO, FHDCMPOPerState

    mro = FHDCMPOPerState.__mro__
    assert mro[:4] == (FHDCMPOPerState, CVPOPerState, FHDCMPO, CVPO), (
        f"MRO changed to {[c.__name__ for c in mro[:4]]}; reversing CVPOPerState and FHDCMPO makes "
        "this arm run a SHARED lambda while still calling itself per-state."
    )


def test_the_per_state_dual_and_the_fh_cost_statistic_are_both_reachable():
    from safe_rl.algorithms import CVPOPerState, FHDCMPO, FHDCMPOPerState

    alg = _make()
    # the ramp override lives on the subclass and must delegate into the per-state E-step
    assert type(alg)._estep_weights.__qualname__.split(".")[0] == "FHDCMPOPerState"
    assert "_estep_weights" in CVPOPerState.__dict__, "per-state E-step vanished"
    # the cost statistic must be the finite-horizon one, not CVPO's discounted reading
    assert type(alg)._estep_cost.__qualname__.split(".")[0] == FHDCMPO.__name__


def test_fh_contract_survives_the_composition():
    """FHDCMPO.__init__ must actually run: skipping it leaves a discounted, non-FH arm."""
    alg = _make()
    assert alg.cost_gamma == 1.0, "gamma_c != 1: the cost channel is discounted again"
    assert alg.qc_thres == pytest.approx(25.0), "threshold is no longer the episodic limit"
    assert alg.cost_n_step == 64 and alg.cost_td_lambda == pytest.approx(0.995)
    assert alg._cost_bootstrap_discount(None) == 1.0
    dones = torch.tensor([[0.0], [1.0]])
    assert torch.allclose(alg._cost_bootstrap_mask(dones, None), 1.0 - dones), (
        "the cost channel is bootstrapping across the horizon boundary again"
    )


@pytest.mark.parametrize("mode,target", [("mean", 0.0), ("cvar", 1.0)])
def test_kappa_target_is_not_silently_zeroed(mode, target):
    alg = _make(fh_risk_mode=mode, fh_kappa=1.0)
    assert alg.fh_kappa_target == pytest.approx(target), (
        f"fh_risk_mode={mode!r} gave kappa target {alg.fh_kappa_target}; a cvar arm with target 0 "
        "trains as a mean arm and nothing in the logs says so."
    )


def test_the_kappa_ramp_actually_advances_through_the_per_state_estep():
    """The ramp lives in FHDCMPO._estep_weights, which this MRO bypasses. Without the override it
    would never advance -- the failure the subclass exists to prevent."""
    from safe_rl.common.fh_cost import kappa_at

    alg = _make(fh_risk_mode="cvar", fh_kappa=1.0, fh_kappa_warmup=2, fh_kappa_ramp=4)
    obs = torch.randn(6, NUM_OBS)
    seen = []
    for _ in range(8):
        before = alg._fh_updates
        with torch.no_grad():
            actor_obs = alg.policy.actor_obs_normalizer(obs)
            _, actions, q, _, _ = alg._estep_sample(actor_obs, obs)
            w = alg._estep_weights(q, actions, obs)
        assert alg._fh_updates == before + 1, "the ramp counter did not advance"
        assert torch.allclose(w.sum(dim=0), torch.ones(obs.shape[0]), atol=1e-5), (
            "E-step weights are not a per-state distribution"
        )
        seen.append(alg._fh_kappa)
    assert seen[0] == pytest.approx(kappa_at(0, 1.0, 2, 4))
    assert seen[-1] > seen[0], f"kappa never moved: {seen}"
    assert seen[-1] == pytest.approx(1.0), f"kappa did not reach its target: {seen}"


def test_lambda_is_actually_per_state_and_bounded():
    """A single lambda shared across states would mean the shared-lambda path is running.

    The limit is deliberately tiny: with a randomly-initialised cost critic and the real budget of
    25 the constraint is INACTIVE everywhere, every lambda_b is correctly 0, and an equality-free
    assertion would pass without the per-state solve ever doing anything.
    """
    alg = _make(cost_limits=[0.05])
    obs = torch.randn(16, NUM_OBS)
    obs[:8] *= 4.0  # states with genuinely different cost structure
    with torch.no_grad():
        actor_obs = alg.policy.actor_obs_normalizer(obs)
        _, actions, q, _, _ = alg._estep_sample(actor_obs, obs)
        w = alg._estep_weights(q, actions, obs)
    info = alg.get_penalty_info()

    assert torch.allclose(w.sum(dim=0), torch.ones(obs.shape[0]), atol=1e-5)
    assert info["lambda_min"] <= info["lambda_median"] <= info["lambda_max"]
    assert info["lambda_min"] >= 0.0
    assert info["lambda_max"] <= alg._current_lambda_max() + 1e-9, "lambda escaped its cap"
    assert info["lambda_max"] > info["lambda_min"] + 1e-6, (
        f"lambda is constant across states ({info['lambda_min']} == {info['lambda_max']}): the "
        "shared-lambda E-step is running, not the per-state one."
    )


def test_lambda_is_zero_everywhere_when_the_constraint_is_slack():
    """The other KKT corner: a budget nothing violates must leave the cost term entirely inert."""
    alg = _make(cost_limits=[1.0e6])
    obs = torch.randn(12, NUM_OBS)
    with torch.no_grad():
        actor_obs = alg.policy.actor_obs_normalizer(obs)
        _, actions, q, _, _ = alg._estep_sample(actor_obs, obs)
        alg._estep_weights(q, actions, obs)
    info = alg.get_penalty_info()
    assert info["lambda_max"] == pytest.approx(0.0, abs=1e-6), (
        f"lambda_max = {info['lambda_max']} under a slack constraint; the inactive corner is wrong."
    )
