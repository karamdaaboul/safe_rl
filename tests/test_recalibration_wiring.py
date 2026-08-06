"""Item 2b: wiring recalibration into CVPO behind `recalibrate_cvar` (default off).

Two properties matter beyond "it runs":
  1. the MEAN constraint path must never touch recalibration -- it is a CVaR-only correction,
     and silently shifting the mean arm would re-confound the very comparison this work exists
     to fix;
  2. with the flag on, the recalibrated CVaR must exceed the raw one for an under-dispersed
     critic (the measured situation), i.e. the correction moves tail risk in the safe direction.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

NUM_OBS = 6
NUM_ACT = 2


def _make_alg(**overrides):
    from safe_rl.algorithms import CVPO
    from safe_rl.modules.safe_actor_critic import SafeActorCritic

    policy = SafeActorCritic(
        NUM_OBS, NUM_OBS, NUM_ACT,
        critic_type="distributional", cost_critic_type="distributional", num_costs=1,
        actor_kwargs={"hidden_dims": [16, 16]},
        critic_kwargs={"num_atoms": 51, "v_min": -5.0, "v_max": 15.0,
                       "network_kwargs": {"hidden_dims": [16, 16]}},
        cost_critic_kwargs={"num_atoms": 51, "v_min": 0.0, "v_max": 50.0,
                            "network_kwargs": {"hidden_dims": [16, 16]}},
    )
    kwargs = dict(cost_limits=[25.0], batch_size=16, num_updates_per_step=1,
                  sample_action_num=8, mstep_iteration_num=1, device="cpu")
    kwargs.update(overrides)
    return CVPO(policy, **kwargs)


def test_flag_defaults_off_and_recalibrator_absent() -> None:
    alg = _make_alg()
    assert alg.recalibrate_cvar is False
    assert alg._recalibrator is None


def test_mean_mode_never_recalibrates(monkeypatch) -> None:
    """The mean arm's computation must not reach the recalibration code at all."""
    import safe_rl.algorithms.cvpo as cvpo_mod

    alg = _make_alg(cost_constraint_mode="mean", recalibrate_cvar=True)
    called = {"n": 0}

    def spy(self, probs):
        called["n"] += 1
        return probs

    monkeypatch.setattr(cvpo_mod.CVPO, "_recalibrate_probs", spy, raising=True)
    obs, act = torch.randn(32, NUM_OBS), torch.randn(32, NUM_ACT)
    with torch.no_grad():
        alg._estep_cost(obs, act, target=False)
    assert called["n"] == 0, "mean constraint must never call recalibration"


def test_cvar_mode_calls_recalibration_when_enabled(monkeypatch) -> None:
    import safe_rl.algorithms.cvpo as cvpo_mod

    alg = _make_alg(cost_constraint_mode="cvar", recalibrate_cvar=True)
    called = {"n": 0}

    def spy(self, probs):
        called["n"] += 1
        return probs

    monkeypatch.setattr(cvpo_mod.CVPO, "_recalibrate_probs", spy, raising=True)
    with torch.no_grad():
        alg._estep_cost(torch.randn(8, NUM_OBS), torch.randn(8, NUM_ACT), target=False)
    assert called["n"] > 0


def test_unfitted_recalibrator_leaves_cvar_unchanged() -> None:
    """Before the buffer fills, the CVaR arm must behave exactly as the raw one."""
    torch.manual_seed(0)
    raw = _make_alg(cost_constraint_mode="cvar", recalibrate_cvar=False)
    torch.manual_seed(0)
    rec = _make_alg(cost_constraint_mode="cvar", recalibrate_cvar=True)
    obs, act = torch.randn(16, NUM_OBS), torch.randn(16, NUM_ACT)
    with torch.no_grad():
        assert torch.allclose(raw._estep_cost(obs, act, False), rec._estep_cost(obs, act, False))


def test_fitted_recalibration_raises_cvar_for_under_dispersed_critic() -> None:
    """Feed PITs from a too-narrow forecast; recalibrated CVaR must exceed raw CVaR."""
    alg = _make_alg(cost_constraint_mode="cvar", cvar_alpha=0.9, recalibrate_cvar=True,
                    recal_min_samples=200)
    rng = np.random.default_rng(0)
    # U-shaped PIT == predicted distribution too narrow (the measured signature).
    pits = np.concatenate([rng.uniform(0.0, 0.05, 800), rng.uniform(0.95, 1.0, 800),
                           rng.uniform(0.0, 1.0, 400)])
    alg._recalibrator.update(pits)
    assert alg._recalibrator.refit()

    obs, act = torch.randn(64, NUM_OBS), torch.randn(64, NUM_ACT)
    with torch.no_grad():
        recal_cvar = alg._estep_cost(obs, act, target=False)
        alg.recalibrate_cvar = False
        raw_cvar = alg._estep_cost(obs, act, target=False)
    assert recal_cvar.mean() > raw_cvar.mean(), (
        f"recalibrated CVaR {recal_cvar.mean():.4f} should exceed raw {raw_cvar.mean():.4f}"
    )


def test_recalibrated_probs_stay_a_valid_distribution() -> None:
    alg = _make_alg(cost_constraint_mode="cvar", recalibrate_cvar=True, recal_min_samples=100)
    rng = np.random.default_rng(1)
    alg._recalibrator.update(rng.beta(0.4, 3.0, size=1000))
    alg._recalibrator.refit()
    probs = torch.softmax(torch.randn(32, 51), dim=-1)
    out = alg._recalibrate_probs(probs)
    assert out.shape == probs.shape
    assert torch.all(out >= 0)
    assert torch.allclose(out.sum(-1), torch.ones(32), atol=1e-5)


def test_pit_collection_only_when_enabled() -> None:
    off = _make_alg(cost_constraint_mode="cvar", recalibrate_cvar=False)
    assert off._recalibrator is None
    on = _make_alg(cost_constraint_mode="cvar", recalibrate_cvar=True)
    assert on._recalibrator is not None and len(on._recalibrator) == 0
