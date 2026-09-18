"""Pre-tanh L2 penalty on the actor mean (arm T4).

WHY IT IS A PENALTY AND NOT A CLIP. Measured on the a2 arms, `pretanh_mean_absmax` reaches
9.3-12.6 with `frac_saturated` at 0.28. `sech^2(5.3)` is ~5e-5, so 64 candidate actions in that
region differ by almost nothing and the cost critic has no across-action variation to rank. A clip
would be WORSE, not better: it maps everything past the boundary onto exactly one action, removing
the variation rather than restoring it. This pulls the mean back toward tanh's responsive region
and leaves the squash in place.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

NUM_OBS, NUM_ACT, N_Q = 10, 2, 16
SMALL = {"hidden_dims": [32, 32], "activation": "relu"}


def _alg(**over):
    from safe_rl.algorithms import FHDCMPO
    from safe_rl.modules import SafeActorCritic

    pol = SafeActorCritic(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACT,
        critic_type="quantile",
        cost_critic_type="quantile",
        num_costs=1,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"n_quantiles": N_Q, "nonneg": False, "network_kwargs": SMALL},
        cost_critic_kwargs={"n_quantiles": N_Q, "nonneg": True, "network_kwargs": SMALL},
    )
    kw = dict(
        cost_limits=[25.0],
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=8,
        mstep_iteration_num=3,
        cost_horizon=1000,
        fh_risk_mode="mean",
        device="cpu",
    )
    kw.update(over)
    return FHDCMPO(pol, **kw)


def _mstep(alg, seed=0, n=8, b=12):
    torch.manual_seed(seed)
    obs = torch.randn(b, NUM_OBS)
    x = torch.randn(n, b, NUM_ACT) * 3.0
    w = torch.softmax(torch.randn(n, b), dim=0)
    mean_old, log_std_old = alg.policy.actor(obs)
    return alg._mstep(obs, x, w, mean_old.detach(), log_std_old.exp().detach())


def test_off_by_default_and_bit_identical() -> None:
    """Every arm before T4 ran with 0.0; the penalty must not perturb them."""
    a, b = _alg(), _alg()
    b.policy.load_state_dict(a.policy.state_dict())
    assert a.mean_l2_coef == 0.0
    la, ia = _mstep(a)
    lb, ib = _mstep(b)
    assert la == lb and ia["pretanh_mean_absmax"] == ib["pretanh_mean_absmax"]


def test_the_penalty_shrinks_the_pre_tanh_mean() -> None:
    """The whole point: same data, same seed, smaller |mean| with the penalty on."""
    off, on = _alg(mean_l2_coef=0.0), _alg(mean_l2_coef=1.0)
    on.policy.load_state_dict(off.policy.state_dict())
    # Drive the mean head far into saturation first, so there is something to pull back.
    for p_ in (off, on):
        with torch.no_grad():
            p_.policy.actor.mean_head.bias.fill_(8.0)
    _, i_off = _mstep(off, seed=1)
    _, i_on = _mstep(on, seed=1)
    assert i_on["pretanh_mean_absmean"] < i_off["pretanh_mean_absmean"]
    assert i_on["frac_saturated"] <= i_off["frac_saturated"]


def test_the_coefficient_is_monotone_at_equilibrium_not_per_step() -> None:
    """A larger coefficient pulls to a smaller fixed point -- but only over MANY steps.

    Per-step monotonicity does NOT hold, and that is Adam, not a bug: Adam normalizes by the
    gradient's running second moment, so a bigger penalty changes the step DIRECTION but leaves
    the magnitude bounded by the learning rate. Measured over 3 M-step iterations from a saturated
    start, coef 0.5 left |mean| at 8.060 vs 7.988 for coef 0.0 -- the wrong way round.

    Worse, the knob SATURATES. Measured from a saturated start, |mean| after 200 M-step iterations:

        coef      0.0    1.0   20.0  100.0
        |mean|  7.433  7.022  6.680  6.610

    A 5x increase from 20 to 100 buys 0.6%. So the coefficient cannot be tuned by reasoning about
    gradient magnitudes, and cranking it does not rescue a weak effect. Whether it can HOLD an
    equilibrium over 60k updates -- fighting a slow drift rather than undoing a large offset -- is
    a different question this test cannot answer, and is what the 2k gate measures.
    """
    prev = None
    for c in (0.0, 1.0, 100.0):
        alg = _alg(mean_l2_coef=c, mstep_iteration_num=200)
        with torch.no_grad():
            alg.policy.actor.mean_head.bias.fill_(8.0)
        _, info = _mstep(alg, seed=2)
        if prev is not None:
            assert info["pretanh_mean_absmean"] < prev, c
        prev = info["pretanh_mean_absmean"]


def test_it_is_logged_so_an_inert_coefficient_is_visible() -> None:
    alg = _alg(mean_l2_coef=0.01)
    _, info = _mstep(alg)
    assert info["mean_l2_coef"] == 0.01
    assert "pretanh_mean_absmean" in info and "pretanh_mean_absmax" in info


def test_a_negative_coefficient_is_rejected() -> None:
    with pytest.raises(ValueError, match="mean_l2_coef"):
        _alg(mean_l2_coef=-1.0)


def test_the_penalty_reaches_the_covariance_through_the_shared_trunk() -> None:
    """It is NOT mean-only in effect, and this is worth knowing before reading T4.

    `kl_var` is a function of `std` alone, so the penalty cannot touch it through the loss. It
    touches it anyway: `mean_head` and `log_std_head` sit on a SHARED backbone
    (`StochasticActor.forward`), so the penalty's gradient flows into the trunk and moves the
    log-std head with it. Measured here at ~0.2% of `kl_var` over three M-step iterations.

    So T4 is not a clean "T2 plus mean control" -- it perturbs the covariance too. Small, but if
    T4 and T2 diverge on `pi_std_max` this is the first thing to check.
    """
    off, on = _alg(mean_l2_coef=0.0), _alg(mean_l2_coef=1.0)
    on.policy.load_state_dict(off.policy.state_dict())
    _, i_off = _mstep(off, seed=3)
    _, i_on = _mstep(on, seed=3)
    assert i_on["kl_var"] != pytest.approx(i_off["kl_var"], rel=1e-9), "expected trunk coupling"
    rel = abs(i_on["kl_var"] - i_off["kl_var"]) / i_off["kl_var"]
    assert rel < 0.05, f"trunk coupling is larger than expected ({rel:.1%}) -- T4 is not isolating"
