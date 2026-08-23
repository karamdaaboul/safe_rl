"""FH-DCMPO with a diffusion (denoising) M-step: the composition must actually compose.

Almost everything here is inherited, so the risk is not a wrong formula -- it is a silent wiring
failure. The worst case is specific and plausible: an MRO change makes the actor update stop using
the constrained weights, the run trains happily, and the constraint is simply not enforced.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("safe_rl.networks.dime.path_mle")

from safe_rl.algorithms import FHDCMPO, FHDCMPODIME, MPODIME  # noqa: E402
from safe_rl.modules import SafeMPODIMEActorCritic  # noqa: E402

OBS, ACT = 6, 2
DIFF = {
    "diff_steps": 3,
    "init_std": 2.5,
    "friction": 1.0,
    "learn_prior": False,
    "score_model": {"num_layers": 2, "num_hid": 32, "num_time_hid": 8, "num_time_out": 8},
}
CKW = {"n_quantiles": 8, "nonneg": True, "network_kwargs": {"hidden_dims": [16, 16], "activation": "relu"}}


def _policy(**kw):
    return SafeMPODIMEActorCritic(
        num_actor_obs=OBS, num_critic_obs=OBS, num_actions=ACT,
        diffusion=dict(DIFF), critic_kwargs={"hidden_dims": [16, 16]},
        cost_critic_kwargs=dict(CKW), **kw,
    )


def _alg(**kw):
    kw.setdefault("cost_limits", [25.0])
    kw.setdefault("device", "cpu")
    kw.setdefault("sample_action_num", 4)
    kw.setdefault("batch_size", 8)
    kw.setdefault("fh_risk_mode", "mean")
    return FHDCMPODIME(_policy(), **kw)


# -- the composition ---------------------------------------------------------------------------


def test_mro_gives_the_constrained_estep_and_the_diffusion_mstep() -> None:
    """The single property the whole class exists for."""
    assert FHDCMPODIME._estep_weights is FHDCMPO._estep_weights
    assert FHDCMPODIME._update_actor_and_alpha is MPODIME._update_actor_and_alpha
    assert FHDCMPODIME._estep_cost is FHDCMPO._estep_cost
    assert FHDCMPODIME._mstep_dime is MPODIME._mstep_dime
    # ...and the finite-horizon cost semantics survive the mix-in.
    assert FHDCMPODIME._cost_bootstrap_mask is FHDCMPO._cost_bootstrap_mask
    assert FHDCMPODIME._cost_bootstrap_discount is FHDCMPO._cost_bootstrap_discount


def test_constructor_guards_against_an_mro_regression() -> None:
    """A reordering would silently produce UNCONSTRAINED MPO-DIME, which still trains."""
    a = _alg()
    assert type(a)._estep_weights.__qualname__.split(".")[0] == "FHDCMPO"
    assert type(a)._update_actor_and_alpha.__qualname__.split(".")[0] == "MPODIME"


def test_rejects_a_gaussian_actor() -> None:
    from safe_rl.modules import SafeActorCritic
    net = {"hidden_dims": [16, 16], "activation": "relu"}
    gauss = SafeActorCritic(
        num_actor_obs=OBS, num_critic_obs=OBS, num_actions=ACT,
        critic_type="quantile", cost_critic_type="quantile", num_costs=1,
        actor_kwargs={"hidden_dims": [16, 16]},
        critic_kwargs={"n_quantiles": 8, "nonneg": False, "network_kwargs": net},
        cost_critic_kwargs=dict(CKW),
    )
    # MPODIME.__init__ catches this first, which is why FHDCMPODIME carries no second check.
    with pytest.raises(TypeError, match="requires a diffusion policy"):
        FHDCMPODIME(gauss, cost_limits=[25.0], device="cpu", sample_action_num=4, batch_size=8)


# -- the finite-horizon cost contract still holds ------------------------------------------------


def test_cost_units_are_still_episodic_and_undiscounted() -> None:
    r = _alg().cost_units_report()
    assert r["qc_thres"] == pytest.approx(25.0)
    assert r["qc_scale"] == pytest.approx(1.0)
    assert r["cost_gamma"] == pytest.approx(1.0)
    assert r["truncation_mask"] == pytest.approx(0.0)  # no bootstrap across the horizon
    assert r["truncation_mask_reward"] == pytest.approx(1.0)


def test_two_kl_budgets_are_distinct_and_both_reported() -> None:
    """CLAUDE.md rule 2: the E-step eps and the M-step path-space beta are different objects."""
    a = _alg(dual_constraint=0.1, kl_path_constraint=0.05)
    assert a.eps_dual == pytest.approx(0.1)
    assert a.eps_kl_path == pytest.approx(0.05)
    info = a.get_penalty_info()
    assert info["kl_path_budget"] == pytest.approx(0.05)
    assert "alpha_path" in info


# -- the policy module -------------------------------------------------------------------------


def test_safe_dime_policy_pairs_a_diffusion_actor_with_quantile_cost_critics() -> None:
    p = _policy()
    assert p.actor_type == "dime"
    assert p.is_quantile_cost_critic and not p.is_distributional_cost_critic
    assert len(p.cost_critics) == 1 and len(p.cost_critic_targets) == 1
    # Targets frozen, exactly as in SafeActorCritic.
    assert all(not q.requires_grad for q in p.cost_critic_targets[0].parameters())
    qc = p.evaluate_cost_q(torch.randn(5, OBS), torch.rand(5, ACT) * 2 - 1)
    assert qc.shape == (5, 1)
    assert bool((qc >= 0).all()), "nonneg softplus head must make Q_c >= 0 structurally"


def test_safe_dime_policy_rejects_multiple_constraints() -> None:
    with pytest.raises(ValueError, match="single-constraint"):
        _policy(num_costs=2)


def test_sample_with_log_prob_exists_for_the_cost_target() -> None:
    """The cost-critic backup needs next actions from the policy; log_prob is discarded."""
    p = _policy()
    a, _ = p.sample_with_log_prob(torch.randn(4, OBS))
    assert a.shape == (4, ACT)


# -- end to end --------------------------------------------------------------------------------


def test_one_actor_update_runs_through_the_denoiser_with_constrained_weights() -> None:
    a = _alg()
    obs = torch.randn(8, OBS)
    before = [p.detach().clone() for p in a.policy.actor.parameters()]
    loss, _ = a._update_actor_and_alpha(obs, obs)
    assert np.isfinite(loss)
    after = list(a.policy.actor.parameters())
    assert any(not torch.equal(b, x) for b, x in zip(before, after)), "denoiser did not move"
    # The constrained E-step ran: lambda/eta diagnostics are populated by FHDCMPO's weights.
    info = a.get_penalty_info()
    assert np.isfinite(info["eta"]) and info["eta"] > 0
    assert "fh_kappa" in info


def test_no_gradient_reaches_a_cost_critic_through_the_denoiser() -> None:
    """CLAUDE.md rule 3, asserted rather than assumed.

    The safety signal must arrive as scalar weights. If anyone ever backpropagates a cost critic
    through the denoising chain, this fails.
    """
    a = _alg()
    for c in a.policy.cost_critics:
        for p in c.parameters():
            p.grad = None
    a._update_actor_and_alpha(torch.randn(8, OBS), torch.randn(8, OBS))
    assert all(p.grad is None for c in a.policy.cost_critics for p in c.parameters())
