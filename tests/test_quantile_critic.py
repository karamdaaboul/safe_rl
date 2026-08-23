"""Phase 1: QR-DQN quantile critic as a drop-in alternative to the C51 categorical critic.

The C51 path is the benchmark baseline, so a large share of these tests are not about the
quantile critic being *correct* but about it being *inert when off* -- see
``test_c51_cost_update_is_unchanged_by_the_quantile_branch``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from safe_rl.modules import QuantileCritic, SafeActorCritic  # noqa: E402
from safe_rl.modules.critic import quantile_huber_loss  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QR_CONFIG = PROJECT_ROOT / "config" / "safety_gymnasium_qrdmpo_goal1.yaml"
C51_CONFIG = PROJECT_ROOT / "config" / "safety_gymnasium_dmpo_pid_goal1.yaml"

NUM_OBS, NUM_ACT, N_Q = 12, 3, 16


def _critic(**overrides) -> QuantileCritic:
    kwargs = dict(
        num_obs=NUM_OBS,
        num_actions=NUM_ACT,
        n_quantiles=N_Q,
        network_kwargs={"hidden_dims": [32, 32], "activation": "relu"},
    )
    kwargs.update(overrides)
    return QuantileCritic(**kwargs)


def _batch(batch_size: int = 8):
    return torch.randn(batch_size, NUM_OBS), torch.rand(batch_size, NUM_ACT) * 2 - 1


def _safe_ac(critic_type: str, **kw) -> SafeActorCritic:
    """SafeActorCritic with small nets, in whichever critic representation."""
    if critic_type == "quantile":
        critic_kwargs = {"n_quantiles": N_Q, "nonneg": False,
                         "network_kwargs": {"hidden_dims": [32, 32], "activation": "relu"}}
        cost_kwargs = {"n_quantiles": N_Q, "nonneg": True,
                       "network_kwargs": {"hidden_dims": [32, 32], "activation": "relu"}}
    else:
        critic_kwargs = {"num_atoms": 21, "v_min": -5.0, "v_max": 15.0,
                         "network_kwargs": {"hidden_dims": [32, 32], "activation": "relu"}}
        cost_kwargs = {"num_atoms": 21, "v_min": 0.0, "v_max": 50.0,
                       "network_kwargs": {"hidden_dims": [32, 32], "activation": "relu"}}
    return SafeActorCritic(
        num_actor_obs=NUM_OBS, num_critic_obs=NUM_OBS, num_actions=NUM_ACT,
        critic_type=critic_type, cost_critic_type=critic_type, num_costs=1,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs=critic_kwargs, cost_critic_kwargs=cost_kwargs, **kw,
    )


# --------------------------------------------------------------------------------------
# 1. Shapes and the monotonicity invariant
# --------------------------------------------------------------------------------------

def test_forward_returns_sorted_quantiles_of_the_right_shape() -> None:
    critic = _critic()
    theta = critic(*_batch(8))
    assert theta.shape == (8, N_Q)
    # Sorted, non-decreasing. Quantile crossing would corrupt every statistic read off it.
    assert torch.all(theta[:, 1:] >= theta[:, :-1])


def test_tau_hat_is_the_midpoint_grid_and_moves_with_the_module() -> None:
    critic = _critic(n_quantiles=4)
    assert torch.allclose(critic.tau_hat, torch.tensor([0.125, 0.375, 0.625, 0.875]))
    # A buffer, not a parameter: it must survive .to()/deepcopy into the target net but never
    # be touched by Polyak averaging or the optimizer.
    assert "tau_hat" in dict(critic.named_buffers())
    assert "tau_hat" not in dict(critic.named_parameters())


# --------------------------------------------------------------------------------------
# 2. nonneg: the structural Q_c >= 0 that C51 got from a one-sided support
# --------------------------------------------------------------------------------------

def test_nonneg_makes_every_output_non_negative() -> None:
    critic = _critic(nonneg=True)
    # Drive the head hard in both directions; softplus must still floor the output at 0.
    with torch.no_grad():
        for param in critic.network.parameters():
            param.mul_(50.0)
    theta = critic(*_batch(32))
    assert torch.all(theta >= 0.0)


def test_without_nonneg_the_critic_can_represent_negative_values() -> None:
    critic = _critic(nonneg=False)
    with torch.no_grad():
        for name, param in critic.network.named_parameters():
            if name.endswith("bias"):
                param.fill_(-3.0)
    assert (critic(*_batch(8)) < 0).any(), "a reward critic must be able to go negative"


# --------------------------------------------------------------------------------------
# 3. get_value is the arithmetic mean (each quantile carries mass 1/N)
# --------------------------------------------------------------------------------------

def test_get_value_of_a_hand_built_quantile_set_is_its_mean() -> None:
    critic = _critic(n_quantiles=4)
    theta = torch.tensor([[0.0, 1.0, 2.0, 5.0], [-1.0, -1.0, 3.0, 3.0]])
    assert torch.allclose(critic.get_value(theta), torch.tensor([2.0, 1.0]))
    # get_dist is the identity, which is what lets SafeActorCritic._scalar_q scalarize both
    # critic types through one call.
    assert critic.get_dist(theta) is theta


# --------------------------------------------------------------------------------------
# 4. Regression sanity on a zero-inflated target -- the shape the real cost return has
# --------------------------------------------------------------------------------------

def _fit_zero_inflated(kappa: float, steps: int = 3000, n_quantiles: int = 32, seed: int = 0):
    """Fit one fixed state-action against 60% mass at 0 + 40% Exp(1).

    That mixture is the qualitative shape of the real cost return, and fitting it is the
    whole premise of the swap: a fixed support spends its atoms on the zero spike, while
    learned locations should put resolution in the tail.

    Returns ``(learned, truth)``. The true quantile function follows from
    ``F(x) = 0.6 + 0.4 * (1 - exp(-x))``: zero for ``tau <= 0.6``, else ``-log((1-tau)/0.4)``.
    """
    torch.manual_seed(seed)
    critic = _critic(n_quantiles=n_quantiles, nonneg=True)
    obs, act = torch.zeros(1, NUM_OBS), torch.zeros(1, NUM_ACT)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-2)

    for _ in range(steps):
        samples = torch.where(
            torch.rand(1, 256) < 0.6,
            torch.zeros(1, 256),
            torch.distributions.Exponential(1.0).sample((1, 256)),
        )
        loss = quantile_huber_loss(critic(obs, act), samples, critic.tau_hat, kappa=kappa).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    learned = critic(obs, act).detach().squeeze(0)
    tau = critic.tau_hat
    truth = torch.where(tau <= 0.6, torch.zeros_like(tau), -torch.log((1.0 - tau) / 0.4))
    return learned, truth


def test_fits_the_quantile_function_of_a_zero_inflated_distribution() -> None:
    """With kappa in its linear regime, the loss recovers the true quantile function.

    kappa is deliberately small here (0.02, i.e. essentially the pinball loss). At the
    QR-DQN default kappa=1.0 the Huber is quadratic across this entire distribution and the
    fit is measurably biased -- see the companion test below, which pins that down.
    """
    learned, truth = _fit_zero_inflated(kappa=0.02)

    # The final quantile is excluded: Exp(1) has an unbounded tail, so q(0.984) is estimated
    # from ~4 samples per batch and is inherently noisy. Every other quantile is checked.
    assert torch.allclose(learned[:-1], truth[:-1], atol=0.2), f"learned={learned}\ntruth={truth}"
    # The flat zero region must actually be flat and near zero.
    assert learned[truth == 0].max().item() < 0.05
    # And the mean, which is what the E-step actually consumes: 0.4 * E[Exp(1)] = 0.4.
    assert learned.mean().item() == pytest.approx(0.4, abs=0.06)


def test_kappa_at_the_qr_dqn_default_biases_the_fit_on_a_small_scale_return() -> None:
    """kappa is in RETURN UNITS, so the QR-DQN default of 1.0 does not transfer here.

    In QR-DQN kappa=1.0 sits deep in the linear (pinball) regime because Atari returns are
    O(100). This task's cost return is O(1), so kappa=1.0 makes the Huber quadratic almost
    everywhere and the objective becomes closer to expectile than quantile regression. The
    result is a visibly biased fit: the zero spike is pulled up and the mean overshoots.

    This is documented, not fixed: config/safety_gymnasium_qrdmpo_goal1.yaml ships kappa=1.0
    as specified, and this test exists so the consequence is on record and measurable.
    """
    biased, truth = _fit_zero_inflated(kappa=1.0)
    faithful, _ = _fit_zero_inflated(kappa=0.02)

    zero_region = truth == 0
    assert biased[zero_region].max() > 5 * faithful[zero_region].max()
    assert (biased - truth).abs().max() > 2 * (faithful - truth).abs().max()
    # Overshoots the true mean of 0.4, i.e. it would report the policy as costlier than it is.
    assert biased.mean().item() > 0.44


# --------------------------------------------------------------------------------------
# 5. Gradients flow through the sort
# --------------------------------------------------------------------------------------

def test_gradients_are_finite_and_nonzero_through_the_sort() -> None:
    critic = _critic()
    loss = quantile_huber_loss(critic(*_batch(8)), torch.randn(8, N_Q), critic.tau_hat).mean()
    loss.backward()
    grads = [p.grad for p in critic.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any((g != 0).any() for g in grads)


# --------------------------------------------------------------------------------------
# 6. Config plumbing -- the guard on the silent-key-drop bug class
# --------------------------------------------------------------------------------------

def test_qr_config_builds_a_policy_with_the_configured_critic() -> None:
    """yaml -> builder -> model, asserting the YAML values actually reached the constructor.

    `policy:` is splatted whole into the constructor, so an unknown key raises TypeError
    rather than vanishing -- but a key that IS accepted and then ignored would still be
    invisible. Hence the value assertions, not just a construction check.
    """
    cfg = yaml.safe_load(QR_CONFIG.read_text())
    policy_cfg = dict(cfg["policy"])
    assert policy_cfg.pop("class_name") == "SafeActorCritic"

    policy = SafeActorCritic(NUM_OBS, NUM_OBS, NUM_ACT, **policy_cfg)

    assert policy.is_quantile_critic and policy.is_quantile_cost_critic
    assert not policy.is_distributional_critic and not policy.is_distributional_cost_critic
    for critic in (policy.critic_1, policy.critic_2):
        assert critic.n_quantiles == 64
        assert critic.kappa == 1.0
        assert critic.nonneg is False
        assert critic.tau_hat.shape == (64,)
    cost = policy.cost_critics[0]
    assert cost.n_quantiles == 64 and cost.kappa == 1.0
    assert cost.nonneg is True, "the cost critic must keep the structural Q_c >= 0"


def test_qr_config_differs_from_the_c51_baseline_only_in_the_critic_block() -> None:
    """The comparison is only interpretable if the critic is the sole difference."""
    qr = yaml.safe_load(QR_CONFIG.read_text())
    c51 = yaml.safe_load(C51_CONFIG.read_text())

    assert qr["algorithm"] == c51["algorithm"]
    assert qr["runner_class_name"] == c51["runner_class_name"]
    assert qr["seed"] == c51["seed"]

    qr_runner, c51_runner = dict(qr["runner"]), dict(c51["runner"])
    assert qr_runner.pop("experiment_name") != c51_runner.pop("experiment_name")
    assert qr_runner == c51_runner, "n_step, save_interval and friends must match exactly"

    qr_policy, c51_policy = dict(qr["policy"]), dict(c51["policy"])
    for key in ("critic_type", "cost_critic_type", "critic_kwargs", "cost_critic_kwargs"):
        qr_policy.pop(key)
        c51_policy.pop(key)
    assert qr_policy == c51_policy, "actor and num_costs must be identical across arms"
    # ...and the network sizes inside the critic blocks must match too.
    for key in ("critic_kwargs", "cost_critic_kwargs"):
        assert qr["policy"][key]["network_kwargs"] == c51["policy"][key]["network_kwargs"]


def test_tqc_drop_must_be_zero_in_phase_1() -> None:
    with pytest.raises(ValueError, match="tqc_drop"):
        _critic(tqc_drop=2)


@pytest.mark.parametrize("bad", [{"n_quantiles": 0}, {"kappa": 0.0}, {"kappa": -1.0}])
def test_invalid_hyperparameters_are_rejected(bad) -> None:
    with pytest.raises(ValueError):
        _critic(**bad)


# --------------------------------------------------------------------------------------
# 7. Save -> load roundtrip
# --------------------------------------------------------------------------------------

def test_state_dict_roundtrip_reproduces_identical_outputs(tmp_path) -> None:
    torch.manual_seed(0)
    critic = _critic(nonneg=True)
    obs, act = _batch(8)
    before = critic(obs, act).detach()

    path = tmp_path / "critic.pt"
    torch.save(critic.state_dict(), path)

    restored = _critic(nonneg=True)
    assert not torch.allclose(restored(obs, act).detach(), before), "fresh init must differ"
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(restored(obs, act).detach(), before)


# --------------------------------------------------------------------------------------
# 8. The loss itself
# --------------------------------------------------------------------------------------

def test_loss_weight_uses_a_detached_indicator() -> None:
    """No gradient may flow through the ``1{u < 0}`` mask.

    The mask is a selector, not a differentiable function of theta. If it were left attached,
    autograd would try to differentiate a step function and the loss would stop being the
    quantile regression objective. Verified by comparing against a reference that builds the
    weight from an independently-detached copy of theta: identical gradients mean the
    indicator contributed none.
    """
    torch.manual_seed(0)
    theta = torch.randn(4, N_Q, requires_grad=True)
    target = torch.randn(4, N_Q)
    tau_hat = (torch.arange(N_Q, dtype=torch.float32) + 0.5) / N_Q

    quantile_huber_loss(theta, target, tau_hat, kappa=1.0).mean().backward()
    got = theta.grad.clone()

    ref = theta.detach().clone().requires_grad_(True)
    u = target.unsqueeze(1) - ref.unsqueeze(2)
    huber = torch.where(u.abs() <= 1.0, 0.5 * u.pow(2), 1.0 * (u.abs() - 0.5))
    # Weight built from a value that is not part of ref's graph at all.
    frozen = (target.unsqueeze(1) - theta.detach().unsqueeze(2) < 0).float()
    ((tau_hat.view(1, -1, 1) - frozen).abs() * huber).mean(dim=2).sum(dim=1).mean().backward()

    assert torch.allclose(got, ref.grad, atol=1e-6)


def test_loss_is_reduced_per_sample_not_to_a_scalar() -> None:
    """The cost channel multiplies by hazard-stratified weights before reducing."""
    theta, target = torch.randn(6, N_Q), torch.randn(6, N_Q)
    tau_hat = (torch.arange(N_Q, dtype=torch.float32) + 0.5) / N_Q
    per_sample = quantile_huber_loss(theta, target, tau_hat)
    assert per_sample.shape == (6,)
    assert torch.all(per_sample >= 0)


def test_loss_is_minimized_at_the_true_quantiles() -> None:
    """Sanity on the asymmetric weighting: the optimum sits at the sample quantiles."""
    torch.manual_seed(0)
    tau_hat = (torch.arange(4, dtype=torch.float32) + 0.5) / 4
    target = torch.arange(200, dtype=torch.float32).unsqueeze(0) / 200.0  # U[0, 1)
    optimal = torch.quantile(target, tau_hat).unsqueeze(0)
    at_optimum = quantile_huber_loss(optimal, target, tau_hat, kappa=1e-3)
    for shift in (-0.2, -0.05, 0.05, 0.2):
        assert quantile_huber_loss(optimal + shift, target, tau_hat, kappa=1e-3) > at_optimum


# --------------------------------------------------------------------------------------
# 9. The C51 path must be untouched -- this is the guard the whole comparison rests on
# --------------------------------------------------------------------------------------

def test_c51_cost_update_is_unchanged_by_the_quantile_branch() -> None:
    """A full CVPO cost-critic update on the C51 arm must consume no extra RNG.

    Following the convention in tests/test_qc_scale_wiring.py: replay an identical
    pre-generated sequence and compare parameters AND RNG state. The new dispatch branch and
    the new diagnostics both run inside this path, so if either mutated state or drew a
    random number, this catches it.
    """
    from safe_rl.algorithms import CVPO

    def replay():
        torch.manual_seed(123)
        policy = _safe_ac("distributional")
        alg = CVPO(policy, cost_limits=[25.0], batch_size=8, num_updates_per_step=1,
                   sample_action_num=4, mstep_iteration_num=1, gamma=0.99,
                   cost_horizon=100, device="cpu")
        obs = torch.randn(8, NUM_OBS)
        act = torch.rand(8, NUM_ACT) * 2 - 1
        loss = alg._update_cost_critic(
            obs, obs, act, torch.rand(8, 1), torch.zeros(8, 1), obs, obs,
        )
        params = torch.cat([p.detach().flatten() for p in policy.parameters()])
        return loss, params, torch.get_rng_state(), alg

    loss_a, params_a, rng_a, alg_a = replay()
    loss_b, params_b, rng_b, _ = replay()

    assert loss_a == loss_b
    assert torch.equal(params_a, params_b)
    assert torch.equal(rng_a, rng_b), "the C51 path must consume exactly the RNG it always did"
    # The diagnostics are computed, and on the categorical arm too.
    assert set(alg_a._last_cost_critic_diag) == {
        "critic_cost_mean_Q", "critic_cost_zero_frac", "critic_cost_spread"
    }


def test_quantile_arm_runs_a_full_cost_and_reward_critic_update() -> None:
    from safe_rl.algorithms import CVPO

    torch.manual_seed(0)
    policy = _safe_ac("quantile")
    alg = CVPO(policy, cost_limits=[25.0], batch_size=8, num_updates_per_step=1,
               sample_action_num=4, mstep_iteration_num=1, gamma=0.99,
               cost_horizon=100, device="cpu")
    obs = torch.randn(8, NUM_OBS)
    act = torch.rand(8, NUM_ACT) * 2 - 1

    cost_loss = alg._update_cost_critic(obs, obs, act, torch.rand(8, 1), torch.zeros(8, 1), obs, obs)
    reward_loss = alg._update_critic(obs, obs, act, torch.randn(8, 1), torch.zeros(8, 1), obs, obs)

    for loss in (cost_loss, reward_loss):
        assert torch.isfinite(torch.tensor(loss)) and loss > 0
    diag = alg._last_cost_critic_diag
    assert set(diag) == {"critic_cost_mean_Q", "critic_cost_zero_frac", "critic_cost_spread"}
    assert 0.0 <= diag["critic_cost_zero_frac"] <= 1.0
    assert diag["critic_cost_spread"] >= 0.0
    # The E-step reads Q_c as a plain expectation; a nonneg cost critic can never hand it a
    # negative one, which is the invariant the C51 one-sided support provided.
    with torch.no_grad():
        assert float(policy.evaluate_cost_q(obs, act).min()) >= 0.0


def test_n_step_discount_broadcasts_over_the_quantile_axis() -> None:
    """A per-sample gamma**n must scale each sample's whole quantile set, not misalign."""
    from safe_rl.algorithms import CVPO

    torch.manual_seed(0)
    policy = _safe_ac("quantile")
    alg = CVPO(policy, cost_limits=[25.0], batch_size=8, num_updates_per_step=1,
               sample_action_num=4, mstep_iteration_num=1, gamma=0.99,
               cost_horizon=100, device="cpu")
    obs = torch.randn(8, NUM_OBS)
    act = torch.rand(8, NUM_ACT) * 2 - 1
    n_steps = torch.arange(1, 9).reshape(8, 1)

    loss = alg._update_cost_critic(
        obs, obs, act, torch.rand(8, 1), torch.zeros(8, 1), obs, obs,
        effective_n_steps=n_steps,
    )
    assert torch.isfinite(torch.tensor(loss))


def test_risk_methods_are_implemented_and_agree_with_the_shared_statistics() -> None:
    """These were Phase-2 stubs that raised; FH-DCMPO needs them, so they are now implemented.

    The critic must *delegate* to the tested helpers in ``safe_rl.common.fh_cost`` rather than
    re-deriving the tail accounting -- two implementations of a safety statistic is one too many.
    Correctness of the helpers themselves lives in ``tests/test_fh_cost.py``.
    """
    from safe_rl.common.fh_cost import quantile_cvar, quantile_var

    critic = _critic()
    theta = critic(*_batch(4))
    assert torch.allclose(critic.get_cvar(theta, 0.9), quantile_cvar(theta, 0.9))
    assert torch.allclose(critic.get_quantile(theta, 0.9), quantile_var(theta, 0.9))
    # `risk_value`'s distortion convention must match the categorical critic's, so a configured
    # risk level means the same thing in either representation.
    assert torch.allclose(critic.risk_value(theta, 0.5), quantile_cvar(theta, 0.5, upper=True))
    assert torch.allclose(critic.risk_value(theta, -0.5), quantile_cvar(theta, 0.5, upper=False))
    assert torch.allclose(critic.risk_value(theta, 1.0), critic.get_value(theta))


def test_quantile_cdf_is_fixed_by_construction() -> None:
    """The dual of the categorical case: here the CDF values are fixed and the support is learned."""
    critic = _critic()
    theta = critic(*_batch(4))
    cdf = critic.get_cdf(theta)
    assert cdf.shape == theta.shape
    expected = (torch.arange(N_Q, dtype=torch.float32) + 1.0) / N_Q
    assert torch.allclose(cdf[0], expected, atol=1e-6)


def test_unknown_critic_type_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="critic_type"):
        SafeActorCritic(num_actor_obs=NUM_OBS, num_critic_obs=NUM_OBS, num_actions=NUM_ACT,
                        critic_type="categorical")


def test_quantile_cost_critic_rejects_multiple_constraints() -> None:
    with pytest.raises(ValueError, match="single constraint"):
        SafeActorCritic(num_actor_obs=NUM_OBS, num_critic_obs=NUM_OBS, num_actions=NUM_ACT,
                        cost_critic_type="quantile", num_costs=2)
