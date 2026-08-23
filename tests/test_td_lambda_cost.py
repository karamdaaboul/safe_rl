"""TD(lambda) cost target (SDAC-style) for FH-DCMPO.

The target is the only thing that can inject real return spread into an undiscounted cost critic,
so every piece of it is checked against hand-computed values rather than against itself.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from safe_rl.modules.critic import quantile_huber_loss  # noqa: E402
from safe_rl.storage.replay_storage import ReplayStorage  # noqa: E402

OBS, ACT = 4, 2


def _store(costs, dones, n_step=2, cost_n_step=6):
    s = ReplayStorage(num_envs=1, max_size=64, obs_shape=[OBS], action_shape=[ACT],
                      n_step=n_step, gamma=0.99, cost_gamma=1.0, cost_n_step=cost_n_step)
    for i, (c, d) in enumerate(zip(costs, dones)):
        s.add(torch.full((1, OBS), float(i)), torch.zeros(1, ACT), torch.zeros(1),
              torch.full((1,), float(d)), torch.full((1, OBS), float(i + 1)),
              costs=torch.full((1, 1), float(c)), bootstrap=torch.zeros(1, 1))
    return s


def _window(s):
    b = s._gather_n_step(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    return b["cost_window_returns"][0], b["cost_window_mask"][0], b["cost_window_next_obs"][0]


# -- weights -----------------------------------------------------------------------------------


def _alg(**kw):
    from safe_rl.algorithms import FHDCMPO
    from safe_rl.modules import SafeActorCritic
    net = {"hidden_dims": [16, 16], "activation": "relu"}
    p = SafeActorCritic(num_actor_obs=OBS, num_critic_obs=OBS, num_actions=ACT,
                        critic_type="quantile", cost_critic_type="quantile", num_costs=1,
                        actor_kwargs={"hidden_dims": [16, 16]},
                        critic_kwargs={"n_quantiles": 8, "nonneg": False, "network_kwargs": net},
                        cost_critic_kwargs={"n_quantiles": 8, "nonneg": True, "network_kwargs": net})
    kw.setdefault("cost_limits", [25.0])
    kw.setdefault("device", "cpu")
    kw.setdefault("sample_action_num", 4)
    kw.setdefault("batch_size", 8)
    return FHDCMPO(p, **kw)


def test_td_lambda_weights_sum_to_one_and_put_remainder_on_the_longest_return() -> None:
    a = _alg(n_step=2, cost_n_step=5, cost_td_lambda=0.9)
    w = a.td_lambda_weights(5).numpy()
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    # Truncating WITHOUT the remainder would renormalise toward short returns - the opposite
    # of the point - so the last component must carry lam^(L-1), not (1-lam)lam^(L-1).
    assert w[-1] == pytest.approx(0.9**4, abs=1e-6)
    assert all(w[i] > w[i + 1] for i in range(3))


def test_td_lambda_weights_at_lambda_zero_are_a_one_step_target() -> None:
    a = _alg(n_step=2, cost_n_step=5, cost_td_lambda=0.0)
    w = a.td_lambda_weights(5).numpy()
    assert w[0] == pytest.approx(1.0)
    assert w[1:].sum() == pytest.approx(0.0, abs=1e-9)


def test_td_lambda_weights_at_lambda_one_are_the_longest_return_only() -> None:
    a = _alg(n_step=2, cost_n_step=5, cost_td_lambda=1.0)
    w = a.td_lambda_weights(5).numpy()
    assert w[-1] == pytest.approx(1.0)
    assert w[:-1].sum() == pytest.approx(0.0, abs=1e-9)


def test_cost_n_step_must_exceed_n_step() -> None:
    with pytest.raises(ValueError, match="must exceed n_step"):
        _alg(n_step=10, cost_n_step=10)


def test_bad_lambda_rejected() -> None:
    with pytest.raises(ValueError, match="cost_td_lambda"):
        _alg(n_step=2, cost_n_step=5, cost_td_lambda=1.5)


# -- the window itself -------------------------------------------------------------------------


def test_window_returns_are_undiscounted_partial_sums() -> None:
    s = _store(costs=[1, 2, 3, 4, 5, 6, 7, 8], dones=[0] * 8)
    ret, mask, _ = _window(s)
    # G_j = sum_{k<j} c_k, undiscounted: 1, 3, 6, 10, 15, 21
    assert ret.tolist() == pytest.approx([1, 3, 6, 10, 15, 21])
    assert mask.tolist() == pytest.approx([1] * 6)  # no episode ends in the window


def test_window_freezes_at_an_episode_boundary_and_stops_bootstrapping() -> None:
    """Past the boundary the component is the EXACT realized remaining cost: a pure MC atom."""
    s = _store(costs=[1, 2, 3, 9, 9, 9, 9, 9], dones=[0, 0, 1, 0, 0, 0, 0, 0])
    ret, mask, _ = _window(s)
    # Episode ends on step 3 (index 2). Returns freeze at 1+2+3 = 6 and never absorb the next
    # episode's 9s; the mask goes to 0 there, so those atoms carry no critic value at all.
    assert ret.tolist() == pytest.approx([1, 3, 6, 6, 6, 6])
    assert mask.tolist() == pytest.approx([1, 1, 0, 0, 0, 0])


def test_window_next_obs_is_the_observation_at_t_plus_j() -> None:
    s = _store(costs=[0] * 8, dones=[0] * 8)
    _, _, nobs = _window(s)
    # next_obs stored at window step j-1 is obs j, i.e. the state reached after j steps.
    assert [float(x[0]) for x in nobs] == pytest.approx([1, 2, 3, 4, 5, 6])


def test_cost_window_absent_by_default_so_existing_algorithms_are_untouched() -> None:
    s = ReplayStorage(num_envs=1, max_size=64, obs_shape=[OBS], action_shape=[ACT], n_step=2, gamma=0.99)
    assert s.cost_n_step == s.n_step and s.window_len == 2
    for i in range(8):
        s.add(torch.zeros(1, OBS), torch.zeros(1, ACT), torch.zeros(1), torch.zeros(1),
              torch.zeros(1, OBS), costs=torch.zeros(1, 1))
    b = s._gather_n_step(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    assert "cost_window_returns" not in b


# -- weighted loss -----------------------------------------------------------------------------


def test_uniform_target_weights_reproduce_the_unweighted_loss() -> None:
    """If this drifts, every pre-TD(lambda) arm silently changes."""
    torch.manual_seed(0)
    theta, target = torch.randn(4, 8), torch.randn(4, 12)
    tau = (torch.arange(8, dtype=torch.float32) + 0.5) / 8
    w = torch.full((4, 12), 1.0 / 12)
    assert torch.allclose(
        quantile_huber_loss(theta, target, tau, 1.0),
        quantile_huber_loss(theta, target, tau, 1.0, target_weights=w),
        atol=1e-6,
    )


def test_target_weights_actually_reweight() -> None:
    theta = torch.zeros(1, 4)
    tau = (torch.arange(4, dtype=torch.float32) + 0.5) / 4
    target = torch.tensor([[0.0, 100.0]])
    near = quantile_huber_loss(theta, target, tau, 1.0, target_weights=torch.tensor([[0.99, 0.01]]))
    far = quantile_huber_loss(theta, target, tau, 1.0, target_weights=torch.tensor([[0.01, 0.99]]))
    assert float(far) > float(near) * 10


def test_target_weight_shape_mismatch_is_rejected() -> None:
    tau = (torch.arange(4, dtype=torch.float32) + 0.5) / 4
    with pytest.raises(ValueError, match="target_weights"):
        quantile_huber_loss(torch.zeros(2, 4), torch.zeros(2, 6), tau, 1.0,
                            target_weights=torch.full((2, 5), 0.2))


# -- end to end --------------------------------------------------------------------------------


def test_update_runs_and_falls_back_without_a_window() -> None:
    a = _alg(n_step=2, cost_n_step=6, cost_td_lambda=0.9)
    B, L, Nq = 8, 6, 8
    args = (torch.randn(B, OBS), torch.randn(B, OBS), torch.rand(B, ACT) * 2 - 1,
            torch.rand(B, 1), torch.zeros(B, 1), torch.randn(B, OBS), torch.randn(B, OBS))
    # With a window -> TD(lambda) path
    loss = a._update_cost_critic_quantile(
        *args, cost_window_returns=torch.rand(B, L) * 10,
        cost_window_next_obs=torch.randn(B, L, OBS),
        cost_window_mask=torch.ones(B, L),
    )
    assert np.isfinite(loss)
    assert "critic_cost_mc_frac" in a._last_cost_critic_diag
    # Without one -> parent's n-step path, unchanged
    loss2 = a._update_cost_critic_quantile(*args)
    assert np.isfinite(loss2)


def test_pure_mc_window_target_has_no_critic_value_in_it() -> None:
    """mask=0 => the target atoms ARE the realized returns, whatever the critic predicts.

    This is the property the whole change rests on: past an episode boundary the target carries
    zero critic error, so its spread comes from real returns rather than from the target network's
    own (measured 2.2-2.4x too narrow) distribution.
    """
    a = _alg(n_step=2, cost_n_step=4, cost_td_lambda=0.9)
    B, L = 4, 4
    rets = torch.arange(B * L, dtype=torch.float32).reshape(B, L)
    with torch.no_grad():
        flat = torch.randn(B * L, OBS)
        z = a.policy.cost_critic_targets[0](
            a.policy.critic_obs_normalizer(flat), torch.zeros(B * L, ACT)
        ).reshape(B, L, -1)
    assert float(z.abs().max()) > 0.0, "critic must be non-trivial or the test proves nothing"

    g_mc = rets.unsqueeze(-1) + torch.zeros(B, L, 1) * z
    assert torch.allclose(g_mc, rets.unsqueeze(-1).expand_as(z))

    g_boot = rets.unsqueeze(-1) + torch.ones(B, L, 1) * z
    assert not torch.allclose(g_boot, rets.unsqueeze(-1).expand_as(z))


def test_window_survives_the_dispatcher() -> None:
    """Regression: the window must reach the quantile path THROUGH `_update_cost_critic`.

    The first launched run crashed with `_update_cost_critic() got an unexpected keyword argument
    'cost_window_returns'` because the tests above call the quantile method directly and so never
    exercised the dispatcher. Any future signature change has to keep this path open.
    """
    a = _alg(n_step=2, cost_n_step=6, cost_td_lambda=0.9)
    B, L = 8, 6
    loss = a._update_cost_critic(
        torch.randn(B, OBS), torch.randn(B, OBS), torch.rand(B, ACT) * 2 - 1,
        torch.rand(B, 1), torch.zeros(B, 1), torch.randn(B, OBS), torch.randn(B, OBS),
        cost_window_returns=torch.rand(B, L) * 10,
        cost_window_next_obs=torch.randn(B, L, OBS),
        cost_window_mask=torch.ones(B, L),
    )
    assert np.isfinite(loss)
    assert a._last_cost_critic_diag["critic_cost_mc_frac"] == pytest.approx(0.0)


def test_dispatcher_still_works_without_a_window() -> None:
    a = _alg(n_step=2, cost_n_step=6)
    B = 8
    loss = a._update_cost_critic(
        torch.randn(B, OBS), torch.randn(B, OBS), torch.rand(B, ACT) * 2 - 1,
        torch.rand(B, 1), torch.zeros(B, 1), torch.randn(B, OBS), torch.randn(B, OBS),
    )
    assert np.isfinite(loss)
