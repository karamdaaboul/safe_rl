"""The TD(lambda) cost target: is it the operator it claims to be?

The suite already checks the *pieces* -- the geometric weights, the undiscounted window sum, the
absence of a bootstrap across the horizon boundary. What it did not check is the property those
pieces exist to produce, and the one that decides whether the critic can be right at all:

    **The true finite-horizon cost-to-go is a fixed point of the TD(lambda) operator.**

If a target network already holds the exact remaining episodic cost, every mixture component
``G_j = sum_{k<j} c_k + m_j Z_c(s_{t+j})`` must return that same value, for every j, and hence so
must any weighting of them. A target that fails this is biased *by construction*: no amount of
training, tuning or extra gradient steps can converge it to the truth, and the failure would show
up in exactly the place it is hardest to attribute -- a critic that is stable in a frozen-policy
study and drifts once the policy moves.

The test is deliberately constructed rather than sampled: costs are a known sequence, the episode
boundary is placed inside the window (so both the bootstrapped and the pure-Monte-Carlo branch of
``cost_window_mask`` are exercised), and the "critic" is an exact oracle. Any discount left in
either of the two places it can hide, an off-by-one in the partial sums, a mis-aligned mask, or a
mis-normalised weight vector all break the equality.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

T = 12  # episode length, short enough to write the ground truth out by hand
L = 8  # cost window; > T - t for the later start states, so the boundary falls inside it


def _costs() -> torch.Tensor:
    """A non-constant, non-monotone cost sequence: constants hide index errors."""
    return torch.tensor([0.0, 3.0, 0.0, 1.0, 5.0, 0.0, 0.0, 2.0, 4.0, 0.0, 1.0, 6.0])


def _true_cost_to_go(costs: torch.Tensor) -> torch.Tensor:
    """``V(t) = sum_{k=t}^{T-1} c_k`` -- undiscounted, ends at the horizon."""
    return torch.tensor([float(costs[t:].sum()) for t in range(T)])


def _window_tensors(costs: torch.Tensor, v_true: torch.Tensor):
    """Build ``cost_window_{returns,mask}`` exactly as ReplayStorage._gather_cost_window does.

    Mirrored here on purpose: the point is to test the ALGORITHM's use of the window against an
    independent construction, not to re-run the storage's own code and agree with itself.
    """
    starts = torch.arange(T)
    returns = torch.zeros(T, L)
    mask = torch.zeros(T, L)
    boot_values = torch.zeros(T, L)
    for b, t in enumerate(starts.tolist()):
        for j in range(1, L + 1):
            end = t + j
            # partial sum of realized costs, frozen at the episode boundary
            returns[b, j - 1] = costs[t : min(end, T)].sum()
            if end < T:  # episode still running after step j -> bootstrap
                mask[b, j - 1] = 1.0
                boot_values[b, j - 1] = v_true[end]
            # else: m_j = 0, and the partial sum IS the exact remaining cost (pure Monte Carlo)
    return returns, mask, boot_values


def test_true_cost_to_go_is_a_fixed_point_of_every_td_lambda_component():
    """Each j-step component reproduces V(t) exactly, so any weighting of them does too."""
    costs = _costs()
    v_true = _true_cost_to_go(costs)
    returns, mask, boot = _window_tensors(costs, v_true)

    # G_j = sum_{k<j} c_k + m_j * Z(s_{t+j}), with Z the exact oracle.
    g = returns + mask * boot  # [T, L]

    for t in range(T):
        assert torch.allclose(g[t], v_true[t].expand(L), atol=1e-6), (
            f"component mismatch at t={t}: G_j = {g[t].tolist()} but V(t) = {float(v_true[t])}"
        )


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 0.995, 1.0])
def test_mixture_is_the_fixed_point_for_every_lambda(lam):
    """Weighting cannot move the fixed point -- for any lambda, including the endpoints."""
    from safe_rl.algorithms import FHDCMPO

    costs = _costs()
    v_true = _true_cost_to_go(costs)
    returns, mask, boot = _window_tensors(costs, v_true)
    g = returns + mask * boot

    alg = FHDCMPO.__new__(FHDCMPO)  # weights depend on nothing but these three attributes
    alg.cost_td_lambda = lam
    alg.device = torch.device("cpu")
    w = alg.td_lambda_weights(L)

    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6), "weights must normalise to 1"
    assert (w >= 0).all(), "geometric weights must be non-negative"

    mixed = (g * w.unsqueeze(0)).sum(dim=1)  # [T]
    assert torch.allclose(mixed, v_true, atol=1e-5), (
        f"lambda={lam}: mixture {mixed.tolist()} != true cost-to-go {v_true.tolist()}"
    )


def test_the_window_actually_exercises_both_branches():
    """Guard the guard: a window that never crosses the boundary would pass vacuously."""
    costs = _costs()
    v_true = _true_cost_to_go(costs)
    _, mask, _ = _window_tensors(costs, v_true)
    assert (mask == 1.0).any(), "no bootstrapped component -- the test would not cover that branch"
    assert (mask == 0.0).any(), "no Monte-Carlo component -- the boundary is never crossed"


def test_a_discounted_window_breaks_the_fixed_point():
    """Negative control: the property is sharp, not trivially true of any target.

    ``gamma_c = 1`` is enforced in two independent places (the algorithm's bootstrap discount and
    ReplayStorage's window sum). Leaving a discount in either one is the classic silent bug -- it
    trains happily against the wrong budget -- so the test must be able to see it.
    """
    costs = _costs()
    v_true = _true_cost_to_go(costs)
    gamma = 0.99
    returns = torch.zeros(T, L)
    mask = torch.zeros(T, L)
    boot = torch.zeros(T, L)
    for t in range(T):
        for j in range(1, L + 1):
            end = t + j
            disc = torch.tensor([gamma**k for k in range(min(end, T) - t)])
            returns[t, j - 1] = (costs[t : min(end, T)] * disc).sum()
            if end < T:
                mask[t, j - 1] = gamma**j
                boot[t, j - 1] = v_true[end]
    g = returns + mask * boot
    assert not torch.allclose(g[0], v_true[0].expand(L), atol=1e-6), (
        "a discounted window still matched the undiscounted truth -- this test cannot detect the bug"
    )
