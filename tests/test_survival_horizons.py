"""Stochastic decision horizons: the continuation model and the survival-shaped n-step return.

The survival shaping rewrites the reward channel's Bellman target, which is the single most
load-bearing quantity in every off-policy arm in this repo. Every failure mode it can have is
silent -- a run trains happily and answers a different question:

* if ``alpha`` comes out 1 everywhere the arm is unshaped MPO wearing another name;
* if the shaped path leaks into the unshaped one, every existing arm's results move;
* if ``survival_discount`` is ``[B]`` instead of ``[B, 1]`` it broadcasts to ``[B, B]`` against the
  standard critic's targets rather than failing;
* if the gamma factors are double-counted the effective horizon is wrong by a power of gamma.

The anchor test is :func:`test_lambda_zero_reproduces_the_scalar_gamma_return_exactly` -- at
``lam = 0`` the whole new path must reduce to the old one bit for bit.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from safe_rl.common.continuation import (  # noqa: E402
    cat_continuation,
    continuation_scale_at,
    exponential_continuation,
)
from safe_rl.storage.replay_storage import ReplayStorage  # noqa: E402

GAMMA = 0.99
N = 5


def _filled_storage(n_step: int = N, seed: int = 0, done_p: float = 0.08, envs: int = 2, steps: int = 40):
    torch.manual_seed(seed)
    st = ReplayStorage(num_envs=envs, max_size=envs * steps, obs_shape=[3], action_shape=[2],
                       device="cpu", n_step=n_step, gamma=GAMMA)
    for _ in range(steps):
        st.add(obs=torch.randn(envs, 3), action=torch.randn(envs, 2), reward=torch.randn(envs, 1),
               done=(torch.rand(envs, 1) < done_p).float(), next_obs=torch.randn(envs, 3),
               costs=torch.rand(envs, 1) * (torch.rand(envs, 1) < 0.3).float())
    return st


# -- the continuation model --------------------------------------------------------------------
def test_exponential_continuation_is_one_at_zero_scale():
    """``lam = 0`` must be an exact off switch, not merely close to one."""
    costs = torch.rand(64, 2) * 5.0
    assert torch.equal(exponential_continuation(costs, 0.0), torch.ones(64))


def test_exponential_continuation_is_monotone_and_bounded():
    costs = torch.linspace(0.0, 10.0, 50).unsqueeze(-1)
    alpha = exponential_continuation(costs, 0.4)
    assert ((alpha >= 0.0) & (alpha <= 1.0)).all()
    assert (alpha[1:] <= alpha[:-1] + 1e-9).all(), "alpha must not increase with violation magnitude"
    assert alpha[0] == pytest.approx(1.0), "zero cost must not attenuate"


def test_exponential_continuation_sums_over_constraints():
    """Multi-constraint costs aggregate additively inside the exponent."""
    c = torch.tensor([[1.0, 2.0]])
    assert exponential_continuation(c, 0.5).item() == pytest.approx(float(np.exp(-0.5 * 3.0)))


def test_negative_scale_is_rejected():
    """A negative scale would give alpha > 1, i.e. a discount ABOVE gamma, breaking contraction."""
    with pytest.raises(ValueError):
        exponential_continuation(torch.rand(4, 1), -0.1)


def test_cat_continuation_saturates_and_takes_the_worst_constraint():
    c = torch.tensor([[0.0, 10.0]])  # second constraint far past its scale
    alpha = cat_continuation(c, c_max=2.0, p_max=1.0)
    assert alpha.item() == pytest.approx(0.0), "saturated constraint should drive alpha to 0"
    assert cat_continuation(torch.zeros(1, 2), c_max=2.0).item() == pytest.approx(1.0)


def test_schedule_endpoints_and_monotonicity():
    vals = [continuation_scale_at(u, 0.8, warmup=10, ramp=20) for u in range(45)]
    assert vals[0] == 0.0 and vals[9] == 0.0, "held at lam_init through warmup"
    assert vals[30] == pytest.approx(0.8) and vals[44] == pytest.approx(0.8), "reaches lam_final"
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), "schedule must be non-decreasing"
    assert continuation_scale_at(5, 0.8, warmup=0, ramp=0) == pytest.approx(0.8), "0/0 = fixed scale"


# -- the shaped n-step return ------------------------------------------------------------------
def test_off_by_default_and_byte_identical():
    """``survival_lambda = None`` must leave the existing path untouched, keys included."""
    st = _filled_storage()
    t_idx, e_idx = torch.arange(12), torch.zeros(12, dtype=torch.long)
    a = st._gather_n_step(t_idx, e_idx)
    st.survival_lambda = None
    b = st._gather_n_step(t_idx, e_idx)
    assert set(a) == set(b) and "survival_discount" not in a
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_lambda_zero_reproduces_the_scalar_gamma_return_exactly():
    """THE anchor. At lam = 0 the shaped return and the shaped discount must equal the originals.

    If this drifts, every arm trained before this feature is no longer comparable with every arm
    trained after it, and nothing else in the suite would notice.
    """
    st = _filled_storage()
    t_idx, e_idx = torch.arange(12), torch.zeros(12, dtype=torch.long)
    base = st._gather_n_step(t_idx, e_idx)
    st.survival_lambda = 0.0
    shaped = st._gather_n_step(t_idx, e_idx)

    assert torch.allclose(base["rewards"], shaped["rewards"], atol=1e-6)
    expected = GAMMA ** base["effective_n_steps"].float()
    assert torch.allclose(shaped["survival_discount"], expected, atol=1e-6), (
        "at alpha == 1 the survival factor IS gamma**n_eff; a mismatch means the gamma factors are "
        "double-counted or missing"
    )
    assert torch.equal(base["costs"], shaped["costs"]), "shaping must not touch the cost channel"


def test_survival_discount_is_column_shaped():
    """``[B, 1]``: a ``[B]`` discount broadcasts to ``[B, B]`` in the standard-critic target."""
    st = _filled_storage()
    st.survival_lambda = 0.5
    batch = st._gather_n_step(torch.arange(9), torch.zeros(9, dtype=torch.long))
    assert batch["survival_discount"].shape == (9, 1)
    # the shape that actually matters downstream: rewards + discount * mask * q  stays [B, 1]
    q = torch.randn(9, 1)
    assert (batch["rewards"] + batch["survival_discount"] * q).shape == (9, 1)


def test_matches_an_explicit_python_reference_including_a_done_inside_the_window():
    """Hand-rolled loop over a constructed window, with the episode ending mid-window."""
    n, lam = 4, 0.7
    st = ReplayStorage(num_envs=1, max_size=16, obs_shape=[2], action_shape=[1], device="cpu",
                       n_step=n, gamma=GAMMA)
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    costs = [0.0, 0.5, 0.0, 2.0, 1.0, 0.0]
    dones = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # episode ends at index 2, inside the window from t=0
    for r, c, d in zip(rewards, costs, dones):
        st.add(obs=torch.zeros(1, 2), action=torch.zeros(1, 1), reward=torch.tensor([[r]]),
               done=torch.tensor([[d]]), next_obs=torch.zeros(1, 2), costs=torch.tensor([[c]]))
    st.survival_lambda = lam
    batch = st._gather_n_step(torch.tensor([0]), torch.tensor([0]))

    # reference: alive until the step AFTER the first done; u_k = prod_{j<k} gamma*alpha_j
    ref, u, alive = 0.0, 1.0, 1.0
    n_eff = None
    for k in range(n):
        alpha_k = float(np.exp(-lam * costs[k]))
        ref += alive * u * alpha_k * rewards[k]
        if dones[k] > 0 and n_eff is None:
            n_eff = k + 1
        if dones[k] > 0:
            alive = 0.0
        u *= GAMMA * alpha_k
    assert batch["rewards"].item() == pytest.approx(ref, abs=1e-6)
    assert int(batch["effective_n_steps"].item()) == n_eff == 3

    u_eff = 1.0
    for k in range(n_eff):
        u_eff *= GAMMA * float(np.exp(-lam * costs[k]))
    assert batch["survival_discount"].item() == pytest.approx(u_eff, abs=1e-6)


def test_larger_scale_attenuates_more():
    st = _filled_storage()
    t_idx, e_idx = torch.arange(16), torch.zeros(16, dtype=torch.long)
    st.survival_lambda = 0.1
    weak = st._gather_n_step(t_idx, e_idx)["survival_discount"]
    st.survival_lambda = 1.0
    strong = st._gather_n_step(t_idx, e_idx)["survival_discount"]
    assert (strong <= weak + 1e-9).all()
    assert float(strong.mean()) < float(weak.mean()), "a larger scale must shorten the horizon"


def test_shaping_without_stored_costs_fails_loudly():
    """Silently shaping with alpha == 1 would be an unshaped run reported as a shaped one."""
    st = ReplayStorage(num_envs=1, max_size=8, obs_shape=[2], action_shape=[1], device="cpu",
                       n_step=2, gamma=GAMMA)
    for _ in range(8):
        st.add(obs=torch.zeros(1, 2), action=torch.zeros(1, 1), reward=torch.ones(1, 1),
               done=torch.zeros(1, 1), next_obs=torch.zeros(1, 2))
    st.survival_lambda = 0.5
    with pytest.raises(RuntimeError, match="per-step costs"):
        st._gather_n_step(torch.tensor([0]), torch.tensor([0]))
