"""Item 3: the lambda controller, tested in isolation from RL.

Scripted delta sequences only -- no env, no critic. The behaviours that matter:

  sgd equivalence  : with Ki alone and no anti-windup, PID must reproduce the existing
                     integral update exactly, so switching controller is a no-op by default.
  anti-windup      : the failure this exists to fix. lambda pinned at lam_max integrates a
                     huge reserve; when delta finally goes negative the reserve must be
                     released promptly, not after thousands of steps. Measured in training:
                     lambda sat at the cap for ~13000 of 15000 iterations.
  rescaling        : 1/(1+lambda) must preserve the SIGN and ordering of the combined
                     advantage -- it may scale the objective, never flip it.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from safe_rl.common.lambda_controller import LambdaController, rescale_advantage  # noqa: E402


def _run(ctrl: LambdaController, deltas) -> list[float]:
    return [ctrl.update(float(d)) for d in deltas]


# --- (a) sgd equivalence --------------------------------------------------------


def test_pid_with_only_ki_reproduces_sgd() -> None:
    deltas = [0.5, -0.2, 0.3, 0.9, -1.1, 0.4] * 5
    sgd = LambdaController(mode="sgd", lr=0.03, lam_max=100.0)
    pid = LambdaController(mode="pid", kp=0.0, ki=0.03, kd=0.0, lam_max=100.0, anti_windup=False)
    assert _run(sgd, deltas) == pytest.approx(_run(pid, deltas), abs=1e-12)


def test_sgd_matches_the_reference_clip_formula() -> None:
    """lambda <- clip(lambda + lr * delta, 0, lam_max), the existing CVPO update."""
    ctrl = LambdaController(mode="sgd", lr=0.1, lam_max=5.0)
    lam, expected = 0.0, []
    for d in [1.0, 2.0, -0.5, 100.0, -100.0]:
        lam = float(np.clip(lam + 0.1 * d, 0.0, 5.0))
        expected.append(lam)
    assert _run(ctrl, [1.0, 2.0, -0.5, 100.0, -100.0]) == pytest.approx(expected)


def test_lambda_stays_within_bounds() -> None:
    for mode in ("sgd", "pid"):
        ctrl = LambdaController(mode=mode, lr=1.0, kp=1.0, ki=1.0, kd=1.0, lam_max=4.0)
        out = _run(ctrl, [50.0] * 20 + [-50.0] * 20)
        assert min(out) >= 0.0 and max(out) <= 4.0


# --- (b) anti-windup ------------------------------------------------------------


def test_anti_windup_releases_promptly_after_saturation() -> None:
    """After a long stretch pinned at lam_max, lambda must fall quickly once delta flips."""
    long_violation = [5.0] * 2000
    recovery = [-5.0] * 50

    with_aw = LambdaController(mode="pid", kp=0.0, ki=0.01, kd=0.0, lam_max=4.0, anti_windup=True)
    without = LambdaController(mode="pid", kp=0.0, ki=0.01, kd=0.0, lam_max=4.0, anti_windup=False)
    _run(with_aw, long_violation)
    _run(without, long_violation)
    assert with_aw.lam == pytest.approx(4.0)
    assert without.lam == pytest.approx(4.0)

    aw_traj = _run(with_aw, recovery)
    no_traj = _run(without, recovery)

    def steps_to_release(traj):
        for i, v in enumerate(traj):
            if v < 4.0 - 1e-9:
                return i
        return len(traj)

    n_aw, n_no = steps_to_release(aw_traj), steps_to_release(no_traj)
    assert n_aw <= 2, f"anti-windup should release within a step or two, took {n_aw}"
    assert n_no > 10 * max(n_aw, 1), f"without anti-windup release took {n_no}, expected much longer"


def test_anti_windup_does_not_change_unsaturated_behaviour() -> None:
    """While lambda is inside the bounds, anti-windup must be inert."""
    deltas = [0.1, -0.05, 0.2, -0.1] * 10
    a = LambdaController(mode="pid", kp=0.1, ki=0.01, kd=0.0, lam_max=100.0, anti_windup=True)
    b = LambdaController(mode="pid", kp=0.1, ki=0.01, kd=0.0, lam_max=100.0, anti_windup=False)
    assert _run(a, deltas) == pytest.approx(_run(b, deltas))


def test_pid_responds_to_derivative() -> None:
    ctrl = LambdaController(mode="pid", kp=0.0, ki=0.0, kd=1.0, lam_max=100.0)
    out = _run(ctrl, [0.0, 1.0, 2.0, 2.0])
    assert out[1] > 0.0            # rising delta -> positive derivative term
    assert out[3] == pytest.approx(0.0, abs=1e-12)   # flat delta -> no derivative term


def test_controller_is_finite_under_extreme_input() -> None:
    ctrl = LambdaController(mode="pid", kp=1e3, ki=1e3, kd=1e3, lam_max=100.0)
    out = _run(ctrl, [1e6, -1e6, 1e6, 0.0, np.nan if False else 1e-9])
    assert all(np.isfinite(v) for v in out)


def test_reset_clears_state() -> None:
    ctrl = LambdaController(mode="pid", kp=0.1, ki=0.1, kd=0.1, lam_max=10.0)
    _run(ctrl, [1.0] * 50)
    assert ctrl.lam > 0
    ctrl.reset()
    assert ctrl.lam == 0.0 and ctrl.integral == 0.0


# --- (c) advantage rescaling ----------------------------------------------------


def test_rescale_preserves_sign_and_order() -> None:
    q_r = np.array([1.0, 0.5, -0.3, 2.0])
    q_c = np.array([0.2, 1.0, 0.1, 0.5])
    for lam in (0.0, 1.0, 4.0, 100.0):
        raw = q_r - lam * q_c
        scaled = rescale_advantage(q_r, q_c, lam)
        assert np.all(np.sign(scaled) == np.sign(raw)), f"sign flipped at lambda={lam}"
        assert np.array_equal(np.argsort(scaled), np.argsort(raw)), f"order changed at lambda={lam}"


def test_rescale_is_identity_at_zero_lambda() -> None:
    q_r = np.array([1.0, -2.0, 0.5])
    q_c = np.array([0.3, 0.4, 0.5])
    assert rescale_advantage(q_r, q_c, 0.0) == pytest.approx(q_r)


def test_rescale_bounds_the_combination() -> None:
    """The point of 1/(1+lambda): the combined advantage must not grow without bound in lambda."""
    q_r, q_c = np.array([1.0]), np.array([1.0])
    mags = [abs(float(rescale_advantage(q_r, q_c, lam)[0])) for lam in (0.0, 1.0, 10.0, 100.0)]
    assert max(mags) <= 1.0 + 1e-9
    unscaled = [abs(1.0 - lam * 1.0) for lam in (0.0, 1.0, 10.0, 100.0)]
    assert max(unscaled) > 50.0, "unscaled combination does grow with lambda (contrast)"
