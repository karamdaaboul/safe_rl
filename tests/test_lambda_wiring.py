"""Item 3b: wiring the lambda controller into CVPO.

`lambda_update` defaults to "sgd" and MUST be bit-identical to the previous inline update --
the smoke oracle depends on it. `rescale_by_lambda` defaults off.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

NUM_OBS = 6
NUM_ACT = 2


def _make_alg(**overrides):
    from safe_rl.algorithms import CVPO
    from safe_rl.modules import SafeSACActorCritic

    policy = SafeSACActorCritic(
        num_actor_obs=NUM_OBS, num_critic_obs=NUM_OBS, num_actions=NUM_ACT, num_costs=1,
        actor_kwargs={"hidden_dims": [16, 16]},
        critic_kwargs={"hidden_dims": [16, 16]},
        cost_critic_kwargs={"hidden_dims": [16, 16]},
    )
    kwargs = dict(cost_limits=[25.0], batch_size=16, num_updates_per_step=1,
                  sample_action_num=8, mstep_iteration_num=1, device="cpu")
    kwargs.update(overrides)
    return CVPO(policy, **kwargs)


def test_defaults_are_sgd_and_no_rescale() -> None:
    alg = _make_alg()
    assert alg.lambda_update == "sgd"
    assert alg.rescale_by_lambda is False


def test_sgd_mode_reproduces_the_legacy_inline_update() -> None:
    """Bit-identical to clip(lam + lr*(eqc - thres), 0, lam_max) -- the smoke baseline."""
    alg = _make_alg(lambda_lr=0.03, lambda_max=100.0)
    alg.lam = 1.0
    alg._lambda_ctrl.lam = 1.0
    alg._lambda_ctrl.integral = 1.0

    lam_ref = 1.0
    for eqc in [3.0, 2.6, 2.4, 10.0, 0.0, 1.2]:
        lam_ref = float(np.clip(lam_ref + 0.03 * (eqc - alg.qc_thres), 0.0, 100.0))
        got = alg._update_lambda(eqc)
        assert got == lam_ref, f"{got} != {lam_ref}"
        assert alg.lam == lam_ref


def test_passive_mode_still_pins_lambda_at_zero() -> None:
    alg = _make_alg(cost_critic_passive=True)
    for eqc in [50.0] * 20:
        alg._update_lambda(eqc)
    assert alg.lam == 0.0


def test_pid_mode_is_selectable_and_bounded() -> None:
    alg = _make_alg(lambda_update="pid", lambda_kp=0.1, lambda_lr=0.01, lambda_kd=0.01,
                    lambda_max=4.0)
    traj = [alg._update_lambda(eqc) for eqc in [50.0] * 500 + [0.0] * 50]
    assert all(np.isfinite(t) for t in traj)
    assert max(traj) <= 4.0 and min(traj) >= 0.0
    assert traj[len(traj) // 2] == pytest.approx(4.0), "should saturate under sustained violation"
    assert traj[-1] < 4.0, "anti-windup should let it come back down"


def test_rescale_flag_changes_estep_logits_only_when_on() -> None:
    from safe_rl.common.lambda_controller import rescale_advantage

    q = np.array([[1.0, 2.0], [0.5, -1.0]])
    qc = np.array([[0.2, 0.4], [0.1, 0.9]])
    lam = 3.0
    off = q - lam * qc
    on = rescale_advantage(q, qc, lam)
    assert not np.allclose(off, on)
    # softmax over candidate actions is invariant to the positive rescale up to temperature
    assert np.array_equal(np.argsort(off, axis=0), np.argsort(on, axis=0))


def test_lambda_controller_state_survives_roundtrip() -> None:
    alg = _make_alg(lambda_update="pid", lambda_kp=0.1, lambda_lr=0.02)
    for eqc in [5.0] * 10:
        alg._update_lambda(eqc)
    state = alg._lambda_ctrl.state_dict()
    fresh = _make_alg(lambda_update="pid", lambda_kp=0.1, lambda_lr=0.02)
    fresh._lambda_ctrl.load_state_dict(state)
    assert fresh._lambda_ctrl.lam == alg._lambda_ctrl.lam
    assert fresh._lambda_ctrl.integral == alg._lambda_ctrl.integral
