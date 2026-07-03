"""Tests for the CLF-RL reward-shaping components."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")


# -- CLF matrix ----------------------------------------------------------------


def test_build_clf_P_is_spd_and_V_nonnegative() -> None:
    from safe_rl.utils.clf import build_clf_P

    n_out = 4
    P = build_clf_P(n_out, q_pos=1.0, q_vel=0.1, r=1.0, device="cpu")
    assert P.shape == (2 * n_out, 2 * n_out)
    # Symmetric.
    assert torch.allclose(P, P.T, atol=1e-5)
    # Positive definite -> V(eta) = eta^T P eta >= 0 for random eta, > 0 for nonzero.
    assert torch.linalg.eigvalsh(P).min() > 0
    eta = torch.randn(100, 2 * n_out)
    V = torch.einsum("bi,ij,bj->b", eta, P, eta)
    assert (V >= 0).all()
    assert V.max() > 0


def test_clf_sigma_positive() -> None:
    from safe_rl.utils.clf import build_clf_P, clf_sigma

    P = build_clf_P(4, device="cpu")
    sigma_v, sigma_vdot = clf_sigma(P, eta_max=1.0, eta_dot_max=5.0, lam=1.0)
    assert sigma_v > 0
    assert sigma_vdot > sigma_v  # decay constant includes the tracking term plus more


# -- H-LIP reference -----------------------------------------------------------


def test_lip_velocity_profile_mean_matches_command() -> None:
    from safe_rl.envs.reference.hlip import lip_velocity_profile

    vd = torch.tensor([0.5, 1.0, 0.0])
    T = 0.4
    # Average of v(t) over a step should equal vd.
    phases = torch.linspace(0, 1, 2001)[:-1]
    means = []
    for v in vd:
        vs, _ = lip_velocity_profile(v.repeat(phases.shape[0]), phases, T, com_height=0.74)
        means.append(vs.mean())
    means = torch.stack(means)
    assert torch.allclose(means, vd, atol=1e-2)


def test_bezier5_endpoints_and_deriv() -> None:
    from safe_rl.envs.reference.hlip import bezier5, bezier5_deriv

    ctrl = torch.tensor([[0.0, 0.0, 1.5, 1.5, 0.0, 0.0]], dtype=torch.float64)
    s0 = torch.tensor([0.0], dtype=torch.float64)
    s1 = torch.tensor([1.0], dtype=torch.float64)
    assert torch.allclose(bezier5(ctrl, s0), torch.tensor([0.0], dtype=torch.float64), atol=1e-9)
    assert torch.allclose(bezier5(ctrl, s1), torch.tensor([0.0], dtype=torch.float64), atol=1e-9)
    # Numerical derivative check at mid-curve (float64 for finite-difference precision).
    s = torch.tensor([0.37], dtype=torch.float64)
    h = 1e-5
    num = (bezier5(ctrl, s + h) - bezier5(ctrl, s - h)) / (2 * h)
    ana = bezier5_deriv(ctrl, s)
    assert torch.allclose(num, ana, atol=1e-3)


def test_hlip_reference_shapes() -> None:
    from safe_rl.envs.reference.hlip import HLIPReference

    ref = HLIPReference(num_envs=8, device="cpu", outputs=("lin_vel_x", "lin_vel_y", "ang_vel_z"))
    cmd = torch.zeros(8, 3)
    cmd[:, 0] = 0.5
    phase = torch.rand(8)
    yd, yd_dot = ref.compute(cmd, phase)
    assert yd.shape == (8, 3)
    assert yd_dot.shape == (8, 3)
    assert ref.n_out == 3


def test_hlip_unknown_output_raises() -> None:
    from safe_rl.envs.reference.hlip import HLIPReference

    with pytest.raises(ValueError):
        HLIPReference(num_envs=2, device="cpu", outputs=("not_a_real_output",))


# -- Wrapper -------------------------------------------------------------------


class _FakeData:
    def __init__(self, num_envs: int) -> None:
        self.root_link_lin_vel_b = torch.zeros(num_envs, 3)
        self.root_link_ang_vel_b = torch.zeros(num_envs, 3)


class _FakeRobot:
    def __init__(self, num_envs: int) -> None:
        self.data = _FakeData(num_envs)


class _FakeScene:
    def __init__(self, num_envs: int) -> None:
        self._robot = _FakeRobot(num_envs)

    def __getitem__(self, name: str) -> _FakeRobot:
        return self._robot


class _FakeCommandManager:
    def __init__(self, num_envs: int) -> None:
        self._cmd = torch.zeros(num_envs, 3)
        self._cmd[:, 0] = 0.5

    def get_command(self, name: str) -> torch.Tensor:
        return self._cmd


class _FakeMjlabEnv:
    """Minimal stand-in for an mjlab VecEnv satisfying the wrapper's needs."""

    def __init__(self, num_envs: int = 4) -> None:
        self.num_envs = num_envs
        self.num_actions = 12
        self.device = torch.device("cpu")
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.cfg = {}
        self.step_dt = 0.02
        self.scene = _FakeScene(num_envs)
        self.command_manager = _FakeCommandManager(num_envs)
        self._step = 0

    @property
    def unwrapped(self):
        return self

    def get_observations(self):
        return torch.zeros(self.num_envs, 5), {"observations": {}}

    def reset(self):
        self.episode_length_buf.zero_()
        return torch.zeros(self.num_envs, 5), {"observations": {}}

    def step(self, actions):
        self._step += 1
        self.episode_length_buf += 1
        # Make the robot move a bit so finite differences are nonzero.
        self.scene["robot"].data.root_link_lin_vel_b[:, 0] = 0.4 + 0.01 * self._step
        rewards = torch.ones(self.num_envs)
        dones = torch.zeros(self.num_envs)
        if self._step == 3:
            dones[0] = 1.0  # trigger a done on env 0
        return torch.zeros(self.num_envs, 5), rewards, dones, {"observations": {}}

    def close(self):
        pass


def test_clf_wrapper_step_produces_finite_shaped_rewards() -> None:
    from safe_rl.envs.clf_reward_wrapper import CLFRewardWrapper

    env = _FakeMjlabEnv(num_envs=4)
    wrapped = CLFRewardWrapper(env, {"outputs": ["lin_vel_x", "lin_vel_y", "ang_vel_z"], "w_v": 10.0, "w_vdot": 2.0})

    actions = torch.zeros(4, 12)
    for _ in range(5):
        obs, reward, dones, extras = wrapped.step(actions)
        assert reward.shape == (4,)
        assert torch.isfinite(reward).all()
        assert "clf/V" in extras["log"]
        assert "clf/reward_tracking" in extras["log"]
        assert "clf/reward_decay" in extras["log"]

    # CLF tracking reward is bounded by w_v (exp(-V/sigma) in [0, 1]).
    assert (extras["log"]["clf/reward_tracking"] <= 10.0 + 1e-5)


def test_clf_wrapper_masks_decay_on_done() -> None:
    from safe_rl.envs.clf_reward_wrapper import CLFRewardWrapper

    env = _FakeMjlabEnv(num_envs=4)
    wrapped = CLFRewardWrapper(env, {"outputs": ["lin_vel_x"]})

    actions = torch.zeros(4, 12)
    last_extras = None
    for _ in range(3):  # env 0 gets done on step 3
        _, _, dones, last_extras = wrapped.step(actions)
    # After a done step, prev_V for that env is zeroed so the next decay is masked.
    assert wrapped._prev_V[0].item() == 0.0


def test_clf_wrapper_reset_clears_memory() -> None:
    from safe_rl.envs.clf_reward_wrapper import CLFRewardWrapper

    env = _FakeMjlabEnv(num_envs=4)
    wrapped = CLFRewardWrapper(env, {"outputs": ["lin_vel_x"]})
    wrapped.step(torch.zeros(4, 12))
    wrapped.reset()
    assert wrapped._prev_V.abs().sum().item() == 0.0
    assert wrapped._prev_y.abs().sum().item() == 0.0
    assert wrapped._initialized is False
