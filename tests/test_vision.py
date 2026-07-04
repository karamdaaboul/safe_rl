from __future__ import annotations

import os

import pytest

from tests.conftest import has_module

torch = pytest.importorskip("torch")


class _StubBackbone(torch.nn.Module):
    """Deterministic stand-in for a pretrained backbone (no download)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # One value per channel-mean so the output depends on the input.
        return x.mean(dim=(2, 3)).repeat(1, 171)[:, :512]


class _StubVisionVecEnv:
    """Minimal dict-obs VecEnv double matching SafetyGymnasiumVecEnv's contract."""

    def __init__(self, num_envs: int = 3) -> None:
        self.device = torch.device("cpu")
        self.num_envs = num_envs
        self.num_actions = 2
        self.max_episode_length = 10
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.cfg = {}
        self.obs_key_slices = {
            "accelerometer": slice(0, 3),
            "gyro": slice(3, 6),
            "goal_lidar": slice(6, 22),
        }
        self.closed = False

    def _obs(self):
        state = torch.arange(self.num_envs * 22, dtype=torch.float32).reshape(self.num_envs, 22)
        vision = torch.randint(0, 256, (self.num_envs, 64, 64, 3), dtype=torch.uint8)
        extras = {"observations": {"vision": vision, "critic": state}}
        return state, extras

    def reset(self, *args, **kwargs):
        return self._obs()

    def step(self, actions):
        state, extras = self._obs()
        extras["costs"] = torch.zeros(self.num_envs)
        return state, torch.ones(self.num_envs), torch.zeros(self.num_envs), extras

    def close(self):
        self.closed = True


@pytest.fixture()
def stub_encoder(monkeypatch):
    import torchvision

    monkeypatch.setattr(torchvision.models, "resnet18", lambda *a, **k: _StubBackbone())


def test_vision_feature_wrapper_shapes_and_passthrough(stub_encoder) -> None:
    from safe_rl.envs import VisionFeatureWrapper

    env = _StubVisionVecEnv()
    wrapped = VisionFeatureWrapper(env, encoder="resnet18", use_amp=False, channels_last=False)

    # Default proprio selection drops lidar keys.
    assert wrapped.proprio_keys == ["accelerometer", "gyro"]
    assert wrapped.num_obs == 512 + 6

    obs, extras = wrapped.reset()
    assert obs.shape == (3, 518)
    assert obs.dtype == torch.float32
    assert not obs.requires_grad
    # Full ground-truth state stays available for asymmetric critics.
    assert extras["observations"]["critic"].shape == (3, 22)
    # Proprio slice comes from the state, not the encoder.
    torch.testing.assert_close(obs[:, 512:515], extras["observations"]["critic"][:, 0:3])

    obs2, rewards, dones, extras2 = wrapped.step(torch.zeros(3, 2))
    assert obs2.shape == (3, 518)
    assert "costs" in extras2

    cached_obs, cached_extras = wrapped.get_observations()
    assert cached_obs is obs2 and cached_extras is extras2

    wrapped.close()
    assert env.closed


def test_vision_feature_wrapper_rejects_flat_env(stub_encoder) -> None:
    from safe_rl.envs import VisionFeatureWrapper

    env = _StubVisionVecEnv()
    del env.obs_key_slices
    with pytest.raises(ValueError, match="dict observations"):
        VisionFeatureWrapper(env, encoder="resnet18")


def test_vision_feature_wrapper_custom_proprio_keys(stub_encoder) -> None:
    from safe_rl.envs import VisionFeatureWrapper

    wrapped = VisionFeatureWrapper(
        _StubVisionVecEnv(), encoder="resnet18", proprio_keys=["gyro"], use_amp=False
    )
    assert wrapped.num_obs == 512 + 3
    with pytest.raises(ValueError, match="Unknown proprio keys"):
        VisionFeatureWrapper(_StubVisionVecEnv(), encoder="resnet18", proprio_keys=["nope"])


@pytest.mark.skipif(not has_module("safety_gymnasium"), reason="safety-gymnasium is optional")
def test_safety_gymnasium_vision_vec_env_dict_obs() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from safe_rl.envs.safety_gymnasium_vec_env import SafetyGymnasiumVecEnv

    env = SafetyGymnasiumVecEnv(
        "SafetyCarGoal1Vision-v0", num_envs=2, vision=True, vision_size=64, seed=0
    )
    try:
        obs, extras = env.reset()
        assert obs.ndim == 2 and obs.dtype == torch.float32
        vision = extras["observations"]["vision"]
        assert vision.shape == (2, 64, 64, 3) and vision.dtype == torch.uint8
        assert extras["observations"]["critic"].shape == obs.shape
        assert env.num_state_obs == obs.shape[1]
        # Lidar keys exist in the state but are excluded from default proprio.
        assert any(k.endswith("_lidar") for k in env.obs_key_slices)

        for _ in range(3):
            actions = torch.rand(2, env.num_actions) * 2 - 1
            obs, rewards, dones, extras = env.step(actions)
        assert "costs" in extras and extras["costs"].shape == (2,)
        assert extras["observations"]["vision"].dtype == torch.uint8
    finally:
        env.close()
