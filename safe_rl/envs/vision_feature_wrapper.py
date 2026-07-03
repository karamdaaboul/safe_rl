"""Frozen pretrained-encoder wrapper for vision-based safe RL.

Wraps a :class:`~safe_rl.envs.vec_env.VecEnv` whose ``extras["observations"]``
carries a batched uint8 ``"vision"`` image (as produced by
:class:`~safe_rl.envs.safety_gymnasium_vec_env.SafetyGymnasiumVecEnv` on the
``*Vision-v0`` envs) and replaces the primary observation with

    ``[frozen encoder features, selected proprioceptive sensors]``

so that everything downstream (rollout storage, MLP actor/critics, the on-policy
runner) keeps seeing a flat float32 tensor. The full ground-truth state placed in
``extras["observations"]["critic"]`` by the inner env is passed through untouched,
which gives asymmetric reward/cost critics for free.

The encoder runs exactly once per env step on the whole batch, under
``torch.no_grad`` with optional fp16 autocast and channels_last layout; images
stay uint8 until inside :meth:`_encode`. This follows the frozen
pretrained-visual-representation (PVR) line of work (arXiv:2407.17238) — cheap,
storage-friendly, and fully parallel — as the first stage before an end-to-end
shared CNN.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .vec_env import VecEnv

# Keys of the ground-truth state that leak object positions; excluded from the
# actor's proprio slice by default so the policy must rely on the camera.
_NONPROPRIO_SUFFIXES = ("_lidar", "_comp")

_ENCODER_DIMS = {"resnet18": 512, "dinov2_vits14": 384}


class VisionFeatureWrapper(VecEnv):
    """Encode uint8 vision observations with a frozen pretrained backbone.

    Args:
        env: Wrapped VecEnv. Must expose ``obs_key_slices`` (mapping state key ->
            slice into the flat state obs) and put a uint8 image batch in
            ``extras["observations"]["vision"]``.
        encoder: ``"resnet18"`` (ImageNet, 512-d) or ``"dinov2_vits14"`` (384-d).
        encoder_weights: Optional local checkpoint path (state dict) to load
            instead of downloading — needed on offline machines (JUWELS).
        proprio_keys: State keys to append to the encoder features. Defaults to
            all non-lidar/compass keys of the inner env.
        use_amp: Run the encoder under fp16 autocast (CUDA only).
        channels_last: Use channels_last memory format for the conv backbone.
    """

    def __init__(
        self,
        env: VecEnv,
        encoder: str = "resnet18",
        encoder_weights: str | None = None,
        device: str | torch.device | None = None,
        proprio_keys: list[str] | None = None,
        use_amp: bool = True,
        channels_last: bool = True,
    ) -> None:
        self.env = env
        self.device = torch.device(device) if device is not None else env.device
        self.num_envs = int(env.num_envs)
        self.num_actions = int(env.num_actions)
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf
        self.cfg = env.cfg
        self.step_dt = float(getattr(env, "step_dt", 1.0))

        self.encoder_name = encoder
        self._amp_enabled = bool(use_amp) and self.device.type == "cuda"
        self._channels_last = bool(channels_last)

        obs_key_slices = getattr(env, "obs_key_slices", None)
        if obs_key_slices is None:
            raise ValueError(
                "VisionFeatureWrapper requires an inner env with dict observations "
                "(e.g. SafetyGymnasiumVecEnv on a *Vision-v0 env with vision=True)."
            )
        if proprio_keys is None:
            proprio_keys = [
                k for k in obs_key_slices if not k.endswith(_NONPROPRIO_SUFFIXES)
            ]
        unknown = [k for k in proprio_keys if k not in obs_key_slices]
        if unknown:
            raise ValueError(f"Unknown proprio keys {unknown}; available: {list(obs_key_slices)}")
        self.proprio_keys = list(proprio_keys)
        self._proprio_idx = torch.cat(
            [torch.arange(obs_key_slices[k].start, obs_key_slices[k].stop) for k in self.proprio_keys]
        ).to(self.device)

        self.encoder, self.num_features = self._build_encoder(encoder, encoder_weights)
        self.num_proprio = int(self._proprio_idx.numel())
        self.num_obs = self.num_features + self.num_proprio

        # ImageNet normalization, applied on-device after the /255 rescale.
        self._img_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._img_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self._last_obs: torch.Tensor | None = None
        self._last_extras: dict | None = None

    # -- encoder ------------------------------------------------------------------

    def _build_encoder(self, name: str, weights_path: str | None) -> tuple[torch.nn.Module, int]:
        if name == "resnet18":
            import torchvision

            if weights_path is not None:
                net = torchvision.models.resnet18()
                net.load_state_dict(torch.load(weights_path, map_location="cpu"))
            else:
                net = torchvision.models.resnet18(
                    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                )
            net.fc = torch.nn.Identity()
        elif name == "dinov2_vits14":
            if weights_path is not None:
                net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)
                net.load_state_dict(torch.load(weights_path, map_location="cpu"))
            else:
                net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        else:
            raise ValueError(f"Unknown vision encoder '{name}'; choose from {list(_ENCODER_DIMS)}")

        net.eval().requires_grad_(False)
        net = net.to(self.device)
        if self._channels_last and name == "resnet18":
            net = net.to(memory_format=torch.channels_last)
        return net, _ENCODER_DIMS[name]

    def _encode(self, vision_u8: torch.Tensor) -> torch.Tensor:
        """One batched forward: (N, H, W, 3) uint8 -> (N, num_features) float32."""
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self._amp_enabled
        ):
            x = vision_u8.to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
            if self.encoder_name == "dinov2_vits14":
                side = x.shape[-1] - x.shape[-1] % 14
                if x.shape[-1] % 14 or x.shape[-2] % 14:
                    x = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False)
            x = (x - self._img_mean) / self._img_std
            if self._channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            feats = self.encoder(x)
        return feats.float()

    def _transform(self, obs: torch.Tensor, extras: dict) -> tuple[torch.Tensor, dict]:
        vision = extras.get("observations", {}).get("vision")
        if vision is None:
            raise RuntimeError(
                "No 'vision' entry in extras['observations'] — the inner env did not "
                "produce image observations (is this a *Vision-v0 env with vision=True?)."
            )
        feats = self._encode(vision)
        actor_obs = torch.cat([feats, obs[:, self._proprio_idx]], dim=1)
        self._last_obs, self._last_extras = actor_obs, extras
        return actor_obs, extras

    # -- VecEnv API ----------------------------------------------------------------

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        if self._last_obs is None or self._last_extras is None:
            return self.reset()
        return self._last_obs, self._last_extras

    def reset(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, dict]:
        obs, extras = self.env.reset(*args, **kwargs)
        return self._transform(obs, extras)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, rewards, dones, extras = self.env.step(actions)
        obs, extras = self._transform(obs, extras)
        return obs, rewards, dones, extras

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails; delegates e.g. cost_limits,
        # set_task, render to the wrapped env without shadowing __init__ attrs.
        return getattr(self.__dict__["env"], name)
