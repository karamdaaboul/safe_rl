from __future__ import annotations

import torch


class NStepReturnAggregator:
    """Per-env n-step return accumulator that sits between the runner and `ReplayStorage`.

    Buffers the last `n` transitions per environment and emits aggregated n-step
    transitions of the form::

        (s_t, a_t, R_n, s_{t+n}, done_n)

    where ``R_n = sum_{k=0}^{n-1} gamma^k * r_{t+k}``. The emitted ``done_n`` flag
    is 1 only when the episode ended with a Bellman-terminal (not a timeout).
    Truncated episodes are flushed but marked ``done_n = 0`` so the downstream
    critic can still bootstrap from the truncation-boundary observation.

    Slot convention: ``buf[0]`` = oldest, ``buf[n-1]`` = newest. When the buffer
    fills, the oldest slot is emitted as an n-step transition, shifted out, and
    the new transition is inserted at slot ``n-1``.
    """

    def __init__(
        self,
        n_step: int,
        gamma: float,
        num_envs: int,
        device: str,
    ) -> None:
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}")
        self.n = int(n_step)
        self.gamma = float(gamma)
        self.num_envs = int(num_envs)
        self.device = device

        self._count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._discounts = (self.gamma ** torch.arange(self.n, device=device)).float()
        self._bufs: dict[str, torch.Tensor] = {}

    @property
    def effective_gamma(self) -> float:
        """Discount factor for the n-step Bellman target: ``gamma ** n``."""
        return self.gamma ** self.n

    def reset(self) -> None:
        """Drop all pending transitions (e.g., at the start of a new training run)."""
        self._count.zero_()

    def _alloc(self, fields: dict[str, torch.Tensor]) -> None:
        for name, val in fields.items():
            if name not in self._bufs:
                self._bufs[name] = torch.zeros(
                    (self.n, *val.shape), device=self.device, dtype=val.dtype
                )

    def push(
        self,
        storage,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        terminal: torch.Tensor,
        truncated: torch.Tensor | None = None,
        critic_obs: torch.Tensor | None = None,
        next_critic_obs: torch.Tensor | None = None,
    ) -> None:
        """Push a batched transition; emit matured transitions to ``storage``.

        Args:
            storage: Target replay buffer (must implement ``add``).
            obs, action, next_obs: ``[num_envs, *]`` tensors.
            reward: ``[num_envs]`` or ``[num_envs, 1]``.
            terminal: ``[num_envs]`` or ``[num_envs, 1]`` — 1 for Bellman-terminal.
            truncated: Optional same-shape — 1 for timeout. Truncation triggers
                a flush but the emitted ``done`` stays 0 so the critic bootstraps.
            critic_obs, next_critic_obs: Optional privileged observations.
        """
        if reward.dim() == 1:
            reward = reward.view(-1, 1)
        if terminal.dim() == 1:
            terminal = terminal.view(-1, 1)
        if truncated is None:
            truncated = torch.zeros_like(terminal)
        elif truncated.dim() == 1:
            truncated = truncated.view(-1, 1)

        fields: dict[str, torch.Tensor] = {
            "observations": obs,
            "actions": action,
            "rewards": reward,
            "next_observations": next_obs,
        }
        if critic_obs is not None:
            fields["critic_observations"] = critic_obs
        if next_critic_obs is not None:
            fields["next_critic_observations"] = next_critic_obs

        self._alloc(fields)

        # Shift oldest out, insert newest at slot n-1.
        for name, val in fields.items():
            buf = self._bufs[name]
            buf[:-1] = buf[1:].clone()
            buf[-1] = val

        self._count = torch.clamp(self._count + 1, max=self.n)

        terminal_mask = terminal.view(-1).bool()
        truncated_mask = truncated.view(-1).bool()
        episode_end = terminal_mask | truncated_mask

        # Case A: episode ended — flush every pending slot for these envs.
        if episode_end.any():
            for h in range(1, self.n + 1):
                mask_h = episode_end & (self._count == h)
                if not mask_h.any():
                    continue
                idxs = torch.nonzero(mask_h, as_tuple=True)[0]
                ends_terminal = terminal_mask[idxs]  # [k] — terminal vs truncated
                for k in range(h):
                    self._emit(
                        storage,
                        idxs=idxs,
                        src_slot=self.n - h + k,
                        rem_h=h - k,
                        is_terminal=ends_terminal,
                    )
            self._count[episode_end] = 0

        # Case B: buffer full and NOT ending this step — emit oldest slot as n-step.
        full_continuing = (~episode_end) & (self._count == self.n)
        if full_continuing.any():
            idxs = torch.nonzero(full_continuing, as_tuple=True)[0]
            self._emit(
                storage,
                idxs=idxs,
                src_slot=0,
                rem_h=self.n,
                is_terminal=None,
            )

    def _emit(
        self,
        storage,
        idxs: torch.Tensor,
        src_slot: int,
        rem_h: int,
        is_terminal: torch.Tensor | None,
    ) -> None:
        """Emit a batch of aggregated transitions from ``src_slot`` with horizon ``rem_h``."""
        rewards_slc = self._bufs["rewards"][src_slot:src_slot + rem_h, idxs, :]  # [h, k, 1]
        disc = self._discounts[:rem_h].view(rem_h, 1, 1)
        R_n = (disc * rewards_slc).sum(dim=0)  # [k, 1]

        emit_obs = self._bufs["observations"][src_slot, idxs]
        emit_action = self._bufs["actions"][src_slot, idxs]
        emit_next_obs = self._bufs["next_observations"][self.n - 1, idxs]

        if is_terminal is None:
            emit_done = torch.zeros(len(idxs), 1, device=self.device)
        else:
            emit_done = is_terminal.view(-1, 1).float()

        extras: dict[str, torch.Tensor] = {}
        if "critic_observations" in self._bufs:
            extras["critic_observations"] = self._bufs["critic_observations"][src_slot, idxs]
        if "next_critic_observations" in self._bufs:
            extras["next_critic_observations"] = self._bufs["next_critic_observations"][self.n - 1, idxs]

        storage.add(emit_obs, emit_action, R_n, emit_done, emit_next_obs, **extras)
