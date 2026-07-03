from __future__ import annotations

from typing import Callable, Generator

import torch


class ReplayStorage:
    """Replay buffer for off-policy algorithms (SAC, TD3).

    This implements a circular buffer that stores transitions for experience replay.
    It supports efficient random sampling for mini-batch training.

    Features:
    - Flexible storage: Can store arbitrary named tensors
    - Warm-up: Optional minimum samples before training starts
    - Processing pipelines: Register pre/post-processors for data normalization
    - Efficient batched operations for vectorized environments
    """

    def __init__(
        self,
        num_envs: int,
        max_size: int,
        obs_shape: list[int],
        action_shape: list[int],
        device: str = "cpu",
        initial_size: int = 0,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        """Initialize the replay buffer.

        Args:
            num_envs: Number of parallel environments.
            max_size: Maximum number of transitions to store.
            obs_shape: Shape of observations.
            action_shape: Shape of actions.
            device: Device to store tensors on.
            initial_size: Minimum transitions before sampling is allowed.
            n_step: Horizon for in-buffer n-step return aggregation. ``1`` (default)
                keeps the plain flat random-sampling behaviour. ``> 1`` enables
                sample-time n-step targets computed from the per-env transition
                stride (requires ``max_size`` to be a multiple of ``num_envs``).
            gamma: Discount factor used for n-step reward aggregation (only used
                when ``n_step > 1``).
        """
        self.device = device
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # N-step aggregation (in-buffer). For n_step > 1 the flat buffer is viewed
        # as a per-env ring of shape [T, num_envs] (row t, env e -> flat t*num_envs+e),
        # which requires max_size to be an exact multiple of num_envs so the stride
        # stays aligned across wrap-around.
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        if self.n_step > 1:
            max_size = (max_size // num_envs) * num_envs
            if max_size < num_envs * self.n_step:
                raise ValueError(
                    f"max_size ({max_size}) too small for n_step={self.n_step} "
                    f"with num_envs={num_envs}; need at least num_envs * n_step transitions."
                )

        # Buffer sizing
        self._max_size = max_size
        self._initial_size = initial_size

        # Buffer state
        self._ptr = 0  # Current write position
        self._size = 0  # Current number of stored transitions
        self._initialized = initial_size == 0

        # Data storage (lazy allocation)
        self._data: dict[str, torch.Tensor] = {}

        # Processing pipelines: {key: [(process_fn, undo_fn), ...]}
        self._processors: dict[str, list[tuple[Callable | None, Callable | None]]] = {}

    @property
    def max_size(self) -> int:
        """Maximum buffer capacity."""
        return self._max_size

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self._size

    @property
    def initialized(self) -> bool:
        """Whether the buffer has enough samples for training."""
        return self._initialized

    def _allocate_tensor(self, name: str, value: torch.Tensor) -> None:
        """Lazily allocate storage for a new data field."""
        if name not in self._data:
            self._data[name] = torch.zeros(
                self._max_size, *value.shape[1:], device=self.device, dtype=value.dtype
            )

    def _process(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Apply registered processors to data before storage."""
        if name not in self._processors:
            return value
        for process_fn, _ in self._processors[name]:
            if process_fn is not None:
                value = process_fn(value)
        return value

    def _process_undo(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Undo processing when retrieving data."""
        if name not in self._processors:
            return value
        for _, undo_fn in reversed(self._processors[name]):
            if undo_fn is not None:
                value = undo_fn(value)
        return value

    def register_processor(
        self,
        key: str,
        process: Callable | None = None,
        undo: Callable | None = None,
    ) -> None:
        """Register a processor for a data field.

        Processors are applied before storage, undo functions when retrieving.

        Args:
            key: Name of the data field.
            process: Function to apply before storage (e.g., normalize).
            undo: Function to reverse the processing (e.g., denormalize).
        """
        if key not in self._processors:
            self._processors[key] = []
        self._processors[key].append((process, undo))

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
        **extras: torch.Tensor,
    ) -> None:
        """Add transitions from all environments to the buffer.

        Args:
            obs: Observations [num_envs, *obs_shape]
            action: Actions [num_envs, *action_shape]
            reward: Rewards [num_envs] or [num_envs, 1]
            done: Done flags [num_envs] or [num_envs, 1]
            next_obs: Next observations [num_envs, *obs_shape]
            **extras: Additional data to store (e.g., costs, log_probs)
        """
        batch_size = obs.shape[0]

        # Ensure proper shapes
        reward = reward.view(-1, 1) if reward.dim() == 1 else reward
        done = done.view(-1, 1) if done.dim() == 1 else done

        # Build transition dict
        transition = {
            "observations": obs,
            "actions": action,
            "rewards": reward,
            "dones": done,
            "next_observations": next_obs,
            **extras,
        }

        # Store each field
        for name, value in transition.items():
            value = value.to(self.device)
            value = self._process(name, value)

            # Allocate storage if needed
            self._allocate_tensor(name, value)

            # Calculate indices for circular buffer
            indices = torch.arange(self._ptr, self._ptr + batch_size, device=self.device) % self._max_size
            self._data[name][indices] = value

        # Update pointer and size
        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)

        # Check if we've reached initial_size
        if not self._initialized and self._size >= self._initial_size:
            self._initialized = True

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary containing sampled data for all stored fields.
        """
        if not self._initialized:
            raise RuntimeError(
                f"Buffer not initialized. Need {self._initial_size} samples, have {self._size}."
            )

        if self.n_step > 1:
            return self._sample_n_step(batch_size)

        indices = torch.randint(0, self._size, (batch_size,), device=self.device)

        batch = {}
        for name, data in self._data.items():
            batch[name] = self._process_undo(name, data[indices].clone())

        return batch

    def _sample_n_step(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch with in-buffer n-step return aggregation.

        Views the flat buffer as a per-env ring ``[T, num_envs]`` (T = max_size //
        num_envs) and, for each sampled (env, start) transition, aggregates the
        discounted reward over the next ``n_step`` transitions of the same env,
        truncating at the first episode end. Mirrors the reference rsl_rl_sac
        ``ReplayBuffer._generate_batch`` logic.

        Returns the same keys as :meth:`sample`, with ``rewards`` holding the
        n-step discounted return, ``next_observations`` / ``dones`` / ``bootstrap``
        taken at the (possibly truncated) horizon, plus an ``effective_n_steps``
        tensor so the algorithm can apply ``gamma ** effective_n_steps``.
        """
        n = self.n_step
        num_envs = self.num_envs
        per_env_len = self._max_size // num_envs  # T (capacity per env)
        filled_t = self._size // num_envs  # number of fully written columns
        head = (self._ptr // num_envs) % per_env_len  # next column to write per env
        full = self._size >= self._max_size

        # Per-env time view of each stored field: [T, num_envs, *feat].
        def view2d(data: torch.Tensor) -> torch.Tensor:
            return data.reshape(per_env_len, num_envs, *data.shape[1:])

        # Build the set of valid (start_t, env) starts whose n-step window does not
        # cross the write head (the oldest/newest boundary in the ring).
        time_len = per_env_len if full else filled_t
        start_t = torch.arange(time_len, device=self.device)
        max_offset = n - 1
        if full:
            starts_before_head = start_t < head
            safe_before = (start_t + max_offset) < head
            safe_after = (start_t + max_offset) < (per_env_len + head)
            safe_mask = torch.where(starts_before_head, safe_before, safe_after)
        else:
            safe_mask = (start_t + max_offset) < time_len
        valid_t = start_t[safe_mask]
        if valid_t.numel() == 0:
            raise RuntimeError("Not enough contiguous transitions for n-step sampling.")

        # Sample (start_t, env) pairs uniformly over the valid grid.
        t_idx = valid_t[torch.randint(0, valid_t.numel(), (batch_size,), device=self.device)]
        e_idx = torch.randint(0, num_envs, (batch_size,), device=self.device)

        step_offsets = torch.arange(n, device=self.device)
        window_t = (t_idx.unsqueeze(-1) + step_offsets) % per_env_len  # [B, n]
        e_exp = e_idx.unsqueeze(-1).expand(batch_size, n)

        rewards2d = view2d(self._data["rewards"])  # [T, E, 1]
        dones2d = view2d(self._data["dones"])  # [T, E, 1]
        all_rewards = rewards2d[window_t, e_exp].squeeze(-1)  # [B, n]
        all_dones = dones2d[window_t, e_exp].squeeze(-1)  # [B, n]

        # Zero out rewards after the first done; discounted sum gamma^k * r_k.
        dones_shifted = torch.cat([torch.zeros_like(all_dones[..., :1]), all_dones[..., :-1]], dim=-1)
        done_masks = torch.cumprod(1.0 - dones_shifted, dim=-1)
        discounts = torch.pow(self.gamma, step_offsets)
        n_step_rewards = (all_rewards * done_masks * discounts.view(1, -1)).sum(dim=-1, keepdim=True)

        # Effective horizon = index of first done (+1), else full n.
        first_done = torch.argmax((all_dones > 0).float(), dim=-1)
        no_dones = all_dones.sum(dim=-1) == 0
        first_done = torch.where(no_dones, torch.full_like(first_done, n - 1), first_done)
        effective_n_steps = (first_done + 1).unsqueeze(-1).to(torch.long)
        final_t = window_t.gather(1, first_done.unsqueeze(-1)).squeeze(-1)  # [B]

        batch: dict[str, torch.Tensor] = {}
        for name, data in self._data.items():
            d2d = view2d(data)
            if name == "rewards":
                value = n_step_rewards
            elif name in ("next_observations", "next_critic_observations", "dones", "bootstrap"):
                # Taken at the (possibly truncated) horizon step.
                value = d2d[final_t, e_idx]
            else:
                # State/action/critic-obs taken at the start step.
                value = d2d[t_idx, e_idx]
            batch[name] = self._process_undo(name, value.clone())

        batch["effective_n_steps"] = effective_n_steps
        return batch

    def batch_generator(
        self, batch_size: int, num_batches: int
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Generate multiple random batches.

        Args:
            batch_size: Number of transitions per batch.
            num_batches: Number of batches to generate.

        Yields:
            Dictionary containing sampled data for each batch.
        """
        for _ in range(num_batches):
            yield self.sample(batch_size)

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self._size

    def clear(self) -> None:
        """Clear the buffer."""
        self._ptr = 0
        self._size = 0
        self._initialized = self._initial_size == 0
        self._data.clear()

    def state_dict(self) -> dict:
        """Get state for serialization."""
        return {
            "data": {k: v.cpu() for k, v in self._data.items()},
            "ptr": self._ptr,
            "size": self._size,
            "initialized": self._initialized,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from serialization."""
        self._data = {k: v.to(self.device) for k, v in state["data"].items()}
        self._ptr = state["ptr"]
        self._size = state["size"]
        self._initialized = state["initialized"]
