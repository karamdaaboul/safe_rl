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
    ):
        """Initialize the replay buffer.

        Args:
            num_envs: Number of parallel environments.
            max_size: Maximum number of transitions to store.
            obs_shape: Shape of observations.
            action_shape: Shape of actions.
            device: Device to store tensors on.
            initial_size: Minimum transitions before sampling is allowed.
        """
        self.device = device
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape

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

        indices = torch.randint(0, self._size, (batch_size,), device=self.device)

        batch = {}
        for name, data in self._data.items():
            batch[name] = self._process_undo(name, data[indices].clone())

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
