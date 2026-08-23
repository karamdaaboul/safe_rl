from __future__ import annotations

import os
import warnings
from typing import Callable, Generator

import torch

from safe_rl.common.continuation import exponential_continuation

# Pool identifiers for hazard-stratified replay. A stored slot belongs to exactly one.
SAFE = 0
HAZARD = 1

# O(size) pool-invariant checks are off by default; tests call check_pool_invariants()
# directly, and SAFE_RL_REPLAY_DEBUG=1 turns them on inside add()/sample().
_REPLAY_DEBUG = os.environ.get("SAFE_RL_REPLAY_DEBUG", "0") == "1"

# Bounded retry budgets for the rejection-style draws (see _draw_positions_wor and
# _draw_valid_flat). Both fall back to a correct-but-slower path if exhausted.
_WOR_MAX_ROUNDS = 5
_NSTEP_MAX_ROUNDS = 6


class ReplayStorage:
    """Replay buffer for off-policy algorithms (SAC, TD3).

    This implements a circular buffer that stores transitions for experience replay.
    It supports efficient random sampling for mini-batch training.

    Features:
    - Flexible storage: Can store arbitrary named tensors
    - Warm-up: Optional minimum samples before training starts
    - Processing pipelines: Register pre/post-processors for data normalization
    - Efficient batched operations for vectorized environments
    - Optional hazard-stratified sampling (``hazard_fraction``): a second sampling
      *view* over the same transitions that over-represents cost-bearing ones, with
      importance weights that restore the uniform-replay objective.
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
        cost_gamma: float | None = None,
        cost_n_step: int | None = None,
        hazard_fraction: float = 0.0,
        cost_window_extras: bool = False,
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
            cost_gamma: Discount for n-step **cost** aggregation. ``None`` (default) reuses
                ``gamma``, which is every existing algorithm's behaviour. FH-DCMPO passes ``1.0``:
                the benchmark constrains an undiscounted episodic cost sum, so the n-step cost
                target must be ``sum_k c_k`` rather than ``sum_k gamma^k c_k``. Without this the
                critic's target would be discounted no matter what the algorithm's bootstrap
                discount says -- a second, easily-missed discount.
            hazard_fraction: Target fraction of cost-bearing ("hazard") transitions in
                a batch drawn with ``sample(..., stratified=True)``. ``0.0`` (default)
                disables the machinery entirely — no metadata is allocated and every
                code path is identical to plain uniform replay. Transitions are stored
                exactly once regardless; the pools are index views over that storage.
            cost_window_extras: Off-policy mismatch DIAGNOSTIC (default off). When on,
                :meth:`_gather_cost_window` additionally returns per-step actions,
                actor observations, slot age, and — if stored — behavior log-probs and
                policy versions for the TD(lambda) window, so the algorithm can measure
                how far the current policy has drifted from the one that generated each
                window. Pure extra gathers; the training fields are untouched.
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
        self.cost_gamma = self.gamma if cost_gamma is None else float(cost_gamma)
        self.cost_n_step = self.n_step if cost_n_step is None else max(1, int(cost_n_step))
        # Stochastic-decision-horizon shaping (arXiv:2602.04599). None = OFF, and the reward
        # aggregation stays bit-identical to the scalar-gamma path. The algorithm sets this each
        # update from its schedule, which is why it is a mutable attribute rather than a ctor arg:
        # the scale ramps during training and every sample must use the CURRENT value.
        self.survival_lambda: float | None = None
        self.cost_window_extras = bool(cost_window_extras)
        # Every window the sampler must not run off the end of: the longer of the two.
        self.window_len = max(self.n_step, self.cost_n_step)
        if self.window_len > 1:
            max_size = (max_size // num_envs) * num_envs
            if max_size < num_envs * self.window_len:
                raise ValueError(
                    f"max_size ({max_size}) too small for window_len={self.window_len} "
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

        # --- Hazard-stratified sampling metadata (index views, not a second buffer) ---
        if not 0.0 <= float(hazard_fraction) <= 1.0:
            raise ValueError(f"hazard_fraction must lie in [0, 1], got {hazard_fraction}.")
        self._hazard_fraction = float(hazard_fraction)
        self._warned_missing_costs = False
        # Composition of the most recent stratified batch, for logging. Exact by
        # construction -- deriving it from the sampled costs would be wrong under
        # n_step > 1, where an aggregated cost can be positive for a safe-stratum start.
        self._last_stratified_info: dict[str, float] = {}
        if self._hazard_fraction > 0.0:
            if num_envs > self._max_size:
                # add() writes num_envs contiguous slots; if they wrapped onto each other
                # the indices would repeat and every pool invariant would silently break.
                raise ValueError(
                    f"num_envs ({num_envs}) must not exceed max_size ({self._max_size}) "
                    "when hazard_fraction > 0."
                )
            # pools[c][:counts[c]] holds the flat buffer indices currently in class c.
            # Unordered; membership is a set, so any permutation is equivalent.
            self._pools = [
                torch.zeros(self._max_size, dtype=torch.long, device=self.device),
                torch.zeros(self._max_size, dtype=torch.long, device=self.device),
            ]
            self._counts = [0, 0]  # python ints -> comparing them costs no device sync
            # Inverse map, shared across pools since a slot is in exactly one of them.
            self._slot_pos = torch.full((self._max_size,), -1, dtype=torch.long, device=self.device)
            # -1 = never written, 0 = SAFE, 1 = HAZARD.
            self._slot_class = torch.full((self._max_size,), -1, dtype=torch.int8, device=self.device)
        else:
            self._pools = []
            self._counts = [0, 0]
            self._slot_pos = None
            self._slot_class = None

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

    @property
    def hazard_fraction(self) -> float:
        """Target hazard fraction for stratified sampling (0 = disabled)."""
        return self._hazard_fraction

    @property
    def hazard_pool_size(self) -> int:
        """Number of stored transitions classified as hazardous."""
        return self._counts[HAZARD]

    @property
    def safe_pool_size(self) -> int:
        """Number of stored transitions classified as safe."""
        return self._counts[SAFE]

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

        # Calculate indices for the circular buffer once (identical for every field).
        indices = torch.arange(self._ptr, self._ptr + batch_size, device=self.device) % self._max_size

        # Store each field
        stored_costs = None
        for name, value in transition.items():
            value = value.to(self.device)
            value = self._process(name, value)

            # Allocate storage if needed
            self._allocate_tensor(name, value)

            self._data[name][indices] = value
            if name == "costs":
                # Classify on the post-process value so the incremental path and the
                # load_state_dict rebuild always agree.
                stored_costs = value

        # Update pointer and size
        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)

        # Refresh the hazard/safe index views for the slots just overwritten.
        if self._hazard_fraction > 0.0:
            self._update_pools(indices, self._classify(stored_costs, batch_size))

        # Check if we've reached initial_size
        if not self._initialized and self._size >= self._initial_size:
            self._initialized = True

    # ------------------------------------------------------------------
    # Hazard/safe index pools
    # ------------------------------------------------------------------

    def _classify(self, costs: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        """Class of each transition in an ``add()`` batch: [B] int8 in {SAFE, HAZARD}.

        A transition is hazardous when any of its constraint costs is positive. Slots
        stored without a ``costs`` field (plain SAC/TD3) are all SAFE.
        """
        if costs is None:
            return torch.zeros(batch_size, dtype=torch.int8, device=self.device)
        return (costs.view(batch_size, -1) > 0).any(dim=-1).to(torch.int8)

    def _update_pools(self, indices: torch.Tensor, new_class: torch.Tensor) -> None:
        """Move the just-written slots into their pools.

        ``indices`` are pairwise distinct (contiguous mod max_size, and num_envs <=
        max_size is enforced in __init__), which is what makes the batched swap-delete
        below safe. Only slots whose class *flips* need a pool move: an unchanged-class
        slot is removed and re-inserted into the same pool, so we skip it entirely.
        """
        old_class = self._slot_class[indices]
        fresh = old_class < 0  # never written before -> pure append
        flip = (~fresh) & (old_class != new_class)
        is_hazard = new_class == HAZARD

        move_h2s = indices[flip & (old_class == HAZARD)]
        move_s2h = indices[flip & (old_class == SAFE)]

        # Removes before inserts, so no element is ever momentarily in two pools.
        self._pool_remove(HAZARD, move_h2s)
        self._pool_remove(SAFE, move_s2h)
        self._pool_insert(HAZARD, torch.cat([move_s2h, indices[fresh & is_hazard]]))
        self._pool_insert(SAFE, torch.cat([move_h2s, indices[fresh & ~is_hazard]]))

        self._slot_class[indices] = new_class

        # O(1): three python ints, no device sync.
        assert self._counts[SAFE] + self._counts[HAZARD] == self._size
        if _REPLAY_DEBUG:
            self.check_pool_invariants()

    def _pool_remove(self, pool_id: int, elems: torch.Tensor) -> None:
        """Remove distinct ``elems`` from pool ``pool_id`` in O(k log k).

        Swap-delete generalized to a batch: positions below ``M-k`` are holes that must
        be refilled; the last ``k`` pool entries are the candidate fillers. Fillers that
        are themselves being deleted are excluded via an O(k) mask over the tail (never
        an O(M) mask over the pool). Since ``holes < M-k <= survivors``, the read and
        write index sets are disjoint, so the advanced-index assignment is well defined.
        """
        k = elems.numel()
        if k == 0:
            return
        pool = self._pools[pool_id]
        tail_lo = self._counts[pool_id] - k

        positions, _ = torch.sort(self._slot_pos[elems])  # [k], distinct, ascending
        holes = positions[positions < tail_lo]
        deleted_in_tail = positions[positions >= tail_lo]

        keep = torch.ones(k, dtype=torch.bool, device=self.device)
        keep[deleted_in_tail - tail_lo] = False
        survivors = tail_lo + keep.nonzero(as_tuple=True)[0]  # same length as holes

        moved = pool[survivors]
        pool[holes] = moved
        self._slot_pos[moved] = holes
        self._slot_pos[elems] = -1
        self._counts[pool_id] = tail_lo

    def _pool_insert(self, pool_id: int, elems: torch.Tensor) -> None:
        """Append distinct ``elems`` (not already members) to pool ``pool_id``."""
        k = elems.numel()
        if k == 0:
            return
        start = self._counts[pool_id]
        positions = torch.arange(start, start + k, device=self.device)
        self._pools[pool_id][positions] = elems
        self._slot_pos[elems] = positions
        self._counts[pool_id] = start + k

    def _rebuild_pools(self) -> None:
        """Recompute the pools from stored costs (used after load_state_dict).

        Pools are pure derived state, so they are rebuilt rather than serialized: that
        keeps checkpoints interchangeable in both directions and avoids ~25 MB per
        checkpoint at max_size=1e6. One O(max_size) pass, dominated by moving the
        replay tensors themselves.
        """
        if self._hazard_fraction <= 0.0:
            return
        self._counts = [0, 0]
        self._slot_pos.fill_(-1)
        self._slot_class.fill_(-1)
        if self._size == 0:
            return
        # The written set is always [0, _size): the ring writes contiguously from 0, and
        # any wrap implies total writes >= max_size, hence _size == max_size.
        written = torch.arange(self._size, device=self.device)
        costs = self._data.get("costs")
        if costs is None:
            hazard = torch.zeros(self._size, dtype=torch.bool, device=self.device)
        else:
            hazard = (costs[written].view(self._size, -1) > 0).any(dim=-1)
        self._slot_class[written] = hazard.to(torch.int8)
        for pool_id, members in ((HAZARD, written[hazard]), (SAFE, written[~hazard])):
            n = members.numel()
            self._pools[pool_id][:n] = members
            self._slot_pos[members] = torch.arange(n, device=self.device)
            self._counts[pool_id] = n

    def check_pool_invariants(self) -> None:
        """Assert pool bookkeeping is self-consistent. O(size); safe to call in tests.

        Within-pool distinctness and cross-pool disjointness follow from these checks
        and need no separate O(N) set intersection: ``_slot_pos`` is a left inverse of
        each pool (so members are distinct), and ``_slot_class`` is single-valued (so
        the pools cannot overlap).
        """
        if self._hazard_fraction <= 0.0:
            return
        assert self._counts[SAFE] + self._counts[HAZARD] == self._size
        for pool_id in (SAFE, HAZARD):
            n = self._counts[pool_id]
            members = self._pools[pool_id][:n]
            assert torch.equal(self._slot_pos[members], torch.arange(n, device=self.device))
            assert bool((self._slot_class[members] == pool_id).all())
        assert bool((self._slot_class[: self._size] >= 0).all())
        if self._size < self._max_size:
            assert bool((self._slot_class[self._size :] < 0).all())

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, stratified: bool = False) -> dict[str, torch.Tensor]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            stratified: Draw a hazard-stratified batch (requires ``hazard_fraction >
                0``). Falls back to uniform sampling when the pools cannot supply the
                request. Uniform batches carry all-ones ``cost_is_weights``.

        Returns:
            Dictionary containing sampled data for all stored fields, plus
            ``cost_is_weights`` [batch_size, 1] — the importance weights that restore
            the uniform-replay objective for the cost-critic regression.
        """
        if not self._initialized:
            raise RuntimeError(
                f"Buffer not initialized. Need {self._initial_size} samples, have {self._size}."
            )

        if stratified and self._hazard_fraction > 0.0:
            if "costs" not in self._data and not self._warned_missing_costs:
                self._warned_missing_costs = True
                warnings.warn(
                    "hazard_fraction > 0 but no 'costs' field has been stored; every "
                    "transition is classified safe and stratified sampling degenerates "
                    "to uniform. Is this buffer being fed by a safe-RL algorithm?",
                    RuntimeWarning,
                    stacklevel=2,
                )
            batch = self._sample_stratified(batch_size)
            if batch is not None:
                return batch
            # Fell back (pools cannot serve the request). Record that honestly rather
            # than leaving the previous stratified draw's numbers to be logged again.
            self._last_stratified_info = self._uniform_composition(batch_size)

        batch = self._sample_n_step(batch_size) if self.n_step > 1 else self._sample_uniform(batch_size)
        batch["cost_is_weights"] = torch.ones(batch_size, 1, device=self.device)
        return batch

    def _uniform_composition(self, batch_size: int) -> dict[str, float]:
        """Composition record for a uniform draw: weights are 1 and the batch tracks the
        buffer in expectation."""
        n_hazard, n_safe = self._counts[HAZARD], self._counts[SAFE]
        total = max(n_hazard + n_safe, 1)
        return {
            "replay_hazard_fraction_buffer": n_hazard / total,
            "replay_hazard_fraction_batch": n_hazard / total,
            "replay_cost_is_weight_hazard": 1.0,
            "replay_cost_is_weight_safe": 1.0,
            "replay_hazard_pool_size": float(n_hazard),
            "replay_safe_pool_size": float(n_safe),
        }

    def _sample_uniform(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Plain 1-step uniform sampling (with replacement)."""
        indices = torch.randint(0, self._size, (batch_size,), device=self.device)

        batch = {}
        for name, data in self._data.items():
            batch[name] = self._process_undo(name, data[indices].clone())

        return batch

    def _is_valid_start(self, start_t: torch.Tensor) -> torch.Tensor:
        """Elementwise O(k) test: does the n-step window from row ``start_t`` avoid the
        write head? Same predicate the uniform n-step sampler builds as a full mask."""
        per_env_len = self._max_size // self.num_envs
        max_offset = self.window_len - 1
        if self._size >= self._max_size:
            head = (self._ptr // self.num_envs) % per_env_len
            return torch.where(
                start_t < head,
                (start_t + max_offset) < head,
                (start_t + max_offset) < (per_env_len + head),
            )
        return (start_t + max_offset) < (self._size // self.num_envs)

    def _valid_start_t(self) -> torch.Tensor:
        """Ascending time rows whose n-step window does not cross the write head."""
        per_env_len = self._max_size // self.num_envs
        start_t = torch.arange(per_env_len, device=self.device)
        valid_t = start_t[self._is_valid_start(start_t)]
        if valid_t.numel() == 0:
            raise RuntimeError("Not enough contiguous transitions for n-step sampling.")
        return valid_t

    def _sample_n_step(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch with in-buffer n-step return aggregation.

        Views the flat buffer as a per-env ring ``[T, num_envs]`` (T = max_size //
        num_envs) and, for each sampled (env, start) transition, aggregates the
        discounted reward over the next ``n_step`` transitions of the same env,
        truncating at the first episode end. Mirrors the reference rsl_rl_sac
        ``ReplayBuffer._generate_batch`` logic.
        """
        valid_t = self._valid_start_t()
        # Sample (start_t, env) pairs uniformly over the valid grid.
        t_idx = valid_t[torch.randint(0, valid_t.numel(), (batch_size,), device=self.device)]
        e_idx = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        return self._gather_n_step(t_idx, e_idx)

    def _gather_n_step(self, t_idx: torch.Tensor, e_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build an n-step batch from externally chosen ``(start_t, env)`` pairs.

        Returns the same keys as :meth:`_sample_uniform`, with ``rewards`` holding the
        n-step discounted return, ``next_observations`` / ``dones`` / ``bootstrap``
        taken at the (possibly truncated) horizon, plus an ``effective_n_steps``
        tensor so the algorithm can apply ``gamma ** effective_n_steps``.
        """
        n = self.n_step
        num_envs = self.num_envs
        batch_size = t_idx.numel()
        per_env_len = self._max_size // num_envs  # T (capacity per env)

        # Per-env time view of each stored field: [T, num_envs, *feat].
        def view2d(data: torch.Tensor) -> torch.Tensor:
            return data.reshape(per_env_len, num_envs, *data.shape[1:])

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

        # Effective horizon = index of first done (+1), else full n.
        first_done = torch.argmax((all_dones > 0).float(), dim=-1)
        no_dones = all_dones.sum(dim=-1) == 0
        first_done = torch.where(no_dones, torch.full_like(first_done, n - 1), first_done)
        effective_n_steps = (first_done + 1).unsqueeze(-1).to(torch.long)
        final_t = window_t.gather(1, first_done.unsqueeze(-1)).squeeze(-1)  # [B]

        # Per-step costs over the same window, needed by BOTH the cost aggregation below and the
        # survival shaping. Gathered once here rather than twice.
        all_costs = None
        if "costs" in self._data:
            costs2d = view2d(self._data["costs"])  # [T, E, C]
            all_costs = costs2d[window_t, e_exp]  # [B, n, C]

        survival_discount = None
        if self.survival_lambda is None:
            n_step_rewards = (all_rewards * done_masks * discounts.view(1, -1)).sum(dim=-1, keepdim=True)
        else:
            # Stochastic decision horizons (arXiv:2602.04599): a continuation probability
            # alpha(s,a) shapes reward and discount as r~ = alpha r, gamma~ = gamma alpha, so the
            # n-step return becomes sum_k u_k r~_k with u_k = prod_{j<k} gamma~_j.
            #
            # Computed HERE, at sample time, and NOT precomputed at rollout time as the paper does.
            # `survival_lambda` is scheduled during training, and this buffer holds up to 1M
            # transitions -- baking alpha in at insertion would make every replayed sample carry a
            # stale scale. Reading the stored per-step costs each time uses the live one. Do not
            # "optimise" this back into the rollout path.
            if all_costs is None:
                raise RuntimeError(
                    "survival shaping needs per-step costs, but this buffer stores none. "
                    "The algorithm must be a safe-RL one (it must call store_transition with `cost`)."
                )
            alpha = exponential_continuation(all_costs, self.survival_lambda)  # [B, n]
            gamma_tilde = self.gamma * alpha  # [B, n]
            # inclusive prod_{j<=k}; exclusive u_k = prod_{j<k} with u_0 = 1
            ubar = torch.cumprod(gamma_tilde, dim=-1)
            u = torch.cat([torch.ones_like(ubar[..., :1]), ubar[..., :-1]], dim=-1)
            n_step_rewards = (all_rewards * done_masks * u * alpha).sum(dim=-1, keepdim=True)
            # The bootstrap factor at the (possibly truncated) horizon: prod_{j<n_eff} gamma~_j.
            # [B, 1] and NOT [B]: the standard-critic target multiplies [batch, 1] tensors, where a
            # [batch] discount would broadcast to [batch, batch] instead of failing loudly.
            survival_discount = ubar.gather(1, first_done.unsqueeze(-1))

        # Aggregate costs over the same window as rewards (safe RL): the cost
        # Bellman backup must see the n-step discounted cost sum, not the 1-step
        # cost, or reward and cost critics would learn on mismatched horizons.
        n_step_costs = None
        if all_costs is not None:
            # `cost_gamma` defaults to `gamma`, so this is bit-identical to the shared-discount
            # version for every existing algorithm. FH-DCMPO sets it to 1.0 to get the plain
            # undiscounted window sum, matching the units the cost limit is actually stated in.
            cost_discounts = (
                discounts if self.cost_gamma == self.gamma else torch.pow(self.cost_gamma, step_offsets)
            )
            n_step_costs = (all_costs * done_masks.unsqueeze(-1) * cost_discounts.view(1, -1, 1)).sum(dim=1)

        batch: dict[str, torch.Tensor] = {}
        for name, data in self._data.items():
            d2d = view2d(data)
            if name == "rewards":
                value = n_step_rewards
            elif name == "costs" and n_step_costs is not None:
                value = n_step_costs
            elif name in ("next_observations", "next_critic_observations", "dones", "bootstrap"):
                # Taken at the (possibly truncated) horizon step.
                value = d2d[final_t, e_idx]
            else:
                # State/action/critic-obs taken at the start step.
                value = d2d[t_idx, e_idx]
            batch[name] = self._process_undo(name, value.clone())

        batch["effective_n_steps"] = effective_n_steps
        if survival_discount is not None:
            batch["survival_discount"] = survival_discount

        if self.cost_n_step > n and "costs" in self._data:
            batch.update(self._gather_cost_window(t_idx, e_idx, per_env_len, view2d))
        return batch

    def _gather_cost_window(self, t_idx, e_idx, per_env_len, view2d) -> dict[str, torch.Tensor]:
        """Per-step data for a TD(lambda) cost target over ``cost_n_step`` steps.

        Returns, for j = 1..L (L = ``cost_n_step``), everything needed to form the j-step
        UNDISCOUNTED cost return ``G_j = sum_{k<j} c_k + m_j * Z_c(s_{t+j})``:

        * ``cost_window_returns`` [B, L] -- the realized partial sums ``sum_{k<j} c_k``, frozen
          once the episode ends so later components do not accumulate the next episode's cost;
        * ``cost_window_next_obs`` [B, L, obs] -- the critic observation at ``t+j``;
        * ``cost_window_mask`` [B, L] -- 1 while the episode is still running after step j, else 0.

        Note what the mask buys past an episode boundary: ``m_j = 0`` there, so ``G_j`` is the
        exact realized remaining episodic cost -- a pure Monte-Carlo target, with no bootstrap and
        no critic error in it at all. For a fixed finite horizon that is the ground truth, which is
        precisely the spread the bootstrapped target cannot manufacture on its own.

        Undiscounted by construction: FH-DCMPO is the only consumer and its cost channel has
        ``gamma_c = 1``. ``cost_gamma`` is deliberately not applied here -- a discounted TD(lambda)
        cost target is not a thing this method wants, and silently honouring it would hide that.
        """
        L = self.cost_n_step
        batch_size = t_idx.numel()
        offs = torch.arange(L, device=self.device)
        wt = (t_idx.unsqueeze(-1) + offs) % per_env_len  # [B, L]
        ee = e_idx.unsqueeze(-1).expand(batch_size, L)

        costs_w = view2d(self._data["costs"])[wt, ee]  # [B, L, C]
        costs_w = costs_w.sum(dim=-1) if costs_w.dim() == 3 else costs_w  # single-constraint
        dones_w = view2d(self._data["dones"])[wt, ee].squeeze(-1)  # [B, L]

        # alive_j = 1 while step j is still inside the ORIGINAL episode (done not yet seen).
        dones_shifted = torch.cat([torch.zeros_like(dones_w[..., :1]), dones_w[..., :-1]], dim=-1)
        alive = torch.cumprod(1.0 - dones_shifted, dim=-1)  # [B, L]
        returns = torch.cumsum(costs_w * alive, dim=-1)  # [B, L], frozen after the boundary

        key = "next_critic_observations" if "next_critic_observations" in self._data else "next_observations"
        next_obs_w = view2d(self._data[key])[wt, ee]  # [B, L, obs]

        # still_j = 1 only if the episode survives step j itself -> bootstrap there, else pure MC.
        still = torch.cumprod(1.0 - dones_w, dim=-1)  # [B, L]

        out = {
            "cost_window_returns": returns,
            "cost_window_next_obs": next_obs_w,
            "cost_window_mask": still,
        }
        if self.cost_window_extras:
            # Off-policy mismatch diagnostic (flag-gated): per-step behavior actions and the
            # observations they were taken from, so the consumer can evaluate
            # log pi_current(a_j | s_j) along the stored window. `alive` marks the steps whose
            # cost is still inside the return (mismatch past the boundary is irrelevant).
            out["cost_window_actions"] = view2d(self._data["actions"])[wt, ee]  # [B, L, A]
            out["cost_window_obs"] = view2d(self._data["observations"])[wt, ee]  # [B, L, obs]
            out["cost_window_alive"] = alive  # [B, L]
            # Age of the window START in stored transitions, counted back from the write head
            # (age 0 = most recently written slot). Ring arithmetic, valid even after wrap.
            flat = wt[:, 0] * self.num_envs + e_idx
            head = (self._ptr - 1) % self._max_size
            out["cost_window_age"] = ((head - flat) % self._max_size).to(torch.float32)  # [B]
            if "behavior_log_prob" in self._data:
                out["cost_window_blp"] = view2d(self._data["behavior_log_prob"])[wt, ee].squeeze(-1)  # [B, L]
            if "policy_version" in self._data:
                out["cost_window_version"] = view2d(self._data["policy_version"])[wt, ee].squeeze(-1)  # [B, L]
        return out

    # ------------------------------------------------------------------
    # Hazard-stratified sampling
    # ------------------------------------------------------------------

    def _draw_positions_wor(self, pool_size: int, k: int) -> torch.Tensor:
        """``k`` distinct positions uniform over ``[0, pool_size)``.

        Expected O(k) when pool_size >> k. ``torch.randperm(pool_size)`` would be O(M)
        with M up to ~1e6 per sample, so instead draw-and-dedupe with bounded top-up
        rounds. The randperm fallback is only reachable when collisions are frequent,
        which only happens when pool_size is within a small factor of k -- so its O(M)
        cost is bounded in practice.
        """
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if k >= pool_size:
            return torch.arange(pool_size, device=self.device)

        drawn = torch.unique(torch.randint(0, pool_size, (k,), device=self.device))
        for _ in range(_WOR_MAX_ROUNDS):
            need = k - drawn.numel()
            if need <= 0:
                break
            extra = torch.randint(0, pool_size, (2 * need + 8,), device=self.device)
            drawn = torch.unique(torch.cat([drawn, extra]))
        if drawn.numel() < k:
            return torch.randperm(pool_size, device=self.device)[:k]
        # torch.unique returns SORTED values. Truncating with drawn[:k] would take the k
        # smallest positions, and pool position correlates with insertion recency -- a
        # silent bias toward stale transitions. Shuffle before truncating.
        return drawn[torch.randperm(drawn.numel(), device=self.device)[:k]]

    def _stratified_counts(self, batch_size: int) -> tuple[int, int] | None:
        """Actual (hazard, safe) counts for a stratified batch, or None if infeasible.

        When a stratum cannot supply its target, all of it is taken once (never
        duplicated) and the remainder comes from the other stratum.
        """
        n_hazard, n_safe = self._counts[HAZARD], self._counts[SAFE]
        total = n_hazard + n_safe
        if total == 0 or batch_size > total:
            return None

        target_hazard = int(round(batch_size * self._hazard_fraction))
        count_hazard = min(target_hazard, n_hazard)
        count_safe = min(batch_size - target_hazard, n_safe)

        # Whichever stratum is short, the other has headroom because batch_size <= total.
        deficit = batch_size - count_hazard - count_safe
        if deficit > 0:
            take = min(deficit, n_safe - count_safe)
            count_safe += take
            deficit -= take
        if deficit > 0:
            take = min(deficit, n_hazard - count_hazard)
            count_hazard += take
            deficit -= take
        assert deficit == 0
        return count_hazard, count_safe

    def _stratified_weights(self, batch_size: int, count_hazard: int, count_safe: int) -> torch.Tensor:
        """Per-transition cost importance weights, in stratum-blocked order.

        Under uniform replay an element is drawn with expected multiplicity B/N; under
        stratified sampling an element of stratum c is drawn with B_c/N_c. The
        correction is the ratio:

            w_c = (B/N) / (B_c/N_c) = (N_c/N) / (B_c/B)

        Deliberately NOT normalized so max(w) == 1: the point is to preserve the
        absolute expectation of the uniform-replay objective, not just its shape.
        """
        n_hazard, n_safe = self._counts[HAZARD], self._counts[SAFE]
        total = n_hazard + n_safe
        weights = torch.ones(batch_size, 1, device=self.device)
        if count_hazard > 0:
            weights[:count_hazard] = (n_hazard / total) / (count_hazard / batch_size)
        if count_safe > 0:
            weights[count_hazard:] = (n_safe / total) / (count_safe / batch_size)
        return weights

    def _draw_valid_flat(
        self, pool_id: int, k: int, already: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Up to ``k`` distinct flat indices from ``pool_id`` that are valid n-step starts.

        May return fewer than ``k``; the caller rebalances. ``already`` seeds the
        dedupe set so a top-up draw cannot repeat an index the caller already holds.
        """
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        pool_size = self._counts[pool_id]
        if k <= 0 or pool_size == 0:
            return empty

        drawn = empty if already is None else already
        target = k + (0 if already is None else already.numel())
        for _ in range(_NSTEP_MAX_ROUNDS):
            need = target - drawn.numel()
            if need <= 0:
                break
            positions = self._draw_positions_wor(pool_size, min(pool_size, 2 * need + 8))
            candidates = self._pools[pool_id][positions]
            candidates = candidates[self._is_valid_start(candidates // self.num_envs)]
            drawn = torch.unique(torch.cat([drawn, candidates]))

        if already is not None and already.numel() > 0:
            drawn = drawn[~torch.isin(drawn, already)]
        if drawn.numel() <= k:
            return drawn
        # Same torch.unique-is-sorted trap as in _draw_positions_wor.
        return drawn[torch.randperm(drawn.numel(), device=self.device)[:k]]

    def _draw_valid_flat_stratified(
        self, batch_size: int, count_hazard: int, count_safe: int
    ) -> tuple[torch.Tensor, int, int] | None:
        """Stratified flat indices restricted to valid n-step starts.

        A stratum can come up short *after* validity filtering even when its pool count
        said otherwise, so rebalance a second time against the actual yield. Returns
        None if the batch still cannot be filled, which sends the caller to the plain
        uniform n-step path.
        """
        hazard = self._draw_valid_flat(HAZARD, count_hazard)
        safe = self._draw_valid_flat(SAFE, count_safe)

        deficit = batch_size - hazard.numel() - safe.numel()
        if deficit > 0:
            extra = self._draw_valid_flat(SAFE, deficit, already=safe)
            safe = torch.cat([safe, extra])
            deficit -= extra.numel()
        if deficit > 0:
            extra = self._draw_valid_flat(HAZARD, deficit, already=hazard)
            hazard = torch.cat([hazard, extra])
            deficit -= extra.numel()
        if deficit > 0:
            return None
        return torch.cat([hazard, safe]), hazard.numel(), safe.numel()

    def _sample_stratified(self, batch_size: int) -> dict[str, torch.Tensor] | None:
        """Hazard-stratified batch over the existing storage, or None to fall back."""
        counts = self._stratified_counts(batch_size)
        if counts is None:
            return None
        count_hazard, count_safe = counts

        if self.n_step > 1:
            repaired = self._draw_valid_flat_stratified(batch_size, count_hazard, count_safe)
            if repaired is None:
                return None
            flat, count_hazard, count_safe = repaired
        else:
            flat = torch.cat(
                [
                    self._pools[HAZARD][self._draw_positions_wor(self._counts[HAZARD], count_hazard)],
                    self._pools[SAFE][self._draw_positions_wor(self._counts[SAFE], count_safe)],
                ]
            )

        # Weights use the ACTUAL composition, never the configured target. Indices and
        # weights are permuted together -- that pairing must not drift.
        weights = self._stratified_weights(batch_size, count_hazard, count_safe)
        perm = torch.randperm(batch_size, device=self.device)
        flat = flat[perm]
        weights = weights[perm]

        if self.n_step > 1:
            batch = self._gather_n_step(flat // self.num_envs, flat % self.num_envs)
        else:
            batch = {
                name: self._process_undo(name, data[flat].clone()) for name, data in self._data.items()
            }

        assert bool(torch.isfinite(weights).all()), "non-finite cost_is_weights"
        assert bool((weights >= 0).all()), "negative cost_is_weights"
        if _REPLAY_DEBUG:
            # E[w] == 1 exactly; catches almost any weight-formula or pairing mistake.
            assert abs(weights.mean().item() - 1.0) < 1e-4

        batch["cost_is_weights"] = weights
        n_hazard, n_safe = self._counts[HAZARD], self._counts[SAFE]
        total = max(n_hazard + n_safe, 1)
        self._last_stratified_info = {
            "replay_hazard_fraction_buffer": n_hazard / total,
            "replay_hazard_fraction_batch": count_hazard / batch_size,
            "replay_cost_is_weight_hazard": (
                (n_hazard / total) / (count_hazard / batch_size) if count_hazard else 0.0
            ),
            "replay_cost_is_weight_safe": (
                (n_safe / total) / (count_safe / batch_size) if count_safe else 0.0
            ),
            "replay_hazard_pool_size": float(n_hazard),
            "replay_safe_pool_size": float(n_safe),
        }
        return batch

    @property
    def last_stratified_info(self) -> dict[str, float]:
        """Composition metrics for the most recent stratified batch (empty if none)."""
        return self._last_stratified_info

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
        if self._hazard_fraction > 0.0:
            # Keep the pool tensors: they are sized by max_size and would just be
            # reallocated identically.
            self._counts = [0, 0]
            self._slot_pos.fill_(-1)
            self._slot_class.fill_(-1)

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
        # Pools are derived state and are not serialized (see _rebuild_pools).
        self._rebuild_pools()
