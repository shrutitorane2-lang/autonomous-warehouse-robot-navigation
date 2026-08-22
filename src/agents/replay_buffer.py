"""
Experience Replay Buffer for DQN.

Stores (state, action, reward, next_state, done) transitions.
Mini-batches are sampled uniformly at random to break temporal correlations
between consecutive transitions — a key stabilisation technique in DQN
(Mnih et al., 2015).

Design choices:
    - Circular buffer (deque) for O(1) append and bounded memory.
    - NumPy arrays for fast batch retrieval.
    - No dependencies beyond NumPy.
"""

from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity circular experience replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store. Oldest transitions are
        evicted when the buffer is full.
    obs_dim : int
        Dimensionality of the observation / state vector.
    seed : int, optional
        Random seed for reproducible sampling.
    """

    def __init__(
        self, capacity: int, obs_dim: int, seed: int = 42
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self._rng = np.random.default_rng(seed)

        # Pre-allocate NumPy arrays for speed
        self._states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

        self._ptr: int = 0    # Write pointer (circular)
        self._size: int = 0   # Current number of stored transitions

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a single transition.

        Parameters
        ----------
        state : np.ndarray, shape (obs_dim,)
        action : int
        reward : float
        next_state : np.ndarray, shape (obs_dim,)
        done : bool
        """
        self._states[self._ptr] = state
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._next_states[self._ptr] = next_state
        self._dones[self._ptr] = float(done)

        # Advance circular pointer
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random mini-batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        states : np.ndarray, shape (batch_size, obs_dim)
        actions : np.ndarray, shape (batch_size,)
        rewards : np.ndarray, shape (batch_size,)
        next_states : np.ndarray, shape (batch_size, obs_dim)
        dones : np.ndarray, shape (batch_size,)

        Raises
        ------
        ValueError
            If the buffer contains fewer than ``batch_size`` transitions.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has only {self._size} transitions; "
                f"requested batch_size={batch_size}."
            )
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """True when the buffer has been populated at least once."""
        return self._size > 0

    def can_sample(self, batch_size: int) -> bool:
        """Return True if a batch of the given size can be sampled."""
        return self._size >= batch_size
