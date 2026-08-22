"""
Tabular Q-Learning agent.

Implements the classic Q-learning algorithm (Watkins & Dayan, 1992):

    Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]

where:
    α  = learning rate          (how fast we update Q-values)
    γ  = discount factor        (importance of future rewards)
    ε  = exploration rate       (probability of random action)

The state is discretised from the continuous observation vector
because the standard Q-table requires hashable discrete states.

State discretisation strategy
------------------------------
The 4-D or 8-D observation is normalised to [0,1]. We bucket each
dimension into ``n_bins`` equal-width bins and convert to an integer
tuple, making it suitable as a dictionary key.

Exploration
-----------
ε-greedy with exponential decay:
    ε_t = max(ε_min, ε_0 × decay^t)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.utils.config import Config, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Number of discrete bins per observation dimension
_N_BINS = 10


class QLearningAgent:
    """
    Tabular Q-Learning with epsilon-greedy exploration and epsilon decay.

    Parameters
    ----------
    config : Config, optional
        Project config (loaded from config.yaml if omitted).
    learning_rate : float, optional
        Override alpha.
    discount_factor : float, optional
        Override gamma.
    epsilon_start : float, optional
        Override starting epsilon.
    epsilon_end : float, optional
        Override minimum epsilon.
    epsilon_decay : float, optional
        Override per-episode multiplicative decay.
    n_actions : int
        Number of discrete actions (4).
    n_bins : int
        Number of bins per observation dimension for state discretisation.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        learning_rate: Optional[float] = None,
        discount_factor: Optional[float] = None,
        epsilon_start: Optional[float] = None,
        epsilon_end: Optional[float] = None,
        epsilon_decay: Optional[float] = None,
        n_actions: int = 4,
        n_bins: int = _N_BINS,
    ) -> None:
        cfg = config if config is not None else load_config()
        ql = cfg.q_learning

        self.alpha: float = learning_rate if learning_rate is not None else ql.learning_rate
        self.gamma: float = discount_factor if discount_factor is not None else ql.discount_factor
        self.epsilon: float = epsilon_start if epsilon_start is not None else ql.epsilon_start
        self.epsilon_end: float = epsilon_end if epsilon_end is not None else ql.epsilon_end
        self.epsilon_decay: float = epsilon_decay if epsilon_decay is not None else ql.epsilon_decay
        self.n_actions: int = n_actions
        self.n_bins: int = n_bins

        # Q-table: state_tuple → [Q(s,a0), Q(s,a1), Q(s,a2), Q(s,a3)]
        self.q_table: Dict[Tuple, np.ndarray] = {}

        self._rng = np.random.default_rng(42)
        self.name = "Q-Learning"

        # Training history (populated by train_episode)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_successes: list[bool] = []
        self.episode_collisions: list[int] = []
        self.epsilon_history: list[float] = []
        self.total_episodes: int = 0

    # ------------------------------------------------------------------
    # State discretisation
    # ------------------------------------------------------------------

    def _discretise(self, obs: np.ndarray) -> Tuple:
        """
        Convert a continuous observation vector into a discrete tuple
        suitable for use as a dictionary key.

        Each dimension is bucketed into [0, n_bins-1].
        """
        bins = np.clip(
            (obs * self.n_bins).astype(int), 0, self.n_bins - 1
        )
        return tuple(bins.tolist())

    def _get_q_values(self, state: Tuple) -> np.ndarray:
        """Return Q-values for the given state, initialising to zeros if new."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=np.float64)
        return self.q_table[state]

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        ε-greedy action selection.

        Parameters
        ----------
        obs : np.ndarray
            Current observation vector.
        deterministic : bool
            If True, always select the greedy action (evaluation mode).

        Returns
        -------
        int
            Selected action.
        """
        if not deterministic and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))

        state = self._discretise(obs)
        q_values = self._get_q_values(state)
        return int(np.argmax(q_values))

    def get_q_values_for_obs(self, obs: np.ndarray) -> np.ndarray:
        """Return Q-values for a given observation (for display / debugging)."""
        state = self._discretise(obs)
        return self._get_q_values(state).copy()

    # ------------------------------------------------------------------
    # Q-value update (Bellman equation)
    # ------------------------------------------------------------------

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> float:
        """
        Apply the Q-learning Bellman update:

            Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]

        Parameters
        ----------
        obs : np.ndarray
        action : int
        reward : float
        next_obs : np.ndarray
        done : bool
            True if the episode ended.

        Returns
        -------
        float
            TD error magnitude (useful for monitoring convergence).
        """
        state = self._discretise(obs)
        next_state = self._discretise(next_obs)

        q_current = self._get_q_values(state)[action]
        q_next_max = 0.0 if done else float(np.max(self._get_q_values(next_state)))

        td_target = reward + self.gamma * q_next_max
        td_error = td_target - q_current
        self.q_table[state][action] = q_current + self.alpha * td_error
        return abs(td_error)

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Apply one step of epsilon decay (call once per episode)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Episode training loop
    # ------------------------------------------------------------------

    def train_episode(self, env) -> dict[str, Any]:
        """
        Run one full training episode.

        Parameters
        ----------
        env : WarehouseEnv

        Returns
        -------
        dict
            Episode metrics: reward, length, success, collisions, epsilon.
        """
        obs, _ = env.reset()
        total_reward = 0.0
        total_td_error = 0.0
        steps = 0
        done = False

        while not done:
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            td_err = self.update(obs, action, reward, next_obs, done)
            total_td_error += td_err
            total_reward += reward
            steps += 1
            obs = next_obs

        self.decay_epsilon()
        self.total_episodes += 1

        result: dict[str, Any] = {
            "episode": self.total_episodes,
            "total_reward": total_reward,
            "steps": steps,
            "success": terminated and not truncated,
            "collisions": info["collision_count"],
            "epsilon": self.epsilon,
            "avg_td_error": total_td_error / max(steps, 1),
            "q_table_size": len(self.q_table),
        }

        # Accumulate history
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.episode_successes.append(result["success"])
        self.episode_collisions.append(info["collision_count"])
        self.epsilon_history.append(self.epsilon)

        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, env, num_episodes: int = 100) -> dict[str, Any]:
        """
        Evaluate the trained policy (no exploration).

        Parameters
        ----------
        env : WarehouseEnv
        num_episodes : int

        Returns
        -------
        dict
        """
        rewards, steps, collisions, successes = [], [], [], []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            rewards.append(total_reward)
            steps.append(info["step_count"])
            collisions.append(info["collision_count"])
            successes.append(float(terminated and not truncated))

        return {
            "algorithm": self.name,
            "num_episodes": num_episodes,
            "success_rate": float(np.mean(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "avg_steps": float(np.mean(steps)),
            "avg_collisions": float(np.mean(collisions)),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save Q-table and training state to disk.

        Parameters
        ----------
        path : str or Path
            File path for the pickle file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "q_table": self.q_table,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "n_bins": self.n_bins,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_successes": self.episode_successes,
            "episode_collisions": self.episode_collisions,
            "epsilon_history": self.epsilon_history,
            "total_episodes": self.total_episodes,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
        logger.info(f"Q-Learning model saved -> {path}")

    @classmethod
    def load(cls, path: str | Path, config: Optional[Config] = None) -> "QLearningAgent":
        """
        Load a previously saved Q-Learning agent.

        Parameters
        ----------
        path : str or Path
        config : Config, optional

        Returns
        -------
        QLearningAgent
        """
        path = Path(path)
        with open(path, "rb") as fh:
            state = pickle.load(fh)

        agent = cls(config=config, n_bins=state["n_bins"])
        agent.q_table = state["q_table"]
        agent.alpha = state["alpha"]
        agent.gamma = state["gamma"]
        agent.epsilon = state["epsilon"]
        agent.epsilon_end = state["epsilon_end"]
        agent.epsilon_decay = state["epsilon_decay"]
        agent.episode_rewards = state.get("episode_rewards", [])
        agent.episode_lengths = state.get("episode_lengths", [])
        agent.episode_successes = state.get("episode_successes", [])
        agent.episode_collisions = state.get("episode_collisions", [])
        agent.epsilon_history = state.get("epsilon_history", [])
        agent.total_episodes = state.get("total_episodes", 0)
        logger.info(
            f"Q-Learning model loaded <- {path} "
            f"(Q-table size: {len(agent.q_table)})"
        )
        return agent
