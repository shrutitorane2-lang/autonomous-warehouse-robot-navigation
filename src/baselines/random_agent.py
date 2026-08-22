"""
Random Agent baseline.

Takes uniformly-random actions. Used as the lowest possible performance
baseline to sanity-check that RL agents learn something meaningful.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RandomAgent:
    """
    Random agent that selects actions uniformly at random.

    This baseline helps answer: "Is the RL agent doing better than chance?"

    Parameters
    ----------
    action_space_n : int
        Number of discrete actions (4 for warehouse navigation).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, action_space_n: int = 4, seed: Optional[int] = None) -> None:
        self.action_space_n = action_space_n
        self._rng = np.random.default_rng(seed)
        self.name = "Random Agent"

    def select_action(self, obs: Optional[np.ndarray] = None) -> int:
        """
        Select a random action.

        Parameters
        ----------
        obs : np.ndarray, optional
            Observation (ignored by this agent, kept for API consistency).

        Returns
        -------
        int
            Random action in [0, action_space_n).
        """
        return int(self._rng.integers(self.action_space_n))

    def run_episode(self, env) -> dict:
        """
        Run a single episode using random actions.

        Parameters
        ----------
        env : WarehouseEnv
            Environment to interact with.

        Returns
        -------
        dict
            Episode metrics: total_reward, steps, collisions, success.
        """
        obs, info = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = self.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        return {
            "total_reward": total_reward,
            "steps": info["step_count"],
            "collisions": info["collision_count"],
            "success": terminated and not truncated,
            "path": info["path"],
        }

    def evaluate(self, env, num_episodes: int = 100, seed: Optional[int] = None) -> dict:
        """
        Evaluate over multiple episodes and aggregate metrics.

        Parameters
        ----------
        env : WarehouseEnv
        num_episodes : int
        seed : int, optional
            Re-seed for reproducibility.

        Returns
        -------
        dict
            Aggregated evaluation metrics.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        rewards, steps, collisions, successes = [], [], [], []

        for ep in range(num_episodes):
            result = self.run_episode(env)
            rewards.append(result["total_reward"])
            steps.append(result["steps"])
            collisions.append(result["collisions"])
            successes.append(float(result["success"]))

        metrics = {
            "algorithm": self.name,
            "num_episodes": num_episodes,
            "success_rate": float(np.mean(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "avg_steps": float(np.mean(steps)),
            "avg_collisions": float(np.mean(collisions)),
        }
        logger.info(
            f"Random Agent — success: {metrics['success_rate']:.1%}, "
            f"avg_reward: {metrics['avg_reward']:.1f}, "
            f"avg_steps: {metrics['avg_steps']:.1f}"
        )
        return metrics
