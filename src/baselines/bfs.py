"""
BFS (Breadth-First Search) pathfinding baseline.

Finds the shortest path (fewest steps) from the robot's current position
to the goal, ignoring movement costs. Used as an exact shortest-path
baseline to show what RL must compete against.

Key characteristic vs RL:
    - BFS has perfect knowledge of the grid.
    - BFS always finds the globally optimal path if one exists.
    - BFS cannot generalise; it replans from scratch every episode.
"""

from __future__ import annotations

import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from src.environment.warehouse_env import CELL_OBSTACLE, ACTION_DELTAS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BFSAgent:
    """
    Breadth-first search agent.

    Plans the shortest obstacle-free path at the start of each episode,
    then executes it step-by-step.

    Parameters
    ----------
    name : str
        Human-readable algorithm name.
    """

    def __init__(self) -> None:
        self.name = "BFS"

    def find_path(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[int]]:
        """
        Run BFS and return the sequence of actions from start to goal.

        Parameters
        ----------
        grid : np.ndarray
            The warehouse grid (CELL_OBSTACLE marks blocked cells).
        start : tuple
            (row, col) of the starting position.
        goal : tuple
            (row, col) of the goal position.

        Returns
        -------
        list of int or None
            Ordered list of actions (0-3) leading from start to goal,
            or None if the goal is unreachable.
        """
        rows, cols = grid.shape
        queue: deque[Tuple[Tuple[int, int], List[int]]] = deque([(start, [])])
        visited: set[Tuple[int, int]] = {start}

        while queue:
            (r, c), actions = queue.popleft()

            if (r, c) == goal:
                return actions

            for action, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and (nr, nc) not in visited
                    and grid[nr, nc] != CELL_OBSTACLE
                ):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), actions + [action]))

        return None  # No path found

    def run_episode(self, env) -> dict:
        """
        Run a single BFS-guided episode.

        Steps
        -----
        1. Reset environment and snapshot the grid/positions.
        2. Plan the full path via BFS.
        3. Execute each action in sequence.
        4. Return episode metrics.
        """
        obs, info = env.reset()
        grid = env.get_grid()
        start = env.get_start_pos()
        goal = env.get_goal_pos()

        t0 = time.perf_counter()
        actions = self.find_path(grid, start, goal)
        plan_time = time.perf_counter() - t0

        total_reward = 0.0
        success = False
        terminated = truncated = False

        if actions is None:
            # No path found; return failure immediately
            logger.warning("BFS found no path — environment may be unsolvable.")
            return {
                "total_reward": -env.max_steps,
                "steps": 0,
                "collisions": 0,
                "success": False,
                "path": [start],
                "plan_time": plan_time,
                "path_length": 0,
            }

        for action in actions:
            if terminated or truncated:
                break
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        success = terminated and not truncated

        return {
            "total_reward": total_reward,
            "steps": info["step_count"],
            "collisions": info["collision_count"],
            "success": success,
            "path": info["path"],
            "plan_time": plan_time,
            "path_length": len(actions),
        }

    def evaluate(self, env, num_episodes: int = 100, seed: Optional[int] = None) -> dict:
        """
        Evaluate BFS over multiple episodes with different random layouts.

        Parameters
        ----------
        env : WarehouseEnv
        num_episodes : int
        seed : int, optional

        Returns
        -------
        dict
            Aggregated evaluation metrics.
        """
        rewards, steps, collisions, successes, plan_times = [], [], [], [], []
        rng = np.random.default_rng(seed)

        for ep in range(num_episodes):
            ep_seed = int(rng.integers(1_000_000))
            env.reset(seed=ep_seed)
            result = self.run_episode(env)

            rewards.append(result["total_reward"])
            steps.append(result["steps"])
            collisions.append(result["collisions"])
            successes.append(float(result["success"]))
            plan_times.append(result["plan_time"])

        metrics = {
            "algorithm": self.name,
            "num_episodes": num_episodes,
            "success_rate": float(np.mean(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "avg_steps": float(np.mean(steps)),
            "avg_collisions": float(np.mean(collisions)),
            "avg_plan_time_ms": float(np.mean(plan_times)) * 1000,
        }
        logger.info(
            f"BFS — success: {metrics['success_rate']:.1%}, "
            f"avg_steps: {metrics['avg_steps']:.1f}, "
            f"plan_time: {metrics['avg_plan_time_ms']:.2f} ms"
        )
        return metrics
