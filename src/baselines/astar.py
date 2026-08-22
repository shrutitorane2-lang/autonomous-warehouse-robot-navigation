"""
A* (A-star) pathfinding baseline.

Uses the Manhattan distance heuristic to find the optimal path
more efficiently than BFS. Demonstrates informed search as a
second classical baseline.

Key characteristic vs BFS:
    - A* typically expands fewer nodes than BFS (O(b^d) → O(b^(d/2))).
    - Both find the same optimal solution for unit-cost grids.
    - Neither generalises across different layouts.

Key characteristic vs RL:
    - A* requires a perfect grid map (full observability).
    - RL can operate with partial / noisy state and generalise.
"""

from __future__ import annotations

import heapq
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.environment.warehouse_env import CELL_OBSTACLE, ACTION_DELTAS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def _euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


_HEURISTICS: Dict[str, Callable] = {
    "manhattan": _manhattan,
    "euclidean": _euclidean,
}


class AStarAgent:
    """
    A* search agent with configurable heuristic.

    Parameters
    ----------
    heuristic : str
        "manhattan" (default) or "euclidean".
    """

    def __init__(self, heuristic: str = "manhattan") -> None:
        if heuristic not in _HEURISTICS:
            raise ValueError(
                f"Unknown heuristic '{heuristic}'. Choose from {list(_HEURISTICS)}"
            )
        self._heuristic_fn = _HEURISTICS[heuristic]
        self.heuristic = heuristic
        self.name = f"A* ({heuristic})"

    def find_path(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[int]]:
        """
        Run A* and return the optimal sequence of actions.

        Parameters
        ----------
        grid : np.ndarray
        start : tuple
        goal : tuple

        Returns
        -------
        list of int or None
        """
        rows, cols = grid.shape
        h = self._heuristic_fn

        # Priority queue: (f_cost, g_cost, position, actions_so_far)
        # g = cost so far, h = heuristic, f = g + h
        heap: list = [(h(start, goal), 0.0, start, [])]
        # g_scores: best known cost to reach each cell
        g_scores: Dict[Tuple[int, int], float] = {start: 0.0}

        while heap:
            f, g, (r, c), actions = heapq.heappop(heap)

            if (r, c) == goal:
                return actions

            # Skip stale entries
            if g > g_scores.get((r, c), float("inf")):
                continue

            for action, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and grid[nr, nc] != CELL_OBSTACLE
                ):
                    new_g = g + 1.0  # Unit step cost
                    neighbour = (nr, nc)
                    if new_g < g_scores.get(neighbour, float("inf")):
                        g_scores[neighbour] = new_g
                        new_f = new_g + h(neighbour, goal)
                        heapq.heappush(heap, (new_f, new_g, neighbour, actions + [action]))

        return None  # No path found

    def run_episode(self, env) -> dict:
        """
        Run a single A*-guided episode.

        Returns
        -------
        dict
            Episode metrics including plan_time and path_length.
        """
        obs, info = env.reset()
        grid = env.get_grid()
        start = env.get_start_pos()
        goal = env.get_goal_pos()

        t0 = time.perf_counter()
        actions = self.find_path(grid, start, goal)
        plan_time = time.perf_counter() - t0

        total_reward = 0.0
        terminated = truncated = False

        if actions is None:
            logger.warning("A* found no path — environment may be unsolvable.")
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
        Evaluate A* over multiple episodes.

        Parameters
        ----------
        env : WarehouseEnv
        num_episodes : int
        seed : int, optional

        Returns
        -------
        dict
        """
        rewards, steps, collisions, successes, plan_times = [], [], [], [], []
        rng = np.random.default_rng(seed)

        for _ in range(num_episodes):
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
            f"A* ({self.heuristic}) — success: {metrics['success_rate']:.1%}, "
            f"avg_steps: {metrics['avg_steps']:.1f}, "
            f"plan_time: {metrics['avg_plan_time_ms']:.2f} ms"
        )
        return metrics
