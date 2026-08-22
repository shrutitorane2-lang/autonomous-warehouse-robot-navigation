"""
Tests for BFS and A* pathfinding baselines.

Covers:
    - Path found in simple grid
    - Path never crosses obstacles
    - Unreachable environments handled gracefully
    - A* with different heuristics
    - Both find equal-length optimal paths on unit-cost grids
    - Random agent stays within action bounds
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.astar import AStarAgent
from src.baselines.bfs import BFSAgent
from src.baselines.random_agent import RandomAgent
from src.environment.warehouse_env import (
    CELL_FREE,
    CELL_OBSTACLE,
    CELL_ROBOT,
    CELL_GOAL,
    WarehouseEnv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_simple_grid():
    """
    5x5 grid with clear corridor:
    S . . . .
    . X X X .
    . X . X .
    . X . X .
    . . . . G
    """
    grid = np.zeros((5, 5), dtype=np.int32)
    for r, c in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 3)]:
        grid[r, c] = CELL_OBSTACLE
    grid[0, 0] = CELL_ROBOT
    grid[4, 4] = CELL_GOAL
    return grid, (0, 0), (4, 4)


def _make_blocked_grid():
    """5x5 grid where goal is completely surrounded."""
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[0, 0] = CELL_ROBOT
    grid[4, 4] = CELL_GOAL
    # Wall off the goal
    for r, c in [(3, 4), (4, 3), (3, 3)]:
        grid[r, c] = CELL_OBSTACLE
    # Block remaining access
    for c in range(5):
        grid[3, c] = CELL_OBSTACLE
    return grid, (0, 0), (4, 4)


@pytest.fixture
def bfs():
    return BFSAgent()


@pytest.fixture
def astar_manhattan():
    return AStarAgent(heuristic="manhattan")


@pytest.fixture
def astar_euclidean():
    return AStarAgent(heuristic="euclidean")


@pytest.fixture
def random_agent():
    return RandomAgent(seed=0)


# ---------------------------------------------------------------------------
# BFS tests
# ---------------------------------------------------------------------------

class TestBFS:

    def test_finds_path_in_open_grid(self, bfs):
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[0, 0] = CELL_ROBOT
        grid[4, 4] = CELL_GOAL
        actions = bfs.find_path(grid, (0, 0), (4, 4))
        assert actions is not None
        assert len(actions) > 0

    def test_path_is_optimal_length(self, bfs):
        """BFS must find shortest path (Manhattan distance in open grid)."""
        grid = np.zeros((5, 5), dtype=np.int32)
        path = bfs.find_path(grid, (0, 0), (4, 4))
        assert path is not None
        assert len(path) == 8  # min steps from (0,0) to (4,4) on 5x5

    def test_path_avoids_obstacles(self, bfs):
        grid, start, goal = _make_simple_grid()
        actions = bfs.find_path(grid, start, goal)
        assert actions is not None

        # Replay path and check no step lands on an obstacle
        r, c = start
        for action in actions:
            from src.environment.warehouse_env import ACTION_DELTAS
            dr, dc = ACTION_DELTAS[action]
            r, c = r + dr, c + dc
            assert grid[r, c] != CELL_OBSTACLE, f"BFS path crosses obstacle at ({r},{c})"

    def test_returns_none_for_unreachable_goal(self, bfs):
        grid, start, goal = _make_blocked_grid()
        actions = bfs.find_path(grid, start, goal)
        assert actions is None, "BFS should return None when goal is unreachable"

    def test_run_episode_returns_correct_keys(self, bfs):
        env = WarehouseEnv(grid_size=8, obstacle_density=0.1, seed=42)
        env.reset()
        result = bfs.run_episode(env)
        assert "success" in result
        assert "steps" in result
        assert "collisions" in result
        assert "total_reward" in result
        assert "path" in result

    def test_bfs_achieves_success_in_reachable_env(self, bfs):
        """BFS should always succeed in solvable environments."""
        successes = 0
        rng = np.random.default_rng(42)
        for _ in range(20):
            seed = int(rng.integers(10000))
            env = WarehouseEnv(grid_size=8, obstacle_density=0.15, seed=seed)
            env.reset()
            result = bfs.run_episode(env)
            if result["success"]:
                successes += 1
        assert successes >= 18, f"BFS should solve most solvable envs, got {successes}/20"


# ---------------------------------------------------------------------------
# A* tests
# ---------------------------------------------------------------------------

class TestAStar:

    def test_finds_path_in_open_grid(self, astar_manhattan):
        grid = np.zeros((5, 5), dtype=np.int32)
        actions = astar_manhattan.find_path(grid, (0, 0), (4, 4))
        assert actions is not None
        assert len(actions) == 8

    def test_astar_path_avoids_obstacles(self, astar_manhattan):
        grid, start, goal = _make_simple_grid()
        actions = astar_manhattan.find_path(grid, start, goal)
        assert actions is not None

        r, c = start
        from src.environment.warehouse_env import ACTION_DELTAS
        for action in actions:
            dr, dc = ACTION_DELTAS[action]
            r, c = r + dr, c + dc
            assert grid[r, c] != CELL_OBSTACLE

    def test_returns_none_for_unreachable(self, astar_manhattan):
        grid, start, goal = _make_blocked_grid()
        result = astar_manhattan.find_path(grid, start, goal)
        assert result is None

    def test_astar_euclidean_finds_same_length_as_manhattan(self):
        """On unit-cost grid both heuristics should give equal-length paths."""
        bfs_agent = BFSAgent()
        a_man = AStarAgent("manhattan")
        a_euc = AStarAgent("euclidean")

        grid = np.zeros((8, 8), dtype=np.int32)
        # Add some obstacles
        for r, c in [(1, 1), (2, 2), (3, 3), (4, 4), (2, 5)]:
            grid[r, c] = CELL_OBSTACLE

        start, goal = (0, 0), (7, 7)
        p_bfs = bfs_agent.find_path(grid, start, goal)
        p_man = a_man.find_path(grid, start, goal)
        p_euc = a_euc.find_path(grid, start, goal)

        assert p_bfs is not None and p_man is not None and p_euc is not None
        # All should find the same optimal path length
        assert len(p_bfs) == len(p_man) == len(p_euc)

    def test_astar_run_episode_keys(self, astar_manhattan):
        env = WarehouseEnv(grid_size=8, obstacle_density=0.1, seed=42)
        env.reset()
        result = astar_manhattan.run_episode(env)
        assert "success" in result
        assert "plan_time" in result
        assert "path_length" in result

    def test_invalid_heuristic_raises(self):
        with pytest.raises(ValueError):
            AStarAgent(heuristic="fake_heuristic")


# ---------------------------------------------------------------------------
# Random Agent tests
# ---------------------------------------------------------------------------

class TestRandomAgent:

    def test_action_always_in_range(self, random_agent):
        obs = np.zeros(4, dtype=np.float32)
        for _ in range(50):
            action = random_agent.select_action(obs)
            assert 0 <= action <= 3

    def test_run_episode_returns_metrics(self, random_agent):
        env = WarehouseEnv(grid_size=8, obstacle_density=0.1, seed=42)
        env.reset()
        result = random_agent.run_episode(env)
        assert "total_reward" in result
        assert "success" in result
        assert "collisions" in result

    def test_evaluate_returns_stats(self, random_agent):
        env = WarehouseEnv(grid_size=8, obstacle_density=0.1, seed=42)
        metrics = random_agent.evaluate(env, num_episodes=10)
        assert "success_rate" in metrics
        assert 0.0 <= metrics["success_rate"] <= 1.0
        assert "avg_steps" in metrics
