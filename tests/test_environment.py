"""
Tests for WarehouseEnv.

Covers:
    - Initialization
    - Valid start / goal placement
    - Obstacle validity
    - Boundary handling
    - Collision handling
    - Goal termination
    - Reachability guarantee
    - State vector shape and normalisation
    - Clone functionality
"""

from __future__ import annotations

import numpy as np
import pytest

from src.environment.warehouse_env import (
    CELL_FREE,
    CELL_GOAL,
    CELL_OBSTACLE,
    CELL_ROBOT,
    WarehouseEnv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env_8x8():
    env = WarehouseEnv(grid_size=8, obstacle_density=0.2, seed=42)
    env.reset()
    return env


@pytest.fixture
def env_10x10():
    env = WarehouseEnv(grid_size=10, obstacle_density=0.2, seed=7)
    env.reset()
    return env


@pytest.fixture
def env_advanced():
    env = WarehouseEnv(grid_size=8, obstacle_density=0.15, seed=1, state_type="advanced")
    env.reset()
    return env


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInitialization:

    def test_grid_size_8(self):
        env = WarehouseEnv(grid_size=8, seed=42)
        env.reset()
        assert env.get_grid().shape == (8, 8)

    def test_grid_size_10(self):
        env = WarehouseEnv(grid_size=10, seed=42)
        env.reset()
        assert env.get_grid().shape == (10, 10)

    def test_grid_size_15(self):
        env = WarehouseEnv(grid_size=15, seed=42)
        env.reset()
        assert env.get_grid().shape == (15, 15)

    def test_action_space_is_discrete_4(self, env_8x8):
        assert env_8x8.action_space.n == 4

    def test_observation_space_basic_shape(self):
        env = WarehouseEnv(grid_size=8, state_type="basic", seed=42)
        env.reset()
        assert env.observation_space.shape == (4,)

    def test_observation_space_advanced_shape(self, env_advanced):
        assert env_advanced.observation_space.shape == (8,)


# ---------------------------------------------------------------------------
# Start / Goal placement
# ---------------------------------------------------------------------------

class TestStartGoal:

    def test_start_is_valid_cell(self, env_8x8):
        start = env_8x8.get_start_pos()
        grid = env_8x8.get_grid()
        r, c = start
        assert 0 <= r < 8 and 0 <= c < 8
        # Start should not be an obstacle
        assert grid[r, c] != CELL_OBSTACLE

    def test_goal_is_valid_cell(self, env_8x8):
        goal = env_8x8.get_goal_pos()
        grid = env_8x8.get_grid()
        r, c = goal
        assert 0 <= r < 8 and 0 <= c < 8
        assert grid[r, c] != CELL_OBSTACLE

    def test_start_and_goal_differ(self, env_8x8):
        assert env_8x8.get_start_pos() != env_8x8.get_goal_pos()

    def test_start_in_upper_left_quadrant(self, env_8x8):
        r, c = env_8x8.get_start_pos()
        # Upper-left quadrant means row < n//2 and col < n//2
        assert r < 4 and c < 4

    def test_goal_in_lower_right_quadrant(self, env_8x8):
        r, c = env_8x8.get_goal_pos()
        assert r >= 4 and c >= 4


# ---------------------------------------------------------------------------
# Obstacle validity
# ---------------------------------------------------------------------------

class TestObstacles:

    def test_obstacle_density_roughly_correct(self, env_8x8):
        grid = env_8x8.get_grid()
        n_obstacles = np.sum(grid == CELL_OBSTACLE)
        n_free = 8 * 8 - 2  # Excluding start and goal
        # Allow ±5% tolerance
        expected = int(0.2 * n_free)
        assert abs(n_obstacles - expected) <= max(3, int(0.05 * n_free))

    def test_start_has_no_obstacle(self, env_8x8):
        grid = env_8x8.get_grid()
        r, c = env_8x8.get_start_pos()
        assert grid[r, c] != CELL_OBSTACLE

    def test_goal_has_no_obstacle(self, env_8x8):
        grid = env_8x8.get_grid()
        r, c = env_8x8.get_goal_pos()
        assert grid[r, c] != CELL_OBSTACLE

    def test_obstacles_are_within_bounds(self, env_8x8):
        grid = env_8x8.get_grid()
        positions = list(zip(*np.where(grid == CELL_OBSTACLE)))
        for r, c in positions:
            assert 0 <= r < 8 and 0 <= c < 8


# ---------------------------------------------------------------------------
# Boundary and collision handling
# ---------------------------------------------------------------------------

class TestBoundaryAndCollisions:

    def test_boundary_collision_stays_in_place(self):
        """Robot hitting a wall should stay in its current position."""
        env = WarehouseEnv(grid_size=8, obstacle_density=0.0, seed=42)
        env.reset()

        # Manually place robot at top-left corner
        env._robot_pos = (0, 0)
        old_pos = env._robot_pos

        # Action 0 = UP → should hit boundary
        obs, reward, terminated, truncated, info = env.step(0)
        assert env.get_robot_pos() == old_pos, "Robot should not move through boundary"
        assert reward < 0, "Boundary collision should give negative reward"
        assert info["collision_count"] >= 1

    def test_obstacle_collision_stays_in_place(self):
        """Robot hitting an obstacle should stay in its current position."""
        env = WarehouseEnv(grid_size=8, obstacle_density=0.0, seed=42)
        env.reset()

        # Force an obstacle at (1, 0)
        env._grid[1, 0] = CELL_OBSTACLE
        env._robot_pos = (0, 0)
        old_pos = env._robot_pos

        # Action 1 = DOWN → should hit obstacle at (1,0)
        _, reward, _, _, info = env.step(1)
        assert env.get_robot_pos() == old_pos, "Robot should not pass through obstacle"
        assert info["collision_count"] >= 1
        assert reward <= -20 + (-1)  # collision + step

    def test_collision_count_increments(self):
        env = WarehouseEnv(grid_size=8, obstacle_density=0.0, seed=42)
        env.reset()
        env._robot_pos = (0, 0)

        # Hit boundary twice
        env.step(0)  # UP — boundary
        env.step(2)  # LEFT — boundary
        assert env.get_collision_count() == 2


# ---------------------------------------------------------------------------
# Goal termination
# ---------------------------------------------------------------------------

class TestGoalTermination:

    def test_reaching_goal_terminates_episode(self):
        env = WarehouseEnv(grid_size=8, obstacle_density=0.0, seed=42)
        env.reset()

        # Place robot one step away from goal
        goal = env.get_goal_pos()
        env._robot_pos = (goal[0] - 1, goal[1])
        env._prev_dist = env._manhattan(env._robot_pos, goal)

        # Move DOWN → reach goal
        _, reward, terminated, truncated, info = env.step(1)
        assert terminated, "Episode should terminate on reaching goal"
        assert not truncated
        assert reward > 90, f"Reward at goal should include +100, got {reward}"

    def test_success_flag_set(self):
        env = WarehouseEnv(grid_size=8, obstacle_density=0.0, seed=42)
        env.reset()
        goal = env.get_goal_pos()
        env._robot_pos = (goal[0], goal[1] - 1)
        env._prev_dist = env._manhattan(env._robot_pos, goal)

        _, _, terminated, truncated, _ = env.step(3)  # RIGHT
        assert terminated and not truncated


# ---------------------------------------------------------------------------
# Reachability guarantee
# ---------------------------------------------------------------------------

class TestReachability:

    @pytest.mark.parametrize("seed", [1, 42, 100, 999, 1234])
    def test_generated_layout_is_reachable(self, seed):
        env = WarehouseEnv(grid_size=10, obstacle_density=0.25, seed=seed)
        env.reset()
        grid = env.get_grid()
        start = env.get_start_pos()
        goal = env.get_goal_pos()
        reachable = WarehouseEnv._bfs_reachable(grid, start, goal)
        assert reachable, f"Layout with seed={seed} is not reachable"

    def test_high_density_still_reachable(self):
        env = WarehouseEnv(grid_size=10, obstacle_density=0.35, seed=77)
        env.reset()
        grid = env.get_grid()
        assert WarehouseEnv._bfs_reachable(grid, env.get_start_pos(), env.get_goal_pos())


# ---------------------------------------------------------------------------
# Observation shape and normalisation
# ---------------------------------------------------------------------------

class TestObservations:

    def test_basic_obs_in_range(self):
        # Create a dedicated basic-state env (not affected by config default)
        env = WarehouseEnv(grid_size=8, state_type="basic", seed=42)
        obs, _ = env.reset()
        assert obs.shape == (4,)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_advanced_obs_in_range(self, env_advanced):
        obs, _ = env_advanced.reset()
        assert obs.shape == (8,)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_obs_dtype_is_float32(self, env_8x8):
        obs, _ = env_8x8.reset()
        assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# Clone and max_steps
# ---------------------------------------------------------------------------

class TestMisc:

    def test_clone_is_independent(self, env_8x8):
        env2 = env_8x8.clone()
        env_8x8.step(1)  # Move in original
        # Clone should not have changed
        assert env2.get_robot_pos() == env_8x8.get_start_pos() or \
               env2.get_robot_pos() != env_8x8.get_robot_pos()

    def test_max_steps_truncation(self):
        env = WarehouseEnv(grid_size=8, obstacle_density=0.0, seed=42)
        # Reduce max_steps for speed
        env.max_steps = 5
        env.reset()
        # Bounce against a wall repeatedly
        env._robot_pos = (0, 0)
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(0)  # keep hitting UP
            if terminated or truncated:
                break
        assert truncated, "Episode should truncate after max_steps"
