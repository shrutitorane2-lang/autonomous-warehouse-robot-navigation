"""
Custom Gymnasium environment for warehouse robot navigation.

The warehouse is a configurable 2-D grid:
    S  = robot start
    G  = goal (package location)
    X  = obstacle (shelf / wall fixture)
    .  = free space

Supported grid sizes: 8x8, 10x10, 15x15, 20x20.
Random layouts are generated using a seeded RNG and guaranteed reachable
(BFS-verified from start to goal).

State representations
---------------------
basic (4-D):
    [robot_x, robot_y, goal_x, goal_y]  -- normalised to [0, 1]

advanced (8-D):
    [robot_x, robot_y, goal_x, goal_y,
     obstacle_up, obstacle_down, obstacle_left, obstacle_right]

Action space
------------
    0 = UP     (row - 1)
    1 = DOWN   (row + 1)
    2 = LEFT   (col - 1)
    3 = RIGHT  (col + 1)

Reward shaping (all values loaded from config):
    +100  reach goal
    - 1   every valid step (encourages shortest paths)
    + 2   step that reduces Manhattan distance to goal
    - 2   step that increases Manhattan distance to goal
    -20   collision with obstacle or grid boundary
    -10   episode times out (max_steps exceeded)
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.config import Config, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Grid cell integer codes
CELL_FREE = 0
CELL_OBSTACLE = 1
CELL_ROBOT = 2
CELL_GOAL = 3

# Action deltas: (row_delta, col_delta)
ACTION_DELTAS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
}
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


class WarehouseEnv(gym.Env):
    """
    Warehouse robot navigation environment.

    Parameters
    ----------
    config : Config, optional
        Project configuration object. Loaded from config.yaml when omitted.
    grid_size : int, optional
        Override the grid size from config.
    obstacle_density : float, optional
        Override the obstacle density from config.
    seed : int, optional
        Override the random seed.
    state_type : str, optional
        "basic" or "advanced".
    render_mode : str, optional
        "human" for matplotlib rendering; None disables rendering.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: Optional[Config] = None,
        grid_size: Optional[int] = None,
        obstacle_density: Optional[float] = None,
        seed: Optional[int] = None,
        state_type: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # ----- Load configuration -----
        self.cfg = config if config is not None else load_config()
        env_cfg = self.cfg.environment

        self.grid_size: int = grid_size if grid_size is not None else env_cfg.grid_size
        self.obstacle_density: float = (
            obstacle_density if obstacle_density is not None else env_cfg.obstacle_density
        )
        self.master_seed: int = seed if seed is not None else env_cfg.random_seed
        self.max_steps: int = env_cfg.max_steps
        self.state_type: str = state_type if state_type is not None else env_cfg.state_type

        # ----- Reward values -----
        r = self.cfg.rewards
        self.R_GOAL: float = r.goal_reached
        self.R_STEP: float = r.step_penalty
        self.R_CLOSER: float = r.closer_reward
        self.R_FARTHER: float = r.farther_penalty
        self.R_COLLISION: float = r.collision_penalty
        self.R_BOUNDARY: float = r.boundary_penalty
        self.R_TIMEOUT: float = r.timeout_penalty

        # ----- Spaces -----
        self.action_space = spaces.Discrete(4)

        obs_dim = 4 if self.state_type == "basic" else 8
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # ----- Runtime state (populated by reset()) -----
        self._grid: np.ndarray = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.int32
        )
        self._robot_pos: Tuple[int, int] = (0, 0)
        self._goal_pos: Tuple[int, int] = (self.grid_size - 1, self.grid_size - 1)
        self._start_pos: Tuple[int, int] = (0, 0)
        self._step_count: int = 0
        self._prev_dist: float = 0.0
        self._visited: set[Tuple[int, int]] = set()
        self._collision_count: int = 0
        self._episode_reward: float = 0.0
        self._path_taken: list[Tuple[int, int]] = []

        self.render_mode = render_mode
        self._rng = np.random.default_rng(self.master_seed)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        Generates a new random warehouse layout (or uses a fixed layout
        supplied via options["grid"]). Guarantees start→goal reachability.

        Parameters
        ----------
        seed : int, optional
            If provided, re-seeds the internal RNG.
        options : dict, optional
            "grid": np.ndarray — supply a fixed grid (for evaluation).
            "start": tuple — fixed start position.
            "goal": tuple — fixed goal position.

        Returns
        -------
        obs : np.ndarray
            Initial observation.
        info : dict
            Metadata (grid, start, goal).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}

        if "grid" in options:
            # Use externally supplied layout
            self._grid = np.array(options["grid"], dtype=np.int32)
            self._start_pos = options.get("start", self._find_cell(CELL_ROBOT))
            self._goal_pos = options.get("goal", self._find_cell(CELL_GOAL))
        else:
            # Generate a new random layout
            self._generate_grid()

        # Reset episode state
        self._robot_pos = self._start_pos
        self._step_count = 0
        self._visited = {self._robot_pos}
        self._collision_count = 0
        self._episode_reward = 0.0
        self._path_taken = [self._robot_pos]
        self._prev_dist = self._manhattan(self._robot_pos, self._goal_pos)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply an action and advance the environment by one timestep.

        Parameters
        ----------
        action : int
            One of {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}.

        Returns
        -------
        obs : np.ndarray
        reward : float
        terminated : bool   — episode ended naturally (goal or timeout penalty applied)
        truncated : bool    — episode cut off by time limit
        info : dict
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        dr, dc = ACTION_DELTAS[action]
        new_row = self._robot_pos[0] + dr
        new_col = self._robot_pos[1] + dc

        reward = self.R_STEP  # Base step cost
        terminated = False
        truncated = False

        # ----- Check boundary -----
        if not self._in_bounds(new_row, new_col):
            reward += self.R_BOUNDARY
            self._collision_count += 1
            # Robot stays in place
        # ----- Check obstacle -----
        elif self._grid[new_row, new_col] == CELL_OBSTACLE:
            reward += self.R_COLLISION
            self._collision_count += 1
            # Robot stays in place
        else:
            # Valid move — update position
            new_pos = (new_row, new_col)
            new_dist = self._manhattan(new_pos, self._goal_pos)
            old_dist = self._prev_dist

            # Distance-based shaping
            if new_dist < old_dist:
                reward += self.R_CLOSER
            elif new_dist > old_dist:
                reward += self.R_FARTHER

            self._robot_pos = new_pos
            self._prev_dist = new_dist
            self._visited.add(new_pos)
            self._path_taken.append(new_pos)

            # ----- Check goal -----
            if self._robot_pos == self._goal_pos:
                reward += self.R_GOAL
                terminated = True

        self._step_count += 1
        self._episode_reward += reward

        # ----- Check timeout -----
        if not terminated and self._step_count >= self.max_steps:
            reward += self.R_TIMEOUT
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render using the warehouse renderer (if render_mode is set)."""
        if self.render_mode == "human":
            from src.visualization.warehouse_renderer import WarehouseRenderer
            renderer = WarehouseRenderer(self)
            renderer.render_matplotlib()
        elif self.render_mode == "rgb_array":
            from src.visualization.warehouse_renderer import WarehouseRenderer
            renderer = WarehouseRenderer(self)
            return renderer.render_rgb_array()
        return None

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Grid Generation
    # ------------------------------------------------------------------

    def _generate_grid(self) -> None:
        """
        Generate a random obstacle layout guaranteed to be solvable.

        Strategy:
        1. Start with an empty grid.
        2. Place start (top-left area) and goal (bottom-right area) randomly.
        3. Randomly scatter obstacles at the configured density.
        4. Verify reachability via BFS; retry up to 100 times.
        """
        max_attempts = 100
        for attempt in range(max_attempts):
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

            # Candidate positions: avoid placing start/goal at same cell
            all_cells = [
                (r, c)
                for r in range(self.grid_size)
                for c in range(self.grid_size)
            ]
            # Start: upper-left quadrant
            start_candidates = [
                (r, c)
                for r, c in all_cells
                if r < self.grid_size // 2 and c < self.grid_size // 2
            ]
            # Goal: lower-right quadrant
            goal_candidates = [
                (r, c)
                for r, c in all_cells
                if r >= self.grid_size // 2 and c >= self.grid_size // 2
            ]

            start_idx = self._rng.integers(len(start_candidates))
            goal_idx = self._rng.integers(len(goal_candidates))
            start = start_candidates[start_idx]
            goal = goal_candidates[goal_idx]

            if start == goal:
                continue

            # Place obstacles
            num_obstacles = int(
                self.obstacle_density * (self.grid_size * self.grid_size - 2)
            )
            obstacle_candidates = [
                cell for cell in all_cells if cell not in {start, goal}
            ]
            shuffled = self._rng.permutation(len(obstacle_candidates))
            chosen = [obstacle_candidates[i] for i in shuffled[:num_obstacles]]
            for r, c in chosen:
                grid[r, c] = CELL_OBSTACLE

            grid[start] = CELL_ROBOT
            grid[goal] = CELL_GOAL

            # BFS reachability check
            if self._bfs_reachable(grid, start, goal):
                self._grid = grid
                self._start_pos = start
                self._goal_pos = goal
                return

        raise RuntimeError(
            f"Could not generate a solvable warehouse in {max_attempts} attempts. "
            "Try reducing obstacle_density."
        )

    @staticmethod
    def _bfs_reachable(
        grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> bool:
        """Return True if goal is reachable from start on the given grid."""
        rows, cols = grid.shape
        queue: deque[Tuple[int, int]] = deque([start])
        visited: set[Tuple[int, int]] = {start}
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in ACTION_DELTAS.values():
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and (nr, nc) not in visited
                    and grid[nr, nc] != CELL_OBSTACLE
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    # ------------------------------------------------------------------
    # Observation & Info
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector (normalised to [0,1])."""
        n = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
        rr, rc = self._robot_pos
        gr, gc = self._goal_pos

        obs = np.array(
            [rr / n, rc / n, gr / n, gc / n],
            dtype=np.float32,
        )

        if self.state_type == "advanced":
            # Local obstacle indicators (1.0 = blocked, 0.0 = passable)
            local = np.array(
                [
                    self._cell_blocked(rr - 1, rc),  # UP
                    self._cell_blocked(rr + 1, rc),  # DOWN
                    self._cell_blocked(rr, rc - 1),  # LEFT
                    self._cell_blocked(rr, rc + 1),  # RIGHT
                ],
                dtype=np.float32,
            )
            obs = np.concatenate([obs, local])

        return obs

    def _get_info(self) -> dict[str, Any]:
        """Return auxiliary episode information."""
        return {
            "robot_pos": self._robot_pos,
            "goal_pos": self._goal_pos,
            "step_count": self._step_count,
            "collision_count": self._collision_count,
            "episode_reward": self._episode_reward,
            "distance_to_goal": self._manhattan(self._robot_pos, self._goal_pos),
            "visited_cells": len(self._visited),
            "path": list(self._path_taken),
        }

    def _cell_blocked(self, row: int, col: int) -> float:
        """Return 1.0 if cell is out-of-bounds or an obstacle, else 0.0."""
        if not self._in_bounds(row, col):
            return 1.0
        return 1.0 if self._grid[row, col] == CELL_OBSTACLE else 0.0

    # ------------------------------------------------------------------
    # Helper Utilities
    # ------------------------------------------------------------------

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def _find_cell(self, cell_type: int) -> Tuple[int, int]:
        """Find the first occurrence of a cell type in the grid."""
        positions = list(zip(*np.where(self._grid == cell_type)))
        if not positions:
            raise ValueError(f"Cell type {cell_type} not found in grid.")
        return tuple(positions[0])  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public Utilities
    # ------------------------------------------------------------------

    def get_grid(self) -> np.ndarray:
        """Return a copy of the current grid."""
        return self._grid.copy()

    def get_robot_pos(self) -> Tuple[int, int]:
        return self._robot_pos

    def get_goal_pos(self) -> Tuple[int, int]:
        return self._goal_pos

    def get_start_pos(self) -> Tuple[int, int]:
        return self._start_pos

    def get_visited(self) -> set[Tuple[int, int]]:
        return set(self._visited)

    def get_path(self) -> list[Tuple[int, int]]:
        return list(self._path_taken)

    def get_collision_count(self) -> int:
        return self._collision_count

    def print_grid(self) -> None:
        """Pretty-print the grid to stdout (useful for debugging)."""
        symbols = {CELL_FREE: ".", CELL_OBSTACLE: "X", CELL_ROBOT: "S", CELL_GOAL: "G"}
        grid = self._grid.copy()
        # Show current robot position
        grid[self._robot_pos] = CELL_ROBOT
        print("\n".join(" ".join(symbols[c] for c in row) for row in grid))
        print()

    def clone(self) -> "WarehouseEnv":
        """
        Return a deep copy of the environment.
        Useful for planning algorithms that need to simulate ahead.
        """
        new_env = WarehouseEnv(
            config=self.cfg,
            grid_size=self.grid_size,
            obstacle_density=self.obstacle_density,
            seed=self.master_seed,
            state_type=self.state_type,
        )
        new_env._grid = self._grid.copy()
        new_env._robot_pos = self._robot_pos
        new_env._goal_pos = self._goal_pos
        new_env._start_pos = self._start_pos
        new_env._step_count = self._step_count
        new_env._prev_dist = self._prev_dist
        new_env._visited = set(self._visited)
        new_env._collision_count = self._collision_count
        new_env._episode_reward = self._episode_reward
        new_env._path_taken = list(self._path_taken)
        return new_env
