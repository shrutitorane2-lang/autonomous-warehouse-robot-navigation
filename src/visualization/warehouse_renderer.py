"""
Warehouse grid renderer.

Supports three output modes:
    1. matplotlib figure  — for saving PNGs and Streamlit display
    2. RGB array          — for video recording / Gymnasium render()
    3. Plotly figure      — interactive HTML charts

Grid cell colour scheme (configurable via config.yaml):
    Free cell    → light grey
    Obstacle     → dark slate
    Start        → purple
    Goal         → green
    Robot        → blue
    Visited cell → light blue
    Path         → yellow-gold

Usage
-----
    renderer = WarehouseRenderer(env)
    fig = renderer.render_matplotlib()        # returns matplotlib Figure
    fig = renderer.render_plotly()            # returns plotly Figure
    img = renderer.render_rgb_array()         # returns np.ndarray H×W×3

    # Animate a path step-by-step (saves to PNG files)
    renderer.animate_path(path, output_dir="results/animation/")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.environment.warehouse_env import (
    CELL_FREE,
    CELL_GOAL,
    CELL_OBSTACLE,
    CELL_ROBOT,
)
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WarehouseRenderer:
    """
    Renders the warehouse environment as matplotlib, Plotly, or RGB array.

    Parameters
    ----------
    env : WarehouseEnv
        The environment to render.
    config : Config, optional
        Project configuration (for colour scheme).
    """

    # Fallback colours if config is missing
    _DEFAULT_COLOURS = {
        "free": "#F0F4F8",
        "obstacle": "#2D3748",
        "robot": "#3182CE",
        "goal": "#38A169",
        "visited": "#BEE3F8",
        "path": "#ECC94B",
        "start": "#9F7AEA",
    }

    def __init__(self, env, config=None) -> None:
        self.env = env
        cfg = config if config is not None else load_config()
        viz = cfg.visualization
        self.cell_size: int = viz.cell_size
        self.colours: dict = {
            k: v
            for k, v in (viz.colors.to_dict() if hasattr(viz.colors, "to_dict") else vars(viz.colors)).items()
        }
        # Fill missing colours with defaults
        for k, v in self._DEFAULT_COLOURS.items():
            self.colours.setdefault(k, v)

    # ------------------------------------------------------------------
    # Matplotlib rendering (primary method for Streamlit)
    # ------------------------------------------------------------------

    def render_matplotlib(
        self,
        path: Optional[List[Tuple[int, int]]] = None,
        title: str = "Warehouse",
        show: bool = False,
        ax=None,
    ):
        """
        Render the warehouse grid as a matplotlib figure.

        Parameters
        ----------
        path : list of (row, col) tuples, optional
            If provided, these cells are highlighted as the robot's path.
        title : str
            Figure title.
        show : bool
            If True, call plt.show() (interactive mode).
        ax : matplotlib Axes, optional
            If provided, draw on this axes instead of creating a new figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba

        grid = self.env.get_grid()
        n = grid.shape[0]
        robot = self.env.get_robot_pos()
        goal = self.env.get_goal_pos()
        start = self.env.get_start_pos()
        visited = self.env.get_visited()
        path = path or self.env.get_path()

        create_fig = ax is None
        if create_fig:
            fig, ax = plt.subplots(figsize=(n * 0.7, n * 0.7))
        else:
            fig = ax.figure

        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        ax.axis("off")

        # Draw cells
        for row in range(n):
            for col in range(n):
                cell_type = grid[row, col]
                pos = (row, col)

                # Determine colour
                if pos == robot:
                    colour = self.colours["robot"]
                elif pos == goal:
                    colour = self.colours["goal"]
                elif pos == start and pos != robot:
                    colour = self.colours["start"]
                elif cell_type == CELL_OBSTACLE:
                    colour = self.colours["obstacle"]
                elif pos in path and pos != start:
                    colour = self.colours["path"]
                elif pos in visited:
                    colour = self.colours["visited"]
                else:
                    colour = self.colours["free"]

                # Grid cell rectangle (matplotlib y-axis is inverted: row 0 → top)
                rect = patches.FancyBboxPatch(
                    (col + 0.02, n - row - 1 + 0.02),
                    0.96, 0.96,
                    boxstyle="round,pad=0.02",
                    facecolor=colour,
                    edgecolor="#CBD5E0",
                    linewidth=0.5,
                )
                ax.add_patch(rect)

                # Cell labels
                if pos == robot:
                    ax.text(col + 0.5, n - row - 0.5, "🤖", ha="center", va="center", fontsize=max(6, 14 - n // 2))
                elif pos == goal:
                    ax.text(col + 0.5, n - row - 0.5, "🎯", ha="center", va="center", fontsize=max(6, 14 - n // 2))
                elif cell_type == CELL_OBSTACLE:
                    ax.text(col + 0.5, n - row - 0.5, "■", ha="center", va="center",
                            fontsize=max(5, 10 - n // 3), color="#718096")

        # Draw path arrows
        if len(path) > 1:
            for i in range(len(path) - 1):
                r0, c0 = path[i]
                r1, c1 = path[i + 1]
                ax.annotate(
                    "",
                    xy=(c1 + 0.5, n - r1 - 0.5),
                    xytext=(c0 + 0.5, n - r0 - 0.5),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.25,head_length=0.2",
                        color="#D69E2E",
                        lw=1.5,
                    ),
                )

        # Legend
        legend_items = [
            patches.Patch(facecolor=self.colours["robot"], label="Robot"),
            patches.Patch(facecolor=self.colours["goal"], label="Goal"),
            patches.Patch(facecolor=self.colours["start"], label="Start"),
            patches.Patch(facecolor=self.colours["obstacle"], label="Obstacle"),
            patches.Patch(facecolor=self.colours["path"], label="Path"),
            patches.Patch(facecolor=self.colours["visited"], label="Visited"),
        ]
        ax.legend(
            handles=legend_items,
            loc="upper right",
            fontsize=7,
            framealpha=0.8,
            ncol=2,
        )

        if create_fig:
            fig.tight_layout()

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Plotly rendering (interactive, for Streamlit)
    # ------------------------------------------------------------------

    def render_plotly(
        self,
        path: Optional[List[Tuple[int, int]]] = None,
        title: str = "Warehouse Navigation",
    ):
        """
        Render the warehouse as an interactive Plotly figure.

        Parameters
        ----------
        path : list of (row, col), optional
        title : str

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        grid = self.env.get_grid()
        n = grid.shape[0]
        robot = self.env.get_robot_pos()
        goal = self.env.get_goal_pos()
        start = self.env.get_start_pos()
        visited = self.env.get_visited()
        path = path or self.env.get_path()

        # Build colour matrix
        colour_matrix = []
        hover_matrix = []
        for row in range(n):
            row_colours = []
            hover_row = []
            for col in range(n):
                pos = (row, col)
                cell_type = grid[row, col]

                if pos == robot:
                    c = self.colours["robot"]
                    label = "Robot"
                elif pos == goal:
                    c = self.colours["goal"]
                    label = "Goal"
                elif pos == start:
                    c = self.colours["start"]
                    label = "Start"
                elif cell_type == CELL_OBSTACLE:
                    c = self.colours["obstacle"]
                    label = "Obstacle"
                elif pos in path:
                    c = self.colours["path"]
                    label = "Path"
                elif pos in visited:
                    c = self.colours["visited"]
                    label = "Visited"
                else:
                    c = self.colours["free"]
                    label = f"Free ({row},{col})"

                row_colours.append(c)
                hover_row.append(label)
            colour_matrix.append(row_colours)
            hover_matrix.append(hover_row)

        # Convert hex to numeric for heatmap z-values
        cell_codes = {
            "free": 0,
            "visited": 1,
            "path": 2,
            "start": 3,
            "goal": 4,
            "robot": 5,
            "obstacle": 6,
        }

        z_matrix = []
        for row in range(n):
            z_row = []
            for col in range(n):
                pos = (row, col)
                cell_type = grid[row, col]
                if pos == robot:
                    z_row.append(cell_codes["robot"])
                elif pos == goal:
                    z_row.append(cell_codes["goal"])
                elif pos == start:
                    z_row.append(cell_codes["start"])
                elif cell_type == CELL_OBSTACLE:
                    z_row.append(cell_codes["obstacle"])
                elif pos in path:
                    z_row.append(cell_codes["path"])
                elif pos in visited:
                    z_row.append(cell_codes["visited"])
                else:
                    z_row.append(cell_codes["free"])
            z_matrix.append(z_row)

        colour_scale = [
            [0.0, self.colours["free"]],
            [1 / 6, self.colours["visited"]],
            [2 / 6, self.colours["path"]],
            [3 / 6, self.colours["start"]],
            [4 / 6, self.colours["goal"]],
            [5 / 6, self.colours["robot"]],
            [1.0, self.colours["obstacle"]],
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=z_matrix,
                colorscale=colour_scale,
                showscale=False,
                hovertext=hover_matrix,
                hovertemplate="%{hovertext}<extra></extra>",
                zmin=0,
                zmax=6,
            )
        )

        # Overlay text annotations
        annotations = []
        for row in range(n):
            for col in range(n):
                pos = (row, col)
                cell_type = grid[row, col]
                text = ""
                if pos == robot:
                    text = "R"
                elif pos == goal:
                    text = "G"
                elif cell_type == CELL_OBSTACLE:
                    text = "X"
                elif pos == start:
                    text = "S"

                if text:
                    annotations.append(
                        dict(
                            x=col,
                            y=n - 1 - row,
                            text=f"<b>{text}</b>",
                            showarrow=False,
                            font=dict(size=max(8, 18 - n), color="white"),
                        )
                    )

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            annotations=annotations,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            height=max(400, n * 45),
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="#F7FAFC",
            paper_bgcolor="#F7FAFC",
        )

        return fig

    # ------------------------------------------------------------------
    # RGB array rendering
    # ------------------------------------------------------------------

    def render_rgb_array(
        self,
        path: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        Render the warehouse as a NumPy RGB array (H × W × 3).

        Parameters
        ----------
        path : list of (row, col), optional

        Returns
        -------
        np.ndarray
            RGB image array.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        fig = self.render_matplotlib(path=path)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        buf.seek(0)
        img = np.array(Image.open(buf).convert("RGB"))
        plt.close(fig)
        return img

    # ------------------------------------------------------------------
    # Step-by-step animation
    # ------------------------------------------------------------------

    def animate_path(
        self,
        path: List[Tuple[int, int]],
        output_dir: str | Path,
        prefix: str = "frame",
    ) -> list[str]:
        """
        Save one PNG frame per step of the path.

        Parameters
        ----------
        path : list of (row, col)
            Full path from start to goal.
        output_dir : str or Path
            Directory to save frame PNGs.
        prefix : str
            Filename prefix for each frame.

        Returns
        -------
        list of str
            Paths to saved frame files.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        frame_paths: list[str] = []
        visited: set[Tuple[int, int]] = set()

        for step_idx, pos in enumerate(path):
            visited.add(pos)

            # Temporarily move robot to current position
            original_pos = self.env._robot_pos
            original_visited = self.env._visited.copy()
            original_path = list(self.env._path_taken)

            self.env._robot_pos = pos
            self.env._visited = set(visited)
            self.env._path_taken = path[: step_idx + 1]

            title = f"Step {step_idx} — Robot at {pos}"
            fig = self.render_matplotlib(path=path[: step_idx + 1], title=title)
            frame_file = out_dir / f"{prefix}_{step_idx:04d}.png"
            fig.savefig(frame_file, dpi=80)
            plt.close(fig)
            frame_paths.append(str(frame_file))

            # Restore
            self.env._robot_pos = original_pos
            self.env._visited = original_visited
            self.env._path_taken = original_path

        logger.info(f"Animation frames saved -> {out_dir} ({len(frame_paths)} frames)")
        return frame_paths

    # ------------------------------------------------------------------
    # State info panel (for Streamlit RL explanation)
    # ------------------------------------------------------------------

    def get_state_info(self, obs: np.ndarray, action: int, reward: float) -> dict:
        """
        Return a human-readable dict of the current RL state for display.

        Parameters
        ----------
        obs : np.ndarray
        action : int
        reward : float

        Returns
        -------
        dict
        """
        from src.environment.warehouse_env import ACTION_NAMES

        n = self.env.grid_size - 1
        info = {
            "robot_pos": self.env.get_robot_pos(),
            "goal_pos": self.env.get_goal_pos(),
            "action": f"{action} ({ACTION_NAMES[action]})",
            "reward": round(reward, 2),
            "distance_to_goal": int(self.env._manhattan(
                self.env.get_robot_pos(), self.env.get_goal_pos()
            )),
            "steps_taken": self.env._step_count,
            "collisions": self.env._collision_count,
        }
        if len(obs) >= 8:
            info["local_obstacles"] = {
                "UP": bool(obs[4] > 0.5),
                "DOWN": bool(obs[5] > 0.5),
                "LEFT": bool(obs[6] > 0.5),
                "RIGHT": bool(obs[7] > 0.5),
            }
        return info
