"""
Evaluation and algorithm comparison engine.

Compares all five approaches on multiple scenarios:
    1. Random Agent
    2. BFS
    3. A*
    4. Q-Learning (trained)
    5. DQN (trained)

Generates:
    - Algorithm comparison table (CSV + printed)
    - Bar charts for success rate, steps, collisions, reward
    - Scenario-specific evaluations (low/medium/high obstacle density)
    - Evaluation results JSON

All metrics come from actual environment interactions — no hard-coded values.

Usage
-----
    from src.evaluation.evaluate import Evaluator
    evaluator = Evaluator()
    results = evaluator.compare_all(grid_size=10)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.baselines.astar import AStarAgent
from src.baselines.bfs import BFSAgent
from src.baselines.random_agent import RandomAgent
from src.environment.warehouse_env import WarehouseEnv
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Evaluates and compares all algorithms on the warehouse environment.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to config.yaml. Defaults to project config.
    """

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        self.cfg = load_config(config_path)
        self._results: list[dict] = []

    # ------------------------------------------------------------------
    # Single-algorithm evaluation
    # ------------------------------------------------------------------

    def evaluate_random(
        self, env: WarehouseEnv, num_episodes: int = 100, seed: int = 42
    ) -> dict:
        """Evaluate the Random Agent."""
        logger.info(f"Evaluating Random Agent ({num_episodes} episodes)...")
        t0 = time.perf_counter()
        agent = RandomAgent(seed=seed)
        metrics = agent.evaluate(env, num_episodes=num_episodes)
        metrics["runtime_s"] = time.perf_counter() - t0
        return metrics

    def evaluate_bfs(
        self, env: WarehouseEnv, num_episodes: int = 100, seed: int = 42
    ) -> dict:
        """Evaluate BFS."""
        logger.info(f"Evaluating BFS ({num_episodes} episodes)...")
        t0 = time.perf_counter()
        agent = BFSAgent()
        metrics = agent.evaluate(env, num_episodes=num_episodes, seed=seed)
        metrics["runtime_s"] = time.perf_counter() - t0
        return metrics

    def evaluate_astar(
        self,
        env: WarehouseEnv,
        num_episodes: int = 100,
        seed: int = 42,
        heuristic: str = "manhattan",
    ) -> dict:
        """Evaluate A*."""
        logger.info(f"Evaluating A* ({heuristic}) ({num_episodes} episodes)...")
        t0 = time.perf_counter()
        agent = AStarAgent(heuristic=heuristic)
        metrics = agent.evaluate(env, num_episodes=num_episodes, seed=seed)
        metrics["runtime_s"] = time.perf_counter() - t0
        return metrics

    def evaluate_q_learning(
        self,
        env: WarehouseEnv,
        model_path: str | Path,
        num_episodes: int = 100,
    ) -> dict:
        """Evaluate a trained Q-Learning agent."""
        from src.agents.q_learning import QLearningAgent

        logger.info(f"Evaluating Q-Learning ({num_episodes} episodes)...")
        t0 = time.perf_counter()
        agent = QLearningAgent.load(model_path)
        metrics = agent.evaluate(env, num_episodes=num_episodes)
        metrics["runtime_s"] = time.perf_counter() - t0
        return metrics

    def evaluate_dqn(
        self,
        env: WarehouseEnv,
        model_path: str | Path,
        num_episodes: int = 100,
    ) -> dict:
        """Evaluate a trained DQN agent."""
        from src.agents.dqn import DQNAgent

        logger.info(f"Evaluating DQN ({num_episodes} episodes)...")
        t0 = time.perf_counter()
        agent = DQNAgent.load(model_path)
        metrics = agent.evaluate(env, num_episodes=num_episodes)
        metrics["runtime_s"] = time.perf_counter() - t0
        return metrics

    # ------------------------------------------------------------------
    # Full comparison
    # ------------------------------------------------------------------

    def compare_all(
        self,
        grid_size: Optional[int] = None,
        obstacle_density: Optional[float] = None,
        seed: Optional[int] = None,
        num_episodes: int = 100,
        ql_model_path: Optional[str | Path] = None,
        dqn_model_path: Optional[str | Path] = None,
        save_dir: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """
        Run all algorithms and return a comparison DataFrame.

        Parameters
        ----------
        grid_size : int, optional
        obstacle_density : float, optional
        seed : int, optional
        num_episodes : int
        ql_model_path : str or Path, optional
        dqn_model_path : str or Path, optional
        save_dir : str or Path, optional
            Where to save comparison CSV and plots.

        Returns
        -------
        pd.DataFrame
            Comparison table with one row per algorithm.
        """
        _grid = grid_size or self.cfg.environment.grid_size
        _density = obstacle_density or self.cfg.environment.obstacle_density
        _seed = seed or self.cfg.environment.random_seed

        env = WarehouseEnv(
            grid_size=_grid,
            obstacle_density=_density,
            seed=_seed,
        )

        results: list[dict] = []

        # Random
        r = self.evaluate_random(env, num_episodes=num_episodes, seed=_seed)
        results.append(r)

        # BFS
        r = self.evaluate_bfs(env, num_episodes=num_episodes, seed=_seed)
        results.append(r)

        # A*
        r = self.evaluate_astar(env, num_episodes=num_episodes, seed=_seed)
        results.append(r)

        # Q-Learning (optional — skip if no model provided)
        if ql_model_path and Path(ql_model_path).exists():
            r = self.evaluate_q_learning(env, ql_model_path, num_episodes=num_episodes)
            results.append(r)
        else:
            logger.warning("Q-Learning model not found — skipping.")

        # DQN (optional)
        if dqn_model_path and Path(dqn_model_path).exists():
            r = self.evaluate_dqn(env, dqn_model_path, num_episodes=num_episodes)
            results.append(r)
        else:
            logger.warning("DQN model not found — skipping.")

        df = pd.DataFrame(results)

        # Print comparison table
        self._print_comparison_table(df)

        # Save outputs
        if save_dir:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / "comparison_table.csv", index=False)
            self._save_comparison_plots(df, out_dir)
            logger.info(f"Comparison results saved -> {out_dir}")

        self._results = results
        return df

    def evaluate_scenarios(
        self,
        grid_size: Optional[int] = None,
        ql_model_path: Optional[str | Path] = None,
        dqn_model_path: Optional[str | Path] = None,
        save_dir: Optional[str | Path] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Evaluate all algorithms across low/medium/high obstacle density scenarios.

        Returns
        -------
        dict
            Maps scenario name to comparison DataFrame.
        """
        scenarios = self.cfg.evaluation.scenarios.to_dict()
        _grid = grid_size or self.cfg.environment.grid_size
        all_scenario_results: dict[str, pd.DataFrame] = {}

        for scenario_name, scenario_cfg in scenarios.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Scenario: {scenario_name.upper()} density "
                        f"({scenario_cfg['obstacle_density']})")
            logger.info(f"{'='*60}")

            df = self.compare_all(
                grid_size=_grid,
                obstacle_density=scenario_cfg["obstacle_density"],
                num_episodes=scenario_cfg["num_episodes"],
                ql_model_path=ql_model_path,
                dqn_model_path=dqn_model_path,
                save_dir=Path(save_dir) / scenario_name if save_dir else None,
            )
            all_scenario_results[scenario_name] = df

        return all_scenario_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_comparison_table(df: pd.DataFrame) -> None:
        """Pretty-print the comparison table to the logger."""
        cols = ["algorithm", "success_rate", "avg_steps", "avg_collisions", "avg_reward", "runtime_s"]
        display_cols = [c for c in cols if c in df.columns]
        table_df = df[display_cols].copy()
        if "success_rate" in table_df:
            table_df["success_rate"] = table_df["success_rate"].map("{:.1%}".format)
        if "avg_steps" in table_df:
            table_df["avg_steps"] = table_df["avg_steps"].map("{:.1f}".format)
        if "avg_collisions" in table_df:
            table_df["avg_collisions"] = table_df["avg_collisions"].map("{:.2f}".format)
        if "avg_reward" in table_df:
            table_df["avg_reward"] = table_df["avg_reward"].map("{:.1f}".format)
        if "runtime_s" in table_df:
            table_df["runtime_s"] = table_df["runtime_s"].map("{:.3f}s".format)

        logger.info("\n" + table_df.to_string(index=False))

    @staticmethod
    def _save_comparison_plots(df: pd.DataFrame, out_dir: Path) -> None:
        """Generate bar chart comparisons for all key metrics."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        algorithms = df["algorithm"].tolist()
        colours = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12"][:len(algorithms)]

        metrics = [
            ("success_rate", "Success Rate", "Fraction", True),
            ("avg_steps", "Avg Steps to Goal", "Steps", False),
            ("avg_collisions", "Avg Collisions", "Collisions", False),
            ("avg_reward", "Avg Episode Reward", "Reward", False),
        ]

        for col, title, ylabel, pct in metrics:
            if col not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            vals = df[col].values
            bars = ax.bar(algorithms, vals, color=colours, edgecolor="black", linewidth=0.5)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Algorithm")
            ax.grid(axis="y", alpha=0.3)
            for bar, val in zip(bars, vals):
                label = f"{val:.1%}" if pct else f"{val:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * (max(vals) - min(vals) + 1),
                    label,
                    ha="center", va="bottom", fontsize=9,
                )
            fig.tight_layout()
            fig.savefig(out_dir / f"comparison_{col}.png", dpi=120)
            plt.close(fig)

        logger.info(f"Comparison plots saved -> {out_dir}")
