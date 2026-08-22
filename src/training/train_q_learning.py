"""
Q-Learning training pipeline.

Trains a QLearningAgent on a WarehouseEnv for a configurable number of
episodes and saves:
    - Trained Q-table (pickle)
    - Training metrics (CSV)
    - Learning curve plots (PNG)

Experiment outputs are written to a unique timestamped directory:
    results/q_learning/YYYYMMDD_HHMMSS/

Usage
-----
    python -m src.training.train_q_learning               # uses config defaults
    python -m src.training.train_q_learning --episodes 5000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.q_learning import QLearningAgent
from src.environment.warehouse_env import WarehouseEnv
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_q_learning(
    episodes: int | None = None,
    grid_size: int | None = None,
    obstacle_density: float | None = None,
    seed: int | None = None,
    learning_rate: float | None = None,
    discount_factor: float | None = None,
    epsilon_start: float | None = None,
    epsilon_end: float | None = None,
    epsilon_decay: float | None = None,
    save_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    Train a Q-Learning agent and return experiment results.

    All parameters default to values in config/config.yaml when omitted.

    Parameters
    ----------
    episodes : int, optional
    grid_size : int, optional
    obstacle_density : float, optional
    seed : int, optional
    learning_rate : float, optional
    discount_factor : float, optional
    epsilon_start : float, optional
    epsilon_end : float, optional
    epsilon_decay : float, optional
    save_dir : str or Path, optional
        Root directory for experiment outputs.
    verbose : bool
        Print per-interval progress.

    Returns
    -------
    dict
        Training results including history and paths to saved artefacts.
    """
    cfg = load_config()

    # Apply overrides
    n_episodes = episodes if episodes is not None else cfg.q_learning.episodes
    _grid_size = grid_size if grid_size is not None else cfg.environment.grid_size
    _density = obstacle_density if obstacle_density is not None else cfg.environment.obstacle_density
    _seed = seed if seed is not None else cfg.environment.random_seed
    _lr = learning_rate if learning_rate is not None else cfg.q_learning.learning_rate
    _gamma = discount_factor if discount_factor is not None else cfg.q_learning.discount_factor
    _eps_start = epsilon_start if epsilon_start is not None else cfg.q_learning.epsilon_start
    _eps_end = epsilon_end if epsilon_end is not None else cfg.q_learning.epsilon_end
    _eps_decay = epsilon_decay if epsilon_decay is not None else cfg.q_learning.epsilon_decay
    eval_interval = cfg.q_learning.eval_interval
    window = cfg.training.moving_avg_window
    checkpoint_interval = cfg.training.checkpoint_interval

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(save_dir) if save_dir else Path(cfg.training.save_dir)
    exp_dir = base_dir / "q_learning" / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_dir = exp_dir / "models"
    model_dir.mkdir(exist_ok=True)

    logger.info(f"Starting Q-Learning training — {n_episodes} episodes")
    logger.info(f"Experiment directory: {exp_dir}")

    # Instantiate environment and agent
    env = WarehouseEnv(
        grid_size=_grid_size,
        obstacle_density=_density,
        seed=_seed,
        state_type=cfg.environment.state_type,
    )
    agent = QLearningAgent(
        learning_rate=_lr,
        discount_factor=_gamma,
        epsilon_start=_eps_start,
        epsilon_end=_eps_end,
        epsilon_decay=_eps_decay,
    )

    # Save experiment config
    exp_config = {
        "algorithm": "Q-Learning",
        "episodes": n_episodes,
        "grid_size": _grid_size,
        "obstacle_density": _density,
        "seed": _seed,
        "learning_rate": _lr,
        "discount_factor": _gamma,
        "epsilon_start": _eps_start,
        "epsilon_end": _eps_end,
        "epsilon_decay": _eps_decay,
        "timestamp": timestamp,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    # ----- Training loop -----
    all_metrics: list[dict] = []
    t_start = time.time()

    for ep in range(1, n_episodes + 1):
        result = agent.train_episode(env)
        all_metrics.append(result)

        # Periodic logging
        if verbose and ep % eval_interval == 0:
            recent = all_metrics[-eval_interval:]
            avg_r = np.mean([m["total_reward"] for m in recent])
            avg_s = np.mean([m["steps"] for m in recent])
            success_r = np.mean([m["success"] for m in recent])
            elapsed = time.time() - t_start
            logger.info(
                f"Episode {ep:5d}/{n_episodes} | "
                f"AvgReward: {avg_r:7.1f} | "
                f"AvgSteps: {avg_s:5.1f} | "
                f"SuccessRate: {success_r:.1%} | "
                f"Epsilon: {result['epsilon']:.4f} | "
                f"QTableSize: {result['q_table_size']:6d} | "
                f"Elapsed: {elapsed:.0f}s"
            )

        # Checkpoint
        if ep % checkpoint_interval == 0:
            ckpt_path = model_dir / f"q_learning_ep{ep:05d}.pkl"
            agent.save(ckpt_path)

    training_time = time.time() - t_start
    logger.info(f"Training completed in {training_time:.1f}s")

    # ----- Save final model -----
    final_model_path = exp_dir / "q_learning_final.pkl"
    agent.save(final_model_path)

    # ----- Save metrics CSV -----
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df["moving_avg_reward"] = (
        metrics_df["total_reward"]
        .rolling(window=window, min_periods=1)
        .mean()
    )
    metrics_path = exp_dir / "training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved -> {metrics_path}")

    # ----- Generate and save plots -----
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    _save_training_plots(metrics_df, plots_dir, window, "Q-Learning")

    # ----- Final evaluation -----
    logger.info("Running post-training evaluation (100 episodes, deterministic)...")
    eval_metrics = agent.evaluate(env, num_episodes=100)
    eval_metrics["training_time_s"] = training_time
    eval_metrics["total_episodes"] = n_episodes
    eval_metrics["q_table_size"] = len(agent.q_table)

    with open(exp_dir / "eval_results.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info(
        f"Final evaluation -> "
        f"Success: {eval_metrics['success_rate']:.1%} | "
        f"AvgReward: {eval_metrics['avg_reward']:.1f} | "
        f"AvgSteps: {eval_metrics['avg_steps']:.1f}"
    )

    return {
        "agent": agent,
        "exp_dir": str(exp_dir),
        "model_path": str(final_model_path),
        "metrics_df": metrics_df,
        "eval_metrics": eval_metrics,
        "training_time_s": training_time,
    }


def _save_training_plots(
    metrics_df: pd.DataFrame,
    plots_dir: Path,
    window: int,
    title_prefix: str,
) -> None:
    """Save reward curve, episode length, and success rate plots."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for file output
    import matplotlib.pyplot as plt

    episodes = metrics_df["episode"].values

    # --- Reward curve ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, metrics_df["total_reward"], alpha=0.3, color="#4A90D9", label="Episode Reward")
    ax.plot(
        episodes,
        metrics_df["moving_avg_reward"],
        color="#E74C3C",
        linewidth=2,
        label=f"Moving Avg (window={window})",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(f"{title_prefix} — Reward Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "reward_curve.png", dpi=120)
    plt.close(fig)

    # --- Episode length ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, metrics_df["steps"], alpha=0.3, color="#27AE60")
    ax.plot(
        episodes,
        metrics_df["steps"].rolling(window=window, min_periods=1).mean(),
        color="#1A6B3C",
        linewidth=2,
        label=f"Moving Avg",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title(f"{title_prefix} — Episode Length")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "episode_length.png", dpi=120)
    plt.close(fig)

    # --- Epsilon decay ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(episodes, metrics_df["epsilon"], color="#9B59B6", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title(f"{title_prefix} — Exploration Rate (ε)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "epsilon_decay.png", dpi=120)
    plt.close(fig)

    # --- Success rate rolling ---
    fig, ax = plt.subplots(figsize=(10, 4))
    success_roll = metrics_df["success"].astype(float).rolling(window=window, min_periods=1).mean()
    ax.plot(episodes, success_roll, color="#F39C12", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"{title_prefix} — Success Rate (rolling {window})")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "success_rate.png", dpi=120)
    plt.close(fig)

    logger.info(f"Training plots saved -> {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-Learning agent")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--grid-size", type=int, dest="grid_size")
    parser.add_argument("--density", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gamma", type=float)
    args = parser.parse_args()

    train_q_learning(
        episodes=args.episodes,
        grid_size=args.grid_size,
        obstacle_density=args.density,
        seed=args.seed,
        learning_rate=args.lr,
        discount_factor=args.gamma,
    )
