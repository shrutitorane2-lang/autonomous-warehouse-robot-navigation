"""
DQN training pipeline.

Trains a DQNAgent on a WarehouseEnv and saves:
    - Model checkpoint (.pt)
    - Training metrics (CSV)
    - Learning curve plots (PNG)
    - Evaluation results (JSON)

Experiment outputs go to:
    results/dqn/YYYYMMDD_HHMMSS/

Usage
-----
    python -m src.training.train_dqn
    python -m src.training.train_dqn --episodes 2000 --grid-size 10
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.dqn import DQNAgent
from src.environment.warehouse_env import WarehouseEnv
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_dqn(
    episodes: int | None = None,
    grid_size: int | None = None,
    obstacle_density: float | None = None,
    seed: int | None = None,
    learning_rate: float | None = None,
    discount_factor: float | None = None,
    epsilon_start: float | None = None,
    epsilon_end: float | None = None,
    epsilon_decay: float | None = None,
    batch_size: int | None = None,
    hidden_size: int | None = None,
    save_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    Train a DQN agent and return experiment results.

    Parameters
    ----------
    (All parameters default to config.yaml values when omitted.)

    Returns
    -------
    dict
        Training results including the agent, metrics, and paths.
    """
    cfg = load_config()

    n_episodes = episodes if episodes is not None else cfg.dqn.episodes
    _grid_size = grid_size if grid_size is not None else cfg.environment.grid_size
    _density = obstacle_density if obstacle_density is not None else cfg.environment.obstacle_density
    _seed = seed if seed is not None else cfg.environment.random_seed
    _lr = learning_rate if learning_rate is not None else cfg.dqn.learning_rate
    _gamma = discount_factor if discount_factor is not None else cfg.dqn.discount_factor
    _eps_start = epsilon_start if epsilon_start is not None else cfg.dqn.epsilon_start
    _eps_end = epsilon_end if epsilon_end is not None else cfg.dqn.epsilon_end
    _eps_decay = epsilon_decay if epsilon_decay is not None else cfg.dqn.epsilon_decay
    _batch_size = batch_size if batch_size is not None else cfg.dqn.batch_size
    _hidden = hidden_size if hidden_size is not None else cfg.dqn.hidden_size
    eval_interval = cfg.dqn.eval_interval
    window = cfg.training.moving_avg_window
    checkpoint_interval = cfg.training.checkpoint_interval
    state_type = cfg.environment.state_type

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(save_dir) if save_dir else Path(cfg.training.save_dir)
    exp_dir = base_dir / "dqn" / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_dir = exp_dir / "models"
    model_dir.mkdir(exist_ok=True)

    logger.info(f"Starting DQN training — {n_episodes} episodes")
    logger.info(f"Experiment directory: {exp_dir}")

    # Environment
    env = WarehouseEnv(
        grid_size=_grid_size,
        obstacle_density=_density,
        seed=_seed,
        state_type=state_type,
    )

    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)

    # Agent
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        learning_rate=_lr,
        discount_factor=_gamma,
        epsilon_start=_eps_start,
        epsilon_end=_eps_end,
        epsilon_decay=_eps_decay,
        batch_size=_batch_size,
        hidden_size=_hidden,
        seed=_seed,
    )

    # Save experiment config
    exp_config = {
        "algorithm": "DQN",
        "episodes": n_episodes,
        "grid_size": _grid_size,
        "obstacle_density": _density,
        "seed": _seed,
        "learning_rate": _lr,
        "discount_factor": _gamma,
        "epsilon_start": _eps_start,
        "epsilon_end": _eps_end,
        "epsilon_decay": _eps_decay,
        "batch_size": _batch_size,
        "hidden_size": _hidden,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
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

        if verbose and ep % eval_interval == 0:
            recent = all_metrics[-eval_interval:]
            avg_r = np.mean([m["total_reward"] for m in recent])
            avg_s = np.mean([m["steps"] for m in recent])
            success_r = np.mean([m["success"] for m in recent])
            avg_loss = np.mean([m["avg_loss"] for m in recent if m["avg_loss"] > 0])
            elapsed = time.time() - t_start
            logger.info(
                f"Episode {ep:5d}/{n_episodes} | "
                f"AvgReward: {avg_r:7.1f} | "
                f"AvgSteps: {avg_s:5.1f} | "
                f"SuccessRate: {success_r:.1%} | "
                f"Epsilon: {result['epsilon']:.4f} | "
                f"AvgLoss: {avg_loss:.4f} | "
                f"Buffer: {result['buffer_size']:5d} | "
                f"Elapsed: {elapsed:.0f}s"
            )

        if ep % checkpoint_interval == 0:
            ckpt_path = model_dir / f"dqn_ep{ep:05d}.pt"
            agent.save(ckpt_path)

    training_time = time.time() - t_start
    logger.info(f"DQN training completed in {training_time:.1f}s")

    # ----- Save final model -----
    final_model_path = exp_dir / "dqn_final.pt"
    agent.save(final_model_path)

    # ----- Save metrics CSV -----
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df["moving_avg_reward"] = (
        metrics_df["total_reward"].rolling(window=window, min_periods=1).mean()
    )
    metrics_path = exp_dir / "training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # ----- Save plots -----
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    _save_training_plots(metrics_df, plots_dir, window)

    # ----- Final evaluation -----
    logger.info("Running post-training evaluation (100 episodes, deterministic)...")
    eval_metrics = agent.evaluate(env, num_episodes=100)
    eval_metrics["training_time_s"] = training_time
    eval_metrics["total_episodes"] = n_episodes

    with open(exp_dir / "eval_results.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info(
        f"DQN final evaluation -> "
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
) -> None:
    """Save DQN training plots including the loss curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = metrics_df["episode"].values

    # Reward curve
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
    ax.set_title("DQN — Reward Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "reward_curve.png", dpi=120)
    plt.close(fig)

    # Loss curve (DQN-specific)
    loss_vals = metrics_df["avg_loss"].values
    non_zero_loss = loss_vals[loss_vals > 0]
    if len(non_zero_loss) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        loss_eps = episodes[loss_vals > 0]
        ax.plot(loss_eps, non_zero_loss, alpha=0.4, color="#E67E22", linewidth=0.8)
        roll = pd.Series(non_zero_loss).rolling(window=min(window, len(non_zero_loss)), min_periods=1).mean()
        ax.plot(loss_eps, roll.values, color="#C0392B", linewidth=2, label="Smoothed Loss")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Huber Loss")
        ax.set_title("DQN — Training Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "dqn_loss.png", dpi=120)
        plt.close(fig)

    # Episode length
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, metrics_df["steps"], alpha=0.3, color="#27AE60")
    ax.plot(
        episodes,
        metrics_df["steps"].rolling(window=window, min_periods=1).mean(),
        color="#1A6B3C",
        linewidth=2,
        label="Moving Avg",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("DQN — Episode Length")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "episode_length.png", dpi=120)
    plt.close(fig)

    # Epsilon decay
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(episodes, metrics_df["epsilon"], color="#9B59B6", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("DQN — Exploration Rate (ε)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "epsilon_decay.png", dpi=120)
    plt.close(fig)

    # Success rate
    fig, ax = plt.subplots(figsize=(10, 4))
    success_roll = metrics_df["success"].astype(float).rolling(window=window, min_periods=1).mean()
    ax.plot(episodes, success_roll, color="#F39C12", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"DQN — Success Rate (rolling {window})")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "success_rate.png", dpi=120)
    plt.close(fig)

    logger.info(f"DQN plots saved -> {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--grid-size", type=int, dest="grid_size")
    parser.add_argument("--density", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    args = parser.parse_args()

    train_dqn(
        episodes=args.episodes,
        grid_size=args.grid_size,
        obstacle_density=args.density,
        seed=args.seed,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        batch_size=args.batch_size,
    )
