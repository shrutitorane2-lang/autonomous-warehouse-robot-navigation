"""
Deep Q-Network (DQN) agent.

Implements DQN (Mnih et al., 2015) with the following stabilisation tricks:
    1. Experience Replay   — breaks temporal correlations in training data.
    2. Target Network      — frozen copy of the online network updated
                             every ``target_update_freq`` episodes to
                             provide stable Q-value targets.
    3. Gradient Clipping   — clips gradients to prevent exploding updates.

Network Architecture
--------------------
    state (obs_dim)
        │
    Linear(obs_dim → hidden_size)
        │  ReLU
    Linear(hidden_size → hidden_size)
        │  ReLU
    Linear(hidden_size → n_actions)
        │
    Q-values for each action

All computation runs on CPU (no CUDA required).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.agents.replay_buffer import ReplayBuffer
from src.utils.config import Config, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Force CPU execution
DEVICE = torch.device("cpu")


class _DQNNetwork(nn.Module):
    """
    Feedforward Q-network.

    Parameters
    ----------
    obs_dim : int
        Input (state) dimension.
    n_actions : int
        Output (Q-value per action) dimension.
    hidden_size : int
        Number of neurons in each hidden layer.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        # Kaiming initialisation for ReLU networks
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for all actions."""
        return self.net(x)


class DQNAgent:
    """
    DQN agent with online/target networks, experience replay, and epsilon-greedy
    exploration.

    Parameters
    ----------
    obs_dim : int
        Observation (state) dimension.
    config : Config, optional
        Project config. Loaded from config.yaml when omitted.
    learning_rate : float, optional
        Override from config.
    discount_factor : float, optional
        Override gamma.
    epsilon_start : float, optional
    epsilon_end : float, optional
    epsilon_decay : float, optional
    batch_size : int, optional
    replay_buffer_size : int, optional
    target_update_freq : int, optional
    hidden_size : int, optional
    gradient_clip : float, optional
    seed : int, optional
    """

    def __init__(
        self,
        obs_dim: int = 8,
        config: Optional[Config] = None,
        learning_rate: Optional[float] = None,
        discount_factor: Optional[float] = None,
        epsilon_start: Optional[float] = None,
        epsilon_end: Optional[float] = None,
        epsilon_decay: Optional[float] = None,
        batch_size: Optional[int] = None,
        replay_buffer_size: Optional[int] = None,
        target_update_freq: Optional[int] = None,
        hidden_size: Optional[int] = None,
        gradient_clip: Optional[float] = None,
        n_actions: int = 4,
        seed: int = 42,
    ) -> None:
        cfg = config if config is not None else load_config()
        dqn_cfg = cfg.dqn

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = discount_factor if discount_factor is not None else dqn_cfg.discount_factor
        self.epsilon = epsilon_start if epsilon_start is not None else dqn_cfg.epsilon_start
        self.epsilon_end = epsilon_end if epsilon_end is not None else dqn_cfg.epsilon_end
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else dqn_cfg.epsilon_decay
        self.batch_size = batch_size if batch_size is not None else dqn_cfg.batch_size
        self.target_update_freq = (
            target_update_freq if target_update_freq is not None else dqn_cfg.target_update_freq
        )
        self.grad_clip = gradient_clip if gradient_clip is not None else dqn_cfg.gradient_clip
        hidden = hidden_size if hidden_size is not None else dqn_cfg.hidden_size
        buf_size = replay_buffer_size if replay_buffer_size is not None else dqn_cfg.replay_buffer_size
        lr = learning_rate if learning_rate is not None else dqn_cfg.learning_rate

        # Reproducibility
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        # ----- Networks -----
        self.online_net = _DQNNetwork(obs_dim, n_actions, hidden).to(DEVICE)
        self.target_net = copy.deepcopy(self.online_net).to(DEVICE)
        self.target_net.eval()  # Target network is never trained directly

        # ----- Optimiser -----
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # ----- Replay buffer -----
        self.replay_buffer = ReplayBuffer(buf_size, obs_dim, seed=seed)

        # ----- Training history -----
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_successes: list[bool] = []
        self.episode_collisions: list[int] = []
        self.episode_losses: list[float] = []
        self.epsilon_history: list[float] = []
        self.total_episodes: int = 0
        self.total_steps: int = 0

        self.name = "DQN"

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        ε-greedy action selection.

        Parameters
        ----------
        obs : np.ndarray
        deterministic : bool
            If True, disable exploration (evaluation mode).

        Returns
        -------
        int
        """
        if not deterministic and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))

        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            q_values = self.online_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions given an observation."""
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            q_values = self.online_net(state_t)
            return q_values.squeeze(0).numpy()

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def _learn(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient descent step.

        Returns
        -------
        float or None
            MSE loss value, or None if the buffer is not ready.
        """
        if not self.replay_buffer.can_sample(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(DEVICE)
        actions_t = torch.LongTensor(actions).to(DEVICE)
        rewards_t = torch.FloatTensor(rewards).to(DEVICE)
        next_states_t = torch.FloatTensor(next_states).to(DEVICE)
        dones_t = torch.FloatTensor(dones).to(DEVICE)

        # Current Q-values: Q(s, a) from online network
        q_values = self.online_net(states_t)
        q_current = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + γ max Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t)
            next_q_max = next_q_values.max(dim=1)[0]
            q_target = rewards_t + self.gamma * next_q_max * (1.0 - dones_t)

        # MSE / Huber loss
        loss = F.smooth_l1_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Target network sync
    # ------------------------------------------------------------------

    def _sync_target_network(self) -> None:
        """Copy online network weights to the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Episode training loop
    # ------------------------------------------------------------------

    def train_episode(self, env) -> dict[str, Any]:
        """
        Run one full training episode.

        Interaction loop:
            observe → select action → step env → store transition
            → learn from replay → repeat

        Parameters
        ----------
        env : WarehouseEnv

        Returns
        -------
        dict
            Episode metrics.
        """
        obs, _ = env.reset()
        total_reward = 0.0
        episode_losses: list[float] = []
        steps = 0
        done = False

        while not done:
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            self.replay_buffer.push(obs, action, reward, next_obs, done)

            # Learn
            loss = self._learn()
            if loss is not None:
                episode_losses.append(loss)

            total_reward += reward
            steps += 1
            self.total_steps += 1
            obs = next_obs

        # Epsilon decay
        self.decay_epsilon()
        self.total_episodes += 1

        # Sync target network periodically
        if self.total_episodes % self.target_update_freq == 0:
            self._sync_target_network()

        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

        result: dict[str, Any] = {
            "episode": self.total_episodes,
            "total_reward": total_reward,
            "steps": steps,
            "success": terminated and not truncated,
            "collisions": info["collision_count"],
            "epsilon": self.epsilon,
            "avg_loss": avg_loss,
            "buffer_size": len(self.replay_buffer),
        }

        # Accumulate history
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.episode_successes.append(result["success"])
        self.episode_collisions.append(info["collision_count"])
        self.episode_losses.append(avg_loss)
        self.epsilon_history.append(self.epsilon)

        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, env, num_episodes: int = 100) -> dict[str, Any]:
        """
        Evaluate the trained policy (no exploration).

        Parameters
        ----------
        env : WarehouseEnv
        num_episodes : int

        Returns
        -------
        dict
        """
        rewards, steps, collisions, successes = [], [], [], []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            rewards.append(total_reward)
            steps.append(info["step_count"])
            collisions.append(info["collision_count"])
            successes.append(float(terminated and not truncated))

        return {
            "algorithm": self.name,
            "num_episodes": num_episodes,
            "success_rate": float(np.mean(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "avg_steps": float(np.mean(steps)),
            "avg_collisions": float(np.mean(collisions)),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model weights and training history."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net_state_dict": self.online_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "episode_successes": self.episode_successes,
                "episode_collisions": self.episode_collisions,
                "episode_losses": self.episode_losses,
                "epsilon_history": self.epsilon_history,
                "total_episodes": self.total_episodes,
                "total_steps": self.total_steps,
            },
            path,
        )
        logger.info(f"DQN model saved -> {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: Optional[Config] = None,
        hidden_size: Optional[int] = None,
    ) -> "DQNAgent":
        """
        Load a previously saved DQN agent.

        Parameters
        ----------
        path : str or Path
        config : Config, optional
        hidden_size : int, optional
            Must match the saved model's architecture.

        Returns
        -------
        DQNAgent
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        obs_dim = checkpoint["obs_dim"]
        n_actions = checkpoint["n_actions"]

        cfg = config if config is not None else load_config()
        hs = hidden_size if hidden_size is not None else cfg.dqn.hidden_size

        agent = cls(obs_dim=obs_dim, config=config, hidden_size=hs, n_actions=n_actions)
        agent.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.gamma = checkpoint["gamma"]
        agent.epsilon = checkpoint["epsilon"]
        agent.episode_rewards = checkpoint.get("episode_rewards", [])
        agent.episode_lengths = checkpoint.get("episode_lengths", [])
        agent.episode_successes = checkpoint.get("episode_successes", [])
        agent.episode_collisions = checkpoint.get("episode_collisions", [])
        agent.episode_losses = checkpoint.get("episode_losses", [])
        agent.epsilon_history = checkpoint.get("epsilon_history", [])
        agent.total_episodes = checkpoint.get("total_episodes", 0)
        agent.total_steps = checkpoint.get("total_steps", 0)

        logger.info(f"DQN model loaded <- {path}")
        return agent
