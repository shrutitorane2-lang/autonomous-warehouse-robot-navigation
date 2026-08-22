"""
Tests for DQN agent and ReplayBuffer.

Covers:
    - Network forward pass
    - Replay buffer push/sample
    - Training step
    - Action selection
    - Save and load
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents.dqn import DQNAgent, _DQNNetwork, DEVICE
from src.agents.replay_buffer import ReplayBuffer
from src.environment.warehouse_env import WarehouseEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OBS_DIM = 8  # advanced state type
N_ACTIONS = 4

@pytest.fixture
def network():
    return _DQNNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_size=64).to(DEVICE)


@pytest.fixture
def replay_buffer():
    return ReplayBuffer(capacity=1000, obs_dim=OBS_DIM, seed=42)


@pytest.fixture
def dqn_agent():
    return DQNAgent(
        obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99,
        batch_size=32,
        replay_buffer_size=1000,
        target_update_freq=5,
        hidden_size=64,
        seed=42,
    )


@pytest.fixture
def env():
    e = WarehouseEnv(grid_size=8, obstacle_density=0.1, seed=42, state_type="advanced")
    e.reset()
    return e


# ---------------------------------------------------------------------------
# Network forward pass
# ---------------------------------------------------------------------------

class TestDQNNetwork:

    def test_forward_pass_output_shape(self, network):
        batch_size = 16
        x = torch.randn(batch_size, OBS_DIM).to(DEVICE)
        out = network(x)
        assert out.shape == (batch_size, N_ACTIONS), \
            f"Expected ({batch_size}, {N_ACTIONS}), got {out.shape}"

    def test_single_obs_forward(self, network):
        x = torch.randn(1, OBS_DIM).to(DEVICE)
        out = network(x)
        assert out.shape == (1, N_ACTIONS)

    def test_output_is_finite(self, network):
        x = torch.randn(8, OBS_DIM).to(DEVICE)
        out = network(x)
        assert torch.all(torch.isfinite(out))

    def test_weights_initialised_nonzero(self, network):
        for layer in network.net:
            if isinstance(layer, torch.nn.Linear):
                assert not torch.all(layer.weight == 0)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class TestReplayBuffer:

    def test_push_and_len(self, replay_buffer):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        replay_buffer.push(obs, 0, 1.0, obs, False)
        assert len(replay_buffer) == 1

    def test_cannot_sample_below_capacity(self, replay_buffer):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        replay_buffer.push(obs, 0, 1.0, obs, False)
        with pytest.raises(ValueError):
            replay_buffer.sample(batch_size=32)

    def test_sample_returns_correct_shapes(self, replay_buffer):
        obs = np.random.rand(OBS_DIM).astype(np.float32)
        for _ in range(100):
            replay_buffer.push(obs, 1, 0.5, obs, False)
        states, actions, rewards, next_states, dones = replay_buffer.sample(32)
        assert states.shape == (32, OBS_DIM)
        assert actions.shape == (32,)
        assert rewards.shape == (32,)
        assert next_states.shape == (32, OBS_DIM)
        assert dones.shape == (32,)

    def test_circular_buffer_overwrites_oldest(self):
        buf = ReplayBuffer(capacity=10, obs_dim=OBS_DIM)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for i in range(15):
            buf.push(obs, i % 4, float(i), obs, False)
        assert len(buf) == 10  # Capped at capacity

    def test_can_sample_returns_correct_bool(self, replay_buffer):
        assert not replay_buffer.can_sample(32)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for _ in range(32):
            replay_buffer.push(obs, 0, 0.0, obs, False)
        assert replay_buffer.can_sample(32)


# ---------------------------------------------------------------------------
# DQN Agent — action selection
# ---------------------------------------------------------------------------

class TestDQNActionSelection:

    def test_random_action_in_range(self, dqn_agent):
        dqn_agent.epsilon = 1.0
        obs = np.random.rand(OBS_DIM).astype(np.float32)
        for _ in range(20):
            action = dqn_agent.select_action(obs)
            assert 0 <= action < N_ACTIONS

    def test_deterministic_action_in_range(self, dqn_agent):
        obs = np.random.rand(OBS_DIM).astype(np.float32)
        for _ in range(20):
            action = dqn_agent.select_action(obs, deterministic=True)
            assert 0 <= action < N_ACTIONS

    def test_get_q_values_shape(self, dqn_agent):
        obs = np.random.rand(OBS_DIM).astype(np.float32)
        q_vals = dqn_agent.get_q_values(obs)
        assert q_vals.shape == (N_ACTIONS,)


# ---------------------------------------------------------------------------
# DQN Agent — training step
# ---------------------------------------------------------------------------

class TestDQNTrainingStep:

    def test_train_episode_returns_valid_result(self, dqn_agent, env):
        result = dqn_agent.train_episode(env)
        assert "total_reward" in result
        assert "steps" in result
        assert "success" in result
        assert "epsilon" in result
        assert "avg_loss" in result
        assert result["steps"] > 0

    def test_loss_is_computed_after_buffer_fills(self, dqn_agent, env):
        # Run enough episodes to fill the replay buffer
        losses = []
        for _ in range(50):
            result = dqn_agent.train_episode(env)
            if result["avg_loss"] > 0:
                losses.append(result["avg_loss"])
        assert len(losses) > 0, "Should compute loss once buffer has enough samples"

    def test_online_net_weights_change_during_training(self, dqn_agent, env):
        # Snapshot initial weights
        initial_weights = [p.clone() for p in dqn_agent.online_net.parameters()]
        for _ in range(20):
            dqn_agent.train_episode(env)
        final_weights = [p for p in dqn_agent.online_net.parameters()]
        changed = any(
            not torch.allclose(i, f)
            for i, f in zip(initial_weights, final_weights)
        )
        assert changed, "Network weights should update during training"


# ---------------------------------------------------------------------------
# DQN Save / Load
# ---------------------------------------------------------------------------

class TestDQNSaveLoad:

    def test_save_and_reload(self, dqn_agent, env, tmp_path):
        for _ in range(3):
            dqn_agent.train_episode(env)
        save_path = tmp_path / "dqn_model.pt"
        dqn_agent.save(save_path)
        assert save_path.exists()

        loaded = DQNAgent.load(save_path, hidden_size=64)
        assert loaded.total_episodes == dqn_agent.total_episodes
        assert loaded.obs_dim == dqn_agent.obs_dim

    def test_loaded_model_produces_same_q_values(self, dqn_agent, env, tmp_path):
        for _ in range(3):
            dqn_agent.train_episode(env)
        save_path = tmp_path / "dqn_tmp.pt"
        dqn_agent.save(save_path)

        loaded = DQNAgent.load(save_path, hidden_size=64)
        obs = np.random.rand(OBS_DIM).astype(np.float32)
        q1 = dqn_agent.get_q_values(obs)
        q2 = loaded.get_q_values(obs)
        np.testing.assert_allclose(q1, q2, atol=1e-5)
