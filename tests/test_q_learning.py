"""
Tests for Q-Learning agent.

Covers:
    - Q-table creation
    - Action selection (greedy and random)
    - Q-value update (Bellman equation)
    - Epsilon decay
    - Training episode completion
    - Save and load
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.agents.q_learning import QLearningAgent
from src.environment.warehouse_env import WarehouseEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    return QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99,
        n_bins=10,
    )


@pytest.fixture
def env():
    e = WarehouseEnv(grid_size=8, obstacle_density=0.1, seed=42)
    e.reset()
    return e


@pytest.fixture
def sample_obs():
    return np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)


# ---------------------------------------------------------------------------
# Q-table
# ---------------------------------------------------------------------------

class TestQTable:

    def test_q_table_starts_empty(self, agent):
        assert len(agent.q_table) == 0

    def test_q_values_initialised_on_first_access(self, agent, sample_obs):
        state = agent._discretise(sample_obs)
        q_vals = agent._get_q_values(state)
        assert q_vals.shape == (4,)
        assert np.all(q_vals == 0.0)

    def test_q_table_grows_with_unique_states(self, agent):
        obs1 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        obs2 = np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32)
        agent._get_q_values(agent._discretise(obs1))
        agent._get_q_values(agent._discretise(obs2))
        assert len(agent.q_table) == 2


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

class TestActionSelection:

    def test_greedy_selects_best_action(self, agent, sample_obs):
        state = agent._discretise(sample_obs)
        agent.q_table[state] = np.array([0.0, 5.0, 0.0, 0.0])
        action = agent.select_action(sample_obs, deterministic=True)
        assert action == 1, "Greedy should select action with highest Q-value"

    def test_random_action_in_valid_range(self, agent, sample_obs):
        agent.epsilon = 1.0  # Always explore
        for _ in range(20):
            action = agent.select_action(sample_obs)
            assert 0 <= action <= 3

    def test_deterministic_mode_never_explores(self, agent, sample_obs):
        agent.epsilon = 1.0
        state = agent._discretise(sample_obs)
        agent.q_table[state] = np.array([0.0, 0.0, 99.0, 0.0])
        for _ in range(20):
            action = agent.select_action(sample_obs, deterministic=True)
            assert action == 2, "Deterministic mode must always be greedy"


# ---------------------------------------------------------------------------
# Q-value update
# ---------------------------------------------------------------------------

class TestQValueUpdate:

    def test_update_increases_q_for_positive_reward(self, agent, sample_obs):
        obs = sample_obs
        next_obs = np.array([0.2, 0.3, 0.8, 0.9], dtype=np.float32)
        state = agent._discretise(obs)
        agent._get_q_values(state)  # Initialise to zero

        td_err = agent.update(obs, action=1, reward=100.0, next_obs=next_obs, done=True)
        q_after = agent.q_table[state][1]
        assert q_after > 0.0, "Q-value should increase after positive reward"
        assert td_err > 0.0

    def test_bellman_update_correct(self, agent):
        """Verify the Bellman update formula numerically."""
        obs = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
        next_obs = np.array([0.2, 0.1, 0.9, 0.9], dtype=np.float32)
        state = agent._discretise(obs)
        next_state = agent._discretise(next_obs)

        # Set up known Q-values
        agent.q_table[state] = np.array([0.0, 0.0, 0.0, 0.0])
        agent.q_table[next_state] = np.array([10.0, 20.0, 5.0, 15.0])

        alpha, gamma = 0.1, 0.99
        reward = 5.0
        q_before = 0.0
        expected_q = q_before + alpha * (reward + gamma * 20.0 - q_before)

        agent.update(obs, action=0, reward=reward, next_obs=next_obs, done=False)
        assert abs(agent.q_table[state][0] - expected_q) < 1e-6

    def test_terminal_state_ignores_next_q(self, agent, sample_obs):
        """When done=True, the target should be just r (no future reward)."""
        obs = sample_obs
        next_obs = np.array([0.9, 0.9, 0.8, 0.8], dtype=np.float32)
        next_state = agent._discretise(next_obs)
        agent.q_table[next_state] = np.array([1000.0, 1000.0, 1000.0, 1000.0])

        state = agent._discretise(obs)
        agent.q_table[state] = np.array([0.0, 0.0, 0.0, 0.0])

        reward = 100.0
        alpha = agent.alpha
        expected = 0.0 + alpha * (reward + 0.0 - 0.0)  # γ * max(Q') = 0 when done

        agent.update(obs, action=0, reward=reward, next_obs=next_obs, done=True)
        assert abs(agent.q_table[state][0] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Epsilon decay
# ---------------------------------------------------------------------------

class TestEpsilonDecay:

    def test_epsilon_decays_each_episode(self, agent):
        eps_before = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < eps_before

    def test_epsilon_never_below_minimum(self, agent):
        agent.epsilon = 0.001
        for _ in range(100):
            agent.decay_epsilon()
        assert agent.epsilon >= agent.epsilon_end

    def test_epsilon_decay_rate(self, agent):
        agent.epsilon = 1.0
        agent.decay_epsilon()
        expected = max(agent.epsilon_end, 1.0 * agent.epsilon_decay)
        assert abs(agent.epsilon - expected) < 1e-10


# ---------------------------------------------------------------------------
# Training episode
# ---------------------------------------------------------------------------

class TestTrainingEpisode:

    def test_train_episode_returns_correct_keys(self, agent, env):
        result = agent.train_episode(env)
        assert "total_reward" in result
        assert "steps" in result
        assert "success" in result
        assert "collisions" in result
        assert "epsilon" in result

    def test_q_table_grows_during_training(self, agent, env):
        for _ in range(10):
            agent.train_episode(env)
        assert len(agent.q_table) > 0

    def test_history_accumulates(self, agent, env):
        n = 5
        for _ in range(n):
            agent.train_episode(env)
        assert len(agent.episode_rewards) == n
        assert len(agent.epsilon_history) == n


# ---------------------------------------------------------------------------
# Save and Load
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_and_load_q_table(self, agent, env, tmp_path):
        # Train briefly
        for _ in range(5):
            agent.train_episode(env)
        save_path = tmp_path / "q_model.pkl"
        agent.save(save_path)
        assert save_path.exists()

        loaded = QLearningAgent.load(save_path)
        assert len(loaded.q_table) == len(agent.q_table)
        assert loaded.alpha == agent.alpha
        assert loaded.total_episodes == agent.total_episodes
