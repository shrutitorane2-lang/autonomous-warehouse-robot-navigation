import sys
sys.path.insert(0, '.')
from src.environment.warehouse_env import WarehouseEnv
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.agents.replay_buffer import ReplayBuffer
from src.baselines.random_agent import RandomAgent
from src.baselines.bfs import BFSAgent
from src.baselines.astar import AStarAgent
from src.training.train_q_learning import train_q_learning
from src.training.train_dqn import train_dqn
from src.evaluation.evaluate import Evaluator
from src.visualization.warehouse_renderer import WarehouseRenderer
from src.utils.config import load_config
print("All imports successful!")

env = WarehouseEnv(grid_size=8, obstacle_density=0.2, seed=42)
obs, info = env.reset()
print(f"Env obs shape: {obs.shape}")
print(f"Start: {env.get_start_pos()}, Goal: {env.get_goal_pos()}")

bfs = BFSAgent()
result = bfs.run_episode(env)
print(f"BFS: success={result['success']}, steps={result['steps']}")

agent = QLearningAgent()
r = agent.train_episode(env)
print(f"Q-Learning: reward={r['total_reward']:.1f}, steps={r['steps']}")

print("Smoke test PASSED!")
