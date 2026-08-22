# Autonomous Warehouse Robot Navigation Using Reinforcement Learning

> **MCA Applied Reinforcement Learning Project**  
> Comparing RL agents (Q-Learning, DQN) against classical pathfinding (BFS, A*) on autonomous warehouse navigation.

---

## 🎯 Problem Statement

Can reinforcement learning learn an efficient collision-free warehouse navigation policy, and how does it compare with traditional pathfinding algorithms?

A simulated warehouse robot must learn to:
- Navigate from a start position to a target/package
- Avoid shelves (obstacles) and boundaries
- Minimise steps and collisions
- Generalise to unseen warehouse layouts

---

## 🏗️ Project Structure

```
warehouse_robot_rl/
├── app.py                          # Streamlit dashboard (main entry point)
├── config/
│   └── config.yaml                 # All hyperparameters (centralised)
├── src/
│   ├── environment/
│   │   └── warehouse_env.py        # Custom Gymnasium environment
│   ├── agents/
│   │   ├── q_learning.py           # Tabular Q-Learning
│   │   ├── dqn.py                  # Deep Q-Network (PyTorch)
│   │   └── replay_buffer.py        # Experience replay buffer
│   ├── baselines/
│   │   ├── random_agent.py         # Random action baseline
│   │   ├── bfs.py                  # BFS pathfinding
│   │   └── astar.py                # A* pathfinding
│   ├── training/
│   │   ├── train_q_learning.py     # Q-Learning training pipeline
│   │   └── train_dqn.py            # DQN training pipeline
│   ├── evaluation/
│   │   └── evaluate.py             # Algorithm comparison engine
│   ├── visualization/
│   │   └── warehouse_renderer.py   # Matplotlib/Plotly/RGB renderer
│   └── utils/
│       ├── config.py               # Config loader
│       └── logger.py               # Coloured logger
├── tests/
│   ├── test_environment.py
│   ├── test_q_learning.py
│   ├── test_dqn.py
│   └── test_pathfinding.py
├── models/                         # Saved model checkpoints
├── results/                        # Training metrics and plots
│   ├── q_learning/
│   └── dqn/
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Linux/macOS
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

---

## 🤖 Algorithms

| Algorithm | Type | Needs Map? | Generalises? |
|-----------|------|-----------|-------------|
| Random Agent | Baseline | No | No |
| BFS | Classical | Yes (full) | No |
| A* (Manhattan) | Classical | Yes (full) | No |
| Q-Learning | RL (tabular) | No | Partially |
| DQN | RL (neural) | No | Yes |

---

## 🌍 Environment

A configurable 2D grid warehouse:

```
S . . X . . . .
. . . X . X . .
. X . . . X . .
. X X . . . . .
. . . . X . . .
. . X . . . X .
. . . . . . . .
. . . X . . . G
```

- `S` = Robot start  
- `G` = Goal  
- `X` = Obstacle (shelf)  
- `.` = Free space  

Supported sizes: **8×8, 10×10, 15×15, 20×20**  
Random layouts with **guaranteed BFS reachability**.

---

## 🎁 Reward Shaping

| Event | Reward | Reason |
|-------|--------|--------|
| Reach goal | **+100** | Primary objective |
| Valid step | -1 | Encourage shortest paths |
| Move closer | +2 | Guide toward goal |
| Move farther | -2 | Penalise backtracking |
| Collision | -20 | Strong obstacle avoidance |
| Boundary | -20 | Same as obstacle |
| Timeout | -10 | Penalise wandering |

---

## 🧠 Q-Learning

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]
```

- State discretised into `n_bins` × `obs_dim` buckets
- ε-greedy exploration with exponential decay
- Configurable via `config/config.yaml`

**Train from command line:**
```bash
python -m src.training.train_q_learning --episodes 3000
```

---

## 🔥 DQN

Neural network architecture:
```
State (obs_dim)
    ↓ Linear
    ↓ ReLU
    ↓ Linear  
    ↓ ReLU
    → Q-values (4)
```

Stabilisation features:
- Experience replay buffer (10,000 transitions)
- Target network (synced every 10 episodes)
- Gradient clipping
- Adam optimizer

**Train from command line:**
```bash
python -m src.training.train_dqn --episodes 2000
```

---

## 🧪 Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src
```

Test coverage:
- **Environment**: Initialization, placement, collisions, reachability (5 seeds)
- **Q-Learning**: Bellman update, epsilon decay, save/load
- **DQN**: Network forward pass, replay buffer, training mechanics
- **Pathfinding**: BFS/A* path validity, obstacle avoidance, unreachable handling

---

## 📊 Training Outputs

Each training run creates a timestamped experiment directory:

```
results/
├── q_learning/
│   └── 20240813_153000/
│       ├── config.json
│       ├── training_metrics.csv
│       ├── eval_results.json
│       ├── q_learning_final.pkl
│       └── plots/
│           ├── reward_curve.png
│           ├── episode_length.png
│           ├── epsilon_decay.png
│           └── success_rate.png
└── dqn/
    └── 20240813_154500/
        ├── dqn_final.pt
        ├── training_metrics.csv
        └── plots/
            ├── reward_curve.png
            ├── dqn_loss.png
            └── success_rate.png
```

---

## 🖥️ Streamlit Dashboard Features

| Tab | Features |
|-----|---------|
| 🏭 Simulation | Create warehouse, run any algorithm, visualise path |
| 🎓 Training | Train Q-Learning / DQN with live progress |
| 📊 Evaluation | Success rate, reward, steps, collision metrics |
| ⚖️ Comparison | Run all 5 algorithms and compare with charts |
| 📚 RL Concepts | Explain MDP, Q-Learning, DQN with live Q-values |

---

## 🔬 Evaluation Scenarios

The evaluator tests all algorithms under:

1. **Low density** (10% obstacles)
2. **Medium density** (20% obstacles)  
3. **High density** (35% obstacles)
4. **Unseen layouts** (new random seeds)

---

## 📈 Configuration

All hyperparameters in `config/config.yaml`:

```yaml
q_learning:
  learning_rate: 0.1
  discount_factor: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  episodes: 3000

dqn:
  learning_rate: 0.001
  batch_size: 64
  replay_buffer_size: 10000
  target_update_freq: 10
  hidden_size: 128
```

---

## 🎓 Academic Context

This project demonstrates:
1. **RL vs Classical AI** — side-by-side comparison
2. **MDP formulation** — state, action, reward, transition
3. **Exploration vs Exploitation** — ε-greedy with decay
4. **Neural function approximation** — DQN replacing Q-table
5. **Generalisation** — training on one set, testing on new layouts
6. **Reward engineering** — shaping agent behaviour

---

## 📚 References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Watkins & Dayan (1992). *Q-learning*. Machine Learning.
- Hart, Nilsson & Raphael (1968). *A formal basis for the heuristic determination of minimum cost paths*. IEEE.
- Gymnasium: https://gymnasium.farama.org
- PyTorch: https://pytorch.org
