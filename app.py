"""
Streamlit Dashboard — Autonomous Warehouse Robot Navigation Using RL
=====================================================================
Professional interactive dashboard for demonstrating and explaining the
full RL pipeline:

    Sidebar controls → create environment → run baselines →
    train Q-Learning / DQN → compare algorithms → visualise path

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure the project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.environment.warehouse_env import WarehouseEnv
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.visualization.warehouse_renderer import WarehouseRenderer

logger = get_logger(__name__)

# ===========================================================================
# Page configuration
# ===========================================================================
st.set_page_config(
    page_title="Warehouse Robot RL",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# Custom CSS — premium dark dashboard
# ===========================================================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.85rem;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: rgba(255,255,255,0.6);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: opacity 0.2s, transform 0.1s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* Info boxes */
    .info-box {
        background: rgba(102,126,234,0.1);
        border-left: 3px solid #667eea;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: rgba(255,255,255,0.85);
        font-size: 0.9rem;
    }

    /* Reward / state display */
    .rl-state-box {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 12px;
        padding: 16px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #a8edea;
    }

    /* DataFrame tables */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ===========================================================================
# Session state helpers
# ===========================================================================
def _init_session() -> None:
    defaults = {
        "env": None,
        "renderer": None,
        "ql_agent": None,
        "dqn_agent": None,
        "ql_trained": False,
        "dqn_trained": False,
        "ql_metrics": None,
        "dqn_metrics": None,
        "comparison_df": None,
        "last_path": [],
        "last_obs": None,
        "last_action": None,
        "last_reward": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ===========================================================================
# Sidebar — configuration panel
# ===========================================================================
def _sidebar() -> dict[str, Any]:
    """Render sidebar and return user-selected configuration."""
    st.sidebar.markdown(
        """
        <div style='text-align:center; padding:16px 0 8px;'>
            <span style='font-size:2.5rem;'>🤖</span>
            <h2 style='color:white; margin:4px 0; font-size:1.1rem; font-weight:600;'>
                Warehouse Robot RL
            </h2>
            <p style='color:rgba(255,255,255,0.4); font-size:0.75rem; margin:0;'>
                MCA Applied RL Project
            </p>
        </div>
        <hr style='border-color:rgba(255,255,255,0.1);'>
        """,
        unsafe_allow_html=True,
    )

    cfg = load_config()

    st.sidebar.markdown("### 🏭 Environment")
    grid_size = st.sidebar.selectbox(
        "Grid Size", [8, 10, 15, 20],
        index=[8, 10, 15, 20].index(cfg.environment.grid_size),
        key="grid_size",
    )
    obstacle_density = st.sidebar.slider(
        "Obstacle Density", 0.05, 0.45, float(cfg.environment.obstacle_density),
        step=0.05, key="obstacle_density",
    )
    seed = st.sidebar.number_input(
        "Random Seed", min_value=0, max_value=9999,
        value=int(cfg.environment.random_seed), key="seed",
    )
    state_type = st.sidebar.radio(
        "State Type", ["basic", "advanced"],
        index=1 if cfg.environment.state_type == "advanced" else 0,
        key="state_type",
    )

    st.sidebar.markdown("### 🎯 Algorithm")
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["Random Agent", "BFS", "A*", "Q-Learning", "DQN"],
        key="algorithm",
    )

    st.sidebar.markdown("### 🧠 Training Hyperparameters")
    episodes = st.sidebar.number_input(
        "Training Episodes", min_value=100, max_value=10000,
        value=int(cfg.q_learning.episodes), step=100, key="episodes",
    )
    learning_rate = st.sidebar.select_slider(
        "Learning Rate (α)",
        options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
        value=float(cfg.q_learning.learning_rate), key="learning_rate",
    )
    gamma = st.sidebar.slider(
        "Discount Factor (γ)", 0.8, 0.999, float(cfg.q_learning.discount_factor),
        step=0.001, format="%.3f", key="gamma",
    )
    epsilon = st.sidebar.slider(
        "Initial Epsilon (ε₀)", 0.1, 1.0, float(cfg.q_learning.epsilon_start),
        step=0.05, key="epsilon",
    )
    epsilon_decay = st.sidebar.slider(
        "Epsilon Decay", 0.990, 0.9999, float(cfg.q_learning.epsilon_decay),
        step=0.0001, format="%.4f", key="epsilon_decay",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<p style='color:rgba(255,255,255,0.3); font-size:0.7rem; text-align:center;'>"
        "Built with PyTorch · Gymnasium · Streamlit</p>",
        unsafe_allow_html=True,
    )

    return {
        "grid_size": grid_size,
        "obstacle_density": obstacle_density,
        "seed": seed,
        "state_type": state_type,
        "algorithm": algorithm,
        "episodes": episodes,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon": epsilon,
        "epsilon_decay": epsilon_decay,
    }


# ===========================================================================
# Tab helpers
# ===========================================================================

def _tab_simulation(params: dict) -> None:
    """Warehouse Simulation tab."""
    st.markdown('<div class="section-header">🏭 Warehouse Simulation</div>', unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("🔄 Create New Warehouse", key="btn_create_env", use_container_width=True):
            env = WarehouseEnv(
                grid_size=params["grid_size"],
                obstacle_density=params["obstacle_density"],
                seed=params["seed"],
                state_type=params["state_type"],
            )
            env.reset()
            st.session_state["env"] = env
            st.session_state["renderer"] = WarehouseRenderer(env)
            st.session_state["last_path"] = []
            st.success(f"✅ {params['grid_size']}×{params['grid_size']} warehouse created!")

    with col_btn2:
        if st.button("▶️ Run Selected Algorithm", key="btn_run_algo", use_container_width=True):
            if st.session_state["env"] is None:
                st.warning("⚠️ Create a warehouse first.")
            else:
                _run_algorithm(params)

    with col_btn3:
        if st.button("📊 Show Path Animation", key="btn_animate", use_container_width=True):
            st.info("Path shown in grid below (arrows indicate movement).")

    # Show warehouse grid
    env = st.session_state.get("env")
    if env is not None:
        renderer = st.session_state.get("renderer") or WarehouseRenderer(env)

        col_grid, col_info = st.columns([2, 1])
        with col_grid:
            path = st.session_state.get("last_path", [])
            fig = renderer.render_plotly(path=path, title=f"Warehouse ({params['grid_size']}×{params['grid_size']})")
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.markdown("#### 🗺️ Grid Information")
            st.markdown(
                f"""
                <div class="info-box">
                    <b>Grid Size:</b> {params['grid_size']} × {params['grid_size']}<br>
                    <b>Obstacle Density:</b> {params['obstacle_density']:.0%}<br>
                    <b>Start:</b> {env.get_start_pos()}<br>
                    <b>Goal:</b> {env.get_goal_pos()}<br>
                    <b>Algorithm:</b> {params['algorithm']}<br>
                    <b>Path Length:</b> {len(st.session_state.get('last_path', []))} steps<br>
                    <b>Collisions:</b> {env.get_collision_count()}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Last run metrics
            if st.session_state.get("last_obs") is not None:
                st.markdown("#### 🧠 RL State Info")
                obs = st.session_state["last_obs"]
                action = st.session_state["last_action"]
                reward = st.session_state["last_reward"]
                state_info = renderer.get_state_info(obs, action or 0, reward or 0.0)
                st.markdown(
                    f"""<div class="rl-state-box">
                    Robot: {state_info['robot_pos']}<br>
                    Goal:  {state_info['goal_pos']}<br>
                    Dist:  {state_info['distance_to_goal']} cells<br>
                    Steps: {state_info['steps_taken']}<br>
                    Collisions: {state_info['collisions']}<br>
                    Last Action: {state_info['action']}<br>
                    Last Reward: {state_info['reward']}
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.info("👆 Click **Create New Warehouse** to get started.")


def _run_algorithm(params: dict) -> None:
    """Run the selected algorithm on the current environment."""
    env: WarehouseEnv = st.session_state["env"]
    algorithm = params["algorithm"]

    with st.spinner(f"Running {algorithm}..."):
        try:
            if algorithm == "Random Agent":
                from src.baselines.random_agent import RandomAgent
                agent = RandomAgent(seed=params["seed"])
                env.reset(seed=params["seed"])
                result = agent.run_episode(env)

            elif algorithm == "BFS":
                from src.baselines.bfs import BFSAgent
                agent = BFSAgent()
                env.reset(seed=params["seed"])
                result = agent.run_episode(env)

            elif algorithm == "A*":
                from src.baselines.astar import AStarAgent
                agent = AStarAgent()
                env.reset(seed=params["seed"])
                result = agent.run_episode(env)

            elif algorithm == "Q-Learning":
                if not st.session_state["ql_trained"]:
                    st.warning("⚠️ Train Q-Learning first (Training tab).")
                    return
                agent = st.session_state["ql_agent"]
                obs, _ = env.reset(seed=params["seed"])
                path = [env.get_robot_pos()]
                total_reward = 0.0
                done = False
                while not done:
                    action = agent.select_action(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    path.append(env.get_robot_pos())
                    done = terminated or truncated
                result = {
                    "total_reward": total_reward,
                    "steps": info["step_count"],
                    "collisions": info["collision_count"],
                    "success": terminated and not truncated,
                    "path": path,
                }
                st.session_state["last_obs"] = obs
                st.session_state["last_action"] = action
                st.session_state["last_reward"] = reward

            elif algorithm == "DQN":
                if not st.session_state["dqn_trained"]:
                    st.warning("⚠️ Train DQN first (Training tab).")
                    return
                agent = st.session_state["dqn_agent"]
                obs, _ = env.reset(seed=params["seed"])
                path = [env.get_robot_pos()]
                total_reward = 0.0
                done = False
                while not done:
                    action = agent.select_action(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    path.append(env.get_robot_pos())
                    done = terminated or truncated
                result = {
                    "total_reward": total_reward,
                    "steps": info["step_count"],
                    "collisions": info["collision_count"],
                    "success": terminated and not truncated,
                    "path": path,
                }
                st.session_state["last_obs"] = obs
                st.session_state["last_action"] = action
                st.session_state["last_reward"] = reward

            else:
                st.error(f"Unknown algorithm: {algorithm}")
                return

            st.session_state["last_path"] = result.get("path", [])
            success_icon = "✅" if result["success"] else "❌"
            st.success(
                f"{success_icon} {algorithm} finished — "
                f"Steps: {result['steps']} | "
                f"Reward: {result['total_reward']:.1f} | "
                f"Collisions: {result['collisions']}"
            )

        except Exception as e:
            st.error(f"Error running {algorithm}: {e}")
            logger.exception(f"Algorithm execution error: {e}")


def _tab_training(params: dict) -> None:
    """Training tab."""
    st.markdown('<div class="section-header">🎓 Training</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Q-Learning")
        if st.button("🚀 Train Q-Learning", key="btn_train_ql", use_container_width=True):
            _train_q_learning(params)

        if st.session_state["ql_trained"]:
            metrics = st.session_state["ql_metrics"]
            st.markdown(
                f"""<div class="metric-card">
                <div class="metric-value">{metrics['eval_metrics']['success_rate']:.1%}</div>
                <div class="metric-label">Success Rate</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### DQN (Deep Q-Network)")
        if st.button("🚀 Train DQN", key="btn_train_dqn", use_container_width=True):
            _train_dqn(params)

        if st.session_state["dqn_trained"]:
            metrics = st.session_state["dqn_metrics"]
            st.markdown(
                f"""<div class="metric-card">
                <div class="metric-value">{metrics['eval_metrics']['success_rate']:.1%}</div>
                <div class="metric-label">Success Rate</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # Show reward curves
    if st.session_state["ql_trained"] or st.session_state["dqn_trained"]:
        _tab_training_analysis()


def _train_q_learning(params: dict) -> None:
    """Train Q-Learning and store results."""
    from src.training.train_q_learning import train_q_learning

    progress_bar = st.progress(0, text="Initialising Q-Learning training...")
    status_text = st.empty()

    try:
        with st.spinner("Training Q-Learning..."):
            results = train_q_learning(
                episodes=params["episodes"],
                grid_size=params["grid_size"],
                obstacle_density=params["obstacle_density"],
                seed=params["seed"],
                learning_rate=params["learning_rate"],
                discount_factor=params["gamma"],
                epsilon_start=params["epsilon"],
                epsilon_decay=params["epsilon_decay"],
                verbose=False,
            )

        st.session_state["ql_agent"] = results["agent"]
        st.session_state["ql_trained"] = True
        st.session_state["ql_metrics"] = results
        progress_bar.progress(1.0)
        status_text.success(
            f"✅ Q-Learning trained! "
            f"Success: {results['eval_metrics']['success_rate']:.1%} | "
            f"Time: {results['training_time_s']:.1f}s"
        )

    except Exception as e:
        progress_bar.empty()
        status_text.error(f"Training failed: {e}")
        logger.exception(e)


def _train_dqn(params: dict) -> None:
    """Train DQN and store results."""
    from src.training.train_dqn import train_dqn

    status_text = st.empty()

    try:
        with st.spinner("Training DQN (this may take a few minutes)..."):
            results = train_dqn(
                episodes=params["episodes"],
                grid_size=params["grid_size"],
                obstacle_density=params["obstacle_density"],
                seed=params["seed"],
                learning_rate=0.001,
                discount_factor=params["gamma"],
                epsilon_start=params["epsilon"],
                epsilon_decay=params["epsilon_decay"],
                verbose=False,
            )

        st.session_state["dqn_agent"] = results["agent"]
        st.session_state["dqn_trained"] = True
        st.session_state["dqn_metrics"] = results
        status_text.success(
            f"✅ DQN trained! "
            f"Success: {results['eval_metrics']['success_rate']:.1%} | "
            f"Time: {results['training_time_s']:.1f}s"
        )

    except Exception as e:
        status_text.error(f"DQN training failed: {e}")
        logger.exception(e)


def _tab_training_analysis() -> None:
    """Training analysis sub-section with reward/loss curves."""
    st.markdown("---")
    st.markdown("#### 📈 Training Analysis")

    tabs = []
    if st.session_state["ql_trained"]:
        tabs.append("Q-Learning")
    if st.session_state["dqn_trained"]:
        tabs.append("DQN")

    if not tabs:
        return

    selected = st.tabs(tabs)

    idx = 0
    if st.session_state["ql_trained"]:
        with selected[idx]:
            df: pd.DataFrame = st.session_state["ql_metrics"]["metrics_df"]
            _plot_training_curves(df, "Q-Learning", include_loss=False)
        idx += 1

    if st.session_state["dqn_trained"]:
        with selected[idx]:
            df = st.session_state["dqn_metrics"]["metrics_df"]
            _plot_training_curves(df, "DQN", include_loss=True)


def _plot_training_curves(df: pd.DataFrame, label: str, include_loss: bool) -> None:
    """Plot reward, episode length, and optionally loss curves using Plotly."""
    episodes = df["episode"].values
    window = 50

    # Reward curve
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        x=episodes, y=df["total_reward"].values,
        name="Episode Reward", opacity=0.3, line=dict(color="#4A90D9", width=1),
    ))
    fig_r.add_trace(go.Scatter(
        x=episodes, y=df["total_reward"].rolling(window, min_periods=1).mean().values,
        name=f"Moving Avg ({window})", line=dict(color="#E74C3C", width=2),
    ))
    fig_r.update_layout(
        title=f"{label} — Reward Curve",
        xaxis_title="Episode", yaxis_title="Reward",
        template="plotly_dark", height=300, margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # Episode length
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=episodes, y=df["steps"].values,
        name="Steps", opacity=0.3, line=dict(color="#27AE60", width=1),
    ))
    fig_s.add_trace(go.Scatter(
        x=episodes, y=df["steps"].rolling(window, min_periods=1).mean().values,
        name="Moving Avg", line=dict(color="#1A6B3C", width=2),
    ))
    fig_s.update_layout(
        title=f"{label} — Episode Length",
        xaxis_title="Episode", yaxis_title="Steps",
        template="plotly_dark", height=280, margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_s, use_container_width=True)

    # DQN loss
    if include_loss and "avg_loss" in df.columns:
        loss_df = df[df["avg_loss"] > 0]
        if not loss_df.empty:
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(
                x=loss_df["episode"].values, y=loss_df["avg_loss"].values,
                name="Loss", opacity=0.4, line=dict(color="#E67E22", width=1),
            ))
            fig_l.add_trace(go.Scatter(
                x=loss_df["episode"].values,
                y=loss_df["avg_loss"].rolling(window, min_periods=1).mean().values,
                name="Smoothed", line=dict(color="#C0392B", width=2),
            ))
            fig_l.update_layout(
                title="DQN — Training Loss (Huber)",
                xaxis_title="Episode", yaxis_title="Loss",
                template="plotly_dark", height=280, margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_l, use_container_width=True)

    # Success rate
    success_roll = df["success"].astype(float).rolling(window, min_periods=1).mean()
    fig_sr = go.Figure()
    fig_sr.add_trace(go.Scatter(
        x=episodes, y=success_roll.values,
        name="Success Rate", fill="tozeroy", line=dict(color="#F39C12", width=2),
        fillcolor="rgba(243,156,18,0.1)",
    ))
    fig_sr.update_layout(
        title=f"{label} — Success Rate (rolling {window})",
        xaxis_title="Episode", yaxis_title="Success Rate",
        yaxis_range=[0, 1],
        template="plotly_dark", height=280, margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_sr, use_container_width=True)


def _tab_evaluation(params: dict) -> None:
    """Evaluation tab — metrics cards."""
    st.markdown('<div class="section-header">📊 Evaluation Metrics</div>', unsafe_allow_html=True)

    if not st.session_state["ql_trained"] and not st.session_state["dqn_trained"]:
        st.info("Train at least one RL agent to see evaluation metrics here.")
        return

    for algo, key in [("Q-Learning", "ql"), ("DQN", "dqn")]:
        if st.session_state[f"{key}_trained"]:
            metrics = st.session_state[f"{key}_metrics"]["eval_metrics"]
            st.markdown(f"#### {algo}")
            c1, c2, c3, c4 = st.columns(4)
            for col, label, value, fmt in [
                (c1, "Success Rate", metrics["success_rate"], ".1%"),
                (c2, "Avg Reward", metrics["avg_reward"], ".1f"),
                (c3, "Avg Steps", metrics["avg_steps"], ".1f"),
                (c4, "Avg Collisions", metrics["avg_collisions"], ".2f"),
            ]:
                with col:
                    formatted = f"{value:{fmt}}"
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="metric-value">{formatted}</div>
                        <div class="metric-label">{label}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            st.markdown("")


def _tab_comparison(params: dict) -> None:
    """Algorithm comparison tab."""
    st.markdown('<div class="section-header">⚖️ Algorithm Comparison</div>', unsafe_allow_html=True)

    if st.button("🔄 Run Full Comparison", key="btn_compare", use_container_width=False):
        _run_comparison(params)

    df = st.session_state.get("comparison_df")
    if df is not None:
        # Show table
        st.markdown("#### 📋 Comparison Table")
        display_df = df.copy()
        if "success_rate" in display_df:
            display_df["success_rate"] = display_df["success_rate"].map("{:.1%}".format)
        if "avg_reward" in display_df:
            display_df["avg_reward"] = display_df["avg_reward"].map("{:.1f}".format)
        if "avg_steps" in display_df:
            display_df["avg_steps"] = display_df["avg_steps"].map("{:.1f}".format)
        if "avg_collisions" in display_df:
            display_df["avg_collisions"] = display_df["avg_collisions"].map("{:.2f}".format)
        st.dataframe(display_df, use_container_width=True)

        # Charts
        _comparison_charts(df)


def _run_comparison(params: dict) -> None:
    """Run all available algorithms and store comparison results."""
    from src.evaluation.evaluate import Evaluator

    with st.spinner("Running comparison across all algorithms (100 episodes each)..."):
        evaluator = Evaluator()
        ql_path = None
        dqn_path = None

        if st.session_state["ql_trained"]:
            # Save temp model for evaluator
            tmp = Path("results/tmp_ql.pkl")
            st.session_state["ql_agent"].save(tmp)
            ql_path = tmp

        if st.session_state["dqn_trained"]:
            tmp = Path("results/tmp_dqn.pt")
            st.session_state["dqn_agent"].save(tmp)
            dqn_path = tmp

        df = evaluator.compare_all(
            grid_size=params["grid_size"],
            obstacle_density=params["obstacle_density"],
            seed=params["seed"],
            num_episodes=50,  # Keep quick for UI
            ql_model_path=ql_path,
            dqn_model_path=dqn_path,
        )
        st.session_state["comparison_df"] = df
        st.success("✅ Comparison complete!")


def _comparison_charts(df: pd.DataFrame) -> None:
    """Plot bar charts for the comparison."""
    import plotly.express as px

    algorithms = df["algorithm"].tolist()
    colour_seq = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12"][:len(algorithms)]

    col1, col2 = st.columns(2)
    with col1:
        if "success_rate" in df:
            fig = px.bar(
                df, x="algorithm", y="success_rate",
                title="Success Rate", color="algorithm",
                color_discrete_sequence=colour_seq,
                text_auto=".1%",
            )
            fig.update_layout(
                template="plotly_dark", showlegend=False, height=350,
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig, use_container_width=True)

        if "avg_collisions" in df:
            fig = px.bar(
                df, x="algorithm", y="avg_collisions",
                title="Avg Collisions per Episode", color="algorithm",
                color_discrete_sequence=colour_seq, text_auto=".2f",
            )
            fig.update_layout(template="plotly_dark", showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "avg_steps" in df:
            fig = px.bar(
                df, x="algorithm", y="avg_steps",
                title="Avg Steps to Goal", color="algorithm",
                color_discrete_sequence=colour_seq, text_auto=".1f",
            )
            fig.update_layout(template="plotly_dark", showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        if "avg_reward" in df:
            fig = px.bar(
                df, x="algorithm", y="avg_reward",
                title="Avg Episode Reward", color="algorithm",
                color_discrete_sequence=colour_seq, text_auto=".1f",
            )
            fig.update_layout(template="plotly_dark", showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)


def _tab_rl_explainer(params: dict) -> None:
    """RL Concepts explainer tab for demonstrations."""
    st.markdown('<div class="section-header">📚 RL Concepts Explainer</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
        This section explains how Reinforcement Learning works in this warehouse problem.
        Use this during your project demonstration!
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🌍 Environment (MDP)
        The warehouse is modelled as a **Markov Decision Process (MDP)**:
        - **State**: Robot position + Goal position (+ local obstacles in advanced mode)
        - **Actions**: UP, DOWN, LEFT, RIGHT (4 discrete actions)
        - **Transition**: Deterministic (action always succeeds unless blocked)
        - **Reward**: Shaped to guide the robot

        #### 🎁 Reward Shaping
        | Event | Reward |
        |-------|--------|
        | Reach goal | **+100** |
        | Valid step | -1 |
        | Move closer | +2 |
        | Move farther | -2 |
        | Collision | -20 |
        | Timeout | -10 |
        """)

    with col2:
        st.markdown("""
        #### 🧠 Q-Learning
        Uses a **Q-table** to store value estimates:
        ```
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        ```
        - α = learning rate (how fast to update)
        - γ = discount factor (future reward importance)
        - ε = exploration rate (try new things vs exploit known)

        #### 🔥 DQN
        Replaces the Q-table with a **neural network**:
        - **Online net**: learns Q-values
        - **Target net**: provides stable training targets
        - **Replay buffer**: breaks temporal correlation
        - **Gradient clipping**: prevents instability
        """)

    st.markdown("---")
    st.markdown("""
    #### 🆚 RL vs Traditional Pathfinding

    | Property | BFS | A* | Q-Learning | DQN |
    |----------|-----|----|-----------|-----|
    | **Needs full map?** | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
    | **Generalises?** | ❌ No | ❌ No | ✅ Partially | ✅ Yes |
    | **Optimal?** | ✅ Yes | ✅ Yes | Sometimes | Sometimes |
    | **Scales to unknown env?** | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
    | **Handles continuous state?** | ❌ No | ❌ No | ❌ No | ✅ Yes |
    """)

    # Current Q-values display
    if st.session_state["ql_trained"] and st.session_state["env"] is not None:
        st.markdown("---")
        st.markdown("#### 🔢 Live Q-Values (Q-Learning)")
        env = st.session_state["env"]
        obs, _ = env.reset()
        agent = st.session_state["ql_agent"]
        q_vals = agent.get_q_values_for_obs(obs)
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

        fig = go.Figure(go.Bar(
            x=action_names, y=q_vals,
            marker_color=["#E74C3C" if v == max(q_vals) else "#3498DB" for v in q_vals],
            text=[f"{v:.2f}" for v in q_vals],
            textposition="outside",
        ))
        fig.update_layout(
            title="Q-Values for Current State (best action = red)",
            template="plotly_dark", height=300,
            yaxis_title="Q-Value",
            xaxis_title="Action",
        )
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# Main app
# ===========================================================================
def main() -> None:
    # Header
    st.markdown(
        """
        <div style='text-align:center; padding:20px 0 10px;'>
            <h1 style='
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 4px;
            '>🤖 Autonomous Warehouse Robot Navigation</h1>
            <p style='color:rgba(255,255,255,0.5); font-size:0.9rem;'>
                MCA Applied Reinforcement Learning · Comparing RL vs Classical Pathfinding
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    params = _sidebar()

    # Main tabs
    tab_sim, tab_train, tab_eval, tab_compare, tab_explain = st.tabs([
        "🏭 Simulation",
        "🎓 Training",
        "📊 Evaluation",
        "⚖️ Comparison",
        "📚 RL Concepts",
    ])

    with tab_sim:
        _tab_simulation(params)

    with tab_train:
        _tab_training(params)

    with tab_eval:
        _tab_evaluation(params)

    with tab_compare:
        _tab_comparison(params)

    with tab_explain:
        _tab_rl_explainer(params)


if __name__ == "__main__":
    main()
