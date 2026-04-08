import os
import sys
from copy import deepcopy
from typing import Any, Dict

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.append(project_root)

from env.dp_vt_env import DPVTEnv
from env.env_visualiser import EnvVisualiser


def _build_env(
    config: Dict[Any, Any],
    init_state: np.ndarray,
    enable_visualisation: bool,
) -> Monitor:
    env_config = deepcopy(config)
    dt = env_config["dt"]
    visualiser = EnvVisualiser(init_state, dt) if enable_visualisation else None
    env_config["visualiser"] = visualiser
    env_config["render_mode"] = "human" if enable_visualisation else None
    return Monitor(DPVTEnv(init_state, **env_config))


def envaluate(
    model_path: str,
    config: Dict[Any, Any],
    init_state: np.ndarray,
    n_eval_episodes: int,
    enable_visualisation: bool,
):
    model = SAC.load(model_path)
    env = _build_env(config, init_state, enable_visualisation)
    if not enable_visualisation:
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_eval_episodes, deterministic=True
        )
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
        return

    vis = env.env.visualiser
    seed = config["seed"]
    np.random.seed(seed)
    terminated = False
    truncated = False
    state, _ = env.reset(seed=seed)
    assert vis is not None
    vis.start_animation()
    while not terminated and not truncated:
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        print(f"reward={reward:.3f}, events={info.get('events', {})}")
    plt.show()


def diagnose(model_path: str, config: Dict[Any, Any], init_state: np.ndarray) -> None:
    model = SAC.load(model_path)
    env = _build_env(config, init_state, enable_visualisation=False)
    raw_env = env.env
    seed = config["seed"]
    dt = config["dt"]
    for _ in range(10):
        _, ax = plt.subplots(2, 3)
        terminated = False
        truncated = False
        states, _ = env.reset(seed=seed)
        s, v, a, j, r = (
            [raw_env.state["array"][0]],
            [raw_env.state["array"][1]],
            [raw_env.state["array"][2]],
            [],
            [],
        )
        progress_r, st_r, v_r, a_r, j_r, d_j_r, terminal_r = [], [], [], [], [], [], []
        st = raw_env.st
        speed_limits = raw_env.speed_limit
        while not terminated and not truncated:
            action, _ = model.predict(states, deterministic=True)
            states, reward, terminated, truncated, info = env.step(action)
            state = raw_env.state
            reward_terms = info.get("reward_terms", {})
            s.append(state["array"][0])
            v.append(state["array"][1])
            a.append(state["array"][2])
            j.append(float(action[0] if np.ndim(action) else action))
            r.append(reward)
            progress_r.append(reward_terms.get("progress", 0.0))
            st_r.append(-reward_terms.get("st_cost", 0.0))
            v_r.append(-reward_terms.get("speed_cost", 0.0))
            a_r.append(-reward_terms.get("acc_cost", 0.0))
            j_r.append(-reward_terms.get("jerk_cost", 0.0))
            d_j_r.append(-reward_terms.get("djerk_cost", 0.0))
            terminal_r.append(-reward_terms.get("terminal_cost", 0.0))
        ax[0, 0].plot(s)
        for boundary in st:
            x, y = [], []
            for point in boundary.points:
                x.append(point.t() / dt)
                y.append(point.s())
            x += [x[0]]
            y += [y[0]]
            color = "g" if boundary.soft else "r"
            ax[0, 0].plot(x, y, color)
        ax[0, 1].plot(v)
        ax[0, 2].plot(a)
        ax[1, 0].plot(j)
        ax[1, 1].plot(r, label="reward")
        ax[1, 1].plot(progress_r, label="progress")
        ax[1, 1].plot(st_r, label="st")
        ax[1, 1].plot(v_r, label="speed")
        ax[1, 1].plot(a_r, label="acc")
        ax[1, 1].plot(j_r, label="jerk")
        ax[1, 1].plot(d_j_r, label="djerk")
        ax[1, 1].plot(terminal_r, label="terminal")
        ss, vv = zip(*speed_limits.speed_limit_points)
        ax[1, 2].plot(ss, vv, label="limit")
        ax[1, 2].plot(s, v, label="real")
        ax[0, 0].set_title("s-t")
        ax[0, 1].set_title("v-t")
        ax[0, 2].set_title("a-t")
        ax[1, 0].set_title("j-t")
        ax[1, 1].set_title("reward")
        ax[1, 2].set_title("v-s")
        ax[1, 1].legend()
        ax[1, 2].legend()
        plt.pause(0.5)
        plt.show()
