import os
import sys

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from typing import Dict, Any
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from env.dp_vt_env import DPVTEnv, EnvVisualiser
from matplotlib import pyplot as plt


def envaluate(
    model_path: str,
    config: Dict[Any, Any],
    init_state: np.ndarray[float],
    n_eval_episodes: int,
    enable_visualisation: bool,
):
    model = SAC.load(model_path)
    dt = config["dt"]
    vis = EnvVisualiser(init_state, dt)
    config["render_mode"] = "human" if enable_visualisation else None
    seed = config["seed"]
    env = DPVTEnv(init_state, **config, visualiser=vis)
    if not enable_visualisation:
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_eval_episodes
        )
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
        return

    np.random.seed(seed)
    terminate = False
    trancate = False
    state, _ = env.reset(seed)
    vis.start_animation()
    while not terminate and not trancate:
        u, _ = model.predict(state, deterministic=True)
        state, reward, terminate, trancate, _ = env.step(u)
    plt.show()


def diagnose(model: str, config: Dict[Any, Any], init_state: np.ndarray):
    model = SAC.load(model)
    config["visualiser"] = None
    env = DPVTEnv(init_state, **config)
    seed = config["seed"]
    dt = config["dt"]
    for _ in range(10):
        fig, ax = plt.subplots(2, 3)
        terminate = False
        trancate = False
        states, _ = env.reset(seed)
        s, v, a, j, r = (
            [env.state["array"][0]],
            [env.state["array"][1]],
            [env.state["array"][2]],
            [],
            [],
        )
        st_r, v_r, a_r, j_r, d_j_r = [], [], [], [], []
        st = env.st
        speed_limits = env.speed_limit
        while not terminate and not trancate:
            u, _ = model.predict(states, deterministic=True)
            states, reward, terminate, trancate, _ = env.step(u)
            state = env.state
            s.append(state["array"][0])
            v.append(state["array"][1])
            a.append(state["array"][2])
            j.append(u[0, 0])
            r.append(reward)
            st_r.append(-env.st_cost)
            v_r.append(-env.speed_cost)
            a_r.append(-env.acc_cost)
            j_r.append(-env.jerk_cost)
            d_j_r.append(-env.d_j_cost)
        ax[0, 0].plot(s)
        for boundary in st:
            x, y = [], []
            for p in boundary.points:
                x.append(p.t() / dt)
                y.append(p.s())

            x += [x[0]]
            y += [y[0]]
            color = "g" if boundary.soft else "r"
            ax[0, 0].plot(x, y, color)
        ax[0, 1].plot(v)
        ax[0, 2].plot(a)
        ax[1, 0].plot(j)
        ax[1, 1].plot(r, label="reward")
        ax[1, 1].plot(st_r, label="st")
        ax[1, 1].plot(v_r, label="v")
        ax[1, 1].plot(a_r, label="a")
        ax[1, 1].plot(j_r, label="j")
        ax[1, 1].plot(d_j_r, label="dj")
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
