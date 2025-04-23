import os
import sys

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.append(project_root)

from typing import Dict, Any
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.type_aliases import TrainFrequencyUnit

from env.dp_vt_env import DPVTEnv
from env.env_visualiser import EnvVisualiser
from train.train_utils import (
    register_schedule,
    MultiEnvEpisodeEndTrainingCallback,
    SACTensorboardCallBack,
    SaveBestModelCallback,
    DictReplayBuffer,
)
from train.model_components import MixedFeaturesExtractor
import matplotlib.pyplot as plt


def train(
    init_state: np.ndarray,
    config: Dict[str, Any],
    mode_name: str,
    device_id: int = 0,
    use_multi_env: bool = True,
    n_envs: int = 1,
    use_custom_log: bool = False,
    show_plot: bool = False,
) -> None:
    dt = config["dp_vt_config"]["dt"]
    vis = EnvVisualiser(init_state, dt)
    params = config["dp_vt_config"]
    render_mode = None
    if show_plot:
        vis.start_animation()
        render_mode = "human"
    params["render_mode"] = render_mode
    params["visualiser"] = vis
    params["init_dynamic_state"] = init_state
    if th.cuda.is_available():
        th.cuda.set_device(device_id)
    env = make_vec_env(DPVTEnv, n_envs=n_envs, env_kwargs=params)

    sac_constructor_config = config["model_config"]["SAC_constructor"]
    if "activation_fn" in sac_constructor_config["policy_kwargs"]:
        # TODO
        sac_constructor_config["policy_kwargs"]["activation_fn"] = eval(
            sac_constructor_config["policy_kwargs"]["activation_fn"]
        )

    sac_constructor_config["policy"] = "MultiInputPolicy"
    sac_constructor_config["policy_kwargs"][
        "features_extractor_class"
    ] = MixedFeaturesExtractor

    sac_constructor_config["buffer_size"] = int(sac_constructor_config["buffer_size"])
    lr_schedule = sac_constructor_config["learning_rate"]
    if "learning_schedule" in config["model_config"]:
        learning_schedule = config["model_config"]["learning_schedule"]
        lr_schedule = get_schedule_fn(register_schedule(**learning_schedule))
    sac_constructor_config["learning_rate"] = lr_schedule
    action_noise = NormalActionNoise(mean=np.array([0]), sigma=np.array([0.1]))
    sac_constructor_config["action_noise"] = action_noise
    if not use_multi_env:
        assert n_envs == 1
        sac_constructor_config["train_freq"] = (1, "episode")
        sac_constructor_config["gradient_steps"] = 1
    sac_constructor_config["device"] = f"cuda:{device_id}"
    model = SAC(env=env, **sac_constructor_config)
    learn_params = config["model_config"]["train_param"]
    callbacks = [SaveBestModelCallback(mode_name)]
    if use_custom_log:
        callbacks.append(SACTensorboardCallBack())
    if use_multi_env:
        callbacks.append(MultiEnvEpisodeEndTrainingCallback())

    learn_params["callback"] = CallbackList(callbacks)
    model.learn(**learn_params)
    model.save(mode_name)
    if show_plot:
        plt.show()


def train_from_local_model(
    model_name: str,
    model_out: str,
    config: Dict[str, Any],
    init_state: np.ndarray,
    device_id: int = 0,
    use_multi_env: bool = True,
    use_custom_log: bool = False,
) -> None:
    params = config["dp_vt_config"]
    th.cuda.set_device(device_id)
    env = DPVTEnv(init_state, **params, visualiser=None)
    model = SAC.load(model_name, env=env, device=f"cuda:{device_id}")
    sac_constructor_config = config["model_config"]["SAC_constructor"]
    model.ent_coef = sac_constructor_config["ent_coef"]
    model.tau = sac_constructor_config["tau"]
    model.replay_buffer.buffer_size = int(sac_constructor_config["buffer_size"])
    model.batch_size = sac_constructor_config["batch_size"]
    model.train_freq = model.train_freq._replace(
        frequency=1, unit=TrainFrequencyUnit.EPISODE
    )
    model.gradient_steps = 1
    if "learning_schedule" in config["model_config"]:
        learning_schedule = config["model_config"]["learning_schedule"]
        model.learning_rate = get_schedule_fn(register_schedule(**learning_schedule))

    model.replay_buffer = DictReplayBuffer(
        model.replay_buffer.buffer_size,
        model.replay_buffer.observation_space,
        model.replay_buffer.action_space,
        device=model.replay_buffer.device,
        n_envs=model.replay_buffer.n_envs,
        optimize_memory_usage=model.replay_buffer.optimize_memory_usage,
    )
    learn_params = config["model_config"]["train_param"]
    callbacks = [SaveBestModelCallback(model_out)]
    if use_custom_log:
        callbacks.append(SACTensorboardCallBack())
    if use_multi_env:
        callbacks.append(MultiEnvEpisodeEndTrainingCallback())

    learn_params["callback"] = CallbackList(callbacks)
    model.learn(**learn_params)
    model.save(model_out)
