import os
import sys
from copy import deepcopy
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_schedule_fn

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.append(project_root)

from env.dp_vt_env import DPVTEnv
from env.env_visualiser import EnvVisualiser
from train.model_components import MixedFeaturesExtractor
from train.train_utils import EvalCallback, SACTensorboardCallBack, SaveBestModelCallback, register_schedule


def train(
    init_state: np.ndarray,
    config: Dict[str, Any],
    mode_name: str,
    device_id: int = 0,
    use_multi_env: bool = False,
    n_envs: int = 1,
    use_custom_log: bool = False,
    show_plot: bool = False,
) -> None:
    del use_multi_env, n_envs
    train_env = _build_env(init_state, config["dp_vt_config"], render=show_plot)
    eval_env = _build_env(init_state, config["dp_vt_config"], render=False)
    if th.cuda.is_available():
        th.cuda.set_device(device_id)
    model = SAC(env=train_env, **_build_sac_kwargs(config, device_id))
    learn_params = deepcopy(config["model_config"]["train_param"])
    callbacks = [SaveBestModelCallback(mode_name)]
    if use_custom_log:
        callbacks.append(SACTensorboardCallBack())
    callbacks.append(EvalCallback(eval_env, **_get_eval_config(config)))
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
    use_multi_env: bool = False,
    use_custom_log: bool = False,
) -> None:
    del use_multi_env
    if th.cuda.is_available():
        th.cuda.set_device(device_id)
    train_env = _build_env(init_state, config["dp_vt_config"], render=False)
    eval_env = _build_env(init_state, config["dp_vt_config"], render=False)
    model = SAC.load(model_name, env=train_env, device=_resolve_device(device_id))
    _apply_runtime_sac_config(model, config)

    learn_params = deepcopy(config["model_config"]["train_param"])
    callbacks = [SaveBestModelCallback(model_out)]
    if use_custom_log:
        callbacks.append(SACTensorboardCallBack())
    callbacks.append(EvalCallback(eval_env, **_get_eval_config(config)))
    learn_params["callback"] = CallbackList(callbacks)
    model.learn(**learn_params)
    model.save(model_out)


def _build_env(
    init_state: np.ndarray, dp_vt_config: Dict[str, Any], render: bool = False
) -> Monitor:
    env_kwargs = deepcopy(dp_vt_config)
    dt = env_kwargs["dt"]
    visualiser = EnvVisualiser(init_state, dt) if render else None
    env_kwargs["visualiser"] = visualiser
    env_kwargs["render_mode"] = "human" if render else None
    env_kwargs["init_dynamic_state"] = init_state
    return Monitor(DPVTEnv(**env_kwargs))


def _build_sac_kwargs(config: Dict[str, Any], device_id: int) -> Dict[str, Any]:
    sac_constructor_config = deepcopy(config["model_config"]["SAC_constructor"])
    policy_kwargs = sac_constructor_config.get("policy_kwargs", {})
    if "activation_fn" in policy_kwargs:
        policy_kwargs["activation_fn"] = eval(policy_kwargs["activation_fn"])
    policy_kwargs["features_extractor_class"] = MixedFeaturesExtractor
    sac_constructor_config["policy_kwargs"] = policy_kwargs
    sac_constructor_config["policy"] = "MultiInputPolicy"
    sac_constructor_config["buffer_size"] = int(sac_constructor_config["buffer_size"])
    sac_constructor_config["learning_rate"] = _get_learning_rate(config, sac_constructor_config)
    sac_constructor_config.setdefault("train_freq", 1)
    sac_constructor_config.setdefault("gradient_steps", 1)
    sac_constructor_config.setdefault("learning_starts", 5000)
    sac_constructor_config["device"] = _resolve_device(device_id)
    sac_constructor_config.pop("action_noise", None)
    return sac_constructor_config


def _apply_runtime_sac_config(model: SAC, config: Dict[str, Any]) -> None:
    sac_constructor_config = deepcopy(config["model_config"]["SAC_constructor"])
    model.ent_coef = sac_constructor_config["ent_coef"]
    model.tau = sac_constructor_config["tau"]
    model.batch_size = sac_constructor_config["batch_size"]
    model.learning_rate = _get_learning_rate(config, sac_constructor_config)
    model.buffer_size = int(sac_constructor_config["buffer_size"])
    train_freq = sac_constructor_config.get("train_freq", 1)
    if isinstance(train_freq, int):
        model.train_freq = TrainFreq(train_freq, TrainFrequencyUnit.STEP)
    elif isinstance(train_freq, (tuple, list)) and len(train_freq) == 2:
        unit = TrainFrequencyUnit(train_freq[1])
        model.train_freq = TrainFreq(int(train_freq[0]), unit)
    model.gradient_steps = int(sac_constructor_config.get("gradient_steps", 1))
    model.learning_starts = sac_constructor_config.get("learning_starts", 5000)


def _get_learning_rate(
    config: Dict[str, Any], sac_constructor_config: Dict[str, Any]
):
    lr_schedule = sac_constructor_config["learning_rate"]
    if "learning_schedule" in config["model_config"]:
        learning_schedule = config["model_config"]["learning_schedule"]
        lr_schedule = get_schedule_fn(register_schedule(**learning_schedule))
    return lr_schedule


def _resolve_device(device_id: int) -> str:
    return f"cuda:{device_id}" if th.cuda.is_available() else "cpu"


def _get_eval_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(
        config.get(
            "eval_config",
            {
                "eval_freq": 20000,
                "n_eval_episodes": 5,
                "deterministic": True,
                "verbose": 1,
            },
        )
    )
