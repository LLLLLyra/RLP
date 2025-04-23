from functools import partial
from typing import Callable
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


def cosine_annealing(
    initial_lr: float, final_lr: float, progress_remaining: float
) -> float:
    return final_lr + 0.5 * (initial_lr - final_lr) * (
        1 + np.cos(np.pi * (1 - progress_remaining))
    )


def linear_decay(
    initial_lr: float,
    final_lr: float,
    transition_fraction: float,
    progress_remaining: float,
) -> float:
    if progress_remaining > transition_fraction:
        return initial_lr
    return final_lr + (initial_lr - final_lr) * (
        progress_remaining / transition_fraction
    )


def constant(initial_lr: float, progress_remaining: float) -> float:
    return initial_lr


def register_schedule(schedule: str, **kwargs) -> Callable[[float], float]:
    if schedule == "cosine":
        return partial(cosine_annealing, kwargs["initial_lr"], kwargs["final_lr"])
    elif schedule == "linear":
        return partial(
            linear_decay,
            kwargs["initial_lr"],
            kwargs["final_lr"],
            kwargs["transition_fraction"],
        )
    elif schedule == "constant":
        return partial(constant, kwargs["initial_lr"])

    return partial(constant, kwargs["initial_lr"])


class MultiEnvEpisodeEndTrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MultiEnvEpisodeEndTrainingCallback, self).__init__(verbose)
        self.episode_end = False

    def _on_step(self) -> bool:
        if any(self.locals.get("dones")):
            self.episode_end = True
        return True

    def _on_rollout_end(self) -> None:
        if self.episode_end:
            self.model.train(gradient_steps=-1)
            self.episode_end = False


class SACTensorboardCallBack(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        replay_data = self.model.replay_buffer.sample(
            self.model.batch_size, env=self.model._vec_normalize_env
        )
        obs = replay_data.next_observations
        with th.no_grad():
            q_value_1 = th.cat(
                self.model.critic_target(obs, self.model.actor(obs)), dim=1
            )
            q_value_1, _ = th.min(q_value_1, dim=1, keepdim=False)
            q_value_2 = th.cat(
                self.model.critic_target(replay_data.observations, replay_data.actions),
                dim=1,
            )
            q_value_2, _ = th.min(q_value_2, dim=1, keepdim=False)

        self.logger.record("train/q1_value", q_value_1.mean().item())
        self.logger.record("train/q2_value", q_value_2.mean().item())
        self.logger.record("train/q1_std", q_value_1.std().item())
        self.logger.record("train/q2_std", q_value_2.std().item())

        # figure, ax = plt.subplots(1, 2)
        # ax[0].hist(q_value_1.cpu().numpy())
        # ax[1].hist(q_value_2.cpu().numpy())
        # self.logger.record("train/Q_hist", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        # plt.close()


class SaveBestModelCallback(BaseCallback):
    def __init__(self, model_name: str, check_freq: int = 10000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.model_name = model_name
        self.best_reward = -float("inf")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            mean_reward = safe_mean(
                [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            )
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                model_name = f"{self.model_name}_best0"
                self.model.save(model_name)
                if self.verbose > 0:
                    print(
                        f"[CheckPoint]: Best Model has been saved at {self.num_timesteps} step with reward {self.best_reward}."
                    )
        return True
