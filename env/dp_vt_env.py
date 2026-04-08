from typing import Any, Dict, List, Optional, Tuple
import json
import os
from copy import deepcopy

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np

from data_generator.speed_limit_generator import SpeedLimitGenerator, SpeedLimit
from data_generator.st_polygon_generator import STGenerator, STBoundary, STPoint
from env.env_visualiser import EnvVisualiser


class DPVTEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        init_dynamic_state: np.ndarray,
        max_jerk: float,
        max_v: float,
        dt: float,
        max_s: float,
        max_a: float,
        min_a: float,
        max_path_s: float,
        max_t: float,
        max_abs_slope: float,
        max_min_s: float,
        max_max_s: float,
        visualiser: Optional[EnvVisualiser],
        max_st_num: int = 10,
        valid_max_st_num: int = 4,
        speed_limit_samples: int = 40,
        single_st_samples: int = 5,
        max_single_speed_limit_points: int = 10,
        max_speed_limit_intervals: int = 5,
        resolution: float = 0.1,
        reward_config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.init_dynamic_state = np.asarray(init_dynamic_state, dtype=np.float32)
        if self.init_dynamic_state.shape[0] != 3:
            raise ValueError(f"initial kinetics state error: {init_dynamic_state}")

        self.max_jerk = float(max_jerk)
        self.max_v = float(max_v)
        self.dt = float(dt)
        self.dt_squared = self.dt**2
        self.dt_cubic = self.dt**3
        self.max_s = float(max_s)
        self.max_a = float(max_a)
        self.min_a = float(min_a)
        self.max_t = float(max_t)
        self.resolution = float(resolution)

        self.speed_limit_generator = SpeedLimitGenerator(max_v, max_path_s, seed)
        self.st_generator = STGenerator(
            max_t, max_abs_slope, max_min_s, max_max_s, seed
        )

        self.max_st_num = max_st_num
        self.valid_max_st_num = valid_max_st_num
        self.speed_limit_samples = speed_limit_samples
        self.single_st_samples = single_st_samples
        self.max_single_speed_limit_points = max_single_speed_limit_points
        self.max_speed_limit_intervals = max_speed_limit_intervals

        self.action_dim = 1
        self.action_space = spaces.Box(
            low=-self.max_jerk,
            high=self.max_jerk,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.array_dim = 5 + speed_limit_samples * 2
        self.image_shape = (
            1,
            int(max_max_s / self.resolution),
            int(self.max_t / self.resolution),
        )

        idx = 5
        observation_high = np.zeros(self.array_dim, dtype=np.float32)
        observation_low = np.zeros(self.array_dim, dtype=np.float32)
        observation_high[:idx] = np.array([max_s, max_v, max_a, max_t, max_v], dtype=np.float32)
        observation_low[:idx] = np.array([0.0, 0.0, self.min_a, 0.0, 0.0], dtype=np.float32)
        observation_high[idx : (idx + self.speed_limit_samples)] = self.max_v
        observation_high[
            (idx + self.speed_limit_samples) : (idx + self.speed_limit_samples * 2)
        ] = max_path_s

        self.observation_space: spaces.Dict = spaces.Dict(
            {
                "st_image": spaces.Box(
                    low=0.0, high=1.0, shape=self.image_shape, dtype=np.float32
                ),
                "array": spaces.Box(
                    low=observation_low,
                    high=observation_high,
                    dtype=np.float32,
                ),
            }
        )

        self.reward_config = self._default_reward_config()
        if reward_config is not None:
            self.reward_config.update(reward_config)

        self.read_from_file = False
        self.speed_limit: SpeedLimit = SpeedLimit()
        self.st: List[STBoundary] = []

        self.visualiser = visualiser
        self.seed = seed
        self.t = 0.0
        self.max_ref_v = self.max_v
        self.state: Dict[str, np.ndarray] = {}

        self.early_return = False
        self.cross_hard_st = False
        self.cross_soft_st = False
        self.overspeed = False
        self.severe_overspeed = False
        self.reverse_speed = False
        self.out_of_bounds = False
        self.interaction = False
        self.exceed_speed = 0.0

        self.progress_reward = 0.0
        self.st_cost = 0.0
        self.speed_cost = 0.0
        self.acc_cost = 0.0
        self.jerk_cost = 0.0
        self.d_j_cost = 0.0
        self.terminal_cost = 0.0
        self.completion_bonus = 0.0

        self.s: List[float] = []
        self.v: List[float] = []
        self.a: List[float] = []
        self.j: List[float] = []

        self.render_mode = render_mode if render_mode in self.metadata["render_modes"] else None
        self.reset(self.seed, normalise=False)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        normalise: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        state = self._generate_static_environment()
        if self.visualiser is not None:
            self.visualiser.reset()
            self.visualiser.set_speed_limit(self.speed_limit)
            self.visualiser.set_st(self.st)

        self.t = 0.0
        self.early_return = False
        self.cross_hard_st = False
        self.cross_soft_st = False
        self.overspeed = False
        self.severe_overspeed = False
        self.reverse_speed = False
        self.out_of_bounds = False
        self.interaction = False
        self.exceed_speed = 0.0

        speed_limit_max = (
            self.speed_limit.speed_limit_points[:, 1].max()
            if self.speed_limit.speed_limit_points.size > 0
            else self.max_v
        )
        init_v = min(
            self.max_v,
            np.random.uniform(0.0, speed_limit_max) + np.random.uniform(0.0, 0.5),
        )
        acc_sign = np.random.binomial(1, 1 - self.max_a / max(-self.min_a, 1e-3))
        if not acc_sign:
            init_a = np.random.uniform(max(self.min_a, -init_v / max(self.dt * 10, 1e-3)), 0.0)
        else:
            init_a = np.random.uniform(0.0, self.max_a)

        self.init_dynamic_state = np.array([0.0, init_v, init_a], dtype=np.float32)
        array = np.concatenate(
            (
                self.init_dynamic_state,
                np.array([0.0, self.max_ref_v], dtype=np.float32),
                state["array"],
            )
        ).astype(np.float32)
        if array.shape[0] != self.array_dim:
            raise ValueError(
                f"inequal array dim: array_dim = {self.array_dim}, array_shape = {array.shape[0]}"
            )

        self.s = [0.0]
        self.v = [float(init_v)]
        self.a = [float(init_a)]
        self.j = []
        self._reset_reward_tracking()

        state["array"] = array
        self.state = state
        info = {"events": self._event_info(), "reward_terms": self._reward_info()}
        if normalise:
            return self._get_normalise_state(), info
        return self._get_obs(), info

    def step(
        self, action: float | np.floating | np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        s_prev, v_prev, a_prev = map(float, self.state["array"][:3])
        u = self._parse_action(action)

        s = s_prev + v_prev * self.dt + 0.5 * a_prev * self.dt_squared + (u * self.dt_cubic) / 6.0
        v = v_prev + a_prev * self.dt + 0.5 * u * self.dt_squared
        if v_prev * v < 0:
            v = 0.0
        v = np.clip(v, -1.0, self.max_v + 2.0)
        a = a_prev + u * self.dt
        if a_prev * a < 0:
            a = 0.0
        a = np.clip(a, self.min_a - 1.0, self.max_a + 1.0)

        self.t = round(self.t + self.dt, 10)
        self.state["array"][:4] = np.array([s, v, a, self.t], dtype=np.float32)
        self.s.append(float(s))
        self.v.append(float(v))
        self.a.append(float(a))
        self.j.append(float(u))

        reward = self._reward_func(s_prev, float(s), float(v), float(a), float(u), self.t)
        terminated = self.t >= self.max_t and not self.early_return
        truncated = self.early_return

        info = {
            "events": self._event_info(),
            "reward_terms": self._reward_info(),
            "speed_limit": float(self.speed_limit.get_speed_limits_by_s(max(float(s), 0.0))),
        }

        self.render(self.state["array"][:3], u, reward)
        return self._get_normalise_state(), reward, terminated, truncated, info

    def _default_reward_config(self) -> Dict[str, Any]:
        return {
            "progress_weight": 0.35,
            "speed_tracking_weight": 0.25,
            "overspeed_weight": 0.8,
            "acc_weight": 0.08,
            "jerk_weight": 0.03,
            "djerk_weight": 0.01,
            "hard_margin_weight": 1.5,
            "soft_margin_weight": 0.6,
            "yield_weight": 0.5,
            "completion_bonus": 1.0,
            "collision_penalty": 10.0,
            "severe_overspeed_penalty": 5.0,
            "reverse_penalty": 5.0,
            "out_of_bounds_penalty": 5.0,
            "overspeed_tolerance": 0.2,
            "severe_overspeed": 1.5,
            "reverse_speed_threshold": -0.2,
            "hard_penetration_tolerance": 0.1,
            "safe_gap_base": 1.0,
            "safe_gap_time": 0.3,
            "safe_gap_back": 0.5,
            "acc_comfort": 2.0,
            "jerk_comfort": 4.0,
            "djerk_comfort": 20.0,
            "terminal_speed_weight": 0.1,
            "terminal_acc_weight": 0.05,
        }

    def _parse_action(self, action: float | np.floating | np.ndarray) -> float:
        if isinstance(action, np.ndarray):
            if action.shape[0] != 1:
                raise ValueError(f"invalid action shape: {action.shape}")
            action_value = action[0] if len(action.shape) == 1 else action[0, 0]
        else:
            action_value = action
        return float(np.clip(float(action_value), -self.max_jerk, self.max_jerk))

    def _reward_func(
        self, s_prev: float, s: float, v: float, a: float, j: float, t: float
    ) -> float:
        self._reset_reward_tracking()

        progress_reward = self.reward_config["progress_weight"] * np.clip(
            (s - s_prev) / max(self.max_v * self.dt, 1e-3), -1.0, 1.5
        )
        speed_cost = self._speed_cost(s, v)
        st_cost = self._st_cost(s, v, t)
        acc_cost = self._acc_cost(a)
        jerk_cost = self._jerk_cost(j)
        djerk_cost = self._d_jerk_cost()
        terminal_penalty = self._terminal_penalty(s, v)
        terminal_bonus = self._completion_bonus(s, v, a, t)

        self.progress_reward = float(progress_reward)
        self.st_cost = float(st_cost)
        self.speed_cost = float(speed_cost)
        self.acc_cost = float(acc_cost)
        self.jerk_cost = float(jerk_cost)
        self.d_j_cost = float(djerk_cost)
        self.terminal_cost = float(terminal_penalty)
        self.completion_bonus = float(terminal_bonus)

        reward = (
            progress_reward
            - st_cost
            - speed_cost
            - acc_cost
            - jerk_cost
            - djerk_cost
            - terminal_penalty
            + terminal_bonus
        )
        return float(reward)

    def _st_cost(self, s: float, v: float, t: float) -> float:
        cost = 0.0
        self.cross_hard_st = False
        self.cross_soft_st = False
        self.interaction = False

        for st in self.st:
            if st.is_empty() or t < st.min_t or t > st.max_t:
                continue
            self.interaction = True
            s_lower, s_upper = st.get_boundary_s_range(t)
            if s_lower is None or s_upper is None:
                continue

            if st.soft:
                cost += self._soft_st_cost(st, s, v, t, s_lower, s_upper)
            else:
                cost += self._hard_st_cost(st, s, v, t, s_lower, s_upper)

        return float(cost)

    def _soft_st_cost(
        self, st: STBoundary, s: float, v: float, t: float, s_lower: float, s_upper: float
    ) -> float:
        safe_front = self.get_yield_distance(st, v, t)
        safe_back = self.reward_config["safe_gap_back"]
        yield_weight = self.reward_config["yield_weight"]
        margin_weight = self.reward_config["soft_margin_weight"]

        if s < s_lower:
            gap = s_lower - s
            return yield_weight * max(0.0, safe_front - gap) ** 2

        if s > s_upper:
            gap = s - s_upper
            return 0.5 * margin_weight * max(0.0, safe_back - gap) ** 2

        self.cross_soft_st = True
        penetration = min(s - s_lower, s_upper - s)
        return margin_weight * (1.0 + penetration) ** 2

    def _hard_st_cost(
        self, st: STBoundary, s: float, v: float, t: float, s_lower: float, s_upper: float
    ) -> float:
        safe_front = self.get_yield_distance(st, v, t)
        safe_back = self.reward_config["safe_gap_back"]
        margin_weight = self.reward_config["hard_margin_weight"]

        if s < s_lower:
            gap = s_lower - s
            return margin_weight * max(0.0, safe_front - gap) ** 2

        if s > s_upper:
            gap = s - s_upper
            return 0.5 * margin_weight * max(0.0, safe_back - gap) ** 2

        penetration = min(s - s_lower, s_upper - s)
        self.cross_hard_st = penetration > self.reward_config["hard_penetration_tolerance"]
        return margin_weight * (1.0 + penetration) ** 2

    def get_yield_distance(self, st: STBoundary, v: float, t: float) -> float:
        ds_lower, _ = st.get_boundary_slopes(t)
        safe_gap = self.reward_config["safe_gap_base"] + self.reward_config["safe_gap_time"] * max(v, 0.0)
        if ds_lower is not None and ds_lower > 0.0:
            safe_gap += 0.1 * ds_lower
        return float(max(safe_gap, self.reward_config["safe_gap_base"]))

    def _speed_cost(self, s: float, v: float) -> float:
        current_speed_limit = float(self.speed_limit.get_speed_limits_by_s(max(s, 0.0)))
        v_ref = min(current_speed_limit, self.max_ref_v)
        tracking_error = v - v_ref
        overspeed_margin = max(0.0, v - current_speed_limit - self.reward_config["overspeed_tolerance"])

        self.exceed_speed = float(v - current_speed_limit)
        self.overspeed = overspeed_margin > 0.0
        self.severe_overspeed = (
            v - current_speed_limit > self.reward_config["severe_overspeed"]
        )
        tracking_cost = self.reward_config["speed_tracking_weight"] * (
            tracking_error / max(self.max_v, 1e-3)
        ) ** 2
        overspeed_cost = self.reward_config["overspeed_weight"] * overspeed_margin**2
        return float(tracking_cost + overspeed_cost)

    def _acc_cost(self, a: float) -> float:
        comfort_scale = max(self.reward_config["acc_comfort"], 1e-3)
        return float(self.reward_config["acc_weight"] * (a / comfort_scale) ** 2)

    def _jerk_cost(self, j: float) -> float:
        comfort_scale = max(self.reward_config["jerk_comfort"], 1e-3)
        return float(self.reward_config["jerk_weight"] * (j / comfort_scale) ** 2)

    def _d_jerk_cost(self) -> float:
        if len(self.j) <= 1:
            return 0.0
        d_jerk = (self.j[-1] - self.j[-2]) / max(self.dt, 1e-3)
        comfort_scale = max(self.reward_config["djerk_comfort"], 1e-3)
        return float(self.reward_config["djerk_weight"] * (d_jerk / comfort_scale) ** 2)

    def _terminal_penalty(self, s: float, v: float) -> float:
        self.reverse_speed = v < self.reward_config["reverse_speed_threshold"]
        self.out_of_bounds = bool(
            np.isnan(s)
            or np.isnan(v)
            or s < -1.0
            or s > self.max_s + 5.0
            or abs(v) > self.max_v + 5.0
        )
        self.early_return = bool(
            self.cross_hard_st
            or self.severe_overspeed
            or self.reverse_speed
            or self.out_of_bounds
        )

        penalty = 0.0
        if self.cross_hard_st:
            penalty += self.reward_config["collision_penalty"]
        if self.severe_overspeed:
            penalty += self.reward_config["severe_overspeed_penalty"]
        if self.reverse_speed:
            penalty += self.reward_config["reverse_penalty"]
        if self.out_of_bounds:
            penalty += self.reward_config["out_of_bounds_penalty"]
        return float(penalty)

    def _completion_bonus(self, s: float, v: float, a: float, t: float) -> float:
        if t < self.max_t or self.early_return:
            return 0.0

        speed_limit = float(self.speed_limit.get_speed_limits_by_s(max(s, 0.0)))
        v_ref = min(speed_limit, self.max_ref_v)
        terminal_state_cost = (
            self.reward_config["terminal_speed_weight"] * (v - v_ref) ** 2
            + self.reward_config["terminal_acc_weight"] * a**2
        )
        return float(self.reward_config["completion_bonus"] - terminal_state_cost)

    def _reset_reward_tracking(self) -> None:
        self.progress_reward = 0.0
        self.st_cost = 0.0
        self.speed_cost = 0.0
        self.acc_cost = 0.0
        self.jerk_cost = 0.0
        self.d_j_cost = 0.0
        self.terminal_cost = 0.0
        self.completion_bonus = 0.0

    def _reward_info(self) -> Dict[str, float]:
        return {
            "progress": float(self.progress_reward),
            "st_cost": float(self.st_cost),
            "speed_cost": float(self.speed_cost),
            "acc_cost": float(self.acc_cost),
            "jerk_cost": float(self.jerk_cost),
            "djerk_cost": float(self.d_j_cost),
            "terminal_cost": float(self.terminal_cost),
            "completion_bonus": float(self.completion_bonus),
        }

    def _event_info(self) -> Dict[str, bool]:
        return {
            "cross_hard_st": bool(self.cross_hard_st),
            "cross_soft_st": bool(self.cross_soft_st),
            "overspeed": bool(self.overspeed),
            "severe_overspeed": bool(self.severe_overspeed),
            "reverse_speed": bool(self.reverse_speed),
            "out_of_bounds": bool(self.out_of_bounds),
        }

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return self.state

    def _get_normalise_state(self) -> Dict[str, np.ndarray]:
        state = deepcopy(self._get_obs())
        array = state["array"].astype(np.float32)
        array = self.min_max_wrapper(
            array,
            self.observation_space["array"].low - 1.0,
            self.observation_space["array"].high + 2.0,
        ).astype(np.float32)
        state["array"] = np.clip(array, 0.0, 1.0)
        state["st_image"] = state["st_image"].astype(np.float32)
        return state

    def _generate_static_environment(self) -> Dict[str, np.ndarray]:
        st_image = self._generate_st()
        speed_limit = self._generate_speed_limit()
        return dict(st_image=st_image, array=speed_limit)

    def _generate_speed_limit(self) -> np.ndarray:
        self.max_ref_v = self.speed_limit_generator.max_speed
        speed_limit = self.speed_limit_generator.generate(
            self.max_single_speed_limit_points,
            self.max_speed_limit_intervals,
            self.read_from_file,
        )
        speed_limit_obv = np.zeros(self.speed_limit_samples * 2, dtype=np.float32)
        s = 0.0
        for i in range(self.speed_limit_samples):
            v = speed_limit.get_speed_limits_by_s(s)
            speed_limit_obv[i] = v
            speed_limit_obv[i + self.speed_limit_samples] = s
            s += 0.5
        self.speed_limit = speed_limit
        return speed_limit_obv

    def _generate_st(self) -> np.ndarray:
        valid_num_hard_st = self.valid_max_st_num // 2
        valid_num_soft_st = self.valid_max_st_num - valid_num_hard_st
        st_polygons = self.st_generator.generate(
            valid_num_hard_st, valid_num_soft_st, self.read_from_file
        )
        image = self.st_generator.encode_st(st_polygons)
        st_image = np.zeros_like(image, dtype=np.float32)
        st_image[(image >= 150) & (image < 230)] = 0.5
        st_image[image < 150] = 1.0
        self.st = st_polygons
        return np.expand_dims(st_image, axis=0).astype(np.float32)

    def min_max_wrapper(
        self,
        state: float | np.ndarray,
        low: float | np.ndarray,
        high: float | np.ndarray,
    ) -> np.ndarray:
        return (state - low) / (high - low)

    def min_max_reverse_wrapper(
        self,
        state: float | np.ndarray,
        low: float | np.ndarray,
        high: float | np.ndarray,
    ) -> np.ndarray:
        return state * (high - low) + low

    def render(self, s: np.ndarray, a: float, r: float) -> None:
        if self.render_mode:
            assert self.visualiser is not None
            self.visualiser.render(s, a, r)
            plt.pause(0.01)
        elif self.visualiser is not None:
            self.visualiser.reset()

    def read_samples_from_file(self, file_path: str) -> None:
        files = os.listdir(file_path)
        speed_buffer: List[SpeedLimit] = []
        st_buffer: List[List[STBoundary]] = []
        for filename in files:
            with open(os.path.join(file_path, filename), encoding="utf-8") as file:
                data = json.load(file)[0]
                for details in data["detail"]:
                    detail = json.loads(details["dp_data"])
                    speed_info = detail["speed_limits"]
                    st_info = detail["st_polygons"]
                    speed_buffer.append(SpeedLimit(np.array(speed_info)))
                    st_boundaries = [
                        STBoundary(
                            list(map(STPoint, st["lower_points"])),
                            list(map(STPoint, st["upper_points"])),
                            bool(st["soft"]),
                        )
                        for st in st_info
                    ]
                    st_buffer.append(st_boundaries)
        self.speed_limit_generator.load_from_local(speed_buffer)
        self.st_generator.load_from_local(st_buffer)
        self.read_from_file = bool(speed_buffer) and bool(st_buffer)


if __name__ == "__main__":
    init_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    vis = EnvVisualiser(init_state, 0.1)
    params = dict(
        max_jerk=10,
        max_v=15,
        dt=0.1,
        max_s=60,
        max_a=0.5,
        min_a=-5,
        max_path_s=20,
        max_t=5,
        max_abs_slope=20,
        max_min_s=10,
        max_max_s=30,
    )
    seed = 11
    env = DPVTEnv(init_state, **params, visualiser=vis, render_mode="human", seed=seed)
    file_path = "local_data"
    env.read_samples_from_file(file_path)
    np.random.seed(seed)
    terminated = False
    truncated = False
    _, _ = env.reset(seed)
    vis.start_animation()
    while not terminated and not truncated:
        action = np.random.uniform(-params["max_jerk"], params["max_jerk"])
        _, reward, terminated, truncated, info = env.step(action)
        print(f"reward={reward:.3f}, events={info['events']}")
    plt.show()
