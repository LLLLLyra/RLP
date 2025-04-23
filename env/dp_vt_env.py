from typing import *
import json
import os
from copy import deepcopy
import torch as th
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

from data_generator.speed_limit_generator import SpeedLimitGenerator, SpeedLimit
from data_generator.st_polygon_generator import STGenerator, STBoundary, STPoint
from common_utils.math_utils import Vec2d
from env.env_visualiser import EnvVisualiser

KMAX_PEN = 1e8


class DPVTEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        init_dynamic_state: np.ndarray[float],
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
        visualiser: EnvVisualiser,
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
        super(DPVTEnv, self).__init__()
        self.init_dynamic_state = init_dynamic_state
        if self.init_dynamic_state.shape[0] != 3:
            raise ValueError(f"initial kinetics state error: {init_dynamic_state}")

        self.max_jerk = max_jerk
        self.max_v = max_v
        self.dt = dt
        self.dt_squared = self.dt**2
        self.dt_cubic = self.dt**3
        self.max_s = max_s
        self.max_a = max_a
        self.min_a = min_a
        self.max_t = max_t

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
            int(max_max_s / resolution),
            int(self.max_t / resolution),
        )

        idx = 5
        observation_high = np.zeros(self.array_dim)
        observation_low = np.zeros(self.array_dim)
        observation_high[:idx] = np.array([max_s, max_v, max_a, max_t, max_v])
        observation_low[:idx] = np.array([0.0, 0.0, self.min_a, 0.0, 0.0])
        observation_high[idx : (idx + self.speed_limit_samples)] = self.max_v
        observation_high[
            (idx + self.speed_limit_samples) : (idx + self.speed_limit_samples * 2)
        ] = max_path_s

        self.observation_space: spaces.Dict[str, spaces.Box] = spaces.Dict(
            {
                "st_image": spaces.Box(low=0, high=255, shape=self.image_shape),
                "array": spaces.Box(low=observation_low, high=observation_high),
            }
        )

        self.early_return = False
        self.cross_hard_st = False
        self.cross_soft_st = False
        self.overspeed = False
        self.prev_cross_hard_st = False
        self.prev_cross_soft_st = False
        self.prev_overspeed = False
        self.prev_neg_speed = False
        self.first_cross_hard_st = self.max_t
        self.first_cross_soft_st = self.max_t
        self.first_overspeed = self.max_t
        self.first_neg_speed = self.max_t
        self.yield_cost = 0
        self.interaction = False
        self.exceed_speed = 0
        self.s = []
        self.v = []
        self.a = []
        self.j = []

        self.read_from_file = False

        self.speed_limit: SpeedLimit = SpeedLimit()
        self.st: List[STBoundary] = []

        self.visualiser = visualiser
        self.seed = seed
        self.t = 0
        self.max_ref_v = self.max_v
        self.reset(self.seed, normalise=False)

        self.st_cost = -1
        self.speed_cost = -1
        self.acc_cost = -1
        self.jerk_cost = -1
        self.terminal_cost = -1
        self.d_j_cost = -1

        if reward_config is None:
            self.reward_config = self._default_reward_config()
        else:
            self.reward_config = reward_config

        if render_mode and render_mode in self.metadata["render_modes"]:
            self.render_mode = render_mode
        else:
            self.render_mode = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
        normalise: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        state = self._generate_static_environment()
        if self.visualiser is not None:
            self.visualiser.reset()
            self.visualiser.set_speed_limit(self.speed_limit)
            self.visualiser.set_st(self.st)
        self.t = 0
        self.early_return = False
        self.cross_hard_st = False
        self.cross_soft_st = False
        self.overspeed = False
        self.prev_cross_hard_st = False
        self.prev_cross_soft_st = False
        self.prev_overspeed = False
        self.prev_neg_speed = False
        self.first_cross_hard_st = self.max_t
        self.first_cross_soft_st = self.max_t
        self.first_overspeed = self.max_t
        self.first_neg_speed = self.max_t
        self.yield_cost = 0
        self.interaction = False
        array = state["array"]
        init_v = np.random.uniform(
            0, self.speed_limit.speed_limit_points[:, 1].max()
        ) + np.random.uniform(0, 1)
        acc = np.random.binomial(1, 1 - self.max_a / -self.min_a)
        if not acc:
            init_a = np.random.uniform(max(self.min_a, -init_v / self.dt / 10), 0)
        else:
            init_a = np.random.uniform(0, self.max_a)
        self.init_dynamic_state = np.array([0, init_v, init_a])
        array = np.concatenate(
            (self.init_dynamic_state, np.array([0.0, self.max_ref_v]), array)
        )
        self.s = [0]
        self.v = [init_v]
        self.a = [init_a]
        self.j = []
        if array.shape[0] != self.array_dim:
            raise ValueError(
                f"inequal array dim: array_dim = {self.array_dim}, array_shape = {array.shape[0]}"
            )
        state["array"] = array
        self.state = state
        if normalise:
            return self._get_normalise_state(), {}
        return self._get_obs(), {}

    def step(
        self, action: float | np.float64 | np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        s, v, a = self.state["array"][:3]
        v_prev, a_prev = self.state["array"][1:3]
        u = None
        if isinstance(action, float):
            u = action
        elif isinstance(action, np.ndarray):
            assert action.shape[0] == 1
            u = action[0] if len(action.shape) == 1 else action[0, 0]
        else:
            raise TypeError(f"action should be either float or ndarray type: {type(u)}")
        u = np.clip(u, -self.max_jerk, self.max_jerk)

        s += v * self.dt + 0.5 * a * self.dt_squared + 1.0 / 6.0 * u * self.dt_cubic
        v += a * self.dt + 0.5 * u * self.dt_squared
        if v_prev * v < 0:
            v = 0
        v = np.clip(v, -1, self.max_v + 2)
        a += u * self.dt
        if a * a_prev < 0:
            a = 0
        a = np.clip(a, self.min_a - 2, self.max_a + 2)
        self.t += self.dt
        self.state["array"][:4] = np.array([s, v, a, self.t])
        self.s.append(s)
        self.v.append(v)
        self.a.append(a)
        self.j.append(u)
        terminated = self.t >= self.max_t
        truncated = v <= -1 and False
        reward = self._reward_func(s, v, a, u, self.t)
        self.early_return = self.cross_hard_st or self.overspeed
        # truncated = truncated or self.early_return

        self.render(self.state["array"][:3], u, reward)

        return self._get_normalise_state(), reward, terminated, truncated, {}

    def _default_reward_config(self) -> Dict[str, Any]:
        config = {}
        config["exceed_speed"] = 0.2
        config["beyond_speed_limit_cost"] = 1e5
        config["speed_coef"] = 1.7
        config["default_speed_cost"] = 1e5

        config["accel_coef"] = 2
        config["accel_penalty"] = 3000
        config["decel_penalty"] = 3000

        config["jerk_coef"] = 4
        config["positive_jerk_coeff"] = 5e3
        config["negative_jerk_coeff"] = 5e3

        config["default_cost"] = 1e6

        config["speed_lon_decision_horizon"] = 35

        config["default_yield_cost"] = 1e5
        config["obstacle_weight"] = 1
        config["yield_exp_coef"] = 2.5
        config["upper_safe_dist"] = 3
        config["default_obstacle_cost"] = 1e5

        config["yield_min_dis"] = 0.5

        config["extra_cost_weights"] = 5e5
        config["first_t_thres"] = 2
        config["front_edge_to_center"] = 3.4

        config["max_exceed_speed"] = 1.0

        return config

    def _reward_func(self, s: float, v: float, a: float, j: float, t: float) -> float:
        # TODO: rewards
        st_cost = self._st_cost(s, v, t)
        speed_cost = self._speed_cost(s, v)
        acc_cost = self._acc_cost(a, v, t)
        jerk_cost = self._jerk_cost(j, v, t)
        dj_cost = self._d_jerk_cost()
        terminate_cost = self._terminal_cost(t)

        self.prev_cross_hard_st = self.prev_cross_hard_st or self.cross_hard_st
        self.prev_cross_soft_st = self.prev_cross_soft_st or self.cross_soft_st
        self.prev_overspeed = self.prev_overspeed or self.overspeed
        self.prev_neg_speed = self.prev_neg_speed or v < 0

        if self.prev_cross_hard_st:
            self.first_cross_hard_st = min(self.first_cross_hard_st, t)

        if self.prev_cross_soft_st:
            self.first_cross_soft_st = min(self.first_cross_soft_st, t)

        if self.prev_overspeed:
            self.first_overspeed = min(self.first_overspeed, t)

        if self.prev_neg_speed:
            self.first_neg_speed = min(self.first_neg_speed, t)

        if self.prev_cross_hard_st:
            st_cost += (
                max(self.max_t - self.first_cross_hard_st, 0)
                + self.cross_hard_st * max(0, t - self.first_cross_hard_st)
            ) * KMAX_PEN
        if self.prev_cross_soft_st:
            st_cost += (
                max(self.max_t - self.first_cross_soft_st, 0)
                + self.cross_hard_st * max(0, t - self.first_cross_soft_st)
            ) * KMAX_PEN
        if self.prev_overspeed:
            speed_cost += (
                max(self.max_t - self.first_overspeed, 0)
                + self.overspeed * max(0, t - self.first_overspeed)
            ) * KMAX_PEN
        if self.prev_neg_speed:
            speed_cost += KMAX_PEN * (
                max(self.max_t - self.first_neg_speed, 0)
                + (v < 0) * max(0, t - self.first_neg_speed)
            )
        cost = st_cost + speed_cost + acc_cost + jerk_cost + terminate_cost + dj_cost
        self.st_cost = st_cost / KMAX_PEN
        self.speed_cost = speed_cost / KMAX_PEN
        self.acc_cost = acc_cost / KMAX_PEN
        self.jerk_cost = jerk_cost / KMAX_PEN
        self.d_j_cost = dj_cost / KMAX_PEN
        self.terminal_cost = terminate_cost / KMAX_PEN
        return -cost / KMAX_PEN

    def _st_cost(self, s: float, v: float, t: float) -> float:
        self.cross_hard_st = False
        self.cross_soft_st = False
        self.yield_cost = 0
        if v < 0.0 or s < 0.0:
            return 0.0
        cost = 0.0
        soft_cost = 0.0
        hard_cost = 0.0
        extra_cost = 0.0
        min_boundary_s = self.reward_config["speed_lon_decision_horizon"]
        self.interaction = False
        for st in self.st:
            if st.is_empty():
                continue

            if t < st.min_t or t > st.max_t:
                continue

            self.interaction = True

            if st.soft:
                soft_cost += self._soft_default_cost(st, s, v, t)
            else:
                c, min_boundary_s = self._hard_st_cost(st, s, v, t, min_boundary_s)
                hard_cost += c

            extra_cost = max(extra_cost, self._extra_cost(st, s, v, t))
        self.yield_cost += extra_cost
        cost = soft_cost + hard_cost + extra_cost
        return cost

    def _soft_default_cost(self, st: STBoundary, s: float, v: float, t: float) -> float:
        cost = 0.0
        s_lower, s_upper = st.get_boundary_s_range(t)
        if s_lower is None or s_upper is None:
            return cost

        l = self.get_yield_distance(st, v, t)
        if s < s_lower:
            if s + l < s_lower:
                pass
            else:
                c = (l - s_lower + s) ** 2 * self.reward_config["default_cost"]
                cost += c
                self.yield_cost += c
        elif s > s_upper:
            cost = (
                self.reward_config["upper_safe_dist"] + s_upper - s
            ) ** 2 * self.reward_config["default_cost"]
        else:
            upper_point = max(st.upper_points, key=lambda x: (x[1], -x[0]))
            can_overtake = (
                self.init_dynamic_state[1] * upper_point.t()
                + 0.5 * self.max_a * upper_point.t()
                >= upper_point.s()
            )
            ds_lower = s - s_lower
            ds_upper = s_upper - s
            self.cross_soft_st = True
            if not can_overtake or ds_lower < ds_upper:
                c = self.reward_config["default_cost"] * min(
                    5e2, np.exp(2.0 * ds_lower)
                )
                cost += c
                self.yield_cost += c

            else:
                cost += self.reward_config["default_cost"] * min(
                    5e2, np.exp(2.0 * ds_upper)
                )

        return cost

    def _hard_st_cost(
        self, st: STBoundary, s: float, v: float, t: float, min_boundary_s: float
    ) -> Tuple[float]:
        cost = 0
        s_lower, s_upper = st.get_boundary_s_range(t)
        if s_lower is None or s_upper is None:
            return cost, min_boundary_s

        l = self.get_yield_distance(st, v, t)
        if s < s_lower:
            if s + l < s_lower:
                pass
            elif s_lower >= min_boundary_s:
                pass
            else:
                min_boundary_s = s_lower
                c = (
                    self.reward_config["obstacle_weight"]
                    * self.reward_config["default_obstacle_cost"]
                    * (l - s_lower + s) ** 2
                )
                cost += c
                self.yield_cost += c
        elif s > s_upper:
            if s > s_upper + self.reward_config["upper_safe_dist"]:
                pass
            else:
                cost += (
                    self.reward_config["obstacle_weight"]
                    * self.reward_config["default_obstacle_cost"]
                    * (self.reward_config["upper_safe_dist"] + s_upper - s) ** 2
                )
        else:
            upper_point = max(st.upper_points, key=lambda x: (x[1], -x[0]))
            can_overtake = (
                self.init_dynamic_state[1] * upper_point.t()
                + 0.5 * self.max_a * upper_point.t()
                >= upper_point.s()
            )
            ds_lower = s - s_lower
            ds_upper = s_upper - s
            penetration = (
                ds_lower if not can_overtake or ds_lower < ds_upper else ds_upper
            )
            if 0.5 < abs(penetration) <= 1:
                cost += KMAX_PEN * np.exp(0.8 * abs(penetration))
            elif abs(penetration) > 1:
                cost += KMAX_PEN * min(20.0, np.exp(0.8 * abs(penetration)))
                self.cross_hard_st = True
            else:
                cost += self.reward_config["default_obstacle_cost"] * np.exp(
                    0.8 * abs(penetration)
                )

        return cost, min_boundary_s

    def _extra_cost(self, st: STBoundary, s: float, v: float, t: float) -> float:
        cost = 0
        if t > self.reward_config["first_t_thres"]:
            cost += 0
        else:
            safe_distance = self.get_yield_distance(st, v, t)
            if st.min_t < 0.5:
                safe_distance = max(
                    safe_distance, self.reward_config["front_edge_to_center"]
                )

            lower_s, upper_s = st.get_boundary_s_range(t)
            if lower_s is not None and upper_s is not None:
                if s >= upper_s:
                    cost += 0
                else:
                    expected_s = lower_s - safe_distance
                    if s > expected_s:
                        cost += (
                            min(1000, np.exp(0.5 * abs(s - expected_s)))
                            * self.reward_config["extra_cost_weights"]
                        )

        return cost

    def get_yield_distance(self, st: STBoundary, v: float, t: float) -> float:
        ds_lower, ds_upper = st.get_boundary_slopes(t)
        if ds_lower is None or ds_upper is None:
            return 0
        if ds_lower <= 1e-2:
            return 0.0
        safe_dist = 0
        response_t = 0.5
        max_dec = 1
        max_acc = 0.5
        sim_v = v + max_acc * response_t
        safe_dist = (
            v * response_t
            + 0.5 * max_acc * response_t**2
            + sim_v**2 / 2 / max_dec
            - ds_lower**2 / 2 / max_dec
        )
        safe_dist = max(safe_dist, self.reward_config["yield_min_dis"])
        return safe_dist

    def _speed_cost(self, s: float, v: float) -> float:
        self.overspeed = False
        curr_speed_limit = self.speed_limit.get_speed_limits_by_s(s)
        cost = 0
        v_ref = min(curr_speed_limit, self.max_ref_v)
        d_v = v - v_ref
        self.exceed_speed = d_v
        if d_v > self.reward_config["max_exceed_speed"]:
            cost += KMAX_PEN * min(10.0, np.exp(0.5 * d_v))
            self.overspeed = True
        if abs(d_v) <= 0.0:
            return 0.0
        cost += (
            self.reward_config["speed_coef"]
            * self.reward_config["default_speed_cost"]
            * min(1000, np.exp(10.0 * abs(d_v)))
        ) * self.speed_cost_factor()
        if v < 0:
            cost += KMAX_PEN * np.exp(-2.0 * v)
        return cost

    def _acc_cost(self, a: float, v: float, t: float) -> float:
        cost = 0
        acc_pen = self.reward_config["accel_coef"] * self.reward_config["accel_penalty"]
        dec_pen = self.reward_config["accel_coef"] * self.reward_config["decel_penalty"]

        if a > 0.0:
            cost = acc_pen * min(3e4, np.exp(5.0 * a))
        elif a < 0.0:
            cost = dec_pen * min(3e4, np.exp(-5.0 * a))
        cost *= self.smooth_cost_factor()
        if v < 0 and a >= 0:
            cost *= 0.5
        if a > self.max_a:
            cost += min(2, np.exp(a - self.max_a)) * KMAX_PEN
        elif a < self.min_a:
            cost += min(2, np.exp(self.min_a - a)) * KMAX_PEN

        if len(self.a) >= 2:
            cost += (self.a[-1] * self.a[-2] < 0) * KMAX_PEN

        return cost

    def _jerk_cost(self, j: float, v: float, t: float) -> float:
        cost = 0
        j_pos_pen = (
            self.reward_config["jerk_coef"] * self.reward_config["positive_jerk_coeff"]
        )
        j_neg_pen = (
            self.reward_config["jerk_coef"] * self.reward_config["negative_jerk_coeff"]
        )

        if j > 0.0:
            cost = j_pos_pen * min(1e4, np.exp(2.0 * j))
        else:
            cost = j_neg_pen * min(1e4, np.exp(-2.0 * j))

        if v < 0 and j >= 0:
            cost *= 0.5

        if len(self.j) >= 2:
            cost += (self.j[-1] * self.j[-2] < 0) * KMAX_PEN

        cost *= self.smooth_cost_factor()
        cost *= self.speed_cost_factor()

        return cost

    def _d_jerk_cost(self) -> float:
        if len(self.j) <= 1:
            return 0.0

        d_j = (self.j[-1] - self.j[-2]) / self.dt
        cost = KMAX_PEN * min(1, 0.005 * abs(d_j))
        cost *= self.smooth_cost_factor()
        cost *= self.speed_cost_factor()
        return cost

    def _terminal_cost(self, current_t) -> float:
        if current_t < self.max_t:
            return 0.0
        cross_begin = [self.max_t, 0]
        cross_end = [0, 0]
        find = False
        for t, s in enumerate(self.s):
            if find:
                break
            t = t * self.dt
            for st in self.st:
                if st.is_point_in(Vec2d([t, s])):
                    cross_begin = [t, s]
                    find = True
                    break
        find = False
        for t, s in enumerate(self.s[::-1]):
            if find:
                break
            t = self.max_t - t * self.dt
            for st in self.st:
                if st.is_point_in(Vec2d([t, s])):
                    cross_end = [t, s]
                    find = True
                    break

        if cross_begin[0] == self.max_t or cross_end[0] == 0:
            return 0.0
        if cross_begin[0] >= cross_end[0]:
            return 0.0

        avg_v = (cross_end[1] - cross_begin[1]) / (cross_end[0] - cross_begin[0])
        if avg_v != 0.0:
            return KMAX_PEN * min(10, np.exp(2.0 * abs(avg_v)))
        return 0.0

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return self.state

    def _get_normalise_state(self) -> np.ndarray:
        state = deepcopy(self._get_obs())
        array = state["array"]
        array = self.min_max_wrapper(
            array,
            self.observation_space["array"].low - 1,
            self.observation_space["array"].high + 2,
        )
        state["array"] = array
        return state

    def _generate_static_environment(self) -> Dict[str, np.ndarray | th.Tensor]:
        st_image = self._generate_st()
        st_tensor = th.from_numpy(st_image).float()
        st_tensor = st_tensor.unsqueeze(0).unsqueeze(0)
        speed_limit = self._generate_speed_limit()
        return dict(st_image=st_tensor, array=speed_limit)

    def _generate_speed_limit(self) -> np.ndarray:
        self.max_ref_v = self.speed_limit_generator.max_speed
        speed_limit = self.speed_limit_generator.generate(
            self.max_single_speed_limit_points,
            self.max_speed_limit_intervals,
            self.read_from_file,
        )
        speed_limit_obv = np.zeros(self.speed_limit_samples * 2)
        s = 0
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
        self.st = st_polygons
        return image

    def speed_cost_factor(self) -> float:
        min_coef = 0.1
        k = 1e-3
        factor = min_coef + (1 - min_coef) * np.exp(-k * self.yield_cost)
        if self.interaction:
            factor = min(factor, 0.5)
        return factor

    def smooth_cost_factor(self) -> float:
        if self.exceed_speed <= 0:
            return 1.0
        min_coef = 0.5
        return min_coef + (1 - min_coef) * np.exp(-self.exceed_speed)

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

    def render(self, s: np.ndarray, a: float, r: float):
        if self.render_mode:
            self.visualiser.render(s, a, r)
            plt.pause(0.01)

        else:
            if self.visualiser is not None:
                self.visualiser.reset()

    def read_samples_from_file(self, file_path) -> None:
        files = os.listdir(file_path)
        speed_buffer = []
        st_buffer = []
        for f in files:
            with open(os.path.join(file_path, f)) as file:
                data = json.load(file)[0]
                for details in data["detail"]:
                    detail = json.loads(details["dp_data"])
                    speed_info = detail["speed_limits"]
                    st_info = detail["st_polygons"]
                    speed_buffer.append(SpeedLimit(np.array(speed_info)))
                    st_boudaries = [
                        STBoundary(
                            list(map(lambda x: STPoint(x), st["lower_points"])),
                            list(map(lambda x: STPoint(x), st["upper_points"])),
                            bool(st["soft"]),
                        )
                        for st in st_info
                    ]
                    st_buffer.append(st_boudaries)
        self.speed_limit_generator.load_from_local(speed_buffer)
        self.st_generator.load_from_local(st_buffer)
        self.read_from_file = len(speed_buffer) and len(st_buffer)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    init_state = np.array([0, 0, 0])
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
    terminate = False
    trancate = False
    state, _ = env.reset(seed)
    vis.start_animation()
    while not terminate and not trancate:
        u = np.random.uniform(-params["max_jerk"], params["max_jerk"])
        state, reward, terminate, trancate, _ = env.step(u)
    plt.show()
