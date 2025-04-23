from typing import List, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from common_utils.speed_limit import SpeedLimit
from common_utils.st_polygon import STBoundary


class EnvVisualiser:
    def __init__(
        self,
        init_physical_state: np.ndarray,
        dt: float,
        speed_limit: Optional[SpeedLimit] = None,
        st: Optional[List[STBoundary]] = None,
    ):
        self.dt = dt
        self.fig, self.axs = plt.subplots(2, 3, figsize=(8, 10))
        self.states = defaultdict(list)
        self.states["s"].append(init_physical_state[0])
        self.states["v"].append(init_physical_state[1])
        self.states["a"].append(init_physical_state[2])

        self.speed_limit = speed_limit
        self.st = st
        self.ani: FuncAnimation = None

    def set_st(self, st: List[STBoundary]) -> None:
        self.st = st

    def set_speed_limit(self, speed_limit: SpeedLimit) -> None:
        self.speed_limit = speed_limit

    def render(self, states: np.ndarray, action: float, reward: float) -> None:
        self.states["s"].append(states[0])
        self.states["v"].append(states[1])
        self.states["a"].append(states[2])
        self.states["j"].append(action)
        self.states["rewards"].append(reward)

    def reset(self) -> None:
        s, v, a = self.states["s"][0], self.states["v"][0], self.states["a"][0]
        self.states = defaultdict(list)
        self.states["s"].append(s)
        self.states["v"].append(v)
        self.states["a"].append(a)

    def _plot_single_state(self, i: int, j: int, data: List[float], name: str) -> None:
        self.axs[i, j].clear()
        self.axs[i, j].plot(data)
        self.axs[i, j].set_title(name)

    def _plot_st(self, i: int, j: int):
        for boundary in self.st:
            x, y = [], []
            for p in boundary.points:
                x.append(p.t() / self.dt)
                y.append(p.s())

            x += [x[0]]
            y += [y[0]]
            color = "g" if boundary.soft else "r"
            self.axs[i, j].plot(x, y, color)

    def _plot_speed_limit(self, i: int, j: int):
        s, v = zip(*self.speed_limit.speed_limit_points)
        self.axs[i, j].clear()
        self.axs[i, j].plot(s, v)
        self.axs[i, j].set_title("s-v")

    def plot(self, frame) -> None:
        self._plot_single_state(0, 0, self.states["s"], "s-t")
        self._plot_single_state(0, 1, self.states["v"], "v-t")
        self._plot_single_state(0, 2, self.states["a"], "a-t")
        self._plot_single_state(1, 0, self.states["j"], "j-t")
        self._plot_single_state(1, 1, self.states["rewards"], "rewards")

        self._plot_st(0, 0)

        self._plot_speed_limit(1, 2)
        return self.axs

    def start_animation(self) -> None:
        self.ani = FuncAnimation(
            self.fig, self.plot, interval=1, blit=False, init_func=lambda: self.axs
        )
        plt.show(block=False)
