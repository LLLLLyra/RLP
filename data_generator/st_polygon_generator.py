from typing import Optional, Tuple, List, Callable
import numpy as np
from common_utils.st_polygon import STBoundary, STPoint
from matplotlib.path import Path


class STGenerator:
    def __init__(
        self,
        max_t: float,
        max_abs_slope: float,
        max_min_s: float = 10,
        max_max_s: float = 30,
        seed: Optional[int] = None,
    ):
        self.max_t = max_t
        self.max_abs_slope = max_abs_slope
        self.max_min_s = max_min_s
        self.max_max_s = max_max_s
        self.generators = self.random_type_generator()
        self.local_sts = []
        np.random.seed(seed)

    def sample_slope(self, max_abs_slope: float) -> float:
        max_abs_slope = abs(max_abs_slope)
        is_oncoming = np.random.binomial(1, 0.5)
        h = 0.0 if is_oncoming else max_abs_slope
        l = -max_abs_slope if is_oncoming else 0.0
        slope = np.random.uniform(l, h)
        return slope

    def sample_t_range(self, max_t: float) -> Tuple[float]:
        t = np.random.uniform(0, max_t, size=2)
        return t.min(), t.max()

    def sample_s_range(self, max_min_s: float, max_max_s: float) -> Tuple[float]:
        min_s = np.random.uniform(0, max_min_s)
        max_s = np.random.uniform(max_min_s, max_max_s)
        return min_s, max_s

    def generate_virtual_stop(self, max_min_s: float) -> STBoundary:
        stop_s = np.random.uniform(max_min_s)
        lower_points = [STPoint([0, stop_s]), STPoint([self.max_t, stop_s])]
        upper_points = [
            STPoint([0, 100]),
            STPoint([self.max_t, 100]),
        ]
        return STBoundary(lower_points, upper_points)

    def generate_rectangular(
        self, max_max_t: float, max_min_s: float, max_max_s: float
    ) -> STBoundary:
        min_t, max_t = self.sample_t_range(max_max_t)
        min_s, max_s = self.sample_s_range(max_min_s, max_max_s)
        lower_points = [STPoint([min_t, min_s]), STPoint([max_t, min_s])]
        upper_points = [STPoint([min_t, max_s]), STPoint([max_t, max_s])]
        return STBoundary(lower_points, upper_points)

    def generate_parallelogram(
        self, max_max_t: float, max_min_s: float, max_max_s: float, max_abs_slope: float
    ) -> STBoundary:
        min_t, max_t = self.sample_t_range(max_max_t)
        min_s, max_s = self.sample_s_range(max_min_s, max_max_s)
        slope = self.sample_slope(max_abs_slope)
        dt = max_t - min_t
        lower_points, upper_points = [], []
        if slope >= 0:
            lower_max_s = min_s + slope * dt
            upper_min_s = max_s - slope * dt
            if lower_max_s > max_s:
                ds = lower_max_s - max_s
                max_s = lower_max_s
                upper_min_s = min_s + ds

            lower_points = [STPoint([min_t, min_s]),
                            STPoint([max_t, lower_max_s])]
            upper_points = [STPoint([min_t, upper_min_s]),
                            STPoint([max_t, max_s])]

        else:
            upper_min_s = max_s + slope * dt
            lower_max_s = min_s - slope * dt
            if upper_min_s > 0:
                if upper_min_s < min_s:
                    ds = min_s - upper_min_s
                    min_s = upper_min_s
                    lower_max_s = max_s - ds

                lower_points = [
                    STPoint([min_t, lower_max_s]), STPoint([max_t, min_s])]
                upper_points = [STPoint([min_t, max_s]),
                                STPoint([max_t, upper_min_s])]
            else:
                t_0 = -max_s / slope
                if not (t_0 < dt and t_0 > 0):
                    raise ValueError(f"invalid time: t_0 = {t_0}, dt = {dt}")
                lower_max_s = min_s
                min_s = 0
                lower_points = [
                    STPoint([min_t, lower_max_s]), STPoint([t_0, 0.0])]
                upper_points = [STPoint([min_t, max_s]), STPoint([t_0, 0.0])]

        return STBoundary(lower_points, upper_points)

    def generate_trapezium(
        self, max_max_t: float, max_min_s: float, max_max_s: float, max_abs_slope: float
    ) -> STBoundary:
        slope = self.sample_slope(max_abs_slope)
        min_t, max_t = self.sample_t_range(max_max_t)
        min_s, max_s = self.sample_s_range(max_min_s, max_max_s)
        lower_points, upper_points = [], []
        dt = max_t - min_t
        if slope > 0:
            lower_s = max_s - slope * dt
            t_ratio = abs(lower_s / (min_s + 1e-4))
            if t_ratio > 1:
                t_ratio = 1.0 / t_ratio
            lower_s = max(lower_s, 0.0)
            slope = (max_s - lower_s) / dt
            upper_s = lower_s + slope * dt * (1 - t_ratio)
            lower_points = [
                STPoint([min_t, lower_s]),
                STPoint([min_t + dt * t_ratio, lower_s]),
                STPoint([max_t, upper_s]),
            ]
            upper_points = [
                STPoint([min_t, lower_s]),
                STPoint([min_t + dt * t_ratio, lower_s + slope * dt * t_ratio]),
                STPoint([max_t, max_s]),
            ]

        else:
            lower_s = max_s + slope * dt
            t_ratio = abs(lower_s / (min_s + 1e-4))
            if t_ratio > 1:
                t_ratio = 1.0 / t_ratio
            lower_s = max(lower_s, 0.0)
            slope = -(max_s - lower_s) / dt
            upper_s = lower_s - slope * dt * t_ratio
            lower_points = [
                STPoint([min_t, upper_s]),
                STPoint([min_t + dt * t_ratio, lower_s]),
                STPoint([max_t, lower_s]),
            ]
            upper_points = [
                STPoint([min_t, max_s]),
                STPoint([min_t + dt * t_ratio, lower_s -
                        slope * (1 - t_ratio) * dt]),
                STPoint([max_t, lower_s]),
            ]

        return STBoundary(lower_points, upper_points)

    def random_type_generator(self) -> List[Callable[[], STBoundary]]:
        random_generator = [
            lambda: self.generate_trapezium(
                self.max_t, self.max_min_s, self.max_max_s, self.max_abs_slope
            ),
            lambda: self.generate_parallelogram(
                self.max_t, self.max_min_s, self.max_max_s, self.max_abs_slope
            ),
            lambda: self.generate_rectangular(
                self.max_t, self.max_min_s, self.max_max_s
            ),
            lambda: self.generate_virtual_stop(self.max_min_s),
        ]
        return random_generator
    
    def load_from_local(self, sts: List[STBoundary]) -> None:
        self.local_sts = sts

    def generate(
        self, max_soft_polygons: int, max_hard_polygons: int, from_local: bool = False
    ) -> List[STBoundary]:
        if from_local:
            idx = np.random.choice(range(len(self.local_sts)))
            return self.local_sts[idx]

        num_soft_polygon = np.random.randint(0, max_soft_polygons + 1)
        num_hard_polygon = np.random.randint(0, max_hard_polygons + 1)

        polygons = []

        for _ in range(num_hard_polygon):
            idx = np.random.randint(0, len(self.generators))
            st_boundary = self.generators[idx]()
            st_boundary.soft = False
            polygons.append(st_boundary)

        for _ in range(num_soft_polygon):
            idx = np.random.randint(0, len(self.generators))
            st_boundary = self.generators[idx]()
            st_boundary.soft = True
            polygons.append(st_boundary)

        return polygons

    def encode_st(self, st_boundaries: List[STBoundary]) -> np.ndarray:
        resolution = 0.1
        width = int(self.max_t / resolution)
        height = int(self.max_max_s / resolution)
        image = np.zeros((height, width), dtype=np.uint8) + 254
        for st in st_boundaries:
            if st.is_empty():
                continue
            points = np.array(st.points)
            vertices = (points / resolution).astype(int)
            poly_path = Path(vertices)
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            mask = poly_path.contains_points(points).reshape(height, width)
            image[mask] = np.minimum(image[mask], (200 if st.soft else 100))

        return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot(ax, st_boundary):
        x, y = [], []
        for p in st_boundary.points:
            x.append(p.t())
            y.append(p.s())

        x += [x[0]]
        y += [y[0]]
        col = "r" if not st_boundary.soft else "g"
        ax.plot(x, y, col)

    gen = STGenerator(5, 15, seed=2)
    st = gen.generate_parallelogram(5, 3, 10, 15)
    fig, ax = plt.subplots(1, 1)
    res = gen.generate(5, 5)
    for polygon in res:
        plot(ax, polygon)
    image = gen.encode_st(res)
    plt.figure()
    plt.imshow(
        image, cmap="rainbow", origin="lower", extent=[0, gen.max_t, 0, gen.max_max_s]
    )
    plt.show()
