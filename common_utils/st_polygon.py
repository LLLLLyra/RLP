from typing import List, Optional, Tuple
import bisect
from common_utils.math_utils import Polygon2d, kMathEpsilon
from common_utils.points import STPoint


class STBoundary(Polygon2d):
    def __init__(self, lower_points: List[STPoint], upper_points: List[STPoint], soft: bool = False):
        if not self._is_valid(lower_points, upper_points):
            raise AssertionError("ST points are invalid")

        self.lower_points = lower_points
        self.upper_points = upper_points
        self.points = lower_points + upper_points[::-1]
        super().__init__(self.points)

        self.min_s: float = min(self.lower_points, key=lambda x: x.s()).s()
        self.max_s: float = max(self.upper_points, key=lambda x: x.s()).s()

        self.min_t = self.lower_points[0].t()
        self.max_t = self.lower_points[-1].t()

        self.soft = soft

    def _is_valid(
        self, lower_points: List[STPoint], upper_points: List[STPoint]
    ) -> bool:
        if len(upper_points) != len(lower_points) or len(lower_points) < 2:
            raise ValueError("invalid ST points number")

        st_pair = list(zip(lower_points, upper_points))

        for i, (lower_point, upper_point) in enumerate(st_pair):
            if lower_point.s() > upper_point.s():
                raise ValueError("invalid lower s")

            if abs(lower_point.t() - upper_point.t()) > kMathEpsilon:
                raise ValueError(
                    "Points in every ST-point pair should be at the same time"
                )

            if i > 1 and st_pair[i - 1][0].t() > lower_point.t():
                raise ValueError("Latter points should have larger t")

        return True

    def get_index_range(
        self, points: List[STPoint], t: float
    ) -> Optional[Tuple[int, int]]:
        if t < points[0].t() or t > points[-1].t():
            return None, None

        idx = bisect.bisect_left(points, t, key=lambda x: x.t())
        if idx == 0:
            return 0, 0
        elif idx == len(points):
            return len(points) - 1, len(points) - 1
        else:
            return idx - 1, idx

    def get_boundary_s_range(self, t: float) -> Optional[Tuple[float, float]]:
        if t < self.min_t or t > self.max_t:
            return None, None

        left, right = self.get_index_range(self.lower_points, t)
        if left is None or right is None:
            return None, None

        r = (
            0.0
            if left == right
            else (t - self.upper_points[left].t())
            / (self.upper_points[right].t() - self.upper_points[left].t())
        )

        s_upper = self.upper_points[left].s() + r * (
            self.upper_points[right].s() - self.upper_points[left].s()
        )
        s_lower = self.lower_points[left].s() + r * (
            self.lower_points[right].s() - self.lower_points[left].s()
        )

        s_upper = min(s_upper, 200)
        s_lower = max(s_lower, 0.0)
        return s_lower, s_upper

    def get_boundary_slopes(self, t: float) -> Optional[Tuple[float, float]]:
        if t < self.min_t or t > self.max_t:
            return None, None

        kdt = 0.05
        t_prev = t - kdt
        t_next = t + kdt

        prev_s_lower, prev_s_upper = self.get_boundary_s_range(t_prev)
        nexr_s_lower, next_s_upper = self.get_boundary_s_range(t_next)
        curr_s_lower, curr_s_upper = self.get_boundary_s_range(t)

        if prev_s_lower is None and nexr_s_lower is None:
            return None, None
        ds_upper, ds_lower = 0, 0
        if prev_s_lower is not None and nexr_s_lower is not None:
            ds_upper = 0.5 * (
                (next_s_upper - curr_s_upper) / kdt
                + (curr_s_upper - prev_s_upper) / kdt
            )
            ds_lower = 0.5 * (
                (nexr_s_lower - curr_s_lower) / kdt
                + (curr_s_lower - prev_s_lower) / kdt
            )

        elif prev_s_lower is not None:
            ds_upper = (curr_s_upper - prev_s_upper) / kdt
            ds_lower = (curr_s_lower - prev_s_lower) / kdt

        else:
            ds_upper = (next_s_upper - curr_s_upper) / kdt
            ds_lower = (nexr_s_lower - curr_s_lower) / kdt

        return ds_lower, ds_upper

    def is_empty(self):
        return len(self.lower_points) == 0
