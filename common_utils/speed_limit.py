import bisect
import numpy as np


class SpeedLimit:
    def __init__(self, speed_limit: np.ndarray = np.array([[]])):
        speed_limit = speed_limit.reshape((-1, 2))
        if (
            len(speed_limit.shape) != 2
            and speed_limit.shape[1] != 2
            and speed_limit.shape[1] != 0
        ):
            raise IndexError("invalid speed limit shapes: " + str(speed_limit.shape))
        self.speed_limit_points: np.ndarray = speed_limit
        self.index = 0

    def append_speed_limit(self, s: float, v: float) -> None:
        if self.speed_limit_points.size > 0:
            if s <= self.speed_limit_points[-1, 0]:
                return
        if self.speed_limit_points.size == 0:
            self.speed_limit_points = np.array([[s, v]])
        else:
            self.speed_limit_points = np.append(
                self.speed_limit_points, np.array([[s, v]]), axis=0
            )

    def get_speed_limits_by_s(self, s: float) -> float:
        if self.speed_limit_points.shape[0] < 2:
            return 20.0

        idx = bisect.bisect_left(self.speed_limit_points, s, key=lambda x: x[0])

        idx = min(idx, self.speed_limit_points.shape[0] - 1)
        return self.speed_limit_points[idx, 1]

    def deep_copy(self) -> "SpeedLimit":
        limit = self.speed_limit_points.copy()
        return SpeedLimit(limit)

    def __len__(self):
        return self.speed_limit_points.shape[0]

    def __iter__(self) -> "SpeedLimit":
        self.index = 0
        return self

    def __next__(self) -> np.ndarray[float]:
        if self.index >= len(self):
            raise StopIteration

        value = self.speed_limit_points[self.index]
        self.index += 1
        return value

    def __getitem__(self, key: int | slice | tuple) -> np.ndarray | float:
        ele = self.speed_limit_points[key]

        return ele

    def __setitem__(self, key: int | slice | tuple, value: float | np.ndarray) -> None:
        ori_value = self[key]
        self.speed_limit_points[key] = value

    def __str__(self) -> str:
        return str(self.speed_limit_points)

    def __repr__(self) -> str:
        return self.speed_limit_points.__repr__()


if __name__ == "__main__":
    speedlimit = SpeedLimit()
    for i, s in enumerate(np.arange(0, 20, 0.5)):
        speedlimit.append_speed_limit(s, 5.5)
        assert speedlimit[i][0] == s
        assert speedlimit[i][1] == 5.5
        assert speedlimit[i, 0] == s

    sub_speed_limit = speedlimit.deep_copy()
    sub_speed_limit = SpeedLimit(sub_speed_limit[:10])
    assert len(sub_speed_limit) == 10
    sub_speed_limit[0, 1] = 10
    for i, speed in enumerate(sub_speed_limit):
        if i == 0:
            assert speed[1] == 10
        else:
            assert speed[1] == 5.5

        assert speedlimit[i, 1] == 5.5

    assert speedlimit.get_speed_limits_by_s(25) == 5.5
    assert speedlimit.get_speed_limits_by_s(7.89) == 5.5
    assert sub_speed_limit.get_speed_limits_by_s(0) == 10.