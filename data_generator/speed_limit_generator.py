from typing import Optional, Tuple, List
import bisect
from common_utils.speed_limit import SpeedLimit
import numpy as np


class SpeedLimitGenerator:
    def __init__(
        self,
        max_speed: float,
        max_path_len: float,
        seed: Optional[int] = None,
    ):
        self.max_speed = max_speed
        self.max_path_length = max_path_len
        np.random.seed(seed)

        self.speed_limit = SpeedLimit()

        for s in np.linspace(0, self.max_path_length, 50):
            self.speed_limit.append_speed_limit(s, self.max_speed)

        self.local_speed_limit = []

    def sample_path_s(self, speed_limit: SpeedLimit, max_path_length: float) -> int:
        """uniformly ramdomly sample a path s

        Args:
            speed_limit (SpeedLimit):
            max_path_length (float):

        Returns:
            int: index of the speed limit
        """
        l = np.random.uniform(0, max_path_length)
        idx = bisect.bisect_right(speed_limit, l, key=lambda x: x[0])
        idx = np.clip(0, len(speed_limit) - 1, idx)
        return idx

    def sample_path_length(self, speed_limit: SpeedLimit) -> Optional[SpeedLimit]:
        """sample total path length

        Args:
            speed_limit (SpeedLimit):

        Returns:
            Optional[SpeedLimit]: speed limit corresponding to the sampled s
        """
        idx = self.sample_path_s(speed_limit, self.max_path_length)
        idx = np.clip(0, len(speed_limit) - 1, idx)

        return SpeedLimit(speed_limit[: idx + 1])

    def sample_single_speed_limit(self, max_speed: float) -> float:
        """ramdomly sample a speed limit

        Args:
            max_speed (float):

        Returns:
            float: speed limit
        """
        return np.random.uniform(0, max_speed)

    def sample_single_speed_limit_s(
        self, speed_limit: SpeedLimit, max_speed: float
    ) -> SpeedLimit:
        path_len = speed_limit[-1][0]
        speed = self.sample_single_speed_limit(max_speed)
        idx = self.sample_path_s(speed_limit, path_len)
        if idx < 0 or idx > len(speed_limit):
            return speed_limit

        sampled_speed_limit = speed_limit
        sampled_speed_limit[idx][1] = speed
        return sampled_speed_limit

    def sample_path_interval(
        self, speed_limit: SpeedLimit, max_path: float
    ) -> Tuple[int]:
        interval = np.random.uniform(0, max_path, size=2)
        l, h = interval.min(), interval.max()
        idx_l = bisect.bisect_right(speed_limit, l, key=lambda x: x[0])
        idx_h = bisect.bisect_right(speed_limit, h, key=lambda x: x[0])
        idx_l = np.clip(0, len(speed_limit) - 1, idx_l)
        idx_h = np.clip(0, len(speed_limit) - 1, idx_h)
        return idx_l, idx_h

    def sample_speed_limit_interval(
        self, speed_limit: SpeedLimit, max_speed: float
    ) -> SpeedLimit:
        path_len = speed_limit[-1][0]
        idx_l, idx_h = self.sample_path_interval(speed_limit, path_len)
        sample_speed = self.sample_single_speed_limit(max_speed)
        sampled_speed_limit = speed_limit
        for i in range(idx_l, idx_h + 1):
            sampled_speed_limit[i][1] = sample_speed

        return sampled_speed_limit

    def combine_speed_limit(
        self, speedlimit1: SpeedLimit, speedlimit2: SpeedLimit
    ) -> SpeedLimit:
        combine_speed_limit = SpeedLimit()
        for i in range(min(len(speedlimit1), len(speedlimit2))):
            combine_speed_limit.append_speed_limit(
                min(speedlimit1[i, 0], speedlimit2[i, 0]),
                min(speedlimit1[i, 1], speedlimit2[i, 1]),
            )

        return combine_speed_limit
    
    def load_from_local(self, speed_limits: List[SpeedLimit]) -> None:
        self.local_speed_limit = speed_limits

    def generate(
        self,
        max_num_candidate_single_speed: int,
        max_num_candidate_intervals: int,
        from_local: bool = False
    ) -> SpeedLimit:
        
        if from_local:
            idx = np.random.choice(len(self.local_speed_limit))
            return self.local_speed_limit[idx]

        max_speed = self.max_speed
        ori_speed_limit = self.speed_limit

        num_candidate_single_speed = np.random.randint(
            max_num_candidate_single_speed + 1
        )
        num_candidate_intervals = np.random.randint(max_num_candidate_intervals + 1)

        speed_limit1 = self.sample_path_length(ori_speed_limit.deep_copy())

        final_speed_limit = speed_limit1.deep_copy()
        for _ in range(num_candidate_single_speed):
            speed_limit1_copy = speed_limit1.deep_copy()
            speed_limit2 = self.sample_single_speed_limit_s(
                speed_limit1_copy, max_speed
            )
            final_speed_limit = self.combine_speed_limit(
                final_speed_limit, speed_limit2
            )

        for _ in range(num_candidate_intervals):
            speed_limit1_copy = speed_limit1.deep_copy()
            speed_limit3 = self.sample_speed_limit_interval(
                speed_limit1_copy, max_speed
            )
            final_speed_limit = self.combine_speed_limit(
                final_speed_limit, speed_limit3
            )

        return final_speed_limit


if __name__ == "__main__":
    gen = SpeedLimitGenerator(2.5, 20, 1234)
    sampled_speed = gen.generate(5, 5, True)
    print(sampled_speed)
