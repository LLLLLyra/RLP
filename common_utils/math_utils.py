from typing import TypeVar, Iterable, List, Optional
import numpy as np

kMathEpsilon = 1e-9


class Vec2d(np.ndarray[float]):
    def __new__(cls, input_array: np.ndarray[float] | Iterable[float]) -> "Vec2d":
        obj = np.asarray(input_array).view(cls)
        if len(obj) > 2 and len(obj) != 0:
            raise ValueError("Input array must contain two elements or empty elements.")
        elif len(obj) == 0:
            obj = np.array([0, 0]).view(cls)

        return obj

    def __array_finalize__(self, obj: "Vec2d") -> None:
        if obj is None:
            return

    def x(self) -> float:
        return self[0]

    def y(self) -> float:
        return self[1]

    def length(self) -> float:
        return np.sqrt(self.sum())

    def normalise(self) -> "Vec2d":
        l = self.length()
        return self / l if l > 0 else Vec2d([0, 0])

    def __eq__(self, other: "Vec2d") -> bool:
        if not isinstance(other, Vec2d):
            raise TypeError("unconsistent type comparison")
        return self[0] == other[0] and self[1] == other[1]


class LineSegment2d:
    def __init__(self, start: Vec2d, end: Vec2d):
        self.start: Vec2d = start
        self.end: Vec2d = end

    def length(self) -> float:
        return (self.end - self.start).length()

    def unit_direction(self) -> Vec2d:
        return (self.end - self.start).normalize()

    def project_onto_unit(self, point: Vec2d) -> float:
        direction = self.unit_direction()
        return np.dot(point - self.start, direction)

    def distance_to(self, point: Vec2d) -> float:
        projection = self.project_onto_unit(point)
        if projection < 0:
            return (point - self.start).length()
        elif projection > self.length():
            return (point - self.end).length()
        else:
            return (
                abs(np.cross(self.end - self.start, point - self.start)) / self.length()
            )

    def is_point_in(self, point: Vec2d) -> bool:
        if self.length() <= kMathEpsilon:
            return (
                abs(point.x() - self.start.x()) <= kMathEpsilon
                and abs(point.y() - self.start.y()) <= kMathEpsilon
            )

        prod = self._ccw(point, self.start, self.end)
        if abs(prod) > kMathEpsilon:
            return False

        return self._is_within(
            point.x(), self.start.x(), self.end.x()
        ) and self._is_within(point.y(), self.start.y(), self.end.y())

    def _is_within(self, value: float, bound1: float, bound2: float) -> bool:
        lower = min(bound1, bound2) - kMathEpsilon
        upper = max(bound1, bound2) + kMathEpsilon
        return lower <= value <= upper

    def has_intersect(self, other: "LineSegment2d") -> bool:
        A, B = self.start, self.end
        C, D = other.start, other.end

        if self._ccw(A, B, C) == 0 and self._ccw(A, B, D) == 0:
            if (
                min(A.x, B.x) <= max(C.x, D.x)
                and max(A.x, B.x) >= min(C.x, D.x)
                and min(A.y, B.y) <= max(C.y, D.y)
                and max(A.y, B.y) >= min(C.y, D.y)
            ):
                return True
            return False

        if (
            self._ccw(A, B, C) * self._ccw(A, B, D) < 0
            and self._ccw(C, D, A) * self._ccw(C, D, B) < 0
        ):
            return True

        return False

    def _ccw(self, A: Vec2d, B: Vec2d, C: Vec2d) -> float:
        return (B.x() - A.x()) * (C.y() - A.y()) - (B.y() - A.y()) * (C.x() - A.x())


class Polygon2d:
    def __init__(self, points: List[Vec2d]):
        self.points = points
        self.num_points = len(points)
        self.build_from_points()

    def build_from_points(self) -> None:
        if self.num_points < 3:
            raise ValueError("A polygon must have at least 3 points.")

        # Calculate area to ensure points are in CCW order
        area = 0.0
        for i in range(1, self.num_points):
            area += (
                self.points[i - 1].x() * self.points[i].y()
                - self.points[i].x() * self.points[i - 1].y()
            )
        area /= 2.0

        if area < 0:
            self.points = self.points[::-1]  # Reverse to make CCW

        # Construct line segments
        self.line_segments = [
            LineSegment2d(self.points[i], self.points[(i + 1) % self.num_points])
            for i in range(self.num_points)
        ]

        # Calculate bounding box
        self.min_x = min(p.x() for p in self.points)
        self.max_x = max(p.x() for p in self.points)
        self.min_y = min(p.y() for p in self.points)
        self.max_y = max(p.y() for p in self.points)

    def is_point_in(self, point: Vec2d) -> bool:
        # Ray casting algorithm to check if a point is inside the polygon
        inside = False
        j = self.num_points - 1
        for i in range(self.num_points):
            if (
                (self.points[i].y() > point.y()) != (self.points[j].y() > point.y())
            ) and (
                point.x()
                < (self.points[j].x() - self.points[i].x())
                * (point.y() - self.points[i].y())
                / (self.points[j].y() - self.points[i].y())
                + self.points[i].x()
            ):
                inside = not inside
            j = i
        return inside

    def distance_to(self, point: Vec2d) -> float:
        if self.is_point_in(point):
            return 0.0
        min_distance = float("inf")
        for segment in self.line_segments:
            min_distance = min(min_distance, segment.distance_to(point))
        return min_distance

    def distance_to_boundary(self, point: Vec2d) -> float:
        min_distance = float("inf")
        for segment in self.line_segments:
            min_distance = min(min_distance, segment.distance_to(point))
        return min_distance

    def has_overlap(self, other: "Polygon2d") -> bool:
        if (
            other.max_x < self.min_x
            or other.min_x > self.max_x
            or other.max_y < self.min_y
            or other.min_y > self.max_y
        ):
            return False

        if self.is_point_in(other.points[0]):
            return True

        if other.is_point_in(self.points[0]):
            return True

        for i in range(other.num_points):
            if self.has_overlap_with_segment(other.line_segments[i]):
                return True

        return False

    def has_overlap_with_segment(self, line_segment: LineSegment2d) -> bool:
        if self.num_points < 3:
            raise ValueError("polygons must have at least 3 points")

        if (
            (line_segment.start.x() < self.min_x and line_segment.end.x() < self.min_x)
            or (
                line_segment.start.x() > self.max_x
                and line_segment.end.x() > self.max_x
            )
            or (
                line_segment.start.y() < self.min_y
                and line_segment.end.y() < self.min_y
            )
            or (
                line_segment.start.y() > self.max_y
                and line_segment.end.y() > self.max_y
            )
        ):
            return False

        if line_segment.length() < kMathEpsilon:
            return self.is_point_in(line_segment.start)

        if self.is_point_in(line_segment.start):
            return True

        if self.is_point_in(line_segment.end):
            return True

        if any(map(lambda x: x.has_intersect(line_segment), self.line_segments)):
            return True

        return False

    @staticmethod
    def is_valid_polygon_points(points: List[Vec2d]) -> bool:
        num_points = len(points)

        if num_points < 3:
            return False

        area = 0.0
        for i in range(1, num_points):
            area += Polygon2d._cross_prod(points[0], points[i - 1], points[i])
        area = abs(area) / 2.0

        if area < kMathEpsilon:
            return False

        return True

    @staticmethod
    def _cross_prod(a: Vec2d, b: Vec2d, c: Vec2d) -> float:
        return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x())

    @staticmethod
    def compute_convex_hull(points: List[Vec2d]) -> Optional["Polygon2d"]:

        n = len(points)
        if n < 3:
            return None

        sorted_indices = sorted(range(n), key=lambda i: (points[i].x(), points[i].y()))

        results = []
        count = 0

        for i in range(n):
            while (
                count > 1
                and Polygon2d._cross_prod(
                    points[results[count - 2]],
                    points[results[count - 1]],
                    points[sorted_indices[i]],
                )
                <= kMathEpsilon
            ):
                results.pop()
                count -= 1
            results.append(sorted_indices[i])
            count += 1

        last_count = count
        for i in range(n - 2, -1, -1):
            while (
                count > last_count
                and Polygon2d._cross_prod(
                    points[results[count - 2]],
                    points[results[count - 1]],
                    points[sorted_indices[i]],
                )
                <= kMathEpsilon
            ):
                results.pop()
                count -= 1
            results.append(sorted_indices[i])
            count += 1

        if count > 1:
            results.pop()
            count -= 1

        if count < 3:
            return None

        convex_hull_points = [points[i] for i in results]

        if not Polygon2d.is_valid_polygon_points(convex_hull_points):
            return None

        return Polygon2d(convex_hull_points)


if __name__ == "__main__":
    valid_points = [Vec2d([0, 0]), Vec2d([4, 0]), Vec2d([4, 3])]

    invalid_points = [Vec2d([0, 0]), Vec2d([1, 1]), Vec2d([2, 2])]

    assert Polygon2d.is_valid_polygon_points(valid_points) == True
    assert Polygon2d.is_valid_polygon_points(invalid_points) == False

    points = [
        Vec2d([0, 0]),
        Vec2d([4, 0]),
        Vec2d([4, 3]),
        Vec2d([2, 2]),
        Vec2d([1, 1]),
        Vec2d([0, 3]),
    ]

    convex_hull = Polygon2d.compute_convex_hull(points)
    s = convex_hull.points
    s_test = [Vec2d([0, 0]), Vec2d([4, 0]), Vec2d([4, 3]), Vec2d([0, 3])]
    assert s == s_test

    polygon1 = Polygon2d([Vec2d([0, 0]), Vec2d([4, 0]), Vec2d([4, 3]), Vec2d([0, 3])])
    polygon2 = Polygon2d([Vec2d([2, 2]), Vec2d([6, 2]), Vec2d([6, 5]), Vec2d([2, 5])])

    assert polygon1.has_overlap(polygon2)

    polygon = Polygon2d([Vec2d([0, 0]), Vec2d([4, 0]), Vec2d([4, 3]), Vec2d([0, 3])])

    line_segment = LineSegment2d(Vec2d([2, 2]), Vec2d([5, 5]))
    assert polygon.has_overlap_with_segment(line_segment)

    line_segment = LineSegment2d(Vec2d([0, 0]), Vec2d([4, 4]))

    point1 = Vec2d([2, 2])
    point2 = Vec2d([5, 5])

    assert line_segment.is_point_in(point1)
    assert not line_segment.is_point_in(point2)
