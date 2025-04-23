from common_utils.math_utils import Vec2d


class STPoint(Vec2d):
    def s(self) -> float:
        return super().y()

    def t(self) -> float:
        return super().x()
