from dataclasses import dataclass
from math import cos, hypot, radians, sin

import cv2
import numpy as np


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def moved(self, dx: float, dy: float) -> "Point":
        return Point(self.x + dx, self.y + dy)


@dataclass(frozen=True)
class RotatedRect:
    center: Point
    width: float
    height: float
    angle: float

    @classmethod
    def from_cv2(cls, rect: tuple[tuple[float, float], tuple[float, float], float]) -> "RotatedRect":
        (cx, cy), (width, height), angle = rect
        return cls(Point(float(cx), float(cy)), float(width), float(height), float(angle))

    def to_cv2(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        return (self.center.x, self.center.y), (self.width, self.height), self.angle

    def points(self) -> tuple[Point, Point, Point, Point]:
        box = cv2.boxPoints(self.to_cv2())
        return tuple(Point(float(x), float(y)) for x, y in box)

    def edge_lengths(self) -> tuple[float, float, float, float]:
        points = self.ordered_points()
        return tuple(_distance(points[index], points[(index + 1) % 4]) for index in range(4))

    def ordered_points(self) -> tuple[Point, Point, Point, Point]:
        width_axis, height_axis = self.axes()
        center = np.array([self.center.x, self.center.y], dtype=np.float64)
        half_width = self.width / 2.0
        half_height = self.height / 2.0

        top_left = center - width_axis * half_width - height_axis * half_height
        top_right = center + width_axis * half_width - height_axis * half_height
        bottom_right = center + width_axis * half_width + height_axis * half_height
        bottom_left = center - width_axis * half_width + height_axis * half_height
        return (
            Point(float(top_left[0]), float(top_left[1])),
            Point(float(top_right[0]), float(top_right[1])),
            Point(float(bottom_right[0]), float(bottom_right[1])),
            Point(float(bottom_left[0]), float(bottom_left[1])),
        )

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        theta = radians(self.angle)
        width_axis = np.array([cos(theta), sin(theta)], dtype=np.float64)
        height_axis = np.array([-sin(theta), cos(theta)], dtype=np.float64)
        return width_axis, height_axis

    def moved(self, dx: float, dy: float) -> "RotatedRect":
        return RotatedRect(self.center.moved(dx, dy), self.width, self.height, self.angle)

    def rotated(self, delta_angle: float) -> "RotatedRect":
        return RotatedRect(self.center, self.width, self.height, self.angle + delta_angle)

    def scaled(self, delta: float) -> "RotatedRect":
        return RotatedRect(
            self.center,
            max(1.0, self.width + delta),
            max(1.0, self.height + delta),
            self.angle,
        )

    def moved_edge_by_vector(self, edge: str, dx: float, dy: float) -> "RotatedRect":
        width_axis, height_axis = self.axes()
        vector = np.array([dx, dy], dtype=np.float64)

        if edge == "right":
            return self._resized("width", float(vector.dot(width_axis)), width_axis)
        if edge == "left":
            return self._resized("width", float(vector.dot(-width_axis)), -width_axis)
        if edge == "bottom":
            return self._resized("height", float(vector.dot(height_axis)), height_axis)
        if edge == "top":
            return self._resized("height", float(vector.dot(-height_axis)), -height_axis)
        raise ValueError(f"Unknown edge: {edge}")

    def moved_corner_by_vector(self, corner: str, dx: float, dy: float) -> "RotatedRect":
        if corner == "top_left":
            return self.moved_edge_by_vector("top", dx, dy).moved_edge_by_vector("left", dx, dy)
        if corner == "top_right":
            return self.moved_edge_by_vector("top", dx, dy).moved_edge_by_vector("right", dx, dy)
        if corner == "bottom_right":
            return self.moved_edge_by_vector("bottom", dx, dy).moved_edge_by_vector("right", dx, dy)
        if corner == "bottom_left":
            return self.moved_edge_by_vector("bottom", dx, dy).moved_edge_by_vector("left", dx, dy)
        raise ValueError(f"Unknown corner: {corner}")

    def _resized(self, dimension: str, delta: float, normal: np.ndarray) -> "RotatedRect":
        if dimension == "width":
            new_width = max(1.0, self.width + delta)
            actual_delta = new_width - self.width
            new_height = self.height
        else:
            new_height = max(1.0, self.height + delta)
            actual_delta = new_height - self.height
            new_width = self.width

        center_dx, center_dy = normal * (actual_delta / 2.0)
        return RotatedRect(
            self.center.moved(float(center_dx), float(center_dy)),
            new_width,
            new_height,
            self.angle,
        )


def _distance(start: Point, end: Point) -> float:
    return hypot(end.x - start.x, end.y - start.y)
