from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile

from firec.core.geometry import Point, RotatedRect


@dataclass(frozen=True)
class FieldGeometry:
    points: tuple[Point, ...]
    center: Point
    width: float
    height: float
    angle: float
    edge_lengths: tuple[float, float, float, float]


@dataclass(frozen=True)
class AnalysisResult:
    radiation_field: FieldGeometry
    light_field: FieldGeometry
    width_difference: float
    height_difference: float
    width_ratio: float
    height_ratio: float
    center_dx: float
    center_dy: float


@dataclass(frozen=True)
class RadiationDetection:
    rect: RotatedRect
    otsu_threshold: float
    threshold: float
    contour_area: float
    image_area: int


def load_image(path: str | Path) -> np.ndarray:
    """Load a TIFF image as a numpy array."""
    image = tifffile.imread(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def otsu_threshold(image: np.ndarray) -> float:
    normalized = normalize_for_display(image)
    threshold, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return display_threshold_to_image_value(image, threshold)


def detect_radiation_field(image: np.ndarray, threshold: float | None = None) -> RadiationDetection:
    otsu = otsu_threshold(image)
    selected_threshold = otsu if threshold is None else threshold
    mask = make_dark_region_mask(image, selected_threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No radiation field contour was detected.")

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    image_area = int(image.shape[0] * image.shape[1])
    area_ratio = area / image_area
    if area_ratio < 0.001 or area_ratio > 0.95:
        raise ValueError("Detected contour area is outside the expected range.")

    rect = RotatedRect.from_cv2(cv2.minAreaRect(contour))
    return RadiationDetection(rect, otsu, float(selected_threshold), area, image_area)


def initial_radiation_rect(image: np.ndarray) -> tuple[RotatedRect, float, str]:
    otsu = otsu_threshold(image)
    try:
        detection = detect_radiation_field(image, otsu)
        return detection.rect, otsu, "Otsu rectangle"
    except ValueError:
        height, width = image.shape[:2]
        size = min(width, height) * 0.5
        rect = RotatedRect(Point(width / 2.0, height / 2.0), size, size, 0.0)
        return rect, otsu, "Fallback rectangle"


def rotate_image_to_align_rect(image: np.ndarray, rect: RotatedRect) -> tuple[np.ndarray, RotatedRect]:
    return rotate_image_around_rect(image, rect, rect.angle)


def rotate_image_around_rect(image: np.ndarray, rect: RotatedRect, angle: float) -> tuple[np.ndarray, RotatedRect]:
    gray = ensure_grayscale(image)
    height, width = gray.shape[:2]
    center = (rect.center.x, rect.center.y)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    rotated_width = int((height * sin_value) + (width * cos_value))
    rotated_height = int((height * cos_value) + (width * sin_value))

    matrix[0, 2] += rotated_width / 2.0 - center[0]
    matrix[1, 2] += rotated_height / 2.0 - center[1]

    border_value = float(np.max(gray)) if gray.size else 0.0
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    new_center = _transform_point(rect.center, matrix)
    return rotated, RotatedRect(new_center, rect.width, rect.height, 0.0)


def make_dark_region_mask(image: np.ndarray, threshold: float) -> np.ndarray:
    gray = ensure_grayscale(image)
    mask = np.where(gray <= threshold, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def line_profile(image: np.ndarray, start: Point, end: Point) -> np.ndarray:
    gray = ensure_grayscale(image)
    length = max(2, int(round(_distance(start, end))) + 1)
    xs = np.linspace(start.x, end.x, length)
    ys = np.linspace(start.y, end.y, length)
    x_indices = np.clip(np.rint(xs).astype(np.int64), 0, gray.shape[1] - 1)
    y_indices = np.clip(np.rint(ys).astype(np.int64), 0, gray.shape[0] - 1)
    return gray[y_indices, x_indices].astype(np.float64)


def invert_profile(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return float(np.max(values)) - values


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    gray = ensure_grayscale(image)
    min_value = float(np.min(gray))
    max_value = float(np.max(gray))
    if max_value <= min_value:
        return np.zeros(gray.shape, dtype=np.uint8)
    normalized = (gray.astype(np.float64) - min_value) * 255.0 / (max_value - min_value)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def display_threshold_to_image_value(image: np.ndarray, display_threshold: float) -> float:
    gray = ensure_grayscale(image)
    min_value = float(np.min(gray))
    max_value = float(np.max(gray))
    return min_value + (float(display_threshold) / 255.0) * (max_value - min_value)


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError("Unsupported image dimensions.")


def compare_fields(radiation_field: RotatedRect, light_field: RotatedRect) -> AnalysisResult:
    radiation = FieldGeometry(
        points=radiation_field.ordered_points(),
        center=radiation_field.center,
        width=radiation_field.width,
        height=radiation_field.height,
        angle=radiation_field.angle,
        edge_lengths=radiation_field.edge_lengths(),
    )
    light = FieldGeometry(
        points=light_field.ordered_points(),
        center=light_field.center,
        width=light_field.width,
        height=light_field.height,
        angle=light_field.angle,
        edge_lengths=light_field.edge_lengths(),
    )
    return AnalysisResult(
        radiation_field=radiation,
        light_field=light,
        width_difference=light.width - radiation.width,
        height_difference=light.height - radiation.height,
        width_ratio=_safe_ratio(light.width, radiation.width),
        height_ratio=_safe_ratio(light.height, radiation.height),
        center_dx=light.center.x - radiation.center.x,
        center_dy=light.center.y - radiation.center.y,
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _distance(start: Point, end: Point) -> float:
    return float(np.hypot(end.x - start.x, end.y - start.y))


def _transform_point(point: Point, matrix: np.ndarray) -> Point:
    x = matrix[0, 0] * point.x + matrix[0, 1] * point.y + matrix[0, 2]
    y = matrix[1, 0] * point.x + matrix[1, 1] * point.y + matrix[1, 2]
    return Point(float(x), float(y))
