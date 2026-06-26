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
    angle: float
    edge_lengths: tuple[float, float, float, float]
    area: float
    area_length_x: float
    area_length_y: float
    average_edge_length: float


@dataclass(frozen=True)
class AnalysisResult:
    radiation_field: FieldGeometry
    light_field: FieldGeometry
    origin_field: str
    dpi: float
    unit: str
    laser_center: Point | None = None


@dataclass(frozen=True)
class RadiationDetection:
    rect: RotatedRect
    otsu_threshold: float
    threshold: float
    contour_area: float
    image_area: int


@dataclass(frozen=True)
class ProfileBoundaryDetection:
    left_position: float
    right_position: float
    smoothed_values: np.ndarray
    radiation_pixel_value: float
    threshold_pixel_value: float
    center_start_position: float
    center_end_position: float


def load_image(path: str | Path) -> np.ndarray:
    """Load a TIFF image as a numpy array."""
    image = tifffile.imread(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def tiff_image_dpi(path: str | Path) -> float | None:
    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.pages:
                return None
            page = tif.pages[0]
            x_tag = page.tags.get("XResolution")
            y_tag = page.tags.get("YResolution")
            x_resolution = _tag_rational_to_float(x_tag.value) if x_tag is not None else None
            y_resolution = _tag_rational_to_float(y_tag.value) if y_tag is not None else None
            values = [value for value in (x_resolution, y_resolution) if value is not None and value > 0]
            if not values:
                return None
            unit_tag = page.tags.get("ResolutionUnit")
            unit_value = int(unit_tag.value) if unit_tag is not None else 2
            dpi_values = [_resolution_to_dpi(value, unit_value) for value in values]
            return float(sum(dpi_values) / len(dpi_values))
    except Exception:
        return None


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


def moving_average_profile(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    window = max(1, int(window))
    if window == 1:
        return values.astype(np.float64)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values.astype(np.float64), (left, right), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def detect_profile_boundaries(
    positions: np.ndarray,
    values: np.ndarray,
    film_pixel_value: float,
    threshold_percent: float = 50.0,
    center_range_mm: float = 20.0,
    dpi: float = 72.0,
    smoothing_window: int = 5,
) -> ProfileBoundaryDetection:
    if positions.size != values.size or values.size < 3:
        raise ValueError("Profile must contain at least three samples.")

    smoothed = moving_average_profile(values, smoothing_window)
    center_position = (float(np.min(positions)) + float(np.max(positions))) / 2.0
    range_px = max(0.0, float(center_range_mm)) * float(dpi) / 25.4
    center_mask = np.abs(positions - center_position) <= range_px
    if np.any(center_mask):
        center_indices = np.flatnonzero(center_mask)
    else:
        center_indices = np.array([int(np.argmin(np.abs(positions - center_position)))], dtype=np.int64)
    center_start = int(center_indices[0])
    center_end = int(center_indices[-1] + 1)
    radiation_pixel_value = float(np.mean(smoothed[center_indices]))

    denominator = float(film_pixel_value) - radiation_pixel_value
    if np.isclose(denominator, 0.0):
        raise ValueError("Film and radiation pixel values are too close.")
    threshold_pixel_value = float(film_pixel_value) - (float(threshold_percent) / 100.0) * denominator

    left = _find_outward_crossing(positions, smoothed, threshold_pixel_value, center_start, -1)
    right = _find_outward_crossing(positions, smoothed, threshold_pixel_value, center_end - 1, 1)
    if left is None or right is None:
        raise ValueError("Could not find threshold crossings on both sides of the profile.")
    return ProfileBoundaryDetection(
        left,
        right,
        smoothed,
        radiation_pixel_value,
        threshold_pixel_value,
        float(positions[center_start]),
        float(positions[max(center_start, center_end - 1)]),
    )


def circular_region_mean(image: np.ndarray, center: Point, radius: float) -> float:
    gray = ensure_grayscale(image).astype(np.float64)
    if radius <= 0:
        raise ValueError("Radius must be greater than zero.")
    height, width = gray.shape[:2]
    y_indices, x_indices = np.ogrid[:height, :width]
    mask = (x_indices - center.x) ** 2 + (y_indices - center.y) ** 2 <= radius**2
    if not np.any(mask):
        raise ValueError("Circle does not include any pixels.")
    return float(np.mean(gray[mask]))


def rectangular_region_mean(image: np.ndarray, top_left: Point, bottom_right: Point) -> float:
    gray = ensure_grayscale(image).astype(np.float64)
    left = min(top_left.x, bottom_right.x)
    right = max(top_left.x, bottom_right.x)
    top = min(top_left.y, bottom_right.y)
    bottom = max(top_left.y, bottom_right.y)
    if np.isclose(left, right) or np.isclose(top, bottom):
        raise ValueError("Rectangle must include a non-zero area.")
    height, width = gray.shape[:2]
    y_indices, x_indices = np.ogrid[:height, :width]
    mask = (x_indices + 0.5 >= left) & (x_indices + 0.5 <= right) & (y_indices + 0.5 >= top) & (y_indices + 0.5 <= bottom)
    if not np.any(mask):
        raise ValueError("Rectangle does not include any pixels.")
    return float(np.mean(gray[mask]))


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


def _resolution_to_dpi(value: float, unit_value: int) -> float:
    if unit_value == 3:
        return float(value) * 2.54
    return float(value)


def _tag_rational_to_float(value: object) -> float | None:
    if isinstance(value, tuple) and len(value) == 2:
        numerator, denominator = value
        denominator = float(denominator)
        if denominator == 0:
            return None
        return float(numerator) / denominator
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError("Unsupported image dimensions.")


def _find_outward_crossing(
    positions: np.ndarray,
    values: np.ndarray,
    threshold: float,
    start_index: int,
    step: int,
) -> float | None:
    index = start_index
    while 0 <= index + step < values.size:
        next_index = index + step
        value = float(values[index])
        next_value = float(values[next_index])
        if (value - threshold) * (next_value - threshold) <= 0.0 and not np.isclose(value, next_value):
            ratio = (threshold - value) / (next_value - value)
            return float(positions[index] + ratio * (positions[next_index] - positions[index]))
        index = next_index
    return None


def compare_fields(radiation_field: RotatedRect, light_field: RotatedRect) -> AnalysisResult:
    return compare_field_polygons(
        radiation_field.ordered_points(),
        light_field.ordered_points(),
        radiation_field.angle,
        light_field.angle,
    )


def compare_field_polygons(
    radiation_points: tuple[Point, Point, Point, Point],
    light_points: tuple[Point, Point, Point, Point],
    radiation_angle: float = 0.0,
    light_angle: float = 0.0,
    origin_field: str = "radiation",
    dpi: float = 0.0,
    origin_point: Point | None = None,
    laser_center: Point | None = None,
) -> AnalysisResult:
    if origin_field not in ("laser", "radiation", "light"):
        raise ValueError(f"Unknown origin field: {origin_field}")

    pixel_radiation = _field_geometry(radiation_points, radiation_angle)
    pixel_light = _field_geometry(light_points, light_angle)
    if origin_field == "laser":
        if origin_point is None:
            raise ValueError("Laser origin requires an origin point.")
        origin = origin_point
    elif origin_field == "radiation":
        origin = pixel_radiation.center
    else:
        origin = pixel_light.center
    scale = _pixel_to_unit_scale(dpi)
    unit = "mm" if scale != 1.0 else "px"

    radiation = _converted_field_geometry(pixel_radiation, origin, scale)
    light = _converted_field_geometry(pixel_light, origin, scale)
    return AnalysisResult(
        radiation_field=radiation,
        light_field=light,
        origin_field=origin_field,
        dpi=round(float(dpi), 1) if dpi > 0 else 0.0,
        unit=unit,
        laser_center=_converted_point(laser_center or origin_point, origin, scale)
        if (laser_center or origin_point) is not None
        else None,
    )


def _field_geometry(points: tuple[Point, Point, Point, Point], angle: float) -> FieldGeometry:
    edge_lengths = tuple(_distance(points[index], points[(index + 1) % 4]) for index in range(4))
    area_length_x_start = _midpoint(points[3], points[0])
    area_length_x_end = _midpoint(points[1], points[2])
    area_length_y_start = _midpoint(points[0], points[1])
    area_length_y_end = _midpoint(points[2], points[3])
    return FieldGeometry(
        points=points,
        center=_line_intersection(
            area_length_x_start,
            area_length_x_end,
            area_length_y_start,
            area_length_y_end,
        )
        or _polygon_center(points),
        angle=angle,
        edge_lengths=edge_lengths,
        area=_polygon_area(points),
        area_length_x=_distance(area_length_x_start, area_length_x_end),
        area_length_y=_distance(area_length_y_start, area_length_y_end),
        average_edge_length=sum(edge_lengths) / len(edge_lengths),
    )


def _converted_field_geometry(field: FieldGeometry, origin: Point, scale: float) -> FieldGeometry:
    return FieldGeometry(
        points=tuple(_converted_point(point, origin, scale) for point in field.points),
        center=_converted_point(field.center, origin, scale),
        angle=_round1(field.angle),
        edge_lengths=tuple(_round1(length * scale) for length in field.edge_lengths),
        area=_round1(field.area * scale * scale),
        area_length_x=_round1(field.area_length_x * scale),
        area_length_y=_round1(field.area_length_y * scale),
        average_edge_length=_round1(field.average_edge_length * scale),
    )


def _converted_point(point: Point, origin: Point, scale: float) -> Point:
    return Point(_round1((point.x - origin.x) * scale), _round1((point.y - origin.y) * scale))


def _scaled_point(point: Point, scale: float) -> Point:
    return Point(_round1(point.x * scale), _round1(point.y * scale))


def _pixel_to_unit_scale(dpi: float) -> float:
    if dpi <= 0:
        return 1.0
    return 25.4 / float(dpi)


def _round1(value: float) -> float:
    return round(float(value), 1)


def _polygon_center(points: tuple[Point, Point, Point, Point]) -> Point:
    return Point(
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def _polygon_area(points: tuple[Point, Point, Point, Point]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point.x * next_point.y - next_point.x * point.y
    return abs(area) / 2.0


def _midpoint(start: Point, end: Point) -> Point:
    return Point((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)


def _line_intersection(a: Point, b: Point, c: Point, d: Point) -> Point | None:
    denominator = (a.x - b.x) * (c.y - d.y) - (a.y - b.y) * (c.x - d.x)
    if abs(denominator) < 1e-9:
        return None
    px = (
        (a.x * b.y - a.y * b.x) * (c.x - d.x)
        - (a.x - b.x) * (c.x * d.y - c.y * d.x)
    ) / denominator
    py = (
        (a.x * b.y - a.y * b.x) * (c.y - d.y)
        - (a.y - b.y) * (c.x * d.y - c.y * d.x)
    ) / denominator
    return Point(float(px), float(py))


def _distance(start: Point, end: Point) -> float:
    return float(np.hypot(end.x - start.x, end.y - start.y))


def _transform_point(point: Point, matrix: np.ndarray) -> Point:
    x = matrix[0, 0] * point.x + matrix[0, 1] * point.y + matrix[0, 2]
    y = matrix[1, 0] * point.x + matrix[1, 1] * point.y + matrix[1, 2]
    return Point(float(x), float(y))
