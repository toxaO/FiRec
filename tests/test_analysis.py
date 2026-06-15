import numpy as np
import cv2

from firec.core.analysis import (
    compare_fields,
    detect_radiation_field,
    invert_profile,
    line_profile,
    load_image,
    rotate_image_around_rect,
    rotate_image_to_align_rect,
)
from firec.core.geometry import Point, RotatedRect
from firec.storage.repository import connect_database, fetch_analysis_rows, save_analysis


def test_load_image_reads_tiff(tmp_path):
    import tifffile

    image_path = tmp_path / "image.tif"
    expected = np.zeros((8, 8), dtype=np.uint8)
    tifffile.imwrite(image_path, expected)

    actual = load_image(image_path)

    assert actual.shape == expected.shape


def test_detect_radiation_field_finds_dark_rectangle():
    image = np.full((120, 160), 40000, dtype=np.uint16)
    rect = ((80, 60), (50, 30), 15)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(image, box, 10000)

    detection = detect_radiation_field(image)

    assert detection.rect.width > 25
    assert detection.rect.height > 25
    assert 70 <= detection.rect.center.x <= 90
    assert 50 <= detection.rect.center.y <= 70


def test_rotated_rect_moves_selected_edge():
    rect = RotatedRect(Point(50, 50), width=40, height=20, angle=0)

    moved = rect.moved_edge_by_vector("right", 10, 0)

    assert moved.width == 50
    assert moved.height == 20
    assert moved.center.x == 55
    assert moved.center.y == 50


def test_line_profile_reads_horizontal_values():
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)

    profile = line_profile(image, Point(1, 2), Point(3, 2))

    assert profile.tolist() == [11, 12, 13]


def test_line_profile_reads_vertical_values():
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)

    profile = line_profile(image, Point(2, 1), Point(2, 3))

    assert profile.tolist() == [7, 12, 17]


def test_invert_profile_flips_values_against_maximum():
    profile = invert_profile(np.array([2, 5, 3], dtype=np.float64))

    assert profile.tolist() == [3, 0, 2]


def test_rotate_image_to_align_rect_sets_rect_angle_to_zero():
    image = np.full((120, 160), 40000, dtype=np.uint16)
    rect = RotatedRect(Point(80, 60), width=50, height=30, angle=15)

    rotated, aligned_rect = rotate_image_to_align_rect(image, rect)

    assert rotated.shape[0] > image.shape[0]
    assert rotated.shape[1] > image.shape[1]
    assert aligned_rect.angle == 0
    assert aligned_rect.width == rect.width
    assert aligned_rect.height == rect.height


def test_rotate_image_around_rect_keeps_rect_unrotated():
    image = np.full((50, 60), 255, dtype=np.uint8)
    rect = RotatedRect(Point(30, 25), width=20, height=10, angle=0)

    rotated, aligned_rect = rotate_image_around_rect(image, rect, 5)

    assert rotated.size > 0
    assert aligned_rect.angle == 0
    assert aligned_rect.width == rect.width
    assert aligned_rect.height == rect.height


def test_save_and_fetch_analysis(tmp_path):
    database_path = tmp_path / "firec.sqlite"
    connection = connect_database(database_path)
    radiation = RotatedRect(Point(10, 10), width=20, height=30, angle=0)
    light = RotatedRect(Point(11, 12), width=25, height=35, angle=0)
    result = compare_fields(radiation, light)

    save_analysis(connection, "image.tif", result)
    rows = fetch_analysis_rows(connection)

    assert len(rows) == 1
    assert rows[0]["image_path"] == "image.tif"
    assert rows[0]["width_difference"] == 5
    assert rows[0]["height_difference"] == 5
    assert rows[0]["width_ratio"] == 1.25
    assert rows[0]["center_dx"] == 1
