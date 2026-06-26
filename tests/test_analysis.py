import sqlite3

import numpy as np
import cv2

from firec.core.analysis import (
    circular_region_mean,
    compare_field_polygons,
    detect_radiation_field,
    detect_profile_boundaries,
    invert_profile,
    line_profile,
    load_image,
    moving_average_profile,
    rectangular_region_mean,
    rotate_image_around_rect,
    rotate_image_to_align_rect,
    tiff_image_dpi,
)
from firec.core.geometry import Point, RotatedRect
from firec.gui.app import _record_display_row
from firec.storage.repository import connect_database, delete_analysis, fetch_analysis_rows, save_analysis


def test_load_image_reads_tiff(tmp_path):
    import tifffile

    image_path = tmp_path / "image.tif"
    expected = np.zeros((8, 8), dtype=np.uint8)
    tifffile.imwrite(image_path, expected)

    actual = load_image(image_path)

    assert actual.shape == expected.shape


def test_tiff_image_dpi_reads_resolution_tags(tmp_path):
    import tifffile

    image_path = tmp_path / "image.tif"
    tifffile.imwrite(image_path, np.zeros((4, 4), dtype=np.uint8), resolution=(300, 300), resolutionunit="INCH")

    assert tiff_image_dpi(image_path) == 300.0


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


def test_moving_average_profile_preserves_length_with_edge_padding():
    profile = moving_average_profile(np.array([10, 20, 40], dtype=np.float64), 3)

    np.testing.assert_allclose(profile, [40 / 3, 70 / 3, 100 / 3])


def test_detect_profile_boundaries_finds_first_crossings_from_center():
    positions = np.arange(11, dtype=np.float64)
    values = np.array([100, 100, 100, 50, 10, 10, 10, 50, 100, 100, 100], dtype=np.float64)

    detection = detect_profile_boundaries(
        positions,
        values,
        film_pixel_value=100,
        threshold_percent=50,
        center_range_mm=1.0,
        dpi=25.4,
        smoothing_window=1,
    )

    assert detection.radiation_pixel_value == 10
    assert detection.threshold_pixel_value == 55
    assert detection.center_start_position == 4
    assert detection.center_end_position == 6
    assert detection.left_position == 2.9
    assert detection.right_position == 7.1


def test_circular_region_mean_uses_raw_pixels_inside_radius():
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)

    mean = circular_region_mean(image, Point(2, 2), 1.0)

    assert mean == 12.0


def test_rectangular_region_mean_uses_raw_pixels_inside_bounds():
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)

    mean = rectangular_region_mean(image, Point(1, 1), Point(4, 4))

    assert mean == 12.0


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
    result = compare_field_polygons(
        radiation.ordered_points(),
        light.ordered_points(),
        origin_field="laser",
        origin_point=Point(0, 0),
    )

    save_analysis(connection, "image.tif", result)
    rows = fetch_analysis_rows(connection)

    assert len(rows) == 1
    assert rows[0]["image_path"] == "image.tif"
    assert rows[0]["source_dpi"] == 0
    assert rows[0]["laser_center_x_px"] == 0
    assert rows[0]["laser_center_y_px"] == 0
    assert rows[0]["radiation_center_x_px"] == 10
    assert rows[0]["radiation_center_y_px"] == 10
    assert rows[0]["light_center_x_px"] == 11
    assert rows[0]["light_center_y_px"] == 12
    assert rows[0]["radiation_edge_length_x_px"] == 20
    assert rows[0]["radiation_edge_length_y_px"] == 30
    assert rows[0]["radiation_area_px2"] == 600


def test_record_display_row_converts_origin_and_dpi():
    row = {
        "created_at": "2026-01-01T00:00:00",
        "image_path": "image.tif",
        "source_dpi": 0.0,
        "laser_center_x_px": 0.0,
        "laser_center_y_px": 0.0,
        "radiation_center_x_px": 10.0,
        "radiation_center_y_px": 20.0,
        "light_center_x_px": 20.0,
        "light_center_y_px": 40.0,
        "radiation_edge_length_x_px": 30.0,
        "radiation_edge_length_y_px": 40.0,
        "radiation_area_px2": 1200.0,
        "light_area_px2": 2400.0,
    }

    display = _record_display_row(row, "light", 254.0)

    assert display["origin"] == "light"
    assert display["unit"] == "mm"
    assert display["dpi"] == 254.0
    assert display["laser_center_x"] == -2.0
    assert display["laser_center_y"] == -4.0
    assert display["radiation_center_x"] == -1.0
    assert display["radiation_center_y"] == -2.0
    assert display["light_center_x"] == 0.0
    assert display["light_center_y"] == 0.0
    assert display["radiation_edge_length_x"] == 3.0
    assert display["radiation_edge_length_y"] == 4.0
    assert display["radiation_area"] == 12.0


def test_delete_analysis_removes_selected_row(tmp_path):
    database_path = tmp_path / "firec.sqlite"
    connection = connect_database(database_path)
    radiation = RotatedRect(Point(10, 10), width=20, height=30, angle=0)
    light = RotatedRect(Point(11, 12), width=25, height=35, angle=0)
    result = compare_field_polygons(
        radiation.ordered_points(),
        light.ordered_points(),
        origin_field="laser",
        origin_point=Point(0, 0),
    )
    save_analysis(connection, "first.tif", result)
    save_analysis(connection, "second.tif", result)
    first_id = int(fetch_analysis_rows(connection)[0]["id"])

    delete_analysis(connection, first_id)
    rows = fetch_analysis_rows(connection)

    assert len(rows) == 1
    assert rows[0]["image_path"] == "second.tif"


def test_compare_field_polygons_converts_origin_and_dpi():
    radiation = RotatedRect(Point(10, 10), width=20, height=30, angle=0).ordered_points()
    light = RotatedRect(Point(20, 20), width=20, height=30, angle=0).ordered_points()

    result = compare_field_polygons(radiation, light, origin_field="light", dpi=254)

    assert result.unit == "mm"
    assert result.dpi == 254.0
    assert result.radiation_field.center.x == -1.0
    assert result.radiation_field.center.y == -1.0
    assert result.light_field.center.x == 0.0
    assert result.light_field.center.y == 0.0
    assert result.radiation_field.area_length_x == 2.0
    assert result.radiation_field.area_length_y == 3.0
    assert result.radiation_field.area == 6.0


def test_compare_field_polygons_reports_laser_center_relative_to_origin():
    radiation = RotatedRect(Point(10, 10), width=20, height=30, angle=0).ordered_points()
    light = RotatedRect(Point(20, 20), width=20, height=30, angle=0).ordered_points()
    laser_center = Point(5, 5)

    laser_origin = compare_field_polygons(
        radiation,
        light,
        origin_field="laser",
        origin_point=laser_center,
        laser_center=laser_center,
    )
    radiation_origin = compare_field_polygons(
        radiation,
        light,
        origin_field="radiation",
        laser_center=laser_center,
    )

    assert laser_origin.laser_center == Point(0.0, 0.0)
    assert radiation_origin.laser_center == Point(-5.0, -5.0)


def test_compare_field_polygons_uses_midline_intersection_as_center():
    points = (
        Point(0, 0),
        Point(10, 0),
        Point(8, 10),
        Point(0, 9),
    )

    result = compare_field_polygons(points, points, origin_field="laser", origin_point=Point(0, 0))

    assert result.radiation_field.center.x == 4.5
    assert result.radiation_field.center.y == 4.8


def test_connect_database_replaces_outdated_schema(tmp_path):
    database_path = tmp_path / "firec.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            image_path TEXT NOT NULL,
            radiation_width REAL NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()

    connection = connect_database(database_path)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(analyses)")}

    assert "radiation_width" not in columns
    assert "radiation_edge_length_x_px" in columns
    assert "light_area_px2" in columns
