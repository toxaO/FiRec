import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from firec.gui import app as app_module
from firec.gui.image_view import ImageView
from firec.core.analysis import compare_field_polygons
from firec.core.geometry import Point, RotatedRect
from firec.storage.repository import connect_database
from firec.storage.repository import fetch_analysis_rows
from firec.storage.repository import save_analysis


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(settings_dir))
    QSettings.setPath(QSettings.IniFormat, QSettings.SystemScope, str(settings_dir))
    settings = QSettings("FiRec", "FiRec")
    settings.clear()
    settings.sync()
    monkeypatch.setattr(
        app_module,
        "connect_database",
        lambda _: connect_database(tmp_path / "firec.sqlite"),
    )
    return settings


def _make_window(qapp):
    window = app_module.MainWindow()
    window.hide()
    return window


def test_main_window_restores_user_settings(qapp, isolated_settings):
    window = _make_window(qapp)

    window._set_analyse_dpi_mode("manual")
    window.dpi_spin.setValue(96.0)
    window.film_pixel_spin.setValue(321.0)
    window.radiation_threshold_spin.setValue(42.0)
    window.radiation_center_spin.setValue(18.0)
    window.smoothing_window_spin.setValue(9)
    window.radiation_profile_manual_radio.setChecked(True)
    window.radiation_profile_distance_spin.setValue(55.0)
    window.raw_profile_check.setChecked(False)
    window.smoothed_profile_check.setChecked(False)
    window.origin_combo.setCurrentIndex(window.origin_combo.findData("light"))
    window.display_option_checks["照射野中心"].setChecked(False)
    window.display_option_checks["光照射野境界"].setChecked(False)
    window.line_width_spin.setValue(4)
    window.point_radius_spin.setValue(4)
    window.point_fill_opacity_spin.setValue(42)
    window.point_label_check.setChecked(False)
    window.dash_interval_spin.setValue(9)
    window.settings.sync()
    window.close()

    restored = _make_window(qapp)

    assert restored.main_tabs.tabText(0) == "Analyse"
    assert restored.main_tabs.tabText(1) == "Records"
    assert restored.analyse_dpi_mode == "manual"
    assert restored.analyse_manual_dpi == 96.0
    assert restored.film_pixel_spin.value() == 321.0
    assert restored.radiation_threshold_spin.value() == 42.0
    assert restored.radiation_center_spin.value() == 18.0
    assert restored.smoothing_window_spin.value() == 9
    assert restored.radiation_profile_mode == "manual"
    assert restored.radiation_profile_distance_spin.value() == 55.0
    assert restored.raw_profile_check.isChecked() is False
    assert restored.smoothed_profile_check.isChecked() is False
    assert restored.origin_combo.currentData() == "light"
    assert restored.line_width_spin.value() == 4
    assert restored.point_radius_spin.value() == 4
    assert restored.point_fill_opacity_spin.value() == 42
    assert restored.point_fill_opacity_spin.singleStep() == 10
    assert restored.point_label_check.isChecked() is False
    assert restored.display_option_checks["照射野中心"].isChecked() is False
    assert restored.display_option_checks["光照射野境界"].isChecked() is False
    assert restored.view.line_width == 4
    assert restored.view.point_radius == 4
    assert restored.view.point_fill_opacity == 42
    assert restored.view.show_point_labels is False
    assert restored.dash_interval_spin.value() == 9
    assert restored.view.dash_interval == 9


def test_manual_dpi_updates_on_focus_loss(qapp, isolated_settings):
    window = _make_window(qapp)
    window.show()
    qapp.processEvents()

    window._set_analyse_dpi_mode("manual")
    window.dpi_spin.setFocus()
    qapp.processEvents()

    window.dpi_spin.lineEdit().selectAll()
    QTest.keyClicks(window.dpi_spin.lineEdit(), "96")
    window.path_edit.setFocus()
    qapp.processEvents()

    assert window.analyse_manual_dpi == 96.0
    assert window.dpi_spin.value() == 96.0


def test_manual_dpi_spinbox_steps_apply_immediately(qapp, isolated_settings):
    window = _make_window(qapp)

    window._set_analyse_dpi_mode("manual")
    window.dpi_spin.setValue(96.0)
    window.dpi_spin.stepUp()

    assert window.analyse_manual_dpi == 97.0
    assert window.dpi_spin.value() == 97.0


def test_main_window_starts_with_no_tool_selected(qapp, isolated_settings):
    window = _make_window(qapp)

    assert window.tool_mode is None
    assert window.pan_tool_button.isChecked() is False


def test_tool_button_click_toggles_tool_mode_off(qapp, isolated_settings):
    window = _make_window(qapp)

    window.circle_tool_button.click()
    assert window.tool_mode == "circle"
    assert window.circle_tool_button.isChecked() is True

    window.circle_tool_button.click()
    assert window.tool_mode is None
    assert window.circle_tool_button.isChecked() is False
    assert window.zoom_tool_button.isChecked() is False


def test_tool_button_click_switches_between_tools(qapp, isolated_settings):
    window = _make_window(qapp)

    window.circle_tool_button.click()
    assert window.tool_mode == "circle"

    window.rect_tool_button.click()
    assert window.tool_mode == "rect"
    assert window.circle_tool_button.isChecked() is False
    assert window.rect_tool_button.isChecked() is True


def test_tool_toggle_clears_measurement_state(qapp, isolated_settings):
    window = _make_window(qapp)

    window.circle_tool_button.click()
    window.circle_roi = (Point(1.0, 2.0), 3.0)
    window.rect_roi = (Point(0.0, 0.0), Point(1.0, 1.0))
    window.ruler_points = (Point(0.0, 0.0), Point(2.0, 2.0))
    window.tool_result_label.setText("value")

    window.circle_tool_button.click()

    assert window.tool_mode is None
    assert window.circle_roi is None
    assert window.rect_roi is None
    assert window.ruler_points is None
    assert window.tool_result_label.text() == ""


def test_reset_radiation_field_does_not_overwrite_saved_defaults(qapp, isolated_settings):
    window = _make_window(qapp)

    window.radiation_profile_manual_radio.setChecked(True)
    window.radiation_threshold_spin.setValue(37.0)
    window.settings.sync()

    window.reset_radiation_field()
    assert window.radiation_profile_mode == "auto"
    window.settings.sync()
    window.close()

    restored = _make_window(qapp)

    assert restored.radiation_profile_mode == "manual"
    assert restored.radiation_threshold_spin.value() == 37.0


def test_record_rows_edit_origin_and_dpi_independently(qapp, isolated_settings):
    window = _make_window(qapp)
    radiation = RotatedRect(Point(10, 10), width=20, height=30, angle=0)
    light = RotatedRect(Point(11, 12), width=25, height=35, angle=0)
    result = compare_field_polygons(
        radiation.ordered_points(),
        light.ordered_points(),
        origin_field="laser",
        origin_point=Point(0, 0),
    )
    save_analysis(window.connection, "image-1.tif", result, "laser", 0.0)
    save_analysis(window.connection, "image-2.tif", result, "laser", 0.0)

    window.refresh_results_table()
    window.results_table.selectRow(0)

    origin_widget = window.results_table.cellWidget(0, 2)
    assert origin_widget is not None
    origin_widget.setCurrentIndex(origin_widget.findData("radiation"))

    rows = fetch_analysis_rows(window.connection)
    assert rows[0]["origin"] == "radiation"
    assert window.results_table.item(0, 7).text() == "0.0"
    assert window.results_table.item(0, 5).text() == "-10.0"
    assert window.results_table.selectedIndexes()[0].row() == 0

    dpi_widget = window.results_table.cellWidget(0, 3)
    assert dpi_widget is not None
    dpi_widget.setValue(254.0)

    rows = fetch_analysis_rows(window.connection)
    assert rows[0]["dpi"] == 254.0
    assert window.results_table.item(0, 4).text() == "mm"
    assert window.results_table.item(0, 5).text() == "-1.0"
    assert window.results_table.item(0, 7).text() == "0.0"
    assert window.results_table.selectedIndexes()[0].row() == 0
    assert window.results_table.item(1, 4).text() == "px"


def test_record_columns_can_be_hidden_with_checkboxes(qapp, isolated_settings):
    window = _make_window(qapp)

    assert all(check.isChecked() for check in window.record_column_checks.values())
    window.record_column_checks["origin"].setChecked(False)
    window.record_column_checks["dpi"].setChecked(False)
    window.settings.sync()
    window.close()

    restored = _make_window(qapp)

    assert restored.results_table.isColumnHidden(2) is True
    assert restored.results_table.isColumnHidden(3) is True
    assert restored.results_table.isColumnHidden(0) is False


def test_film_baseline_uses_saved_value_across_images(qapp, isolated_settings, monkeypatch, tmp_path):
    image_a = np.array([[0.0, 1000.0], [1000.0, 0.0]])
    image_b = np.full((4, 4), 900.0)
    images = {
        str(tmp_path / "a.tif"): image_a,
        str(tmp_path / "b.tif"): image_b,
    }

    monkeypatch.setattr(app_module, "load_image", lambda path: images[str(path)])
    monkeypatch.setattr(app_module, "tiff_image_dpi", lambda path: None)

    first = _make_window(qapp)
    first.load_path(Path(tmp_path / "a.tif"))
    assert first.film_pixel_spin.value() == 1000.0
    first.film_pixel_spin.setValue(555.0)
    first.settings.sync()
    first.close()

    restored = _make_window(qapp)
    restored.load_path(Path(tmp_path / "b.tif"))

    assert restored.film_pixel_spin.value() == 555.0


def test_image_view_uses_japanese_floating_labels(qapp):
    view = ImageView()
    view.set_image(np.zeros((4, 4), dtype=np.float32))
    view.set_laser_center(Point(1.0, 2.0))
    texts = [item.toPlainText() for item in view.scene().items() if hasattr(item, "toPlainText")]

    assert "レーザー" in texts


def test_image_view_applies_custom_display_styles(qapp):
    view = ImageView()
    view.set_image(np.zeros((8, 8), dtype=np.float32))
    view.set_laser_center(Point(2.0, 3.0))
    view.set_line_width(4)
    view.set_point_radius(4)
    view.set_point_fill_opacity(42)
    view.set_dash_interval(11)
    view.center_points = {"radiation": Point(1.0, 1.0)}
    view._draw_center_points()
    view.set_radiation_points({"a": Point(3.0, 3.0)})
    view.set_radiation_rect(RotatedRect(Point(4.0, 4.0), width=2.0, height=2.0, angle=0))

    line_items = [item for item in view.scene().items() if hasattr(item, "line")]
    polygon_items = [item for item in view.scene().items() if hasattr(item, "polygon")]
    ellipse_items = [item for item in view.scene().items() if hasattr(item, "rect")]

    assert any(item.pen().width() == 4 for item in line_items)
    assert any(item.pen().dashPattern() == [10.0, 11.0] for item in polygon_items)
    assert ellipse_items
    assert any(item.rect().width() == 8 for item in ellipse_items)
    assert any(item.brush().color().alpha() == 107 for item in ellipse_items)


def test_point_labels_can_be_hidden_and_restored(qapp):
    view = ImageView()
    view.set_image(np.zeros((8, 8), dtype=np.float32))
    view.set_laser_center(Point(2.0, 3.0))
    view.set_result_center_points({"radiation": Point(1.0, 1.0)})
    view.set_radiation_points({"a": Point(3.0, 3.0)})
    view.set_radiation_rect(RotatedRect(Point(4.0, 4.0), width=2.0, height=2.0, angle=0))
    view.set_profile_cursor_points({"L1": Point(2.0, 2.0)})
    assert any(hasattr(item, "toPlainText") for item in view.scene().items())

    view.set_show_point_labels(False)
    assert not any(hasattr(item, "toPlainText") for item in view.scene().items())

    view.set_show_point_labels(True)
    assert any(hasattr(item, "toPlainText") for item in view.scene().items())


def test_profile_cursor_points_follow_point_settings_immediately(qapp):
    view = ImageView()
    view.set_image(np.zeros((8, 8), dtype=np.float32))
    view.set_profile_cursor_points({"L1": Point(2.0, 2.0)})
    view.set_point_radius(6)
    view.set_point_fill_opacity(40)

    ellipse_items = [item for item in view.scene().items() if hasattr(item, "rect")]

    assert any(item.rect().width() == 12 for item in ellipse_items)
    assert any(item.brush().color().alpha() == 102 for item in ellipse_items)


def test_selected_profile_line_uses_configured_width(qapp):
    view = ImageView()
    view.set_image(np.zeros((8, 8), dtype=np.float32))
    view.set_line_width(5)
    view.select_profile_line("top")

    line_items = [item for item in view.scene().items() if hasattr(item, "line")]

    assert line_items
    assert all(item.pen().width() == 5 for item in line_items)


def test_laser_stage_updates_center_from_profile_lines(qapp, isolated_settings):
    window = _make_window(qapp)
    image = np.arange(25, dtype=np.float32).reshape(5, 5)

    window.view.set_image(image)
    window.image = image
    window.activate_stage("laser")

    window.view.set_profile_line_positions(left_x=1.0, bottom_y=3.0)

    assert window.laser_center == Point(1.0, 3.0)
    assert window.profile_cursors["laser_x"] == 1.0
    assert window.profile_cursors["laser_y"] == 3.0


def test_laser_stage_populates_left_and_bottom_profiles_only(qapp, isolated_settings):
    window = _make_window(qapp)
    image = np.arange(25, dtype=np.float32).reshape(5, 5)

    window.view.set_image(image)
    window.image = image
    window.activate_stage("laser")

    window.view.set_profile_line_positions(left_x=1.0, bottom_y=3.0)

    assert window.left_profile_plot.values is not None
    assert window.bottom_profile_plot.values is not None
    assert window.left_profile_plot.positions is not None
    assert window.bottom_profile_plot.positions is not None
    assert window.top_profile_plot.values is None
    assert window.right_profile_plot.values is None


def test_laser_stage_forces_raw_profile_visible_even_when_saved_toggles_are_off(qapp, isolated_settings):
    window = _make_window(qapp)
    image = np.arange(25, dtype=np.float32).reshape(5, 5)

    window.raw_profile_check.setChecked(False)
    window.smoothed_profile_check.setChecked(False)
    window.view.set_image(image)
    window.image = image
    window.activate_stage("laser")

    assert window.left_profile_plot.raw_profile_visible is True
    assert window.bottom_profile_plot.raw_profile_visible is True
    assert window.left_profile_plot.smoothed_profile_visible is False
    assert window.bottom_profile_plot.smoothed_profile_visible is False
    assert window.left_profile_plot.values is not None
    assert window.bottom_profile_plot.values is not None


def test_radiation_stage_keeps_profile_visibility_toggles(qapp, isolated_settings):
    window = _make_window(qapp)
    image = np.arange(25, dtype=np.float32).reshape(5, 5)

    window.raw_profile_check.setChecked(False)
    window.smoothed_profile_check.setChecked(False)
    window.view.set_image(image)
    window.image = image
    window.activate_stage("laser")
    window.activate_stage("radiation")

    assert window.left_profile_plot.raw_profile_visible is False
    assert window.bottom_profile_plot.raw_profile_visible is False
    assert window.left_profile_plot.smoothed_profile_visible is False
    assert window.bottom_profile_plot.smoothed_profile_visible is False
