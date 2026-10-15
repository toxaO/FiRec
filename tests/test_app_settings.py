import os
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from firec.gui import app as app_module
from firec.gui.image_view import ImageView
from firec.core.analysis import compare_field_polygons
from firec.core.geometry import Point, RotatedRect
from firec.storage.repository import connect_database
from firec.storage.repository import fetch_analysis_rows
from firec.storage.repository import save_analysis


def test_dpi_input_is_default_and_persists(qapp, isolated_settings):
    window = _make_window(qapp)
    assert window.calibration_method.currentData() == "dpi"
    window.dpi_input.setText("300")
    assert window._effective_dpi() == 300
    assert "300 DPI" in window.calibration_label.text()
    restored = _make_window(qapp)
    assert restored.calibration_method.currentData() == "dpi"
    assert restored._effective_dpi() == 300
    window._loaded_image_dpi = 254.0
    window._set_analyse_dpi_mode("tag")
    assert window.calibration_label.text() == "254 DPI"
    window._set_analyse_dpi_mode("dpi")
    assert window._effective_dpi() == 300
    for value in ("0", "-1", "nan", "inf", "abc"):
        window.dpi_input.setText(value)
        assert window._calibration_error()
        assert window._effective_dpi() == 0
    window.close()
    restored.close()


def test_python_entry_point_sets_application_icon(qapp, monkeypatch):
    application = Mock()
    application.exec.return_value = 0
    window = Mock()
    monkeypatch.setattr(app_module, "QApplication", lambda args: application)
    monkeypatch.setattr(app_module, "MainWindow", lambda: window)
    assert app_module.main() == 0
    application.setWindowIcon.assert_called_once()
    icon = application.setWindowIcon.call_args.args[0]
    assert not icon.isNull()
    assert not icon.pixmap(64, 64).isNull()
    window.show.assert_called_once()


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


def _find_label(window, text: str) -> QLabel:
    for label in window.findChildren(QLabel):
        if label.text() == text:
            return label
    raise AssertionError(f"label not found: {text}")


def test_main_window_restores_user_settings(qapp, isolated_settings):
    window = _make_window(qapp)

    window._set_analyse_dpi_mode("numeric")
    window.pixel_length.setText("0.1")
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

    assert restored.main_tabs.tabText(0) == "解析"
    assert restored.main_tabs.tabText(1) == "記録"
    assert restored.analyse_dpi_mode == "numeric"
    assert float(restored.pixel_length.text()) == 0.1
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


def test_analyse_parameter_labels_show_hover_help(qapp, isolated_settings):
    window = _make_window(qapp)

    assert _find_label(window, "ベース画素値").toolTip().replace("\n", "") == (
        "未照射部分の画素値として扱う値です。未照射フィルムや照射野外の ROI の値を入れてください。"
    )
    assert "黄色線" in _find_label(window, "閾値").toolTip().replace("\n", "")
    assert "緑線" in _find_label(window, "照射野画素値平均範囲").toolTip().replace("\n", "")
    assert "スムージング" in _find_label(window, "smoothed px").toolTip().replace("\n", "")
    assert "距離" in _find_label(window, "レーザー中心からのoffset").toolTip().replace("\n", "")
    assert window.radiation_profile_auto_radio.toolTip().replace("\n", "") == "レーザー中心から一定距離に 4 本のラインプロファイル線を自動配置します。"
    assert window.radiation_profile_manual_radio.toolTip().replace("\n", "") == "ラインプロファイル線を直接ドラッグして位置を決めます。"
    assert window.raw_profile_check.toolTip().replace("\n", "") == "ラインプロファイルを表示します。"
    assert window.smoothed_profile_check.toolTip().replace("\n", "") == "スムージングをかけたラインプロファイルを表示します。"
    assert window.point_label_check.toolTip().replace("\n", "") == "画像上の点が何を示すかのラベルを表示します。"
    assert _find_label(window, "線幅").toolTip().replace("\n", "") == "画像上の線の太さを設定します。"
    assert _find_label(window, "点線間隔").toolTip().replace("\n", "") == "画像上の点線の間隔を設定します。"
    assert _find_label(window, "点の半径").toolTip().replace("\n", "") == "画像上の点の半径を設定します。"
    assert "0%で透明" in _find_label(window, "点の塗りつぶし不透明度").toolTip().replace("\n", "")
    assert window.display_option_checks["照射野境界点"].toolTip().replace("\n", "") == "ラインプロファイル上の閾値で決まる境界の位置です。"
    assert "表示・非表示" in window.display_option_checks["レーザー中心"].toolTip().replace("\n", "")
    assert not window.film_pixel_spin.toolTip()


def test_input_fields_have_no_hover_help_after_dpi_mode_changes(qapp, isolated_settings):
    from PySide6.QtWidgets import QAbstractSpinBox, QComboBox, QLineEdit

    window = _make_window(qapp)
    for mode in ("none", "numeric", "tag"):
        window._set_analyse_dpi_mode(mode)
        for widget in window.findChildren(QLineEdit) + window.findChildren(QAbstractSpinBox) + window.findChildren(QComboBox):
            assert not widget.toolTip()
    window.close()


def test_label_help_ignores_padding_but_appears_over_text(qapp):
    from PySide6.QtCore import QEvent, QPoint
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtWidgets import QToolTip

    label = app_module._tooltip_label("線幅", "線の太さを指定します。")
    label.setFixedSize(300, 30)
    label.show()
    qapp.processEvents()
    QToolTip.hideText()
    outside = QPoint(290, 15)
    qapp.sendEvent(label, QHelpEvent(QEvent.ToolTip, outside, label.mapToGlobal(outside)))
    assert not QToolTip.isVisible()
    inside = QPoint(10, 15)
    qapp.sendEvent(label, QHelpEvent(QEvent.ToolTip, inside, label.mapToGlobal(inside)))
    assert QToolTip.isVisible()
    QToolTip.hideText()
    label.close()


@pytest.mark.parametrize("stage", ["laser", "radiation", "light", "result"])
def test_stage_help_shows_on_hover_without_changing_analysis(qapp, isolated_settings, stage):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtWidgets import QToolTip

    window = _make_window(qapp)
    window.show()
    qapp.processEvents()
    original_stage = window.current_stage
    window.completed_stages = {"laser", "radiation", "light"}
    original_completed = window.completed_stages.copy()
    button = window.stage_help_buttons[stage]
    assert button.isEnabled()
    assert not button.isCheckable()
    position = button.rect().center()
    qapp.sendEvent(button, QHelpEvent(QEvent.ToolTip, position, button.mapToGlobal(position)))
    assert QToolTip.isVisible()
    assert "操作の流れ" in QToolTip.text()
    assert window.current_stage == original_stage
    assert window.completed_stages == original_completed
    QToolTip.hideText()
    window.close()


@pytest.mark.parametrize("stage", ["laser", "radiation", "light", "result"])
def test_stage_help_click_shows_explanation_without_changing_stage(qapp, isolated_settings, stage):
    from PySide6.QtWidgets import QToolTip

    window = _make_window(qapp)
    window.show()
    qapp.processEvents()
    current_stage = window.current_stage
    window.stage_help_buttons[stage].click()
    assert QToolTip.isVisible()
    assert "操作の流れ" in QToolTip.text()
    assert window.current_stage == current_stage
    QToolTip.hideText()
    window.close()


def test_numeric_resolution_updates_on_focus_loss(qapp, isolated_settings):
    window = _make_window(qapp)
    window.show()
    window._set_analyse_dpi_mode("numeric")
    window.pixel_length.setFocus()
    window.pixel_length.selectAll()
    QTest.keyClicks(window.pixel_length, "0.1")
    window.path_edit.setFocus()
    qapp.processEvents()
    assert window._effective_dpi() == pytest.approx(254)


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


@pytest.mark.parametrize("dpi", [12700.0, 25.4 / 0.09633])
def test_record_origin_changes_preserve_calibration(qapp, isolated_settings, dpi):
    window = _make_window(qapp)
    rectangle = RotatedRect(Point(10, 10), width=20, height=30, angle=0)
    result = compare_field_polygons(
        rectangle.ordered_points(), rectangle.ordered_points(),
        origin_field="laser", origin_point=Point(0, 0),
    )
    save_analysis(window.connection, "calibrated.tif", result, "laser", dpi)
    assert fetch_analysis_rows(window.connection)[0]["dpi"] == dpi
    window.refresh_results_table()
    assert window.results_table.cellWidget(0, 3).value() == pytest.approx(dpi, abs=0.05)
    origin_widget = window.results_table.cellWidget(0, 2)
    for origin in ("radiation", "light"):
        origin_widget.setCurrentIndex(origin_widget.findData(origin))
        assert fetch_analysis_rows(window.connection)[0]["dpi"] == dpi
    window.close()


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


def test_display_options_toggle_reveals_all_groups_and_preserves_choices(qapp, isolated_settings):
    window = _make_window(qapp)
    window.show()
    qapp.processEvents()
    assert not window.line_width_spin.isVisible()
    window.display_toggle.click()
    qapp.processEvents()
    controls = [window.line_width_spin, window.point_label_check, *window.display_option_checks.values()]
    assert all(control.isVisible() for control in controls)
    window.display_option_checks["照射野境界"].setChecked(False)
    window.display_toggle.click()
    qapp.processEvents()
    assert not any(control.isVisible() for control in controls)
    window.display_toggle.click()
    qapp.processEvents()
    assert all(control.isVisible() for control in controls)
    assert not window.display_option_checks["照射野境界"].isChecked()
    assert not window.view.show_radiation_edges
    window.close()


def test_open_button_loads_image_from_entered_path(qapp, isolated_settings, tmp_path):
    import tifffile

    path = tmp_path / "image.tif"
    image = np.arange(100, dtype=np.uint16).reshape(10, 10)
    tifffile.imwrite(path, image)
    window = _make_window(qapp)
    window.path_edit.setText(str(path))
    window.open_button.click()
    assert window.image_path == path
    np.testing.assert_array_equal(window.image, image)
    window.close()


def test_operation_help_tracks_tool_and_returns_to_current_stage(qapp, isolated_settings):
    window = _make_window(qapp)
    stage_help = window.operation_help.text()
    window.pan_tool_button.click()
    assert "移動" in window.operation_help.text()
    assert window.operation_help.text() != stage_help
    window.ruler_tool_button.click()
    assert "2点" in window.operation_help.text()
    window.ruler_tool_button.click()
    assert window.operation_help.text() == stage_help
    window.close()


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


def _loaded_navigation_window(qapp):
    window = _make_window(qapp)
    window.load_path(Path(__file__).parents[1] / "sample" / "img.tif")
    window.radiation_profile_distance_spin.setValue(13.0)
    window.go_to_stage("radiation")
    return window


def test_step_navigation_preserves_fields_cursors_and_completion(qapp, isolated_settings):
    window = _loaded_navigation_window(qapp)
    window.detect_radiation_field()
    window.light_rect = RotatedRect(Point(360, 350), 210, 220, 0)
    window.view.set_light_rect(window.light_rect)
    before = (window.radiation_rect, window.radiation_polygon, window.light_rect,
              dict(window.radiation_points), dict(window.profile_cursors), set(window.completed_stages))
    window.go_to_stage("result")
    window.go_to_stage("laser")
    window.go_to_stage("radiation")
    after = (window.radiation_rect, window.radiation_polygon, window.light_rect,
             dict(window.radiation_points), dict(window.profile_cursors), set(window.completed_stages))
    assert after == before
    window.close()


def test_frame_click_selects_without_scrolling_or_moving_cards(qapp, isolated_settings):
    from PySide6.QtCore import QPoint, Qt

    window = _make_window(qapp)
    window.show()
    qapp.processEvents()
    window.analysis_scroll.verticalScrollBar().setValue(40)
    window.path_edit.setFocus()
    qapp.processEvents()
    scroll = window.analysis_scroll.verticalScrollBar().value()
    geometry = {key: frame.geometry() for key, frame in window.stage_frames.items()}
    QTest.mouseClick(window.stage_frames["light"], Qt.LeftButton, pos=QPoint(2, 2))
    assert window.current_stage == "light"
    assert window.path_edit.hasFocus()
    qapp.processEvents()
    assert window.analysis_scroll.verticalScrollBar().value() == scroll
    assert {key: frame.geometry() for key, frame in window.stage_frames.items()} == geometry
    window.close()


def test_tab_from_image_keeps_stage_and_moves_focus(qapp, isolated_settings):
    from PySide6.QtCore import Qt

    window = _make_window(qapp)
    window.show()
    window.view.setFocus()
    qapp.processEvents()
    QTest.keyClick(window.view, Qt.Key_Tab)
    assert window.current_stage == "laser"
    assert not window.view.hasFocus()
    window.close()


def test_results_process_automatically_and_upstream_changes_block_save(qapp, isolated_settings):
    window = _loaded_navigation_window(qapp)
    window.detect_radiation_field()
    assert window.current_stage == "radiation"
    assert window.light_rect is not None
    assert window.save_button.isEnabled()
    assert window.stage_frames["result"].property("stepState") == "complete"
    assert window.result_tree.topLevelItemCount() > 0
    assert not hasattr(window, "review_button")
    window.save_button.click()
    assert len(fetch_analysis_rows(window.connection)) == 1
    window.origin_combo.setCurrentIndex(window.origin_combo.findData("radiation"))
    assert window.result_tree.topLevelItem(0).text(1) == "radiation 中心"
    assert window.save_button.isEnabled()
    assert window.current_stage == "radiation"
    original_light = window.light_rect
    window.radiation_threshold_spin.setValue(60)
    assert window.light_rect == original_light
    assert not window.save_button.isEnabled()
    assert window.result_tree.topLevelItemCount() == 0
    assert window.stage_frames["result"].property("stepState") == "stale"
    assert window.stage_frames["radiation"].property("stepState") == "stale"
    window.go_to_stage("radiation")
    window.detect_radiation_field()
    assert window.light_rect == original_light
    window.go_to_stage("result")
    assert window.save_button.isEnabled()
    window.close()


@pytest.mark.parametrize("change", ["distance", "mode"])
def test_radiation_controls_update_stale_status_from_result_stage(qapp, isolated_settings, change):
    window = _loaded_navigation_window(qapp)
    window.detect_radiation_field()
    assert window.save_button.isEnabled()
    if change == "distance":
        window.radiation_profile_distance_spin.setValue(14)
    else:
        window.radiation_profile_manual_radio.setChecked(True)
    assert not window.save_button.isEnabled()
    assert window.stage_frames["radiation"].property("stepState") == "stale"
    window.close()


def test_radiation_mode_edit_does_not_disable_laser_line_editing(qapp, isolated_settings):
    window = _loaded_navigation_window(qapp)
    window.go_to_stage("laser")
    window.radiation_profile_manual_radio.setChecked(True)
    window.radiation_profile_auto_radio.setChecked(True)
    assert window.view.profile_lines_editable
    window.close()


def test_result_automatically_reports_all_missing_inputs(qapp, isolated_settings):
    window = _make_window(qapp)
    assert not hasattr(window, "review_button")
    message = window.result_validation_label.text()
    assert "画像" in message
    assert "レーザー" in message
    assert "照射野" in message
    assert "光照射野" in message
    assert not window.save_button.isEnabled()
    window.close()


def test_light_field_can_be_adjusted_before_radiation_detection(qapp, isolated_settings):
    window = _make_window(qapp)
    window.load_path(Path(__file__).parents[1] / "sample" / "img.tif")
    window.go_to_stage("light")
    assert window.light_rect is not None
    adjusted = RotatedRect(Point(350, 340), 220, 240, 0)
    window.view.set_light_rect(adjusted)
    window._on_rect_changed("light", adjusted)
    window.radiation_profile_distance_spin.setValue(13)
    window.detect_radiation_field()
    assert window.light_rect == adjusted
    assert window.save_button.isEnabled()
    assert window.stage_frames["light"].property("stepState") == "complete"
    window.close()


def test_resolution_none_tag_numeric_and_invalid(qapp, isolated_settings):
    w = _make_window(qapp)
    w._loaded_image_dpi = 254
    w._set_analyse_dpi_mode("none")
    assert w._effective_dpi() == 0
    assert w.radiation_center_spin.suffix() == " px"
    w._set_analyse_dpi_mode("tag")
    assert w._effective_dpi() == 254
    w._loaded_image_dpi = 0
    assert w._calibration_error()
    assert w._effective_dpi() == 0
    w._set_analyse_dpi_mode("numeric")
    w.pixel_length.setText("0.1")
    assert w._effective_dpi() == pytest.approx(254)
    for value in ("", "0", "-1", "nan", "inf", "abc"):
        w.pixel_length.setText(value)
        assert w._effective_dpi() == 0
        assert w._calibration_error()
    assert "mm/px" in w.result_validation_label.text()


def test_resolution_legacy_preferences_convert_dpi(qapp, isolated_settings):
    isolated_settings.setValue("analyse_dpi_mode", "manual")
    isolated_settings.setValue("analyse_manual_dpi", 254)
    w = _make_window(qapp)
    assert w.analyse_dpi_mode == "numeric"
    assert float(w.pixel_length.text()) == pytest.approx(0.1)
    assert w._effective_dpi() == pytest.approx(254)


def test_uncalibrated_profile_offsets_use_pixels(qapp, isolated_settings):
    w = _make_window(qapp)
    w.image = np.zeros((200, 200))
    w.laser_center = Point(100, 100)
    w._set_analyse_dpi_mode("none")
    w.radiation_profile_distance_mm = 20
    assert w._auto_radiation_profile_positions() == dict(top=80, bottom=120, left=80, right=120)


def test_resolution_change_invalidates_review_and_save(qapp, isolated_settings, monkeypatch):
    monkeypatch.setattr(app_module.QMessageBox, "warning", lambda *args: None)
    w = _loaded_navigation_window(qapp)
    w.detect_radiation_field()
    assert w.save_button.isEnabled()
    w._set_analyse_dpi_mode("numeric")
    w.pixel_length.setText("0.1")
    assert not w.save_button.isEnabled()
    w.detect_radiation_field()
    assert w.save_button.isEnabled()
    w.pixel_length.setText("nan")
    assert not w.save_button.isEnabled()
    w.save_current_result()
    assert fetch_analysis_rows(w.connection) == []


def test_ruler_uses_pixel_units_without_resolution(qapp, isolated_settings):
    w = _make_window(qapp)
    w.image = np.zeros((20, 20))
    w.tool_mode = "ruler"
    w._set_analyse_dpi_mode("none")
    w._on_ruler_changed((Point(0, 0), Point(3, 4)))
    assert "5.0 px" in w.tool_result_label.text()
    w._set_analyse_dpi_mode("numeric")
    w.pixel_length.setText("0.1")
    assert "0.5 mm" in w.tool_result_label.text()


def test_legacy_image_mode_maps_to_tag(qapp, isolated_settings):
    isolated_settings.setValue("analyse_dpi_mode", "image")
    w = _make_window(qapp)
    assert w.calibration_method.currentData() == "tag"
    assert w._effective_dpi() == 0


def test_resolution_card_follows_laser(qapp, isolated_settings):
    w = _make_window(qapp)
    layout = w.stage_frames["laser"].parentWidget().layout()
    assert layout.indexOf(w.calibration_method.parentWidget()) == layout.indexOf(w.stage_frames["laser"]) + 1


def test_square_image_and_profiles_resize_without_changing_analysis(qapp, isolated_settings):
    w = _loaded_navigation_window(qapp)
    w.detect_radiation_field()
    radiation, light, dpi = w.radiation_rect, w.light_rect, w._effective_dpi()
    w.view.scale(1.5, 1.5)
    scale = w.view.transform().m11()
    w.show()
    for width, height in ((1000, 740), (1300, 700), (900, 1000)):
        w.resize(width, height)
        qapp.processEvents()
        assert w.view.width() == w.view.height()
        assert w.square_image.rect().contains(w.view.geometry())
        for profile in w.profile_plots.values():
            assert w.square_image.rect().contains(profile.geometry())
        assert w.top_profile_plot.width() == w.view.width()
        assert w.left_profile_plot.height() == w.view.height()
        assert w.radiation_rect == radiation and w.light_rect == light
        assert w._effective_dpi() == dpi
        assert w.view.transform().m11() == scale
    w.analysis_splitter.setSizes([450, 450])
    qapp.processEvents()
    assert w.view.width() == w.view.height()
    assert w.square_image.rect().contains(w.view.geometry())
    w.close()


def test_resolution_stage_preserves_analysis_and_image_info(qapp, isolated_settings):
    w = _loaded_navigation_window(qapp)
    w.detect_radiation_field()
    radiation = w.radiation_rect
    light = w.light_rect
    dpi = w._effective_dpi()
    w.show()
    qapp.processEvents()
    geometry = {key: frame.geometry() for key, frame in w.stage_frames.items()}
    w.go_to_stage("calibration")
    qapp.processEvents()
    assert w.current_stage == "calibration"
    assert w.radiation_rect == radiation
    assert w.light_rect == light
    assert w._effective_dpi() == dpi
    assert {key: frame.geometry() for key, frame in w.stage_frames.items()} == geometry
    w.status.showMessage("検出完了")
    qapp.processEvents()
    assert w.image_info_label.isVisible()
    assert w.image_info_label.parentWidget() is w.status
    w.close()


def test_numeric_resolution_survives_image_loading(qapp, isolated_settings, monkeypatch):
    monkeypatch.setattr(app_module, "load_image", lambda path: np.zeros((30, 30)))
    monkeypatch.setattr(app_module, "tiff_image_dpi", lambda path: None)
    w = _make_window(qapp)
    w._set_analyse_dpi_mode("numeric")
    w.pixel_length.setText("0.1")
    w.load_path(Path("first.tif"))
    w.load_path(Path("second.tif"))
    assert w.pixel_length.text() == "0.1"
    assert w._effective_dpi() == pytest.approx(254)
    w._set_analyse_dpi_mode("tag")
    assert "解像度タグ" in w.result_validation_label.text()
    assert "照射野" in w.result_validation_label.text()
