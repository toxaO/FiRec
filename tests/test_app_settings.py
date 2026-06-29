import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from firec.gui import app as app_module
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
    window.display_option_checks["Radiation Center"].setChecked(False)
    window.display_option_checks["Light Edge"].setChecked(False)
    window.settings.sync()
    window.close()

    restored = _make_window(qapp)

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
    assert restored.display_option_checks["Radiation Center"].isChecked() is False
    assert restored.display_option_checks["Light Edge"].isChecked() is False


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
    save_analysis(window.connection, "image.tif", result, "laser", 0.0)

    window.refresh_results_table()

    origin_widget = window.results_table.cellWidget(0, 2)
    assert origin_widget is not None
    origin_widget.setCurrentIndex(origin_widget.findData("radiation"))

    rows = fetch_analysis_rows(window.connection)
    assert rows[0]["origin"] == "radiation"
    assert window.results_table.item(0, 7).text() == "0.0"
    assert window.results_table.item(0, 5).text() == "-10.0"

    dpi_widget = window.results_table.cellWidget(0, 3)
    assert dpi_widget is not None
    dpi_widget.setValue(254.0)

    rows = fetch_analysis_rows(window.connection)
    assert rows[0]["dpi"] == 254.0
    assert window.results_table.item(0, 4).text() == "mm"
    assert window.results_table.item(0, 5).text() == "-1.0"
    assert window.results_table.item(0, 7).text() == "0.0"


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
