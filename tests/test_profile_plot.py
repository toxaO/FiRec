import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor

from firec.gui.profile_plot import ProfilePlot
from firec.gui.image_view import PROFILE_LINE_COLOR


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_profile_overlays_render_and_toggle_independently(qapp, orientation):
    plot = ProfilePlot(orientation)
    plot.resize(101, 101)
    plot.set_profile(np.arange(101, dtype=float), np.arange(101, dtype=float), (20, 80), reference_value=50)
    plot.set_raw_profile_visible(False)
    plot.set_smoothed_profile_visible(False)
    plot.set_cross_profile_positions((30, 70))
    plot.set_cursors({"L1": 50})
    length = plot.width() - 1 if orientation == "horizontal" else plot.height() - 1

    def pixel(position, across=8):
        image = plot.grab().toImage()
        coordinate = round(position * length / 100)
        return image.pixelColor(coordinate, across) if orientation == "horizontal" else image.pixelColor(across, coordinate)

    background = QColor(28, 28, 28)
    assert pixel(30) == PROFILE_LINE_COLOR
    assert pixel(70) == PROFILE_LINE_COLOR
    assert pixel(40).green() > background.green()
    assert pixel(10) == background
    assert pixel(50) == QColor(255, 255, 255)
    plot.set_cross_profiles_visible(False)
    assert pixel(30) == pixel(40)
    plot.set_average_range_visible(False)
    assert pixel(40) == background
    plot.set_boundary_cursors_visible(False)
    assert pixel(50) == background
    assert plot.cursors == {"L1": 50}
    assert plot._hit_cursor(QPointF(plot.width() / 2, plot.height() / 2)) is None
    # Threshold crosses the full profile independently of the position markers.
    point = (8, round((plot.height() - 1) / 2)) if orientation == "horizontal" else (round((plot.width() - 1) / 2), 8)
    assert plot.grab().toImage().pixelColor(*point) == QColor(255, 170, 0)
    plot.set_threshold_visible(False)
    assert plot.grab().toImage().pixelColor(*point) == background
    plot.close()
