from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


class ProfilePlot(QWidget):
    def __init__(self, orientation: str) -> None:
        super().__init__()
        if orientation not in ("vertical", "horizontal"):
            raise ValueError(f"Unknown orientation: {orientation}")
        self.orientation = orientation
        self.values: np.ndarray | None = None
        self.positions: np.ndarray | None = None
        self.edge_positions: tuple[float, ...] = ()
        self.visible_range: tuple[float, float] | None = None
        self.cursors: dict[str, float] = {}
        self.active_cursor: str | None = None
        self.selected = False
        self.on_cursor_moved: Callable[[str, float], None] | None = None
        self.on_selected: Callable[[], None] | None = None
        if self.orientation == "vertical":
            self.setFixedWidth(40)
        else:
            self.setFixedHeight(40)
        self.setMinimumSize(self.sizeHint())

    def sizeHint(self) -> QSize:
        if self.orientation == "vertical":
            return QSize(40, 320)
        return QSize(320, 40)

    def set_values(self, values: np.ndarray | None) -> None:
        self.values = values
        if values is None:
            self.positions = None
            self.edge_positions = ()
        else:
            self.positions = np.linspace(0.0, float(values.size - 1), values.size)
        self.update()

    def set_profile(
        self,
        values: np.ndarray | None,
        positions: np.ndarray | None,
        edge_positions: tuple[float, ...],
    ) -> None:
        self.values = values
        self.positions = positions
        self.edge_positions = edge_positions
        self.update()

    def set_visible_range(self, visible_range: tuple[float, float] | None) -> None:
        self.visible_range = visible_range
        self.update()

    def set_cursors(self, cursors: dict[str, float]) -> None:
        self.cursors = cursors
        if self.active_cursor not in self.cursors:
            self.active_cursor = None
        self.update()

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(28, 28, 28))
        painter.setRenderHint(QPainter.Antialiasing, False)
        self._draw_border(painter)

        values = self.values
        positions = self.positions
        if values is None or positions is None or values.size < 2 or positions.size != values.size:
            return

        visible_min, visible_max = self._plot_range(positions)
        visible_mask = (positions >= visible_min) & (positions <= visible_max)
        if np.count_nonzero(visible_mask) < 2:
            return

        visible_values = values[visible_mask]
        minimum = float(np.min(visible_values))
        maximum = float(np.max(visible_values))
        span = maximum - minimum
        if span <= 0:
            normalized = np.full(values.shape, 0.5, dtype=np.float64)
        else:
            normalized = (values.astype(np.float64) - minimum) / span

        width = max(1, self.width() - 1)
        height = max(1, self.height() - 1)
        points = QPolygonF()
        for position_value, value in zip(positions, normalized):
            if position_value < visible_min or position_value > visible_max:
                continue
            position = (position_value - visible_min) / (visible_max - visible_min)
            if self.orientation == "vertical":
                x = value * width
                y = position * height
            else:
                x = position * width
                y = height - value * height
            points.append(QPointF(x, y))

        painter.setPen(QPen(QColor(0, 220, 255), 1))
        painter.drawPolyline(points)
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        if self.orientation == "vertical":
            painter.drawLine(0, 0, 0, height)
            painter.drawLine(width, 0, width, height)
        else:
            painter.drawLine(0, 0, width, 0)
            painter.drawLine(0, height, width, height)

        painter.setPen(QPen(QColor(255, 210, 0), 1))
        for edge_position in self.edge_positions:
            if edge_position < visible_min or edge_position > visible_max:
                continue
            position = (edge_position - visible_min) / (visible_max - visible_min)
            if self.orientation == "vertical":
                y = round(position * height)
                painter.drawLine(0, y, width, y)
            else:
                x = round(position * width)
                painter.drawLine(x, 0, x, height)

        for name, cursor_position in self.cursors.items():
            if cursor_position < visible_min or cursor_position > visible_max:
                continue
            ratio = (cursor_position - visible_min) / (visible_max - visible_min)
            color = QColor(255, 120, 0) if name == self.active_cursor else QColor(255, 255, 255)
            painter.setPen(QPen(color, 2))
            if self.orientation == "vertical":
                y = round(ratio * height)
                painter.drawLine(0, y, width, y)
            else:
                x = round(ratio * width)
                painter.drawLine(x, 0, x, height)

        self._draw_border(painter)

    def _plot_range(self, positions: np.ndarray) -> tuple[float, float]:
        if self.visible_range is None:
            return float(np.min(positions)), float(np.max(positions))
        start, end = self.visible_range
        minimum = max(float(np.min(positions)), min(start, end))
        maximum = min(float(np.max(positions)), max(start, end))
        if maximum <= minimum:
            return float(np.min(positions)), float(np.max(positions))
        return minimum, maximum

    def mousePressEvent(self, event) -> None:
        if self.on_selected is not None:
            self.on_selected()
        cursor = self._hit_cursor(event.position())
        if cursor is not None:
            self.active_cursor = cursor
            self._move_cursor(cursor, event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.active_cursor is None:
            super().mouseMoveEvent(event)
            return
        self._move_cursor(self.active_cursor, event.position())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self.active_cursor = None
        self.update()
        event.accept()

    def _hit_cursor(self, position: QPointF) -> str | None:
        if not self.cursors:
            return None
        visible_min, visible_max = self._current_visible_range()
        span = visible_max - visible_min
        if span <= 0:
            return None
        length = max(1, self.height() - 1 if self.orientation == "vertical" else self.width() - 1)
        coordinate = position.y() if self.orientation == "vertical" else position.x()
        tolerance = 8.0
        best_name = None
        best_distance = tolerance
        for name, cursor_position in self.cursors.items():
            pixel = (cursor_position - visible_min) * length / span
            distance = abs(coordinate - pixel)
            if distance <= best_distance:
                best_name = name
                best_distance = distance
        return best_name

    def _move_cursor(self, name: str, position: QPointF) -> None:
        visible_min, visible_max = self._current_visible_range()
        length = max(1, self.height() - 1 if self.orientation == "vertical" else self.width() - 1)
        coordinate = position.y() if self.orientation == "vertical" else position.x()
        ratio = max(0.0, min(1.0, coordinate / length))
        value = visible_min + ratio * (visible_max - visible_min)
        self.cursors[name] = value
        if self.on_cursor_moved is not None:
            self.on_cursor_moved(name, value)
        self.update()

    def _current_visible_range(self) -> tuple[float, float]:
        if self.positions is None or self.positions.size == 0:
            if self.visible_range is not None:
                return min(self.visible_range), max(self.visible_range)
            return 0.0, 1.0
        return self._plot_range(self.positions)

    def _draw_border(self, painter: QPainter) -> None:
        color = QColor(0, 210, 255) if self.selected else QColor(80, 80, 80)
        width = 2 if self.selected else 1
        painter.setPen(QPen(color, width))
        painter.drawRect(0, 0, max(0, self.width() - 1), max(0, self.height() - 1))
