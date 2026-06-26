from __future__ import annotations

from collections.abc import Callable
from math import hypot

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QKeyEvent, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QSizePolicy,
)

from firec.core.analysis import normalize_for_display
from firec.core.geometry import Point, RotatedRect


EDGE_NAMES = ("top", "right", "bottom", "left")
PROFILE_LINE_COLOR = QColor(45, 135, 220)
PROFILE_LINE_SELECTED_COLOR = QColor(0, 210, 255)
RADIATION_COLOR = QColor(60, 210, 100)
RADIATION_FILL_COLOR = QColor(80, 255, 120, 55)
LIGHT_EDGE_COLOR = QColor(255, 145, 205)
LIGHT_EDGE_SELECTED_COLOR = QColor(220, 0, 0)
POINT_COLOR = QColor(255, 120, 0)


class ImageView(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setBackgroundBrush(QColor(214, 214, 214))
        self.setStyleSheet("QGraphicsView { border: 1px solid #8c8c8c; }")

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._radiation_item: QGraphicsPolygonItem | None = None
        self._light_item: QGraphicsPolygonItem | None = None
        self._profile_line_items: list[QGraphicsItem] = []
        self._profile_cursor_point_items: list[QGraphicsItem] = []
        self._point_items: list[QGraphicsItem] = []
        self._center_point_items: list[QGraphicsItem] = []
        self._vertex_items: list[QGraphicsItem] = []
        self._edge_length_items: list[QGraphicsItem] = []
        self._laser_center_items: list[QGraphicsItem] = []
        self._circle_overlay_items: list[QGraphicsItem] = []
        self._target_items: list[QGraphicsItem] = []
        self._handle_items: list[QGraphicsItem] = []
        self._image_shape: tuple[int, int] | None = None
        self._image_aspect_ratio: float | None = None
        self.radiation_rect: RotatedRect | None = None
        self.radiation_polygon: tuple[Point, Point, Point, Point] | None = None
        self.light_rect: RotatedRect | None = None
        self.light_polygon: tuple[Point, Point, Point, Point] | None = None
        self.radiation_points: dict[str, Point] = {}
        self.center_points: dict[str, Point] = {}
        self.laser_center: Point | None = None
        self.top_profile_y = 0.0
        self.bottom_profile_y = 0.0
        self.left_profile_x = 0.0
        self.right_profile_x = 0.0
        self.active_profile_orientation: str | None = None
        self.visible_profile_lines: set[str] | None = None
        self.profile_lines_visible = True
        self.profile_lines_editable = True
        self.profile_line_offsets = {}
        self.selected_profile_line: str | None = None
        self.active_field = "radiation"
        self.selected_target = "top"
        self.show_radiation_edges = True
        self.show_radiation_center = True
        self.show_radiation_area = True
        self.show_radiation_points = True
        self.show_radiation_edge_lengths = True
        self.show_radiation_vertices = True
        self.show_light_edges = True
        self.show_light_center = True
        self.show_light_edge_lengths = True
        self.show_light_vertices = True
        self.show_laser_center = True
        self.editing_enabled = True
        self.pan_enabled = False
        self.on_rect_changed: Callable[[str, RotatedRect], None] | None = None
        self.on_tab_navigation: Callable[[bool], None] | None = None
        self.on_profile_lines_changed: Callable[[dict[str, tuple[Point, Point]]], None] | None = None
        self.on_profile_line_selected: Callable[[str | None], None] | None = None
        self.on_visible_scene_rect_changed: Callable[[QRectF], None] | None = None

        self.horizontalScrollBar().valueChanged.connect(lambda value: self._emit_visible_scene_rect_changed())
        self.verticalScrollBar().valueChanged.connect(lambda value: self._emit_visible_scene_rect_changed())

        self._drag_mode: str | None = None
        self._drag_edge: str | None = None
        self._last_scene_pos: QPointF | None = None
        self._last_view_pos: QPointF | None = None

    def set_image(self, image: np.ndarray) -> None:
        display = normalize_for_display(image)
        height, width = display.shape
        qimage = QImage(display.data, width, height, display.strides[0], QImage.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.scene().clear()
        self._target_items = []
        self._handle_items = []
        self._profile_line_items = []
        self._profile_cursor_point_items = []
        self._point_items = []
        self._center_point_items = []
        self._vertex_items = []
        self._edge_length_items = []
        self._laser_center_items = []
        self._circle_overlay_items = []
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self._image_shape = (height, width)
        self._image_aspect_ratio = width / height if height > 0 else None
        self.radiation_rect = None
        self.radiation_polygon = None
        self.light_rect = None
        self.light_polygon = None
        self.radiation_points = {}
        self.center_points = {}
        self.laser_center = None
        self.top_profile_y = height * 0.25
        self.bottom_profile_y = height * 0.75
        self.left_profile_x = width * 0.25
        self.right_profile_x = width * 0.75
        self.active_profile_orientation = None
        self.visible_profile_lines = None
        self.profile_lines_visible = True
        self.profile_line_offsets = {}
        self.selected_profile_line = "top"
        self._radiation_item = None
        self._light_item = None
        self._emit_profile_lines_changed()
        self._emit_profile_line_selected()
        self.reset_view()
        self.updateGeometry()

    def set_profile_orientation(self, orientation: str | None) -> None:
        if orientation not in (None, "horizontal", "vertical"):
            raise ValueError(f"Unknown profile orientation: {orientation}")
        self.active_profile_orientation = orientation
        self._draw_profile_lines()
        self._emit_profile_lines_changed()

    def set_profile_lines_visible(self, visible: bool) -> None:
        self.profile_lines_visible = visible
        self._draw_profile_lines()

    def set_profile_lines_editable(self, editable: bool) -> None:
        self.profile_lines_editable = editable

    def set_visible_profile_lines(self, names: set[str] | None) -> None:
        self.visible_profile_lines = names
        self._draw_profile_lines()

    def set_profile_line_positions(
        self,
        top_y: float | None = None,
        bottom_y: float | None = None,
        left_x: float | None = None,
        right_x: float | None = None,
    ) -> None:
        if self._image_shape is None:
            return
        height, width = self._image_shape
        if top_y is not None:
            self.top_profile_y = max(0.0, min(height - 1.0, top_y))
        if bottom_y is not None:
            self.bottom_profile_y = max(0.0, min(height - 1.0, bottom_y))
        if left_x is not None:
            self.left_profile_x = max(0.0, min(width - 1.0, left_x))
        if right_x is not None:
            self.right_profile_x = max(0.0, min(width - 1.0, right_x))
        self._draw_profile_lines()
        self._emit_profile_lines_changed()

    def image_shape(self) -> tuple[int, int] | None:
        return self._image_shape

    def hasHeightForWidth(self) -> bool:
        return self._image_aspect_ratio is not None

    def heightForWidth(self, width: int) -> int:
        if self._image_aspect_ratio is None or self._image_aspect_ratio <= 0:
            return super().heightForWidth(width)
        return max(1, round(width / self._image_aspect_ratio))

    def sizeHint(self):  # type: ignore[override]
        if self._image_aspect_ratio is None:
            return super().sizeHint()
        width = 640
        height = max(1, round(width / self._image_aspect_ratio))
        return QSize(width, height)

    def select_profile_line(self, line_name: str | None) -> None:
        self._set_selected_profile_line(line_name)

    def set_profile_cursor_points(self, points: dict[str, Point]) -> None:
        self._clear_profile_cursor_point_items()
        for name, point in points.items():
            radius = 5
            dot = self.scene().addEllipse(
                point.x - radius,
                point.y - radius,
                radius * 2,
                radius * 2,
                QPen(Qt.white, 2),
                QBrush(POINT_COLOR),
            )
            label = self.scene().addText(name)
            label.setDefaultTextColor(QColor(255, 255, 255))
            label.setPos(point.x + 6, point.y + 6)
            dot.setZValue(13)
            label.setZValue(13)
            self._profile_cursor_point_items.extend([dot, label])

    def set_result_center_points(self, points: dict[str, Point]) -> None:
        self.center_points = points
        self._draw_center_points()

    def set_laser_center(self, point: Point | None) -> None:
        self.laser_center = point
        self._draw_laser_center()

    def set_circle_overlay(self, center: Point | None, radius: float | None) -> None:
        self._clear_circle_overlay_items()
        if center is None or radius is None or radius <= 0:
            return
        outline = self.scene().addEllipse(
            center.x - radius,
            center.y - radius,
            radius * 2.0,
            radius * 2.0,
            QPen(QColor(255, 170, 0), 2),
            QBrush(Qt.NoBrush),
        )
        horizontal = self.scene().addLine(center.x - radius, center.y, center.x + radius, center.y, QPen(QColor(255, 170, 0), 1))
        vertical = self.scene().addLine(center.x, center.y - radius, center.x, center.y + radius, QPen(QColor(255, 170, 0), 1))
        dot = self.scene().addEllipse(
            center.x - 3,
            center.y - 3,
            6,
            6,
            QPen(Qt.white, 1),
            QBrush(QColor(255, 170, 0)),
        )
        label = self.scene().addText("X,Y,R")
        label.setDefaultTextColor(QColor(255, 170, 0))
        label.setPos(center.x + radius + 4, center.y + radius + 4)
        for item in (outline, horizontal, vertical, dot, label):
            item.setZValue(16)
            self._circle_overlay_items.append(item)

    def _draw_center_points(self) -> None:
        self._clear_center_point_items()
        for name, point in self.center_points.items():
            if name == "radiation" and not self.show_radiation_center:
                continue
            if name == "light" and not self.show_light_center:
                continue
            if name == "laser" and not self.show_laser_center:
                continue
            radius = 5
            dot = self.scene().addEllipse(
                point.x - radius,
                point.y - radius,
                radius * 2,
                radius * 2,
                QPen(Qt.white, 1),
                QBrush(LIGHT_EDGE_SELECTED_COLOR if name == "light" else RADIATION_COLOR),
            )
            label = self.scene().addText(name)
            label.setDefaultTextColor(QColor(255, 255, 255))
            label.setPos(point.x + 6, point.y + 6)
            dot.setZValue(14)
            label.setZValue(14)
            self._center_point_items.extend([dot, label])

    def set_radiation_points(self, points: dict[str, Point]) -> None:
        self.radiation_points = points
        self._draw_radiation_points()

    def set_show_radiation_edges(self, show: bool) -> None:
        self.show_radiation_edges = show
        self._sync_item_visibility()

    def set_show_radiation_center(self, show: bool) -> None:
        self.show_radiation_center = show
        self._draw_center_points()

    def set_show_radiation_area(self, show: bool) -> None:
        self.show_radiation_area = show
        self._sync_item_visibility()

    def set_show_radiation_points(self, show: bool) -> None:
        self.show_radiation_points = show
        self._draw_radiation_points()

    def set_show_radiation_edge_lengths(self, show: bool) -> None:
        self.show_radiation_edge_lengths = show
        self._draw_edge_length_lines()

    def set_show_radiation_vertices(self, show: bool) -> None:
        self.show_radiation_vertices = show
        self._draw_vertices()

    def set_show_light_edges(self, show: bool) -> None:
        self.show_light_edges = show
        self._sync_item_visibility()
        self._draw_selection_overlays()

    def set_show_light_center(self, show: bool) -> None:
        self.show_light_center = show
        self._draw_center_points()

    def set_show_light_edge_lengths(self, show: bool) -> None:
        self.show_light_edge_lengths = show
        self._draw_edge_length_lines()

    def set_show_light_vertices(self, show: bool) -> None:
        self.show_light_vertices = show
        self._draw_vertices()

    def set_show_laser_center(self, show: bool) -> None:
        self.show_laser_center = show
        self._draw_laser_center()

    def set_radiation_rect(self, rect: RotatedRect | None, reset_profile_lines: bool = True) -> None:
        self.radiation_rect = rect
        self.radiation_polygon = None if rect is None else rect.ordered_points()
        if rect is None:
            if self._radiation_item is not None:
                self.scene().removeItem(self._radiation_item)
                self._radiation_item = None
            self._draw_profile_lines()
            self._emit_profile_lines_changed()
            self._draw_edge_length_lines()
            self._draw_vertices()
            return
        if reset_profile_lines:
            self.profile_line_offsets = {}
        self._radiation_item = self._set_polygon_item(
            self._radiation_item,
            rect,
            _dashed_pen(RADIATION_COLOR),
            QBrush(Qt.NoBrush),
        )
        self._sync_item_visibility()
        self._draw_profile_lines()
        self._emit_profile_lines_changed()
        self._draw_edge_length_lines()
        self._draw_vertices()

    def set_radiation_polygon(
        self,
        points: tuple[Point, Point, Point, Point],
        reset_profile_lines: bool = True,
        emit_profile_lines: bool = True,
    ) -> None:
        self.radiation_polygon = points
        if reset_profile_lines:
            self.profile_line_offsets = {}
        self._radiation_item = self._set_polygon_points_item(
            self._radiation_item,
            points,
            _dashed_pen(RADIATION_COLOR),
            QBrush(Qt.NoBrush),
        )
        self._sync_item_visibility()
        self._draw_profile_lines()
        if emit_profile_lines:
            self._emit_profile_lines_changed()
        self._draw_edge_length_lines()
        self._draw_vertices()

    def set_light_rect(self, rect: RotatedRect | None) -> None:
        self.light_rect = rect
        self.light_polygon = None if rect is None else rect.ordered_points()
        if rect is None:
            if self._light_item is not None:
                self.scene().removeItem(self._light_item)
                self._light_item = None
            self._draw_selection_overlays()
            self._draw_edge_length_lines()
            self._draw_vertices()
            return
        self._light_item = self._set_polygon_item(
            self._light_item,
            rect,
            _dashed_pen(LIGHT_EDGE_COLOR, 2, [3, 3]),
            QBrush(Qt.NoBrush),
        )
        self._sync_item_visibility()
        self._draw_selection_overlays()
        self._draw_edge_length_lines()
        self._draw_vertices()

    def set_light_polygon(self, points: tuple[Point, Point, Point, Point]) -> None:
        self.light_polygon = points
        self._light_item = self._set_polygon_points_item(
            self._light_item,
            points,
            _dashed_pen(LIGHT_EDGE_COLOR, 2, [3, 3]),
            QBrush(Qt.NoBrush),
        )
        self._sync_item_visibility()
        self._draw_selection_overlays()
        self._draw_edge_length_lines()
        self._draw_vertices()

    def set_editing_enabled(self, enabled: bool) -> None:
        self.editing_enabled = enabled
        self._draw_selection_overlays()

    def set_active_field(self, field: str) -> None:
        if field not in ("laser", "radiation", "light"):
            raise ValueError(f"Unknown field: {field}")
        self.active_field = field
        if self.selected_target not in EDGE_NAMES:
            self.selected_target = "top"
        self._sync_item_visibility()
        self._draw_selection_overlays()

    def select_next_target(self) -> str:
        targets = EDGE_NAMES
        if self.selected_target not in targets:
            self.selected_target = "top"
        else:
            index = targets.index(self.selected_target)
            self.selected_target = targets[(index + 1) % len(targets)]
        self._draw_selection_overlays()
        return self.selected_target

    def select_target(self, target: str) -> None:
        if target not in EDGE_NAMES:
            raise ValueError(f"Unknown target: {target}")
        self.selected_target = target
        self._draw_selection_overlays()

    def reset_view(self) -> None:
        if self._pixmap_item is None:
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._emit_visible_scene_rect_changed()

    def set_pan_enabled(self, enabled: bool) -> None:
        self.pan_enabled = enabled
        self.setCursor(Qt.OpenHandCursor if enabled else Qt.ArrowCursor)

    def zoom_in(self) -> None:
        self._zoom_by(1.1)

    def zoom_out(self) -> None:
        self._zoom_by(1.0 / 1.1)

    def _zoom_by(self, factor: float) -> None:
        if self._pixmap_item is None:
            return
        self.scale(factor, factor)
        self._emit_visible_scene_rect_changed()

    def wheelEvent(self, event) -> None:
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._emit_visible_scene_rect_changed()

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_view()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        profile_line = (
            self._hit_profile_line(scene_pos)
            if self.editing_enabled and self.profile_lines_editable and self.active_field in ("laser", "radiation")
            else None
        )
        target = self._hit_active_target(scene_pos) if self.editing_enabled and self.active_field == "light" else None
        self._last_scene_pos = scene_pos
        self._last_view_pos = event.position()
        if profile_line is not None:
            self._drag_mode = "profile"
            self._drag_edge = profile_line
            self._set_selected_profile_line(profile_line)
        elif target is not None:
            self._drag_mode = "target"
            self._drag_edge = target
            self._set_selected_profile_line(None)
            self.select_target(target)
        elif self.pan_enabled:
            self._drag_mode = "pan"
            self._drag_edge = None
            self.setCursor(Qt.ClosedHandCursor)
        else:
            self._drag_mode = None
            self._drag_edge = None
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_mode is None or self._last_scene_pos is None or self._last_view_pos is None:
            super().mouseMoveEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        active_rect = self._active_rect()
        if (
            self.editing_enabled
            and self._drag_mode == "profile"
            and self._drag_edge is not None
        ):
            dx = scene_pos.x() - self._last_scene_pos.x()
            dy = scene_pos.y() - self._last_scene_pos.y()
            self._move_profile_line(self._drag_edge, dx, dy)
            self._last_scene_pos = scene_pos
        elif (
            self.editing_enabled
            and self._drag_mode == "target"
            and active_rect is not None
            and self._drag_edge is not None
        ):
            dx = scene_pos.x() - self._last_scene_pos.x()
            dy = scene_pos.y() - self._last_scene_pos.y()
            dx, dy = self._constrained_light_delta(self._drag_edge, dx, dy)
            self._update_active_rect(self._move_target(active_rect, self._drag_edge, dx, dy))
            self._last_scene_pos = scene_pos
        elif self._drag_mode == "pan":
            delta = event.position() - self._last_view_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            self._last_view_pos = event.position()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_mode = None
        self._drag_edge = None
        self._last_scene_pos = None
        self._last_view_pos = None
        if self.pan_enabled:
            self.setCursor(Qt.OpenHandCursor)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Tab, Qt.Key_Backtab):
            if self.on_tab_navigation is not None:
                self.on_tab_navigation(bool(event.modifiers() & Qt.ShiftModifier) or event.key() == Qt.Key_Backtab)
            event.accept()
            return
        if event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            if not self.editing_enabled:
                event.accept()
                return
            if self.selected_profile_line is not None and self.profile_lines_editable:
                self._nudge_selected_profile_line(event)
                event.accept()
                return
            if self.active_field == "radiation" and not self.profile_lines_editable:
                event.accept()
                return
            self._nudge_selected_target(event)
            event.accept()
            return
        super().keyPressEvent(event)

    def _nudge_selected_profile_line(self, event: QKeyEvent) -> None:
        if self.selected_profile_line is None or not self.profile_lines_editable:
            return
        step = 10.0 if event.modifiers() & Qt.ShiftModifier else 1.0
        dx = dy = 0.0
        if event.key() == Qt.Key_Left:
            dx = -step
        elif event.key() == Qt.Key_Right:
            dx = step
        elif event.key() == Qt.Key_Up:
            dy = -step
        elif event.key() == Qt.Key_Down:
            dy = step
        self._move_profile_line(self.selected_profile_line, dx, dy)

    def _nudge_selected_target(self, event: QKeyEvent) -> None:
        active_rect = self._active_rect()
        if active_rect is None or (self.active_field == "radiation" and not self.profile_lines_editable):
            return
        step = 10.0 if event.modifiers() & Qt.ShiftModifier else 1.0
        dx = dy = 0.0
        if event.key() == Qt.Key_Left:
            dx = -step
        elif event.key() == Qt.Key_Right:
            dx = step
        elif event.key() == Qt.Key_Up:
            dy = -step
        elif event.key() == Qt.Key_Down:
            dy = step

        if self.active_field == "radiation" and event.modifiers() & Qt.ShiftModifier:
            if event.key() in (Qt.Key_Up, Qt.Key_Down):
                size_delta = 1.0 if event.key() == Qt.Key_Up else -1.0
                self._update_active_rect(active_rect.scaled(size_delta))
            return

        if self._nudge_profile_line_for_selected_edge(event, dx, dy):
            return

        dx, dy = self._constrained_light_delta(self.selected_target, dx, dy)
        self._update_active_rect(self._move_target(active_rect, self.selected_target, dx, dy))

    def _nudge_profile_line_for_selected_edge(self, event: QKeyEvent, dx: float, dy: float) -> bool:
        return False

    def _move_target(self, rect: RotatedRect, target: str, dx: float, dy: float) -> RotatedRect:
        if self.active_field == "light" and self.light_polygon is not None:
            self.light_polygon = self._move_light_polygon_target(target, dx, dy)
        if target in EDGE_NAMES:
            return rect.moved_edge_by_vector(self._visual_edge_map(rect)[target], dx, dy)
        raise ValueError(f"Unknown target: {target}")

    def move_active_rect(self, dx: float, dy: float) -> None:
        active_rect = self._active_rect()
        if active_rect is not None:
            self._update_active_rect(active_rect.moved(dx, dy))

    def rotate_active_rect(self, delta_angle: float) -> None:
        return

    def scale_active_rect(self, delta: float) -> None:
        active_rect = self._active_rect()
        if active_rect is not None:
            self._update_active_rect(active_rect.scaled(delta))

    def _active_rect(self) -> RotatedRect | None:
        if self.active_field == "radiation":
            return self.radiation_rect
        if self.active_field == "laser":
            return None
        return self.light_rect

    def _update_active_rect(self, rect: RotatedRect) -> None:
        if self.active_field == "radiation":
            self.set_radiation_rect(rect, reset_profile_lines=False)
        else:
            light_polygon = self.light_polygon
            self.set_light_rect(rect)
            if light_polygon is not None:
                self.set_light_polygon(light_polygon)
        if self.on_rect_changed is not None:
            self.on_rect_changed(self.active_field, rect)
        self._draw_selection_overlays()

    def _set_polygon_item(
        self,
        item: QGraphicsPolygonItem | None,
        rect: RotatedRect,
        pen: QPen,
        brush: QBrush,
    ) -> QGraphicsPolygonItem:
        return self._set_polygon_points_item(item, rect.ordered_points(), pen, brush)

    def _set_polygon_points_item(
        self,
        item: QGraphicsPolygonItem | None,
        points: tuple[Point, Point, Point, Point],
        pen: QPen,
        brush: QBrush,
    ) -> QGraphicsPolygonItem:
        polygon = QPolygonF([QPointF(point.x, point.y) for point in points])
        if item is None:
            item = self.scene().addPolygon(polygon, pen, brush)
        else:
            item.setPolygon(polygon)
            item.setPen(pen)
            item.setBrush(brush)
        return item

    def _draw_selection_overlays(self) -> None:
        self._clear_overlay_items()
        if not self.editing_enabled:
            return
        if self.selected_profile_line is not None:
            return
        rect = self._active_rect()
        if rect is None:
            return
        if self.active_field != "light":
            return
        if not self.show_light_edges:
            return
        if self.selected_target in EDGE_NAMES:
            edge_points = self._edge_points(rect, self.selected_target)
            if edge_points is None:
                return
            start, end = edge_points
            item = self.scene().addLine(
                start.x,
                start.y,
                end.x,
                end.y,
                _dashed_pen(LIGHT_EDGE_SELECTED_COLOR, 3, [3, 3]),
            )
            self._target_items.append(item)

    def _sync_item_visibility(self) -> None:
        if self._radiation_item is not None:
            visible = self.show_radiation_edges or self.show_radiation_area
            self._radiation_item.setVisible(visible)
            self._radiation_item.setPen(_dashed_pen(RADIATION_COLOR) if self.show_radiation_edges else QPen(Qt.NoPen))
            self._radiation_item.setBrush(QBrush(RADIATION_FILL_COLOR) if self.show_radiation_area else QBrush(Qt.NoBrush))
            self._radiation_item.setOpacity(1.0)
        if self._light_item is not None:
            self._light_item.setVisible(self.active_field == "light" and self.show_light_edges)
        self._draw_edge_length_lines()
        self._draw_vertices()

    def _clear_overlay_items(self) -> None:
        for item in self._target_items + self._handle_items:
            self.scene().removeItem(item)
        self._target_items = []
        self._handle_items = []

    def _draw_profile_lines(self) -> None:
        self._clear_profile_line_items()
        if not self.profile_lines_visible:
            return
        lines = self.profile_lines()
        for name, points in lines.items():
            if self.visible_profile_lines is not None and name not in self.visible_profile_lines:
                continue
            if self.active_profile_orientation is not None and name != self.active_profile_orientation:
                continue
            start, end = points
            width = 3 if name == self.selected_profile_line else 2
            color = PROFILE_LINE_SELECTED_COLOR if name == self.selected_profile_line else PROFILE_LINE_COLOR
            item = self.scene().addLine(start.x, start.y, end.x, end.y, _dashed_pen(color, width))
            item.setZValue(10)
            self._profile_line_items.append(item)

    def _clear_profile_line_items(self) -> None:
        for item in self._profile_line_items:
            self.scene().removeItem(item)
        self._profile_line_items = []

    def _clear_profile_cursor_point_items(self) -> None:
        for item in self._profile_cursor_point_items:
            self.scene().removeItem(item)
        self._profile_cursor_point_items = []

    def _clear_center_point_items(self) -> None:
        for item in self._center_point_items:
            self.scene().removeItem(item)
        self._center_point_items = []

    def _clear_vertex_items(self) -> None:
        for item in self._vertex_items:
            self.scene().removeItem(item)
        self._vertex_items = []

    def _clear_edge_length_items(self) -> None:
        for item in self._edge_length_items:
            self.scene().removeItem(item)
        self._edge_length_items = []

    def _clear_laser_center_items(self) -> None:
        for item in self._laser_center_items:
            self.scene().removeItem(item)
        self._laser_center_items = []

    def _clear_circle_overlay_items(self) -> None:
        for item in self._circle_overlay_items:
            self.scene().removeItem(item)
        self._circle_overlay_items = []

    def profile_lines(self) -> dict[str, tuple[Point, Point]]:
        if self._image_shape is None:
            return {}
        height, width = self._image_shape
        return {
            "top": (Point(0.0, self.top_profile_y), Point(width - 1.0, self.top_profile_y)),
            "bottom": (Point(0.0, self.bottom_profile_y), Point(width - 1.0, self.bottom_profile_y)),
            "left": (Point(self.left_profile_x, 0.0), Point(self.left_profile_x, height - 1.0)),
            "right": (Point(self.right_profile_x, 0.0), Point(self.right_profile_x, height - 1.0)),
        }

    def _offset_line(self, start: Point, end: Point, offset: QPointF) -> tuple[Point, Point]:
        return (
            Point(start.x + offset.x(), start.y + offset.y()),
            Point(end.x + offset.x(), end.y + offset.y()),
        )

    def _extend_line_to_image(self, line: tuple[Point, Point], width: int, height: int) -> tuple[Point, Point]:
        start, end = line
        dx = end.x - start.x
        dy = end.y - start.y
        intersections: list[Point] = []
        if dx != 0:
            for x in (0.0, float(width - 1)):
                t = (x - start.x) / dx
                y = start.y + t * dy
                if 0.0 <= y <= height - 1:
                    _append_unique_point(intersections, Point(x, y))
        if dy != 0:
            for y in (0.0, float(height - 1)):
                t = (y - start.y) / dy
                x = start.x + t * dx
                if 0.0 <= x <= width - 1:
                    _append_unique_point(intersections, Point(x, y))
        if len(intersections) < 2:
            return line
        return intersections[0], intersections[-1]

    def _move_profile_line(self, name: str, dx: float, dy: float) -> None:
        if self._image_shape is None:
            return
        height, width = self._image_shape
        if name == "top":
            self.top_profile_y = max(0.0, min(height - 1.0, self.top_profile_y + dy))
        elif name == "bottom":
            self.bottom_profile_y = max(0.0, min(height - 1.0, self.bottom_profile_y + dy))
        elif name == "left":
            self.left_profile_x = max(0.0, min(width - 1.0, self.left_profile_x + dx))
        elif name == "right":
            self.right_profile_x = max(0.0, min(width - 1.0, self.right_profile_x + dx))
        else:
            return
        self._draw_profile_lines()
        self._emit_profile_lines_changed()

    def _set_selected_profile_line(self, line_name: str | None) -> None:
        if line_name is not None and line_name not in self.profile_lines():
            return
        if self.selected_profile_line == line_name:
            return
        self.selected_profile_line = line_name
        self._clear_overlay_items()
        self._draw_profile_lines()
        self._emit_profile_line_selected()

    def _draw_edge_length_lines(self) -> None:
        self._clear_edge_length_items()
        fields = (
            ("radiation", self.radiation_polygon, RADIATION_COLOR, self.show_radiation_edge_lengths),
            ("light", self.light_polygon, LIGHT_EDGE_COLOR, self.show_light_edge_lengths),
        )
        for name, points, color, visible in fields:
            if points is None or not visible:
                continue
            if name == "light" and self.active_field != "light":
                continue
            left_mid, right_mid, top_mid, bottom_mid = _edge_length_points(points)
            for start, end in ((left_mid, right_mid), (top_mid, bottom_mid)):
                item = self.scene().addLine(start.x, start.y, end.x, end.y, _dashed_pen(color, 1, [5, 4]))
                item.setZValue(11)
                self._edge_length_items.append(item)

    def _draw_vertices(self) -> None:
        self._clear_vertex_items()
        fields = (
            ("radiation", self.radiation_polygon, RADIATION_COLOR, self.show_radiation_vertices),
            ("light", self.light_polygon, LIGHT_EDGE_COLOR, self.show_light_vertices),
        )
        vertex_labels = ("TL", "TR", "BR", "BL")
        for name, points, color, visible in fields:
            if points is None or not visible:
                continue
            if name == "light" and self.active_field != "light":
                continue
            for label_text, point in zip(vertex_labels, points, strict=True):
                radius = 3
                dot = self.scene().addEllipse(
                    point.x - radius,
                    point.y - radius,
                    radius * 2,
                    radius * 2,
                    QPen(color, 1),
                    QBrush(color),
                )
                label = self.scene().addText(f"{name[0].upper()}{label_text}")
                label.setDefaultTextColor(color)
                label.setPos(point.x + 4, point.y + 4)
                dot.setZValue(12)
                label.setZValue(12)
                self._vertex_items.extend([dot, label])

    def _draw_laser_center(self) -> None:
        self._clear_laser_center_items()
        if self.laser_center is None or not self.show_laser_center:
            return
        point = self.laser_center
        radius = 6
        items = [
            self.scene().addLine(point.x - 10, point.y, point.x + 10, point.y, QPen(POINT_COLOR, 2)),
            self.scene().addLine(point.x, point.y - 10, point.x, point.y + 10, QPen(POINT_COLOR, 2)),
            self.scene().addEllipse(
                point.x - radius,
                point.y - radius,
                radius * 2,
                radius * 2,
                QPen(Qt.white, 1),
                QBrush(POINT_COLOR),
            ),
            self.scene().addText("laser"),
        ]
        items[-1].setDefaultTextColor(POINT_COLOR)
        items[-1].setPos(point.x + 7, point.y + 7)
        for item in items:
            item.setZValue(15)
            self._laser_center_items.append(item)

    def _draw_radiation_points(self) -> None:
        self._clear_point_items()
        if not self.show_radiation_points:
            return
        for name, point in self.radiation_points.items():
            radius = 4
            dot = self.scene().addEllipse(
                point.x - radius,
                point.y - radius,
                radius * 2,
                radius * 2,
                QPen(POINT_COLOR, 1),
                QBrush(POINT_COLOR),
            )
            label = self.scene().addText(name)
            font = QFont()
            font.setPointSize(14)
            label.setFont(font)
            label.setDefaultTextColor(POINT_COLOR)
            label.setPos(point.x + 5, point.y + 5)
            dot.setZValue(12)
            label.setZValue(12)
            self._point_items.extend([dot, label])

    def _clear_point_items(self) -> None:
        for item in self._point_items:
            self.scene().removeItem(item)
        self._point_items = []

    def _emit_profile_lines_changed(self) -> None:
        if self.on_profile_lines_changed is None:
            return
        lines = self.profile_lines()
        self.on_profile_lines_changed(lines)

    def _emit_profile_line_selected(self) -> None:
        if self.on_profile_line_selected is not None:
            self.on_profile_line_selected(self.selected_profile_line)

    def visible_scene_rect(self) -> QRectF:
        return self.mapToScene(self.viewport().rect()).boundingRect()

    def _emit_visible_scene_rect_changed(self) -> None:
        if self.on_visible_scene_rect_changed is not None:
            self.on_visible_scene_rect_changed(self.visible_scene_rect())

    def _hit_profile_line(self, scene_pos: QPointF) -> str | None:
        if not self.profile_lines_visible:
            return None
        tolerance = 12.0 / max(self.transform().m11(), 0.01)
        best_name = None
        best_distance = tolerance
        for name, points in self.profile_lines().items():
            if self.visible_profile_lines is not None and name not in self.visible_profile_lines:
                continue
            if self.active_profile_orientation is not None and name != self.active_profile_orientation:
                continue
            distance = _distance_to_segment(scene_pos, points[0], points[1])
            if distance <= best_distance:
                best_name = name
                best_distance = distance
        return best_name

    def _hit_active_target(self, scene_pos: QPointF) -> str | None:
        rect = self._active_rect()
        if rect is None or self.active_field != "light" or not self.show_light_edges:
            return None
        tolerance = 12.0 / max(self.transform().m11(), 0.01)
        best_target = None
        best_distance = tolerance
        for edge in EDGE_NAMES:
            points = self._edge_points(rect, edge)
            if points is None:
                continue
            distance = _distance_to_segment(scene_pos, points[0], points[1])
            if distance <= best_distance:
                best_target = edge
                best_distance = distance
        return best_target

    def _edge_points(self, rect: RotatedRect, edge: str) -> tuple[Point, Point] | None:
        if self.active_field == "light" and self.light_polygon is not None:
            local_edge = self._visual_edge_map_points(self.light_polygon).get(edge)
            if local_edge is None:
                return None
            return self._polygon_edge_points(self.light_polygon).get(local_edge)
        local_edge = self._visual_edge_map(rect).get(edge)
        if local_edge is None:
            return None
        return self._local_edge_points(rect).get(local_edge)

    def _polygon_edge_points(self, points: tuple[Point, Point, Point, Point]) -> dict[str, tuple[Point, Point]]:
        top_left, top_right, bottom_right, bottom_left = points
        return {
            "top": (top_left, top_right),
            "right": (top_right, bottom_right),
            "bottom": (bottom_right, bottom_left),
            "left": (bottom_left, top_left),
        }

    def _move_light_polygon_target(self, target: str, dx: float, dy: float) -> tuple[Point, Point, Point, Point]:
        if self.light_polygon is None:
            raise ValueError("No light polygon")
        points = list(self.light_polygon)
        visual_edge = self._visual_edge_map_points(self.light_polygon).get(target)
        index_pairs = {
            "top": (0, 1),
            "right": (1, 2),
            "bottom": (2, 3),
            "left": (3, 0),
        }
        if visual_edge not in index_pairs:
            return self.light_polygon
        first, second = index_pairs[visual_edge]
        points[first] = points[first].moved(dx, dy)
        points[second] = points[second].moved(dx, dy)
        return tuple(points)

    def _constrained_light_delta(self, target: str, dx: float, dy: float) -> tuple[float, float]:
        if self.active_field != "light":
            return dx, dy
        if target in ("top", "bottom"):
            return 0.0, dy
        if target in ("left", "right"):
            return dx, 0.0
        return dx, dy

    def _visual_edge_map_points(self, points: tuple[Point, Point, Point, Point]) -> dict[str, str]:
        edge_points = self._polygon_edge_points(points)
        edge_centers = {
            name: Point((edge[0].x + edge[1].x) / 2.0, (edge[0].y + edge[1].y) / 2.0)
            for name, edge in edge_points.items()
        }
        top = min(edge_centers, key=lambda name: edge_centers[name].y)
        bottom = max(edge_centers, key=lambda name: edge_centers[name].y)
        remaining = [name for name in edge_centers if name not in (top, bottom)]
        left = min(remaining, key=lambda name: edge_centers[name].x)
        right = max(remaining, key=lambda name: edge_centers[name].x)
        return {"top": top, "right": right, "bottom": bottom, "left": left}

    def _local_edge_points(self, rect: RotatedRect) -> dict[str, tuple[Point, Point]]:
        top_left, top_right, bottom_right, bottom_left = rect.ordered_points()
        return {
            "top": (top_left, top_right),
            "right": (top_right, bottom_right),
            "bottom": (bottom_right, bottom_left),
            "left": (bottom_left, top_left),
        }

    def _visual_edge_map(self, rect: RotatedRect) -> dict[str, str]:
        edge_centers = {
            name: Point((points[0].x + points[1].x) / 2.0, (points[0].y + points[1].y) / 2.0)
            for name, points in self._local_edge_points(rect).items()
        }
        top = min(edge_centers, key=lambda name: edge_centers[name].y)
        bottom = max(edge_centers, key=lambda name: edge_centers[name].y)
        remaining = [name for name in edge_centers if name not in (top, bottom)]
        left = min(remaining, key=lambda name: edge_centers[name].x)
        right = max(remaining, key=lambda name: edge_centers[name].x)
        return {"top": top, "right": right, "bottom": bottom, "left": left}


def _midpoint(start: Point, end: Point) -> Point:
    return Point((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)


def _edge_length_points(points: tuple[Point, Point, Point, Point]) -> tuple[Point, Point, Point, Point]:
    return (
        _midpoint(points[3], points[0]),
        _midpoint(points[1], points[2]),
        _midpoint(points[0], points[1]),
        _midpoint(points[2], points[3]),
    )


def _append_unique_point(points: list[Point], point: Point) -> None:
    for existing in points:
        if abs(existing.x - point.x) < 0.001 and abs(existing.y - point.y) < 0.001:
            return
    points.append(point)


def _distance_to_segment(point: QPointF, start: Point, end: Point) -> float:
    px, py = point.x(), point.y()
    sx, sy = start.x, start.y
    ex, ey = end.x, end.y
    dx = ex - sx
    dy = ey - sy
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_squared))
    closest_x = sx + t * dx
    closest_y = sy + t * dy
    return hypot(px - closest_x, py - closest_y)


def _dashed_pen(color, width: int = 2, pattern: list[int] | None = None) -> QPen:
    pen = QPen(color, width)
    pen.setStyle(Qt.CustomDashLine)
    pen.setDashPattern(pattern or [10, 7])
    return pen
