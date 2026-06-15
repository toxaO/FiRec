import sys
from math import atan2, degrees, hypot
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from firec.core.analysis import (
    compare_fields,
    invert_profile,
    line_profile,
    load_image,
)
from firec.core.geometry import Point, RotatedRect
from firec.gui.image_view import ImageView
from firec.gui.profile_plot import ProfilePlot
from firec.storage.repository import connect_database, export_rows_to_csv, fetch_analysis_rows, save_analysis


PROFILE_POINT_NAMES = {
    "top": ("L1", "R1"),
    "bottom": ("L2", "R2"),
    "left": ("U2", "D2"),
    "right": ("U1", "D1"),
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FiRec")
        self.resize(900, 680)
        self.original_image: np.ndarray | None = None
        self.image: np.ndarray | None = None
        self.image_path: Path | None = None
        self.radiation_rect: RotatedRect | None = None
        self.radiation_polygon: tuple[Point, Point, Point, Point] | None = None
        self.light_rect: RotatedRect | None = None
        self.radiation_points: dict[str, Point] = {}
        self.profile_cursors: dict[str, float] = {}
        self.selected_profile_line: str | None = "top"
        self.connection = connect_database("firec.sqlite")

        self.view = ImageView()
        self.view.on_rect_changed = self._on_rect_changed
        self.view.on_tab_navigation = self.move_stage
        self.view.on_profile_lines_changed = self._on_profile_lines_changed
        self.view.on_profile_line_selected = self._on_profile_line_selected
        self.view.on_visible_scene_rect_changed = self._on_visible_scene_rect_changed

        self.main_tabs = QTabWidget()
        self.main_tabs.currentChanged.connect(self._on_main_tab_changed)
        self.path_edit = QLineEdit()
        self.top_profile_plot = ProfilePlot("horizontal")
        self.bottom_profile_plot = ProfilePlot("horizontal")
        self.left_profile_plot = ProfilePlot("vertical")
        self.right_profile_plot = ProfilePlot("vertical")
        self.profile_plots = {
            "top": self.top_profile_plot,
            "bottom": self.bottom_profile_plot,
            "left": self.left_profile_plot,
            "right": self.right_profile_plot,
        }
        for line_name, plot in self.profile_plots.items():
            plot.on_cursor_moved = self._on_profile_cursor_moved
            plot.on_selected = lambda line_name=line_name: self._on_profile_plot_selected(line_name)
        self.result_text = QPlainTextEdit("No result")
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(260)
        self.results_table = QTableWidget()
        self.current_stage = "radiation"
        self.stage_frames: dict[str, QGroupBox] = {}
        self.stage_controls: dict[str, list[QWidget]] = {}
        self.completed_stages: set[str] = set()

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self._build_layout()
        self.view.set_editing_enabled(False)
        self._sync_stage_controls()
        self.refresh_results_table()

    def _build_layout(self) -> None:
        self.main_tabs.addTab(self._analyse_tab(), "Analyse")
        self.main_tabs.addTab(self._record_tab(), "Record")
        self.setCentralWidget(self.main_tabs)

    def _analyse_tab(self) -> QWidget:
        side_layout = QVBoxLayout()
        side_layout.setContentsMargins(6, 6, 6, 6)
        side_layout.setSpacing(6)
        radiation_frame, radiation_controls = self._radiation_frame()
        light_frame, light_controls = self._light_frame()
        result_frame, result_controls = self._result_frame()
        self.stage_frames = {
            "radiation": radiation_frame,
            "light": light_frame,
            "result": result_frame,
        }
        self.stage_controls = {
            "radiation": radiation_controls,
            "light": light_controls,
            "result": result_controls,
        }
        side_layout.addWidget(radiation_frame)
        side_layout.addWidget(light_frame)
        side_layout.addWidget(result_frame)
        side_layout.addLayout(self._step_nav_buttons())
        side_layout.addStretch(1)
        side_layout.addWidget(self._options_frame())

        side_panel = _plain_frame(side_layout)
        side_panel.setMinimumWidth(280)
        side_panel.setMaximumWidth(340)

        side_scroll = QScrollArea()
        side_scroll.setWidget(side_panel)
        side_scroll.setWidgetResizable(True)
        side_scroll.setMinimumWidth(300)
        side_scroll.setMaximumWidth(360)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_profile_layout = QVBoxLayout()
        left_profile_layout.setContentsMargins(0, 0, 0, 0)
        left_profile_layout.setSpacing(3)
        left_profile_layout.addWidget(self.left_profile_plot, 1)

        right_profile_layout = QVBoxLayout()
        right_profile_layout.setContentsMargins(0, 0, 0, 0)
        right_profile_layout.setSpacing(3)
        right_profile_layout.addWidget(self.right_profile_plot, 1)

        image_row = QHBoxLayout()
        image_row.setContentsMargins(0, 0, 0, 0)
        image_row.setSpacing(3)
        image_row.addLayout(left_profile_layout)
        image_row.addWidget(self.view, 1)
        image_row.addLayout(right_profile_layout)

        top_profile_layout = QVBoxLayout()
        top_profile_layout.setContentsMargins(0, 0, 0, 0)
        top_profile_layout.setSpacing(3)
        top_profile_layout.addWidget(self.top_profile_plot)

        bottom_profile_layout = QVBoxLayout()
        bottom_profile_layout.setContentsMargins(0, 0, 0, 0)
        bottom_profile_layout.setSpacing(3)
        bottom_profile_layout.addWidget(self.bottom_profile_plot)

        top_profile_row = QHBoxLayout()
        top_profile_row.setContentsMargins(0, 0, 0, 0)
        top_profile_row.setSpacing(3)
        top_profile_row.addSpacing(40)
        top_profile_row.addLayout(top_profile_layout, 1)
        top_profile_row.addSpacing(40)

        bottom_profile_row = QHBoxLayout()
        bottom_profile_row.setContentsMargins(0, 0, 0, 0)
        bottom_profile_row.setSpacing(3)
        bottom_profile_row.addSpacing(40)
        bottom_profile_row.addLayout(bottom_profile_layout, 1)
        bottom_profile_row.addSpacing(40)

        image_layout = QVBoxLayout()
        image_layout.setContentsMargins(8, 8, 8, 8)
        image_layout.setSpacing(3)
        image_layout.addLayout(top_profile_row)
        image_layout.addLayout(image_row, 1)
        image_layout.addLayout(bottom_profile_row)

        image_panel_layout = QVBoxLayout()
        image_panel_layout.setContentsMargins(0, 0, 0, 0)
        image_panel_layout.setSpacing(6)
        image_panel_layout.addWidget(_plain_frame(self._load_bar()))
        image_panel_layout.addWidget(_plain_frame(image_layout), 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addLayout(image_panel_layout, 1)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        content_layout.addLayout(layout, 1)
        content_layout.addWidget(side_scroll)

        widget = QWidget()
        widget.setLayout(content_layout)
        return widget

    def _record_tab(self) -> QWidget:
        refresh_button = _button("Refresh")
        refresh_button.clicked.connect(self.refresh_results_table)
        export_button = _button("Export CSV")
        export_button.clicked.connect(self.export_csv)

        controls = QHBoxLayout()
        controls.addWidget(export_button)
        controls.addWidget(refresh_button)
        controls.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.results_table)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _load_bar(self) -> QHBoxLayout:
        browse_button = _button("Browse")
        browse_button.clicked.connect(self.browse_image_path)

        self.path_edit.returnPressed.connect(self.load_from_path_edit)
        self.path_edit.setPlaceholderText("TIFF image path")
        self.path_edit.setMinimumWidth(320)

        layout = QHBoxLayout()
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(browse_button)
        return layout

    def _radiation_frame(self) -> tuple[QGroupBox, list[QWidget]]:
        self.radiation_status_label = QLabel("Set points on four profiles.")

        layout = QVBoxLayout()
        layout.addWidget(self.radiation_status_label)

        frame = _frame("Radiation")
        frame.setLayout(layout)
        frame.mousePressEvent = lambda event: self.go_to_stage("radiation")
        return frame, []

    def _light_frame(self) -> tuple[QGroupBox, list[QWidget]]:
        reset_button = _button("Reset")
        reset_button.clicked.connect(self.reset_light_field)

        layout = QHBoxLayout()
        layout.addWidget(reset_button)
        layout.addStretch(1)

        frame = _frame("Light")
        frame.setLayout(layout)
        frame.mousePressEvent = lambda event: self.go_to_stage("light")
        return frame, [reset_button]

    def _result_frame(self) -> tuple[QGroupBox, list[QWidget]]:
        save_button = _button("Save")
        save_button.clicked.connect(self.save_current_result)

        layout = QVBoxLayout()
        layout.addWidget(self.result_text)
        layout.addWidget(save_button)

        frame = _frame("Result")
        frame.setLayout(layout)
        frame.mousePressEvent = lambda event: self.go_to_stage("result")
        return frame, [save_button]

    def _options_frame(self) -> QGroupBox:
        options = (
            ("Radiation Edge", self.view.set_show_radiation_edges),
            ("Radiation Center", self.view.set_show_radiation_center),
            ("Radiation Area", self.view.set_show_radiation_area),
            ("Set Points", self.view.set_show_radiation_points),
            ("Light Edge", self.view.set_show_light_edges),
            ("Light Center", self.view.set_show_light_center),
        )
        layout = QGridLayout()
        for index, (label, callback) in enumerate(options):
            check = QCheckBox(label)
            check.setChecked(True)
            check.toggled.connect(callback)
            layout.addWidget(check, index // 2, index % 2)
        frame = _frame("Options")
        frame.setLayout(layout)
        return frame

    def _step_nav_buttons(self) -> QHBoxLayout:
        step_label = QLabel("step")
        back_button = _button("▲ previous")
        next_button = _button("▼ next")
        back_button.clicked.connect(lambda: self.move_stage(True))
        next_button.clicked.connect(lambda: self.move_stage(False))

        layout = QHBoxLayout()
        layout.addWidget(step_label)
        layout.addStretch(1)
        layout.addWidget(back_button)
        layout.addWidget(next_button)
        return layout

    def browse_image_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TIFF image",
            str(Path.cwd()),
            "Images (*.tif *.tiff);;All files (*)",
        )
        if path:
            self.path_edit.setText(path)
            self.load_path(Path(path))

    def load_from_path_edit(self) -> None:
        path_text = self.path_edit.text().strip()
        if not path_text:
            QMessageBox.warning(self, "No image path", "Enter or browse a TIFF image path.")
            return
        self.load_path(Path(path_text))

    def load_path(self, path: Path) -> None:
        try:
            self.original_image = load_image(path)
            self.image = self.original_image
            self.image_path = path
            self.radiation_rect = None
            self.radiation_polygon = None
            self.light_rect = None
            self.radiation_points = {}
            self.profile_cursors = {}
            self.selected_profile_line = "top"
            self.view.set_image(self.image)
            self._set_default_profile_lines()
            self.view.set_radiation_rect(None)
            self.view.set_light_rect(None)
            self.completed_stages = set()
            self.path_edit.setText(str(path))
            self.activate_stage("radiation")
            self.status.showMessage("Loaded image.")
            self._update_result_label()
        except Exception as error:
            QMessageBox.critical(self, "Load failed", str(error))

    def activate_stage(self, stage: str) -> None:
        self.current_stage = stage
        if stage == "radiation":
            self.view.set_active_field("radiation")
            if self.selected_profile_line is None:
                self.view.select_profile_line("top")
            self.view.set_profile_orientation(None)
            self.view.set_profile_lines_visible(True)
        elif stage == "light":
            self.view.select_profile_line(None)
            self.view.set_active_field("light")
            self.view.set_profile_orientation(None)
            self.view.set_profile_lines_visible(False)
        else:
            self.view.select_profile_line(None)
            self.view.set_profile_orientation(None)
            self.view.set_profile_lines_visible(False)
        self.view.set_editing_enabled(stage in ("radiation", "light"))
        self._sync_stage_controls()
        if stage == "radiation":
            self._on_profile_lines_changed(self.view.profile_lines())
        self._sync_radiation_step_ui()
        self._sync_result_center_points()
        self.view.setVisible(True)
        self.main_tabs.setCurrentIndex(0)
        self.view.setFocus()

    def go_to_stage(self, stage: str) -> None:
        stages = ("radiation", "light", "result")
        if stage not in stages:
            return
        current_index = stages.index(self.current_stage)
        target_index = stages.index(stage)
        if target_index < current_index:
            self._reset_after_stage(stage)
            self.activate_stage(stage)
            return
        while self.current_stage != stage:
            previous_stage = self.current_stage
            self.move_stage(False)
            if self.current_stage == previous_stage:
                return

    def move_stage(self, backward: bool = False) -> None:
        if self.current_stage == "radiation" and self.image is None:
            if not backward:
                QMessageBox.warning(self, "No image", "Load an image before moving to Light.")
            return
        if self.current_stage == "radiation" and self.image is not None:
            if backward:
                return
            self._capture_radiation_points()
            if self._build_radiation_from_points():
                self.confirm_radiation_field()
                return
            QMessageBox.warning(self, "Incomplete radiation field", "Set all radiation points before moving to Light.")
            return
        if self.current_stage == "light" and not backward:
            self.confirm_light_field()
            return
        stages = ("radiation", "light", "result")
        index = stages.index(self.current_stage)
        next_index = index - 1 if backward else index + 1
        next_index = max(0, min(len(stages) - 1, next_index))
        if backward:
            self._reset_after_stage(stages[next_index])
        self.activate_stage(stages[next_index])

    def _reset_after_stage(self, stage: str) -> None:
        if stage == "radiation":
            self.light_rect = None
            self.view.set_light_rect(None)
            self.completed_stages.discard("radiation")
            self.completed_stages.discard("light")
        elif stage == "light":
            self.completed_stages.discard("light")
        self._sync_result_center_points()
        self._update_result_label()

    def _set_default_profile_lines(self) -> None:
        shape = self.view.image_shape()
        if shape is None:
            return
        height, width = shape
        self.view.set_profile_line_positions(
            top_y=height * 0.25,
            bottom_y=height * 0.75,
            left_x=width * 0.25,
            right_x=width * 0.75,
        )

    def confirm_radiation_field(self) -> None:
        if self.current_stage != "radiation":
            return
        if self.radiation_rect is None:
            if not self._build_radiation_from_points():
                QMessageBox.warning(self, "No radiation field", "Set all radiation points before setting radiation field.")
                return
        if self.radiation_rect is None:
            return
        self.light_rect = self.radiation_rect
        self.view.set_light_rect(self.light_rect)
        if self.radiation_polygon is not None:
            self.view.set_light_polygon(self.radiation_polygon)
        self.completed_stages.add("radiation")
        self.activate_stage("light")
        self._update_result_label()

    def reset_radiation_field(self) -> None:
        self.radiation_points = {}
        self.radiation_rect = None
        self.radiation_polygon = None
        self.light_rect = None
        self.view.set_radiation_points(self.radiation_points)
        self.view.set_radiation_rect(None)
        self.view.set_light_rect(None)
        self.status.showMessage("Cleared radiation points.")
        self._update_result_label()

    def reset_light_field(self) -> None:
        if self.current_stage != "light":
            return
        if self.radiation_rect is None:
            QMessageBox.warning(self, "No radiation field", "Set radiation field before resetting light field.")
            return
        self.light_rect = self.radiation_rect
        self.view.set_light_rect(self.light_rect)
        if self.radiation_polygon is not None:
            self.view.set_light_polygon(self.radiation_polygon)
        self.status.showMessage("Reset light field to radiation field.")
        self._update_result_label()

    def confirm_light_field(self) -> None:
        if self.current_stage != "light":
            return
        if self.radiation_rect is None or self.light_rect is None:
            QMessageBox.warning(self, "No light field", "Set radiation field before setting light field.")
            return
        self.completed_stages.add("light")
        self.activate_stage("result")
        self._update_result_label()

    def move_active_rect(self, dx: float, dy: float) -> None:
        if self.current_stage not in ("radiation", "light"):
            return
        self.view.move_active_rect(dx, dy)
        self.view.setFocus()

    def scale_active_rect(self, delta: float) -> None:
        if self.current_stage not in ("radiation", "light"):
            return
        self.view.scale_active_rect(delta)
        self.view.setFocus()

    def save_current_result(self) -> None:
        if self.current_stage != "result":
            return
        if self.image_path is None or self.radiation_rect is None or self.light_rect is None:
            QMessageBox.warning(self, "Cannot save", "Set both rectangles before saving.")
            return
        result = compare_fields(self.radiation_rect, self.light_rect)
        save_analysis(self.connection, self.image_path, result)
        self.refresh_results_table()
        self.status.showMessage("Saved analysis result.")

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            str(Path.cwd() / "firec_results.csv"),
            "CSV (*.csv)",
        )
        if not path:
            return
        rows = fetch_analysis_rows(self.connection)
        if not rows:
            QMessageBox.information(self, "No data", "There are no saved results to export.")
            return
        export_rows_to_csv(rows, path)
        self.status.showMessage(f"Exported {path}")

    def refresh_results_table(self) -> None:
        rows = fetch_analysis_rows(self.connection)
        columns = [
            "created_at",
            "image_path",
            "radiation_width",
            "radiation_height",
            "light_width",
            "light_height",
            "width_difference",
            "height_difference",
            "width_ratio",
            "height_ratio",
            "center_dx",
            "center_dy",
        ]
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        self.results_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, column in enumerate(columns):
                value = row.get(column, "")
                self.results_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.results_table.resizeColumnsToContents()

    def _on_main_tab_changed(self, index: int) -> None:
        if index == 1:
            self.refresh_results_table()
        else:
            self._sync_stage_controls()
            self.view.setFocus()

    def _on_rect_changed(self, field: str, rect: RotatedRect) -> None:
        if field == "radiation":
            self.radiation_rect = rect
        else:
            self.light_rect = rect
        self._update_result_label()

    def _on_profile_lines_changed(self, lines: dict[str, tuple[Point, Point]]) -> None:
        if self.image is None:
            for plot in self.profile_plots.values():
                plot.set_profile(None, None, ())
            return
        for name, line in lines.items():
            values = invert_profile(line_profile(self.image, line[0], line[1]))
            if name in ("top", "bottom"):
                positions = np.linspace(line[0].x, line[1].x, values.size)
                axis = "x"
            else:
                positions = np.linspace(line[0].y, line[1].y, values.size)
                axis = "y"
            positions, values = _sort_profile(positions, values)
            self.profile_plots[name].set_profile(
                values,
                positions,
                _line_polygon_edge_positions(self.radiation_polygon, line, axis),
            )
        self._sync_radiation_step_ui()

    def _on_profile_cursor_moved(self, name: str, value: float) -> None:
        self.profile_cursors[name] = value
        self._sync_radiation_step_ui()

    def _on_profile_line_selected(self, line_name: str | None) -> None:
        self.selected_profile_line = line_name
        self._sync_radiation_step_ui()

    def _on_profile_plot_selected(self, line_name: str) -> None:
        if self.current_stage != "radiation":
            return
        self.view.select_profile_line(line_name)
        self.view.setFocus()

    def _capture_radiation_points(self) -> None:
        lines = self.view.profile_lines()
        for line_name, point_names in PROFILE_POINT_NAMES.items():
            line = lines.get(line_name)
            if line is None:
                continue
            for point_name in point_names:
                cursor_value = self.profile_cursors.get(point_name)
                if cursor_value is None:
                    continue
                if line_name in ("top", "bottom"):
                    self.radiation_points[point_name] = Point(cursor_value, line[0].y)
                else:
                    self.radiation_points[point_name] = Point(line[0].x, cursor_value)
        self.view.set_radiation_points(self.radiation_points)
        self._build_radiation_from_points()
        self._sync_radiation_step_ui()
        self._update_result_label()

    def _sync_radiation_step_ui(self) -> None:
        if not hasattr(self, "radiation_status_label"):
            return
        self.radiation_status_label.setText(f"Set radiation points: {len(self.radiation_points)}/8")
        if self.current_stage != "radiation":
            for plot in self.profile_plots.values():
                plot.set_selected(False)
                plot.set_cursors({})
                plot.set_profile(None, None, ())
            self.view.set_profile_cursor_points({})
            return
        selected_line = self.selected_profile_line if self.current_stage == "radiation" else None
        preview_points: dict[str, Point] = {}
        for line_name, point_names in PROFILE_POINT_NAMES.items():
            orientation = "horizontal" if line_name in ("top", "bottom") else "vertical"
            self._ensure_step_cursors(point_names, orientation)
            plot = self.profile_plots[line_name]
            plot.set_selected(line_name == selected_line)
            cursors = {name: self.profile_cursors[name] for name in point_names if name in self.profile_cursors}
            plot.set_cursors(cursors)
            preview_points.update(self._profile_cursor_points(line_name, cursors))
        self.view.set_profile_cursor_points(preview_points)

    def _ensure_step_cursors(self, names: tuple[str, str], orientation: str) -> None:
        shape = self.view.image_shape()
        if shape is None:
            return
        height, width = shape
        if orientation == "horizontal":
            default_positions = (width * 0.2, width * 0.8)
        else:
            default_positions = (height * 0.2, height * 0.8)
        for name, position in zip(names, default_positions):
            self.profile_cursors.setdefault(name, position)

    def _profile_cursor_points(self, line_name: str, cursors: dict[str, float]) -> dict[str, Point]:
        line = self.view.profile_lines().get(line_name)
        if line is None:
            return {}
        points: dict[str, Point] = {}
        for name, value in cursors.items():
            if line_name in ("top", "bottom"):
                points[name] = Point(value, line[0].y)
            else:
                points[name] = Point(line[0].x, value)
        return points

    def _build_radiation_from_points(self) -> bool:
        required = ("L1", "R1", "L2", "R2", "U1", "D1", "U2", "D2")
        if any(name not in self.radiation_points for name in required):
            return False
        left_line = (self.radiation_points["L1"], self.radiation_points["L2"])
        right_line = (self.radiation_points["R1"], self.radiation_points["R2"])
        top_line = (self.radiation_points["U1"], self.radiation_points["U2"])
        bottom_line = (self.radiation_points["D1"], self.radiation_points["D2"])
        top_left = _line_intersection(left_line[0], left_line[1], top_line[0], top_line[1])
        top_right = _line_intersection(right_line[0], right_line[1], top_line[0], top_line[1])
        bottom_right = _line_intersection(right_line[0], right_line[1], bottom_line[0], bottom_line[1])
        bottom_left = _line_intersection(left_line[0], left_line[1], bottom_line[0], bottom_line[1])
        if None in (top_left, top_right, bottom_right, bottom_left):
            return False
        self.radiation_polygon = (top_left, top_right, bottom_right, bottom_left)
        self.radiation_rect = _rect_from_ordered_points(top_left, top_right, bottom_right, bottom_left)
        self.view.set_radiation_rect(self.radiation_rect, reset_profile_lines=False)
        self.view.set_radiation_polygon(self.radiation_polygon, reset_profile_lines=False)
        return True

    def _sync_result_center_points(self) -> None:
        if self.current_stage != "result":
            self.view.set_result_center_points({})
            return
        points: dict[str, Point] = {}
        if self.radiation_polygon is not None:
            points["radiation"] = _polygon_center(self.radiation_polygon)
        elif self.radiation_rect is not None:
            points["radiation"] = _polygon_center(self.radiation_rect.ordered_points())
        light_polygon = self._current_light_polygon()
        if light_polygon is not None:
            points["light"] = _polygon_center(light_polygon)
        self.view.set_result_center_points(points)

    def _on_visible_scene_rect_changed(self, scene_rect) -> None:
        for plot in self.profile_plots.values():
            plot.set_visible_range(None)

    def _update_result_label(self) -> None:
        if self.radiation_rect is None:
            if self.radiation_points:
                points = ", ".join(sorted(self.radiation_points))
                self.result_text.setPlainText(f"Radiation points: {points}")
            else:
                self.result_text.setPlainText("No result")
            return
        lines = _field_summary_lines("Radiation", self.radiation_polygon or self.radiation_rect.ordered_points())
        light_polygon = self._current_light_polygon()
        if light_polygon is not None:
            lines.append("")
            lines.extend(_field_summary_lines("Light", light_polygon))
        self.result_text.setPlainText("\n".join(lines))

    def _current_light_polygon(self) -> tuple[Point, Point, Point, Point] | None:
        if self.view.light_polygon is not None:
            return self.view.light_polygon
        if self.light_rect is not None:
            return self.light_rect.ordered_points()
        return None

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Tab, Qt.Key_Backtab) and self.main_tabs.currentIndex() == 0:
            self.move_stage(bool(event.modifiers() & Qt.ShiftModifier) or event.key() == Qt.Key_Backtab)
            event.accept()
            return
        super().keyPressEvent(event)

    def _sync_stage_controls(self) -> None:
        for stage, frame in self.stage_frames.items():
            selected = stage == self.current_stage
            frame.setTitle(_stage_title(stage, stage in self.completed_stages))
            frame.setStyleSheet(_selected_group_style() if selected else _group_style())
            for control in self.stage_controls[stage]:
                control.setEnabled(selected)


def _frame(title: str) -> QGroupBox:
    frame = QGroupBox(title)
    frame.setStyleSheet(_group_style())
    return frame


def _plain_frame(layout: QHBoxLayout | QVBoxLayout) -> QFrame:
    frame = QFrame()
    frame.setFrameShape(QFrame.NoFrame)
    frame.setStyleSheet("QFrame { background: #efefef; border: 0; border-radius: 4px; }")
    frame.setLayout(layout)
    return frame


def _stage_title(stage: str, completed: bool) -> str:
    labels = {
        "radiation": "Radiation",
        "light": "Light",
        "result": "Result",
    }
    prefix = "✓ " if completed else ""
    return f"{prefix}{labels[stage]}"


def _button(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setMaximumWidth(120)
    button.setMinimumWidth(72)
    button.setFocusPolicy(Qt.NoFocus)
    return button


def _group_style() -> str:
    return "QGroupBox { background: #f3f3f3; border: 0; border-radius: 4px; margin-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"


def _selected_group_style() -> str:
    return "QGroupBox { background: #dfe9f6; border: 0; border-radius: 4px; margin-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"


def _sort_profile(positions: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(positions)
    return positions[order], values[order]


def _field_summary_lines(label: str, points: tuple[Point, Point, Point, Point]) -> list[str]:
    center = _polygon_center(points)
    edge_lengths = _polygon_edge_lengths(points)
    area = _polygon_area(points)
    vertex_labels = ("TL", "TR", "BR", "BL")
    edge_labels = ("top", "right", "bottom", "left")
    lines = [f"{label}"]
    lines.append(f"  center: {_format_point(center)}")
    lines.append("  vertices:")
    lines.extend(f"    {name}: {_format_point(point)}" for name, point in zip(vertex_labels, points))
    lines.append("  edge lengths:")
    lines.extend(f"    {name}: {length:.2f}" for name, length in zip(edge_labels, edge_lengths))
    lines.append(f"  area: {area:.2f}")
    return lines


def _polygon_center(points: tuple[Point, Point, Point, Point]) -> Point:
    return Point(
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def _polygon_edge_lengths(points: tuple[Point, Point, Point, Point]) -> tuple[float, float, float, float]:
    return tuple(
        hypot(points[(index + 1) % len(points)].x - point.x, points[(index + 1) % len(points)].y - point.y)
        for index, point in enumerate(points)
    )


def _polygon_area(points: tuple[Point, Point, Point, Point]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point.x * next_point.y - next_point.x * point.y
    return abs(area) / 2.0


def _format_point(point: Point) -> str:
    return f"({point.x:.2f}, {point.y:.2f})"


def _line_polygon_edge_positions(
    polygon: tuple[Point, Point, Point, Point] | None,
    line: tuple[Point, Point],
    axis: str,
) -> tuple[float, ...]:
    if polygon is None:
        return ()
    edges = (
        (polygon[0], polygon[1]),
        (polygon[1], polygon[2]),
        (polygon[2], polygon[3]),
        (polygon[3], polygon[0]),
    )
    positions: list[float] = []
    for edge in edges:
        intersection = _segment_intersection(line[0], line[1], edge[0], edge[1])
        if intersection is None:
            continue
        position = intersection.y if axis == "y" else intersection.x
        if not any(abs(position - existing) < 0.001 for existing in positions):
            positions.append(position)
    return tuple(sorted(positions))


def _segment_intersection(a: Point, b: Point, c: Point, d: Point) -> Point | None:
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
    if not (_between(px, a.x, b.x) and _between(py, a.y, b.y)):
        return None
    if not (_between(px, c.x, d.x) and _between(py, c.y, d.y)):
        return None
    return Point(float(px), float(py))


def _between(value: float, start: float, end: float) -> bool:
    return min(start, end) - 0.001 <= value <= max(start, end) + 0.001


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


def _rect_from_ordered_points(top_left: Point, top_right: Point, bottom_right: Point, bottom_left: Point) -> RotatedRect:
    center = Point(
        (top_left.x + top_right.x + bottom_right.x + bottom_left.x) / 4.0,
        (top_left.y + top_right.y + bottom_right.y + bottom_left.y) / 4.0,
    )
    top_width = hypot(top_right.x - top_left.x, top_right.y - top_left.y)
    bottom_width = hypot(bottom_right.x - bottom_left.x, bottom_right.y - bottom_left.y)
    left_height = hypot(bottom_left.x - top_left.x, bottom_left.y - top_left.y)
    right_height = hypot(bottom_right.x - top_right.x, bottom_right.y - top_right.y)
    angle = degrees(atan2(top_right.y - top_left.y, top_right.x - top_left.x))
    return RotatedRect(
        center,
        (top_width + bottom_width) / 2.0,
        (left_height + right_height) / 2.0,
        angle,
    )


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
