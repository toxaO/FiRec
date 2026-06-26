import sys
from math import atan2, degrees, hypot
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    QScrollArea,
    QStatusBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from firec.core.analysis import (
    AnalysisResult,
    FieldGeometry,
    circular_region_mean,
    compare_field_polygons,
    detect_profile_boundaries,
    invert_profile,
    line_profile,
    load_image,
    moving_average_profile,
)
from firec.core.geometry import Point, RotatedRect
from firec.gui.image_view import ImageView
from firec.gui.profile_plot import ProfilePlot
from firec.storage.repository import connect_database, delete_analysis, export_rows_to_csv, fetch_analysis_rows, save_analysis


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
        self.laser_center: Point | None = None
        self.radiation_points: dict[str, Point] = {}
        self.profile_cursors: dict[str, float] = {}
        self.manually_adjusted_radiation_cursors: set[str] = set()
        self._auto_updating_profile_cursor = False
        self._last_profile_lines: dict[str, tuple[Point, Point]] = {}
        self._profile_display_markers: dict[str, tuple[np.ndarray | None, float | None, tuple[float, float] | None]] = {}
        self._auto_detection_failures: dict[str, str] = {}
        self.selected_profile_line: str | None = "bottom"
        self.settings = QSettings("FiRec", "FiRec")
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
        self.image_info_label = QLabel("No image")
        self.dpi_spin = QDoubleSpinBox()
        self.origin_combo = QComboBox()
        self.record_origin_combo = QComboBox()
        self.record_dpi_spin = QDoubleSpinBox()
        self.film_pixel_spin = QDoubleSpinBox()
        self.radiation_threshold_spin = QDoubleSpinBox()
        self.radiation_center_spin = QDoubleSpinBox()
        self.smoothing_window_spin = QSpinBox()
        self.raw_profile_check = QCheckBox("Raw profile")
        self.smoothed_profile_check = QCheckBox("Smoothed profile")
        self.circle_x_spin = QDoubleSpinBox()
        self.circle_y_spin = QDoubleSpinBox()
        self.circle_radius_spin = QDoubleSpinBox()
        self.circle_mean_label = QLabel("")
        self.circle_options_visible = False
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
        self.result_tree = QTreeWidget()
        self.result_tree.setHeaderLabels(["Item", "Value"])
        self.result_tree.setMinimumHeight(260)
        self.results_table = QTableWidget()
        self.result_row_ids: list[int] = []
        self.current_stage = "laser"
        self.stage_frames: dict[str, QGroupBox] = {}
        self.stage_controls: dict[str, list[QWidget]] = {}
        self.completed_stages: set[str] = set()

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self._build_layout()
        self._restore_settings()
        self.view.set_editing_enabled(False)
        self._update_profile_visibility()
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
        laser_frame, laser_controls = self._laser_frame()
        radiation_frame, radiation_controls = self._radiation_frame()
        light_frame, light_controls = self._light_frame()
        result_frame, result_controls = self._result_frame()
        self.stage_frames = {
            "laser": laser_frame,
            "radiation": radiation_frame,
            "light": light_frame,
            "result": result_frame,
        }
        self.stage_controls = {
            "laser": laser_controls,
            "radiation": radiation_controls,
            "light": light_controls,
            "result": result_controls,
        }
        side_layout.addLayout(self._step_nav_buttons())
        side_layout.addWidget(laser_frame)
        side_layout.addWidget(radiation_frame)
        side_layout.addWidget(light_frame)
        side_layout.addWidget(result_frame)
        side_layout.addWidget(self._display_options_frame())
        side_layout.addStretch(1)

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
        image_layout.addLayout(self._zoom_buttons())

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
        self.record_origin_combo.addItem("Origin: Laser", "laser")
        self.record_origin_combo.addItem("Origin: Radiation", "radiation")
        self.record_origin_combo.addItem("Origin: Light", "light")
        self.record_origin_combo.currentIndexChanged.connect(lambda index: self.refresh_results_table())

        self.record_dpi_spin.setRange(0.0, 10000.0)
        self.record_dpi_spin.setDecimals(1)
        self.record_dpi_spin.setSingleStep(1.0)
        self.record_dpi_spin.setSpecialValueText("px")
        self.record_dpi_spin.setSuffix(" dpi")
        self.record_dpi_spin.setMaximumWidth(110)
        self.record_dpi_spin.valueChanged.connect(lambda value: self._on_record_dpi_changed())

        refresh_button = _button("Refresh")
        refresh_button.clicked.connect(self.refresh_results_table)
        export_button = _button("Export CSV")
        export_button.clicked.connect(self.export_csv)
        delete_button = _button("Delete")
        delete_button.clicked.connect(self.delete_selected_records)

        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.ExtendedSelection)

        controls = QHBoxLayout()
        controls.addWidget(self.record_origin_combo)
        controls.addWidget(QLabel("DPI"))
        controls.addWidget(self.record_dpi_spin)
        controls.addWidget(export_button)
        controls.addWidget(delete_button)
        controls.addWidget(refresh_button)
        controls.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.results_table)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _load_bar(self) -> QVBoxLayout:
        browse_button = _button("Browse")
        browse_button.clicked.connect(self.browse_image_path)

        self.dpi_spin.setRange(0.0, 10000.0)
        self.dpi_spin.setDecimals(1)
        self.dpi_spin.setSingleStep(1.0)
        self.dpi_spin.setSpecialValueText("px")
        self.dpi_spin.setSuffix(" dpi")
        self.dpi_spin.setMaximumWidth(110)
        self.dpi_spin.valueChanged.connect(lambda value: self._on_dpi_changed())

        self.path_edit.returnPressed.connect(self.load_from_path_edit)
        self.path_edit.setPlaceholderText("TIFF image path")
        self.path_edit.setMinimumWidth(320)
        self.image_info_label.setMinimumWidth(180)

        top_row = QHBoxLayout()
        top_row.addWidget(self.path_edit, 1)
        top_row.addWidget(browse_button)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.image_info_label, 1)
        bottom_row.addWidget(QLabel("DPI"))
        bottom_row.addWidget(self.dpi_spin)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addLayout(top_row)
        layout.addLayout(bottom_row)
        return layout

    def _zoom_buttons(self) -> QHBoxLayout:
        zoom_in_button = _button("Zoom In")
        zoom_in_button.clicked.connect(self.view.zoom_in)
        zoom_out_button = _button("Zoom Out")
        zoom_out_button.clicked.connect(self.view.zoom_out)
        pan_button = _button("Pan")
        pan_button.setCheckable(True)
        pan_button.toggled.connect(self.view.set_pan_enabled)
        reset_button = _button("Reset")
        reset_button.clicked.connect(self.view.reset_view)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)
        for button in (zoom_out_button, zoom_in_button, pan_button, reset_button):
            layout.addWidget(button)
        layout.addStretch(1)
        return layout

    def _laser_frame(self) -> tuple[QGroupBox, list[QWidget]]:
        self.laser_status_label = QLabel("Set laser center.")
        reset_button = _button("Reset")
        reset_button.clicked.connect(self.reset_laser_center)

        layout = QVBoxLayout()
        layout.addWidget(self.laser_status_label)
        layout.addWidget(reset_button)

        frame = _frame("Laser Center")
        frame.setLayout(layout)
        frame.mousePressEvent = lambda event: self.go_to_stage("laser")
        return frame, [reset_button]

    def _radiation_frame(self) -> tuple[QGroupBox, list[QWidget]]:
        self.radiation_status_label = QLabel("Set radiation boundary points")
        reset_button = _button("Reset")
        reset_button.clicked.connect(self.reset_radiation_field)

        self.film_pixel_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.film_pixel_spin.setDecimals(1)
        self.film_pixel_spin.setSingleStep(100.0)
        self.film_pixel_spin.setToolTip("Raw pixel value for unirradiated film.")
        self.film_pixel_spin.valueChanged.connect(lambda value: self._redetect_radiation_lines())

        self.radiation_threshold_spin.setRange(0.0, 100.0)
        self.radiation_threshold_spin.setDecimals(1)
        self.radiation_threshold_spin.setSingleStep(1.0)
        self.radiation_threshold_spin.setSuffix(" %")
        self.radiation_threshold_spin.setValue(50.0)
        self.radiation_threshold_spin.valueChanged.connect(lambda value: self._redetect_radiation_lines())

        self.radiation_center_spin.setRange(1.0, 100.0)
        self.radiation_center_spin.setDecimals(1)
        self.radiation_center_spin.setSingleStep(1.0)
        self.radiation_center_spin.setSuffix(" %")
        self.radiation_center_spin.setValue(20.0)
        self.radiation_center_spin.valueChanged.connect(lambda value: self._redetect_radiation_lines())

        self.smoothing_window_spin.setRange(1, 501)
        self.smoothing_window_spin.setSingleStep(2)
        self.smoothing_window_spin.setValue(5)
        self.smoothing_window_spin.valueChanged.connect(lambda value: self._redetect_radiation_lines())

        for spin in (self.circle_x_spin, self.circle_y_spin, self.circle_radius_spin):
            spin.setRange(0.0, 1_000_000.0)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setMaximumWidth(64)
        self.circle_x_spin.valueChanged.connect(lambda value: self._update_circle_overlay())
        self.circle_y_spin.valueChanged.connect(lambda value: self._update_circle_overlay())
        self.circle_radius_spin.valueChanged.connect(lambda value: self._update_circle_overlay())
        self.circle_radius_spin.setValue(10.0)
        self.circle_mean_label.setFixedWidth(60)
        self.circle_mean_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.circle_mean_label.setText("")
        circle_button = _button("Mean")
        circle_button.setMaximumWidth(64)
        circle_button.setToolTip("Calculate circle mean.")
        circle_button.clicked.connect(self.calculate_circle_mean)

        settings_layout = QGridLayout()
        settings_layout.addWidget(QLabel("Film baseline px"), 0, 0)
        settings_layout.addWidget(self.film_pixel_spin, 0, 1)
        settings_layout.addWidget(QLabel("Boundary"), 1, 0)
        settings_layout.addWidget(self.radiation_threshold_spin, 1, 1)
        settings_layout.addWidget(QLabel("Center +/-"), 2, 0)
        settings_layout.addWidget(self.radiation_center_spin, 2, 1)
        settings_layout.addWidget(QLabel("Smooth px"), 3, 0)
        settings_layout.addWidget(self.smoothing_window_spin, 3, 1)

        layout = QVBoxLayout()
        layout.addWidget(self.radiation_status_label)
        layout.addLayout(settings_layout)
        profile_toggle_layout = QHBoxLayout()
        profile_toggle_layout.addWidget(self.raw_profile_check)
        profile_toggle_layout.addWidget(self.smoothed_profile_check)
        profile_toggle_layout.addStretch(1)
        layout.addLayout(profile_toggle_layout)
        layout.addWidget(self._circle_mean_frame(circle_button))
        layout.addWidget(reset_button)

        frame = _frame("Radiation")
        frame.setLayout(layout)
        frame.mousePressEvent = lambda event: self.go_to_stage("radiation")
        return frame, [
            self.film_pixel_spin,
            self.radiation_threshold_spin,
            self.radiation_center_spin,
            self.smoothing_window_spin,
            self.raw_profile_check,
            self.smoothed_profile_check,
            circle_button,
            reset_button,
        ]

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

        self.origin_combo.addItem("Origin: Laser", "laser")
        self.origin_combo.addItem("Origin: Radiation", "radiation")
        self.origin_combo.addItem("Origin: Light", "light")
        self.origin_combo.currentIndexChanged.connect(lambda index: self._update_result_label())

        controls = QHBoxLayout()
        controls.addStretch(1)
        controls.addWidget(self.origin_combo)
        controls.addWidget(save_button)

        layout = QVBoxLayout()
        layout.addWidget(self.result_tree)
        layout.addLayout(controls)

        frame = _frame("Result")
        frame.setLayout(layout)
        frame.mousePressEvent = lambda event: self.go_to_stage("result")
        return frame, [self.origin_combo, save_button]

    def _display_options_frame(self) -> QGroupBox:
        options = (
            ("Laser Center", self.view.set_show_laser_center),
            ("Radiation Edge", self.view.set_show_radiation_edges),
            ("Radiation Center", self.view.set_show_radiation_center),
            ("Radiation Area", self.view.set_show_radiation_area),
            ("Radiation Length", self.view.set_show_radiation_edge_lengths),
            ("Radiation Vertices", self.view.set_show_radiation_vertices),
            ("Radiation boundary", self.view.set_show_radiation_points),
            ("Light Edge", self.view.set_show_light_edges),
            ("Light Center", self.view.set_show_light_center),
            ("Light Length", self.view.set_show_light_edge_lengths),
            ("Light Vertices", self.view.set_show_light_vertices),
        )
        visibility_layout = QGridLayout()
        for index, (label, callback) in enumerate(options):
            check = QCheckBox(label)
            check.setChecked(True)
            check.toggled.connect(callback)
            visibility_layout.addWidget(check, index // 2, index % 2)
        frame = _frame("Options")
        frame.setLayout(visibility_layout)
        return frame

    def _circle_mean_frame(self, circle_button: QPushButton) -> QGroupBox:
        self.raw_profile_check.setChecked(True)
        self.smoothed_profile_check.setChecked(True)
        self.raw_profile_check.toggled.connect(lambda visible: self._update_profile_visibility())
        self.smoothed_profile_check.toggled.connect(lambda visible: self._update_profile_visibility())

        circle_row = QHBoxLayout()
        circle_row.setContentsMargins(0, 0, 0, 0)
        circle_row.setSpacing(3)
        circle_row.addStretch(1)
        circle_row.addWidget(QLabel("X"))
        circle_row.addWidget(self.circle_x_spin)
        circle_row.addWidget(QLabel("Y"))
        circle_row.addWidget(self.circle_y_spin)
        circle_row.addWidget(QLabel("R"))
        circle_row.addWidget(self.circle_radius_spin)
        circle_row.addStretch(1)

        circle_action_row = QHBoxLayout()
        circle_action_row.setContentsMargins(0, 0, 0, 0)
        circle_action_row.setSpacing(3)
        circle_action_row.addStretch(1)
        circle_action_row.addWidget(circle_button)
        circle_action_row.addWidget(self.circle_mean_label)
        circle_action_row.addStretch(1)

        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addLayout(circle_row)
        content_layout.addLayout(circle_action_row)
        content.setLayout(content_layout)

        frame, toggle = self._collapsible_section("Options", content, expanded=False)
        toggle.toggled.connect(self._set_circle_options_visible)
        self._set_circle_options_visible(False)
        return frame

    def _collapsible_section(self, title: str, content: QWidget, expanded: bool = False) -> tuple[QGroupBox, QToolButton]:
        toggle = QToolButton()
        toggle.setText(title)
        toggle.setCheckable(True)
        toggle.setChecked(expanded)
        toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggle.setAutoRaise(True)

        body = QWidget()
        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(2)
        body_layout.addWidget(content)
        body.setLayout(body_layout)
        body.setVisible(expanded)

        def _sync(checked: bool) -> None:
            body.setVisible(checked)
            toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

        toggle.toggled.connect(_sync)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(toggle)
        layout.addWidget(body)

        frame = QGroupBox()
        frame.setLayout(layout)
        _sync(expanded)
        return frame, toggle

    def _step_nav_buttons(self) -> QHBoxLayout:
        step_label = QLabel("step")
        back_button = _nav_button("▲ previous")
        next_button = _nav_button("▼ next")
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
            self.path_edit.text().strip() or str(Path.cwd()),
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
            self.laser_center = None
            self.radiation_points = {}
            self.profile_cursors = {}
            self.manually_adjusted_radiation_cursors = set()
            self._last_profile_lines = {}
            self._profile_display_markers = {}
            self._auto_detection_failures = {}
            self.selected_profile_line = "bottom"
            self.view.set_image(self.image)
            self._update_image_info()
            self._set_default_radiation_settings()
            self._update_circle_overlay()
            self.laser_center = self._default_laser_center()
            self._set_laser_profile_lines()
            self._set_laser_profile_cursors()
            self.view.set_laser_center(self.laser_center)
            self.view.set_radiation_rect(None)
            self.view.set_light_rect(None)
            self.completed_stages = set()
            self.path_edit.setText(str(path))
            self._save_settings()
            self.activate_stage("laser")
            self.status.showMessage("Loaded image.")
            self._update_result_label()
        except Exception as error:
            QMessageBox.critical(self, "Load failed", str(error))

    def _update_image_info(self) -> None:
        if self.image is None:
            self.image_info_label.setText("No image")
            return
        minimum = float(np.min(self.image))
        maximum = float(np.max(self.image))
        self.image_info_label.setText(f"{self.image.dtype} min {minimum:.1f} max {maximum:.1f}")

    def _set_default_radiation_settings(self) -> None:
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        minimum = float(np.min(self.image))
        maximum = float(np.max(self.image))
        self.circle_x_spin.setMaximum(float(width - 1))
        self.circle_y_spin.setMaximum(float(height - 1))
        self.circle_x_spin.setValue((width - 1.0) / 2.0)
        self.circle_y_spin.setValue((height - 1.0) / 2.0)
        self.film_pixel_spin.blockSignals(True)
        self.film_pixel_spin.setRange(minimum - abs(maximum - minimum) * 2.0 - 1.0, maximum + abs(maximum - minimum) * 2.0 + 1.0)
        self.film_pixel_spin.setValue(maximum)
        self.film_pixel_spin.blockSignals(False)
        self.circle_mean_label.setText("")

    def _restore_settings(self) -> None:
        last_path = self.settings.value("last_image_path", "", str)
        if last_path:
            self.path_edit.setText(last_path)
        dpi = self.settings.value("dpi", 0.0, float)
        self.dpi_spin.setValue(float(dpi or 0.0))
        self.record_dpi_spin.setValue(float(dpi or 0.0))

    def _save_settings(self) -> None:
        self.settings.setValue("last_image_path", self.path_edit.text().strip())
        self.settings.setValue("dpi", self.dpi_spin.value())

    def _on_dpi_changed(self) -> None:
        self._save_settings()
        self._update_result_label()

    def _on_record_dpi_changed(self) -> None:
        self.settings.setValue("dpi", self.record_dpi_spin.value())
        self.refresh_results_table()

    def activate_stage(self, stage: str) -> None:
        self.current_stage = stage
        if stage == "laser":
            self._set_laser_profile_lines()
            self.view.set_active_field("laser")
            self.view.set_visible_profile_lines({"left", "bottom"})
            if self.selected_profile_line not in ("left", "bottom"):
                self.selected_profile_line = "bottom"
            self.view.select_profile_line(self.selected_profile_line)
            self.view.set_profile_orientation(None)
            self.view.set_profile_lines_visible(True)
        elif stage == "radiation":
            self._set_default_profile_lines()
            self.view.set_active_field("radiation")
            if self.selected_profile_line is None:
                self.view.select_profile_line("top")
            self.view.set_visible_profile_lines(None)
            self.view.set_profile_orientation(None)
            self.view.set_profile_lines_visible(True)
        elif stage == "light":
            self.view.select_profile_line(None)
            self.view.set_active_field("light")
            self.view.set_visible_profile_lines(None)
            self.view.set_profile_orientation(None)
            self.view.set_profile_lines_visible(False)
        else:
            self.view.select_profile_line(None)
            self.view.set_visible_profile_lines(None)
            self.view.set_profile_orientation(None)
            self.view.set_profile_lines_visible(False)
        self.view.set_editing_enabled(stage in ("laser", "radiation", "light"))
        self._sync_stage_controls()
        if stage in ("laser", "radiation"):
            self._on_profile_lines_changed(self.view.profile_lines())
        self._sync_laser_step_ui()
        self._sync_radiation_step_ui()
        self._sync_result_center_points()
        self.view.setVisible(True)
        self.main_tabs.setCurrentIndex(0)
        self.view.setFocus()

    def go_to_stage(self, stage: str) -> None:
        stages = _stage_names()
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
        if self.current_stage == "laser":
            if backward:
                return
            if self.image is None:
                QMessageBox.warning(self, "No image", "Load an image before moving to Radiation.")
                return
            self.confirm_laser_center()
            return
        if self.current_stage == "radiation" and self.image is None:
            if not backward:
                QMessageBox.warning(self, "No image", "Load an image before moving to Light.")
            return
        if self.current_stage == "radiation" and self.image is not None and not backward:
            self._capture_radiation_points()
            if self._build_radiation_from_points():
                self.confirm_radiation_field()
                return
            QMessageBox.warning(self, "Incomplete radiation field", "Set all radiation points before moving to Light.")
            return
        if self.current_stage == "light" and not backward:
            self.confirm_light_field()
            return
        stages = _stage_names()
        index = stages.index(self.current_stage)
        next_index = index - 1 if backward else index + 1
        next_index = max(0, min(len(stages) - 1, next_index))
        if backward:
            self._reset_after_stage(stages[next_index])
        self.activate_stage(stages[next_index])

    def _reset_after_stage(self, stage: str) -> None:
        if stage == "laser":
            self.radiation_rect = None
            self.radiation_polygon = None
            self.light_rect = None
            self.radiation_points = {}
            self.view.set_radiation_points(self.radiation_points)
            self.view.set_radiation_rect(None)
            self.view.set_light_rect(None)
            self.completed_stages.clear()
        elif stage == "radiation":
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

    def _default_laser_center(self) -> Point | None:
        shape = self.view.image_shape()
        if shape is None:
            return None
        height, width = shape
        return Point((width - 1.0) / 2.0, (height - 1.0) / 2.0)

    def _set_laser_profile_lines(self) -> None:
        if self.laser_center is None:
            self.laser_center = self._default_laser_center()
        if self.laser_center is None:
            return
        self.view.set_profile_line_positions(
            bottom_y=self.laser_center.y,
            left_x=self.laser_center.x,
        )

    def _set_laser_profile_cursors(self) -> None:
        if self.laser_center is None:
            self.laser_center = self._default_laser_center()
        if self.laser_center is None:
            return
        self.profile_cursors["laser_x"] = self.laser_center.x
        self.profile_cursors["laser_y"] = self.laser_center.y

    def confirm_laser_center(self) -> None:
        if self.current_stage != "laser":
            return
        if self.laser_center is None:
            self.laser_center = self._default_laser_center()
        if self.laser_center is None:
            QMessageBox.warning(self, "No laser center", "Load an image before setting laser center.")
            return
        self.completed_stages.add("laser")
        self.activate_stage("radiation")

    def reset_laser_center(self) -> None:
        if self.image is None:
            return
        center = self._default_laser_center()
        if center is None:
            return
        self.laser_center = center
        self._set_laser_profile_lines()
        self._set_laser_profile_cursors()
        self.view.set_laser_center(self.laser_center)
        self._sync_laser_step_ui()
        self._update_result_label()
        self.status.showMessage("Reset laser center.")

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
        for point_names in PROFILE_POINT_NAMES.values():
            for point_name in point_names:
                self.profile_cursors.pop(point_name, None)
        self.manually_adjusted_radiation_cursors = set()
        self._profile_display_markers = {}
        self._auto_detection_failures = {}
        self.view.set_radiation_points(self.radiation_points)
        self.view.set_radiation_rect(None)
        self.view.set_light_rect(None)
        if self.current_stage == "radiation":
            self._redetect_radiation_lines()
        self.status.showMessage("Reset radiation boundary detection.")
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
        result = self._current_analysis_result("laser", raw_pixels=True)
        if result is None:
            QMessageBox.warning(self, "Cannot save", "Set both rectangles before saving.")
            return
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
        rows = self._display_record_rows(fetch_analysis_rows(self.connection))
        if not rows:
            QMessageBox.information(self, "No data", "There are no saved results to export.")
            return
        export_rows_to_csv(rows, path)
        self.status.showMessage(f"Exported {path}")

    def delete_selected_records(self) -> None:
        selected_rows = sorted({index.row() for index in self.results_table.selectedIndexes()}, reverse=True)
        if not selected_rows:
            return
        for row_index in selected_rows:
            if 0 <= row_index < len(self.result_row_ids):
                delete_analysis(self.connection, self.result_row_ids[row_index])
        self.refresh_results_table()
        self.status.showMessage(f"Deleted {len(selected_rows)} record(s).")

    def refresh_results_table(self) -> None:
        raw_rows = fetch_analysis_rows(self.connection)
        rows = self._display_record_rows(raw_rows)
        columns = [
            "created_at",
            "image_path",
            "origin",
            "dpi",
            "unit",
            "laser_center_x",
            "laser_center_y",
            "radiation_center_x",
            "radiation_center_y",
            "light_center_x",
            "light_center_y",
            "radiation_edge_length_x",
            "radiation_edge_length_y",
            "radiation_area",
            "light_area",
        ]
        self.result_row_ids = [int(row["id"]) for row in raw_rows]
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        self.results_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, column in enumerate(columns):
                value = row.get(column, "")
                self.results_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.results_table.resizeColumnsToContents()

    def _display_record_rows(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        origin = self.record_origin_combo.currentData() or "laser"
        dpi = self.record_dpi_spin.value()
        return [_record_display_row(row, origin, dpi) for row in rows]

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
        active_lines = {"left", "bottom"} if self.current_stage == "laser" else set(lines)
        changed_lines = self._changed_profile_lines(lines)
        if self.current_stage == "radiation":
            self._auto_update_radiation_cursors(lines, changed_lines)
        for name, plot in self.profile_plots.items():
            if name not in active_lines:
                plot.set_profile(None, None, ())
        for name, line in lines.items():
            if name not in active_lines:
                continue
            raw_values = line_profile(self.image, line[0], line[1])
            values = invert_profile(raw_values)
            if name in ("top", "bottom"):
                positions = np.linspace(line[0].x, line[1].x, values.size)
            else:
                positions = np.linspace(line[0].y, line[1].y, values.size)
            positions, values = _sort_profile(positions, values)
            smoothed_values, reference_value, window_positions = self._profile_display_markers.get(name, (None, None, None))
            self.profile_plots[name].set_profile(
                values,
                positions,
                window_positions or (),
                smoothed_values,
                reference_value,
            )
        self._sync_laser_step_ui()
        self._sync_radiation_step_ui()

    def _on_profile_cursor_moved(self, name: str, value: float) -> None:
        self.profile_cursors[name] = value
        if not self._auto_updating_profile_cursor and name not in ("laser_x", "laser_y"):
            self.manually_adjusted_radiation_cursors.add(name)
        if name in ("laser_x", "laser_y"):
            self._update_laser_center_from_cursors()
            self._sync_laser_step_ui()
            self._update_result_label()
            return
        self._sync_radiation_step_ui()

    def _on_profile_line_selected(self, line_name: str | None) -> None:
        self.selected_profile_line = line_name
        self._sync_laser_step_ui()
        self._sync_radiation_step_ui()

    def _on_profile_plot_selected(self, line_name: str) -> None:
        if self.current_stage not in ("laser", "radiation"):
            return
        if self.current_stage == "laser" and line_name not in ("left", "bottom"):
            return
        self.view.select_profile_line(line_name)
        self.view.setFocus()

    def _update_profile_visibility(self) -> None:
        raw_visible = self.raw_profile_check.isChecked()
        smoothed_visible = self.smoothed_profile_check.isChecked()
        for plot in self.profile_plots.values():
            plot.set_raw_profile_visible(raw_visible)
            plot.set_smoothed_profile_visible(smoothed_visible)

    def _changed_profile_lines(self, lines: dict[str, tuple[Point, Point]]) -> set[str]:
        if not self._last_profile_lines:
            changed = set(lines)
        else:
            changed = {name for name, line in lines.items() if not _same_line(line, self._last_profile_lines.get(name))}
        self._last_profile_lines = dict(lines)
        return changed

    def _auto_update_radiation_cursors(
        self,
        lines: dict[str, tuple[Point, Point]],
        line_names: set[str],
    ) -> None:
        if self.image is None:
            return
        for line_name in line_names:
            point_names = PROFILE_POINT_NAMES.get(line_name)
            line = lines.get(line_name)
            if point_names is None or line is None:
                continue
            if all(name in self.manually_adjusted_radiation_cursors for name in point_names):
                continue
            raw_values = None
            try:
                raw_values = line_profile(self.image, line[0], line[1])
                if line_name in ("top", "bottom"):
                    positions = np.linspace(line[0].x, line[1].x, raw_values.size)
                else:
                    positions = np.linspace(line[0].y, line[1].y, raw_values.size)
                positions, raw_values = _sort_profile(positions, raw_values)
                detection = detect_profile_boundaries(
                    positions,
                    raw_values,
                    self.film_pixel_spin.value(),
                    self.radiation_threshold_spin.value(),
                    self.radiation_center_spin.value(),
                    self.smoothing_window_spin.value(),
                )
            except ValueError as error:
                self._auto_detection_failures[line_name] = str(error)
                self._profile_display_markers[line_name] = self._profile_markers_without_detection(raw_values)
                for point_name in point_names:
                    if point_name not in self.manually_adjusted_radiation_cursors:
                        self.profile_cursors.pop(point_name, None)
                        self.radiation_points.pop(point_name, None)
                continue

            max_raw = float(np.max(raw_values)) if raw_values.size else 0.0
            self._profile_display_markers[line_name] = (
                max_raw - detection.smoothed_values,
                max_raw - detection.threshold_pixel_value,
                (float(detection.center_start_position), float(detection.center_end_position)),
            )
            self._auto_detection_failures.pop(line_name, None)
            self._auto_updating_profile_cursor = True
            try:
                for point_name, position in zip(point_names, (detection.left_position, detection.right_position), strict=True):
                    if point_name not in self.manually_adjusted_radiation_cursors:
                        self.profile_cursors[point_name] = position
            finally:
                self._auto_updating_profile_cursor = False
        self._update_radiation_auto_status()

    def _profile_markers_without_detection(self, raw_values: np.ndarray | None) -> tuple[np.ndarray | None, float | None, tuple[float, float] | None]:
        if raw_values is None:
            return None, None, None
        smoothed = moving_average_profile(raw_values, self.smoothing_window_spin.value())
        max_raw = float(np.max(raw_values)) if raw_values.size else 0.0
        return max_raw - smoothed, None, None

    def _update_circle_overlay(self) -> None:
        if self.image is None or not self.circle_options_visible:
            self.view.set_circle_overlay(None, None)
            return
        self.view.set_circle_overlay(
            Point(self.circle_x_spin.value(), self.circle_y_spin.value()),
            self.circle_radius_spin.value(),
        )

    def _set_circle_options_visible(self, visible: bool) -> None:
        self.circle_options_visible = visible
        self._update_circle_overlay()

    def _redetect_radiation_lines(self) -> None:
        if self.current_stage != "radiation" or self.image is None:
            return
        self._auto_update_radiation_cursors(self.view.profile_lines(), set(self.view.profile_lines()))
        self._on_profile_lines_changed(self.view.profile_lines())

    def _update_radiation_auto_status(self) -> None:
        if self._auto_detection_failures:
            names = ", ".join(sorted(self._auto_detection_failures))
            self.status.showMessage(f"Radiation boundary auto detection failed: {names}")
            return
        if self.current_stage == "radiation":
            self.status.showMessage("Radiation boundary points updated.")

    def calculate_circle_mean(self) -> None:
        if self.image is None:
            QMessageBox.warning(self, "No image", "Load an image before calculating a circle mean.")
            return
        try:
            value = circular_region_mean(
                self.image,
                Point(self.circle_x_spin.value(), self.circle_y_spin.value()),
                self.circle_radius_spin.value(),
            )
        except ValueError as error:
            QMessageBox.warning(self, "Circle mean failed", str(error))
            return
        self.circle_mean_label.setText(f"{value:.1f}")

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

    def _sync_laser_step_ui(self) -> None:
        if not hasattr(self, "laser_status_label"):
            return
        if self.laser_center is None:
            self.laser_status_label.setText("Set laser center.")
        else:
            self.laser_status_label.setText(f"Laser center: {_format_point(self.laser_center)}")
        if self.current_stage != "laser":
            return
        self._set_laser_profile_cursors()
        for line_name, plot in self.profile_plots.items():
            if line_name not in ("left", "bottom"):
                plot.set_selected(False)
                plot.set_cursors({})
                continue
            cursor_name = "laser_y" if line_name == "left" else "laser_x"
            if cursor_name not in self.profile_cursors:
                self._set_laser_profile_cursors()
            cursor_value = self.profile_cursors.get(cursor_name)
            if cursor_value is None:
                plot.set_cursors({})
                continue
            plot.set_selected(line_name == self.selected_profile_line)
            plot.set_cursors({cursor_name: cursor_value})
        self.view.set_profile_cursor_points({})
        self.view.set_laser_center(self.laser_center)

    def _update_laser_center_from_cursors(self) -> None:
        if self.laser_center is None:
            self.laser_center = self._default_laser_center()
        if self.laser_center is None:
            return
        x = self.profile_cursors.get("laser_x", self.laser_center.x)
        y = self.profile_cursors.get("laser_y", self.laser_center.y)
        self.laser_center = Point(x, y)
        self.view.set_laser_center(self.laser_center)
        self.view.set_profile_line_positions(bottom_y=y, left_x=x)

    def _sync_radiation_step_ui(self) -> None:
        if not hasattr(self, "radiation_status_label"):
            return
        self._update_profile_visibility()
        if self._auto_detection_failures:
            names = ", ".join(sorted(self._auto_detection_failures))
            self.radiation_status_label.setText(f"Auto detection failed: {names}")
        else:
            count = sum(1 for point_names in PROFILE_POINT_NAMES.values() for name in point_names if name in self.profile_cursors)
            self.radiation_status_label.setText(f"Radiation boundary points: {count}/8")
        if self.current_stage != "radiation":
            if self.current_stage != "laser":
                for plot in self.profile_plots.values():
                    plot.set_selected(False)
                    plot.set_cursors({})
                    plot.set_profile(None, None, ())
                self.view.set_profile_cursor_points({})
            return
        selected_line = self.selected_profile_line if self.current_stage == "radiation" else None
        preview_points: dict[str, Point] = {}
        for line_name, point_names in PROFILE_POINT_NAMES.items():
            plot = self.profile_plots[line_name]
            plot.set_selected(line_name == selected_line)
            cursors = {name: self.profile_cursors[name] for name in point_names if name in self.profile_cursors}
            plot.set_cursors(cursors)
            preview_points.update(self._profile_cursor_points(line_name, cursors))
        self.view.set_profile_cursor_points(preview_points)
        self._update_radiation_preview(preview_points)

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
        polygon = _radiation_polygon_from_points(self.radiation_points)
        if polygon is None:
            return False
        top_left, top_right, bottom_right, bottom_left = polygon
        self.radiation_polygon = polygon
        self.radiation_rect = _rect_from_ordered_points(top_left, top_right, bottom_right, bottom_left)
        self.view.set_radiation_rect(self.radiation_rect, reset_profile_lines=False)
        self.view.set_radiation_polygon(self.radiation_polygon, reset_profile_lines=False)
        return True

    def _update_radiation_preview(self, points: dict[str, Point]) -> None:
        polygon = _radiation_polygon_from_points(points)
        if polygon is None:
            return
        top_left, top_right, bottom_right, bottom_left = polygon
        self.radiation_polygon = polygon
        self.radiation_rect = _rect_from_ordered_points(top_left, top_right, bottom_right, bottom_left)
        self.view.set_radiation_polygon(
            self.radiation_polygon,
            reset_profile_lines=False,
            emit_profile_lines=False,
        )

    def _sync_result_center_points(self) -> None:
        if self.current_stage != "result":
            self.view.set_result_center_points({})
            return
        points: dict[str, Point] = {}
        if self.radiation_polygon is not None:
            points["radiation"] = _field_center(self.radiation_polygon)
        elif self.radiation_rect is not None:
            points["radiation"] = _field_center(self.radiation_rect.ordered_points())
        light_polygon = self._current_light_polygon()
        if light_polygon is not None:
            points["light"] = _field_center(light_polygon)
        self.view.set_result_center_points(points)

    def _on_visible_scene_rect_changed(self, scene_rect) -> None:
        for plot in self.profile_plots.values():
            plot.set_visible_range(None)

    def _update_result_label(self) -> None:
        if self.current_stage != "result":
            self._clear_result_tree()
            return
        if self.radiation_rect is None:
            self._clear_result_tree()
            return
        result = self._current_analysis_result()
        if result is None:
            self._clear_result_tree()
            return
        self._populate_result_tree(result)

    def _clear_result_tree(self) -> None:
        self.result_tree.clear()

    def _populate_result_tree(self, result: AnalysisResult) -> None:
        self.result_tree.clear()
        self.result_tree.setHeaderLabels(["Item", "Value"])
        area_unit = f"{result.unit}^2"
        _add_tree_item(self.result_tree, "Origin", f"{result.origin_field} center")
        _add_tree_item(self.result_tree, "Unit", result.unit if result.dpi <= 0 else f"{result.unit} (DPI {result.dpi:.1f})")
        if result.laser_center is not None:
            _add_tree_item(self.result_tree, "Laser Center", _format_point(result.laser_center))
        radiation_item = _add_field_tree(self.result_tree, "Radiation", result.radiation_field, area_unit)
        light_item = _add_field_tree(self.result_tree, "Light", result.light_field, area_unit)
        radiation_item.setExpanded(True)
        light_item.setExpanded(True)
        self.result_tree.resizeColumnToContents(0)

    def _current_analysis_result(self, origin_field: str | None = None, raw_pixels: bool = False) -> AnalysisResult | None:
        if self.radiation_rect is None:
            return None
        radiation_polygon = self.radiation_polygon or self.radiation_rect.ordered_points()
        light_polygon = self._current_light_polygon()
        if light_polygon is None:
            return None
        selected_origin = origin_field or self.origin_combo.currentData() or "laser"
        if selected_origin == "laser" and self.laser_center is None:
            return None
        if raw_pixels:
            selected_origin = "laser"
            origin_point = Point(0.0, 0.0)
            dpi = 0.0
        else:
            origin_point = self.laser_center if selected_origin == "laser" else None
            dpi = self.dpi_spin.value()
        return compare_field_polygons(
            radiation_polygon,
            light_polygon,
            self.radiation_rect.angle,
            self.light_rect.angle if self.light_rect is not None else 0.0,
            selected_origin,
            dpi,
            origin_point,
            self.laser_center,
        )

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


def _radiation_polygon_from_points(points: dict[str, Point]) -> tuple[Point, Point, Point, Point] | None:
    required = ("L1", "R1", "L2", "R2", "U1", "D1", "U2", "D2")
    if any(name not in points for name in required):
        return None
    left_line = (points["L1"], points["L2"])
    right_line = (points["R1"], points["R2"])
    top_line = (points["U1"], points["U2"])
    bottom_line = (points["D1"], points["D2"])
    top_left = _line_intersection(left_line[0], left_line[1], top_line[0], top_line[1])
    top_right = _line_intersection(right_line[0], right_line[1], top_line[0], top_line[1])
    bottom_right = _line_intersection(right_line[0], right_line[1], bottom_line[0], bottom_line[1])
    bottom_left = _line_intersection(left_line[0], left_line[1], bottom_line[0], bottom_line[1])
    if None in (top_left, top_right, bottom_right, bottom_left):
        return None
    return top_left, top_right, bottom_right, bottom_left


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
        "laser": "Laser Center",
        "radiation": "Radiation",
        "light": "Light",
        "result": "Result",
    }
    prefix = "✓ " if completed else ""
    return f"{prefix}{labels[stage]}"


def _stage_names() -> tuple[str, str, str, str]:
    return ("laser", "radiation", "light", "result")


def _record_display_row(row: dict[str, object], origin: str, dpi: float) -> dict[str, object]:
    scale = 25.4 / dpi if dpi > 0 else 1.0
    unit = "mm" if dpi > 0 else "px"
    laser = Point(float(row["laser_center_x_px"]), float(row["laser_center_y_px"]))
    radiation = Point(float(row["radiation_center_x_px"]), float(row["radiation_center_y_px"]))
    light = Point(float(row["light_center_x_px"]), float(row["light_center_y_px"]))
    origin_point = {
        "laser": laser,
        "radiation": radiation,
        "light": light,
    }.get(origin, laser)

    return {
        "created_at": row["created_at"],
        "image_path": row["image_path"],
        "origin": origin,
        "dpi": _round1(dpi) if dpi > 0 else 0.0,
        "unit": unit,
        "laser_center_x": _round1((laser.x - origin_point.x) * scale),
        "laser_center_y": _round1((laser.y - origin_point.y) * scale),
        "radiation_center_x": _round1((radiation.x - origin_point.x) * scale),
        "radiation_center_y": _round1((radiation.y - origin_point.y) * scale),
        "light_center_x": _round1((light.x - origin_point.x) * scale),
        "light_center_y": _round1((light.y - origin_point.y) * scale),
        "radiation_edge_length_x": _round1(float(row["radiation_edge_length_x_px"]) * scale),
        "radiation_edge_length_y": _round1(float(row["radiation_edge_length_y_px"]) * scale),
        "radiation_area": _round1(float(row["radiation_area_px2"]) * scale * scale),
        "light_area": _round1(float(row["light_area_px2"]) * scale * scale),
    }


def _round1(value: float) -> float:
    return round(float(value), 1)


def _button(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setMaximumWidth(120)
    button.setMinimumWidth(72)
    button.setFocusPolicy(Qt.NoFocus)
    return button


def _nav_button(text: str) -> QPushButton:
    button = _button(text)
    button.setMinimumWidth(96)
    button.setMinimumHeight(32)
    button.setMaximumWidth(150)
    return button


def _group_style() -> str:
    return "QGroupBox { background: #f3f3f3; border: 1px solid #b8b8b8; border-radius: 4px; margin-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"


def _selected_group_style() -> str:
    return "QGroupBox { background: #dfe9f6; border: 1px solid #5f8fc4; border-radius: 4px; margin-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"


def _sort_profile(positions: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(positions)
    return positions[order], values[order]


def _same_line(
    line: tuple[Point, Point],
    other: tuple[Point, Point] | None,
) -> bool:
    if other is None:
        return False
    return (
        np.isclose(line[0].x, other[0].x)
        and np.isclose(line[0].y, other[0].y)
        and np.isclose(line[1].x, other[1].x)
        and np.isclose(line[1].y, other[1].y)
    )


def _add_field_tree(parent: QTreeWidget | QTreeWidgetItem, label: str, field: FieldGeometry, area_unit: str) -> QTreeWidgetItem:
    vertex_labels = ("TL", "TR", "BR", "BL")
    edge_labels = ("top", "right", "bottom", "left")
    field_item = _add_tree_item(parent, label)
    _add_tree_item(field_item, "Center", _format_point(field.center))
    _add_tree_item(field_item, "Area Length X", f"{field.area_length_x:.1f}")
    _add_tree_item(field_item, "Area Length Y", f"{field.area_length_y:.1f}")
    _add_tree_item(field_item, "Average Edge", f"{field.average_edge_length:.1f}")
    _add_tree_item(field_item, "Area", f"{field.area:.1f} {area_unit}")

    vertices_item = _add_tree_item(field_item, "Vertices")
    for name, point in zip(vertex_labels, field.points, strict=True):
        _add_tree_item(vertices_item, name, _format_point(point))
    vertices_item.setExpanded(False)

    edge_item = _add_tree_item(field_item, "Edge Lengths")
    for name, length in zip(edge_labels, field.edge_lengths, strict=True):
        _add_tree_item(edge_item, name, f"{length:.1f}")
    edge_item.setExpanded(False)
    return field_item


def _add_tree_item(parent: QTreeWidget | QTreeWidgetItem, label: str, value: str = "") -> QTreeWidgetItem:
    item = QTreeWidgetItem([label, value])
    if isinstance(parent, QTreeWidget):
        parent.addTopLevelItem(item)
    else:
        parent.addChild(item)
    return item


def _polygon_center(points: tuple[Point, Point, Point, Point]) -> Point:
    return Point(
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def _field_center(points: tuple[Point, Point, Point, Point]) -> Point:
    left_mid, right_mid, top_mid, bottom_mid = _edge_length_points(points)
    return _line_intersection(left_mid, right_mid, top_mid, bottom_mid) or _polygon_center(points)


def _edge_length_points(points: tuple[Point, Point, Point, Point]) -> tuple[Point, Point, Point, Point]:
    return (
        _midpoint(points[3], points[0]),
        _midpoint(points[1], points[2]),
        _midpoint(points[0], points[1]),
        _midpoint(points[2], points[3]),
    )


def _midpoint(start: Point, end: Point) -> Point:
    return Point((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)


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
    return f"({point.x:.1f}, {point.y:.1f})"


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


if __name__ == "__main__":
    raise SystemExit(main())
