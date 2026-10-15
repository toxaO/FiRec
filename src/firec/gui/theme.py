"""FiStar-inspired styling for the FiRec desktop interface."""

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QProxyStyle, QStyle, QStyleOptionButton, QToolTip


class HoverHelpTextFilter(QObject):
    """Limit checkbox and radio help to their text, excluding empty space."""

    def eventFilter(self, widget, event):
        if event.type() == QEvent.ToolTip:
            option = QStyleOptionButton()
            widget.initStyleOption(option)
            element = QStyle.SE_CheckBoxContents if isinstance(widget, QCheckBox) else QStyle.SE_RadioButtonContents
            contents = widget.style().subElementRect(element, option, widget)
            text_rect = widget.style().itemTextRect(
                widget.fontMetrics(), contents, Qt.AlignLeft | Qt.AlignVCenter,
                widget.isEnabled(), widget.text(),
            )
            if not text_rect.contains(event.pos()):
                QToolTip.hideText()
                event.accept()
                return True
        return super().eventFilter(widget, event)


class HoverHelpLabel(QLabel):
    """Show field help only over the rendered label text."""

    def event(self, event):
        if event.type() == QEvent.ToolTip:
            text_rect = self.style().itemTextRect(
                self.fontMetrics(), self.contentsRect(), self.alignment(),
                self.isEnabled(), self.text(),
            )
            if not text_rect.contains(event.pos()):
                QToolTip.hideText()
                event.accept()
                return True
        return super().event(event)


class HoverHelpStyle(QProxyStyle):
    """Wait before showing help, including when moving between controls."""

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_ToolTip_WakeUpDelay:
            return 1000
        if hint == QStyle.SH_ToolTip_FallAsleepDelay:
            return 0
        return super().styleHint(hint, option, widget, returnData)

STYLE = """
QWidget { color: #253b49; font-size: 13px; }
QMainWindow, QDialog { background: #f3f6f8; }
QTabWidget::pane { border: 0; }
QTabBar::tab { background: #e5edf2; padding: 9px 22px; border: 0; }
QTabBar::tab:selected { background: #f3f6f8; color: #087b87; font-weight: 600; }
QTabBar::tab:hover { color: #087b87; }
QGroupBox { background: #ffffff; border: 1px solid #dce5eb; border-radius: 7px;
            margin-top: 14px; padding: 10px 6px 6px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QFrame[card="true"] { background: #ffffff; border: 1px solid #dce5eb; border-radius: 7px; }
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
    background: #ffffff; border: 1px solid #cbd8e0; border-radius: 4px;
    min-height: 24px; padding: 3px 6px; selection-background-color: #087f8c;
}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus { border-color: #0794a3; }
QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {
    background: #eef2f5; color: #7b8e9a; border-color: #dce5eb;
}
QComboBox { padding-right: 28px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right;
                      border: 0; border-left: 1px solid #dce5eb; width: 24px; background: #eef4f7;
                      border-top-right-radius: 4px; border-bottom-right-radius: 4px; }
QComboBox::down-arrow { image: url("@DOWN_ARROW@"); width: 12px; height: 8px; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 16px; }
QPushButton { background: #eef4f7; border: 1px solid #cbd8e0; border-radius: 5px;
              padding: 6px 10px; min-height: 20px; }
QPushButton:hover { background: #e3eef3; border-color: #91b9c5; }
QPushButton:pressed { background: #d6e8ed; }
QPushButton[primary="true"] { background: #087f8c; border-color: #087f8c; color: #ffffff; font-weight: 600; }
QPushButton[primary="true"]:hover { background: #066c78; }
QPushButton:disabled { background: #e9eef1; border-color: #dce5eb; color: #85949e; }
QToolButton { border: 1px solid transparent; border-radius: 4px; padding: 3px; }
QToolButton:hover, QToolButton:checked { background: #e1eff2; border-color: #a8ced6; }
QLabel[stageHeading="true"] { font-weight: 600; }
QToolButton[helpButton="true"], QToolButton[helpButton="true"]:hover,
QToolButton[helpButton="true"]:pressed { background: transparent; border-color: transparent; }
QToolButton[imageTool="true"] { background: #ffffff; border: 1px solid #b9cbd5; }
QToolButton[imageTool="true"]:hover { background: #e1eff2; border-color: #78b4c1; }
QToolButton[imageTool="true"]:checked { background: #d6edf1; border-color: #087f8c; }
QToolButton[imageTool="true"]:disabled { background: #f1f4f6; border-color: #dce5eb; }
QScrollArea { border: 0; background: transparent; }
QTreeWidget, QTableWidget { background: #ffffff; alternate-background-color: #f3f7fa;
    border: 1px solid #dce5eb; border-radius: 4px;
    selection-background-color: #d6edf1; selection-color: #253b49; gridline-color: #e5edf2; }
QTreeWidget::item, QTableWidget::item { padding: 4px; }
QHeaderView::section { background: #eaf1f5; padding: 6px; border: 0; border-bottom: 1px solid #dce5eb; }
QCheckBox, QRadioButton { spacing: 5px; }
QLabel[muted="true"] { color: #516978; font-size: 12px; }
QLabel:disabled, QCheckBox:disabled, QRadioButton:disabled { color: #85949e; }
QStatusBar { background: #eaf0f4; color: #647b89; }
QStatusBar::item { border: 0; }
QToolTip { background: #253b49; color: #ffffff; border: 0; padding: 5px; }
""".replace("@DOWN_ARROW@", (Path(__file__).parent / "assets/icons/chevron_down.svg").as_posix())

GROUP_STYLE = """
QGroupBox { background: #ffffff; border: 1px solid #dce5eb; border-radius: 7px;
            margin-top: 14px; padding: 10px 6px 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
"""

SELECTED_GROUP_STYLE = GROUP_STYLE + """
QGroupBox { background: #f2fafb; border-color: #0794a3; }
QGroupBox::title { color: #087b87; }
"""


def stage_card_style(state: str, selected: bool) -> str:
    background = {"complete": "#edf8f2", "incomplete": "#fff8ed", "stale": "#fff0dc"}[state]
    border = "#087f8c" if selected else "#dce5eb"
    return f"""
    QGroupBox#analysisStep {{ background: {background}; border: 2px solid {border};
        border-radius: 7px; margin-top: 12px; padding: 8px 6px 6px;
        font-size: 15px; font-weight: 600; }}
    QGroupBox#analysisStep::title {{ subcontrol-origin: margin; left: 10px;
        padding: 0 5px; }}
    """
