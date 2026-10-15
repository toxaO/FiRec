"""Square image area with aligned profiles and flexible outer space."""

from PySide6.QtWidgets import QWidget


class SquareImagePanel(QWidget):
    def __init__(self, view, profiles):
        super().__init__()
        self.view = view
        self.profiles = profiles
        self.setMinimumSize(212, 212)
        for widget in (view, *profiles.values()):
            widget.setParent(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        thickness, gap = 40, 3
        inset = thickness + gap
        side = max(0, min(self.width(), self.height()) - 2 * inset)
        x = (self.width() - side) // 2
        y = (self.height() - side) // 2
        self.view.setGeometry(x, y, side, side)
        self.profiles["left"].setGeometry(x - inset, y, thickness, side)
        self.profiles["right"].setGeometry(x + side + gap, y, thickness, side)
        self.profiles["top"].setGeometry(x, y - inset, side, thickness)
        self.profiles["bottom"].setGeometry(x, y + side + gap, side, thickness)
