from PySide6.QtWidgets import (
    QFontDialog,
    QInputDialog,
    QFileDialog,
    QMessageBox,
    QWidget,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
)
from PySide6.QtCore import Qt, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Callable


class CustomNavigationToolbar(NavigationToolbar2QT):
    def __init__(
        self,
        canvas: FigureCanvasQTAgg,
        parent: QWidget,
        coordinates: bool = True,
        color_callback: Optional[Callable[[], None]] = None,
        export_data_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(canvas, parent, coordinates)
        self.addSeparator()
        self.addAction("Font", self.change_font)
        self.addAction("Export", self.export_figure)
        if color_callback:
            self.addAction("Colors", color_callback)
        if export_data_callback:
            self.addAction("Export Data", export_data_callback)

    def export_figure(self) -> None:
        dpi, ok = QInputDialog.getInt(
            self, "Export Settings", "DPI:", value=300, minValue=72, maxValue=1200
        )
        if not ok:
            return
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if fname:
            try:
                self.canvas.figure.savefig(fname, dpi=dpi, bbox_inches="tight")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save figure: {e}")

    def change_font(self) -> None:
        ok, font = QFontDialog.getFont(self)
        if ok:
            size = font.pointSize()
            family = font.family()

            # Update rcParams for future plots
            plt.rcParams.update(
                {
                    "font.size": size,
                    "font.family": family,
                    "axes.labelsize": size,
                    "axes.titlesize": size + 2,
                    "xtick.labelsize": size,
                    "ytick.labelsize": size,
                    "legend.fontsize": size,
                }
            )

            # Update current figure elements
            for ax in self.canvas.figure.axes:
                for item in (
                    [ax.title, ax.xaxis.label, ax.yaxis.label]
                    + ax.get_xticklabels()
                    + ax.get_yticklabels()
                ):
                    item.set_fontsize(size)
                    item.set_fontfamily(family)

                legend = ax.get_legend()
                if legend:
                    for text in legend.get_texts():
                        text.set_fontsize(size)
                        text.set_fontfamily(family)

            self.canvas.draw()


class MplCanvas(FigureCanvasQTAgg):
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        width: float = 5,
        height: float = 4,
        dpi: int = 100,
        is_3d: bool = False,
    ) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        if is_3d:
            self.axes = self.fig.add_subplot(111, projection="3d")
        else:
            self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class Plot3DControls(QWidget):
    view_changed = Signal(int, int, int)
    xlim_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.init_ui()

    def init_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        def create_control_row(slider: QSlider, spin: QSpinBox) -> QWidget:
            w = QWidget()
            l = QHBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            l.addWidget(slider)
            l.addWidget(spin)
            return w

        self.slider_elev = QSlider(Qt.Orientation.Horizontal)
        self.spin_elev = QSpinBox()
        self.slider_elev.setRange(0, 90)
        self.spin_elev.setRange(0, 90)
        self.slider_elev.setValue(30)
        self.spin_elev.setValue(30)
        self.slider_elev.valueChanged.connect(self.spin_elev.setValue)
        self.spin_elev.valueChanged.connect(self.slider_elev.setValue)
        self.slider_elev.valueChanged.connect(self._emit_view_changed)
        layout.addRow("Elevation:", create_control_row(self.slider_elev, self.spin_elev))

        self.slider_azim = QSlider(Qt.Orientation.Horizontal)
        self.spin_azim = QSpinBox()
        self.slider_azim.setRange(-180, 180)
        self.spin_azim.setRange(-180, 180)
        self.slider_azim.setValue(-60)
        self.spin_azim.setValue(-60)
        self.slider_azim.valueChanged.connect(self.spin_azim.setValue)
        self.spin_azim.valueChanged.connect(self.slider_azim.setValue)
        self.slider_azim.valueChanged.connect(self._emit_view_changed)
        layout.addRow("Azimuth:", create_control_row(self.slider_azim, self.spin_azim))

        self.slider_roll = QSlider(Qt.Orientation.Horizontal)
        self.spin_roll = QSpinBox()
        self.slider_roll.setRange(-180, 180)
        self.spin_roll.setRange(-180, 180)
        self.slider_roll.setValue(0)
        self.spin_roll.setValue(0)
        self.slider_roll.valueChanged.connect(self.spin_roll.setValue)
        self.spin_roll.valueChanged.connect(self.slider_roll.setValue)
        self.slider_roll.valueChanged.connect(self._emit_view_changed)
        layout.addRow("Roll:", create_control_row(self.slider_roll, self.spin_roll))

        self.spin_xlim_min = QDoubleSpinBox()
        self.spin_xlim_min.setRange(-100, 100)
        self.spin_xlim_min.setValue(5.5)
        self.spin_xlim_min.setDecimals(2)
        self.spin_xlim_min.setSingleStep(0.1)
        self.spin_xlim_min.valueChanged.connect(self.xlim_changed.emit)

        self.spin_xlim_max = QDoubleSpinBox()
        self.spin_xlim_max.setRange(-100, 100)
        self.spin_xlim_max.setValue(11.0)
        self.spin_xlim_max.setDecimals(2)
        self.spin_xlim_max.setSingleStep(0.1)
        self.spin_xlim_max.valueChanged.connect(self.xlim_changed.emit)

        xlim_widget = QWidget()
        xlim_layout = QHBoxLayout(xlim_widget)
        xlim_layout.setContentsMargins(0, 0, 0, 0)
        xlim_layout.addWidget(self.spin_xlim_min)
        xlim_layout.addWidget(QLabel("to"))
        xlim_layout.addWidget(self.spin_xlim_max)
        layout.addRow("X-Limit (ppm):", xlim_widget)

    def _emit_view_changed(self) -> None:
        self.view_changed.emit(self.slider_elev.value(), self.slider_azim.value(), self.slider_roll.value())

    def get_view_params(self) -> tuple[int, int, int]:
        return self.slider_elev.value(), self.slider_azim.value(), self.slider_roll.value()

    def get_xlim(self) -> tuple[float, float]:
        return self.spin_xlim_min.value(), self.spin_xlim_max.value()
