from PySide6.QtWidgets import (
    QFontDialog,
    QInputDialog,
    QFileDialog,
    QMessageBox,
    QWidget,
)
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
    ) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
