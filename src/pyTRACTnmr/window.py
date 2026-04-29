import os
import csv
import importlib.resources
import numpy as np
from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QSplitter,
    QTabWidget,
    QMessageBox,
    QFormLayout,
    QTableWidgetItem,
    QMenu,
    QColorDialog,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
)
from PySide6.QtGui import QAction, QDesktopServices, QDragEnterEvent, QDropEvent, QColor
from PySide6.QtCore import Qt, QPoint, QUrl, QThread

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import SpanSelector

try:
    from .widgets import MplCanvas, CustomNavigationToolbar, Plot3DControls
    from .data_panel import ExperimentPanel
    from .analysis_panel import ProcessingTab, FittingTab
    from . import processing
    from .workers import SlidingWindowWorker
    from . import exporters
except ImportError:
    from widgets import MplCanvas, CustomNavigationToolbar, Plot3DControls
    from data_panel import ExperimentPanel
    from analysis_panel import ProcessingTab, FittingTab
    import processing
    from workers import SlidingWindowWorker
    import exporters


class TractApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRACT Analysis GUI")
        self.resize(1200, 800)
        self.setAcceptDrops(True)

        # Data State
        self.datasets: List[Dict[str, Any]] = []
        self.current_idx: int = -1
        self.selector: Optional[SpanSelector] = None
        self.baseline_nodes: List[float] = []
        self.picking_baseline: bool = False

        # Plot colors
        self.fill_color_alpha = "blue"
        self.fill_color_beta = "red"
        self.fill_color_sliding = "blue"

        self.init_ui()

    def init_ui(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- Panel 1: Data Loading ---
        self.panel_experiment = ExperimentPanel()
        self.panel_experiment.load_clicked.connect(self.load_data)
        self.panel_experiment.load_demo_clicked.connect(self.load_demo_data)
        self.panel_experiment.help_clicked.connect(self.open_help)

        # Connect table signals
        self.panel_experiment.table_data.cellDoubleClicked.connect(
            self.on_table_double_click
        )
        self.panel_experiment.table_data.itemChanged.connect(self.on_table_item_changed)
        self.panel_experiment.table_data.customContextMenuRequested.connect(
            self.show_context_menu
        )

        # --- Panel 2: Visualization ---
        splitter_center = QSplitter(Qt.Orientation.Vertical)

        # Top: Spectrum
        self.tabs_spectrum = QTabWidget()

        # Tab 1: Phase Check
        self.canvas_spec = MplCanvas(self)
        self.toolbar_spec = CustomNavigationToolbar(self.canvas_spec, self)
        self.canvas_spec.mpl_connect("button_press_event", self.on_canvas_click)
        self.canvas_spec.mpl_connect("scroll_event", self.on_scroll)

        widget_phase = QWidget()
        layout_phase = QVBoxLayout()
        layout_phase.addWidget(self.toolbar_spec)
        layout_phase.addWidget(self.canvas_spec)
        widget_phase.setLayout(layout_phase)
        self.tabs_spectrum.addTab(widget_phase, "Phase Check")

        # Tab 2: Alpha Stack
        self.canvas_alpha = MplCanvas(self, is_3d=True)
        self.toolbar_alpha = CustomNavigationToolbar(self.canvas_alpha, self)
        self.canvas_alpha.mpl_connect("scroll_event", self.on_scroll)
        widget_alpha = QWidget()
        layout_alpha = QVBoxLayout()
        layout_alpha.addWidget(self.toolbar_alpha)
        layout_alpha.addWidget(self.canvas_alpha)

        self.controls_alpha = Plot3DControls()
        self.controls_alpha.view_changed.connect(self.update_alpha_view)
        self.controls_alpha.xlim_changed.connect(self.plot_stacked_spectra)
        layout_alpha.addWidget(self.controls_alpha)

        widget_alpha.setLayout(layout_alpha)
        self.tabs_spectrum.addTab(widget_alpha, "Alpha Stack")

        # Tab 3: Beta Stack
        self.canvas_beta = MplCanvas(self, is_3d=True)
        self.toolbar_beta = CustomNavigationToolbar(self.canvas_beta, self)
        self.canvas_beta.mpl_connect("scroll_event", self.on_scroll)
        widget_beta = QWidget()
        layout_beta = QVBoxLayout()
        layout_beta.addWidget(self.toolbar_beta)
        layout_beta.addWidget(self.canvas_beta)

        self.controls_beta = Plot3DControls()
        self.controls_beta.view_changed.connect(self.update_beta_view)
        self.controls_beta.xlim_changed.connect(self.plot_stacked_spectra)
        layout_beta.addWidget(self.controls_beta)

        widget_beta.setLayout(layout_beta)
        self.tabs_spectrum.addTab(widget_beta, "Beta Stack")

        self.tabs_spectrum.setTabEnabled(1, False)
        self.tabs_spectrum.setTabEnabled(2, False)

        # Bottom: Fits
        self.tabs_results = QTabWidget()

        # Tab 1: Relaxation Fits
        self.canvas_fit = MplCanvas(self)
        self.toolbar_fit = CustomNavigationToolbar(
            self.canvas_fit, self, color_callback=self.pick_fit_colors
        )
        widget_fit_std = QWidget()
        layout_fit = QVBoxLayout()
        # lbl_fit = QLabel("<b>Relaxation Fits</b>")
        # lbl_fit.setFixedHeight(30)
        # layout_fit.addWidget(lbl_fit)
        layout_fit.addWidget(self.toolbar_fit)
        layout_fit.addWidget(self.canvas_fit)
        widget_fit_std.setLayout(layout_fit)
        self.tabs_results.addTab(widget_fit_std, "Relaxation Fits")

        # Tab 2: Sliding Window Analysis
        self.canvas_sliding = MplCanvas(self)
        self.toolbar_sliding = CustomNavigationToolbar(
            self.canvas_sliding, self, color_callback=self.pick_sliding_colors
        )
        widget_fit_slide = QWidget()
        layout_fit_slide = QVBoxLayout()
        layout_fit_slide.addWidget(self.toolbar_sliding)
        layout_fit_slide.addWidget(self.canvas_sliding)
        widget_fit_slide.setLayout(layout_fit_slide)
        self.tabs_results.addTab(widget_fit_slide, "Sliding Window")

        widget_results = QWidget()
        layout_results = QVBoxLayout()
        lbl_results = QLabel("<b>Analysis Results</b>")
        lbl_results.setFixedHeight(30)
        layout_results.addWidget(lbl_results)
        layout_results.addWidget(self.tabs_results)
        widget_results.setLayout(layout_results)

        splitter_center.addWidget(self.tabs_spectrum)
        splitter_center.addWidget(widget_results)

        # --- Panel 3: Controls ---
        panel3 = QTabWidget()

        # Tab 1: Processing
        self.tab_processing = ProcessingTab()
        self.tab_processing.param_changed.connect(self.process_data)
        self.tab_processing.pick_nodes_toggled.connect(self.toggle_picking)
        self.tab_processing.clear_nodes_clicked.connect(self.clear_baseline)
        self.tab_processing.input_p0.editingFinished.connect(
            self.update_phase_from_text
        )
        self.tab_processing.input_p1.editingFinished.connect(
            self.update_phase_from_text
        )

        # Tab 2: Fitting
        self.tab_fitting = FittingTab()
        self.tab_fitting.fit_requested.connect(self.run_fitting)
        self.tab_fitting.export_fit_requested.connect(self.export_fit_data)
        self.tab_fitting.export_sliding_requested.connect(self.export_sliding_data)
        self.tab_fitting.export_report_requested.connect(self.export_pdf_report)

        panel3.addTab(self.tab_processing, "Processing")
        panel3.addTab(self.tab_fitting, "Fitting")

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(self.panel_experiment)
        main_splitter.addWidget(splitter_center)
        main_splitter.addWidget(panel3)
        main_splitter.setSizes([400, 500, 300])
        main_layout.addWidget(main_splitter)

    def toggle_picking(self, checked: bool) -> None:
        self.picking_baseline = checked
        self.process_data()

    def clear_baseline(self) -> None:
        self.baseline_nodes = []
        self.process_data()

    def on_canvas_click(self, event) -> None:
        if not self.picking_baseline or event.inaxes != self.canvas_spec.axes:
            return
        if event.button == 1 and event.xdata is not None:
            self.baseline_nodes.append(event.xdata)
            self.process_data()

    def update_alpha_view(self, elev: int, azim: int, roll: int) -> None:
        if hasattr(self.canvas_alpha.axes, "view_init"):
            self.canvas_alpha.axes.view_init(elev=elev, azim=azim, roll=roll)
            self.canvas_alpha.draw()

    def update_beta_view(self, elev: int, azim: int, roll: int) -> None:
        if hasattr(self.canvas_beta.axes, "view_init"):
            self.canvas_beta.axes.view_init(elev=elev, azim=azim, roll=roll)
            self.canvas_beta.draw()

    def on_scroll(self, event) -> None:
        ax = None
        canvas = None
        if event.inaxes == self.canvas_spec.axes:
            ax = self.canvas_spec.axes
            canvas = self.canvas_spec
        elif event.inaxes == self.canvas_alpha.axes:
            ax = self.canvas_alpha.axes
            canvas = self.canvas_alpha
        elif event.inaxes == self.canvas_beta.axes:
            ax = self.canvas_beta.axes
            canvas = self.canvas_beta

        if ax is None or hasattr(ax, "get_zlim"):  # Ignore 3D plots
            return

        ymin, ymax = ax.get_ylim()
        mouse_y = event.ydata

        # Zoom factor
        factor = 0.8 if event.button == "up" else 1.2

        new_ymin = mouse_y + (ymin - mouse_y) * factor
        new_ymax = mouse_y + (ymax - mouse_y) * factor

        ax.set_ylim(new_ymin, new_ymax)
        canvas.draw()

    def update_phase_from_text(self) -> None:
        try:
            p0 = float(self.tab_processing.input_p0.text())
            p1 = float(self.tab_processing.input_p1.text())

            self.tab_processing.slider_p0_coarse.blockSignals(True)
            self.tab_processing.slider_p0_fine.blockSignals(True)
            self.tab_processing.slider_p1_coarse.blockSignals(True)
            self.tab_processing.slider_p1_fine.blockSignals(True)

            self.tab_processing.slider_p0_coarse.setValue(int(p0))
            self.tab_processing.slider_p0_fine.setValue(int(round((p0 - int(p0)) * 10)))

            self.tab_processing.slider_p1_coarse.setValue(int(p1))
            self.tab_processing.slider_p1_fine.setValue(int(round((p1 - int(p1)) * 10)))

            self.tab_processing.slider_p0_coarse.blockSignals(False)
            self.tab_processing.slider_p0_fine.blockSignals(False)
            self.tab_processing.slider_p1_coarse.blockSignals(False)
            self.tab_processing.slider_p1_fine.blockSignals(False)

            self.process_data()
        except ValueError:
            pass

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                self.load_experiment(path)

    def load_data(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Bruker Directory")
        if folder:
            self.load_experiment(folder)

    def load_demo_data(self) -> None:
        try:
            # Use importlib.resources to get a path to the bundled demo data.
            # This is the modern approach (Python 3.9+) and handles packaged apps (e.g., in zip files).
            demo_exp_resource = importlib.resources.files("pyTRACTnmr").joinpath(
                "demo_data/1"
            )

            with importlib.resources.as_file(demo_exp_resource) as demo_path:
                if demo_path.is_dir():
                    self.load_experiment(str(demo_path), exp_name="DRB2 dsRBD1")
                    self.statusBar().showMessage("Loaded demo data", 5000)
                else:
                    # This case should ideally not happen if packaging is correct
                    QMessageBox.critical(
                        self,
                        "Demo Data Error",
                        f"Demo data resource found, but it is not a directory: {demo_path}",
                    )

        except (ModuleNotFoundError, FileNotFoundError) as e:
            QMessageBox.critical(
                self,
                "Demo Data Error",
                f"Could not find the demo dataset. It may not be installed correctly.\nError: {e}",
            )

    def load_experiment(self, folder: str, exp_name: Optional[str] = None) -> None:
        try:
            delay_list = None
            if not os.path.exists(os.path.join(folder, "vdlist")):
                QMessageBox.information(
                    self,
                    "Delay List Missing",
                    "The standard 'vdlist' file was not found. Please select a delay list file manually.",
                )
                delay_list, _ = QFileDialog.getOpenFileName(
                    self, "Select delay list file", folder
                )

            tb = processing.TractBruker(folder, delay_list=delay_list)
            name = exp_name if exp_name else os.path.basename(folder)
            dataset = {
                "name": name,
                "path": folder,
                "handler": tb,
                "p0": tb.phc0,
                "p1": tb.phc1,
            }
            self.datasets.append(dataset)
            self.update_table()
            self.switch_dataset(len(self.datasets) - 1)
            self.statusBar().showMessage(f"Loaded experiment: {name}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def process_data(self) -> None:
        if self.current_idx < 0:
            return
        try:
            # Capture current zoom level
            xlim = self.canvas_spec.axes.get_xlim()
            ylim = self.canvas_spec.axes.get_ylim()
            has_zoom = len(self.canvas_spec.axes.lines) > 0

            p0 = self.tab_processing.slider_p0_coarse.value() + (
                self.tab_processing.slider_p0_fine.value() / 10.0
            )
            p1 = self.tab_processing.slider_p1_coarse.value() + (
                self.tab_processing.slider_p1_fine.value() / 10.0
            )

            self.tab_processing.input_p0.setText(f"{p0:.1f}")
            self.tab_processing.input_p1.setText(f"{p1:.1f}")

            points = int(self.tab_processing.input_points.text())
            apod_func = (
                "sp"
                if self.tab_processing.combo_apod.currentText().startswith("Sine")
                else "em"
            )
            lb = float(self.tab_processing.input_lb.text())

            off = float(self.tab_processing.input_off.text())
            end = float(self.tab_processing.input_end.text())
            pow_val = float(self.tab_processing.input_pow.text())

            if self.current_idx >= 0:
                self.datasets[self.current_idx]["p0"] = p0
                self.datasets[self.current_idx]["p1"] = p1

            tb = self.datasets[self.current_idx]["handler"]

            nodes_idx = []
            if self.baseline_nodes and tb.unit_converter:
                for ppm in self.baseline_nodes:
                    idx = tb.unit_converter(ppm, "ppm")
                    nodes_idx.append(int(idx))
                nodes_idx.sort()

            trace = tb.process_first_trace(
                p0,
                p1,
                points=points,
                apod_func=apod_func,
                lb=lb,
                off=off,
                end=end,
                pow=pow_val,
                nodes=nodes_idx,
            )

            # Cache the processed trace
            self.datasets[self.current_idx]["processed_trace"] = trace
            if tb.unit_converter:
                self.datasets[self.current_idx]["ppm_scale"] = (
                    tb.unit_converter.ppm_scale()
                )

            self.canvas_spec.axes.clear()
            if tb.unit_converter:
                ppm_scale = tb.unit_converter.ppm_scale()
                self.canvas_spec.axes.plot(ppm_scale, trace, label="First Plane")
                self.canvas_spec.axes.invert_xaxis()
                self.canvas_spec.axes.set_xlabel(r"$^{1}H (ppm)$")
                self.canvas_spec.axes.set_ylabel("Intensity")
            else:
                self.canvas_spec.axes.plot(trace, label="First Plane")
            self.canvas_spec.axes.legend()

            for node in self.baseline_nodes:
                self.canvas_spec.axes.axvline(
                    x=node, color="r", linestyle="--", alpha=0.5
                )

            if has_zoom:
                self.canvas_spec.axes.set_xlim(xlim)
                self.canvas_spec.axes.set_ylim(ylim)

            if not self.picking_baseline:
                self.selector = SpanSelector(
                    self.canvas_spec.axes,
                    self.on_span_select,
                    "horizontal",
                    useblit=True,
                    props=dict(alpha=0.2, facecolor="green"),
                    interactive=True,
                    drag_from_anywhere=True,
                )
            else:
                self.selector = None

            if self.selector:
                try:
                    s = float(self.tab_processing.input_int_start.text())
                    e = float(self.tab_processing.input_int_end.text())
                    self.selector.extents = (min(s, e), max(s, e))
                except ValueError:
                    pass

            self.canvas_spec.draw()
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", str(e))

    def run_fitting(self) -> None:
        if self.current_idx < 0:
            return
        try:
            tb = self.datasets[self.current_idx]["handler"]
            p0 = self.tab_processing.slider_p0_coarse.value() + (
                self.tab_processing.slider_p0_fine.value() / 10.0
            )
            p1 = self.tab_processing.slider_p1_coarse.value() + (
                self.tab_processing.slider_p1_fine.value() / 10.0
            )

            points = int(self.tab_processing.input_points.text())
            apod_func = (
                "sp"
                if self.tab_processing.combo_apod.currentText().startswith("Sine")
                else "em"
            )
            lb = float(self.tab_processing.input_lb.text())

            off = float(self.tab_processing.input_off.text())
            end_param = float(self.tab_processing.input_end.text())
            pow_val = float(self.tab_processing.input_pow.text())
            start_ppm = float(self.tab_processing.input_int_start.text())
            end_ppm = float(self.tab_processing.input_int_end.text())

            # Update physics constants
            tb.CSA_15N = float(self.tab_fitting.input_csa.text()) * 1e-6
            tb.CSA_BOND_ANGLE = float(self.tab_fitting.input_angle.text()) * np.pi / 180
            s2_val = float(self.tab_fitting.input_s2.text())

            try:
                n_boot = int(self.tab_fitting.input_bootstraps.text())
                if n_boot < 10:
                    n_boot = 10
                    self.tab_fitting.input_bootstraps.setText("10")
            except ValueError:
                n_boot = 1000
                self.tab_fitting.input_bootstraps.setText("1000")

            nodes_idx = []
            if self.baseline_nodes and tb.unit_converter:
                for ppm in self.baseline_nodes:
                    idx = tb.unit_converter(ppm, "ppm")
                    nodes_idx.append(int(idx))
                nodes_idx.sort()

            tb.split_process(
                p0,
                p1,
                points=points,
                apod_func=apod_func,
                lb=lb,
                off=off,
                end=end_param,
                pow=pow_val,
                nodes=nodes_idx,
            )

            b0 = (
                float(self.tab_fitting.input_field.text())
                if self.tab_fitting.input_field.text()
                else None
            )

            tb.integrate_ppm(start_ppm, end_ppm)
            tb.calc_relaxation()

            tb.calc_tc(B0=b0, S2=s2_val, n_bootstrap=n_boot)

            self.plot_stacked_spectra()

            x, y_a, y_b, popt_a, popt_b, pcov_a, pcov_b = tb.get_fit_data()

            self.canvas_fit.axes.clear()
            self.canvas_fit.axes.plot(x, y_a, "bo", label=r"$\alpha -spin\ state$")
            self.canvas_fit.axes.plot(x, y_b, "ro", label=r"$\beta -spin\ state$")

            # Smooth lines for fit and CI
            x_smooth = np.linspace(0, np.max(x) * 1.1, 100)
            fit_a = processing.TractBruker._relax(x_smooth, *popt_a)
            ci_a = tb.calc_confidence_interval(x_smooth, popt_a, pcov_a)
            fit_b = processing.TractBruker._relax(x_smooth, *popt_b)
            ci_b = tb.calc_confidence_interval(x_smooth, popt_b, pcov_b)

            self.canvas_fit.axes.plot(x_smooth, fit_a, "b-")
            self.canvas_fit.axes.fill_between(
                x_smooth,
                fit_a - ci_a,
                fit_a + ci_a,
                color=self.fill_color_alpha,
                alpha=0.2,
            )
            self.canvas_fit.axes.plot(x_smooth, fit_b, "r-")
            self.canvas_fit.axes.fill_between(
                x_smooth,
                fit_b - ci_b,
                fit_b + ci_b,
                color=self.fill_color_beta,
                alpha=0.2,
            )

            self.canvas_fit.axes.set_xlabel("Delay (s)")
            self.canvas_fit.axes.set_ylabel(r"$I/I_0$")

            res_text = (
                f"Ra: {tb.Ra:.2f} +/- {tb.err_Ra:.2f} Hz\n"
                f"Rb: {tb.Rb:.2f} +/- {tb.err_Rb:.2f} Hz\n"
                f"Tau_c: {tb.tau_c:.2f} +/- {tb.err_tau_c:.2f} ns"
            )
            self.tab_fitting.lbl_results.setText(res_text)

            self.canvas_fit.axes.legend()
            self.canvas_fit.draw()

            self.update_table()

            if self.tab_fitting.chk_sliding.isChecked():
                self.tab_fitting.btn_fit.setEnabled(False)
                self.tab_fitting.lbl_results.setText(
                    self.tab_fitting.lbl_results.text()
                    + "\nCalculating sliding window..."
                )
                self.statusBar().showMessage("Calculated Tau_c. Starting sliding window...", 5000)

                self.thread = QThread()
                self.worker = SlidingWindowWorker(
                    tb, start_ppm, end_ppm, b0, s2_val, n_boot, self.current_idx
                )
                self.worker.moveToThread(self.thread)
                self.thread.started.connect(self.worker.run)
                self.worker.finished.connect(self.on_sliding_finished)
                self.worker.error.connect(self.on_sliding_error)
                self.worker.finished.connect(self.thread.quit)
                self.worker.finished.connect(self.worker.deleteLater)
                self.thread.finished.connect(self.thread.deleteLater)
                self.thread.start()
            else:
                self.tabs_results.setCurrentIndex(0)
                self.statusBar().showMessage("Finished calculating Tau_c", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Fit Error", str(e))

    def on_sliding_finished(self, ppms, taus, errs, idx):
        self.tab_fitting.btn_fit.setEnabled(True)

        if 0 <= idx < len(self.datasets):
            self.datasets[idx]["sliding_results"] = (ppms, taus, errs)

        if idx == self.current_idx:
            self.canvas_sliding.axes.clear()
            self.canvas_sliding.axes.plot(ppms, taus, "b-", label=r"$\tau_c$")
            self.canvas_sliding.axes.fill_between(
                ppms, taus - errs, taus + errs, color=self.fill_color_sliding, alpha=0.2
            )
            self.canvas_sliding.axes.set_xlabel(r"$^{1}H (ppm)$")
            self.canvas_sliding.axes.set_ylabel(r"$\tau_c (ns)$")
            self.canvas_sliding.axes.invert_xaxis()
            self.canvas_sliding.axes.grid(True)
            self.canvas_sliding.axes.legend()
            self.canvas_sliding.draw()
            self.tabs_results.setCurrentIndex(1)

            current_text = self.tab_fitting.lbl_results.text().replace(
                "\nCalculating sliding window...", ""
            )
            self.tab_fitting.lbl_results.setText(current_text)
            self.statusBar().showMessage("Finished sliding window analysis", 5000)

    def on_sliding_error(self, msg):
        self.tab_fitting.btn_fit.setEnabled(True)
        QMessageBox.critical(self, "Sliding Window Error", msg)

    def export_table_to_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                headers = []
                for col in range(self.panel_experiment.table_data.columnCount()):
                    item = self.panel_experiment.table_data.horizontalHeaderItem(col)
                    headers.append(item.text() if item else "")
                writer.writerow(headers)
                for row in range(self.panel_experiment.table_data.rowCount()):
                    row_data = [
                        self.panel_experiment.table_data.item(row, col).text()
                        if self.panel_experiment.table_data.item(row, col)
                        else ""
                        for col in range(self.panel_experiment.table_data.columnCount())
                    ]
                    writer.writerow(row_data)
                self.statusBar().showMessage("Saved CSV successfully", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def update_table(self) -> None:
        self.panel_experiment.table_data.blockSignals(True)
        self.panel_experiment.table_data.setRowCount(len(self.datasets))
        for i, ds in enumerate(self.datasets):
            # Experiment Name (Editable)
            item_name = QTableWidgetItem(ds["name"])
            item_name.setFlags(item_name.flags() | Qt.ItemFlag.ItemIsEditable)
            self.panel_experiment.table_data.setItem(i, 0, item_name)

            # Temperature
            try:
                temp = ds["handler"].attributes["acqus"]["TE"]
            except (KeyError, TypeError):
                temp = "N/A"
            item_temp = QTableWidgetItem(str(temp))
            item_temp.setFlags(item_temp.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.panel_experiment.table_data.setItem(i, 1, item_temp)

            # Delays
            n_delays = (
                len(ds["handler"].delays) if ds["handler"].delays is not None else 0
            )
            item_delays = QTableWidgetItem(str(n_delays))
            item_delays.setFlags(item_delays.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.panel_experiment.table_data.setItem(i, 2, item_delays)

            # Helper for values
            def get_val(attr):
                if hasattr(ds["handler"], attr):
                    return f"{getattr(ds['handler'], attr):.2f}"
                return "N/A"

            # Ra
            item_ra = QTableWidgetItem(get_val("Ra"))
            item_ra.setFlags(item_ra.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.panel_experiment.table_data.setItem(i, 3, item_ra)

            # Rb
            item_rb = QTableWidgetItem(get_val("Rb"))
            item_rb.setFlags(item_rb.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.panel_experiment.table_data.setItem(i, 4, item_rb)

            # Tau_C
            item_tau = QTableWidgetItem(get_val("tau_c"))
            item_tau.setFlags(item_tau.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.panel_experiment.table_data.setItem(i, 5, item_tau)

            # Errors
            for col, attr in enumerate(["err_Ra", "err_Rb", "err_tau_c"], start=6):
                item_err = QTableWidgetItem(get_val(attr))
                item_err.setFlags(item_err.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.panel_experiment.table_data.setItem(i, col, item_err)

        self.panel_experiment.table_data.blockSignals(False)

    def on_table_double_click(self, row: int, col: int) -> None:
        if col == 0:
            return
        self.switch_dataset(row)

    def on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            row = item.row()
            new_name = item.text()
            if row < len(self.datasets):
                self.datasets[row]["name"] = new_name
                if row == self.current_idx:
                    self.panel_experiment.current_experiment.setText(new_name)

    def switch_dataset(self, index: int) -> None:
        if index < 0 or index >= len(self.datasets):
            return

        # Save state of current dataset
        if self.current_idx >= 0 and self.current_idx < len(self.datasets):
            self.datasets[self.current_idx]["baseline_nodes"] = list(
                self.baseline_nodes
            )

        self.current_idx = index
        ds = self.datasets[index]
        tb = ds["handler"]

        self.panel_experiment.current_experiment.setText(ds["name"])

        # Update Field Strength from parameters
        try:
            self.tab_fitting.input_field.setText(
                "{:.2f}".format(tb.attributes["acqus"]["SFO1"])
            )
        except (KeyError, AttributeError):
            pass

        self.tab_processing.slider_p0_coarse.blockSignals(True)
        self.tab_processing.slider_p0_fine.blockSignals(True)
        self.tab_processing.slider_p1_coarse.blockSignals(True)
        self.tab_processing.slider_p1_fine.blockSignals(True)

        p0 = ds["p0"]
        self.tab_processing.slider_p0_coarse.setValue(int(p0))
        self.tab_processing.slider_p0_fine.setValue(round((p0 - int(p0)) * 10))
        self.tab_processing.input_p0.setText(f"{p0:.1f}")

        p1 = ds["p1"]
        self.tab_processing.slider_p1_coarse.setValue(int(p1))
        self.tab_processing.slider_p1_fine.setValue(round((p1 - int(p1)) * 10))
        self.tab_processing.input_p1.setText(f"{p1:.1f}")

        self.tab_processing.slider_p0_coarse.blockSignals(False)
        self.tab_processing.slider_p0_fine.blockSignals(False)
        self.tab_processing.slider_p1_coarse.blockSignals(False)
        self.tab_processing.slider_p1_fine.blockSignals(False)

        # Restore baseline nodes
        self.baseline_nodes = ds.get("baseline_nodes", [])
        self.picking_baseline = False
        self.tab_processing.btn_pick_bl.setChecked(False)

        self.canvas_spec.axes.clear()

        # Use cached trace if available to avoid reprocessing
        if "processed_trace" in ds:
            trace = ds["processed_trace"]
            ppm_scale = ds.get("ppm_scale")

            if ppm_scale is not None:
                self.canvas_spec.axes.plot(ppm_scale, trace, label="First Plane")
                self.canvas_spec.axes.invert_xaxis()
                self.canvas_spec.axes.set_xlabel(r"$^{1}H (ppm)$")
                self.canvas_spec.axes.set_ylabel("Intensity")
            else:
                self.canvas_spec.axes.plot(trace, label="First Plane")
            self.canvas_spec.axes.legend()

            for node in self.baseline_nodes:
                self.canvas_spec.axes.axvline(
                    x=node, color="r", linestyle="--", alpha=0.5
                )

            self.selector = SpanSelector(
                self.canvas_spec.axes,
                self.on_span_select,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.2, facecolor="green"),
                interactive=True,
                drag_from_anywhere=True,
            )
            try:
                s = float(self.tab_processing.input_int_start.text())
                e = float(self.tab_processing.input_int_end.text())
                self.selector.extents = (min(s, e), max(s, e))
            except ValueError:
                pass
            self.canvas_spec.draw()
        else:
            self.process_data()

        # Update fit display
        self.canvas_fit.axes.clear()
        self.tab_fitting.lbl_results.setText("Results will appear here.")

        if hasattr(tb, "Ra") and hasattr(tb, "popt_alpha"):
            try:
                x, y_a, y_b, popt_a, popt_b, pcov_a, pcov_b = tb.get_fit_data()
                self.canvas_fit.axes.plot(x, y_a, "bo", label="Alpha (Anti-TROSY)")
                self.canvas_fit.axes.plot(x, y_b, "ro", label="Beta (TROSY)")

                x_smooth = np.linspace(0, np.max(x) * 1.1, 100)
                fit_a = processing.TractBruker._relax(x_smooth, *popt_a)
                ci_a = tb.calc_confidence_interval(x_smooth, popt_a, pcov_a)
                fit_b = processing.TractBruker._relax(x_smooth, *popt_b)
                ci_b = tb.calc_confidence_interval(x_smooth, popt_b, pcov_b)

                self.canvas_fit.axes.plot(x_smooth, fit_a, "b-")
                self.canvas_fit.axes.fill_between(
                    x_smooth,
                    fit_a - ci_a,
                    fit_a + ci_a,
                    color=self.fill_color_alpha,
                    alpha=0.2,
                )
                self.canvas_fit.axes.plot(x_smooth, fit_b, "r-")
                self.canvas_fit.axes.fill_between(
                    x_smooth,
                    fit_b - ci_b,
                    fit_b + ci_b,
                    color=self.fill_color_beta,
                    alpha=0.2,
                )

                tau_c_val = getattr(tb, "tau_c", 0.0)
                err_tau_c_val = getattr(tb, "err_tau_c", 0.0)
                res_text = (
                    f"Ra: {tb.Ra:.2f} +/- {tb.err_Ra:.2f} Hz\n"
                    f"Rb: {tb.Rb:.2f} +/- {tb.err_Rb:.2f} Hz\n"
                    f"Tau_c: {tau_c_val:.2f} +/- {err_tau_c_val:.2f} ns"
                )
                self.tab_fitting.lbl_results.setText(res_text)
                self.canvas_fit.axes.legend()
            except Exception:
                pass

        self.canvas_fit.draw()

        self.canvas_sliding.axes.clear()
        if "sliding_results" in ds:
            ppms, taus, errs = ds["sliding_results"]
            self.canvas_sliding.axes.plot(ppms, taus, "b-", label=r"$\tau_c$")
            self.canvas_sliding.axes.fill_between(
                ppms,
                taus - errs,
                taus + errs,
                color=self.fill_color_sliding,
                alpha=0.2,
                label=r"$\sigma$",
            )
            self.canvas_sliding.axes.set_xlabel(r"$^{1}H (ppm)$")
            self.canvas_sliding.axes.set_ylabel(r"$\tau_c (ns)$")
            self.canvas_sliding.axes.invert_xaxis()
            self.canvas_sliding.axes.grid(True)
            self.canvas_sliding.axes.legend()
        self.canvas_sliding.draw()

        self.plot_stacked_spectra()

    def show_context_menu(self, pos: QPoint) -> None:
        menu = QMenu()
        action_change = QAction("Change Experiment", self)
        action_delete = QAction("Delete Experiment", self)
        action_export = QAction("Export Table to CSV", self)

        action_change.triggered.connect(self.change_experiment)
        action_delete.triggered.connect(self.delete_experiment)
        action_export.triggered.connect(self.export_table_to_csv)

        menu.addAction(action_change)
        menu.addAction(action_delete)
        menu.addSeparator()
        menu.addAction(action_export)
        menu.exec(self.panel_experiment.table_data.mapToGlobal(pos))

    def change_experiment(self) -> None:
        row = self.panel_experiment.table_data.currentRow()
        if row < 0:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Bruker Directory")
        if folder:
            try:
                delay_list = None
                if not os.path.exists(os.path.join(folder, "vdlist")):
                    delay_list, _ = QFileDialog.getOpenFileName(
                        self, "vdlist not found. Select delay list file:", folder
                    )

                tb = processing.TractBruker(folder, delay_list=delay_list)
                self.datasets[row]["handler"] = tb
                self.datasets[row]["path"] = folder
                self.datasets[row]["name"] = os.path.basename(folder)
                self.datasets[row]["p0"] = tb.phc0
                self.datasets[row]["p1"] = tb.phc1
                self.update_table()
                if row == self.current_idx:
                    self.switch_dataset(row)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to change data: {str(e)}")

    def delete_experiment(self) -> None:
        row = self.panel_experiment.table_data.currentRow()
        if row < 0:
            return
        del self.datasets[row]
        self.update_table()
        if len(self.datasets) == 0:
            self.current_idx = -1
            self.canvas_spec.axes.clear()
            self.canvas_spec.draw()
            self.canvas_fit.axes.clear()
            self.canvas_fit.draw()
            self.canvas_sliding.axes.clear()
            self.canvas_sliding.draw()
            self.canvas_alpha.axes.clear()
            self.canvas_alpha.draw()
            self.canvas_beta.axes.clear()
            self.canvas_beta.draw()
            self.panel_experiment.current_experiment.clear()
        elif row == self.current_idx:
            self.current_idx = -1  # Prevent saving state to wrong index
            self.switch_dataset(max(0, row - 1))
        elif row < self.current_idx:
            self.current_idx -= 1

    def on_span_select(self, vmin: float, vmax: float) -> None:
        self.tab_processing.input_int_start.setText(f"{vmax:.3f}")
        self.tab_processing.input_int_end.setText(f"{vmin:.3f}")

    def open_help(self) -> None:
        QDesktopServices.openUrl(QUrl("https://github.com/debadutta-patra/pyTRACTnmr"))

    def update_fill_colors(
        self,
        alpha_color: Optional[str] = None,
        beta_color: Optional[str] = None,
        sliding_color: Optional[str] = None,
    ) -> None:
        """Update the colors used for fill_between plots."""
        if alpha_color:
            self.fill_color_alpha = alpha_color
        if beta_color:
            self.fill_color_beta = beta_color
        if sliding_color:
            self.fill_color_sliding = sliding_color

        if self.current_idx >= 0:
            self.switch_dataset(self.current_idx)

    def pick_fit_colors(self) -> None:
        c = QColorDialog.getColor(
            QColor(self.fill_color_alpha), self, "Select Alpha Fill Color"
        )
        if c.isValid():
            self.fill_color_alpha = c.name()

        c = QColorDialog.getColor(
            QColor(self.fill_color_beta), self, "Select Beta Fill Color"
        )
        if c.isValid():
            self.fill_color_beta = c.name()
        self.update_fill_colors()

    def pick_sliding_colors(self) -> None:
        c = QColorDialog.getColor(
            QColor(self.fill_color_sliding), self, "Select Sliding Window Fill Color"
        )
        if c.isValid():
            self.fill_color_sliding = c.name()
        self.update_fill_colors()

    def plot_stacked_spectra(self) -> None:
        if self.current_idx < 0:
            return

        tb = self.datasets[self.current_idx]["handler"]

        if not tb.alpha_spectra or not tb.beta_spectra:
            self.tabs_spectrum.setTabEnabled(1, False)
            self.tabs_spectrum.setTabEnabled(2, False)
            self.canvas_alpha.axes.clear()
            self.canvas_alpha.draw()
            self.canvas_beta.axes.clear()
            self.canvas_beta.draw()
            return

        self.tabs_spectrum.setTabEnabled(1, True)
        self.tabs_spectrum.setTabEnabled(2, True)

        def _plot(
            canvas,
            spectra,
            delays,
            controls,
        ):
            ax = canvas.axes
            ax.clear()
            if not spectra:
                canvas.draw()
                return

            ppm_scale = tb.unit_converter.ppm_scale() if tb.unit_converter else None
            n_pts = min(len(spectra), len(delays))

            cmap = plt.get_cmap("viridis")
            norm = None
            if n_pts > 0:
                norm = plt.Normalize(np.min(delays[:n_pts]), np.max(delays[:n_pts]))

            xlim = controls.get_xlim()

            for i in range(n_pts):
                s = spectra[i]
                y_val = delays[i]
                x = ppm_scale if ppm_scale is not None else np.arange(len(s))

                if x is not None:
                    x_min, x_max = min(xlim), max(xlim)
                    mask = (x >= x_min) & (x <= x_max)
                    x_plot, s_plot = x[mask], s[mask]
                else:
                    x_plot, s_plot = x, s

                if x_plot is None or len(x_plot) == 0:
                    continue

                ys_plot = np.full_like(s_plot, y_val)
                color = cmap(norm(y_val)) if norm else "k"
                ax.plot(x_plot, ys_plot, s_plot, color=color, linewidth=1)

            ax.set_xlabel(r"$^{1}H\ (ppm)$")
            ax.set_ylabel("Delay (s)")
            ax.set_zlabel("Intensity")

            if ppm_scale is not None:
                ax.set_xlim(xlim[1], xlim[0])
                ax.invert_xaxis()

            if delays is not None and n_pts > 0:
                relevant_delays = delays[:n_pts]
                ax.set_ylim(max(relevant_delays) * 1.1, min(relevant_delays) * 0.9)

            ax.set_zlim(bottom=0)
            ax.get_zaxis().set_ticklabels([])

            elev, azim, roll = controls.get_view_params()
            ax.view_init(elev=elev, azim=azim, roll=roll)
            canvas.draw()

        delays = tb.delays if tb.delays is not None else np.array([])
        _plot(
            self.canvas_alpha,
            tb.alpha_spectra,
            delays,
            self.controls_alpha,
        )
        _plot(
            self.canvas_beta,
            tb.beta_spectra,
            delays,
            self.controls_beta,
        )

    def export_fit_data(self) -> None:
        if self.current_idx < 0:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Fit Data", "", "All Files (*)"
        )
        if not path:
            return

        base_path = os.path.splitext(path)[0]
        csv_path = base_path + ".csv"
        py_path = base_path + ".py"

        try:
            tb = self.datasets[self.current_idx]["handler"]
            if not hasattr(tb, "popt_alpha"):
                QMessageBox.warning(
                    self, "No Data", "No fit data available. Run fitting first."
                )
                return

            x, y_a, y_b, popt_a, popt_b, pcov_a, pcov_b = tb.get_fit_data()

            # Generate smooth data for plotting
            x_smooth = np.linspace(0, np.max(x) * 1.1, 100)
            fit_a = processing.TractBruker._relax(x_smooth, *popt_a)
            ci_a = tb.calc_confidence_interval(x_smooth, popt_a, pcov_a)
            fit_b = processing.TractBruker._relax(x_smooth, *popt_b)
            ci_b = tb.calc_confidence_interval(x_smooth, popt_b, pcov_b)

            # Save CSV
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["# Experiment", self.datasets[self.current_idx]["name"]]
                )
                writer.writerow(["# Ra (Hz)", f"{getattr(tb, 'Ra', 'N/A')}"])
                writer.writerow(["# Rb (Hz)", f"{getattr(tb, 'Rb', 'N/A')}"])
                writer.writerow(["# Tau_c (ns)", f"{getattr(tb, 'tau_c', 'N/A')}"])
                writer.writerow([])
                writer.writerow(
                    [
                        "Delay (s)",
                        "Alpha Intensity",
                        "Beta Intensity",
                        "",
                        "Smooth Delay (s)",
                        "Alpha Fit",
                        "Alpha CI",
                        "Beta Fit",
                        "Beta CI",
                    ]
                )

                n_raw = len(x)
                n_smooth = len(x_smooth)
                for i in range(max(n_raw, n_smooth)):
                    row = []
                    if i < n_raw:
                        row.extend([x[i], y_a[i], y_b[i]])
                    else:
                        row.extend(["", "", ""])

                    row.append("")  # Spacer

                    if i < n_smooth:
                        row.extend([x_smooth[i], fit_a[i], ci_a[i], fit_b[i], ci_b[i]])
                    else:
                        row.extend(["", "", "", "", ""])
                    writer.writerow(row)

            # Save Python Script
            csv_filename = os.path.basename(csv_path)
            with open(py_path, "w") as f:
                script_content = exporters.generate_fit_script(
                    csv_filename, self.fill_color_alpha, self.fill_color_beta
                )
                f.write(script_content)

            QMessageBox.information(
                self, "Export Successful", f"Saved:\n{csv_path}\n{py_path}"
            )
            self.statusBar().showMessage("Exported relaxation data successfully", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_pdf_report(self) -> None:
        if self.current_idx < 0:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF Report", "", "PDF Files (*.pdf)"
        )
        if not path:
            return

        if not path.endswith(".pdf"):
            path += ".pdf"

        try:
            ds = self.datasets[self.current_idx]

            with PdfPages(path) as pdf:
                # Page 1: Text Summary
                fig_text = plt.figure(figsize=(8.5, 11))
                txt = "TRACT Analysis Report\n\n"
                txt += f"Experiment: {ds['name']}\n"
                txt += f"Path: {ds['path']}\n\n"

                txt += "Processing Parameters:\n"
                txt += f"  P0: {ds.get('p0', 0):.2f}\n"
                txt += f"  P1: {ds.get('p1', 0):.2f}\n"
                txt += f"  Points: {self.tab_processing.input_points.text()}\n"
                txt += (
                    f"  Apodization: {self.tab_processing.combo_apod.currentText()}\n"
                )
                if "Sine" in self.tab_processing.combo_apod.currentText():
                    txt += f"  Offset: {self.tab_processing.input_off.text()}\n"
                    txt += f"  End: {self.tab_processing.input_end.text()}\n"
                    txt += f"  Power: {self.tab_processing.input_pow.text()}\n"
                else:
                    txt += f"  LB: {self.tab_processing.input_lb.text()}\n"

                txt += "\nFitting Parameters:\n"
                txt += f"  Field: {self.tab_fitting.input_field.text()} MHz\n"
                txt += f"  CSA: {self.tab_fitting.input_csa.text()} ppm\n"
                txt += f"  Angle: {self.tab_fitting.input_angle.text()} deg\n"
                txt += f"  S2: {self.tab_fitting.input_s2.text()}\n"
                txt += f"  Bootstraps: {self.tab_fitting.input_bootstraps.text()}\n"

                txt += "\nResults:\n"
                txt += self.tab_fitting.lbl_results.text()

                fig_text.text(
                    0.1, 0.9, txt, fontsize=12, va="top", fontfamily="monospace"
                )
                pdf.savefig(fig_text)
                plt.close(fig_text)

                pdf.savefig(self.canvas_spec.figure, bbox_inches="tight")
                pdf.savefig(self.canvas_alpha.figure, bbox_inches="tight")
                pdf.savefig(self.canvas_beta.figure, bbox_inches="tight")
                pdf.savefig(self.canvas_fit.figure, bbox_inches="tight")
                if "sliding_results" in ds:
                    pdf.savefig(self.canvas_sliding.figure, bbox_inches="tight")

            QMessageBox.information(
                self, "Export Successful", f"Report saved to:\n{path}"
            )
            self.statusBar().showMessage("Exported PDF report successfully", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_sliding_data(self) -> None:
        if self.current_idx < 0:
            return

        ds = self.datasets[self.current_idx]
        if "sliding_results" not in ds:
            QMessageBox.warning(self, "No Data", "No sliding window results available.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Sliding Window Data", "", "All Files (*)"
        )
        if not path:
            return

        base_path = os.path.splitext(path)[0]
        csv_path = base_path + ".csv"
        py_path = base_path + ".py"

        try:
            ppms, taus, errs = ds["sliding_results"]

            # Save CSV
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["# Experiment", ds["name"]])
                writer.writerow([])
                writer.writerow(["PPM", "Tau_c (ns)", "Error (ns)"])
                for i in range(len(ppms)):
                    writer.writerow([ppms[i], taus[i], errs[i]])

            # Save Python Script
            csv_filename = os.path.basename(csv_path)
            with open(py_path, "w") as f:
                script_content = exporters.generate_sliding_script(
                    csv_filename, self.fill_color_sliding
                )
                f.write(script_content)

            QMessageBox.information(
                self, "Export Successful", f"Saved:\n{csv_path}\n{py_path}"
            )
            self.statusBar().showMessage("Exported sliding window data successfully", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
