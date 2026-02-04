import os
import csv
import numpy as np
from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QSplitter,
    QTabWidget,
    QFormLayout,
    QGroupBox,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMenu,
    QSlider,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QPoint

from widgets import MplCanvas, CustomNavigationToolbar
from matplotlib.widgets import SpanSelector
import processing


class TractApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRACT Analysis GUI")
        self.resize(1200, 800)

        # Data State
        self.dic = None
        self.data = None
        self.proc_data = None
        self.time_points = None
        self.datasets: List[Dict[str, Any]] = []
        self.current_idx: int = -1
        self.selector: Optional[SpanSelector] = None

        self.init_ui()

    def init_ui(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- Panel 1: Data Loading ---
        panel1 = QGroupBox("Experiment Info")
        layout1 = QVBoxLayout()

        self.btn_load = QPushButton("Load Bruker Directory")
        self.btn_load.clicked.connect(self.load_data)

        self.current_experiment = QLineEdit()
        self.current_experiment.setPlaceholderText("Current Experiment")
        self.current_experiment.setReadOnly(True)

        self.table_data = QTableWidget()
        self.table_data.setColumnCount(9)
        self.table_data.setHorizontalHeaderLabels(
            ["Experiment", "Temperature (K)", "Delays", "Ra (Hz)", "Rb (Hz)", "Tau_C (ns)", "Err Ra", "Err Rb", "Err Tau_C"]
        )
        self.table_data.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table_data.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_data.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
        )
        self.table_data.cellDoubleClicked.connect(self.on_table_double_click)
        self.table_data.itemChanged.connect(self.on_table_item_changed)
        self.table_data.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_data.customContextMenuRequested.connect(self.show_context_menu)

        layout1.addWidget(QLabel("Load Data:"))
        layout1.addWidget(self.btn_load)
        layout1.addSpacing(10)
        layout1.addWidget(QLabel("Current Experiment:"))
        layout1.addWidget(self.current_experiment)
        layout1.addSpacing(10)
        layout1.addWidget(self.table_data)
        layout1.addStretch()
        panel1.setLayout(layout1)

        # --- Panel 2: Visualization ---
        splitter_center = QSplitter(Qt.Orientation.Vertical)

        # Top: Spectrum
        self.canvas_spec = MplCanvas(self)
        self.toolbar_spec = CustomNavigationToolbar(self.canvas_spec, self)
        widget_spec = QWidget()
        layout_spec = QVBoxLayout()
        lbl_spec = QLabel("<b>Processed Spectrum (Phase Check)</b>")
        lbl_spec.setFixedHeight(30)
        layout_spec.addWidget(lbl_spec)
        layout_spec.addWidget(self.toolbar_spec)
        layout_spec.addWidget(self.canvas_spec)
        widget_spec.setLayout(layout_spec)

        # Bottom: Fits
        self.canvas_fit = MplCanvas(self)
        self.toolbar_fit = CustomNavigationToolbar(self.canvas_fit, self)
        widget_fit = QWidget()
        layout_fit = QVBoxLayout()
        lbl_fit = QLabel("<b>Relaxation Fits</b>")
        lbl_fit.setFixedHeight(30)
        layout_fit.addWidget(lbl_fit)
        layout_fit.addWidget(self.toolbar_fit)
        layout_fit.addWidget(self.canvas_fit)
        widget_fit.setLayout(layout_fit)

        splitter_center.addWidget(widget_spec)
        splitter_center.addWidget(widget_fit)

        # --- Panel 3: Controls ---
        panel3 = QTabWidget()

        # Tab 1: Processing
        tab1 = QWidget()
        layout_t1 = QFormLayout()

        self.slider_p0_coarse = QSlider(Qt.Orientation.Horizontal)
        self.slider_p0_coarse.setRange(-180, 180)
        self.slider_p0_coarse.setValue(0)
        self.slider_p0_coarse.valueChanged.connect(self.process_data)

        self.slider_p0_fine = QSlider(Qt.Orientation.Horizontal)
        self.slider_p0_fine.setRange(-50, 50)
        self.slider_p0_fine.setValue(0)
        self.slider_p0_fine.valueChanged.connect(self.process_data)
        self.input_p0 = QLineEdit("0.0")
        self.input_p0.setFixedWidth(50)
        self.input_p0.editingFinished.connect(self.update_phase_from_text)

        self.slider_p1_coarse = QSlider(Qt.Orientation.Horizontal)
        self.slider_p1_coarse.setRange(-360, 360)
        self.slider_p1_coarse.setValue(0)
        self.slider_p1_coarse.valueChanged.connect(self.process_data)

        self.slider_p1_fine = QSlider(Qt.Orientation.Horizontal)
        self.slider_p1_fine.setRange(-50, 50)
        self.slider_p1_fine.setValue(0)
        self.slider_p1_fine.valueChanged.connect(self.process_data)
        self.input_p1 = QLineEdit("0.0")
        self.input_p1.setFixedWidth(50)
        self.input_p1.editingFinished.connect(self.update_phase_from_text)

        self.input_points = QLineEdit("2048")
        self.input_points.editingFinished.connect(self.process_data)

        self.input_off = QLineEdit("0.35")
        self.input_off.editingFinished.connect(self.process_data)

        self.input_end = QLineEdit("0.98")
        self.input_end.editingFinished.connect(self.process_data)

        self.input_pow = QLineEdit("2.0")
        self.input_pow.editingFinished.connect(self.process_data)

        self.input_int_start = QLineEdit("9.5")
        self.input_int_start.editingFinished.connect(self.process_data)
        self.input_int_end = QLineEdit("7.5")
        self.input_int_end.editingFinished.connect(self.process_data)

        layout_t1.addRow("P0 Coarse:", self.slider_p0_coarse)
        layout_t1.addRow(
            "P0 Fine (+/- 5):",
            self.create_slider_layout(self.slider_p0_fine, self.input_p0),
        )
        layout_t1.addRow("P1 Coarse:", self.slider_p1_coarse)
        layout_t1.addRow(
            "P1 Fine (+/- 5):",
            self.create_slider_layout(self.slider_p1_fine, self.input_p1),
        )
        layout_t1.addRow(QLabel("<b>Apodization & ZF</b>"))
        layout_t1.addRow("Points (ZF):", self.input_points)
        layout_t1.addRow("Sine Offset:", self.input_off)
        layout_t1.addRow("Sine End:", self.input_end)
        layout_t1.addRow("Sine Power:", self.input_pow)
        layout_t1.addRow(QLabel("<b>Integration Range</b>"))
        layout_t1.addRow("Start (ppm):", self.input_int_start)
        layout_t1.addRow("End (ppm):", self.input_int_end)
        tab1.setLayout(layout_t1)

        # Tab 2: Fitting
        tab2 = QWidget()
        layout_t2 = QFormLayout()
        self.input_field = QLineEdit("600")
        self.input_csa = QLineEdit("160")
        self.input_angle = QLineEdit("17")
        self.input_s2 = QLineEdit("1.0")
        self.btn_fit = QPushButton("Calculate Tau_c")
        self.btn_fit.clicked.connect(self.run_fitting)
        self.lbl_results = QLabel("Results will appear here.")
        self.lbl_results.setWordWrap(True)

        layout_t2.addRow("Field Strength (MHz):", self.input_field)
        layout_t2.addRow("CSA (ppm):", self.input_csa)
        layout_t2.addRow("CSA Angle (deg):", self.input_angle)
        layout_t2.addRow("Order Parameter (S2):", self.input_s2)
        layout_t2.addRow(self.btn_fit)
        layout_t2.addRow(self.lbl_results)
        tab2.setLayout(layout_t2)

        panel3.addTab(tab1, "Processing")
        panel3.addTab(tab2, "Fitting")

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(panel1)
        main_splitter.addWidget(splitter_center)
        main_splitter.addWidget(panel3)
        main_splitter.setSizes([400, 500, 300])
        main_layout.addWidget(main_splitter)

    def create_slider_layout(self, slider: QSlider, label: QWidget) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(slider)
        layout.addWidget(label)
        layout.setContentsMargins(0, 0, 0, 0)
        return widget

    def update_phase_from_text(self) -> None:
        try:
            p0 = float(self.input_p0.text())
            p1 = float(self.input_p1.text())

            self.slider_p0_coarse.blockSignals(True)
            self.slider_p0_fine.blockSignals(True)
            self.slider_p1_coarse.blockSignals(True)
            self.slider_p1_fine.blockSignals(True)

            self.slider_p0_coarse.setValue(int(p0))
            self.slider_p0_fine.setValue(int(round((p0 - int(p0)) * 10)))

            self.slider_p1_coarse.setValue(int(p1))
            self.slider_p1_fine.setValue(int(round((p1 - int(p1)) * 10)))

            self.slider_p0_coarse.blockSignals(False)
            self.slider_p0_fine.blockSignals(False)
            self.slider_p1_coarse.blockSignals(False)
            self.slider_p1_fine.blockSignals(False)

            self.process_data()
        except ValueError:
            pass

    def load_data(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Bruker Directory")
        if folder:
            try:
                delay_list = None
                if not os.path.exists(os.path.join(folder, "vdlist")):
                    delay_list, _ = QFileDialog.getOpenFileName(
                        self, "vdlist not found. Select delay list file:", folder
                    )

                tb = processing.TractBruker(folder, delay_list=delay_list)
                name = os.path.basename(folder)
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
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def process_data(self) -> None:
        if self.current_idx < 0:
            return
        try:
            p0 = self.slider_p0_coarse.value() + (self.slider_p0_fine.value() / 10.0)
            p1 = self.slider_p1_coarse.value() + (self.slider_p1_fine.value() / 10.0)

            self.input_p0.setText(f"{p0:.1f}")
            self.input_p1.setText(f"{p1:.1f}")

            points = int(self.input_points.text())
            off = float(self.input_off.text())
            end = float(self.input_end.text())
            pow_val = float(self.input_pow.text())

            if self.current_idx >= 0:
                self.datasets[self.current_idx]["p0"] = p0
                self.datasets[self.current_idx]["p1"] = p1

            tb = self.datasets[self.current_idx]["handler"]
            trace = tb.process_first_trace(
                p0, p1, points=points, off=off, end=end, pow=pow_val
            )

            self.canvas_spec.axes.clear()
            if tb.unit_converter:
                ppm_scale = tb.unit_converter.ppm_scale()
                self.canvas_spec.axes.plot(ppm_scale, trace, label="First Trace")
                self.canvas_spec.axes.invert_xaxis()
                self.canvas_spec.axes.set_xlabel(r"$^{1}H (ppm)$")
                self.canvas_spec.axes.set_ylabel("Intensity")
            else:
                self.canvas_spec.axes.plot(trace, label="First Trace")
            self.canvas_spec.axes.legend()

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
                s = float(self.input_int_start.text())
                e = float(self.input_int_end.text())
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
            p0 = self.slider_p0_coarse.value() + (self.slider_p0_fine.value() / 10.0)
            p1 = self.slider_p1_coarse.value() + (self.slider_p1_fine.value() / 10.0)

            points = int(self.input_points.text())
            off = float(self.input_off.text())
            end_param = float(self.input_end.text())
            pow_val = float(self.input_pow.text())
            start_ppm = float(self.input_int_start.text())
            end_ppm = float(self.input_int_end.text())

            # Update physics constants
            tb.CSA_15N = float(self.input_csa.text()) * 1e-6
            tb.CSA_BOND_ANGLE = float(self.input_angle.text()) * np.pi / 180
            s2_val = float(self.input_s2.text())

            tb.split_process(p0, p1, points=points, off=off, end=end_param, pow=pow_val)
            tb.integrate_ppm(start_ppm, end_ppm)
            tb.calc_relaxation()

            b0 = float(self.input_field.text()) if self.input_field.text() else None
            tb.calc_tc(B0=b0, S2=s2_val)

            x, y_a, y_b, popt_a, popt_b = tb.get_fit_data()

            self.canvas_fit.axes.clear()
            self.canvas_fit.axes.plot(x, y_a, "bo", label="Alpha (Anti-TROSY)")
            self.canvas_fit.axes.plot(x, y_b, "ro", label="Beta (TROSY)")
            self.canvas_fit.axes.plot(
                x, processing.TractBruker._relax(x, *popt_a), "b-"
            )
            self.canvas_fit.axes.plot(
                x, processing.TractBruker._relax(x, *popt_b), "r-"
            )
            self.canvas_fit.axes.set_xlabel("Delay (s)")
            self.canvas_fit.axes.set_ylabel(r"$I/I_0$")

            res_text = (
                f"Ra: {tb.Ra:.2f} +/- {tb.err_Ra:.2f} Hz\n"
                f"Rb: {tb.Rb:.2f} +/- {tb.err_Rb:.2f} Hz\n"
                f"Tau_c: {tb.tau_c:.2f} +/- {tb.err_tau_c:.2f} ns"
            )
            self.lbl_results.setText(res_text)

            self.canvas_fit.axes.legend()
            self.canvas_fit.draw()

            self.update_table()
        except Exception as e:
            QMessageBox.critical(self, "Fit Error", str(e))

    def export_table_to_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = []
                for col in range(self.table_data.columnCount()):
                    item = self.table_data.horizontalHeaderItem(col)
                    headers.append(item.text() if item else "")
                writer.writerow(headers)
                for row in range(self.table_data.rowCount()):
                    row_data = [self.table_data.item(row, col).text() if self.table_data.item(row, col) else "" for col in range(self.table_data.columnCount())]
                    writer.writerow(row_data)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def update_table(self) -> None:
        self.table_data.blockSignals(True)
        self.table_data.setRowCount(len(self.datasets))
        for i, ds in enumerate(self.datasets):
            # Experiment Name (Editable)
            item_name = QTableWidgetItem(ds["name"])
            item_name.setFlags(item_name.flags() | Qt.ItemFlag.ItemIsEditable)
            self.table_data.setItem(i, 0, item_name)

            # Temperature
            try:
                temp = ds["handler"].attributes["acqus"]["TE"]
            except (KeyError, TypeError):
                temp = "N/A"
            item_temp = QTableWidgetItem(str(temp))
            item_temp.setFlags(item_temp.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_data.setItem(i, 1, item_temp)

            # Delays
            n_delays = (
                len(ds["handler"].delays) if ds["handler"].delays is not None else 0
            )
            item_delays = QTableWidgetItem(str(n_delays))
            item_delays.setFlags(item_delays.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_data.setItem(i, 2, item_delays)

            # Helper for values
            def get_val(attr):
                if hasattr(ds["handler"], attr):
                    return f"{getattr(ds['handler'], attr):.2f}"
                return "N/A"

            # Ra
            item_ra = QTableWidgetItem(get_val("Ra"))
            item_ra.setFlags(item_ra.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_data.setItem(i, 3, item_ra)

            # Rb
            item_rb = QTableWidgetItem(get_val("Rb"))
            item_rb.setFlags(item_rb.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_data.setItem(i, 4, item_rb)

            # Tau_C
            item_tau = QTableWidgetItem(get_val("tau_c"))
            item_tau.setFlags(item_tau.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_data.setItem(i, 5, item_tau)

            # Errors
            for col, attr in enumerate(["err_Ra", "err_Rb", "err_tau_c"], start=6):
                item_err = QTableWidgetItem(get_val(attr))
                item_err.setFlags(item_err.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table_data.setItem(i, col, item_err)

        self.table_data.blockSignals(False)

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
                    self.current_experiment.setText(new_name)

    def switch_dataset(self, index: int) -> None:
        if index < 0 or index >= len(self.datasets):
            return
        self.current_idx = index
        ds = self.datasets[index]
        tb = ds["handler"]

        self.current_experiment.setText(ds["name"])

        # Update Field Strength from parameters
        try:
            self.input_field.setText("{:.2f}".format(tb.attributes["acqus"]["SFO1"]))
        except (KeyError, AttributeError):
            pass

        self.slider_p0_coarse.blockSignals(True)
        self.slider_p0_fine.blockSignals(True)
        self.slider_p1_coarse.blockSignals(True)
        self.slider_p1_fine.blockSignals(True)

        p0 = ds["p0"]
        self.slider_p0_coarse.setValue(int(p0))
        self.slider_p0_fine.setValue(round((p0 - int(p0)) * 10))
        self.input_p0.setText(f"{p0:.1f}")

        p1 = ds["p1"]
        self.slider_p1_coarse.setValue(int(p1))
        self.slider_p1_fine.setValue(round((p1 - int(p1)) * 10))
        self.input_p1.setText(f"{p1:.1f}")

        self.slider_p0_coarse.blockSignals(False)
        self.slider_p0_fine.blockSignals(False)
        self.slider_p1_coarse.blockSignals(False)
        self.slider_p1_fine.blockSignals(False)

        self.process_data()

        # Update fit display
        self.canvas_fit.axes.clear()
        self.lbl_results.setText("Results will appear here.")

        if hasattr(tb, "Ra") and hasattr(tb, "popt_alpha"):
            try:
                x, y_a, y_b, popt_a, popt_b = tb.get_fit_data()
                self.canvas_fit.axes.plot(x, y_a, "bo", label="Alpha (Anti-TROSY)")
                self.canvas_fit.axes.plot(x, y_b, "ro", label="Beta (TROSY)")
                self.canvas_fit.axes.plot(
                    x, processing.TractBruker._relax(x, *popt_a), "b-"
                )
                self.canvas_fit.axes.plot(
                    x, processing.TractBruker._relax(x, *popt_b), "r-"
                )

                tau_c_val = getattr(tb, "tau_c", 0.0)
                err_tau_c_val = getattr(tb, "err_tau_c", 0.0)
                res_text = (
                    f"Ra: {tb.Ra:.2f} +/- {tb.err_Ra:.2f} Hz\n"
                    f"Rb: {tb.Rb:.2f} +/- {tb.err_Rb:.2f} Hz\n"
                    f"Tau_c: {tau_c_val:.2f} +/- {err_tau_c_val:.2f} ns"
                )
                self.lbl_results.setText(res_text)
                self.canvas_fit.axes.legend()
            except Exception:
                pass

        self.canvas_fit.draw()

    def update_sample_name(self) -> None:
        if self.current_idx >= 0:
            name = self.current_experiment.text()
            self.datasets[self.current_idx]["name"] = name
            self.update_table()

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
        menu.exec(self.table_data.mapToGlobal(pos))

    def change_experiment(self) -> None:
        row = self.table_data.currentRow()
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
        row = self.table_data.currentRow()
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
            self.current_experiment.clear()
        elif row == self.current_idx:
            self.switch_dataset(max(0, row - 1))
        elif row < self.current_idx:
            self.current_idx -= 1

    def on_span_select(self, vmin: float, vmax: float) -> None:
        self.input_int_start.setText(f"{vmax:.3f}")
        self.input_int_end.setText(f"{vmin:.3f}")
