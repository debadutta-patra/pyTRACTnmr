from PySide6.QtWidgets import (
    QWidget,
    QFormLayout,
    QSlider,
    QLineEdit,
    QComboBox,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QCheckBox,
)
from PySide6.QtCore import Qt, Signal


class ProcessingTab(QWidget):
    param_changed = Signal()
    pick_nodes_toggled = Signal(bool)
    clear_nodes_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.slider_p0_coarse = QSlider(Qt.Orientation.Horizontal)
        self.slider_p0_coarse.setRange(-180, 180)
        self.slider_p0_coarse.setValue(0)
        self.slider_p0_coarse.valueChanged.connect(lambda _: self.param_changed.emit())

        self.slider_p0_fine = QSlider(Qt.Orientation.Horizontal)
        self.slider_p0_fine.setRange(-50, 50)
        self.slider_p0_fine.setValue(0)
        self.slider_p0_fine.valueChanged.connect(lambda _: self.param_changed.emit())

        self.input_p0 = QLineEdit("0.0")
        self.input_p0.setFixedWidth(50)

        self.slider_p1_coarse = QSlider(Qt.Orientation.Horizontal)
        self.slider_p1_coarse.setRange(-360, 360)
        self.slider_p1_coarse.setValue(0)
        self.slider_p1_coarse.valueChanged.connect(lambda _: self.param_changed.emit())

        self.slider_p1_fine = QSlider(Qt.Orientation.Horizontal)
        self.slider_p1_fine.setRange(-50, 50)
        self.slider_p1_fine.setValue(0)
        self.slider_p1_fine.valueChanged.connect(lambda _: self.param_changed.emit())

        self.input_p1 = QLineEdit("0.0")
        self.input_p1.setFixedWidth(50)

        self.input_points = QLineEdit("2048")
        self.input_points.editingFinished.connect(self.param_changed.emit)

        self.combo_apod = QComboBox()
        self.combo_apod.addItems(["Sine Bell (sp)", "Exponential (em)"])
        self.combo_apod.currentIndexChanged.connect(lambda _: self.update_apod_ui())
        self.combo_apod.currentIndexChanged.connect(lambda _: self.param_changed.emit())

        self.input_lb = QLineEdit("5.0")
        self.input_lb.editingFinished.connect(self.param_changed.emit)

        self.input_off = QLineEdit("0.35")
        self.input_off.editingFinished.connect(self.param_changed.emit)

        self.input_end = QLineEdit("0.98")
        self.input_end.editingFinished.connect(self.param_changed.emit)

        self.input_pow = QLineEdit("2.0")
        self.input_pow.editingFinished.connect(self.param_changed.emit)

        self.input_int_start = QLineEdit("9.5")
        self.input_int_start.editingFinished.connect(self.param_changed.emit)
        self.input_int_end = QLineEdit("7.5")
        self.input_int_end.editingFinished.connect(self.param_changed.emit)

        layout.addRow("P0 Coarse:", self.slider_p0_coarse)
        layout.addRow(
            "P0 Fine (+/- 5):",
            self.create_slider_layout(self.slider_p0_fine, self.input_p0),
        )
        layout.addRow("P1 Coarse:", self.slider_p1_coarse)
        layout.addRow(
            "P1 Fine (+/- 5):",
            self.create_slider_layout(self.slider_p1_fine, self.input_p1),
        )
        layout.addRow(QLabel("<b>Apodization & ZF</b>"))
        layout.addRow("Function:", self.combo_apod)
        layout.addRow("Points (ZF):", self.input_points)
        layout.addRow("Line Broadening (Hz):", self.input_lb)
        layout.addRow("Sine Offset:", self.input_off)
        layout.addRow("Sine End:", self.input_end)
        layout.addRow("Sine Power:", self.input_pow)
        layout.addRow(QLabel("<b>Integration Range</b>"))
        layout.addRow("Start (ppm):", self.input_int_start)
        layout.addRow("End (ppm):", self.input_int_end)

        layout.addRow(QLabel("<b>Baseline Correction</b>"))
        self.btn_pick_bl = QPushButton("Pick Nodes")
        self.btn_pick_bl.setCheckable(True)
        self.btn_pick_bl.clicked.connect(lambda checked: self.pick_nodes_toggled.emit(checked))
        self.btn_clear_bl = QPushButton("Clear Nodes")
        self.btn_clear_bl.clicked.connect(lambda _: self.clear_nodes_clicked.emit())
        layout_bl = QHBoxLayout()
        layout_bl.addWidget(self.btn_pick_bl)
        layout_bl.addWidget(self.btn_clear_bl)
        layout.addRow(layout_bl)

        self.setLayout(layout)
        self.update_apod_ui()

    def create_slider_layout(self, slider: QSlider, label: QWidget) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(slider)
        layout.addWidget(label)
        layout.setContentsMargins(0, 0, 0, 0)
        return widget

    def update_apod_ui(self) -> None:
        is_sp = self.combo_apod.currentText().startswith("Sine")
        layout = self.layout()
        if isinstance(layout, QFormLayout):
            layout.setRowVisible(self.input_lb, not is_sp)
            layout.setRowVisible(self.input_off, is_sp)
            layout.setRowVisible(self.input_end, is_sp)
            layout.setRowVisible(self.input_pow, is_sp)


class FittingTab(QWidget):
    fit_requested = Signal()
    export_fit_requested = Signal()
    export_sliding_requested = Signal()
    export_report_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()
        self.input_field = QLineEdit("600")
        self.input_csa = QLineEdit("172")
        self.input_angle = QLineEdit("17")
        self.input_s2 = QLineEdit("1.0")
        self.input_bootstraps = QLineEdit("1000")
        self.chk_sliding = QCheckBox("Sliding Window Analysis")

        self.btn_fit = QPushButton("Calculate Tau_c")
        self.btn_fit.clicked.connect(self.fit_requested.emit)

        self.btn_export_fit = QPushButton("Export Relaxation Data")
        self.btn_export_fit.clicked.connect(self.export_fit_requested.emit)

        self.btn_export_sliding = QPushButton("Export Sliding Window Data")
        self.btn_export_sliding.clicked.connect(self.export_sliding_requested.emit)

        self.btn_export_report = QPushButton("Export PDF Report")
        self.btn_export_report.clicked.connect(self.export_report_requested.emit)

        self.lbl_results = QLabel("Results will appear here.")
        self.lbl_results.setWordWrap(True)

        layout.addRow("Field Strength (MHz):", self.input_field)
        layout.addRow("CSA (ppm):", self.input_csa)
        layout.addRow("CSA Angle (deg):", self.input_angle)
        layout.addRow("Order Parameter (S2):", self.input_s2)
        layout.addRow("Bootstraps:", self.input_bootstraps)
        layout.addRow(self.chk_sliding)
        layout.addRow(self.btn_fit)
        layout.addRow(self.btn_export_fit)
        layout.addRow(self.btn_export_sliding)
        layout.addRow(self.btn_export_report)
        layout.addRow(self.lbl_results)
        self.setLayout(layout)