from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QTableWidget,
    QHBoxLayout,
    QLabel,
    QHeaderView,
)
from PySide6.QtCore import Qt, Signal


class ExperimentPanel(QGroupBox):
    load_clicked = Signal()
    load_demo_clicked = Signal()
    help_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__("Experiment Info", parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.btn_load = QPushButton("Load Bruker Directory")
        self.btn_load.clicked.connect(self.load_clicked.emit)

        self.btn_load_demo = QPushButton("Load Demo Data")
        self.btn_load_demo.clicked.connect(self.load_demo_clicked.emit)

        load_buttons_layout = QHBoxLayout()
        load_buttons_layout.addWidget(self.btn_load)
        load_buttons_layout.addWidget(self.btn_load_demo)

        self.current_experiment = QLineEdit()
        self.current_experiment.setPlaceholderText("Current Experiment")
        self.current_experiment.setReadOnly(True)

        self.table_data = QTableWidget()
        self.table_data.setColumnCount(9)
        self.table_data.setHorizontalHeaderLabels(
            [
                "Experiment",
                "Temperature (K)",
                "Delays",
                "Ra (Hz)",
                "Rb (Hz)",
                "Tau_C (ns)",
                "Err Ra",
                "Err Rb",
                "Err Tau_C",
            ]
        )
        self.table_data.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table_data.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_data.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
        )
        self.table_data.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        layout.addWidget(QLabel("Load Data:"))
        layout.addLayout(load_buttons_layout)
        layout.addSpacing(10)
        layout.addWidget(QLabel("Current Experiment:"))
        layout.addWidget(self.current_experiment)
        layout.addSpacing(10)
        layout.addWidget(self.table_data)
        layout.addStretch()

        layout_bottom = QHBoxLayout()
        layout_bottom.addStretch()
        self.btn_help = QPushButton("?")
        self.btn_help.setFixedSize(20, 20)
        self.btn_help.setToolTip("Open GitHub Repository")
        self.btn_help.clicked.connect(self.help_clicked.emit)
        layout_bottom.addWidget(self.btn_help)
        layout.addLayout(layout_bottom)

        self.setLayout(layout)