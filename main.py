import sys
from PySide6.QtWidgets import QApplication
from window import TractApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TractApp()
    window.show()
    sys.exit(app.exec())
