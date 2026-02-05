import sys
from PySide6.QtWidgets import QApplication

try:
    from .window import TractApp
except ImportError:
    from window import TractApp


def main():
    app = QApplication(sys.argv)
    window = TractApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
