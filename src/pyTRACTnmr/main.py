import sys
from PySide6.QtWidgets import QApplication

from window import TractApp  # type: ignore


def main():
    app = QApplication(sys.argv)
    window = TractApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
