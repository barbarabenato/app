import sys

from PyQt5.QtWidgets import QApplication, QWidget
from ui.main_window import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    UI = MainWindow()
    sys.exit(app.exec_())


