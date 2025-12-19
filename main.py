"""
Entry point for the minimal DICOM viewer demo.
"""
import sys
from PyQt5.QtWidgets import QApplication
from viewer.app_mainwindow import MainWindow


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
