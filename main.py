"""
Entry point for the advanced DICOM viewer.
"""
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from viewer.app_mainwindow import MainWindow


def main():
    # Enable high DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName('DICOM Viewer')
    app.setOrganizationName('Medical Imaging')
    
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
