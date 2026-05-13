#!/usr/bin/env python3
import sys
import os
import traceback

# Run from the project root so "data/" resolves correctly
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
from main_window import MainWindow


def _excepthook(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    detail = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    sys.stderr.write(detail)
    try:
        app = QApplication.instance()
        if app is not None:
            dlg = QMessageBox()
            dlg.setIcon(QMessageBox.Critical)
            dlg.setWindowTitle("Unexpected Error")
            dlg.setText(
                "An unexpected error occurred. The application may be unstable.\n\n"
                "Please save your work and restart if needed."
            )
            dlg.setDetailedText(detail)
            dlg.exec_()
    except Exception:
        pass


def main():
    sys.excepthook = _excepthook
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Label & Track")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
