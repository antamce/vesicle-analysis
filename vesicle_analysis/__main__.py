from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from vesicle_analysis.main_window import MainWindow, application_icon_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browse and analyze ND2 recordings.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Optional ND2 file or directory to open at launch.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if sys.platform == "win32":
        try:
            # Give Windows a stable application identity so it uses our Qt
            # window icon in the taskbar instead of the Python/Conda icon.
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "VesicleAnalysis.DesktopApp"
            )
        except (AttributeError, OSError):
            pass
    app = QApplication(sys.argv)
    app.setApplicationName("Vesicle Analysis")
    app.setOrganizationName("Vesicle Analysis")
    icon_path = application_icon_path()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    # Starting maximized keeps every platform inside the available desktop and
    # avoids a later layout update changing the window's normal/maximized state.
    window.showMaximized()
    if args.path is not None:
        window.open_path(args.path)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
