import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from src.data_visualization.suite.app.bootstrap import build_visualization_data_service
from src.data_visualization.suite.app.main_window import VisualizationMainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch IMU visualization suite")
    parser.add_argument(
        "--sqlite-db-path",
        type=Path,
        required=True,
        help="Path to SQLite metadata index (index.db)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_service = build_visualization_data_service(args.sqlite_db_path)

    app = QApplication(sys.argv)
    window = VisualizationMainWindow(data_service)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
