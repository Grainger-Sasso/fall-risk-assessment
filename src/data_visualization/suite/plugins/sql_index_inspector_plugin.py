from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.data_visualization.suite.plugins.base import PluginContext, VisualizationPlugin


class SQLIndexInspectorPlugin(VisualizationPlugin):
    plugin_id = "sql_index_inspector"
    display_name = "SQL Index Inspector"

    def __init__(self) -> None:
        self._context: Optional[PluginContext] = None
        self._status_label: Optional[QLabel] = None
        self._tables: Dict[str, QTableWidget] = {}

    def build_panel(self, parent: QWidget, context: PluginContext) -> QWidget:
        self._context = context
        root = QWidget(parent)
        layout = QVBoxLayout(root)

        header_row = QHBoxLayout()
        title = QLabel("SQLite Metadata Index Contents", root)
        refresh_button = QPushButton("Refresh", root)
        refresh_button.clicked.connect(self._refresh_tables)
        rollback_button = QPushButton("Rollback Last Feature Run", root)
        rollback_button.clicked.connect(self._rollback_last_feature_run)
        cleanup_button = QPushButton("Cleanup All Features", root)
        cleanup_button.clicked.connect(self._cleanup_all_features)
        header_row.addWidget(title)
        header_row.addStretch(1)
        header_row.addWidget(rollback_button)
        header_row.addWidget(cleanup_button)
        header_row.addWidget(refresh_button)

        self._status_label = QLabel("-", root)
        self._status_label.setWordWrap(True)

        tabs = QTabWidget(root)
        self._tables["imu_records"] = self._create_table(
            tabs, ["IMU ID", "Type", "Path"]
        )
        self._tables["user_records"] = self._create_table(
            tabs, ["User ID", "Type", "Path"]
        )
        self._tables["feature_records"] = self._create_table(
            tabs, ["Feature ID", "Type", "Path"]
        )
        self._tables["instrument_spec_records"] = self._create_table(
            tabs, ["Spec ID", "Type", "Path"]
        )
        self._tables["relations"] = self._create_table(
            tabs, ["Source Type", "Source ID", "Target Type", "Target ID"]
        )

        tabs.addTab(self._tables["imu_records"], "IMU")
        tabs.addTab(self._tables["user_records"], "Users")
        tabs.addTab(self._tables["feature_records"], "Features")
        tabs.addTab(self._tables["instrument_spec_records"], "Instrument Specs")
        tabs.addTab(self._tables["relations"], "Relations")

        layout.addLayout(header_row)
        layout.addWidget(self._status_label)
        layout.addWidget(tabs)
        self._refresh_tables()
        return root

    def _create_table(self, parent: QWidget, headers: List[str]) -> QTableWidget:
        table = QTableWidget(parent)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        return table

    def _refresh_tables(self, status_message: Optional[str] = None) -> None:
        if self._context is None:
            return
        snapshot = self._context.data_service.get_sql_index_snapshot()
        self._populate_record_table(
            self._tables["imu_records"],
            snapshot.get("imu_records", []),
        )
        self._populate_record_table(
            self._tables["user_records"],
            snapshot.get("user_records", []),
        )
        self._populate_record_table(
            self._tables["feature_records"],
            snapshot.get("feature_records", []),
        )
        self._populate_record_table(
            self._tables["instrument_spec_records"],
            snapshot.get("instrument_spec_records", []),
        )
        self._populate_relation_table(self._tables["relations"], snapshot.get("relations", []))

        if self._status_label is not None:
            if status_message is not None:
                self._status_label.setText(status_message)
            else:
                self._status_label.setText(
                    "Loaded "
                    f"{len(snapshot.get('imu_records', []))} IMU, "
                    f"{len(snapshot.get('user_records', []))} user, "
                    f"{len(snapshot.get('feature_records', []))} feature, "
                    f"{len(snapshot.get('instrument_spec_records', []))} instrument spec records "
                    f"and {len(snapshot.get('relations', []))} relations."
                )

    def _cleanup_all_features(self) -> None:
        if self._context is None or self._status_label is None:
            return
        deleted_count = self._context.data_service.cleanup_all_features()
        self._refresh_tables(
            f"Feature cleanup completed. Removed {deleted_count} feature record(s)."
        )

    def _rollback_last_feature_run(self) -> None:
        if self._context is None or self._status_label is None:
            return
        deleted_count = self._context.data_service.rollback_last_feature_generation_run()
        if deleted_count == 0:
            self._refresh_tables(
                "No rollback was performed. Last run metadata may be absent or already rolled back."
            )
        else:
            self._refresh_tables(
                f"Rollback completed. Removed {deleted_count} feature record(s)."
            )

    @staticmethod
    def _populate_record_table(table: QTableWidget, rows: List[Dict[str, str]]) -> None:
        table.setRowCount(len(rows))
        for idx, row in enumerate(rows):
            table.setItem(idx, 0, QTableWidgetItem(row.get("id", "")))
            table.setItem(idx, 1, QTableWidgetItem(row.get("type", "")))
            table.setItem(idx, 2, QTableWidgetItem(row.get("path", "")))
        table.resizeColumnsToContents()

    @staticmethod
    def _populate_relation_table(table: QTableWidget, rows: List[Dict[str, str]]) -> None:
        table.setRowCount(len(rows))
        for idx, row in enumerate(rows):
            table.setItem(idx, 0, QTableWidgetItem(row.get("source_type", "")))
            table.setItem(idx, 1, QTableWidgetItem(row.get("source_id", "")))
            table.setItem(idx, 2, QTableWidgetItem(row.get("target_type", "")))
            table.setItem(idx, 3, QTableWidgetItem(row.get("target_id", "")))
        table.resizeColumnsToContents()
