"""Feature validation panel: missingness, physiological range violations, index drift.

Complements the Feature Quality plugin. Quality answers "is this feature present and
separable?"; validation answers "is this feature's value physically believable?".
Violations are reported in two tiers so clinically slow-but-real gait is never
presented as an extraction defect.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.plugins.base import PluginContext, VisualizationPlugin
from src.feature_validation.aggregation import ValidationSummary
from src.feature_validation.csv_export import (
    timestamped_path,
    write_integrity_csv,
    write_per_feature_csv,
    write_per_user_csv,
)
from src.feature_validation.range_catalog import default_range_catalog
from src.feature_validation.validation_service import ValidationRun, run_feature_validation

VIEW_PER_FEATURE = "Per feature type"
VIEW_PER_USER = "Per participant"
VIEW_INTEGRITY = "Index integrity"

SORT_WORST_MISSING = "Most missing first"
SORT_WORST_IMPLAUSIBLE = "Most implausible first"
SORT_WORST_ATYPICAL = "Most atypical first"
SORT_NAME = "Name"

PER_FEATURE_HEADERS = [
    "Feature type",
    "Unit",
    "Implausible provenance",
    "Atypical provenance",
    "Participants",
    "Values",
    "Missing %",
    "Implausible",
    "Implausible %",
    "Atypical",
    "Atypical %",
    "Min",
    "Max",
    "Worst implausible",
]

PER_USER_HEADERS = [
    "Participant",
    "Class",
    "Records",
    "Usable samples",
    "Values",
    "Missing %",
    "Implausible",
    "Atypical",
    "Meaningful %",
    "Feature types fully missing",
]

INTEGRITY_HEADERS = ["Metric", "Value", "Detail"]


class FeatureValidationPlugin(VisualizationPlugin):
    """Post-hoc physiological validation of generated features."""

    plugin_id = "feature_validation"
    display_name = "Feature Validation"

    def __init__(self) -> None:
        self._context: Optional[PluginContext] = None
        self._run: Optional[ValidationRun] = None
        self._catalog = default_range_catalog()

    def build_panel(self, parent: QWidget, context: PluginContext) -> QWidget:
        self._context = context
        panel = QWidget(parent)
        root_layout = QHBoxLayout(panel)
        root_layout.addWidget(self._build_controls(panel), stretch=0)
        root_layout.addWidget(self._build_results(panel), stretch=1)
        return panel

    def _build_controls(self, parent: QWidget) -> QWidget:
        container = QWidget(parent)
        container.setMaximumWidth(360)
        layout = QVBoxLayout(container)

        group = QGroupBox("Validation", container)
        form = QFormLayout(group)

        self.basis_selector = QComboBox(group)
        self.basis_selector.addItems([basis.value for basis in SampleBasis])
        form.addRow("Sample basis", self.basis_selector)

        self.view_selector = QComboBox(group)
        self.view_selector.addItems([VIEW_PER_FEATURE, VIEW_PER_USER, VIEW_INTEGRITY])
        self.view_selector.currentTextChanged.connect(self._render_current)
        form.addRow("View", self.view_selector)

        self.sort_selector = QComboBox(group)
        self.sort_selector.addItems(
            [SORT_WORST_MISSING, SORT_WORST_IMPLAUSIBLE, SORT_WORST_ATYPICAL, SORT_NAME]
        )
        self.sort_selector.currentTextChanged.connect(self._render_current)
        form.addRow("Sort", self.sort_selector)

        layout.addWidget(group)

        self.run_button = QPushButton("Run validation", container)
        self.run_button.clicked.connect(self._run_validation)
        layout.addWidget(self.run_button)

        self.export_button = QPushButton("Export CSV...", container)
        self.export_button.clicked.connect(self._export_csv)
        self.export_button.setEnabled(False)
        layout.addWidget(self.export_button)

        self.catalog_label = QLabel(
            "Range catalog: {} feature types (schema {})".format(
                len(self._catalog.ranges), self._catalog.schema_version
            ),
            container,
        )
        self.catalog_label.setWordWrap(True)
        layout.addWidget(self.catalog_label)

        layout.addStretch(1)
        return container

    def _build_results(self, parent: QWidget) -> QWidget:
        container = QWidget(parent)
        layout = QVBoxLayout(container)

        self.summary_label = QLabel("Click 'Run validation' to check features.", container)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.summary_label, stretch=0)

        self.table = QTableWidget(container)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, stretch=1)

        return container

    def _run_validation(self) -> None:
        if self._context is None:
            return
        self.run_button.setEnabled(False)
        self.summary_label.setText("Validating features...")
        try:
            self._run = run_feature_validation(
                data_service=self._context.data_service,
                catalog=self._catalog,
            )
        except Exception as exc:
            self._run = None
            self.summary_label.setText("Validation failed: {}: {}".format(type(exc).__name__, exc))
            return
        finally:
            self.run_button.setEnabled(True)

        self.export_button.setEnabled(True)
        self._render_current()

    def _current_summary(self) -> Optional[ValidationSummary]:
        if self._run is None:
            return None
        return self._run.summary_for(SampleBasis(self.basis_selector.currentText()))

    def _render_current(self) -> None:
        if self._run is None:
            return
        view = self.view_selector.currentText()
        if view == VIEW_INTEGRITY:
            self._render_integrity()
        elif view == VIEW_PER_USER:
            self._render_per_user()
        else:
            self._render_per_feature()

    def _set_table(self, headers: List[str], rows: List[List[str]]) -> None:
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                self.table.setItem(row_index, column_index, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    def _render_per_feature(self) -> None:
        summary = self._current_summary()
        if summary is None or summary.is_empty:
            self.summary_label.setText("No validated records for this basis.")
            self._set_table(PER_FEATURE_HEADERS, [])
            return

        features = list(summary.per_feature)
        sort_mode = self.sort_selector.currentText()
        if sort_mode == SORT_WORST_MISSING:
            features.sort(key=lambda f: -f.missing_fraction)
        elif sort_mode == SORT_WORST_IMPLAUSIBLE:
            features.sort(key=lambda f: -f.implausible_fraction)
        elif sort_mode == SORT_WORST_ATYPICAL:
            features.sort(
                key=lambda f: -(f.atypical_values / max(f.finite_values, 1))
            )
        else:
            features.sort(key=lambda f: str(getattr(f.feature_type, "value", f.feature_type)))

        rows = []
        for feature in features:
            atypical_fraction = feature.atypical_values / max(feature.finite_values, 1)
            rows.append(
                [
                    str(getattr(feature.feature_type, "value", feature.feature_type)),
                    feature.unit or "-",
                    feature.implausible_provenance or "-",
                    feature.atypical_provenance or "-",
                    str(feature.participants),
                    str(feature.total_values),
                    "{:.2f}".format(100.0 * feature.missing_fraction),
                    str(feature.implausible_values),
                    "{:.3f}".format(100.0 * feature.implausible_fraction),
                    str(feature.atypical_values),
                    "{:.2f}".format(100.0 * atypical_fraction),
                    _fmt_optional(feature.min_value),
                    _fmt_optional(feature.max_value),
                    _fmt_optional(feature.worst_implausible),
                ]
            )
        self.summary_label.setText(summary.summary_text())
        self._set_table(PER_FEATURE_HEADERS, rows)

    def _render_per_user(self) -> None:
        summary = self._current_summary()
        if summary is None or summary.is_empty:
            self.summary_label.setText("No validated records for this basis.")
            self._set_table(PER_USER_HEADERS, [])
            return

        users = list(summary.per_user)
        sort_mode = self.sort_selector.currentText()
        if sort_mode == SORT_WORST_MISSING:
            users.sort(key=lambda u: u.finite_fraction)
        elif sort_mode == SORT_WORST_IMPLAUSIBLE:
            users.sort(key=lambda u: -u.implausible_fraction)
        elif sort_mode == SORT_WORST_ATYPICAL:
            users.sort(key=lambda u: -(u.atypical_values / max(u.finite_values, 1)))
        else:
            users.sort(key=lambda u: u.participant_id)

        rows = []
        for user in users:
            rows.append(
                [
                    user.participant_id,
                    user.class_label,
                    str(user.feature_records),
                    str(user.usable_samples),
                    str(user.total_values),
                    "{:.2f}".format(100.0 * (1.0 - user.finite_fraction)),
                    str(user.implausible_values),
                    str(user.atypical_values),
                    "{:.2f}".format(100.0 * user.meaningful_fraction),
                    ", ".join(user.feature_types_all_missing) or "-",
                ]
            )
        self.summary_label.setText(summary.summary_text())
        self._set_table(PER_USER_HEADERS, rows)

    def _render_integrity(self) -> None:
        if self._run is None:
            return
        report = self._run.integrity
        rows = [
            ["Indexed feature records", str(report.indexed_feature_count), ""],
            ["Loadable feature records", str(report.loadable_feature_count), ""],
            ["Index overstatement", str(report.index_overstatement), ""],
            [
                "Dangling index rows",
                str(report.dangling_count),
                ", ".join(report.dangling_feature_ids[:8]),
            ],
            [
                "Unreadable payloads",
                str(report.unreadable_count),
                ", ".join(sorted(report.unreadable_feature_ids)[:8]),
            ],
            [
                "Orphaned payload dirs",
                str(report.orphaned_count),
                ", ".join(Path(p).name for p in report.orphaned_payload_dirs[:8]),
            ],
            ["Indexed IMU records", str(report.indexed_imu_count), ""],
            [
                "IMU records without features",
                str(len(report.imu_ids_without_features)),
                ", ".join(report.imu_ids_without_features[:8]),
            ],
            [
                "Participants with features",
                str(len(report.participants_with_features)),
                "",
            ],
        ]
        self.summary_label.setText(report.summary_text())
        self._set_table(INTEGRITY_HEADERS, rows)

    def _export_csv(self) -> None:
        if self._run is None:
            return
        directory = QFileDialog.getExistingDirectory(None, "Select CSV output directory")
        if not directory:
            return
        generated_at = datetime.now()
        base = Path(directory) / "feature_validation.csv"
        try:
            written = [
                write_per_feature_csv(
                    timestamped_path(base, "per_feature", generated_at), self._run.summaries
                ),
                write_per_user_csv(
                    timestamped_path(base, "per_user", generated_at), self._run.summaries
                ),
                write_integrity_csv(
                    timestamped_path(base, "index_integrity", generated_at),
                    self._run.integrity,
                ),
            ]
        except OSError as exc:
            QMessageBox.warning(None, "Export failed", str(exc))
            return
        QMessageBox.information(
            None,
            "Export complete",
            "Wrote:\n" + "\n".join(path.name for path in written),
        )


def _fmt_optional(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return "{:.4g}".format(value)
