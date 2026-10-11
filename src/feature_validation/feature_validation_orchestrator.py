"""CLI entry point for post-hoc feature validation.

Reads features already written to disk, checks them for finite and physiologically
meaningful values, and writes per-feature, per-participant and index-integrity CSVs.

This never modifies features or the index; it only reports.

    python -m src.feature_validation.feature_validation_orchestrator \
        --sqlite-db-path path/to/index.db --output-dir out/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.app.bootstrap import build_visualization_data_service
from src.feature_validation.csv_export import (
    timestamped_path,
    write_integrity_csv,
    write_per_feature_csv,
    write_per_user_csv,
)
from src.feature_validation.range_catalog import (
    default_range_catalog,
    load_range_catalog,
)
from src.feature_validation.validation_service import run_feature_validation


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc validation of generated gait features"
    )
    parser.add_argument(
        "--sqlite-db-path",
        type=Path,
        required=True,
        help="Path to SQLite metadata index (index.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for validation CSV output",
    )
    parser.add_argument(
        "--range-catalog",
        type=Path,
        default=None,
        help="Override the physiological range catalog JSON",
    )
    parser.add_argument(
        "--basis",
        choices=[basis.value for basis in SampleBasis],
        action="append",
        default=None,
        help="Sample basis to validate; repeatable. Defaults to all.",
    )
    parser.add_argument(
        "--feature-payload-root",
        type=Path,
        action="append",
        default=None,
        help="Directory to scan for orphaned feature payloads; repeatable.",
    )
    return parser.parse_args(argv)


def run_validation(
    sqlite_db_path: Path,
    output_dir: Path,
    catalog_path: Optional[Path] = None,
    bases: Optional[List[SampleBasis]] = None,
    payload_roots: Optional[List[Path]] = None,
) -> int:
    catalog = (
        load_range_catalog(catalog_path) if catalog_path is not None else default_range_catalog()
    )
    resolved_bases = bases or list(SampleBasis)
    data_service = build_visualization_data_service(sqlite_db_path)
    generated_at = datetime.now()

    print("Range catalog: {} feature types (schema {})".format(
        len(catalog.ranges), catalog.schema_version
    ))

    run = run_feature_validation(
        data_service=data_service,
        bases=resolved_bases,
        catalog=catalog,
        feature_payload_roots=payload_roots,
    )
    integrity = run.integrity
    summaries = run.summaries

    print("\n=== Index integrity ===")
    print(integrity.summary_text())

    if run.load_warnings:
        print(
            "\n{} record(s) could not be validated; see the integrity report.".format(
                len(run.load_warnings)
            ),
            file=sys.stderr,
        )

    for summary in summaries:
        print("\n=== Validation: {} basis ===".format(summary.basis.value))
        print(summary.summary_text())

    base = output_dir / "feature_validation.csv"
    feature_csv = write_per_feature_csv(
        timestamped_path(base, "per_feature", generated_at), summaries
    )
    user_csv = write_per_user_csv(
        timestamped_path(base, "per_user", generated_at), summaries
    )
    integrity_csv = write_integrity_csv(
        timestamped_path(base, "index_integrity", generated_at), integrity
    )

    print("\n=== Output ===")
    for label, path in (
        ("per-feature", feature_csv),
        ("per-user", user_csv),
        ("integrity", integrity_csv),
    ):
        print("  {:<12} {}".format(label, path))

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    bases = [SampleBasis(value) for value in args.basis] if args.basis else None
    return run_validation(
        sqlite_db_path=args.sqlite_db_path,
        output_dir=args.output_dir,
        catalog_path=args.range_catalog,
        bases=bases,
        payload_roots=args.feature_payload_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
