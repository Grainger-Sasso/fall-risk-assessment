"""CLI: assess a test user's fall risk with a trained model bundle.

Two input modes:

1. Loose record directories (a "new user" walking in the door):
    python -m src.fall_risk_assessment.assess_fall_risk \\
        --bundle-dir /data/models/fall_risk_model_xxx \\
        --imu-dir /data/imu_data/imu_data_imu_abc \\
        --user-dir /data/user_data/user_CO001 \\
        --instrument-spec-dir /data/instrument_specs/ltmm \\
        --output-dir /data/assessments

2. A record already indexed in a SQLite database:
    python -m src.fall_risk_assessment.assess_fall_risk \\
        --bundle-dir /data/models/fall_risk_model_xxx \\
        --sqlite-db-path /data/sql_db_indexes/ltmm/index.db \\
        --imu-id imu_abc \\
        --output-dir /data/assessments
"""

import argparse
from pathlib import Path

from src.database_manager.database_generator import DatabaseGenerator
from src.fall_risk_assessment.inference import FallRiskAssessor
from src.fall_risk_assessment.reporting import render_assessment_html
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess one user's fall risk from an IMU record."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # mode 1: loose dirs
    parser.add_argument("--imu-dir", type=Path, default=None)
    parser.add_argument("--user-dir", type=Path, default=None)
    parser.add_argument("--instrument-spec-dir", type=Path, default=None)
    # mode 2: database record
    parser.add_argument("--sqlite-db-path", type=Path, default=None)
    parser.add_argument("--imu-id", type=str, default=None)
    parser.add_argument(
        "--no-html", action="store_true", help="Skip the HTML report (JSON only)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assessor = FallRiskAssessor.from_bundle_dir(args.bundle_dir)

    if args.sqlite_db_path is not None and args.imu_id:
        db_manager = DatabaseGenerator().generate_database(
            args.sqlite_db_path, validate=True
        )
        assessment = assessor.assess_db_record(
            db_manager, IMUDataIdentifier(args.imu_id)
        )
    elif args.imu_dir is not None and args.user_dir is not None:
        assessment = assessor.assess_record_dirs(
            imu_dir=args.imu_dir,
            user_dir=args.user_dir,
            instrument_spec_dir=args.instrument_spec_dir,
        )
    else:
        raise SystemExit(
            "Provide either --sqlite-db-path with --imu-id, or --imu-dir with "
            "--user-dir."
        )

    output_dir = Path(args.output_dir)
    stem = f"fall_risk_assessment_{assessment.user_id}"
    json_path = assessment.write_json(output_dir / f"{stem}.json")
    print(f"[fall-risk-assess] user={assessment.user_id} "
          f"probability={assessment.risk_probability:.3f} "
          f"category={assessment.risk_category} "
          f"predicted_faller={assessment.predicted_faller}")
    print(f"[fall-risk-assess] JSON: {json_path}")
    if not args.no_html:
        html_path = render_assessment_html(assessment, output_dir / f"{stem}.html")
        print(f"[fall-risk-assess] HTML: {html_path}")


if __name__ == "__main__":
    main()
