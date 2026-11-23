"""Convert the Upstate (Carapace) ActiGraph share into the assessment-database layout.

Output, under ``--assessment-db-root``::

    imu_data/<dataset>/imu_data_<imu id>/imu_data.h5        4-day free-living recordings
    imu_data/<dataset>_lab/imu_data_<imu id>/imu_data.h5    clinic lab sessions
    user_data/<dataset>/user_FaFRA_NNN/{user_data,clinical_demographic_data}.json
    instrument_specs/<dataset>/instrument_spec.json
    sql_db_indexes/<dataset>/index.db, sql_db_indexes/<dataset>_lab/index.db
    conversion_reports/<dataset>/{conversion_report.json, participants.csv}

* Users are keyed by study ID (``FaFRA_NNN``); folder initials, subject names
  and dates of birth in the share are never written out.
* Age, sex, height and mass come from the device ``info.txt`` (present for
  every participant). Height is cross-checked against the demographics
  workbook when one is given.
* ``faller_status`` comes from "Fallen in 12 months" in the demographics
  workbook when present, otherwise ``unknown``. CSM risk groups are recorded
  in ``participants.csv`` only; they are not fall outcomes.
* IMU IDs are deterministic (uuid5 of dataset/participant/session), so a
  re-run overwrites the same records instead of duplicating them.

Run::

    python -m src.util.data_converters.upstate.upstate_converter \\
        --share-zip E:/fafra_data/upstate/Carapace_ActigraphDataShare.zip \\
        --demographics-xlsx E:/fafra_data/raw_data/2024_Upstate/upstate_data/demographics_2025_03_13.xlsx \\
        --risk-groups-xlsx "E:/fafra_data/raw_data/Upstate_2026/Carapace Risk Groups_Pilot for CSM 2026.xlsx" \\
        --assessment-db-root E:/fafra_data/assessment_database
"""

import argparse
import csv
import json
import re
import uuid
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.data_io.import_export.importers.data.imu.imu_data_importer import (
    IMUDataFileNames,
    IMUDataImporter,
)
from src.data_io.import_export.importers.data.user.user_data_importer import (
    UserDataFileNames,
)
from src.data_io.model_fields.data.user.clinical_demographic_data_fields import (
    ClinicalDemographicDataFields,
)
from src.data_io.model_fields.data.user.user_data_fields import UserDataFields
from src.data_model.data.user.clinical.clinical_demographic_data import FallerStatus
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from src.util.data_converters.imu.gt3x.gt3x_to_h5_converter import (
    GT3XInfo,
    GT3XToH5Converter,
    read_gt3x,
)
from src.util.dataset_generator.dataset_generator import build_sql_index

FREE_LIVING = "free_living"
LAB = "lab"
SESSIONS = (FREE_LIVING, LAB)
PARTICIPANT_DIR = re.compile(r"^Fa+FRA_[A-Za-z]{2}_(\d{3})$")
IMU_NAMESPACE = uuid.UUID("6f1f0b0e-5a0c-4c55-9d43-2f7e0b7c1a11")
HEIGHT_DISCREPANCY_CM = 5.0


def user_id_for(participant: int) -> str:
    return f"FaFRA_{participant:03d}"


def imu_id_for(dataset: str, participant: int, session: str) -> str:
    return f"imu_{uuid.uuid5(IMU_NAMESPACE, f'{dataset}/{user_id_for(participant)}/{session}')}"


@dataclass(frozen=True)
class ShareRecording:
    participant: int
    session: str
    member: str  # path inside the zip


def classify_session(member: str) -> Optional[str]:
    """Session of a ``.gt3x`` inside a participant folder, or None to skip it."""
    parts = member.split("/")
    try:
        start = next(i for i, part in enumerate(parts) if PARTICIPANT_DIR.match(part))
    except StopIteration:
        return None
    subfolders = [p.lower() for p in parts[start + 1 : -1]]
    if any(p == "1-sec" for p in subfolders):
        return None  # byte-identical duplicate of the lab file (P016)
    if any("activ" in p or "actiiv" in p for p in subfolders):
        return FREE_LIVING
    if not subfolders or any("lab" in p for p in subfolders):
        return LAB
    return None


def index_share(members: Sequence[str]) -> List[ShareRecording]:
    """Find one lab and one free-living ``.gt3x`` per participant folder."""
    found: Dict[tuple, str] = {}
    for member in members:
        if not member.lower().endswith(".gt3x"):
            continue
        session = classify_session(member)
        if session is None:
            continue
        folder = next(p for p in member.split("/") if PARTICIPANT_DIR.match(p))
        participant = int(PARTICIPANT_DIR.match(folder).group(1))
        key = (participant, session)
        if key in found:
            raise ValueError(f"Two {session} recordings for {user_id_for(participant)}: {found[key]} and {member}")
        found[key] = member
    return [ShareRecording(p, s, m) for (p, s), m in sorted(found.items())]


def load_fall_history(path: Optional[Path]) -> Dict[int, dict]:
    """participant -> {fallen_12mo, height_m, weight_kg} from the demographics workbook."""
    if path is None:
        return {}
    import pandas as pd

    frame = pd.read_excel(path, sheet_name="Demographics")
    fall_col = next(c for c in frame.columns if str(c).lower().startswith("fallen in 12 months"))
    out: Dict[int, dict] = {}
    for _, row in frame.iterrows():
        match = re.search(r"(\d{3})", str(row.get("Participant ID", "")))
        if not match:
            continue
        fallen = row[fall_col]
        out[int(match.group(1))] = {
            "fallen_12mo": None if pd.isna(fallen) else int(fallen),
            "height_m": None if pd.isna(row.get("Height (m)")) else float(row["Height (m)"]),
            "weight_kg": None if pd.isna(row.get("Weight (Kg)")) else float(row["Weight (Kg)"]),
        }
    return out


def load_risk_groups(path: Optional[Path]) -> Dict[int, str]:
    """participant -> risk group; participants listed in several groups get 'conflict:<a>|<b>'."""
    if path is None:
        return {}
    import pandas as pd

    groups: Dict[int, List[str]] = {}
    with pd.ExcelFile(path) as book:
        for sheet in book.sheet_names:
            label = re.sub(r"^final\s+|\s+group$", "", sheet.strip(), flags=re.I).lower().replace(" ", "_")
            frame = book.parse(sheet_name=sheet)
            for value in frame["Participant ID"].dropna():
                match = re.search(r"(\d{3})", str(value))
                if match:
                    groups.setdefault(int(match.group(1)), []).append(label)
    return {
        p: labels[0] if len(set(labels)) == 1 else "conflict:" + "|".join(sorted(set(labels)))
        for p, labels in groups.items()
    }


def instrument_spec_json(spec_id: str, info: GT3XInfo) -> dict:
    def value(v):
        return {"value": v}

    return {
        "spec_id": spec_id,
        "imu_name": info.instrument_name,
        "sensor_specifications": [
            {
                "sensor_type": "accelerometer",
                "sensor_name": f"{info.instrument_name}_accelerometer",
                "units": "g",
                "range": value([info.acceleration_min, info.acceleration_max]),
                "sensitivity": value(None),
                "resolution": value(12),
                "sampling_rate": value(float(info.sample_rate)),
                "noise_density": value(None),
                "bias_stability": value(None),
                "alignment_error": value(None),
                "cross_axis_sensitivity": value(None),
                "power_consumption": value(None),
                "operating_conditions": value({"temperature": None, "humidity": None}),
                "physical_size": {"length": None, "width": None, "height": None},
                "mass": None,
            }
        ],
    }


def _record_dir(imu_parent: Path, imu_id: str) -> Path:
    return imu_parent / f"imu_data_{imu_id}"


def _already_converted(record_dir: Path, imu_id: str) -> bool:
    if not (record_dir / f"{IMUDataFileNames.IMU_DATA.value}.h5").exists():
        return False
    try:
        return IMUDataImporter().import_header(record_dir).imu_data_identifier.value == imu_id
    except Exception:
        return False


def convert_recording(
    share_zip: str, recording: ShareRecording, imu_parent: str, imu_id: str, overwrite: bool
) -> dict:
    """Worker: parse (with repair), export and summarise one recording."""
    imu_parent_path = Path(imu_parent)
    summary = {
        "user_id": user_id_for(recording.participant),
        "session": recording.session,
        "imu_id": imu_id,
    }
    with zipfile.ZipFile(share_zip) as archive:
        payload = archive.read(recording.member)
    parsed = read_gt3x(payload)
    del payload
    info = parsed.info
    summary.update(
        n_samples=parsed.n_samples,
        duration_hours=round(parsed.duration_hours, 4),
        idle_sleep_fraction=round(float(parsed.idle_mask.mean()), 4),
        max_gap_s=round(float(np.diff(parsed.time).max()), 3) if parsed.n_samples > 1 else None,
        monotonic=bool(np.all(np.diff(parsed.time) > 0)),
        repaired=parsed.repair.repaired,
        repair_dropped_bytes=parsed.repair.dropped_bytes,
        repair_junk_runs=parsed.repair.junk_runs,
        info={
            "serial_number": info.serial_number,
            "device_type": info.device_type,
            "firmware": info.firmware,
            "sample_rate": info.sample_rate,
            "acceleration_scale": info.acceleration_scale,
            "acceleration_min": info.acceleration_min,
            "acceleration_max": info.acceleration_max,
            "sex": info.sex,
            "height_cm": info.height_cm,
            "mass_kg": info.mass_kg,
            "age": info.age,
            "limb": info.limb,
        },
    )
    record_dir = _record_dir(imu_parent_path, imu_id)
    if not overwrite and _already_converted(record_dir, imu_id):
        summary["status"] = "skipped_existing"
        summary["record_dir"] = str(record_dir)
        return summary
    converter = GT3XToH5Converter()
    imu_data = converter.build_imu_data(
        parsed, IMUDataIdentifier(imu_id), UserIdentifier(user_id_for(recording.participant))
    )
    del parsed
    summary["record_dir"] = str(converter.export(imu_data, imu_parent_path))
    summary["status"] = "converted"
    return summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_user_records(
    user_parent: Path,
    participant: int,
    info: dict,
    fall_history: Optional[dict],
) -> str:
    user_id = user_id_for(participant)
    user_dir = user_parent / f"user_{user_id}"
    fallen = (fall_history or {}).get("fallen_12mo")
    if fallen is None:
        status = FallerStatus.UNKNOWN
    else:
        status = FallerStatus.from_bool(bool(fallen))
    _write_json(
        user_dir / f"{UserDataFileNames.USER_DATA.value}.json",
        {UserDataFields.USER_DATA_IDENTIFIER.value: user_id},
    )
    _write_json(
        user_dir / f"{UserDataFileNames.CLININCAL_DEMOGRAPHIC_DATA.value}.json",
        {
            ClinicalDemographicDataFields.NAME.value: "",
            ClinicalDemographicDataFields.AGE.value: info["age"],
            ClinicalDemographicDataFields.SEX.value: (info["sex"] or "").strip().lower() or None,
            ClinicalDemographicDataFields.WEIGHT.value: info["mass_kg"],
            ClinicalDemographicDataFields.HEIGHT.value: round(info["height_cm"] / 100.0, 4)
            if info["height_cm"]
            else None,
            ClinicalDemographicDataFields.IDENTIFIER.value: user_id,
            ClinicalDemographicDataFields.FALLER_STATUS.value: status.value,
        },
    )
    return status.value


def convert_share(
    share_zip: Path,
    assessment_db_root: Path,
    dataset: str = "upstate",
    demographics_xlsx: Optional[Path] = None,
    risk_groups_xlsx: Optional[Path] = None,
    workers: int = 2,
    overwrite: bool = False,
    participants: Optional[Sequence[int]] = None,
    build_index: bool = True,
) -> dict:
    root = Path(assessment_db_root)
    imu_parents = {FREE_LIVING: root / "imu_data" / dataset, LAB: root / "imu_data" / f"{dataset}_lab"}
    user_parent = root / "user_data" / dataset
    spec_dir = root / "instrument_specs" / dataset
    report_dir = root / "conversion_reports" / dataset
    index_paths = {
        FREE_LIVING: root / "sql_db_indexes" / dataset / "index.db",
        LAB: root / "sql_db_indexes" / f"{dataset}_lab" / "index.db",
    }
    for path in [*imu_parents.values(), user_parent, spec_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(share_zip) as archive:
        recordings = index_share(archive.namelist())
    if participants:
        wanted = set(participants)
        recordings = [r for r in recordings if r.participant in wanted]
    fall_history = load_fall_history(demographics_xlsx)
    risk_groups = load_risk_groups(risk_groups_xlsx)
    print(f"[upstate] {len(recordings)} recordings from "
          f"{len({r.participant for r in recordings})} participants", flush=True)

    results: List[dict] = []
    failures: List[dict] = []
    ordered = sorted(recordings, key=lambda r: r.session != FREE_LIVING)  # big files first

    def job(rec: ShareRecording) -> tuple:
        return (str(share_zip), rec, str(imu_parents[rec.session]),
                imu_id_for(dataset, rec.participant, rec.session), overwrite)

    def record(rec: ShareRecording, outcome) -> None:
        try:
            summary = outcome()
        except Exception as exc:  # noqa: BLE001 - recorded and reported
            failures.append({"user_id": user_id_for(rec.participant), "session": rec.session,
                             "error": f"{type(exc).__name__}: {exc}"})
            print(f"[upstate] FAILED {user_id_for(rec.participant)} {rec.session}: {exc}", flush=True)
            return
        results.append(summary)
        flag = " (repaired)" if summary["repaired"] else ""
        print(f"[upstate] {summary['status']} {summary['user_id']} {summary['session']}: "
              f"{summary['duration_hours']:.2f} h{flag}", flush=True)

    if workers <= 1:
        for rec in ordered:
            record(rec, lambda rec=rec: convert_recording(*job(rec)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(convert_recording, *job(rec)): rec for rec in ordered}
            for future in as_completed(futures):
                record(futures[future], future.result)
    if not results:
        raise RuntimeError(f"No recordings converted; {len(failures)} failed.")

    # Session sanity: free-living recordings are multi-day, lab sessions under 2 h.
    for summary in results:
        hours = summary["duration_hours"]
        if (summary["session"] == FREE_LIVING and hours < 24) or (summary["session"] == LAB and hours > 2):
            summary.setdefault("warnings", []).append(f"unexpected duration {hours:.2f} h for {summary['session']}")

    # Users, from device info (prefer the free-living file; both agree in this share).
    by_user: Dict[str, Dict[str, dict]] = {}
    for summary in results:
        by_user.setdefault(summary["user_id"], {})[summary["session"]] = summary
    participant_rows: List[dict] = []
    for user_id, sessions in sorted(by_user.items()):
        participant = int(user_id[-3:])
        info = (sessions.get(FREE_LIVING) or sessions[LAB])["info"]
        history = fall_history.get(participant)
        status = write_user_records(user_parent, participant, info, history)
        sheet_height = (history or {}).get("height_m")
        device_height = round(info["height_cm"] / 100.0, 4) if info["height_cm"] else None
        discrepancy = (
            round(abs(sheet_height - device_height) * 100, 1)
            if sheet_height is not None and device_height is not None
            else None
        )
        participant_rows.append({
            "user_id": user_id,
            "limb_label": info["limb"],
            "faller_status": status,
            "fall_history_source": "demographics_xlsx" if history and history.get("fallen_12mo") is not None else "",
            "risk_group": risk_groups.get(participant, ""),
            "device_height_m": device_height,
            "sheet_height_m": sheet_height,
            "height_discrepancy_cm": discrepancy,
            "height_discrepancy_flag": bool(discrepancy is not None and discrepancy > HEIGHT_DISCREPANCY_CM),
            **{f"{s}_imu_id": sessions[s]["imu_id"] if s in sessions else "" for s in SESSIONS},
            **{f"{s}_hours": sessions[s]["duration_hours"] if s in sessions else None for s in SESSIONS},
            **{f"{s}_repaired": sessions[s]["repaired"] if s in sessions else None for s in SESSIONS},
        })

    # One instrument spec: every device in the share is the same model and configuration.
    infos = {(s["info"]["device_type"], s["info"]["sample_rate"], s["info"]["acceleration_min"],
              s["info"]["acceleration_max"]) for s in results}
    if len(infos) != 1:
        raise ValueError(f"Expected one device configuration, found {sorted(infos)}")
    first = results[0]["info"]
    spec_info = GT3XInfo(
        serial_number="", device_type=first["device_type"], firmware=first["firmware"],
        sample_rate=first["sample_rate"], acceleration_scale=first["acceleration_scale"],
        acceleration_min=first["acceleration_min"], acceleration_max=first["acceleration_max"],
        timezone=None, sex=None, height_cm=None, mass_kg=None, age=None, limb=None,
    )
    spec_path = spec_dir / "instrument_spec.json"
    _write_json(spec_path, instrument_spec_json(f"{dataset}_actigraph_{first['device_type'].lower()}_v1", spec_info))

    indexes = {}
    if build_index:
        for session in SESSIONS:
            if any(r["session"] == session for r in results):
                index_paths[session].parent.mkdir(parents=True, exist_ok=True)
                indexes[session] = str(build_sql_index(
                    imu_parent_dir=imu_parents[session],
                    user_parent_dir=user_parent,
                    instrument_spec_inputs=spec_path,
                    sqlite_db_path=index_paths[session],
                ))

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "share_zip": str(share_zip),
        "dataset": dataset,
        "n_recordings": len(results),
        "n_failures": len(failures),
        "n_repaired": sum(1 for r in results if r["repaired"]),
        "imu_parents": {k: str(v) for k, v in imu_parents.items()},
        "user_parent": str(user_parent),
        "instrument_spec": str(spec_path),
        "indexes": indexes,
        "failures": failures,
        "recordings": sorted(
            [{k: v for k, v in r.items() if k != "info"} for r in results],
            key=lambda r: (r["user_id"], r["session"]),
        ),
    }
    _write_json(report_dir / "conversion_report.json", report)
    with open(report_dir / "participants.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(participant_rows[0].keys()))
        writer.writeheader()
        writer.writerows(participant_rows)
    print(f"[upstate] converted={sum(r['status'] == 'converted' for r in results)} "
          f"skipped={sum(r['status'] == 'skipped_existing' for r in results)} "
          f"repaired={report['n_repaired']} failed={len(failures)}; report: {report_dir}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the Upstate ActiGraph share to the assessment database.")
    parser.add_argument("--share-zip", type=Path, required=True)
    parser.add_argument("--assessment-db-root", type=Path, required=True)
    parser.add_argument("--dataset", default="upstate")
    parser.add_argument("--demographics-xlsx", type=Path, default=None,
                        help="Workbook with a 'Demographics' sheet holding 'Fallen in 12 months'.")
    parser.add_argument("--risk-groups-xlsx", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel recordings; each 4-day file needs ~5 GB RAM.")
    parser.add_argument("--participants", type=str, default="",
                        help="Optional comma-separated study numbers to convert (e.g. 1,13,52).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-index", action="store_true")
    args = parser.parse_args()
    participants = [int(p) for p in args.participants.split(",") if p.strip()]
    report = convert_share(
        share_zip=args.share_zip,
        assessment_db_root=args.assessment_db_root,
        dataset=args.dataset,
        demographics_xlsx=args.demographics_xlsx,
        risk_groups_xlsx=args.risk_groups_xlsx,
        workers=args.workers,
        overwrite=args.overwrite,
        participants=participants or None,
        build_index=not args.no_index,
    )
    if report["n_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
