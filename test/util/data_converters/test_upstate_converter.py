import csv
import json
import sqlite3
from contextlib import closing
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from src.util.data_converters.upstate.upstate_converter import (
    FREE_LIVING,
    LAB,
    classify_session,
    convert_share,
    imu_id_for,
    index_share,
    load_risk_groups,
)
from test.util.data_converters.gt3x_fixtures import (
    DATE_OF_BIRTH_TICKS,
    SUBJECT_NAME,
    make_gt3x,
)

ROOT = "Carapace_ActigraphDataShare"


class TestShareIndexing(unittest.TestCase):
    def test_classify_session(self):
        cases = {
            f"{ROOT}/FaFRA_AB_001/MOS (2024-12-02).gt3x": LAB,
            f"{ROOT}/FaFRA_AB_001/Activity Data/MOS (2024-12-09).gt3x": FREE_LIVING,
            f"{ROOT}/FaFRA_AB_037/Actiivity Data_AB/MOS.gt3x": FREE_LIVING,
            f"{ROOT}/FaFRA_XY_001/XY_001_ActivityData/MOS.gt3x": FREE_LIVING,
            f"{ROOT}/FaFRA_XY_001/XY_001_Lab Tests/MOS.gt3x": LAB,
            f"{ROOT}/FaFRA_QR_016/1-sec/MOS.gt3x": None,
            f"{ROOT}/CN_Sept2024PilotData/MOS.gt3x": None,
            f"{ROOT}/FaFRA_AB_001/Misc/MOS.gt3x": None,
        }
        for member, expected in cases.items():
            self.assertEqual(classify_session(member), expected, member)

    def test_index_share_and_duplicate_detection(self):
        members = [
            f"{ROOT}/FaaFRA_ZZ_011/MOS.gt3x",
            f"{ROOT}/FaaFRA_ZZ_011/Activity Data/MOS.gt3x",
            f"{ROOT}/FaaFRA_ZZ_011/Activity Data/MOS1sec.agd",
            f"{ROOT}/EmoryShare/ZZ_011/x.csv",
        ]
        found = index_share(members)
        self.assertEqual([(r.participant, r.session) for r in found], [(11, FREE_LIVING), (11, LAB)])
        with self.assertRaises(ValueError):
            index_share(members + [f"{ROOT}/FaaFRA_ZZ_011/second lab.gt3x"])

    def test_imu_ids_are_deterministic_and_distinct(self):
        self.assertEqual(imu_id_for("upstate", 1, LAB), imu_id_for("upstate", 1, LAB))
        self.assertNotEqual(imu_id_for("upstate", 1, LAB), imu_id_for("upstate", 1, FREE_LIVING))
        self.assertTrue(imu_id_for("upstate", 1, LAB).startswith("imu_"))

    def test_risk_groups_flag_participants_listed_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "groups.xlsx"
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame({"Participant ID": ["FaFRA_003", "FaFRA_031"]}).to_excel(
                    writer, sheet_name="Final Risk Group", index=False)
                pd.DataFrame({"Participant ID": ["FaFRA_031"]}).to_excel(
                    writer, sheet_name="Final Middle Group", index=False)
                pd.DataFrame({"Participant ID": ["FaFRA_001"]}).to_excel(
                    writer, sheet_name="Final Non-Risk Group", index=False)
            groups = load_risk_groups(path)
        self.assertEqual(groups, {3: "risk", 31: "conflict:middle|risk", 1: "non-risk"})


class TestConvertShare(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            share = tmp / "share.zip"
            with zipfile.ZipFile(share, "w") as archive:
                archive.writestr(f"{ROOT}/FaFRA_AB_001/MOS (lab).gt3x", make_gt3x(3))
                archive.writestr(f"{ROOT}/FaFRA_AB_001/Activity Data/MOS (fl).gt3x", make_gt3x(6))
                archive.writestr(f"{ROOT}/FaFRA_AB_001/1-sec/MOS (lab).gt3x", make_gt3x(3))
                archive.writestr(
                    f"{ROOT}/FaaFRA_CD_002/Actiivity Data_CD/MOS (fl).gt3x",
                    make_gt3x(5, junk=b"\x00\x07", limb="Waist"),
                )
                archive.writestr(f"{ROOT}/EmoryShare/AB_001/summary.csv", "Subject\nsomeone\n")
            demographics = tmp / "demographics.xlsx"
            pd.DataFrame({
                "Participant ID": ["FaFRA_001"],
                " DOB": ["1950-01-01"],
                "Height (m)": [1.70],
                "Weight (Kg)": [60.0],
                "Fallen in 12 months (n= 0, y = 1)": [1],
            }).to_excel(demographics, sheet_name="Demographics", index=False)

            db = tmp / "db"
            report = convert_share(share, db, demographics_xlsx=demographics, workers=1)

            self.assertEqual(report["n_recordings"], 3)
            self.assertEqual(report["n_failures"], 0)
            self.assertEqual(report["n_repaired"], 1)
            self.assertEqual(len(list((db / "imu_data/upstate").glob("imu_data_*/imu_data.h5"))), 2)
            self.assertEqual(len(list((db / "imu_data/upstate_lab").glob("imu_data_*/imu_data.h5"))), 1)

            demo_1 = json.loads((db / "user_data/upstate/user_FaFRA_001/clinical_demographic_data.json").read_text())
            self.assertEqual(demo_1["faller_status"], "faller")
            self.assertEqual(demo_1["name"], "")
            self.assertEqual((demo_1["sex"], demo_1["height"], demo_1["age"]), ("female", 1.6002, 74.0))
            demo_2 = json.loads((db / "user_data/upstate/user_FaFRA_002/clinical_demographic_data.json").read_text())
            self.assertEqual(demo_2["faller_status"], "unknown")

            with open(db / "conversion_reports/upstate/participants.csv") as handle:
                rows = {r["user_id"]: r for r in csv.DictReader(handle)}
            self.assertEqual(rows["FaFRA_001"]["height_discrepancy_flag"], "True")  # 1.70 vs 1.6002
            self.assertEqual(rows["FaFRA_002"]["limb_label"], "Waist")
            self.assertEqual(rows["FaFRA_002"]["free_living_repaired"], "True")

            spec = json.loads((db / "instrument_specs/upstate/instrument_spec.json").read_text())
            self.assertEqual(spec["imu_name"], "actigraph_wGT3XBT")
            self.assertEqual(spec["sensor_specifications"][0]["range"]["value"], [-8.0, 8.0])

            for name in ("upstate", "upstate_lab"):
                with closing(sqlite3.connect(db / "sql_db_indexes" / name / "index.db")) as conn:
                    tables = [r[0] for r in conn.execute("select name from sqlite_master where type='table'")]
                    self.assertTrue(tables)

            for path in db.rglob("*"):
                if path.is_file():
                    data = path.read_bytes()
                    self.assertNotIn(SUBJECT_NAME.encode(), data, path)
                    self.assertNotIn(DATE_OF_BIRTH_TICKS.encode(), data, path)
                    self.assertNotIn(b"_AB_", data, path)  # folder initials

            # Re-running without --overwrite reuses the records.
            again = convert_share(share, db, demographics_xlsx=demographics, workers=1)
            self.assertEqual({r["status"] for r in again["recordings"]}, {"skipped_existing"})


if __name__ == "__main__":
    unittest.main()
