import io
import json
import logging
import tempfile
import unittest
import zipfile
from pathlib import Path

from pygt3x.reader import FileReader

from src.util.data_converters.imu.gt3x.gt3x_log_repair import (
    record_checksum,
    repair_gt3x_bytes,
    repair_share,
    resync_log,
    scan_record_chain,
)
from test.util.data_converters.gt3x_fixtures import (
    SAMPLE_RATE,
    make_gt3x,
    make_log,
    make_record,
)

N_SECONDS = 6


def _samples(gt3x: bytes) -> int:
    logging.disable(logging.WARNING)
    try:
        with FileReader(io.BytesIO(gt3x)) as reader:
            return int(reader.acceleration.shape[0])
    finally:
        logging.disable(logging.NOTSET)


class TestRecordChecksum(unittest.TestCase):
    def test_is_ones_complement_of_xor(self):
        self.assertEqual(record_checksum(bytes([0x1E, 0x00])), (~0x1E) & 0xFF)
        self.assertEqual(record_checksum(b""), 0xFF)

    def test_made_records_validate(self):
        record = make_record(26, 123, b"\x01\x02\x03")
        self.assertEqual(record_checksum(record[:-1]), record[-1])


class TestScanAndResync(unittest.TestCase):
    def test_intact_chain(self):
        scan = scan_record_chain(make_log(N_SECONDS))
        self.assertTrue(scan.intact)
        self.assertEqual(scan.n_records, N_SECONDS + 1)

    def test_break_is_located_at_the_stray_bytes(self):
        clean = make_log(N_SECONDS)
        first_record_len = len(make_log(0))
        broken = make_log(N_SECONDS, junk=b"\x00\x07\x00")
        scan = scan_record_chain(broken)
        self.assertFalse(scan.intact)
        self.assertEqual(scan.first_break_offset, first_record_len)

        repaired, report = resync_log(broken)
        self.assertEqual(repaired, clean)
        self.assertEqual(report.dropped_bytes, 3)
        self.assertEqual(report.junk_runs, [(first_record_len, 3)])
        self.assertEqual(report.n_records, N_SECONDS + 1)

    def test_resync_drops_a_corrupt_record_mid_stream(self):
        log = bytearray(make_log(N_SECONDS))
        second_record = len(make_log(0))
        third_record = len(make_log(1))
        log[second_record + 10] ^= 0xFF  # corrupt a payload byte -> bad checksum
        log[second_record + 6] ^= 0x01  # and its size -> chain breaks
        repaired, report = resync_log(bytes(log))
        self.assertEqual(report.n_records, N_SECONDS)
        self.assertEqual(report.junk_runs, [(second_record, third_record - second_record)])
        self.assertEqual(repaired, make_log(0) + make_log(N_SECONDS)[third_record:])


class TestRepairGt3x(unittest.TestCase):
    def test_intact_file_is_returned_unchanged(self):
        gt3x = make_gt3x(N_SECONDS)
        repaired, report = repair_gt3x_bytes(gt3x)
        self.assertIs(repaired, gt3x)
        self.assertFalse(report.repaired)
        self.assertEqual(report.dropped_bytes, 0)

    def test_stray_bytes_break_pygt3x_and_repair_restores_every_sample(self):
        expected = N_SECONDS * SAMPLE_RATE
        self.assertEqual(_samples(make_gt3x(N_SECONDS)), expected)

        broken = make_gt3x(N_SECONDS, junk=b"\x00\x07")
        try:
            broken_samples = _samples(broken)
        except Exception:
            broken_samples = -1
        self.assertNotEqual(broken_samples, expected)

        repaired, report = repair_gt3x_bytes(broken)
        self.assertTrue(report.repaired)
        self.assertEqual(report.dropped_bytes, 2)
        self.assertEqual(_samples(repaired), expected)
        with zipfile.ZipFile(io.BytesIO(repaired)) as archive:
            self.assertEqual(archive.read("log.bin"), make_log(N_SECONDS))
            self.assertEqual(sorted(archive.namelist()), ["info.txt", "log.bin"])


class TestRepairShare(unittest.TestCase):
    def test_writes_only_defective_files_and_a_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            share = tmp / "share.zip"
            with zipfile.ZipFile(share, "w") as archive:
                archive.writestr("Share/P_A_001/ok.gt3x", make_gt3x(3))
                archive.writestr("Share/P_A_001/Activity Data/bad.gt3x", make_gt3x(3, junk=b"\x01"))
                archive.writestr("Share/notes.csv", "x")
            out = tmp / "repaired"
            rows = repair_share(share, out)

            self.assertEqual([r["file"] for r in rows], ["Share/P_A_001/Activity Data/bad.gt3x"])
            self.assertTrue((out / "Share/P_A_001/Activity Data/bad.gt3x").exists())
            self.assertFalse((out / "Share/P_A_001/ok.gt3x").exists())
            report = json.loads((out / "repair_report.json").read_text())
            self.assertEqual(report[0]["dropped_bytes"], 1)
            self.assertEqual(_samples((out / "Share/P_A_001/Activity Data/bad.gt3x").read_bytes()), 3 * SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
