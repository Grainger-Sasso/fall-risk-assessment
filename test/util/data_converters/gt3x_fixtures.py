"""Synthetic .gt3x files that pygt3x can parse, with optional stray bytes in log.bin."""

import io
import struct
import zipfile
from typing import Optional

import numpy as np

from src.util.data_converters.imu.gt3x.gt3x_log_repair import record_checksum

SAMPLE_RATE = 10
START_EPOCH_S = 1_733_000_000
TICKS_AT_UNIX_EPOCH = 621_355_968_000_000_000
ACTIVITY2 = 26
METADATA = 6
SUBJECT_NAME = "Test Subject Person"
DATE_OF_BIRTH_TICKS = "615569479800000000"


def make_record(record_type: int, timestamp: int, payload: bytes) -> bytes:
    body = bytes([0x1E, record_type]) + struct.pack("<IH", timestamp, len(payload)) + payload
    return body + bytes([record_checksum(body)])


def counts_for_second(second: int) -> np.ndarray:
    """Deterministic int16 counts (x, y, z) for one second; y holds ~1 g of gravity."""
    i = np.arange(SAMPLE_RATE)
    x = (second * 7 + i * 3) % 41 - 20
    y = 256 + (i % 5)
    z = -((second + i) % 13)
    return np.column_stack([x, y, z]).astype("<i2")


def make_log(n_seconds: int, junk: bytes = b"", junk_after_records: int = 1) -> bytes:
    records = [make_record(METADATA, START_EPOCH_S, b'{"MetadataType":"Test"}')]
    for s in range(n_seconds):
        records.append(make_record(ACTIVITY2, START_EPOCH_S + s, counts_for_second(s).tobytes()))
    return b"".join(records[:junk_after_records]) + junk + b"".join(records[junk_after_records:])


def make_info(n_seconds: int, limb: str = "Back") -> str:
    start = TICKS_AT_UNIX_EPOCH + START_EPOCH_S * 10_000_000
    last = start + n_seconds * 10_000_000
    return "\n".join([
        "Serial Number: TEST0000001",
        "Device Type: wGT3XBT",
        "Firmware: 1.9.2",
        "Battery Voltage: 4.10",
        f"Sample Rate: {SAMPLE_RATE}",
        f"Start Date: {start}",
        "Stop Date: 0",
        f"Last Sample Time: {last}",
        "TimeZone: -05:00:00",
        f"Download Date: {last}",
        "Acceleration Scale: 256.0",
        "Acceleration Min: -8.0",
        "Acceleration Max: 8.0",
        "Sex: Female",
        "Height: 160.02",
        "Mass: 60.5",
        "Age: 74",
        f"Limb: {limb}",
        f"DateOfBirth: {DATE_OF_BIRTH_TICKS}",
        f"Subject Name: {SUBJECT_NAME}",
    ]) + "\n"


def make_gt3x(n_seconds: int = 5, junk: bytes = b"", limb: str = "Back", log: Optional[bytes] = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("log.bin", log if log is not None else make_log(n_seconds, junk))
        archive.writestr("info.txt", make_info(n_seconds, limb))
    return buffer.getvalue()
