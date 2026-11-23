"""Detect and repair misaligned record streams in ActiGraph ``.gt3x`` files.

A ``.gt3x`` file is a zip holding ``info.txt`` and ``log.bin``. ``log.bin`` is
a chain of records::

    0x1E | type (1 B) | timestamp (4 B, LE) | payload size (2 B, LE) | payload | checksum (1 B)

where ``checksum`` is the one's complement of the XOR of every preceding byte
of the record (separator through payload).

Some files carry a few stray bytes between records. ``pygt3x`` does not
resynchronise after them, and it fails in one of three ways depending on what
the misaligned bytes decode to:

* zero samples are returned;
* hours of data are **silently** dropped;
* a garbage record passes its 8-bit checksum and parsing raises.

The repair rebuilds ``log.bin`` from the records that start with the
separator, carry a known record type and pass their checksum. Only bytes that
belong to no valid record are dropped. Files whose chain is intact are
returned unchanged.

Command line (writes repaired copies of every defective file in a share)::

    python -m src.util.data_converters.imu.gt3x.gt3x_log_repair \\
        --source E:/fafra_data/upstate/Carapace_ActigraphDataShare.zip \\
        --output-dir E:/fafra_data/upstate/repaired
"""

import argparse
import io
import json
import struct
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

LOG_MEMBER = "log.bin"
RECORD_SEPARATOR = 0x1E
HEADER_SIZE = 8  # separator, type, timestamp (4), payload size (2)
# Record types known to pygt3x (pygt3x.Types); anything else is treated as junk.
KNOWN_RECORD_TYPES = frozenset(range(0, 28))


def record_checksum(record: bytes) -> int:
    """One's complement of the XOR of ``record`` (header + payload)."""
    x = int(np.bitwise_xor.reduce(np.frombuffer(record, dtype=np.uint8))) if record else 0
    return (~x) & 0xFF


def _record_end(log: bytes, offset: int) -> Optional[int]:
    """Offset of the checksum byte of a structurally plausible record at ``offset``."""
    if offset + HEADER_SIZE >= len(log) or log[offset] != RECORD_SEPARATOR:
        return None
    payload_size = struct.unpack_from("<H", log, offset + 6)[0]
    end = offset + HEADER_SIZE + payload_size
    if end >= len(log):
        return None
    return end


@dataclass(frozen=True)
class LogChainScan:
    """Result of walking the record chain from the start of ``log.bin``."""

    total_bytes: int
    n_records: int
    first_break_offset: Optional[int]

    @property
    def intact(self) -> bool:
        return self.first_break_offset is None


def scan_record_chain(log: bytes) -> LogChainScan:
    """Follow record sizes from offset 0; report where the chain breaks, if anywhere.

    Only structure is checked (separator and size). pygt3x itself skips records
    with bad checksums or unknown types, so those don't need repair.
    """
    offset = 0
    n_records = 0
    while offset < len(log):
        end = _record_end(log, offset)
        if end is None:
            return LogChainScan(len(log), n_records, offset)
        n_records += 1
        offset = end + 1
    return LogChainScan(len(log), n_records, None)


@dataclass
class LogRepairReport:
    """What a repair did. ``junk_runs`` holds (offset, length) pairs in the original log."""

    original_bytes: int
    repaired_bytes: int
    n_records: int
    first_break_offset: Optional[int]
    junk_runs: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        return self.first_break_offset is not None

    @property
    def dropped_bytes(self) -> int:
        return self.original_bytes - self.repaired_bytes


def resync_log(log: bytes) -> Tuple[bytes, LogRepairReport]:
    """Rebuild ``log`` from checksum-valid records, dropping bytes that belong to none."""
    scan = scan_record_chain(log)
    out = bytearray()
    junk_runs: List[Tuple[int, int]] = []
    offset = 0
    n_records = 0
    while offset < len(log):
        end = _record_end(log, offset)
        if (
            end is not None
            and log[offset + 1] in KNOWN_RECORD_TYPES
            and record_checksum(log[offset:end]) == log[end]
        ):
            out += log[offset : end + 1]
            n_records += 1
            offset = end + 1
            continue
        if junk_runs and junk_runs[-1][0] + junk_runs[-1][1] == offset:
            junk_runs[-1] = (junk_runs[-1][0], junk_runs[-1][1] + 1)
        else:
            junk_runs.append((offset, 1))
        offset += 1
    report = LogRepairReport(
        original_bytes=len(log),
        repaired_bytes=len(out),
        n_records=n_records,
        first_break_offset=scan.first_break_offset,
        junk_runs=junk_runs,
    )
    return bytes(out), report


def repair_gt3x_bytes(gt3x: bytes) -> Tuple[bytes, LogRepairReport]:
    """Return ``gt3x`` with a repaired ``log.bin``; unchanged if its chain is intact."""
    with zipfile.ZipFile(io.BytesIO(gt3x)) as archive:
        log = archive.read(LOG_MEMBER)
        scan = scan_record_chain(log)
        if scan.intact:
            report = LogRepairReport(
                original_bytes=len(log),
                repaired_bytes=len(log),
                n_records=scan.n_records,
                first_break_offset=None,
            )
            return gt3x, report
        repaired_log, report = resync_log(log)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as out:
            for info in archive.infolist():
                payload = repaired_log if info.filename == LOG_MEMBER else archive.read(info)
                out.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED)
    return buffer.getvalue(), report


def repair_gt3x_file(source: Path, destination: Path) -> LogRepairReport:
    """Write a repaired copy of ``source`` to ``destination`` (always written)."""
    repaired, report = repair_gt3x_bytes(Path(source).read_bytes())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(repaired)
    return report


def iter_gt3x_sources(source: Path) -> Iterator[Tuple[str, bytes]]:
    """Yield (relative path, bytes) for every ``.gt3x`` in a directory tree or a zip."""
    source = Path(source)
    if source.is_dir():
        for path in sorted(source.rglob("*.gt3x")):
            yield path.relative_to(source).as_posix(), path.read_bytes()
        return
    with zipfile.ZipFile(source) as archive:
        for name in sorted(archive.namelist()):
            if name.lower().endswith(".gt3x"):
                yield name, archive.read(name)


def repair_share(source: Union[Path, str], output_dir: Union[Path, str]) -> List[dict]:
    """Write repaired copies of every defective ``.gt3x`` in ``source`` under ``output_dir``.

    Copies keep their relative path. Intact files are not copied. A
    ``repair_report.json`` listing every repaired file is written alongside.
    """
    output_dir = Path(output_dir)
    rows: List[dict] = []
    for relative_path, payload in iter_gt3x_sources(Path(source)):
        repaired, report = repair_gt3x_bytes(payload)
        if not report.repaired:
            continue
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(repaired)
        row = {"file": relative_path, **asdict(report), "dropped_bytes": report.dropped_bytes}
        rows.append(row)
        print(
            f"[gt3x-repair] {relative_path}: dropped {report.dropped_bytes} byte(s) "
            f"in {len(report.junk_runs)} run(s); {report.n_records} records kept",
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "repair_report.json").write_text(json.dumps(rows, indent=2))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="Zip or directory of .gt3x files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = repair_share(args.source, args.output_dir)
    print(f"[gt3x-repair] repaired {len(rows)} file(s) -> {args.output_dir}")


if __name__ == "__main__":
    main()
