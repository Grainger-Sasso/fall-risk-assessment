"""Re-root the record paths stored in SQLite metadata indexes.

An index stores each record's absolute directory, so a copied assessment
database keeps reading from wherever it was indexed. This rewrites paths that
start with ``--from`` so they start with ``--to`` instead. Paths under any
other root are left untouched and reported.

The rewrite is all-or-nothing per index. Nothing is written unless every
rewritten path exists (override with ``--allow-missing``).

    python -m src.database_manager.relocate_index \\
        --from E:/fafra_data/assessment_database \\
        --to C:/fafra_data/assessment_database \\
        C:/fafra_data/assessment_database/sql_db_indexes   # an index.db or a folder of them

Add ``--dry-run`` to report without writing.
"""

import argparse
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, List, Optional, Sequence

INDEX_FILE = "index.db"


def _normalise(path: str) -> str:
    """Forward slashes, no trailing slash, lower-case drive letter."""
    text = str(path).replace("\\", "/").rstrip("/")
    return re.sub(r"^([A-Za-z]):", lambda m: m.group(1).lower() + ":", text)


def _is_windows_style(root: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", root))


def _root_label(path: str, depth: int = 2) -> str:
    parts = _normalise(path).split("/")
    return "/".join(parts[: depth + 1])


def rebase_path(stored: str, old_root: str, new_root: str) -> Optional[str]:
    """``stored`` moved from under ``old_root`` to under ``new_root``; None if not under it."""
    old = _normalise(old_root)
    candidate = _normalise(stored)
    if _is_windows_style(old_root):
        matches = candidate.lower() == old.lower() or candidate.lower().startswith(old.lower() + "/")
    else:
        matches = candidate == old or candidate.startswith(old + "/")
    if not matches:
        return None
    relative = candidate[len(old):].lstrip("/")
    parts = [p for p in relative.split("/") if p]
    if _is_windows_style(new_root):
        return str(PureWindowsPath(new_root).joinpath(*parts))
    return "/".join([new_root.rstrip("/")] + parts)


@dataclass
class RelocationReport:
    index: Path
    rewritten: int = 0
    already_under_new_root: int = 0
    other_roots: Dict[str, int] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    written: bool = False


def relocate_index(
    index: Path,
    old_root: str,
    new_root: str,
    dry_run: bool = False,
    allow_missing: bool = False,
) -> RelocationReport:
    """Rewrite record paths under ``old_root`` in one index to sit under ``new_root``."""
    index = Path(index)
    if not index.is_file():
        raise FileNotFoundError(index)
    report = RelocationReport(index=index)
    updates: List[tuple] = []
    other_roots: Counter = Counter()
    # Read-only: indexes that need no rewrite are never modified.
    reader = sqlite3.connect(f"{index.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = reader.execute("SELECT id_type, id, path FROM records;").fetchall()
    finally:
        reader.close()
    for id_type, identifier, stored in rows:
        new_path = rebase_path(stored, old_root, new_root)
        if new_path is None:
            if rebase_path(stored, new_root, new_root) is not None:
                report.already_under_new_root += 1
            else:
                other_roots[_root_label(stored)] += 1
            continue
        if not Path(new_path).exists():
            report.missing.append(new_path)
        updates.append((new_path, id_type, identifier))
    report.rewritten = len(updates)
    report.other_roots = dict(other_roots)
    if dry_run or not updates:
        return report
    if report.missing and not allow_missing:
        raise FileNotFoundError(
            f"{len(report.missing)} relocated path(s) in {index} do not exist "
            f"(first: {report.missing[0]}); nothing was written."
        )
    timestamp = datetime.now(timezone.utc).isoformat()
    writer = sqlite3.connect(index)
    try:
        with writer:  # one transaction: all rows or none
            writer.executemany(
                "UPDATE records SET path=?, updated_at=? WHERE id_type=? AND id=?;",
                [(path, timestamp, id_type, identifier) for path, id_type, identifier in updates],
            )
    finally:
        writer.close()
    report.written = True
    return report


def find_indexes(targets: Iterable[Path]) -> List[Path]:
    """Expand folders to the ``index.db`` files beneath them (skipping macOS ``._`` files)."""
    found: List[Path] = []
    for target in targets:
        target = Path(target)
        if target.is_dir():
            found.extend(sorted(p for p in target.rglob(INDEX_FILE) if p.is_file()))
        else:
            found.append(target)
    return found


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Re-root record paths stored in SQLite metadata indexes.")
    parser.add_argument("targets", nargs="+", type=Path, help="index.db files and/or folders containing them")
    parser.add_argument("--from", dest="old_root", required=True)
    parser.add_argument("--to", dest="new_root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Rewrite even if some relocated paths do not exist.")
    args = parser.parse_args(argv)
    failed = False
    for index in find_indexes(args.targets):
        try:
            report = relocate_index(index, args.old_root, args.new_root, args.dry_run, args.allow_missing)
        except (FileNotFoundError, sqlite3.DatabaseError) as exc:
            print(f"[relocate] SKIPPED {index}: {exc}")
            failed = True
            continue
        action = "would rewrite" if args.dry_run else "rewrote"
        print(
            f"[relocate] {index}: {action} {report.rewritten}, already at new root "
            f"{report.already_under_new_root}, missing {len(report.missing)}, "
            f"left under other roots {report.other_roots or '{}'}"
        )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
