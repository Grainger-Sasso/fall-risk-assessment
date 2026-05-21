"""
Generate an executive summary PDF from gait visualization outputs.

Summary content includes:
- Dataset/class snapshot
- Missingness highlights
- Top separating aggregate features (AUC)
- Correlation-group redundancy highlights
- Recommended shortlist of robust candidate features
"""

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.data_visualization.gait_features.dataset_report import (
    _compute_class_representation,
    _compute_feature_missingness,
    _get_database_manager,
)


def _add_date_to_output_path(output_path: Path) -> Path:
    date_str = date.today().isoformat()
    return output_path.parent / f"{output_path.stem}_{date_str}{output_path.suffix}"


def _load_auc_rows(auc_csv_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with auc_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            auc_text = (row.get("auc_roc") or "").strip()
            dev_text = (row.get("auc_deviation_from_0_5") or "").strip()
            auc = None if not auc_text else float(auc_text)
            deviation = None if not dev_text else float(dev_text)
            rows.append(
                {
                    "feature_type": (row.get("feature_type") or "").strip(),
                    "stat_type": (row.get("stat_type") or "").strip(),
                    "n_faller": int((row.get("n_faller") or "0").strip()),
                    "n_non_faller": int((row.get("n_non_faller") or "0").strip()),
                    "auc_roc": auc,
                    "auc_deviation_from_0_5": deviation,
                }
            )
    return rows


def _load_correlation_groups(groups_output_path: Path) -> List[List[str]]:
    groups: List[List[str]] = []
    current_group: List[str] = []
    for line in groups_output_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("Group "):
            if current_group:
                groups.append(current_group)
                current_group = []
            continue
        if stripped.startswith("- "):
            current_group.append(stripped[2:].strip())
    if current_group:
        groups.append(current_group)
    return groups


def _format_top_missingness(missingness: Dict[str, float], n_top: int) -> List[Tuple[str, float]]:
    ordered = sorted(missingness.items(), key=lambda item: item[1], reverse=True)
    return ordered[:n_top]


def _build_feature_to_group_map(groups: List[List[str]]) -> Dict[str, int]:
    feature_to_group: Dict[str, int] = {}
    for idx, group in enumerate(groups, start=1):
        for feature_name in group:
            feature_to_group[feature_name] = idx
    return feature_to_group


def _build_shortlist(
    auc_rows: List[Dict[str, object]],
    missingness_by_feature: Dict[str, float],
    feature_to_group: Dict[str, int],
    max_missingness_pct: float = 30.0,
    top_k: int = 12,
) -> List[Dict[str, object]]:
    sortable = [
        row
        for row in auc_rows
        if row["auc_deviation_from_0_5"] is not None
    ]
    sortable.sort(key=lambda row: float(row["auc_deviation_from_0_5"]), reverse=True)

    selected: List[Dict[str, object]] = []
    used_groups: set[int] = set()
    for row in sortable:
        feature_type = str(row["feature_type"])
        group_id = feature_to_group.get(feature_type)
        missing_pct = missingness_by_feature.get(feature_type, 100.0)
        if missing_pct > max_missingness_pct:
            continue
        if group_id is not None and group_id in used_groups:
            continue
        selected.append(
            {
                **row,
                "missingness_pct": missing_pct,
                "correlation_group": group_id,
            }
        )
        if group_id is not None:
            used_groups.add(group_id)
        if len(selected) >= top_k:
            break
    return selected


def _wrap_text(value: str, max_len: int = 38) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _add_text_page(
    pdf: PdfPages,
    title: str,
    lines: List[str],
    subtitle: Optional[str] = None,
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.9])
    ax.axis("off")
    ax.text(0.0, 0.98, title, fontsize=18, weight="bold", va="top")
    if subtitle:
        ax.text(0.0, 0.93, subtitle, fontsize=11, color="#444444", va="top")
    y = 0.88
    for line in lines:
        ax.text(0.0, y, line, fontsize=11, va="top")
        y -= 0.04
        if y < 0.05:
            break
    plt.tight_layout()
    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _add_table_page(
    pdf: PdfPages,
    title: str,
    headers: List[str],
    rows: List[List[str]],
    subtitle: Optional[str] = None,
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax.axis("off")
    ax.text(0.0, 0.98, title, fontsize=16, weight="bold", va="top")
    if subtitle:
        ax.text(0.0, 0.93, subtitle, fontsize=10, color="#444444", va="top")
    table = ax.table(
        cellText=rows if rows else [["No data available"] + [""] * (len(headers) - 1)],
        colLabels=headers,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.02, 1.0, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if row_idx == 0:
            cell.set_facecolor("#E9EEF6")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row_idx % 2 else "#F8F9FB")
    plt.tight_layout()
    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _chunk_items(items: List[str], chunk_size: int) -> List[List[str]]:
    return [items[idx : idx + chunk_size] for idx in range(0, len(items), chunk_size)]


def _format_group_feature_lines(
    features: List[str], max_line_chars: int = 100
) -> List[str]:
    """
    Render full feature list as wrapped lines without truncation.
    """
    if not features:
        return ["  - (no features)"]

    lines: List[str] = []
    current_line = "  - "
    for feature in features:
        token = feature if current_line == "  - " else f", {feature}"
        if len(current_line) + len(token) > max_line_chars and current_line != "  - ":
            lines.append(current_line)
            current_line = f"    {feature}"
        else:
            current_line += token
    lines.append(current_line)
    return lines


def generate_executive_summary(
    base_path: Path,
    auc_csv_path: Path,
    correlation_groups_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    db_manager = _get_database_manager(base_path)
    (
        n_fallers,
        n_non_fallers,
        n_unknown,
        n_epochs_fallers,
        n_epochs_non_fallers,
        n_epochs_unknown,
    ) = _compute_class_representation(db_manager)
    missingness = _compute_feature_missingness(db_manager)
    missingness_by_feature = {feature.value: pct for feature, pct in missingness.items()}

    auc_rows = _load_auc_rows(auc_csv_path)
    auc_ranked = [
        row for row in auc_rows if row["auc_deviation_from_0_5"] is not None
    ]
    auc_ranked.sort(key=lambda row: float(row["auc_deviation_from_0_5"]), reverse=True)
    unique_feature_types = sorted({str(row["feature_type"]) for row in auc_rows})
    unique_stat_types = sorted({str(row["stat_type"]) for row in auc_rows})

    groups = _load_correlation_groups(correlation_groups_path)
    feature_to_group = _build_feature_to_group_map(groups)
    shortlist = _build_shortlist(
        auc_rows=auc_ranked,
        missingness_by_feature=missingness_by_feature,
        feature_to_group=feature_to_group,
    )

    if output_path is None:
        output_path = base_path / "gait_feature_executive_summary.pdf"
    output_path = _add_date_to_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overview_lines: List[str] = [
        f"Generated: {date.today().isoformat()}",
        "",
        "Dataset snapshot",
        f"- Participants: fallers={n_fallers}, non-fallers={n_non_fallers}, unknown={n_unknown}",
        f"- Epoch counts: fallers={n_epochs_fallers}, non-fallers={n_epochs_non_fallers}, unknown={n_epochs_unknown}",
        "",
        "Interpretation guidance",
        "- Prioritize features with high AUC deviation, low missingness, and low redundancy.",
        "- Correlated groups indicate duplicate signal; prefer one robust representative.",
        "- corr_group in the shortlist table is the ID of a high-correlation feature cluster; "
        "features sharing the same corr_group are strongly correlated and potentially redundant.",
        "- corr_group = none means the feature was not part of any high-correlation cluster.",
        "",
        "Redundancy overview",
        f"- Highly correlated groups found: {len(groups)}",
        "- In plain terms: if several features move together almost all the time, "
        "they are telling us nearly the same thing.",
        "- Keeping all of them can over-complicate analysis, so we usually keep one "
        "strong representative from each group.",
    ]
    if groups:
        largest_groups = sorted(groups, key=lambda group: len(group), reverse=True)[:5]
        for idx, group in enumerate(largest_groups, start=1):
            overview_lines.append(
                f"- Group {idx}: size={len(group)}, sample={', '.join(group[:5])}"
            )

    missing_rows = [
        [_wrap_text(name, 46), f"{pct:.2f}%"]
        for name, pct in _format_top_missingness(missingness_by_feature, n_top=20)
    ]
    top_auc_rows = [
        [
            _wrap_text(str(row["feature_type"]), 30),
            str(row["stat_type"]),
            f"{float(row['auc_roc']):.3f}",
            f"{float(row['auc_deviation_from_0_5']):.3f}",
            f"{row['n_faller']}",
            f"{row['n_non_faller']}",
        ]
        for row in auc_ranked[:20]
    ]
    shortlist_rows = [
        [
            _wrap_text(str(row["feature_type"]), 30),
            str(row["stat_type"]),
            f"{float(row['auc_roc']):.3f}",
            f"{float(row['auc_deviation_from_0_5']):.3f}",
            f"{float(row['missingness_pct']):.2f}%",
            (
                "none"
                if row["correlation_group"] is None
                else f"{row['correlation_group']}"
            ),
        ]
        for row in shortlist
    ]

    with PdfPages(str(output_path)) as pdf:
        dataset_label = base_path.name
        feature_set_summary = (
            f"{len(unique_feature_types)} feature types x "
            f"{len(unique_stat_types)} stat types "
            f"({len(auc_rows)} feature-stat records)"
        )
        _add_text_page(
            pdf=pdf,
            title="Gait Feature Executive Summary",
            subtitle=(
                f"Dataset: {dataset_label} | "
                f"Feature set analyzed: {feature_set_summary}"
            ),
            lines=overview_lines,
        )
        _add_table_page(
            pdf=pdf,
            title="Top Missingness Features",
            subtitle="Highest missingness can limit downstream model reliability.",
            headers=["Feature", "Missingness"],
            rows=missing_rows,
        )
        _add_table_page(
            pdf=pdf,
            title="Top Separating Features",
            subtitle="Ranked by AUC deviation from 0.5 (higher indicates stronger separation).",
            headers=[
                "Feature",
                "Stat",
                "AUC",
                "AUC Dev",
                "n_faller",
                "n_non_faller",
            ],
            rows=top_auc_rows,
        )
        if groups:
            groups_per_page = 6
            for group_chunk in _chunk_items([str(i) for i in range(len(groups))], groups_per_page):
                redundancy_lines: List[str] = []
                for group_idx_text in group_chunk:
                    group_idx = int(group_idx_text)
                    group = groups[group_idx]
                    redundancy_lines.append(
                        f"Group {group_idx + 1} (n={len(group)} features)"
                    )
                    redundancy_lines.extend(_format_group_feature_lines(group))
                    redundancy_lines.append("")
                _add_text_page(
                    pdf=pdf,
                    title="Redundancy Groups (Full Membership)",
                    subtitle=(
                        "Complete feature lists for each high-correlation group. "
                        "Features in the same group are likely redundant."
                    ),
                    lines=redundancy_lines,
                )
        _add_table_page(
            pdf=pdf,
            title="Recommended Shortlist",
            subtitle="Filtered for low missingness and reduced intra-group redundancy.",
            headers=[
                "Feature",
                "Stat",
                "AUC",
                "AUC Dev",
                "Missingness",
                "Corr Group",
            ],
            rows=shortlist_rows,
        )
    print(f"Saved executive summary to {output_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate executive summary report for gait visualization outputs."
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Base path to converted data (e.g. .../ltmm_2026_02_21).",
    )
    parser.add_argument(
        "--auc-csv",
        type=Path,
        required=True,
        help="Path to aggregate feature AUC summary CSV.",
    )
    parser.add_argument(
        "--correlation-groups",
        type=Path,
        required=True,
        help="Path to feature correlation groups text output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PDF path "
            "(default: <base-path>/gait_feature_executive_summary_<date>.pdf)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_executive_summary(
        base_path=args.base_path,
        auc_csv_path=args.auc_csv,
        correlation_groups_path=args.correlation_groups,
        output_path=args.output,
    )
