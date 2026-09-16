"""Unified HTML validation artifact for gait feature integration tests.

Combines feature-quality report content (feature- and participant-level),
physiological plausibility audits, and clinical gait-speed reference checks
into a single self-contained visual report.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.data_model.features.record_features import RecordFeatures
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.services.feature_quality_report import (
    FeatureLevelQualityReport,
    FeatureQualityReportBuilder,
    ParticipantLevelQualityReport,
    feature_level_summary_text,
    participant_level_summary_text,
)
from src.data_visualization.suite.services.visualization_data_service import (
    VisualizationDataService,
)
from test.gait_features.validation.physiological_validators import (
    PhysiologicalAuditResult,
    audit_epoch_physiological_plausibility,
    audit_stride_physiological_plausibility,
)
from test.gait_features.validation.reference_validators import (
    GaitSpeedReferenceAudit,
    audit_gait_speed_matches_reference,
)

DEFAULT_ARTIFACT_PATH = Path("test/gait_features/integration/artifacts/gait_validation_report.html")


@dataclass
class RecordValidationSnapshot:
    participant_id: str
    feature_id: str
    class_label: str
    stride_physiological: PhysiologicalAuditResult
    epoch_physiological: PhysiologicalAuditResult
    gait_speed_reference: Optional[GaitSpeedReferenceAudit] = None


@dataclass
class UnifiedValidationReport:
    generated_at: datetime
    dataset: str
    extraction_profile: str
    extraction_backend: str
    feature_level_epoch: FeatureLevelQualityReport
    feature_level_stride: FeatureLevelQualityReport
    participant_level_epoch: ParticipantLevelQualityReport
    participant_level_stride: ParticipantLevelQualityReport
    record_snapshots: List[RecordValidationSnapshot] = field(default_factory=list)

    @property
    def all_checks_passed(self) -> bool:
        for snapshot in self.record_snapshots:
            if not snapshot.stride_physiological.passed:
                return False
            if not snapshot.epoch_physiological.passed:
                return False
            if snapshot.gait_speed_reference is not None and not snapshot.gait_speed_reference.passed:
                return False
        if self.participant_level_epoch.inconsistent_count_records:
            return False
        if self.participant_level_stride.inconsistent_count_records:
            return False
        return True


def build_unified_validation_report(
    *,
    dataset: str,
    extraction_profile: str,
    extraction_backend: str,
    data_service: VisualizationDataService,
    records: Sequence[Tuple[str, str, RecordFeatures]],
) -> UnifiedValidationReport:
    """Assemble quality reports and per-record validation audits."""
    builder = FeatureQualityReportBuilder(data_service=data_service)
    snapshots: List[RecordValidationSnapshot] = []

    for participant_id, feature_id, record in records:
        class_label = record.feature_metadata.user_identifier.value
        snapshots.append(
            RecordValidationSnapshot(
                participant_id=participant_id,
                feature_id=feature_id,
                class_label=class_label,
                stride_physiological=audit_stride_physiological_plausibility(record),
                epoch_physiological=audit_epoch_physiological_plausibility(record),
                gait_speed_reference=audit_gait_speed_matches_reference(record, participant_id),
            )
        )

    return UnifiedValidationReport(
        generated_at=datetime.now(timezone.utc),
        dataset=dataset,
        extraction_profile=extraction_profile,
        extraction_backend=extraction_backend,
        feature_level_epoch=builder.build_feature_level(SampleBasis.EPOCH),
        feature_level_stride=builder.build_feature_level(SampleBasis.STRIDE),
        participant_level_epoch=builder.build_participant_level(SampleBasis.EPOCH),
        participant_level_stride=builder.build_participant_level(SampleBasis.STRIDE),
        record_snapshots=list(snapshots),
    )


def write_unified_validation_artifact(
    report: UnifiedValidationReport,
    output_path: Path = DEFAULT_ARTIFACT_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_unified_validation_html(report), encoding="utf-8")
    return output_path


def render_unified_validation_html(report: UnifiedValidationReport) -> str:
    status_class = "pass" if report.all_checks_passed else "fail"
    status_label = "ALL CHECKS PASSED" if report.all_checks_passed else "ISSUES DETECTED"

    sections = [
        _render_header(report, status_class, status_label),
        _render_summary_cards(report),
        _render_quality_section(
            "Feature quality — stride basis",
            report.feature_level_stride,
            report.participant_level_stride,
        ),
        _render_quality_section(
            "Feature quality — epoch basis",
            report.feature_level_epoch,
            report.participant_level_epoch,
        ),
        _render_record_validation_section(report),
        _render_footer(report),
    ]
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gait Validation Report — {html.escape(report.dataset)}</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --card: #ffffff;
      --text: #1a1f2e;
      --muted: #5c677d;
      --border: #dde3ea;
      --pass: #0f7b4a;
      --pass-bg: #e6f4ed;
      --fail: #b42318;
      --fail-bg: #fdecec;
      --warn: #b54708;
      --warn-bg: #fff4e5;
      --accent: #175cd3;
      --bar-bg: #e9eef5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 24px 20px 48px; }}
    h1, h2, h3 {{ margin: 0 0 12px; line-height: 1.2; }}
    h1 {{ font-size: 1.75rem; }}
    h2 {{ font-size: 1.25rem; margin-top: 28px; }}
    h3 {{ font-size: 1rem; color: var(--muted); font-weight: 600; }}
    p {{ margin: 0 0 8px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
      margin-bottom: 16px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }}
    .hero {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
    }}
    .badge {{
      display: inline-block;
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 0.8125rem;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .badge.pass {{ background: var(--pass-bg); color: var(--pass); }}
    .badge.fail {{ background: var(--fail-bg); color: var(--fail); }}
    .badge.warn {{ background: var(--warn-bg); color: var(--warn); }}
    .meta {{ color: var(--muted); font-size: 0.875rem; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .stat {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
    }}
    .stat .label {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    .stat .value {{ font-size: 1.35rem; font-weight: 700; margin-top: 4px; }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
    pre.summary {{
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 0.8125rem;
      white-space: pre-wrap;
      margin: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.03em; }}
    .bar-row {{ margin-bottom: 8px; }}
    .bar-label {{
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
      margin-bottom: 3px;
      gap: 8px;
    }}
    .bar-track {{
      height: 10px;
      background: var(--bar-bg);
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, #84caff, var(--accent));
      border-radius: 999px;
    }}
    .bar-fill.low {{ background: linear-gradient(90deg, #fda29b, var(--fail)); }}
    .speed-compare {{
      display: grid;
      gap: 14px;
    }}
    .speed-item {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
      background: #fbfcfe;
    }}
    .speed-scale {{
      position: relative;
      height: 28px;
      background: var(--bar-bg);
      border-radius: 8px;
      margin: 10px 0 6px;
    }}
    .speed-band {{
      position: absolute;
      top: 0;
      bottom: 0;
      background: rgba(15, 123, 74, 0.12);
      border-left: 1px dashed var(--pass);
      border-right: 1px dashed var(--pass);
    }}
    .speed-marker {{
      position: absolute;
      top: 4px;
      width: 10px;
      height: 20px;
      border-radius: 3px;
      transform: translateX(-50%);
    }}
    .speed-marker.extracted {{ background: var(--accent); }}
    .speed-marker.reference {{ background: var(--pass); width: 3px; height: 28px; top: 0; }}
    .legend {{ display: flex; gap: 16px; font-size: 0.75rem; color: var(--muted); flex-wrap: wrap; }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 2px;
      margin-right: 6px;
      vertical-align: middle;
    }}
    .legend .extracted::before {{ background: var(--accent); }}
    .legend .reference::before {{ background: var(--pass); }}
    .legend .band::before {{ background: rgba(15, 123, 74, 0.25); }}
    footer {{ margin-top: 28px; color: var(--muted); font-size: 0.8125rem; }}
  </style>
</head>
<body>
  <div class="wrap">
{body}
  </div>
</body>
</html>
"""


def _render_header(report: UnifiedValidationReport, status_class: str, status_label: str) -> str:
    timestamp = report.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    return f"""
    <section class="card hero">
      <div>
        <h1>Gait Feature Validation Report</h1>
        <p class="meta">Dataset: <strong>{html.escape(report.dataset)}</strong> ·
        Profile: <strong>{html.escape(report.extraction_profile)}</strong> ·
        Backend: <strong>{html.escape(report.extraction_backend)}</strong></p>
        <p class="meta">Generated {html.escape(timestamp)}</p>
      </div>
      <span class="badge {status_class}">{html.escape(status_label)}</span>
    </section>
    """


def _render_summary_cards(report: UnifiedValidationReport) -> str:
    physio_pass = sum(
        1
        for snap in report.record_snapshots
        if snap.stride_physiological.passed and snap.epoch_physiological.passed
    )
    speed_audits = [snap.gait_speed_reference for snap in report.record_snapshots if snap.gait_speed_reference]
    speed_pass = sum(1 for audit in speed_audits if audit.passed)
    cards = [
        ("Feature records", str(len(report.record_snapshots))),
        ("Physiological checks passed", f"{physio_pass}/{len(report.record_snapshots)}"),
        ("Gait speed reference passed", f"{speed_pass}/{len(speed_audits)}" if speed_audits else "n/a"),
        ("Stride usable samples", str(report.feature_level_stride.total_usable_samples)),
        ("Epoch usable samples", str(report.feature_level_epoch.total_usable_samples)),
    ]
    items = "".join(
        f'<div class="stat"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )
    return f'<section class="grid">{items}</section>'


def _render_quality_section(
    title: str,
    feature_report: FeatureLevelQualityReport,
    participant_report: ParticipantLevelQualityReport,
) -> str:
    feature_summary = html.escape(feature_level_summary_text(feature_report))
    participant_summary = html.escape(participant_level_summary_text(participant_report))
    coverage_bars = _render_coverage_bars(feature_report)
    participant_table = _render_participant_table(participant_report)
    return f"""
    <section>
      <h2>{html.escape(title)}</h2>
      <div class="two-col">
        <div class="card">
          <h3>Feature-level summary</h3>
          <pre class="summary">{feature_summary}</pre>
          <h3 style="margin-top:16px">Feature validity (lowest coverage)</h3>
          {coverage_bars}
        </div>
        <div class="card">
          <h3>Participant-level summary</h3>
          <pre class="summary">{participant_summary}</pre>
          <h3 style="margin-top:16px">Per-participant coverage</h3>
          {participant_table}
        </div>
      </div>
    </section>
    """


def _render_coverage_bars(feature_report: FeatureLevelQualityReport, limit: int = 12) -> str:
    if not feature_report.metrics:
        return "<p class='meta'>No feature metrics available.</p>"
    ranked = sorted(feature_report.metrics, key=lambda metric: metric.valid_fraction)
    rows = []
    for metric in ranked[:limit]:
        pct = max(0.0, min(100.0, metric.valid_fraction * 100.0))
        css_class = "low" if pct < 80.0 else ""
        rows.append(
            f"""<div class="bar-row">
              <div class="bar-label">
                <span>{html.escape(metric.feature_type.value)}</span>
                <span>{pct:.1f}% valid</span>
              </div>
              <div class="bar-track"><div class="bar-fill {css_class}" style="width:{pct:.1f}%"></div></div>
            </div>"""
        )
    return "\n".join(rows)


def _render_participant_table(participant_report: ParticipantLevelQualityReport) -> str:
    if not participant_report.participants:
        return "<p class='meta'>No participant metrics available.</p>"
    rows = []
    for metric in participant_report.participants:
        status = "pass" if metric.count_consistent else "fail"
        status_text = "consistent" if metric.count_consistent else "mismatch"
        rows.append(
            f"<tr>"
            f"<td>{html.escape(metric.participant_id)}</td>"
            f"<td>{html.escape(metric.class_label)}</td>"
            f"<td>{metric.usable_sample_count}</td>"
            f"<td>{metric.padded_slot_count}</td>"
            f"<td>{metric.mean_coverage:.2f}</td>"
            f'<td><span class="badge {status}">{status_text}</span></td>'
            f"</tr>"
        )
    return f"""
    <table>
      <thead><tr>
        <th>Participant</th><th>Class</th><th>Usable</th><th>Padded</th><th>Coverage</th><th>Counts</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def _render_record_validation_section(report: UnifiedValidationReport) -> str:
    blocks = []
    for snapshot in report.record_snapshots:
        blocks.append(_render_record_block(snapshot))
    return f"""
    <section>
      <h2>Per-record validation</h2>
      <p class="meta">Physiological plausibility bounds and clinical gait-speed reference checks for each extracted record.</p>
      {''.join(blocks)}
    </section>
    """


def _render_record_block(snapshot: RecordValidationSnapshot) -> str:
    overall_pass = (
        snapshot.stride_physiological.passed
        and snapshot.epoch_physiological.passed
        and (snapshot.gait_speed_reference is None or snapshot.gait_speed_reference.passed)
    )
    badge = "pass" if overall_pass else "fail"
    physio_table = _render_physio_audit_table(snapshot.stride_physiological, snapshot.epoch_physiological)
    speed_block = _render_gait_speed_block(snapshot.gait_speed_reference)
    return f"""
    <div class="card">
      <div class="hero" style="margin-bottom:12px">
        <div>
          <h3 style="margin:0;color:var(--text)">{html.escape(snapshot.participant_id)}</h3>
          <p class="meta">Feature ID: {html.escape(snapshot.feature_id)} · Class: {html.escape(snapshot.class_label)}</p>
        </div>
        <span class="badge {badge}">{'PASS' if overall_pass else 'FAIL'}</span>
      </div>
      <div class="two-col">
        <div>{physio_table}</div>
        <div>{speed_block}</div>
      </div>
    </div>
    """


def _render_physio_audit_table(
    stride: PhysiologicalAuditResult,
    epoch: PhysiologicalAuditResult,
) -> str:
    rows = []
    for audit in (stride, epoch):
        status = "pass" if audit.passed else "fail"
        detail = "No violations"
        if audit.error:
            detail = audit.error
        elif audit.degenerate_row_indices:
            detail = f"{len(audit.degenerate_row_indices)} degenerate row(s)"
        elif audit.violations:
            worst = audit.violations[0]
            detail = (
                f"{worst.feature_type.value}: {worst.out_of_range_count}/"
                f"{worst.finite_count} out of range"
            )
            if len(audit.violations) > 1:
                detail += f" (+{len(audit.violations) - 1} more)"
        rows.append(
            f"<tr><td>{html.escape(audit.basis)}</td>"
            f'<td><span class="badge {status}">{"pass" if audit.passed else "fail"}</span></td>'
            f"<td>{html.escape(detail)}</td></tr>"
        )
    violation_details = _render_violation_details(stride, epoch)
    return f"""
    <h3>Physiological plausibility</h3>
    <table>
      <thead><tr><th>Basis</th><th>Status</th><th>Detail</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    {violation_details}
    """


def _render_violation_details(
    stride: PhysiologicalAuditResult,
    epoch: PhysiologicalAuditResult,
) -> str:
    items: List[str] = []
    for audit in (stride, epoch):
        for violation in audit.violations[:5]:
            examples = ", ".join(f"{value:.4g}" for value in violation.example_values)
            items.append(
                f"<li><strong>{html.escape(violation.feature_type.value)}</strong> "
                f"({html.escape(audit.basis)}): {violation.out_of_range_count}/"
                f"{violation.finite_count} out of range · bounds {html.escape(violation.bounds)}"
                f"{f' · e.g. {html.escape(examples)}' if examples else ''}</li>"
            )
    if not items:
        return ""
    return f"<ul style='margin:10px 0 0;padding-left:18px;font-size:0.8125rem;color:var(--muted)'>{''.join(items)}</ul>"


def _render_gait_speed_block(audit: Optional[GaitSpeedReferenceAudit]) -> str:
    if audit is None:
        return "<h3>Gait speed reference</h3><p class='meta'>No clinical reference configured.</p>"
    status = "pass" if audit.passed else "fail"
    if audit.error and audit.extracted_mps is None:
        return f"""
        <h3>Gait speed reference</h3>
        <span class="badge fail">fail</span>
        <p class="meta" style="margin-top:8px">{html.escape(audit.error)}</p>
        """
    extracted = audit.extracted_mps or 0.0
    reference = audit.reference_mps or 0.0
    tolerance = audit.tolerance_relative or 0.0
    rel_err = audit.relative_error or 0.0
    chart = _render_speed_comparison_chart(extracted, reference, tolerance)
    return f"""
    <h3>Gait speed reference</h3>
    <span class="badge {status}">{'pass' if audit.passed else 'fail'}</span>
    <div class="speed-compare" style="margin-top:10px">
      <div class="speed-item">
        <p><strong>Extracted median:</strong> {extracted:.3f} m/s</p>
        <p><strong>Clinical base(velocity):</strong> {reference:.3f} m/s</p>
        <p><strong>Relative error:</strong> {rel_err:.1%} (tolerance ±{tolerance:.0%})</p>
        {chart}
      </div>
    </div>
    """


def _render_speed_comparison_chart(extracted: float, reference: float, tolerance: float) -> str:
    low = reference * (1.0 - tolerance)
    high = reference * (1.0 + tolerance)
    span_min = min(low, extracted, reference) * 0.9
    span_max = max(high, extracted, reference) * 1.1
    span = max(span_max - span_min, 0.01)

    def pos(value: float) -> float:
        return (value - span_min) / span * 100.0

    band_left = pos(low)
    band_width = pos(high) - band_left
    return f"""
    <div class="speed-scale">
      <div class="speed-band" style="left:{band_left:.1f}%; width:{band_width:.1f}%"></div>
      <div class="speed-marker reference" style="left:{pos(reference):.1f}%"></div>
      <div class="speed-marker extracted" style="left:{pos(extracted):.1f}%"></div>
    </div>
    <div class="legend">
      <span class="reference">Clinical reference</span>
      <span class="extracted">Extracted median</span>
      <span class="band">±{tolerance:.0%} tolerance band</span>
    </div>
    """


def _render_footer(report: UnifiedValidationReport) -> str:
    return f"""
    <footer>
      <p>Unified artifact combining feature-quality reports, physiological plausibility audits,
      and clinical gait-speed reference checks. Regenerate with
      <code>GENERATE_GAIT_VALIDATION_ARTIFACTS=1</code>.</p>
    </footer>
    """
