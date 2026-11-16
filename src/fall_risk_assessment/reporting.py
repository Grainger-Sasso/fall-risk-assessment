"""Render a FallRiskAssessment as a self-contained HTML report."""

from pathlib import Path
from typing import List, Optional

from src.fall_risk_assessment.inference import BasisQuality, FallRiskAssessment

_CATEGORY_COLORS = {
    "low": "#2e7d32",
    "moderate": "#f9a825",
    "high": "#c62828",
}

_STYLE = """
  body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
         margin: 2rem auto; max-width: 900px; color: #222; }
  h1 { font-size: 1.6rem; border-bottom: 2px solid #444; padding-bottom: .4rem; }
  h2 { font-size: 1.15rem; margin-top: 2rem; }
  .meta { color: #666; font-size: .85rem; }
  .risk-banner { border-radius: 10px; padding: 1.2rem 1.5rem; color: #fff;
                 display: flex; justify-content: space-between; align-items: center;
                 margin: 1.2rem 0; }
  .risk-banner .category { font-size: 1.5rem; font-weight: 700; text-transform: uppercase; }
  .risk-banner .prob { font-size: 2.2rem; font-weight: 700; }
  .prob-track { background: #e0e0e0; border-radius: 6px; height: 14px; margin: .6rem 0 1.4rem; position: relative; }
  .prob-fill { height: 14px; border-radius: 6px; }
  .prob-tick { position: absolute; top: -4px; width: 2px; height: 22px; background: #222; }
  table { border-collapse: collapse; width: 100%; font-size: .85rem; }
  th, td { text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #ddd; }
  th { background: #f5f5f5; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .pill { display: inline-block; border-radius: 10px; padding: .05rem .55rem;
          font-size: .75rem; color: #fff; }
  .quality-grid { display: flex; gap: 1rem; }
  .quality-card { flex: 1; border: 1px solid #ddd; border-radius: 8px; padding: .8rem 1rem; }
  .quality-card h3 { margin: 0 0 .5rem; font-size: .95rem; }
  .warn { color: #c62828; }
  .footnote { color: #888; font-size: .75rem; margin-top: 2rem; }
"""


def _percentile_pill(percentile: Optional[float]) -> str:
    if percentile is None:
        return "<span class='pill' style='background:#9e9e9e'>n/a</span>"
    if percentile <= 25 or percentile >= 75:
        color = "#c62828" if percentile >= 75 else "#1565c0"
    else:
        color = "#2e7d32"
    return f"<span class='pill' style='background:{color}'>{percentile:.0f}%</span>"


def _quality_card(title: str, quality: Optional[BasisQuality]) -> str:
    if quality is None:
        return (
            f"<div class='quality-card'><h3>{title}</h3>"
            "<p class='warn'>No data</p></div>"
        )
    missing_html = ""
    if quality.missing_features:
        shown = ", ".join(quality.missing_features[:6])
        suffix = " …" if len(quality.missing_features) > 6 else ""
        missing_html = (
            f"<p class='warn'>Missing features: {shown}{suffix}</p>"
        )
    return (
        f"<div class='quality-card'><h3>{title}</h3>"
        f"<p>Usable samples: <strong>{quality.n_usable_samples}</strong></p>"
        f"<p>Features observed: {quality.n_features_observed}"
        f"/{quality.n_features_expected}</p>"
        f"<p>NaN fraction: {quality.nan_fraction:.1%}</p>"
        f"{missing_html}</div>"
    )


def render_assessment_html(assessment: FallRiskAssessment, output_path: Path) -> Path:
    category_color = _CATEGORY_COLORS.get(assessment.risk_category, "#546e7a")
    probability_pct = assessment.risk_probability * 100.0
    threshold_pct = assessment.decision_threshold * 100.0

    rows: List[str] = []
    for name, value in assessment.feature_values.items():
        percentile = assessment.feature_percentiles.get(name)
        value_text = f"{value:.4g}" if value is not None else "—"
        rows.append(
            f"<tr><td>{name}</td><td class='num'>{value_text}</td>"
            f"<td>{_percentile_pill(percentile)}</td></tr>"
        )

    known_html = ""
    if assessment.known_faller_status is not None:
        known_html = (
            "<p class='meta'>Known clinical faller status (reference): "
            f"<strong>{assessment.known_faller_status}</strong></p>"
        )

    decision_text = (
        "FALLER (above decision threshold)"
        if assessment.predicted_faller
        else "NON-FALLER (below decision threshold)"
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Fall Risk Assessment — {assessment.user_id}</title>
<style>{_STYLE}</style></head>
<body>
<h1>Fall Risk Assessment</h1>
<p class="meta">User <strong>{assessment.user_id}</strong> ·
IMU record {assessment.imu_record_id} ·
Assessed {assessment.assessed_at_utc} ·
Model <strong>{assessment.model_name}</strong></p>

<div class="risk-banner" style="background:{category_color}">
  <div>
    <div class="category">{assessment.risk_category} risk</div>
    <div>Binary classification: {decision_text}</div>
  </div>
  <div class="prob">{probability_pct:.0f}%</div>
</div>

<div class="prob-track">
  <div class="prob-fill" style="width:{probability_pct:.1f}%;background:{category_color}"></div>
  <div class="prob-tick" style="left:{threshold_pct:.1f}%" title="decision threshold"></div>
</div>
<p class="meta">Calibrated faller probability {probability_pct:.1f}% ·
decision threshold {threshold_pct:.1f}% (marker)</p>
{known_html}

<h2>Recording quality</h2>
<div class="quality-grid">
  {_quality_card('Stride basis', assessment.stride_quality)}
  {_quality_card('Epoch basis', assessment.epoch_quality)}
</div>

<h2>Gait features vs training cohort</h2>
<p class="meta">Percentile shows where this user falls within the training
cohort distribution for each feature (nanmean across usable samples).</p>
<table>
<thead><tr><th>Feature</th><th style="text-align:right">Value</th><th>Cohort percentile</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>

<p class="footnote">Generated by the fall-risk-assessment inference pipeline.
This report is a research output, not a medical diagnosis.</p>
</body></html>
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
