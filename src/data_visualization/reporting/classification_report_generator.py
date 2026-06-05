import argparse
from datetime import datetime
from pathlib import Path
from typing import Union

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from src.classification.evaluation.evaluation_artifact import EvaluationArtifact
from src.data_visualization.suite.plots.classification_plot_engine import (
    ClassificationPlotEngine,
)

matplotlib.use("Agg")

FILENAME_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def _timestamped_path(base_path: Path, generated_at: datetime, suffix: str) -> Path:
    timestamp = generated_at.strftime(FILENAME_TIMESTAMP_FORMAT)
    return base_path.with_name(f"{base_path.stem}_{timestamp}{suffix}")


def _render_cover_page(
    pdf: PdfPages, artifact: EvaluationArtifact, generated_at: datetime
) -> None:
    figure = Figure(figsize=(11, 8.5))
    axis = figure.add_subplot(111)
    axis.set_axis_off()
    participant_counts = artifact.participant_counts
    sample_counts = artifact.sample_counts
    lines = [
        "Fall-Risk Classification Report",
        "",
        f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Participants (common): {participant_counts.get('common', 0)} "
        f"(faller={participant_counts.get('faller', 0)}, "
        f"non-faller={participant_counts.get('non_faller', 0)})",
        f"Stride samples: {sample_counts.get('stride', 0)}   "
        f"Epoch samples: {sample_counts.get('epoch', 0)}",
        f"Model families: {len(artifact.model_results)}",
        f"Fusion strategies: {', '.join(artifact.fusion_strategies)}",
        f"CV: {artifact.cv_config.get('resolved_n_splits')} splits x "
        f"{artifact.cv_config.get('n_repeats')} repeats",
    ]
    axis.text(0.5, 0.6, "\n".join(lines), ha="center", va="center", fontsize=14, linespacing=1.8)
    figure.tight_layout()
    pdf.savefig(figure)


def generate_classification_report(
    artifact: Union[EvaluationArtifact, Path, str],
    output_pdf: Path,
) -> Path:
    if not isinstance(artifact, EvaluationArtifact):
        artifact = EvaluationArtifact.from_json(Path(artifact))

    generated_at = datetime.now()
    output_pdf = Path(output_pdf)
    resolved_pdf = _timestamped_path(output_pdf, generated_at, ".pdf")
    resolved_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(resolved_pdf) as pdf:
        _render_cover_page(pdf, artifact, generated_at)

        for metric in ("roc_auc", "pr_auc"):
            figure = Figure(figsize=(11, 8.5))
            ClassificationPlotEngine(figure).render_model_comparison(artifact, metric=metric)
            pdf.savefig(figure)

        ranking_figure = Figure(figsize=(11, 8.5))
        ClassificationPlotEngine(ranking_figure).render_ranking_table(artifact)
        pdf.savefig(ranking_figure)

        for result in artifact.model_results:
            model_name = result.model_name

            fusion_figure = Figure(figsize=(11, 8.5))
            ClassificationPlotEngine(fusion_figure).render_fusion_comparison(
                artifact, model_name=model_name, metric="roc_auc"
            )
            pdf.savefig(fusion_figure)

            roc_figure = Figure(figsize=(11, 8.5))
            ClassificationPlotEngine(roc_figure).render_roc_overlay(artifact, model_name)
            pdf.savefig(roc_figure)

            pr_figure = Figure(figsize=(11, 8.5))
            ClassificationPlotEngine(pr_figure).render_pr_overlay(artifact, model_name)
            pdf.savefig(pr_figure)

            confusion_figure = Figure(figsize=(11, 8.5))
            ClassificationPlotEngine(confusion_figure).render_confusion(
                artifact, model_name=model_name
            )
            pdf.savefig(confusion_figure)

    return resolved_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a PDF report from a classification evaluation artifact."
    )
    parser.add_argument("--artifact-json", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_pdf = generate_classification_report(
        artifact=args.artifact_json,
        output_pdf=args.output_pdf,
    )
    print(f"Classification report PDF: {resolved_pdf}")


if __name__ == "__main__":
    main()
