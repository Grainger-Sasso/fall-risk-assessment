"""Evaluation mode identifiers for classification orchestration."""

LATE_FUSION_SAMPLE = "late_fusion_sample"
EARLY_FUSION_PARTICIPANT = "early_fusion_participant"

EARLY_FUSION_RESULT_KEY = "early_fusion"

SUPPORTED_EVALUATION_MODES = (
    LATE_FUSION_SAMPLE,
    EARLY_FUSION_PARTICIPANT,
)

SUPPORTED_AGGREGATIONS = ("mean",)
