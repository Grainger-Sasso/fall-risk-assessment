from typing import List, Optional, Tuple

import numpy as np


def pool_scores_by_participant(
    sample_groups: np.ndarray,
    sample_scores: np.ndarray,
    participant_order: Optional[List[str]] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Aggregate per-sample probabilities into one score per participant by mean.

    Returns the participant order and the pooled score array aligned to it. If
    ``participant_order`` is provided, scores are aligned to that order
    (participants with no samples receive NaN).
    """
    sample_groups = np.asarray(sample_groups, dtype=object)
    sample_scores = np.asarray(sample_scores, dtype=float)

    if participant_order is None:
        participant_order = list(dict.fromkeys(sample_groups.tolist()))

    pooled = np.full(len(participant_order), np.nan, dtype=float)
    for index, participant in enumerate(participant_order):
        mask = sample_groups == participant
        if mask.any():
            pooled[index] = float(np.nanmean(sample_scores[mask]))
    return list(participant_order), pooled
