"""Aggregate sample-level classification rows to participant-level vectors."""

from typing import List, Tuple

import numpy as np

from src.classification.data.classification_dataset import BasisSampleDataset


def aggregate_samples_to_participants(
    dataset: BasisSampleDataset,
    participant_order: List[str],
    aggregation: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Collapse sample rows into one feature vector per participant.

    Returns (X, y, participant_ids) aligned to ``participant_order`` while
    skipping participants with no rows in ``dataset``.
    """
    if aggregation != "mean":
        raise ValueError(
            f"Unsupported aggregation '{aggregation}'. Supported: mean."
        )

    rows: List[np.ndarray] = []
    labels: List[int] = []
    kept_ids: List[str] = []
    label_map = dataset.participant_labels()

    for participant_id in participant_order:
        mask = dataset.groups == participant_id
        if not np.any(mask):
            continue
        block = dataset.X[mask]
        rows.append(np.nanmean(block, axis=0))
        labels.append(int(label_map[participant_id]))
        kept_ids.append(participant_id)

    if not rows:
        return (
            np.empty((0, dataset.n_features)),
            np.empty((0,), dtype=int),
            [],
        )

    return np.vstack(rows), np.asarray(labels, dtype=int), kept_ids
