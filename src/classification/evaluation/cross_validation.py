"""Leakage-safe cross-validation helpers for the classification stack.

Outer and inner cross-validation operate at the PARTICIPANT level: each
participant is a single unit, split with stratified k-fold, then expanded to
that participant's samples. This makes leakage impossible by construction
(a participant's samples never straddle train/test).
"""

from typing import Iterator, List, Tuple

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


class InsufficientDataError(ValueError):
    """Raised when the data cannot support the requested cross-validation."""


def resolve_n_splits(labels: np.ndarray, desired: int, minimum: int = 2) -> int:
    """Cap desired folds at the smallest per-class count; require >= minimum."""
    labels = np.asarray(labels).astype(int)
    if labels.size == 0:
        raise InsufficientDataError("No labels available for cross-validation.")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise InsufficientDataError(
            "Both classes are required for stratified cross-validation."
        )
    min_class_count = int(counts.min())
    n_splits = min(desired, min_class_count)
    if n_splits < minimum:
        raise InsufficientDataError(
            f"Need at least {minimum} samples in the smallest class; "
            f"found {min_class_count}."
        )
    return n_splits


def iter_participant_folds(
    participant_labels: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) into the participant arrays."""
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    placeholder = np.zeros((len(participant_labels), 1))
    for train_idx, test_idx in splitter.split(placeholder, participant_labels):
        yield train_idx, test_idx


def iter_single_stratified_folds(
    participant_labels: np.ndarray,
    n_splits: int,
    random_state: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Single (non-repeated) stratified k-fold over participants (inner CV)."""
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    placeholder = np.zeros((len(participant_labels), 1))
    for train_idx, test_idx in splitter.split(placeholder, participant_labels):
        yield train_idx, test_idx


def sample_mask_for_participants(
    sample_groups: np.ndarray, participants: List[str]
) -> np.ndarray:
    """Boolean mask selecting samples whose participant is in ``participants``."""
    participant_set = set(participants)
    return np.array(
        [group in participant_set for group in sample_groups.tolist()],
        dtype=bool,
    )
