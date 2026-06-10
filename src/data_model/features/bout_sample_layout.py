"""Helpers for the (bout, feature, sample-slot) feature tensor layout.

Stride and epoch features are stored as dense ``(J, M, N)`` arrays where ``N``
is padded to the maximum sample count across bouts in a record. Unused slots are
filled with NaN. Analytics must ignore those slots and operate only on *usable*
samples: rows that contain at least one finite feature value.
"""

from __future__ import annotations

import numpy as np


def flatten_bout_features(features: np.ndarray) -> np.ndarray:
    """Return a ``(num_bouts * num_samples, num_features)`` per-sample matrix."""
    array = np.asarray(features, dtype=float)
    if array.size == 0:
        return np.empty((0, 0), dtype=float)
    if array.ndim != 3:
        raise ValueError("Features tensor must be 3D with shape (J, M, N).")
    num_bouts, num_features, num_samples = array.shape
    return np.transpose(array, (0, 2, 1)).reshape(num_bouts * num_samples, num_features)


def usable_sample_mask(per_sample: np.ndarray) -> np.ndarray:
    """True for rows that are not entirely NaN (i.e. not structural padding)."""
    matrix = np.asarray(per_sample, dtype=float)
    if matrix.size == 0:
        return np.empty(0, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError("per_sample matrix must be 2D.")
    return ~np.all(np.isnan(matrix), axis=1)


def count_usable_samples(features: np.ndarray) -> int:
    """Count sample slots with at least one populated feature value."""
    per_sample = flatten_bout_features(features)
    return int(np.sum(usable_sample_mask(per_sample)))


def select_usable_samples(per_sample: np.ndarray) -> np.ndarray:
    """Drop structurally padded rows from a per-sample matrix."""
    matrix = np.asarray(per_sample, dtype=float)
    mask = usable_sample_mask(matrix)
    if mask.size == 0:
        return matrix
    return matrix[mask]
