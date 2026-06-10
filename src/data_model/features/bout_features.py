from dataclasses import dataclass
from typing import List, Union

import numpy as np

from src.data_types.feature.feature_type import FeatureType
from src.data_types.feature.mobgap_feature_type import MobgapFeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis


@dataclass
class BoutFeatures:
    """
    Container for all feature values for a single sampling basis.

    The tensor shape is:
        features[bout_index][feature_index][sample_index]
    where J = number of bouts, M = number of feature types, and N = samples
    for this basis (epoch or stride).
    """

    sample_basis: SampleBasis
    features: np.ndarray
    bout_starts: np.ndarray
    bout_ends: np.ndarray
    feature_names: List[Union[FeatureType, MobgapFeatureType]]
    sample_starts: np.ndarray
    sample_ends: np.ndarray
    units: List[str]

    def __post_init__(self) -> None:
        # Normalize numeric collections to ndarrays for consistent storage.
        self.features = np.asarray(self.features, dtype=float)
        self.bout_starts = np.asarray(self.bout_starts, dtype=float)
        self.bout_ends = np.asarray(self.bout_ends, dtype=float)
        self.sample_starts = np.asarray(self.sample_starts, dtype=float)
        self.sample_ends = np.asarray(self.sample_ends, dtype=float)

        self._validate_dimensions()
        self._validate_times()

    @property
    def num_bouts(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_feature_types(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[2])

    def _validate_dimensions(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("Features tensor must be a 3D ndarray with shape (J, M, N).")

        num_bouts, num_feature_types, num_samples = self.features.shape

        if self.bout_starts.ndim != 1 or self.bout_ends.ndim != 1:
            raise ValueError("Bout time axes must be 1D ndarrays.")
        if self.sample_starts.ndim != 1 or self.sample_ends.ndim != 1:
            raise ValueError("Sample time axes must be 1D ndarrays.")

        if len(self.bout_starts) != num_bouts or len(self.bout_ends) != num_bouts:
            raise ValueError(
                "Bout axis mismatch: expected len(bout_starts) and len(bout_ends) to "
                "match number of bout feature matrices."
            )

        if len(self.sample_starts) != len(self.sample_ends):
            raise ValueError(
                "Sample axis mismatch: sample_starts and sample_ends must have equal length."
            )

        num_feature_types = len(self.feature_names)
        if len(self.units) != num_feature_types:
            raise ValueError(
                "Feature axis mismatch: units must align one-to-one with feature_names."
            )

        if len(self.sample_starts) != num_samples:
            raise ValueError(
                "Sample axis mismatch: sample_starts must align with the tensor sample axis."
            )

    def _validate_times(self) -> None:
        for bout_start, bout_end in zip(self.bout_starts, self.bout_ends):
            if bout_start >= bout_end:
                raise ValueError("Invalid bout timestamps: bout_start must be < bout_end.")

        for sample_start, sample_end in zip(self.sample_starts, self.sample_ends):
            if sample_start >= sample_end:
                raise ValueError(
                    "Invalid sample timestamps: sample_start must be < sample_end."
                )
