from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis


@dataclass
class BasisSampleDataset:
    """
    Sample-level feature matrix for a single sampling basis (stride or epoch).

    Each row is one sample (one stride or one epoch). ``y`` carries the binary
    participant label inherited by every sample of that participant, and
    ``groups`` carries the participant identifier for each row so that
    cross-validation can keep a participant's samples together.
    """

    basis: SampleBasis
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: List[FeatureType] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=int)
        self.groups = np.asarray(self.groups, dtype=object)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0]) if self.X.ndim == 2 else 0

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1]) if self.X.ndim == 2 else 0

    @property
    def is_empty(self) -> bool:
        return self.n_samples == 0 or self.n_features == 0

    def participant_ids(self) -> List[str]:
        """Unique participant identifiers in stable first-seen order."""
        return list(dict.fromkeys(self.groups.tolist()))

    def participant_labels(self) -> Dict[str, int]:
        """Map participant id -> binary label (first observed for that group)."""
        labels: Dict[str, int] = {}
        for group, label in zip(self.groups.tolist(), self.y.tolist()):
            if group not in labels:
                labels[group] = int(label)
        return labels


@dataclass
class BasisParticipantDataset:
    """
    Participant-level feature matrix for a single sampling basis.

    Each row is one participant with features aggregated across that
    participant's sample rows (for example by nanmean).
    """

    basis: SampleBasis
    X: np.ndarray
    y: np.ndarray
    participant_ids: List[str]
    feature_names: List[FeatureType] = field(default_factory=list)
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=int)
        self.participant_ids = list(self.participant_ids)

    @property
    def n_participants(self) -> int:
        return int(self.X.shape[0]) if self.X.ndim == 2 else 0

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1]) if self.X.ndim == 2 else 0

    @property
    def is_empty(self) -> bool:
        return self.n_participants == 0 or self.n_features == 0


@dataclass
class EarlyFusionParticipantDataset:
    """
    Participant-level matrix with stride and epoch features concatenated.

    Early fusion happens at the feature level: each participant has one row
    containing aggregated stride features followed by aggregated epoch features.
    """

    X: np.ndarray
    y: np.ndarray
    participant_ids: List[str]
    stride_feature_names: List[FeatureType] = field(default_factory=list)
    epoch_feature_names: List[FeatureType] = field(default_factory=list)
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=int)
        self.participant_ids = list(self.participant_ids)

    @property
    def n_participants(self) -> int:
        return int(self.X.shape[0]) if self.X.ndim == 2 else 0

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1]) if self.X.ndim == 2 else 0

    @property
    def is_empty(self) -> bool:
        return self.n_participants == 0 or self.n_features == 0


@dataclass
class ClassificationDataset:
    """Paired stride and epoch sample datasets for late-fusion classification."""

    stride: BasisSampleDataset
    epoch: BasisSampleDataset

    def get_basis(self, basis: SampleBasis) -> BasisSampleDataset:
        if basis == SampleBasis.STRIDE:
            return self.stride
        if basis == SampleBasis.EPOCH:
            return self.epoch
        raise ValueError(f"Unsupported sample basis: {basis}.")

    def common_participant_ids(self) -> List[str]:
        """Participants present in BOTH bases (required for fusion), stable order."""
        epoch_ids = set(self.epoch.participant_ids())
        return [pid for pid in self.stride.participant_ids() if pid in epoch_ids]

    def participant_labels(self) -> Dict[str, int]:
        """Labels for common participants, sourced from the stride basis."""
        stride_labels = self.stride.participant_labels()
        return {pid: stride_labels[pid] for pid in self.common_participant_ids()}
