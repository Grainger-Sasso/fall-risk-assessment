from dataclasses import dataclass

from src.data_model.assessment_data import AssessmentData
from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import (
    FeatureMetadata,
)
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.identifiers.identifier import Identifier


@dataclass
class RecordFeatures(AssessmentData):
    """
    Represents all extracted features for a given IMU record.

    Features are separated by sampling basis into epoch and stride containers.
    """

    epoch_features: BoutFeatures
    stride_features: BoutFeatures
    feature_metadata: FeatureMetadata

    def __post_init__(self) -> None:
        if self.epoch_features.sample_basis != SampleBasis.EPOCH:
            raise ValueError("epoch_features must use SampleBasis.EPOCH.")
        if self.stride_features.sample_basis != SampleBasis.STRIDE:
            raise ValueError("stride_features must use SampleBasis.STRIDE.")

    def get_data_id(self) -> Identifier:
        return self.feature_metadata.feature_identifier

    def get_associated_data_id(self) -> Identifier:
        return self.feature_metadata.imu_data_identifier

    def get_features_by_basis(self, sample_basis: SampleBasis) -> BoutFeatures:
        if sample_basis == SampleBasis.EPOCH:
            return self.epoch_features
        if sample_basis == SampleBasis.STRIDE:
            return self.stride_features
        raise ValueError(f"Unsupported sample basis: {sample_basis}.")
