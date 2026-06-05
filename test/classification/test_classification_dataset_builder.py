import unittest
from dataclasses import dataclass

import numpy as np

from src.classification.data.classification_dataset_builder import (
    ClassificationDatasetBuilder,
)
from src.data_model.data.user.clinical.clinical_demographic_data import FallerStatus
from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis


@dataclass
class _FakeId:
    value: str


@dataclass
class _FakeClinical:
    faller_status: FallerStatus


@dataclass
class _FakeUser:
    clinical_demographic_data: _FakeClinical


def _bout(basis: SampleBasis, features: np.ndarray) -> BoutFeatures:
    num_samples = features.shape[2]
    return BoutFeatures(
        sample_basis=basis,
        features=features,
        bout_starts=np.array([0.0]),
        bout_ends=np.array([float(num_samples)]),
        feature_names=[FeatureType.GAIT_SPEED, FeatureType.CADENCE],
        sample_starts=np.arange(num_samples, dtype=float),
        sample_ends=np.arange(1, num_samples + 1, dtype=float),
        units=["m/s", "steps/min"],
    )


def _record(user_value: str, stride: np.ndarray, epoch: np.ndarray) -> RecordFeatures:
    return RecordFeatures(
        epoch_features=_bout(SampleBasis.EPOCH, epoch),
        stride_features=_bout(SampleBasis.STRIDE, stride),
        feature_metadata=FeatureMetadata(
            feature_identifier=_FakeId(f"feat_{user_value}"),
            user_identifier=_FakeId(user_value),
            imu_data_identifier=_FakeId(f"imu_{user_value}"),
        ),
    )


class _FakeDB:
    def __init__(self):
        self._records = {
            "feat_u1": _record(
                "u1",
                stride=np.array([[[1.0, 2.0], [3.0, 4.0]]]),
                epoch=np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]),
            ),
            "feat_u2": _record(
                "u2",
                stride=np.array([[[5.0, 6.0], [7.0, 8.0]]]),
                epoch=np.array([[[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]),
            ),
            # Unknown faller status -> dropped entirely.
            "feat_u3": _record(
                "u3",
                stride=np.array([[[9.0], [10.0]]]),
                epoch=np.array([[[1.3], [1.4]]]),
            ),
            # All-NaN stride rows -> dropped from stride basis only.
            "feat_u4": _record(
                "u4",
                stride=np.array([[[np.nan, np.nan], [np.nan, np.nan]]]),
                epoch=np.array([[[2.0, 2.1], [2.2, 2.3]]]),
            ),
        }
        self._users = {
            "u1": _FakeUser(_FakeClinical(FallerStatus.FALLER)),
            "u2": _FakeUser(_FakeClinical(FallerStatus.NON_FALLER)),
            "u3": _FakeUser(_FakeClinical(FallerStatus.UNKNOWN)),
            "u4": _FakeUser(_FakeClinical(FallerStatus.NON_FALLER)),
        }

    def list_feature_ids(self):
        return [_FakeId(key) for key in self._records]

    def load_features(self, feature_id):
        return self._records[feature_id.value]

    def load_user(self, user_id):
        return self._users[user_id.value]


class TestClassificationDatasetBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = ClassificationDatasetBuilder(db_manager=_FakeDB())

    def test_stride_basis_drops_unknown_and_all_nan(self):
        stride = self.builder.build_basis(SampleBasis.STRIDE)
        # u1 (2 samples) + u2 (2 samples); u3 unknown dropped; u4 all-NaN dropped.
        self.assertEqual(stride.n_samples, 4)
        self.assertEqual(set(stride.participant_ids()), {"u1", "u2"})
        self.assertEqual(stride.feature_names, [FeatureType.GAIT_SPEED, FeatureType.CADENCE])

    def test_epoch_basis_keeps_u4_but_drops_unknown(self):
        epoch = self.builder.build_basis(SampleBasis.EPOCH)
        # u1(3) + u2(3) + u4(2 samples) = 8; u3 unknown dropped.
        self.assertEqual(epoch.n_samples, 8)
        self.assertEqual(set(epoch.participant_ids()), {"u1", "u2", "u4"})

    def test_common_participants_are_intersection(self):
        dataset = self.builder.build()
        # Stride has u1,u2; epoch has u1,u2,u4 -> common is u1,u2.
        self.assertEqual(set(dataset.common_participant_ids()), {"u1", "u2"})
        labels = dataset.participant_labels()
        self.assertEqual(labels["u1"], 1)
        self.assertEqual(labels["u2"], 0)


if __name__ == "__main__":
    unittest.main()
