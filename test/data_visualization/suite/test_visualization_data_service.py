import unittest
from dataclasses import dataclass
from typing import Optional

import numpy as np
from src.data_model.data.user.clinical.clinical_demographic_data import FallerStatus
from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.services.visualization_data_service import (
    VisualizationDataService,
)


@dataclass
class _FakeIdentifier:
    value: str


@dataclass
class _FakeClinicalData:
    faller_status: FallerStatus


@dataclass
class _FakeUserWithClinical:
    user_identifier: _FakeIdentifier
    clinical_demographic_data: _FakeClinicalData


class _FakeDBManager:
    def __init__(self):
        self.imu_ids = [_FakeIdentifier("imu_1"), _FakeIdentifier("imu_2")]
        self.feature_ids = [_FakeIdentifier("feature_1")]
        self.user_for_imu = {"imu_1": _FakeIdentifier("user_1")}
        self.spec_for_imu = {"imu_1": _FakeIdentifier("spec_1")}
        self.feature_records = {
            "feature_1": RecordFeatures(
                epoch_features=BoutFeatures(
                    sample_basis=SampleBasis.EPOCH,
                    features=np.array([[[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]]]),
                    bout_starts=np.array([0.0]),
                    bout_ends=np.array([3.0]),
                    feature_names=[FeatureType.GAIT_SPEED, FeatureType.CADENCE],
                    sample_starts=np.array([0.0, 1.0, 2.0]),
                    sample_ends=np.array([1.0, 2.0, 3.0]),
                    units=["m/s", "steps/min"],
                ),
                stride_features=BoutFeatures(
                    sample_basis=SampleBasis.STRIDE,
                    features=np.array([[[10.0, 20.0], [40.0, 50.0]]]),
                    bout_starts=np.array([0.0]),
                    bout_ends=np.array([2.0]),
                    feature_names=[FeatureType.GAIT_SPEED, FeatureType.CADENCE],
                    sample_starts=np.array([0.0, 1.0]),
                    sample_ends=np.array([1.0, 2.0]),
                    units=["m/s", "steps/min"],
                ),
                feature_metadata=FeatureMetadata(
                    feature_identifier=_FakeIdentifier("feature_1"),
                    user_identifier=_FakeIdentifier("user_1"),
                    imu_data_identifier=_FakeIdentifier("imu_1"),
                ),
            )
        }

    def list_imu_ids(self):
        return self.imu_ids

    def list_feature_ids(self):
        return self.feature_ids

    def load_imu(self, imu_id):
        return f"imu:{imu_id.value}"

    def get_user_for_imu(self, imu_id) -> Optional[_FakeIdentifier]:
        return self.user_for_imu.get(imu_id.value)

    def load_user(self, user_id):
        return _FakeUserWithClinical(
            user_identifier=user_id,
            clinical_demographic_data=_FakeClinicalData(
                faller_status=FallerStatus.FALLER if user_id.value == "user_1" else FallerStatus.NON_FALLER
            ),
        )

    def get_instrument_spec_for_imu(self, imu_id):
        return self.spec_for_imu.get(imu_id.value)

    def load_instrument_spec(self, spec_id):
        return f"spec:{spec_id.value}"

    def load_features(self, feature_id):
        return self.feature_records[feature_id.value]


class TestVisualizationDataService(unittest.TestCase):
    def setUp(self) -> None:
        self.db_manager = _FakeDBManager()
        self.service = VisualizationDataService(db_manager=self.db_manager)

    def test_list_ids(self):
        self.assertEqual([item.value for item in self.service.list_imu_ids()], ["imu_1", "imu_2"])
        self.assertEqual([item.value for item in self.service.list_feature_ids()], ["feature_1"])

    def test_load_user_for_imu(self):
        user = self.service.load_user_for_imu(_FakeIdentifier("imu_1"))
        self.assertIsNotNone(user)
        self.assertEqual(user.user_identifier.value, "user_1")

        missing = self.service.load_user_for_imu(_FakeIdentifier("imu_2"))
        self.assertIsNone(missing)

    def test_load_instrument_spec_for_imu(self):
        spec_id, spec = self.service.load_instrument_spec_for_imu(_FakeIdentifier("imu_1"))
        self.assertIsNotNone(spec_id)
        self.assertEqual(spec_id.value, "spec_1")
        self.assertEqual(spec, "spec:spec_1")

        none_id, none_spec = self.service.load_instrument_spec_for_imu(_FakeIdentifier("imu_2"))
        self.assertIsNone(none_id)
        self.assertIsNone(none_spec)

    def test_feature_helpers(self):
        types = self.service.list_feature_types_for_basis(
            _FakeIdentifier("feature_1"), SampleBasis.EPOCH
        )
        self.assertEqual(types, [FeatureType.GAIT_SPEED, FeatureType.CADENCE])

        feature_record = self.service.load_features(_FakeIdentifier("feature_1"))
        values = self.service.extract_feature_values(
            feature_record, SampleBasis.EPOCH, FeatureType.GAIT_SPEED
        )
        self.assertEqual(values.tolist(), [1.0, 2.0])

        grouped = self.service.collect_feature_values_by_class(
            basis=SampleBasis.STRIDE,
            feature_type=FeatureType.CADENCE,
        )
        self.assertIn("faller", grouped)
        self.assertEqual(grouped["faller"].tolist(), [40.0, 50.0])


if __name__ == "__main__":
    unittest.main()
