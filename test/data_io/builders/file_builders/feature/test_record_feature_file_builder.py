import unittest

import numpy as np

from src.data_io.builders.file_builders.feature.record_feature_file_builder import (
    RecordFeatureFileBuilder,
)
from src.data_io.model_fields.features.feature_fields import FeatureFields
from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


def _make_bout_features(sample_basis: SampleBasis, value: float) -> BoutFeatures:
    return BoutFeatures(
        sample_basis=sample_basis,
        features=np.array([[[value]]]),
        bout_starts=np.array([10.0]),
        bout_ends=np.array([11.0]),
        feature_names=[FeatureType.DAY_N],
        sample_starts=np.array([1.0]),
        sample_ends=np.array([1.5]),
        units=[""],
    )


class TestRecordFeatureFileBuilder(unittest.TestCase):
    def test_build_hdf5_group(self):
        record = RecordFeatures(
            epoch_features=_make_bout_features(SampleBasis.EPOCH, 1.0),
            stride_features=_make_bout_features(SampleBasis.STRIDE, 2.0),
            feature_metadata=FeatureMetadata(
                feature_identifier=FeatureIdentifier("feature_1"),
                user_identifier=UserIdentifier("user_1"),
                imu_data_identifier=IMUDataIdentifier("imu_1"),
            ),
        )
        group = RecordFeatureFileBuilder().build(record)
        self.assertEqual(group.name, FeatureFields.RECORD_FEATURES.value)
        self.assertEqual(group.attributes[FeatureFields.FEATURE_IDENTIFIER.value], "feature_1")
        epoch = group.get_item_by_name(FeatureFields.EPOCH_FEATURES.value)
        stride = group.get_item_by_name(FeatureFields.STRIDE_FEATURES.value)
        self.assertIsNotNone(epoch.get_item_by_name(FeatureFields.FEATURES.value))
        self.assertIsNotNone(stride.get_item_by_name(FeatureFields.FEATURES.value))
        self.assertEqual(epoch.attributes[FeatureFields.USABLE_SAMPLE_COUNT.value], 1)
        self.assertEqual(stride.attributes[FeatureFields.USABLE_SAMPLE_COUNT.value], 1)
        self.assertEqual(group.attributes[FeatureFields.VERSION.value], "1.2")


if __name__ == "__main__":
    unittest.main()
