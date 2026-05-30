import unittest

import numpy as np

from src.data_io.builders.model_builders.features.record_feature_builder import (
    RecordFeatureBuilder,
)
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.features.feature_fields import FeatureFields
from src.data_types.sample_basis.sample_basis import SampleBasis


def _build_feature_group() -> HDF5Group:
    epoch_group = HDF5Group(
        name=FeatureFields.EPOCH_FEATURES.value,
        attributes={},
        items=[
            HDF5Dataset(name=FeatureFields.FEATURES.value, data=[[[1.0]]], attributes={}),
            HDF5Dataset(name=FeatureFields.BOUT_STARTS.value, data=[10.0], attributes={}),
            HDF5Dataset(name=FeatureFields.BOUT_ENDS.value, data=[12.0], attributes={}),
            HDF5Dataset(
                name=FeatureFields.FEATURE_NAMES.value,
                data=["Day N"],
                attributes={},
            ),
            HDF5Dataset(name=FeatureFields.SAMPLE_STARTS.value, data=[1.0], attributes={}),
            HDF5Dataset(name=FeatureFields.SAMPLE_ENDS.value, data=[1.5], attributes={}),
            HDF5Dataset(name=FeatureFields.UNITS.value, data=[""], attributes={}),
        ],
    )
    stride_group = HDF5Group(
        name=FeatureFields.STRIDE_FEATURES.value,
        attributes={},
        items=[
            HDF5Dataset(name=FeatureFields.FEATURES.value, data=[[[2.0]]], attributes={}),
            HDF5Dataset(name=FeatureFields.BOUT_STARTS.value, data=[10.0], attributes={}),
            HDF5Dataset(name=FeatureFields.BOUT_ENDS.value, data=[12.0], attributes={}),
            HDF5Dataset(
                name=FeatureFields.FEATURE_NAMES.value,
                data=["Day N"],
                attributes={},
            ),
            HDF5Dataset(name=FeatureFields.SAMPLE_STARTS.value, data=[1.0], attributes={}),
            HDF5Dataset(name=FeatureFields.SAMPLE_ENDS.value, data=[1.5], attributes={}),
            HDF5Dataset(name=FeatureFields.UNITS.value, data=[""], attributes={}),
        ],
    )
    return HDF5Group(
        name=FeatureFields.RECORD_FEATURES.value,
        items=[epoch_group, stride_group],
        attributes={
            FeatureFields.FEATURE_IDENTIFIER.value: "feature_1",
            FeatureFields.USER_DATA_IDENTIFIER.value: "user_1",
            FeatureFields.IMU_DATA_IDENTIFIER.value: "imu_1",
            FeatureFields.VERSION.value: "1.0",
        },
    )


class TestRecordFeatureBuilder(unittest.TestCase):
    def test_build_record_features(self):
        builder = RecordFeatureBuilder()
        result = builder.build(_build_feature_group())
        self.assertEqual(result.feature_metadata.feature_identifier.value, "feature_1")
        self.assertEqual(result.epoch_features.sample_basis, SampleBasis.EPOCH)
        self.assertEqual(result.stride_features.sample_basis, SampleBasis.STRIDE)
        np.testing.assert_array_equal(result.epoch_features.features, np.array([[[1.0]]]))
        np.testing.assert_array_equal(result.stride_features.features, np.array([[[2.0]]]))


if __name__ == "__main__":
    unittest.main()
