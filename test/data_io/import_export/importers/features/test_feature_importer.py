import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_io.import_export.importers.features.feature_importer import FeatureImporter
from src.data_io.read_write.writers.hdf5.hdf5_file_writer import HDF5FileWriter
from src.data_io.builders.file_builders.feature.record_feature_file_builder import (
    RecordFeatureFileBuilder,
)
from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


def _make_record_features() -> RecordFeatures:
    epoch = BoutFeatures(
        sample_basis=SampleBasis.EPOCH,
        features=np.array([[[1.0]]]),
        bout_starts=np.array([10.0]),
        bout_ends=np.array([11.0]),
        feature_names=[FeatureType.DAY_N],
        sample_starts=np.array([1.0]),
        sample_ends=np.array([1.5]),
        units=[""],
    )
    stride = BoutFeatures(
        sample_basis=SampleBasis.STRIDE,
        features=np.array([[[2.0]]]),
        bout_starts=np.array([10.0]),
        bout_ends=np.array([11.0]),
        feature_names=[FeatureType.DAY_N],
        sample_starts=np.array([1.0]),
        sample_ends=np.array([1.5]),
        units=[""],
    )
    return RecordFeatures(
        epoch_features=epoch,
        stride_features=stride,
        feature_metadata=FeatureMetadata(
            feature_identifier=FeatureIdentifier("feature_1"),
            user_identifier=UserIdentifier("user_1"),
            imu_data_identifier=IMUDataIdentifier("imu_1"),
        ),
    )


class TestFeatureImporter(unittest.TestCase):
    def test_import_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_path = Path(tmp)
            file_path = temp_path / "features.h5"
            builder = RecordFeatureFileBuilder()
            group = builder.build(_make_record_features())
            writer = HDF5FileWriter()
            success, error = writer.write(file_path, group)
            self.assertTrue(success, error)

            imported = FeatureImporter().import_data(temp_path)
            self.assertEqual(imported.feature_metadata.feature_identifier.value, "feature_1")
            np.testing.assert_array_equal(imported.epoch_features.features, np.array([[[1.0]]]))


if __name__ == "__main__":
    unittest.main()
