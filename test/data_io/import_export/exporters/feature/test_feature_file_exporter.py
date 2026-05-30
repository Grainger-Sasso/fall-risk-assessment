import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_io.import_export.exporters.feature.feature_file_exporter import (
    FeatureFileExporter,
)
from src.data_io.import_export.importers.features.feature_importer import FeatureImporter
from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


def _record_features() -> RecordFeatures:
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


class TestFeatureFileExporter(unittest.TestCase):
    def test_export_and_import_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = _record_features()
            output_subdir = FeatureFileExporter().export_data(root, data)
            self.assertTrue(output_subdir.exists())
            self.assertEqual(output_subdir.name, "features_feature_1")
            imported = FeatureImporter().import_data(output_subdir)
            self.assertEqual(imported.feature_metadata.feature_identifier.value, "feature_1")
            np.testing.assert_array_equal(imported.stride_features.features, np.array([[[2.0]]]))


if __name__ == "__main__":
    unittest.main()
