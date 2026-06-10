import unittest
from unittest.mock import MagicMock

import numpy as np

from src.classification.data.classification_dataset_builder import ClassificationDatasetBuilder
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.mobgap_feature_type import MobgapFeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class TestCatalogHomogeneity(unittest.TestCase):
    def test_mixed_stride_catalogs_raise(self):
        db_manager = MagicMock()
        db_manager.list_feature_ids.return_value = [
            FeatureIdentifier("feature_1"),
            FeatureIdentifier("feature_2"),
        ]

        skdh_record = MagicMock(spec=RecordFeatures)
        skdh_record.feature_metadata = FeatureMetadata(
            feature_identifier=FeatureIdentifier("feature_1"),
            user_identifier=UserIdentifier("user_1"),
            imu_data_identifier=IMUDataIdentifier("imu_1"),
            stride_feature_catalog="skdh",
        )
        mobgap_record = MagicMock(spec=RecordFeatures)
        mobgap_record.feature_metadata = FeatureMetadata(
            feature_identifier=FeatureIdentifier("feature_2"),
            user_identifier=UserIdentifier("user_2"),
            imu_data_identifier=IMUDataIdentifier("imu_2"),
            stride_feature_catalog="mobgap",
        )
        db_manager.load_features.side_effect = [skdh_record, mobgap_record]

        empty_bout = MagicMock()
        empty_bout.feature_names = []
        empty_bout.features = []
        skdh_record.get_features_by_basis.return_value = empty_bout
        mobgap_record.get_features_by_basis.return_value = empty_bout

        builder = ClassificationDatasetBuilder(db_manager=db_manager)
        with self.assertRaises(ValueError):
            builder.build_basis(SampleBasis.STRIDE)

    def test_stride_catalog_filter_skips_non_matching_records(self):
        db_manager = MagicMock()
        db_manager.list_feature_ids.return_value = [FeatureIdentifier("feature_1")]

        record = MagicMock(spec=RecordFeatures)
        record.feature_metadata = FeatureMetadata(
            feature_identifier=FeatureIdentifier("feature_1"),
            user_identifier=UserIdentifier("user_1"),
            imu_data_identifier=IMUDataIdentifier("imu_1"),
            stride_feature_catalog="mobgap",
        )
        db_manager.load_features.return_value = record

        empty_bout = MagicMock()
        empty_bout.feature_names = [MobgapFeatureType.CADENCE_SPM]
        empty_bout.features = np.empty((0, 1, 0))
        record.get_features_by_basis.return_value = empty_bout

        builder = ClassificationDatasetBuilder(db_manager=db_manager)
        dataset = builder.build_basis(SampleBasis.STRIDE, stride_catalog="skdh")
        self.assertTrue(dataset.is_empty)


if __name__ == "__main__":
    unittest.main()
