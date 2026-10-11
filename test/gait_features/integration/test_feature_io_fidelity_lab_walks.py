import tempfile
import unittest
from pathlib import Path

import h5py

from src.data_io.import_export.exporters.feature.feature_file_exporter import FeatureFileExporter
from src.data_io.import_export.importers.features.feature_importer import FeatureImporter
from src.data_io.model_fields.features.feature_fields import FeatureFields
from src.data_model.features.record_features import RecordFeatures
from src.data_types.sample_basis.sample_basis import SampleBasis
from test.fixtures.assessment_database import participant_fixtures
from test.fixtures.integration_cases import SkdhIntegrationTestCase
from test.fixtures.real_features import extract_participant
from test.gait_features.validation.extraction_validators import assert_io_fidelity


class TestFeatureIoFidelityLabWalks(SkdhIntegrationTestCase):
    """Extracted Lab Walks features survive export/import without value changes."""

    def test_hdf5_roundtrip_preserves_usable_samples(self):
        fixture = participant_fixtures("lab_walks", tier="primary")[0]
        record = extract_participant(fixture).record

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exported_dir = FeatureFileExporter().export_data(root, record)
            reloaded = FeatureImporter().import_data(exported_dir)
            assert_io_fidelity(record, reloaded)
            self._assert_feature_labels_persisted(exported_dir, record)

    def _assert_feature_labels_persisted(
        self, exported_dir: Path, record: RecordFeatures
    ) -> None:
        feature_file = exported_dir / "features.h5"
        self.assertTrue(feature_file.is_file(), "Expected features.h5 in export directory")
        basis_map = {
            FeatureFields.EPOCH_FEATURES.value: SampleBasis.EPOCH,
            FeatureFields.STRIDE_FEATURES.value: SampleBasis.STRIDE,
        }
        with h5py.File(feature_file, "r") as handle:
            root = handle[FeatureFields.RECORD_FEATURES.value]
            for basis_key, sample_basis in basis_map.items():
                group = root[basis_key]
                stored_names = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in group[FeatureFields.FEATURE_NAMES.value][()]
                ]
                tensor = group[FeatureFields.FEATURES.value][()]
                self.assertEqual(len(stored_names), tensor.shape[1])
                expected = record.get_features_by_basis(sample_basis)
                self.assertEqual(
                    [item.value for item in expected.feature_names],
                    stored_names,
                )
                stored_units = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in group[FeatureFields.UNITS.value][()]
                ]
                self.assertEqual(len(stored_units), len(stored_names))
                self.assertEqual(stored_units, expected.units)


if __name__ == "__main__":
    unittest.main()
