from typing import Any, Dict, List

import numpy as np

from src.data_io.builders.file_builders.file_builder import FileBuilder
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.features.feature_fields import FeatureFields
from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.sample_basis.sample_basis import SampleBasis


class RecordFeatureFileBuilder(FileBuilder):
    """Builds HDF5 file format from RecordFeatures model objects."""

    version: str = "1.2"

    def build(self, data: RecordFeatures) -> HDF5Group:
        if not isinstance(data, RecordFeatures):
            raise ValueError("Data must be RecordFeatures.")
        return HDF5Group(
            name=FeatureFields.RECORD_FEATURES.value,
            items=[
                self._build_basis_group(data.epoch_features),
                self._build_basis_group(data.stride_features),
            ],
            attributes=self._build_root_attributes(data.feature_metadata),
        )

    def _build_basis_group(self, basis_features: BoutFeatures) -> HDF5Group:
        if basis_features.sample_basis == SampleBasis.EPOCH:
            group_name = FeatureFields.EPOCH_FEATURES.value
        elif basis_features.sample_basis == SampleBasis.STRIDE:
            group_name = FeatureFields.STRIDE_FEATURES.value
        else:
            raise ValueError(f"Unsupported sample basis: {basis_features.sample_basis}")

        return HDF5Group(
            name=group_name,
            items=self._build_basis_datasets(basis_features),
            attributes={
                FeatureFields.USABLE_SAMPLE_COUNT.value: int(
                    basis_features.usable_sample_count
                ),
            },
        )

    def _build_basis_datasets(self, basis_features: BoutFeatures) -> List[HDF5Dataset]:
        return [
            HDF5Dataset(
                name=FeatureFields.FEATURES.value,
                data=self._serialize_ndarray(basis_features.features),
                attributes={},
            ),
            HDF5Dataset(
                name=FeatureFields.BOUT_STARTS.value,
                data=self._serialize_ndarray(basis_features.bout_starts),
                attributes={},
            ),
            HDF5Dataset(
                name=FeatureFields.BOUT_ENDS.value,
                data=self._serialize_ndarray(basis_features.bout_ends),
                attributes={},
            ),
            HDF5Dataset(
                name=FeatureFields.FEATURE_NAMES.value,
                data=[feature_name.value for feature_name in basis_features.feature_names],
                attributes={},
            ),
            HDF5Dataset(
                name=FeatureFields.SAMPLE_STARTS.value,
                data=self._serialize_ndarray(basis_features.sample_starts),
                attributes={},
            ),
            HDF5Dataset(
                name=FeatureFields.SAMPLE_ENDS.value,
                data=self._serialize_ndarray(basis_features.sample_ends),
                attributes={},
            ),
            HDF5Dataset(
                name=FeatureFields.UNITS.value,
                data=basis_features.units,
                attributes={},
            ),
        ]

    def _build_root_attributes(self, metadata: FeatureMetadata) -> Dict[str, Any]:
        return {
            FeatureFields.FEATURE_IDENTIFIER.value: metadata.feature_identifier.value,
            FeatureFields.USER_DATA_IDENTIFIER.value: metadata.user_identifier.value,
            FeatureFields.IMU_DATA_IDENTIFIER.value: metadata.imu_data_identifier.value,
            FeatureFields.VERSION.value: self.version,
            FeatureFields.EXTRACTION_BACKEND.value: metadata.extraction_backend,
            FeatureFields.STRIDE_FEATURE_CATALOG.value: metadata.stride_feature_catalog,
            FeatureFields.EXTRACTION_PROFILE.value: metadata.extraction_profile,
            FeatureFields.EXTRACTION_LIBRARY_VERSION.value: metadata.extraction_library_version,
            FeatureFields.EXTRACTED_AT_UTC.value: metadata.extracted_at_utc,
        }

    def _serialize_ndarray(self, value: np.ndarray) -> Any:
        return value.tolist() if isinstance(value, np.ndarray) else value
