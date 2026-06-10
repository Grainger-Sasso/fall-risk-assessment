from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.features.feature_fields import FeatureFields
from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.stride_feature_name import parse_stride_feature_name
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier

import numpy as np

class RecordFeatureBuilder(ModelBuilder):
    version: str = "1.1"

    def build(self, input_file: HDF5Group) -> RecordFeatures:
        if not isinstance(input_file, HDF5Group):
            raise ValueError("File must contain record feature data in HDF5 group format.")

        if input_file.name != FeatureFields.RECORD_FEATURES.value:
            raise ValueError(
                f"Expected root group '{FeatureFields.RECORD_FEATURES.value}', got "
                f"'{input_file.name}'."
            )

        metadata = self._build_feature_metadata(input_file)
        epoch_group = self._get_group_by_names(input_file, [FeatureFields.EPOCH_FEATURES.value])
        stride_group = self._get_group_by_names(
            input_file, [FeatureFields.STRIDE_FEATURES.value]
        )

        epoch_features = self._build_basis_features(epoch_group, SampleBasis.EPOCH)
        stride_features = self._build_basis_features(stride_group, SampleBasis.STRIDE)

        return RecordFeatures(
            epoch_features=epoch_features,
            stride_features=stride_features,
            feature_metadata=metadata,
        )

    def _build_feature_metadata(self, root_group: HDF5Group) -> FeatureMetadata:
        attrs = root_group.attributes
        required_attr_fields = [
            FeatureFields.FEATURE_IDENTIFIER,
            FeatureFields.USER_DATA_IDENTIFIER,
            FeatureFields.IMU_DATA_IDENTIFIER,
            FeatureFields.VERSION,
        ]
        for field in required_attr_fields:
            if field.value not in attrs:
                raise ValueError(f"Missing required metadata field: {field.value}")

        feature_id = FeatureIdentifier(self._to_str(attrs[FeatureFields.FEATURE_IDENTIFIER.value]))
        user_id = UserIdentifier(self._to_str(attrs[FeatureFields.USER_DATA_IDENTIFIER.value]))
        imu_id = IMUDataIdentifier(self._to_str(attrs[FeatureFields.IMU_DATA_IDENTIFIER.value]))
        _ = self._to_str(attrs[FeatureFields.VERSION.value])

        return FeatureMetadata(
            feature_identifier=feature_id,
            user_identifier=user_id,
            imu_data_identifier=imu_id,
            extraction_backend=self._optional_attr(
                attrs, FeatureFields.EXTRACTION_BACKEND, "skdh"
            ),
            stride_feature_catalog=self._optional_attr(
                attrs, FeatureFields.STRIDE_FEATURE_CATALOG, "skdh"
            ),
            extraction_profile=self._optional_attr(
                attrs, FeatureFields.EXTRACTION_PROFILE, "free_living"
            ),
            extraction_library_version=self._optional_attr(
                attrs, FeatureFields.EXTRACTION_LIBRARY_VERSION, ""
            ),
            extracted_at_utc=self._optional_attr(attrs, FeatureFields.EXTRACTED_AT_UTC, ""),
        )

    def _build_basis_features(
        self, basis_group: HDF5Group, sample_basis: SampleBasis
    ) -> BoutFeatures:
        features = np.asarray(
            basis_group.get_item_by_name(FeatureFields.FEATURES.value).data, dtype=float
        )
        bout_starts = np.asarray(
            basis_group.get_item_by_name(FeatureFields.BOUT_STARTS.value).data, dtype=float
        )
        bout_ends = np.asarray(
            basis_group.get_item_by_name(FeatureFields.BOUT_ENDS.value).data, dtype=float
        )
        sample_starts = np.asarray(
            basis_group.get_item_by_name(FeatureFields.SAMPLE_STARTS.value).data, dtype=float
        )
        sample_ends = np.asarray(
            basis_group.get_item_by_name(FeatureFields.SAMPLE_ENDS.value).data, dtype=float
        )
        units = self._to_str_list(
            basis_group.get_item_by_name(FeatureFields.UNITS.value).data
        )

        feature_names_raw = basis_group.get_item_by_name(
            FeatureFields.FEATURE_NAMES.value
        ).data
        feature_names = []
        for feature_name in self._to_str_list(feature_names_raw):
            try:
                feature_names.append(parse_stride_feature_name(feature_name))
            except ValueError as exc:
                raise ValueError(
                    f"'{feature_name}' is not a recognized stride feature name."
                ) from exc

        return BoutFeatures(
            sample_basis=sample_basis,
            features=features,
            bout_starts=bout_starts,
            bout_ends=bout_ends,
            feature_names=feature_names,
            sample_starts=sample_starts,
            sample_ends=sample_ends,
            units=units,
        )

    def _get_group_by_names(
        self, parent_group: HDF5Group, candidate_names: list[str]
    ) -> HDF5Group:
        for name in candidate_names:
            try:
                candidate = parent_group.get_item_by_name(name)
                if isinstance(candidate, HDF5Group):
                    return candidate
            except ValueError:
                continue
        raise ValueError(
            f"None of the expected groups were found: {candidate_names}."
        )

    def _to_str(self, value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _to_str_list(self, values) -> list[str]:
        output = []
        for value in values:
            output.append(self._to_str(value))
        return output

    def _optional_attr(self, attrs: dict, field, default: str) -> str:
        if field.value not in attrs:
            return default
        return self._to_str(attrs[field.value])