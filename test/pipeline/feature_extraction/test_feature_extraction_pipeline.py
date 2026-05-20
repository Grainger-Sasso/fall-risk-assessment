from pathlib import Path

from src.database_manager.database_generator import DatabaseGenerator
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from src.pipeline.feature_extraction.feature_extraction_pipeline import (
    FeatureExtractionPipeline,
)


def _build_database_paths(base_path: Path):
    registry_paths = {
        IMUDataIdentifier: base_path / "registries" / "imu_data",
        UserIdentifier: base_path / "registries" / "user_data",
        RawFeatureIdentifier: base_path / "registries" / "raw_feature",
        AggregateFeatureIdentifier: base_path / "registries" / "agg_feature",
    }
    mapping_paths = {
        (IMUDataIdentifier, UserIdentifier): base_path
        / "mappings"
        / "imu_to_user_mapping",
        (RawFeatureIdentifier, IMUDataIdentifier): base_path
        / "mappings"
        / "raw_feat_to_imu_mapping",
        (AggregateFeatureIdentifier, RawFeatureIdentifier): base_path
        / "mappings"
        / "agg_feat_to_raw_feat_mapping",
    }
    output_dir_paths = {
        RawFeatureIdentifier: base_path / "raw_feature",
        AggregateFeatureIdentifier: base_path / "agg_feature",
    }
    return registry_paths, mapping_paths, output_dir_paths


def main():
    dataset_base_path = Path(
        "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_lab_walks_2026_05_09"
    )
    registry_paths, mapping_paths, output_dir_paths = _build_database_paths(
        dataset_base_path
    )

    dataset_path = dataset_base_path / "dataset"
    dataset_name = dataset_base_path.name
    db_generator = DatabaseGenerator()
    db_manager = db_generator.generate_database(
        registry_paths, mapping_paths, output_dir_paths, validate=False
    )
    pipeline = FeatureExtractionPipeline(db_manager,treadmill_profile=True)
    pipeline.run(dataset_path, dataset_name)
    print(f'Finished pipeline run for {dataset_name}')

    # pipeline.run(
    #     Path(
    #         "/Users/graingersasso/Desktop/fafra_testing/feature_extraction/input/dataset"
    #     ),
    #     "test_feature_ex_dataset",
    # )
    # raw_feature_registry: Registry = pipeline.db_manager.registry_manager.get_provider(
    #     RawFeatureIdentifier
    # )
    # agg_feature_registry: Registry = pipeline.db_manager.registry_manager.get_provider(
    #     AggregateFeatureIdentifier
    # )
    # for identifier, path in raw_feature_registry.registry.items():
    #     raw_feat_id = RawFeatureIdentifier(identifier)
    #     raw_feature: RawFeature = pipeline.db_manager.import_data([raw_feat_id])[0]
    # for identifier, path in agg_feature_registry.registry.items():
    #     agg_feat_id = AggregateFeatureIdentifier(identifier)
    #     agg_feature: AggregateFeature = pipeline.db_manager.import_data([agg_feat_id])[
    #         0
    #     ]
    #     pass


if __name__ == "__main__":
    main()
