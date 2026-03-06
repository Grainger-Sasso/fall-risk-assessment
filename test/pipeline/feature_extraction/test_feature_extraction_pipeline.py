from pathlib import Path

from src.data_model.features.aggregate.aggregate_feature import AggregateFeature
from src.data_model.features.raw.raw_feature import RawFeature
from src.database_generator.database_generator import DatabaseGenerator
from src.database_manager.registry.registry import Registry
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from src.pipeline.feature_extraction.feature_extraction_pipeline import (
    FeatureExtractionPipeline,
)


def main():
    registry_paths = {
        IMUDataIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/registries/imu_data"
        ),
        UserIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/registries/user_data"
        ),
        RawFeatureIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/registries/raw_feature"
        ),
        AggregateFeatureIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/registries/agg_feature"
        ),
    }
    mapping_paths = {
        (IMUDataIdentifier, UserIdentifier): Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/mappings/imu_to_user_mapping"
        ),
        (RawFeatureIdentifier, IMUDataIdentifier): Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/mappings/raw_feat_to_imu_mapping"
        ),
        (AggregateFeatureIdentifier, RawFeatureIdentifier): Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/mappings/agg_feat_to_raw_feat_mapping"
        ),
    }
    output_dir_paths = {
        RawFeatureIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/raw_feat"
        ),
        AggregateFeatureIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/agg_feat"
        ),
    }
    dataset_path = "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/dataset"
    db_generator = DatabaseGenerator()
    db_manager = db_generator.generate_database(
        registry_paths, mapping_paths, output_dir_paths
    )
    pipeline = FeatureExtractionPipeline(db_manager)
    # pipeline.run(
    #     Path(
    #         "/Users/graingersasso/Desktop/fafra_testing/feature_extraction/input/dataset"
    #     ),
    #     "test_feature_ex_dataset",
    # )
    raw_feature_registry: Registry = pipeline.db_manager.registry_manager.get_provider(
        RawFeatureIdentifier
    )
    agg_feature_registry: Registry = pipeline.db_manager.registry_manager.get_provider(
        AggregateFeatureIdentifier
    )
    for identifier, path in raw_feature_registry.registry.items():
        raw_feat_id = RawFeatureIdentifier(identifier)
        raw_feature: RawFeature = pipeline.db_manager.import_data([raw_feat_id])[0]
    for identifier, path in agg_feature_registry.registry.items():
        agg_feat_id = AggregateFeatureIdentifier(identifier)
        agg_feature: AggregateFeature = pipeline.db_manager.import_data([agg_feat_id])[
            0
        ]
        pass


if __name__ == "__main__":
    main()
