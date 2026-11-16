# fall-risk-assessment

End-to-end pipeline for assessing fall risk in older adults from wearable
IMU (inertial measurement unit) data: raw recordings in, trained risk models
and per-user assessment reports out.

Built and validated on the
[Long Term Movement Monitoring (LTMM) dataset](https://physionet.org/content/ltmm/1.0.0/)
— 3-day free-living lower-back accelerometer recordings from community-dwelling
older adults with known faller / non-faller status — plus its short in-lab
walking segments.

## Pipeline

```
raw recordings (.dat/.gt3x)                       new user IMU record
        │                                                 │
        ▼                                                 ▼
  data converters                              ┌─────────────────────┐
        │                                      │  FallRiskAssessor   │
        ▼                                      │  (inference)        │
  assessment database        TRAIN             │                     │
  (HDF5/JSON payloads  ──►  gait feature  ──►  │ extraction with the │
  + SQLite index)           extraction         │ bundle's config     │
        │                   (skdh/mobgap)      │        │            │
        │                        │             │        ▼            │
        │                        ▼             │ feature vector      │
        │                 stride + epoch       │ (schema-aligned,    │
        │                 RecordFeatures       │  nanmean-aggregated)│
        │                        │             │        │            │
        │                        ▼             │        ▼            │
        │                 model selection      │ calibrated faller   │
        │                 (repeated CV) +      │ probability         │
        │                 final calibrated fit │        │            │
        │                        │             │        ▼            │
        │                        ▼             │ risk category +     │
        └──────────────►  FallRiskModelBundle ─┤ JSON/HTML report    │
                                               └─────────────────────┘
```

### 1. Convert raw data

`src/util/data_converters/` converts raw recordings (LTMM WFDB `.dat`,
ActiGraph `.gt3x`) into the assessment-database layout: one directory per
record holding `imu_data.h5`, plus per-user `user_data.json` and
`clinical_demographic_data.json` (with faller status), and instrument
specification JSON.

### 2. Index the database

```bash
python -m src.util.dataset_generator.dataset_generator   # or call build_sql_index(...)
```

Builds a SQLite index (records + typed relations: IMU→user, IMU→spec,
feature→IMU) that `DatabaseManager` serves data from.

### 3. Extract gait features

```bash
python -m src.gait_features.run_feature_extraction \
    --sqlite-db-path .../sql_db_indexes/ltmm/index.db \
    --feature-output-dir .../features/ltmm \
    --extraction-backend skdh \
    --extraction-profile free_living \
    --continue-on-error
```

Each IMU record becomes `RecordFeatures`: gait features on two sampling bases
— per **stride** and per fixed-length **epoch** — stored as
`(bout, feature, sample)` tensors. Use `--extraction-profile treadmill` for
short lab walks.

### 4. Train and export a fall-risk model

```bash
python -m src.fall_risk_assessment.train_fall_risk_model \
    --sqlite-db-path .../sql_db_indexes/ltmm/index.db \
    --output-dir .../models/ltmm \
    --holdout-participants CO002,CO010,FL004,FL010
```

Ranks five model families (elastic-net logistic regression, linear/RBF SVM,
shallow LightGBM, random forest) with repeated participant-stratified
cross-validation on the early-fusion participant dataset (aggregated stride +
epoch features per participant), picks the champion, derives the binary
operating point from calibrated out-of-fold probabilities, fits the tuned +
CV-calibrated champion on all training participants, and exports a
**FallRiskModelBundle**: a versioned directory with `model.joblib` and a
`manifest.json` carrying the feature schema, extraction config, decision
thresholds, CV metrics, training provenance, and training-cohort feature
statistics. Held-out participants are excluded from every training step so
they can be assessed later as genuinely unseen users.

### 5. Assess a user

```bash
python -m src.fall_risk_assessment.assess_fall_risk \
    --bundle-dir .../models/ltmm/fall_risk_model_<champion>_<ts> \
    --imu-dir .../imu_data/imu_data_imu_xxx \
    --user-dir .../user_data/user_XXX \
    --instrument-spec-dir .../instrument_specs/ltmm \
    --output-dir .../assessments
# or assess a DB-indexed record: --sqlite-db-path ... --imu-id imu_xxx
```

Reruns extraction with the bundle's exact configuration, aligns the record's
features onto the bundle's schema, aggregates to one participant vector, and
produces a calibrated faller probability, a low/moderate/high risk category,
a binary classification at the trained operating point, per-basis data-quality
metrics, and a JSON + HTML report with cohort-percentile context per feature.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/data_model/`, `src/data_io/` | Typed domain models (IMU, user, features, specs) and HDF5/JSON/CSV persistence |
| `src/database_manager/` | SQLite-indexed record store with typed relations and validation |
| `src/gait_features/` | Extraction pipeline, skdh/mobgap backends, dataset orchestration, run logs |
| `src/feature_validation/` | Post-hoc physiological range validation of generated features |
| `src/classification/` | Sample/participant datasets, model families, fusion, CV evaluation |
| `src/fall_risk_assessment/` | **Product layer**: trainable, deployable model bundles + single-user inference and reports |
| `src/data_visualization/` | PySide6 analysis suite and report generators |
| `test/` | Unit, integration, and real-data validation tests |

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest test
```

The skdh backend requires Python ≥3.9; the optional mobgap backend requires
Python ≥3.10. Integration tests that need the real assessment database locate
it via `test/fixtures/assessment_fixture_catalog.json` (override the root with
`FRA_ASSESSMENT_DATABASE_ROOT`).

Outputs of this pipeline are research artifacts, not medical diagnoses.
