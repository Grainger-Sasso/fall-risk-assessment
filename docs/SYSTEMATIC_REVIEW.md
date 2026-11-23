# Systematic review: is this the right setup for discriminating fall risk in LTMM?

Scope: the classification and evaluation pipeline in this repository, reviewed
from first principles (what the task is and what information the data
contains), compared with every published analysis of the LTMM cohort we could
find and with comparable cohorts, and tested empirically with selection-free
diagnostics on the same data. Companion to `METHODOLOGY_REVIEW.md`.

Evidence base (2026-11-18):

- Literature: Europe PMC full-text search for "Long Term Movement Monitoring"
  plus targeted searches; abstracts or full text of 20 papers (references at
  the end). Paywalled full texts were not available — where a validation
  method is "not reported", that is the abstract's silence, not a finding.
- Production results: `predictive_value_report_20261117_155925.json`,
  `aggregation_comparison_20261117_180437.json`.
- Diagnostics: `src/fall_risk_assessment/experiments/setup_diagnostics.py`
  → `setup_diagnostics_20261118_150253.json` (reproduced exactly by a second
  run). Every model in it is fixed in advance (median imputation,
  standardization, L2 logistic regression C = 1, balanced weights; no tuning;
  any feature selection nested inside each training fold and each
  permutation), evaluated with the production protocol: 5-fold × 5-repeat
  participant-stratified CV, participant-averaged out-of-fold probabilities,
  2000-sample participant bootstrap CIs, 200-permutation p-values. Same 67
  participants as production (CO002, CO010, FL004, FL010 held out); 64 for
  lab-walk models; 59 with both.

## Verdict

**Correct: yes. Optimal: no.**

The evaluation is leakage-free and unbiased: participant-level splits, nested
selection, bootstrap CIs, and a matched permutation null. That makes it
stricter than almost every published LTMM analysis, and its numbers are more
trustworthy than theirs.

The *configuration*, however, sits near the low end of what this cohort
supports. It uses 3-day free-living features only, searches 76 features, and
searches five model families. Selection-free diagnostics on the same people
show that simpler choices do measurably better:

| Setup (honest estimate) | n | AUC (95% CI) | Permutation p |
|---|---|---|---|
| Production: 76 free-living features, nested 5-family search | 67 | 0.533 (0.386–0.685) | 0.338* |
| Same free-living data, one pre-specified feature (gait speed) | 67 | 0.600 (0.459–0.744) | 0.124 |
| 1-minute lab walk, one pre-specified feature (gait speed) | 64 | **0.674 (0.529–0.804)** | **0.025** |
| Stopwatch lab gait speed from the clinical sheet (matched) | 59 | **0.677 (0.530–0.815)** | **0.015** |

\* The production permutation test reruns the pipeline with reduced grids,
so its p-value compares a matched observed AUC of 0.525 against a null mean
of 0.499.

The realistic ceiling for this cohort and task is **AUC ≈ 0.65–0.75**. That
range is where every honestly validated comparable analysis lands, and where
this cohort's own clinical tests land (TUG 0.68, gait speed 0.67, FSST 0.66).

Published LTMM claims of AUC 0.80–0.88 come from analyses that chose models,
features or fusion weights on the evaluation data, or that do not report their
validation.

The production pipeline under-performs that band for two reasons, both
measured below:

1. **Too much flexibility for n = 67** (costs ≈ 0.07 AUC).
2. **A free-living feature representation that carries less signal than the
   same participants' 1-minute lab walk** (costs ≈ 0.04–0.10).

No setup can turn 67 participants into a deployable model. At a true AUC of
0.70, standard sample-size criteria support about **one** pre-specified
predictor.

## 1. The problem from first principles

### 1.1 What is being discriminated

- **The label is recurrent-faller status, looked back over 12 months.** LTMM
  classes participants as fallers when they self-report **≥ 2 falls in the
  past year**; everyone else is a "non-faller". In our 67, 8 of 38 non-fallers
  had exactly one fall, and one faller (FL008) has fewer than two falls
  recorded in the spreadsheet. All 67 labels match the spreadsheet groups.
- **Retrospective design.** This is cross-sectional discrimination of known
  status, not prediction. Falls can *change* the measurements (fear of
  falling, cautious gait, low mood), which helps retrospective discrimination
  but says nothing about prospective risk. The "6 Months Fall" column is a
  subset of the past-year count (it never exceeds it), so no prospective
  label is available.
- **Label noise is intrinsic.** Twelve-month recall of falls is known to miss
  a substantial share of falls (Cummings et al., 1988). Single fallers sit in
  the control group.
- **The cohort is restricted, which compresses gait differences.** Ages are
  65–87, diagnosed gait or balance disorders were excluded, and MMSE > 24 was
  required. Consistent with this, age and sex carry no signal here (AUC 0.50
  each; §6.2).
- **Gait is only one causal pathway to a fall.** Cognition, medication,
  vision, environment and exposure all contribute. For gait measures, the
  plausible effect size between fallers and non-fallers is moderate: a
  standardized difference of 0.4–0.8 corresponds to an AUC of about
  0.61–0.71 under a binormal model.

### 1.2 What information the data contains

- **Per participant:**
  - ~3 days of 100 Hz tri-axial lower-back acceleration in free living.
  - A 1-minute lab walk.
  - A clinical battery: TUG, Berg, DGI, FSST, ABC, lab gait speed (normal and
    dual-task), and cognitive and mood scales.
- **Millions of samples per person, but one label per person.** Windows,
  strides and epochs from one person are pseudo-replicates of that person's
  label, not independent examples. The number of independent training
  examples is 67 people, of whom 29 are fallers. Within-person data improves
  the *reliability* of each person's features (one week of daily-life gait
  gives ICC > 0.75; Rispens et al., 2015). It does not add labeled examples.
- **Window-level splits leak identity.** In a 296-person cohort, the same deep
  model scored AUC 0.65 with subject-wise splits and 0.94 with sample-wise
  splits (Nait Aicha et al., 2018). In a 163-person cohort, window-level CV
  gave about 0.99–1.00 while subject-wise CV gave 84–88% accuracy (Sci Rep
  2026, GSTRIDE).

### 1.3 What the sample size allows

These numbers were computed for this cohort's prevalence (29/67). They use
Riley's shrinkage criterion (S = 0.9) and the Hanley–McNeil standard error.

| True AUC | Participants needed per candidate predictor | Predictors supportable at n = 67 | Power to detect (single pre-specified score, two-sided α = 0.05) |
|---|---|---|---|
| 0.60 | 288 | 0.2 | 0.28 |
| 0.65 | 128 | 0.5 | 0.55 |
| 0.70 | 71 | 0.95 | 0.82 |
| 0.75 | 44 | 1.5 | 0.96 |
| 0.80 | 30 | 2.3 | 1.00 |

- **Uncertainty under the null:** the standard error of the AUC is 0.072,
  so any estimate carries a ±0.14 band.
- **Overall risk alone needs a larger sample:** estimating the overall risk
  to ±0.05 needs n ≈ 377.
- **Penalization does not rescue small samples.** Ridge or elastic-net
  shrinkage is itself estimated unreliably at this size (Van Calster et al.,
  2020; Riley et al., 2021).

### 1.4 The implied optimal strategy

When n is this small and the expected signal is moderate:

- **Fix the model in advance and use the data mainly to evaluate it.** That
  means one to three literature-justified features, a simple monotone model,
  permutation inference, and every participant in the evaluation.
- **Expect data-driven search to hurt.** Searching many features, families
  and hyperparameters spends the degrees of freedom the data does not have,
  and nested CV will honestly report the cost. That pattern is exactly what
  §6 finds.

## 2. Audit of the current setup

| Component | Current choice | Assessment |
|---|---|---|
| Label | LTMM recurrent-faller status | Used correctly. Claims should name it: retrospective, ≥ 2 falls. |
| Unit of analysis | Participant-level splits, imputation, scaling, calibration and thresholds | **Correct.** Most published LTMM results fail here (§3). |
| Gait detection (free living) | skdh LightGBM gait classifier, bouts ≥ 8 s, all bouts pooled | **Weak point.** Bout length strongly shifts free-living gait measures (Del Din et al., 2016). Pooling every bout of 8 s or more mixes context with gait quality. Extracted speed is valid but noisy (ρ = 0.67 vs stopwatch speed, compared with 0.82 for the lab walk; §6.3). |
| Features | 36 skdh spatiotemporal stride features + 40 generic 8-s epoch signal features | **Partly misaligned with what is known to work on LTMM.** Missing: dominant-frequency *amplitude and width* (Weiss 2013's discriminating measures), multiscale entropy (Ihlen 2016a), local dynamic stability (Ihlen 2016b), daily-life postural transitions (Iluz 2016), and amount and bout-structure of walking. |
| Aggregation | Mean per participant | Reasonable. Dispersion aggregates were tested and gave no gain (`aggregation_comparison`), consistent with Rispens 2015 (extremes are not significantly better than medians). |
| Model search | 5 families × grids, family chosen per fold | **Too flexible for n = 67.** Selection is unstable (5 different families won across 25 folds). Every diagnostic shows AUC falling as flexibility rises (§6.5). |
| Evaluation | Nested 5×5 CV, participant bootstrap, matched permutation test | **Gold standard for internal validation.** Keep it. Report the participant-averaged AUC (pooling across 25 different models mixes their calibrations; 0.476 pooled vs 0.533 averaged). |
| Below-chance results | Flat-CV family AUCs of 0.36–0.44 | **Not a bug.** Under weak or no signal, small-sample CV is biased below 0.5: an age + sex model scores 0.264 against a permutation null mean of 0.447. Always judge against the permutation null, never against 0.5. |
| Holdouts | 4 participants excluded from every analysis | Costs 6% of the sample. Fine for the product demo, but scientific estimates should use all 71. |
| Product bundle | Random forest on the 76 free-living features | Its honest discrimination is about chance. The risk reports it produces should not be read as informative. |

## 3. Published LTMM analyses

| Study | Data | n (F/NF) | Approach | Validation | Reported result | Appraisal |
|---|---|---|---|---|---|---|
| Weiss 2013 (original) | 3-day | 71 (≥ 2 falls) | Steps, activity, dominant-frequency amplitude and width per axis | Not in abstract | Steps did not differ (p = .42). Vertical amplitude lower in fallers (0.58 vs 0.71, p = .008). ML axis also differed. 3-day measures improved identification over performance tests (p < .005). | No AUC in abstract. Group differences, not a validated classifier. |
| Ihlen 2016a | 3-day, 50-s epochs | 71 (32/39) | Multiscale entropy (RCME, RMPE), PLS-DA, vs 35 conventional Weiss features | Not in abstract | AUC > 0.8 for entropy; conventional features lower | Many scales and axes. Unknown whether the number of components or scales was chosen inside CV, so likely optimistic. |
| Ihlen 2016b | 3-day | 70 (31/39) | Local dynamic stability, 3 algorithms | Not in abstract | AUC 0.59 / 0.73 / 0.75 depending on algorithm | Best algorithm chosen after seeing results; the authors note it depends on parameters. |
| Iluz 2016 | 3-day transitions | 33 fallers vs 38 older adults | Sit-to-walk and walk-to-sit transitions, ML | Not in abstract (5,027 transitions) | 88% accuracy vs ~72% for lab tests | Transition-level data. Identity leakage cannot be ruled out. |
| Altunkaya 2024 | Lab walk | 73 (35/38) | 168 features, ReliefF, MLP | 10-fold CV. Feature count and hidden size chosen by test accuracy. | AUC 0.84–0.88, accuracy 82% | Selection on the evaluation data; optimistic. |
| Askhatova 2026 | Lab walk + 3-day features | 63 (34/29) | Hybrid TCN + 5 MI-selected features | **Subject-wise 5-fold, selection inside CV** | Mean AUC 0.749, folds 0.525–0.929 | Best-designed. Architecture not nested; folds spread widely. |
| Mono 2026 | Lab walk | 69 (31/38) | SVM baseline; CNN; clinical-concept bottleneck | Leave-one-subject-out. Configurations and fusion weight chosen on the same data (the authors call it exploratory). | SVM 0.62; CNN 0.88; concept models 0.74–0.80 | Many configurations on one evaluation. Concepts include ABC. |
| Mohan 2025 | 3-day ADL means/SDs | 62 usable, 10–15 test | DBN; label = Morse Fall Scale | Single small hold-out; synthetic merged participants | Accuracy 91–93% | Not comparable: different label, tiny test set. |
| **This repo** | 3-day | 67 (29/38) | 76 features, nested 5-family search | Nested 5×5 CV + permutation | 0.533 (0.386–0.685), p = 0.34 | Leakage-free and selection-free. |
| **This repo** | Lab walk | 64 (29/35) | Gait speed, pre-specified | 5×5 CV + permutation | 0.674 (0.529–0.804), p = 0.025 | Selection-free. |

What the table shows:

- **Validation quality predicts the result.** The two analyses with
  subject-wise validation and selection inside the folds (Askhatova 2026; this
  repo's diagnostics) give 0.65–0.75. Results above 0.80 come only from
  analyses that selected on the evaluation data or do not report validation.
  Our own data reproduces that inflation: picking the best lab feature on all
  the data gives a univariate AUC of 0.768, while selecting it inside each
  fold gives 0.655. That is an inflation of about 0.11 from selection alone.
- **Most published LTMM work uses the 1-minute lab walk.** That data also
  carries more signal in our hands (§6).

## 4. Comparable cohorts and clinical benchmarks

| Study | Cohort | Outcome | Validation | Discrimination |
|---|---|---|---|---|
| Nait Aicha 2018 | FARAO, 296, 1-week trunk accelerometry | Prospective, ≥ 1 fall in 6 months | Subject-wise split | Handcrafted gait features **0.67** (0.59–0.73). Deep models 0.52–0.65. Multi-task (age/sex auxiliary, gait only) up to 0.75. Sample-wise split 0.90–0.94 (identity leakage). |
| van Schooten 2016 | 319, 1-week trunk accelerometry | Prospective, 6–12 months | Cross-validated | Time-dependent AUC **0.66–0.72** (first fall) and **0.69–0.76** (second fall), with fall history among predictors |
| van Schooten 2015 | 169, 1-week | Prospective, 6 months | Not in abstract | Clinical 0.68 → 0.82 with gait added. Likely apparent (in-sample) performance. |
| Rispens 2015 (JMIR) | 202, 1-week | Prospective | — | Extreme percentiles were not significantly better than medians |
| Gade 2021 (review) | 72 models, 86,369 people | Prospective falls | Mostly none | AUC 0.49–0.87. The 3 validated models scored **0.62–0.69**. All rated high risk of bias. |
| Barry 2014 (meta-analysis) | Community-dwelling, TUG ≥ 13.5 s | Falls | — | Sensitivity 0.31, specificity 0.74: limited predictive ability |
| Howcroft 2013 (review) | 40 inertial-sensor studies | Mixed | Mostly weak | Accuracy 62–100%. 65% of studies used the lower back. |
| Shany 2015 (review) | Sensor-based fall prediction | — | — | Documents over-optimism: small samples, many features, feature selection before validation, no external validation |

## 5. What is feasible

- **The honest band.** Leakage-free, selection-free estimates on comparable
  data cluster at AUC 0.62–0.75 (Nait Aicha's baseline 0.67; van Schooten
  2016 0.66–0.76; validated models in Gade 2021 0.62–0.69; Askhatova 0.749;
  Mono's SVM baseline 0.62).
- **This cohort's own clinical benchmark sits in the same band.** The best
  performance-based tests reach TUG 0.68, gait speed 0.67 and FSST 0.66.
- **Measures that are partly consequences of falling score higher**: ABC
  (balance confidence) 0.71, SF-36 0.71, near falls 0.69. They are not valid
  risk factors in a retrospective design.
- **A realistic target for gait-based discrimination of LTMM recurrent-faller
  status is AUC 0.65–0.75**, with a ±0.14 CI at this n.
- **Treat AUCs above 0.80 as evidence of optimism** unless they come with
  nested selection and subject-wise validation.

## 6. Diagnostics on this data

### 6.1 Labels

- All 67 labels match the spreadsheet groups.
- 8 non-fallers had exactly one fall.
- FL008 is labeled a faller with fewer than two recorded falls.

### 6.2 Clinical benchmark

These are univariate AUCs with the direction fixed in advance and no model
fitting. Picking the best of the 16 is descriptive only.

| Measure | AUC (95% CI) | n | Note |
|---|---|---|---|
| ABC balance confidence | 0.713 (0.581–0.835) | 65 | Partly a consequence of falling |
| SF-36 | 0.709 (0.564–0.841) | 62 | Partly a consequence of falling |
| Near falls, past year | 0.686 (0.534–0.830) | 52 | Partly a consequence of falling |
| TUG | 0.677 (0.544–0.799) | 67 | |
| Lab gait speed (stopwatch) | 0.668 (0.529–0.792) | 67 | |
| Four Square Step Test | 0.664 (0.523–0.801) | 65 | |
| GDS (depression) | 0.655 (0.509–0.781) | 64 | Partly a consequence of falling |
| Dual-task gait speed | 0.638 (0.497–0.772) | 65 | |
| Berg | 0.591 (0.446–0.728) | 67 | |
| DGI | 0.590 (0.441–0.726) | 67 | |
| MMSE / MoCA / PASE / TMT-B | 0.52–0.55 | 64–66 | |
| Sex / age | 0.503 / 0.495 | 67 / 66 | No signal in this restricted cohort |

### 6.3 Is the free-living extraction valid?

Extracted gait speed was compared against stopwatch lab speed (Spearman ρ).

| Extracted gait speed | ρ | Median extracted vs lab (m/s) |
|---|---|---|
| 3-day free living, `GAIT_SPEED` | 0.671 (n = 67) | 1.12 vs 1.10 |
| Lab walk, `GAIT_SPEED` | 0.824 (n = 64) | 1.14 vs 1.10 |
| Either, `GAIT_SPEED_M1` | 0.64 / 0.77 | 1.43–1.48 vs 1.10 |

- The extraction works. Free-living speed tracks lab speed but is noisier,
  as expected for daily-life walking.
- **The `*_M1` step-length variants overestimate speed by about 30%.** They
  rank participants plausibly but are biased in absolute terms.
- Two fallers have implausibly little detected gait over 3 days: FL015 with
  13 strides and FL014 with 467. Excluding them changes nothing (§6.6).

### 6.4 Is there individually discriminative gait signal?

Each data source was tested with a family-wise (Westfall–Young max-T)
permutation test over all 76 features.

**3-day free living: no feature survives** (family-wise p of the best = 0.235).

- The strongest features are mediolateral dominant frequency (AUC 0.31,
  i.e. 0.69 inverted), mediolateral zero-crossing rate, `GAIT_SPEED_M1`,
  anteroposterior jerk, and vertical high-band power.
- All of them are lower in fallers.
- The mediolateral axis was also where Weiss 2013 found a difference.

**Lab walk: four features survive** (family-wise p of the best = 0.010).

- Double support (AUC 0.768; initial 0.767, terminal 0.763).
- Stance time (0.762).
- Vertical dominant (step) frequency (0.242, i.e. 0.758 inverted).
- Stride time (0.740).
- Together: fallers walk with longer double support and stance, and a slower
  step rhythm. This is the classic, well-replicated profile.

### 6.5 Models

All of these use the production CV protocol. Each is selection-free; the
"nested best-k" rows select features inside every training fold and every
permutation.

| Model | Data | n | AUC (95% CI) | p | Null mean |
|---|---|---|---|---|---|
| Production: nested 5-family search, 76 features | 3-day | 67 | 0.533 (0.386–0.685) | 0.338* | 0.499* |
| Ridge LR, all 76 | 3-day | 67 | 0.520 (0.380–0.665) | 0.373 | 0.489 |
| Pre-specified 6 gait features | 3-day | 67 | 0.537 (0.381–0.693) | 0.338 | 0.487 |
| 6 gait features + stride-time SD | 3-day | 67 | 0.515 (0.364–0.660) | 0.433 | 0.493 |
| Nested best-3 of 76 | 3-day | 67 | 0.560 (0.418–0.711) | 0.259 | 0.487 |
| Nested best-1 of 76 | 3-day | 67 | 0.563 (0.420–0.711) | 0.239 | 0.485 |
| Gait speed only | 3-day | 67 | 0.600 (0.459–0.744) | 0.124 | 0.481 |
| Ridge LR, all 76 | Lab | 64 | 0.611 (0.456–0.756) | 0.104 | 0.487 |
| Pre-specified 6 gait features | Lab | 64 | 0.640 (0.493–0.785) | 0.075 | 0.482 |
| Nested best-1 / best-3 of 76 | Lab | 64 | 0.655 / 0.656 (0.51–0.79) | 0.095 / 0.085 | 0.49 |
| **Gait speed only** | **Lab** | 64 | **0.674 (0.529–0.804)** | **0.025** | 0.476 |
| Age + sex | Clinical | 67 | 0.264 (0.143–0.391) | 0.920 | **0.447** |
| Pre-specified clinical (age, sex, TUG, Berg, DGI, speed) | Clinical | 67 | 0.598 (0.465–0.732) | 0.114 | 0.479 |
| Pre-specified clinical + ABC | Clinical | 67 | 0.624 (0.494–0.755) | 0.060 | 0.481 |
| Clinical (6) + 3-day gait (6) | Combined | 67 | 0.515 (0.376–0.658) | 0.408 | 0.487 |
| Clinical (6) + lab gait (6) | Combined | 64 | 0.651 (0.509–0.787) | 0.075 | 0.494 |

\* From the production report's reduced-grid matched permutation test
(observed 0.525).

Matched comparison on the 59 participants who have both recordings:

| Gait speed source | AUC (95% CI) | p |
|---|---|---|
| 3-day free living (extracted) | 0.600 (0.453–0.745) | 0.119 |
| 1-minute lab walk (extracted) | 0.644 (0.499–0.781) | 0.045 |
| Stopwatch (clinical sheet) | 0.677 (0.530–0.815) | 0.015 |

### 6.6 Sensitivity analyses

- **Clean contrast** (non-fallers with zero falls, n = 58): 3-day gait set
  0.526, 3-day all features 0.430, clinical set 0.532. Removing single
  fallers does not unmask a signal.
- **Excluding implausibly low gait** (FL014, FL015; n = 65): 3-day gait set
  0.480, all features 0.525. Data-quality outliers are not the cause.

### 6.7 What the diagnostics establish

1. **Flexibility costs discrimination at this n, in every data source.** The
   ordering is consistent: 1 pre-specified feature, then nested best-1 or
   best-3, then 6 pre-specified features, then all 76 features (ridge or the
   5-family search). For 3-day data that runs 0.60 → 0.56 → 0.54 → 0.52–0.53;
   for the lab walk, 0.67 → 0.66 → 0.64 → 0.61. Even the
   pre-specified clinical model (0.60) scores below TUG alone (0.68), because
   age and sex add noise in this cohort. This is the small-n bias–variance
   trade-off that §1.3 predicts.
2. **The 1-minute lab walk carries more fall-related gait signal than 3 days
   of free-living recording.** This holds as processed here, for the same
   people, in every analysis: lab-walk models beat 3-day models by about
   0.04–0.10; four lab features survive family-wise correction and no
   free-living feature does. This is the opposite of Weiss 2013's
   conclusion. Their free-living measures (dominant-frequency amplitude and
   width on ambulation segments) are not in our feature set, so the
   free-living *representation* is the weak link, not free-living data as
   such.
3. **The ceiling is modest.** Even the best measures (a stopwatch, TUG) reach
   only about 0.68. At n = 67, an honest result of 0.67 with
   p ≈ 0.02 is about the strongest finding this cohort can produce.
4. **Ruled out as causes:** leakage (there is none), label noise (the clean
   contrast does not help), gait-detection failures (exclusion does not
   help), and mean-only aggregation (dispersion aggregates do not help).

## 7. Recommendations

Ranked by expected value. Each is pre-specifiable, so it can be tested without
fishing.

1. **Make the primary analysis a pre-registered, one- or two-predictor
   model.** For example, lab-walk gait speed, or double support plus gait
   speed, as plain logistic regression evaluated with 5×5 CV and a
   permutation test on all 71 participants. Keep the five-family search as a
   secondary, exploratory analysis. Adopt a complexity budget of about n / 70
   predictors at an expected AUC of 0.70.
2. **Add the lab walk as a first-class input.** The pipeline already extracts
   it (`--extraction-profile treadmill`), and it is the strongest gait source
   here. Report lab, free-living, clinical and combined models side by side.
3. **Rebuild the free-living representation from what worked on this
   cohort, fixed from the papers in advance:**
   - Weiss 2013 dominant-frequency amplitude and width per axis.
   - Bout-length standardization: features only from bouts longer than a
     pre-set length, medians over bouts.
   - Amount and bout structure of daily walking.
   - Optionally, the multiscale-entropy and local-dynamic-stability features
     of Ihlen 2016, with their published parameters.
4. **Use all 71 participants for scientific estimates.** Run the
   unseen-person product demo with leave-one-out bundles, so 4 participants
   are not permanently removed from the evidence.
5. **Do not pursue deep learning on n = 67.** On 296 participants with
   subject-wise splits, deep models matched handcrafted features at best.
   Here, any architecture search would have to be nested to mean anything.
6. **External validation is the only route to deployment claims.** The
   public GSTRIDE dataset (163 older adults, foot-mounted IMU, fall history)
   or the FARAO cohort (on request) would test whether a fixed LTMM-derived
   model transfers.
7. **Keep the evaluation as it is**: participant-level, nested, bootstrap and
   permutation. Report the participant-averaged AUC and the permutation null
   mean. Phrase claims as in `METHODOLOGY_REVIEW.md` §7. A bounded positive
   claim is now supportable for lab-walk gait speed: AUC 0.67 (0.53–0.80),
   p = 0.025, for retrospective recurrent-faller status. That is one of
   several reported analyses, so it should be confirmed as a pre-registered
   primary analysis.

## 8. Limitations of this review

- **Abstracts only for paywalled papers.** Validation details for Weiss 2013,
  Ihlen 2016a/b, Iluz 2016 and van Schooten 2015 come from abstracts. Their
  appraisals are flagged "not in abstract", not "absent".
- **About 20 models were compared.** Each estimate is selection-free, but
  choosing the best of them is not. The lab-walk advantage is treated as real
  because it is consistent across five independent analyses (pre-specified,
  nested best-1 and best-3, matched, family-wise screen), not because any
  single CI excludes the 3-day result. The CIs overlap.
- **Univariate clinical AUCs** use directions fixed in advance and are
  descriptive.
- **Everything here concerns retrospective, recurrent-faller
  discrimination.** Prospective validity is untested.

## References

LTMM cohort:

- Weiss A, et al. Does the evaluation of gait quality during daily life provide insight into fall risk? *Neurorehabil Neural Repair* 2013;27:742–752. [doi:10.1177/1545968313491004](https://doi.org/10.1177/1545968313491004)
- PhysioNet Long Term Movement Monitoring Database v1.0.0. [physionet.org/content/ltmm](https://physionet.org/content/ltmm/1.0.0/)
- Ihlen EAF, et al. The complexity of daily life walking in older adult community-dwelling fallers and non-fallers. *J Biomech* 2016;49:1420–1428. [doi:10.1016/j.jbiomech.2016.02.055](https://doi.org/10.1016/j.jbiomech.2016.02.055)
- Ihlen EAF, et al. A comparison study of local dynamic stability measures of daily life walking in older adult community-dwelling fallers and non-fallers. *J Biomech* 2016;49:1498–1503. [doi:10.1016/j.jbiomech.2016.03.019](https://doi.org/10.1016/j.jbiomech.2016.03.019)
- Iluz T, et al. Can a body-fixed sensor reduce Heisenberg's uncertainty when it comes to the evaluation of mobility? *J Gerontol A* 2016;71:1459–1465. [record](https://cris.bgu.ac.il/en/publications/can-a-body-fixed-sensor-reduce-heisenbergs-uncertainty-when-it-co-2/)
- Altunkaya S. Leveraging feature selection for enhanced fall risk prediction in elderly using gait analysis. *Med Biol Eng Comput* 2024;62. [doi:10.1007/s11517-024-03180-2](https://doi.org/10.1007/s11517-024-03180-2)
- Askhatova A, et al. A temporal convolutional network approach for gait-based fall risk prediction. *MethodsX* 2026;16:103845. [doi:10.1016/j.mex.2026.103845](https://doi.org/10.1016/j.mex.2026.103845)
- Mono L, Herke A-K, Jashari R. Neuro-symbolic fall risk detection from wearable IMU sensors. AIxIA 2026 Discussion Papers, CEUR-WS Vol-4270. [pdf](https://ceur-ws.org/Vol-4270/short7.pdf)
- Mohan D, et al. A novel cooperative AI-based fall risk prediction model for older adults. *Sensors* 2025;25:3991. [doi:10.3390/s25133991](https://doi.org/10.3390/s25133991)

Comparable cohorts and reviews:

- Nait Aicha A, et al. Deep learning to predict falls in older adults based on daily-life trunk accelerometry. *Sensors* 2018;18:1654. [doi:10.3390/s18051654](https://doi.org/10.3390/s18051654)
- van Schooten KS, et al. Ambulatory fall-risk assessment: amount and quality of daily-life gait predict falls in older adults. *J Gerontol A* 2015;70:608–615. [doi:10.1093/gerona/glu225](https://doi.org/10.1093/gerona/glu225)
- van Schooten KS, et al. Daily-life gait quality as predictor of falls in older people: a 1-year prospective cohort study. *PLoS One* 2016;11:e0158623. [doi:10.1371/journal.pone.0158623](https://doi.org/10.1371/journal.pone.0158623)
- Rispens SM, et al. Identification of fall risk predictors in daily life measurements. *Neurorehabil Neural Repair* 2015;29:54–61. [doi:10.1177/1545968314532031](https://doi.org/10.1177/1545968314532031)
- Rispens SM, et al. Do extreme values of daily-life gait characteristics provide more information about fall risk than median values? *JMIR Res Protoc* 2015;4:e4. [doi:10.2196/resprot.3931](https://doi.org/10.2196/resprot.3931)
- Weiss A, et al. Objective assessment of fall risk in Parkinson's disease using a body-fixed sensor worn for 3 days. *PLoS One* 2014;9:e96675. [PMC4011791](https://pmc.ncbi.nlm.nih.gov/articles/PMC4011791/)
- Classification of fallers and non-fallers in older adults using IMU gait signals and explainable deep learning (GSTRIDE). *Sci Rep* 2026. [nature.com](https://www.nature.com/articles/s41598-026-44368-w)
- Shany T, et al. Are we stumbling in our quest to find the best predictor? Over-optimism in sensor-based models for predicting falls in older adults. *Healthc Technol Lett* 2015;2:79–88. [doi:10.1049/htl.2015.0019](https://doi.org/10.1049/htl.2015.0019)
- Montesinos L, et al. Wearable inertial sensors for fall risk assessment and prediction in older adults: a systematic review and meta-analysis. *IEEE TNSRE* 2018;26:573–582. [doi:10.1109/TNSRE.2017.2771383](https://doi.org/10.1109/TNSRE.2017.2771383)
- Howcroft J, et al. Review of fall risk assessment in geriatric populations using inertial sensors. *J Neuroeng Rehabil* 2013;10:91. [doi:10.1186/1743-0003-10-91](https://doi.org/10.1186/1743-0003-10-91)
- Barry E, et al. Is the Timed Up and Go test a useful predictor of risk of falls in community dwelling older adults? *BMC Geriatr* 2014;14:14. [doi:10.1186/1471-2318-14-14](https://doi.org/10.1186/1471-2318-14-14)
- Gade GV, et al. Predicting falls in community-dwelling older adults: a systematic review of prognostic models. *BMJ Open* 2021;11:e044170. [doi:10.1136/bmjopen-2020-044170](https://doi.org/10.1136/bmjopen-2020-044170)

Methods (cited from the standard literature; not re-verified for this review):

- Riley RD, et al. Calculating the sample size required for developing a clinical prediction model. *BMJ* 2020;368:m441.
- Riley RD, et al. Penalization and shrinkage methods produced unreliable clinical prediction models especially when sample size was small. *J Clin Epidemiol* 2021;132:88–96.
- Van Calster B, et al. Regression shrinkage methods for clinical prediction models do not guarantee improved performance. *Stat Methods Med Res* 2020;29:3166–3178.
- Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics* 2006;7:91.
- Saeb S, et al. The need to approximate the use-case in clinical machine learning. *GigaScience* 2017;6:gix019.
- Hanley JA, McNeil BJ. The meaning and use of the area under a ROC curve. *Radiology* 1982;143:29–36.
- Del Din S, et al. Free-living gait characteristics in ageing and Parkinson's disease: impact of environment and ambulatory bout length. *J Neuroeng Rehabil* 2016;13:46.
- Cummings SR, Nevitt MC, Kidd S. Forgetting falls: the limited accuracy of recall of falls in the elderly. *J Am Geriatr Soc* 1988;36:613–616.
