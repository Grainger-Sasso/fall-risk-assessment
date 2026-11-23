# Methodology review: is this the right classifier approach?

A first-principles audit of the fall-risk classification pipeline, written to
answer one question: **can results from this pipeline definitively establish
that the model does — or does not — have predictive value?**

## 1. What is actually being predicted

LTMM labels are **retrospective faller status**: participants were enrolled as
fallers based on reported fall history, then wore the sensor for 3 days. The
pipeline therefore measures *cross-sectional discrimination of known faller
status from gait*, not *prospective prediction of future falls*. This is the
correct first rung of the validation ladder (if gait cannot even separate
known fallers, prospective prediction is hopeless), but no result here —
however strong — supports the claim "this model predicts who will fall."
Claims must be phrased as discrimination of faller status in this cohort.

## 2. Unit of analysis and leakage (sound)

Every split, stratification, aggregation, and statistic operates at the
participant level: a participant's strides/epochs never straddle train/test;
imputation and scaling are fit inside sklearn pipelines on training folds
only; calibration and decision thresholds come from training/out-of-fold data
only; cohort reference statistics in the bundle come from training
participants only; holdout participants are excluded before every training
step. The design is leakage-free by construction. This is the non-negotiable
foundation and it is correct.

## 3. Features and aggregation (sound, with a known ceiling)

Engineered gait features (skdh) on two sampling bases — per-stride temporal
/spatial parameters and per-epoch signal features — aggregated to one vector
per participant by nanmean (early fusion, ~76 features). Justification: at
n≈70 participants, engineered features plus linear/regularized models are the
statistically appropriate operating point.

Known ceilings, in order of likely impact:

- **Mean-only aggregation discards distributional information.** Gait
  *variability* (e.g., stride-time CV) is among the strongest fall-risk
  markers in the literature; a mean-only summary keeps within-feature
  variability only insofar as the epoch features encode it. Adding dispersion
  aggregates (std/IQR per feature) is the single highest-value feature-side
  upgrade — at the cost of doubling dimensionality, which nested CV can
  arbitrate honestly.
- **Self-supervised pretrained accelerometer encoders** (e.g., UK-Biobank-
  scale SSL models) are the current representation-learning frontier for
  wearables and the right move *if* this work ever has hundreds of
  participants or adopts transfer learning. Training any deep model from
  scratch on 71 participants would be methodologically indefensible —
  classical ML is not a compromise here; it is the correct choice.

## 4. Model families (appropriate)

Elastic-net logistic regression, linear/RBF SVM, shallow LightGBM, random
forest — all with class weighting, within-family hyperparameter grids, and
median imputation + robust scaling. This spans the bias/variance range that
n≈70, p≈76 data can support. Nothing more expressive is justified.

> **Update (2026-11-18, `SYSTEMATIC_REVIEW.md`):** selection-free diagnostics
> show that *searching* over these families and all 76 features costs about
> 0.07 AUC at this n. A single pre-specified feature does better on the same
> data (gait speed 0.60 free-living; 0.67 lab walk, p = 0.025). Sample-size
> criteria support about one pre-specified predictor at an expected AUC of
> 0.70. The family search should therefore be the exploratory analysis, not
> the primary one.

## 5. Evaluation protocol — the original gap, now closed

The original evaluator ranked the five families on the same repeated-CV folds
it reported metrics from. Selecting the best of five on those folds and then
quoting its fold metrics incurs **selection optimism (winner's curse)** — the
champion's reported AUC is biased upward, and by an amount that is material at
this sample size. Three additions close the gap
(`src/fall_risk_assessment/statistical_validation.py`):

1. **Fully nested CV** (`nested_family_selection_cv`): family selection runs
   on inner folds of each outer-train split; outer test folds measure the
   whole selection-plus-tuning procedure. Reported out-of-fold AUC is an
   unbiased estimate of the deployed pipeline's performance on new
   same-cohort participants.
2. **Percentile-bootstrap CIs over participants** on participant-averaged
   out-of-fold probabilities: quantifies cohort-sampling uncertainty, the
   dominant error term at n≈70.
3. **Permutation test of the identical pipeline** on label-shuffled data:
   observed and null statistics computed by the same procedure, giving an
   exact-style p-value for "better than chance". Property tests verify the
   machinery reports chance on noise and significance on signal.

Residual, acknowledged subtleties: repeated-CV folds are correlated (handled
by bootstrapping participants, not folds); the production bundle still uses
the flat evaluator to *choose* its champion (fine — selection bias affects
reported metrics, not the legitimacy of deploying the selected model, and the
nested result is what gets quoted as performance).

## 6. Calibration and decision thresholds (adequate, clearly scoped)

Sigmoid (Platt) CV-calibration is the right choice at this n (isotonic
overfits below ~1000 samples). The binary operating point maximizes balanced
accuracy on out-of-fold probabilities; the low/moderate/high bands at 1/3 and
2/3 are product UX, not clinically validated cutoffs, and are labeled as such.

## 7. Statistical power — the hard constraint no method can remove

With ~31 fallers / ~36 non-fallers, the 95% CI on an AUC spans roughly ±0.10
to ±0.14. Power to declare AUC > 0.5 at α = 0.05 is adequate (~80%+) only if
the true AUC is ≳ 0.70. Consequences for claim-making:

- A **significant** result (permutation p < 0.05 with CI excluding 0.5) *does*
  definitively establish predictive value in this cohort.
- A **null** result does **not** definitively establish absence of value — it
  rules out strong effects (true AUC ≳ 0.70) but is underpowered against
  modest ones (0.55–0.65). The honest negative claim is bounded: "no evidence
  of discrimination; effects larger than X are unlikely given the CI."

## 8. The 4-participant holdout demo

CO002, CO010, FL004, FL010 are excluded from all training and assessed through
the full raw-record → report path. This demonstrates the *inference product*
end-to-end on genuinely unseen people. Four people carry no statistical
weight; the nested-CV numbers are the performance evidence.

## Verdict

With the statistical-inference layer in place, the approach is methodologically
sound and at the defensible frontier for this data regime: participant-level
leakage-free design, appropriate model class for n≈70, nested selection-free
performance estimation, cohort-level uncertainty quantification, and an
exchangeable permutation null. The results it produces support definitive,
*bounded* claims: discrimination of retrospective faller status in the LTMM
cohort, with uncertainty made explicit. What no analysis of this dataset can
support: prospective fall prediction, generalization to other cohorts/devices,
or confident null claims about modest effect sizes. Dispersion-aware feature
aggregation has since been tested and did not help (see Results); the
remaining high-value step is external/prospective validation, which requires
new data.

## Results (LTMM, free-living, skdh backend)

> From `predictive_value_report_20261117_155925.json` (run 2026-11-17;
> 5-fold × 5-repeat outer CV, 3-fold inner CV, 2000 bootstrap resamples,
> 200 permutations, seed 42; scikit-learn 1.9.1, LightGBM 4.7.0, skdh 0.17.9).

- **Population:** 67 participants (29 fallers / 38 non-fallers, prevalence
  0.433), 76 mean-aggregated features; CO002, CO010, FL004, FL010 held out.
- **Nested-CV out-of-fold ROC-AUC:** 0.476 pooled; participant-averaged
  0.533, **95% CI 0.386–0.685**. Per-fold ROC-AUC 0.468 ± 0.156. Inner
  selection was unstable — random forest 12/25 folds, linear SVM 7, RBF SVM
  3, LightGBM 2, elastic-net 1 — itself a sign there is no consistent signal
  for any family to lock onto.
- **PR-AUC vs prevalence:** participant-averaged 0.539 (95% CI 0.370–0.716)
  against a chance level of 0.433; the CI spans chance.
- **Operating point:** balanced accuracy 0.533 ± 0.110; sensitivity 0.26,
  specificity 0.81 — the models mostly default to "non-faller".
- **Permutation test:** observed AUC 0.525 (matched fast procedure) vs null
  0.499 ± 0.076 (95th percentile 0.636); **p = 0.338**.
- **Reproducibility:** an earlier, interrupted run on a different machine and
  library stack gave nested AUC 0.473 (participant-averaged 0.516, CI
  0.369–0.670) and observed permutation AUC 0.558 — the conclusion is stable
  across environments.
- **Conclusion under §7's claim grammar:** **no evidence that mean-aggregated
  gait features discriminate retrospective faller status in the LTMM cohort.**
  The CI upper bound (≈0.69) makes strong discrimination (true AUC ≳ 0.70)
  unlikely; modest effects (AUC 0.55–0.65) can neither be claimed nor ruled
  out at n = 67. This is not evidence that gait carries no fall-risk
  information.

### Aggregation experiment (§3 follow-up)

> From `aggregation_comparison_20261117_180437.json` (completed 2026-11-18);
> identical 5×5 nested CV, seed 42, same 67 participants per variant.

| Variant | Features | Nested AUC (pooled) | Participant-averaged AUC (95% CI) | Most-selected family |
|---|---|---|---|---|
| mean | 76 | 0.476 | 0.533 (0.386–0.685) | random forest 12/25 |
| mean + std | 152 | 0.454 | 0.453 (0.311–0.610) | random forest 15/25 |
| mean + std + IQR | 228 | 0.458 | 0.440 (0.303–0.584) | elastic-net 10/25 |

**Conclusion:** dispersion-aware aggregation does not rescue discrimination.
Neither dispersion variant beats mean-only; both point estimates sit below
chance and their CI upper bounds *fall* (0.685 → 0.610 → 0.584) as
dimensionality doubles and triples — the signature of added noise, not added
signal, at n = 67. §3's highest-value feature-side upgrade is therefore tested
and negative for this cohort; mean-only remains the production aggregation,
and the remaining lever is more (or external/prospective) data, not more
summary statistics of the same features.

**Numerical note:** the mean + std + IQR variant initially hung. Several
within-participant IQR features (e.g. stride double-support asymmetry) are
quantized — 0.02 for nearly every participant — so their cross-participant
IQR is floating-point noise (~1e-10); `RobustScaler` scaled outliers to ~1e9
and the uncapped linear SVM never converged. All pipelines now use
`DegenerateSafeRobustScaler` (std scaling for columns whose IQR is negligible
relative to their spread) and SVMs are capped at `SVM_MAX_ITER`. The scaler
flags no column for mean or mean + std across all 300 outer/inner training
splits, so those rows are unaffected; it flags three stride-IQR columns for
mean + std + IQR.
