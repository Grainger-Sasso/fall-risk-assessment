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
or confident null claims about modest effect sizes. The highest-value next
steps are dispersion-aware feature aggregation (testable immediately) and
external/prospective validation (requires new data).

## Results (LTMM, free-living, skdh backend)

> Filled in from `predictive_value_report_*.json` after the full run.

- Population: _pending_
- Nested-CV out-of-fold ROC-AUC (95% CI): _pending_
- PR-AUC vs prevalence: _pending_
- Permutation test: _pending_
- Conclusion under §7's claim grammar: _pending_
