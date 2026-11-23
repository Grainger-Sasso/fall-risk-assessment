# Upstate (Carapace ActiGraph) dataset: evaluation for the fall-risk pipeline

*Evaluated 2026-11-23 against branch `fall-risk-pipeline`.*
*Source: `E:\fafra_data\upstate\`, which holds `Carapace_ActigraphDataShare.zip` (5.0 GB, 850 entries) and
`Participant Demographics_to share(Lab Testing ).csv` (53 rows × 31 columns).*

Participants are referred to by study number only (P001–P054). This document deliberately
contains no names, dates of birth or folder initials. See [§6 Privacy](#6-privacy-the-share-is-not-de-identified).

---

## 1. Bottom line

| Question | Answer |
| --- | --- |
| Is the IMU data compatible with the pipeline? | **Yes, closely.** Raw 100 Hz tri-axial accelerometry (ActiGraph wGT3X-BT, ±8 g, 256 counts/g), the same sampling rate and units the pipeline already handles. `pygt3x` and `GT3XToH5Converter` are already in the repo. |
| Is it readily convertible? | **Yes, with one required fix.** 77 of 98 recordings parse correctly as-is. **21 recordings from 11 participants** have 1–3 junk bytes at a fixed offset (byte ~1006) of `log.bin`. `pygt3x` mishandles these: lab files decode to **zero samples**, and free-living files **silently lose 0.7–25.6 h** with no error, or crash (P052). Rebuilding `log.bin` from checksum-valid records (a checksum-based resync) restores all 21 to full length (verified). The converter itself is still a prototype (dummy IDs, debug prints, no user/spec JSON) and needs a batch driver. |
| Does it pass the pipeline's own IMU validation? | **Yes.** All 98 recordings pass `ImuValidationMixin.prepare_accelerometer` (monotonic time, declared vs. inferred rate and unit, variance) once resynced. |
| Does gait extraction produce sensible output? | **Yes.** skdh (free-living profile) finds walking in 50 of 51 lab sessions (median 211 strides). Device gait speed correlates with the clinic's stopwatch gait speed (r = 0.59 overall, r = 0.73 for "Back"-worn devices), which also confirms the file-to-CSV participant mapping. Free-living recordings contain a median of ~3 h of walking-like 10-s windows. |
| Can it be used to **train or validate** a fall-risk model? | **Not yet.** No faller status or fall history appears anywhere in the share. An older demographics workbook gives 12-month fall history for P001–P018 only (§3.1). |
| Can it be used for **inference** with the LTMM-trained bundle? | Yes, technically: inference only needs IMU data plus height. Without outcomes, though, the predictions can't be scored. |

**Usable cohort once converted:** 46 participants with a usable 4-day free-living recording
(29 "Back", 17 "Waist"); 45 of them also have clinical test data. 51 participants have a
usable lab session.

---

## 2. What is in the share

| Component | Contents | Pipeline relevance |
| --- | --- | --- |
| `FaFRA_XX_NNN/` (53 folders; P011's is misspelled `FaaFRA_…`) | Per participant: a **lab-session** `.gt3x` (8–38 min, clinic visit) and a **4-day free-living** `.gt3x` in an `Activity Data*` subfolder (one spelled `Actiivity`), each with an ActiLife `1sec.agd` (1-s counts, SQLite). | **Primary input.** The raw `.gt3x` is what the pipeline needs. The `.agd` files are useful for wear-time QC. |
| `EmoryShare/` (52 folders + `Processed ActiGraph Master File/`) | ActiLife-derived 60-s epoch reports: Freedson cut-points, sedentary bouts, sleep scores, daily/hourly totals. | Not usable for gait extraction (count-based, 60 s). Could provide secondary physical-activity covariates. One folder is labelled with participant number 045 but different initials from the P045 device folder. |
| `CN_Aug2024WearTest/`, `CN_Sept2024PilotData/` | Device wear tests and pilot runs (not study participants). | Exclude. |
| P001 folder: `…RAW.csv` (704 MB), `…_log.pdf`, `…Sleep.*` | ActiLife raw CSV export, a paper log and a sleep report for P001. | Redundant with the `.gt3x`. |
| P039 folder: `Lab Data_Output.*`, `Vector Mag All Lab Tests.pdf` | ActiLife scoring of P039's lab session. | Not needed. |
| P016 folder: `1-sec/` subfolder | Byte-identical duplicate of P016's lab `.gt3x`. | Ignore. |
| Demographics CSV | Lab clinical tests: MoCA-blind, ABC (%), GDS, grip ×3, comfortable gait speed ×2, TUG / TUG-cognitive / TUG-manual ×2, 6 static balance stances, single-leg stance R/L, 30-s sit-to-stand, 2-minute walk distance. | No age, sex, height, weight or fall history (see §3). |

**Device metadata is uniform:** all 99 `.gt3x` files are wGT3X-BT, firmware 1.9.2, 100 Hz,
±8 g, scale 256, 0 unexpected resets, idle-sleep mode enabled, no `calibration.json`
(pre-calibrated). These are exactly the defaults hard-coded in `GT3XToH5Converter`.
Free-living recordings were all programmed for exactly 96 h.

**Demographics live in the device files, not the CSV.** Every `info.txt` and `.agd` carries
sex, age, height (cm) and mass (kg), consistent across each participant's files. These can
populate `clinical_demographic_data.json`. Height is required by both skdh and mobgap.

---

## 3. Missing data

### 3.1 Labels: partial, and outside this share

There is **no faller / non-faller status, retrospective fall count or prospective fall
follow-up** in the CSV, the device metadata or the EmoryShare reports. The training path
(`train_fall_risk_model`) requires `FallerStatus.FALLER` / `NON_FALLER`.

> **Update (2026-11-23, conversion):** two older files elsewhere on `E:` fill part of this gap.
>
> - `raw_data/2024_Upstate/upstate_data/demographics_2025_03_13.xlsx` has **"Fallen in 12 months"** for **P001–P018** (10 fallers, 8 non-fallers). That is retrospective fall history, the same kind of label as LTMM. The converter writes it to `faller_status`; everyone else is `unknown`.
> - `raw_data/Upstate_2026/Carapace Risk Groups_Pilot for CSM 2026.xlsx` puts 49 participants into Risk (12) / Middle (27) / Non-Risk (11) groups. P031 is listed under both Risk and Middle. These are groupings, not fall outcomes, so the converter records them only in `participants.csv`.
> - The workbook also has height and weight for P001–P018. They agree with the device metadata within 2 cm except **P016 (device 160.0 cm vs sheet 152 cm)** and **P018 (device 162.6 cm vs sheet 193 cm)**. It confirms that P004 really weighs ~52 kg.
>
> Eighteen labelled participants are enough for criterion checks, not for training. Ask the study team for fall history on P019–P054.

The CSV does include established clinical fall-risk screens that could serve as **surrogate**
labels, with caveats:

| Surrogate cutoff | n flagged / 51 |
| --- | --- |
| TUG ≥ 12 s (CDC STEADI) | 6 |
| TUG ≥ 13.5 s (Shumway-Cook) | 2 |
| ABC < 67 % | 4 |
| Comfortable gait speed < 0.8 m/s | 3 |
| Any of TUG ≥ 12 / ABC < 67 / gait speed < 0.8 | 10 |

This is a relatively high-functioning cohort (median TUG 9.2 s, range 6.7–16.6). With ≤ 10
positives, a surrogate-label classifier would be badly underpowered. A surrogate also isn't
the same target as the LTMM model, which was trained on fall history. **Ask the study team
whether fall history (e.g., falls in the prior 12 months) or prospective fall calendars were
collected.**

### 3.2 Participant coverage

52 participants have device data (P001–P054 except P030 and P035).

| Gap | Participants | Impact |
| --- | --- | --- |
| No free-living raw `.gt3x` | P019, P022 (empty `Activity Data_*` folders); P034, P039 (no folder); P020 (only the 1-s `.agd`, no raw) | 5 participants have lab data only. The `.agd` can't be up-sampled to raw. |
| No lab `.gt3x` | P052 | Free-living only. |
| Free-living recording barely worn | P051 (~24 h of 96 h worn, 1 valid day, 1,918 steps total) | Exclude or flag. |
| Large non-wear | P045 (~66 h worn), P016 (~71 h) | Still ≥ 3 valid days; usable. |
| Very little walking detected in free-living | P013 (49 walking-like 10-s windows), P051 (77), P014 (94) | Expect sparse stride features. P014 is also the slowest participant (TUG 16.6 s). |
| Clinical CSV row blank | P030, P035 | No device data either, so no impact. |
| Device data but no CSV row | P054 | No clinical tests. |
| Partial clinical rows | P013 and P052 (no grip), P018 (dominant hand "?"), P023 (no 2MWT), P036 (no gait-speed trial 1), P041 (no 30STS) | Minor. |

### 3.3 Data-entry issues in the CSV

- Balance columns mix `P`, `p`, `F`, `F (12.3)`, `F(12.3)`, `F (<1)` and `Not attempted`. These need a parser that yields pass/fail plus hold time.
- `30 STS` for P013 is `11(arms)`, meaning arms were used, which is a protocol deviation.
- `ABC` is a percent string. One `TUG Cog` average is given to 3 decimals (P049).
- The ID format differs from the folders (`FaFRA_013` vs `FaFRA_XX_013`). Join on the 3-digit number. The gait-speed correlation in §4.1 suggests that join is correct.
- Device-metadata plausibility: P004's low mass (52.2 kg at 177.8 cm) is confirmed by the demographics workbook. Device heights for P016 (160.0 cm vs 152 cm) and P018 (162.6 cm vs 193 cm) disagree with the workbook. Verify them before relying on those participants' gait speed, which uses height.

---

## 4. Signal quality

### 4.1 Lab sessions (51)

- 8–38 min each. Timing is perfect: every second has exactly 100 samples, with no gaps and no non-monotonic timestamps. No clipping at ±8 g.
- Idle-sleep mode is active for 1–59 % of each session. `pygt3x` fills those periods with the last value and flags them in the idle mask. They are non-walking periods, so gait detection skips them.
- **Gait extraction** (skdh `GaitLumbar` with LightGBM gait detection, i.e., the pipeline's `free_living` profile) on all 51 sessions:
  - 50/51 yield gait bouts (median 211 strides, min 59). P036 yields none.
  - Device median gait speed vs. clinic comfortable gait speed: **r = 0.59** (p < 1e-5, n = 49), mean bias +0.13 m/s. vs. Avg TUG: r = −0.61.
  - By placement label: **"Back" r = 0.73, bias +0.08 m/s (n = 30); "Waist" r = 0.49, bias +0.21 m/s (n = 19).**
  - The `treadmill` profile is **not** appropriate for these sessions. It treats the whole mixed-task recording as one bout (4–23 strides). The share has no task timestamps for segmenting the walking tests.

### 4.2 Free-living recordings (47)

- After resync, **all 47 are exactly 96.0 h** (34.56 M samples). No non-monotonic timestamps. 6 files have 1–4 gaps of at most 2 s, which is negligible. Clipping is effectively zero (worst file: 1 sample in 34 M).
- Idle-sleep mode covers a median 61 % of each recording (range 49–93 %). Combined with nights, that is expected.
- **Wear time** (from the 1-s `.agd`, ≥ 90 min of zero counts = non-wear): median 93.9 h of 96. 44 of 47 recordings have ≥ 4 valid days (≥ 10 h worn), 2 have 3, and P051 has 1.
- **Walking content:** a median of 1,072 walking-like 10-s windows (~3 h) per recording. The lowest are P013, P051 and P014 (see §3.2).
- All 47 pass the pipeline's IMU validation. Full skdh free-living extraction over 4 days wasn't run here; the lab sessions exercise the same code path.

### 4.3 Sensor placement

The `Limb` field is **"Back" for 32 participants and "Waist" for 20** (consistent between each
participant's lab and free-living files). Both labels occur throughout the enrollment period.

- A spectral check (device Z axis shows step-frequency energy like an AP axis; X shows stride-frequency energy like an ML axis) gives a similar orientation signature for both groups: every recording is worn upright with Z ≈ anteroposterior. The check is slightly weaker for "Waist" (median index 4.3 vs 4.9).
- The weaker gait-speed agreement and larger positive bias for "Waist" (§4.1) suggest a real difference in mounting: hip or anterior waist rather than the lumbar midline.
- skdh `GaitLumbar` and the LTMM training data both assume **lumbar** placement. Treat placement as a known covariate. Run a "Back"-only primary analysis (29 participants with usable free-living data, 28 of them with clinical data) and confirm the actual mounting protocol with the study team.

---

## 5. Conversion to the pipeline format

### 5.1 Mapping

| Pipeline artifact | Source in this share | Notes |
| --- | --- | --- |
| `imu_data.h5` (`IMUData`, one tri-axial accelerometer `SensorData`) | `.gt3x` → `pygt3x.FileReader.acceleration` ÷ 256 → g | Already implemented in `GT3XToH5Converter._build_sensor_data`. 100 Hz declared and inferred. The idle-sleep flag maps to `idle_mask`. |
| `user_data.json` | Study number | Use `P0NN`, not the folder initials. |
| `clinical_demographic_data.json` | `.gt3x` `info.txt`: `Sex`, `Age`, `Height` (cm → m), `Mass` (kg) | `faller_status` = `unknown` until labels exist. `name` must stay `""`. |
| Instrument spec JSON | Constant: ActiGraph wGT3X-BT, 100 Hz, ±8 g, 256 counts/g | Write one `actigraph_wgt3xbt` spec. |
| Clinical test scores | Demographics CSV | `ClinicalAssessment` is still a stub, so these have nowhere to go yet. Keep them as a side table for surrogate labels and validation. |

### 5.2 Required changes

1. **Validate and resync `log.bin` before every parse.**
   - **Affected participants:** P008, P024, P032, P036, P040, P046, P047, P048, P049 and P050, in both their lab and free-living files, plus P052's free-living file.
   - **The defect:** 1–3 junk bytes sit right after the 4 header records (byte 1005–1007). P049's free-living file also has a 459-byte corrupt run mid-file, which costs about 4 s.
   - **How `pygt3x` fails:** its `LogReader` doesn't resynchronise. Lab files return 0 samples. Free-living files **silently** return truncated data: P046 loses 20.5 h, P049 25.6 h, P040 8.6 h, P050 6.2 h, the others 0.7–3.3 h. P052 raises `ValueError: cannot reshape array…` because a garbage record passes its 8-bit checksum.
   - **The fix:** walk the record chain, which is cheap. If it breaks, rebuild `log.bin` from records that start with `0x1E` and have a valid checksum, then hand that to `FileReader` as an in-memory zip. Verified on all 21 files: each comes back at full length and passes validation.
   - **Why the check must be unconditional:** a "did we get any samples?" check is not enough, because the free-living failure is silent.
2. **Turn `GT3XToH5Converter` into a batch converter**, as `DATToHDF5Converter.convert_all_dat_to_h5` already is:
   - generate real IMU IDs (`IdentifierGenerator`) instead of `"dummy_imu_id"`;
   - pass the participant ID;
   - export through `IMUDataFileExporter`;
   - write the user and clinical JSONs from `info.txt`;
   - remove the debug `print`s and the hard-coded macOS paths in `main()`.
3. **Register lab and free-living sessions as separate IMU records per participant**, both linked to one user, and pick the right one at feature-extraction time. Use the `free_living` profile for both (see §4.1).
4. **Storage:** the HDF5 writer stores float64 with no compression. A 4-day record (34.6 M samples × 5 columns) is about 1.4 GB, so roughly 65 GB for all 47 free-living files. That fits on E: (1.1 TB free). Adding `compression="gzip"` to `HDF5FileWriter` or storing g as float32 would cut this substantially.
5. **Never copy `Subject Name` / `DateOfBirth`** into converted artifacts. The current converter only copies device type and serial, which is correct. Keep it that way.

### 5.3 Domain-shift caveats for applying LTMM-trained models

- LTMM is a lumbar sensor at 100 Hz, 3 days, with **gyroscope**. This share is accelerometer-only. The skdh backend only uses accelerometry, so it is fine. mobgap's gyro-dependent steps are not.
- 4-day vs. 3-day recordings: aggregate per-day or truncate to match the training distribution.
- Mixed placement ("Waist") is out-of-distribution for a lumbar-trained model.

---

## 6. Privacy: the share is not de-identified

- **Every** `.gt3x` `info.txt` (99/99) and every `.agd` `settings` table (101/101) contains the participant's **full name** (`Subject Name`) and **date of birth** (`DateOfBirth`).
- The EmoryShare CSVs (`Subject` column) and P039's lab output CSV also contain names.
- Folder names contain participant initials.
- Recording dates and timezone are in every file.

Handle the raw share as identifiable data under the study's DUA/IRB terms. Keep it off cloud
sync and out of git. Strip these fields during conversion. If you need to share it, ask the
study team for a scrubbed re-export.

---

## 7. Conversion and feature extraction (done 2026-11-23)

**Converter.** `src/util/data_converters/upstate/upstate_converter.py` is a batch driver. It builds on
`gt3x_log_repair.py` (record-chain check + resync) and a rewritten `GT3XToH5Converter`
(`read_gt3x` repairs in memory and drops `Subject Name` / `DateOfBirth` at parse time). The
README has the command. Repaired copies of the 21 defective files are in
`E:\fafra_data\upstate\repaired\` (with `repair_report.json`); each opens at full length in
unmodified `pygt3x`.

**Converted data** (in `C:\fafra_data\assessment_database\`; converted on the external `E:` drive, then copied to the PC and its indexes re-rooted with `relocate_index`):

| Dataset | Records | Index | Size |
| --- | --- | --- | --- |
| `imu_data/upstate` (4-day free-living) | 47, all 96.0 h | `sql_db_indexes/upstate/index.db` | 32 GB (float32 axes) |
| `imu_data/upstate_lab` (clinic session) | 51 | `sql_db_indexes/upstate_lab/index.db` | 139 MB |
| `user_data/upstate` | 52 users: 10 faller, 8 non-faller, 34 unknown | both indexes | — |

All 98 converted (21 repaired, 0 failed). Timestamps are monotonic and the largest gap is 2 s.
`conversion_reports/upstate/` holds `conversion_report.json` and `participants.csv` (study ID,
limb label, faller status and source, risk group, height cross-check, IMU IDs, repair flags).

**Feature extraction** (`run_feature_extraction`, skdh, `free_living` profile):

| | Free-living | Lab |
| --- | --- | --- |
| Records with features | **47 / 47** (27 min) | **50 / 51** (P036: no stride events, as in §4.1) |
| Strides per record, median (min) | 7,073 (532, P014) | 206 (59) |
| Device gait speed vs clinic comfortable gait speed | r = 0.36 (n = 46, p = 0.015) | r = 0.59 (n = 49, p < 1e-5) |
| …"Back" / "Waist" | r = 0.48 / 0.25 | r = 0.73 / 0.49 |
| Device gait speed vs Avg TUG | r = −0.43 | r = −0.61 |

The lab numbers match the pre-conversion analysis in §4.1 exactly, so conversion changed
nothing. Free-living speed agrees less with the clinic stopwatch than lab speed does, as it did
in LTMM; the two measure different things.

Both indexes validate and load into `ClassificationDatasetBuilder`, which keeps the 18 labelled
participants (10 fallers).

**Flag for review:** median free-living gait speed of 1.9–2.0 m/s for P026, P052 ("Waist") and
P036 ("Back"). That is implausible for this cohort. P036's lab session also yielded no strides.
Check those devices' mounting before using their features.

## 8. Recommended next steps

1. **Ask the study team for fall history on P019–P054** (and any prospective follow-up). Confirm the "Back" vs "Waist" mounting protocol, and P016/P018 heights.
2. Use the 18 labelled participants and the clinic measures for **external validity** of the features. Use the full cohort for **distribution-shift checks** of the LTMM bundle (cohort-percentile context), not for performance claims.

---

### Appendix: how this was evaluated

All checks ran read-only against the zip without extracting it, using the project `.venv`
(`pygt3x`, skdh 0.17.9):

- metadata survey of every `.gt3x` `info.txt` and `.agd` settings/data table;
- a record-chain scan of every `log.bin`;
- full raw parse of every `.gt3x` with `pygt3x` (resync applied where the chain is broken) for timing, idle-sleep, clipping and orientation;
- `ImuValidationMixin.prepare_accelerometer` on each converted `SensorData`;
- `SkdhBackend.extract` (free-living profile) on every lab session;
- wear time from the 1-s `.agd` counts (≥ 90 consecutive zero-count minutes = non-wear; valid day = ≥ 10 h worn).

A per-participant inventory (no names or initials; includes device-reported age, sex, height
and mass) is at `E:\fafra_data\upstate\evaluation\participant_inventory.csv`. It is kept beside
the data rather than in this repository.
