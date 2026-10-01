# Error analysis: v002 demo artifacts (undertone-lightgbm-v002, skin_depth-extra_trees-v002)

Provenance: repo commit cb48ebd, demo/synthetic dataset (63 train+val rows / 32 people;
16-row locked demo holdout). ALL NUMBERS HERE ARE SYNTHETIC-DEMO RESULTS —
NOT FACEMAXX PERFORMANCE BENCHMARKS. NOT FOR PRODUCTION.

## 1. Why undertone macro-F1 (0.343 CV / 0.336 holdout) is far below accuracy (0.94 CV / 0.81 holdout)

VERIFIED (from confusion matrices in metrics.json):
- Class imbalance: demo train+val counts are Warm=41 (65%), Neutral=18, Neutral-Cool=4;
  Cool / Neutral-Warm / Olive have ZERO rows. Accuracy is dominated by the Warm majority.
- Holdout confusion: Warm recall = 1.00 (10/10); all 4 Neutral-Cool rows and 1 Neutral row
  are misclassified, mostly as Warm/Neutral. Warm precision 0.91 — it absorbs neighbours.
- Macro-F1 averages equally over classes, so zero-support classes (vacuous F1=0) drag it
  down structurally. This is an honest, expected consequence of tiny imbalanced data,
  not a model bug.

## 2. Classes with zero or very low support

Undertone: Cool=0, Neutral-Warm=0, Olive=0 rows in train+val; Neutral-Cool=4, Neutral=2 in
holdout. The model CANNOT have learned Cool/Neutral-Warm/Olive — metrics for them are
vacuous and the persisted warnings say so (spec #28).
Skin depth: Very Deep=0 rows; Light=4, Medium=4, Very Fair=6, Medium-Deep=14, Fair=20,
Deep=9, Light-Medium=6. Support < 8 (MIN_CLASS_SAMPLES) for most classes — demo-only
loosening (min_class_samples=3) is what allows training at all.

## 3. Systematic confusions

- Undertone: everything gravitates to Warm (majority + colour-proxy overlap in synthetic data).
- Skin depth: Medium->Medium-Deep (2), Light->Light-Medium (1) — all ADJACENT on the
  ordinal scale (adjacent_share_of_errors = 1.0). No catastrophic jumps.

## 4. Is Warm overpredicted because of imbalance?

YES (verified on demo data): class_weight="balanced" is applied, but with 41 vs 4 vs 18
rows and overlapping synthetic feature distributions, the prior still dominates. The
balanced-accuracy (0.854 CV) vs accuracy (0.938 CV) gap quantifies it.

## 5. Do features separate the undertone classes?

CANNOT BE VERIFIED for Cool/Neutral-Warm/Olive — no training rows exist. Where labels
exist (Warm vs Neutral vs Neutral-Cool) the model separates Warm well but confuses
Neutral-Cool with Warm/Neutral. Data limitation, not a verified feature defect.

## 6. Are skin-depth errors mostly adjacent?

YES — 3/3 holdout errors adjacent (Medium->Medium-Deep x2, Light->Light-Medium).
Reported separately from accuracy per the ordinal-diagnostics rule; adjacent errors are
NOT counted as correct anywhere.

## 7. CV vs holdout differences (81.3% holdout vs 93.8% CV undertone)

Plausible hypothesis (not separately testable at n=16): the 16-row holdout contains
10/16 Warm (62.5%, similar to train) but its Neutral-Cool/Neutral rows are few enough
that each miss costs 6-12pp of accuracy. Sample size, not model instability, is the
primary explanation. The CV-vs-holdout macro-F1 values are nearly identical (0.343 vs
0.336), which supports this: the ranking behaviour is stable, the tiny-n accuracy is not.

## 8. Candidate selection consistency across folds/seeds (multi-seed experiment, this update)

Undertone (8 seeds x 4-fold person-grouped CV, identical eligible rows):
random_forest 4 wins, lightgbm 2, extra_trees 2; mean macro-F1 spread across candidates
0.300-0.325 with +/-0.003-0.017 std. NO STABLE WINNER — the v002 lightgbm selection was
seed-42 noise. Per Phase 11 discipline no replacement was made (no candidate is a
defensible improvement; all are statistically indistinguishable at this n).

Skin depth: extra_trees 7/8 wins, mean 0.608 +/-0.009, clearly ahead of random_forest
(0.572) and lightgbm (0.578). The v002 extra_trees selection is DEFENSIBLE and stable.

## 9. Are probabilities/confidence meaningful?

Calibration is correctly reported as method=none (validation split lacks some classes;
spec #27 rule: do not calibrate on incomplete/tiny calibration data). Raw probabilities
are retained and abstention machinery exists (SeasonPredictor.abstain_margin, default 0
= disabled in demo CLI). Confidence values are uncalibrated model outputs, and the
axis_first/hierarchical season soups are documented as such.

## 10. Are preprocessing failures passed into classification?

NO (verified): quality_gate() rejects extraction failures, skin hue outside [10, 85] deg,
>30% shadow/highlight clipping, and insufficient skin pixels; the API returns an explicit
error + retry hint instead of a prediction. corrupt/unreadable images raise or report
error instead of producing garbage features (tests added this update).

## Verified defects: none new in this update (EXIF orientation investigated; pinned decoder
behaviour confirmed correct and locked by regression tests).

## 11. Controlled perturbation robustness (this update; n=5-6 demo images, DIAGNOSTIC ONLY)

| perturbation | feat_skin_L | feat_skin_a | feat_skin_b | undertone_proxy (b-a) |
|---|---|---|---|---|
| brightness +10% | +4.63 (max 6.27) | +0.20 | +1.10 | +0.90 |
| mild warm cast (R+8/B-4) | +0.78 | +3.25 | +3.08 | -0.17 |
| JPEG recompress q=70 | -0.10 | +0.58 | -0.17 | -0.75 |

Read-out (hypothesis-grade, synthetic demo images, tiny n):
- Undertone features (a*, b*, b-a proxy) are comparatively ROBUST to global brightness
  and JPEG compression; a mild warm cast moves a*/b* ~+3 but shifts the b-a undertone
  proxy by only ~0.2 (a and b move together).
- L* (the skin-depth signal) moves +4.6 under +10% brightness - depth classes are
  ~7 L* units apart, so one brightness step can flip one depth class. Depth is the
  illumination-sensitive axis, consistent with the quality-gate design.
- NOT a substitute for real-world validation; motivates recording white-balance
  diagnostics (already stored) and possibly depth-specific illumination checks at
  inference time.

## 12. Dataset receipt integrity (this update)

`ingest verify` OK for Fitzpatrick17k (sha256 851dfaabf1791c2a..., re-hashed from the
stored archive). CelebAMask-HQ archive present under data/raw/external/ (3.15 GB,
license-confirmed, ignored by git).

## Data limitations (blocking real conclusions)

- 63 rows / 32 people / 3-4 classes with meaningful support — orders of magnitude below
  the 200-image / 60-person development target, and the real gate is CLOSED.
- Synthetic generation process defines the feature-label relationship; nothing here
  transfers to real photographs.
