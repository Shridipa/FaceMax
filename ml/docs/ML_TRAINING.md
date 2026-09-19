# FaceMaxx ML Training

How validated real annotations become trained, calibrated, bias-audited model
artifacts — and every safeguard the pipeline enforces mechanically.

## The pipeline (stages are never collapsed)

```text
Image
 ↓
CV colour extraction            facemax_color.colour (canonical feat_* columns)
 ↓
Validated feature vector        processed/colour_features.csv
 ↓
READINESS GATE                  facemax_color.ready  (NOT READY blocks training)
 ↓
Person-grouped dataset          training_data.py (manifest-driven, exclusions reported)
 ↓
Baseline models                 LogReg | RandomForest | XGBoost | LightGBM
 ↓
Person-grouped cross-validation StratifiedGroupKFold (fold stats recorded)
 ↓
Model selection                 macro-F1 (never plain accuracy)
 ↓
Calibration                     sigmoid/isotonic, fitted on VALIDATION only
 ↓
Robustness evaluation           facemax_bias subgroup machinery (frozen)
 ↓
Model artifact                  data/models/<target>/<model_id>/
 ↓
LOCKED HOLDOUT                  evaluate (scored once, never tuned against)
 ↓
Approved model                  metrics.json + MODEL_CARD.md updated
```

## Gate contract (do not weaken)

`train` refuses real data unless the readiness audit says otherwise:

| Readiness status | `train` behaviour |
|---|---|
| `NOT READY` | refused, exit 1, blockers printed, **no model trained** |
| `SMOKE-READY` | refused unless `--smoke`; smoke artifacts are marked `"smoke": true` |
| `TRAIN-READY` | allowed; artifacts are real candidates |

`--demo` is the only way to train on synthetic data; it reads and writes
**only** `data/demo/` and its artifacts are explicitly labelled
`DATA STATUS: SYNTHETIC DEMO - NOT A FACEMAXX PERFORMANCE BENCHMARK`.
The demo tree gets its own person-grouped split manifests (same frozen
`grouped_split` logic) so the manifest-driven flow is identical.

`evaluate` refuses to score a demo artifact against real data (or vice versa)
and refuses any artifact whose target differs from the prepared data.

## Training-data preparation (`prepare`)

`prepare_training_data` joins annotations to features by `image_id` and
**reports every exclusion** (never silently drops):

```text
Training data preparation
==================================
target:                    season (column: season_primary)
total annotated rows:      842
rows with valid features:  831
rows excluded:             11

excluded (each exclusion is reported, never silent):
  locked_holdout_row             4  e.g. ['IMG0012', ...]
  invalid_season_primary_label   3  e.g. ['IMG0044:Bogus Season']
  missing_person_id              2  e.g. ['IMG0101']
  duplicate_image_id             2  e.g. ['IMG0208']
```

Rules enforced here:

- **locked holdout exclusion** — test-split rows (from the frozen split
  manifests) are carried separately for `evaluate` and can never enter
  training matrices;
- **manifest-driven splits** — train/val/test come from `data/splits/*.txt`
  (person-grouped, written once by `audit --splits/--regen-splits`); training
  code never re-splits real data;
- **person_id mandatory** — rows without a person are excluded (person-grouped
  evaluation is the anti-leakage backbone);
- **canonical labels only** — labels outside the frozen vocabularies
  (`facemax_bias.vocab`) are excluded and listed;
- **class balance report** — per class: images AND people (resampling never
  manufactures people); imbalance is handled with class weights, not
  duplication.

## Feature leakage audit

The feature matrix may contain ONLY the canonical `feat_*` columns produced by
`colour.extract_image().as_feature_row()`. Any other column — season labels,
axis labels, annotator votes, agreement scores, metadata — raises
`LeakageError` before any model sees data:

```text
LeakageError: feature leakage: these columns are NOT canonical CV features and
must never be model inputs: ['season_primary', 'annotator_1'] ...
```

This is structural: there is no code path that can feed labels to a model.

## Baselines and validation

Four tabular candidates per target (all present in the environment): 
`LogisticRegression`, `RandomForestClassifier`, `XGBoost` (string-label
adapter), `LightGBM`. Selection metric: **macro-F1** (config:
`selection_metric`). Validation: `StratifiedGroupKFold` on `person_id` —
every fold records images/people/class distribution **and full person-id
lists**, so fold-level person disjointness is provable, not assumed.
Classes below `min_class_samples` (default 5) are refused loudly:

```text
CV SKIPPED: classes below min_class_samples=5: Cool Winter=4 - collect more
data instead of training on unstable classes
DATA INSUFFICIENT for supervised training of this target
```

That is a successful, honest outcome — fix it with data, not thresholds.

## Season model: three approaches compared objectively

The 12-season target is trained three ways on the SAME person-grouped folds:

1. **direct** — features → 12 seasons (the baseline candidates);
2. **axis_first** — the existing `ColourModelBundle` axis models + rule
   combiner (`models.py`, unchanged);
3. **hierarchical** — stage-1 axis models → season head on
   [features | axis one-hots].

CV macro-F1 decides; the winner and all three scores are recorded in the
run manifest and metrics. No approach is assumed better — measured.

## Metrics

Every evaluation reports accuracy, balanced accuracy, macro precision/recall/
F1, weighted F1, per-class precision/recall/F1/support, and the confusion
matrix. Season models additionally get **boundary analysis** (see below).
Holdout evaluations add **top-2 accuracy** and the **abstain rate**.

### Metric persistence (do not regress)

Everything `compute_metrics()` computes is persisted: every CV candidate
records per-candidate `mean_accuracy`, `mean_balanced_accuracy`,
`mean_macro_precision`, `mean_macro_recall`, `mean_weighted_f1`,
`per_class_mean` (precision/recall/f1 per class) and `confusion_matrix_sum`
alongside the fold table — not just the selection metric. The holdout block
adds `top2_accuracy` and abstention counts.

## Secondary-season rule (documented, configurable)

```json
"secondary_rule": { "secondary_min_prob": 0.10, "abstain_margin": 0.0 }
```

> Primary = highest calibrated probability.
> Secondary = highest-probability season **distinct from the primary**,
> emitted only when its probability ≥ `secondary_min_prob`
> (otherwise no secondary is reported at all).

Do not treat the second-highest raw softmax as a secondary season; the rule
and threshold live in `configs/color_training.json` and every artifact
records the threshold it used.

### Abstention (uncertainty is surfaced, never faked)

When `abstain_margin > 0` and the calibrated top-2 probability gap falls
below the margin, `predict_row` returns `"abstained": true` with an **empty
primary** and the runner-up as the honest alternative ("between Deep Autumn
and Deep Winter"). Abstained rows count as errors in accuracy (never hidden)
but are reported separately as `abstain_rate` in the holdout metrics.
Keep the margin at 0 until a threshold is validated on real data.

## Selection evidence + feature audit + ablation

Every run persists (and `train` prints):

- **selection_reason** — winner, runner-up, CV margin, metric used;
- **feature_audit.json** — per feature: missing %, min/max/mean/std, invalid
  count, and flags for `constant` / `near_constant` / `heavily_missing`
  columns (reported, never silently dropped);
- **ablation** — the winner re-scored on the SAME folds with cumulative
  feature groups (skin_core → +hair_eye → +region_variability →
  +quality_lighting → full). Diagnostic only: the full set stays deployed
  unless a variant's advantage is judged meaningful.

`evaluate` additionally writes **error_analysis.json** (worst classes, worst
subgroups, largest confusion pairs) and refreshes the model card.

## Candidate registry (one authoritative comparison)

`models` in the config selects from: `logreg`, `random_forest`,
`extra_trees`, `xgboost`, `lightgbm`, `hist_gb` — all scored on identical
folds/features/seed. `hist_gb` is sklearn's HistGradientBoostingClassifier:
the same family as the historical `hist_gb = 0.725` line in old
`model_comparison.txt` artifacts, which came from an earlier dataset/feature
generation and is NOT comparable to current runs. Every comparison output now
starts with a provenance header (dataset rows/people + feature version + a
"not comparable across provenance" warning) so stale numbers can never be
read as one experiment again.

## Feature schema versioning

`FEATURE_VERSION = "facemax-color-features-v2"` adds `feat_skin_hue`,
`feat_skin_region_count`, `feat_skin_L_spread`, `feat_skin_a_spread`,
`feat_skin_b_spread`, `feat_highlight_clip_frac`, `feat_shadow_clip_frac`
on top of the unchanged v1 block. `feature_schema.json` is persisted with
every artifact and `predict_row`/`verify_feature_schema` refuses to predict
when the input features do not carry the model's columns (loud failure
instead of silent garbage).

## Quality gate at inference

`models.quality_gate()` rejects images whose extraction is dominated by
cast/mask/clipping failure (skin hue outside the plausible human band,
>30% shadow or highlight clipping, too few skin pixels) with
"Photo quality is too low for a reliable analysis." — a warning is more
honest than a confident wrong season (the old demo low-light = 0.0 result).

## Calibration

- Fitted on the **validation split only** (never the holdout, never train).
- `auto`: isotonic when ≥ `isotonic_min_samples` (default 1000) calibration
  rows, else sigmoid (Platt) — small calibration sets make isotonic overfit.
- Implementation uses a frozen base model + a single precomputed split
  (prefit semantics) so tiny validation splits cannot silently trigger
  sklearn's internal 5-fold stratification.
- Calibration requires every model class to appear in the calibration data;
  otherwise raw probabilities are kept and the gap is reported.
- Scores: **ECE** and **Brier**, computed on the LOCKED HOLDOUT (the honest
  test of the calibrator) by `evaluate`.

## Seasonal boundary analysis

Errors are classified by the seasonal-wheel neighbour table from the
readiness audit:

- `Deep Autumn → Deep Winter` — **boundary** error (neighbours; one axis apart;
  genuine ambiguity at the seasonal edge, informs handbook sharpening);
- `Deep Autumn → Light Summer` — **non-boundary** error (label or feature
  poison; investigate individually).

`evaluate` reports correct / boundary / non-boundary counts, the boundary
share of errors, and the top confusion pairs. Boundary confusion is never
called random error.

## Robustness / bias (frozen evaluator)

Holdout predictions are pushed through `facemax_bias.subgroups` (verdicts,
thresholds, `GROUP_SIZE_WARN` untouched) across `skin_depth`, `lighting_type`,
`camera_type`, `makeup`, `image_quality`, `source`. Each subgroup reports
`n_images`, `n_people`, accuracy, balanced accuracy, macro-F1, and a
`small_n` flag — small subgroups are **flagged, never hidden**, and their
metrics are never presented as definitive. No race/ethnicity inference exists
anywhere in the pipeline; skin depth is never a proxy for anything else.

## Model artifacts

```text
data/models/
  season/
    season-hierarchical_random_forest-v001/
      model.joblib          the loaded SeasonPredictor
      metadata.json         model_id, type, target, feature/dataset version,
                            timestamp, rows/people, validation strategy,
                            hyperparameters, seed, git commit, smoke flag
      metrics.json          CV + calibration + holdout + robustness
      feature_schema.json   canonical feature columns + leakage-audit note
      confusion_matrix.json holdout confusion matrix
      MODEL_CARD.md         intended use, inputs/outputs, training data,
                            evaluation, limitations, robustness, not-intended-for
      holdout_report.txt    the final evaluation report
```

No anonymous `.pkl`; every artifact is reproducible from its metadata
(seed, config, dataset digest `dataset_version`, `feature_version`).
Model binaries are not committed to Git.

## CLI

```bash
python -m facemax_color prepare  --targets season,undertone,... [--demo]
python -m facemax_color train    --targets ... [--config configs/color_training.json] \
                                 [--seed N] [--folds N] [--smoke] [--demo]
python -m facemax_color evaluate --model <model_id> [--demo]
python -m facemax_color ready    # the gate input
python -m facemax_color gate     # the ML-training gate summary
```

npm: `colour:prepare`, `colour:train`, `colour:evaluate`, `colour:ready`,
`colour:gate`.

## Configuration

`configs/color_training.json` (deep-merged over built-in defaults):

```json
{
  "random_seed": 42,
  "cv_folds": 5,
  "min_class_samples": 5,
  "selection_metric": "macro_f1",
  "class_weight": "balanced",
  "n_jobs": 1,
  "models": ["logreg", "random_forest", "xgboost", "lightgbm"],
  "season_approaches": ["direct", "axis_first", "hierarchical"],
  "hierarchical_axis_backend": "random_forest",
  "calibration": { "method": "auto", "isotonic_min_samples": 1000, "bins": 15 },
  "secondary_rule": { "secondary_min_prob": 0.10 }
}
```

Real-data training never lowers `min_class_samples`; only the isolated demo
wiring test does (and says so).

## Data governance (unchanged by this milestone)

No label rewriting, no threshold lowering, no season merging, no synthetic
labels for real images, no race/ethnicity inference, no silent drops, no
tuning against the locked holdout. When data is insufficient the pipeline
prints `DATA INSUFFICIENT` — the remedy is more annotated data, collected
through the annotation workflow.
