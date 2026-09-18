# FaceMaxx bias-evaluation pipeline (Python, stdlib-only)

> **Definition of bias used throughout:** *A systematic difference in prediction
> quality, confidence, or reliability across meaningful visual conditions or
> groups in the data.*
>
> So the question is never "is the dataset big?" but **"does FaceMaxx give
> similarly reliable colour-analysis results across skin depths, lighting
> conditions, hair colours, image qualities, and other appearance variations?"**

This package is the automated **Model Bias & Robustness Report** that runs after
every training/evaluation cycle. It is pure Python 3 standard library — no
numpy/pandas/sklearn required — so it runs anywhere, including CI.

```
ml/
├── facemax_bias/            FROZEN evaluation package (stdlib-only, no deps)
│   ├── vocab.py             canonical labels (skin depth, undertone, seasons, ...)
│   ├── schema.py            annotation-table validation
│   ├── metrics.py           accuracy, macro F1, balanced accuracy, confusion
│   ├── distribution.py      representation-bias analysis
│   ├── subgroups.py         subgroup performance + fairness gaps + verdicts
│   ├── labels.py            annotator agreement, ambiguity, seasonal boundaries
│   ├── robustness.py        lighting/camera/makeup/quality stability, quality gate
│   ├── leakage.py           person-level split integrity
│   ├── splits.py            person-grouped K-fold
│   ├── report.py            the fairness dashboard (text + JSON)
│   ├── recommendations.py   bias report -> remediation actions (data/model fixes,
│   │                        quality-gate only for confidence warnings)
│   └── demo.py              synthetic dataset with *planted* biases
├── facemax_color/           the colour-analysis system the evaluator scores
│   ├── dataset.py           real-data tree, extended annotation schema, registry
│   ├── colour.py            face detection -> skin mask -> LAB/ITA features
│   ├── models.py            per-axis baselines (LogReg/RF/HGB) + season combiner
│   ├── evaluate.py          person-split holdout -> predictions -> bias report
│   ├── training_data.py     canonical training-data prep (leakage-audited)
│   ├── training.py          gated supervised training + CV + calibration
│   ├── evaluate_model.py    locked-holdout evaluation + robustness + artifacts
│   ├── synth.py             synthetic face-image generator (demo only)
│   ├── demo.py              end-to-end demo orchestration
│   └── audit.py             real-data quality audit + split manifests
├── configs/                 color_training.json (seed/folds/thresholds/rule)
├── tests/                   unit tests (stdlib unittest)
├── docs/                    ANNOTATION_HANDBOOK.md, ML_TRAINING.md, MODEL_CARD.md
└── data/                    dataset tree (see "Real-data dataset foundation")
```

## Quick start

```bash
cd ml

python -m facemax_bias selfcheck         # evaluator smoke test
python -m unittest discover tests        # full unit tests (both packages)

# Generate the synthetic demo dataset (2,000 images, 1,000 people) + report:
python -m facemax_bias demo --out data --people 1000

# Report from your own annotation table (CSV or JSONL):
python -m facemax_bias report path/to/annotations.csv --out data/report.txt

# Or from Python:
#   from facemax_bias.report import generate_report
#   report = generate_report(rows)
#   print(report.text)
#   report.save_json("report.json")
```

## Colour-extraction + baseline models (`facemax_color`)

The real colour-analysis pipeline. Uses numpy/OpenCV/sklearn when available,
with graceful fallbacks (heuristic labelling) when not.

```bash
cd ml

python -m facemax_color init-data        # create dataset tree + annotation template
# (human) fill metadata/facemaxx_color_annotations.csv, drop images into data/raw/
python -m facemax_color validate         # per-row lint (required fields, vocab, ranges)
python -m facemax_color extract          # face detection -> skin LAB/ITA features
python -m facemax_color audit --splits   # data-quality audit + person-grouped manifests
python -m facemax_color ready            # data-readiness audit (gate input)
python -m facemax_color prepare          # training-data prep report (exclusions + class balance)
python -m facemax_color train            # GATED supervised training (NOT READY blocks)
python -m facemax_color evaluate --model <model_id>   # LOCKED holdout, scored once

# End-to-end synthetic demo (no real data needed):
python -m facemax_color demo-image --people 40 --images 2
```

### Real-data acquisition + annotation workflow

The human pipeline from camera to evaluable labels (full guide:
`docs/ANNOTATION_HANDBOOK.md`):

```bash
python -m facemax_color annot manifest                        # capture-manifest template
# ...shoot per the handbook protocol; fill the manifest while shooting...
python -m facemax_color annot import --manifest data/annotations/capture_manifest.csv --staging photos/
python -m facemax_color annot queue --annotator alice --ids-file ids.txt
# ...each annotator fills their round file INDEPENDENTLY...
python -m facemax_color annot record --round-file data/annotations/queues/round-alice.csv
python -m facemax_color annot merge        # majority season + agreement; split votes stay unmerged
python -m facemax_color annot coverage     # progress vs. handbook targets
python -m facemax_color annot pending      # images awaiting annotation
```

Mechanically enforced: sha256 content-addressed ids (identical files collapse;
people are never double-counted), per-annotator independence, vocabulary
rejection with suggestions (never auto-fixed), missing labels stay pending
(never fabricated), axis labels merged only on full annotator agreement,
Deep-Autumn/Deep-Winter boundary candidates tracked.

### The ML-training gate

```bash
python -m facemax_color gate   # OPEN/CLOSED + explicit reasons
```

`gate` is the single checkpoint before any model training. It reports OPEN
only when: enough annotated images from enough distinct people exist, every
row passes `validate`, the full `audit` passes, and the holdout is locked and
verified. Until then, training is explicitly out of scope.

### Data-readiness audit (the last checkpoint before training)

```bash
python -m facemax_color ready  # NOT READY / SMOKE-READY / TRAIN-READY + reasons
npm run colour:ready           # same, from the repo root
```

`ready` answers - with numbers - whether the annotated dataset is actually
usable for supervised learning. It re-verifies every gate condition, then
adds the training-level analyses:

- **season/depth/axis coverage matrices** — all 12 seasons, all 8 skin depths,
  every undertone/chroma/contrast label, each flagged MISSING or
  under-represented against explicit minimums;
- **annotator disagreement, classified** — disagreement between seasonal
  wheel *neighbours* (differ on one axis, e.g. Deep Autumn <-> Deep Winter)
  is a BOUNDARY signal (expected, informative); disagreement between unrelated
  seasons is a CONFLICT (label poison: re-annotate or retire the row);
- **capture-condition stability** — dissent rate per lighting/camera/quality/
  makeup bucket; a bucket with enough rows and >= 25% dissent is flagged
  UNSTABLE (collect better data or control the condition);
- **person grouping & consistency** — multi-image people counted, people whose
  photos carry conflicting merged seasons listed (annotation error or id error),
  rows with missing person_id counted;
- **holdout untouchedness** — the locked manifest is digest-verified, and every
  test-split image must still exist in the annotation table *with* season truth
  (a deleted or unlabelled holdout row blocks readiness);
- **train gaps** — a SMOKE-READY dataset passes the gate but lists exactly
  what to collect next; TRAIN-READY additionally requires all 12 seasons,
  every depth/axis label, >= 0.80 mean agreement on mostly multi-annotated
  rows, zero conflicts, consistent people, and no unstable conditions.

Exit codes: `0` TRAIN-READY, `1` NOT READY, `2` SMOKE-READY (safe to smoke-test
the training *wiring*, never to quote a benchmark). Reports are written to
`data/reports/readiness_report.txt` + `.json`.

```bash
python -m facemax_color ready --root data --out data/reports
```

### External dataset ingestion (license-gated, traceable)

```bash
python -m facemax_color ingest plan      # official sources + license gates + status
# human: review official terms -> complete the registry license field ->
#        download from the OFFICIAL source themselves (no scraping)
python -m facemax_color ingest record --dataset CelebAMask-HQ --file ~/Downloads/celeba.zip --license-confirmed
python -m facemax_color ingest verify --dataset CelebAMask-HQ --rehash
```

- `ingest record` refuses to run unless the dataset's registry block has a
  completed license field AND `--license-confirmed` is passed (a human
  certifies the official terms were reviewed). It computes sha256 and writes
  `data/ingest/<name>.receipt.json` + updates `data/ingest/sources.json`.
- `ingest verify --rehash` re-hashes the archive; any post-receipt change
  fails verification.
- **No external dataset is ever seasonal ground truth** — CelebAMask-HQ for
  parsing masks, FFHQ for preprocessing robustness, Fitzpatrick17k for
  skin-depth evaluation only. Season labels come only from the FaceMaxx
  custom annotated set.

### Per-row annotation validation

```bash
python -m facemax_color validate              # text report, exit 1 on invalid rows
# -> data/reports/row_lint.json (machine-readable, for the annotation UI)
```

Checks per row: required fields (`image_id`, `person_id`, `season_primary`,
`undertone`, `skin_depth`), controlled vocabularies (with "did you mean"
suggestions for typos like `Wram` -> `Warm`), numeric ranges (LAB, 0-1
confidences), and draping-colour names in `best_colors`/`neutral_colors`/
`caution_colors`.

`demo-image` proves the full wiring on synthetic Haar-detectable faces:

```
DATA STATUS: SYNTHETIC DEMO - not a real-world fairness benchmark.
MODEL STATUS: DEMO/SYNTHETIC - trained on generated images.
```

- Generated images live ONLY under `data/demo/raw/`; annotations under
  `data/demo/annotations.csv`. The real `data/raw/` and
  `metadata/facemaxx_color_annotations.csv` are never touched
  (enforced by `test_demo_data_isolated_from_real_data`).
- If the Haar detector refuses a synthetic face, extraction falls back to a
  known-geometry box, logged as `DEMO_FALLBACK_FACE_BOX` and recorded in
  `data/demo/fallback_boxes.json`.
- Demo numbers describe pipeline wiring, NOT model quality. Production
  fairness evaluation requires real, person-grouped, annotated data.

Extraction outputs the feature vector the models consume (all measured from
pixels, never hard-coded): median/mean skin `L*`, `a*`, `b*`, ITA°,
undertone/olive/chroma/depth proxies, hair `L*`, eye `L*`, hair-vs-skin
contrast, white-balance shift, brightness/exposure flag, skin-pixel count.

The generated sample report lives at `ml/data/sample_report.txt` (JSON twin in
`sample_report.json`).

## What the report checks (and why)

| # | Check | Question it answers |
|---|-------|---------------------|
| 1 | **Representation** | Which skin depths / undertones / lighting / cameras / sources are missing or rare? |
| 2 | **Subgroup performance** | Does accuracy hold across every visual group, or only on average? Reports worst-vs-best gap + verdict (`strong` / `needs investigation` / `serious problem`). |
| 3 | **Confusion structure** | Are errors confined to adjacent seasons (Deep Autumn ↔ Deep Winter = OK) or jumping the palette (Deep Autumn → Light Summer = red flag)? |
| 4 | **Label quality** | How often do 3 annotators agree? Which images are inherently ambiguous? Is the headline accuracy sensitive to the choice of ground truth? |
| 5 | **Robustness** | Does the same person keep the same season across lighting/camera/makeup conditions? Per-condition accuracy + top-2 stability. |
| 6 | **Quality gate** | Below which image-quality bucket should the app show "results may be inaccurate" instead of a firm season? |
| 7 | **Split integrity** | Are all images of one person confined to a single split (no train/test leakage)? |

## Annotation schema (the full spec)

```
image_id, person_id, source, license,
image_quality, lighting_type, camera_type, makeup,
skin_depth, skin_L, skin_a, skin_b,
undertone, undertone_confidence, olive_score,
hair_depth, hair_temperature, eye_depth,
contrast_score, chroma_score,
season_primary, season_secondary, season_confidence,
annotator_1, annotator_2, annotator_3, annotation_agreement,
split
```

Rules baked into the pipeline:

- **`person_id` is never a model feature.** It exists solely to group images of
  the same human so all of them land in one split (`grouped_split`,
  `group_kfold`) and so leakage is detected (`leakage_check`).
- **Ambiguity is preserved, not crushed.** Images where annotators disagree are
  flagged, and `season_secondary` + `majority_label` keep the boundary
  information (Deep Autumn 51% / Deep Winter 47% stays a boundary, not a false
  certainty). The report lists the most common seasonal boundaries.
- Every record needs at minimum `image_id`, `person_id`, `season_primary`;
  `schema.validate()` checks the rest against the canonical vocabularies in
  `vocab.py` (free-form lighting/quality labels are auto-normalized).

## Acceptance criteria (engineering targets, not scientific constants)

| Verdict | Per-subgroup performance | Worst-vs-best gap |
|---|---|---|
| **Strong** | ≥ 80% | ≤ 10 pts |
| **Needs investigation** | 70–80% | 10–15 pts |
| **Serious problem** | < 70% | > 15 pts |

Tune `STRONG_MIN_RATE`, `STRONG_MAX_GAP`, etc. in `subgroups.py` as the project
matures. Subgroups with n < 30 are still reported but flagged `(small n)`.

## Interpreting the demo report

`ml/data/sample_report.txt` is generated from synthetic data with **deliberately
planted biases** so you can see what the dashboard looks like when things go
wrong:

- Very Deep skin: 1.5% of images → flagged under-represented
- Accuracy monotonically drops from Very Fair → Very Deep (24.5-pt gap,
  "serious problem")
- Poor/Very Poor quality buckets collapse (47-pt gap) → quality gate fires
- Warm/cool indoor light shifts predictions warm/cool → visible in lighting
  robustness and in the Deep Autumn ↔ Deep Winter confusion pair
- 9.6% of images have annotator disagreement, concentrated on the
  Deep Autumn / Deep Winter boundary

**Overall accuracy is 62.5% — but the per-subgroup sections are where the real
story is.** That is exactly the point of the pipeline.

## Real-data dataset foundation

Canonical tree (created by `init-data` / `audit`):

```
data/
├── raw/            untouched original images (never edit in place)
├── interim/        detection crops, masks, white-balance intermediates
├── processed/      extracted colour-feature tables
├── metadata/       facemaxx_color_annotations.csv  <- the human annotation table
├── annotations/    harmony_scores.csv (per-colour draping scores, long format)
├── splits/         train.txt / val.txt / test.txt + person_map.json (freeze!)
├── demo/           SYNTHETIC demo artifacts - never mixed with real data
└── reports/        data_quality_report.txt, bias reports
```

- `DATASET_REGISTRY.md` tracks every external dataset (CelebAMask-HQ for face
  parsing masks only — **non-commercial research only**; FFHQ for
  preprocessing/robustness — research use, per-image Flickr licensing;
  Fitzpatrick17k for skin-depth robustness evaluation only — CC BY-NC-SA 3.0;
  the two project-owned primary sets). License facts are pre-filled with
  `verified:` citations to the official pages; the `human-check:` items are
  the human's approval act. `audit` lints the registry — no dataset enters
  `raw/` without a completed block, and no external dataset is ever labelled
  with 12-season colour truth.
- The extended annotation schema adds traceability and harmony targets
  (`image_path`, `quality_score`, `white_balance_quality`, `hair_color`,
  `hair_warmth`, `eye_color`, `contrast_level`, `temperature_axis`,
  `best_colors`, `neutral_colors`, `caution_colors`, `agreement_score`,
  `notes`, ...) on top of the frozen evaluator schema. No race/ethnicity
  columns exist anywhere — skin tone is never a proxy for race.
- `audit` validates schema, leakage, class/subgroup distributions, annotator
  agreement, file-layer integrity (missing raw images, byte-identical
  duplicate files), row-lint status, holdout-lock state and ingestion
  receipts; rows with missing season truth are counted, never fabricated.
  `--splits` writes person-grouped train/val/test manifests (one person =
  one split, via the frozen evaluator's `grouped_split`) that later training
  runs must REUSE.

### Locked holdout

```bash
python -m facemax_color audit --regen-splits          # split + FREEZE (write holdout.lock.json)
python -m facemax_color audit --regen-splits --force  # explicitly break an existing lock
```

`audit --regen-splits` re-splits person-grouped, refuses to lock a leaked
split, then writes `data/splits/holdout.lock.json` containing sha256 digests
of every manifest. Every later `audit` re-verifies: modified or deleted
manifests report `LOCK BROKEN`, and re-splitting without `--force` is
refused. The locked test split is the fairness holdout: training/tuning
touches only train/val, forever.
- Colour-harmony ground truth is per-person, per-colour scores (1–5) for the
  18 draping colours (burgundy ... plum) in `annotations/harmony_scores.csv`;
  `audit` warns when a person's colour coverage is incomplete.

## Supervised ML training (the gated pipeline)

Full documentation: `docs/ML_TRAINING.md` (pipeline, safeguards, config,
secondary-season rule). Pipeline (stages never collapsed):

```text
ready (gate) -> prepare -> person-grouped CV over LogReg/RF/XGBoost/LightGBM
  -> model selection (macro-F1) -> calibration (val-only; sigmoid/isotonic)
  -> artifact (metadata/metrics/card) -> evaluate on the LOCKED holdout
  -> robustness via the frozen evaluator -> model card refreshed
```

Key properties, all mechanically enforced:

- **The readiness gate decides.** `train` on NOT READY data prints
  `ML TRAINING BLOCKED`, the blockers, and `No real-data model was trained.`
  SMOKE-READY data needs explicit `--smoke` (artifacts marked `"smoke": true`);
  only TRAIN-READY data produces real candidates.
- **Feature leakage is structurally impossible.** The feature matrix accepts
  ONLY canonical `feat_*` CV columns; any label/annotator/metadata column
  raises `LeakageError` before a model sees data.
- **Person grouping end-to-end.** Splits come from the frozen manifests;
  CV is StratifiedGroupKFold on `person_id` and fold records carry full
  person-id lists so disjointness is provable; the locked holdout is carried
  separately and excluded from training.
- **Every exclusion is reported** (missing feature row, invalid label,
  duplicate id, missing person, locked holdout row) — nothing is dropped
  silently; class balance reports images AND people per class.
- **Rare classes refuse to train** (`DATA INSUFFICIENT`) instead of producing
  unstable models; fix with data, not lowered thresholds.
- **Season approaches compared objectively** on the same folds: direct 12-class,
  axis-first rule combiner (existing `models.py`), hierarchical
  axes-then-head. The winner is measured, not assumed.
- **Calibration is honest.** Fitted on validation only; every-class-coverage
  required; ECE + Brier scored on the holdout by `evaluate`. Secondary season
  follows the documented margin rule (`secondary_rule.secondary_min_prob`).
- **Boundary errors are separated**: Deep Autumn→Deep Winter (wheel neighbours,
  genuine ambiguity) is reported differently from Deep Autumn→Light Summer
  (label/feature poison).
- **Demo training is isolated**: `train --demo` reads/writes only `data/demo/`,
  marks every artifact `smoke`, and every demo report carries the
  `NOT A FACEMAXX PERFORMANCE BENCHMARK` banner. `evaluate` refuses to mix
  demo artifacts with real data.

```bash
npm run colour:prepare       # training-data prep report
npm run colour:train         # gated training (uses configs/color_training.json)
npm run colour:evaluate      # locked-holdout evaluation (--model <id>)
npm run colour:train-demo    # isolated synthetic wiring test
```

Artifacts land in `data/models/<target>/<model_id>/` (demo ones under
`data/demo/models/`) with `model.joblib`, `metadata.json`, `metrics.json`,
`feature_schema.json`, `confusion_matrix.json`, `holdout_report.txt` and a
generated `MODEL_CARD.md` (format: `docs/MODEL_CARD.md`). No anonymous `.pkl`.

## Locking the holdout

The spec's gold standard: a locked, never-trained-on evaluation set. Use
`grouped_split(rows, test_frac=...)` once, freeze the split column, and treat
the test people as the holdout. Then track report-over-report:

```
Model V1 -> holdout -> 78%
Model V2 -> holdout -> 83%
```

and compare the *subgroup sections*, not just the headline.
