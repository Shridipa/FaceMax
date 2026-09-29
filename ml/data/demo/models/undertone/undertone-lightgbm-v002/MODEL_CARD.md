# Model card: undertone-lightgbm-v002

## Intended use
FaceMaxx undertone prediction from CV colour features. One input of a person's photo -> model output used by the product's colour analysis. Not a medical or identity tool.

## Inputs
Canonical CV features only: feat_skin_L, feat_skin_a, feat_skin_b, feat_ita, feat_undertone_proxy, feat_olive_proxy, feat_chroma_proxy, feat_depth_proxy, feat_skin_hue, feat_hair_L, feat_eye_L, feat_contrast_proxy, feat_skin_region_count, feat_skin_L_spread, feat_skin_a_spread, feat_skin_b_spread, feat_wb_b_minus_a, feat_grey_world_shift, feat_brightness, feat_highlight_clip_frac, feat_shadow_clip_frac
Labels, annotator votes, agreement scores and metadata are structurally excluded (LeakageError).

## Outputs
- `undertone` prediction

## Training data
- source: demo
- rows: 63 from 32 people
- dataset_version: 4aff0f893ce2caf6
- feature_version: facemax-color-features-v2

## Evaluation
- validation: person-grouped StratifiedGroupKFold (5 folds)
- CV selection metric macro_f1: best=lightgbm (0.343)
- selection: lightgbm chosen over runner-up random_forest (margin 0.0074915824915824936)
- feature-group ablation (same folds/backend, macro-F1):
    full                         0.343
    skin_core                    0.317
    skin_core+hair_eye           0.309
    skin_core+hair_eye+region_variability 0.361
    skin_core+hair_eye+region_variability+quality_lighting 0.343
- locked holdout: accuracy=0.812 macro_f1=0.337 balanced_accuracy=0.750 n=16

### Holdout per-class metrics

| label | support | precision | recall | f1 |
|---|---|---|---|---|
| Cool | 0 | 0.000 | 0.000 | 0.000 |
| Neutral-Cool | 4 | 1.000 | 0.250 | 0.400 |
| Neutral | 2 | 0.500 | 1.000 | 0.667 |
| Neutral-Warm | 0 | 0.000 | 0.000 | 0.000 |
| Warm | 10 | 0.909 | 1.000 | 0.952 |
| Olive | 0 | 0.000 | 0.000 | 0.000 |
- warning: classes with zero support (metrics for them are vacuous): Cool, Neutral-Warm, Olive
- warning: classes with support < 5 (unstable metrics): Neutral-Cool=4, Neutral=2
- calibration: method=none ECE=0.1888 Brier=0.3765
  (validation split does not contain classes ['Neutral-Cool']; raw probabilities kept (calibration requires every class to appear in the calibration data))

## Known limitations
- **SYNTHETIC DEMO / SMOKE TEST ARTIFACT: trained on synthetic demo data. NOT A FACEMAXX PERFORMANCE BENCHMARK. NOT FOR PRODUCTION.**
- trained on the annotated FaceMaxx dataset only; small classes are unstable
- lighting/makeup/quality shifts can move predictions (see robustness section)
- seasonal boundaries (e.g. Deep Autumn <-> Deep Winter) carry genuine ambiguity
- olive undertone is partly culture-dependent; labels reflect this project's handbook
- NOT a benchmark beyond the recorded holdout numbers
- flagged features (see feature_audit.json): feat_highlight_clip_frac, feat_shadow_clip_frac, feat_skin_region_count

## Bias / robustness
```
skin_depth: worst=None (None) best=None (None) gap=None
lighting_type: worst=None (None) best=None (None) gap=None
camera_type: worst=None (None) best=None (None) gap=None
makeup: worst=None (None) best=None (None) gap=None
image_quality: worst=None (None) best=None (None) gap=None
source: worst=None (None) best=None (None) gap=None
```

## Not intended for
- race, ethnicity, health, attractiveness or identity inference of any kind
- decisions about individuals beyond colour-analysis styling suggestions
