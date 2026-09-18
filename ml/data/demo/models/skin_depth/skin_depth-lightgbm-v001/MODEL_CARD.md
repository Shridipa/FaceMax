# Model card: skin_depth-lightgbm-v001

## Intended use
FaceMaxx skin_depth prediction from CV colour features. One input of a person's photo -> model output used by the product's colour analysis. Not a medical or identity tool.

## Inputs
Canonical CV features only: feat_skin_L, feat_skin_a, feat_skin_b, feat_ita, feat_undertone_proxy, feat_olive_proxy, feat_chroma_proxy, feat_depth_proxy, feat_hair_L, feat_eye_L, feat_contrast_proxy, feat_wb_b_minus_a, feat_grey_world_shift, feat_brightness
Labels, annotator votes, agreement scores and metadata are structurally excluded (LeakageError).

## Outputs
- `skin_depth` prediction

## Training data
- source: demo
- rows: 63 from 32 people
- dataset_version: 6467ba063e297f7b
- feature_version: facemax-color-features-v1

## Evaluation
- validation: person-grouped StratifiedGroupKFold (5 folds)
- CV selection metric macro_f1: best=lightgbm (0.575)
- calibration: method=none ECE=None Brier=None

## Known limitations
- trained on the annotated FaceMaxx dataset only; small classes are unstable
- lighting/makeup/quality shifts can move predictions (see robustness section)
- seasonal boundaries (e.g. Deep Autumn <-> Deep Winter) carry genuine ambiguity
- olive undertone is partly culture-dependent; labels reflect this project's handbook
- NOT a benchmark beyond the recorded holdout numbers

## Bias / robustness
not yet evaluated on the locked holdout

## Not intended for
- race, ethnicity, health, attractiveness or identity inference of any kind
- decisions about individuals beyond colour-analysis styling suggestions
