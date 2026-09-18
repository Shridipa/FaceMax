# FaceMaxx real-data dataset

- `raw/` original images, read-only by convention
- `interim/` face crops, masks, white-balance intermediates
- `processed/` extracted colour-feature tables (parquet/csv)
- `metadata/facemaxx_color_annotations.csv` human labels (template provided)

Fill one row per image. season_secondary/annotator_* may be empty early on; the evaluator treats missing annotator columns as single-label ground truth. person_id groups images of the same human and is NEVER a model feature.
