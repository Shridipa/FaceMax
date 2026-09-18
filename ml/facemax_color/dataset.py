"""FaceMaxx real-data dataset layout.

    data/
    ├── raw/                      untouched original images (never edit in place)
    ├── interim/                  detection crops, masks, intermediate arrays
    ├── processed/                model-ready feature tables
    └── metadata/
        └── facemaxx_color_annotations.csv   the human annotation table

Annotation columns follow the fairness spec exactly (see ml/README.md). The
template is created with ``init_dataset`` and must stay in sync with
``facemax_bias.vocab.COLUMNS`` (validated by the tests).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

from facemax_bias.vocab import COLUMNS

ANNOTATIONS_FILENAME = "facemaxx_color_annotations.csv"
HARMONY_FILENAME = "harmony_scores.csv"
REGISTRY_FILENAME = "DATASET_REGISTRY.md"

# Canonical data tree. metadata/ holds the primary annotation CSV (kept for
# backward compatibility with earlier scaffolding); annotations/ holds harmony
# scores and free-form notes; splits/ holds person-grouped split manifests.
DATA_SUBDIRS = (
    "raw",
    "interim",
    "processed",
    "metadata",
    "annotations",
    "splits",
    "demo",
    "reports",
)

# Extended annotation schema (fairness spec columns + real-data additions:
# traceability, white-balance quality, colour-harmony targets, notes).
# The frozen facemax_bias evaluator never depended on the exact column list;
# these extras ride along in row dicts and are ignored where irrelevant.
EXTENDED_COLUMNS = tuple(COLUMNS) + (
    "image_path",
    "quality_score",
    "white_balance_quality",
    "hair_color",
    "hair_warmth",
    "eye_color",
    "eye_depth",
    "contrast_level",
    "chroma_level",
    "temperature_axis",
    "depth_axis",
    "chroma_axis",
    "best_colors",
    "neutral_colors",
    "caution_colors",
    "agreement_score",
    "notes",
)

# Colour-harmony draping targets: per-colour compatibility scores (1-5) are
# annotated per person and stored long-format in annotations/harmony_scores.csv.
HARMONY_COLOURS = (
    "burgundy", "red", "orange", "yellow", "olive", "emerald",
    "forest_green", "teal", "cobalt", "lavender", "pink", "black",
    "white", "cream", "navy", "brown", "rust", "plum",
)

HARMONY_COLUMNS = (
    "person_id",
    "colour",
    "score_1_to_5",
    "annotator",
    "notes",
)

TEMPLATE_NOTE = (
    "Fill one row per image. season_secondary/annotator_* may be empty early on; "
    "the evaluator treats missing annotator columns as single-label ground truth. "
    "person_id groups images of the same human and is NEVER a model feature."
)


def init_dataset(root: Path) -> Path:
    """Create the canonical dataset tree + annotation template. Idempotent."""
    root = Path(root)
    for sub in DATA_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    annotations = root / "metadata" / ANNOTATIONS_FILENAME
    if not annotations.exists():
        write_annotations(annotations, [])
    write_dataset_registry(root / REGISTRY_FILENAME)
    harmony = root / "annotations" / HARMONY_FILENAME
    if not harmony.exists():
        write_harmony_scores(harmony, [])
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# FaceMaxx real-data dataset\n\n"
            "- `raw/` original images, read-only by convention\n"
            "- `interim/` face crops, masks, white-balance intermediates\n"
            "- `processed/` extracted colour-feature tables (parquet/csv)\n"
            f"- `metadata/{ANNOTATIONS_FILENAME}` human labels (extended schema template)\n"
            f"- `annotations/{HARMONY_FILENAME}` per-colour draping scores (long format)\n"
            f"- `splits/` person-grouped train/val/test manifests (write once, freeze)\n"
            "- `demo/` SYNTHETIC demo artifacts - never mixed with real data\n"
            "- `reports/` bias/robustness reports\n"
            f"- `{REGISTRY_FILENAME}` external-dataset license/provenance registry\n\n"
            "Data rules: no fabricated labels; no race/ethnicity inference; skin tone is\n"
            "never a proxy for race; no duplication of minority samples to fake balance;\n"
            "all images of one person stay in exactly one split (see leakage.py).\n\n"
            + TEMPLATE_NOTE
            + "\n",
            encoding="utf-8",
        )
    return annotations


def write_annotations(path: Path, rows: List[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        # extrasaction=ignore: rows enriched with feature columns stay writable
        # while the file itself keeps the canonical spec schema.
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS), extrasaction="ignore")
        writer.writeheader()
        if not rows:
            fh.write(f"# {TEMPLATE_NOTE}\n")
        writer.writerows(rows)


def write_extended_annotations(path: Path, rows: List[Dict]) -> None:
    """Write the extended real-data schema (traceability + harmony targets)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(EXTENDED_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_harmony_scores(path: Path, rows: List[Dict]) -> None:
    """Long-format per-colour draping scores (see HARMONY_COLOURS)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(HARMONY_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_registry(path: Path) -> None:
    """Create DATASET_REGISTRY.md with one empty tracking block per source."""
    path = Path(path)
    if path.exists():
        return
    path.write_text(REGISTRY_TEMPLATE, encoding="utf-8")


REGISTRY_TEMPLATE = """# FaceMaxx DATASET REGISTRY

Every external dataset considered for training or evaluation is tracked here
BEFORE download. No dataset enters `data/raw/` without a completed block and a
license that permits the intended use.

Rules:
- CelebAMask-HQ: face PARSING (skin/hair/eyes/lips masks) only - never seasonal ground truth.
- FFHQ: preprocessing/robustness research only; respect per-image licensing.
- Fitzpatrick17k (or equivalent): skin-DEPTH robustness evaluation only where the
  license permits; Fitzpatrick labels are NOT undertone/season labels.
- FaceMaxx custom annotated set: PRIMARY source for undertone, depth, chroma,
  contrast, temperature, season, secondary season, confidence.
- FaceMaxx colour-harmony/draping set: PRIMARY source for colour compatibility
  (see annotations/harmony_scores.csv).

---

## 1. CelebAMask-HQ

- **official source:**
- **URL:**
- **license:**
- **commercial use:** unknown / [confirm]
- **redistribution:** unknown / [confirm]
- **attribution required:** unknown / [confirm]
- **intended use:** face parsing masks (skin/hair/eyes/lips) for segmentation supervision
- **known limitations:** constricted pose/lighting diversity; celebrity skew; NOT seasonal labels
- **downloaded:** no
- **checksum (sha256):**
- **preprocessing version:**

---

## 2. FFHQ (Flickr-Faces-HQ)

- **official source:** NVIDIA
- **URL:** https://github.com/NVlabs/ffhq-dataset
- **license:** research; images retain original Flickr licenses - verify per image
- **commercial use:** no (dataset card); per-image rights vary
- **redistribution:** do not redistribute images; distribute metadata/scripts
- **attribution required:** yes (cite FFHQ paper; respect Flickr source)
- **intended use:** face preprocessing/robustness (detection, alignment, white balance)
- **known limitations:** licensing heterogeneity; not undertone/season labeled
- **downloaded:** no
- **checksum (sha256):**
- **preprocessing version:**

---

## 3. Fitzpatrick17k (skin-tone benchmark)

- **official source:**
- **URL:**
- **license:**
- **commercial use:** unknown / [confirm]
- **redistribution:** unknown / [confirm]
- **attribution required:** unknown / [confirm]
- **intended use:** skin-DEPTH robustness EVALUATION ONLY (never undertone/season truth)
- **known limitations:** dermatoscopic/web images; Fitzpatrick skin-type labels are not colour-analysis axes
- **downloaded:** no
- **checksum (sha256):**
- **preprocessing version:**

---

## 4. FaceMaxx custom annotated dataset (PRIMARY)

- **official source:** this project
- **URL:** data/raw + data/metadata/facemaxx_color_annotations.csv
- **license:** project-owned
- **commercial use:** yes
- **redistribution:** no
- **attribution required:** n/a
- **intended use:** undertone / skin depth / chroma / contrast / temperature / season / secondary season / confidence
- **known limitations:** size grows with annotation effort; annotator agreement tracked per image
- **downloaded:** n/a (self-collected)
- **checksum:** n/a
- **preprocessing version:**

---

## 5. FaceMaxx colour-harmony / draping dataset (PRIMARY)

- **official source:** this project
- **URL:** data/annotations/harmony_scores.csv
- **license:** project-owned
- **commercial use:** yes
- **redistribution:** no
- **attribution required:** n/a
- **intended use:** per-colour compatibility scores (burgundy ... plum) for the harmony model
- **known limitations:** subjective; multi-annotator scores stored long-format
- **downloaded:** n/a (self-collected)
- **checksum:** n/a
- **preprocessing version:**
"""


def read_annotations(path: Path) -> List[Dict]:
    """Read the annotation CSV, skipping template comment lines."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"annotations not found: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    return [
        {k: (v if v not in ("", None) else None) for k, v in row.items() if k}
        for row in reader
    ]


def join_with_features(
    annotations: List[Dict],
    features: Dict[str, Dict],
    require_features: bool = False,
) -> List[Dict]:
    """Left-join extracted colour features onto annotation rows by image_id.

    ``features`` maps image_id -> feature dict (from colour.extract_image).
    Rows without features are dropped when ``require_features`` is set (e.g.
    before training); kept otherwise (e.g. distribution-only reports).
    """
    joined: List[Dict] = []
    for row in annotations:
        image_id = str(row.get("image_id") or "")
        feats = features.get(image_id)
        if feats is None:
            if require_features:
                continue
            merged = dict(row)
            merged["features_ok"] = False
        else:
            merged = {**row, **coerce_feature_row(feats), "features_ok": True}
        joined.append(merged)
    return joined


def find_image(raw_dir: Path, image_id: str, extensions=(".jpg", ".jpeg", ".png", ".webp")) -> Optional[Path]:
    """Locate the raw image file for an image_id."""
    raw_dir = Path(raw_dir)
    for ext in extensions:
        candidate = raw_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


# Feature columns that are labels, not numbers (everything else feat_* -> float).
STRING_FEATURES = frozenset({"feat_exposure_flag", "feat_face_box_source"})


def coerce_feature_row(row: Dict) -> Dict:
    """CSV rows arrive as strings; coerce feat_* numerics to float.

    Models and heuristics compare features numerically (``L* >= 78``), so a
    string "68.6" from the feature table would crash them. Non-numeric values
    become NaN and are imputed downstream by feature_matrix().
    """
    out: Dict = {}
    for key, value in row.items():
        if key.startswith("feat_") and key not in STRING_FEATURES:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                out[key] = float("nan")
        else:
            out[key] = value
    return out
