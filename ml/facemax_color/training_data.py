"""Canonical training-data preparation layer (Step 6/7/8 of the ML milestone).

One place where validated annotations become model-ready matrices:

    annotations (metadata/facemaxx_color_annotations.csv)
      x features  (processed/colour_features.csv, from the CV pipeline)
      -> joined rows, EVERY exclusion reported (never silently dropped)
      -> person-grouped split assignment FROM the locked manifests
      -> leakage-audited feature matrix (canonical feat_* columns ONLY)

Hard rules enforced here (fail loudly, never auto-fix):
  - the feature matrix may contain ONLY the canonical feat_* columns produced
    by colour.extract_image().as_feature_row(); any other column (labels,
    annotator votes, agreement scores, metadata) raises LeakageError.
  - rows in the LOCKED test split (holdout) can never enter train/val data.
  - a person_id is mandatory for training (person-grouped evaluation).
  - labels outside the canonical vocabularies are excluded AND reported.
  - duplicate image_ids are excluded AND reported.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from facemax_bias.vocab import (
    CHROMA_ORDER,
    CONTRAST_ORDER,
    SEASON_ORDER,
    SKIN_DEPTH_ORDER,
    UNDERTONE_ORDER,
)

from .models import FEATURE_COLUMNS

# The canonical supervised targets. Columns/vocabularies are the REPO's
# controlled vocabularies - never new spellings (spec: reuse, don't invent).
TARGETS: Dict[str, Dict[str, Any]] = {
    "undertone": {"column": "undertone", "order": UNDERTONE_ORDER},
    "skin_depth": {"column": "skin_depth", "order": SKIN_DEPTH_ORDER},
    "chroma": {"column": "chroma_level", "order": CHROMA_ORDER},
    "contrast": {"column": "contrast_level", "order": CONTRAST_ORDER},
    "season": {"column": "season_primary", "order": SEASON_ORDER},
}

FEATURE_VERSION = "facemax-color-features-v1"


class LeakageError(RuntimeError):
    """Raised when a non-feature column tries to enter the feature matrix."""


def audit_feature_columns(columns: List[str]) -> List[str]:
    """Reject any requested feature column outside the canonical feat_* set.

    This is the structural guarantee that labels (season_primary, undertone,
    skin_depth, chroma_level, contrast_level), annotator votes, agreement
    scores and metadata can never become model inputs. Fail loudly.
    """
    allowed = set(FEATURE_COLUMNS)
    bad = [c for c in columns if c not in allowed]
    if bad:
        raise LeakageError(
            "feature leakage: these columns are NOT canonical CV features and "
            f"must never be model inputs: {bad}. Allowed: {sorted(allowed)}"
        )
    return list(columns)


def dataset_digest(paths: Tuple[Path, ...]) -> str:
    """Short deterministic digest of the exact data files used (traceability)."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()[:16]


def read_feature_table(path: Path) -> Dict[str, Dict[str, Any]]:
    """image_id -> raw feature row (strings; coerced later by coerce_feature_row)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"feature table missing (run `extract` first): {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["image_id"]: row for row in csv.DictReader(fh)}


@dataclass
class TrainingData:
    """Everything a training run needs, plus the full exclusion report."""

    target: str                      # canonical target key (see TARGETS)
    label_column: str                # annotation column holding the truth
    rows: List[Dict[str, Any]] = field(default_factory=list)   # joined + split-tagged
    exclusions: Counter = field(default_factory=Counter)       # reason -> count
    exclusion_detail: Dict[str, List[str]] = field(default_factory=dict)
    class_balance: List[Dict[str, Any]] = field(default_factory=list)
    total_annotated_rows: int = 0
    source: str = "real"             # real | demo
    dataset_version: str = ""
    feature_version: str = FEATURE_VERSION
    split_counts: Dict[str, int] = field(default_factory=dict)
    holdout_ids: List[str] = field(default_factory=list)
    # Rows in the LOCKED test split, kept for `evaluate` only - they are never
    # returned by usable_rows and never enter training matrices.
    holdout_rows: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def usable_rows(self) -> List[Dict[str, Any]]:
        """Rows that carry a valid label and a person_id (train/val candidates)."""
        return [
            r for r in self.rows
            if r.get("_label_ok") and r.get("_person_ok")
        ]

    def summary(self) -> str:
        width = 34
        lines = ["Training data preparation", "=" * width]
        lines.append(f"target:                    {self.target} (column: {self.label_column})")
        lines.append(f"source:                    {self.source}")
        lines.append(f"dataset_version:           {self.dataset_version}")
        lines.append(f"feature_version:           {self.feature_version}")
        lines.append(f"total annotated rows:      {self.total_annotated_rows}")
        lines.append(f"rows with valid features:  {sum(self.split_counts.values())}")
        lines.append(f"rows excluded:             {sum(self.exclusions.values())}")
        if self.exclusions:
            lines.append("")
            lines.append("excluded (each exclusion is reported, never silent):")
            for reason, count in self.exclusions.most_common():
                ids = self.exclusion_detail.get(reason, [])
                suffix = f"  e.g. {ids[:3]}" if ids else ""
                lines.append(f"  {reason:<28} {count:>4}{suffix}")
        if self.class_balance:
            lines.append("")
            lines.append("class balance (images / people — never resampled blindly):")
            header = f"  {'label':<16} {'images':>6} {'people':>7} {'share':>7}"
            lines.append(header)
            lines.append("  " + "-" * (len(header) - 2))
            total = sum(entry["images"] for entry in self.class_balance)
            for entry in self.class_balance:
                share = entry["images"] / total if total else 0.0
                lines.append(
                    f"  {entry['label']:<16} {entry['images']:>6} {entry['people']:>7} "
                    f"{share * 100:>6.1f}%"
                )
        if self.split_counts:
            lines.append("")
            lines.append(f"usable rows by split: {dict(sorted(self.split_counts.items()))}")
        if self.holdout_ids:
            lines.append(
                f"locked holdout excluded from training: {len(self.holdout_ids)} images"
            )
        return "\n".join(lines)


def _load_split_manifest(root: Path, name: str) -> List[str]:
    path = root / "splits" / f"{name}.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assign_splits(rows: List[Dict[str, Any]], root: Path) -> Tuple[List[str], Dict[str, int]]:
    """Tag rows with their split FROM the manifests (never re-split here).

    Real data must reuse the frozen person-grouped manifests written by
    `audit --splits/--regen-splits`. Re-splitting inside a training script
    would silently create a different dataset than the locked holdout.
    """
    ids = {"train": set(_load_split_manifest(root, "train")),
           "val": set(_load_split_manifest(root, "val")),
           "test": set(_load_split_manifest(root, "test"))}
    for row in rows:
        image_id = str(row.get("image_id") or "")
        row["split"] = "train" if image_id in ids["train"] else \
                       "val" if image_id in ids["val"] else \
                       "test" if image_id in ids["test"] else "unsplit"
    counts = Counter(str(r["split"]) for r in rows)
    return rows, dict(counts)


def prepare_training_data(
    annotations_path: Path,
    features_path: Path,
    target: str,
    data_root: Optional[Path] = None,
    source: str = "real",
) -> TrainingData:
    """Join annotations + features for one target with full exclusion reporting.

    ``data_root`` (the dataset tree) is required for real data so the locked
    split manifests can be honoured; demo data may pass None (CV-only).
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {sorted(TARGETS)}")
    spec = TARGETS[target]
    label_column = spec["column"]
    order = list(spec["order"])

    annotations_path = Path(annotations_path)
    rows = _read_annotation_rows(annotations_path)
    data = TrainingData(
        target=target,
        label_column=label_column,
        total_annotated_rows=len(rows),
        source=source,
    )

    # --- duplicate image_ids (subsequent occurrences excluded, all reported) -
    seen_ids: set = set()
    for row in rows:
        image_id = str(row.get("image_id") or "")
        if not image_id:
            _exclude(data, "missing_image_id", "")
            row["_exclude"] = "missing_image_id"
            continue
        if image_id in seen_ids:
            _exclude(data, "duplicate_image_id", image_id)
            row["_exclude"] = "duplicate_image_id"
            continue
        seen_ids.add(image_id)

    # --- join features -------------------------------------------------------
    features = read_feature_table(features_path)
    for row in rows:
        if row.get("_exclude"):
            continue
        image_id = str(row.get("image_id") or "")
        feats = features.get(image_id)
        if feats is None:
            _exclude(data, "missing_feature_row", image_id)
            row["_exclude"] = "missing_feature_row"
            continue
        row.update({k: v for k, v in feats.items() if k != "image_id"})
        if str(row.get("feat_ok", "")).lower() in ("false", "0", "no"):
            _exclude(data, "failed_feature_extraction", image_id)
            row["_exclude"] = "failed_feature_extraction"

    # --- label + person validity ---------------------------------------------
    for row in rows:
        if row.get("_exclude"):
            continue
        label = str(row.get(label_column) or "").strip()
        if label not in order:
            _exclude(data, f"invalid_{label_column}_label", f"{row.get('image_id')}:{label or '<empty>'}")
            row["_exclude"] = "invalid_label"
            row["_label_ok"] = False
        else:
            row["_label_ok"] = True
        person = str(row.get("person_id") or "").strip()
        row["_person_ok"] = bool(person)
        if not person:
            _exclude(data, "missing_person_id", str(row.get("image_id")))
            row["_exclude"] = "missing_person_id"

    # --- split assignment from the locked manifests --------------------------
    if data_root is not None:
        rows, counts = _assign_splits(rows, Path(data_root))
        for row in rows:
            if row.get("_exclude"):
                continue
            if row["split"] == "test":
                data.holdout_ids.append(str(row.get("image_id")))
                data.holdout_rows.append(row)
                _exclude(data, "locked_holdout_row", str(row.get("image_id")))
                row["_exclude"] = "locked_holdout_row"
            elif row["split"] == "unsplit":
                _exclude(data, "not_in_split_manifests", str(row.get("image_id")))
                row["_exclude"] = "not_in_split_manifests"
    else:
        for row in rows:
            if not row.get("_exclude"):
                row["split"] = "train"

    data.rows = [r for r in rows if not r.get("_exclude")]
    data.split_counts = Counter(str(r.get("split")) for r in data.rows)
    data.split_counts = dict(data.split_counts)

    # --- class balance report (images AND people per class) ------------------
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in data.rows:
        by_label[str(row[label_column])].append(row)
    data.class_balance = [
        {
            "label": label,
            "images": len(members),
            "people": len({str(m.get("person_id")) for m in members}),
        }
        for label, members in sorted(by_label.items(), key=lambda kv: order.index(kv[0]))
    ]

    paths = [annotations_path, Path(features_path)]
    data.dataset_version = dataset_digest(tuple(paths))
    return data


def _read_annotation_rows(annotations_path: Path) -> List[Dict[str, Any]]:
    from .dataset import read_annotations

    return read_annotations(annotations_path)


def _exclude(data: TrainingData, reason: str, detail: str) -> None:
    data.exclusions[reason] += 1
    data.exclusion_detail.setdefault(reason, []).append(detail)


def matrices(
    data: TrainingData,
    splits: Tuple[str, ...] = ("train",),
    extra_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature matrix X, label vector y and group vector for the given splits.

    ``extra_columns`` must pass the leakage audit (it always does: only
    canonical feat_* columns can ever be appended).
    """
    from .dataset import coerce_feature_row

    columns = audit_feature_columns(list(FEATURE_COLUMNS) + list(extra_columns or []))
    rows = [r for r in data.rows if r.get("split") in splits and r.get("_label_ok") and r.get("_person_ok")]
    if not rows:
        raise ValueError(
            f"no usable rows for splits {splits} (target={data.target}); "
            "check the split manifests and exclusions in the preparation report"
        )
    X = np.array(
        [[float(coerce_feature_row({c: r.get(c)}).get(c)) if r.get(c) not in (None, "") else np.nan
          for c in columns] for r in rows],
        dtype=np.float64,
    )
    # median-impute NaNs per column (deterministic; documented)
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        X[~np.isfinite(col), j] = float(np.median(finite)) if finite.size else 0.0
    y = np.array([str(r[data.label_column]) for r in rows])
    groups = np.array([str(r["person_id"]) for r in rows])
    return X, y, groups


def save_report(data: TrainingData, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": data.target,
        "label_column": data.label_column,
        "source": data.source,
        "dataset_version": data.dataset_version,
        "feature_version": data.feature_version,
        "total_annotated_rows": data.total_annotated_rows,
        "usable_rows": data.split_counts,
        "exclusions": dict(data.exclusions),
        "exclusion_detail": data.exclusion_detail,
        "class_balance": data.class_balance,
        "holdout_ids": data.holdout_ids,
        "feature_version_note": "canonical feat_* columns only (leakage-audited)",
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
