"""Registered external datasets: what they may and may NOT be used for.

Every claim in this module traces back to data/DATASET_REGISTRY.md (license
facts verified there) and data/ingest/sources.json (ingestion state). The
module exists so the COMPATIBILITY RULES are executable, not prose:

- FFHQ: images only (identity/label-free) -> preprocessing/robustness/quality
  research. No FaceMaxx axis may be learned from FFHQ.
- CelebAMask-HQ: face-parsing masks -> skin-region segmentation supervision
  and mask validation. Masks are geometry, never colour semantics.
- Fitzpatrick17k: Fitzpatrick types I-VI are dermatology sun-reactivity
  labels. They stay untouched in ``fitzpatrick_type`` and never become
  undertone, skin-depth, chroma or season labels.

Nothing here downloads data, fabricates labels, or bypasses the license gate
in ingest.py. Status functions REPORT; they do not mutate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ingest import read_sources
from .dataset import REGISTRY_FILENAME

# Canonical CelebAMask-HQ face-parsing label set (19 classes), grouped by the
# FaceMaxx region schema. Mapping is based on the official CelebAMask-HQ
# annotation classes; pixel values in .png masks follow the official
# 'g_mask' convention (index into this list after palette removal).
CELEBAMASK_LABELS: Tuple[str, ...] = (
    "background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow",
    "r_brow", "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat",
    "ear_r", "neck_l", "neck", "cloth",
)

# FaceMaxx region groups over the CelebAMask labels.
CELEBAMASK_SKIN_GROUP: Tuple[str, ...] = ("skin", "neck")
CELEBAMASK_EXCLUDE_GROUPS: Tuple[str, ...] = (
    "hair", "u_lip", "l_lip", "mouth",           # visible features to EXCLUDE
    "l_eye", "r_eye", "l_brow", "r_brow",        # eyes/brows to EXCLUDE
    "background", "cloth", "hat", "ear_r", "neck_l", "eye_g",  # non-skin
)
CELEBAMASK_KEEP_GROUPS: Tuple[str, ...] = CELEBAMASK_SKIN_GROUP

# Tasks a dataset may contribute to. Keys are the FaceMaxx task names; only
# tasks listed in a dataset's entry may receive that dataset's data.
DATASET_TASK_COMPATIBILITY: Dict[str, Tuple[str, ...]] = {
    "FFHQ": (
        "face_detection",
        "face_alignment",
        "image_preprocessing",
        "image_quality_evaluation",
        "lighting_robustness",
        "appearance_robustness",
        "skin_region_feature_extraction_research",
    ),
    "CelebAMask-HQ": (
        "skin_region_segmentation",
        "face_parsing_mask_validation",
        "skin_vs_nonskin_separation",
        "image_preprocessing",
    ),
    "Fitzpatrick17k": (
        "skin_tone_representation_analysis",
        "skin_depth_robustness_evaluation_only",
    ),
}

# Targets that require expert FaceMaxx annotation. No external dataset may
# feed these, under any mapping, ever.
EXPERT_ONLY_TARGETS: Tuple[str, ...] = (
    "undertone", "skin_depth", "chroma", "contrast", "temperature", "season",
)

# Task names that amount to supervised use of an expert-only target.
_SUPERVISED_TASKS = {
    "undertone", "skin_depth", "chroma", "contrast", "temperature", "season",
    "undertone_classification", "skin_depth_classification",
    "season_classification", "seasonal_colour_classification",
}


class LabelCompatibilityError(ValueError):
    """A dataset was asked to label a task its actual labels cannot support."""


def assert_labels_allowed(dataset: str, task: str) -> None:
    """Guard: refuse dataset->task pairs the dataset's labels cannot support.

    Expert-only colour axes (undertone/season/...) are refused outright for
    external datasets; other tasks must appear in DATASET_TASK_COMPATIBILITY.
    """
    if dataset in DATASET_TASK_COMPATIBILITY and task in _SUPERVISED_TASKS:
        raise LabelCompatibilityError(
            f"{dataset} cannot label {task!r}: {sorted(_SUPERVISED_TASKS)} are "
            f"supervised ONLY by the expert-annotated FaceMaxx custom set "
            f"(see DATASET_REGISTRY.md rules)"
        )
    allowed = DATASET_TASK_COMPATIBILITY.get(dataset)
    if allowed is None:
        raise LabelCompatibilityError(
            f"unknown dataset {dataset!r}; registered sources: "
            f"{sorted(DATASET_TASK_COMPATIBILITY)}"
        )
    if task not in allowed:
        raise LabelCompatibilityError(
            f"{dataset} may not support task {task!r}; permitted tasks: {list(allowed)}"
        )


def map_celebamask_labels(raw_labels: List[str]) -> Dict[str, List[str]]:
    """Map raw CelebAMask-HQ mask class names -> FaceMaxx region groups.

    Returns {"skin": [...], "exclude": [...], "unknown": [...]} so callers can
    build skin masks WITHOUT ever touching colour semantics. Unknown labels
    are preserved (never silently dropped) and reported to the caller.
    """
    known = set(CELEBAMASK_LABELS)
    skin, exclude, unknown = [], [], []
    for raw in raw_labels:
        label = str(raw).strip().lower()
        if label in CELEBAMASK_SKIN_GROUP:
            skin.append(label)
        elif label in CELEBAMASK_EXCLUDE_GROUPS:
            exclude.append(label)
        elif label in known:
            # Known CelebAMask class not part of either group (defensive).
            unknown.append(label)
        else:
            unknown.append(label)
    return {"skin": skin, "exclude": exclude, "unknown": unknown}


def validate_celebamask_pairing(
    image_sizes: List[Optional[Tuple[int, int]]],
    mask_sizes: List[Optional[Tuple[int, int]]],
    mask_labels: List[Optional[List[str]]],
) -> Dict[str, Any]:
    """Image/mask alignment report for a CelebAMask-HQ pairing set.

    Checks per sample: presence, dimension equality, and label vocabulary.
    READ-ONLY diagnostics: problems are counted and returned, never repaired
    or silently skipped, per the pipeline's reporting rules.
    """
    n = max(len(image_sizes), len(mask_sizes), len(mask_labels))
    missing_masks = 0
    missing_images = 0
    dim_mismatches: List[int] = []
    bad_labels: List[int] = []
    for i in range(n):
        img = image_sizes[i] if i < len(image_sizes) else None
        msk = mask_sizes[i] if i < len(mask_sizes) else None
        labels = mask_labels[i] if i < len(mask_labels) else None
        if img is None:
            missing_images += 1
        if msk is None or labels is None:
            missing_masks += 1
            continue
        if img is not None and tuple(img) != tuple(msk):
            dim_mismatches.append(i)
        unknown = [lab for lab in labels if str(lab).strip().lower() not in set(CELEBAMASK_LABELS)]
        if unknown:
            bad_labels.append(i)
    return {
        "n": n,
        "missing_images": missing_images,
        "missing_masks": missing_masks,
        "dimension_mismatch_count": len(dim_mismatches),
        "dimension_mismatch_indices": dim_mismatches[:50],
        "invalid_mask_label_count": len(bad_labels),
        "invalid_mask_label_indices": bad_labels[:50],
        "ok": n > 0 and missing_masks == 0 and missing_images == 0
        and not dim_mismatches and not bad_labels,
    }


def validate_fitzpatrick_frame(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fitzpatrick17k label-preservation guard.

    Fitzpatrick types (including the documented -1 'unknown' value) must be
    carried in a dedicated ``fitzpatrick_type`` column and must NOT appear in
    any FaceMaxx axis column (undertone/skin_depth/season/...). Missing
    Fitzpatrick values stay missing (-1/None): they are never defaulted to a
    class, because Fitzpatrick types are NOT FaceMaxx labels at all.
    """
    forbidden = ("undertone", "skin_depth", "chroma", "contrast",
                 "temperature", "season")
    problems: List[str] = []
    unknown_types = 0
    labelled = 0
    for i, row in enumerate(rows):
        for axis in forbidden:
            if axis in row and row[axis] not in (None, ""):
                problems.append(f"row {i}: forbidden FaceMaxx axis {axis!r} present")
        ft = row.get("fitzpatrick_type", row.get("fitzpatrick", None))
        if ft in (None, "", -1, "-1"):
            unknown_types += 1
        else:
            labelled += 1
    return {
        "n": len(rows),
        "labelled": labelled,
        "unknown_type": unknown_types,
        "problems": problems,
        "ok": not problems,
        "note": "Fitzpatrick types are sun-reactivity labels; they are never "
                "converted to FaceMaxx undertone/depth/season labels.",
    }


def external_dataset_status(root: Path) -> List[Dict[str, Any]]:
    """Discovery/status of every registered external dataset (read-only).

    Combines data/ingest/sources.json (ingestion state) with a check for
    extracted files under the dataset's designated real-data directory
    data/raw/external/<name>/ so 'downloaded' vs 'extracted' vs 'ingested'
    are distinguishable without guessing.
    """
    root = Path(root)
    status: List[Dict[str, Any]] = []
    for entry in read_sources(root):
        name = entry["name"]
        extract_dir = root / "raw" / "external" / name
        has_files = extract_dir.exists() and any(extract_dir.iterdir())
        receipt = entry.get("receipt")
        receipt_ok = bool(receipt) and (root / "ingest" / str(receipt)).exists()
        status.append(
            {
                "name": name,
                "registered": True,
                "registry_file": (root / REGISTRY_FILENAME).exists(),
                "license_gate": entry.get("license_gate", "unconfirmed"),
                "downloaded": entry.get("status") == "downloaded",
                "receipt_present": receipt_ok,
                "extracted": has_files,
                "extract_dir": str(extract_dir),
                "ingested": bool(entry.get("ingested", False)),
                "tasks": list(DATASET_TASK_COMPATIBILITY.get(name, ())),
                "official_url": entry.get("official_url", ""),
                "intended_use": entry.get("intended_use", ""),
            }
        )
    return status
