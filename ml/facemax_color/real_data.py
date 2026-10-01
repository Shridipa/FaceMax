"""REAL-image CV testing mode (evidence class: REAL IMAGE - UNLABELLED).

This module runs GENUINE photographs from a licence-confirmed, registered
external dataset through the existing FaceMaxx CV pipeline:

    decode -> face detection -> box plausibility -> skin segmentation
           -> colour extraction (-> optional perturbation stability probes)

What this DOES prove: the CV machinery operates on real photographs, with
real failure modes (decode errors, missed detections, implausible boxes,
degenerate skin masks) counted and reported - never silently discarded.

What this does NOT prove: undertone / skin-depth / season ACCURACY. Real
photographs alone carry no FaceMaxx colour ground truth. External datasets
can never label these axes (external_datasets.EXPERT_ONLY_TARGETS); only the
expert annotation workflow can. Model predictions shown here are made by
SYNTHETIC-DEMO-TRAINED artifacts used as probe instruments and are labelled
as such everywhere they appear.

NO SILENT DEMO FALLBACK (mandatory): if no usable real images are available
(missing dataset, unconfirmed licence, all-decode-failed selection), the run
ends status=BLOCKED with an explicit reason and
``synthetic_demo_substituted: false``. Demo evaluation remains a separate,
explicitly selected mode (``demo-image``).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .colour import (
    detect_all_face_boxes,
    extract_image,
    load_image,
    rgb_skin_mask,
)
from .dataset import ANNOTATIONS_FILENAME, read_annotations
from .external_datasets import build_skin_mask_with_exclusions
from .ingest import read_sources

EVIDENCE_CLASS = "REAL IMAGE - UNLABELLED"

# Real archive layout (verified in the ingestion receipts; the extracted
# archive nests one level deeper than the dataset name).
_CELEBA_BASE = ("raw", "external", "CelebAMask-HQ", "CelebAMask-HQ")
_CELEBA_IMAGES = "CelebA-HQ-img"
_CELEBA_MASKS = "CelebAMask-HQ-mask-anno"
_CELEBA_MASK_SHARD_SIZE = 2000  # shards 0..14 of 2000 indices each

# Skin-region GT classes; every other present class excludes (exclusion-wins).
_GT_SKIN_CLASSES = ("skin", "neck")

# Feature keys tracked under perturbation stability probes.
_STABILITY_FEATURES = (
    "feat_skin_L",
    "feat_skin_a",
    "feat_skin_b",
    "feat_ita",
    "feat_undertone_proxy",
    "feat_depth_proxy",
    "feat_chroma_proxy",
)

_PROBE_AXES = ("undertone", "skin_depth")


class RealDataError(RuntimeError):
    """Real-data mode asked for something the repo cannot currently do."""


# ---------------------------------------------------------------------------
# Provenance / licence / annotation status (read-only)
# ---------------------------------------------------------------------------

def _license_state(root: Path, dataset: str) -> Dict[str, Any]:
    for entry in read_sources(root):
        if str(entry.get("name", "")).lower() == dataset.lower():  # ingest.py convention
            return {
                "registered": True,
                "license_gate": entry.get("license_gate", "unconfirmed"),
                "status": entry.get("status"),
                "official_url": entry.get("official_url", ""),
            }
    return {"registered": False, "license_gate": "unregistered", "status": None,
            "official_url": ""}


def _provenance(root: Path, dataset: str) -> Dict[str, Any]:
    path = Path(root) / "ingest" / f"{dataset}.receipt.json"
    if not path.exists():
        return {"receipt_present": False}
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"receipt_present": True, "receipt_readable": False}
    return {
        "receipt_present": True,
        "sha256": receipt.get("sha256"),
        "archive_bytes": receipt.get("bytes"),
        "recorded": receipt.get("recorded"),
        "license_summary": str(receipt.get("license", ""))[:200],
    }


def _annotation_status(root: Path) -> Dict[str, Any]:
    """Expert-labelled (Category C) availability: REPORTED, never inferred."""
    try:
        rows = read_annotations(Path(root) / "metadata" / ANNOTATIONS_FILENAME)
    except Exception:
        rows = []
    targets = {"undertone": 0, "skin_depth": 0, "season": 0}
    people = set()
    for row in rows:
        person = str(row.get("person_id") or "").strip()
        if person:
            people.add(person)
        for target in targets:
            if str(row.get(target if target != "season" else "season_primary") or "").strip():
                targets[target] += 1
    return {
        "annotated_images": sum(
            1 for r in rows if str(r.get("season_primary") or "").strip()
        ),
        "people": len(people),
        "expert_labelled_available": bool(rows),
        "target_counts": targets,
        "note": (
            "expert FaceMaxx labels come ONLY from the annotation workflow "
            "(ANNOTATION_HANDBOOK.md); external datasets cannot supply them"
            if rows else
            "no expert annotations exist yet: model-accuracy evaluation ("
            "Category C) is unavailable"
        ),
    }


# ---------------------------------------------------------------------------
# Sample selection (deterministic, bounded)
# ---------------------------------------------------------------------------

def _list_celeba_images(images_dir: Path) -> List[int]:
    """Sorted numeric indices of the extracted CelebA-HQ-img/*.jpg files."""
    if not images_dir.is_dir():
        return []
    indices = []
    for path in images_dir.glob("*.jpg"):
        stem = path.stem
        if stem.isdigit():
            indices.append(int(stem))
    return sorted(indices)


def _sample_indices(universe: List[int], limit: int, offset: int) -> List[int]:
    """Deterministic, evenly spread sample; --offset slides the window.

    Not random: the same --limit/--offset always selects the same images so
    runs are reproducible and resumable (offset advances the window).
    """
    if not universe or limit <= 0:
        return []
    step = max(1, len(universe) // limit)
    effective = offset % step if step > 1 else 0
    return list(universe[effective::step])[:limit]


# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------

def box_plausibility(box: Tuple[int, int, int, int], image_shape) -> List[str]:
    """A detection counts as plausible only if it passes these geometry checks.

    A non-empty bounding box alone is NOT a successful detection.
    """
    x, y, w, h = box
    ih, iw = image_shape[:2]
    reasons: List[str] = []
    if w < 80 or h < 80:
        reasons.append("box_below_min_size")
    if x < 0 or y < 0 or x + w > iw or y + h > ih:
        reasons.append("box_out_of_frame")
    area = float(w * h)
    if area < 0.01 * iw * ih:
        reasons.append("box_tiny_fraction")
    if area > 0.98 * iw * ih:
        reasons.append("box_dominates_frame")
    aspect = w / float(h) if h else 0.0
    if not (0.5 <= aspect <= 2.0):
        reasons.append("implausible_aspect_ratio")
    return reasons


def _ground_truth_face_mask(
    image_shape,
    box: Tuple[int, int, int, int],
    masks_dir: Path,
    idx: int,
) -> Optional[Dict[str, Any]]:
    """GT skin mask (exclusion-wins) cropped to the face box, or None.

    Raw per-class mask PNGs are combined with the SAME composition semantics
    as the ingestion validator (features sit ON skin -> exclusion wins), then
    nearest-resized to the image geometry and cropped to the face box.
    """
    import cv2

    shard = idx // _CELEBA_MASK_SHARD_SIZE
    paths = sorted((Path(masks_dir) / str(shard)).glob(f"{idx:05d}_*.png"))
    if not paths:
        return None
    ih, iw = image_shape[:2]
    skin = np.zeros((ih, iw), dtype=bool)
    exclude = np.zeros((ih, iw), dtype=bool)
    for path in paths:
        raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        label = path.stem.split("_", 1)[1].strip().lower()
        if raw.shape != (ih, iw):
            raw = cv2.resize(raw, (iw, ih), interpolation=cv2.INTER_NEAREST)
        mask = raw > 0
        if label in _GT_SKIN_CLASSES:
            skin |= mask
        else:
            exclude |= mask
    composed = build_skin_mask_with_exclusions(skin, exclude)
    x, y, w, h = box
    return {
        "skin": composed["skin"][y : y + h, x : x + w],
        "overlap_px": composed["overlap_px"],
    }


def confusion_counts(pred_mask, gt_mask) -> Tuple[int, int, int]:
    """(tp, fp, fn) between the predicted skin mask and the GT mask."""
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(
            f"mask shape mismatch: predicted {pred.shape} vs gt {gt.shape}"
        )
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    return tp, fp, fn


def segmentation_metrics(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
    """IoU / Dice / pixel precision / pixel recall from additive confusion counts."""

    def _ratio(num: int, den: int) -> Optional[float]:
        return round(num / den, 4) if den else None

    return {
        "iou": _ratio(tp, tp + fp + fn),
        "dice": _ratio(2 * tp, 2 * tp + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
    }


def build_perturbations(image_bgr: "np.ndarray") -> Dict[str, "np.ndarray"]:
    """Controlled, technically justified variants (never written to datasets)."""
    import cv2

    variants: Dict[str, "np.ndarray"] = {}
    try:
        variants["brightness+10"] = cv2.convertScaleAbs(image_bgr, alpha=1.0, beta=10)
        variants["brightness-10"] = cv2.convertScaleAbs(image_bgr, alpha=1.0, beta=-10)
    except Exception:
        pass
    try:
        ok, encoded = cv2.imencode(
            ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        )
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                variants["jpeg_q50"] = decoded
    except Exception:
        pass
    try:
        h, w = image_bgr.shape[:2]
        small = cv2.resize(image_bgr, (max(1, w // 2), max(1, h // 2)),
                           interpolation=cv2.INTER_AREA)
        variants["resolution_half_restored"] = cv2.resize(
            small, (w, h), interpolation=cv2.INTER_NEAREST
        )
    except Exception:
        pass
    return variants


# ---------------------------------------------------------------------------
# SYNTHETIC-DEMO-TRAINED artifacts as probe instruments (read-only)
# ---------------------------------------------------------------------------

def load_probe_predictors(root: Path) -> Dict[str, Dict[str, Any]]:
    """Newest per-axis artifact from the isolated demo tree, if any.

    These are used ONLY to report prediction STABILITY of demo-trained models
    on real inputs. They are labelled ``trained_on: synthetic_demo`` wherever
    they appear. Artifacts are loaded read-only; nothing is retrained,
    overwritten, or promoted.
    """
    import joblib

    base = Path(root) / "demo" / "models"
    loaded: Dict[str, Dict[str, Any]] = {}
    if not base.is_dir():
        return loaded
    for axis in _PROBE_AXES:
        axis_dir = base / axis
        files = sorted(axis_dir.glob("*/model.joblib")) if axis_dir.is_dir() else []
        if not files:
            continue
        try:
            predictor = joblib.load(files[-1])
        except Exception:
            continue
        loaded[axis] = {
            "predictor": predictor,
            "artifact": files[-1].parent.name,
            "trained_on": "synthetic_demo",
        }
    return loaded


def _predict_safe(predictor: Any, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        if hasattr(predictor, "verify_feature_schema"):
            predictor.verify_feature_schema(features)
        return predictor.predict_row(features)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def _blocked_report(root: Path, dataset: str, reason: str, **extra: Any) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "report": "real_cv_validation",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_class": EVIDENCE_CLASS,
        "dataset": {"name": dataset},
        "status": "BLOCKED",
        "blocked_reason": reason,
        "synthetic_demo_substituted": False,
        "counts": {"images_selected": 0, "images_processed": 0},
        "per_image": [],
        "model_evaluation": {
            "status": "BLOCKED",
            "reason": "no expert-labelled real data exists; external datasets "
                      "cannot supply FaceMaxx colour ground truth",
            "synthetic_demo_substituted": False,
        },
        "limitations": REAL_CV_LIMITATIONS,
    }
    report.update(extra)
    return report


REAL_CV_LIMITATIONS = [
    "convenience sample: detection/extraction rates on this sample are NOT "
    "general-population performance claims",
    "segmentation metrics compare the heuristic skin mask with CelebAMask-HQ "
    "annotation geometry INSIDE the Haar face box only; they validate mask "
    "composition, not colour analysis",
    "colour values from uncontrolled photographs are not laboratory-grade "
    "measurements",
    "any predictions shown come from SYNTHETIC-DEMO-TRAINED artifacts used as "
    "stability probes; they are not accuracy measurements",
    "CelebAMask-HQ identity annotations are not part of this repository, so "
    "unique-people counts are unknown, not zero",
]


def run_real_cv(
    root: Path,
    dataset: str = "CelebAMask-HQ",
    limit: int = 24,
    offset: int = 0,
    perturb: bool = False,
) -> Dict[str, Any]:
    """Run the real-image CV validation and return the machine-readable report.

    Never raises for expected unavailability: returns status=BLOCKED instead.
    """
    root = Path(root)

    # --- dataset gate: registered + licence-confirmed + implemented harness
    license_state = _license_state(root, dataset)
    if not license_state["registered"]:
        return _blocked_report(
            root, dataset,
            f"dataset {dataset!r} is not registered in data/ingest/sources.json",
            license=license_state,
            provenance=_provenance(root, dataset),
            annotation_status=_annotation_status(root),
        )
    if license_state["license_gate"] != "confirmed":
        return _blocked_report(
            root, dataset,
            f"licence gate for {dataset!r} is "
            f"{license_state['license_gate']!r}; human licence review required "
            "before any real-image use",
            license=license_state,
            provenance=_provenance(root, dataset),
            annotation_status=_annotation_status(root),
        )
    if dataset != "CelebAMask-HQ":
        return _blocked_report(
            root, dataset,
            "no real-image harness is implemented for this dataset yet "
            "(implemented: CelebAMask-HQ); do not substitute demo data",
            license=license_state,
            provenance=_provenance(root, dataset),
            annotation_status=_annotation_status(root),
        )

    base = root.joinpath(*_CELEBA_BASE)
    images_dir = base / _CELEBA_IMAGES
    masks_dir = base / _CELEBA_MASKS
    universe = _list_celeba_images(images_dir)
    if not universe:
        return _blocked_report(
            root, dataset,
            f"no extracted images found under {images_dir} - run the licensed "
            "download + extraction first; synthetic demo results were not "
            "substituted",
            license=license_state,
            provenance=_provenance(root, dataset),
            annotation_status=_annotation_status(root),
        )

    indices = _sample_indices(universe, limit, offset)
    if not indices:
        return _blocked_report(
            root, dataset, "sample selection produced zero images for the "
            "requested --limit/--offset",
            license=license_state,
            provenance=_provenance(root, dataset),
            annotation_status=_annotation_status(root),
        )

    predictors = load_probe_predictors(root) if perturb else {}

    counts = {
        "images_selected": len(indices),
        "images_universe": len(universe),
        "decode_failures": 0,
        "no_face_detected": 0,
        "implausible_face_box": 0,
        "multi_face_images": 0,
        "skin_mask_too_small": 0,
        "extraction_failed": 0,
        "images_processed": 0,
        "images_extracted_ok": 0,
        "images_with_seg_metrics": 0,
    }
    tp_total = fp_total = fn_total = 0
    per_image: List[Dict[str, Any]] = []
    stability = {
        "images_probed": 0,
        "feature_abs_deltas": {key: [] for key in _STABILITY_FEATURES},
        "prediction_flips": {axis: {"flips": 0, "compared": 0} for axis in predictors},
    }

    for idx in indices:
        # CelebA-HQ images are UNpadded (0.jpg ... 29999.jpg) while the
        # per-class masks ARE 5-padded (00000_skin.png) - both verified on the
        # extracted archive.
        image_path = images_dir / f"{idx}.jpg"
        record: Dict[str, Any] = {"image": image_path.name, "person": "unknown"}
        t0 = time.perf_counter()
        try:
            image = load_image(image_path)
        except Exception as exc:
            counts["decode_failures"] += 1
            record.update({"status": "decode_failed", "error": str(exc)[:120]})
            per_image.append(record)
            continue
        record["decode_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

        faces = detect_all_face_boxes(image)
        record["faces_detected"] = len(faces)
        if len(faces) > 1:
            counts["multi_face_images"] += 1
        box = max(faces, key=lambda f: f[2] * f[3]) if faces else None
        if box is None:
            counts["no_face_detected"] += 1
            record["status"] = "no_face_detected"
            per_image.append(record)
            continue
        plausible_problems = box_plausibility(box, image.shape)
        record["face_box_plausible"] = not plausible_problems
        if plausible_problems:
            counts["implausible_face_box"] += 1
            record.update({"status": "implausible_face_box",
                           "plausibility_problems": plausible_problems})
            per_image.append(record)
            continue
        record["face_box"] = list(int(v) for v in box)

        mask = rgb_skin_mask(image, box)
        skin_fraction = float(mask.mean()) if mask.size else 0.0
        record["skin_mask_fraction"] = round(skin_fraction, 4)
        if skin_fraction < 0.02:
            counts["skin_mask_too_small"] += 1
            record["status"] = "skin_mask_too_small"
            per_image.append(record)
            continue

        extraction = extract_image(image_path.stem, image_path)
        record["extraction_ok"] = extraction.ok
        if not extraction.ok:
            counts["extraction_failed"] += 1
            record.update({"status": "extraction_failed",
                           "error": str(extraction.error or "")[:120]})
            per_image.append(record)
            continue

        counts["images_extracted_ok"] += 1
        features = extraction.as_feature_row()
        record["skin_pixels"] = int(extraction.skin_pixels)
        record["face_fraction"] = _finite(extraction.face_fraction)
        record["exposure_flag"] = extraction.exposure_flag
        record["skin_L"] = _finite(extraction.skin_L)
        record["skin_a"] = _finite(extraction.skin_a)
        record["skin_b"] = _finite(extraction.skin_b)

        # --- segmentation vs GT (geometry validation, NOT colour accuracy)
        gt = _ground_truth_face_mask(image.shape, box, masks_dir, idx)
        if gt is None or not gt["skin"].any():
            record["segmentation"] = {"status": "unavailable",
                                      "reason": "no GT mask classes present"}
        else:
            tp, fp, fn = confusion_counts(mask, gt["skin"])
            tp_total += tp
            fp_total += fp
            fn_total += fn
            counts["images_with_seg_metrics"] += 1
            record["segmentation"] = {
                "status": "scored",
                "iou": segmentation_metrics(tp, fp, fn)["iou"],
                "gt_overlap_px": int(gt["overlap_px"]),
            }

        # --- optional perturbation stability probes (demo-trained models)
        if perturb:
            _probe_stability(record, image, image_path, features, predictors,
                             stability)

        counts["images_processed"] += 1
        record["status"] = "ok"
        per_image.append(record)

    micro = segmentation_metrics(tp_total, fp_total, fn_total)
    stability_out = _finalize_stability(stability)

    report = {
        "report": "real_cv_validation",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_class": EVIDENCE_CLASS,
        "status": "OK",
        "blocked_reason": None,
        "synthetic_demo_substituted": False,
        "dataset": {
            "name": "CelebAMask-HQ",
            "official_url": license_state["official_url"],
            "license_gate": license_state["license_gate"],
            "note": "non-commercial research only; masks are geometry, never "
                    "colour ground truth (DATASET_REGISTRY.md)",
        },
        "provenance": _provenance(root, dataset),
        "selection": {
            "limit": limit,
            "offset": offset,
            "deterministic": True,
            "method": "evenly spaced indices over the sorted image list; "
                      "--offset slides the window",
        },
        "annotation_status": _annotation_status(root),
        "counts": counts,
        "per_image": per_image,
        "segmentation_metrics": {
            "scope": "face box only (Haar box crop); GT neck/chin outside the "
                     "box is not counted",
            "images_scored": counts["images_with_seg_metrics"],
            "micro": micro,
            "validates": "skin-mask composition vs annotation geometry",
            "does_not_validate": "undertone / skin-depth / season accuracy",
        },
        "perturbation_stability": stability_out if perturb else None,
        "probe_predictors": {
            axis: {"artifact": info["artifact"], "trained_on": info["trained_on"]}
            for axis, info in predictors.items()
        } if perturb else None,
        "model_evaluation": {
            "status": "BLOCKED",
            "reason": (
                "no expert-labelled real data exists (annotation table empty); "
                "external dataset labels cannot become FaceMaxx colour ground "
                "truth"
            ),
            "valid_real_labelled_samples": _annotation_status(root)[
                "annotated_images"
            ],
            "synthetic_demo_substituted": False,
        },
        "limitations": REAL_CV_LIMITATIONS,
    }
    return report


# ---------------------------------------------------------------------------
# Perturbation probe plumbing
# ---------------------------------------------------------------------------

def _finite(value: float) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, 4) if np.isfinite(value) else None


def _probe_stability(
    record: Dict[str, Any],
    image: "np.ndarray",
    image_path: Path,
    features: Dict[str, Any],
    predictors: Dict[str, Dict[str, Any]],
    stability: Dict[str, Any],
) -> None:
    """Perturb ONE image in memory; record feature deltas + prediction flips.

    Derived variants never enter any dataset, split, or manifest.
    """
    import tempfile

    from .colour import extract_image as _extract

    base_preds = {
        axis: _predict_safe(info["predictor"], features)
        for axis, info in predictors.items()
    }
    base_labels = {
        axis: (out or {}).get("prediction") for axis, out in base_preds.items()
    }

    variants = build_perturbations(image)
    if not variants:
        return
    stability["images_probed"] += 1
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "variant.jpg"
        for name, variant in variants.items():
            try:
                import cv2

                ok, encoded = cv2.imencode(".jpg", variant)
                if not ok:
                    continue
                tmp_path.write_bytes(encoded.tobytes())
            except Exception:
                continue
            vext = _extract(f"{image_path.stem}-{name}", tmp_path)
            if not vext.ok:
                continue
            vfeatures = vext.as_feature_row()
            for key in _STABILITY_FEATURES:
                base_val, var_val = features.get(key), vfeatures.get(key)
                try:
                    if base_val is not None and var_val is not None and \
                            np.isfinite(base_val) and np.isfinite(var_val):
                        stability["feature_abs_deltas"][key].append(
                            abs(float(var_val) - float(base_val))
                        )
                except (TypeError, ValueError):
                    continue
            for axis, info in predictors.items():
                vout = _predict_safe(info["predictor"], vfeatures)
                vlabel = (vout or {}).get("prediction")
                bucket = stability["prediction_flips"].setdefault(
                    axis, {"flips": 0, "compared": 0}
                )
                if base_labels[axis] is not None and vlabel is not None:
                    bucket["compared"] += 1
                    if vlabel != base_labels[axis]:
                        bucket["flips"] += 1
    record["perturbation_probed"] = True
    record["baseline_predictions"] = {
        axis: label for axis, label in base_labels.items()
    }


def _finalize_stability(stability: Dict[str, Any]) -> Dict[str, Any]:
    deltas = {}
    for key, values in stability["feature_abs_deltas"].items():
        deltas[key] = {
            "n_compared": len(values),
            "mean_abs_delta": round(sum(values) / len(values), 4) if values else None,
        }
    flips = {}
    for axis, bucket in stability["prediction_flips"].items():
        flips[axis] = {
            "flips": bucket["flips"],
            "compared": bucket["compared"],
            "flip_rate": (
                round(bucket["flips"] / bucket["compared"], 4)
                if bucket["compared"] else None
            ),
        }
    return {
        "images_probed": stability["images_probed"],
        "feature_mean_abs_delta": deltas,
        "prediction_flips": flips,
        "note": "flip counts measure stability of SYNTHETIC-DEMO-TRAINED "
                "models on real inputs, not correctness (no ground truth)",
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_real_cv(report: Dict[str, Any]) -> str:
    lines: List[str] = [
        "REAL-IMAGE CV VALIDATION (Category B)",
        "=" * 46,
        f"evidence class: {report['evidence_class']}",
        f"status:         {report['status']}",
    ]
    if report["status"] == "BLOCKED":
        lines += [
            "",
            "REAL-DATA TEST: BLOCKED",
            f"Reason: {report['blocked_reason']}",
            "Synthetic demo results were not substituted.",
            "",
            "Demo evaluation remains available as an explicitly selected "
            "separate mode (`demo-image`).",
        ]
        return "\n".join(lines)

    ds = report["dataset"]
    lines += [
        f"dataset:        {ds['name']} ({ds['license_gate']}) {ds['official_url']}",
        f"receipt:        sha256 {str(report['provenance'].get('sha256'))[:16]}... "
        f"recorded {report['provenance'].get('recorded')}",
        f"selection:      {report['selection']['limit']} images, offset "
        f"{report['selection']['offset']} (deterministic)",
        "",
        "counts:",
    ]
    for key, value in report["counts"].items():
        lines.append(f"  {key:<24} {value}")

    seg = report["segmentation_metrics"]
    micro = seg["micro"]
    lines += [
        "",
        f"segmentation vs GT (micro over {seg['images_scored']} scored images, "
        f"{seg['scope']}):",
        f"  IoU {micro['iou']}  Dice {micro['dice']}  "
        f"precision {micro['precision']}  recall {micro['recall']}",
        "  (validates mask composition vs annotation geometry; NOT colour "
        "accuracy)",
    ]

    stability = report.get("perturbation_stability")
    if stability:
        lines += ["", f"perturbation stability ({stability['images_probed']} images probed):"]
        for key, stats in stability["feature_mean_abs_delta"].items():
            lines.append(f"  {key:<22} mean|delta| {stats['mean_abs_delta']} "
                         f"(n={stats['n_compared']})")
        for axis, stats in stability["prediction_flips"].items():
            lines.append(f"  {axis:<22} prediction flips {stats['flips']}/"
                         f"{stats['compared']} (rate {stats['flip_rate']})")

    ann = report["annotation_status"]
    lines += [
        "",
        "model evaluation (Category C):",
        f"  status: {report['model_evaluation']['status']} - "
        f"{report['model_evaluation']['reason']}",
        f"  expert-labelled real images available: "
        f"{ann['annotated_images']} (people: {ann['people']})",
        "  Synthetic demo results were not substituted.",
        "",
        "limitations:",
    ]
    for lim in report["limitations"]:
        lines.append(f"  - {lim}")
    return "\n".join(lines)
