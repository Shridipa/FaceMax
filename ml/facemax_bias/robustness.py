"""Robustness & stability tests: lighting, camera, makeup, image quality.

Uses per-person comparisons wherever possible (same face, different capture
conditions) and falls back to cohort-level comparisons otherwise.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .metrics import accuracy, macro_f1
from .vocab import MAJOR_AXES, normalize_lighting, normalize_quality, quality_at_least


def _normalized(row: Dict, column: str) -> str:
    value = str(row.get(column) or "").strip()
    if column == "lighting_type":
        return normalize_lighting(value)
    if column == "image_quality":
        return normalize_quality(value)
    return value


def condition_stability(
    rows: List[Dict],
    condition_column: str,
    prediction_key: str = "prediction",
    truth_key: str = "season_primary",
    secondary_key: str = "prediction_secondary",
) -> Dict[str, Dict[str, float]]:
    """Per-condition accuracy plus a stability score.

    Stability = fraction of rows where the *ranked* prediction (primary or
    secondary) contains the truth, i.e. how often the model's shortlist is
    stable even when its top pick flips.
    """
    buckets: Dict[str, List[Dict]] = {}
    for row in rows:
        key = _normalized(row, condition_column)
        if not key or row.get(truth_key) in (None, ""):
            continue
        buckets.setdefault(key, []).append(row)

    result: Dict[str, Dict[str, float]] = {}
    for key, bucket in buckets.items():
        actual = [str(r[truth_key]) for r in bucket]
        predicted = [str(r.get(prediction_key) or "") for r in bucket]
        n = len(bucket)
        top2_hits = sum(
            1
            for r in bucket
            if str(r.get(truth_key)) in {
                str(r.get(prediction_key) or ""),
                str(r.get(secondary_key) or ""),
            }
        )
        result[key] = {
            "n": float(n),
            "accuracy": accuracy(actual, predicted),
            "macro_f1": macro_f1(actual, predicted),
            "top2_stability": top2_hits / n,
        }
    return result


def person_flip_rate(
    rows: List[Dict],
    condition_column: str,
    prediction_key: str = "prediction",
) -> Optional[float]:
    """Fraction of same-person pairs whose prediction changes across a condition.

    The gold-standard illumination-stability test: the same face photographed
    under different lights should not change season.
    """
    by_person: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        pid = str(row.get("person_id") or "").strip()
        cond = _normalized(row, condition_column)
        if pid and cond and row.get(prediction_key):
            by_person[pid].append(row)

    pairs = flips = 0
    for images in by_person.values():
        conditions: Dict[str, List[Dict]] = defaultdict(list)
        for image in images:
            conditions[_normalized(image, condition_column)].append(image)
        if len(conditions) < 2:
            continue
        cond_keys = sorted(conditions)
        for i in range(len(cond_keys)):
            for j in range(i + 1, len(cond_keys)):
                left = conditions[cond_keys[i]][0]
                right = conditions[cond_keys[j]][0]
                pairs += 1
                if str(left.get(prediction_key)) != str(right.get(prediction_key)):
                    flips += 1
    return flips / pairs if pairs else None


def quality_bucket_performance(
    rows: List[Dict],
    prediction_key: str = "prediction",
    truth_key: str = "season_primary",
) -> Dict[str, Dict[str, float]]:
    """Accuracy by image-quality bucket → drives the refuse-to-predict gate."""
    return condition_stability(
        rows, "image_quality", prediction_key, truth_key
    )


def recommend_quality_floor(
    quality_perf: Dict[str, Dict[str, float]],
    min_accuracy: float = 0.75,
) -> Tuple[Optional[str], Optional[str]]:
    """Choose the lowest quality bucket that still clears min_accuracy.

    Scans from the worst bucket upward: the floor is the first (worst) bucket
    that fails min_accuracy, raised one step so everything better is allowed.
    Returns (floor_label, message). quality_at_least() can then be used by the
    product to decide when to show a low-confidence warning instead of a season.
    """
    from .vocab import QUALITY_ORDER

    order = list(QUALITY_ORDER)
    for i in range(len(order) - 1, -1, -1):  # worst -> best
        bucket = order[i]
        stats = quality_perf.get(bucket)
        if not stats or stats.get("n", 0) < 1:
            continue  # skip buckets we have no data for
        if stats["accuracy"] < min_accuracy:
            if i == 0:
                return None, (
                    f"Even '{bucket}' images clear {min_accuracy * 100:.0f}% "
                    f"- no quality gate needed."
                )
            floor = order[i - 1]
            return floor, (
                f"Predictions below '{floor}' quality fall under {min_accuracy * 100:.0f}% "
                f"accuracy - show a low-confidence warning instead of a firm season."
            )
    return None, f"All measured quality buckets clear {min_accuracy * 100:.0f}%."


def makeup_filter_risk(
    rows: List[Dict],
    prediction_key: str = "prediction",
    truth_key: str = "season_primary",
) -> Dict[str, Dict[str, float]]:
    """Accuracy for makeup/filter categories; the product layer can use this to
    display 'strong colour filter detected' style warnings."""
    return condition_stability(
        rows, "makeup", prediction_key, truth_key
    )
