"""Subgroup performance and fairness-gap analysis.

Implements the "does the model work equally well across visual groups?" test:
per-subgroup accuracy/macro-F1 for a chosen attribute column, worst-case
subgroups, worst-vs-best gap, and acceptance thresholds.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .metrics import accuracy, macro_f1
from .vocab import GROUP_SIZE_WARN, MAJOR_AXES


# Engineering targets (documented in ml/README.md) — not universal scientific
# thresholds, just FaceMaxx acceptance criteria.
STRONG_MIN_RATE = 0.80
STRONG_MAX_GAP = 0.10
INVESTIGATE_MIN_RATE = 0.70
INVESTIGATE_MAX_GAP = 0.15


def subgroup_performance(
    rows: List[Dict],
    attribute: str,
    prediction_key: str = "prediction",
    truth_key: str = "season_primary",
    order: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Accuracy + macro-F1 per subgroup of ``attribute``.

    Subgroups with fewer than GROUP_SIZE_WARN rows are flagged via "n" but
    still reported (small-n results should be read with suspicion).
    """
    order = order or list(MAJOR_AXES.get(attribute, []))
    buckets: Dict[str, List[Dict]] = {}
    for row in rows:
        key = str(row.get(attribute) or "").strip()
        if not key:
            continue
        if row.get(truth_key) in (None, ""):
            continue
        buckets.setdefault(key, []).append(row)

    extra = sorted(set(buckets) - set(order))
    ordered = [label for label in order if label in buckets] + extra

    result: Dict[str, Dict[str, float]] = {}
    for label in ordered:
        bucket = buckets[label]
        actual = [str(r[truth_key]) for r in bucket]
        predicted = [str(r.get(prediction_key) or "") for r in bucket]
        result[label] = {
            "n": float(len(bucket)),
            "accuracy": accuracy(actual, predicted),
            "macro_f1": macro_f1(actual, predicted),
            "small_n": 1.0 if len(bucket) < GROUP_SIZE_WARN else 0.0,
        }
    return result


def worst_best(
    performance: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
    min_n: int = GROUP_SIZE_WARN,
) -> Optional[Dict[str, float]]:
    """Worst and best subgroup by metric, ignoring tiny (n < min_n) groups."""
    eligible = {
        label: m
        for label, m in performance.items()
        if m.get("n", 0) >= min_n
    }
    if not eligible:
        return None
    worst_label = min(eligible, key=lambda k: eligible[k][metric])
    best_label = max(eligible, key=lambda k: eligible[k][metric])
    return {
        "worst_group": worst_label,
        "worst": eligible[worst_label][metric],
        "best_group": best_label,
        "best": eligible[best_label][metric],
        "gap": eligible[best_label][metric] - eligible[worst_label][metric],
    }


def verdict(rate: float, gap: float) -> str:
    """Classify a subgroup axis against the FaceMaxx engineering targets."""
    if rate >= STRONG_MIN_RATE and gap <= STRONG_MAX_GAP:
        return "strong"
    if rate >= INVESTIGATE_MIN_RATE and gap <= INVESTIGATE_MAX_GAP:
        return "needs investigation"
    return "serious problem"


def fairness_gaps(
    rows: List[Dict],
    prediction_key: str = "prediction",
    truth_key: str = "season_primary",
    attributes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Run subgroup analysis across every requested attribute axis."""
    attributes = list(attributes) if attributes is not None else list(MAJOR_AXES)
    report: Dict[str, Dict[str, object]] = {}
    for attribute in attributes:
        perf = subgroup_performance(rows, attribute, prediction_key, truth_key)
        wb = worst_best(perf)
        report[attribute] = {
            "performance": perf,
            "worst_best": wb,
            "verdict": verdict(wb["worst"], wb["gap"]) if wb else "insufficient data",
        }
    return report
