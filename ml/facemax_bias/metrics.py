"""Metric helpers: per-group metrics, macro F1, balanced accuracy, confusion matrix.

Implemented with the Python standard library so the pipeline runs anywhere;
pandas/numpy/sklearn can wrap these later without changing semantics.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Label = str


def confusion_matrix(actual: Sequence[Label], predicted: Sequence[Label]) -> List[List[int]]:
    labels = sorted(set(actual) | set(predicted))
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for a, p in zip(actual, predicted):
        matrix[index[a]][index[p]] += 1
    return matrix


def confusion_labels(actual: Sequence[Label], predicted: Sequence[Label]) -> List[Label]:
    return sorted(set(actual) | set(predicted))


def per_class_rates(
    actual: Sequence[Label],
    predicted: Sequence[Label],
    labels: Optional[Iterable[Label]] = None,
) -> Dict[Label, Dict[str, float]]:
    """Per-class recall/precision (and support) for each label."""
    labels = list(labels) if labels is not None else sorted(set(actual) | set(predicted))
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for a, p in zip(actual, predicted):
        if a == p:
            tp[a] += 1
        else:
            fp[p] += 1
            fn[a] += 1
    result: Dict[Label, Dict[str, float]] = {}
    for label in labels:
        support = tp[label] + fn[label]
        pred_total = tp[label] + fp[label]
        recall = tp[label] / support if support else 0.0
        precision = tp[label] / pred_total if pred_total else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        result[label] = {
            "support": float(support),
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
    return result


def macro_f1(actual: Sequence[Label], predicted: Sequence[Label]) -> float:
    rates = per_class_rates(actual, predicted)
    scores = [m["f1"] for m in rates.values() if m["support"] > 0]
    return sum(scores) / len(scores) if scores else 0.0


def accuracy(actual: Sequence[Label], predicted: Sequence[Label]) -> float:
    if not actual:
        return 0.0
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def balanced_accuracy(actual: Sequence[Label], predicted: Sequence[Label]) -> float:
    rates = per_class_rates(actual, predicted)
    recalls = [m["recall"] for m in rates.values() if m["support"] > 0]
    return sum(recalls) / len(recalls) if recalls else 0.0


def top2_accuracy(actual: Sequence[Label], predicted: Sequence[Label], runner_up: Sequence[Label]) -> float:
    if not actual:
        return 0.0
    hits = sum(a in (p, r) for a, p, r in zip(actual, predicted, runner_up))
    return hits / len(actual)


def subgroup_metric(
    rows: Iterable[Dict],
    prediction_key: str,
    truth_key: str = "season_primary",
    group_key: str = "skin_depth",
) -> Dict[str, Dict[str, float]]:
    """Split rows by a column, then compute accuracy / macro-F1 per subgroup.

    Rows with missing truth are skipped. Returned dict maps subgroup label ->
    {"n": support, "accuracy": ..., "macro_f1": ...}.
    """
    buckets: Dict[str, List[Dict]] = {}
    for row in rows:
        key = str(row.get(group_key) or "").strip() or "Unknown"
        if row.get(truth_key) in (None, ""):
            continue
        buckets.setdefault(key, []).append(row)

    result: Dict[str, Dict[str, float]] = {}
    for key, bucket in buckets.items():
        actual = [str(row[truth_key]) for row in bucket]
        predicted = [str(row.get(prediction_key) or "") for row in bucket]
        result[key] = {
            "n": float(len(bucket)),
            "accuracy": accuracy(actual, predicted),
            "macro_f1": macro_f1(actual, predicted),
        }
    return result

