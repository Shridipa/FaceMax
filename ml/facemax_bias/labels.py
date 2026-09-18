"""Annotator agreement and ambiguity handling.

Label bias check: store every annotator's label, compute majority + agreement,
and flag inherently ambiguous images instead of forcing them into one season.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

from .metrics import accuracy


def parse_annotators(row: Dict) -> List[str]:
    labels: List[str] = []
    for i in (1, 2, 3):
        value = str(row.get(f"annotator_{i}") or "").strip()
        if value:
            labels.append(value)
    return labels


def majority_label(labels: List[str]) -> Optional[str]:
    if not labels:
        return None
    counter = Counter(labels)
    top_count = max(counter.values())
    winners = sorted(l for l, c in counter.items() if c == top_count)
    return winners[0] if len(winners) == 1 else None  # None = split vote


def agreement_score(labels: List[str]) -> float:
    """Fraction of annotators agreeing with the majority (1.0 when unanimous)."""
    if not labels:
        return 0.0
    counter = Counter(labels)
    return max(counter.values()) / len(labels)


def annotate_row(row: Dict) -> Dict:
    """Attach majority_label / annotation_agreement if not already present."""
    labels = parse_annotators(row)
    row.setdefault("majority_label", majority_label(labels) or row.get("season_primary"))
    row.setdefault(
        "annotation_agreement",
        row.get("annotation_agreement") or agreement_score(labels),
    )
    return row


def agreement_distribution(rows: List[Dict]) -> Dict[str, int]:
    buckets = {"unanimous": 0, "majority": 0, "split": 0}
    for row in rows:
        labels = parse_annotators(row)
        if not labels:
            continue
        score = agreement_score(labels)
        if score >= 1.0:
            buckets["unanimous"] += 1
        elif len(set(labels)) == len(labels):
            buckets["split"] += 1
        else:
            buckets["majority"] += 1
    return buckets


def ambiguous_rows(rows: List[Dict], threshold: float = 0.99) -> List[str]:
    """image_ids where annotators were not unanimous (majority or split).

    Default threshold 0.99 flags any non-unanimous image; pass e.g. 0.67 to
    flag only two-thirds-agreement-or-worse cases.
    """
    flagged = []
    for row in rows:
        labels = parse_annotators(row)
        if labels and agreement_score(labels) < threshold:
            flagged.append(str(row.get("image_id")))
    return flagged


def seasonal_boundary_stats(rows: List[Dict]) -> List[Tuple[str, str, int]]:
    """Most common (primary, secondary) pairs — reveals seasonal boundaries."""
    counter: Counter = Counter()
    for row in rows:
        primary = str(row.get("season_primary") or "").strip()
        secondary = str(row.get("season_secondary") or "").strip()
        if primary and secondary and primary != secondary:
            counter[(primary, secondary)] += 1
    return counter.most_common()


def label_source_accuracy(
    rows: List[Dict],
    prediction_key: str = "prediction",
    use_majority: bool = True,
) -> Dict[str, float]:
    """Compare model accuracy against majority-label truth vs. single-annotator truth.

    A large gap means disagreement-heavy images are distorting the headline number.
    """
    results: Dict[str, float] = {}
    for name, truth_key in (
        ("majority_label", "majority_label"),
        ("annotator_1", "annotator_1"),
    ):
        if use_majority and truth_key == "majority_label":
            eligible = [r for r in rows if r.get("majority_label")]
        else:
            eligible = [r for r in rows if r.get("annotator_1")]
        if not eligible:
            continue
        actual = [str(r[truth_key]) for r in eligible]
        predicted = [str(r.get(prediction_key) or "") for r in eligible]
        results[name] = accuracy(actual, predicted)
    return results
