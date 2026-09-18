"""Dataset distribution analysis: representation bias across every major axis."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .vocab import MAJOR_AXES, GROUP_SIZE_WARN


def _counter(rows: List[Dict], column: str) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        value = str(row.get(column) or "").strip()
        if value:
            counter[value] += 1
    return counter


def distribution(rows: List[Dict], columns: List[str] | None = None) -> Dict[str, Dict[str, Any]]:
    """Count images per category for each axis; returns counts + shares."""
    columns = list(columns) if columns is not None else list(MAJOR_AXES)
    total = len(rows)
    result: Dict[str, Dict[str, Any]] = {}
    for column in columns:
        counter = _counter(rows, column)
        order = MAJOR_AXES.get(column)
        labels = [l for l in (order or sorted(counter)) if counter.get(l)]
        extra = [l for l in counter if l not in (order or [])]
        labels += sorted(extra)  # keep unrecognised labels visible instead of hiding them
        result[column] = {
            "total": total,
            "counts": {label: counter[label] for label in labels},
            "shares": {label: (counter[label] / total if total else 0.0) for label in labels},
        }
    return result


def flag_underrepresented(
    dist: Dict[str, Dict[str, Any]],
    min_share: float = 0.03,
    min_count: int = GROUP_SIZE_WARN,
) -> List[str]:
    """Human-readable flags for categories with too few examples."""
    flags: List[str] = []
    for column, info in dist.items():
        for label, count in info["counts"].items():
            share = info["shares"][label]
            if count < min_count or share < min_share:
                flags.append(
                    f"{column}/{label}: {count} images ({share * 100:.1f}%) "
                    f"- under-represented (need >={min_count}, >={min_share * 100:.0f}%)"
                )
    return flags


def imbalance_ratio(dist: Dict[str, Dict[str, Any]], column: str) -> float:
    """max/min frequency ratio for one axis (inf when the smallest bucket is empty)."""
    counts = [c for c in dist[column]["counts"].values() if c > 0]
    if not counts:
        return float("inf")
    return max(counts) / min(counts)


def summary(dist: Dict[str, Dict[str, Any]]) -> str:
    lines = [f"Total images: {dist['season_primary']['total'] if 'season_primary' in dist else '?'}"]
    for column, info in dist.items():
        lines.append("")
        lines.append(column.replace("_", " ").title() + ":")
        for label, count in info["counts"].items():
            lines.append(f"  {label:<15} {count:>6}  {info['shares'][label] * 100:5.1f}%")
    return "\n".join(lines)
