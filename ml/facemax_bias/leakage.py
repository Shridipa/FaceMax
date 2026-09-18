"""Person-level split integrity (duplicate-people leakage).

The dataset mistake: images of the same person scattered across train/val/test
inflate test scores. All images of one person_id must stay in a single split.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set


def split_by_person(rows: List[Dict]) -> Dict[str, Set[str]]:
    """Map split -> set of person_ids that appear in it."""
    mapping: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        split = str(row.get("split") or "unsplit")
        person = str(row.get("person_id") or "").strip()
        if person:
            mapping[split].add(person)
    return mapping


def leakage_check(rows: List[Dict]) -> Dict[str, object]:
    """Find person_ids spanning multiple splits.

    Returns {"leaked_person_ids": [...], "pairs": [(person, split_a, split_b)...],
    "clean": bool, "person_counts": {split: n_persons}}.
    """
    spans: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        person = str(row.get("person_id") or "").strip()
        split = str(row.get("split") or "unsplit")
        if person:
            spans[person].add(split)

    leaked = sorted(p for p, s in spans.items() if len(s) > 1)
    pairs = []
    for person in leaked:
        parts = sorted(spans[person])
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                pairs.append((person, parts[i], parts[j]))

    mapping = split_by_person(rows)
    return {
        "leaked_person_ids": leaked,
        "pairs": pairs,
        "clean": not leaked,
        "person_counts": {split: len(people) for split, people in sorted(mapping.items())},
    }


def grouped_split(rows: List[Dict], test_frac: float = 0.2, seed: int = 0) -> List[Dict]:
    """Re-split rows so every person lands in exactly one split.

    Deterministic for a given seed; writes a 'split' field onto each row dict.
    """
    import random

    rng = random.Random(seed)
    people = sorted({str(r.get("person_id") or "") for r in rows if r.get("person_id")})
    rng.shuffle(people)
    n_test = int(len(people) * test_frac)
    test_people = set(people[:n_test])
    rest = people[n_test:]
    n_val = int(len(rest) * 0.15)
    val_people = set(rest[:n_val])

    for row in rows:
        person = str(row.get("person_id") or "")
        if person in test_people:
            row["split"] = "test"
        elif person in val_people:
            row["split"] = "val"
        else:
            row["split"] = "train"
    return rows


def split_stats(rows: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Images + people per split — sanity check before training/eval."""
    images: Dict[str, int] = defaultdict(int)
    people: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        split = str(row.get("split") or "unsplit")
        images[split] += 1
        person = str(row.get("person_id") or "").strip()
        if person:
            people[split].add(person)
    return {
        split: {"images": images[split], "people": len(people[split])}
        for split in sorted(images)
    }
