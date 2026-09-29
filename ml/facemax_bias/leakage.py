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


def grouped_split_class_aware(
    rows: List[Dict],
    test_frac: float = 0.2,
    seed: int = 0,
    label_column: str = "season_primary",
    val_frac: float = 0.15,
) -> List[Dict]:
    """Person-grouped split that TRIES to keep every label class represented.

    Motivation (ML spec #6/#24): a val split that misses a class cannot be used
    for calibration, and a test split that misses a class silently reports
    perfect recall for it. Purely random person splits hit that often on small
    data. This variant still keeps every person whole (mandatory) but chooses
    WHICH whole people go to val/test to preserve class coverage.

    Algorithm: order people by label diversity (then image count), then
    round-robin assign one person per uncovered label to test, then to val,
    and fill the remaining quota deterministically. If full coverage is not
    achievable with whole people, it falls back to the plain grouped_split
    (honest behaviour: coverage is attempted, never manufactured by splitting
    a person's images across splits or duplicating rows).
    """
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    people = sorted({str(r.get("person_id") or "") for r in rows if r.get("person_id")})
    if not people:
        return grouped_split(rows, test_frac=test_frac, seed=seed)
    rng.shuffle(people)

    by_person = defaultdict(list)
    for row in rows:
        person = str(row.get("person_id") or "")
        if person:
            by_person[person].append(row)

    def labels_of(person: str) -> Set[str]:
        return {
            str(r.get(label_column) or "").strip()
            for r in by_person[person]
        } - {""}

    all_labels = {
        str(r.get(label_column) or "").strip() for r in rows
    } - {""}

    # Label-diverse people first: one such person covers several classes at
    # once, which is what makes whole-person class coverage possible.
    order = sorted(people, key=lambda p: (-len(labels_of(p)), -len(by_person[p]), p))

    n_test = int(len(people) * test_frac)
    n_val = max(1, int(len(people) * val_frac))
    test_people: Set[str] = set()
    val_people: Set[str] = set()

    # Round-robin: cover each label in test first, then val.
    for target_set in (test_people, val_people):
        covered = set().union(*(labels_of(p) for p in target_set)) if target_set else set()
        for label in sorted(all_labels):
            if label in covered:
                continue
            for person in order:
                if person in test_people or person in val_people:
                    continue
                if label in labels_of(person):
                    target_set.add(person)
                    covered |= labels_of(person)
                    break

    # Fill the remaining quota deterministically (no overlaps).
    for person in order:
        if len(test_people) >= n_test:
            break
        if person not in val_people:
            test_people.add(person)
    for person in order:
        if len(val_people) >= n_val:
            break
        if person not in test_people:
            val_people.add(person)

    def covers(split_people: Set[str]) -> Set[str]:
        have: Set[str] = set()
        for person in split_people:
            have |= labels_of(person)
        return have

    if all_labels and (not all_labels <= covers(test_people) or not all_labels <= covers(val_people)):
        # Whole-person coverage impossible - fall back to the plain policy
        # rather than manufacturing coverage.
        return grouped_split(rows, test_frac=test_frac, seed=seed)

    for row in rows:
        person = str(row.get("person_id") or "")
        if person in test_people:
            row["split"] = "test"
        elif person in val_people:
            row["split"] = "val"
        else:
            row["split"] = "train"
    return rows


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
