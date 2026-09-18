"""Person-grouped K-fold cross-validation helpers.

GroupKFold without sklearn: folds are assigned by person_id so no person's
images straddle train/test folds.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple


def group_kfold(
    rows: List[Dict],
    k: int = 5,
    seed: int = 0,
) -> List[Tuple[int, List[Dict], List[Dict]]]:
    """Yield (fold_index, train_rows, eval_rows) with person-disjoint folds.

    Deterministic for a given seed.
    """
    people = sorted({str(r.get("person_id") or "") for r in rows if r.get("person_id")})
    rng = random.Random(seed)
    rng.shuffle(people)

    assignment: Dict[str, int] = {}
    for i, person in enumerate(people):
        assignment[person] = i % k

    folds: List[Tuple[int, List[Dict], List[Dict]]] = []
    for fold in range(k):
        train = [r for r in rows if assignment.get(str(r.get("person_id") or "")) != fold]
        eval_rows = [r for r in rows if assignment.get(str(r.get("person_id") or "")) == fold]
        folds.append((fold, train, eval_rows))
    return folds


def fold_sizes(folds: List[Tuple[int, List[Dict], List[Dict]]]) -> List[Dict[str, int]]:
    """Rows per fold, for a quick sanity print."""
    return [
        {"fold": fold, "train": len(train), "eval": len(eval_rows)}
        for fold, train, eval_rows in folds
    ]
