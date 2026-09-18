"""Real-data quality & annotation-readiness audit.

The LAST analysis before ML training. Sits between the annotation workflow
and any model fitting, and answers - with numbers, not impressions:

  1. Are all 12 seasons represented?
  2. Are skin-depth groups sufficiently represented?
  3. Which seasons/depths/axis labels are under-represented?
  4. Where do annotators disagree: seasonal BOUNDARIES (expected, informative)
     or fundamental CONFLICTS (label poison)?
  5. Do certain capture conditions (lighting/camera/quality/makeup)
     destabilise labels?
  6. Are multiple photos of one person grouped and labelled consistently?
  7. Is the locked holdout genuinely untouched?
  8. Are there enough examples for each axis (undertone/depth/chroma/contrast)?
  9. Are the labels actually usable for supervised learning?

Verdict tiers:
  NOT READY    - hard blockers (the ML gate conditions fail)
  SMOKE-READY  - enough clean, grouped, holdout-protected data to smoke-test
                 the training wiring (NOT a real benchmark; the gate minimum)
  TRAIN-READY  - additionally all 12 seasons + every depth/axis label covered,
                 agreement high, no conflicts, no unstable conditions

This module fabricates nothing and never edits data: under-representation is
reported so humans can collect the missing data.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from facemax_bias.vocab import (
    CHROMA_ORDER,
    CONTRAST_ORDER,
    SKIN_DEPTH_ORDER,
    SEASON_ORDER,
    UNDERTONE_ORDER,
)

from .dataset import init_dataset, read_annotations
from .validate import lint_rows

# ---------------------------------------------------------------------------
# Thresholds (deliberately explicit - they ARE the readiness definition)
# ---------------------------------------------------------------------------

# Smoke tier = the frozen ML-gate minimums (see __main__.cmd_gate): >= 40
# annotated images from >= 20 people + clean lint + audit PASS + locked holdout.
# 40 images is a wiring smoke test, NOT a training set.

TRAIN_MIN_SEASON = 10        # annotated examples per season for full training
TRAIN_MIN_DEPTH = 5          # annotated examples per skin-depth group
TRAIN_MIN_AXIS_LABEL = 8     # annotated examples per undertone/chroma/contrast label
TRAIN_MIN_MEAN_AGREEMENT = 0.80
TRAIN_MIN_MULTI_ANNOTATOR_FRAC = 0.5  # agreement must be measurable on most rows

STABILITY_MIN_ROWS = 5       # judge a condition only with >= this many rows
STABILITY_RATE = 0.25        # dissent rate that flags a capture condition

# Seasons considered each other's NEAREST neighbours on the seasonal wheel:
# they differ on exactly one axis (temperature, depth or chroma). Annotator
# disagreement BETWEEN neighbours is expected at boundaries and is useful
# data; disagreement between non-neighbours signals a labelling problem.
SEASON_NEIGHBOURS: Dict[str, frozenset] = {
    "Light Spring": frozenset({"Bright Spring", "Warm Spring", "Light Summer"}),
    "Bright Spring": frozenset({"Light Spring", "Warm Spring", "Light Winter"}),
    "Warm Spring": frozenset({"Light Spring", "Bright Spring", "Warm Autumn", "Light Summer"}),
    "Light Summer": frozenset({"Light Spring", "Soft Summer", "Cool Summer", "Light Winter"}),
    "Soft Summer": frozenset({"Light Summer", "Cool Summer", "Soft Autumn"}),
    "Cool Summer": frozenset({"Light Summer", "Soft Summer", "Cool Winter"}),
    "Soft Autumn": frozenset({"Soft Summer", "Warm Autumn", "Deep Autumn"}),
    "Warm Autumn": frozenset({"Warm Spring", "Soft Autumn", "Deep Autumn"}),
    "Deep Autumn": frozenset({"Soft Autumn", "Warm Autumn", "Deep Winter"}),
    "Light Winter": frozenset({"Bright Spring", "Light Summer", "Cool Winter", "Deep Winter"}),
    "Cool Winter": frozenset({"Cool Summer", "Light Winter", "Deep Winter"}),
    "Deep Winter": frozenset({"Deep Autumn", "Cool Winter", "Light Winter"}),
}


def _pair(a: str, b: str) -> str:
    return " <-> ".join(sorted((a, b)))


def season_distance(a: str, b: str) -> Optional[int]:
    """1 for wheel-neighbours (boundary pair), None for unrelated seasons."""
    if a == b:
        return 0
    if b in SEASON_NEIGHBOURS.get(a, frozenset()):
        return 1
    return None


# ---------------------------------------------------------------------------
# report model
# ---------------------------------------------------------------------------


@dataclass
class ReadinessReport:
    status: str = "NOT READY"
    blockers: List[str] = field(default_factory=list)   # why NOT READY at all
    train_gaps: List[str] = field(default_factory=list)  # why not TRAIN-READY
    advisories: List[str] = field(default_factory=list)

    annotated_rows: int = 0
    pending_rows: int = 0
    people: int = 0

    season_counts: Dict[str, int] = field(default_factory=dict)
    missing_seasons: List[str] = field(default_factory=list)
    underrepresented_seasons: List[str] = field(default_factory=list)

    depth_counts: Dict[str, int] = field(default_factory=dict)
    underrepresented_depths: List[str] = field(default_factory=list)

    axis_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    underrepresented_axis_labels: Dict[str, List[str]] = field(default_factory=dict)

    multi_annotator_rows: int = 0
    mean_agreement: float = 0.0
    boundary_pairs: Dict[str, int] = field(default_factory=dict)  # "A <-> B": n
    boundary_disagreements: int = 0
    conflicting_disagreements: int = 0

    condition_dissent: Dict[str, Dict[str, float]] = field(default_factory=dict)
    unstable_conditions: List[str] = field(default_factory=list)

    multi_image_people: int = 0
    inconsistent_people: List[str] = field(default_factory=list)
    missing_person_ids: int = 0

    holdout_untouched: bool = False
    holdout_test_without_truth: int = 0
    holdout_detail: Dict[str, Any] = field(default_factory=dict)

    lint_invalid_annotated: int = 0

    # -- helpers ------------------------------------------------------------

    @property
    def dissent_rows(self) -> int:
        return self.boundary_disagreements + self.conflicting_disagreements

    @property
    def conflict_rate(self) -> float:
        if not self.multi_annotator_rows:
            return 0.0
        return self.conflicting_disagreements / self.multi_annotator_rows

    @property
    def boundary_rate(self) -> float:
        if not self.dissent_rows:
            return 0.0
        return self.boundary_disagreements / self.dissent_rows

    def to_json(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "blockers": self.blockers,
            "train_gaps": self.train_gaps,
            "advisories": self.advisories,
            "annotated_rows": self.annotated_rows,
            "pending_rows": self.pending_rows,
            "people": self.people,
            "season_counts": self.season_counts,
            "missing_seasons": self.missing_seasons,
            "underrepresented_seasons": self.underrepresented_seasons,
            "depth_counts": self.depth_counts,
            "underrepresented_depths": self.underrepresented_depths,
            "axis_counts": self.axis_counts,
            "underrepresented_axis_labels": self.underrepresented_axis_labels,
            "multi_annotator_rows": self.multi_annotator_rows,
            "mean_agreement": round(self.mean_agreement, 4),
            "boundary_pairs": self.boundary_pairs,
            "boundary_disagreements": self.boundary_disagreements,
            "conflicting_disagreements": self.conflicting_disagreements,
            "condition_dissent": self.condition_dissent,
            "unstable_conditions": self.unstable_conditions,
            "multi_image_people": self.multi_image_people,
            "inconsistent_people": self.inconsistent_people,
            "missing_person_ids": self.missing_person_ids,
            "holdout_untouched": self.holdout_untouched,
            "holdout_test_without_truth": self.holdout_test_without_truth,
            "holdout_detail": self.holdout_detail,
            "lint_invalid_annotated": self.lint_invalid_annotated,
        }

    # -- rendering ----------------------------------------------------------

    def render(self) -> str:
        lines = ["FACEMAXX DATA READINESS AUDIT", "=" * 50]
        lines.append(f"annotated rows: {self.annotated_rows}   pending: {self.pending_rows}   people: {self.people}")
        lines.append("")
        lines.append(f"VERDICT: {self.status}")
        for blocker in self.blockers:
            lines.append(f"  BLOCKED: {blocker}")
        for gap in self.train_gaps:
            lines.append(f"  TRAIN GAP: {gap}")
        for advisory in self.advisories:
            lines.append(f"  note: {advisory}")

        lines.append("")
        lines.append("1-3. SEASON COVERAGE (all 12 must exist for training)")
        width = max(len(s) for s in SEASON_ORDER)
        for season in SEASON_ORDER:
            count = self.season_counts.get(season, 0)
            mark = ""
            if count == 0:
                mark = "  <-- MISSING"
            elif season in self.underrepresented_seasons:
                mark = f"  <-- under-represented (< {TRAIN_MIN_SEASON})"
            lines.append(f"  {season:<{width}} {count:>5}  {'#' * min(count, 60)}{mark}")

        lines.append("")
        lines.append("2. SKIN-DEPTH COVERAGE")
        for depth in SKIN_DEPTH_ORDER:
            count = self.depth_counts.get(depth, 0)
            mark = "  <-- MISSING" if count == 0 else ""
            if count and depth in self.underrepresented_depths:
                mark = f"  <-- under-represented (< {TRAIN_MIN_DEPTH})"
            lines.append(f"  {depth:<{width}} {count:>5}  {'#' * min(count, 60)}{mark}")

        lines.append("")
        lines.append("8. AXIS COVERAGE (undertone / chroma / contrast)")
        for axis, order in (
            ("undertone", UNDERTONE_ORDER),
            ("chroma_level", CHROMA_ORDER),
            ("contrast_level", CONTRAST_ORDER),
        ):
            counts = self.axis_counts.get(axis, {})
            missing = self.underrepresented_axis_labels.get(axis, [])
            summary = ", ".join(f"{label}:{counts.get(label, 0)}" for label in order)
            lines.append(f"  {axis:<15} {summary}")
            if missing:
                lines.append(f"                  under-represented: {', '.join(missing)}")

        lines.append("")
        lines.append("4. ANNOTATOR DISAGREEMENT (multi-annotator rows only)")
        lines.append(f"  multi-annotator rows:  {self.multi_annotator_rows}")
        lines.append(f"  mean agreement:        {self.mean_agreement:.3f}")
        lines.append(f"  disagreement rows:     {self.dissent_rows}")
        lines.append(
            f"  boundary disagreements:{self.boundary_disagreements:>4} "
            f"({self.boundary_rate * 100:.0f}% of dissent)"
        )
        lines.append(
            f"  conflicting disagreements:{self.conflicting_disagreements:>3} "
            f"({(1 - self.boundary_rate) * 100:.0f}% of dissent)"
        )
        if self.boundary_pairs:
            lines.append("  disagreement by season pair:")
            for pair, count in sorted(self.boundary_pairs.items(), key=lambda kv: -kv[1])[:8]:
                kind = "boundary" if " <-> " in pair and self._pair_is_boundary(pair) else "CONFLICT"
                lines.append(f"    {pair}: {count}  ({kind})")
        if self.conflicting_disagreements:
            lines.append("  ! conflicts are label poison: re-annotate or retire these rows before training")

        lines.append("")
        lines.append("5. CAPTURE-CONDITION LABEL STABILITY")
        if self.condition_dissent:
            for column, values in self.condition_dissent.items():
                for value, rate in values.items():
                    flag = "  <-- UNSTABLE" if f"{column}={value}" in self.unstable_conditions else ""
                    lines.append(f"  {column:<15} {value:<15} dissent {rate * 100:5.1f}%{flag}")
        else:
            lines.append("  (not yet measurable - needs multi-annotator rows)")

        lines.append("")
        lines.append("6. PERSON GROUPING & CONSISTENCY")
        lines.append(f"  people:                     {self.people}")
        lines.append(f"  people with 2+ images:      {self.multi_image_people}")
        lines.append(f"  rows with no person_id:     {self.missing_person_ids}")
        if self.inconsistent_people:
            lines.append(f"  people with CONFLICTING seasons across their photos: {len(self.inconsistent_people)}")
            for entry in self.inconsistent_people[:8]:
                lines.append(f"    ! {entry}")

        lines.append("")
        lines.append("7. LOCKED HOLDOUT")
        state = self.holdout_detail.get("state", "unknown")
        lines.append(f"  state: {state}")
        lines.append(f"  untouched: {'yes' if self.holdout_untouched else 'NO'}")
        if self.holdout_test_without_truth:
            lines.append(
                f"  ! {self.holdout_test_without_truth} test-split images have no season truth"
            )
        for problem in self.holdout_detail.get("problems", [])[:5]:
            lines.append(f"  ! {problem}")

        lines.append("")
        lines.append("9. USABILITY FOR SUPERVISED LEARNING")
        lines.append(
            f"  annotated rows failing row lint: {self.lint_invalid_annotated}"
            + ("  <-- fix before training" if self.lint_invalid_annotated else "")
        )
        lines.append("")
        lines.append(f"overall: {self.status}")
        return "\n".join(lines)

    def _pair_is_boundary(self, pair: str) -> bool:
        a, b = pair.split(" <-> ")
        return season_distance(a, b) == 1


def save_report(report: ReadinessReport, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "readiness_report.txt").write_text(report.render() + "\n", encoding="utf-8")
    (out_dir / "readiness_report.json").write_text(
        json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def _row_labels(row: Dict[str, Any]) -> List[str]:
    """Independent annotator season labels carried on the merged row."""
    return [
        str(row.get(f"annotator_{i}") or "").strip()
        for i in (1, 2, 3)
        if str(row.get(f"annotator_{i}") or "").strip()
    ]


def _majority_fraction(labels: List[str]) -> float:
    counts: Dict[str, int] = defaultdict(int)
    for label in labels:
        counts[label] += 1
    return max(counts.values()) / len(labels)


def _disagreement_analysis(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, int], int, int, float, int]:
    """Boundary vs conflict classification over merged annotator labels.

    Returns (boundary_pairs, boundary_count, conflict_count, mean_agreement,
    multi_annotator_rows). Only rows with 2+ annotator labels qualify.
    """
    pairs: Dict[str, int] = defaultdict(int)
    boundary = conflict = 0
    agreements: List[float] = []
    multi = 0
    for row in rows:
        labels = _row_labels(row)
        if len(labels) < 2:
            continue
        multi += 1
        agreements.append(_majority_fraction(labels))
        distinct = sorted(set(labels))
        if len(distinct) < 2:
            continue  # unanimous
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                a, b = distinct[i], distinct[j]
                pair = _pair(a, b)
                pairs[pair] += 1
                if season_distance(a, b) == 1:
                    boundary += 1
                else:
                    conflict += 1
    mean_agreement = sum(agreements) / len(agreements) if agreements else 0.0
    return dict(pairs), boundary, conflict, mean_agreement, multi


def _condition_stability(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Dissent rate per capture condition; flag conditions that destabilise labels.

    A condition is UNSTABLE when it has >= STABILITY_MIN_ROWS multi-annotator
    rows and its dissent rate >= STABILITY_RATE (dissent = any annotator
    disagreement, i.e. agreement < 1.0).
    """
    by_condition: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        labels = _row_labels(row)
        if len(labels) < 2:
            continue
        dissent = 1 if len(set(labels)) > 1 else 0
        for column in ("lighting_type", "camera_type", "image_quality", "makeup"):
            value = str(row.get(column) or "").strip()
            if value:
                by_condition[column][value].append(dissent)
    rates: Dict[str, Dict[str, float]] = {}
    unstable: List[str] = []
    for column, values in sorted(by_condition.items()):
        rates[column] = {}
        for value, flags in sorted(values.items()):
            rate = sum(flags) / len(flags)
            rates[column][value] = round(rate, 4)
            if len(flags) >= STABILITY_MIN_ROWS and rate >= STABILITY_RATE:
                unstable.append(f"{column}={value}")
    return rates, unstable


def _person_consistency(rows: List[Dict[str, Any]]) -> Tuple[int, List[str], int]:
    """Multi-image people with conflicting merged seasons across their photos."""
    by_person: Dict[str, List[str]] = defaultdict(list)
    missing = 0
    for row in rows:
        person = str(row.get("person_id") or "").strip()
        season = str(row.get("season_primary") or "").strip()
        if not person:
            missing += 1
            continue
        if season:
            by_person[person].append(season)
    multi = 0
    inconsistent: List[str] = []
    for person, seasons in sorted(by_person.items()):
        if len(seasons) < 2:
            continue
        multi += 1
        distinct = sorted(set(seasons))
        if len(distinct) > 1:
            inconsistent.append(f"{person}: {', '.join(distinct)}")
    return multi, inconsistent, missing


def _holdout_status(root: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Is the locked holdout genuinely untouched, and is it fully labelled?"""
    from .audit import verify_holdout

    check = verify_holdout(root)
    detail: Dict[str, Any] = {"state": "not locked", "problems": list(check.get("problems", []))}
    if not check.get("locked"):
        return detail
    detail["state"] = "locked+verified" if check.get("ok") else "LOCK BROKEN"

    # Read the holdout membership from the manifest (its digest is verified
    # above), so this works with any existing lock without format changes.
    test_manifest = root / "splits" / "test.txt"
    test_ids = (
        {line.strip() for line in test_manifest.read_text(encoding="utf-8").splitlines() if line.strip()}
        if test_manifest.exists()
        else set()
    )
    by_id = {str(r.get("image_id") or ""): r for r in rows}
    missing_rows = sorted(tid for tid in test_ids if tid not in by_id)
    without_truth = sorted(
        tid for tid in test_ids
        if tid in by_id and not str(by_id[tid].get("season_primary") or "").strip()
    )
    detail["test_images"] = len(test_ids)
    if missing_rows:
        detail["problems"].append(f"{len(missing_rows)} test rows deleted from annotation table since lock")
    detail["test_without_truth"] = len(without_truth)
    detail["test_ids_missing"] = missing_rows[:10]
    return detail


def readiness(root: Path) -> ReadinessReport:
    """Run every readiness analysis on the real annotation table."""
    from .annotation import coverage
    from .audit import audit_dataset

    root = Path(root)
    init_dataset(root)
    all_rows = read_annotations(root / "metadata" / "facemaxx_color_annotations.csv")
    annotated = [r for r in all_rows if str(r.get("season_primary") or "").strip()]
    pending = len(all_rows) - len(annotated)

    report = ReadinessReport()
    report.annotated_rows = len(annotated)
    report.pending_rows = pending
    report.people = len({str(r.get("person_id") or "").strip() for r in all_rows} - {""})

    # ---- coverage matrices -------------------------------------------------
    report.season_counts = {s: 0 for s in SEASON_ORDER}
    report.depth_counts = {d: 0 for d in SKIN_DEPTH_ORDER}
    report.axis_counts = {
        "undertone": {u: 0 for u in UNDERTONE_ORDER},
        "chroma_level": {c: 0 for c in CHROMA_ORDER},
        "contrast_level": {c: 0 for c in CONTRAST_ORDER},
    }
    for row in annotated:
        season = str(row.get("season_primary") or "").strip()
        if season in report.season_counts:
            report.season_counts[season] += 1
        depth = str(row.get("skin_depth") or "").strip()
        if depth in report.depth_counts:
            report.depth_counts[depth] += 1
        for axis in report.axis_counts:
            value = str(row.get(axis) or "").strip()
            if value in report.axis_counts[axis]:
                report.axis_counts[axis][value] += 1

    report.missing_seasons = [s for s in SEASON_ORDER if report.season_counts[s] == 0]
    report.underrepresented_seasons = [
        s for s in SEASON_ORDER if 0 < report.season_counts[s] < TRAIN_MIN_SEASON
    ]
    report.underrepresented_depths = [
        d for d in SKIN_DEPTH_ORDER if 0 < report.depth_counts[d] < TRAIN_MIN_DEPTH
    ]
    report.underrepresented_axis_labels = {
        axis: [
            label for label, count in counts.items()
            if 0 < count < TRAIN_MIN_AXIS_LABEL
        ]
        for axis, counts in report.axis_counts.items()
    }

    # ---- disagreement / stability / people ---------------------------------
    pairs, boundary, conflict, mean_agreement, multi = _disagreement_analysis(annotated)
    report.boundary_pairs = pairs
    report.boundary_disagreements = boundary
    report.conflicting_disagreements = conflict
    report.mean_agreement = mean_agreement
    report.multi_annotator_rows = multi

    report.condition_dissent, report.unstable_conditions = _condition_stability(annotated)
    report.multi_image_people, report.inconsistent_people, report.missing_person_ids = (
        _person_consistency(annotated)
    )

    # ---- holdout ------------------------------------------------------------
    report.holdout_detail = _holdout_status(root, all_rows)
    report.holdout_untouched = bool(
        report.holdout_detail.get("state") == "locked+verified"
        and not report.holdout_detail.get("problems")
        and not report.holdout_detail.get("test_without_truth")
    )
    report.holdout_test_without_truth = int(report.holdout_detail.get("test_without_truth") or 0)

    # ---- lint (annotated rows only; pending rows cannot be linted) ----------
    lint = lint_rows(annotated)
    report.lint_invalid_annotated = lint.total_rows - lint.valid_rows

    # ---- verdict ------------------------------------------------------------
    cov = coverage(root)
    audit = audit_dataset(root, rows=all_rows)

    if cov["images_annotated"] < 40:
        report.blockers.append(
            f"only {cov['images_annotated']} annotated images (< 40 gate minimum)"
        )
    if cov["people"] < 20:
        report.blockers.append(f"only {cov['people']} people (< 20 gate minimum)")
    if report.lint_invalid_annotated:
        report.blockers.append(
            f"{report.lint_invalid_annotated} annotated rows fail row lint (run `validate`)"
        )
    if not audit.ok():
        report.blockers.append("dataset audit FAIL (run `audit` for details)")
    if not report.holdout_untouched:
        state = report.holdout_detail.get("state", "not locked")
        report.blockers.append(f"holdout not a verified untouched lock (state: {state})")
    if report.holdout_test_without_truth:
        report.blockers.append(
            f"{report.holdout_test_without_truth} holdout test images have no season truth"
        )
    if pending:
        report.advisories.append(
            f"{pending} imported images still pending annotation (excluded from training)"
        )

    if report.blockers:
        report.status = "NOT READY"
        return report

    # Smoke tier reached: gate conditions satisfied. Now the training tier.
    report.status = "SMOKE-READY"
    if report.missing_seasons:
        report.train_gaps.append(
            f"{len(report.missing_seasons)} seasons have zero examples: "
            + ", ".join(report.missing_seasons)
        )
    if report.underrepresented_seasons:
        report.train_gaps.append(
            f"{len(report.underrepresented_seasons)} seasons below {TRAIN_MIN_SEASON} examples: "
            + ", ".join(report.underrepresented_seasons)
        )
    if report.underrepresented_depths:
        report.train_gaps.append(
            f"skin depths below {TRAIN_MIN_DEPTH} examples: "
            + ", ".join(report.underrepresented_depths)
        )
    for axis, labels in report.underrepresented_axis_labels.items():
        if labels:
            report.train_gaps.append(
                f"{axis} labels below {TRAIN_MIN_AXIS_LABEL} examples: " + ", ".join(labels)
            )
    if report.multi_annotator_rows < report.annotated_rows * TRAIN_MIN_MULTI_ANNOTATOR_FRAC:
        report.train_gaps.append(
            f"only {report.multi_annotator_rows}/{report.annotated_rows} rows multi-annotated; "
            "agreement is not yet measurable on most of the data"
        )
    elif report.mean_agreement < TRAIN_MIN_MEAN_AGREEMENT:
        report.train_gaps.append(
            f"mean agreement {report.mean_agreement:.3f} < {TRAIN_MIN_MEAN_AGREEMENT}"
        )
    if report.conflicting_disagreements:
        report.train_gaps.append(
            f"{report.conflicting_disagreements} conflicting (non-boundary) annotator disagreements"
        )
    if report.inconsistent_people:
        report.train_gaps.append(
            f"{len(report.inconsistent_people)} people have conflicting seasons across their own photos"
        )
    if report.unstable_conditions:
        report.train_gaps.append(
            "unstable capture conditions: " + ", ".join(report.unstable_conditions)
        )

    if not report.train_gaps:
        report.status = "TRAIN-READY"
    return report
