"""Build the Model Bias & Robustness Report after every training run.

Usage:
    from facemax_bias.report import generate_report
    report = generate_report(rows, predictions=rows, prediction_key="prediction")
    print(report.text)

``rows``: annotation records (see ml/README.md for the CSV schema). Each row
may carry the model's outputs: ``prediction``, ``prediction_secondary`` and
``prediction_confidence``. Rows without a prediction are excluded from
performance sections but still counted in the distribution section.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import labels as labels_mod
from . import leakage as leakage_mod
from . import robustness as robustness_mod
from . import schema as schema_mod
from . import subgroups as subgroups_mod
from .distribution import distribution as dataset_distribution
from .distribution import flag_underrepresented, imbalance_ratio
from .recommendations import actions_from_report, render_actions
from .metrics import accuracy, balanced_accuracy, confusion_labels, confusion_matrix, macro_f1, per_class_rates
from .vocab import (
    GROUP_SIZE_WARN,
    MAJOR_AXES,
    QUALITY_ORDER,
    SEASON_ORDER,
    SKIN_DEPTH_ORDER,
    normalize_quality,
)


class BiasReport:
    """Container for the computed report with .text and .as_dict()."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    @property
    def text(self) -> str:
        return render_text(self.data)

    def as_dict(self) -> Dict[str, Any]:
        return self.data

    def save_json(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, default=str)


def _pct(value: Optional[float]) -> str:
    return f"{value * 100:5.1f}%" if value is not None else "  n/a "


def _bar(rate: float, width: int = 24) -> str:
    filled = int(round(rate * width))
    return "#" * filled + "." * (width - filled)


def generate_report(
    rows: List[Dict],
    prediction_key: str = "prediction",
    truth_key: str = "season_primary",
    secondary_key: str = "prediction_secondary",
    title: str = "FACEMAXX MODEL FAIRNESS REPORT",
    lock_condition: Optional[str] = None,
) -> BiasReport:
    """Compute every section of the bias dashboard from annotated rows."""
    # Derive majority_label / annotation_agreement from annotator_1..3 when absent.
    for row in rows:
        labels_mod.annotate_row(row)
    predicted_rows = [r for r in rows if r.get(prediction_key) not in (None, "")]

    distribution = dataset_distribution(rows)
    schema_report = schema_mod.validate(rows)
    leakage = leakage_mod.leakage_check(rows)
    subgroup_report = subgroups_mod.fairness_gaps(
        predicted_rows, prediction_key, truth_key
    )
    confusion = _confusion(predicted_rows, prediction_key, truth_key)
    overall = _overall(predicted_rows, prediction_key, truth_key, secondary_key)
    label_report = labels_mod.label_source_accuracy(predicted_rows, prediction_key)
    agreement = labels_mod.agreement_distribution(rows)
    ambiguous = labels_mod.ambiguous_rows(rows)
    boundaries = labels_mod.seasonal_boundary_stats(rows)
    lighting = robustness_mod.condition_stability(
        predicted_rows, "lighting_type", prediction_key, truth_key, secondary_key
    )
    camera = robustness_mod.condition_stability(
        predicted_rows, "camera_type", prediction_key, truth_key, secondary_key
    )
    makeup = robustness_mod.makeup_filter_risk(predicted_rows, prediction_key, truth_key)
    quality = robustness_mod.quality_bucket_performance(
        predicted_rows, prediction_key, truth_key
    )
    floor_label, floor_message = robustness_mod.recommend_quality_floor(quality)

    data: Dict[str, Any] = {
        "title": title,
        "generated_for": lock_condition or "all conditions",
        "total_images": len(rows),
        "scored_images": len(predicted_rows),
        "distribution": distribution,
        "representation_flags": flag_underrepresented(distribution),
        "imbalance_ratios": {
            column: imbalance_ratio(distribution, column)
            for column in ("skin_depth", "season_primary", "source")
            if column in distribution
        },
        "schema": {
            "missing_required": schema_report.missing_required,
            "duplicate_images": len(schema_report.duplicate_images),
            "unknown_values": schema_report.unknown_values,
        },
        "leakage": leakage,
        "overall": overall,
        "subgroups": subgroup_report,
        "confusion": confusion,
        "label_quality": {
            "agreement_distribution": agreement,
            "ambiguous_count": len(ambiguous),
            "ambiguous_examples": ambiguous[:20],
            "season_boundaries": [
                {"primary": a, "secondary": b, "count": c} for (a, b), c in boundaries[:10]
            ],
            "truth_source_comparison": label_report,
        },
        "robustness": {
            "lighting": lighting,
            "camera": camera,
            "makeup_filters": makeup,
            "image_quality": quality,
            "recommended_quality_floor": {
                "bucket": floor_label,
                "message": floor_message,
            },
        },
    }
    return BiasReport(data)


def _overall(rows: List[Dict], prediction_key: str, truth_key: str, secondary_key: str) -> Dict[str, Any]:
    actual = [str(r[truth_key]) for r in rows if r.get(truth_key)]
    predicted = [str(r.get(prediction_key) or "") for r in rows if r.get(truth_key)]
    runner_up = [str(r.get(secondary_key) or "") for r in rows if r.get(truth_key)]
    return {
        "accuracy": accuracy(actual, predicted),
        "macro_f1": macro_f1(actual, predicted),
        "balanced_accuracy": balanced_accuracy(actual, predicted),
        "top2_accuracy": _top2(actual, predicted, runner_up),
        "n": len(actual),
    }


def _top2(actual: List[str], predicted: List[str], runner_up: List[str]) -> Optional[float]:
    if not actual:
        return None
    hits = sum(a in (p, r) for a, p, r in zip(actual, predicted, runner_up))
    return hits / len(actual)


def _confusion(rows: List[Dict], prediction_key: str, truth_key: str) -> Dict[str, Any]:
    actual = [str(r[truth_key]) for r in rows if r.get(truth_key)]
    predicted = [str(r.get(prediction_key) or "") for r in rows if r.get(truth_key)]
    labels = confusion_labels(actual, predicted)
    matrix = confusion_matrix(actual, predicted)
    total_actual = [sum(row) for row in matrix]
    # Row-normalized rates per (actual -> predicted) cell.
    pairs = []
    for i, actual_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            if i == j or not total_actual[i]:
                continue
            rate = matrix[i][j] / total_actual[i]
            if rate >= 0.10:
                pairs.append(
                    {
                        "actual": actual_label,
                        "predicted": predicted_label,
                        "rate": rate,
                        "count": matrix[i][j],
                    }
                )
    pairs.sort(key=lambda p: -p["rate"])
    return {"labels": labels, "matrix": matrix, "top_confusions": pairs[:12]}


def render_text(data: Dict[str, Any]) -> str:
    """Render the console dashboard exactly like the spec sheet."""
    lines: List[str] = []
    width = 56

    def rule(char: str = "=") -> str:
        return char * width

    def section(name: str) -> None:
        lines.append("")
        lines.append("-" * width)
        lines.append(name)
        lines.append("-" * width)

    lines.append(rule())
    lines.append(data.get("title", "FACEMAXX MODEL FAIRNESS REPORT").center(width))
    lines.append(rule())
    lines.append(f"Total images:      {data['total_images']}")
    lines.append(f"Scored images:     {data['scored_images']}")

    overall = data.get("overall") or {}
    if overall.get("n"):
        lines.append("")
        lines.append(f"Overall Accuracy:    {_pct(overall['accuracy'])}")
        lines.append(f"Macro F1:            {_pct(overall['macro_f1'])}")
        lines.append(f"Balanced Accuracy:   {_pct(overall['balanced_accuracy'])}")
        if overall.get("top2_accuracy") is not None:
            lines.append(f"Top-2 Accuracy:      {_pct(overall['top2_accuracy'])}")

    section("REPRESENTATION (dataset distribution)")
    dist = data.get("distribution") or {}
    for column in ("skin_depth", "undertone", "contrast", "chroma", "lighting_type", "camera_type", "image_quality", "source", "makeup", "season_primary"):
        info = dist.get(column)
        if not info or not info["counts"]:
            continue
        lines.append(f"{column.replace('_', ' ').title()}:")
        for label, count in info["counts"].items():
            share = info["shares"][label]
            flag = "  <-- low" if share < 0.03 else ""
            lines.append(f"  {label:<16} {count:>6}  {share * 100:5.1f}%{flag}")
    for flag in (data.get("representation_flags") or [])[:8]:
        lines.append(f"  ! {flag}")

    section("SUBGROUP PERFORMANCE (accuracy by visual group)")
    for attribute, block in (data.get("subgroups") or {}).items():
        perf = block.get("performance") or {}
        if not perf:
            continue
        title = attribute.replace("_", " ").title()
        lines.append("")
        lines.append(f"{title}:")
        for label, m in perf.items():
            mark = "  (small n)" if m.get("small_n") else ""
            lines.append(
                f"  {label:<16} {_pct(m['accuracy'])}  (n={int(m['n'])}){mark}"
            )
        wb = block.get("worst_best")
        if wb:
            lines.append(
                f"  gap: {wb['best_group']} {_pct(wb['best'])} vs "
                f"{wb['worst_group']} {_pct(wb['worst'])} -> {abs(wb['gap']) * 100:.1f} pts "
                f"[{block.get('verdict')}]"
            )

    section("CONFUSION (top season confusions, row-normalized)")
    confusion = data.get("confusion") or {}
    top = confusion.get("top_confusions") or []
    if top:
        for pair in top:
            lines.append(
                f"  {pair['actual']:<14} -> {pair['predicted']:<14} {pair['rate'] * 100:5.1f}%  ({pair['count']})"
            )
    else:
        lines.append("  (no confusion pairs >= 10%)")

    section("LABEL QUALITY (annotator agreement)")
    lq = data.get("label_quality") or {}
    agreement = lq.get("agreement_distribution") or {}
    total_agreement = sum(agreement.values())
    if total_agreement:
        for key in ("unanimous", "majority", "split"):
            count = agreement.get(key, 0)
            lines.append(f"  {key:<10} {count:>6}  ({count / total_agreement * 100:.1f}%)")
        lines.append(f"  ambiguous images flagged: {lq.get('ambiguous_count')}")
        for boundary in (lq.get("season_boundaries") or [])[:5]:
            lines.append(
                f"  boundary: {boundary['primary']} <-> {boundary['secondary']}  ({boundary['count']} images)"
            )
        comparison = lq.get("truth_source_comparison") or {}
        if comparison:
            lines.append("  accuracy by truth source:")
            for source, value in comparison.items():
                lines.append(f"    {source:<14} {_pct(value)}")

    section("ROBUSTNESS (stability across conditions)")
    robustness = data.get("robustness") or {}
    for name, key in (
        ("Lighting", "lighting"),
        ("Camera", "camera"),
        ("Makeup/Filters", "makeup_filters"),
        ("Image Quality", "image_quality"),
    ):
        stats = robustness.get(key) or {}
        if not stats:
            continue
        lines.append("")
        lines.append(f"{name}:")
        for label, m in stats.items():
            extra = f"  top2-stable {_pct(m['top2_stability'])}" if m.get("top2_stability") is not None else ""
            lines.append(f"  {label:<16} {_pct(m['accuracy'])}  (n={int(m['n'])}){extra}")

    floor = (robustness.get("recommended_quality_floor") or {})
    if floor.get("message"):
        lines.append("")
        lines.append(f"  quality gate: {floor['message']}")

    leakage = data.get("leakage") or {}
    section("SPLIT INTEGRITY (duplicate-people leakage)")
    lines.append(f"  leakage-free splits: {'YES' if leakage.get('clean') else 'NO'}")
    for split, counts in (leakage.get("person_counts") or {}).items():
        lines.append(f"  {split:<8} people={counts}")
    for person, split_a, split_b in (leakage.get("pairs") or [])[:10]:
        lines.append(f"  ! {person} appears in both {split_a} and {split_b}")

    lines.append("")
    lines.append(rule())
    lines.append(render_actions(actions_from_report(data)))
    lines.append(rule())
    return "\n".join(lines)
