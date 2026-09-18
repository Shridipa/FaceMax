"""Recommendation engine: turn a bias report into remediation actions.

Governing principle (from the FaceMaxx fairness spec):

    Subgroup under-performance is a DATA/MODEL problem. It is remediated by
    collecting better, more varied data and improving the model -- never by
    lowering user-facing confidence for the affected group.

    Image-quality warnings are the ONLY confidence-gating mechanism. They
    reflect the capture condition (blur, exposure, compression) and gate
    every user equally.

This module inspects a ``facemax_bias.report.generate_report`` dict and emits
prioritized, actionable items. It deliberately does NOT propose per-group
confidence scaling.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .subgroups import INVESTIGATE_MAX_GAP, INVESTIGATE_MIN_RATE, STRONG_MAX_GAP, STRONG_MIN_RATE

# Suggested minimum additions per under-represented subgroup (engineering
# target; tune as the dataset matures).
MIN_TARGET_PER_SUBGROUP = 200


def _fmt(value: Any) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else str(value)


def actions_from_report(report: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """Produce {'collect': [...], 'model': [...], 'gate': [...]} action lists."""
    collect: List[Dict[str, str]] = []
    model: List[Dict[str, str]] = []
    gate: List[Dict[str, str]] = []

    # ---- Representation gaps -> data collection -------------------------
    for flag in report.get("representation_flags", []):
        collect.append(
            {
                "action": "collect_data",
                "target": flag,
                "why": "under-represented subgroup; the model cannot learn it from a handful of images",
            }
        )

    # ---- Subgroup gaps -> data AND model work ---------------------------
    subgroups = report.get("subgroups", {})
    for attribute, block in subgroups.items():
        wb = block.get("worst_best")
        if not wb:
            continue
        worst = wb["worst_group"]
        gap = wb["gap"]
        perf = wb["worst"]
        if perf < INVESTIGATE_MIN_RATE or gap > INVESTIGATE_MAX_GAP:
            collect.append(
                {
                    "action": "collect_data",
                    "target": f"{attribute}/{worst}",
                    "why": (
                        f"worst subgroup {_fmt(perf)} vs best {_fmt(wb['best'])} "
                        f"(gap {gap * 100:.1f} pts) - add genuinely varied examples, not duplicates"
                    ),
                }
            )
            model.append(
                {
                    "action": "model_improvement",
                    "target": attribute,
                    "why": (
                        f"audit features/preprocessing for {attribute}: check LAB sampling, "
                        f"white balance and segmentation errors concentrated on '{worst}'"
                    ),
                }
            )

    # ---- Confusion structure -> model/feature work ----------------------
    for pair in (report.get("confusion", {}) or {}).get("top_confusions", []):
        if pair["rate"] >= 0.20:
            model.append(
                {
                    "action": "model_improvement",
                    "target": f"{pair['actual']} -> {pair['predicted']}",
                    "why": (
                        f"{pair['rate'] * 100:.0f}% of {pair['actual']} predicted as "
                        f"{pair['predicted']}; add boundary examples and re-check undertone features"
                    ),
                }
            )

    # ---- Label quality -> annotation process ----------------------------
    label_quality = report.get("label_quality", {}) or {}
    agreement = label_quality.get("agreement_distribution", {}) or {}
    total = sum(agreement.values())
    if total and agreement.get("split", 0) / max(total, 1) > 0.05:
        collect.append(
            {
                "action": "annotation_process",
                "target": "split-vote images",
                "why": f"{agreement['split']} images had no majority label - add a 4th annotator or arbitration",
            }
        )

    # ---- Quality -> the ONLY legitimate confidence gate ------------------
    robustness = report.get("robustness", {}) or {}
    floor = robustness.get("recommended_quality_floor", {}) or {}
    if floor.get("message"):
        gate.append(
            {
                "action": "quality_gate",
                "target": floor.get("bucket") or "all uploads",
                "why": floor["message"],
            }
        )

    return {"collect": collect, "model": model, "gate": gate}


def render_actions(actions: Dict[str, List[Dict[str, str]]]) -> str:
    """Text block appended to the fairness report."""
    lines = ["RECOMMENDED ACTIONS (subgroup bias => data/model fixes; quality => gate)"]
    labels = {
        "collect": "DATA COLLECTION",
        "model": "MODEL / FEATURE WORK",
        "gate": "QUALITY GATE (the only confidence mechanism)",
    }
    for key in ("collect", "model", "gate"):
        items = actions.get(key, [])
        if not items:
            continue
        lines.append("")
        lines.append(f"  {labels[key]}:")
        for item in items[:12]:
            lines.append(f"    - [{item['action']}] {item['target']}")
            lines.append(f"        {item['why']}")
    if len(lines) == 1:
        lines.append("  (no actions required - all checks passed)")
    return "\n".join(lines)
