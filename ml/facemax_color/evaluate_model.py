"""Locked-holdout evaluation (the LAST step of the ML milestone).

Sequence enforced by the CLI:

    train  -> model selection via person-grouped CV  (holdout untouched)
    evaluate -> ONE final pass over the LOCKED holdout  (this module)

The holdout is never used for tuning: this module has no training code at
all. It loads a saved artifact, scores the frozen test split, and writes
metrics back into the artifact (metrics.json + confusion_matrix.json) so the
model card and registry reflect the final unbiased numbers.

Robustness reuses the FROZEN evaluator (facemax_bias.subgroups) - verdicts,
thresholds and the GROUP_SIZE_WARN small-sample flag stay canonical.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .training import (
    SeasonPredictor,
    boundary_analysis,
    calibration_report,
    compute_metrics,
    robustness_report,
)
from .training_data import TARGETS, TrainingData


class EvaluationError(RuntimeError):
    pass


def load_predictor(models_root: Path, model_id: str) -> Tuple[SeasonPredictor, Dict[str, Any]]:
    """Load a saved artifact by id; refuses unknown/mismatched ids."""
    import joblib

    parts = model_id.split("-")
    if len(parts) < 3:
        raise EvaluationError(f"malformed model_id {model_id!r} (expected <target>-<type>-vNNN)")
    target = parts[0]
    artifact_dir = Path(models_root) / target / model_id
    model_path = artifact_dir / "model.joblib"
    meta_path = artifact_dir / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise EvaluationError(
            f"artifact not found: {artifact_dir} (train first; see `train --help`)"
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    predictor = joblib.load(model_path)
    return predictor, metadata


def _prob_matrix(
    predictor: SeasonPredictor,
    rows: List[Dict[str, Any]],
    labels: List[str],
) -> np.ndarray:
    """Row probabilities aligned to the canonical label order."""
    matrix = np.zeros((len(rows), len(labels)), dtype=np.float64)
    for i, row in enumerate(rows):
        features = {c: row.get(c) for c in predictor.feature_columns}
        out = predictor.predict_row(features)
        probs = out.get("probabilities") or {}
        for j, label in enumerate(labels):
            matrix[i, j] = float(probs.get(label, 0.0))
    return matrix


def run_holdout_evaluation(
    data: TrainingData,
    models_root: Path,
    model_id: str,
    bins: int = 15,
    data_source_expected: Optional[str] = None,
) -> Dict[str, Any]:
    """Score the LOCKED holdout rows with a trained artifact.

    ``data`` must come from prepare_training_data(...) on the SAME annotation +
    feature tables the model was trained on (its holdout_rows carry the frozen
    test split). This function never trains and never tunes.
    """
    predictor, metadata = load_predictor(models_root, model_id)
    if data_source_expected and metadata.get("data_source") != data_source_expected:
        raise EvaluationError(
            f"artifact {model_id} was trained on {metadata.get('data_source')!r} data "
            f"but is being evaluated against {data_source_expected!r} data - refusing "
            "to mix demo and real artifacts"
        )
    if metadata.get("target") != data.target:
        raise EvaluationError(
            f"artifact target {metadata.get('target')!r} != prepared data target {data.target!r}"
        )

    spec = TARGETS[data.target]
    labels = list(spec["order"])
    rows = [
        r for r in data.holdout_rows
        if r.get("_label_ok") and r.get("_person_ok")
    ]
    if not rows:
        raise EvaluationError(
            "no evaluable holdout rows (labelled, person-tagged, feature-backed); "
            "the holdout cannot be scored"
        )

    truth: List[str] = []
    preds: List[str] = []
    for row in rows:
        features = {c: row.get(c) for c in predictor.feature_columns}
        out = predictor.predict_row(features)
        row["prediction"] = out["prediction"]
        row["prediction_secondary"] = out["prediction_secondary"]
        row["prediction_confidence"] = out["prediction_confidence"]
        truth.append(str(row[data.label_column]))
        preds.append(out["prediction"])

    metrics = compute_metrics(truth, preds, labels)

    # Calibration scores on holdout probabilities (the honest test of the
    # val-fitted calibrator; never refitted here).
    prob_matrix = _prob_matrix(predictor, rows, labels)
    label_index = {label: i for i, label in enumerate(labels)}
    y_idx = np.array([label_index[t] for t in truth])
    calib_scores = calibration_report(prob_matrix, y_idx, bins)

    # Boundary analysis (season only - the wheel is a season concept).
    boundary = boundary_analysis(truth, preds) if data.target == "season" else None

    # Robustness via the FROZEN evaluator (small-n flags preserved).
    robustness = robustness_report(rows, truth_key=data.label_column)

    evaluated = {
        "model_id": model_id,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "holdout_rows": len(rows),
        "holdout_people": len({str(r.get("person_id")) for r in rows}),
        "metrics": metrics,
        "calibration_scores": calib_scores,
        "boundary": boundary,
        "robustness": robustness,
    }

    # Write results back into the artifact so the card/registry stay truthful.
    artifact_dir = Path(models_root) / data.target / model_id
    metrics_path = artifact_dir / "metrics.json"
    stored: Dict[str, Any] = {}
    if metrics_path.exists():
        try:
            stored = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            stored = {}
    stored.update(
        {
            "holdout": metrics,
            "calibration_scores": calib_scores,
            "boundary": boundary,
            "robustness": robustness,
            "holdout_evaluated_at": evaluated["evaluated_at"],
            "holdout_rows": len(rows),
            "holdout_people": evaluated["holdout_people"],
        }
    )
    metrics_path.write_text(json.dumps(stored, indent=2, default=str) + "\n", encoding="utf-8")
    (artifact_dir / "confusion_matrix.json").write_text(
        json.dumps(metrics["confusion_matrix"], indent=2) + "\n", encoding="utf-8"
    )
    # Refresh the model card so robustness/holdout sections are populated.
    from .training import load_config  # default config supplies card scaffolding

    config = load_config(None)
    (artifact_dir / "MODEL_CARD.md").write_text(
        _render_card_from_stored(metadata, stored, config), encoding="utf-8"
    )
    return evaluated


def _render_card_from_stored(metadata: Dict[str, Any], stored: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Model card rendered from artifact metrics (post-holdout refresh)."""
    from .training import render_model_card

    metrics = {
        "cv": stored.get("cv") or {},
        "holdout": stored.get("holdout"),
        "calibration": {
            "method": (stored.get("calibration") or {}).get("method"),
            "ece": (stored.get("calibration_scores") or {}).get("ece"),
            "brier": (stored.get("calibration_scores") or {}).get("brier"),
        },
        "robustness": stored.get("robustness"),
    }
    return render_model_card(metadata, metrics, config)


def render_holdout_report(evaluated: Dict[str, Any], target: str, demo: bool = False) -> str:
    """Human-readable locked-holdout evaluation report."""
    width = 64
    lines: List[str] = []
    if demo:
        lines.append("DATA STATUS: SYNTHETIC DEMO - NOT A FACEMAXX PERFORMANCE BENCHMARK")
        lines.append("MODEL STATUS: DEMO/SYNTHETIC")
        lines.append("")
    lines.append("LOCKED HOLDOUT EVALUATION")
    lines.append("=" * width)
    lines.append(f"model:            {evaluated['model_id']}  (target: {target})")
    lines.append(f"evaluated:        {evaluated['evaluated_at']}")
    lines.append(f"holdout size:     {evaluated['holdout_rows']} images / {evaluated['holdout_people']} people")
    m = evaluated["metrics"]
    lines.append("")
    lines.append(
        f"accuracy={m['accuracy']:.3f}  balanced_accuracy={m['balanced_accuracy']:.3f}"
    )
    lines.append(
        f"macro P/R/F1={m['macro_precision']:.3f}/{m['macro_recall']:.3f}/{m['macro_f1']:.3f}"
        f"  weighted_f1={m['weighted_f1']:.3f}"
    )
    lines.append("")
    lines.append("per-class (support = holdout images):")
    lines.append(f"  {'label':<16} {'prec':>6} {'rec':>6} {'f1':>6} {'support':>8}")
    for label, stats in m["per_class"].items():
        lines.append(
            f"  {label:<16} {stats['precision']:>6.3f} {stats['recall']:>6.3f} "
            f"{stats['f1']:>6.3f} {stats['support']:>8}"
        )
    calib = evaluated.get("calibration_scores") or {}
    if calib:
        lines.append("")
        lines.append(f"calibration (holdout): ECE={calib.get('ece')}  Brier={calib.get('brier')}")
    boundary = evaluated.get("boundary")
    if boundary:
        lines.append("")
        lines.append("seasonal boundary analysis (errors by wheel distance):")
        lines.append(
            f"  correct={boundary['correct']}  boundary-neighbour errors={boundary['boundary_errors']} "
            f"({boundary['boundary_share_of_errors'] * 100:.0f}% of errors)  "
            f"non-boundary errors={boundary['non_boundary_errors']}"
        )
        if boundary["boundary_pairs"]:
            lines.append("  boundary confusion pairs:")
            for pair, count in list(boundary["boundary_pairs"].items())[:8]:
                lines.append(f"    {pair}: {count}")
        lines.append("  boundary confusion is ambiguity at seasonal edges - NOT random error;")
        lines.append("  it informs where the annotation handbook needs sharper criteria.")
    robustness = evaluated.get("robustness") or {}
    if robustness:
        lines.append("")
        lines.append("robustness / subgroup performance (frozen evaluator verdicts):")
        for attribute, info in robustness.items():
            wb = info.get("worst_best") or {}
            verdict = info.get("verdict")
            if wb:
                lines.append(
                    f"  {attribute:<15} worst={wb.get('worst_group')}({wb.get('worst')}) "
                    f"best={wb.get('best_group')}({wb.get('best')}) gap={wb.get('gap'):.3f}  [{verdict}]"
                )
            small = [
                value for value, stats in (info.get("subgroups") or {}).items()
                if stats.get("small_n")
            ]
            if small:
                lines.append(f"    small-sample subgroups (unstable, flagged not hidden): {', '.join(small)}")
    lines.append("")
    lines.append("the holdout was scored exactly once for this artifact; no tuning used it.")
    return "\n".join(lines)
