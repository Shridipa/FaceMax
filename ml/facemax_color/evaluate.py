"""Step 5: connect the real colour models to the bias evaluator.

    annotations + colour features
        -> person-level split (train/test, no leakage)
        -> ColourModelBundle fitted on train rows
        -> predictions written onto test rows
        -> facemax_bias.generate_report   (the FINAL BIAS REPORT)
        -> comparison table + recommended actions

Every number in the report is computed on held-out people.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from facemax_bias.leakage import grouped_split, leakage_check, split_stats
from facemax_bias.report import BiasReport, generate_report
from facemax_bias.recommendations import actions_from_report, render_actions
from facemax_bias.vocab import SEASON_ORDER

from .models import FEATURE_COLUMNS, ColourModelBundle, train_bundle
from .dataset import coerce_feature_row, join_with_features


@dataclass
class EvaluationOutput:
    bundle: ColourModelBundle
    report: BiasReport
    actions_text: str
    split_summary: Dict[str, Any]
    leakage: Dict[str, Any]
    test_rows: List[Dict] = field(default_factory=list)

    @property
    def report_text(self) -> str:
        return self.report.text

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "bias_report.txt").write_text(self.report_text + "\n\n" + self.actions_text, encoding="utf-8")
        self.report.save_json(str(out_dir / "bias_report.json"))
        (out_dir / "model_comparison.txt").write_text(render_comparison(self.bundle), encoding="utf-8")


def _features_ok(row: Dict) -> bool:
    return bool(row.get("features_ok")) and row.get("season_primary") in SEASON_ORDER


def run_evaluation(
    rows: List[Dict],
    test_frac: float = 0.25,
    seed: int = 7,
    season_cv: bool = True,
    lock_split: bool = False,
    min_rows: int = 40,
) -> EvaluationOutput:
    """Full train -> predict -> evaluate cycle on person-grouped splits.

    ``lock_split=True`` reuses an existing 'split' column (the frozen holdout)
    instead of re-splitting; rows must then carry split=train/test(/val).
    ``min_rows`` guards real evaluations (default 40); the synthetic demo may
    lower it because its report is explicitly labeled as a smoke test.
    """
    annotated = [r for r in rows if _features_ok(r)]
    if len(annotated) < min_rows:
        raise ValueError(
            f"need >= {min_rows} fully annotated+extracted rows, got {len(annotated)}. "
            "Run `python -m facemax_color extract` first."
        )

    if lock_split:
        for row in annotated:
            row.setdefault("split", "test")
        leakage = leakage_check(annotated)
        if not leakage["clean"]:
            raise ValueError(f"locked split has person leakage: {leakage['pairs'][:5]}")
        train_rows = [r for r in annotated if r.get("split") == "train"]
        test_rows = [r for r in annotated if r.get("split") == "test"]
        if not train_rows or not test_rows:
            raise ValueError("locked split needs both train and test rows")
    else:
        all_rows = grouped_split([dict(r) for r in annotated], test_frac=test_frac, seed=seed)
        leakage = leakage_check(all_rows)
        train_rows = [r for r in all_rows if r["split"] == "train"]
        test_rows = [r for r in all_rows if r["split"] == "test"]
        # Evaluated rows keep their split tags for the report.
        rows = all_rows

    bundle = train_bundle(train_rows, season_cv=season_cv)

    predicted = 0
    for row in test_rows:
        features = coerce_feature_row({k: row.get(k) for k in FEATURE_COLUMNS})
        if all(v is None or (isinstance(v, float) and np.isnan(v)) for v in features.values()):
            continue
        result = bundle.predict(features)
        row["prediction"] = result["prediction"]
        row["prediction_secondary"] = result["prediction_secondary"]
        row["prediction_confidence"] = result["prediction_confidence"]
        row["prediction_axes"] = result["axes"]
        predicted += 1

    if not predicted:
        raise ValueError("no test rows could be predicted; check feature extraction coverage")

    report = generate_report(test_rows)

    return EvaluationOutput(
        bundle=bundle,
        report=report,
        actions_text=render_actions(actions_from_report(report.as_dict())),
        split_summary=split_stats(all_rows if not lock_split else annotated),
        leakage=leakage,
        test_rows=test_rows,
    )


def render_comparison(bundle: ColourModelBundle) -> str:
    """Human-readable model-comparison table (LogReg vs RF vs HGB per axis)."""
    lines = [
        "MODEL COMPARISON (person-grouped CV, macro-F1)",
        "=" * 52,
    ]
    for axis, info in bundle.comparison.items():
        scores = info.get("cv_macro_f1") or {}
        backend = info.get("backend")
        if axis == "season_primary":
            lines.append("")
            lines.append(f"season pipeline [{backend}]")
            value = scores if isinstance(scores, float) else float("nan")
            lines.append(f"  full-pipeline macro-F1: {value:.3f}")
            continue
        lines.append("")
        lines.append(f"{axis}  (selected: {backend})")
        for name, score in sorted(scores.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1)):
            marker = "  <-- selected" if name == backend else ""
            value = "  n/a " if score != score else f"{score:.3f}"
            lines.append(f"  {name:<16} {value}{marker}")
    lines.append("")
    lines.append("macro-F1 is the selection metric; accuracy alone hides minority-class failure.")
    return "\n".join(lines)
