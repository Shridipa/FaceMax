"""Supervised ML training pipeline (the ML-training milestone).

Scientific rules baked in (see docs/ML_TRAINING.md):

    readiness gate -> training data -> person-grouped CV -> model selection
        -> calibration -> (only then) LOCKED HOLDOUT evaluation

- Four tabular baselines per target: LogisticRegression, RandomForest,
  XGBoost, LightGBM (the last two degrade gracefully when not installed).
- Person-grouped StratifiedGroupKFold for every comparison; the same person
  can never appear in train and test folds.
- Season target compares three approaches objectively:
    direct       features -> 12 seasons
    axis_first   features -> axis models -> rule combiner (existing models.py)
    hierarchical features -> axis models -> season head on features+axes
- Class imbalance via class weights / sample weights (never resampling that
  could duplicate people) and always reported per class (images AND people).
- Calibration: Platt (sigmoid) for smaller sets, isotonic above the config
  threshold, fitted on the validation split only (never on holdout), scored
  with Brier + ECE.
- Secondary-season rule is configurable and documented, never arbitrary.
- Every trained model lands in a structured artifact directory with
  metadata.json / metrics.json / feature_schema.json / confusion_matrix.json /
  MODEL_CARD.md - no anonymous .pkl files.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from facemax_bias.vocab import CHROMA_ORDER, CONTRAST_ORDER, SEASON_ORDER, SKIN_DEPTH_ORDER, UNDERTONE_ORDER

from .models import FEATURE_COLUMNS, ColourModelBundle, train_bundle
from .readiness import SEASON_NEIGHBOURS
from .training_data import (
    TARGETS,
    TrainingData,
    LeakageError,
    audit_feature_columns,
    feature_audit,
    matrices,
)

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for ML training: pip install scikit-learn") from exc

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None

# Axis label columns for the season stages (canonical repo vocabularies only).
AXIS_SPECS = (
    ("undertone", "undertone", UNDERTONE_ORDER),
    ("skin_depth", "skin_depth", SKIN_DEPTH_ORDER),
    ("chroma", "chroma_level", CHROMA_ORDER),
    ("contrast", "contrast_level", CONTRAST_ORDER),
)

DEFAULT_CONFIG = {
    "random_seed": 42,
    "cv_folds": 5,
    "min_class_samples": 5,
    "selection_metric": "macro_f1",
    "class_weight": "balanced",
    "n_jobs": 1,  # 1 avoids joblib/loky pool storms on small tabular data; raise for large runs
    "models": [
        "logreg", "random_forest", "extra_trees", "xgboost", "lightgbm", "hist_gb",
    ],
    "season_approaches": ["direct", "axis_first", "hierarchical"],
    "hierarchical_axis_backend": "random_forest",
    "calibration": {
        "method": "auto",              # auto | sigmoid | isotonic | none
        "isotonic_min_samples": 1000,  # below this, sigmoid (Platt) is used
        "bins": 15,
    },
    "secondary_rule": {
        # DOCUMENTED RULE (docs/ML_TRAINING.md #secondary-season):
        # secondary = highest-probability season distinct from the primary,
        # emitted only when its calibrated probability >= secondary_min_prob.
        "secondary_min_prob": 0.10,
    },
}


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if path is not None:
        user = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, value in user.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# candidates + class weights
# ---------------------------------------------------------------------------

def _class_weights(y: "np.ndarray") -> Dict[str, float]:
    values, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {str(v): total / (len(values) * c) for v, c in zip(values, counts)}


def _sample_weights(y: "np.ndarray") -> "np.ndarray":
    weights = _class_weights(y)
    return np.array([weights[str(label)] for label in y], dtype=np.float64)


def make_candidates(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """The candidate registry (missing optional deps are skipped with a note).

    hist_gb is sklearn's HistGradientBoostingClassifier - the same family as
    the historical `hist_gb` line in old model_comparison artifacts, now a
    first-class registered candidate (see the provenance header note in
    models.render_comparison_header for why old comparison files cannot be
    compared with these runs).
    """
    requested = list(config.get("models") or DEFAULT_CONFIG["models"])
    cw = config.get("class_weight") or None
    n_jobs = int(config.get("n_jobs", 1))
    candidates: Dict[str, Any] = {}
    if "logreg" in requested:
        candidates["logreg"] = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight=cw, random_state=seed)),
        ])
    if "random_forest" in requested:
        candidates["random_forest"] = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight=cw,
            random_state=seed, n_jobs=n_jobs,
        )
    if "extra_trees" in requested:
        candidates["extra_trees"] = ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight=cw,
            random_state=seed, n_jobs=n_jobs,
        )
    if "hist_gb" in requested:
        from sklearn.ensemble import HistGradientBoostingClassifier

        candidates["hist_gb"] = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, random_state=seed,
        )
    if "xgboost" in requested and XGBClassifier is not None:
        candidates["xgboost"] = _XGBAdapter(
            n_estimators=300, learning_rate=0.08, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            eval_metric="mlogloss", random_state=seed, n_jobs=n_jobs, verbosity=0,
        )
    elif "xgboost" in requested:
        print("  note: xgboost not installed - skipped (pip install xgboost)")
    if "lightgbm" in requested and LGBMClassifier is not None:
        candidates["lightgbm"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.08, num_leaves=31,
            min_child_samples=5, class_weight=cw,
            random_state=seed, n_jobs=n_jobs, verbosity=-1,
        )
    elif "lightgbm" in requested:
        print("  note: lightgbm not installed - skipped (pip install lightgbm)")
    return candidates


def _fit(name: str, model: Any, X: "np.ndarray", y: "np.ndarray") -> Any:
    """Fit with class-weight sample weights for models lacking class_weight."""
    if name in ("xgboost", "hist_gb"):
        model.fit(X, y, sample_weight=_sample_weights(y))
    else:
        model.fit(X, y)
    return model


class _XGBAdapter(ClassifierMixin, BaseEstimator):
    """XGBClassifier with string-label encoding (xgboost needs 0..k-1 ints).

    A proper sklearn estimator (clone-compatible) whose fit/predict surface
    speaks the canonical string labels.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.08,
        max_depth: int = 4,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        tree_method: str = "hist",
        eval_metric: str = "mlogloss",
        random_state: int = 0,
        n_jobs: int = -1,
        verbosity: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tree_method = tree_method
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity

    def _new_xgb(self) -> Any:
        assert XGBClassifier is not None
        return XGBClassifier(**self.get_params(deep=False))

    def fit(self, X: "np.ndarray", y: "np.ndarray", sample_weight: Optional["np.ndarray"] = None) -> "_XGBAdapter":
        from sklearn.preprocessing import LabelEncoder

        self.encoder_ = LabelEncoder().fit(y)
        y_enc = self.encoder_.transform(y)
        self.model_ = self._new_xgb()
        self.model_.fit(X, y_enc, sample_weight=sample_weight)
        return self

    @property
    def classes_(self) -> "np.ndarray":
        return self.encoder_.classes_

    def predict(self, X: "np.ndarray") -> "np.ndarray":
        return self.encoder_.inverse_transform(self.model_.predict(X).astype(int))

    def predict_proba(self, X: "np.ndarray") -> "np.ndarray":
        return self.model_.predict_proba(X)


# ---------------------------------------------------------------------------
# metrics + boundary analysis + calibration scoring
# ---------------------------------------------------------------------------

def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average="macro", zero_division=0
    )
    _, _, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average="weighted", zero_division=0
    )
    per_class: Dict[str, Dict[str, float]] = {}
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(p_c[i]),
            "recall": float(r_c[i]),
            "f1": float(f_c[i]),
            "support": int(s_c[i]),
        }
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    block = {
        "n": len(y_true),
        "accuracy": acc,
        "balanced_accuracy": bal,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f_macro),
        "weighted_f1": float(f_weighted),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(labels),
            "matrix": cm.astype(int).tolist(),
        },
    }
    # Persisted honesty warnings (zero/tiny-support classes, degenerate
    # predictions) so no reader mistakes imbalance-inflated accuracy for
    # balanced skill (spec #28: never discard per-class context).
    block["warnings"] = metric_warnings(block, labels)
    return block


def metric_warnings(
    metrics: Dict[str, Any],
    labels: Sequence[str],
    min_support: int = 5,
) -> List[str]:
    """Honesty warnings derived ONLY from a metrics block (spec #28/#42).

    - classes with zero holdout/fold support (their recall is vacuous)
    - classes whose support is below ``min_support`` (metrics are unstable)
    - degenerate predictions (one predicted class covers everything)
    Never modifies the metrics themselves - flags so nobody mistakes an
    imbalance-inflated score for balanced skill (the demo Warm-class effect).
    """
    warnings: List[str] = []
    per_class = metrics.get("per_class") or {}
    low, zero = [], []
    for label in labels:
        support = int((per_class.get(label) or {}).get("support", 0))
        if support == 0:
            zero.append(label)
        elif support < min_support:
            low.append(f"{label}={support}")
    if zero:
        warnings.append("classes with zero support (metrics for them are vacuous): " + ", ".join(zero))
    if low:
        warnings.append(f"classes with support < {min_support} (unstable metrics): " + ", ".join(low))
    preds = metrics.get("confusion_matrix") or {}
    matrix = preds.get("matrix")
    if matrix:
        col_sums = [sum(row[j] for row in matrix) for j in range(len(matrix))]
        nonzero = [c for c in col_sums if c > 0]
        if len(nonzero) == 1 and len(matrix) > 1:
            warnings.append(
                "degenerate predictions: every row was predicted as a single class "
                "(accuracy equals that class's share - not skill)"
            )
    return warnings


def ordinal_adjacency_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    order: Sequence[str],
) -> Dict[str, Any]:
    """For ORDERED targets (skin_depth): how far off are the errors?

    Spec #5: a Medium->Medium-Deep mistake (adjacent on the depth scale) is a
    different failure from a Very Fair->Deep one. Reports the share of errors
    that are ADJACENT (distance 1 on the ordinal scale) plus the worst pairs.
    Diagnostic only - never used to restate accuracy.
    """
    index = {label: i for i, label in enumerate(order)}
    adjacent = 0
    pairs: Dict[str, int] = {}
    errors = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == pred or truth not in index or pred not in index:
            continue
        errors += 1
        distance = abs(index[truth] - index[pred])
        if distance == 1:
            adjacent += 1
        pair = " -> ".join(sorted((truth, pred)))
        pairs[pair] = pairs.get(pair, 0) + 1
    return {
        "n_errors": errors,
        "adjacent_errors": adjacent,
        "adjacent_share_of_errors": round(adjacent / errors, 4) if errors else 0.0,
        "worst_pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:6]),
    }


def top2_accuracy(truth: Sequence[str], probs_rows: Sequence[Dict[str, float]]) -> float:
    """Share of rows where the truth is one of the two highest-probability labels."""
    if not truth:
        return 0.0
    hits = 0
    for actual, probs in zip(truth, probs_rows):
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])[:2]
        if any(label == actual for label, _ in ranked):
            hits += 1
    return float(hits / len(truth))


def boundary_analysis(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Any]:
    """Errors classified by seasonal-wheel distance (readiness.SEASON_NEIGHBOURS).

    A Deep Autumn -> Deep Winter mistake is a BOUNDARY error (neighbours on the
    wheel, differ on one axis); Deep Autumn -> Light Summer is a real CONFLICT.
    Boundary confusion is not "random error" and is reported separately.
    """
    correct = boundary = conflict = 0
    pairs: Dict[str, int] = {}
    for truth, pred in zip(y_true, y_pred):
        if truth == pred:
            correct += 1
        elif pred in SEASON_NEIGHBOURS.get(truth, frozenset()):
            boundary += 1
            pair = " <-> ".join(sorted((truth, pred)))
            pairs[pair] = pairs.get(pair, 0) + 1
        else:
            conflict += 1
    errors = boundary + conflict
    return {
        "n": len(list(y_true)),
        "correct": correct,
        "boundary_errors": boundary,
        "non_boundary_errors": conflict,
        "boundary_share_of_errors": (boundary / errors) if errors else 0.0,
        "boundary_pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
    }


def expected_calibration_error(probs: "np.ndarray", y_idx: "np.ndarray", bins: int = 15) -> float:
    """ECE over the max-probability class (equal-width bins)."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_idx).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i else (confidences >= lo) & (confidences <= hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def brier_score(probs: "np.ndarray", y_idx: "np.ndarray") -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# person-grouped cross-validation
# ---------------------------------------------------------------------------

def _n_splits(y: "np.ndarray", groups: "np.ndarray", requested: int, min_class: int) -> int:
    counts = {v: int((y == v).sum()) for v in np.unique(y)}
    rarest = min(counts.values())
    return max(2, min(requested, rarest, len(set(groups))))


def grouped_cv(
    X: "np.ndarray",
    y: "np.ndarray",
    groups: "np.ndarray",
    candidates: Dict[str, Any],
    labels: Sequence[str],
    seed: int,
    n_splits: int,
    min_class_samples: int,
    selection_metric: str = "macro_f1",
) -> Dict[str, Any]:
    """Person-grouped stratified CV with full fold statistics per candidate."""
    counts = {v: int((y == v).sum()) for v in np.unique(y)}
    too_rare = {v: c for v, c in counts.items() if c < min_class_samples}
    if too_rare:
        return {
            "skipped": True,
            "reason": (
                f"classes below min_class_samples={min_class_samples}: "
                + ", ".join(f"{v}={c}" for v, c in sorted(too_rare.items()))
                + " - collect more data instead of training on unstable classes"
            ),
            "class_counts": counts,
        }

    skf = StratifiedGroupKFold(n_splits=_n_splits(y, groups, n_splits, min_class_samples), shuffle=True, random_state=seed)
    folds = list(skf.split(X, y, groups))
    results: Dict[str, Any] = {}
    for name, template in candidates.items():
        fold_records: List[Dict[str, Any]] = []
        scores: List[float] = []
        fold_metrics: List[Dict[str, Any]] = []
        for fold_idx, (tr, te) in enumerate(folds, 1):
            model = _fit(name, clone(template), X[tr], y[tr])
            pred = model.predict(X[te])
            metric = compute_metrics(y[te], pred, labels)
            fold_metrics.append(metric)
            fold_records.append({
                "fold": fold_idx,
                "train_images": int(len(tr)),
                "train_people": int(len(set(groups[tr]))),
                "test_images": int(len(te)),
                "test_people": int(len(set(groups[te]))),
                # full id lists kept so audits can PROVE fold-level person
                # disjointness (no person in two folds' test sets)
                "train_people_ids": sorted(set(str(g) for g in groups[tr])),
                "test_people_ids": sorted(set(str(g) for g in groups[te])),
                "test_class_distribution": {str(v): int(c) for v, c in zip(*np.unique(y[te], return_counts=True))},
                selection_metric: metric[selection_metric],
            })
            scores.append(metric[selection_metric])
        results[name] = {
            "folds": fold_records,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            # PERSISTED per-candidate aggregates (spec #41: never discard
            # computed evaluation information) - means over the same folds.
            "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in fold_metrics])),
            "mean_macro_precision": float(np.mean([m["macro_precision"] for m in fold_metrics])),
            "mean_macro_recall": float(np.mean([m["macro_recall"] for m in fold_metrics])),
            "mean_weighted_f1": float(np.mean([m["weighted_f1"] for m in fold_metrics])),
            "per_class_mean": {
                label: {
                    key: float(np.mean([m["per_class"][label][key] for m in fold_metrics]))
                    for key in ("precision", "recall", "f1")
                }
                for label in labels
            },
            "confusion_matrix_sum": {
                "labels": list(labels),
                "matrix": np.sum(
                    [np.array(m["confusion_matrix"]["matrix"]) for m in fold_metrics], axis=0
                ).astype(int).tolist(),
            },
        }
    ranked = sorted(
        ((name, r["mean"]) for name, r in results.items()),
        key=lambda kv: -kv[1],
    )
    return {
        "skipped": False,
        "class_counts": counts,
        "candidates": results,
        "ranking": [{"model": name, "mean": score} for name, score in ranked],
        "best": ranked[0][0] if ranked else None,
    }


# ---------------------------------------------------------------------------
# season approaches
# ---------------------------------------------------------------------------

def _rows_for(data: TrainingData, splits: Tuple[str, ...]) -> List[Dict[str, Any]]:
    return [r for r in data.rows if r.get("split") in splits and r.get("_label_ok") and r.get("_person_ok")]


def _train_axis_stage(rows: List[Dict[str, Any]], axis_column: str, vocab: Sequence[str], backend: str, seed: int) -> Any:
    """One hierarchical stage: features -> axis label (canonical vocab only)."""
    usable = [r for r in rows if str(r.get(axis_column) or "") in vocab]
    y = np.array([str(r[axis_column]) for r in usable])
    if len(usable) < len(vocab) or len(set(y)) < 2:
        return None  # stage unusable; caller falls back
    X = matrices_from_rows(usable)
    template = make_candidates({"models": [backend]}, seed)[backend]
    return _fit(backend, template, X, y)


def matrices_from_rows(rows: List[Dict[str, Any]], extra_columns: Optional[List[str]] = None) -> "np.ndarray":
    """Feature matrix straight from joined rows (leakage-audited columns only)."""
    from .dataset import coerce_feature_row

    columns = audit_feature_columns(list(FEATURE_COLUMNS) + list(extra_columns or []))
    X = np.array(
        [
            [float(coerce_feature_row({c: r.get(c)}).get(c)) if r.get(c) not in (None, "") else np.nan
             for c in columns]
            for r in rows
        ],
        dtype=np.float64,
    )
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        X[~np.isfinite(col), j] = float(np.median(finite)) if finite.size else 0.0
    return X


def _axis_onehot(axis_pred: str, vocab: Sequence[str]) -> List[float]:
    return [1.0 if axis_pred == v else 0.0 for v in vocab]


def train_hierarchical_head(
    train_rows: List[Dict[str, Any]],
    head_name: str,
    candidates: Dict[str, Any],
    axis_backend: str,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Approach 3: stage-1 axis models -> season head on [features | axis one-hots].

    The head consumes the canonical feature block plus ONE-HOT vectors of the
    stage-1 axis predictions. Stage-1 predictions are model outputs (not
    annotation labels), so this adds no label leakage; the feature block is
    still leakage-audited.
    """
    axis_models: Dict[str, Any] = {}
    for _, column, vocab in AXIS_SPECS:
        model = _train_axis_stage(train_rows, column, vocab, axis_backend, seed)
        if model is None:
            return None  # cannot build the chain from this data
        axis_models[column] = model

    X_num = matrices_from_rows(train_rows)
    # Stage-1 predictions are computed VECTORISED per axis (single-row RF
    # predicts are pathologically slow); identical values to row-by-row.
    X_aug = np.hstack([X_num, _hierarchical_augment(axis_models, train_rows)[:, len(FEATURE_COLUMNS):]])
    y = np.array([str(r["season_primary"]) for r in train_rows])
    head = _fit(head_name, clone(candidates[head_name]), X_aug, y)
    return {"axis_models": axis_models, "head": head, "head_name": head_name}


# ---------------------------------------------------------------------------
# the unified predictor artifact (all approaches expose the same interface)
# ---------------------------------------------------------------------------

@dataclass
class SeasonPredictor:
    """Saved season model. approach: direct | axis_first | hierarchical."""

    approach: str
    feature_columns: List[str]
    medians: Dict[str, float]
    classes: List[str]
    secondary_min_prob: float
    pipeline: Any = None               # direct
    bundle: Any = None                 # axis_first (ColourModelBundle)
    axis_models: Dict[str, Any] = field(default_factory=dict)  # hierarchical
    head: Any = None                   # hierarchical
    calibrated: bool = False
    abstain_margin: float = 0.0        # top1-top2 prob gap below this -> abstain
    feature_version: str = ""         # schema check at inference (spec #45)

    def _numeric_row(self, features: Dict[str, Any]) -> "np.ndarray":
        from .dataset import coerce_feature_row

        feats = coerce_feature_row(features)
        values = []
        for col in self.feature_columns:
            v = feats.get(col)
            v = float(v) if isinstance(v, (int, float)) and np.isfinite(v) else np.nan
            if np.isnan(v):
                v = self.medians.get(col, 0.0)
            values.append(v)
        return np.array([values], dtype=np.float64)

    def predict_row(self, features: Dict[str, Any]) -> Dict[str, Any]:
        X = self._numeric_row(features)
        if self.approach == "direct":
            probs = self.pipeline.predict_proba(X)[0]
            classes = [str(c) for c in self.pipeline.classes_]
        elif self.approach == "axis_first":
            out = self.bundle.predict(features)
            # normalize the axis soup into a probability vector over all seasons
            probs_full = {s: 0.0 for s in self.classes}
            primary = out["prediction"]
            probs_full[primary] = max(out["prediction_confidence"], 1e-6)
            rest = 1.0 - probs_full[primary]
            if out["prediction_secondary"] and out["prediction_secondary"] in probs_full:
                probs_full[out["prediction_secondary"]] += rest
            probs = np.array([probs_full[c] for c in self.classes])
            classes = list(self.classes)
        else:  # hierarchical
            preds = {}
            for _, column, _vocab in AXIS_SPECS:
                preds[column] = str(self.axis_models[column].predict(X)[0])
            onehots = [
                v for _, column, vocab in AXIS_SPECS
                for v in _axis_onehot(preds[column], vocab)
            ]
            X_aug = np.hstack([X, np.array([onehots], dtype=np.float64)])
            probs = self.head.predict_proba(X_aug)[0]
            classes = [str(c) for c in self.head.classes_]

        order = np.argsort(-probs)
        primary = classes[order[0]]
        confidence = float(probs[order[0]])
        runner_up = classes[order[1]] if len(order) > 1 else ""
        runner_up_prob = float(probs[order[1]]) if len(order) > 1 else 0.0
        secondary = ""
        if runner_up and runner_up_prob >= self.secondary_min_prob:
            secondary = runner_up

        # Abstention (spec #53): when the top two calibrated probabilities are
        # genuinely close, say so instead of manufacturing certainty.
        abstain = bool(self.abstain_margin > 0.0 and (confidence - runner_up_prob) < self.abstain_margin)
        return {
            "prediction": "" if abstain else primary,
            "prediction_secondary": runner_up if abstain else secondary,
            "prediction_confidence": round(confidence, 4),
            "probabilities": {c: round(float(p), 6) for c, p in zip(classes, probs)},
            "abstained": abstain,
            "abstain_reason": (
                f"top-2 probability gap {confidence - runner_up_prob:.3f} below "
                f"abstain_margin {self.abstain_margin:.3f}; treat as ambiguous between "
                f"{primary} and {runner_up}" if abstain else ""
            ),
        }

    def verify_feature_schema(self, features: Dict[str, Any]) -> None:
        """Refuse to predict when the feature schema does not match the model.

        A silent column mismatch would produce confident nonsense; failing
        loudly here is the whole point of persisting feature_schema.json.
        """
        from .dataset import coerce_feature_row

        feats = coerce_feature_row(features)
        missing = [c for c in self.feature_columns if c not in feats]
        if missing:
            raise ValueError(
                "feature schema mismatch: model expects columns missing from the "
                f"input features: {missing} (feature_version={self.feature_version!r}); "
                "re-extract with a matching feature version instead of predicting blindly"
            )


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def calibrate(model: Any, X_val: "np.ndarray", y_val: "np.ndarray", config: Dict[str, Any]) -> Tuple[Any, str]:
    """Fit a probability calibrator on the validation split (never holdout).

    auto: isotonic when >= isotonic_min_samples calibration rows, else sigmoid
    (Platt). Small calibration sets make isotonic overfit - documented choice.

    Implementation note: the base model is FROZEN and calibration uses one
    precomputed split (all validation rows). This is prefit semantics and -
    unlike leaving cv unset - does not silently require sklearn's internal
    5-fold stratification, which tiny validation splits cannot satisfy.
    """
    method = (config.get("calibration") or {}).get("method", "auto")
    n = len(y_val)
    if method == "auto":
        method = "isotonic" if n >= (config.get("calibration") or {}).get("isotonic_min_samples", 1000) else "sigmoid"
    if method == "none":
        return model, "none"
    idx = np.arange(len(y_val))
    try:
        frozen = FrozenEstimator(model)
    except ImportError:  # pragma: no cover - sklearn < 1.6
        calibrator = CalibratedClassifierCV(model, method=method, cv="prefit")
    else:
        calibrator = CalibratedClassifierCV(frozen, method=method, cv=[(idx, idx)])
    calibrator.fit(X_val, y_val)
    return calibrator, method


def calibration_report(probs: "np.ndarray", y_idx: "np.ndarray", bins: int) -> Dict[str, float]:
    return {
        "ece": round(expected_calibration_error(probs, y_idx, bins), 4),
        "brier": round(brier_score(probs, y_idx), 4),
    }


# ---------------------------------------------------------------------------
# artifact registry
# ---------------------------------------------------------------------------

ARTIFACT_FILES = ("model.joblib", "metadata.json", "metrics.json", "feature_schema.json", "confusion_matrix.json")

MODELS_ROOT = "data/models"


def next_model_id(models_root: Path, target: str, model_type: str) -> str:
    base = models_root / target
    existing = {p.name for p in base.glob("*") if p.is_dir()} if base.exists() else set()
    version = 1
    while f"{target}-{model_type}-v{version:03d}" in existing:
        version += 1
    return f"{target}-{model_type}-v{version:03d}"


def save_artifact(
    models_root: Path,
    target: str,
    model_type: str,
    predictor: Any,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    data: TrainingData,
    seed: int,
    smoke: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Path]:
    if joblib is None:  # pragma: no cover
        raise ImportError("joblib required to persist models: pip install joblib")
    model_id = next_model_id(models_root, target, model_type)
    out_dir = models_root / target / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_id": model_id,
        "model_type": model_type,
        "target": target,
        "label_column": data.label_column,
        "feature_version": data.feature_version,
        "dataset_version": data.dataset_version,
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_rows": int(sum(data.split_counts.values())),
        "training_people": len({str(r.get("person_id")) for r in data.rows if r.get("_person_ok")}),
        "validation_strategy": "person-grouped StratifiedGroupKFold",
        "hyperparameters": {k: str(v) for k, v in config.items() if k not in ("secondary_rule", "calibration")},
        "random_seed": seed,
        "git_commit": git_commit(),
        "data_source": data.source,
        "smoke": bool(smoke),
        "selection_reason": metrics.get("selection_reason"),
    }
    if extra:
        metadata.update(extra)

    joblib.dump(predictor, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_version": data.feature_version,
                "feature_columns": list(FEATURE_COLUMNS),
                "leakage_audit": "canonical feat_* columns only; labels/annotator columns rejected by audit_feature_columns",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    cm = metrics.get("holdout", {}).get("confusion_matrix") or metrics.get("cv", {}).get("confusion_matrix")
    if cm:
        (out_dir / "confusion_matrix.json").write_text(json.dumps(cm, indent=2) + "\n", encoding="utf-8")
    (out_dir / "MODEL_CARD.md").write_text(render_model_card(metadata, metrics, config), encoding="utf-8")
    return model_id, out_dir


def render_model_card(metadata: Dict[str, Any], metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Per-model model card (docs/MODEL_CARD.md documents the full format)."""
    target = metadata.get("target")
    lines = [f"# Model card: {metadata.get('model_id')}", ""]
    lines.append("## Intended use")
    lines.append(
        f"FaceMaxx {target} prediction from CV colour features. "
        "One input of a person's photo -> model output used by the product's "
        "colour analysis. Not a medical or identity tool."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"Canonical CV features only: {', '.join(FEATURE_COLUMNS)}")
    lines.append("Labels, annotator votes, agreement scores and metadata are structurally excluded (LeakageError).")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- `{target}` prediction")
    if target == "season":
        lines.append("- primary season, secondary season (documented margin rule), calibrated confidence")
    lines.append("")
    lines.append("## Training data")
    lines.append(f"- source: {metadata.get('data_source')}")
    lines.append(f"- rows: {metadata.get('training_rows')} from {metadata.get('training_people')} people")
    lines.append(f"- dataset_version: {metadata.get('dataset_version')}")
    lines.append(f"- feature_version: {metadata.get('feature_version')}")
    lines.append("")
    lines.append("## Evaluation")
    lines.append(f"- validation: {metadata.get('validation_strategy')} ({config.get('cv_folds')} folds)")
    cv = metrics.get("cv") or {}
    if cv and not cv.get("skipped"):
        best = cv.get("best")
        lines.append(f"- CV selection metric {config.get('selection_metric')}: best={best} "
                     f"({cv.get('candidates', {}).get(best, {}).get('mean', float('nan')):.3f})")
        reason = metrics.get("selection_reason") or {}
        if reason:
            lines.append(
                f"- selection: {reason.get('selected')} chosen over runner-up "
                f"{reason.get('runner_up')} (margin {reason.get('margin')})"
            )
        ablation = metrics.get("ablation") or {}
        if ablation and ablation.get("variants"):
            lines.append("- feature-group ablation (same folds/backend, macro-F1):")
            for variant, stats in ablation["variants"].items():
                lines.append(f"    {variant:<28} {stats.get('mean'):.3f}")
    holdout = metrics.get("holdout")
    if holdout:
        lines.append(f"- locked holdout: accuracy={holdout.get('accuracy'):.3f} "
                     f"macro_f1={holdout.get('macro_f1'):.3f} "
                     f"balanced_accuracy={holdout.get('balanced_accuracy'):.3f} n={holdout.get('n')}")
        # Per-class table on the card (spec #29): support + P/R/F1 must be
        # visible where the headline number is read, not buried in metrics.json.
        per_class = holdout.get("per_class") or {}
        if per_class:
            lines.append("")
            lines.append("### Holdout per-class metrics")
            lines.append("")
            lines.append("| label | support | precision | recall | f1 |")
            lines.append("|---|---|---|---|---|")
            for label, stats in per_class.items():
                lines.append(
                    f"| {label} | {int(stats.get('support', 0))} "
                    f"| {stats.get('precision', 0.0):.3f} | {stats.get('recall', 0.0):.3f} "
                    f"| {stats.get('f1', 0.0):.3f} |"
                )
        for warning in holdout.get("warnings") or []:
            lines.append(f"- warning: {warning}")
        ordinal = metrics.get("ordinal_adjacency")
        if ordinal:
            lines.append(
                f"- ordinal adjacency: {ordinal.get('adjacent_errors')}/{ordinal.get('n_errors')} "
                f"errors are adjacent classes on the depth scale "
                f"({ordinal.get('adjacent_share_of_errors'):.0%}) - see metrics.json"
            )
    calib = metrics.get("calibration")
    if calib:
        lines.append(f"- calibration: method={calib.get('method')} ECE={calib.get('ece')} Brier={calib.get('brier')}")
        if calib.get("note"):
            lines.append(f"  ({calib.get('note')})")
    lines.append("")
    lines.append("## Known limitations")
    limitations = [
        "- trained on the annotated FaceMaxx dataset only; small classes are unstable",
        "- lighting/makeup/quality shifts can move predictions (see robustness section)",
        "- seasonal boundaries (e.g. Deep Autumn <-> Deep Winter) carry genuine ambiguity",
        "- olive undertone is partly culture-dependent; labels reflect this project's handbook",
        "- NOT a benchmark beyond the recorded holdout numbers",
    ]
    if metadata.get("smoke") or metadata.get("data_source") == "demo":
        limitations.insert(
            0,
            "- **SYNTHETIC DEMO / SMOKE TEST ARTIFACT: trained on synthetic demo data. "
            "NOT A FACEMAXX PERFORMANCE BENCHMARK. NOT FOR PRODUCTION.**",
        )
    audit = metrics.get("feature_audit") or {}
    if audit.get("flagged"):
        limitations.append(f"- flagged features (see feature_audit.json): {', '.join(audit['flagged'])}")
    lines.extend(limitations)
    lines.append("")
    lines.append("## Bias / robustness")
    robustness = metrics.get("robustness")
    if robustness:
        lines.append("```")
        for attribute, info in robustness.items():
            wb = info.get("worst_best") or {}
            lines.append(
                f"{attribute}: worst={wb.get('worst_group')} ({wb.get('worst')}) "
                f"best={wb.get('best_group')} ({wb.get('best')}) gap={wb.get('gap')}"
            )
        lines.append("```")
    else:
        lines.append("not yet evaluated on the locked holdout")
    lines.append("")
    lines.append("## Not intended for")
    lines.append(
        "- race, ethnicity, health, attractiveness or identity inference of any kind\n"
        "- decisions about individuals beyond colour-analysis styling suggestions"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# robustness (reuses the FROZEN evaluator's subgroup machinery)
# ---------------------------------------------------------------------------

def robustness_report(
    rows: List[Dict[str, Any]],
    truth_key: str,
    attributes: Sequence[str] = ("skin_depth", "lighting_type", "camera_type", "makeup", "image_quality", "source"),
) -> Dict[str, Any]:
    """Per-subgroup n_images/n_people/accuracy/balanced_accuracy/macro_f1.

    Small subgroups are FLAGGED (small_n), never hidden. Additionally runs the
    frozen evaluator's fairness_gaps so its verdicts/thresholds stay canonical.
    """
    from collections import defaultdict

    from facemax_bias.subgroups import fairness_gaps
    from facemax_bias.vocab import GROUP_SIZE_WARN
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    buckets: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    scored = [r for r in rows if r.get(truth_key) and r.get("prediction")]
    for row in scored:
        for attribute in attributes:
            value = str(row.get(attribute) or "").strip()
            if value:
                buckets[attribute][value].append(row)

    report: Dict[str, Any] = {}
    for attribute in attributes:
        groups_out: Dict[str, Dict[str, Any]] = {}
        for value, members in sorted(buckets.get(attribute, {}).items()):
            truth = [str(r[truth_key]) for r in members]
            pred = [str(r["prediction"]) for r in members]
            labels = sorted(set(truth) | set(pred))
            groups_out[value] = {
                "n_images": len(members),
                "n_people": len({str(r.get("person_id")) for r in members}),
                "accuracy": round(float(accuracy_score(truth, pred)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(truth, pred)), 4),
                "macro_f1": round(float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)), 4),
                "small_n": len(members) < GROUP_SIZE_WARN,
            }
        # frozen evaluator's canonical verdicts on the same rows
        frozen = fairness_gaps(scored, truth_key=truth_key, attributes=[attribute]).get(attribute, {})
        report[attribute] = {
            "subgroups": groups_out,
            "worst_best": frozen.get("worst_best"),
            "verdict": frozen.get("verdict", "insufficient data"),
        }
    return report


# ---------------------------------------------------------------------------
# top-level training run
# ---------------------------------------------------------------------------

def run_training(
    data: TrainingData,
    config: Dict[str, Any],
    models_root: Path,
    smoke: bool = False,
    val_fraction_note: str = "",
) -> Dict[str, Any]:
    """Train every candidate for the target via person-grouped CV; save the winner.

    The locked holdout is NEVER touched here - only `evaluate` uses it, after
    model selection is complete (spec #17).
    """
    seed = int(config.get("random_seed", 42))
    labels = list(TARGETS[data.target]["order"])
    train_rows = _rows_for(data, ("train", "val"))
    X, y, groups = matrices(data, splits=("train", "val"))

    # Feature audit FIRST (spec: inspect before trusting models). Reported
    # into metrics.json + feature_audit.json; never silently drops columns.
    audit = feature_audit(train_rows)

    candidates = make_candidates(config, seed)
    cv = grouped_cv(
        X, y, groups, candidates, labels, seed,
        n_splits=int(config.get("cv_folds", 5)),
        min_class_samples=int(config.get("min_class_samples", 5)),
        selection_metric=str(config.get("selection_metric", "macro_f1")),
    )
    result: Dict[str, Any] = {
        "target": data.target,
        "cv": cv,
        "n_train_val_rows": len(train_rows),
        "n_people": len(set(groups)),
        "feature_audit": audit,
    }
    if cv.get("skipped"):
        return result

    best_name = cv["best"]
    selection_metric = str(config.get("selection_metric", "macro_f1"))
    # Evidence, not just a name (spec #58): why this candidate won, and what
    # the runner-up scored, recorded alongside the artifact.
    ranking = cv.get("ranking") or []
    runner_up = ranking[1] if len(ranking) > 1 else None
    selection_reason = {
        "metric": selection_metric,
        "selected": best_name,
        "selected_mean": ranking[0]["mean"] if ranking else None,
        "runner_up": runner_up["model"] if runner_up else None,
        "runner_up_mean": runner_up["mean"] if runner_up else None,
        "margin": (
            ranking[0]["mean"] - runner_up["mean"]
            if runner_up else None
        ),
        "note": (
            f"highest person-grouped CV {selection_metric} over the same folds; "
            "heuristic baseline kept separately as model zero"
        ),
    }
    result["selection_reason"] = selection_reason

    # ---- refit the winner on all train+val rows -----------------------------
    best_model = _fit(best_name, candidates[best_name], X, y)

    secondary_min = float((config.get("secondary_rule") or {}).get("secondary_min_prob", 0.10))

    if data.target == "season":
        # Compare approaches 1-3 objectively (spec #3); pick by CV macro-F1.
        approach_scores = _season_approach_cv(data, config, seed, selection_metric)
        result["season_approaches"] = approach_scores
        # Only approaches that actually ran have a numeric mean (skipped ones
        # carry {skipped: True, reason: ...} and must not crash selection).
        approach_rows = {
            k: v for k, v in approach_scores.items()
            if isinstance(v.get("mean"), (int, float)) and v["mean"] == v["mean"]
        }
        best_approach = max(approach_rows, key=lambda k: approach_rows[k]["mean"]) if approach_rows else "direct"
        best_approach_score = approach_rows.get(best_approach, {}).get("mean", float("-inf"))
        direct_score = cv["candidates"][best_name]["mean"]
        if best_approach_score > direct_score:
            predictor, model_type = _build_season_predictor(
                data, config, seed, best_approach, secondary_min, train_rows, best_approach_score
            )
        else:
            predictor = SeasonPredictor(
                approach="direct",
                feature_columns=list(FEATURE_COLUMNS),
                medians=_medians(X, list(FEATURE_COLUMNS)),
                classes=labels,
                secondary_min_prob=secondary_min,
                pipeline=best_model,
            )
            model_type = best_name
        result["selected_approach"] = predictor.approach
    else:
        predictor = SeasonPredictor(
            approach="direct",
            feature_columns=list(FEATURE_COLUMNS),
            medians=_medians(X, list(FEATURE_COLUMNS)),
            classes=labels,
            secondary_min_prob=secondary_min,
            pipeline=best_model,
        )
        model_type = best_name
    predictor.abstain_margin = float((config.get("secondary_rule") or {}).get("abstain_margin", 0.0))
    predictor.feature_version = data.feature_version

    # ---- calibration on the val split only (if it exists) --------------------
    val_rows = _rows_for(data, ("val",))
    calibration_info: Dict[str, Any] = {"method": "none", "note": "no validation rows; raw probabilities kept"}
    if val_rows:
        if data.target == "season" and predictor.approach == "axis_first":
            # bundle confidences are already normalized axis soups; kept as-is
            calibration_info = {"method": "none", "note": "axis_first confidence is a normalized axis soup"}
        else:
            if predictor.approach == "hierarchical":
                X_val = _hierarchical_augment(predictor.axis_models, val_rows)
            else:
                X_val, _, _ = matrices(data, splits=("val",))
            y_val = np.array([str(r[data.label_column]) for r in val_rows])
            base = predictor.pipeline if predictor.approach == "direct" else predictor.head
            # Calibration needs every base-model class represented in val;
            # a missing class cannot be calibrated, only reported.
            base_classes = {str(c) for c in np.ravel(getattr(base, "classes_", []))}
            val_classes = {str(v) for v in np.unique(y_val)}
            missing = sorted(base_classes - val_classes)
            # Guard: a calibrator fitted on ~n rows/classes cannot be trusted
            # (spec #27: do NOT over-calibrate tiny data). Rule of thumb kept
            # conservative: >= 20 rows AND >= 5 rows per class, else report
            # insufficiency honestly instead of manufacturing confidence.
            n_classes = max(len(base_classes), 1)
            insufficient = len(y_val) < 20 or len(y_val) < 5 * n_classes
            if missing:
                calibration_info = {
                    "method": "none",
                    "sufficient": False,
                    "note": (
                        "validation split does not contain classes "
                        f"{missing}; raw probabilities kept (calibration "
                        "requires every class to appear in the calibration data)"
                    ),
                }
            elif insufficient:
                calibration_info = {
                    "method": "none",
                    "sufficient": False,
                    "note": (
                        f"insufficient data for reliable calibration: {len(y_val)} "
                        f"validation rows across {n_classes} classes (want >= 20 rows "
                        "and >= 5 per class); raw probabilities kept"
                    ),
                }
            else:
                calibrated, method = calibrate(base, X_val, y_val, config)
                if predictor.approach == "direct":
                    predictor.pipeline = calibrated
                else:
                    predictor.head = calibrated
                predictor.calibrated = method != "none"
                calibration_info = {
                    "method": method,
                    "sufficient": True,
                    "n_calibration_rows": len(y_val),
                    "note": "fitted on the validation split only; holdout untouched",
                }
    result["calibration"] = calibration_info

    # Same-folds feature-group ablation (reported, never auto-applied).
    # Reuses the already-selected winner: no extra candidate CV round.
    try:
        result["ablation"] = _ablation_inner(data, config, seed, selection_metric, backend=best_name)
    except Exception as exc:  # ablation is diagnostic; never block training
        result["ablation"] = {"skipped": True, "reason": str(exc)}

    model_id, artifact_dir = save_artifact(
        models_root, data.target, model_type, predictor,
        {
            "cv": cv,
            "calibration": calibration_info,
            "selection_reason": selection_reason,
            "feature_audit": audit,
            "ablation": result.get("ablation"),
        },
        config, data, seed, smoke,
    )
    (artifact_dir / "feature_audit.json").write_text(
        json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8"
    )
    result.update({"model_id": model_id, "artifact_dir": str(artifact_dir)})
    return result


def _medians(X: "np.ndarray", columns: List[str]) -> Dict[str, float]:
    return {col: float(np.median(X[:, j])) for j, col in enumerate(columns)}


# ---------------------------------------------------------------------------
# feature-group ablation (spec #50: prove which features actually help)
# ---------------------------------------------------------------------------

FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "skin_core": (
        "feat_skin_L", "feat_skin_a", "feat_skin_b", "feat_ita",
        "feat_undertone_proxy", "feat_olive_proxy", "feat_chroma_proxy",
        "feat_depth_proxy", "feat_skin_hue",
    ),
    "hair_eye": ("feat_hair_L", "feat_eye_L", "feat_contrast_proxy"),
    "region_variability": (
        "feat_skin_region_count", "feat_skin_L_spread",
        "feat_skin_a_spread", "feat_skin_b_spread",
    ),
    "quality_lighting": (
        "feat_wb_b_minus_a", "feat_grey_world_shift", "feat_brightness",
        "feat_highlight_clip_frac", "feat_shadow_clip_frac",
    ),
}


def cv_best_name(data: TrainingData, config: Dict[str, Any], seed: int) -> str:
    """Standalone best-candidate selection (run_training passes its own winner)."""
    labels = list(TARGETS[data.target]["order"])
    X, y, groups = matrices(data, splits=("train", "val"))
    cv = grouped_cv(
        X, y, groups, make_candidates(config, seed), labels, seed,
        n_splits=int(config.get("cv_folds", 5)),
        min_class_samples=int(config.get("min_class_samples", 5)),
        selection_metric=str(config.get("selection_metric", "macro_f1")),
    )
    return cv.get("best") or "logreg"


def _ablation_inner(
    data: TrainingData,
    config: Dict[str, Any],
    seed: int,
    selection_metric: str,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    from sklearn.model_selection import StratifiedGroupKFold as _SGKF

    if backend is None:
        backend = cv_best_name(data, config, seed)
    labels = list(TARGETS[data.target]["order"])
    X_full, y, groups = matrices(data, splits=("train", "val"))
    rows = _rows_for(data, ("train", "val"))
    skf = _SGKF(
        n_splits=_n_splits(y, groups, int(config.get("cv_folds", 5)), int(config.get("min_class_samples", 5))),
        shuffle=True, random_state=seed,
    )
    folds = list(skf.split(X_full, y, groups))
    candidates = {backend: make_candidates(config, seed)[backend]}

    variants: Dict[str, Optional[Tuple[str, ...]]] = {"full": None}
    cumulative: Tuple[str, ...] = ()
    group_order = ["skin_core", "hair_eye", "region_variability", "quality_lighting"]
    for position, group in enumerate(group_order, 1):
        cumulative = cumulative + FEATURE_GROUPS[group]
        variants["+".join(group_order[:position])] = cumulative

    out: Dict[str, Any] = {"backend": backend, "variants": {}}
    for name, columns in variants.items():
        if columns is None:
            X = X_full
        else:
            idx = [list(FEATURE_COLUMNS).index(c) for c in columns if c in list(FEATURE_COLUMNS)]
            X = X_full[:, idx]
        scores: List[float] = []
        for _fold, (tr, te) in enumerate(folds, 1):
            model = _fit(backend, clone(candidates[backend]), X[tr], y[tr])
            pred = model.predict(X[te])
            metric = compute_metrics(y[te], pred, labels)
            scores.append(metric[selection_metric])
        out["variants"][name] = {
            "n_features": int(X.shape[1]),
            "mean": float(np.mean(scores)) if scores else float("nan"),
            "std": float(np.std(scores)) if scores else float("nan"),
        }
    best_variant = max(
        (k for k in out["variants"] if out["variants"][k]["mean"] == out["variants"][k]["mean"]),
        key=lambda k: out["variants"][k]["mean"],
        default="full",
    )
    out["best_variant"] = best_variant
    out["note"] = (
        "same folds/backend for every variant; the full set stays the deployed "
        "default unless a variant wins by a margin worth acting on"
    )
    return out


def _build_season_predictor(
    data: TrainingData,
    config: Dict[str, Any],
    seed: int,
    approach: str,
    secondary_min: float,
    train_rows: List[Dict[str, Any]],
    approach_score: float,
) -> Tuple[SeasonPredictor, str]:
    if approach == "axis_first":
        bundle = train_bundle(train_rows, season_cv=False, seed=seed)
        return SeasonPredictor(
            approach="axis_first",
            feature_columns=list(FEATURE_COLUMNS),
            medians={c: 0.0 for c in FEATURE_COLUMNS},
            classes=list(SEASON_ORDER),
            secondary_min_prob=secondary_min,
            bundle=bundle,
        ), "axis_first_bundle"
    if approach == "hierarchical":
        axis_backend = str(config.get("hierarchical_axis_backend", "random_forest"))
        built = train_hierarchical_head(
            train_rows, axis_backend, make_candidates({"models": [axis_backend]}, seed), axis_backend, seed
        )
        if built is not None:
            return SeasonPredictor(
                approach="hierarchical",
                feature_columns=list(FEATURE_COLUMNS),
                medians={c: 0.0 for c in FEATURE_COLUMNS},
                classes=list(SEASON_ORDER),
                secondary_min_prob=secondary_min,
                axis_models=built["axis_models"],
                head=built["head"],
            ), f"hierarchical_{axis_backend}"
    # fall through: direct
    raise ValueError(f"cannot build season approach {approach!r} from this data")


def _season_approach_cv(
    data: TrainingData,
    config: Dict[str, Any],
    seed: int,
    selection_metric: str,
) -> Dict[str, Any]:
    """Person-grouped CV of the axis_first and hierarchical season approaches.

    (direct candidate-level CV is already in `cv`.) All approaches are scored
    on the SAME folds so the comparison is objective - spec #3 says do not
    assume the hierarchical model is better; measure it.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    from .dataset import coerce_feature_row

    requested = list(config.get("season_approaches") or ["direct", "axis_first", "hierarchical"])
    X, y, groups = matrices(data, splits=("train", "val"))
    rows = _rows_for(data, ("train", "val"))
    skf = StratifiedGroupKFold(
        n_splits=_n_splits(y, groups, int(config.get("cv_folds", 5)), int(config.get("min_class_samples", 5))),
        shuffle=True, random_state=seed,
    )
    folds = list(skf.split(X, y, groups))
    out: Dict[str, Any] = {}

    # ---- approach 2: axis_first (existing ColourModelBundle rule combiner) ---
    if "axis_first" in requested:
        scores: List[float] = []
        fold_records: List[Dict[str, Any]] = []
        error: Optional[str] = None
        for fold_idx, (tr, te) in enumerate(folds, 1):
            try:
                bundle = train_bundle([rows[i] for i in tr], season_cv=False, seed=seed)
                preds = [
                    bundle.predict(coerce_feature_row({k: rows[i].get(k) for k in FEATURE_COLUMNS}))["prediction"]
                    for i in te
                ]
            except Exception as exc:
                error = f"fold {fold_idx}: {exc}"
                break
            truth = [str(rows[i][data.label_column]) for i in te]
            metric = compute_metrics(truth, preds, list(SEASON_ORDER))
            fold_records.append({
                "fold": fold_idx,
                "test_images": int(len(te)),
                "test_people": int(len(set(groups[te]))),
                selection_metric: metric[selection_metric],
            })
            scores.append(metric[selection_metric])
        if error or not scores:
            out["axis_first"] = {"skipped": True, "reason": error or "no valid folds"}
        else:
            out["axis_first"] = {
                "folds": fold_records,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }

    # ---- approach 3: hierarchical (stage-1 axes -> season head) --------------
    if "hierarchical" in requested:
        axis_backend = str(config.get("hierarchical_axis_backend", "random_forest"))
        head_candidates = make_candidates({"models": [axis_backend]}, seed)
        scores_h: List[float] = []
        fold_records_h: List[Dict[str, Any]] = []
        error_h: Optional[str] = None
        for fold_idx, (tr, te) in enumerate(folds, 1):
            try:
                built = train_hierarchical_head(
                    [rows[i] for i in tr], axis_backend, head_candidates, axis_backend, seed
                )
                if built is None:
                    error_h = f"fold {fold_idx}: axis stage untrainable"
                    break
                preds = [_hierarchical_predict(built, rows[i]) for i in te]
            except Exception as exc:
                error_h = f"fold {fold_idx}: {exc}"
                break
            truth = [str(rows[i][data.label_column]) for i in te]
            metric = compute_metrics(truth, preds, list(SEASON_ORDER))
            fold_records_h.append({
                "fold": fold_idx,
                "test_images": int(len(te)),
                "test_people": int(len(set(groups[te]))),
                selection_metric: metric[selection_metric],
            })
            scores_h.append(metric[selection_metric])
        if error_h or not scores_h:
            out["hierarchical"] = {"skipped": True, "reason": error_h or "no valid folds"}
        else:
            out["hierarchical"] = {
                "folds": fold_records_h,
                "mean": float(np.mean(scores_h)),
                "std": float(np.std(scores_h)),
            }
    return out


def _hierarchical_predict(built: Dict[str, Any], row: Dict[str, Any]) -> str:
    """Predict one row through the stage-1 axis models + season head."""
    X = matrices_from_rows([row])
    onehots = []
    for _, column, vocab in AXIS_SPECS:
        pred = str(built["axis_models"][column].predict(X)[0])
        onehots.extend(_axis_onehot(pred, vocab))
    X_aug = np.hstack([X, np.array([onehots], dtype=np.float64)])
    return str(built["head"].predict(X_aug)[0])


def _hierarchical_augment(axis_models: Dict[str, Any], rows: List[Dict[str, Any]]) -> "np.ndarray":
    """[features | axis one-hots] matrix for rows, using the stage-1 models."""
    X = matrices_from_rows(rows)
    onehots = np.zeros((len(rows), sum(len(vocab) for _, _, vocab in AXIS_SPECS)), dtype=np.float64)
    offset = 0
    for _, column, vocab in AXIS_SPECS:
        preds = axis_models[column].predict(X)
        for row_idx, pred in enumerate(preds):
            onehots[row_idx, offset:offset + len(vocab)] = _axis_onehot(str(pred), vocab)
        offset += len(vocab)
    return np.hstack([X, onehots])
