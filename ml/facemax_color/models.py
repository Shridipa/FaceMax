"""Baseline colour-analysis models (Step 4 of the FaceMaxx ML roadmap).

Architecture (deliberately simple, per spec):

    skin features (LAB, ITA, proxies)
         |
         |-- undertone model   (6 classes)
         |-- skin-depth model  (8 ordered classes)
         |-- chroma model      (3 classes)
         |-- contrast model    (3 classes)
         v
    season classifier          (12 classes)
         |
    season_secondary + confidence for the evaluator

Model comparison: LogisticRegression -> RandomForest -> HistGradientBoosting
(sklearn's LightGBM-class model; XGBoost/LightGBM are available but HGB keeps
the baseline dependency-light and is equivalent for tabular data this small).
Winner is chosen per axis by macro-F1 (never plain accuracy — see fairness
spec section 11) and persisted with joblib.

Fallbacks: models gracefully degrade to rule-based heuristics from the colour
science literature (ITA bands for depth, b*-a* for undertone) when sklearn is
missing or a class has too few samples to train. Heuristics are also exported
for use as "model zero" baselines in comparisons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from facemax_bias.vocab import (
    CHROMA_ORDER,
    CONTRAST_ORDER,
    SEASON_ORDER,
    SKIN_DEPTH_ORDER,
    UNDERTONE_ORDER,
)

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    _SKLEARN_ERROR = (
        "scikit-learn is required for baseline models: pip install scikit-learn"
    )

# Feature columns produced by colour.extract_image().as_feature_row()
FEATURE_COLUMNS = [
    "feat_skin_L",
    "feat_skin_a",
    "feat_skin_b",
    "feat_ita",
    "feat_undertone_proxy",
    "feat_olive_proxy",
    "feat_chroma_proxy",
    "feat_depth_proxy",
    "feat_hair_L",
    "feat_eye_L",
    "feat_contrast_proxy",
    "feat_wb_b_minus_a",
    "feat_grey_world_shift",
    "feat_brightness",
]

AXES = {
    "undertone": UNDERTONE_ORDER,
    "skin_depth": SKIN_DEPTH_ORDER,
    "chroma": CHROMA_ORDER,
    "contrast": CONTRAST_ORDER,
    "season_primary": SEASON_ORDER,
}

MIN_CLASS_SAMPLES = 8  # below this, fallback heuristic is used for the axis


def feature_matrix(rows: List[Dict], columns: Optional[List[str]] = None) -> "np.ndarray":
    """Rows -> numeric matrix; NaNs are imputed to column medians."""
    columns = columns or FEATURE_COLUMNS
    matrix = np.array(
        [[float(row.get(c) if row.get(c) is not None else np.nan) for c in columns] for row in rows],
        dtype=np.float64,
    )
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        finite = col[np.isfinite(col)]
        median = float(np.median(finite)) if finite.size else 0.0
        matrix[~np.isfinite(col), j] = median
    return matrix


def _labels(rows: List[Dict], axis: str) -> "np.ndarray":
    return np.array([str(row.get(axis) or "") for row in rows])


# ---------------------------------------------------------------------------
# Rule-based fallbacks ("model zero")
# ---------------------------------------------------------------------------

def heuristic_undertone(skin_a: float, skin_b: float) -> str:
    """b*-a* balance: cool < -2, neutral band, warm > 8; olive = low a* with high b*."""
    value = skin_b - skin_a
    if skin_a < 6 and skin_b > 12:
        return "Olive"
    if value < -2:
        return "Cool"
    if value < 4:
        return "Neutral-Cool" if value < 1 else "Neutral"
    if value < 10:
        return "Neutral-Warm"
    return "Warm"


def heuristic_depth(skin_L: float) -> str:
    """ITA-style L* bands, ordered shallow -> deep."""
    if skin_L >= 78: return "Very Fair"
    if skin_L >= 71: return "Fair"
    if skin_L >= 64: return "Light"
    if skin_L >= 56: return "Light-Medium"
    if skin_L >= 47: return "Medium"
    if skin_L >= 38: return "Medium-Deep"
    if skin_L >= 29: return "Deep"
    return "Very Deep"


def heuristic_chroma(skin_a: float, skin_b: float) -> str:
    value = math.hypot(skin_a, skin_b)
    if value < 18: return "Muted"
    if value < 30: return "Medium"
    return "Bright"


def heuristic_contrast(contrast_proxy: float) -> str:
    if contrast_proxy < 15: return "Low"
    if contrast_proxy < 30: return "Medium"
    return "High"


HEURISTICS = {
    "undertone": lambda f: heuristic_undertone(f["feat_skin_a"], f["feat_skin_b"]),
    "skin_depth": lambda f: heuristic_depth(f["feat_skin_L"]),
    "chroma": lambda f: heuristic_chroma(f["feat_skin_a"], f["feat_skin_b"]),
    "contrast": lambda f: heuristic_contrast(f["feat_contrast_proxy"]),
}


# ---------------------------------------------------------------------------
# Season logic: combining the axes
# ---------------------------------------------------------------------------

def season_from_axes(
    undertone: str,
    depth: str,
    chroma: str,
    contrast: str,
) -> str:
    """Map the four axes onto the 12 seasons (classic seasonal-colour logic).

    This is the transparent, inspectable "season classifier" the roadmap asks
    for before any learned combiner is attempted.
    """
    depth_index = SKIN_DEPTH_ORDER.index(depth) if depth in SKIN_DEPTH_ORDER else 3
    deep = depth_index >= 5          # Medium-Deep, Deep, Very Deep
    light = depth_index <= 2         # Very Fair, Fair, Light

    if undertone == "Olive":
        base = "Autumn" if not deep else "Winter"
    elif undertone in ("Warm", "Neutral-Warm"):
        base = "Spring" if light else "Autumn"
    elif undertone in ("Cool", "Neutral-Cool"):
        base = "Summer" if light else "Winter"
    else:  # Neutral
        if deep:
            base = "Winter" if chroma in ("Bright", "Medium") else "Autumn"
        elif light:
            base = "Summer" if chroma == "Muted" else "Spring"
        else:
            base = "Autumn" if chroma == "Muted" else "Summer"

    if base == "Spring":
        return "Bright Spring" if chroma == "Bright" else ("Warm Spring" if undertone in ("Warm", "Neutral-Warm") else "Light Spring")
    if base == "Summer":
        if chroma == "Bright":
            return "Light Winter"  # cool + light + bright crosses into winter brightness
        if light and chroma == "Medium":
            return "Light Summer"
        return "Cool Summer" if undertone == "Cool" else "Soft Summer"
    if base == "Autumn":
        if deep:
            return "Deep Autumn"
        return "Soft Autumn" if chroma == "Muted" else "Warm Autumn"
    # Winter
    if deep:
        return "Deep Winter" if contrast in ("High", "Medium") else "Deep Autumn"
    if chroma == "Bright":
        return "Bright Spring" if light else "Cool Winter"
    return "Light Winter" if light else "Cool Winter"


def _season_confidence(probs: Dict[str, float]) -> Tuple[str, str, float]:
    """(primary, secondary, confidence) from an axis-probability soup."""
    if not probs:
        return "", "", 0.0
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    primary = ranked[0][0]
    secondary = ranked[1][0] if len(ranked) > 1 else ""
    confidence = ranked[0][1] / max(sum(p[1] for p in ranked), 1e-9)
    return primary, secondary, float(confidence)


def predict_season_from_axes(
    axis_predictions: Dict[str, str],
    axis_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[str, str, float]:
    """Deterministic season from axis predictions; secondary from prob soup."""
    primary = season_from_axes(
        axis_predictions.get("undertone", "Neutral"),
        axis_predictions.get("skin_depth", "Medium"),
        axis_predictions.get("chroma", "Medium"),
        axis_predictions.get("contrast", "Medium"),
    )
    if not axis_probabilities:
        return primary, "", 0.5
    # Marginalize: p(season) ~= sum over plausible axis values.
    soup: Dict[str, float] = {}
    ut_probs = axis_probabilities.get("undertone") or {axis_predictions.get("undertone", "Neutral"): 1.0}
    dp_probs = axis_probabilities.get("skin_depth") or {axis_predictions.get("skin_depth", "Medium"): 1.0}
    ch_probs = axis_probabilities.get("chroma") or {axis_predictions.get("chroma", "Medium"): 1.0}
    ct_probs = axis_probabilities.get("contrast") or {axis_predictions.get("contrast", "Medium"): 1.0}
    for ut, p_ut in ut_probs.items():
        for dp, p_dp in dp_probs.items():
            for ch, p_ch in ch_probs.items():
                for ct, p_ct in ct_probs.items():
                    season = season_from_axes(ut, dp, ch, ct)
                    soup[season] = soup.get(season, 0.0) + p_ut * p_dp * p_ch * p_ct
    return _season_confidence(soup)


# ---------------------------------------------------------------------------
# Trained models
# ---------------------------------------------------------------------------

def _make_candidates(seed: int) -> Dict[str, Any]:
    return {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "hist_gb": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, random_state=seed
        ),
    }


@dataclass
class AxisModel:
    axis: str
    backend: str = "heuristic"          # heuristic | logreg | random_forest | hist_gb
    pipeline: Any = None
    classes: List[str] = field(default_factory=list)
    macro_f1_cv: float = 0.0

    def predict(self, features: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        if self.pipeline is None:
            value = HEURISTICS[self.axis](features)
            return value, {value: 1.0}
        X = feature_matrix([features])[0:1]
        proba = self.pipeline.predict_proba(X)[0]
        classes = list(self.pipeline.classes_)
        probs = {str(c): float(p) for c, p in zip(classes, proba)}
        return max(probs, key=probs.get), probs


@dataclass
class ColourModelBundle:
    axes: Dict[str, AxisModel]
    comparison: Dict[str, Any] = field(default_factory=dict)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Full prediction for one image's features."""
        axis_preds: Dict[str, str] = {}
        axis_probs: Dict[str, Dict[str, float]] = {}
        for axis, model in self.axes.items():
            value, probs = model.predict(features)
            axis_preds[axis] = value
            axis_probs[axis] = probs
        season, secondary, confidence = predict_season_from_axes(axis_preds, axis_probs)
        return {
            "axes": axis_preds,
            "axis_probabilities": axis_probs,
            "prediction": season,
            "prediction_secondary": secondary,
            "prediction_confidence": confidence,
        }


def train_axis(
    rows: List[Dict],
    axis: str,
    person_key: str = "person_id",
    seed: int = 7,
) -> Tuple[AxisModel, Dict[str, float]]:
    """Train one axis with person-grouped CV model selection.

    Returns (AxisModel, {backend: cv_macro_f1}). If sklearn is missing or any
    class has < MIN_CLASS_SAMPLES samples, falls back to the heuristic.
    """
    labels = _labels(rows, axis)
    known = np.array([l in AXES[axis] for l in labels])
    rows_known = [r for r, k in zip(rows, known) if k]
    labels_known = labels[known]

    counts = {c: int((labels_known == c).sum()) for c in set(labels_known)}
    if not rows_known or min(counts.values()) < MIN_CLASS_SAMPLES:
        model = AxisModel(axis=axis, backend="heuristic")
        model.classes = list(AXES[axis])
        return model, {"heuristic": float("nan")}

    X = feature_matrix(rows_known)
    y = labels_known
    groups = np.array([str(r.get(person_key) or r.get("image_id", i)) for i, r in enumerate(rows_known)])

    scores: Dict[str, float] = {}
    try:
        n_splits = min(5, len(set(groups)))
        if n_splits < 2:
            raise ValueError("not enough groups for CV")
        gkf = GroupKFold(n_splits=n_splits)
        for name, candidate in _make_candidates(seed).items():
            fold_scores = []
            for train_idx, test_idx in gkf.split(X, y, groups):
                clf = clone(candidate)
                clf.fit(X[train_idx], y[train_idx])
                pred = clf.predict(X[test_idx])
                fold_scores.append(float(f1_score(y[test_idx], pred, average="macro", zero_division=0)))
            scores[name] = float(np.mean(fold_scores))
        best = max(scores, key=scores.get)
    except Exception:
        scores = {"heuristic": float("nan")}
        best = "heuristic"

    if best == "heuristic":
        model = AxisModel(axis=axis, backend="heuristic")
    else:
        candidates = _make_candidates(seed)
        candidates[best].fit(X, y)
        model = AxisModel(axis=axis, backend=best, pipeline=candidates[best])
    model.classes = list(AXES[axis])
    model.macro_f1_cv = scores.get(best, float("nan"))
    return model, scores


def train_bundle(
    rows: List[Dict],
    axes: Optional[List[str]] = None,
    seed: int = 7,
    season_cv: bool = True,
) -> ColourModelBundle:
    """Train all axis models + record the comparison table.

    ``season_cv=False`` skips the expensive nested person-grouped CV of the
    full season pipeline (use in tests / quick iterations).
    """
    axes = axes or ["undertone", "skin_depth", "chroma", "contrast"]
    bundle = ColourModelBundle(axes={})
    comparison: Dict[str, Any] = {}
    for axis in axes:
        model, scores = train_axis(rows, axis, seed=seed)
        bundle.axes[axis] = model
        comparison[axis] = {"backend": model.backend, "cv_macro_f1": scores}
    # Season classifier: the transparent rule combiner over the trained axes.
    season_scores = _season_cv_score(rows, bundle, seed=seed) if season_cv else float("nan")
    comparison["season_primary"] = {
        "backend": "rule_combiner+" + "+".join(sorted(m.backend for m in bundle.axes.values())),
        "cv_macro_f1": season_scores,
    }
    bundle.comparison = comparison
    return bundle


def _season_cv_score(rows: List[Dict], bundle: ColourModelBundle, seed: int = 7) -> float:
    """Person-grouped CV macro-F1 for the full season pipeline."""
    eligible = [r for r in rows if r.get("season_primary") in SEASON_ORDER]
    if len(eligible) < 30:
        return float("nan")
    groups = sorted({str(r.get("person_id") or i) for i, r in enumerate(eligible)})
    if len(groups) < 2:
        return float("nan")
    from sklearn.model_selection import GroupKFold  # local import: guarded module

    gkf = GroupKFold(n_splits=min(5, len(groups)))
    scores = []
    y_all = np.array([str(r["season_primary"]) for r in eligible])
    for train_idx, test_idx in gkf.split(eligible, y_all, np.array([str(r.get("person_id") or i) for i, r in enumerate(eligible)])):
        train_rows = [eligible[i] for i in train_idx]
        test_rows = [eligible[i] for i in test_idx]
        fold_bundle = train_bundle(train_rows, axes=["undertone", "skin_depth", "chroma", "contrast"], seed=seed)
        preds = [fold_bundle.predict(_features_of(r))["prediction"] for r in test_rows]
        truth = [str(r["season_primary"]) for r in test_rows]
        scores.append(float(f1_score(truth, preds, average="macro", zero_division=0)))
    return float(np.mean(scores)) if scores else float("nan")


def _features_of(row: Dict) -> Dict[str, Any]:
    from .dataset import coerce_feature_row

    return coerce_feature_row({k: row.get(k) for k in FEATURE_COLUMNS})


def save_bundle(bundle: ColourModelBundle, path: Path) -> None:
    if joblib is None:  # pragma: no cover
        raise ImportError("joblib required to persist models: pip install joblib")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> ColourModelBundle:
    if joblib is None:  # pragma: no cover
        raise ImportError("joblib required to load models: pip install joblib")
    return joblib.load(path)
