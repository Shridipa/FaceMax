"""
POST /api/color-analysis — FaceMax colour analysis (Vercel Python function).

Runs the real CV/ML pipeline (ml/facemax_color) in-process:

    base64 image -> face detection -> skin-masked LAB sampling
      -> features (L*, a*, b*, ITA, contrast, ...)
      -> trained joblib models (undertone / skin_depth / chroma / contrast)
      -> season mapping -> JSON

Response schema matches the Express backend (server.ts) exactly, so the
frontend (src/services/api.ts) works unchanged.

Dependencies (root requirements.txt): numpy, opencv-python-headless,
scikit-learn, joblib, lightgbm.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Make the ml/ packages importable (api/ -> project root -> ml/).
# Vercel's Python functions run with CWD at the project root, but we resolve
# relative to this file so it also works when run locally for testing.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_ML_DIR = _THIS_DIR.parent / "ml"
if not _ML_DIR.is_dir():
    _ML_DIR = Path(os.getcwd()) / "ml"
for _p in (str(_ML_DIR), str(_THIS_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DATA_ROOT = _ML_DIR / "data"
_MODELS_CACHE: Dict[str, Any] = {}

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # reject absurd payloads early


def _preload_libgomp() -> None:
    """dlopen the bundled GNU OpenMP runtime before sklearn/lightgbm load.

    Vercel's Amazon Linux runtime does not ship libgomp.so.1, which the
    scikit-learn / lightgbm manylinux wheels depend on. We bundle a portable
    copy in api/lib/ and load it with RTLD_GLOBAL so joblib.load() works.
    """
    if sys.platform != "linux":
        return
    import ctypes

    candidates = (
        _THIS_DIR / "lib" / "libgomp.so.1",
        Path(os.getcwd()) / "api" / "lib" / "libgomp.so.1",
    )
    for candidate in candidates:
        if candidate.is_file():
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                traceback.print_exc()


_preload_libgomp()


# ---------------------------------------------------------------------------
# Model loading (cached across warm invocations)
# ---------------------------------------------------------------------------
def _load_axis_predictor(axis: str):
    """Return (predictor, model_name) or (None, None) if no model exists."""
    cache_key = f"{axis}:{_DATA_ROOT}"
    if cache_key in _MODELS_CACHE:
        return _MODELS_CACHE[cache_key]

    model_dir = _DATA_ROOT / "models" / axis
    if not model_dir.is_dir() or not any(model_dir.iterdir()):
        # Fall back to the committed demo models.
        demo = _DATA_ROOT / "demo" / "models" / axis
        if demo.is_dir() and any(demo.iterdir()):
            model_dir = demo
        else:
            _MODELS_CACHE[cache_key] = (None, None)
            return None, None

    model_files = sorted(model_dir.glob("*/model.joblib"))
    if not model_files:
        _MODELS_CACHE[cache_key] = (None, None)
        return None, None

    # Pick the NEWEST version per axis (artifact dirs are ...-vNNN; the old
    # code took the alphabetically-first, so a trained v002 was shadowed by
    # v001). Keeps the API contract identical - just the right artifact.

    try:
        import joblib

        predictor = joblib.load(model_files[-1])
        entry = (predictor, model_files[-1].parent.name)
    except Exception:
        traceback.print_exc()
        entry = (None, None)
    _MODELS_CACHE[cache_key] = entry
    return entry


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_image(data_url_or_b64: str) -> Tuple[int, Dict[str, Any]]:
    """Analyse one base64/data-URL image. Returns (http_status, payload)."""
    # 1. Validate + decode ---------------------------------------------------
    if not data_url_or_b64:
        return 400, {"ok": False, "error": "Image data (base64) is required."}

    raw = data_url_or_b64
    if ";base64," in raw:
        raw = raw.split(";base64,", 1)[1]
    raw = raw.strip()

    try:
        image_bytes = base64.b64decode(raw, validate=False)
    except Exception:
        return 400, {"ok": False, "error": "Image data could not be decoded."}
    if len(image_bytes) < 1024:
        return 400, {"ok": False, "error": "Image data looks empty or truncated."}
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return 413, {"ok": False, "error": "Image is too large after decoding."}

    # 2. Run the real pipeline (extract -> features -> models -> season) -----
    try:
        from facemax_color.colour import extract_image
        from facemax_color.models import (
            AxisModel,
            feature_matrix,
            predict_season_from_axes,
        )
    except Exception:
        traceback.print_exc()
        return 500, {
            "ok": False,
            "error": "Colour analysis backend failed to start. Please try again later.",
        }

    # Write to a temp file so np.fromfile-based loading works.
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as fh:
            fh.write(image_bytes)
            tmp_path = Path(fh.name)

        ext = extract_image("selfie", tmp_path)
        if not ext.ok:
            msg = {
                "no_face_detected": "We couldn't detect a face in this photo. Please use a clear, front-facing photo.",
                "insufficient_skin_pixels": "The face is too small or unclear to analyse. Please move closer to the camera.",
                "regions_below_pixel_minimum": "The face is too small or unclear to analyse. Please move closer to the camera.",
            }.get(ext.error or "", "This photo couldn't be analysed. Please try another one with even lighting.")
            return 200, {"ok": False, "error": msg}

        features = ext.as_feature_row()

        # 2b. Quality gate (ML spec #14/#15): refuse to colour-analyse images
        # whose extraction is dominated by colour cast / clipping / mask
        # failure. "Photo quality is insufficient..." is more honest than a
        # confident wrong season. Same gate as the CLI predict path.
        from facemax_color.models import quality_gate

        gate = quality_gate(features)
        if not gate.get("ok"):
            return 200, {
                "ok": False,
                "error": "Photo quality is insufficient for reliable colour analysis.",
                "quality_problems": gate.get("problems", []),
                "retry_hint": (
                    "Use diffuse natural light, avoid backlight and filters, "
                    "and fill the frame with your face."
                ),
            }

        # 3. Per-axis predictions: trained model when present, else heuristic.
        axes_predictions: Dict[str, str] = {}
        axes_probabilities: Dict[str, Dict[str, float]] = {}
        model_used: Dict[str, str] = {}

        for axis in ("undertone", "skin_depth", "chroma", "contrast"):
            predictor, model_name = _load_axis_predictor(axis)
            if predictor is not None:
                try:
                    if hasattr(predictor, "predict_row"):
                        result = predictor.predict_row(features)
                        axes_predictions[axis] = result["prediction"]
                        axes_probabilities[axis] = result.get(
                            "probabilities", {result["prediction"]: 1.0}
                        )
                    else:
                        X = feature_matrix([features])
                        proba = predictor.predict_proba(X)[0]
                        classes = [str(c) for c in predictor.classes_]
                        probs = {c: float(p) for c, p in zip(classes, proba)}
                        axes_predictions[axis] = max(probs, key=probs.get)
                        axes_probabilities[axis] = probs
                    model_used[axis] = model_name or "model"
                    continue
                except Exception:
                    traceback.print_exc()  # fall through to heuristic

            pred, probs = AxisModel(axis=axis).predict(features)
            axes_predictions[axis] = pred
            axes_probabilities[axis] = probs
            model_used[axis] = "heuristic"

        # 4. Season from axes (same logic as the CLI / Express backend).
        season_primary, season_secondary, season_confidence = predict_season_from_axes(
            axes_predictions, axes_probabilities
        )

        payload = {
            "ok": True,
            "season": {
                "primary": season_primary,
                "secondary": season_secondary,
                "confidence": round(season_confidence, 3),
            },
            "axes": {
                "undertone": {
                    "prediction": axes_predictions.get("undertone", ""),
                    "probabilities": axes_probabilities.get("undertone", {}),
                    "model": model_used.get("undertone", ""),
                },
                "skin_depth": {
                    "prediction": axes_predictions.get("skin_depth", ""),
                    "probabilities": axes_probabilities.get("skin_depth", {}),
                    "model": model_used.get("skin_depth", ""),
                },
                "chroma": {
                    "prediction": axes_predictions.get("chroma", ""),
                    "probabilities": axes_probabilities.get("chroma", {}),
                    "model": model_used.get("chroma", ""),
                },
                "contrast": {
                    "prediction": axes_predictions.get("contrast", ""),
                    "probabilities": axes_probabilities.get("contrast", {}),
                    "model": model_used.get("contrast", ""),
                },
            },
            "features": {
                "skin_L": round(features.get("feat_skin_L", 0), 1),
                "skin_a": round(features.get("feat_skin_a", 0), 1),
                "skin_b": round(features.get("feat_skin_b", 0), 1),
                "ita": round(features.get("feat_ita", 0), 1),
                "hair_L": round(features.get("feat_hair_L", 0), 1),
                "eye_L": round(features.get("feat_eye_L", 0), 1),
                "contrast_proxy": round(features.get("feat_contrast_proxy", 0), 1),
                "brightness": features.get("feat_exposure_flag", "ok"),
                "skin_pixels": features.get("feat_skin_pixels", 0),
            },
        }
        return 200, payload
    except Exception:
        traceback.print_exc()
        return 500, {
            "ok": False,
            "error": "Analysis failed. Please try again.",
        }
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Vercel handler (BaseHTTPRequestHandler — no framework needed)
# ---------------------------------------------------------------------------
CORS_HEADERS = (
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "POST, OPTIONS"),
    ("Access-Control-Allow-Headers", "Content-Type"),
)


class handler(BaseHTTPRequestHandler):
    server_version = "FaceMaxColorAnalysis/1.0"

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        for key, value in CORS_HEADERS:
            self.send_header(key, value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(200)
        for key, value in CORS_HEADERS:
            self.send_header(key, value)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 16 * 1024 * 1024:
                self._send_json(400, {"ok": False, "error": "Invalid request body size."})
                return
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                self._send_json(400, {"ok": False, "error": "Request body must be JSON."})
                return

            status, payload = analyze_image(str(data.get("image") or ""))
            self._send_json(status, payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away; nothing to send
        except Exception:
            traceback.print_exc()
            try:
                self._send_json(500, {"ok": False, "error": "Analysis failed. Please try again."})
            except Exception:
                pass

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
        if os.environ.get("FMAX_VERBOSE_LOGS"):
            super().log_message(fmt, *args)
