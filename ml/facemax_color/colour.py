"""FaceMaxx colour-extraction pipeline.

    Image -> face detection -> face landmarks/segmentation -> skin mask
          -> remove hair/eyes/lips/brows/shadows -> skin pixel sampling
          -> white-balance / lighting analysis -> RGB -> LAB
          -> L*, a*, b* -> undertone / depth / chroma / contrast features

Dependencies: numpy + OpenCV (cv2). Both are imported lazily with clear errors
so the rest of the toolkit (evaluator, CLI, tests) still works without them.

Region sampling strategy (all coordinates are normalized 0-100 like the
FaceMaxx landmark schema used by the web app):

- forehead : brow band well above the eyes, centre columns
- left / right cheek : beside the nose, below the eyes
- chin : below the lips

Each region is sampled as the median of skin-masked LAB pixels, which resists
outliers (moles, highlighter, stray hair pixels). A 5x5 median filter plus
shadow rejection (keep pixels within 1.5 MAD of region median L*) cleans the
mask before sampling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_ERROR = (
        "OpenCV (cv2) is required for colour extraction: pip install opencv-python"
    )

# ---------------------------------------------------------------------------
# Region geometry (normalized coordinates, x/y 0-100 within the detected face box)
# ---------------------------------------------------------------------------
# Each region: (x0, y0, x1, y1) as fractions of the face box.
REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "forehead": (0.30, 0.12, 0.70, 0.28),
    "left_cheek": (0.12, 0.45, 0.35, 0.70),
    "right_cheek": (0.65, 0.45, 0.88, 0.70),
    "chin": (0.38, 0.82, 0.62, 0.95),
}
REGION_PIXEL_MIN = 40  # refuse regions with fewer usable skin pixels


@dataclass
class ExtractionResult:
    """Everything extracted from one image; written to processed/ tables."""

    image_id: str
    ok: bool = False
    error: Optional[str] = None

    # Per-region LAB medians (the core colour signals)
    region_lab: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    # Global skin LAB (median over the whole face mask)
    skin_L: float = float("nan")
    skin_a: float = float("nan")
    skin_b: float = float("nan")

    # Derived colour features
    ita_deg: float = float("nan")          # Individual Typology Angle
    undertone_proxy: float = float("nan")  # b* vs a* balance
    olive_proxy: float = float("nan")      # green-shift indicator (a* low vs b*)
    chroma_proxy: float = float("nan")     # saturation-ish (a*, b* magnitude)
    depth_proxy: float = float("nan")      # 100 - L*
    skin_hue_deg: float = float("nan")     # CIELAB hue angle atan2(b*, a*) in degrees
    hair_L: float = float("nan")           # hair region luminance
    eye_L: float = float("nan")            # eye-region luminance
    contrast_proxy: float = float("nan")   # skin minus min(hair, eye) luminance

    # Multi-region skin statistics (variability flags shadow/makeup/extraction issues)
    skin_region_count: int = 0             # how many regions produced usable samples
    skin_L_spread: float = float("nan")    # std of region L* medians (uneven light)
    skin_a_spread: float = float("nan")    # std of region a* medians
    skin_b_spread: float = float("nan")    # std of region b* medians

    # Lighting / quality diagnostics (these feed the ROBUSTNESS sections)
    white_balance_b_minus_a: float = float("nan")  # >0 warm cast, <0 cool cast
    grey_world_shift: float = float("nan")         # magnitude of global colour cast
    brightness: float = float("nan")               # mean face L*
    highlight_clip_frac: float = float("nan")      # share of face-box pixels ~255
    shadow_clip_frac: float = float("nan")         # share of face-box pixels ~0
    exposure_flag: str = ""                        # ok | under | over
    face_fraction: float = float("nan")            # face box / image area
    face_box_source: str = "detector"              # detector | demo_fallback
    skin_pixels: int = 0

    def as_feature_row(self) -> Dict[str, Any]:
        """Flat dict for joining onto annotation rows (prefixed, no collisions)."""
        return {
            "feat_skin_L": self.skin_L,
            "feat_skin_a": self.skin_a,
            "feat_skin_b": self.skin_b,
            "feat_ita": self.ita_deg,
            "feat_undertone_proxy": self.undertone_proxy,
            "feat_olive_proxy": self.olive_proxy,
            "feat_chroma_proxy": self.chroma_proxy,
            "feat_depth_proxy": self.depth_proxy,
            "feat_skin_hue": self.skin_hue_deg,
            "feat_hair_L": self.hair_L,
            "feat_eye_L": self.eye_L,
            "feat_contrast_proxy": self.contrast_proxy,
            "feat_skin_region_count": self.skin_region_count,
            "feat_skin_L_spread": self.skin_L_spread,
            "feat_skin_a_spread": self.skin_a_spread,
            "feat_skin_b_spread": self.skin_b_spread,
            "feat_wb_b_minus_a": self.white_balance_b_minus_a,
            "feat_grey_world_shift": self.grey_world_shift,
            "feat_brightness": self.brightness,
            "feat_highlight_clip_frac": self.highlight_clip_frac,
            "feat_shadow_clip_frac": self.shadow_clip_frac,
            "feat_exposure_flag": self.exposure_flag,
            "feat_face_fraction": self.face_fraction,
            "feat_face_box_source": self.face_box_source,
            "feat_skin_pixels": self.skin_pixels,
            "feat_ok": self.ok,
        }


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise ImportError(_CV2_ERROR)


def load_image(path: Path) -> "np.ndarray":
    """Read an image as BGR ndarray, robust to exotic paths."""
    _require_cv2()
    data = np.fromfile(str(path), dtype=np.uint8)  # handles unicode paths on Windows
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable image: {path}")
    return image


def detect_face_box(image_bgr: "np.ndarray") -> Optional[Tuple[int, int, int, int]]:
    """Haar-cascade face detection; returns (x, y, w, h) or None."""
    _require_cv2()
    cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    # Largest face wins (selfies: the subject dominates the frame).
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def rgb_skin_mask(image_bgr: "np.ndarray", face_box: Tuple[int, int, int, int]) -> "np.ndarray":
    """Boolean skin mask inside the face box (RGB + chroma heuristics).

    Rules (classic skin-chroma filters, tuned to be permissive across depths):
      R > 60, R > G, R > B, |R-G| > 8 and Cb in [77, 130], Cr in [133, 175].
    """
    _require_cv2()
    x, y, w, h = face_box
    roi = image_bgr[y : y + h, x : x + w]
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.int16)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Cr = ycc[..., 1].astype(np.int16)
    Cb = ycc[..., 2].astype(np.int16)

    mask = (
        (R > 60)
        & (R > G)
        & (R > B)
        & (np.abs(R - G) > 8)
        & (Cb >= 77)
        & (Cb <= 130)
        & (Cr >= 133)
        & (Cr <= 175)
    )
    # Clean: median filter then morphological opening to drop speckle.
    mask_u8 = (mask.astype(np.uint8)) * 255
    mask_u8 = cv2.medianBlur(mask_u8, 5)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask_u8 > 0


def _mad_clean(values: "np.ndarray", keep: float = 1.5) -> "np.ndarray":
    """Drop outliers beyond keep*MAD from the median (shadow/specular removal)."""
    if values.size == 0:
        return values
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        return values
    return values[np.abs(values - med) <= keep * mad * 1.4826]


def region_lab_samples(
    image_bgr: "np.ndarray",
    face_box: Tuple[int, int, int, int],
    skin_mask: "np.ndarray",
) -> Dict[str, Tuple[float, float, float]]:
    """Median LAB per named region, masked to skin + MAD-cleaned."""
    _require_cv2()
    x, y, w, h = face_box
    roi = image_bgr[y : y + h, x : x + w]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0  # OpenCV L is 0-255; normalize to 0-100
    lab[..., 1] -= 128.0  # OpenCV a*/b* are 0-255 centered at 128 -> re-center to 0
    lab[..., 2] -= 128.0

    result: Dict[str, Tuple[float, float, float]] = {}
    for name, (fx0, fy0, fx1, fy1) in REGIONS.items():
        x0, y0 = int(fx0 * w), int(fy0 * h)
        x1, y1 = int(fx1 * w), int(fy1 * h)
        sub_mask = skin_mask[y0:y1, x0:x1]
        sub_lab = lab[y0:y1, x0:x1]
        if sub_mask.sum() < REGION_PIXEL_MIN:
            continue
        pixels = sub_lab[sub_mask]
        # Shadow/specular rejection around the region-median L* (keeps a* / b*
        # of surviving pixels coherent; rejects shadow and highlight bands).
        L_med = float(np.median(pixels[:, 0]))
        L_mad = float(np.median(np.abs(pixels[:, 0] - L_med))) * 1.4826
        keep_idx = np.abs(pixels[:, 0] - L_med) <= 1.5 * L_mad + 1e-6
        pixels = pixels[keep_idx]
        if len(pixels) < REGION_PIXEL_MIN:
            continue
        result[name] = (
            float(np.median(pixels[:, 0])),
            float(np.median(pixels[:, 1])),
            float(np.median(pixels[:, 2])),
        )
    return result


def clipping_fractions(image_bgr: "np.ndarray", face_box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Share of face-box pixels at/next-to 0 (shadow) or 255 (highlight).

    A cheap, model-usable proxy for exposure failure: a low-light photo has a
    high shadow_clip_frac; a blown-out one a high highlight_clip_frac. These
    feed the quality gate instead of silently colouring the predictions.
    """
    _require_cv2()
    x, y, w, h = face_box
    roi = image_bgr[y : y + h, x : x + w]
    if roi.size == 0:
        return float("nan"), float("nan")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    total = float(gray.size)
    shadow = float(np.count_nonzero(gray <= 5)) / total
    highlight = float(np.count_nonzero(gray >= 250)) / total
    return highlight, shadow


def estimate_white_balance(image_bgr: "np.ndarray", face_box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Lighting diagnostics inside the face box.

    Returns (wb_b_minus_a, grey_world_shift):
    - wb_b_minus_a : median skin b* - a* imbalance; strongly positive suggests a
      yellow/warm cast, strongly negative a blue/cool cast.
    - grey_world_shift : how far the face-box channel means are from grey
      (0 = neutral illuminant).
    """
    _require_cv2()
    x, y, w, h = face_box
    roi = image_bgr[y : y + h, x : x + w]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    a_med = float(np.median(lab[..., 1]))
    b_med = float(np.median(lab[..., 2]))
    means = roi.reshape(-1, 3).mean(axis=0)  # BGR order
    grey_shift = float(np.abs(means[0] - means[1]) + np.abs(means[1] - means[2])) / 2.0
    return b_med - a_med, grey_shift


def hair_eye_luminance(
    image_bgr: "np.ndarray", face_box: Tuple[int, int, int, int]
) -> Tuple[float, float]:
    """Median L* of hair band (above brow) and eye band for contrast features.

    Hair is sampled just OUTSIDE the top of the face box (forehead above);
    eyes from the classic eye band inside the face.
    """
    _require_cv2()
    x, y, w, h = face_box
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)

    # Hair band: above the face box, same width, 35% of box height.
    top = max(0, y - int(0.35 * h))
    hair_band = gray[top:y, x : x + w]
    hair_L = float(np.median(hair_band)) * 100.0 / 255.0 if hair_band.size else float("nan")

    # Eye band: 30-45% of face height.
    eye_band = gray[y + int(0.30 * h) : y + int(0.45 * h), x : x + w]
    eye_L = float(np.median(eye_band)) * 100.0 / 255.0 if eye_band.size else float("nan")
    return hair_L, eye_L


def extract_image(image_id: str, path: Path, fallback_box=None) -> ExtractionResult:
    """Full pipeline for one image. Never raises; check .ok / .error.

    ``fallback_box``: (x, y, w, h) used ONLY when the detector fails - e.g. the
    synthetic demo, where the drawn face geometry is known. Its use is logged
    loudly as DEMO_FALLBACK_FACE_BOX and recorded in face_box_source.
    """
    result = ExtractionResult(image_id=image_id)
    try:
        image = load_image(path)
        box = detect_face_box(image)
        if box is None and fallback_box is not None:
            print(f"DEMO_FALLBACK_FACE_BOX: {image_id} -> {tuple(fallback_box)}")
            box = tuple(int(v) for v in fallback_box)
            result.face_box_source = "demo_fallback"
        if box is None:
            result.error = "no_face_detected"
            return result

        x, y, w, h = box
        ih, iw = image.shape[:2]
        result.face_fraction = (w * h) / float(iw * ih)

        mask = rgb_skin_mask(image, box)
        result.skin_pixels = int(mask.sum())
        if result.skin_pixels < REGION_PIXEL_MIN * len(REGIONS) // 2:
            result.error = "insufficient_skin_pixels"
            return result

        result.region_lab = region_lab_samples(image, box, mask)
        if len(result.region_lab) < 2:
            result.error = "regions_below_pixel_minimum"
            return result

        # Global skin stats: median across region medians + region variability.
        all_L = np.array([lab[0] for lab in result.region_lab.values()])
        all_a = np.array([lab[1] for lab in result.region_lab.values()])
        all_b = np.array([lab[2] for lab in result.region_lab.values()])
        result.skin_L, result.skin_a, result.skin_b = (
            float(np.median(all_L)),
            float(np.median(all_a)),
            float(np.median(all_b)),
        )
        result.skin_region_count = len(result.region_lab)
        # Cross-region spread: uneven lighting / makeup / shadow shows up here.
        result.skin_L_spread = float(np.std(all_L))
        result.skin_a_spread = float(np.std(all_a))
        result.skin_b_spread = float(np.std(all_b))

        # Derived colour-science features -------------------------------
        L, A, B = result.skin_L, result.skin_a, result.skin_b
        result.ita_deg = float(
            np.degrees(np.arctan((L - 50.0) / max(abs(B), 1e-6))) * (1.0 if B >= 0 else -1.0)
        )
        result.undertone_proxy = float(B - A)          # warm >= 0, cool < 0
        # Olive proxy = a*/b* magnitude ratio (low a* vs b* = green/olive cast).
        # Sign-safe: for genuinely COOL skin a* can be <= 0, and a plain A/B
        # division would flip the signal (negative ratio reads "very olive").
        # We want olive-ness to track a* DEPLETION relative to b*, so use the
        # magnitudes and clamp negatives of a* to 0 (a* <= 0 means NO redness,
        # which is maximum olive-ness, not an anti-olive signal).
        result.olive_proxy = float(max(A, 0.0) / max(abs(B), 1e-6))
        result.chroma_proxy = float(np.hypot(A, B))
        result.depth_proxy = float(100.0 - L)
        result.skin_hue_deg = float(np.degrees(np.arctan2(B, A)) % 360.0)  # CIELAB hue angle

        hair_L, eye_L = hair_eye_luminance(image, box)
        result.hair_L, result.eye_L = hair_L, eye_L
        if not (np.isnan(hair_L) or np.isnan(L)):
            # High contrast = skin much lighter than hair/eyes (dark hair, light skin).
            result.contrast_proxy = float(L - min(hair_L, eye_L) if not np.isnan(eye_L) else L - hair_L)

        wb, grey = estimate_white_balance(image, box)
        result.white_balance_b_minus_a = wb
        result.grey_world_shift = grey
        hi_clip, sh_clip = clipping_fractions(image, box)
        result.highlight_clip_frac = hi_clip
        result.shadow_clip_frac = sh_clip
        result.brightness = L
        result.exposure_flag = "ok"
        if L < 35:
            result.exposure_flag = "under"
        elif L > 88:
            result.exposure_flag = "over"

        result.ok = True
        return result
    except Exception as exc:  # defensive: one bad file must not kill a batch
        result.error = f"exception: {exc}"
        return result


def extract_batch(
    image_ids: List[str],
    raw_dir: Path,
    progress_every: int = 200,
    fallback_boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
) -> Dict[str, ExtractionResult]:
    """Extract features for many images; skips ids whose file is missing.

    ``fallback_boxes`` maps image_id -> (x, y, w, h) used only when the detector
    fails (demo/synthetic use; logged as DEMO_FALLBACK_FACE_BOX).
    """
    from .dataset import find_image

    results: Dict[str, ExtractionResult] = {}
    for i, image_id in enumerate(image_ids, 1):
        path = find_image(Path(raw_dir), image_id)
        if path is None:
            missing = ExtractionResult(image_id=image_id, error="file_missing")
            results[image_id] = missing
            continue
        fallback = (fallback_boxes or {}).get(image_id)
        results[image_id] = extract_image(image_id, path, fallback_box=fallback)
        if progress_every and i % progress_every == 0:
            print(f"  extracted {i}/{len(image_ids)}")
    return results
