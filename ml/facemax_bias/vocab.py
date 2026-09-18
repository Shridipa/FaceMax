"""Canonical vocabularies for FaceMaxx bias evaluation.

Keep these in one place so all analyzers, the report generator, and future
annotation tooling agree on exact label spelling.
"""
from __future__ import annotations

# Ordered shallow -> deep. The dashboard prints subgroups in this order.
SKIN_DEPTH_ORDER = (
    "Very Fair",
    "Fair",
    "Light",
    "Light-Medium",
    "Medium",
    "Medium-Deep",
    "Deep",
    "Very Deep",
)

# Ordered cool -> warm. Olive is treated as its own axis value.
UNDERTONE_ORDER = (
    "Cool",
    "Neutral-Cool",
    "Neutral",
    "Neutral-Warm",
    "Warm",
    "Olive",
)

CONTRAST_ORDER = ("Low", "Medium", "High")

CHROMA_ORDER = ("Muted", "Medium", "Bright")

LIGHTING_ORDER = (
    "Natural",
    "Indoor warm",
    "Indoor cool",
    "Low light",
    "Mixed",
    "Overexposed",
)

CAMERA_ORDER = (
    "iPhone",
    "Samsung",
    "Pixel",
    "Budget Android",
    "Webcam",
    "DSLR",
)

# Ordered excellent -> poor. The dashboard prints subgroups in this order.
QUALITY_ORDER = ("Excellent", "Good", "Acceptable", "Poor", "Very Poor")

SOURCE_ORDER = (
    "Studio",
    "Celeb",
    "Instagram",
    "Phone selfie",
    "Webcam",
    "Synthetic",
)

MAKEUP_ORDER = (
    "No makeup",
    "Light makeup",
    "Heavy makeup",
    "Filter",
    "Skin smoothing",
)

# The 12 standard seasonal colour categories.
SEASON_ORDER = (
    "Light Spring",
    "Bright Spring",
    "Warm Spring",
    "Light Summer",
    "Soft Summer",
    "Cool Summer",
    "Soft Autumn",
    "Warm Autumn",
    "Deep Autumn",
    "Light Winter",
    "Cool Winter",
    "Deep Winter",
)

SEASONS = frozenset(SEASON_ORDER)

MAJOR_AXES = {
    "skin_depth": SKIN_DEPTH_ORDER,
    "undertone": UNDERTONE_ORDER,
    "contrast": CONTRAST_ORDER,
    "chroma": CHROMA_ORDER,
    "lighting": LIGHTING_ORDER,
    "camera": CAMERA_ORDER,
    "image_quality": QUALITY_ORDER,
    "source": SOURCE_ORDER,
    "makeup": MAKEUP_ORDER,
    "season_primary": SEASON_ORDER,
}

QUALITY_RANK = {name: rank for rank, name in enumerate(QUALITY_ORDER)}

COLUMNS = (
    "image_id",
    "person_id",
    "source",
    "license",
    "image_quality",
    "lighting_type",
    "camera_type",
    "makeup",
    "skin_depth",
    "skin_L",
    "skin_a",
    "skin_b",
    "undertone",
    "undertone_confidence",
    "olive_score",
    "hair_depth",
    "hair_temperature",
    "eye_depth",
    "contrast_score",
    "chroma_score",
    "season_primary",
    "season_secondary",
    "season_confidence",
    "annotator_1",
    "annotator_2",
    "annotator_3",
    "annotation_agreement",
    "split",
)

GROUP_SIZE_WARN = 30  # minimum images per subgroup for a trustworthy rate


def normalize_lighting(value: str) -> str:
    """Map free-form lighting labels onto the canonical vocabulary."""
    key = str(value or "").strip().lower()
    aliases = {
        "natural": "Natural",
        "daylight": "Natural",
        "outdoor": "Natural",
        "warm indoor": "Indoor warm",
        "indoor warm": "Indoor warm",
        "tungsten": "Indoor warm",
        "cool indoor": "Indoor cool",
        "indoor cool": "Indoor cool",
        "fluorescent": "Indoor cool",
        "low light": "Low light",
        "dim": "Low light",
        "mixed": "Mixed",
        "overexposed": "Overexposed",
        "blown out": "Overexposed",
    }
    return aliases.get(key, str(value).strip())


def normalize_quality(value: str) -> str:
    """Map free-form quality labels onto the canonical vocabulary."""
    key = str(value or "").strip().lower()
    aliases = {
        "excellent": "Excellent",
        "good": "Good",
        "acceptable": "Acceptable",
        "ok": "Acceptable",
        "poor": "Poor",
        "very poor": "Very Poor",
        "awful": "Very Poor",
    }
    return aliases.get(key, str(value).strip())


def quality_at_least(value: str, floor: str) -> bool:
    """True when image quality is at least as good as ``floor``."""
    return QUALITY_RANK.get(normalize_quality(value), 99) <= QUALITY_RANK[floor]
