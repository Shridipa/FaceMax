"""Synthetic demo dataset with *deliberately planted* biases.

The generator encodes every pathology from the bias spec so the dashboard can
be seen working end-to-end without real user data:

- representation bias  (Very Deep/Deep skin severely under-represented)
- subgroup bias        (deeper skin -> lower simulated accuracy)
- lighting sensitivity (warm/cool indoor light shifts predicted undertone)
- camera bias          (Samsung selfies skew "warm", webcam skews noisy)
- quality decay        (poor-quality images degrade accuracy sharply)
- label disagreement   (Deep Autumn / Deep Winter boundary is ambiguous)
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .vocab import (
    CAMERA_ORDER,
    CHROMA_ORDER,
    CONTRAST_ORDER,
    LIGHTING_ORDER,
    MAKEUP_ORDER,
    QUALITY_ORDER,
    SEASON_ORDER,
    SKIN_DEPTH_ORDER,
    UNDERTONE_ORDER,
    normalize_lighting,
    normalize_quality,
)

# Representation bias: shallower skin buckets are deliberately over-sampled.
SKIN_WEIGHTS = {
    "Very Fair": 0.20,
    "Fair": 0.24,
    "Light": 0.16,
    "Light-Medium": 0.14,
    "Medium": 0.12,
    "Medium-Deep": 0.08,
    "Deep": 0.045,
    "Very Deep": 0.015,
}

# Subgroup bias: base per-skin-depth accuracy used by the fake model.
DEPTH_ACCURACY = {
    "Very Fair": 0.92,
    "Fair": 0.90,
    "Light": 0.88,
    "Light-Medium": 0.87,
    "Medium": 0.85,
    "Medium-Deep": 0.80,
    "Deep": 0.72,
    "Very Deep": 0.66,
}

LIGHTING_ACCURACY = {
    "Natural": 1.00,       # multiplier relative to depth accuracy
    "Indoor warm": 0.90,
    "Indoor cool": 0.93,
    "Low light": 0.70,
    "Mixed": 0.88,
    "Overexposed": 0.75,
}

QUALITY_ACCURACY = {
    "Excellent": 1.00,
    "Good": 0.97,
    "Acceptable": 0.90,
    "Poor": 0.68,
    "Very Poor": 0.45,
}

CAMERA_ACCURACY = {
    "iPhone": 1.00,
    "Samsung": 0.95,
    "Pixel": 1.00,
    "Budget Android": 0.88,
    "Webcam": 0.85,
    "DSLR": 1.00,
}

MAKEUP_ACCURACY = {
    "No makeup": 1.00,
    "Light makeup": 0.97,
    "Heavy makeup": 0.85,
    "Filter": 0.70,
    "Skin smoothing": 0.80,
}

# Season prior: Deep Autumn/Winter dominate the pool (class imbalance).
SEASON_WEIGHTS = {
    "Deep Autumn": 0.17,
    "Deep Winter": 0.15,
    "Warm Autumn": 0.12,
    "Cool Winter": 0.11,
    "Soft Autumn": 0.10,
    "Soft Summer": 0.08,
    "Light Summer": 0.06,
    "Bright Spring": 0.05,
    "Warm Spring": 0.05,
    "Light Winter": 0.04,
    "Light Spring": 0.04,
    "Cool Summer": 0.03,
}

# The DA/DW boundary is ambiguous: annotators disagree on a slice of images.
DA_DW_DISAGREEMENT_RATE = 0.25


def _weighted_pick(rng: random.Random, weights: Dict[str, float]) -> str:
    population = list(weights)
    return rng.choices(population, weights=[weights[p] for p in population], k=1)[0]


def generate_rows(
    n_people: int = 1000,
    seed: int = 7,
    images_per_person: int = 2,
) -> List[Dict]:
    """Create a synthetic but statistically realistic annotation table."""
    rng = random.Random(seed)
    rows: List[Dict] = []
    image_counter = 0

    for person in range(n_people):
        person_id = f"P{person:05d}"
        skin_depth = _weighted_pick(rng, SKIN_WEIGHTS)
        true_season = _weighted_pick(rng, SEASON_WEIGHTS)
        undertone = _weighted_pick(
            rng,
            {u: w for u, w in UNDERTONE_WEIGHTS_FOR(skin_depth).items()},
        )
        contrast = _weighted_pick(rng, {c: 1 / len(CONTRAST_ORDER) for c in CONTRAST_ORDER})
        chroma = _weighted_pick(rng, {c: 1 / len(CHROMA_ORDER) for c in CHROMA_ORDER})

        # Person-level split assignment: ~70% train, 15% val, 15% test.
        # Every image of one person inherits the same split (no leakage).
        roll = rng.random()
        split_for_person = "test" if roll < 0.15 else ("val" if roll < 0.30 else "train")

        # Skin LAB values loosely consistent with depth (L decreases with depth).
        depth_index = SKIN_DEPTH_ORDER.index(skin_depth)
        skin_L = 85 - depth_index * 7 + rng.uniform(-3, 3)
        skin_a = 8 + rng.uniform(-3, 3) + (2 if "Autumn" in true_season else 0)
        skin_b = 16 + rng.uniform(-4, 4) + (4 if "Warm" in true_season or "Autumn" in true_season else 0)

        for image in range(images_per_person):
            image_counter += 1
            lighting = _weighted_pick(
                rng,
                {"Natural": 0.55, "Indoor warm": 0.18, "Indoor cool": 0.12, "Low light": 0.08, "Mixed": 0.05, "Overexposed": 0.02},
            )
            camera = _weighted_pick(
                rng, {"iPhone": 0.35, "Samsung": 0.22, "Pixel": 0.15, "Budget Android": 0.12, "Webcam": 0.10, "DSLR": 0.06}
            )
            quality = _weighted_pick(
                rng, {"Excellent": 0.20, "Good": 0.35, "Acceptable": 0.25, "Poor": 0.15, "Very Poor": 0.05}
            )
            makeup = _weighted_pick(
                rng, {"No makeup": 0.45, "Light makeup": 0.20, "Heavy makeup": 0.10, "Filter": 0.15, "Skin smoothing": 0.10}
            )
            source = {
                "iPhone": "Phone selfie",
                "Samsung": "Phone selfie",
                "Pixel": "Phone selfie",
                "Budget Android": "Phone selfie",
                "Webcam": "Webcam",
                "DSLR": "Studio",
            }[camera]

            # ---- simulate the model -------------------------------------
            base = DEPTH_ACCURACY[skin_depth]
            effective = (
                base
                * LIGHTING_ACCURACY[normalize_lighting(lighting)]
                * QUALITY_ACCURACY[normalize_quality(quality)]
                * CAMERA_ACCURACY[camera]
                * MAKEUP_ACCURACY[makeup]
            )
            correct = rng.random() < effective

            if correct:
                prediction = true_season
            else:
                # Structured errors: confusions happen between adjacent seasons.
                prediction = _adjacent_season(rng, true_season)

            prediction_secondary = true_season if not correct else _adjacent_season(rng, true_season)
            prediction_confidence = round(rng.uniform(0.55, 0.99) if correct else rng.uniform(0.45, 0.8), 3)

            # Lighting shift: yellow/blue light literally pushes predictions warm/cool
            # even when the base prediction was right.
            if lighting == "Indoor warm" and rng.random() < 0.35:
                prediction = _shift_warm(prediction)
            if lighting == "Indoor cool" and rng.random() < 0.25:
                prediction = _shift_cool(prediction)

            # ---- simulate annotators ------------------------------------
            labels: List[str] = []
            for _ in range(3):
                if true_season == "Deep Autumn" and rng.random() < DA_DW_DISAGREEMENT_RATE:
                    labels.append("Deep Winter")
                else:
                    labels.append(true_season)
            annotators = labels + [""] * (3 - len(labels))

            rows.append(
                {
                    "image_id": f"IMG{image_counter:06d}",
                    "person_id": person_id,
                    "source": source,
                    "license": "synthetic-demo",
                    "image_quality": quality,
                    "lighting_type": lighting,
                    "camera_type": camera,
                    "makeup": makeup,
                    "skin_depth": skin_depth,
                    "skin_L": round(skin_L, 1),
                    "skin_a": round(skin_a, 1),
                    "skin_b": round(skin_b, 1),
                    "undertone": undertone,
                    "undertone_confidence": round(rng.uniform(0.4, 0.95), 3),
                    "olive_score": round(rng.uniform(0, 1), 3),
                    "hair_depth": rng.choice(["Light", "Medium", "Deep"]),
                    "hair_temperature": rng.choice(["Cool", "Neutral", "Warm"]),
                    "eye_depth": rng.choice(["Light", "Medium", "Deep"]),
                    "contrast_score": round(rng.uniform(0, 1), 3),
                    "chroma_score": round(rng.uniform(0, 1), 3),
                    "season_primary": true_season,
                    "season_secondary": _adjacent_season(rng, true_season),
                    "season_confidence": round(rng.uniform(0.5, 1.0), 3),
                    "annotator_1": annotators[0],
                    "annotator_2": annotators[1],
                    "annotator_3": annotators[2],
                    "annotation_agreement": round(
                        max(
                            labels.count(max(set(labels), key=labels.count)) / len(labels),
                            0,
                        ),
                        3,
                    ) if labels else 0.0,
                    "split": split_for_person,
                    "prediction": prediction,
                    "prediction_secondary": prediction_secondary,
                    "prediction_confidence": prediction_confidence,
                }
            )
    return rows


def UNDERTONE_WEIGHTS_FOR(skin_depth: str) -> Dict[str, float]:
    """Deep skin skews warm/olive in the demo pool (a real-world pattern)."""
    if skin_depth in ("Deep", "Very Deep", "Medium-Deep"):
        return {"Cool": 0.10, "Neutral-Cool": 0.12, "Neutral": 0.23, "Neutral-Warm": 0.25, "Warm": 0.20, "Olive": 0.10}
    return {"Cool": 0.20, "Neutral-Cool": 0.18, "Neutral": 0.24, "Neutral-Warm": 0.19, "Warm": 0.15, "Olive": 0.04}


_ADJACENT = {
    "Light Spring": ["Bright Spring", "Light Summer"],
    "Bright Spring": ["Light Spring", "Warm Spring"],
    "Warm Spring": ["Bright Spring", "Light Spring"],
    "Light Summer": ["Light Spring", "Soft Summer"],
    "Soft Summer": ["Light Summer", "Cool Summer"],
    "Cool Summer": ["Soft Summer", "Cool Winter"],
    "Soft Autumn": ["Warm Autumn", "Soft Summer"],
    "Warm Autumn": ["Soft Autumn", "Deep Autumn"],
    "Deep Autumn": ["Deep Winter", "Warm Autumn"],
    "Light Winter": ["Cool Winter", "Bright Spring"],
    "Cool Winter": ["Light Winter", "Deep Winter"],
    "Deep Winter": ["Deep Autumn", "Cool Winter"],
}


def _adjacent_season(rng: random.Random, season: str) -> str:
    options = _ADJACENT.get(season, [s for s in SEASON_ORDER if s != season])
    return rng.choice(options)


def _shift_warm(season: str) -> str:
    return {
        "Cool Summer": "Soft Autumn",
        "Soft Summer": "Soft Autumn",
        "Cool Winter": "Deep Autumn",
        "Light Summer": "Soft Autumn",
        "Deep Winter": "Deep Autumn",
        "Light Winter": "Soft Autumn",
    }.get(season, season)


def _shift_cool(season: str) -> str:
    return {
        "Soft Autumn": "Soft Summer",
        "Warm Autumn": "Soft Summer",
        "Deep Autumn": "Deep Winter",
        "Warm Spring": "Light Summer",
    }.get(season, season)


def demo_summary(rows: List[Dict]) -> str:
    """One-line sanity summary of the generated pool."""
    seasons = len({r['season_primary'] for r in rows})
    people = len({r['person_id'] for r in rows})
    return f"{len(rows)} images | {people} people | {seasons} seasons"
