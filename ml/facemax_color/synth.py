"""Synthetic face-image generator for end-to-end pipeline validation.

Draws Haar-detectable cartoon faces (skin ellipse + brows/eyes/nose/mouth
shading + hair band) whose LAB skin colour genuinely encodes the ground-truth
labels. Each person gets images under different lighting conditions, so the
demo exercises lighting robustness through the REAL CV pipeline, not through
faked predictions.

Ground truth: skin L* band -> skin_depth, b*-a* -> undertone, season via the
rule combiner. Labels are written into the standard annotation schema.
"""
from __future__ import annotations

import random
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from facemax_bias.vocab import SKIN_DEPTH_ORDER

from .models import (
    heuristic_chroma,
    heuristic_contrast,
    heuristic_depth,
    heuristic_undertone,
    season_from_axes,
)

LIGHTING_CONDITIONS = ("Natural", "Indoor warm", "Indoor cool", "Low light")
LIGHTING_FACTORS = {
    "Natural": (1.00, 1.00, 1.00),      # B,G,R multipliers
    "Indoor warm": (0.88, 0.98, 1.10),
    "Indoor cool": (1.12, 1.00, 0.88),
    "Low light": (0.55, 0.55, 0.55),
}


def _lab_to_bgr(L: float, a: float, b: float) -> Tuple[int, int, int]:
    lab = np.array([[[L * 255.0 / 100.0, a + 128.0, b + 128.0]]], dtype=np.uint8)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_face(
    skin_bgr: Tuple[int, int, int],
    hair_bgr: Tuple[int, int, int],
    eye_bgr: Tuple[int, int, int],
    size: int = 320,
) -> "np.ndarray":
    img = np.full((320, 256, 3), (105, 98, 92), np.uint8)  # studio-ish backdrop
    h, w = img.shape[:2]
    cx, cy = 128, 170  # face centre inside DEMO_FACE_BOX

    # Hair mass behind the face.
    cv2.ellipse(img, (cx, cy - 20), (int(w * 0.42), int(h * 0.42)), 0, 0, 360, hair_bgr, -1)
    # Face ellipse.
    cv2.ellipse(img, (cx, cy), (int(w * 0.34), int(h * 0.38)), 0, 0, 360, skin_bgr, -1)
    # Brows.
    cv2.ellipse(img, (cx - 40, cy - 55), (26, 7), 8, 0, 360, eye_bgr, -1)
    cv2.ellipse(img, (cx + 40, cy - 55), (26, 7), -8, 0, 360, eye_bgr, -1)
    # Eyes (white + iris) - gives Haar real features to lock onto.
    for dx in (-40, 40):
        cv2.ellipse(img, (cx + dx, cy - 32), (17, 9), 0, 0, 360, (235, 235, 235), -1)
        cv2.circle(img, (cx + dx, cy - 32), 6, eye_bgr, -1)
    # Nose shading + mouth.
    cv2.ellipse(img, (cx, cy + 8), (9, 14), 0, 0, 360, _shade(skin_bgr, 0.92), -1)
    cv2.ellipse(img, (cx, cy + 62), (30, 10), 0, 0, 360, _shade(skin_bgr, 0.8), -1)
    return img


def _shade(bgr: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(int(np.clip(c * factor, 0, 255)) for c in bgr)


def _apply_lighting(img: "np.ndarray", condition: str) -> "np.ndarray":
    b, g, r = LIGHTING_FACTORS.get(condition, (1, 1, 1))
    out = img.astype(np.float32)
    out[..., 0] = np.clip(out[..., 0] * b, 0, 255)
    out[..., 1] = np.clip(out[..., 1] * g, 0, 255)
    out[..., 2] = np.clip(out[..., 2] * r, 0, 255)
    # zlib.crc32 instead of hash(): str hash is salted per process, which would
    # break the demo's determinism guarantee.
    noise = np.random.default_rng(zlib.crc32(condition.encode()) & 0xFFFF).normal(0, 2.5, out.shape)
    return np.clip(out + noise, 0, 255).astype(np.uint8)


# Deterministic face box for demo fallback (the drawn face occupies this region).
DEMO_FACE_BOX = (41, 48, 174, 244)  # (x, y, w, h) in the 256x320 canvas


def generate_synth_faces(
    root: Path,
    n_people: int = 40,
    images_per_person: int = 2,
    seed: int = 11,
    raw_dir: Path = None,
) -> List[Dict]:
    """Create synthetic images + annotation rows; returns rows for the CSV.

    ``raw_dir`` defaults to ``root/demo/raw`` so synthetic images can NEVER
    land in the real ``data/raw/`` tree (demo isolation rule).
    """
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    root = Path(root)
    raw = Path(raw_dir) if raw_dir is not None else root / "demo" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    image_counter = 0
    for person in range(n_people):
        person_id = f"SP{person:04d}"
        # Sample true skin from the FULL depth range (deliberately balanced,
        # unlike the demo bias generator) plus warm/cool undertone.
        depth_index = int(nprng.integers(0, len(SKIN_DEPTH_ORDER)))
        warm = bool(nprng.random() < 0.5)
        skin_L = 82.0 - depth_index * 7.0 + float(nprng.normal(0, 1.5))
        skin_a = (10.0 if warm else 13.5) + float(nprng.normal(0, 0.8))
        skin_b = (24.0 if warm else 15.0) + float(nprng.normal(0, 1.2))

        skin_depth = heuristic_depth(skin_L)
        undertone = heuristic_undertone(skin_a, skin_b)
        # Drawn hair/eye luminance varies per person, so the drawn scenes
        # genuinely encode low/medium/high CONTRAST (extraction must recover it).
        hair_L = max(8.0, skin_L - float(nprng.uniform(15.0, 55.0)))
        eye_L = max(8.0, skin_L - float(nprng.uniform(30.0, 60.0)))
        # chroma/contrast ground truth from the SAME drawn-pixel parameters the
        # images encode (not invented): chroma from the skin a*/b* magnitude,
        # contrast from the known hair/eye vs skin L* gap.
        chroma = heuristic_chroma(skin_a, skin_b)
        contrast = heuristic_contrast(max(skin_L - hair_L, skin_L - eye_L))
        season = season_from_axes(undertone, skin_depth, chroma, contrast)

        skin_bgr = _lab_to_bgr(skin_L, skin_a, skin_b)
        hair_bgr = _lab_to_bgr(hair_L, 2.0, 3.0)
        eye_bgr = _lab_to_bgr(eye_L, 2.0, 3.0)

        conditions = rng.sample(LIGHTING_CONDITIONS, k=min(images_per_person, len(LIGHTING_CONDITIONS)))
        for i, condition in enumerate(conditions):
            image_counter += 1
            image_id = f"SYN{image_counter:05d}"
            img = _apply_lighting(_draw_face(skin_bgr, hair_bgr, eye_bgr), condition)
            cv2.imwrite(str(raw / f"{image_id}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append(
                {
                    "image_id": image_id,
                    "person_id": person_id,
                    "source": "Synthetic",
                    "license": "demo",
                    "image_quality": "Good",
                    "lighting_type": condition,
                    "camera_type": "DSLR",
                    "makeup": "No makeup",
                    "skin_depth": skin_depth,
                    "skin_L": round(skin_L, 1),
                    "skin_a": round(skin_a, 1),
                    "skin_b": round(skin_b, 1),
                    "undertone": undertone,
                    "undertone_confidence": 0.9,
                    "olive_score": 0.0,
                    "hair_depth": "Deep" if hair_L < 30 else "Medium",
                    "hair_temperature": "Neutral",
                    "eye_depth": "Deep",
                    "contrast_score": 0.6,
                    "chroma_score": 0.5,
                    "chroma_level": chroma,
                    "contrast_level": contrast,
                    "season_primary": season,
                    "season_secondary": "",
                    "season_confidence": 0.8,
                    "annotator_1": season,
                    "annotator_2": season,
                    "annotator_3": season,
                    "annotation_agreement": 1.0,
                    "split": "",
                }
            )
    return rows
