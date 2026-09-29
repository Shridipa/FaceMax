"""One-shot validation of the extracted CelebAMask-HQ pairing (tool).

READ-ONLY research/preprocessing validation - no model training, no label
fabrication. For a deterministic sample of images this script:

  1. reads the real image + its per-class mask PNGs (cv2),
  2. aligns masks to images with a DOCUMENTED nearest-neighbour resize
     (official CelebAMask-HQ masks ship at 512x512 while images are
     1024x1024; the resize preserves binary labels and is recorded in the
     report - nothing is repaired silently),
  3. checks mask label vocabulary against the archive's own class names,
  4. merges masks into FaceMaxx region groups (skin+neck vs exclusions),
  5. verifies the merged skin and exclude masks are DISJOINT,
  6. pushes the masked skin pixels through the EXISTING extraction path
     (facemax_color.colour.region_lab_samples, with its 8-bit LAB a*/b*
     recentering) to prove the CelebA pairing works with the pipeline as-is.

Results are written to data/reports/celebamask_pairing_validation.json.
Fitzpatrick-style caution: CelebAMask-HQ masks are GEOMETRY only. Nothing
here derives undertone, skin depth, or seasonal labels from them.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BASE = ROOT / "data/raw/external/CelebAMask-HQ/CelebAMask-HQ"
IMAGES = BASE / "CelebA-HQ-img"
MASKS = BASE / "CelebAMask-HQ-mask-anno"
REPORT = ROOT / "data/reports/celebamask_pairing_validation.json"

SAMPLE = [0, 1, 100, 5000, 15000, 29999]  # deterministic, spans the dataset

# FaceMaxx region groups over the real CelebAMask-HQ mask class names
# (grounded in the archive's own filenames: <idx>_<class>.png).
SKIN_CLASSES = {"skin", "neck"}
EXCLUDE_CLASSES = {
    "hair", "u_lip", "l_lip", "mouth",
    "l_eye", "r_eye", "l_brow", "r_brow",
    "background", "cloth", "hat", "ear_r", "neck_l", "eye_g",
}
KNOWN_CLASSES = SKIN_CLASSES | EXCLUDE_CLASSES | {"nose", "l_ear", "r_ear"}


def list_mask_classes(idx: int):
    d = MASKS / str(idx // 2000)
    return sorted(p.stem.split("_", 1)[1] for p in d.glob(f"{idx:05d}_*.png"))


def main() -> int:
    import cv2  # local import: tool-only dependency path

    per_image = []
    for idx in SAMPLE:
        img_path = IMAGES / f"{idx}.jpg"
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        classes = list_mask_classes(idx)
        entry = {
            "image_id": idx,
            "image_read_ok": image is not None,
            "mask_classes_found": classes,
            "unknown_classes": [c for c in classes if c not in KNOWN_CLASSES],
        }
        if image is None:
            entry["ok"] = False
            per_image.append(entry)
            continue

        h, w = image.shape[:2]
        skin = np.zeros((h, w), dtype=np.uint8)
        exclude = np.zeros((h, w), dtype=np.uint8)
        masks_readable = 0
        native_sizes = set()
        resized_any = False
        for cls in classes:
            d = MASKS / str(idx // 2000)
            mask = cv2.imread(str(next(d.glob(f"{idx:05d}_{cls}.png"))), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            masks_readable += 1
            native_sizes.add((mask.shape[1], mask.shape[0]))
            if mask.shape[:2] != (h, w):
                # Documented transformation: official masks are 512x512
                # (half resolution); nearest-neighbour keeps binary labels.
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                resized_any = True
            if cls in SKIN_CLASSES:
                skin |= (mask > 127).astype(np.uint8)
            elif cls in EXCLUDE_CLASSES:
                exclude |= (mask > 127).astype(np.uint8)

        overlap = int(np.logical_and(skin, exclude).sum())
        entry.update(
            {
                "image_size": [w, h],
                "mask_native_sizes": sorted(native_sizes),
                "mask_resize_applied": resized_any,
                "masks_readable": masks_readable,
                "masks_expected": len(classes),
                "skin_px": int(skin.sum()),
                "exclude_px": int(exclude.sum()),
                "skin_exclude_overlap_px": overlap,
            }
        )

        # Existing extraction path on real CelebA pixels: skin mask must be a
        # full-size boolean array for region_lab_samples (it indexes by box).
        from facemax_color.colour import region_lab_samples

        regions = region_lab_samples(image, (0, 0, w, h), skin.astype(bool))
        entry["region_lab_samples_regions"] = sorted(regions)
        entry["lab_sample"] = {
            k: [round(v, 2) for v in vals] for k, vals in list(regions.items())[:2]
        }
        entry["ok"] = bool(
            masks_readable == len(classes)
            and overlap == 0
            and skin.sum() > 0
            and not entry["unknown_classes"]
            and regions
        )
        per_image.append(entry)

    report = {
        "dataset": "CelebAMask-HQ",
        "sample": SAMPLE,
        "transformation": {
            "mask_to_image": "nearest-neighbour resize (official masks are "
            "512x512; images 1024x1024); binary labels preserved; recorded "
            "per image as mask_resize_applied",
        },
        "per_image": per_image,
        "note": "masks are geometry only; no colour/undertone/season labels derived",
    }
    report["all_ok"] = all(e.get("ok") for e in per_image)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_ok": report["all_ok"], "per_image_ok": [e.get("ok") for e in per_image]}, indent=1))
    print(f"report: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
