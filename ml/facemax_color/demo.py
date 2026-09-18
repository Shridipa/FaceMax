"""End-to-end synthetic demo orchestration (demo-image command).

Proves the full wiring: synthetic images -> REAL CV extraction -> REAL models
-> bias evaluator. This is a pipeline smoke test, NOT a fairness benchmark:

    DATA STATUS: SYNTHETIC DEMO
    MODEL STATUS: DEMO/SYNTHETIC

The demo is deterministic for a fixed seed (fixed generator seeds, no
time-based input) and writes all artifacts under data/demo/.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .dataset import coerce_feature_row, init_dataset, join_with_features, write_extended_annotations
from .evaluate import render_comparison, run_evaluation
from .models import FEATURE_COLUMNS, train_bundle
from .synth import DEMO_FACE_BOX, generate_synth_faces

DEMO_HEADER = (
    "DATA STATUS: SYNTHETIC DEMO - not a real-world fairness benchmark.\n"
    "MODEL STATUS: DEMO/SYNTHETIC - trained on generated images.\n"
)


@dataclass
class DemoSummary:
    people: int
    images_generated: int
    faces_detected: int
    fallback_boxes: int
    extraction_failures: Dict[str, int] = field(default_factory=dict)
    predictions_ok: int = 0
    season_distribution: Dict[str, int] = field(default_factory=dict)
    sample_features: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            DEMO_HEADER.rstrip(),
            "",
            f"people generated:        {self.people}",
            f"images generated:        {self.images_generated}",
            f"faces detected:          {self.faces_detected} (detector)",
            f"fallback boxes used:     {self.fallback_boxes} (DEMO_FALLBACK_FACE_BOX)",
            f"extraction success rate: "
            f"{(self.faces_detected + self.fallback_boxes) / max(self.images_generated, 1) * 100:.1f}%",
            f"prediction success:      {self.predictions_ok}/{self.faces_detected + self.fallback_boxes}",
            "",
            "season distribution (demo predictions):",
        ]
        for season, count in sorted(self.season_distribution.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {season:<16} {count}")
        if self.extraction_failures:
            lines.append("")
            lines.append("extraction failures:")
            for err, count in sorted(self.extraction_failures.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {err}: {count}")
        if self.sample_features:
            lines.append("")
            lines.append("sample extracted features (first 3, straight from pixels):")
            for sample in self.sample_features:
                lines.append(
                    f"  {sample['image_id']}: L*={sample['L']:.1f} a*={sample['a']:.1f} "
                    f"b*={sample['b']:.1f} ITA={sample['ita']:.1f} "
                    f"undertone_proxy={sample['undertone_proxy']:.1f} "
                    f"contrast={sample['contrast_proxy']:.1f} "
                    f"-> {sample['season']} ({sample['confidence']:.2f})"
                )
        lines.append("")
        lines.append(
            "WARNING: demo statistics describe pipeline wiring, not model quality. "
            "Production fairness evaluation requires real, person-grouped, annotated data."
        )
        return "\n".join(lines)


def run_demo(
    root: Path,
    n_people: int = 40,
    images_per_person: int = 2,
    seed: int = 11,
    season_cv: bool = False,
) -> DemoSummary:
    """Generate -> extract (with logged fallback) -> train -> predict -> report.

    Isolation rule: EVERYTHING the demo produces (images, annotations, features,
    reports) lives under ``root/demo/``. The real annotation table
    (metadata/facemaxx_color_annotations.csv) and data/raw/ are never touched,
    so synthetic data can never contaminate real training data.
    """
    root = Path(root)
    init_dataset(root)
    demo_dir = root / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate deterministic synthetic images + annotation rows (demo-local).
    rows = generate_synth_faces(
        root,
        n_people=n_people,
        images_per_person=images_per_person,
        seed=seed,
        raw_dir=demo_dir / "raw",
    )
    write_extended_annotations(demo_dir / "annotations.csv", rows)

    # 2. Extract with the REAL pipeline; known-geometry fallback boxes are used
    #    only when the Haar detector refuses a synthetic face (logged loudly).
    from .colour import extract_batch

    image_ids = [str(r["image_id"]) for r in rows]
    fallback_boxes = {str(r["image_id"]): DEMO_FACE_BOX for r in rows}
    results = extract_batch(image_ids, demo_dir / "raw", fallback_boxes=fallback_boxes)

    detected = sum(1 for r in results.values() if r.ok and r.face_box_source == "detector")
    fallback = sum(1 for r in results.values() if r.ok and r.face_box_source == "demo_fallback")
    failures: Dict[str, int] = {}
    for r in results.values():
        if not r.ok:
            failures[r.error or "unknown"] = failures.get(r.error or "unknown", 0) + 1

    features = {
        image_id: result.as_feature_row() for image_id, result in results.items()
    }
    # Persist the demo feature table so the ML training pipeline can consume
    # the demo tree in isolation (`train --demo`): same canonical format as
    # the real processed/colour_features.csv, but demo-local.
    import csv as _csv

    feat_table = demo_dir / "colour_features.csv"
    with feat_table.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=["image_id"] + list(next(iter(features.values())).keys()))
        writer.writeheader()
        for image_id in image_ids:
            writer.writerow({"image_id": image_id, **features[image_id]})
    # Sidecar: which images used the demo fallback (transparency artifact).
    (demo_dir / "fallback_boxes.json").write_text(
        json.dumps(
            {
                "demo_face_box": list(DEMO_FACE_BOX),
                "used_by": sorted(
                    image_id for image_id, r in results.items()
                    if r.face_box_source == "demo_fallback"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 3. Train on ALL extracted rows (demo model; no held-out claim) ...
    joined_all = join_with_features(rows, features, require_features=True)
    joined_all = [r for r in joined_all if r.get("season_primary")]
    bundle = train_bundle(joined_all, season_cv=season_cv, seed=seed)

    # 4. ... predict the same rows with that demo bundle (extracted rows only).
    predictions_ok = 0
    season_distribution: Dict[str, int] = {}
    sample_features: List[Dict[str, Any]] = []
    for row in [r for r in joined_all if r.get("feat_ok")]:
        feats = coerce_feature_row({k: row.get(k) for k in FEATURE_COLUMNS})
        out = bundle.predict(feats)
        row["prediction"] = out["prediction"]
        row["prediction_secondary"] = out["prediction_secondary"]
        row["prediction_confidence"] = out["prediction_confidence"]
        predictions_ok += 1
        season_distribution[out["prediction"]] = season_distribution.get(out["prediction"], 0) + 1
        if len(sample_features) < 3:
            sample_features.append(
                {
                    "image_id": row["image_id"],
                    "L": feats["feat_skin_L"],
                    "a": feats["feat_skin_a"],
                    "b": feats["feat_skin_b"],
                    "ita": feats["feat_ita"],
                    "undertone_proxy": feats["feat_undertone_proxy"],
                    "contrast_proxy": feats["feat_contrast_proxy"],
                    "season": out["prediction"],
                    "confidence": out["prediction_confidence"],
                }
            )

    # 5. Feed through the EXISTING evaluator bridge -> bias report.
    # min_rows lowered: this demo is explicitly labeled SYNTHETIC, and small
    # subgroup counts are acceptable for a wiring smoke test.
    output = run_evaluation(joined_all, test_frac=0.3, season_cv=season_cv, seed=seed, min_rows=10)
    output.save(demo_dir)
    summary_path = demo_dir / "demo_summary.txt"
    summary = DemoSummary(
        people=n_people,
        images_generated=len(image_ids),
        faces_detected=detected,
        fallback_boxes=fallback,
        extraction_failures=failures,
        predictions_ok=predictions_ok,
        season_distribution=season_distribution,
        sample_features=sample_features,
        warnings=["synthetic data; not a fairness benchmark"],
    )
    summary_path.write_text(summary.render() + "\n", encoding="utf-8")

    # 6. Comparison table alongside the demo report.
    (demo_dir / "model_comparison.txt").write_text(render_comparison(bundle), encoding="utf-8")
    return summary
