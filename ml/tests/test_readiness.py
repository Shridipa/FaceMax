"""Tests for the real-data readiness audit (the pre-ML-training checkpoint)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from facemax_color.readiness import (
    TRAIN_MIN_SEASON,
    season_distance,
    readiness,
)
from facemax_color.dataset import (
    EXTENDED_COLUMNS,
    init_dataset,
    read_annotations,
    write_extended_annotations,
)

SEASONS = [
    "Light Spring", "Bright Spring", "Warm Spring",
    "Light Summer", "Soft Summer", "Cool Summer",
    "Soft Autumn", "Warm Autumn", "Deep Autumn",
    "Light Winter", "Cool Winter", "Deep Winter",
]
DEPTHS = [
    "Very Fair", "Fair", "Light", "Light-Medium",
    "Medium", "Medium-Deep", "Deep", "Very Deep",
]
UNDERTONES = ["Cool", "Neutral-Cool", "Neutral", "Neutral-Warm", "Warm", "Olive"]
CHROMAS = ["Muted", "Medium", "Bright"]
CONTRASTS = ["Low", "Medium", "High"]


def _row(image_id: str, person_id: str, season: str, **kw) -> dict:
    row = {c: "" for c in EXTENDED_COLUMNS}
    row.update(
        {
            "image_id": image_id,
            "person_id": person_id,
            "source": "Phone selfie",
            "license": "project-owned",
            "image_quality": kw.get("image_quality", "Good"),
            "lighting_type": kw.get("lighting_type", "Natural"),
            "camera_type": kw.get("camera_type", "iPhone"),
            "makeup": kw.get("makeup", "No makeup"),
            "skin_depth": kw.get("skin_depth", "Medium"),
            "undertone": kw.get("undertone", "Neutral"),
            "chroma_level": kw.get("chroma_level", "Medium"),
            "contrast_level": kw.get("contrast_level", "Medium"),
            "season_primary": season,
            "season_secondary": kw.get("season_secondary", ""),
            "season_confidence": "0.9",
            "annotator_1": kw.get("annotator_1", season),
            "annotator_2": kw.get("annotator_2", season),
            "annotator_3": kw.get("annotator_3", season),
            "annotation_agreement": kw.get("annotation_agreement", "1.0"),
            "image_path": f"raw/{image_id}.jpg",
        }
    )
    return row


def _write(root: Path, rows: list) -> None:
    init_dataset(root)
    write_extended_annotations(
        root / "metadata" / "facemaxx_color_annotations.csv", rows
    )
    for i, row in enumerate(rows):
        (root / "raw" / f"{row['image_id']}.jpg").write_bytes(f"unique-bytes-{i}".encode())


def _full_dataset(tmp: Path, people: int = 60, per_person: int = 2):
    """Audit-clean dataset covering all 12 seasons / 8 depths / every axis label.

    Axis labels and season are assigned PER PERSON (a person's photos share
    their colouring, varying only capture conditions), mirroring real data.
    """
    root = Path(tmp) / "data"
    rows = []
    counter = 0
    for p in range(people):
        season = SEASONS[p % len(SEASONS)]
        depth = DEPTHS[p % len(DEPTHS)]
        undertone = UNDERTONES[p % len(UNDERTONES)]
        chroma = CHROMAS[p % len(CHROMAS)]
        contrast = CONTRASTS[(p + 1) % len(CONTRASTS)]
        for _ in range(per_person):
            counter += 1
            rows.append(
                _row(
                    f"RD-{counter:04d}",
                    f"RP-{p:03d}",
                    season,
                    skin_depth=depth,
                    undertone=undertone,
                    chroma_level=chroma,
                    contrast_level=contrast,
                )
            )
    _write(root, rows)
    return root, rows


class SeasonNeighbourTests(unittest.TestCase):
    def test_wheel_neighbours_are_distance_one(self):
        self.assertEqual(season_distance("Deep Autumn", "Deep Winter"), 1)
        self.assertEqual(season_distance("Warm Spring", "Light Spring"), 1)

    def test_unrelated_seasons_have_no_wheel_edge(self):
        self.assertIsNone(season_distance("Deep Autumn", "Cool Summer"))
        self.assertIsNone(season_distance("Light Spring", "Deep Winter"))


class EmptyDatasetTests(unittest.TestCase):
    def test_empty_tree_is_not_ready_with_gate_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            report = readiness(root)
            self.assertEqual(report.status, "NOT READY")
            self.assertEqual(report.annotated_rows, 0)
            self.assertEqual(len(report.missing_seasons), 12)
            joined = " ".join(report.blockers)
            self.assertIn("annotated images", joined)
            self.assertIn("people", joined)
            self.assertIn("holdout", joined)


class VerdictTests(unittest.TestCase):
    def test_full_clean_dataset_is_train_ready(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.audit import lock_holdout

            root, rows = _full_dataset(Path(tmp))
            lock_holdout(root, rows, test_frac=0.2, seed=3)
            report = readiness(root)
            self.assertEqual(report.status, "TRAIN-READY", msg=report.blockers + report.train_gaps)
            self.assertEqual(len(report.missing_seasons), 0)
            self.assertEqual(len(report.underrepresented_seasons), 0)
            self.assertEqual(len(report.underrepresented_depths), 0)
            self.assertFalse(report.underrepresented_axis_labels["undertone"])
            self.assertEqual(report.conflicting_disagreements, 0)
            self.assertEqual(report.unstable_conditions, [])
            self.assertEqual(report.inconsistent_people, [])
            self.assertTrue(report.holdout_untouched)
            self.assertEqual(report.multi_annotator_rows, report.annotated_rows)
            self.assertEqual(report.mean_agreement, 1.0)

    def test_missing_season_blocks_train_readiness_but_not_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.audit import lock_holdout

            root, rows = _full_dataset(Path(tmp))
            # Remove every Cool Summer person BEFORE locking (data settles,
            # then the holdout freezes): still 40+ images / 20+ people, but
            # one season has zero examples -> SMOKE-READY, never TRAIN-READY.
            remaining = [r for r in rows if r["season_primary"] != "Cool Summer"]
            write_extended_annotations(
                root / "metadata" / "facemaxx_color_annotations.csv", remaining
            )
            for row in rows:
                if row["season_primary"] == "Cool Summer":
                    (root / "raw" / f"{row['image_id']}.jpg").unlink()
            lock_holdout(root, remaining, test_frac=0.2, seed=3)
            report = readiness(root)
            self.assertEqual(report.status, "SMOKE-READY", msg=report.blockers)
            self.assertIn("Cool Summer", report.missing_seasons)
            self.assertTrue(
                any("zero examples" in gap for gap in report.train_gaps)
            )

    def test_below_minimum_season_is_flagged_underrepresented(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.audit import lock_holdout

            root, rows = _full_dataset(Path(tmp))
            # Drop two Warm Spring rows to 8 examples (< TRAIN_MIN_SEASON),
            # THEN lock so the holdout reflects the reduced dataset.
            victims = [r for r in rows if r["season_primary"] == "Warm Spring"][:2]
            remaining = [r for r in rows if r not in victims]
            write_extended_annotations(
                root / "metadata" / "facemaxx_color_annotations.csv", remaining
            )
            for victim in victims:
                (root / "raw" / f"{victim['image_id']}.jpg").unlink()
            lock_holdout(root, remaining, test_frac=0.2, seed=3)
            report = readiness(root)
            self.assertEqual(report.status, "SMOKE-READY", msg=report.blockers)
            self.assertIn("Warm Spring", report.underrepresented_seasons)
            self.assertLess(report.season_counts["Warm Spring"], TRAIN_MIN_SEASON)


class DisagreementTests(unittest.TestCase):
    def test_boundary_vs_conflict_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = [
                # boundary: DA <-> DW are wheel neighbours
                _row(
                    "RD-B1", "P1", "Deep Autumn",
                    annotator_1="Deep Autumn", annotator_2="Deep Autumn",
                    annotator_3="Deep Winter", annotation_agreement="0.6667",
                    season_secondary="Deep Winter",
                ),
                # conflict: DA vs Cool Summer share no wheel edge
                _row(
                    "RD-B2", "P2", "Cool Summer",
                    annotator_1="Cool Summer", annotator_2="Cool Summer",
                    annotator_3="Deep Autumn", annotation_agreement="0.6667",
                    season_secondary="Deep Autumn",
                ),
                # unanimous row: no dissent
                _row("RD-B3", "P3", "Warm Spring"),
            ]
            _write(root, rows)
            report = readiness(root)
            self.assertEqual(report.boundary_disagreements, 1)
            self.assertEqual(report.conflicting_disagreements, 1)
            self.assertEqual(report.boundary_pairs.get("Deep Autumn <-> Deep Winter"), 1)
            self.assertEqual(report.boundary_pairs.get("Cool Summer <-> Deep Autumn"), 1)
            self.assertEqual(report.dissent_rows, 2)
            self.assertAlmostEqual(report.boundary_rate, 0.5)

    def test_condition_instability_flagged_only_with_enough_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = []
            # 6 dissenting rows in Low light (>= STABILITY_MIN_ROWS)...
            for i in range(6):
                rows.append(
                    _row(
                        f"RD-L{i}", f"PL{i}", "Deep Autumn",
                        lighting_type="Low light",
                        annotator_1="Deep Autumn", annotator_2="Deep Winter",
                        annotator_3="Deep Autumn", annotation_agreement="0.6667",
                        season_secondary="Deep Winter",
                    )
                )
            # ...and 6 dissenting rows in Natural (control, also dissenting)
            for i in range(6):
                rows.append(
                    _row(
                        f"RD-N{i}", f"PN{i}", "Soft Summer",
                        annotator_1="Soft Summer", annotator_2="Cool Summer",
                        annotator_3="Soft Summer", annotation_agreement="0.6667",
                        season_secondary="Cool Summer",
                    )
                )
            _write(root, rows)
            report = readiness(root)
            # Both conditions dissent at 100%, both have >= 5 rows -> both flagged.
            self.assertIn("lighting_type=Low light", report.unstable_conditions)
            self.assertIn("lighting_type=Natural", report.unstable_conditions)
            self.assertAlmostEqual(
                report.condition_dissent["lighting_type"]["Low light"], 1.0
            )

    def test_small_condition_sample_not_judged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = [
                _row(
                    "RD-S1", "PS1", "Deep Autumn",
                    lighting_type="Low light",
                    annotator_1="Deep Autumn", annotator_2="Deep Winter",
                    annotator_3="Deep Autumn", annotation_agreement="0.6667",
                ),
            ]
            _write(root, rows)
            report = readiness(root)
            self.assertEqual(report.unstable_conditions, [])
            self.assertIn("Low light", report.condition_dissent.get("lighting_type", {}))


class PersonConsistencyTests(unittest.TestCase):
    def test_conflicting_seasons_within_one_person_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = [
                _row("RD-P1", "SAME", "Deep Autumn"),
                _row("RD-P2", "SAME", "Cool Summer"),
                _row("RD-P3", "OTHER", "Warm Spring"),
                _row("RD-P4", "OTHER", "Warm Spring"),  # consistent person
            ]
            _write(root, rows)
            report = readiness(root)
            self.assertEqual(report.multi_image_people, 2)
            self.assertEqual(len(report.inconsistent_people), 1)
            self.assertIn("SAME", report.inconsistent_people[0])

    def test_missing_person_id_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = [_row("RD-M1", "", "Warm Spring")]
            _write(root, rows)
            report = readiness(root)
            self.assertEqual(report.missing_person_ids, 1)


class HoldoutTests(unittest.TestCase):
    def test_stale_holdout_detected_when_test_row_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.audit import lock_holdout

            root, rows = _full_dataset(Path(tmp))
            lock_holdout(root, rows, test_frac=0.2, seed=3)
            # Delete one test-split row from the annotation table (manifest
            # untouched) - the holdout is no longer an honest snapshot.
            test_ids = (
                (root / "splits" / "test.txt").read_text(encoding="utf-8").split()
            )
            victim_id = test_ids[0]
            remaining = [
                r for r in read_annotations(root / "metadata" / "facemaxx_color_annotations.csv")
                if r["image_id"] != victim_id
            ]
            write_extended_annotations(
                root / "metadata" / "facemaxx_color_annotations.csv", remaining
            )
            (root / "raw" / f"{victim_id}.jpg").unlink()
            report = readiness(root)
            self.assertFalse(report.holdout_untouched)
            self.assertEqual(report.status, "NOT READY")
            self.assertTrue(
                any("deleted" in p for p in report.holdout_detail.get("problems", []))
            )

    def test_holdout_test_row_without_truth_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.audit import lock_holdout

            root, rows = _full_dataset(Path(tmp))
            lock_holdout(root, rows, test_frac=0.2, seed=3)
            # Empty the season of one test-split row: the holdout image exists
            # but can never be scored - readiness must refuse.
            test_ids = (
                (root / "splits" / "test.txt").read_text(encoding="utf-8").split()
            )
            stored = read_annotations(root / "metadata" / "facemaxx_color_annotations.csv")
            for row in stored:
                if row["image_id"] == test_ids[0]:
                    row["season_primary"] = ""
            write_extended_annotations(
                root / "metadata" / "facemaxx_color_annotations.csv", stored
            )
            report = readiness(root)
            self.assertEqual(report.holdout_test_without_truth, 1)
            self.assertFalse(report.holdout_untouched)
            self.assertEqual(report.status, "NOT READY")
            self.assertTrue(
                any("holdout" in b and "truth" in b for b in report.blockers)
            )


class CliTests(unittest.TestCase):
    def test_cli_ready_exit_codes_and_reports(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.__main__ import main

            root = Path(tmp) / "data"
            out = Path(tmp) / "reports"
            rc = main(["ready", "--root", str(root), "--out", str(out)])
            self.assertEqual(rc, 1)  # NOT READY
            payload = json.loads((out / "readiness_report.json").read_text())
            self.assertEqual(payload["status"], "NOT READY")
            self.assertTrue((out / "readiness_report.txt").exists())


if __name__ == "__main__":
    unittest.main()
