"""Tests for the acquisition + annotation workflow and the ML-training gate."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from facemax_color.annotation import (
    PENDING_FILENAME,
    coverage,
    import_images,
    merge_rounds,
    pending_pool,
    queue_round,
    read_round,
    record_round,
    write_capture_manifest,
)
from facemax_color.dataset import (
    EXTENDED_COLUMNS,
    HARMONY_COLOURS,
    init_dataset,
    read_annotations,
    write_extended_annotations,
)


def _capture_row(name: str, person: str, **kw) -> dict:
    row = {
        "file": name,
        "person_alias": person,
        "lighting_type": "Natural",
        "camera_type": "iPhone",
        "makeup": "No makeup",
        "notes": "",
    }
    row.update(kw)
    return row


class ImportTests(unittest.TestCase):
    def test_import_dedupes_content_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            staging = Path(tmp) / "staging"
            staging.mkdir(parents=True)
            (staging / "a.jpg").write_bytes(b"image-A")
            (staging / "b.jpg").write_bytes(b"image-B")
            (staging / "a_copy.jpg").write_bytes(b"image-A")  # identical content
            manifest = write_capture_manifest(Path(tmp) / "manifest.csv")
            import csv

            with manifest.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=["file", "person_alias", "lighting_type", "camera_type", "makeup", "notes"]).writerows(
                    [
                        _capture_row("a.jpg", "alice"),
                        _capture_row("b.jpg", "alice"),
                        _capture_row("a_copy.jpg", "alice"),
                    ]
                )
            result = import_images(root, manifest, staging)
            self.assertEqual(result["imported"], 2)  # a_copy collapsed into a
            self.assertEqual(result["skipped_duplicate_file"], 1)
            # Idempotent re-import.
            again = import_images(root, manifest, staging)
            self.assertEqual(again["imported"], 0)
            self.assertEqual(again["skipped_already_imported"], 2)
            # Rows carry capture metadata; ids are content-addressed.
            rows = read_annotations(root / "metadata" / "facemaxx_color_annotations.csv")
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["image_id"].startswith("IMG-") for r in rows))
            self.assertEqual(rows[0]["person_id"], "alice")
            self.assertEqual(rows[0]["lighting_type"], "Natural")

    def test_import_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            staging = Path(tmp) / "staging"
            staging.mkdir()
            manifest = write_capture_manifest(Path(tmp) / "manifest.csv")
            import csv

            with manifest.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=["file", "person_alias", "lighting_type", "camera_type", "makeup", "notes"]).writerow(
                    _capture_row("ghost.jpg", "alice")
                )
            with self.assertRaises(FileNotFoundError):
                import_images(root, manifest, staging)


class RoundTests(unittest.TestCase):
    def _seed(self, tmp: Path, n=3):
        root = Path(tmp)
        init_dataset(root)
        rows = []
        for i in range(n):
            row = {c: "" for c in EXTENDED_COLUMNS}
            row.update({"image_id": f"IMG-{i:04d}", "person_id": f"P{i}", "image_path": f"raw/IMG-{i:04d}.jpg"})
            rows.append(row)
        write_extended_annotations(root / "metadata" / "facemaxx_color_annotations.csv", rows)
        return root, rows

    def test_queue_and_pending_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, rows = self._seed(Path(tmp))
            pool = pending_pool(root)
            self.assertEqual(len(pool), 3)
            path = queue_round(root, "alice", [r["image_id"] for r in rows])
            entries = read_round(path)
            self.assertEqual(len(entries), 3)
            self.assertEqual(entries[0]["annotator"], "alice")
            self.assertEqual(entries[0]["season"], "")
            with self.assertRaises(ValueError):
                queue_round(root, "alice", ["IMG-9999"])

    def test_record_accepts_defers_and_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, rows = self._seed(Path(tmp))
            path = queue_round(root, "alice", [r["image_id"] for r in rows])
            entries = read_round(path)
            entries[0]["season"] = "Deep Autumn"
            entries[0]["undertone"] = "Warm"
            entries[0]["skin_depth"] = "Deep"
            entries[1]["season"] = "Wram Autunm"  # typos -> rejected with suggestions
            # entries[2] left empty -> deferred
            import csv

            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(entries[0].keys()))
                writer.writeheader()
                writer.writerows(entries)
            result = record_round(root, path)
            self.assertEqual(result["accepted"], 1)
            self.assertEqual(result["deferred"], 1)
            self.assertEqual(len(result["rejected"]), 1)
            self.assertIn("did you mean", result["rejected"][0])
            # Idempotent per (image, annotator).
            again = record_round(root, path)
            self.assertEqual(again["accepted"], 0)


class MergeTests(unittest.TestCase):
    def _seed(self, tmp: Path):
        root = Path(tmp)
        init_dataset(root)
        row = {c: "" for c in EXTENDED_COLUMNS}
        row.update({"image_id": "IMG-1", "person_id": "P1", "image_path": "raw/IMG-1.jpg"})
        write_extended_annotations(root / "metadata" / "facemaxx_color_annotations.csv", [row])
        return root

    def _round(self, tmp: Path, annotator: str, season: str, **extra) -> Path:
        path = Path(tmp) / f"round-{annotator}.csv"
        import csv

        entry = {
            "image_id": "IMG-1", "image_path": "raw/IMG-1.jpg", "annotator": annotator,
            "season": season, "season_secondary": "", "undertone": extra.get("undertone", "Warm"),
            "skin_depth": extra.get("skin_depth", "Deep"), "chroma": extra.get("chroma", "Muted"),
            "contrast": extra.get("contrast", "Medium"), "olive_score": "0.0",
            "confidence_1to5": "4", "best_colors": "olive;cream", "neutral_colors": "navy",
            "caution_colors": "lavender", "notes": "",
        }
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(entry.keys()))
            writer.writeheader()
            writer.writerow(entry)
        return path

    def _record(self, root: Path, tmp: Path, annotator: str, season: str, **extra):
        record_round(root, self._round(tmp, annotator, season, **extra))

    def test_majority_merge_with_secondary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._seed(Path(tmp))
            self._record(root, Path(tmp), "a", "Deep Autumn")
            self._record(root, Path(tmp), "b", "Deep Autumn")
            self._record(root, Path(tmp), "c", "Deep Winter")
            result = merge_rounds(root)
            self.assertEqual(result["merged"], 1)
            self.assertEqual(result["split_votes"], [])
            row = read_annotations(root / "metadata" / "facemaxx_color_annotations.csv")[0]
            self.assertEqual(row["season_primary"], "Deep Autumn")
            self.assertEqual(row["season_secondary"], "Deep Winter")
            self.assertEqual(row["annotation_agreement"], "0.6667")
            self.assertEqual(row["annotator_1"], "Deep Autumn")
            self.assertEqual(row["annotator_3"], "Deep Winter")
            # Axis merged only on full agreement: undertone unanimous.
            self.assertEqual(row["undertone"], "Warm")

    def test_axis_disagreement_left_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._seed(Path(tmp))
            self._record(root, Path(tmp), "a", "Deep Autumn", chroma="Muted")
            self._record(root, Path(tmp), "b", "Deep Autumn", chroma="Bright")
            merge_rounds(root)
            row = read_annotations(root / "metadata" / "facemaxx_color_annotations.csv")[0]
            self.assertEqual(row["season_primary"], "Deep Autumn")
            self.assertFalse(row["chroma_level"])  # never voted for them

    def test_split_vote_not_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._seed(Path(tmp))
            self._record(root, Path(tmp), "a", "Deep Autumn")
            self._record(root, Path(tmp), "b", "Deep Winter")
            self._record(root, Path(tmp), "c", "Soft Summer")
            result = merge_rounds(root)
            self.assertEqual(result["merged"], 0)
            self.assertEqual(len(result["split_votes"]), 1)
            row = read_annotations(root / "metadata" / "facemaxx_color_annotations.csv")[0]
            self.assertFalse(row["season_primary"])  # left unlabelled, not guessed

    def test_single_annotator_not_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._seed(Path(tmp))
            self._record(root, Path(tmp), "a", "Deep Autumn")
            result = merge_rounds(root)
            self.assertEqual(result["merged"], 0)


class GateTests(unittest.TestCase):
    def _full_dataset(self, tmp: Path, people=25, per_person=2):
        """Build an audit-clean, fully annotated dataset."""
        root = Path(tmp) / "data"
        init_dataset(root)
        seasons = ["Deep Autumn", "Deep Winter", "Warm Spring", "Soft Summer", "Cool Winter"]
        rows = []
        counter = 0
        for p in range(people):
            for j in range(per_person):
                counter += 1
                row = {c: "" for c in EXTENDED_COLUMNS}
                season = seasons[p % len(seasons)]
                row.update(
                    {
                        "image_id": f"GD-{counter:04d}",
                        "person_id": f"GP-{p:03d}",
                        "source": "Phone selfie",
                        "license": "project-owned",
                        "image_quality": "Good",
                        "lighting_type": "Natural",
                        "camera_type": "iPhone",
                        "makeup": "No makeup",
                        "skin_depth": "Deep" if p % 2 else "Light-Medium",
                        "undertone": "Warm" if p % 3 else "Cool",
                        "chroma_level": "Medium",
                        "contrast_level": "Medium",
                        "season_primary": season,
                        "annotator_1": season,
                        "annotator_2": season,
                        "annotator_3": season,
                        "annotation_agreement": "1.0",
                        "season_confidence": "0.9",
                        "image_path": f"raw/GD-{counter:04d}.jpg",
                    }
                )
                rows.append(row)
        write_extended_annotations(root / "metadata" / "facemaxx_color_annotations.csv", rows)
        for i, row in enumerate(rows):
            (root / "raw" / f"{row['image_id']}.jpg").write_bytes(f"unique-{i}".encode())
        return root, rows

    def test_gate_closed_when_empty(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.__main__ import main

            rc = main(["gate", "--root", str(Path(tmp) / "data"), "--out", str(Path(tmp) / "reports")])
            self.assertEqual(rc, 1)  # closed
            gate = json.loads((Path(tmp) / "reports" / "ml_gate.json").read_text())
            self.assertEqual(gate["status"], "CLOSED")
            self.assertTrue(gate["reasons"])

    def test_gate_open_on_clean_annotated_dataset(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.__main__ import main
            from facemax_color.audit import lock_holdout

            root, rows = self._full_dataset(Path(tmp))
            lock_holdout(root, rows, test_frac=0.2, seed=3)
            rc = main(["gate", "--root", str(root), "--out", str(Path(tmp) / "reports")])
            gate = json.loads((Path(tmp) / "reports" / "ml_gate.json").read_text())
            self.assertEqual(gate["status"], "OPEN", msg=gate["reasons"])
            self.assertEqual(rc, 0)

    def test_coverage_counts_boundary_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._full_dataset(Path(tmp))
            info = coverage(root)
            self.assertEqual(info["images_imported"], 50)
            self.assertEqual(info["images_annotated"], 50)
            self.assertEqual(info["people"], 25)
            self.assertEqual(info["people_with_multiple_images"], 25)
            self.assertGreater(info["boundary_candidates_DA_DW"], 0)


if __name__ == "__main__":
    unittest.main()
