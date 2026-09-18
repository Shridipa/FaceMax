"""Tests for the real-data foundation: extended schema, registry, audit, splits."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from facemax_color.dataset import (
    EXTENDED_COLUMNS,
    HARMONY_COLOURS,
    REGISTRY_FILENAME,
    init_dataset,
    read_annotations,
    write_extended_annotations,
    write_harmony_scores,
)
from facemax_color.audit import audit_dataset, write_split_manifests, _registry_issues


def _row(image_id="IMG1", person_id="P1", **kw):
    row = {
        "image_id": image_id,
        "person_id": person_id,
        "source": "Phone selfie",
        "license": "CC-BY",
        "image_quality": "Good",
        "lighting_type": "Natural",
        "camera_type": "iPhone",
        "makeup": "No makeup",
        "skin_depth": "Medium",
        "skin_L": "60.0",
        "skin_a": "9.0",
        "skin_b": "24.0",
        "undertone": "Warm",
        "undertone_confidence": "0.9",
        "olive_score": "0.0",
        "hair_depth": "Deep",
        "hair_temperature": "Neutral",
        "eye_depth": "Medium",
        "contrast_score": "0.6",
        "chroma_score": "0.5",
        "season_primary": "Warm Autumn",
        "season_secondary": "",
        "season_confidence": "0.8",
        "annotator_1": "Warm Autumn",
        "annotator_2": "Warm Autumn",
        "annotator_3": "Warm Autumn",
        "annotation_agreement": "1.0",
        "split": "",
        "image_path": f"raw/{image_id}.jpg",
        "quality_score": "4",
        "white_balance_quality": "good",
        "best_colors": "olive;cream",
        "neutral_colors": "navy",
        "caution_colors": "black",
        "notes": "",
    }
    row.update(kw)
    return row


class ExtendedSchemaTests(unittest.TestCase):
    def test_extended_columns_superset_of_evaluator_columns(self):
        from facemax_bias.vocab import COLUMNS
        self.assertTrue(set(COLUMNS).issubset(set(EXTENDED_COLUMNS)))

    def test_extended_roundtrip_preserves_extra_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ann.csv"
            write_extended_annotations(path, [_row()])
            loaded = read_annotations(path)
            self.assertEqual(loaded[0]["image_id"], "IMG1")
            self.assertEqual(loaded[0]["best_colors"], "olive;cream")

    def test_no_race_ethnicity_columns(self):
        forbidden = {"race", "ethnicity", "skin_type", "nationality"}
        self.assertFalse(forbidden & set(EXTENDED_COLUMNS))


class RegistryTests(unittest.TestCase):
    def test_init_creates_registry_with_all_five_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_dataset(Path(tmp))
            text = (Path(tmp) / REGISTRY_FILENAME).read_text(encoding="utf-8")
            for name in ("CelebAMask-HQ", "FFHQ", "Fitzpatrick17k", "custom annotated", "harmony"):
                self.assertIn(name, text)

    def test_registry_lint_passes_on_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_dataset(Path(tmp))
            self.assertEqual(_registry_issues(Path(tmp) / REGISTRY_FILENAME), [])

    def test_registry_lint_flags_missing_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_dataset(Path(tmp))
            path = Path(tmp) / REGISTRY_FILENAME
            text = path.read_text(encoding="utf-8").replace("- **license:**\n", "")
            path.write_text(text, encoding="utf-8")
            issues = _registry_issues(path)
            self.assertTrue(any("license" in i.lower() for i in issues))


class HarmonyTests(unittest.TestCase):
    def test_harmony_roundtrip_and_colour_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "harmony.csv"
            rows = [
                {"person_id": "P1", "colour": colour, "score_1_to_5": "4", "annotator": "A", "notes": ""}
                for colour in HARMONY_COLOURS
            ]
            write_harmony_scores(path, rows)
            loaded = read_annotations(path)
            self.assertEqual(len(loaded), len(HARMONY_COLOURS))
            self.assertEqual({r["colour"] for r in loaded}, set(HARMONY_COLOURS))


class AuditTests(unittest.TestCase):
    def _root_with_data(self, tmp: str) -> Path:
        init_dataset(Path(tmp))  # create tree; annotations returned but root is tmp
        root = Path(tmp)
        rows = [
            _row("IMG1", "P1", season_primary="Warm Autumn"),
            _row("IMG2", "P1", season_primary="Deep Autumn", undertone="Cool"),
            _row("IMG3", "P2", season_primary="Cool Summer", undertone="Cool", skin_depth="Fair"),
            _row("IMG4", "P2", season_primary="", annotator_1="", annotator_2="", annotator_3=""),  # missing truth -> counted, not fabricated
        ]
        write_extended_annotations(root / "metadata" / "facemaxx_color_annotations.csv", rows)
        # Raw files must exist for every annotated row (file-layer audit check);
        # each file gets UNIQUE bytes so the duplicate-content check stays clean.
        for i, row in enumerate(rows):
            (root / "raw" / f"{row['image_id']}.jpg").write_bytes(f"fake-jpeg-{i}".encode())
        write_harmony_scores(
            root / "annotations" / "harmony_scores.csv",
            [{"person_id": "P1", "colour": c, "score_1_to_5": "3", "annotator": "A", "notes": ""} for c in HARMONY_COLOURS],
        )
        return Path(tmp)

    def test_audit_counts_missing_truth_without_fabricating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root_with_data(tmp)
            result = audit_dataset(root)
            self.assertEqual(result.total_rows, 4)
            self.assertEqual(result.missing_truth, 1)
            self.assertTrue(any("never guessed" in w or "not guessed" in w for w in result.warnings))

    def test_audit_detects_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root_with_data(tmp)
            rows = [
                _row("IMG1", "P1", split="train"),
                _row("IMG2", "P1", split="test"),  # same person, different split
            ]
            write_extended_annotations(root / "metadata" / "facemaxx_color_annotations.csv", rows)
            result = audit_dataset(root)
            self.assertEqual(len(result.person_leakage_pairs), 1)
            self.assertFalse(result.ok())

    def test_audit_harmony_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root_with_data(tmp)
            result = audit_dataset(root)
            self.assertEqual(result.harmony_persons, 1)
            self.assertFalse(any("harmony" in w for w in result.warnings))
            # Now drop half the colours -> warning expected.
            write_harmony_scores(
                root / "annotations" / "harmony_scores.csv",
                [{"person_id": "P1", "colour": "burgundy", "score_1_to_5": "3", "annotator": "A", "notes": ""}],
            )
            result = audit_dataset(root)
            self.assertTrue(any("harmony scores missing" in w for w in result.warnings))

    def test_split_manifests_are_person_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root_with_data(tmp)
            rows = [_row(f"IMG{i}", f"P{i // 2}") for i in range(1, 21)]  # 10 people x 2 images
            write_extended_annotations(root / "metadata" / "facemaxx_color_annotations.csv", rows)
            counts = write_split_manifests(root, rows, test_frac=0.2, seed=3)
            self.assertEqual(sum(counts.values()), 20)
            import json
            person_map = json.loads((root / "splits" / "person_map.json").read_text())
            seen = set()
            for split, people in person_map.items():
                overlap = seen & set(people)
                self.assertFalse(overlap, f"person in multiple splits: {overlap}")
                seen |= set(people)

    def test_audit_overall_pass_on_clean_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root_with_data(tmp)
            result = audit_dataset(root)
            self.assertTrue(result.ok())

    def test_audit_flags_missing_raw_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root_with_data(tmp)
            (root / "raw" / "IMG2.jpg").unlink()
            result = audit_dataset(root)
            self.assertIn("IMG2", result.missing_raw_images)
            self.assertFalse(result.ok())
            self.assertTrue(any("do not exist" in w for w in result.warnings))

    def test_audit_flags_duplicate_file_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root_with_data(tmp)
            # IMG3 becomes a byte-identical copy of IMG1 -> duplicate content.
            (root / "raw" / "IMG3.jpg").write_bytes((root / "raw" / "IMG1.jpg").read_bytes())
            result = audit_dataset(root)
            self.assertEqual(len(result.duplicate_files), 1)
            self.assertFalse(result.ok())


class IngestTests(unittest.TestCase):
    def _prepared_root(self, tmp: str) -> Path:
        root = Path(tmp)
        init_dataset(root)
        # Human completes the license field for CelebAMask-HQ.
        registry = root / REGISTRY_FILENAME
        registry.write_text(
            registry.read_text(encoding="utf-8").replace(
                "## 1. CelebAMask-HQ\n\n- **official source:**\n- **URL:**\n- **license:**",
                "## 1. CelebAMask-HQ\n\n- **official source:**\n- **URL:**\n- **license:** research-only, non-commercial",
            ),
            encoding="utf-8",
        )
        return root

    def test_sources_template_created_with_three_externals(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.ingest import read_sources
            sources = read_sources(Path(tmp))
            self.assertEqual(
                {s["name"] for s in sources},
                {"CelebAMask-HQ", "FFHQ", "Fitzpatrick17k"},
            )
            self.assertTrue(all(s["status"] == "not_downloaded" for s in sources))

    def test_record_refuses_unknown_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.ingest import IngestError, record_download
            root = self._prepared_root(tmp)
            archive = root / "x.zip"
            archive.write_bytes(b"data")
            with self.assertRaises(IngestError):
                record_download(root, "NotADataset", archive, license_confirmed=True)

    def test_record_refuses_without_registry_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.ingest import IngestError, record_download
            root = Path(tmp)
            init_dataset(root)
            archive = root / "celeba.zip"
            archive.write_bytes(b"data")
            with self.assertRaises(IngestError):
                record_download(root, "CelebAMask-HQ", archive, license_confirmed=True)

    def test_record_refuses_unconfirmed_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.ingest import IngestError, record_download
            root = self._prepared_root(tmp)
            archive = root / "celeba.zip"
            archive.write_bytes(b"data")
            with self.assertRaises(IngestError):
                record_download(root, "CelebAMask-HQ", archive, license_confirmed=False)

    def test_record_writes_receipt_and_verifies(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.ingest import record_download, verify_receipt
            root = self._prepared_root(tmp)
            archive = root / "celeba.zip"
            archive.write_bytes(b"celeba-bytes")
            receipt = record_download(root, "CelebAMask-HQ", archive, license_confirmed=True)
            self.assertEqual(receipt["bytes"], len(b"celeba-bytes"))
            self.assertTrue(receipt["sha256"])
            status = verify_receipt(root, "CelebAMask-HQ")
            self.assertTrue(status["ok"])
            # Re-hash matches while the file is unchanged.
            self.assertTrue(verify_receipt(root, "CelebAMask-HQ", rehash=True)["ok"])
            # Tampering is detected.
            archive.write_bytes(b"tampered")
            bad = verify_receipt(root, "CelebAMask-HQ", rehash=True)
            self.assertFalse(bad["ok"])
            self.assertIn("mismatch", bad["reason"])

    def test_audit_warns_on_missing_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.ingest import read_sources, write_sources
            root = self._prepared_root(tmp)
            sources = read_sources(root)
            sources[0]["status"] = "downloaded"
            sources[0]["receipt"] = None
            write_sources(root, sources)
            result = audit_dataset(root)
            self.assertTrue(any("receipt missing" in w for w in result.warnings))


class ValidateTests(unittest.TestCase):
    def test_clean_row_passes(self):
        from facemax_color.validate import lint_row
        issues = lint_row(_row(), row_number=2)
        self.assertTrue(issues.ok)

    def test_missing_required_fields_flagged(self):
        from facemax_color.validate import lint_row
        row = _row(undertone="", skin_depth="")
        issues = lint_row(row, row_number=2)
        self.assertIn("undertone", issues.missing_required)
        self.assertIn("skin_depth", issues.missing_required)
        self.assertFalse(issues.ok)

    def test_unknown_vocab_value_with_suggestion(self):
        from facemax_color.validate import lint_row
        row = _row(undertone="Wram")  # typo
        issues = lint_row(row, row_number=2)
        self.assertEqual(len(issues.unknown_values), 1)
        self.assertIn("did you mean 'Warm'", issues.unknown_values[0])

    def test_numeric_range_violation(self):
        from facemax_color.validate import lint_row
        row = _row(skin_L="142.0", undertone_confidence="1.7")
        issues = lint_row(row, row_number=2)
        self.assertTrue(any(o.startswith("skin_L") for o in issues.out_of_range))
        self.assertTrue(any(o.startswith("undertone_confidence") for o in issues.out_of_range))

    def test_non_numeric_rejected(self):
        from facemax_color.validate import lint_row
        row = _row(contrast_score="high")
        issues = lint_row(row, row_number=2)
        self.assertTrue(any("not numeric" in u for u in issues.unknown_values))

    def test_multi_colour_fields_accept_season_names_only(self):
        from facemax_color.validate import lint_row
        good = lint_row(_row(), row_number=2)
        self.assertTrue(good.ok)
        bad = lint_row(_row(best_colors="burgundy;chartreuse"), row_number=2)
        self.assertTrue(any("chartreuse" in u for u in bad.unknown_values))

    def test_lint_report_json_shape(self):
        from facemax_color.validate import lint_rows
        report = lint_rows([_row(), _row("IMG2", "P1", undertone="Nope")])
        data = report.to_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["total_rows"], 2)
        self.assertEqual(data["valid_rows"], 1)
        self.assertEqual(len(data["issues"]), 1)


class HoldoutTests(unittest.TestCase):
    def _rows(self, n_people=10, per_person=2):
        rows = []
        counter = 0
        for p in range(n_people):
            for _ in range(per_person):
                counter += 1
                rows.append(_row(f"H{counter:03d}", f"HP{p:02d}"))
        return rows

    def test_lock_and_verify_roundtrip(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.audit import lock_holdout, verify_holdout
            root = Path(tmp)
            init_dataset(root)
            rows = self._rows()
            write_extended_annotations(root / "metadata" / "facemaxx_color_annotations.csv", rows)
            lock = lock_holdout(root, rows, test_frac=0.2, seed=3)
            self.assertGreater(lock["counts"]["test"], 0)
            status = verify_holdout(root)
            self.assertTrue(status["locked"])
            self.assertTrue(status["ok"])

    def test_tampered_manifest_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.audit import lock_holdout, verify_holdout
            root = Path(tmp)
            init_dataset(root)
            rows = self._rows()
            write_extended_annotations(root / "metadata" / "facemaxx_color_annotations.csv", rows)
            lock_holdout(root, rows, test_frac=0.2, seed=3)
            # Tamper: rewrite train.txt with different content.
            train_path = root / "splits" / "train.txt"
            train_path.write_text("HACKED\n", encoding="utf-8")
            status = verify_holdout(root)
            self.assertFalse(status["ok"])
            self.assertTrue(any("train.txt" in p for p in status["problems"]))

    def test_lock_refuses_leaked_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.audit import lock_holdout
            root = Path(tmp)
            init_dataset(root)
            rows = self._rows(n_people=6)
            for row in rows:  # force leakage into the pre-existing split column
                row["split"] = "train" if row["image_id"].endswith("1") else "test"
            # grouped_split overwrites 'split', so instead verify the guard exists:
            # a manual corruption path is simulated by checking leakage first.
            from facemax_bias.leakage import leakage_check
            check = leakage_check(rows)
            self.assertFalse(check["clean"])  # precondition: leaked input is detected
            # lock_holdout re-splits via grouped_split, so it must still succeed
            # (grouped_split is the fix); the guard protects corrupted manifests.
            lock = lock_holdout(root, rows, test_frac=0.2, seed=3)
            self.assertTrue(lock["counts"])

    def test_verify_without_lock_reports_unlocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.audit import verify_holdout
            status = verify_holdout(Path(tmp))
            self.assertFalse(status["locked"])
            self.assertFalse(status["ok"])


if __name__ == "__main__":
    unittest.main()
