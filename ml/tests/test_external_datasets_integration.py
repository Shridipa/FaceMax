"""Tests for external-dataset integration (FFHQ / CelebAMask-HQ / Fitzpatrick17k).

Covers the phase-11 checklist for this update:
- dataset->task label-compatibility enforcement (external data can never
  supervise undertone/skin_depth/season/...),
- CelebAMask-HQ mask label mapping + image/mask alignment validation,
- Fitzpatrick17k label preservation (types stay Fitzpatrick types),
- ingestion isolation (demo tree refused) and idempotent re-recording,
- LAB a*/b* recentering on the real extraction path,
- real-data training-gate integrity (never weakened).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from facemax_color.dataset import REGISTRY_FILENAME, init_dataset


def _prepared_root(tmp: str) -> Path:
    """Dataset tree with a completed license field for CelebAMask-HQ."""
    root = Path(tmp)
    init_dataset(root)
    registry = root / REGISTRY_FILENAME
    registry.write_text(
        registry.read_text(encoding="utf-8").replace(
            "## 1. CelebAMask-HQ\n\n- **official source:**\n- **URL:**\n- **license:**",
            "## 1. CelebAMask-HQ\n\n- **official source:**\n- **URL:**\n- **license:** research-only, non-commercial",
        ),
        encoding="utf-8",
    )
    return root


class TaskCompatibilityTests(unittest.TestCase):
    def test_external_datasets_cannot_supervise_expert_targets(self):
        from facemax_color.external_datasets import LabelCompatibilityError, assert_labels_allowed

        for dataset in ("FFHQ", "CelebAMask-HQ", "Fitzpatrick17k"):
            for task in ("undertone", "skin_depth", "season", "chroma", "contrast"):
                with self.assertRaises(LabelCompatibilityError, msg=f"{dataset}->{task}"):
                    assert_labels_allowed(dataset, task)

    def test_permitted_tasks_are_allowed(self):
        from facemax_color.external_datasets import assert_labels_allowed

        assert_labels_allowed("FFHQ", "image_preprocessing")
        assert_labels_allowed("FFHQ", "lighting_robustness")
        assert_labels_allowed("CelebAMask-HQ", "skin_region_segmentation")
        assert_labels_allowed("Fitzpatrick17k", "skin_depth_robustness_evaluation_only")

    def test_unknown_dataset_and_unknown_task_refused(self):
        from facemax_color.external_datasets import LabelCompatibilityError, assert_labels_allowed

        with self.assertRaises(LabelCompatibilityError):
            assert_labels_allowed("ImageNet", "image_preprocessing")
        with self.assertRaises(LabelCompatibilityError):
            assert_labels_allowed("FFHQ", "seasonal_colour_classification")

    def test_status_reports_all_three_registered_not_downloaded(self):
        from facemax_color.external_datasets import external_dataset_status

        with tempfile.TemporaryDirectory() as tmp:
            root = _prepared_root(tmp)
            status = {row["name"]: row for row in external_dataset_status(root)}
            self.assertEqual(
                set(status), {"FFHQ", "CelebAMask-HQ", "Fitzpatrick17k"}
            )
            for row in status.values():
                self.assertTrue(row["registered"])
                self.assertFalse(row["downloaded"])
                self.assertFalse(row["extracted"])
                self.assertEqual(row["license_gate"], "unconfirmed")
                self.assertTrue(row["tasks"])


class CelebAMaskMappingTests(unittest.TestCase):
    def test_mask_labels_map_to_region_groups(self):
        from facemax_color.external_datasets import map_celebamask_labels

        mapped = map_celebamask_labels(
            ["skin", "neck", "hair", "u_lip", "l_lip", "l_eye", "r_eye",
             "background", "cloth", "totally_unknown"]
        )
        self.assertEqual(set(mapped["skin"]), {"skin", "neck"})
        self.assertIn("hair", mapped["exclude"])
        self.assertIn("background", mapped["exclude"])
        # Unknown labels preserved and reported, never silently dropped.
        self.assertEqual(mapped["unknown"], ["totally_unknown"])

    def test_pairing_validation_reports_problems_without_fixing_them(self):
        from facemax_color.external_datasets import validate_celebamask_pairing

        ok = validate_celebamask_pairing(
            image_sizes=[(512, 512), (512, 512)],
            mask_sizes=[(512, 512), (512, 512)],
            mask_labels=[["skin", "hair"], ["skin"]],
        )
        self.assertTrue(ok["ok"])

        bad = validate_celebamask_pairing(
            image_sizes=[(512, 512), (512, 512), (256, 256)],
            mask_sizes=[None, (512, 512), (512, 512)],
            mask_labels=[None, ["skin"], ["skin", "not_a_class"]],
        )
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["missing_masks"], 1)
        self.assertEqual(bad["dimension_mismatch_count"], 1)
        self.assertEqual(bad["invalid_mask_label_count"], 1)


class FitzpatrickPreservationTests(unittest.TestCase):
    def test_fitzpatrick_types_stay_separate(self):
        from facemax_color.external_datasets import validate_fitzpatrick_frame

        rows = [
            {"image": "a", "fitzpatrick_type": 3},
            {"image": "b", "fitzpatrick_type": -1},  # documented unknown
            {"image": "c", "fitzpatrick_type": None},
        ]
        report = validate_fitzpatrick_frame(rows)
        self.assertTrue(report["ok"])
        self.assertEqual(report["labelled"], 1)
        self.assertEqual(report["unknown_type"], 2)

    def test_fitzpatrick_types_cannot_become_facemaxx_labels(self):
        from facemax_color.external_datasets import validate_fitzpatrick_frame

        rows = [
            {"image": "a", "fitzpatrick_type": 3, "skin_depth": "Medium"},
            {"image": "b", "fitzpatrick_type": 2, "undertone": "Warm"},
        ]
        report = validate_fitzpatrick_frame(rows)
        self.assertFalse(report["ok"])
        self.assertEqual(len(report["problems"]), 2)


class IngestIsolationTests(unittest.TestCase):
    def test_demo_tree_archive_refused_for_real_ingestion(self):
        from facemax_color.ingest import IngestError, record_download

        with tempfile.TemporaryDirectory() as tmp:
            root = _prepared_root(tmp)
            demo_archive = root / "demo" / "celeba.zip"
            demo_archive.parent.mkdir(exist_ok=True)
            demo_archive.write_bytes(b"synthetic")
            with self.assertRaises(IngestError):
                record_download(root, "CelebAMask-HQ", demo_archive, license_confirmed=True)

    def test_rerecording_same_archive_is_idempotent(self):
        import json

        from facemax_color.ingest import read_sources, record_download

        with tempfile.TemporaryDirectory() as tmp:
            root = _prepared_root(tmp)
            archive = root / "celeba.zip"
            archive.write_bytes(b"celeba-bytes")

            first = record_download(root, "CelebAMask-HQ", archive, license_confirmed=True)
            sources = read_sources(root)
            entry = next(s for s in sources if s["name"] == "CelebAMask-HQ")
            self.assertEqual(entry["status"], "downloaded")

            second = record_download(root, "CelebAMask-HQ", archive, license_confirmed=True)
            self.assertEqual(first, second)
            sources_after = read_sources(root)
            entry_after = next(s for s in sources_after if s["name"] == "CelebAMask-HQ")
            self.assertEqual(entry_after["receipt"], entry["receipt"])
            # Exactly one receipt file exists - no duplicate records.
            receipts = list((root / "ingest").glob("*.receipt.json"))
            self.assertEqual(len(receipts), 1)

    def test_different_archive_replaces_receipt(self):
        from facemax_color.ingest import read_sources, record_download

        with tempfile.TemporaryDirectory() as tmp:
            root = _prepared_root(tmp)
            archive = root / "celeba.zip"
            archive.write_bytes(b"v1-bytes")
            first = record_download(root, "CelebAMask-HQ", archive, license_confirmed=True)
            archive.write_bytes(b"v2-bytes-different")
            second = record_download(root, "CelebAMask-HQ", archive, license_confirmed=True)
            self.assertNotEqual(first["sha256"], second["sha256"])
            sources = read_sources(root)
            entry = next(s for s in sources if s["name"] == "CelebAMask-HQ")
            self.assertEqual(entry["checksum"], second["sha256"])


class LabRecenteringTests(unittest.TestCase):
    def test_opencv_lab_ab_recentered_to_zero_on_grey(self):
        """8-bit OpenCV LAB has a*/b* centred at 128; the extraction path
        must re-centre so neutral grey reads a*=b*=0 (regression guard for
        any future external-dataset preprocessing)."""
        import numpy as np

        from facemax_color.colour import region_lab_samples

        # Flat mid-grey image (neutral in LAB after recentering); boolean mask.
        image = np.full((120, 120, 3), 128, dtype=np.uint8)
        mask = np.ones((120, 120), dtype=bool)
        box = (0, 0, 120, 120)
        samples = region_lab_samples(image, box, mask)
        self.assertTrue(samples)
        for name, (L, a, b) in samples.items():
            self.assertAlmostEqual(a, 0.0, delta=1.5, msg=f"{name} a* not recentered")
            self.assertAlmostEqual(b, 0.0, delta=1.5, msg=f"{name} b* not recentered")
            self.assertGreater(L, 40.0)
            self.assertLess(L, 60.0)


class GateIntegrityTests(unittest.TestCase):
    def test_real_gate_still_blocks_empty_dataset(self):
        from facemax_color.__main__ import _gate_allows_training

        with tempfile.TemporaryDirectory() as tmp:
            init_dataset(Path(tmp))
            allowed, status, reasons = _gate_allows_training(Path(tmp), smoke=False)
            self.assertFalse(allowed)
            self.assertEqual(status, "NOT READY")
            self.assertTrue(reasons)

    def test_demo_training_path_is_physically_separate(self):
        from facemax_color.__main__ import _training_paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _training_paths(root, demo=True)
            real = _training_paths(root, demo=False)
            self.assertEqual(paths["data_root"], root / "demo")
            self.assertNotEqual(paths["data_root"], real["data_root"])
            # Demo annotations/features/models stay inside the demo tree.
            for key in ("annotations", "features"):
                demo_prefix = str(root / "demo").replace("\\", "/")
                self.assertTrue(
                    str(paths[key]).replace("\\", "/").startswith(demo_prefix),
                    msg=f"{key} escaped the demo tree: {paths[key]}",
                )
                self.assertNotEqual(paths[key], real[key])


if __name__ == "__main__":
    unittest.main()
