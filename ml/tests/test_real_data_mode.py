"""Real-image CV testing mode (Category B) and its guarantees.

These tests run WITHOUT the real dataset (everything real-data-dependent is
either monkeypatched or exercised through the BLOCKED paths), so they belong
to the standard regression suite. RealArchiveIntegrationTests at the bottom
is permanently skipped in the standard suite and runs only when enabled
locally (remove its @unittest.skip) against the extracted licensed archive
- CI never needs the 3 GB dataset.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from facemax_color import real_data as rd


def _sources(root: Path, name: str, gate: str, status: str = "downloaded") -> None:
    ingest_dir = root / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    (ingest_dir / "sources.json").write_text(
        json.dumps(
            [{
                "name": name,
                "official_url": "https://example.org/ds",
                "intended_use": "test",
                "license_gate": gate,
                "status": status,
                "checksum": None,
                "receipt": None,
            }],
            indent=2,
        ),
        encoding="utf-8",
    )


class SegmentationMathTests(unittest.TestCase):
    """Deterministic checks of the additive confusion-count math."""

    def test_perfect_masks(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5] = True
        tp, fp, fn = rd.confusion_counts(mask, mask.copy())
        self.assertEqual((tp, fp, fn), (50, 0, 0))
        m = rd.segmentation_metrics(tp, fp, fn)
        self.assertEqual(m["iou"], 1.0)
        self.assertEqual(m["dice"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)

    def test_known_half_overlap(self):
        pred = np.zeros((10, 10), dtype=bool)
        pred[:, :5] = True
        gt = np.zeros((10, 10), dtype=bool)
        gt[:5] = True
        tp, fp, fn = rd.confusion_counts(pred, gt)
        self.assertEqual((tp, fp, fn), (25, 25, 25))
        m = rd.segmentation_metrics(tp, fp, fn)
        self.assertAlmostEqual(m["iou"], 1 / 3, places=4)
        self.assertAlmostEqual(m["dice"], 0.5, places=4)
        self.assertAlmostEqual(m["precision"], 0.5, places=4)
        self.assertAlmostEqual(m["recall"], 0.5, places=4)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            rd.confusion_counts(np.zeros((4, 4), bool), np.zeros((5, 5), bool))


class PlausibilityTests(unittest.TestCase):
    def test_good_box_passes(self):
        self.assertEqual(rd.box_plausibility((100, 100, 300, 400), (1000, 800, 3)), [])

    def test_rejections(self):
        self.assertIn("box_below_min_size",
                      rd.box_plausibility((0, 0, 40, 400), (1000, 800, 3)))
        self.assertIn("box_out_of_frame",
                      rd.box_plausibility((900, 100, 300, 400), (1000, 800, 3)))
        self.assertIn("implausible_aspect_ratio",
                      rd.box_plausibility((100, 100, 40, 400), (1000, 800, 3)))
        self.assertIn("box_dominates_frame",
                      rd.box_plausibility((2, 2, 995, 795), (1000, 800, 3)))


class SampleSelectionTests(unittest.TestCase):
    def test_deterministic_even_spread(self):
        universe = list(range(100))
        a = rd._sample_indices(universe, 10, 0)
        b = rd._sample_indices(universe, 10, 0)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)
        self.assertEqual(a[0], 0)
        self.assertEqual(a[-1], 90)

    def test_offset_slides_window_without_overlap(self):
        universe = list(range(100))
        base = rd._sample_indices(universe, 10, 0)
        nxt = rd._sample_indices(universe, 10, 1)
        self.assertEqual(set(base) & set(nxt), set())

    def test_limit_overrun_is_bounded(self):
        self.assertEqual(len(rd._sample_indices([1, 2, 3], 100, 0)), 3)
        self.assertEqual(rd._sample_indices([], 10, 0), [])


class NoDemoFallbackTests(unittest.TestCase):
    """The mandatory rule: real-data requests never substitute demo data."""

    def _blocked(self, root: Path, dataset: str = "CelebAMask-HQ"):
        report = rd.run_real_cv(root, dataset=dataset, limit=4)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIsNotNone(report["blocked_reason"])
        self.assertIs(report["synthetic_demo_substituted"], False)
        self.assertEqual(report["model_evaluation"]["synthetic_demo_substituted"], False)
        self.assertEqual(report["counts"]["images_processed"], 0)
        self.assertEqual(report["per_image"], [])
        return report

    def test_unregistered_dataset_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._blocked(Path(tmp), dataset="Unheard-Of")

    def test_unconfirmed_licence_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _sources(root, "FFHQ", gate="unconfirmed", status="not_downloaded")
            self._blocked(root, dataset="FFHQ")

    def test_licence_confirmed_but_no_images_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _sources(root, "CelebAMask-HQ", gate="confirmed")
            report = self._blocked(root)
            self.assertIn("no extracted images", report["blocked_reason"])

    def test_unimplemented_harness_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _sources(root, "Fitzpatrick17k", gate="confirmed")
            report = self._blocked(root, dataset="Fitzpatrick17k")
            self.assertIn("no real-image harness", report["blocked_reason"])

    def test_demo_tree_present_still_blocked(self):
        """A populated demo tree must NOT satisfy a real-data request."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _sources(root, "CelebAMask-HQ", gate="confirmed")
            demo_models = root / "demo" / "models" / "undertone" / "undertone-x-v001"
            demo_models.mkdir(parents=True)
            (demo_models / "model.joblib").write_bytes(b"x")
            self._blocked(root)

    def test_blocked_report_discloses_probe_provenance(self):
        """The demo-trained-probe disclosure must be PRESENT, not absent:
        the rule is 'no demo DATA substituted' (asserted in _blocked), while
        demo-trained probe provenance must stay visible in limitations."""
        report = self._blocked(Path(tempfile.gettempdir()) / "nonexistent-facemaxx-root")
        self.assertIn("synthetic-demo-trained", json.dumps(report).lower())


class AnnotationStatusTests(unittest.TestCase):
    def test_no_annotation_file_reported_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = rd._annotation_status(Path(tmp))
        self.assertFalse(status["expert_labelled_available"])
        self.assertEqual(status["annotated_images"], 0)

    def test_labels_counted_per_target(self):
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "metadata"
            meta.mkdir()
            with (meta / "facemaxx_color_annotations.csv").open(
                "w", newline="", encoding="utf-8"
            ) as fh:
                writer = csv.DictWriter(fh, fieldnames=[
                    "image_id", "person_id", "season_primary", "undertone",
                    "skin_depth",
                ])
                writer.writeheader()
                writer.writerow({"image_id": "IMG1", "person_id": "P1",
                                 "season_primary": "Deep Autumn",
                                 "undertone": "Warm", "skin_depth": "Deep"})
                writer.writerow({"image_id": "IMG2", "person_id": "P1",
                                 "season_primary": "", "undertone": "Warm",
                                 "skin_depth": ""})
            status = rd._annotation_status(Path(tmp))
        self.assertTrue(status["expert_labelled_available"])
        self.assertEqual(status["annotated_images"], 1)
        self.assertEqual(status["target_counts"]["undertone"], 2)
        self.assertEqual(status["target_counts"]["skin_depth"], 1)


class ColourExtractionIntegrationTests(unittest.TestCase):
    """decode -> detect -> skin mask -> extract on a SYNTHETIC photograph.

    Exercises the real code path (extract_image) with a synthetic image so
    the test runs everywhere; real-photo coverage is the opt-in marker below.
    """

    def test_extract_on_synthetic_face_like_image(self):
        import cv2

        img = np.full((400, 400, 3), 200, dtype=np.uint8)
        img[:, :, 0] = 120  # blueish background
        cv2.rectangle(img, (120, 100), (280, 320), (110, 160, 220), -1)  # skin-ish
        cv2.rectangle(img, (150, 160), (190, 180), (30, 30, 30), -1)   # eye
        cv2.rectangle(img, (210, 160), (250, 180), (30, 30, 30), -1)   # eye
        cv2.rectangle(img, (160, 240), (240, 260), (60, 40, 90), -1)   # mouth
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "face.jpg"
            cv2.imwrite(str(path), img)
            ext = rd.extract_image("SYN1", path)
        self.assertTrue(ext.ok, ext.error)
        self.assertGreater(ext.skin_pixels, 0)
        self.assertEqual(ext.face_box_source, "detector")


class ProbePredictorTests(unittest.TestCase):
    def test_no_demo_models_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(rd.load_probe_predictors(Path(tmp)), {})

    def test_corrupt_demo_artifact_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis_dir = root / "demo" / "models" / "undertone" / "undertone-x-v001"
            axis_dir.mkdir(parents=True)
            (axis_dir / "model.joblib").write_bytes(b"not a joblib file")
            self.assertEqual(rd.load_probe_predictors(root), {})


class GroundTruthMaskTests(unittest.TestCase):
    def test_missing_masks_return_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            masks_dir = Path(tmp) / "0"
            masks_dir.mkdir()
            self.assertIsNone(
                rd._ground_truth_face_mask((64, 64, 3), (0, 0, 32, 32), masks_dir, 0)
            )

    def test_exclusion_wins_in_gt_composition(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "0"
            shard.mkdir()
            skin = np.zeros((64, 64), dtype=np.uint8)
            skin[8:56, 8:56] = 255
            excl = np.zeros((64, 64), dtype=np.uint8)
            excl[8:20, 8:20] = 255  # overlaps the skin region
            for name, arr in (("00000_skin.png", skin), ("00000_nose.png", excl)):
                ok, buf = cv2.imencode(".png", arr)
                assert ok
                (shard / name).write_bytes(buf.tobytes())
            gt = rd._ground_truth_face_mask((64, 64, 3), (0, 0, 64, 64), shard.parent, 0)
        self.assertIsNotNone(gt)
        self.assertEqual(int(gt["skin"][8:20, 8:20].sum()), 0)  # excluded wins
        self.assertEqual(int(gt["skin"].sum()), 48 * 48 - 12 * 12)


class ReportRenderTests(unittest.TestCase):
    def test_blocked_render_has_no_demo_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _sources(root, "CelebAMask-HQ", gate="confirmed")
            report = rd.run_real_cv(root, dataset="CelebAMask-HQ", limit=4)
        text = rd.render_real_cv(report)
        self.assertIn("REAL-DATA TEST: BLOCKED", text)
        self.assertIn("Synthetic demo results were not substituted.", text)
        self.assertNotIn("images_processed          1", text)


@unittest.skip(
    "opt-in: enable locally to run real images through the pipeline "
    "(needs the extracted licensed CelebAMask-HQ archive; the standard "
    "suite must never depend on it)"
)
class RealArchiveIntegrationTests(unittest.TestCase):
    """Real images through the real pipeline (Category B evidence).

    Keep skipped in the default run; run it deliberately on a machine with
    the archive by deleting the @unittest.skip decorator above.
    """

    def test_bounded_real_sample(self):
        from facemax_color.real_data import run_real_cv

        root = Path(__file__).resolve().parents[1] / "data"
        report = run_real_cv(root, dataset="CelebAMask-HQ", limit=6)
        self.assertEqual(report["status"], "OK")
        self.assertEqual(report["counts"]["images_processed"], 6)
        self.assertGreater(report["segmentation_metrics"]["micro"]["iou"], 0.3)
        self.assertIs(report["synthetic_demo_substituted"], False)


if __name__ == "__main__":
    unittest.main()
