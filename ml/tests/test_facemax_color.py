"""Tests for facemax_color. Run with:  python -m unittest discover tests -v"""
from __future__ import annotations

import unittest

import numpy as np

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

from facemax_color.dataset import (
    ANNOTATIONS_FILENAME,
    init_dataset,
    join_with_features,
    read_annotations,
    write_annotations,
)
from facemax_color.models import (
    FEATURE_COLUMNS,
    heuristic_depth,
    heuristic_undertone,
    predict_season_from_axes,
    season_from_axes,
    train_axis,
    train_bundle,
)


def _mkrow(**kw):
    row = {
        "image_id": "IMG1",
        "person_id": "P1",
        "season_primary": "Deep Autumn",
        "skin_depth": "Medium",
        "undertone": "Warm",
        "chroma": "Medium",
        "contrast": "Medium",
        "split": "test",
        "features_ok": True,
        "feat_skin_L": 60.0,
        "feat_skin_a": 9.0,
        "feat_skin_b": 24.0,
        "feat_ita": 20.0,
        "feat_undertone_proxy": 15.0,
        "feat_olive_proxy": 0.4,
        "feat_chroma_proxy": 26.0,
        "feat_depth_proxy": 40.0,
        "feat_hair_L": 20.0,
        "feat_eye_L": 30.0,
        "feat_contrast_proxy": 30.0,
        "feat_wb_b_minus_a": 5.0,
        "feat_grey_world_shift": 10.0,
        "feat_brightness": 60.0,
        "feat_exposure_flag": "ok",
        "feat_face_fraction": 0.3,
        "feat_skin_pixels": 50000,
    }
    row.update(kw)
    return row


class DatasetTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_init_dataset_creates_tree(self):
        from pathlib import Path
        annotations = init_dataset(Path(self.root))
        for sub in ("raw", "interim", "processed", "metadata"):
            self.assertTrue((Path(self.root) / sub).exists())
        self.assertEqual(annotations.name, ANNOTATIONS_FILENAME)

    def test_annotations_roundtrip(self):
        from pathlib import Path
        path = Path(self.root) / "ann.csv"
        rows = [_mkrow(image_id="A"), _mkrow(image_id="B", person_id="P2")]
        write_annotations(path, rows)
        loaded = read_annotations(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["image_id"], "A")

    def test_template_has_comment_row(self):
        from pathlib import Path
        path = Path(self.root) / "ann.csv"
        write_annotations(path, [])
        text = path.read_text(encoding="utf-8")
        self.assertIn("#", text)

    def test_join_with_features(self):
        rows = [_mkrow(image_id="A"), _mkrow(image_id="B")]
        features = {"A": {"feat_skin_L": 70.0}}
        joined = join_with_features(rows, features)
        by_id = {r["image_id"]: r for r in joined}
        self.assertTrue(by_id["A"]["features_ok"])
        self.assertFalse(by_id["B"]["features_ok"])
        self.assertEqual(by_id["A"]["feat_skin_L"], 70.0)

    def test_join_requires_features(self):
        rows = [_mkrow(image_id="A"), _mkrow(image_id="B")]
        features = {"A": {"feat_skin_L": 70.0}}
        joined = join_with_features(rows, features, require_features=True)
        self.assertEqual([r["image_id"] for r in joined], ["A"])


@unittest.skipUnless(HAVE_CV2, "OpenCV required")
class ColourTests(unittest.TestCase):
    def _synthetic_face(self):
        import cv2
        img = np.full((400, 400, 3), (128, 118, 108), np.uint8)
        cv2.ellipse(img, (200, 220), (120, 150), 0, 0, 360, (110, 160, 200), -1)
        cv2.rectangle(img, (60, 30), (340, 90), (30, 30, 30), -1)
        return img

    def test_region_lab_recentered(self):
        from facemax_color.colour import region_lab_samples, rgb_skin_mask
        img = self._synthetic_face()
        box = (80, 70, 240, 300)
        mask = rgb_skin_mask(img, box)
        regions = region_lab_samples(img, box, mask)
        self.assertGreaterEqual(len(regions), 3)
        for name, (L, a, b) in regions.items():
            self.assertTrue(0 <= L <= 100, f"L* out of range: {L}")
            self.assertTrue(-60 <= a <= 80, f"a* not re-centered: {a}")
            self.assertTrue(-60 <= b <= 80, f"b* not re-centered: {b}")

    def test_warm_skin_features(self):
        import facemax_color.colour as col
        img = self._synthetic_face()
        col_detect = col.detect_face_box
        col.detect_face_box = lambda image: (80, 70, 240, 300)
        try:
            import tempfile, pathlib
            tmp = pathlib.Path(tempfile.mkdtemp()) / "face.png"
            cv2.imwrite(str(tmp), img)
            res = col.extract_image("T1", tmp)
        finally:
            col.detect_face_box = col_detect
        self.assertTrue(res.ok, res.error)
        # Synthetic skin is warm: b* should exceed a*.
        self.assertGreater(res.undertone_proxy, 0)
        self.assertTrue(0 <= res.skin_L <= 100)
        self.assertEqual(res.exposure_flag, "ok")

    def test_no_face_handled(self):
        import facemax_color.colour as col
        import tempfile, pathlib
        blank = np.full((200, 200, 3), 90, np.uint8)
        tmp = pathlib.Path(tempfile.mkdtemp()) / "blank.png"
        cv2.imwrite(str(tmp), blank)
        res = col.extract_image("T2", tmp)
        self.assertFalse(res.ok)
        self.assertIn("no_face", (res.error or ""))

    def test_feature_row_keys_prefixed(self):
        from facemax_color.colour import ExtractionResult
        row = ExtractionResult(image_id="X").as_feature_row()
        for key in row:
            self.assertTrue(key.startswith("feat_"), key)


class SeasonLogicTests(unittest.TestCase):
    def test_classic_mappings(self):
        self.assertEqual(season_from_axes("Warm", "Light", "Bright", "Medium"), "Bright Spring")
        self.assertEqual(season_from_axes("Cool", "Very Fair", "Muted", "Low"), "Cool Summer")
        self.assertEqual(season_from_axes("Warm", "Deep", "Medium", "Medium"), "Deep Autumn")
        self.assertEqual(season_from_axes("Cool", "Deep", "Bright", "High"), "Deep Winter")
        self.assertEqual(season_from_axes("Neutral", "Very Deep", "Muted", "Medium"), "Deep Autumn")

    def test_all_12_reachable(self):
        seasons = set()
        for ut in ("Cool", "Neutral-Cool", "Neutral", "Neutral-Warm", "Warm", "Olive"):
            for dp in ("Very Fair", "Light", "Light-Medium", "Medium", "Deep", "Very Deep"):
                for ch in ("Muted", "Medium", "Bright"):
                    for ct in ("Low", "Medium", "High"):
                        seasons.add(season_from_axes(ut, dp, ch, ct))
        self.assertEqual(len(seasons), 12)

    def test_secondary_prediction(self):
        probs = {
            "undertone": {"Warm": 0.6, "Neutral-Warm": 0.4},
            "skin_depth": {"Deep": 1.0},
            "chroma": {"Medium": 1.0},
            "contrast": {"Medium": 1.0},
        }
        primary, secondary, confidence = predict_season_from_axes(
            {"undertone": "Warm", "skin_depth": "Deep", "chroma": "Medium", "contrast": "Medium"}, probs
        )
        self.assertEqual(primary, "Deep Autumn")
        self.assertTrue(0 <= confidence <= 1)


class HeuristicTests(unittest.TestCase):
    def test_undertone_bands(self):
        self.assertEqual(heuristic_undertone(8, 32), "Warm")
        self.assertEqual(heuristic_undertone(12, 2), "Cool")
        self.assertEqual(heuristic_undertone(5, 20), "Olive")

    def test_depth_bands_ordered(self):
        order = ["Very Fair", "Fair", "Light", "Light-Medium", "Medium", "Medium-Deep", "Deep", "Very Deep"]
        depths = [heuristic_depth(L) for L in (85, 74, 67, 60, 50, 42, 33, 20)]
        self.assertEqual(depths, order)


class ModelTests(unittest.TestCase):
    def _synthetic_dataset(self, n=240):
        """Rows whose features genuinely encode the labels (learnable signal)."""
        rng = np.random.default_rng(42)
        rows = []
        for i in range(n):
            depth_idx = int(rng.integers(0, 8))
            warm = bool(rng.random() < 0.5)
            L = 82 - depth_idx * 7 + rng.normal(0, 1.5)
            a = (9 if warm else 13) + rng.normal(0, 1.0)
            b = (24 if warm else 14) + rng.normal(0, 1.5)
            row = _mkrow(
                image_id=f"I{i:04d}",
                person_id=f"P{i // 2:04d}",
                feat_skin_L=L,
                feat_skin_a=a,
                feat_skin_b=b,
                feat_undertone_proxy=b - a,
                feat_depth_proxy=100 - L,
                feat_chroma_proxy=float(np.hypot(a, b)),
                feat_ita=30.0,
                feat_olive_proxy=a / max(b, 1),
                feat_hair_L=25.0,
                feat_eye_L=35.0,
                feat_contrast_proxy=L - 25.0,
                feat_wb_b_minus_a=b - a,
                feat_grey_world_shift=8.0,
                feat_brightness=L,
            )
            row["skin_depth"] = heuristic_depth(L)
            row["undertone"] = heuristic_undertone(a, b)
            row["season_primary"] = season_from_axes(
                row["undertone"], row["skin_depth"], "Medium", "Medium"
            )
            rows.append(row)
        return rows

    def test_train_axis_learns_depth(self):
        rows = self._synthetic_dataset()
        model, scores = train_axis(rows, "skin_depth", seed=3)
        self.assertIn(model.backend, ("logreg", "random_forest", "hist_gb"))
        best_score = max(v for v in scores.values() if v == v)
        self.assertGreater(best_score, 0.5)  # signal is strong; weak models fail this

    def test_train_bundle_end_to_end(self):
        rows = self._synthetic_dataset()
        bundle = train_bundle(rows, season_cv=False, seed=3)
        self.assertEqual(set(bundle.axes), {"undertone", "skin_depth", "chroma", "contrast"})
        features = {k: rows[0].get(k) for k in FEATURE_COLUMNS}
        result = bundle.predict(features)
        self.assertIn(result["prediction"], [
            "Light Spring", "Bright Spring", "Warm Spring", "Light Summer", "Soft Summer",
            "Cool Summer", "Soft Autumn", "Warm Autumn", "Deep Autumn", "Light Winter",
            "Cool Winter", "Deep Winter",
        ])
        self.assertIn("axes", result)
        self.assertTrue(0 <= result["prediction_confidence"] <= 1)

    def test_bundle_prediction_consistency(self):
        rows = self._synthetic_dataset(n=60)
        bundle = train_bundle(rows, season_cv=False, seed=5)
        features = {k: rows[1].get(k) for k in FEATURE_COLUMNS}
        out = bundle.predict(features)
        # Axis predictions should feed the deterministic season mapping.
        expected = season_from_axes(
            out["axes"]["undertone"], out["axes"]["skin_depth"],
            out["axes"]["chroma"], out["axes"]["contrast"],
        )
        self.assertEqual(out["prediction"], expected)


class DemoTests(unittest.TestCase):
    """End-to-end synthetic demo: wiring proof, NOT a fairness benchmark."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from pathlib import Path
        from facemax_color.demo import run_demo
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.summary = run_demo(cls.root, n_people=8, images_per_person=2, seed=99)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_summary_labels_synthetic(self):
        text = self.summary.render()
        self.assertIn("DATA STATUS: SYNTHETIC DEMO", text)
        self.assertIn("MODEL STATUS: DEMO/SYNTHETIC", text)
        self.assertIn("not a real-world fairness benchmark", text)

    def test_images_generated_and_counted(self):
        self.assertEqual(self.summary.images_generated, 16)
        raws = list((self.root / "demo" / "raw").glob("SYN*.jpg"))
        self.assertEqual(len(raws), 16)

    def test_image_dimensions_valid(self):
        import cv2
        raws = sorted((self.root / "demo" / "raw").glob("SYN*.jpg"))
        img = cv2.imread(str(raws[0]))
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[:2], (320, 256))

    def test_extraction_consumes_images(self):
        self.assertGreater(self.summary.faces_detected + self.summary.fallback_boxes, 0)

    def test_fallback_is_explicit(self):
        # Every fallback must be counted AND recorded in the sidecar json.
        import json
        sidecar = json.loads((self.root / "demo" / "fallback_boxes.json").read_text())
        self.assertEqual(len(sidecar["used_by"]), self.summary.fallback_boxes)
        self.assertEqual(sidecar["demo_face_box"], [41, 48, 174, 244])

    def test_features_not_hardcoded(self):
        # Extracted values must differ across differently-labelled people
        # (proves they come from pixels, not from constants).
        samples = self.summary.sample_features
        if len(samples) >= 2:
            pair = (samples[0]["L"], samples[1]["L"])
            self.assertNotAlmostEqual(*pair, places=6)

    def test_predictions_complete(self):
        self.assertEqual(self.summary.predictions_ok, self.summary.faces_detected + self.summary.fallback_boxes)
        self.assertTrue(self.summary.season_distribution)

    def test_report_generated_with_status_banner(self):
        report_path = self.root / "demo" / "bias_report.txt"
        self.assertTrue(report_path.exists())
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("FACEMAXX", text)

    def test_deterministic_for_fixed_seed(self):
        import tempfile
        from pathlib import Path
        from facemax_color.demo import run_demo
        with tempfile.TemporaryDirectory() as other:
            again = run_demo(Path(other), n_people=8, images_per_person=2, seed=99)
            self.assertEqual(self.summary.season_distribution, again.season_distribution)
            self.assertEqual(self.summary.faces_detected, again.faces_detected)
            self.assertEqual(self.summary.fallback_boxes, again.fallback_boxes)

    def test_demo_data_isolated_from_real_data(self):
        """Synthetic data must never contaminate the real annotation/raw tree."""
        # No synthetic images in the real raw tree.
        self.assertEqual(list((self.root / "raw").glob("SYN*.jpg")), [])
        # Real annotation CSV stays an empty template.
        from facemax_color.dataset import read_annotations
        real_rows = read_annotations(self.root / "metadata" / "facemaxx_color_annotations.csv")
        self.assertEqual(real_rows, [])
        # Demo rows live in the demo-local table with Synthetic provenance.
        demo_rows = read_annotations(self.root / "demo" / "annotations.csv")
        self.assertEqual(len(demo_rows), 16)
        self.assertTrue(all(r["source"] == "Synthetic" for r in demo_rows))


class EvaluationTests(unittest.TestCase):
    def _dataset(self, n=160):
        rng = np.random.default_rng(7)
        rows = []
        for i in range(n):
            depth_idx = int(rng.integers(0, 8))
            warm = bool(rng.random() < 0.5)
            L = 82 - depth_idx * 7 + rng.normal(0, 1.5)
            a = (9 if warm else 13) + rng.normal(0, 1.0)
            b = (24 if warm else 14) + rng.normal(0, 1.5)
            row = _mkrow(
                image_id=f"E{i:04d}",
                person_id=f"EP{i // 2:04d}",
                feat_skin_L=L, feat_skin_a=a, feat_skin_b=b,
                feat_undertone_proxy=b - a, feat_depth_proxy=100 - L,
                feat_chroma_proxy=float(np.hypot(a, b)), feat_ita=30.0,
                feat_olive_proxy=a / max(b, 1), feat_hair_L=25.0, feat_eye_L=35.0,
                feat_contrast_proxy=L - 25.0, feat_wb_b_minus_a=b - a,
                feat_grey_world_shift=8.0, feat_brightness=L,
            )
            row["skin_depth"] = heuristic_depth(L)
            row["undertone"] = heuristic_undertone(a, b)
            row["season_primary"] = season_from_axes(row["undertone"], row["skin_depth"], "Medium", "Medium")
            rows.append(row)
        return rows

    def test_run_evaluation_produces_report(self):
        from facemax_color.evaluate import run_evaluation
        output = run_evaluation(self._dataset(), season_cv=False, seed=3)
        text = output.report_text
        for token in ("FACEMAXX", "Overall Accuracy", "SUBGROUP PERFORMANCE", "SPLIT INTEGRITY"):
            self.assertIn(token, text)
        self.assertTrue(output.leakage["clean"])
        self.assertTrue(output.test_rows)
        for row in output.test_rows:
            self.assertTrue(row.get("prediction"))

    def test_actions_follow_the_principle(self):
        """Subgroup problems must produce data/model actions, never confidence scaling."""
        from facemax_bias.recommendations import actions_from_report
        from facemax_color.evaluate import run_evaluation
        output = run_evaluation(self._dataset(), season_cv=False, seed=3)
        actions = actions_from_report(output.report.as_dict())
        all_actions = actions["collect"] + actions["model"] + actions["gate"]
        for item in all_actions:
            self.assertNotIn("confidence", item["action"])  # no confidence-lowering 'fixes'
        # Quality gate is the only confidence mechanism and comes from image_quality.
        for item in actions["gate"]:
            self.assertEqual(item["action"], "quality_gate")

    def test_render_comparison(self):
        from facemax_color.evaluate import render_comparison
        from facemax_color.models import train_bundle
        bundle = train_bundle(self._dataset(n=240), season_cv=False, seed=3)
        text = render_comparison(bundle)
        for token in ("MODEL COMPARISON", "macro-F1", "logreg", "random_forest", "hist_gb"):
            self.assertIn(token, text)


class CorruptImageTests(unittest.TestCase):
    """Invalid/corrupt images must fail LOUDLY, never become garbage features."""

    def test_load_image_rejects_unreadable_bytes(self):
        if not HAVE_CV2:
            self.skipTest("opencv not installed")
        import tempfile
        from pathlib import Path

        from facemax_color.colour import load_image

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "garbage.jpg"
            p.write_bytes(b"this is not an image at all\x00\x01\x02")
            with self.assertRaises(ValueError):
                load_image(p)

    def test_extract_image_reports_corrupt_file_instead_of_raising(self):
        if not HAVE_CV2:
            self.skipTest("opencv not installed")
        import tempfile
        from pathlib import Path

        from facemax_color.colour import extract_image

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "garbage.jpg"
            p.write_bytes(b"\xff\xd8\xff" + b"\x00" * 512)  # truncated JPEG header
            result = extract_image("corrupt", p)
        self.assertFalse(result.ok)
        self.assertTrue(result.error)

    def test_extract_batch_reports_missing_file(self):
        if not HAVE_CV2:
            self.skipTest("opencv not installed")
        import tempfile
        from pathlib import Path

        from facemax_color.colour import extract_batch

        with tempfile.TemporaryDirectory() as tmp:
            results = extract_batch(["does_not_exist"], Path(tmp), progress_every=0)
        self.assertFalse(results["does_not_exist"].ok)
        self.assertEqual(results["does_not_exist"].error, "file_missing")


class ExifOrientationDecoderTests(unittest.TestCase):
    """Pin the OpenCV decoder's EXIF orientation behaviour (Phase 3 audit).

    The pinned decoder (opencv-python-headless 4.12, requirements.txt) applies
    the JPEG EXIF orientation tag during imdecode, so phone photos decode
    upright and face detection works. That is a LOAD-BEARING behaviour: if the
    decoder ever stops honouring the tag, rotated phone photos would decode
    sideways and systematically fail detection. These tests turn that silent
    failure mode into an explicit, actionable failure.
    """

    def _jpeg_with_orientation(self, pixels, orientation):
        import io

        ok, enc = cv2.imencode(".jpg", pixels)
        assert ok
        try:
            from PIL import Image

            buf = io.BytesIO()
            exif = Image.Exif()
            exif[274] = orientation
            Image.open(io.BytesIO(enc.tobytes())).save(buf, format="JPEG", exif=exif)
            return buf.getvalue()
            # PIL is a test-only convenience; without it, fall back to a plain
            # JPEG (orientation tag absent, nothing to pin).
        except ImportError:  # pragma: no cover
            return enc.tobytes()

    def test_decoder_applies_exif_orientation(self):
        if not HAVE_CV2:
            self.skipTest("opencv not installed")
        import io
        import tempfile
        from pathlib import Path

        from facemax_color.colour import load_image

        upright = np.zeros((64, 40, 3), dtype=np.uint8)  # 64h x 40w
        sideways = np.transpose(upright, (1, 0, 2))  # 40h x 64w stored sideways
        sideways = np.ascontiguousarray(sideways)
        sideways[:] = 0  # keep content constant; shape is the marker
        jpeg = self._jpeg_with_orientation(sideways, 6)
        if b"Exif" not in jpeg:  # pragma: no cover - Pillow missing
            self.skipTest("Pillow not available to write EXIF")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "phone.jpg"
            p.write_bytes(jpeg)
            img = load_image(p)
        self.assertEqual(
            img.shape[:2],
            (64, 40),
            "OpenCV no longer applies EXIF orientation during decode; rotated "
            "phone photos will systematically fail face detection. Add an "
            "explicit orientation transform in colour.load_image.",
        )

    def test_plain_jpeg_decodes_untouched(self):
        if not HAVE_CV2:  # pragma: no cover
            self.skipTest("opencv not installed")
        import tempfile
        from pathlib import Path

        from facemax_color.colour import load_image

        # No EXIF tag: pixels must pass through byte-identical (shape + data).
        pixels = np.full((50, 70, 3), 128, dtype=np.uint8)
        ok, enc = cv2.imencode(".jpg", pixels)
        assert ok
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "plain.jpg"
            p.write_bytes(enc.tobytes())
            img = load_image(p)
        self.assertEqual(img.shape, (50, 70, 3))
        np.testing.assert_array_equal(img, pixels)


if __name__ == "__main__":
    unittest.main()
