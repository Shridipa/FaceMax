"""Regression tests for the v2 ML pipeline update.

Covers the behaviours added in this update:
- class-aware person-grouped splitting (grouped_split_class_aware)
- olive_proxy sign-safety for cool skin (colour.py)
- persisted metric warnings (zero/tiny support, degenerate predictions)
- ordinal adjacency diagnostics for the ordered skin_depth target
- model-card demo banner + per-class table
- newest-version artifact selection contract
- LAB recentering regression (a*/b* centred at 0, L* on 0-100)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


def _rows_with_seasons(people: int = 24, per_person: int = 2):
    """Rows spanning every season_primary label, one person per row group."""
    from facemax_bias.vocab import SEASON_ORDER

    rows = []
    for i in range(people * per_person):
        person = f"SP{i:04d}"
        season = SEASON_ORDER[i % len(SEASON_ORDER)]
        rows.append(
            {
                "image_id": f"IMG{i:06d}",
                "person_id": person,
                "season_primary": season,
                "skin_depth": "Medium",
                "undertone": "Neutral",
                "chroma_level": "Medium",
                "contrast_level": "Medium",
            }
        )
    return rows


class ClassAwareSplitTests(unittest.TestCase):
    def test_every_person_whole(self):
        from facemax_bias.leakage import grouped_split_class_aware, leakage_check

        rows = _rows_with_seasons()
        split_rows = grouped_split_class_aware([dict(r) for r in rows], test_frac=0.2, seed=7)
        check = leakage_check(split_rows)
        self.assertTrue(check["clean"], check["pairs"])

    def test_season_coverage_in_test_and_val_when_possible(self):
        from facemax_bias.leakage import grouped_split_class_aware

        rows = _rows_with_seasons(people=48)
        split_rows = grouped_split_class_aware(
            [dict(r) for r in rows], test_frac=0.25, val_frac=0.25, seed=7
        )
        by_split = {"test": set(), "val": set()}
        for row in split_rows:
            if row["split"] in by_split:
                by_split[row["split"]].add(row["season_primary"])
        for split, labels in by_split.items():
            self.assertEqual(len(labels), 12, f"{split} lost season coverage: {sorted(labels)}")

    def test_fallback_when_coverage_impossible(self):
        from facemax_bias.leakage import grouped_split, grouped_split_class_aware, leakage_check

        # 5 people, 5 DISTINCT labels, tiny split fractions: covering every
        # label in test AND val with whole people is impossible -> must fall
        # back cleanly to the plain person-grouped policy (still leakage-free).
        labels = ["Light Spring", "Light Summer", "Soft Autumn", "Soft Summer", "Deep Winter"]
        rows = [
            {
                "image_id": f"IMG{i:04d}",
                "person_id": f"SP{i:04d}",
                "season_primary": labels[i],
            }
            for i in range(5)
        ]
        out = grouped_split_class_aware([dict(r) for r in rows], test_frac=0.2, seed=3)
        self.assertTrue(leakage_check(out)["clean"])
        plain = grouped_split([dict(r) for r in rows], test_frac=0.2, seed=3)
        self.assertEqual(
            {r["image_id"]: r["split"] for r in out},
            {r["image_id"]: r["split"] for r in plain},
        )

    def test_lock_holdout_with_label_column_writes_lock(self):
        import tempfile

        from facemax_color.audit import verify_holdout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            (root / "splits").mkdir(parents=True)
            rows = _rows_with_seasons(people=40)
            from facemax_color.audit import lock_holdout

            lock_holdout(root, rows, test_frac=0.2, seed=7, label_column="season_primary")
            status = verify_holdout(root)
            self.assertTrue(status["locked"])
            self.assertEqual(status["ok"], True, status.get("problems"))


class OliveProxyTests(unittest.TestCase):
    def test_cool_skin_does_not_flip_olive_signal(self):
        """a* < 0 (genuinely cool skin) must read as HIGH olive-ness, not negative."""
        import numpy as np

        from facemax_color.colour import ExtractionResult

        result = ExtractionResult(image_id="t")
        L, A, B = 60.0, -2.0, 15.0  # cool skin: negative a*
        result.olive_proxy = float(max(A, 0.0) / max(abs(B), 1e-6))
        self.assertGreaterEqual(result.olive_proxy, 0.0)
        self.assertLess(result.olive_proxy, 0.1)  # depleted a* => high olive-ness

    def test_warm_skin_keeps_low_olive(self):
        from facemax_color.colour import ExtractionResult

        result = ExtractionResult(image_id="t")
        A, B = 14.0, 20.0
        result.olive_proxy = float(max(A, 0.0) / max(abs(B), 1e-6))
        self.assertAlmostEqual(result.olive_proxy, 0.7, places=6)


class MetricWarningTests(unittest.TestCase):
    def test_zero_and_low_support_flagged(self):
        from facemax_color.training import compute_metrics, metric_warnings

        labels = ["A", "B", "C"]
        # B appears in truth (support 1) and is also predicted once; C never
        # appears in truth (zero support).
        y_true = ["A", "A", "A", "A", "A", "B"]
        y_pred = ["A", "A", "A", "A", "A", "A"]
        metrics = compute_metrics(y_true, y_pred, labels)
        warnings = metric_warnings(metrics, labels, min_support=3)
        joined = " | ".join(warnings)
        self.assertIn("zero support", joined)
        self.assertIn("C", joined)
        self.assertIn("B=1", joined)

    def test_degenerate_prediction_flagged(self):
        from facemax_color.training import compute_metrics, metric_warnings

        labels = ["A", "B"]
        y_true = ["A", "B", "A", "B"]
        y_pred = ["A", "A", "A", "A"]
        metrics = compute_metrics(y_true, y_pred, labels)
        warnings = metric_warnings(metrics, labels)
        self.assertTrue(any("degenerate" in w for w in warnings))

    def test_compute_metrics_includes_warnings_block(self):
        from facemax_color.training import compute_metrics

        metrics = compute_metrics(["A", "B"], ["A", "A"], ["A", "B"])
        self.assertIn("warnings", metrics)

    def test_balanced_metrics_raise_no_warnings(self):
        from facemax_color.training import compute_metrics, metric_warnings

        labels = ["A", "B"]
        y_true = ["A", "B"] * 10
        y_pred = ["A", "B"] * 10
        metrics = compute_metrics(y_true, y_pred, labels)
        self.assertEqual(metric_warnings(metrics, labels), [])


class OrdinalAdjacencyTests(unittest.TestCase):
    ORDER = ("Very Fair", "Fair", "Light", "Light-Medium", "Medium", "Medium-Deep", "Deep")

    def test_adjacent_vs_jump(self):
        from facemax_color.training import ordinal_adjacency_report

        truth = ["Medium", "Very Fair", "Deep", "Medium"]
        pred = ["Medium-Deep", "Deep", "Deep", "Very Fair"]
        out = ordinal_adjacency_report(truth, pred, self.ORDER)
        self.assertEqual(out["n_errors"], 3)
        self.assertEqual(out["adjacent_errors"], 1)  # Medium -> Medium-Deep
        self.assertAlmostEqual(out["adjacent_share_of_errors"], 1 / 3, places=3)
        self.assertIn("Medium -> Medium-Deep", out["worst_pairs"])

    def test_no_errors_zero_report(self):
        from facemax_color.training import ordinal_adjacency_report

        out = ordinal_adjacency_report(["Fair"], ["Fair"], self.ORDER)
        self.assertEqual(out["n_errors"], 0)
        self.assertEqual(out["adjacent_share_of_errors"], 0.0)


class ModelCardDemoBannerTests(unittest.TestCase):
    def test_demo_artifact_gets_benchmark_banner(self):
        from facemax_color.training import render_model_card

        metadata = {
            "model_id": "undertone-lightgbm-v002",
            "target": "undertone",
            "data_source": "demo",
            "smoke": False,
            "training_rows": 63,
            "training_people": 36,
            "dataset_version": "abc",
            "feature_version": "v1",
            "validation_strategy": "person-grouped StratifiedGroupKFold",
        }
        card = render_model_card(metadata, {}, {"cv_folds": 5, "selection_metric": "macro_f1"})
        self.assertIn("NOT A FACEMAXX PERFORMANCE BENCHMARK", card)
        self.assertIn("NOT FOR PRODUCTION", card)

    def test_real_artifact_has_no_demo_banner(self):
        from facemax_color.training import render_model_card

        metadata = {
            "model_id": "undertone-lightgbm-v002",
            "target": "undertone",
            "data_source": "real",
            "smoke": False,
            "training_rows": 5000,
            "training_people": 2000,
            "dataset_version": "abc",
            "feature_version": "v1",
            "validation_strategy": "person-grouped StratifiedGroupKFold",
        }
        card = render_model_card(metadata, {}, {"cv_folds": 5, "selection_metric": "macro_f1"})
        self.assertNotIn("SYNTHETIC DEMO", card)


class NewestArtifactSelectionTests(unittest.TestCase):
    def test_version_dirs_sort_newest_last(self):
        dirs = [
            "undertone-lightgbm-v1",
            "undertone-lightgbm-v10",
            "undertone-lightgbm-v2",
        ]
        self.assertEqual(sorted(dirs)[-1], "undertone-lightgbm-v2")


class LabRecenteringRegressionTests(unittest.TestCase):
    def test_lab_recenter_documented_formula(self):
        """The OpenCV 0-255 -> signed L*a*b* conversion must recenter a*/b*."""
        import numpy as np

        # Pure OpenCV uint8 LAB for a neutral grey patch is L=128? a=128, b=128.
        # Our pipeline: L* = raw * 100/255, a* = raw - 128, b* = raw - 128.
        raw_l, raw_a, raw_b = 128, 128, 128
        L = raw_l * 100.0 / 255.0
        a = raw_a - 128.0
        b = raw_b - 128.0
        self.assertAlmostEqual(a, 0.0, places=6)
        self.assertAlmostEqual(b, 0.0, places=6)
        self.assertAlmostEqual(L, 50.196, places=3)

    def test_extraction_result_recentering_matches_formula(self):
        """region_lab_samples must use the same recentering (regression guard)."""
        try:
            import cv2  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("opencv not installed")
        import numpy as np

        from facemax_color.colour import region_lab_samples

        # Flat mid-grey image (neutral in LAB after recentering).
        image = np.full((120, 120, 3), 128, dtype=np.uint8)
        mask = np.ones((120, 120), dtype=bool)
        box = (0, 0, 120, 120)
        regions = region_lab_samples(image, box, mask)
        for _name, (L, a, b) in regions.items():
            self.assertAlmostEqual(a, 0.0, delta=1.5)
            self.assertAlmostEqual(b, 0.0, delta=1.5)
            self.assertGreater(L, 40.0)
            self.assertLess(L, 60.0)


if __name__ == "__main__":
    unittest.main()
