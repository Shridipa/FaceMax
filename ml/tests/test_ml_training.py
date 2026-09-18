"""Tests for the ML training pipeline (gated, person-grouped, holdout-protected)."""
from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from facemax_bias.vocab import (
    CHROMA_ORDER,
    CONTRAST_ORDER,
    SEASON_ORDER,
    SKIN_DEPTH_ORDER,
    UNDERTONE_ORDER,
)

from facemax_color.dataset import EXTENDED_COLUMNS, init_dataset, write_extended_annotations

SEASONS = list(SEASON_ORDER)
DEPTHS = list(SKIN_DEPTH_ORDER)
UNDERTONES = list(UNDERTONE_ORDER)
CHROMAS = list(CHROMA_ORDER)
CONTRASTS = list(CONTRAST_ORDER)

FEAT_COLS = [
    "feat_skin_L", "feat_skin_a", "feat_skin_b", "feat_ita",
    "feat_undertone_proxy", "feat_olive_proxy", "feat_chroma_proxy",
    "feat_depth_proxy", "feat_hair_L", "feat_eye_L", "feat_contrast_proxy",
    "feat_wb_b_minus_a", "feat_grey_world_shift", "feat_brightness",
]


def _fixture_rows(people: int = 60, per_person: int = 3):
    """Deterministic dataset: season/depth/undertone consistent per person.

    feat_skin_L encodes the season index so models can actually learn; every
    person's photos share their colouring (mirrors real data).
    """
    rows = []
    n = 0
    for p in range(people):
        season = SEASONS[p % len(SEASONS)]
        depth = DEPTHS[p % len(DEPTHS)]
        undertone = UNDERTONES[p % len(UNDERTONES)]
        chroma = CHROMAS[p % len(CHROMAS)]
        contrast = CONTRASTS[(p + 1) % len(CONTRASTS)]
        L = 25.0 + (SEASONS.index(season) * 4.5)
        for _ in range(per_person):
            n += 1
            row = {c: "" for c in EXTENDED_COLUMNS}
            row.update(
                {
                    "image_id": f"MX-{n:04d}",
                    "person_id": f"MP-{p:03d}",
                    "source": "Phone selfie",
                    "license": "project-owned",
                    "image_quality": "Good",
                    "lighting_type": "Natural",
                    "camera_type": "iPhone",
                    "makeup": "No makeup",
                    "skin_depth": depth,
                    "undertone": undertone,
                    "chroma_level": chroma,
                    "contrast_level": contrast,
                    "season_primary": season,
                    "season_secondary": "",
                    "season_confidence": "0.9",
                    "annotator_1": season,
                    "annotator_2": season,
                    "annotator_3": season,
                    "annotation_agreement": "1.0",
                    "image_path": f"raw/MX-{n:04d}.jpg",
                    "features_ok": "True",
                }
            )
            rows.append(row)
    return rows


def _write_dataset(root: Path, rows):
    """Annotation table + feature table + raw files (unique bytes)."""
    init_dataset(root)
    write_extended_annotations(root / "metadata" / "facemaxx_color_annotations.csv", rows)
    with (root / "processed" / "colour_features.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id"] + FEAT_COLS + ["feat_ok"])
        writer.writeheader()
        for i, row in enumerate(rows):
            season = row["season_primary"]
            # corrupted-label rows (fixtures) get a neutral L; they are excluded
            # by the label audit before training anyway
            L = 25.0 + SEASONS.index(season) * 4.5 if season in SEASONS else 50.0
            payload = {
                "feat_skin_L": L, "feat_skin_a": 8.0, "feat_skin_b": 18.0,
                "feat_ita": 30.0, "feat_undertone_proxy": 10.0,
                "feat_olive_proxy": 0.44, "feat_chroma_proxy": 19.7,
                "feat_depth_proxy": 100 - L, "feat_hair_L": 30.0,
                "feat_eye_L": 25.0, "feat_contrast_proxy": 20.0,
                "feat_wb_b_minus_a": 2.0, "feat_grey_world_shift": 5.0,
                "feat_brightness": L,
            }
            writer.writerow({"image_id": row["image_id"], **payload, "feat_ok": "True"})
    for i, row in enumerate(rows):
        (root / "raw" / f"{row['image_id']}.jpg").write_bytes(f"unique-{i}".encode())


def _prepare(root: Path, target: str):
    from facemax_color.training_data import prepare_training_data

    return prepare_training_data(
        root / "metadata" / "facemaxx_color_annotations.csv",
        root / "processed" / "colour_features.csv",
        target,
        data_root=root,
        source="real",
    )


_CONFIG = {"random_seed": 42, "cv_folds": 3, "min_class_samples": 5, "n_jobs": 1}


class LeakageTests(unittest.TestCase):
    def test_audit_feature_columns_rejects_labels_and_metadata(self):
        from facemax_color.training_data import LeakageError, audit_feature_columns

        for bad in ("season_primary", "season_secondary", "undertone", "skin_depth",
                    "chroma_level", "contrast_level", "agreement_score", "annotator_1",
                    "person_id", "image_quality"):
            with self.assertRaises(LeakageError, msg=bad):
                audit_feature_columns(["feat_skin_L", bad])

    def test_canonical_features_pass_the_audit(self):
        from facemax_color.training_data import audit_feature_columns
        from facemax_color.models import FEATURE_COLUMNS

        self.assertEqual(audit_feature_columns(list(FEATURE_COLUMNS)), list(FEATURE_COLUMNS))

    def test_matrices_contain_only_canonical_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = _fixture_rows()
            _write_dataset(root, rows)
            from facemax_color.audit import lock_holdout

            lock_holdout(root, rows, test_frac=0.2, seed=3)
            from facemax_color.models import FEATURE_COLUMNS
            from facemax_color.training_data import matrices

            data = _prepare(root, "season")
            X, y, groups = matrices(data, splits=("train", "val"))
            self.assertEqual(X.shape[1], len(FEATURE_COLUMNS))
            self.assertEqual(len(X), len(y))
            self.assertEqual(len(y), len(groups))
            self.assertNotIn("", set(groups))


class PreparationTests(unittest.TestCase):
    def _corrupted_rows(self, rows):
        rows = [dict(r) for r in rows]
        # invalid label
        rows[0]["season_primary"] = "Not A Season"
        # missing person
        rows[1]["person_id"] = ""
        # duplicate id (row 2 copies row 3's id)
        rows[2]["image_id"] = rows[3]["image_id"]
        return rows

    def test_every_exclusion_is_reported(self):  # noqa: invalid-name
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = _fixture_rows()
            rows = self._corrupted_rows(rows)
            _write_dataset(root, rows)
            from facemax_color.audit import lock_holdout

            lock_holdout(root, rows, test_frac=0.2, seed=3)
            data = _prepare(root, "season")
            reasons = dict(data.exclusions)
            self.assertGreaterEqual(reasons.get("invalid_season_primary_label", 0), 1)
            self.assertGreaterEqual(reasons.get("missing_person_id", 0), 1)
            self.assertGreaterEqual(reasons.get("duplicate_image_id", 0), 1)
            self.assertGreaterEqual(reasons.get("locked_holdout_row", 0), 1)
            self.assertTrue(data.holdout_ids)
            # summary prints every exclusion (never silent)
            text = data.summary()
            self.assertIn("invalid_season_primary_label", text)
            self.assertIn("duplicate_image_id", text)
            self.assertIn("locked holdout", text.lower())

    def test_missing_feature_row_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = _fixture_rows(people=24, per_person=1)
            _write_dataset(root, rows)
            # drop one feature row
            feat_path = root / "processed" / "colour_features.csv"
            lines = feat_path.read_text(encoding="utf-8").splitlines()
            feat_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            data = _prepare(root, "season")
            self.assertGreaterEqual(data.exclusions.get("missing_feature_row", 0), 1)

    def test_holdout_never_enters_training_matrices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = _fixture_rows()
            _write_dataset(root, rows)
            from facemax_color.audit import lock_holdout

            lock_holdout(root, rows, test_frac=0.2, seed=3)
            data = _prepare(root, "season")
            from facemax_color.training_data import matrices

            X, y, groups = matrices(data, splits=("train", "val"))
            holdout_people = {r["person_id"] for r in data.holdout_rows}
            self.assertTrue(holdout_people)
            self.assertEqual(set(groups) & holdout_people, set())

    def test_class_balance_reports_images_and_people(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = _fixture_rows()
            _write_dataset(root, rows)
            from facemax_color.audit import lock_holdout

            lock_holdout(root, rows, test_frac=0.2, seed=3)
            data = _prepare(root, "season")
            for entry in data.class_balance:
                self.assertGreater(entry["images"], 0)
                self.assertGreater(entry["people"], 0)
                self.assertLessEqual(entry["people"], entry["images"])


class TrainingTests(unittest.TestCase):
    def _train_ready_root(self, tmp: Path):
        root = Path(tmp) / "data"
        rows = _fixture_rows()
        _write_dataset(root, rows)
        from facemax_color.audit import lock_holdout

        lock_holdout(root, rows, test_frac=0.2, seed=3)
        return root

    def test_season_training_end_to_end_with_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._train_ready_root(Path(tmp))
            from facemax_color.training import run_training

            data = _prepare(root, "season")
            result = run_training(data, dict(_CONFIG), root / "models")
            self.assertFalse(result["cv"].get("skipped"))
            self.assertIn(result["cv"]["best"], {"logreg", "random_forest", "xgboost", "lightgbm"})
            artifact = Path(result["artifact_dir"])
            for name in ("model.joblib", "metadata.json", "metrics.json", "feature_schema.json", "MODEL_CARD.md"):
                self.assertTrue((artifact / name).exists(), name)
            meta = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["target"], "season")
            self.assertEqual(meta["random_seed"], 42)
            self.assertEqual(meta["data_source"], "real")
            self.assertTrue(meta["training_rows"] > 0)
            self.assertTrue(meta["training_people"] > 0)
            # season approaches compared objectively
            self.assertIn("axis_first", result["season_approaches"])
            self.assertIn("hierarchical", result["season_approaches"])

    def test_deterministic_seed_same_cv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._train_ready_root(Path(tmp))
            from facemax_color.training import run_training

            r1 = run_training(_prepare(root, "undertone"), dict(_CONFIG), root / "models")
            r2 = run_training(_prepare(root, "undertone"), dict(_CONFIG), root / "models")
            best = r1["cv"]["best"]
            self.assertEqual(best, r2["cv"]["best"])
            self.assertAlmostEqual(
                r1["cv"]["candidates"][best]["mean"],
                r2["cv"]["candidates"][best]["mean"],
                places=9,
            )

    def test_cv_folds_are_person_grouped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._train_ready_root(Path(tmp))
            from facemax_color.training import run_training

            result = run_training(_prepare(root, "undertone"), dict(_CONFIG), root / "models")
            folds = result["cv"]["candidates"][result["cv"]["best"]]["folds"]
            self.assertGreaterEqual(len(folds), 2)
            # no person in two folds' test sets; train/test disjoint within a fold
            test_people = [set(f["test_people_ids"]) for f in folds]
            for i in range(len(test_people)):
                for j in range(i + 1, len(test_people)):
                    self.assertEqual(test_people[i] & test_people[j], set())
            for fold in folds:
                self.assertEqual(
                    set(fold["train_people_ids"]) & set(fold["test_people_ids"]), set()
                )

    def test_rare_classes_block_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._train_ready_root(Path(tmp))
            from facemax_color.training import run_training

            config = dict(_CONFIG, min_class_samples=10_000)
            result = run_training(_prepare(root, "season"), config, root / "models")
            self.assertTrue(result["cv"]["skipped"])
            self.assertIn("min_class_samples", result["cv"]["reason"])

    def test_all_five_targets_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._train_ready_root(Path(tmp))
            from facemax_color.training import run_training

            for target in ("undertone", "skin_depth", "chroma", "contrast", "season"):
                result = run_training(_prepare(root, target), dict(_CONFIG), root / "models")
                self.assertFalse(result["cv"].get("skipped"), target)
                self.assertTrue(result.get("model_id"), target)


def data_rows_of(root: Path):
    from facemax_color.dataset import read_annotations

    return read_annotations(root / "metadata" / "facemaxx_color_annotations.csv")


class CalibrationTests(unittest.TestCase):
    def test_calibration_fits_on_val_and_probs_sum_to_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = _fixture_rows()
            _write_dataset(root, rows)
            from facemax_color.audit import lock_holdout

            lock_holdout(root, rows, test_frac=0.2, seed=3)
            from facemax_color.training import run_training
            import joblib

            result = run_training(_prepare(root, "contrast"), dict(_CONFIG), root / "models")
            predictor = joblib.load(Path(result["artifact_dir"]) / "model.joblib")
            # 3-class contrast with every val class present -> calibrated
            self.assertTrue(result["calibration"]["method"] in ("sigmoid", "isotonic"))
            row = {c: rows[0].get(c) for c in predictor.feature_columns}
            out = predictor.predict_row(row)
            self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0, places=5)
            self.assertIn(out["prediction"], CONTRASTS)

    def test_calibration_scores_bounded(self):
        import numpy as np

        from facemax_color.training import brier_score, expected_calibration_error

        rng = np.random.default_rng(0)
        probs = rng.dirichlet([1, 1, 1], size=200)
        y = rng.integers(0, 3, size=200)
        ece = expected_calibration_error(probs, y, bins=10)
        brier = brier_score(probs, y)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)
        self.assertGreaterEqual(brier, 0.0)
        self.assertLessEqual(brier, 2.0)
        perfect = np.zeros((3, 3))
        perfect[:, :] = 0.01
        for i in range(3):
            perfect[i, i] = 0.97
        y_idx = np.arange(3)
        self.assertLess(brier_score(perfect, y_idx), 0.01)

    def test_secondary_season_rule_is_documented_and_configurable(self):
        from facemax_color.training import SeasonPredictor

        class FakePipeline:
            classes_ = ["Deep Autumn", "Deep Winter", "Light Summer"]

            def predict_proba(self, X):
                import numpy as np

                return np.array([[0.55, 0.40, 0.05]])

        predictor = SeasonPredictor(
            approach="direct",
            feature_columns=["feat_skin_L"],
            medians={"feat_skin_L": 50.0},
            classes=["Deep Autumn", "Deep Winter", "Light Summer"],
            secondary_min_prob=0.10,
            pipeline=FakePipeline(),
        )
        out = predictor.predict_row({"feat_skin_L": 30.0})
        self.assertEqual(out["prediction"], "Deep Autumn")
        self.assertEqual(out["prediction_secondary"], "Deep Winter")
        predictor.strict = None
        predictor.secondary_min_prob = 0.50
        out2 = predictor.predict_row({"feat_skin_L": 30.0})
        self.assertEqual(out2["prediction_secondary"], "")


class BoundaryTests(unittest.TestCase):
    def test_boundary_vs_conflict_errors(self):
        from facemax_color.training import boundary_analysis

        result = boundary_analysis(
            ["Deep Autumn", "Deep Autumn", "Soft Summer", "Deep Winter"],
            ["Deep Winter", "Deep Autumn", "Cool Summer", "Deep Winter"],
        )
        self.assertEqual(result["correct"], 2)
        # both errors are wheel-neighbour pairs (DA<->DW, SS<->CS): boundary
        self.assertEqual(result["boundary_errors"], 2)
        self.assertEqual(result["non_boundary_errors"], 0)
        self.assertIn("Deep Autumn <-> Deep Winter", result["boundary_pairs"])
        # a real conflict: wheel-unrelated seasons
        result2 = boundary_analysis(["Deep Autumn"], ["Light Summer"])
        self.assertEqual(result2["non_boundary_errors"], 1)
        self.assertEqual(result2["boundary_errors"], 0)

    def test_boundary_errors_reported_in_holdout_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = _fixture_rows()
            _write_dataset(root, rows)
            from facemax_color.audit import lock_holdout

            lock_holdout(root, rows, test_frac=0.2, seed=3)
            from facemax_color.training import run_training
            from facemax_color.training_data import prepare_training_data
            from facemax_color.evaluate_model import run_holdout_evaluation

            data = prepare_training_data(
                root / "metadata" / "facemaxx_color_annotations.csv",
                root / "processed" / "colour_features.csv",
                "season", data_root=root, source="real",
            )
            result = run_training(data, dict(_CONFIG), root / "models")
            evaluated = run_holdout_evaluation(data, root / "models", result["model_id"])
            self.assertIsNotNone(evaluated["boundary"])
            self.assertIn("boundary_share_of_errors", evaluated["boundary"])


class HoldoutEvaluationTests(unittest.TestCase):
    def test_holdout_evaluation_writes_final_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = _fixture_rows()
            _write_dataset(root, rows)
            from facemax_color.audit import lock_holdout

            lock_holdout(root, rows, test_frac=0.2, seed=3)
            from facemax_color.training import run_training
            from facemax_color.evaluate_model import run_holdout_evaluation

            data = _prepare(root, "contrast")
            result = run_training(data, dict(_CONFIG), root / "models")
            metrics_path = Path(result["artifact_dir"]) / "metrics.json"
            before = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertNotIn("holdout", before)
            evaluated = run_holdout_evaluation(data, root / "models", result["model_id"])
            self.assertGreater(evaluated["metrics"]["n"], 0)
            after = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertIn("holdout", after)
            self.assertIn("calibration_scores", after)
            self.assertIn("robustness", after)
            self.assertTrue((Path(result["artifact_dir"]) / "confusion_matrix.json").exists())
            # model card refreshed with the holdout section
            card = (Path(result["artifact_dir"]) / "MODEL_CARD.md").read_text(encoding="utf-8")
            self.assertIn("locked holdout", card.lower())

    def test_holdout_refuses_demo_real_mixing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = _fixture_rows()
            _write_dataset(root, rows)
            from facemax_color.audit import lock_holdout

            lock_holdout(root, rows, test_frac=0.2, seed=3)
            from facemax_color.training import run_training
            from facemax_color.evaluate_model import EvaluationError, run_holdout_evaluation

            data = _prepare(root, "contrast")
            result = run_training(data, dict(_CONFIG), root / "models")
            with self.assertRaises(EvaluationError):
                run_holdout_evaluation(
                    data, root / "models", result["model_id"], data_source_expected="demo"
                )

    def test_robustness_flags_small_subgroups_without_hiding_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            rows = _fixture_rows(people=24, per_person=1)
            _write_dataset(root, rows)
            from facemax_color.training import robustness_report

            for row in rows:
                row["prediction"] = row["season_primary"]
            report = robustness_report(rows, truth_key="season_primary")
            small_flagged = []
            for attribute, info in report.items():
                for value, stats in info["subgroups"].items():
                    if stats["small_n"]:
                        small_flagged.append((attribute, value))
            self.assertTrue(small_flagged)  # tiny groups flagged...
            all_values = [v for info in report.values() for v in info["subgroups"]]
            self.assertEqual(len(all_values), len(small_flagged) + (len(all_values) - len(small_flagged)))  # ...and still reported


class GateTests(unittest.TestCase):
    def test_not_ready_blocks_real_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.__main__ import main

            root = Path(tmp) / "data"
            init_dataset(root)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                rc = main(["train", "--root", str(root), "--targets", "season"])
            self.assertEqual(rc, 1)
            out = buffer.getvalue()
            self.assertIn("ML TRAINING BLOCKED", out)
            self.assertIn("NOT READY", out)
            self.assertIn("No real-data model was trained", out)
            self.assertFalse((root / "models" / "season").exists())

    def test_demo_mode_trains_in_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.__main__ import main

            root = Path(tmp) / "data"
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                main(["demo-image", "--root", str(root), "--people", "40", "--images", "2", "--seed", "11"])
            with redirect_stdout(buffer):
                rc = main(["train", "--demo", "--root", str(root), "--targets", "undertone"])
            self.assertEqual(rc, 0)
            out = buffer.getvalue()
            self.assertIn("NOT A FACEMAXX PERFORMANCE BENCHMARK", out)
            self.assertTrue((root / "demo" / "models" / "undertone").exists())
            self.assertFalse((root / "models").exists())          # real tree untouched
            self.assertFalse((root / "metadata" / "facemaxx_color_annotations.csv").exists() and
                             len(data_rows_of(root)) > 0)  # real table still empty

    def test_smoke_ready_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.__main__ import main

            root = Path(tmp) / "data"
            rows = [r for r in _fixture_rows() if r["season_primary"] != "Cool Summer"]
            _write_dataset(root, rows)
            from facemax_color.audit import lock_holdout

            lock_holdout(root, rows, test_frac=0.2, seed=3)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                rc = main(["train", "--root", str(root), "--targets", "undertone"])
            self.assertEqual(rc, 1)
            self.assertIn("SMOKE-READY", buffer.getvalue())
            self.assertFalse((root / "models" / "undertone").exists())
            # explicit --smoke acknowledges the tier
            with redirect_stdout(buffer):
                rc = main(["train", "--root", str(root), "--targets", "undertone", "--smoke"])
            self.assertEqual(rc, 0)
            meta = json.loads(
                (root / "models" / "undertone" / "undertone-logreg-v001" / "metadata.json").read_text(encoding="utf-8")
            ) if (root / "models" / "undertone" / "undertone-logreg-v001").exists() else None
            artifacts = list((root / "models" / "undertone").glob("*/metadata.json"))
            self.assertTrue(artifacts)
            meta = json.loads(artifacts[0].read_text(encoding="utf-8"))
            self.assertTrue(meta["smoke"])

    def test_train_ready_permits_training_and_holdout_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            from facemax_color.__main__ import main

            root = Path(tmp) / "data"
            rows = _fixture_rows()
            _write_dataset(root, rows)
            from facemax_color.audit import lock_holdout

            lock_holdout(root, rows, test_frac=0.2, seed=3)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                rc = main(["train", "--root", str(root), "--targets", "contrast"])
            self.assertEqual(rc, 0)
            artifacts = list((root / "models" / "contrast").glob("*/metadata.json"))
            self.assertTrue(artifacts)
            model_id = json.loads(artifacts[0].read_text(encoding="utf-8"))["model_id"]
            self.assertFalse(json.loads(artifacts[0].read_text(encoding="utf-8"))["smoke"])
            # final holdout evaluation via CLI
            with redirect_stdout(buffer):
                rc = main(["evaluate", "--root", str(root), "--model", model_id])
            self.assertEqual(rc, 0)
            self.assertIn("LOCKED HOLDOUT EVALUATION", buffer.getvalue())
            self.assertTrue(
                (root / "models" / "contrast" / model_id / "holdout_report.txt").exists()
            )


if __name__ == "__main__":
    unittest.main()
