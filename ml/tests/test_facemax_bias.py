"""Tests for facemax_bias. Run with:  python -m unittest discover ml/tests -v"""
from __future__ import annotations

import unittest

from facemax_bias.demo import generate_rows, demo_summary
from facemax_bias.distribution import distribution, flag_underrepresented, imbalance_ratio
from facemax_bias.labels import (
    agreement_score,
    ambiguous_rows,
    majority_label,
    seasonal_boundary_stats,
)
from facemax_bias.leakage import grouped_split, leakage_check, split_stats
from facemax_bias.metrics import (
    accuracy,
    balanced_accuracy,
    confusion_matrix,
    macro_f1,
    per_class_rates,
)
from facemax_bias.report import generate_report
from facemax_bias.robustness import (
    condition_stability,
    person_flip_rate,
    recommend_quality_floor,
)
from facemax_bias.schema import validate
from facemax_bias.splits import fold_sizes, group_kfold
from facemax_bias.subgroups import fairness_gaps, subgroup_performance, worst_best, verdict
from facemax_bias.vocab import SEASON_ORDER, SKIN_DEPTH_ORDER, quality_at_least


def _mkrow(**kw):
    row = {
        "image_id": "IMG1",
        "person_id": "P1",
        "season_primary": "Deep Autumn",
        "split": "test",
        "skin_depth": "Medium",
        "lighting_type": "Natural",
        "camera_type": "iPhone",
        "image_quality": "Good",
        "makeup": "No makeup",
        "prediction": "Deep Autumn",
    }
    row.update(kw)
    return row


class MetricsTests(unittest.TestCase):
    def test_accuracy(self):
        actual = ["A", "A", "B"]
        pred = ["A", "B", "B"]
        self.assertAlmostEqual(accuracy(actual, pred), 2 / 3)

    def test_macro_f1(self):
        actual = ["A", "A", "B", "B"]
        pred = ["A", "B", "B", "B"]
        # A: p=1.0, r=0.5 -> f1=0.6667; B: p=0.667, r=1.0 -> f1=0.8
        self.assertAlmostEqual(macro_f1(actual, pred), (2 / 3 + 0.8) / 2)

    def test_balanced_accuracy(self):
        actual = ["A", "A", "B", "B"]
        pred = ["A", "A", "A", "B"]
        self.assertAlmostEqual(balanced_accuracy(actual, pred), 0.75)

    def test_confusion_matrix_shape(self):
        matrix = confusion_matrix(["A", "B", "B"], ["A", "A", "B"])
        self.assertEqual(len(matrix), 2)
        self.assertEqual(matrix[0][0], 1)
        self.assertEqual(matrix[1][0], 1)

    def test_per_class_rates(self):
        rates = per_class_rates(["A", "A", "B"], ["A", "B", "B"])
        self.assertAlmostEqual(rates["A"]["recall"], 0.5)
        self.assertAlmostEqual(rates["B"]["precision"], 0.5)
        self.assertAlmostEqual(rates["B"]["recall"], 1.0)


class SubgroupTests(unittest.TestCase):
    def test_subgroup_performance_orders_by_vocab(self):
        rows = [
            _mkrow(image_id="I1", person_id="P1", skin_depth="Deep", prediction="Deep Winter"),
            _mkrow(image_id="I2", person_id="P2", skin_depth="Very Fair", prediction="Deep Autumn"),
        ]
        perf = subgroup_performance(rows, "skin_depth")
        self.assertEqual(list(perf), ["Very Fair", "Deep"])  # shallow -> deep
        self.assertAlmostEqual(perf["Very Fair"]["accuracy"], 1.0)
        self.assertAlmostEqual(perf["Deep"]["accuracy"], 0.0)

    def test_worst_best_and_verdict(self):
        perf = {"A": {"n": 50, "accuracy": 0.95}, "B": {"n": 50, "accuracy": 0.60}}
        wb = worst_best(perf)
        self.assertEqual(wb["worst_group"], "B")
        self.assertEqual(verdict(0.6, wb["gap"]), "serious problem")

    def test_small_groups_flagged(self):
        rows = [_mkrow(image_id=f"I{i}", person_id=f"P{i}", skin_depth="Very Deep", prediction="Deep Winter") for i in range(5)]
        perf = subgroup_performance(rows, "skin_depth")
        self.assertEqual(perf["Very Deep"]["small_n"], 1.0)


class LeakageTests(unittest.TestCase):
    def test_leakage_detected(self):
        rows = [
            _mkrow(image_id="I1", person_id="P1", split="train"),
            _mkrow(image_id="I2", person_id="P1", split="test"),
        ]
        result = leakage_check(rows)
        self.assertFalse(result["clean"])
        self.assertIn("P1", result["leaked_person_ids"])

    def test_grouped_split_keeps_person_whole(self):
        rows = [
            _mkrow(image_id="I1", person_id="P1"),
            _mkrow(image_id="I2", person_id="P1"),
            _mkrow(image_id="I3", person_id="P2"),
        ]
        grouped_split(rows, test_frac=0.5, seed=1)
        for person in ("P1", "P2"):
            splits = {r["split"] for r in rows if r["person_id"] == person}
            self.assertEqual(len(splits), 1)

    def test_split_stats(self):
        rows = [
            _mkrow(image_id="I1", person_id="P1", split="train"),
            _mkrow(image_id="I2", person_id="P1", split="train"),
            _mkrow(image_id="I3", person_id="P2", split="test"),
        ]
        stats = split_stats(rows)
        self.assertEqual(stats["train"]["images"], 2)
        self.assertEqual(stats["train"]["people"], 1)

    def test_group_kfold_disjoint(self):
        rows = generate_rows(n_people=30, images_per_person=2, seed=5)
        folds = group_kfold(rows, k=4, seed=1)
        seen = set()
        for _, train, eval_rows in folds:
            eval_people = {r["person_id"] for r in eval_rows}
            train_people = {r["person_id"] for r in train}
            self.assertFalse(eval_people & train_people)
            self.assertFalse(eval_people & seen)
            seen |= eval_people
        self.assertEqual(len(fold_sizes(folds)), 4)


class LabelTests(unittest.TestCase):
    def test_majority_label(self):
        self.assertEqual(majority_label(["A", "A", "B"]), "A")
        self.assertIsNone(majority_label(["A", "B", "C"]))

    def test_agreement_score(self):
        self.assertAlmostEqual(agreement_score(["A", "A", "B"]), 2 / 3)

    def test_ambiguous_rows(self):
        rows = [
            _mkrow(image_id="I1", annotator_1="Deep Autumn", annotator_2="Deep Autumn", annotator_3="Deep Winter"),
            _mkrow(image_id="I2", annotator_1="Deep Autumn", annotator_2="Deep Winter", annotator_3="Cool Winter"),
            _mkrow(image_id="I3", annotator_1="Deep Autumn", annotator_2="Deep Autumn", annotator_3="Deep Autumn"),
        ]
        # default: anything not unanimous is ambiguous
        self.assertEqual(ambiguous_rows(rows), ["I1", "I2"])
        # stricter: only worse-than-majority agreement (I1 has exactly 2/3)
        self.assertEqual(ambiguous_rows(rows, threshold=0.5), ["I2"])


class DistributionTests(unittest.TestCase):
    def test_distribution_shares(self):
        rows = [_mkrow(image_id="I1", skin_depth="Deep"), _mkrow(image_id="I2", skin_depth="Very Fair")]
        dist = distribution(rows, columns=["skin_depth"])
        self.assertAlmostEqual(dist["skin_depth"]["shares"]["Deep"], 0.5)

    def test_underrepresented_flag(self):
        rows = [_mkrow(image_id=f"I{i}", skin_depth="Very Fair") for i in range(40)]
        rows += [_mkrow(image_id=f"J{i}", skin_depth="Very Deep") for i in range(3)]
        flags = flag_underrepresented(distribution(rows, columns=["skin_depth"]))
        self.assertTrue(any("Very Deep" in f for f in flags))

    def test_imbalance_ratio(self):
        rows = [_mkrow(image_id="I1", skin_depth="Deep"), _mkrow(image_id="I2", skin_depth="Very Fair")]
        self.assertAlmostEqual(imbalance_ratio(distribution(rows, columns=["skin_depth"]), "skin_depth"), 1.0)


class RobustnessTests(unittest.TestCase):
    def test_condition_stability(self):
        rows = [
            _mkrow(image_id="I1", lighting_type="Natural", prediction="Deep Autumn"),
            _mkrow(image_id="I2", lighting_type="Low light", prediction="Deep Winter"),
        ]
        stats = condition_stability(rows, "lighting_type")
        self.assertAlmostEqual(stats["Natural"]["accuracy"], 1.0)
        self.assertAlmostEqual(stats["Low light"]["accuracy"], 0.0)

    def test_person_flip_rate(self):
        rows = [
            _mkrow(image_id="I1", person_id="P1", lighting_type="Natural", prediction="Deep Autumn"),
            _mkrow(image_id="I2", person_id="P1", lighting_type="Low light", prediction="Deep Winter"),
        ]
        self.assertAlmostEqual(person_flip_rate(rows, "lighting_type"), 1.0)

    def test_quality_floor(self):
        quality = {
            "Excellent": {"n": 10, "accuracy": 0.92},
            "Good": {"n": 10, "accuracy": 0.88},
            "Acceptable": {"n": 10, "accuracy": 0.79},
            "Poor": {"n": 10, "accuracy": 0.64},
        }
        floor, message = recommend_quality_floor(quality)
        self.assertEqual(floor, "Acceptable")
        self.assertIn("warning", message)


class SchemaTests(unittest.TestCase):
    def test_validate_ok(self):
        report = validate([_mkrow()])
        self.assertTrue(report.ok)
        self.assertEqual(report.total_rows, 1)

    def test_validate_missing_required(self):
        row = _mkrow()
        del row["season_primary"]
        report = validate([row])
        self.assertIn("season_primary", report.missing_required)
        self.assertFalse(report.ok)

    def test_validate_unknown_value(self):
        report = validate([_mkrow(image_quality="meh")])
        self.assertIn("meh", report.unknown_values.get("image_quality", {}))


class VocabTests(unittest.TestCase):
    def test_quality_at_least(self):
        self.assertTrue(quality_at_least("Excellent", "Good"))
        self.assertFalse(quality_at_least("Poor", "Acceptable"))

    def test_season_vocab_size(self):
        self.assertEqual(len(SEASON_ORDER), 12)


class DemoTests(unittest.TestCase):
    def test_demo_generation(self):
        rows = generate_rows(n_people=25, images_per_person=2, seed=11)
        self.assertEqual(len(rows), 50)
        self.assertEqual(len({r["person_id"] for r in rows}), 25)
        summary = demo_summary(rows)
        self.assertIn("25 people", summary)

    def test_demo_has_planted_bias(self):
        rows = generate_rows(n_people=400, seed=13)
        dist = distribution(rows, columns=["skin_depth", "season_primary"])
        very_deep = dist["skin_depth"]["counts"].get("Very Deep", 0)
        very_fair = dist["skin_depth"]["counts"].get("Very Fair", 0)
        self.assertLess(very_deep, very_fair / 5)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.rows = generate_rows(n_people=200, seed=17)
        self.report = generate_report(self.rows)

    def test_report_sections_present(self):
        text = self.report.text
        for token in (
            "Overall Accuracy",
            "Macro F1",
            "SUBGROUP PERFORMANCE",
            "REPRESENTATION",
            "CONFUSION",
            "LABEL QUALITY",
            "ROBUSTNESS",
            "SPLIT INTEGRITY",
        ):
            self.assertIn(token, text)

    def test_report_json_roundtrip(self):
        data = self.report.as_dict()
        self.assertIn("subgroups", data)
        self.assertIn("lighting", data["robustness"])
        self.assertIn("accuracy", data["overall"])

    def test_report_detects_planted_lighting_bias(self):
        lighting = self.report.as_dict()["robustness"]["lighting"]
        self.assertGreater(lighting["Natural"]["accuracy"], lighting["Low light"]["accuracy"])

    def test_report_flags_ambiguous_boundary(self):
        self.assertGreater(self.report.as_dict()["label_quality"]["ambiguous_count"], 0)

    def test_report_has_mixed_splits_and_no_leakage(self):
        data = self.report.as_dict()
        person_counts = data["leakage"]["person_counts"]
        self.assertEqual(set(person_counts), {"train", "val", "test"})
        self.assertTrue(data["leakage"]["clean"])

    def test_fairness_gaps_over_attributes(self):
        gaps = fairness_gaps(self.rows, attributes=["skin_depth", "lighting_type"])
        self.assertIn("skin_depth", gaps)
        self.assertIn("verdict", gaps["skin_depth"])


if __name__ == "__main__":
    unittest.main()
