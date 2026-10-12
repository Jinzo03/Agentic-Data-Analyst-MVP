import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifier import VerificationLayer


class NumericVerificationTest(unittest.TestCase):
    def setUp(self):
        self.verifier = object.__new__(VerificationLayer)
        self.numeric_results = {
            "dataset_rows": 100,
            "group_a_mean": 125.5,
            "group_b_mean": 99.0,
            "p_value": 0.018,
            "significance_threshold": 0.05,
            "conversion_rate": 0.8,
        }
        self.stdout = "p_value: 0.018\ngroup_a_mean: 125.50"

    def test_accepts_report_numbers_backed_by_runner_json(self):
        report = (
            "The data contained 100 rows. Group A mean was 125.50 and "
            "Group B mean was 99. The p-value was 0.018, below the "
            "0.05 threshold. Conversion rate was 80%."
        )

        result = self.verifier.verify_report(
            report,
            self.stdout,
            numeric_results=self.numeric_results,
        )

        self.assertTrue(result["verified"], result["violations"])
        self.assertEqual(result["unsupported_numbers"], [])

    def test_flags_each_unmatched_report_number(self):
        report = "Revenue increased by 25% and generated a 2.4x return."

        result = self.verifier.verify_report(
            report,
            self.stdout,
            numeric_results=self.numeric_results,
        )

        self.assertFalse(result["verified"])
        self.assertEqual(result["unsupported_numbers"], ["25%", "2.4"])

    def test_same_value_attached_to_wrong_metric_is_incomplete(self):
        report = "Group B mean was 125.5."

        result = self.verifier.verify_report(
            report,
            self.stdout,
            numeric_results=self.numeric_results,
        )

        self.assertFalse(result["verified"])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["unmatched_claims"], ["125.5"])

    def test_significance_claim_without_p_value_is_incomplete(self):
        result = self.verifier.verify_report(
            "The result was statistically significant.",
            "group_mean: 10",
            numeric_results={"group_mean": 10},
        )

        self.assertFalse(result["verified"])
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(any("UNVERIFIED SIGNIFICANCE" in item for item in result["violations"]))

    def test_uses_labeled_runner_p_value_even_when_stdout_has_no_p_line(self):
        result = self.verifier.verify_report(
            "The p-value was 0.01, indicating a statistically significant result.",
            "",
            numeric_results={"metrics.p_value": 0.01},
        )

        self.assertTrue(result["verified"], result["violations"])

    def test_accepts_a_number_rounded_to_its_reported_precision(self):
        result = self.verifier.verify_report(
            "Revenue mean: 17.35.",
            "",
            numeric_results={"revenue_mean": 17.345},
        )

        self.assertTrue(result["verified"], result["violations"])

    def test_collapses_runner_aliases_for_the_same_labeled_metric(self):
        result = self.verifier.verify_report(
            "Group A mean: 12.5.",
            "group_a_mean: 12.5",
            numeric_results={
                "metrics.group_a_mean": 12.5,
                "stdout.group_a_mean": 12.5,
            },
        )

        self.assertTrue(result["verified"], result["violations"])

    def test_value_without_a_confident_metric_label_is_incomplete(self):
        result = self.verifier.verify_report(
            "The value was 99.",
            "",
            numeric_results={"group_a_mean": 99, "group_b_mean": 99},
        )

        self.assertFalse(result["verified"])
        self.assertEqual(result["status"], "incomplete")

    def test_accepts_formatted_r_squared_and_conventional_small_p_value(self):
        result = self.verifier.verify_report(
            "Model R-squared ($R^2$): 0.361. F-test p-value: p < 0.001, "
            "indicating a statistically significant model.",
            "",
            numeric_results={"r_squared": 0.36055, "f_p_value": 0.0},
        )

        self.assertTrue(result["verified"], result["violations"])
        self.assertEqual(result["unsupported_numbers"], [])

    def test_rejects_a_p_value_bound_that_conflicts_with_runner_metric(self):
        result = self.verifier.verify_report(
            "F-test p-value: p < 0.001.",
            "",
            numeric_results={"f_p_value": 0.02},
        )

        self.assertFalse(result["verified"])
        self.assertTrue(any("INCONSISTENT P-VALUE BOUND" in item for item in result["violations"]))

    def test_matches_formatted_metric_table_cells_to_headers_and_rows(self):
        report = """| Predictor | Coefficient | P-value |
|---|---:|---:|
| Alcohol | 0.276 | 0.001 |"""
        result = self.verifier.verify_report(
            report,
            "",
            numeric_results={
                "coef_alcohol": 0.2762,
                "p_alcohol": 0.001,
            },
        )

        self.assertTrue(result["verified"], result["violations"])


if __name__ == "__main__":
    unittest.main()
