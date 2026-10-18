import unittest
import sys
from pathlib import Path
from unittest.mock import Mock

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

    def test_significance_decision_uses_selected_alpha(self):
        report = "The test p-value was 0.04 and the result was significant."
        metrics = {"p_value": 0.04}

        strict = self.verifier.verify_report(report, "", metrics, alpha=0.01)
        default = self.verifier.verify_report(report, "", metrics, alpha=0.05)

        self.assertFalse(strict["verified"])
        self.assertTrue(any("FALSE POSITIVE" in item for item in strict["violations"]))
        self.assertTrue(default["verified"], default["violations"])

    def test_significance_claim_matches_the_named_test_among_multiple_p_values(self):
        result = self.verifier.verify_report(
            "The Welch's t-test was statistically significant.",
            "",
            {
                "welch_t_test_p_value": 0.01,
                "mann_whitney_p_value": 0.20,
            },
        )

        self.assertTrue(result["verified"], result["violations"])
        self.assertEqual(len(result["significance_checks"]), 1)
        self.assertEqual(result["significance_checks"][0]["metric"], "welch_t_test_p_value")
        self.assertTrue(result["significance_checks"][0]["significant"])

    def test_named_test_significance_uses_its_own_p_value_for_contradiction(self):
        result = self.verifier.verify_report(
            "The Mann-Whitney test was statistically significant.",
            "",
            {
                "welch_t_test_p_value": 0.001,
                "mann_whitney_p_value": 0.20,
            },
        )

        self.assertFalse(result["verified"])
        self.assertTrue(any("FALSE POSITIVE" in item for item in result["violations"]))
        self.assertEqual(result["significance_checks"][0]["metric"], "mann_whitney_p_value")

    def test_report_can_state_each_test_beside_its_own_p_value(self):
        report = (
            "Welch's t-test — p-value = 0.01; statistically significant. "
            "Mann-Whitney U test — p-value = 0.20; not statistically significant."
        )
        result = self.verifier.verify_report(
            report,
            "",
            {
                "welch_t_test_p_value": 0.01,
                "mann_whitney_u_test_p_value": 0.20,
            },
        )

        self.assertTrue(result["verified"], result["violations"])
        checks = {check["metric"]: check for check in result["significance_checks"]}
        self.assertEqual(set(checks), {
            "welch_t_test_p_value", "mann_whitney_u_test_p_value"
        })
        self.assertTrue(checks["welch_t_test_p_value"]["significant"])
        self.assertFalse(checks["mann_whitney_u_test_p_value"]["significant"])

    def test_generic_significance_claim_stays_incomplete_with_multiple_tests(self):
        result = self.verifier.verify_report(
            "The result was statistically significant.",
            "",
            {"welch_t_test_p_value": 0.01, "mann_whitney_p_value": 0.20},
        )

        self.assertFalse(result["verified"])
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(any("UNVERIFIED SIGNIFICANCE" in item for item in result["violations"]))

    def test_named_test_claim_is_incomplete_when_its_p_value_label_is_ambiguous(self):
        result = self.verifier.verify_report(
            "The Welch t-test was significant.",
            "",
            {"welch_t_test_p_value": 0.01, "welch_t_test_p": 0.02},
        )

        self.assertFalse(result["verified"])
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(any("matches multiple runner metrics" in item for item in result["violations"]))

    def test_report_generation_prompt_receives_selected_alpha(self):
        self.verifier.client = Mock()
        self.verifier.model = "mock-model"
        self.verifier.fallback_models = ()
        self.verifier.client.models.generate_content.return_value = Mock(text=(
            "## Executive Summary\nSummary.\n"
            "## Key Statistical Findings\nFindings.\n"
            "## Risk & Caveats\nCaveats.\n"
            "## Business Recommendation\nRecommendation."
        ))

        self.verifier.generate_report("Check a difference", "p_value: 0.04", alpha=0.01)

        call = self.verifier.client.models.generate_content.call_args.kwargs
        self.assertIn("alpha = 0.01", call["config"].system_instruction)
        self.assertIn("test's readable name immediately beside its own p-value", call["config"].system_instruction)
        self.assertIn("test-specific labels", call["config"].system_instruction)
        self.assertIn("Selected significance level (alpha): 0.01", call["contents"])

    def test_report_generation_retries_empty_response_on_fallback_model(self):
        self.verifier.model = "mock-model"
        self.verifier.fallback_models = ("fallback-model",)
        self.verifier.client = Mock()
        self.verifier.client.models.generate_content.side_effect = [
            Mock(text=" \n"),
            Mock(text=(
                "## Executive Summary\nSummary.\n"
                "## Key Statistical Findings\nThe groups differ.\n"
                "## Risk & Caveats\nCaveat.\n"
                "## Business Recommendation\nRecommendation."
            )),
        ]

        with self.assertWarnsRegex(RuntimeWarning, "unusable report"):
            report = self.verifier.generate_report("Compare groups", "p_value: 0.02")

        self.assertIn("The groups differ.", report)
        calls = self.verifier.client.models.generate_content.call_args_list
        self.assertEqual([call.kwargs["model"] for call in calls], [
            "mock-model", "fallback-model"
        ])

    def test_report_generation_raises_clear_error_when_all_responses_are_empty(self):
        self.verifier.model = "mock-model"
        self.verifier.fallback_models = ("fallback-model",)
        self.verifier.client = Mock()
        self.verifier.client.models.generate_content.side_effect = [
            Mock(text=None), Mock(text=" ")
        ]

        with self.assertWarnsRegex(RuntimeWarning, "unusable report"):
            with self.assertRaisesRegex(RuntimeError, "could not produce a usable analysis report"):
                self.verifier.generate_report("Compare groups", "p_value: 0.02")

    def test_report_generation_retries_response_missing_required_sections(self):
        self.verifier.model = "mock-model"
        self.verifier.fallback_models = ("fallback-model",)
        self.verifier.client = Mock()
        self.verifier.client.models.generate_content.side_effect = [
            Mock(text="Only a summary."),
            Mock(text=(
                "## Executive Summary\nSummary.\n"
                "## Key Statistical Findings\nFindings.\n"
                "## Risk & Caveats\nCaveats.\n"
                "## Business Recommendation\nRecommendation."
            )),
        ]

        with self.assertWarnsRegex(RuntimeWarning, "unusable report"):
            report = self.verifier.generate_report("Compare groups", "p_value: 0.02")

        self.assertIn("## Business Recommendation", report)
        self.assertEqual(self.verifier.client.models.generate_content.call_count, 2)


if __name__ == "__main__":
    unittest.main()
