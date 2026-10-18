"""Mocked method-selection tests; never contact Gemini."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgenticDataAnalyst


class AgentMethodSelectionTest(unittest.TestCase):
    def setUp(self):
        self.agent = object.__new__(AgenticDataAnalyst)
        self.agent.model = "mock-model"
        self.agent.fallback_models = ()
        self.agent.client = Mock()
        self.profile = {
            "summary": {
                "numeric_columns": ["revenue"],
                "categorical_columns": ["customer_id", "period", "region"],
            }
        }

    def response(self, clarification_needed, pairing_field=None):
        return Mock(
            text=json.dumps(
                {
                    "clarification_needed": clarification_needed,
                    "clarification_question": (
                        "Which field identifies repeated observations?"
                        if clarification_needed else ""
                    ),
                    "outcome": "revenue",
                    "predictors": ["period"],
                    "study_design": "repeated observations by customer",
                    "pairing_field": pairing_field,
                    "time_field": "period",
                    "assumptions_to_check": ["paired observations", "time ordering"],
                    "selected_method": "paired comparison",
                    "rationale": "The question requests within-customer comparison.",
                }
            )
        )

    def test_clarification_turn_then_follow_up_resolves_design(self):
        self.agent.client.models.generate_content.side_effect = [
            self.response(True),
            self.response(False, pairing_field="customer_id"),
        ]

        first = self.agent.select_method(
            "Compare revenue across periods for the same customers.", self.profile
        )
        self.assertTrue(first["clarification_needed"])

        history = [
            {"role": "assistant", "content": first["clarification_question"]},
            {"role": "user", "content": "customer_id identifies each customer."},
        ]
        second = self.agent.select_method(
            "customer_id identifies each customer.",
            self.profile,
            conversation_history=history,
        )

        self.assertFalse(second["clarification_needed"])
        self.assertEqual(second["pairing_field"], "customer_id")
        request = json.loads(
            self.agent.client.models.generate_content.call_args.kwargs["contents"]
        )
        self.assertEqual(request["recent_conversation"], history)

    def test_single_string_predictor_and_assumption_are_normalized(self):
        response = json.loads(self.response(False).text)
        response["predictors"] = "period"
        response["assumptions_to_check"] = "paired observations"
        self.agent.client.models.generate_content.return_value = Mock(text=json.dumps(response))

        selection = self.agent.select_method("Compare revenue", self.profile)

        self.assertEqual(selection["predictors"], ["period"])
        self.assertEqual(selection["assumptions_to_check"], ["paired observations"])

    def test_invalid_array_shape_uses_fallback_model(self):
        self.agent.fallback_models = ("fallback-model",)
        malformed = json.loads(self.response(False).text)
        malformed["predictors"] = {"field": "period"}
        self.agent.client.models.generate_content.side_effect = [
            Mock(text=json.dumps(malformed)),
            self.response(False),
        ]

        selection = self.agent.select_method("Compare revenue", self.profile)

        self.assertEqual(selection["predictors"], ["period"])
        calls = self.agent.client.models.generate_content.call_args_list
        self.assertEqual(calls[0].kwargs["model"], "mock-model")
        self.assertEqual(calls[1].kwargs["model"], "fallback-model")

    def test_all_models_with_invalid_method_schema_raise_clear_error(self):
        malformed = json.loads(self.response(False).text)
        malformed["assumptions_to_check"] = {"missing": "array"}
        self.agent.client.models.generate_content.return_value = Mock(text=json.dumps(malformed))

        with self.assertRaisesRegex(RuntimeError, "invalid method-selection response"):
            self.agent.select_method("Compare revenue", self.profile)

    def test_code_generation_receives_selected_method_plan(self):
        self.agent.client.models.generate_content.return_value = Mock(
            text="REASONING PLAN: Use a paired comparison.\n\n"
            "PYTHON CODE:\n```python\nprint('paired')\n```"
        )
        method_plan = {
            "selected_method": "paired t-test",
            "pairing_field": "customer_id",
            "outcome": "revenue",
        }

        generated = self.agent.generate_analysis_code(
            "Compare revenue by period.",
            self.profile,
            method_plan=method_plan,
        )

        self.assertIn("print('paired')", generated["code"])
        prompt = self.agent.client.models.generate_content.call_args.kwargs["contents"]
        self.assertIn('"pairing_field": "customer_id"', prompt)
        system_prompt = self.agent.client.models.generate_content.call_args.kwargs[
            "config"
        ].system_instruction
        self.assertIn("test-specific metric label", system_prompt)
        self.assertIn("mann_whitney_u_p_value", system_prompt)

    def test_code_generation_retries_malformed_or_empty_code_response(self):
        self.agent.fallback_models = ("fallback-model",)
        self.agent.client.models.generate_content.side_effect = [
            Mock(text="REASONING PLAN: Compare groups.\n\nPYTHON CODE:\n```python\nprint(\n````"),
            Mock(text="REASONING PLAN: Compare groups.\n\nPYTHON CODE:\n```python\nprint('ok')\n```")
        ]

        with self.assertWarnsRegex(RuntimeWarning, "unusable analysis code"):
            generated = self.agent.generate_analysis_code("Compare groups", self.profile)

        self.assertEqual(generated["code"], "print('ok')")
        calls = self.agent.client.models.generate_content.call_args_list
        self.assertEqual([call.kwargs["model"] for call in calls], [
            "mock-model", "fallback-model"
        ])

    def test_code_generation_raises_clear_error_when_all_responses_are_unusable(self):
        self.agent.fallback_models = ("fallback-model",)
        self.agent.client.models.generate_content.side_effect = [
            Mock(text="   "),
            Mock(text="I can help with that, but no code block was returned."),
        ]

        with self.assertWarnsRegex(RuntimeWarning, "unusable analysis code"):
            with self.assertRaisesRegex(RuntimeError, "could not produce usable analysis code"):
                self.agent.generate_analysis_code("Compare groups", self.profile)


if __name__ == "__main__":
    unittest.main()
