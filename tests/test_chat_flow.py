"""Chat orchestration state tests; no Streamlit, Gemini, or Docker required."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chat_flow import (
    active_answer,
    build_conversation_history,
    move_answer,
    prepare_assistant_message,
    regenerate_answer,
    submit_chat_turn,
)


class ChatFlowTest(unittest.TestCase):
    def test_initial_turn_passes_alpha_and_settings_and_stores_retry_context(self):
        profile = {"summary": {"rows": 3}}
        history = [{"role": "user", "content": "Earlier question"}]
        run_turn = Mock(return_value={"role": "assistant", "content": "Result"})

        response = submit_chat_turn(
            "New question", profile, history,
            run_turn=run_turn,
            data_path="data.csv",
            dataset_name="data.csv",
            dataset_hash="data-hash",
            max_code_refinements=4,
            alpha=0.02,
        )

        run_turn.assert_called_once_with(
            "New question", profile, history, "data.csv",
            dataset_name="data.csv", max_code_refinements=4, alpha=0.02,
        )
        self.assertEqual(response["answer_versions"][0]["content"], "Result")
        self.assertEqual(response["retry_context"]["alpha"], 0.02)
        self.assertEqual(response["retry_context"]["conversation_history"], history)

    def test_follow_up_context_uses_selected_answer_version(self):
        messages = [
            {"role": "user", "content": "Compare scores."},
            {
                "role": "assistant",
                "content": "Second answer",
                "active_answer_index": 0,
                "answer_versions": [
                    {"content": "First answer", "context": "First answer context"},
                    {"content": "Second answer", "context": "Second answer context"},
                ],
            },
        ]

        history = build_conversation_history(messages, limit=12)

        self.assertEqual(history, [
            {"role": "user", "content": "Compare scores."},
            {"role": "assistant", "content": "First answer context"},
        ])

    def test_answer_navigation_clamps_to_available_versions(self):
        message = {
            "turn_id": "turn-1",
            "active_answer_index": 0,
            "answer_versions": [{"content": "A"}, {"content": "B"}],
        }
        messages = [message]

        move_answer(messages, "turn-1", -1)
        self.assertEqual(message["active_answer_index"], 0)
        move_answer(messages, "turn-1", 1)
        self.assertEqual(active_answer(message)["content"], "B")
        move_answer(messages, "turn-1", 1)
        self.assertEqual(message["active_answer_index"], 1)

    def test_try_again_reuses_turn_context_and_alpha_then_preserves_versions(self):
        profile = {"summary": {"rows": 3}}
        history = [{"role": "user", "content": "Compare scores"}]
        message = prepare_assistant_message(
            {"role": "assistant", "content": "Original", "context": "Original context"},
            query="Compare scores",
            conversation_history=history,
            profile_report=profile,
            dataset_hash="sha256-1",
            max_code_refinements=3,
            alpha=0.01,
        )
        run_turn = Mock(return_value={
            "role": "assistant", "content": "Alternative", "context": "Alternative context"
        })

        regenerate_answer(
            message,
            run_turn=run_turn,
            data_path="sample.csv",
            dataset_name="sample.csv",
            dataset_hash="sha256-1",
        )

        run_turn.assert_called_once_with(
            "Compare scores", profile, history, "sample.csv",
            dataset_name="sample.csv", max_code_refinements=3, alpha=0.01,
        )
        self.assertEqual([v["content"] for v in message["answer_versions"]], [
            "Original", "Alternative"
        ])
        self.assertEqual(message["active_answer_index"], 1)
        self.assertEqual(active_answer(message)["content"], "Alternative")

    def test_try_again_rejects_different_dataset_without_calling_model(self):
        message = prepare_assistant_message(
            {"role": "assistant", "content": "Original"},
            query="Summarize",
            conversation_history=[],
            profile_report={},
            dataset_hash="old-hash",
            max_code_refinements=2,
            alpha=0.05,
        )
        run_turn = Mock()

        with self.assertRaisesRegex(ValueError, "dataset changed"):
            regenerate_answer(
                message, run_turn=run_turn, data_path="new.csv",
                dataset_name="new.csv", dataset_hash="new-hash",
            )
        run_turn.assert_not_called()
        self.assertEqual(len(message["answer_versions"]), 1)

    def test_clarification_is_not_offered_as_a_retryable_answer(self):
        response = {"role": "assistant", "content": "Which outcome?", "needs_clarification": True}

        prepared = prepare_assistant_message(
            response,
            query="Compare groups",
            conversation_history=[],
            profile_report={},
            dataset_hash="hash",
            max_code_refinements=2,
            alpha=0.05,
        )

        self.assertIs(prepared, response)
        self.assertNotIn("retry_context", prepared)
        self.assertNotIn("answer_versions", prepared)


if __name__ == "__main__":
    unittest.main()
