"""Pure chat-turn and answer-history helpers used by the Streamlit app."""

from __future__ import annotations

import uuid
from typing import Any, Callable


def active_answer(message: dict[str, Any]) -> dict[str, Any]:
    versions = message.get("answer_versions")
    if not versions:
        return message
    index = max(0, min(int(message.get("active_answer_index", len(versions) - 1)), len(versions) - 1))
    message["active_answer_index"] = index
    return versions[index]


def make_answer_version(message: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "answer_versions", "active_answer_index", "turn_id", "retry_context", "retry_error"
    }
    return {key: value for key, value in message.items() if key not in metadata}


def build_conversation_history(
    messages: list[dict[str, Any]], limit: int
) -> list[dict[str, str]]:
    """Build model context, using the currently selected assistant answer."""
    answers_by_turn = {
        message["turn_id"]: message
        for message in messages
        if message.get("role") == "assistant" and message.get("turn_id")
    }
    history = []
    for message in messages[-limit:] if limit > 0 else []:
        if message.get("role") == "assistant":
            selected = active_answer(message)
            content = selected.get("context", selected["content"])
        else:
            paired_answer = answers_by_turn.get(message.get("turn_id"))
            selected = active_answer(paired_answer) if paired_answer else message
            content = selected.get("query", message["content"])
        history.append({
            "role": message["role"],
            "content": content,
        })
    return history


def move_answer(messages: list[dict[str, Any]], message_id: str, offset: int) -> None:
    for message in messages:
        if message.get("turn_id") == message_id and message.get("answer_versions"):
            versions = message.get("answer_versions", [])
            if versions:
                current = int(message.get("active_answer_index", len(versions) - 1))
                message["active_answer_index"] = max(0, min(current + offset, len(versions) - 1))
            return


def find_answer_message(
    messages: list[dict[str, Any]], message_id: str
) -> dict[str, Any] | None:
    """Find the assistant message for a turn whose user and assistant share an ID."""
    return next(
        (
            message for message in messages
            if message.get("role") == "assistant"
            and message.get("turn_id") == message_id
            and message.get("answer_versions")
        ),
        None,
    )


def link_existing_user_messages(messages: list[dict[str, Any]]) -> None:
    """Link pre-edit-feature user turns to their adjacent versioned assistant turn."""
    for index, message in enumerate(messages):
        if (
            message.get("role") != "assistant"
            or not message.get("turn_id")
            or not message.get("answer_versions")
            or index == 0
        ):
            continue
        user_message = messages[index - 1]
        if user_message.get("role") == "user" and not user_message.get("turn_id"):
            user_message["turn_id"] = message["turn_id"]


def prepare_assistant_message(
    response: dict[str, Any], *, query: str, conversation_history: list[dict[str, str]],
    profile_report: dict[str, Any], dataset_hash: str | None,
    max_code_refinements: int, alpha: float, turn_id: str | None = None,
) -> dict[str, Any]:
    """Attach retry metadata to a completed response; clarification stays unversioned."""
    if response.get("needs_clarification"):
        return response
    response["turn_id"] = turn_id or uuid.uuid4().hex
    response["query"] = query
    response["answer_versions"] = [make_answer_version(response)]
    response["active_answer_index"] = 0
    response["retry_context"] = {
        "query": query,
        "conversation_history": conversation_history,
        "profile_report": profile_report,
        "dataset_hash": dataset_hash,
        "max_code_refinements": max_code_refinements,
        "alpha": alpha,
    }
    return response


def submit_chat_turn(
    query: str, profile_report: dict[str, Any], conversation_history: list[dict[str, str]],
    *, run_turn: Callable[..., dict[str, Any]], data_path: str,
    dataset_name: str | None, dataset_hash: str | None,
    max_code_refinements: int, alpha: float, turn_id: str | None = None,
) -> dict[str, Any]:
    """Execute a user turn and prepare its response for chat history."""
    try:
        response = run_turn(
            query, profile_report, conversation_history, data_path,
            dataset_name=dataset_name,
            max_code_refinements=max_code_refinements,
            alpha=alpha,
        )
    except Exception as exc:
        response = {
            "role": "assistant",
            "content": f"I couldn't complete that turn: {type(exc).__name__}: {exc}",
            "context": f"Turn error: {type(exc).__name__}: {exc}",
            "error": f"{type(exc).__name__}: {exc}",
            "audit": {},
        }
    return prepare_assistant_message(
        response,
        query=query,
        conversation_history=conversation_history,
        profile_report=profile_report,
        dataset_hash=dataset_hash,
        max_code_refinements=max_code_refinements,
        alpha=alpha,
        turn_id=turn_id,
    )


def regenerate_answer(
    message: dict[str, Any], *, run_turn: Callable[..., dict[str, Any]],
    data_path: str, dataset_name: str | None, dataset_hash: str | None,
    query: str | None = None,
) -> dict[str, Any]:
    """Run the saved request and append a version while retaining prior answers."""
    context = message.get("retry_context", {})
    if context.get("dataset_hash") != dataset_hash:
        raise ValueError("The dataset changed; this answer can no longer be regenerated.")
    selected_query = query or active_answer(message).get("query", context["query"])
    answer = run_turn(
        selected_query, context["profile_report"], context["conversation_history"],
        data_path, dataset_name=dataset_name,
        max_code_refinements=context["max_code_refinements"], alpha=context["alpha"],
    )
    answer["query"] = selected_query
    versions = message.setdefault("answer_versions", [make_answer_version(message)])
    versions.append(make_answer_version(answer))
    message["active_answer_index"] = len(versions) - 1
    message.pop("retry_error", None)
    return answer


def edit_chat_turn(
    messages: list[dict[str, Any]], turn_id: str, edited_query: str, *,
    run_turn: Callable[..., dict[str, Any]], data_path: str,
    dataset_name: str | None, dataset_hash: str | None,
) -> dict[str, Any]:
    """Regenerate an edited turn, retain its versions, and drop stale later turns."""
    assistant_index = next(
        (
            index for index, message in enumerate(messages)
            if message.get("role") == "assistant" and message.get("turn_id") == turn_id
        ),
        None,
    )
    if assistant_index is None:
        raise ValueError("Could not find the chat turn to edit.")
    if not edited_query.strip():
        raise ValueError("The edited question cannot be empty.")

    assistant = messages[assistant_index]
    answer = regenerate_answer(
        assistant,
        run_turn=run_turn,
        data_path=data_path,
        dataset_name=dataset_name,
        dataset_hash=dataset_hash,
        query=edited_query.strip(),
    )
    del messages[assistant_index + 1:]
    return answer
