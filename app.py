"""Streamlit chat interface for iterative dataset analysis."""

import hashlib
import importlib.metadata
import json
import platform
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import streamlit as st

from agent import AgenticDataAnalyst
from profiler import DataProfiler
from runner import CodeExecutionRunner
from verifier import VerificationLayer


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
MAX_CODE_REFINEMENTS = 2
MAX_HISTORY_MESSAGES = 12
RUNS_DIR = OUTPUT_DIR / "runs"

DATA_HANDLING_PATTERNS = {
    "null removal": r"\.dropna\s*\(",
    "null imputation": r"\.(?:fillna|interpolate)\s*\(",
    "row filtering": r"\.(?:query|where)\s*\(|\.loc\s*\[[^\]]*(?:>|<|==|!=|>=|<=)",
    "outlier filtering": r"(?:outlier|isolationforest|localoutlierfactor|winsor)",
}


def review_data_handling(initial_code: str, final_code: str) -> list[dict[str, Any]]:
    """Flag common data-cleaning operations for human review, not as proof of change."""
    findings = []
    for label, pattern in DATA_HANDLING_PATTERNS.items():
        present = bool(re.search(pattern, final_code, flags=re.IGNORECASE))
        existed_before = bool(re.search(pattern, initial_code, flags=re.IGNORECASE))
        if present:
            findings.append({
                "operation": label,
                "present_in_final_code": True,
                "introduced_during_refinement": bool(initial_code) and not existed_before,
                "note": "Review the executed code and DATA HANDLING output to confirm scope and impact.",
            })
    return findings


def fingerprint_dataset(path: str) -> str:
    """Return a streaming SHA-256 fingerprint for the exact uploaded file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as data_file:
        for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def host_library_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": platform.python_version(), "libraries": {}}
    for distribution in ("google-genai", "streamlit", "duckdb", "pandas"):
        try:
            versions["libraries"][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions["libraries"][distribution] = None
    return versions


def chart_record_paths(paths: list[str]) -> list[str]:
    recorded_paths = []
    for chart_path in paths:
        path = Path(chart_path)
        try:
            recorded_paths.append(str(path.resolve().relative_to(PROJECT_DIR)))
        except ValueError:
            recorded_paths.append(str(path))
    return recorded_paths


def write_run_record(
    *,
    run_id: str,
    created_at: str,
    user_query: str,
    dataset_path: str,
    dataset_sha256: str,
    dataset_name: str | None,
    method_selection: dict[str, Any],
    generated: dict[str, Any],
    attempts: list[dict[str, Any]],
    execution: dict[str, Any],
    report: str | None,
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    """Persist a self-contained record of an analysis turn without copying data."""
    record_path = RUNS_DIR / f"{run_id}.json"
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": created_at,
        "question": user_query,
        "dataset": {
            "name": dataset_name,
            "sha256": dataset_sha256,
            "format": Path(dataset_path).suffix.lower().lstrip("."),
        },
        "method_selection": method_selection,
        "reasoning_plan": generated.get("plan", ""),
        "generated_code": generated.get("code", ""),
        "attempts": attempts,
        "environment": {
            "analysis_sandbox": execution.get("environment", {}),
            "application_host": host_library_versions(),
        },
        "execution": {
            "status": execution.get("status"),
            "stdout": execution.get("stdout", ""),
            "error": execution.get("error"),
            "numeric_results": execution.get("numeric_results", {}),
            "data_changes": execution.get("data_changes", []),
        },
        "report": report,
        "verification": verification,
        "charts": chart_record_paths(execution.get("chart_paths", [])),
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = record_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(record_path)
    return {"run_id": run_id, "path": str(record_path), "created_at_utc": created_at}

st.set_page_config(page_title="Agentic Data Analyst", layout="wide")
st.title("Agentic Data Analyst")
st.subheader("Conversational data analysis with statistical verification")


def profile_dataset(path: str) -> dict[str, Any]:
    """Profile the selected CSV or Parquet file once per uploaded dataset."""
    connection = duckdb.connect(database=":memory:")
    try:
        if Path(path).suffix.lower() == ".parquet":
            connection.execute(
                "CREATE TABLE dataset AS SELECT * FROM read_parquet(?)",
                [path],
            )
        else:
            connection.execute(
                "CREATE TABLE dataset AS SELECT * FROM read_csv_auto(?)",
                [path],
            )
        return DataProfiler(connection).run_preflight_check()
    finally:
        connection.close()


def run_analysis_turn(
    user_query: str,
    profile_report: dict[str, Any],
    conversation_history: list[dict[str, str]],
    data_path: str,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Generate, execute, optionally repair, report, and verify one chat turn."""
    agent = AgenticDataAnalyst()
    method_selection = agent.select_method(
        user_query,
        profile_report,
        conversation_history=conversation_history,
    )
    if method_selection["clarification_needed"]:
        question = str(method_selection.get("clarification_question", "")).strip()
        if not question:
            question = (
                "I need one more detail to choose an appropriate analysis method. "
                "What outcome or study-design detail should I use?"
            )
        return {
            "role": "assistant",
            "content": question,
            "context": (
                "Method selection is waiting for the user's clarification.\n"
                + json.dumps(method_selection, ensure_ascii=False, indent=2, default=str)
            ),
            "audit": {"method_selection": method_selection},
            "needs_clarification": True,
        }

    generated = agent.generate_analysis_code(
        user_query,
        profile_report,
        conversation_history=conversation_history,
        method_plan=method_selection,
    )
    if not generated["code"].strip():
        return {
            "role": "assistant",
            "content": "I couldn't produce executable analysis code for that request.",
            "context": generated["raw_response"],
            "error": generated["raw_response"],
            "audit": {
                "generated": generated,
                "method_selection": method_selection,
                "attempts": [],
            },
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_sha256 = fingerprint_dataset(data_path)
    run_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    attempt_history: list[dict[str, Any]] = []
    initial_code = generated["code"]
    refinement_count = 0
    runner = CodeExecutionRunner(data_path, output_dir=str(OUTPUT_DIR))
    try:
        while True:
            execution = runner.execute_python_code(generated["code"])
            attempt_history.append(
                {
                    "code": generated["code"],
                    "status": execution["status"],
                    "error": execution["error"],
                }
            )

            if execution["status"] == "success":
                break
            if not execution.get("recoverable", True):
                break
            if refinement_count >= MAX_CODE_REFINEMENTS:
                break

            failed_code = generated["code"]
            failed_trace = execution["error"] or "Execution failed without a traceback."
            refinement_count += 1
            try:
                generated = agent.generate_analysis_code(
                    user_query,
                    profile_report,
                    conversation_history=conversation_history,
                    previous_code=failed_code,
                    error_trace=failed_trace,
                    method_plan=method_selection,
                )
            except Exception as refinement_error:
                execution["error"] = (
                    f"{failed_trace}\n\nAuto-refinement request failed: "
                    f"{type(refinement_error).__name__}: {refinement_error}"
                )
                break

            if not generated["code"].strip():
                execution["error"] = (
                    f"{failed_trace}\n\nAuto-refinement returned no executable code.\n"
                    f"Gemini response:\n{generated['raw_response']}"
                )
                break
    finally:
        runner.con.close()

    audit: dict[str, Any] = {
        "generated": generated,
        "method_selection": method_selection,
        "attempts": attempt_history,
        "refinement_count": refinement_count,
        "profile_diagnostics": profile_report.get("advanced_diagnostics", {}),
        "data_changes": execution.get("data_changes", []),
        "data_handling_review": review_data_handling(
            initial_code, generated.get("code", "")
        ),
    }

    try:
        audit["reproducibility_record"] = write_run_record(
            run_id=run_id,
            created_at=created_at,
            user_query=user_query,
            dataset_path=data_path,
            dataset_sha256=dataset_sha256,
            dataset_name=dataset_name,
            method_selection=method_selection,
            generated=generated,
            attempts=attempt_history,
            execution=execution,
            report=None,
            verification=None,
        )
    except (OSError, TypeError, ValueError) as exc:
        audit["reproducibility_record_error"] = f"{type(exc).__name__}: {exc}"

    if execution["status"] != "success":
        error_text = execution["error"] or "The isolated runner returned an unknown error."
        return {
            "role": "assistant",
            "content": (
                f"I couldn't complete that analysis after {len(attempt_history)} "
                f"execution attempt(s) and {refinement_count} automatic repair(s). "
                "The traceback is available in the audit details below."
            ),
            "context": f"Analysis failed. Error and traceback:\n{error_text}",
            "error": error_text,
            "audit": audit,
        }

    numeric_results = execution.get("numeric_results", {})
    verifier = VerificationLayer()
    report = verifier.generate_report(
        user_query,
        execution["stdout"],
        numeric_results=numeric_results,
        conversation_history=conversation_history,
        data_changes=execution.get("data_changes", []),
    )
    verification = verifier.verify_report(
        report,
        execution["stdout"],
        numeric_results=numeric_results,
    )
    audit.update(
        {
            "stdout": execution["stdout"],
            "numeric_results": numeric_results,
            "verification": verification,
        }
    )
    try:
        audit["reproducibility_record"] = write_run_record(
            run_id=run_id,
            created_at=created_at,
            user_query=user_query,
            dataset_path=data_path,
            dataset_sha256=dataset_sha256,
            dataset_name=dataset_name,
            method_selection=method_selection,
            generated=generated,
            attempts=attempt_history,
            execution=execution,
            report=report,
            verification=verification,
        )
        audit.pop("reproducibility_record_error", None)
    except (OSError, TypeError, ValueError) as exc:
        audit["reproducibility_record_error"] = f"{type(exc).__name__}: {exc}"
    history_numeric_results = dict(list(numeric_results.items())[:100])
    numeric_context = json.dumps(
        history_numeric_results,
        indent=2,
        allow_nan=False,
    )
    if len(numeric_results) > len(history_numeric_results):
        numeric_context += "\n[Older numeric entries omitted from chat context.]"
    assistant_context = (
        f"Report:\n{report}\n\n"
        "Recent executed output (may be truncated):\n"
        f"{execution['stdout'][-6000:]}\n\n"
        "Runner numeric results JSON:\n"
        f"{numeric_context}\n\n"
        f"Verification result: {json.dumps(verification, ensure_ascii=False)}"
    )
    return {
        "role": "assistant",
        "content": report,
        "context": assistant_context,
        "charts": execution.get("chart_paths") or [],
        "audit": audit,
    }


def render_message(message: dict[str, Any]) -> None:
    """Render one persisted chat turn and its analysis artifacts."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] != "assistant":
            return
        if message.get("needs_clarification"):
            st.info("I need this detail before choosing the analysis method.")

        audit = message.get("audit", {})
        verification = audit.get("verification")
        if verification:
            if verification["verified"]:
                st.success("Metric-aware numeric and statistical verification passed.")
            elif verification.get("status") == "incomplete":
                st.warning(
                    "Verification incomplete: some report numbers could not be "
                    "matched confidently to their named metrics. Review the audit details."
                )
            else:
                st.error("Verification found unsupported or inconsistent claims.")
                for violation in verification["violations"]:
                    st.warning(violation)

        if audit.get("refinement_count"):
            st.info(
                "Execution recovered after "
                f"{audit['refinement_count']} automatic code repair(s)."
            )

        for item in audit.get("data_handling_review", []):
            if item.get("introduced_during_refinement"):
                st.warning(
                    f"Auto-refinement introduced code for {item['operation']}. "
                    "Review its scope and impact in Analysis details."
                )

        if message.get("error"):
            for index, attempt in enumerate(audit.get("attempts", []), start=1):
                with st.expander(f"Attempt {index}: code and traceback"):
                    st.code(attempt["code"], language="python")
                    st.code(attempt["error"] or "No traceback was captured.")
            if not audit.get("attempts"):
                st.code(message["error"])
            record_info = audit.get("reproducibility_record", {})
            record_path = Path(record_info.get("path", ""))
            if record_path.is_file():
                st.caption(f"Reproducibility record saved at {record_path}")
                st.download_button(
                    "Download run record",
                    data=record_path.read_bytes(),
                    file_name=record_path.name,
                    mime="application/json",
                    key=f"download_failed_run_{record_info['run_id']}",
                )
            elif audit.get("reproducibility_record_error"):
                st.warning(
                    "Could not save the reproducibility record: "
                    + audit["reproducibility_record_error"]
                )
            return

        chart_paths = [
            path for path in message.get("charts", []) if Path(path).is_file()
        ]
        if chart_paths:
            with st.expander("Visualizations", expanded=True):
                for index, chart_path in enumerate(chart_paths, start=1):
                    st.image(
                        chart_path,
                        caption=f"Generated visualization {index}",
                        width="stretch",
                    )

        generated = audit.get("generated", {})
        with st.expander("Analysis details", expanded=False):
            record_info = audit.get("reproducibility_record", {})
            record_path = Path(record_info.get("path", ""))
            if record_path.is_file():
                st.markdown("**Reproducibility record**")
                try:
                    shown_path = record_path.relative_to(PROJECT_DIR)
                except ValueError:
                    shown_path = record_path
                st.caption(f"Saved locally: {shown_path}")
                st.download_button(
                    "Download run record",
                    data=record_path.read_bytes(),
                    file_name=record_path.name,
                    mime="application/json",
                    key=f"download_run_{record_info['run_id']}",
                )
            elif audit.get("reproducibility_record_error"):
                st.warning(
                    "Could not save the reproducibility record: "
                    + audit["reproducibility_record_error"]
                )
            method_selection = audit.get("method_selection")
            if method_selection:
                st.markdown("**Method-selection plan**")
                st.json(method_selection)
            diagnostics = audit.get("profile_diagnostics", {})
            if diagnostics:
                st.markdown("**Preflight methodology diagnostics**")
                st.json(diagnostics)
            if audit.get("data_handling_review"):
                st.markdown("**Potential data-handling operations found in code**")
                st.caption("Static code scan; entries may not have executed.")
                st.json(audit["data_handling_review"])
            if audit.get("data_changes"):
                st.markdown("**Measured data changes during execution**")
                st.dataframe(audit["data_changes"], hide_index=True, width="stretch")
            else:
                st.markdown("**Measured data changes during execution**")
                st.caption(
                    "No changes through the tracked DataFrame operations were observed. "
                    "Other kinds of transformations may not be captured."
                )
            if generated.get("plan"):
                st.markdown("**Analysis plan**")
                st.markdown(generated["plan"])
            if generated.get("code"):
                st.markdown("**Executed code**")
                st.code(generated["code"], language="python")
            if audit.get("stdout"):
                st.markdown("**Captured output**")
                st.code(audit["stdout"])
            if audit.get("numeric_results") is not None:
                st.markdown("**Runner numeric results (JSON)**")
                st.json(audit["numeric_results"])
            verification = audit.get("verification", {})
            if verification.get("matched_claims"):
                st.markdown("**Report number to metric matches**")
                st.json(verification["matched_claims"])

        if audit.get("refinement_count"):
            with st.expander("Auto-refinement history", expanded=False):
                for index, attempt in enumerate(audit.get("attempts", [])[:-1], start=1):
                    st.markdown(f"**Failed attempt {index}**")
                    st.code(attempt["code"], language="python")
                    st.code(attempt["error"] or "No traceback was captured.")


def clear_conversation() -> None:
    st.session_state["chat_messages"] = []


for state_key, initial_value in (
    ("chat_messages", []),
    ("dataset_path", None),
    ("dataset_name", None),
    ("dataset_hash", None),
    ("profile_report", None),
):
    if state_key not in st.session_state:
        st.session_state[state_key] = initial_value


with st.sidebar:
    st.header("Data source")
    uploaded_file = st.file_uploader(
        "Upload CSV or Parquet",
        type=["csv", "parquet"],
        key="dataset_uploader",
    )
    if st.session_state["dataset_path"]:
        st.caption(f"Current dataset: {st.session_state['dataset_name']}")
        st.button(
            "Clear conversation",
            on_click=clear_conversation,
            width="stretch",
        )


if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    upload_hash = hashlib.sha256(uploaded_bytes).hexdigest()
    if upload_hash != st.session_state["dataset_hash"]:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_bytes)
            new_data_path = temp_file.name

        try:
            with st.spinner("Profiling this dataset for the conversation..."):
                new_profile = profile_dataset(new_data_path)
        except Exception as exc:
            Path(new_data_path).unlink(missing_ok=True)
            st.error(f"Could not load or profile the dataset: {exc}")
            st.stop()

        old_data_path = st.session_state["dataset_path"]
        if old_data_path and old_data_path != new_data_path:
            Path(old_data_path).unlink(missing_ok=True)
        st.session_state["dataset_path"] = new_data_path
        st.session_state["dataset_name"] = uploaded_file.name
        st.session_state["dataset_hash"] = upload_hash
        st.session_state["profile_report"] = new_profile
        st.session_state["chat_messages"] = []


if st.session_state["dataset_path"] is None:
    st.info("Upload a CSV or Parquet dataset to start a conversation.")
else:
    summary = st.session_state["profile_report"].get("summary", {})
    st.caption(
        f"Using **{st.session_state['dataset_name']}** · "
        f"{summary.get('total_rows', '?')} rows · "
        f"{summary.get('total_columns', '?')} columns. "
        "Upload a different file whenever you want to change the data."
    )

    if not st.session_state["chat_messages"]:
        with st.chat_message("assistant"):
            st.markdown(
                "Ask a question about the data. Follow up naturally, for example "
                "**“Now group this by region”** or **“Why did you choose that test?”**."
            )

    for stored_message in st.session_state["chat_messages"]:
        render_message(stored_message)

    prompt = st.chat_input("Ask about the data or follow up on an earlier result")
    if prompt:
        user_message = {"role": "user", "content": prompt}
        prior_messages = st.session_state["chat_messages"][-MAX_HISTORY_MESSAGES:]
        conversation_history = [
            {
                "role": message["role"],
                "content": message.get("context", message["content"]),
            }
            for message in prior_messages
        ]
        st.session_state["chat_messages"].append(user_message)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Analyzing your request..."):
            try:
                response_message = run_analysis_turn(
                    prompt,
                    st.session_state["profile_report"],
                    conversation_history,
                    st.session_state["dataset_path"],
                    dataset_name=st.session_state["dataset_name"],
                )
            except Exception as exc:
                response_message = {
                    "role": "assistant",
                    "content": f"I couldn't complete that turn: {type(exc).__name__}: {exc}",
                    "context": f"Turn error: {type(exc).__name__}: {exc}",
                    "error": f"{type(exc).__name__}: {exc}",
                    "audit": {},
                }
        st.session_state["chat_messages"].append(response_message)
        render_message(response_message)
