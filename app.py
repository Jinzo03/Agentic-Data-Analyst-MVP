import tempfile
import streamlit as st
import duckdb
from pathlib import Path

from profiler import DataProfiler
from agent import AgenticDataAnalyst
from runner import CodeExecutionRunner
from verifier import VerificationLayer

st.set_page_config(page_title="Agentic Data Analyst", layout="wide")

st.title(" Agentic Data Analyst")
st.subheader("Autonomous data analytics with statistical verification and review")
PROJECT_DIR = Path(__file__).resolve().parent
MAX_CODE_REFINEMENTS = 2

# Sidebar: File Upload & Configuration
with st.sidebar:
    st.header("1. Data Source")
    uploaded_file = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
    
    st.header("2. Analysis Request")
    user_query = st.text_area(
        "What would you like to analyze?",
        value="Compare revenue across departments and test if differences are statistically significant."
    )
    
    run_button = st.button("Run Agentic Analysis", type="primary")

if run_button and uploaded_file is not None:
    # Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_data_path = tmp_file.name

    output_dir = PROJECT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    with st.spinner("Running pre-flight profiling & statistical diagnostic check..."):
        con = duckdb.connect(database=":memory:")
        con.execute(f"CREATE TABLE dataset AS SELECT * FROM read_csv_auto('{temp_data_path}')")
        profiler = DataProfiler(con)
        profile_report = profiler.run_preflight_check()
        con.close()

    with st.spinner("Agent planning and writing execution code based on diagnostics..."):
        agent = AgenticDataAnalyst()
        generated = agent.generate_analysis_code(user_query, profile_report)

    if not generated["code"].strip():
        st.error("The model did not return executable Python code.")
        st.code(generated["raw_response"])
        st.stop()

    attempt_history = []
    refinement_count = 0
    with st.spinner(
        "Executing code; automatically repairing failures "
        f"(up to {MAX_CODE_REFINEMENTS} retries)..."
    ):
        runner = CodeExecutionRunner(temp_data_path, output_dir=str(output_dir))
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
                if refinement_count >= MAX_CODE_REFINEMENTS:
                    break

                failed_code = generated["code"]
                failed_trace = execution["error"] or "Execution failed without a traceback."
                refinement_count += 1
                try:
                    generated = agent.generate_analysis_code(
                        user_query,
                        profile_report,
                        previous_code=failed_code,
                        error_trace=failed_trace,
                    )
                except Exception as refinement_error:
                    execution["error"] = (
                        f"{failed_trace}\n\n"
                        "Auto-refinement request failed:\n"
                        f"{type(refinement_error).__name__}: {refinement_error}"
                    )
                    break

                if not generated["code"].strip():
                    execution["error"] = (
                        f"{failed_trace}\n\n"
                        "Auto-refinement returned no executable Python code.\n"
                        f"Gemini response:\n{generated['raw_response']}"
                    )
                    break
        finally:
            runner.con.close()

    if execution["status"] != "success":
        st.error(
            "Execution failed after "
            f"{len(attempt_history)} attempt(s), including "
            f"{refinement_count} automatic repair(s)."
        )
        for index, attempt in enumerate(attempt_history, start=1):
            with st.expander(f"Failed attempt {index}: generated code and traceback"):
                st.code(attempt["code"], language="python")
                st.code(attempt["error"] or "No traceback was captured.")
        st.code(execution["error"], language="python")
        st.stop()
    else:
        if refinement_count:
            st.success(
                f"Execution recovered after {refinement_count} automatic code repair(s)."
            )

        with st.spinner("Generating executive report and running verification audit..."):
            verifier = VerificationLayer()
            report = verifier.generate_report(user_query, execution["stdout"])
            verification = verifier.verify_report(report, execution["stdout"])

        # Display UI
        st.divider()

        st.markdown("### Visualizations")
        chart_paths = execution.get("chart_paths") or (
            [execution["chart_path"]] if execution.get("chart_path") else []
        )
        visible_charts = [
            path for path in chart_paths if path and Path(path).is_file()
        ]
        if visible_charts:
            for index, chart_path in enumerate(visible_charts, start=1):
                st.image(
                    chart_path,
                    caption=f"Generated visualization {index}",
                    width="stretch",
                )
        else:
            st.info(
                "No visualization was generated for this request. "
                "Ask for a chart or plot when you run the analysis again."
            )
        
        # Top Metric: Verification Guardrail Status
        if verification["verified"]:
            st.success(" Audit Verification Passed: No statistical hallucinations detected.")
        else:
            st.error(" Audit Verification Failed!")
            for v in verification["violations"]:
                st.warning(f"- {v}")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### Executive Analyst Report")
            st.markdown(report)
            
        with col_right:
            st.markdown("### Human Audit Trail")
            
            with st.expander(" Pre-Flight Diagnostic Report", expanded=False):
                st.json(profile_report)

            with st.expander(" Agent Strategy & Reasoning Plan", expanded=True):
                st.markdown(generated["plan"])

            if refinement_count:
                with st.expander(" Auto-Refinement History", expanded=False):
                    for index, attempt in enumerate(attempt_history[:-1], start=1):
                        st.markdown(f"**Failed attempt {index}**")
                        st.code(attempt["code"], language="python")
                        st.code(attempt["error"] or "No traceback was captured.")

            with st.expander(" Executed Python Code", expanded=True):
                st.code(generated["code"], language="python")

            with st.expander(" Raw Execution Terminal Output (stdout)", expanded=True):
                st.code(execution["stdout"])
