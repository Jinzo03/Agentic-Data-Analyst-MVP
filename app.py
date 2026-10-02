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

    with st.spinner("Executing generated code inside sandbox runner..."):
        runner = CodeExecutionRunner(temp_data_path, output_dir=str(output_dir))
        execution = runner.execute_python_code(generated["code"])
        runner.con.close()

    if execution["status"] != "success":
        st.error("Execution Engine Failed!")
        st.code(execution["error"], language="python")
    else:
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

            with st.expander(" Executed Python Code", expanded=True):
                st.code(generated["code"], language="python")

            with st.expander(" Raw Execution Terminal Output (stdout)", expanded=True):
                st.code(execution["stdout"])
