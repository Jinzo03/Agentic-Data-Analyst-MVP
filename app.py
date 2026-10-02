import os
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

    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    with st.spinner("Running pre-flight profiling & statistical diagnostic check..."):
        con = duckdb.connect(database=":memory:")
        con.execute(f"CREATE TABLE dataset AS SELECT * FROM read_csv_auto('{temp_data_path}')")
        profiler = DataProfiler(con)
        profile_report = profiler.run_preflight_check()
        con.close()

    with st.spinner("Agent planning and writing execution code based on diagnostics..."):
        agent = AgenticDataAnalyst()
        generated = agent.generate_analysis_code(user_query, profile_report)

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
            
            if execution["chart_path"] and os.path.exists(execution["chart_path"]):
                st.image(execution["chart_path"], caption="Generated Chart", use_container_width=True)

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