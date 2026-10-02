"""Run the complete profiling, analysis, execution, and verification pipeline."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from agent import AgenticDataAnalyst
from profiler import DataProfiler
from runner import CodeExecutionRunner
from verifier import VerificationLayer


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "company_data.csv"
OUTPUT_DIR = PROJECT_DIR / "output"
USER_QUERY = (
    "Compare revenue across Engineering and Marketing departments and "
    "test whether the difference is statistically significant."
)


def create_sample_data(path: Path) -> None:
    """Write deterministic sample data with skewed revenue distributions."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "department": ["Engineering"] * 50 + ["Marketing"] * 50,
            "revenue": np.concatenate(
                [
                    rng.exponential(scale=100, size=50),
                    rng.exponential(scale=250, size=50),
                ]
            ),
        }
    )
    data.to_csv(path, index=False)


def run_pipeline() -> int:
    """Run each pipeline stage and return a process exit code."""
    create_sample_data(DATA_PATH)

    profile_connection = duckdb.connect(database=":memory:")
    runner = None
    try:
        profile_connection.execute(
            "CREATE TABLE dataset AS SELECT * FROM read_csv_auto(?)",
            [str(DATA_PATH)],
        )
        profile_report = DataProfiler(profile_connection).run_preflight_check()

        # Both clients read GEMINI_API_KEY from the project .env file.
        agent = AgenticDataAnalyst()
        generated = agent.generate_analysis_code(USER_QUERY, profile_report)
        if not generated["code"].strip():
            raise RuntimeError(
                "Gemini did not return a fenced Python code block. "
                "Raw response:\n" + generated["raw_response"]
            )

        print("=== AGENT REASONING PLAN ===")
        print(generated["plan"])
        print("\n=== GENERATED CODE ===")
        print(generated["code"])

        runner = CodeExecutionRunner(str(DATA_PATH), output_dir=str(OUTPUT_DIR))
        execution = runner.execute_python_code(generated["code"])
        print("\n=== EXECUTION RESULT ===")
        print("Status:", execution["status"])
        print("Captured output:\n", execution["stdout"])
        print("Saved chart:", execution["chart_path"])

        if execution["status"] != "success":
            print("Execution error:\n", execution["error"])
            return 1

        verifier = VerificationLayer()
        report = verifier.generate_report(USER_QUERY, execution["stdout"])
        verification = verifier.verify_report(report, execution["stdout"])

        print("\n=== GENERATED ANALYST REPORT ===")
        print(report)
        print("\n=== AUDIT VERIFICATION STATUS ===")
        print("Passed verification check:", verification["verified"])
        if verification["violations"]:
            for violation in verification["violations"]:
                print("-", violation)

        return 0 if verification["verified"] else 1
    finally:
        profile_connection.close()
        if runner is not None:
            runner.con.close()


if __name__ == "__main__":
    raise SystemExit(run_pipeline())
