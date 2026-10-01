import duckdb
import pandas as pd
import numpy as np

from runner import CodeExecutionRunner
from profiler import DataProfiler
from agent import AgenticDataAnalyst

# 1. Setup synthetic data with non-normal metric
np.random.seed(42)
df_sample = pd.DataFrame({
    "department": ["Engineering"] * 50 + ["Marketing"] * 50,
    # Non-normally distributed revenue (skewed distribution)
    "revenue": list(np.random.exponential(scale=100, size=50)) + list(np.random.exponential(scale=250, size=50))
})
df_sample.to_csv("company_data.csv", index=False)

# 2. Run Pre-flight Profiler (Step 2)
con = duckdb.connect(database=':memory:')
con.execute("CREATE TABLE dataset AS SELECT * FROM read_csv_auto('company_data.csv')")
profiler = DataProfiler(con)
profile_report = profiler.run_preflight_check()

# 3. Instantiate Agent & Generate Code (Step 3)
# agent.py reads GEMINI_API_KEY from .env. Do not pass the old OpenAI
# placeholder here, because it overrides the Gemini key loaded from .env.
agent = AgenticDataAnalyst()

query = "Compare revenue across Engineering and Marketing departments and test if the difference is statistically significant."
generated_output = agent.generate_analysis_code(query, profile_report)

print("=== AGENT REASONING PLAN ===")
print(generated_output["plan"])

print("\n=== GENERATED CODE ===")
print(generated_output["code"])

# 4. Execute Code in Runner Sandbox (Step 1)
runner = CodeExecutionRunner("company_data.csv")
execution_result = runner.execute_python_code(generated_output["code"])

print("\n=== EXECUTION RESULT ===")
print("Status:", execution_result["status"])
print("Captured Output:\n", execution_result["stdout"])
print("Saved Chart:", execution_result["chart_path"])
