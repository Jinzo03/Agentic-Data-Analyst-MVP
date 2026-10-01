import sys
import io
import os
import contextlib
import traceback
import duckdb
import pandas as pd
import matplotlib.pyplot as plt

class CodeExecutionRunner:
    def __init__(self, data_path: str, output_dir: str = "./output"):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset into an in-memory DuckDB table named 'dataset'
        self.con = duckdb.connect(database=':memory:')
        self.con.execute(f"CREATE TABLE dataset AS SELECT * FROM read_csv_auto('{data_path}')")

    def execute_python_code(self, code_str: str) -> dict:
        """Executes LLM-generated Python code safely in context with DuckDB connection."""
        stdout_capture = io.StringIO()
        plt.close('all')  # Clear previous plots
        
        # Expose predefined tools into the code's global execution space
        exec_globals = {
            "con": self.con,
            "pd": pd,
            "duckdb": duckdb,
            "plt": plt,
            "output_dir": self.output_dir
        }
        
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code_str, exec_globals)
            
            # Save any generated plot automatically
            chart_path = None
            if plt.get_fignums():
                chart_path = os.path.join(self.output_dir, "generated_chart.png")
                plt.savefig(chart_path, bbox_inches='tight')
                plt.close('all')

            return {
                "status": "success",
                "stdout": stdout_capture.getvalue().strip(),
                "chart_path": chart_path,
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "stdout": stdout_capture.getvalue().strip(),
                "chart_path": None,
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            }