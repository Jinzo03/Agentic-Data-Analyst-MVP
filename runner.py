import sys
import io
import os
import contextlib
import traceback
import duckdb
import pandas as pd
import matplotlib

matplotlib.use("Agg")

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
            
            # Save every open Matplotlib figure so the dashboard can display it.
            chart_paths = []
            for index, figure_number in enumerate(plt.get_fignums(), start=1):
                chart_path = os.path.join(
                    self.output_dir,
                    f"generated_chart_{index}.png",
                )
                figure = plt.figure(figure_number)
                figure.savefig(chart_path, bbox_inches="tight", dpi=160)
                chart_paths.append(chart_path)

            plt.close("all")

            return {
                "status": "success",
                "stdout": stdout_capture.getvalue().strip(),
                "chart_path": chart_paths[0] if chart_paths else None,
                "chart_paths": chart_paths,
                "error": None
            }
        except Exception as e:
            plt.close('all')
            return {
                "status": "error",
                "stdout": stdout_capture.getvalue().strip(),
                "chart_path": None,
                "chart_paths": [],
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            }
