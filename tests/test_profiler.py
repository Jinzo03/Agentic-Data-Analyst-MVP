"""Deterministic tests for profiler diagnostics; no Gemini or network required."""

import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiler import DataProfiler


class DataProfilerDiagnosticsTest(unittest.TestCase):
    def test_large_dataset_diagnostics_are_capped_at_ten_thousand_rows(self):
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                "CREATE TABLE dataset AS SELECT range AS value FROM range(10050)"
            )

            result = DataProfiler(connection).run_preflight_check()

            self.assertEqual(result["summary"]["total_rows"], 10050)
            self.assertEqual(result["summary"]["diagnostic_sample_size"], 10000)
            self.assertTrue(result["summary"]["diagnostics_sampled"])
        finally:
            connection.close()

    def test_simple_csv_profiles_with_one_all_column_fetch(self):
        class FetchCountingConnection:
            def __init__(self, connection):
                self.connection = connection
                self.fetchdf_calls = 0

            def execute(self, query):
                self.connection.execute(query)
                return self

            def fetchall(self):
                return self.connection.fetchall()

            def fetchone(self):
                return self.connection.fetchone()

            def fetchdf(self):
                self.fetchdf_calls += 1
                return self.connection.fetchdf()

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "simple.csv"
            csv_path.write_text(
                "group,value\nA,1\nA,2\nA,3\nB,4\nB,5\nB,6\n",
                encoding="utf-8",
            )
            connection = duckdb.connect(database=":memory:")
            try:
                connection.execute(
                    "CREATE TABLE dataset AS SELECT * FROM read_csv_auto(?)",
                    [str(csv_path)],
                )
                counted_connection = FetchCountingConnection(connection)

                result = DataProfiler(counted_connection).run_preflight_check()

                self.assertEqual(result["summary"]["total_rows"], 6)
                self.assertEqual(result["summary"]["total_columns"], 2)
                self.assertEqual(result["summary"]["diagnostic_sample_size"], 6)
                self.assertFalse(result["summary"]["diagnostics_sampled"])
                self.assertEqual(result["summary"]["numeric_columns"], ["value"])
                self.assertEqual(
                    result["column_diagnostics"]["value"]["mean"], 3.5
                )
                self.assertEqual(counted_connection.fetchdf_calls, 1)
            finally:
                connection.close()

    def test_reports_missingness_vif_zero_rate_and_time_diagnostic(self):
        frame = pd.DataFrame(
            {
                "event_time": pd.date_range("2025-01-01", periods=8, freq="D"),
                "measurement": [1.0, 2.1, None, 4.2, 5.0, 6.3, 7.1, 8.4],
                "predictor": [2.0, 3.8, 5.1, 7.2, 8.9, 11.2, 13.0, 15.1],
                "count_metric": [0, 1, 0, 2, 3, 1, 4, 2],
            }
        )
        connection = duckdb.connect(database=":memory:")
        try:
            connection.register("source_frame", frame)
            connection.execute("CREATE TABLE dataset AS SELECT * FROM source_frame")

            result = DataProfiler(connection).run_preflight_check()

            self.assertEqual(result["summary"]["total_rows"], 8)
            self.assertEqual(
                result["column_diagnostics"]["measurement"]["null_count"], 1
            )
            advanced = result["advanced_diagnostics"]
            self.assertEqual(
                advanced["missingness_summary"]["measurement"]["missing_count"], 1
            )
            self.assertIn("not_identifiable", advanced["missingness_mechanism"]["status"])
            self.assertTrue(advanced["multicollinearity_vif"]["results"])
            zero_rates = advanced["integer_nonnegative_zero_rates"]
            self.assertTrue(any(item["column"] == "count_metric" for item in zero_rates))
            self.assertEqual(
                advanced["autocorrelation"]["status"], "screened_with_time_column"
            )
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
