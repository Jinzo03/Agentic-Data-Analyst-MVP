"""Run the sandbox worker on fixed data without Docker or Gemini."""

import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner import _SANDBOX_WORKER


class SandboxWorkerOutputTest(unittest.TestCase):
    def test_outputs_metrics_and_measures_dataframe_changes(self):
        csv_data = b"group,value\nA,1\nA,\nB,100\nB,200\n"
        code = """
df = con.execute('SELECT * FROM dataset').fetchdf()
clean = df.dropna()
filtered = clean[clean['value'] < 150]
filled = df.fillna({'value': 0})
df.loc[df['value'] > 150, 'value'] = 150
metrics['filtered_rows'] = int(len(filtered))
print(f'filtered_rows: {len(filtered)}')
"""
        request = {
            "suffix": ".csv",
            "data_base64": base64.b64encode(csv_data).decode("ascii"),
            "code": code,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            worker_path = Path(temp_dir) / "worker.py"
            worker_path.write_text(_SANDBOX_WORKER, encoding="utf-8")
            environment = os.environ.copy()
            environment["TEMP"] = temp_dir
            environment["TMP"] = temp_dir
            environment["TMPDIR"] = temp_dir
            result = subprocess.run(
                [sys.executable, str(worker_path)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=30,
                env=environment,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        response_line = next(
            line for line in reversed(result.stdout.splitlines())
            if line.startswith("__RUNNER_RESULT__")
        )
        payload = json.loads(response_line.removeprefix("__RUNNER_RESULT__"))
        self.assertEqual(payload["status"], "success", payload["error"])
        self.assertIn("filtered_rows: 2", payload["stdout"])
        self.assertEqual(payload["numeric_results"]["metrics.filtered_rows"], 2)

        changes = payload["data_changes"]
        operations = [change["operation"] for change in changes]
        self.assertTrue(any("dropna" in operation for operation in operations))
        self.assertTrue(any("boolean DataFrame selection" in operation for operation in operations))
        dropna_change = next(change for change in changes if "dropna" in change["operation"])
        self.assertEqual(dropna_change["before_rows"], 4)
        self.assertEqual(dropna_change["after_rows"], 3)
        self.assertEqual(dropna_change["missing_cells_filled"], 1)
        fill_change = next(change for change in changes if "fillna" in change["operation"])
        self.assertEqual(fill_change["changed_cells"], 1, fill_change)
        assignment_change = next(
            change for change in changes if "indexer" in change["operation"]
        )
        self.assertEqual(assignment_change["changed_cells"], 1)


if __name__ == "__main__":
    unittest.main()
