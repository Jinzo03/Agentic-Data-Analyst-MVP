"""Smoke test for the project-level CodeExecutionRunner."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")

# Put the project root ahead of tests/ so `runner` resolves to ../runner.py,
# rather than the stale duplicate tests/runner.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from runner import CodeExecutionRunner  # noqa: E402


class CodeExecutionRunnerTest(unittest.TestCase):
    def test_runs_analysis_and_saves_chart(self):
        sandbox_check = object.__new__(CodeExecutionRunner)
        sandbox_check.docker_image = "agentic-data-analyst-sandbox:latest"
        sandbox_error = sandbox_check._check_docker()
        if sandbox_error:
            self.skipTest(sandbox_error)

        data = pd.DataFrame(
            {
                "category": ["A", "A", "A", "B", "B", "B"],
                "score": [12.4, 14.2, 13.1, 22.1, 24.5, 21.8],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "sample.csv"
            data.to_csv(csv_path, index=False)

            runner = CodeExecutionRunner(str(csv_path), output_dir=str(temp_path / "output"))
            try:
                result = runner.execute_python_code(
                    """
import matplotlib.pyplot as plt
import pandas as pd

df = con.execute('SELECT category, score FROM dataset').df()
group_a = df.loc[df['category'] == 'A', 'score']
group_b = df.loc[df['category'] == 'B', 'score']
print(f'rows: {len(df)}')
print(f'mean difference: {group_b.mean() - group_a.mean():.2f}')
df.boxplot(column='score', by='category')
plt.title('Scores by category')
plt.suptitle('')
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Example trend')
"""
                )
            finally:
                runner.con.close()

            self.assertEqual(result["status"], "success", result["error"])
            self.assertIsNone(result["error"])
            self.assertIn("rows: 6", result["stdout"])
            self.assertIn("mean difference: 9.57", result["stdout"])
            self.assertTrue(result["chart_path"])
            self.assertEqual(len(result["chart_paths"]), 2)
            self.assertEqual(result["chart_paths"][0], result["chart_path"])
            self.assertTrue(all(Path(path).is_file() for path in result["chart_paths"]))
            self.assertTrue(Path(result["chart_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
