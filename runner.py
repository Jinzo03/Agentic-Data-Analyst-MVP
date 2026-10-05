"""Execute generated analyses in an isolated Docker container."""

import base64
import json
import re
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

import duckdb


SANDBOX_IMAGE = "agentic-data-analyst-sandbox:latest"
MAX_INPUT_BYTES = 100 * 1024 * 1024
MAX_CODE_BYTES = 1 * 1024 * 1024
MAX_RUNTIME_SECONDS = 120
MAX_CAPTURED_OUTPUT_BYTES = 32 * 1024 * 1024

_SANDBOX_WORKER = r'''
import base64
import ast
import contextlib
import duckdb
import io
import json
import math
import numbers
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

payload = json.load(sys.stdin)
csv_suffix = payload.get("suffix", ".csv")
data_path = Path(tempfile.gettempdir()) / ("uploaded_data" + csv_suffix)
data_path.write_bytes(base64.b64decode(payload["data_base64"]))
output_dir = Path(tempfile.gettempdir()) / "analysis_output"
output_dir.mkdir(exist_ok=True)
class CappedStringIO(io.StringIO):
    def __init__(self, limit=4 * 1024 * 1024):
        super().__init__()
        self.limit = limit
        self.truncated = False

    def write(self, text):
        remaining = self.limit - self.tell()
        if remaining > 0:
            super().write(text[:remaining])
        if len(text) > remaining:
            self.truncated = True
        return len(text)

    def getvalue(self):
        value = super().getvalue()
        if self.truncated:
            value += "\n[stdout truncated at 4 MiB]"
        return value

stdout_capture = CappedStringIO()
metrics = {"significance_threshold": 0.05}
data_changes = []
MAX_CELL_COMPARISON = 250000

def dataframe_snapshot(frame):
    if not isinstance(frame, pd.DataFrame):
        return None
    snapshot = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_cells": int(frame.isna().sum().sum()),
        "index": frame.index.copy(),
        "columns_index": frame.columns.copy(),
        "values": None,
    }
    if frame.size <= MAX_CELL_COMPARISON:
        snapshot["values"] = frame.copy(deep=True)
    return snapshot

def record_data_change(operation, before, after):
    if before is None or not isinstance(after, pd.DataFrame):
        return
    before_rows = before["rows"]
    after_rows = len(after)
    before_missing = before["missing_cells"]
    after_missing = int(after.isna().sum().sum())
    entry = {
        "operation": operation,
        "before_rows": before_rows,
        "after_rows": after_rows,
        "rows_removed": max(0, before_rows - after_rows),
        "rows_added": max(0, after_rows - before_rows),
        "before_columns": before["columns"],
        "after_columns": len(after.columns),
        "missing_cells_before": before_missing,
        "missing_cells_after": after_missing,
        "missing_cells_filled": max(0, before_missing - after_missing),
        "changed_cells": None,
        "value_comparison": "not measured; frame exceeded comparison limit",
    }
    previous = before["values"]
    if previous is not None:
        try:
            shared_columns = previous.columns.intersection(after.columns)
            common_index = previous.index.intersection(after.index)
            if not previous.index.is_unique or not after.index.is_unique:
                raise ValueError("duplicate index labels")
            changed_cells = 0
            for column in shared_columns:
                for index_value in common_index:
                    old_value = previous.at[index_value, column]
                    new_value = after.at[index_value, column]
                    old_missing = bool(pd.isna(old_value))
                    new_missing = bool(pd.isna(new_value))
                    if old_missing != new_missing:
                        changed_cells += 1
                    elif not old_missing:
                        try:
                            changed_cells += bool(old_value != new_value)
                        except (TypeError, ValueError):
                            changed_cells += repr(old_value) != repr(new_value)
            entry["changed_cells"] = int(changed_cells)
            entry["value_comparison"] = "measured for rows and columns present before and after"
        except (ValueError, TypeError, KeyError) as exc:
            entry["value_comparison"] = (
                "could not compare values because index/column alignment was ambiguous: "
                f"{type(exc).__name__}"
            )
    changed = (
        entry["rows_removed"] or entry["rows_added"]
        or entry["missing_cells_filled"] or entry["changed_cells"]
        or entry["before_columns"] != entry["after_columns"]
    )
    if changed:
        data_changes.append(entry)

def track_data_operation(operation, source, result):
    before = dataframe_snapshot(source)
    record_data_change(operation, before, result)
    return result

def track_inplace_operation(operation, source, action):
    before = dataframe_snapshot(source)
    action()
    record_data_change(operation, before, source)

def track_index_assignment(operation, source, indexer, key, value):
    before = dataframe_snapshot(source)
    indexer[key] = value
    record_data_change(operation, before, source)

class DataChangeInstrumenter(ast.NodeTransformer):
    METHOD_LABELS = {
        "dropna": "null removal (dropna)",
        "fillna": "null imputation (fillna)",
        "interpolate": "null imputation (interpolate)",
        "query": "row filter (query)",
        "where": "row filter (where)",
        "drop": "row/column removal (drop)",
        "drop_duplicates": "duplicate-row removal (drop_duplicates)",
        "clip": "value clipping / possible outlier handling (clip)",
        "replace": "value replacement (replace)",
    }

    def operation(self, expression):
        for node in ast.walk(expression):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in self.METHOD_LABELS:
                    source = node.func.value
                    while isinstance(source, (ast.Attribute, ast.Subscript)):
                        source = source.value if isinstance(source, ast.Attribute) else source.value
                    if isinstance(source, ast.Name):
                        inplace = any(
                            keyword.arg == "inplace" and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True
                            for keyword in node.keywords
                        )
                        return source.id, self.METHOD_LABELS[node.func.attr], inplace
            if isinstance(node, ast.Subscript):
                index = node.slice
                is_row_filter = isinstance(index, (ast.Compare, ast.BoolOp)) or (
                    isinstance(index, ast.Call) and isinstance(index.func, ast.Attribute)
                ) or isinstance(index, ast.Name) or (
                    isinstance(node.value, ast.Attribute)
                    and node.value.attr in {"loc", "iloc"}
                )
                source = node.value
                while isinstance(source, (ast.Attribute, ast.Subscript)):
                    source = source.value
                if is_row_filter and isinstance(source, ast.Name):
                    label = "row filter (boolean DataFrame selection)"
                    return source.id, label, False
        return None

    def visit_Assign(self, node):
        expression = node.value
        found = self.operation(expression)
        node = self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
            target = node.targets[0]
            indexer = target.value
            source = indexer
            if isinstance(indexer, ast.Attribute) and indexer.attr in {"loc", "iloc"}:
                source = indexer.value
            if isinstance(source, ast.Name):
                operation = "value assignment (DataFrame indexer)"
                return ast.copy_location(ast.Expr(value=ast.Call(
                    func=ast.Name(id="track_index_assignment", ctx=ast.Load()),
                    args=[ast.Constant(value=operation), source, indexer, target.slice, node.value],
                    keywords=[],
                )), node)
        if found and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            source_name, label, inplace = found
            if not inplace:
                node.value = ast.Call(
                    func=ast.Name(id="track_data_operation", ctx=ast.Load()),
                    args=[ast.Constant(value=label), ast.Name(id=source_name, ctx=ast.Load()), node.value],
                    keywords=[],
                )
        return node

    def visit_Expr(self, node):
        expression = node.value
        found = self.operation(expression)
        node = self.generic_visit(node)
        if found and found[2]:
            source_name, label, _ = found
            node.value = ast.Call(
                func=ast.Name(id="track_inplace_operation", ctx=ast.Load()),
                args=[
                    ast.Constant(value=label),
                    ast.Name(id=source_name, ctx=ast.Load()),
                    ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=node.value),
                ],
                keywords=[],
            )
        return node

def instrument_data_changes(code):
    tree = ast.parse(code)
    tree = DataChangeInstrumenter().visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, "<generated analysis>", "exec")

connection = duckdb.connect(database=":memory:")

try:
    if csv_suffix.lower() == ".parquet":
        connection.execute(
            "CREATE TABLE dataset AS SELECT * FROM read_parquet(?)",
            [str(data_path)],
        )
    else:
        connection.execute(
            "CREATE TABLE dataset AS SELECT * FROM read_csv_auto(?)",
            [str(data_path)],
        )
    dataset_rows = connection.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]

    namespace = {
        "con": connection,
        "pd": pd,
        "np": np,
        "duckdb": duckdb,
        "plt": plt,
        "output_dir": str(output_dir),
        "metrics": metrics,
        "track_data_operation": track_data_operation,
        "track_inplace_operation": track_inplace_operation,
        "track_index_assignment": track_index_assignment,
    }
    with contextlib.redirect_stdout(stdout_capture):
        exec(instrument_data_changes(payload["code"]), namespace)

    numeric_results = {
        "dataset_rows": int(dataset_rows),
        "significance_threshold": 0.05,
    }

    def add_numeric(key, value):
        if isinstance(value, (bool, np.bool_)):
            return
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, numbers.Real) and math.isfinite(float(value)):
            numeric_results[str(key)] = value

    def flatten_metrics(prefix, value):
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                flatten_metrics(f"{prefix}.{child_key}" if prefix else child_key, child_value)
        elif isinstance(value, (list, tuple)):
            for index, child_value in enumerate(value):
                flatten_metrics(f"{prefix}[{index}]", child_value)
        else:
            add_numeric(prefix, value)

    explicit_metrics = namespace.get("metrics")
    if isinstance(explicit_metrics, dict):
        flatten_metrics("metrics", explicit_metrics)

    for name, value in namespace.items():
        if not name.startswith("__") and name != "metrics":
            add_numeric(name, value)

    number = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    for line in stdout_capture.getvalue().splitlines():
        match = re.match(rf"^\s*([^:=\n]{{1,100}})\s*[:=]\s*({number})\s*%?\s*$", line)
        if match:
            label = "stdout." + match.group(1).strip()
            add_numeric(label, float(match.group(2).replace(",", "")))

    for index, change in enumerate(data_changes, start=1):
        for field in (
            "before_rows", "after_rows", "rows_removed", "rows_added",
            "before_columns", "after_columns", "missing_cells_before",
            "missing_cells_after", "missing_cells_filled", "changed_cells",
        ):
            if change[field] is not None:
                add_numeric(f"data_change_{index}.{field}", change[field])

    chart_data = []
    total_chart_bytes = 0
    for index, figure_number in enumerate(plt.get_fignums(), start=1):
        if index > 10:
            break
        figure = plt.figure(figure_number)
        image_buffer = io.BytesIO()
        figure.savefig(image_buffer, format="png", bbox_inches="tight", dpi=160)
        image_bytes = image_buffer.getvalue()
        if (
            len(image_bytes) > 5 * 1024 * 1024
            or total_chart_bytes + len(image_bytes) > 15 * 1024 * 1024
        ):
            continue
        total_chart_bytes += len(image_bytes)
        chart_data.append({
            "name": f"generated_chart_{index}.png",
            "data_base64": base64.b64encode(image_bytes).decode("ascii"),
        })

    print("__RUNNER_RESULT__" + json.dumps({
        "status": "success",
        "stdout": stdout_capture.getvalue().strip(),
        "numeric_results": numeric_results,
        "data_changes": data_changes,
        "charts": chart_data,
        "error": None,
    }, allow_nan=False))
except Exception as exc:
    plt.close("all")
    print("__RUNNER_RESULT__" + json.dumps({
        "status": "error",
        "stdout": stdout_capture.getvalue().strip(),
        "numeric_results": {},
        "data_changes": data_changes,
        "charts": [],
        "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
    }))
finally:
    connection.close()
'''


class CodeExecutionRunner:
    """Run generated Python in a fail-closed, network-disabled container."""

    def __init__(
        self,
        data_path: str,
        output_dir: str = "./output",
        *,
        docker_image: str = SANDBOX_IMAGE,
        timeout: int = MAX_RUNTIME_SECONDS,
    ) -> None:
        self.data_path = str(Path(data_path).resolve())
        self.output_dir = str(Path(output_dir).resolve())
        self.docker_image = docker_image
        self.timeout = timeout
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Retained for existing callers that inspect or close runner.con.
        self.con = duckdb.connect(database=":memory:")
        suffix = Path(self.data_path).suffix.lower()
        if suffix == ".parquet":
            self.con.execute(
                "CREATE TABLE dataset AS SELECT * FROM read_parquet(?)",
                [self.data_path],
            )
        else:
            self.con.execute(
                "CREATE TABLE dataset AS SELECT * FROM read_csv_auto(?)",
                [self.data_path],
            )

    @staticmethod
    def _error_result(message: str, *, recoverable: bool = False) -> dict[str, Any]:
        return {
            "status": "error",
            "stdout": "",
            "numeric_results": {},
            "chart_path": None,
            "chart_paths": [],
            "error": message,
            "recoverable": recoverable,
        }

    def _check_docker(self) -> str | None:
        try:
            info = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return f"Docker is not available: {exc}"
        if info.returncode != 0:
            return (
                "Docker daemon is unavailable. Start Docker Desktop to run the "
                f"isolated analysis. Details: {info.stderr.strip()}"
            )

        try:
            image = subprocess.run(
                ["docker", "image", "inspect", self.docker_image],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return f"Docker image check timed out: {exc}"
        if image.returncode != 0:
            return (
                f"Sandbox image {self.docker_image!r} is missing. Build it from "
                "the project folder with: docker build -f Dockerfile.sandbox "
                f"-t {self.docker_image} ."
            )
        return None

    def execute_python_code(self, code_str: str) -> dict[str, Any]:
        """Execute code in a container with no network or host file mounts."""
        if len(code_str.encode("utf-8")) > MAX_CODE_BYTES:
            return self._error_result(
                f"Generated code exceeds the {MAX_CODE_BYTES // (1024 * 1024)} MiB limit."
            )

        data_path = Path(self.data_path)
        try:
            data_bytes = data_path.read_bytes()
        except OSError as exc:
            return self._error_result(f"Could not read uploaded data: {exc}")
        if len(data_bytes) > MAX_INPUT_BYTES:
            return self._error_result(
                f"Input data exceeds the {MAX_INPUT_BYTES // (1024 * 1024)} MiB limit."
            )

        docker_problem = self._check_docker()
        if docker_problem:
            return self._error_result(docker_problem)

        container_name = f"agent-analysis-{uuid.uuid4().hex[:12]}"
        payload = json.dumps(
            {
                "code": code_str,
                "suffix": data_path.suffix.lower() or ".csv",
                "data_base64": base64.b64encode(data_bytes).decode("ascii"),
            }
        )
        command = [
            "docker", "run", "--rm", "--interactive",
            "--name", container_name,
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--memory", "1g",
            "--cpus", "1",
            "--pids-limit", "64",
            "--user", "65534:65534",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m,mode=1777",
            "--env", "HOME=/tmp",
            "--env", "MPLCONFIGDIR=/tmp/matplotlib",
            "--env", "MPLBACKEND=Agg",
            "--env", "OPENBLAS_NUM_THREADS=1",
            "--env", "OMP_NUM_THREADS=1",
            self.docker_image,
            "python", "-I", "-c", _SANDBOX_WORKER,
        ]

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            return self._error_result(f"Could not start the Docker sandbox: {exc}")

        output_state: dict[str, bytes] = {}

        def drain_output(name: str, pipe: Any, limit: int) -> None:
            captured = bytearray()
            while True:
                chunk = pipe.read(64 * 1024)
                if not chunk:
                    break
                captured.extend(chunk)
                if len(captured) > limit:
                    del captured[: len(captured) - limit]
            output_state[name] = bytes(captured)

        stdout_thread = threading.Thread(
            target=drain_output,
            args=("stdout", process.stdout, MAX_CAPTURED_OUTPUT_BYTES),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=drain_output,
            args=("stderr", process.stderr, 2 * 1024 * 1024),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            try:
                process.stdin.write(payload.encode("utf-8"))
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            return_code = process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            stdout_thread.join()
            stderr_thread.join()
            subprocess.run(
                ["docker", "kill", container_name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return self._error_result(
                f"Generated analysis exceeded the {self.timeout}-second time limit.",
                recoverable=True,
            )
        finally:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            stdout_thread.join()
            stderr_thread.join()

        result_stdout = output_state.get("stdout", b"").decode("utf-8", errors="replace")
        result_stderr = output_state.get("stderr", b"").decode("utf-8", errors="replace")

        if return_code != 0:
            detail = result_stderr.strip() or result_stdout.strip()
            return self._error_result(
                "The isolated sandbox failed before returning an execution result. "
                f"Docker output: {detail}"
            )

        try:
            response_line = next(
                line for line in reversed(result_stdout.splitlines())
                if line.startswith("__RUNNER_RESULT__")
            )
            payload_result = json.loads(response_line.removeprefix("__RUNNER_RESULT__"))
        except (json.JSONDecodeError, StopIteration):
            return self._error_result(
                "The isolated sandbox returned invalid output. "
                f"Details: {result_stderr.strip() or result_stdout[-2000:]}"
            )

        if payload_result.get("status") != "success":
            return self._error_result(
                payload_result.get("error") or "Generated code failed without an error message.",
                recoverable=True,
            ) | {
                "stdout": payload_result.get("stdout", ""),
                "data_changes": payload_result.get("data_changes", []),
            }

        chart_paths = []
        charts_dir = Path(self.output_dir) / f"run_{uuid.uuid4().hex}"
        try:
            for chart in payload_result.get("charts", []):
                name = chart.get("name", "")
                if not re.fullmatch(r"generated_chart_\d+\.png", name):
                    continue
                chart_bytes = base64.b64decode(
                    chart.get("data_base64", ""),
                    validate=True,
                )
                if len(chart_bytes) > 5 * 1024 * 1024:
                    continue
                charts_dir.mkdir(parents=True, exist_ok=True)
                chart_path = charts_dir / name
                chart_path.write_bytes(chart_bytes)
                chart_paths.append(str(chart_path))
        except (ValueError, OSError) as exc:
            return self._error_result(
                f"Could not save sandbox chart output: {exc}",
                recoverable=False,
            )

        return {
            "status": "success",
            "stdout": payload_result.get("stdout", ""),
            "numeric_results": payload_result.get("numeric_results", {}),
            "data_changes": payload_result.get("data_changes", []),
            "chart_path": chart_paths[0] if chart_paths else None,
            "chart_paths": chart_paths,
            "error": None,
            "recoverable": True,
        }
