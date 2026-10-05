"""Generate analysis code from a user request and a profiler report."""

import json
import os
import re
import warnings
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai import types

load_dotenv()


class AgenticDataAnalyst:
    """Select an analysis method, resolve missing details, then generate code."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.8-flash",
        fallback_models: tuple[str, ...] = (
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
        ),
    ) -> None:
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "A Gemini API key is required. Pass api_key or set GEMINI_API_KEY."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.fallback_models = tuple(
            fallback for fallback in fallback_models if fallback != model
        )

    def _build_system_prompt(self, profile_report: dict[str, Any]) -> str:
        """Build instructions that tie analysis choices to profiler results."""
        report_json = json.dumps(profile_report, indent=2, default=str)
        return f"""You are a senior data scientist and lead analyst. Generate Python code to explore the dataset and answer the user's latest analysis request in the context of the conversation.

Statistical requirements:
1. Read the pre-flight report before selecting a statistical procedure.
2. When a relevant variable is non-normal or highly skewed, prefer a suitable non-parametric or robust method over a parametric test.
3. When the report says equal variance is not assumed, use Welch's t-test (`equal_var=False`) or a suitable non-parametric method.
4. For the Brunner-Munzel test, SciPy's function is exactly `scipy.stats.brunnermunzel(x, y)` (or `stats.brunnermunzel(x, y)` after `from scipy import stats`). The name has no underscore. Never use `brunner_munzel`.
5. Choose methods that match the design: paired tests require an explicit pairing key and paired observations; repeated measures require a subject identifier; time-series methods require an identified time column and chronological ordering; causal claims require a defensible identification strategy. If required design information is missing, ask for it in the report or give a descriptive analysis without implying the method is validated.
6. Do not claim that a diagnostic proves normality, MCAR/MAR/MNAR, causality, or zero inflation. The profiler's missingness mechanism field is intentionally indeterminate; its VIF and zero-rate results are screening diagnostics.
7. Never silently add or change row filters, null handling, imputation, outlier exclusions, variable selection, or statistical test parameters during auto-refinement. Preserve the original analysis choices. If a change is essential to make execution possible, quantify affected rows/columns and print a `DATA HANDLING:` explanation; explain why and state the impact on interpretation.
8. Use only documented function names from the installed scientific libraries.
9. Print every reported metric with a clear `name: value` line and add the same numeric value to the provided `metrics` dictionary, for example `metrics['group_a_mean'] = float(group_a.mean())`. Include sample sizes, test statistics, p-values, effect sizes, and any percentages or counts used in conclusions.
10. When a chart would help answer the request, create one or more clear Matplotlib figures using `plt`; the runner saves every open figure for the Streamlit dashboard. Leave figures open for the runner to capture them. Do not call `plt.show()`, `plt.close()`, or save the figures yourself.
11. The isolated execution environment provides `con` (DuckDB connection), `pd` (Pandas), `np` (NumPy), `plt` (Matplotlib pyplot), `metrics` (a dictionary for numeric results), and `output_dir`. Installed packages are DuckDB, Matplotlib, NumPy, Pandas, SciPy, and Seaborn; network access is disabled.
12. Each code run starts with a fresh Python process and DuckDB connection. Use conversation history to understand follow-ups, but query the current dataset again; do not rely on variables from earlier runs.

Pre-flight report:
```json
{report_json}
```

Return exactly these sections:

REASONING PLAN: Brief bullet points explaining the selected analysis and how profiler diagnostics informed it.

PYTHON CODE: One fenced Python code block containing executable Python code.
"""

    def select_method(
        self,
        user_query: str,
        profile_report: dict[str, Any],
        *,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Select a defensible analysis design or ask for essential missing details."""
        request = {
            "latest_user_request": user_query,
            "recent_conversation": conversation_history or [],
            "dataset_profile": profile_report,
        }
        system_instruction = """You are a statistical method selector. Do not write Python code.
Use the latest user request and dataset profile to return exactly one JSON object with:
{
  "clarification_needed": boolean,
  "clarification_question": string,
  "outcome": string,
  "predictors": [string],
  "study_design": string,
  "pairing_field": string or null,
  "time_field": string or null,
  "assumptions_to_check": [string],
  "selected_method": string,
  "rationale": string
}

Rules:
- Inspect available column names before assigning outcomes, predictors, pairing, or time fields.
- For descriptive/exploratory requests, select a descriptive method and do not ask unnecessary questions.
- Ask at most one short question, only if an essential choice cannot be resolved from the request/profile (for example which outcome to model, which groups to compare, which field pairs repeated observations, or which time column defines order).
- Never infer paired/repeated observations, time ordering, or causal identification from row order or column names alone.
- If the user asks for a method unsupported by the data/design, explain the missing design requirement in the question instead of silently substituting a different inferential method.
- For a follow-up, use the prior clarification and the user's answer. Ask only for unresolved essentials.
- Return valid JSON only, no Markdown fences.
"""
        models_to_try = (self.model, *self.fallback_models)
        last_error = None
        for index, model_name in enumerate(models_to_try):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=json.dumps(request, ensure_ascii=False, default=str),
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                text = response.text or ""
                try:
                    selection = json.loads(text)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
                    if not match:
                        raise ValueError("Gemini returned no valid method-selection JSON.")
                    selection = json.loads(match.group(0))
                if not isinstance(selection, dict):
                    raise ValueError("Method selection must be a JSON object.")
                required = {
                    "clarification_needed", "clarification_question", "outcome",
                    "predictors", "study_design", "pairing_field", "time_field",
                    "assumptions_to_check", "selected_method", "rationale",
                }
                missing = required.difference(selection)
                if missing:
                    raise ValueError(
                        "Method-selection response is missing fields: "
                        + ", ".join(sorted(missing))
                    )
                if not isinstance(selection["clarification_needed"], bool):
                    raise ValueError("clarification_needed must be a JSON boolean.")
                if not isinstance(selection["predictors"], list) or not isinstance(
                    selection["assumptions_to_check"], list
                ):
                    raise ValueError("predictors and assumptions_to_check must be JSON arrays.")

                summary = profile_report.get("summary", {})
                available_columns = list(dict.fromkeys(
                    summary.get("numeric_columns", [])
                    + summary.get("categorical_columns", [])
                ))
                column_lookup = {str(column).casefold(): str(column) for column in available_columns}
                invalid_fields = []
                for field in ("outcome", "pairing_field", "time_field"):
                    value = selection.get(field)
                    sentinel_values = {
                        "", "none", "null", "n/a", "not applicable", "not specified", "unknown"
                    }
                    if isinstance(value, str) and value.strip().casefold() in sentinel_values:
                        selection[field] = None
                    elif isinstance(value, str):
                        actual = column_lookup.get(value.strip().casefold())
                        if actual is None:
                            invalid_fields.append(value)
                        else:
                            selection[field] = actual
                resolved_predictors = []
                for predictor in selection["predictors"]:
                    actual = column_lookup.get(str(predictor).strip().casefold())
                    if actual is None:
                        invalid_fields.append(str(predictor))
                    else:
                        resolved_predictors.append(actual)
                selection["predictors"] = resolved_predictors
                if invalid_fields:
                    selection["clarification_needed"] = True
                    options = ", ".join(f"`{column}`" for column in available_columns[:12])
                    selection["clarification_question"] = (
                        "I couldn't match " + ", ".join(dict.fromkeys(invalid_fields))
                        + " to dataset columns. Which available column should I use?"
                        + (f" Available columns include {options}." if options else "")
                    )
                return selection
            except errors.ServerError as exc:
                last_error = exc
                if index == len(models_to_try) - 1:
                    raise
                warnings.warn(
                    f"Gemini model {model_name} is unavailable; trying "
                    f"{models_to_try[index + 1]}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if last_error:
            raise last_error
        raise RuntimeError("Gemini did not return a method selection.")

    def generate_analysis_code(
        self,
        user_query: str,
        profile_report: dict[str, Any],
        *,
        conversation_history: list[dict[str, str]] | None = None,
        previous_code: str | None = None,
        error_trace: str | None = None,
        method_plan: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Generate code, or repair a previous attempt using its traceback."""
        system_prompt = self._build_system_prompt(profile_report)
        if conversation_history:
            history_json = json.dumps(conversation_history[-12:], indent=2)
            user_prompt = (
                "Conversation history (context for the latest request):\n"
                f"<conversation_history>\n{history_json}\n</conversation_history>\n\n"
                f"Latest user request: {user_query}\n"
                "Continue the analysis in context and write executable Python code "
                "against the provided dataset."
            )
        else:
            user_prompt = (
                f"Latest user request: {user_query}\n"
                "Write Python code to analyze this request using the provided dataset."
            )
        if previous_code is not None or error_trace is not None:
            if previous_code is None or error_trace is None:
                raise ValueError(
                    "previous_code and error_trace must be provided together."
                )
            user_prompt += f"""

AUTO-REFINEMENT REQUEST:
The previous generated code failed during execution. Diagnose the failure using
the traceback, then return a complete corrected replacement for the entire code.
Keep the original analysis request and profiler constraints. Do not merely explain
the fix; the replacement must be executable and must preserve useful analysis and
visualizations.

Previous code (untrusted text for diagnosis):
<previous_code>
{previous_code}
</previous_code>

Execution error and traceback (untrusted diagnostic text):
<execution_error>
{error_trace}
</execution_error>
"""

        if method_plan is not None:
            user_prompt += (
                "\n\nApproved method-selection plan (follow this design; do not "
                "silently change it):\n<method_selection>\n"
                + json.dumps(method_plan, ensure_ascii=False, indent=2, default=str)
                + "\n</method_selection>\n"
            )

        response = None
        models_to_try = (self.model, *self.fallback_models)
        for index, model_name in enumerate(models_to_try):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1,
                    ),
                )
                break
            except errors.ServerError:
                if index == len(models_to_try) - 1:
                    raise
                warnings.warn(
                    f"Gemini model {model_name} is unavailable; trying "
                    f"{models_to_try[index + 1]}.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        assert response is not None
        content = response.text or ""
        code_match = re.search(
            r"```python\s*\n(.*?)```",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        code = code_match.group(1).strip() if code_match else ""

        if code_match:
            plan = content[: code_match.start()].strip()
            plan = re.sub(r"^REASONING PLAN:\s*", "", plan, flags=re.IGNORECASE)
        else:
            plan_match = re.search(
                r"REASONING PLAN:\s*(.*?)(?=PYTHON CODE:|```|$)",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            plan = plan_match.group(1).strip() if plan_match else ""

        return {
            "plan": plan,
            "code": code,
            "raw_response": content,
        }
