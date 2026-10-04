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
    """Ask Gemini to plan and produce executable analysis code."""

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

    def generate_analysis_code(
        self,
        user_query: str,
        profile_report: dict[str, Any],
        *,
        conversation_history: list[dict[str, str]] | None = None,
        previous_code: str | None = None,
        error_trace: str | None = None,
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
