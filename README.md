# Agentic Data Analyst

A conversational data analysis MVP built with Streamlit, Gemini, DuckDB, and Python’s scientific stack. Upload a CSV or Parquet dataset, ask a question in plain language, and explore the result through follow-up questions in the same chat.

> **Project status:** Work in progress. Features and documentation will change as the project develops.

## Video demo

<!-- Add a short walkthrough link or embed here when available. -->

_Short video demo coming soon._

## What it does

- Profiles an uploaded dataset and shows a preview and profile summary.
- Uses Gemini to select an analysis method, ask for missing study-design details, and generate analysis code.
- Runs generated code in a restricted Docker container.
- Can send execution errors back to Gemini for an automatic code repair, up to the configured maximum.
- Produces a Markdown report and checks its numeric and significance claims against labeled runner results.
- Keeps chat context for follow-up questions and lets users retry an answer and navigate between answer versions.
- Displays generated charts and measured data changes in the dashboard.
- Saves a reproducibility record for analysis turns under `output/runs/`.

The verification layer is a guardrail, not a proof that an analysis is correct. Review the method, assumptions, report, and audit details before relying on a result.

## Requirements

- Windows with PowerShell, or another environment that can run Python and Docker.
- Python 3.10 or later for the Streamlit app.
- Docker Desktop running with Linux containers enabled.
- A Gemini API key available to the app process.

Generated analysis code is run in Docker. The app does not fall back to executing generated code in its own process if Docker is unavailable. See [SANDBOX.md](SANDBOX.md) for container setup and limits.

## Setup on Windows

Open PowerShell in the project folder and create a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project folder and add your Gemini key:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
```

Keep `.env` private. It is excluded by `.gitignore`; do not paste the key into source files or commit it.

Build the analysis sandbox image once from the project folder:

```powershell
docker build -f Dockerfile.sandbox -t agentic-data-analyst-sandbox:latest .
```

Start the dashboard:

```powershell
streamlit run app.py
```

Streamlit prints a local URL in the terminal. Open it in a browser, upload a CSV or Parquet file, and ask a question about the data.

## Using the dashboard

1. Upload a CSV or Parquet file in the sidebar.
2. Review the dataset preview and profile.
3. Ask a question in the chat. If method selection needs an essential detail, answer its clarification question.
4. Continue with follow-up questions, or use **Try again** to request another answer. Use the answer navigation controls to view earlier versions.
5. Review charts, verification status, settings, and analysis details below the answer.

The sidebar controls the maximum number of automatic code repairs and the significance threshold (`alpha`).

## Tests

With the virtual environment activated and project requirements installed, run the mocked regression tests:

```powershell
python -m unittest discover -s tests -v
```

The unit tests mock Gemini responses and do not need a Gemini API key. Runner and profiling tests may require the project dependencies; tests that exercise isolated code execution also require Docker.

`test_pipeline.py` is a manual end-to-end example. It makes live Gemini calls and requires a working API key, Docker, and the sandbox image.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard and analysis-turn orchestration |
| `agent.py` | Gemini method selection and code generation |
| `profiler.py` | Dataset profiling and preflight diagnostics |
| `runner.py` | Isolated Docker execution and result collection |
| `verifier.py` | Report generation and numeric/significance checks |
| `chat_flow.py` | Chat history, retries, and answer-version state |
| `tests/` | Mocked regression tests and runner tests |
| `SANDBOX.md` | Sandbox setup and isolation details |

## Configuration and artifacts

- `GEMINI_API_KEY` is loaded from `.env` for local app runs.
- `MAX_CODE_REFINEMENTS` is currently a dashboard setting with a default in `app.py`.
- `alpha` is selected in the sidebar and passed through profiling, execution, and verification.
- Charts and run records are written under `output/`; this generated directory is excluded from version control.

## Current scope

The agent chooses methods based on the request, conversation, and available profile information, but it cannot infer missing study-design details reliably. Complex methods may need a clear outcome, predictor, pairing or time field, and design information. Inspect generated code and run details when the method or data handling matters to your conclusions.

This README is an initial project guide and will be updated as capabilities, setup, and deployment evolve.
