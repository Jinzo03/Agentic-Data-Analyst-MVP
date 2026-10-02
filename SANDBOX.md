# Isolated code execution

Generated analysis code runs in a Docker container. The Streamlit process fails
closed if Docker is stopped or the sandbox image is missing; it never falls back
to executing generated code in the app process.

## First-time setup (PowerShell)

1. Install and start Docker Desktop with Linux containers enabled.
2. From the project folder, build the analysis image:

   ```powershell
   docker build -f Dockerfile.sandbox -t agentic-data-analyst-sandbox:latest .
   ```

3. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

Rebuild the image after changing the sandbox's installed packages in
`Dockerfile.sandbox`.

## Container limits

Each run has no network, no host file mounts, a read-only root filesystem, a
non-root user, no Linux capabilities, a 1 GiB memory limit, one CPU, a 64-process
limit, and a 120-second timeout. Uploaded data and generated charts are passed
between the host and container through standard input and output. Only chart
images returned by the worker are written to the project's output folder.

The worker includes DuckDB, Matplotlib, NumPy, Pandas, SciPy, and Seaborn. It
does not receive the Streamlit process's environment variables, including the
Gemini API key.
