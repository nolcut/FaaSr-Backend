import json


IO_SYSTEM_PROMPT = (
    "You are a file selection assistant. Choose files needed to complete the user task.\n"
    "Return ONLY valid JSON: {\"uris\": [<list of uri strings>], \"rationale\": \"<string>\"}\n"
    "Use the exact uri values from the registry entries.\n"
    "Do not include any text outside the JSON."
)

EVAL_SYSTEM_PROMPT = (
    "You are evaluating the output of a coding agent.\n"
    "Return ONLY valid JSON with these keys:\n"
    "  \"decision\": \"continue\"|\"loop_back\"|\"abort\"\n"
    "  \"reasoning\": \"<string>\"\n"
    "  \"file_descriptions\": {\"<filename>\": \"<natural language description of what the file contains>\"}\n"
    "Default to 'continue' unless there is a clear, objective reason not to.\n"
    "- continue: the agent produced output files and no unhandled exception occurred. "
    "Do NOT second-guess meta-context, such as date ranges.\n"
    "- loop_back: ONLY if the agent raised an unhandled exception, produced zero output files, "
    "or output deviates far from what is expected.\n"
    "- abort: ONLY if the exception is truly unrecoverable (e.g. missing credentials, "
    "invalid workflow config). Never abort for missing packages — use loop_back.\n"
    "IMPORTANT: The coding agent CAN install missing packages at runtime using faasr_install(). "
    "A ModuleNotFoundError is always loop_back, never abort.\n"
    "Provide a file_descriptions entry for every output file listed.\n"
    "Do not include any text outside the JSON."
)

AVAILABLE_PACKAGES = """\
AVAILABLE PYTHON PACKAGES (pre-installed, import directly):
- numpy
- scipy
- pandas
- matplotlib
- Pillow (import PIL)
- openai, anthropic
- langgraph
- requests
- PyYAML (import yaml)
- pydantic
Standard library modules (json, os, sys, csv, math, datetime, re, pathlib) are also available.

Commonly needed geo/science packages (NOT pre-installed — you MUST call faasr_install first):
- geopandas, shapely, rioxarray, pyproj, fiona, earthaccess, pystac-client"""


def build_coding_system_prompt(context: dict) -> str:
    registry_entries = context.get("registry_entries", [])
    file_metadata = context.get("file_metadata", {})
    input_dir = context.get("input_dir", "/tmp/agent/input")
    output_dir = context.get("output_dir", "/tmp/agent/output")
    eval_feedback = context.get("eval_feedback", "")
    exception = context.get("exception", "")
    loop_count = context.get("loop_count", 0)

    uri_descriptions = {e.get("file_uri", ""): e.get("description", "") for e in registry_entries}

    file_summary = ""
    if file_metadata:
        parts = []
        max_sample_chars = 2000
        max_schema_chars = 1000
        for uri, meta in file_metadata.items():
            local_path = meta.get("local_path", "")
            sidecar = meta.get("sidecar", {})
            sample = meta.get("sample", "")
            description = uri_descriptions.get(uri, "")
            sidecar_str = json.dumps(sidecar, indent=2) if sidecar else "(no schema)"

            if len(sidecar_str) > max_schema_chars:
                sidecar_str = sidecar_str[:max_schema_chars] + "\n... (schema truncated)"

            if len(sample) > max_sample_chars:
                sample = sample[:max_sample_chars] + f"\n... (truncated, full file at {local_path})"

            part = f"File: {uri}\n  Local path: {local_path}\n"
            if description:
                part += f"  Description: {description}\n"
            part += f"  Schema: {sidecar_str}\n  Sample:\n{sample}"
            parts.append(part)
        file_summary = "Input files:\n" + "\n\n".join(parts)

    retry_block = ""
    if loop_count > 0:
        retry_block = f"\n\nPREVIOUS ATTEMPT FAILED (attempt {loop_count}).\n"
        if exception:
            retry_block += f"Traceback:\n{exception}\n"
        if eval_feedback:
            retry_block += f"Evaluator feedback: {eval_feedback}\n"
        retry_block += (
            "You MUST fix the issue above. Do not repeat the same mistake.\n"
            "If the failure was a missing package, call faasr_install(\"package_name\") "
            "at the top of your code BEFORE any import that uses it.\n"
        )

    return f"""You are a FaaSr coding agent. You process data files and write results to disk.

CRITICAL OUTPUT RULES:
- Generate ONLY pure Python code — no markdown, no triple backticks, no ```python tags
- Start immediately with import statements or code, no pretext

PACKAGE INSTALLATION RULES (IMPORTANT):
For every package NOT in the pre-installed list below, you MUST call faasr_install() on its own
line immediately before the import. Do this unconditionally — even if you think the package
might already be installed. Never import a non-pre-installed package without calling
faasr_install() first. Example:

    faasr_install("geopandas")
    import geopandas as gpd
    faasr_install("rioxarray")
    import rioxarray

{retry_block}

CRITICAL RUNTIME RULES:
- DO NOT use 'faasr' as a variable name or import any 'faasr' module — 'faasr' does not exist in this environment
- Use ONLY the provided functions (faasr_log, faasr_invocation_id, faasr_rank) for meta-context
- DO NOT perform any S3 operations — no faasr_put_file, no faasr_get_file
- Read inputs from: {input_dir}
- Write ALL outputs to: {output_dir}
- Use the input_dir and output_dir variables injected into the runtime
- Never hardcode run IDs or invocation IDs — use faasr_invocation_id() if needed

AVAILABLE FUNCTIONS (injected into runtime, do not import):
- faasr_log(log_message): Append a message to the local log file (uploaded to S3 by the eval agent)
- faasr_invocation_id(): Returns the current invocation ID string
- faasr_rank(): Returns a dict with "rank" and "max_rank"
- faasr_install(package_name): Install a Python package at runtime via pip (call before importing)

{AVAILABLE_PACKAGES}

{file_summary}

Write Python code that reads from input_dir, processes the data, and writes outputs to output_dir.
Log key steps with faasr_log."""
