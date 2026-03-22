"""
Coding agent subprocess entry point.

Usage: python coding_agent_entry.py <context_json_path> <result_json_path>

Reads a context JSON, generates and executes code via LLM, writes a result JSON.
The agent has NO S3 access — it reads from input_dir and writes to output_dir only.
All FaaSr interactions (log, rank, invocation_id) go through the RPC server.
"""
import json
import os
import sys
import traceback
from pathlib import Path


def _get_safe_builtins() -> dict:
    safe_builtins = {
        # I/O
        "print": print,
        "input": input,
        "open": open,
        # Type constructors
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        "bytes": bytes,
        "bytearray": bytearray,
        "complex": complex,
        # Utilities
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "pow": pow,
        "divmod": divmod,
        "slice": slice,
        # Type checking
        "type": type,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "callable": callable,
        # String methods
        "chr": chr,
        "ord": ord,
        "format": format,
        "repr": repr,
        "ascii": ascii,
        "bin": bin,
        "hex": hex,
        "oct": oct,
        # Error handling
        "Exception": Exception,
        "RuntimeError": RuntimeError,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "OSError": OSError,
        "IOError": IOError,
        "FileNotFoundError": FileNotFoundError,
        # Iteration
        "iter": iter,
        "next": next,
        # Functional
        "all": all,
        "any": any,
        "hash": hash,
        "id": id,
        # Attribute access
        "getattr": getattr,
        "setattr": setattr,
        "hasattr": hasattr,
        "delattr": delattr,
        # Import
        "__import__": __import__,
    }
    return safe_builtins


def _build_system_prompt(context: dict) -> str:
    registry_entries = context.get("registry_entries", [])
    file_metadata = context.get("file_metadata", {})
    workflow_spec = context.get("workflow_spec", {})
    input_dir = context.get("input_dir", "/tmp/agent/input")
    output_dir = context.get("output_dir", "/tmp/agent/output")

    registry_summary = ""
    if registry_entries:
        lines = [f"  - {e.get('name', '?')}: {e.get('description', '')}" for e in registry_entries]
        registry_summary = "Available registry entries:\n" + "\n".join(lines)

    file_summary = ""
    if file_metadata:
        parts = []
        for uri, meta in file_metadata.items():
            local_path = meta.get("local_path", "")
            sidecar = meta.get("sidecar", {})
            sample = meta.get("sample", "")
            sidecar_str = json.dumps(sidecar, indent=2) if sidecar else "(no schema)"
            parts.append(
                f"File: {uri}\n  Local path: {local_path}\n"
                f"  Schema: {sidecar_str}\n  Sample:\n{sample}"
            )
        file_summary = "Input files:\n" + "\n\n".join(parts)

    available_packages = """\
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
Standard library modules (json, os, sys, csv, math, datetime, re, pathlib) are also available."""

    return f"""You are a FaaSr coding agent. You process data files and write results to disk.

CRITICAL OUTPUT RULES:
- Generate ONLY pure Python code — no markdown, no triple backticks, no ```python tags
- Start immediately with import statements or code, no pretext

CRITICAL RUNTIME RULES:
- DO NOT import or reference any 'faasr' module
- Use ONLY the provided functions (faasr_log, faasr_invocation_id, faasr_rank) for meta-context
- DO NOT perform any S3 operations — no faasr_put_file, no faasr_get_file
- Read inputs from: {input_dir}
- Write ALL outputs to: {output_dir} (JSON or image files: .png, .jpg, .jpeg)
- Use the input_dir and output_dir variables injected into the runtime
- Never hardcode run IDs or invocation IDs — use faasr_invocation_id() if needed
- If you need a package not listed below, call faasr_install("package_name") before importing it

AVAILABLE FUNCTIONS (injected into runtime, do not import):
- faasr_log(log_message): Append a message to the local log file (uploaded to S3 by the eval agent)
- faasr_invocation_id(): Returns the current invocation ID string
- faasr_rank(): Returns a dict with "rank" and "max_rank"
- faasr_install(package_name): Install a Python package at runtime via pip (call before importing)

{available_packages}

WORKFLOW CONTEXT:
{json.dumps(workflow_spec, indent=2)}

{registry_summary}

{file_summary}

Write Python code that reads inputs from input_dir, processes them, and writes JSON outputs to output_dir.
Handle errors gracefully. Log what you are doing with faasr_log."""


def main():
    if len(sys.argv) < 3:
        print("Usage: coding_agent_entry.py <context_json> <result_json>", file=sys.stderr)
        sys.exit(1)

    ctx_path = sys.argv[1]
    result_path = sys.argv[2]

    def write_result(success: bool, exception: str | None = None):
        with open(result_path, "w") as f:
            json.dump({"success": success, "exception": exception}, f)

    try:
        with open(ctx_path, "r") as f:
            context = json.load(f)
    except Exception as e:
        write_result(False, f"Could not read context: {e}")
        sys.exit(1)

    try:
        # Import agent helpers
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from FaaSr_py.helpers.agent_helper import AgentCodeGenerator, get_agent_api_key, get_agent_provider
    except Exception as e:
        write_result(False, f"Import error: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    try:
        api_key = get_agent_api_key()
        provider = get_agent_provider()
        if not provider:
            write_result(False, "Could not determine LLM provider. Set AGENT_KEY.")
            sys.exit(1)

        generator = AgentCodeGenerator(api_key, provider)
        system_prompt = _build_system_prompt(context)
        prompt = context.get("prompt", "")

        code = generator.generate_text(prompt, system_prompt, temperature=context.get("temperature", 0.2))

        # Clean up markdown formatting
        if "```python" in code:
            code = code.replace("```python\n", "").replace("\n```python", "")
        if "```" in code:
            if code.strip().startswith("```"):
                code = code[code.index("```") + 3:]
            if code.strip().endswith("```"):
                code = code[:code.rindex("```")]
        code = code.strip()

        if not code:
            write_result(False, "LLM returned empty code")
            sys.exit(1)

        # Save generated code to output dir for audit/aggregation
        output_dir = context.get("output_dir", "/tmp/agent/output")
        code_path = Path(output_dir) / "coding_agent_code.py"
        try:
            code_path.write_text(code)
            _faasr_log(f"Saved generated code to {code_path.name}")
        except Exception as e:
            _faasr_log(f"Warning: could not save generated code: {e}")

    except Exception as e:
        write_result(False, f"Code generation error: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    # Ensure output and log dirs exist
    output_dir = context.get("output_dir", "/tmp/agent/output")
    input_dir = context.get("input_dir", "/tmp/agent/input")
    logs_dir = context.get("logs_dir", "/tmp/agent/logs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    _invocation_id = context.get("invocation_id", "")
    _rank = context.get("rank", {})
    _log_path = Path(logs_dir) / "coding_agent.log"

    def _faasr_log(msg):
        with open(_log_path, "a") as _f:
            _f.write(str(msg) + "\n")

    def _faasr_install(package_name: str):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pip install {package_name!r} failed:\n{result.stderr}")
        _faasr_log(f"Installed package: {package_name}")

    # Build execution namespace
    namespace = {
        "__builtins__": _get_safe_builtins(),
        "faasr_log": _faasr_log,
        "faasr_install": _faasr_install,
        "faasr_invocation_id": lambda: _invocation_id,
        "faasr_rank": lambda: _rank,
        # Injected variables
        "input_dir": input_dir,
        "output_dir": output_dir,
        # Common stdlib imports
        "json": __import__("json"),
        "os": __import__("os"),
        "sys": __import__("sys"),
        "csv": __import__("csv"),
        "math": __import__("math"),
        "datetime": __import__("datetime"),
        "re": __import__("re"),
        "pathlib": __import__("pathlib"),
        "Path": Path,
    }

    try:
        exec(code, namespace)
        write_result(True)
    except Exception:
        write_result(False, traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
