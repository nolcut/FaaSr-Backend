"""
Coding agent subprocess entry point.

Usage: python coding_agent_entry.py <context_json_path> <result_json_path>

Reads a context JSON, runs an agentic exploration loop via ClaudeSDKClient,
then executes the finalized function. Writes a result JSON when done.
The agent has NO S3 access — it reads from input_dir and writes to output_dir only.
"""
import asyncio
import io
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path


def _run_snippet(code: str, namespace: dict) -> str:
    """Exec code in shared exploration namespace, capture stdout/stderr, return string."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, namespace)
    except Exception:
        buf.write(traceback.format_exc())
    return buf.getvalue() or "(no output)"


def _download(url: str, filename: str, input_dir: str) -> str:
    """Download a URL to input_dir/filename, return a status string."""
    import requests
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        path = os.path.join(input_dir, filename)
        with open(path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return f"Downloaded to {path} ({os.path.getsize(path)} bytes)"
    except Exception as e:
        return f"Download failed: {e}"


def _build_mcp_server(explore_ns: dict, input_dir: str, state: dict):
    """Build an in-process MCP server with three agent tools."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool(
        "execute_code",
        "Run Python code to explore data. Variables persist between calls. "
        "input_dir is available. output_dir is NOT. Use print() to see values.",
        {"code": str},
    )
    async def execute_code(args: dict) -> dict:
        output = _run_snippet(args["code"], explore_ns)
        return {"content": [{"type": "text", "text": output}]}

    @tool(
        "download_dataset",
        "Download a file from a URL into input_dir. Returns the local path.",
        {"url": str, "filename": str},
    )
    async def download_dataset(args: dict) -> dict:
        result = _download(args["url"], args["filename"], input_dir)
        return {"content": [{"type": "text", "text": result}]}

    @tool(
        "finalize_function",
        "Submit the final aggregate function. Code must be FULLY SELF-CONTAINED "
        "(all imports, faasr_install calls, data loading). Runs in a fresh namespace "
        "with input_dir and output_dir. Nothing from exploration carries over.",
        {"code": str},
    )
    async def finalize_function(args: dict) -> dict:
        state["finalized_code"] = args["code"]
        return {"content": [{"type": "text", "text": "Function saved. You may stop."}]}

    return create_sdk_mcp_server(
        "coding",
        tools=[execute_code, download_dataset, finalize_function],
    )


async def main():
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
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from FaaSr_py.client.agent_prompts import build_agentic_coding_system_prompt
    except Exception as e:
        write_result(False, f"Import error: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    # Ensure directories exist
    output_dir = context.get("output_dir", "/tmp/agent/output")
    input_dir = context.get("input_dir", "/tmp/agent/input")
    logs_dir = context.get("logs_dir", "/tmp/agent/logs")
    code_dir = context.get("code_dir", "/tmp/agent/code")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    Path(code_dir).mkdir(parents=True, exist_ok=True)

    _invocation_id = context.get("invocation_id", "")
    _rank = context.get("rank", {})
    _log_path = Path(logs_dir) / "coding_agent.log"

    def _faasr_log(msg):
        with open(_log_path, "a") as _f:
            _f.write(str(msg) + "\n")

    _installed_packages_path = Path("/tmp/agent/installed_packages.json")

    def _faasr_install(package_name: str):
        import importlib
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pip install {package_name!r} failed:\n{result.stderr}")
        importlib.invalidate_caches()
        _faasr_log(f"Installed package: {package_name}")
        try:
            existing = json.loads(_installed_packages_path.read_text()) if _installed_packages_path.exists() else []
            if package_name not in existing:
                existing.append(package_name)
            _installed_packages_path.write_text(json.dumps(existing))
        except Exception:
            pass

    # If caller provides pre-generated code (e.g. cache hit), skip the agentic loop
    if context.get("pre_generated_code"):
        code = context["pre_generated_code"]
    else:
        try:
            from claude_agent_sdk import (
                ClaudeAgentOptions,
                ClaudeSDKClient,
                AssistantMessage,
                ResultMessage,
                TextBlock,
            )
        except ImportError as e:
            write_result(False, f"claude_agent_sdk not available: {e}")
            sys.exit(1)

        # claude_agent_sdk reads ANTHROPIC_API_KEY; map AGENT_KEY to it
        agent_key = os.environ.get("AGENT_KEY", "")
        if agent_key:
            os.environ["ANTHROPIC_API_KEY"] = agent_key

        # Exploration namespace: input_dir accessible, output_dir intentionally excluded
        explore_ns = {
            "__builtins__": __builtins__,
            "faasr_log": _faasr_log,
            "faasr_install": _faasr_install,
            "faasr_invocation_id": lambda: _invocation_id,
            "faasr_rank": lambda: _rank,
            "input_dir": input_dir,
        }

        state = {"finalized_code": None}
        mcp_server = _build_mcp_server(explore_ns, input_dir, state)

        system_prompt = context.get("system_prompt") or build_agentic_coding_system_prompt(context)
        prompt = context.get("prompt", "")

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            mcp_servers={"coding": mcp_server},
            allowed_tools=[
                "mcp__coding__execute_code",
                "mcp__coding__download_dataset",
                "mcp__coding__finalize_function",
            ],
            max_turns=15,
        )

        try:
            async with ClaudeSDKClient(options) as client:
                await client.query(prompt)
                async for message in client.receive_messages():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                print(block.text, end="", flush=True)
                    elif isinstance(message, ResultMessage):
                        print()
                        break
        except Exception as e:
            write_result(False, f"Agent loop error: {e}\n{traceback.format_exc()}")
            sys.exit(1)

        code = state["finalized_code"]
        if not code:
            write_result(False, "Agent did not call finalize_function")
            sys.exit(1)

    # Scrub API keys before executing generated code
    os.environ.pop("AGENT_KEY", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)

    function_invoke = context.get("function_invoke", "coding_agent")
    code_path = Path(code_dir) / f"{function_invoke}.py"
    try:
        code_path.write_text(code)
        _faasr_log(f"Saved generated code to {code_path}")
    except Exception as e:
        _faasr_log(f"Warning: could not save generated code: {e}")

    # Fresh execution namespace — no exploration state leaks in
    fresh_namespace = {
        "__builtins__": __builtins__,
        "faasr_log": _faasr_log,
        "faasr_install": _faasr_install,
        "faasr_invocation_id": lambda: _invocation_id,
        "faasr_rank": lambda: _rank,
        "input_dir": input_dir,
        "output_dir": output_dir,
    }

    try:
        exec(code, fresh_namespace)
        write_result(True)
    except Exception:
        tb = traceback.format_exc()
        _faasr_log(f"Code execution failed:\n{tb}")
        write_result(False, tb)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
