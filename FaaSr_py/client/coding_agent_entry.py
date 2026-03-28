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
        from FaaSr_py.client.agent_prompts import build_coding_system_prompt
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
        system_prompt = build_coding_system_prompt(context)
        prompt = context.get("prompt", "")

        def _generate_and_clean(gen_prompt):
            raw = generator.generate_text(gen_prompt, system_prompt, temperature=context.get("temperature", 0.2))
            if "```python" in raw:
                raw = raw.replace("```python\n", "").replace("\n```python", "")
            if "```" in raw:
                if raw.strip().startswith("```"):
                    raw = raw[raw.index("```") + 3:]
                if raw.strip().endswith("```"):
                    raw = raw[:raw.rindex("```")]
            return raw.strip()

        code = _generate_and_clean(prompt)
        for attempt in range(1, 3):
            try:
                compile(code, "<generated>", "exec")
                break
            except SyntaxError as e:
                if attempt == 2:
                    write_result(False, f"Syntax error after 3 attempts: {e}")
                    sys.exit(1)
                code = _generate_and_clean(
                    f"{prompt}\n\nYour previous response had a syntax error: {e}\n"
                    f"Return corrected Python only."
                )

        if not code:
            write_result(False, "LLM returned empty code")
            sys.exit(1)

    except Exception as e:
        write_result(False, f"Code generation error: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    # Ensure output and log dirs exist
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

    # Save generated code to dedicated code dir (separate from output data)
    function_invoke = context.get("function_invoke", "coding_agent")
    code_path = Path(code_dir) / f"{function_invoke}.py"
    try:
        code_path.write_text(code)
        _faasr_log(f"Saved generated code to {code_path}")
    except Exception as e:
        _faasr_log(f"Warning: could not save generated code: {e}")

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

    # Build execution namespace
    namespace = {
        "__builtins__": __builtins__,
        "faasr_log": _faasr_log,
        "faasr_install": _faasr_install,
        "faasr_invocation_id": lambda: _invocation_id,
        "faasr_rank": lambda: _rank,
        # Injected variables
        "input_dir": input_dir,
        "output_dir": output_dir,
    }

    # Scrub the agent key before executing generated code
    os.environ.pop("AGENT_KEY", None)

    try:
        exec(code, namespace)
        write_result(True)
    except Exception:
        tb = traceback.format_exc()
        _faasr_log(f"Code execution failed:\n{tb}")
        write_result(False, tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
