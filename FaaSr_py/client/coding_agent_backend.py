import json
import logging
import os
import secrets
import subprocess
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from FaaSr_py.client.agent_stubs import faasr_block_requests, faasr_unblock_requests

logger = logging.getLogger(__name__)

_ENTRY_SCRIPT = Path(__file__).parent / "coding_agent_entry.py"


@dataclass
class CodingResult:
    success: bool
    exception: str | None  # full traceback string if failed


class CodingAgentBackend(ABC):
    @abstractmethod
    def run(self, context: dict) -> CodingResult:
        """
        Run the coding agent with the given context.

        Arguments:
            context: dict with keys prompt, function_invoke, workflow_spec,
                     registry_entries, file_metadata, input_dir, output_dir

        Returns:
            CodingResult
        """


class DirectExecBackend(CodingAgentBackend):
    """
    Runs the coding agent as a plain subprocess with a filtered environment.
    No sandboxing — intended for development and testing.
    """

    def __init__(self, timeout: int = 300):
        self.timeout = timeout

    def run(self, context: dict) -> CodingResult:
        run_id = uuid.uuid4().hex
        ctx_file = f"/tmp/faasr_agent_ctx_{run_id}.json"
        result_file = f"/tmp/faasr_agent_result_{run_id}.json"

        secret = secrets.token_hex(32)
        try:
            with open(ctx_file, "w") as f:
                json.dump(context, f)

            faasr_block_requests(secret)
            try:
                env = _filtered_env()
                proc = subprocess.run(
                    [sys.executable, str(_ENTRY_SCRIPT), ctx_file, result_file],
                    env=env,
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                )
            finally:
                faasr_unblock_requests(secret)

            _append_subprocess_output(proc)
            if proc.returncode != 0:
                logger.warning(f"Coding agent subprocess exited with code {proc.returncode}")
            return _read_result(result_file)

        except subprocess.TimeoutExpired:
            logger.error("Coding agent subprocess timed out")
            return CodingResult(success=False, exception="Subprocess timed out")
        except Exception as e:
            logger.error(f"Coding agent backend error: {e}")
            return CodingResult(success=False, exception=str(e))
        finally:
            for path in (ctx_file, result_file):
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass


class NsjailBackend(CodingAgentBackend):
    """
    Runs the coding agent inside an nsjail sandbox.
    Binds /tmp read-write. Only AGENT_KEY and PYTHONPATH are passed through.
    """

    def __init__(self, timeout: int = 300):
        self.timeout = timeout

    def run(self, context: dict) -> CodingResult:
        run_id = uuid.uuid4().hex
        ctx_file = f"/tmp/faasr_agent_ctx_{run_id}.json"
        result_file = f"/tmp/faasr_agent_result_{run_id}.json"

        secret = secrets.token_hex(32)
        try:
            with open(ctx_file, "w") as f:
                json.dump(context, f)

            faasr_block_requests(secret)
            try:
                env = _filtered_env()
                cmd = [
                    "nsjail",
                    "-Mo",
                    "--time_limit", str(self.timeout),
                    "--bindmount", "/tmp",
                    "--disable_clone_newnet",   # allow network access (needed for pip installs)
                    "--disable_clone_newuser",  # avoid clone() permission errors in containers
                    "--",
                    sys.executable,
                    str(_ENTRY_SCRIPT),
                    ctx_file,
                    result_file,
                ]
                proc = subprocess.run(cmd, env=env, timeout=self.timeout + 10, capture_output=True, text=True)
            finally:
                faasr_unblock_requests(secret)

            _append_subprocess_output(proc)
            if proc.returncode != 0:
                logger.warning(f"Coding agent subprocess exited with code {proc.returncode}")
            return _read_result(result_file)

        except subprocess.TimeoutExpired:
            logger.error("Coding agent nsjail subprocess timed out")
            return CodingResult(success=False, exception="Subprocess timed out")
        except Exception as e:
            logger.error(f"Coding agent backend error: {e}")
            return CodingResult(success=False, exception=str(e))
        finally:
            for path in (ctx_file, result_file):
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass


def _append_subprocess_output(proc) -> None:
    """Append subprocess stdout/stderr to the coding agent log file."""
    if not (proc.stdout or proc.stderr):
        return
    log_path = Path("/tmp/agent/logs/coding_agent.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as lf:
        if proc.stdout:
            lf.write("\n--- subprocess stdout ---\n")
            lf.write(proc.stdout)
        if proc.stderr:
            lf.write("\n--- subprocess stderr ---\n")
            lf.write(proc.stderr)


def _filtered_env() -> dict:
    """Return a minimal environment for the coding agent subprocess.
    Explicitly excludes AWS credentials and any DataStore secrets.
    AGENT_KEY is passed through so the subprocess can call the LLM for code generation,
    but coding_agent_entry.py scrubs it before exec-ing the generated code."""
    allowed = {"PATH", "HOME", "PYTHONPATH", "TMPDIR", "LANG", "LC_ALL", "AGENT_KEY"}
    return {k: v for k, v in os.environ.items() if k in allowed and v is not None}


def _read_result(result_file: str) -> CodingResult:
    """Parse the result JSON written by coding_agent_entry.py."""
    try:
        with open(result_file, "r") as f:
            data = json.load(f)
        return CodingResult(
            success=bool(data.get("success", False)),
            exception=data.get("exception"),
        )
    except FileNotFoundError:
        return CodingResult(success=False, exception="Result file not written by agent")
    except Exception as e:
        return CodingResult(success=False, exception=f"Could not parse result: {e}")


def get_coding_backend(backend_type: str = None) -> CodingAgentBackend:
    """
    Return a CodingAgentBackend based on FAASR_CODING_BACKEND env var or argument.
    Defaults to DirectExecBackend.
    """
    t = backend_type or os.getenv("FAASR_CODING_BACKEND", "direct")
    if t == "nsjail":
        return NsjailBackend()
    return DirectExecBackend()
