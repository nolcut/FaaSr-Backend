import hashlib
import logging
import os
import re

from FaaSr_py.helpers.faasr_start_invoke_helper import faasr_get_github_raw

logger = logging.getLogger(__name__)


def resolve_downstream(faasr: dict, bridge_action_name: str) -> list[dict]:
    """
    Resolve the immediate InvokeNext targets of a Bridge action into their
    code or prompt text so the Bridge can infer expected I/O.

    Returns a list of dicts, one per downstream action:
        {
            "action_name": str,
            "action_type": str,         # "Python", "R", "Agent", "Bridge"
            "code_or_prompt": str,      # source code or agent prompt text
            "code_hash": str,           # SHA-256 of code_or_prompt
            "function_name": str | None
        }
    """
    action_config = faasr["ActionList"][bridge_action_name]
    invoke_next = action_config.get("InvokeNext", [])
    if isinstance(invoke_next, str):
        invoke_next = [invoke_next]

    results = []
    for target in invoke_next:
        # Strip rank suffix e.g. "ActionName(2)"
        action_name = re.split(r"[()]", target)[0]
        spec = _resolve_action(faasr, action_name, visited=set())
        results.extend(spec)

    return results


def _resolve_action(faasr: dict, action_name: str, visited: set) -> list[dict]:
    """
    Resolve a single action. Recurses through Bridge chains to find terminal actions.
    """
    if action_name in visited:
        logger.warning(f"Cycle detected resolving downstream action: {action_name}")
        return []
    visited = visited | {action_name}

    action_config = faasr["ActionList"].get(action_name)
    if not action_config:
        logger.warning(f"Downstream action not found in ActionList: {action_name}")
        return []

    action_type = action_config.get("Type", "")

    if action_type == "Bridge":
        # Recurse to find the terminal non-Bridge consumer
        invoke_next = action_config.get("InvokeNext", [])
        if isinstance(invoke_next, str):
            invoke_next = [invoke_next]
        results = []
        for target in invoke_next:
            next_name = re.split(r"[()]", target)[0]
            results.extend(_resolve_action(faasr, next_name, visited))
        return results

    if action_type == "Agent":
        prompt = action_config.get("Arguments", {}).get("prompt", "")
        return [_make_spec(action_name, action_type, prompt, function_name=None)]

    if action_type in ("Python", "R"):
        function_name = action_config.get("FunctionName")
        code = _fetch_function_code(faasr, function_name, action_type)
        return [_make_spec(action_name, action_type, code, function_name=function_name)]

    logger.warning(f"Unknown action type '{action_type}' for downstream action: {action_name}")
    return []


def _fetch_function_code(faasr: dict, function_name: str | None, action_type: str) -> str:
    """
    Fetch the source code for a Python or R function from GitHub or local file.
    Returns empty string if the code cannot be retrieved.
    """
    if not function_name:
        return ""

    token = os.getenv("GH_PAT")

    # Try remote GitHub repo first
    if "FunctionGitRepo" in faasr:
        paths = faasr["FunctionGitRepo"].get(function_name)
        if paths:
            if isinstance(paths, str):
                paths = [paths]
            for path in paths:
                ext = os.path.splitext(path)[-1].lower()
                if ext in (".py", ".r"):
                    try:
                        code = faasr_get_github_raw(token, path)
                        logger.info(
                            f"Fetched downstream code for {function_name} from {path}"
                        )
                        return code
                    except Exception as e:
                        logger.warning(
                            f"Could not fetch downstream code from {path}: {e}"
                        )

    # Try local file
    if "FunctionLocalFile" in faasr:
        paths = faasr["FunctionLocalFile"].get(function_name)
        if paths:
            if isinstance(paths, str):
                paths = [paths]
            for path in paths:
                ext = os.path.splitext(path)[-1].lower()
                if ext in (".py", ".r") and os.path.isfile(path):
                    try:
                        with open(path, "r") as f:
                            code = f.read()
                        logger.info(
                            f"Fetched downstream code for {function_name} from local file {path}"
                        )
                        return code
                    except Exception as e:
                        logger.warning(
                            f"Could not read local file {path}: {e}"
                        )

    # Fall back to the installed function in /tmp/functions/{InvocationID}/
    invocation_id = faasr.get("InvocationID", "")
    if invocation_id and function_name:
        for ext in (".py", ".r", ".R"):
            local_path = f"/tmp/functions/{invocation_id}/{function_name}{ext}"
            if os.path.isfile(local_path):
                try:
                    with open(local_path, "r") as f:
                        code = f.read()
                    logger.info(
                        f"Fetched downstream code for {function_name} from {local_path}"
                    )
                    return code
                except Exception as e:
                    logger.warning(f"Could not read {local_path}: {e}")

    logger.warning(
        f"Could not retrieve source code for downstream function: {function_name}"
    )
    return ""


def _make_spec(
    action_name: str,
    action_type: str,
    code_or_prompt: str,
    function_name: str | None,
) -> dict:
    code_hash = hashlib.sha256(code_or_prompt.encode()).hexdigest()
    return {
        "action_name": action_name,
        "action_type": action_type,
        "code_or_prompt": code_or_prompt,
        "code_hash": code_hash,
        "function_name": function_name,
    }
