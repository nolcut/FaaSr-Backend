import json
import logging
import sys
import traceback as tb_module
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from FaaSr_py.client.py_client_stubs import faasr_exit, faasr_log, faasr_return
from FaaSr_py.helpers.agent_constraints import AgentContextManager
from FaaSr_py.helpers.agent_helper import (
    AgentCodeGenerator,
    get_agent_api_key,
    get_agent_provider,
)

logger = logging.getLogger(__name__)


class AgentGraphState(TypedDict, total=False):
    """State container for the langgraph-based agent execution"""

    prompt: str
    action_name: str
    exploration_data: Dict[str, Any]
    selected_files: List[str]
    file_previews: Dict[str, str]
    generated_code: str
    result: Any
    cycle_count: int


def run_agent_function(faasr, prompt, action_name):
    """
    Entry point for agent function execution with adaptive exploration

    Arguments:
        faasr: FaaSr payload instance
        prompt: Natural language prompt for the agent
        action_name: Name of the action being executed
    """
    logger.info(f"Starting agent execution for {action_name} with prompt: {prompt[:100]}")

    # Initialize constraints
    context = AgentContextManager(faasr)

    try:
        # Get API key and provider
        api_key = get_agent_api_key()
        provider = get_agent_provider()

        if not provider:
            raise RuntimeError(
                "Could not determine LLM provider. Please set AGENT_KEY with valid OpenAI or Claude key."
            )

        logger.info(f"Using {provider} as LLM provider")

        generator = AgentCodeGenerator(api_key, provider)

        graph = _build_agent_graph(faasr, context, generator)
        final_state = graph.invoke(
            {
                "prompt": prompt,
                "action_name": action_name,
                "exploration_data": {},
                "cycle_count": 0,
            }
        )

        result = final_state.get("result", True)
        faasr_return(result)

    except Exception as e:
        err_msg = f"Agent execution failed: {str(e)}"
        traceback = tb_module.format_exc()
        logger.error(f"{err_msg}\n{traceback}")
        faasr_exit(message=err_msg, traceback=traceback)


def _build_agent_graph(faasr, context, generator):
    """
    Build the langgraph execution flow for the agent

    Steps:
    1) Explore S3 via API
    2) Select relevant files
    3) View file contents (previews)
    4) Generate code
    5) Execute code (uploads final response)
    """

        system_prompt = """You are a file selection assistant for S3 content (code, data, configs, logs).
        logger.info("Phase: explore_s3 - start")
        exploration_data = _explore_s3_context(faasr, context)
        logger.info(
            "Phase: explore_s3 - done (file_count=%s)",
    - Prioritize files most relevant to the user's workflow and the system prompt.
    - If the request is about data or outputs, prefer data files (csv/json/parquet/txt) and their configs.
    - If the request is about behavior/logic, prefer code, configs, and orchestration files.
    - Avoid unrelated test or sample files unless explicitly requested.
    - If unsure, pick the minimal set that most likely answers the request.
        logger.info("Phase: select_files - start")
        exploration_data = state.get("exploration_data", {})
        selected_files = _select_relevant_files(
            generator,
            state.get("prompt", ""),
            exploration_data,
        )
        logger.info("Phase: select_files - done (selected=%s)", len(selected_files))
        return {"selected_files": selected_files}

    def _node_view_files(state: AgentGraphState) -> Dict[str, Any]:
        logger.info("Phase: view_files - start")
        selected_files = state.get("selected_files", [])
        file_previews = _preview_relevant_files(selected_files)
        logger.info("Phase: view_files - done (previewed=%s)", len(file_previews))
        return {"file_previews": file_previews}

    def _node_decide_more_exploration(state: AgentGraphState) -> Dict[str, Any]:
        logger.info("Phase: decide_more_exploration - start")
        decision = _decide_more_exploration(
            generator,
            state.get("prompt", ""),
            state.get("exploration_data", {}),
            state.get("file_previews", {}),
        )
        logger.info(
            "Phase: decide_more_exploration - done (explore_more=%s, prefix=%s)",
            decision.get("explore_more") if isinstance(decision, dict) else None,
            decision.get("prefix") if isinstance(decision, dict) else None,
        )
        return {"exploration_decision": decision}

    def _node_explore_more(state: AgentGraphState) -> Dict[str, Any]:
        logger.info("Phase: explore_more - start")
        decision = state.get("exploration_decision", {})
        prefix = decision.get("prefix", "") if isinstance(decision, dict) else ""
        updated = _explore_more_s3_context(faasr, prefix, state.get("exploration_data", {}))
        cycle_count = int(state.get("cycle_count", 0)) + 1
        logger.info(
            "Phase: explore_more - done (cycle=%s, file_count=%s)",
            cycle_count,
            updated.get("file_count") if isinstance(updated, dict) else None,
        )
        return {"exploration_data": updated, "cycle_count": cycle_count}

    def _node_generate_code(state: AgentGraphState) -> Dict[str, Any]:
        logger.info("Phase: generate_code - start")
        exploration_data = dict(state.get("exploration_data", {}))
        if state.get("file_previews"):
            exploration_data["file_previews"] = state["file_previews"]

        code = generator.generate_code_with_context(state.get("prompt", ""), exploration_data)

        _log_generated_code_to_s3(faasr, state.get("action_name", "agent"), code)

        if not generator.validate_code_safety(code):
            raise RuntimeError("Generated code failed safety validation")

        logger.info("Phase: generate_code - done")
        return {"generated_code": code}

    def _node_execute_code(state: AgentGraphState) -> Dict[str, Any]:
        logger.info("Phase: execute_code - start")
        code = state.get("generated_code", "")
        if not code:
            raise RuntimeError("No generated code to execute")

        agent_namespace = _prepare_agent_namespace(faasr, context)

        agent_namespace["_regenerate_approach"] = lambda new_prompt, discovered_data: _adaptive_regeneration(
            generator, new_prompt, discovered_data, agent_namespace, context
        )

        env_backup = context.sanitize_environment()
        try:
            logger.info("Executing generated agent code")
            exec(code, agent_namespace)
            result = agent_namespace.get("result", True)
            logger.info("Agent execution completed successfully")
        finally:
            context.restore_environment(env_backup)

        logger.info("Phase: execute_code - done")
        return {"result": result}

    graph = StateGraph(AgentGraphState)
    graph.add_node("explore_s3", _node_explore_s3)
    graph.add_node("select_files", _node_select_files)
    graph.add_node("view_files", _node_view_files)
    graph.add_node("decide_more_exploration", _node_decide_more_exploration)
    graph.add_node("explore_more", _node_explore_more)
    graph.add_node("generate_code", _node_generate_code)
    graph.add_node("execute_code", _node_execute_code)

    graph.set_entry_point("explore_s3")
    graph.add_edge("explore_s3", "select_files")
    graph.add_edge("select_files", "view_files")
    graph.add_edge("view_files", "decide_more_exploration")
    graph.add_conditional_edges(
        "decide_more_exploration",
        lambda state: _should_explore_more(state),
        {
            "explore_more": "explore_more",
            "generate_code": "generate_code",
        },
    )
    graph.add_edge("explore_more", "select_files")
    graph.add_edge("generate_code", "execute_code")
    graph.add_edge("execute_code", END)

    return graph.compile()


def _select_relevant_files(generator: AgentCodeGenerator, prompt: str, exploration_data: Dict[str, Any]) -> List[str]:
    """
    Use the LLM to select relevant files from S3 listing
    """
    available_files = exploration_data.get("all_files") or exploration_data.get("files") or exploration_data.get("sample_files") or []

    if not available_files:
        return []

    system_prompt = """You are a file selection assistant for a codebase.
Return ONLY valid JSON with keys: files (list of strings), rationale (string).
Only choose from the provided available files. Choose at most 10 files.

Selection rules:
- Prioritize files most relevant to the user's workflow and the system prompt.
- Prefer entrypoints, orchestration logic, schedulers, config, and helpers directly tied to the request.
- If the request mentions "workflow", "system", "agent", or "execution", prefer files in client/, engine/, helpers/, server/.
- Avoid unrelated test or sample files unless explicitly requested.
- If unsure, pick the minimal set that most likely governs behavior.

Do not include any extra text outside JSON."""

    selection_prompt = (
        f"User request:\n{prompt}\n\nAvailable files:\n"
        + "\n".join(f"- {f}" for f in available_files)
    )

    raw = generator.generate_text(selection_prompt, system_prompt)

    selection = _extract_json(raw)
    if not selection or not isinstance(selection, dict):
        return []

    files = selection.get("files", [])
    if not isinstance(files, list):
        return []

    allowed = set(available_files)
    filtered = [f for f in files if isinstance(f, str) and f in allowed]
    return filtered[:10]


def _decide_more_exploration(
    generator: AgentCodeGenerator,
    prompt: str,
    exploration_data: Dict[str, Any],
    file_previews: Dict[str, str],
) -> Dict[str, Any]:
    """
    Ask the LLM if more S3 exploration is needed and which prefix to explore.
    """
    system_prompt = """You are deciding whether to explore more S3 prefixes.
Return ONLY valid JSON with keys: explore_more (boolean), prefix (string), rationale (string).
If no further exploration is needed, set explore_more=false and prefix="".
Do not include any extra text outside JSON."""

    available_files = exploration_data.get("all_files") or exploration_data.get("files") or exploration_data.get("sample_files") or []
    folders = exploration_data.get("folders") or []

    preview_summary = ""
    if file_previews:
        preview_summary = "\n\nPreviews (truncated):\n" + "\n".join(
            f"- {name}: {preview[:200].replace('\n', ' ')}" for name, preview in file_previews.items()
        )

    decision_prompt = (
        f"User request:\n{prompt}\n\n"
        f"Known folders:\n" + "\n".join(f"- {f}" for f in folders) + "\n\n"
        f"Available files (sample):\n" + "\n".join(f"- {f}" for f in available_files[:50]) +
        preview_summary
    )

    raw = generator.generate_text(decision_prompt, system_prompt)
    decision = _extract_json(raw)
    if not decision or not isinstance(decision, dict):
        return {"explore_more": False, "prefix": "", "rationale": "invalid_json"}

    explore_more = bool(decision.get("explore_more", False))
    prefix = decision.get("prefix", "")
    if not isinstance(prefix, str):
        prefix = ""

    return {
        "explore_more": explore_more,
        "prefix": prefix.strip(),
        "rationale": decision.get("rationale", ""),
    }


def _should_explore_more(state: AgentGraphState) -> str:
    """Return the next node key for conditional routing."""
    decision = state.get("exploration_decision", {})
    explore_more = bool(decision.get("explore_more")) if isinstance(decision, dict) else False
    cycle_count = int(state.get("cycle_count", 0))

    if explore_more and cycle_count < 2:
        return "explore_more"
    return "generate_code"


def _explore_more_s3_context(faasr, prefix: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explore S3 again using a prefix and merge into existing exploration data.
    """
    from FaaSr_py.client.py_client_stubs import faasr_get_folder_list

    updated = dict(current_data or {})
    try:
        files = faasr_get_folder_list(prefix=prefix or "")
        existing = set(updated.get("all_files") or updated.get("files") or [])
        merged = list(existing.union(files))

        updated["all_files"] = merged
        updated["file_count"] = len(merged)
        updated.setdefault("additional_files", [])
        updated["additional_files"] = list(set(updated["additional_files"]) | set(files))

        if "files" in updated:
            updated["files"] = merged[:50]
        else:
            updated["files"] = merged[:50]

        if "folders" in updated:
            updated["folders"] = list(set(updated["folders"]) | {f.split('/')[0] for f in files if '/' in f})
        else:
            updated["folders"] = list({f.split('/')[0] for f in files if '/' in f})

        return updated
    except Exception as e:
        logger.warning(f"Could not explore S3 with prefix '{prefix}': {e}")
        return updated


def _extract_json(text: str) -> Dict[str, Any] | None:
    """Best-effort JSON extraction from LLM response"""
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        return None

    return None


def _preview_relevant_files(selected_files: List[str]) -> Dict[str, str]:
    """
    Download and preview selected files (truncated) for context
    """
    from FaaSr_py.client.py_client_stubs import faasr_get_file

    previews: Dict[str, str] = {}

    if not selected_files:
        return previews

    preview_dir = Path("/tmp/faasr_agent_previews")
    preview_dir.mkdir(parents=True, exist_ok=True)

    for remote_file in selected_files[:10]:
        try:
            local_name = Path(remote_file).name or "preview_file"
            local_path = preview_dir / local_name

            faasr_get_file(
                local_file=local_path.name,
                remote_file=remote_file,
                local_folder=str(preview_dir),
                remote_folder=".",
            )

            with open(local_path, "rb") as f:
                raw = f.read(20000)
            preview_text = raw.decode("utf-8", errors="replace")
            previews[remote_file] = preview_text
        except Exception as e:
            logger.warning(f"Failed to preview {remote_file}: {e}")

    return previews


def _log_generated_code_to_s3(faasr, action_name, code):
    """
    Log the generated agent code to S3 for debugging/auditing
    
    Arguments:
        faasr: FaaSr payload instance
        action_name: Name of the agent action
        code: Generated Python code to log
    """
    try:
        from FaaSr_py.client.py_client_stubs import faasr_put_file
        import time
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        code_filename = f"{action_name}_generated_{timestamp}.py"
        
        # Write code to temp file
        temp_path = f"/tmp/{code_filename}"
        with open(temp_path, "w") as f:
            f.write(code)
        
        # Upload to S3
        faasr_put_file(code_filename, code_filename, local_folder="/tmp", remote_folder="agent_generated_code")
        logger.info(f"Logged generated code to S3: agent_generated_code/{code_filename}")
        
    except Exception as e:
        logger.warning(f"Could not log generated code to S3: {e}")


def _explore_s3_context(faasr, context):
    """
    Explore S3 to provide context for agent code generation
    
    Returns:
        dict: S3 exploration data including file list (if reasonable size)
    """
    from FaaSr_py.client.py_client_stubs import faasr_get_folder_list
    
    try:
        # Get list of files (limited exploration)
        files = faasr_get_folder_list()
        
        # Only include if list is reasonable size
        if len(files) <= 100:
            # Group by prefix/folder
            folders = {}
            for file in files:
                parts = file.split('/')
                if len(parts) > 1:
                    folder = parts[0]
                    if folder not in folders:
                        folders[folder] = []
                    folders[folder].append(file)
                else:
                    if 'root' not in folders:
                        folders['root'] = []
                    folders['root'].append(file)
            
            return {
                "file_count": len(files),
                "files": files[:50],  # First 50 files
                "all_files": files,
                "folders": list(folders.keys()),
                "structure": folders if len(files) <= 20 else None
            }
        else:
            return {
                "file_count": len(files),
                "note": "Too many files to list individually",
                "sample_files": files[:50]
            }
    except Exception as e:
        logger.warning(f"Could not explore S3: {e}")
        return {"error": "Could not list S3 files"}


def _adaptive_regeneration(generator, new_prompt, discovered_data, namespace, context):
    """
    Allow agent to regenerate approach based on discovered data
    
    This function is exposed to agent code as _regenerate_approach()
    """
    logger.info("Agent requesting adaptive regeneration based on discoveries")
    
    # Combine original context with new discoveries
    enhanced_prompt = f"""Based on discovered data: {discovered_data}

{new_prompt}"""
    
    # Generate new code
    new_code = generator.generate_code_with_context(enhanced_prompt, discovered_data)
    
    # Validate safety
    if not generator.validate_code_safety(new_code):
        raise RuntimeError("Regenerated code failed safety validation")
    
    # Execute new code in same namespace
    exec(new_code, namespace)
    

def _prepare_agent_namespace(faasr, context):
    """
    Prepare the execution namespace for agent code

    Arguments:
        faasr: FaaSr payload
        context: AgentContextManager instance

    Returns:
        dict: Namespace for exec()
    """
    # Create safe namespace with only allowed functions
    from FaaSr_py.client.agent_stubs import (
        agent_get_file,
        agent_get_folder_list,
        agent_invocation_id,
        agent_log,
        agent_put_file,
        agent_rank,
    )

    namespace = {
        "__builtins__": _get_safe_builtins(),
        # Core FaaSr functions (with agent constraints)
        "faasr_put_file": agent_put_file,
        "faasr_get_file": agent_get_file,
        "faasr_get_folder_list": agent_get_folder_list,
        "faasr_log": agent_log,
        "faasr_invocation_id": agent_invocation_id,
        "faasr_rank": agent_rank,
        # Common imports
        "json": __import__("json"),
        "os": __import__("os"),
        "sys": __import__("sys"),
        "csv": __import__("csv"),
        "math": __import__("math"),
        "datetime": __import__("datetime"),
        "re": __import__("re"),
        "pathlib": __import__("pathlib"),
        "Path": Path,
        # Store context for validators
        "_agent_context": context,
        "_agent_validator": context.validator,
    }

    return namespace


def _get_safe_builtins():
    """
    Return a restricted set of builtins for agent execution

    Returns:
        dict: Safe builtins
    """
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
