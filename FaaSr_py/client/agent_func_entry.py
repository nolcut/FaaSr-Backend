import csv
import datetime
import json
import logging
import os
import shutil
import threading
import traceback as tb_module
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from FaaSr_py.client.agent_stubs import agent_put_file
from FaaSr_py.client.coding_agent_backend import get_coding_backend
from FaaSr_py.client.py_client_stubs import faasr_exit, faasr_extend, faasr_return
from FaaSr_py.helpers.agent_helper import AgentCodeGenerator, get_agent_api_key, get_agent_provider
from FaaSr_py.helpers.rank import faasr_rank as _faasr_rank
from FaaSr_py.helpers.s3_helper_functions import flush_s3_log
from FaaSr_py.s3_api import faasr_get_file, faasr_put_file
from FaaSr_py.s3_api.registry import faasr_registry_query

logger = logging.getLogger(__name__)

INPUT_DIR = "/tmp/agent/input"
OUTPUT_DIR = "/tmp/agent/output"


class AgentGraphState(TypedDict, total=False):
    prompt: str
    function_invoke: str            # = faasr["FunctionInvoke"]
    workflow_spec: Dict[str, Any]   # faasr["ActionList"][faasr["FunctionInvoke"]]
    registry_entries: List[Dict]    # upstream registry entries (schema_uri included)
    selected_uris: List[str]        # file URIs chosen by IO agent LLM
    file_metadata: Dict[str, Any]   # {uri: {local_path, sidecar, sample}}
    coding_result: Dict[str, Any]   # {success, exception}
    eval_decision: str              # "continue" | "loop_back" | "abort"
    eval_reasoning: str
    loop_count: int


def run_agent_function(faasr, prompt, action_name):
    """
    Entry point for agent function execution.

    Arguments:
        faasr: FaaSr payload instance
        prompt: Natural language prompt for the agent
        action_name: Name of the action being executed (= faasr["FunctionInvoke"])
    """
    logger.info(f"Starting agent execution for {action_name}")

    try:
        api_key = get_agent_api_key()
        provider = get_agent_provider()
        if not provider:
            raise RuntimeError(
                "Could not determine LLM provider. Please set AGENT_KEY."
            )

        generator = AgentCodeGenerator(api_key, provider)
        graph = _build_agent_graph(faasr, generator)

        stop_event = threading.Event()
        _start_duration_monitor(stop_event, faasr)

        try:
            final_state = graph.invoke(
                {
                    "prompt": prompt,
                    "function_invoke": faasr["FunctionInvoke"],
                    "loop_count": 0,
                }
            )
        finally:
            stop_event.set()

        result = final_state.get("eval_decision") != "abort"
        faasr_return(result)

    except Exception as e:
        err_msg = f"Agent execution failed: {str(e)}"
        traceback = tb_module.format_exc()
        logger.error(f"{err_msg}\n{traceback}")
        faasr_exit(message=err_msg, traceback=traceback)
    finally:
        flush_s3_log()


def _build_agent_graph(faasr, generator: AgentCodeGenerator):
    """Build the 4-node LangGraph execution flow."""

    def _node_query_registry(state: AgentGraphState) -> Dict[str, Any]:
        logger.info("Node: query_registry")
        entries = faasr_registry_query(faasr, action_name=state["function_invoke"])
        workflow_spec = faasr["ActionList"].get(state["function_invoke"], {})
        return {"registry_entries": entries, "workflow_spec": workflow_spec}

    def _node_io_agent(state: AgentGraphState) -> Dict[str, Any]:
        logger.info("Node: io_agent")
        registry_entries = state.get("registry_entries", [])
        prompt = state.get("prompt", "")
        workflow_spec = state.get("workflow_spec", {})

        # LLM selects which file URIs to download
        selected_uris = _select_files(generator, prompt, workflow_spec, registry_entries)
        logger.info(f"IO agent selected {len(selected_uris)} files")

        # Build URI→entry lookup for schema_uri
        uri_to_entry = {e.get("file_uri", ""): e for e in registry_entries}

        # Download files + inspect
        os.makedirs(INPUT_DIR, exist_ok=True)
        file_metadata: Dict[str, Any] = {}

        for uri in selected_uris:
            parts = uri.rsplit("/", 1)
            remote_folder = parts[0] if len(parts) == 2 else "."
            remote_file = parts[-1]
            local_path = str(Path(INPUT_DIR) / remote_file)

            try:
                faasr_get_file(
                    faasr_payload=faasr,
                    local_file=remote_file,
                    remote_file=remote_file,
                    local_folder=INPUT_DIR,
                    remote_folder=remote_folder,
                )
            except Exception as e:
                logger.warning(f"IO agent could not download {uri}: {e}")
                continue

            # Download sidecar if available
            sidecar = {}
            entry = uri_to_entry.get(uri, {})
            schema_uri = entry.get("schema_uri", "")
            if schema_uri:
                sidecar_parts = schema_uri.rsplit("/", 1)
                sidecar_remote_folder = sidecar_parts[0] if len(sidecar_parts) == 2 else "."
                sidecar_remote_file = sidecar_parts[-1]
                sidecar_local = str(Path(INPUT_DIR) / sidecar_remote_file)
                try:
                    faasr_get_file(
                        faasr_payload=faasr,
                        local_file=sidecar_remote_file,
                        remote_file=sidecar_remote_file,
                        local_folder=INPUT_DIR,
                        remote_folder=sidecar_remote_folder,
                    )
                    with open(sidecar_local, "r") as f:
                        sidecar = json.load(f)
                except Exception as e:
                    logger.warning(f"IO agent could not download sidecar for {uri}: {e}")

            sample = _sample_file(local_path, sidecar)
            file_metadata[uri] = {
                "local_path": local_path,
                "sidecar": sidecar,
                "sample": sample,
            }

        logger.info(f"IO agent file inventory: {list(file_metadata.keys())}")
        return {"selected_uris": selected_uris, "file_metadata": file_metadata}

    def _node_coding_agent(state: AgentGraphState) -> Dict[str, Any]:
        logger.info("Node: coding_agent")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        eval_reasoning = state.get("eval_reasoning", "")
        loop_count = state.get("loop_count", 0)
        if eval_reasoning:
            logger.info(f"Passing eval feedback to coding agent (loop {loop_count}): {eval_reasoning}")

        # Only pass registry entries for files that were actually selected (to reduce payload)
        selected_uris = set(state.get("selected_uris", []))
        all_registry_entries = state.get("registry_entries", [])
        relevant_entries = [e for e in all_registry_entries if e.get("file_uri") in selected_uris]

        context = {
            "prompt": state.get("prompt", ""),
            "function_invoke": state.get("function_invoke", ""),
            "workflow_spec": state.get("workflow_spec", {}),
            "registry_entries": relevant_entries,
            "file_metadata": state.get("file_metadata", {}),
            "input_dir": INPUT_DIR,
            "output_dir": OUTPUT_DIR,
            "logs_dir": "/tmp/agent/logs",
            "invocation_id": faasr.get("InvocationID", ""),
            "rank": _faasr_rank(faasr_payload=faasr),
            "temperature": 0.2,
            "eval_feedback": eval_reasoning,
            "loop_count": loop_count,
        }
        result = get_coding_backend().run(context)
        logger.info(f"Coding agent finished: success={result.success}")

        if not result.success:
            function_invoke = state.get("function_invoke", "coding_agent")
            code_path = Path(OUTPUT_DIR) / f"{function_invoke}.py"
            if code_path.exists():
                try:
                    agent_put_file(
                        local_file=code_path.name,
                        local_folder=str(code_path.parent),
                        remote_file=f"failed_{code_path.name}",
                        remote_folder=f"{function_invoke}_outputs",
                    )
                    logger.info(f"Uploaded failed code as failed_{code_path.name}")
                except Exception as e:
                    logger.warning(f"Could not upload failed code: {e}")

        return {"coding_result": {"success": result.success, "exception": result.exception}}

    def _node_eval_agent(state: AgentGraphState) -> Dict[str, Any]:
        logger.info("Node: eval_agent")
        coding_result = state.get("coding_result", {})
        prompt = state.get("prompt", "")
        loop_count = state.get("loop_count", 0)

        # Summarise output directory
        output_summary = _summarise_output_dir()

        # LLM evaluation
        system_prompt = (
            "You are evaluating the output of a coding agent.\n"
            "Return ONLY valid JSON with these keys:\n"
            "  \"decision\": \"continue\"|\"loop_back\"|\"abort\"\n"
            "  \"reasoning\": \"<string>\"\n"
            "  \"file_descriptions\": {\"<filename>\": \"<natural language description of what the file contains>\"}\n"
            "Default to 'continue' unless there is a clear, objective reason not to.\n"
            "- continue: the agent produced output files and no unhandled exception occurred. "
            "Do NOT second-guess domain logic, date ranges, values, or whether the approach seems optimal.\n"
            "- loop_back: ONLY if the agent raised an unhandled exception, produced zero output files, "
            "or every output file is empty.\n"
            "- abort: ONLY if the exception is truly unrecoverable (e.g. missing credentials, "
            "invalid workflow config). Never abort for missing packages — use loop_back.\n"
            "IMPORTANT: The coding agent CAN install missing packages at runtime using faasr_install(). "
            "A ModuleNotFoundError is always loop_back, never abort.\n"
            "Provide a file_descriptions entry for every output file listed.\n"
            "Do not include any text outside the JSON."
        )
        coding_log = ""
        _log_file = Path("/tmp/agent/logs/coding_agent.log")
        if _log_file.exists():
            try:
                log_text = _log_file.read_text()
                if log_text:
                    coding_log = f"\nCoding agent log:\n{log_text}\n"
            except Exception:
                pass

        today_str = datetime.date.today().isoformat()
        eval_prompt = (
            f"Today's date: {today_str}\n\n"
            f"User task: {prompt}\n\n"
            f"Coding agent success: {coding_result.get('success')}\n"
            f"Exception: {coding_result.get('exception') or 'none'}\n"
            f"{coding_log}"
            f"\nOutput directory contents:\n{output_summary}"
        )
        raw = generator.generate_text(eval_prompt, system_prompt, temperature=0.6)
        logger.debug(f"Eval LLM raw response:\n{raw}")
        parsed = _extract_json(raw)
        if parsed is None:
            logger.warning(f"Eval agent: JSON extraction failed on raw response: {raw[:500]}")
        decision_data = parsed or {}
        decision = decision_data.get("decision", "continue")
        reasoning = decision_data.get("reasoning", "")
        file_descriptions = decision_data.get("file_descriptions", {})

        logger.info(f"Eval decision: {decision} | reasoning: {reasoning}")
        logger.info(f"Output summary:\n{output_summary}")
        logger.debug(f"File descriptions: {file_descriptions}")

        # Enforce max 1 loopback
        if decision == "loop_back" and loop_count >= 1:
            logger.warning(f"Max loopbacks reached — last reasoning: {reasoning}")
            decision = "abort"
            reasoning = f"Max loopbacks reached — {reasoning}"

        new_loop_count = loop_count + (1 if decision == "loop_back" else 0)

        # On continue: upload outputs
        if decision == "continue":
            _upload_outputs(state.get("function_invoke", "unknown"), file_descriptions)

        # On loop_back: clear working dirs for retry
        if decision == "loop_back":
            _clear_dir(OUTPUT_DIR)
            _clear_dir(INPUT_DIR)

        # Upload coding agent log to S3 (with invocation ID to make it unique)
        _log_file = Path("/tmp/agent/logs/coding_agent.log")
        if _log_file.exists():
            try:
                invocation_id = faasr.get("InvocationID", "unknownID")
                agent_put_file(
                    local_file=_log_file.name,
                    local_folder=str(_log_file.parent),
                    remote_file=f"{state.get('function_invoke', 'agent')}_{invocation_id}_coding_agent.log",
                    remote_folder=f"{state.get('function_invoke', 'agent')}_logs",
                )
            except Exception as e:
                logger.warning(f"Could not upload coding agent log: {e}")

        return {
            "eval_decision": decision,
            "eval_reasoning": reasoning,
            "loop_count": new_loop_count,
        }

    # Build graph
    graph = StateGraph(AgentGraphState)
    graph.add_node("query_registry", _node_query_registry)
    graph.add_node("io_agent", _node_io_agent)
    graph.add_node("coding_agent", _node_coding_agent)
    graph.add_node("eval_agent", _node_eval_agent)

    graph.set_entry_point("query_registry")
    graph.add_edge("query_registry", "io_agent")
    graph.add_edge("io_agent", "coding_agent")
    graph.add_edge("coding_agent", "eval_agent")
    graph.add_conditional_edges(
        "eval_agent",
        _eval_router,
        {"continue": END, "loop_back": "io_agent", "abort": END},
    )

    return graph.compile()


# Routing

def _eval_router(state: AgentGraphState) -> str:
    decision = state.get("eval_decision", "continue")
    if decision == "abort":
        faasr_exit(message=state.get("eval_reasoning", "Agent aborted"))
        return "abort"
    return decision


# IO agent helpers

def _select_files(
    generator: AgentCodeGenerator,
    prompt: str,
    workflow_spec: Dict[str, Any],
    registry_entries: List[Dict],
) -> List[str]:
    """Ask the LLM to select file URIs from registry entries."""
    if not registry_entries:
        return []

    # Show uri + name + description; LLM returns full URIs directly
    visible_entries = [
        {"uri": e.get("file_uri", ""), "name": e.get("name", ""), "description": e.get("description", "")}
        for e in registry_entries
    ]
    valid_uris = {e.get("file_uri", "") for e in registry_entries}

    system_prompt = (
        "You are a file selection assistant. Choose files needed to complete the user task.\n"
        "Return ONLY valid JSON: {\"uris\": [<list of uri strings>], \"rationale\": \"<string>\"}\n"
        "Use the exact uri values from the registry entries. Choose at most 10 files.\n"
        "Do not include any text outside the JSON."
    )
    selection_prompt = (
        f"Task: {prompt}\n\n"
        f"Workflow spec:\n{json.dumps(workflow_spec, indent=2)}\n\n"
        f"Registry entries:\n"
        + "\n".join(f"- uri={e['uri']} name={e['name']}: {e['description']}" for e in visible_entries)
    )

    raw = generator.generate_text(selection_prompt, system_prompt, temperature=0.2)
    data = _extract_json(raw) or {}
    logger.debug(f"IO selection rationale: {data.get('rationale', '')}")
    # Validate returned URIs against the known set to prevent hallucination
    all_returned = data.get("uris", [])
    dropped = [u for u in all_returned if u not in valid_uris]
    if dropped:
        logger.warning(f"IO agent dropped {len(dropped)} hallucinated URIs: {dropped}")
    uris = [u for u in all_returned if u in valid_uris]
    logger.info(f"IO agent selected URIs: {uris}")
    return uris[:10]


def _sample_file(local_path: str, sidecar: dict) -> str:
    """
    Return a short representative sample of the file's content,
    guided by the sidecar schema when available.
    Limits sample to 1000 chars to balance context and payload size.
    """
    max_sample_chars = 1000
    try:
        if local_path.endswith(".json"):
            with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Use sidecar keys to pick specific fields if available
                keys = sidecar.get("properties", {}).keys() if sidecar else data.keys()
                sample = {k: data[k] for k in list(keys)[:5] if k in data}
                result = json.dumps(sample, indent=2)
            elif isinstance(data, list):
                result = json.dumps(data[:3], indent=2)
            else:
                result = str(data)[:max_sample_chars]
            return result[:max_sample_chars]

        if local_path.endswith(".csv"):
            with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                rows = [next(reader, [])]  # header
                rows += [next(reader, []) for _ in range(3)]
            result = "\n".join(",".join(row) for row in rows if row)
            return result[:max_sample_chars]

        # Binary/image files — no sampling
        return ""

    except Exception as e:
        logger.warning(f"Could not sample {local_path}: {e}")
        return ""


# Eval agent helpers

def _summarise_output_dir() -> str:
    """List output files and provide a brief JSON summary for each."""
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        return "(output directory does not exist)"
    files = list(output_path.iterdir())
    if not files:
        return "(output directory is empty)"
    lines = []
    for f in files:
        if f.suffix == ".json":
            try:
                with open(f, "r") as fp:
                    data = json.load(fp)
                if isinstance(data, dict):
                    keys = list(data.keys())[:10]
                    lines.append(f"{f.name}: JSON object with keys {keys}")
                elif isinstance(data, list):
                    lines.append(f"{f.name}: JSON array with {len(data)} items")
                else:
                    lines.append(f"{f.name}: {type(data).__name__}")
            except Exception:
                lines.append(f"{f.name}: (unreadable JSON)")
        else:
            lines.append(f"{f.name}: {f.stat().st_size} bytes")
    return "\n".join(lines)


def _upload_outputs(function_invoke: str, file_descriptions: dict = None):
    """Upload all files in OUTPUT_DIR (including subdirectories) to S3 via agent_put_file."""
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        logger.warning("Output directory does not exist — nothing to upload")
        return
    remote_folder = f"{function_invoke}_outputs"
    descriptions = file_descriptions or {}

    # Recursively find all files in output_dir and subdirectories
    for file in output_path.rglob("*"):
        if file.is_file():
            # Preserve relative directory structure in remote folder
            rel_parent = file.relative_to(output_path).parent
            remote_subfolder = f"{remote_folder}/{rel_parent}" if str(rel_parent) != "." else remote_folder

            try:
                agent_put_file(
                    local_file=file.name,
                    local_folder=str(file.parent),
                    remote_file=file.name,
                    remote_folder=remote_subfolder,
                    description=descriptions.get(file.name, ""),
                )
                logger.info(f"Uploaded output: {remote_subfolder}/{file.name}")
            except Exception as e:
                logger.error(f"Failed to upload {file.name}: {e}")

    _log_generated_code_to_s3(remote_folder)


def _clear_dir(path: str):
    """Remove and recreate a directory."""
    try:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not clear {path}: {e}")



# Duration monitor

def _start_duration_monitor(stop_event: threading.Event, faasr):
    """Start a background thread that checkpoints state if the function nears timeout."""
    def _monitor():
        if not stop_event.wait(800):  # stops function if still running after 800s
            logger.warning("Function approaching timeout — checkpointing")
            _checkpoint_state_to_s3(faasr) # filler right now
            faasr_extend()  # does nothing (todo fix)

    threading.Thread(target=_monitor, daemon=True).start()


def _checkpoint_state_to_s3(faasr):
    """Upload a checkpoint marker to S3. Best-effort."""
    try:
        import time
        function_invoke = faasr.get("FunctionInvoke", "agent")
        marker = f"/tmp/{function_invoke}_checkpoint_{int(time.time())}.json"
        with open(marker, "w") as f:
            json.dump({"status": "timeout_checkpoint", "function": function_invoke}, f)
        faasr_put_file(
            faasr_payload=faasr,
            local_file=Path(marker).name,
            remote_file=Path(marker).name,
            local_folder="/tmp",
            remote_folder=f"{function_invoke}_checkpoints",
        )
    except Exception as e:
        logger.warning(f"Could not checkpoint to S3: {e}")



# Utilities

def _extract_json(text: str) -> Dict[str, Any] | None:
    """Best-effort JSON extraction from LLM response."""
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start: end + 1])
    except Exception:
        return None
    return None


def _log_generated_code_to_s3(remote_folder: str):
    """Upload an audit marker noting this agent run produced outputs."""
    try:
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        marker_name = f"agent_run_{timestamp}.json"
        marker_path = f"/tmp/{marker_name}"
        with open(marker_path, "w") as f:
            json.dump({"agent_run": timestamp, "output_folder": remote_folder}, f)
        # Use py_client_stubs so this doesn't trigger registry add
        from FaaSr_py.client.py_client_stubs import faasr_put_file as rpc_put
        rpc_put(
            local_file=marker_name,
            local_folder="/tmp",
            remote_file=marker_name,
            remote_folder="agent_audit",
        )
    except Exception as e:
        logger.warning(f"Could not log audit marker to S3: {e}")
