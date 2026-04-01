"""
Bridge action entry point.

A Bridge action transforms upstream data into the format expected by a downstream action.
It is LLM-driven: the IO agent selects relevant upstream files, the coding agent generates
transformation code, and the eval agent validates and uploads outputs.

Unlike Agent actions, Bridge has no user-supplied prompt. The task is auto-generated from:
  - upstream file metadata (schemas, samples, descriptions from registry)
  - downstream action code or prompt (fetched by the downstream resolver)
"""
import datetime
import json
import logging
import os
import sys
import threading
import traceback as tb_module
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from FaaSr_py.client.agent_func_entry import (
    _clear_dir,
    _extract_json,
    _run_prefix,
    _sample_file,
    _summarise_output_dir,
    _upload_generated_code,
    _upload_outputs,
    _write_agent_result,
    _write_manifest,
    _start_duration_monitor,
)
from FaaSr_py.client.agent_prompts import EVAL_SYSTEM_PROMPT, IO_SYSTEM_PROMPT
from FaaSr_py.client.agent_s3_ops import AgentS3Ops
from FaaSr_py.client.bridge_prompts import build_bridge_system_prompt
from FaaSr_py.client.coding_agent_backend import get_coding_backend
from FaaSr_py.helpers.agent_helper import AgentCodeGenerator, get_agent_api_key, get_agent_provider
from FaaSr_py.helpers.downstream_resolver import resolve_downstream
from FaaSr_py.helpers.s3_helper_functions import flush_s3_log
from FaaSr_py.s3_api import faasr_get_file
from FaaSr_py.s3_api.registry import faasr_registry_query, faasr_snapshot_existing_keys

logger = logging.getLogger(__name__)

INPUT_DIR = "/tmp/agent/input"
OUTPUT_DIR = "/tmp/agent/output"
CODE_DIR = "/tmp/agent/code"
LOGS_DIR = "/tmp/agent/logs"
INSTALLED_PACKAGES_FILE = "/tmp/agent/installed_packages.json"

IO_TEMP = 0.0
CODING_TEMP = 0.2
EVALUATOR_TEMP = 0.0
SAMPLE_BUDGET_CHARS = 20_000


class BridgeGraphState(TypedDict, total=False):
    function_invoke: str
    downstream_specs: List[Dict]         # from downstream_resolver
    registry_entries: List[Dict]         # upstream registry entries
    io_prompt: str                       # auto-generated prompt for IO agent file selection
    selected_uris: List[str]             # file URIs chosen by IO agent LLM
    file_metadata: Dict[str, Any]        # {uri: {local_path, sidecar, sample}}
    bridge_system_prompt: str            # auto-generated coding system prompt
    coding_result: Dict[str, Any]        # {success, exception, installed_packages}
    eval_decision: str                   # "continue" | "loop_back" | "abort"
    eval_reasoning: str
    loop_count: int


def run_bridge_function(faasr, action_name: str, result_file: str):
    """
    Entry point for Bridge action execution.

    Arguments:
        faasr: FaaSr payload instance
        action_name: Name of the bridge action (= faasr["FunctionInvoke"])
        result_file: Path to write the result JSON for the executor to read
    """
    logger.info(f"Starting bridge execution for {action_name}")

    try:
        api_key = get_agent_api_key()
        provider = get_agent_provider()
        if not provider:
            raise RuntimeError(
                "Could not determine LLM provider. Please set AGENT_KEY."
            )

        snapshot = faasr_snapshot_existing_keys(faasr)
        s3_ops = AgentS3Ops(faasr, snapshot)

        generator = AgentCodeGenerator(api_key, provider)
        graph = _build_bridge_graph(faasr, generator, s3_ops, result_file)

        stop_event = threading.Event()
        _start_duration_monitor(stop_event, faasr)

        try:
            final_state = graph.invoke(
                {
                    "function_invoke": action_name,
                    "loop_count": 0,
                }
            )
        finally:
            stop_event.set()

        result = final_state.get("eval_decision") != "abort"
        _write_agent_result(result_file, function_result=result)

    except Exception as e:
        err_msg = f"Bridge execution failed: {str(e)}"
        traceback = tb_module.format_exc()
        logger.error(f"{err_msg}\n{traceback}")
        _write_agent_result(result_file, error=True, message=err_msg, traceback=traceback)
        sys.exit(1)
    finally:
        flush_s3_log()


def _build_bridge_graph(faasr, generator: AgentCodeGenerator, s3_ops: AgentS3Ops, result_file: str):
    """Build the 4-node Bridge LangGraph execution flow."""

    def _node_query_and_resolve(state: BridgeGraphState) -> Dict[str, Any]:
        logger.info("Node: query_and_resolve")
        action_name = state["function_invoke"]

        # Query upstream registry (exclude this action's own outputs)
        entries = faasr_registry_query(faasr, action_name=action_name)
        logger.info(f"Bridge found {len(entries)} upstream registry entries")

        # Resolve downstream action code/prompts
        downstream_specs = resolve_downstream(faasr, action_name)
        logger.info(
            f"Bridge resolved {len(downstream_specs)} downstream spec(s): "
            + ", ".join(s["action_name"] for s in downstream_specs)
        )

        # Build a concise IO selection prompt describing what downstream needs
        io_prompt = _build_io_prompt(downstream_specs)

        return {
            "registry_entries": entries,
            "downstream_specs": downstream_specs,
            "io_prompt": io_prompt,
        }

    def _node_io_agent(state: BridgeGraphState) -> Dict[str, Any]:
        logger.info("Node: io_agent")
        registry_entries = state.get("registry_entries", [])
        io_prompt = state.get("io_prompt", "")
        downstream_specs = state.get("downstream_specs", [])

        # LLM selects which upstream URIs are relevant to satisfy the downstream
        selected_uris = _select_files(generator, io_prompt, registry_entries)
        logger.info(f"IO agent selected {len(selected_uris)} files for bridge")

        uri_to_entry = {e.get("file_uri", ""): e for e in registry_entries}

        os.makedirs(INPUT_DIR, exist_ok=True)
        file_metadata: Dict[str, Any] = {}
        per_file_chars = SAMPLE_BUDGET_CHARS // len(selected_uris) if selected_uris else 0

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
                raise RuntimeError(f"IO agent failed to download {uri}: {e}") from e

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

            sample = _sample_file(local_path, sidecar, per_file_chars)
            file_metadata[uri] = {
                "local_path": local_path,
                "sidecar": sidecar,
                "sample": sample,
            }

        logger.info(f"IO agent file inventory: {list(file_metadata.keys())}")

        # Build the bridge coding system prompt now that we have file metadata
        bridge_system_prompt = build_bridge_system_prompt({
            "registry_entries": registry_entries,
            "file_metadata": file_metadata,
            "downstream_specs": downstream_specs,
            "input_dir": INPUT_DIR,
            "output_dir": OUTPUT_DIR,
            "eval_feedback": state.get("eval_reasoning", ""),
            "exception": state.get("coding_result", {}).get("exception", ""),
            "loop_count": state.get("loop_count", 0),
        })

        return {
            "selected_uris": selected_uris,
            "file_metadata": file_metadata,
            "bridge_system_prompt": bridge_system_prompt,
        }

    def _node_bridge_coding(state: BridgeGraphState) -> Dict[str, Any]:
        logger.info("Node: bridge_coding")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        eval_reasoning = state.get("eval_reasoning", "")
        loop_count = state.get("loop_count", 0)
        if eval_reasoning:
            logger.info(f"Passing eval feedback to bridge coding (loop {loop_count}): {eval_reasoning}")

        selected_uris = set(state.get("selected_uris", []))
        all_registry_entries = state.get("registry_entries", [])
        relevant_entries = [e for e in all_registry_entries if e.get("file_uri") in selected_uris]

        # Derive the user-facing prompt from downstream specs (used as the LLM turn prompt)
        downstream_specs = state.get("downstream_specs", [])
        coding_prompt = _build_coding_turn_prompt(downstream_specs)

        context = {
            # Bridge-specific: pre-built system prompt injected into coding_agent_entry.py
            "system_prompt": state.get("bridge_system_prompt", ""),
            "prompt": coding_prompt,
            "function_invoke": state.get("function_invoke", ""),
            "registry_entries": relevant_entries,
            "file_metadata": state.get("file_metadata", {}),
            "downstream_specs": downstream_specs,
            "input_dir": INPUT_DIR,
            "output_dir": OUTPUT_DIR,
            "code_dir": CODE_DIR,
            "logs_dir": LOGS_DIR,
            "temperature": CODING_TEMP,
            "eval_feedback": eval_reasoning,
            "exception": state.get("coding_result", {}).get("exception", ""),
            "loop_count": loop_count,
        }
        result = get_coding_backend().run(context)
        logger.info(f"Bridge coding finished: success={result.success}")

        installed_packages = []
        try:
            pkg_file = Path(INSTALLED_PACKAGES_FILE)
            if pkg_file.exists():
                installed_packages = json.loads(pkg_file.read_text())
        except Exception:
            pass

        if not result.success:
            function_invoke = state.get("function_invoke", "bridge")
            code_path = Path(CODE_DIR) / f"{function_invoke}.py"
            if code_path.exists():
                try:
                    s3_ops.agent_put_file(
                        local_file=code_path.name,
                        local_folder=str(code_path.parent),
                        remote_file=f"failed_{code_path.name}",
                        remote_folder=f"{_run_prefix(faasr)}/{function_invoke}_outputs",
                    )
                    logger.info(f"Uploaded failed bridge code as failed_{code_path.name}")
                except Exception as e:
                    logger.warning(f"Could not upload failed bridge code: {e}")

        return {
            "coding_result": {
                "success": result.success,
                "exception": result.exception,
                "installed_packages": installed_packages,
            }
        }

    def _node_bridge_eval(state: BridgeGraphState) -> Dict[str, Any]:
        logger.info("Node: bridge_eval")
        coding_result = state.get("coding_result", {})
        downstream_specs = state.get("downstream_specs", [])
        loop_count = state.get("loop_count", 0)

        output_summary = _summarise_output_dir()

        downstream_desc = ", ".join(
            f"{s['action_name']} ({s['action_type']})" for s in downstream_specs
        ) or "(unknown)"
        eval_task_desc = f"Bridge transformation for downstream action(s): {downstream_desc}"

        system_prompt = EVAL_SYSTEM_PROMPT
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
            f"Task: {eval_task_desc}\n\n"
            f"Coding agent success: {coding_result.get('success')}\n"
            f"Exception: {coding_result.get('exception') or 'none'}\n"
            f"{coding_log}"
            f"\nOutput directory contents:\n{output_summary}"
        )
        raw = generator.generate_text(eval_prompt, system_prompt, temperature=EVALUATOR_TEMP)
        logger.debug(f"Bridge eval LLM raw response:\n{raw}")
        parsed = _extract_json(raw)
        if parsed is None:
            logger.warning(f"Bridge eval: JSON extraction failed on raw response: {raw[:500]}")
            parsed = {"decision": "abort", "reasoning": "Bridge eval returned unparseable response"}
        decision_data = parsed
        decision = decision_data.get("decision", "continue")
        reasoning = decision_data.get("reasoning", "")
        file_descriptions = decision_data.get("file_descriptions", {})

        logger.info(f"Bridge eval decision: {decision} | reasoning: {reasoning}")
        logger.info(f"Output summary:\n{output_summary}")

        if decision == "loop_back" and loop_count >= 1:
            logger.warning(f"Max loopbacks reached — last reasoning: {reasoning}")
            decision = "abort"
            reasoning = f"Max loopbacks reached — {reasoning}"

        new_loop_count = loop_count + (1 if decision == "loop_back" else 0)

        if decision == "continue":
            function_invoke = state.get("function_invoke", "unknown")
            _write_manifest(
                faasr,
                state,
                file_descriptions,
                state.get("coding_result", {}).get("installed_packages", []),
            )
            _upload_outputs(function_invoke, _run_prefix(faasr), file_descriptions, s3_ops)
            _upload_generated_code(function_invoke, _run_prefix(faasr), s3_ops)

        if decision == "loop_back":
            _clear_dir(OUTPUT_DIR)
            _clear_dir(INPUT_DIR)

        loop_back_state_reset = {}
        if decision == "loop_back":
            loop_back_state_reset = {
                "selected_uris": [],
                "file_metadata": {},
                "bridge_system_prompt": "",
            }

        _log_file = Path("/tmp/agent/logs/coding_agent.log")
        if _log_file.exists():
            try:
                invocation_id = faasr.get("InvocationID", "unknownID")
                s3_ops.agent_put_file(
                    local_file=_log_file.name,
                    local_folder=str(_log_file.parent),
                    remote_file=f"{state.get('function_invoke', 'bridge')}_{invocation_id}_coding_agent.log",
                    remote_folder=f"{_run_prefix(faasr)}/{state.get('function_invoke', 'bridge')}_logs",
                )
            except Exception as e:
                logger.warning(f"Could not upload bridge coding log: {e}")

        return {
            "eval_decision": decision,
            "eval_reasoning": reasoning,
            "loop_count": new_loop_count,
            **loop_back_state_reset,
        }

    def _eval_router(state: BridgeGraphState) -> str:
        decision = state.get("eval_decision", "continue")
        if decision == "abort":
            _write_agent_result(
                result_file,
                error=True,
                message=state.get("eval_reasoning", "Bridge aborted"),
            )
            return "abort"
        return decision

    graph = StateGraph(BridgeGraphState)
    graph.add_node("query_and_resolve", _node_query_and_resolve)
    graph.add_node("io_agent", _node_io_agent)
    graph.add_node("bridge_coding", _node_bridge_coding)
    graph.add_node("bridge_eval", _node_bridge_eval)

    graph.set_entry_point("query_and_resolve")
    graph.add_edge("query_and_resolve", "io_agent")
    graph.add_edge("io_agent", "bridge_coding")
    graph.add_edge("bridge_coding", "bridge_eval")
    graph.add_conditional_edges(
        "bridge_eval",
        _eval_router,
        {"continue": END, "loop_back": "io_agent", "abort": END},
    )

    return graph.compile()


# Helpers

def _build_io_prompt(downstream_specs: List[Dict]) -> str:
    """
    Build a concise selection prompt for the IO agent describing what the downstream needs.
    """
    if not downstream_specs:
        return "Select all available upstream files."

    parts = []
    for spec in downstream_specs:
        action_name = spec.get("action_name", "")
        action_type = spec.get("action_type", "")
        code_or_prompt = spec.get("code_or_prompt", "")
        # Truncate to keep the IO prompt short
        snippet = code_or_prompt[:800] if code_or_prompt else "(not available)"
        parts.append(
            f"Downstream action: {action_name} (Type: {action_type})\n"
            f"Code/prompt excerpt:\n{snippet}"
        )

    downstream_desc = "\n\n".join(parts)
    return (
        "Select the upstream files needed to produce inputs for the following downstream action(s).\n\n"
        + downstream_desc
    )


def _build_coding_turn_prompt(downstream_specs: List[Dict]) -> str:
    """
    Build the LLM turn prompt (user message) for the coding agent.
    The system prompt already contains all the context; this is a short directive.
    """
    if not downstream_specs:
        return "Transform the upstream data files as needed. If no transformation is required, produce no output."

    names = ", ".join(s["action_name"] for s in downstream_specs)
    return (
        f"Generate Python transformation code to convert the upstream data into the format "
        f"expected by: {names}. "
        "Use the file schemas and code/prompts in the system prompt to infer expected filenames and formats."
    )


def _select_files(
    generator: AgentCodeGenerator,
    prompt: str,
    registry_entries: List[Dict],
) -> List[str]:
    """Ask the LLM to select file URIs from registry entries."""
    if not registry_entries:
        return []

    visible_entries = [
        {
            "uri": e.get("file_uri", ""),
            "name": e.get("name", ""),
            "description": e.get("description", ""),
        }
        for e in registry_entries
    ]
    valid_uris = {e.get("file_uri", "") for e in registry_entries}

    selection_prompt = (
        f"Task: {prompt}\n\n"
        "Registry entries:\n"
        + "\n".join(
            f"- uri={e['uri']} name={e['name']}: {e['description']}"
            for e in visible_entries
        )
    )

    raw = generator.generate_text(selection_prompt, IO_SYSTEM_PROMPT, temperature=IO_TEMP)
    data = _extract_json(raw) or {}
    logger.debug(f"Bridge IO selection rationale: {data.get('rationale', '')}")

    all_returned = data.get("uris", [])
    dropped = [u for u in all_returned if u not in valid_uris]
    if dropped:
        logger.warning(f"Bridge IO agent dropped {len(dropped)} hallucinated URIs: {dropped}")
    uris = [u for u in all_returned if u in valid_uris]
    logger.info(f"Bridge IO agent selected URIs: {uris}")
    return uris
