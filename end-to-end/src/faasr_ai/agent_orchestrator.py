# src/faasr_ai/agent_orchestrator.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from faasr_ai.agents.clarification import (
    build_clarification_graph,
    make_initial_clarification_state,
)
from faasr_ai.agents.agent_workflow import (
    generate_agent_workflow_node,
)
from faasr_ai.agents.agent_reflection import (
    build_agent_reflection_graph,
    make_initial_agent_reflection_state,
)


class AgentOrchestratorState(TypedDict, total=False):
    # Input
    user_description: str

    # After clarification
    clarified_description: str

    # Global inputs (deterministically captured during clarification)
    global_input_files: List[str]
    global_input_folders: List[str]

    # Workflow generation
    faasr_json: Dict[str, Any]
    generation_errors: List[str]

    # Reflection
    reflection_validation_result: Dict[str, Any]
    reflection_count: int


def _clarification_node(
    state: AgentOrchestratorState, llm_call: Callable[[str, Optional[str]], str]
) -> AgentOrchestratorState:
    """Run the clarification sub-graph with agent-workflow intent pre-set."""
    user_description = (state.get("user_description") or "").strip()
    if not user_description:
        state["clarified_description"] = ""
        state["generation_errors"] = ["No workflow description provided."]
        return state

    clarifier = build_clarification_graph(llm_call)
    clar_state = make_initial_clarification_state(
        user_description,
        intent="agent-workflow",
    )
    clar_final = clarifier.invoke(clar_state)

    state["clarified_description"] = (
        clar_final.get("enhanced_natural_language_request", "") or ""
    )
    final_req = clar_final.get("final_structured_request", {})
    state["global_input_files"] = final_req.get("global_input_files", [])
    state["global_input_folders"] = final_req.get("global_input_folders", [])
    return state


def _generate_node(
    state: AgentOrchestratorState, llm_call: Callable[[str, Optional[str]], str]
) -> AgentOrchestratorState:
    """Pass orchestrator state to agent workflow generation."""
    from faasr_ai.agents.agent_workflow import AgentWorkflowState
    wf_state: AgentWorkflowState = {
        "clarified_description": state.get("clarified_description", ""),
        "faasr_json": state.get("faasr_json", {}),
        "generation_errors": [],
        "user_approved": False,
        "review_feedback": "",
        "global_input_files": state.get("global_input_files", []),
        "global_input_folders": state.get("global_input_folders", []),
    }
    result = generate_agent_workflow_node(wf_state, llm_call)
    faasr_json = result.get("faasr_json", {})
    if faasr_json:
        gi_files = state.get("global_input_files", [])
        gi_folders = state.get("global_input_folders", [])
        if gi_files:
            faasr_json["GlobalInputFiles"] = gi_files
        if gi_folders:
            faasr_json["GlobalInputFolders"] = gi_folders
    state["faasr_json"] = faasr_json
    state["generation_errors"] = result.get("generation_errors", [])
    return state


def _reflect_node(
    state: AgentOrchestratorState, llm_call: Callable[[str, Optional[str]], str]
) -> AgentOrchestratorState:
    """Pass orchestrator state to the agent reflection sub-graph.

    Skips reflection entirely if generation failed (faasr_json is empty).

    Strips GlobalInputFiles/GlobalInputFolders before reflection (so the revision
    LLM doesn't see system-injected fields), then re-injects them afterward.
    """
    faasr_json = dict(state.get("faasr_json", {}))

    # Skip reflection entirely if generation failed
    if not faasr_json:
        return state

    # Strip system-injected fields before reflection
    gi_files = faasr_json.pop("GlobalInputFiles", state.get("global_input_files", []))
    gi_folders = faasr_json.pop("GlobalInputFolders", state.get("global_input_folders", []))

    reflector = build_agent_reflection_graph(llm_call)
    reflection_state = make_initial_agent_reflection_state(
        faasr_json=faasr_json,
        clarified_description=state.get("clarified_description", ""),
    )
    reflection_final = reflector.invoke(reflection_state)

    result_json = reflection_final.get("faasr_json", {})

    # Re-inject system-injected fields
    if gi_files:
        result_json["GlobalInputFiles"] = gi_files
    if gi_folders:
        result_json["GlobalInputFolders"] = gi_folders

    # Final validation gate: ensure the reflected workflow is still structurally valid
    if result_json:
        from faasr_ai.agents.agent_workflow import _validate_schema, _validate_dag
        schema_errors = _validate_schema(result_json)
        dag_errors = _validate_dag(result_json) if not schema_errors else []
        if schema_errors or dag_errors:
            all_errors = schema_errors + dag_errors
            import logging
            logging.getLogger(__name__).warning(
                f"Post-reflection validation failed: {'; '.join(all_errors[:3])}"
            )
            state["generation_errors"] = [
                f"Workflow is invalid after reflection:\n"
                + "\n".join(f"  - {e}" for e in all_errors)
            ]
            state["faasr_json"] = {}
            return state

    state["faasr_json"] = result_json
    state["reflection_validation_result"] = reflection_final.get("validation_result", {})
    state["reflection_count"] = reflection_final.get("reflection_count", 0)
    return state


def build_agent_orchestrator(
    llm_call_clarify: Callable[[str, Optional[str]], str],
    llm_call_workflow: Callable[[str, Optional[str]], str],
    llm_call_reflect: Callable[[str, Optional[str]], str],
) -> Any:
    """Build the LangGraph for agent workflow generation.

    Flow: clarification -> generate -> reflect (includes human approval) -> END
    """
    g = StateGraph(AgentOrchestratorState)

    g.add_node("clarification", lambda s: _clarification_node(s, llm_call_clarify))
    g.add_node("generate", lambda s: _generate_node(s, llm_call_workflow))
    g.add_node("reflect", lambda s: _reflect_node(s, llm_call_reflect))

    g.set_entry_point("clarification")
    g.add_edge("clarification", "generate")
    g.add_edge("generate", "reflect")
    g.add_edge("reflect", END)

    return g.compile()
