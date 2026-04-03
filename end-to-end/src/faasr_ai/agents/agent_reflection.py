from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, TypedDict

from faasr_ai.agents.agent_workflow import _extract_json, _validate_schema, _validate_dag
from faasr_ai.prompts.agent_reflection_prompts import (
    AGENT_VALIDATION_PROMPT_TEMPLATE,
    AGENT_REVISION_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_REFLECTIONS = 3


# -------------------------
# State
# -------------------------
class AgentReflectionState(TypedDict, total=False):
    """State for the agent reflection sub-graph."""
    # Input
    faasr_json: Dict[str, Any]
    clarified_description: str

    # Validation outputs
    validation_result: Dict[str, Any]
    has_issues: bool
    reflection_count: int
    max_reflections: int

    # For applying fixes
    suggestions: List[str]

    # Human validation
    human_approved: bool
    human_feedback: Optional[str]


# -------------------------
# Validation / Revision Logic
# -------------------------
def validate_agent_workflow(
    llm_call: Callable[[str, Optional[str]], str],
    faasr_json: Dict[str, Any],
    clarified_description: str,
) -> Dict[str, Any]:
    """Use LLM to validate the agent workflow and identify issues.

    Returns:
        Dict with keys: has_issues, issues, suggestions, overall_quality
    """
    faasr_json_str = json.dumps(faasr_json, indent=2, ensure_ascii=False)

    prompt = AGENT_VALIDATION_PROMPT_TEMPLATE.format(
        clarified_description=clarified_description,
        faasr_json=faasr_json_str,
    )

    raw = llm_call(prompt, "You are a workflow validation expert. Output only valid JSON.")
    clean_raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean_raw)
    except json.JSONDecodeError:
        return {
            "has_issues": True,
            "issues": [{
                "action_name": "general",
                "severity": "critical",
                "category": "parsing",
                "description": "Failed to parse validation response",
            }],
            "suggestions": ["Re-run validation"],
            "overall_quality": "poor",
        }


def revise_agent_workflow(
    llm_call: Callable[[str, Optional[str]], str],
    faasr_json: Dict[str, Any],
    clarified_description: str,
    validation_result: Dict[str, Any],
    human_feedback: Optional[str] = None,
) -> tuple[Dict[str, Any], bool]:
    """Use LLM to revise the agent workflow based on validation feedback.

    Re-validates schema and DAG after revision. Returns (original, False) if the
    revision cannot be parsed or breaks structural validity.

    Returns:
        (faasr_json, success)
    """
    faasr_json_str = json.dumps(faasr_json, indent=2, ensure_ascii=False)
    issues_json = json.dumps(validation_result.get("issues", []), indent=2, ensure_ascii=False)
    suggestions_json = json.dumps(validation_result.get("suggestions", []), indent=2, ensure_ascii=False)

    prompt = AGENT_REVISION_PROMPT_TEMPLATE.format(
        clarified_description=clarified_description,
        faasr_json=faasr_json_str,
        issues_json=issues_json,
        suggestions_json=suggestions_json,
    )

    if human_feedback:
        prompt += f"\n\nAdditional feedback from the user:\n{human_feedback}"

    raw = llm_call(prompt, "You are a workflow revision expert. Output only valid JSON.")
    revised = _extract_json(raw)

    if revised is None:
        logger.warning("Failed to parse revised workflow JSON; keeping original.")
        return faasr_json, False

    # Discard revision if it breaks schema or DAG validity
    schema_errors = _validate_schema(revised)
    if schema_errors:
        logger.warning(f"Revised workflow has schema violations; keeping original. Errors: {schema_errors[:3]}")
        return faasr_json, False

    dag_errors = _validate_dag(revised)
    if dag_errors:
        logger.warning(f"Revised workflow has DAG violations; keeping original. Errors: {dag_errors}")
        return faasr_json, False

    return revised, True


# -------------------------
# Nodes
# -------------------------
def reflection_check_node(state: AgentReflectionState) -> AgentReflectionState:
    """Initialize counters and reset approval flags."""
    if "reflection_count" not in state:
        state["reflection_count"] = 0
    if "max_reflections" not in state:
        state["max_reflections"] = DEFAULT_MAX_REFLECTIONS
    state["human_approved"] = False
    state["human_feedback"] = None
    return state


def reflection_validation_node(
    state: AgentReflectionState,
    llm_call: Callable[[str, Optional[str]], str],
) -> AgentReflectionState:
    """Validate the agent workflow and display results."""
    faasr_json = state.get("faasr_json", {})
    clarified_description = state.get("clarified_description", "")

    if not faasr_json:
        state["validation_result"] = {
            "has_issues": True,
            "issues": [{
                "action_name": "general",
                "severity": "critical",
                "category": "coverage",
                "description": "No workflow JSON to validate",
            }],
            "suggestions": ["Generate a workflow first"],
            "overall_quality": "poor",
        }
        state["has_issues"] = True
        return state

    action_count = len(faasr_json.get("ActionList", {}))
    print(f"\n[Reflection Agent] Validating workflow with {action_count} action(s)...")

    validation_result = validate_agent_workflow(
        llm_call, faasr_json, clarified_description,
    )

    state["validation_result"] = validation_result
    state["has_issues"] = validation_result.get("has_issues", False)
    state["suggestions"] = validation_result.get("suggestions", [])

    print(f"\n--- Validation Results ---")
    print(f"Overall Quality: {validation_result.get('overall_quality', 'unknown')}")
    print(f"Has Issues: {state['has_issues']}")

    issues = validation_result.get("issues", [])
    if issues:
        print(f"\nFound {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            severity = issue.get("severity", "unknown")
            category = issue.get("category", "unknown")
            description = issue.get("description", "")
            action_name = issue.get("action_name", "general")
            print(f"  {i}. [{severity.upper()}] {action_name} ({category}): {description}")

    suggestions = validation_result.get("suggestions", [])
    if suggestions:
        print(f"\nSuggestions ({len(suggestions)}):")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")

    return state


def human_validation_node(
    state: AgentReflectionState,
    input_fn=None,
) -> AgentReflectionState:
    """Present the workflow to the user and ask for approval.

    input_fn: callable(prompt) -> str; defaults to built-in input().
    Accepts input_fn to support thread-safe input via _HumanInputBroker.
    """
    if input_fn is None:
        input_fn = input

    faasr = state.get("faasr_json", {})
    action_list = faasr.get("ActionList", {})

    print("\n" + "=" * 60)
    print("  WORKFLOW SUMMARY")
    print("=" * 60)
    print(f"  Name:        {faasr.get('WorkflowName', 'N/A')}")
    print(f"  Entry Point: {faasr.get('FunctionInvoke', 'N/A')}")
    print(f"  Data Store:  {faasr.get('DefaultDataStore', 'N/A')}")
    print("-" * 60)

    for i, (name, action) in enumerate(action_list.items(), 1):
        invoke_next = action.get("InvokeNext", [])
        prompt_text = action.get("Arguments", {}).get("prompt", "")
        arrow = " -> " + ", ".join(str(n) for n in invoke_next) if invoke_next else " -> [END]"
        print(f"\n  [{i}] {name}{arrow}")
        print(f"      Prompt: {prompt_text}")

    print("=" * 60)

    answer = input_fn("\nDo you approve this workflow? [y/n]: ").strip().lower()

    if answer in ("y", "yes"):
        state["human_approved"] = True
        state["human_feedback"] = None
        print("Workflow approved.")
    else:
        feedback = input_fn("Provide feedback for revision (or press Enter to skip): ").strip()
        state["human_approved"] = False
        state["human_feedback"] = feedback or None
        print("Workflow rejected.")

    return state


def reflection_revision_node(
    state: AgentReflectionState,
    llm_call: Callable[[str, Optional[str]], str],
) -> AgentReflectionState:
    """Revise the workflow based on validation feedback and optional human feedback."""
    faasr_json = state.get("faasr_json", {})
    clarified_description = state.get("clarified_description", "")
    validation_result = state.get("validation_result", {})
    human_feedback = state.get("human_feedback")

    print(f"\n[Reflection Agent] Revising workflow based on feedback...")
    if human_feedback:
        print(f"  Incorporating user feedback: {human_feedback}")

    revised, success = revise_agent_workflow(
        llm_call,
        faasr_json,
        clarified_description,
        validation_result,
        human_feedback=human_feedback,
    )

    if not success:
        print("  Revision was invalid; workflow unchanged.")

    state["faasr_json"] = revised
    state["human_feedback"] = None

    reflection_count = state.get("reflection_count", 0)
    state["reflection_count"] = reflection_count + 1

    print(f"Workflow revised (iteration {state['reflection_count']})")

    return state


# -------------------------
# Routers
# -------------------------
def router_after_validation(state: AgentReflectionState) -> str:
    """Route based on LLM validation results."""
    has_issues = state.get("has_issues", False)
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", DEFAULT_MAX_REFLECTIONS)

    if not has_issues:
        print(f"\nLLM validation passed. Requesting human review...")
        return "human_validate"

    if reflection_count >= max_reflections:
        print(f"\nMax reflections ({max_reflections}) reached. Requesting human review anyway...")
        validation_result = state.get("validation_result", {})
        issues = validation_result.get("issues", [])
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        if critical_issues:
            print(f"WARNING: {len(critical_issues)} critical issue(s) remain unresolved!")
        return "human_validate"

    print(f"\nIssues found. Proceeding to revision (iteration {reflection_count + 1}/{max_reflections})...")
    return "revise"


def router_after_human_validation(state: AgentReflectionState) -> str:
    """Route based on human approval."""
    if state.get("human_approved"):
        return "done"
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", DEFAULT_MAX_REFLECTIONS)
    if reflection_count >= max_reflections:
        print(f"\nMax revisions ({max_reflections}) reached. Accepting current workflow.")
        return "done"
    return "revise"


# -------------------------
# Build Graph
# -------------------------
def build_agent_reflection_graph(
    llm_call: Callable[[str, Optional[str]], str],
    input_fn=None,
):
    """Build the agent reflection sub-graph.

    Graph structure:
    check -> validate -> human_validate -> (revise -> validate) -> END
    """
    from langgraph.graph import StateGraph, END

    g = StateGraph(AgentReflectionState)

    g.add_node("check", reflection_check_node)
    g.add_node("validate", lambda s: reflection_validation_node(s, llm_call))
    g.add_node("human_validate", lambda s: human_validation_node(s, input_fn=input_fn))
    g.add_node("revise", lambda s: reflection_revision_node(s, llm_call))

    g.set_entry_point("check")

    g.add_edge("check", "validate")

    g.add_conditional_edges(
        "validate",
        router_after_validation,
        {
            "revise": "revise",
            "human_validate": "human_validate",
        },
    )

    g.add_conditional_edges(
        "human_validate",
        router_after_human_validation,
        {
            "revise": "revise",
            "done": END,
        },
    )

    g.add_edge("revise", "validate")

    return g.compile()


# -------------------------
# Initialize State
# -------------------------
def make_initial_agent_reflection_state(
    faasr_json: Dict[str, Any],
    clarified_description: str,
    max_reflections: int = DEFAULT_MAX_REFLECTIONS,
) -> AgentReflectionState:
    return {
        "faasr_json": faasr_json,
        "clarified_description": clarified_description,
        "validation_result": {},
        "has_issues": False,
        "reflection_count": 0,
        "max_reflections": max_reflections,
        "suggestions": [],
        "human_approved": False,
        "human_feedback": None,
    }
