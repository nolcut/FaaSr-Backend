# src/faasr_ai/orchestrator.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, TypedDict
import asyncio
import sys
import logging

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

from faasr_ai.agents.clarification import build_clarification_graph, make_initial_clarification_state
from faasr_ai.agents.workflow import workflow_agent_node, make_initial_workflow_state
from faasr_ai.agents.reflection import build_reflection_graph, make_initial_reflection_state
from faasr_ai.agents.coding import coding_node, make_initial_coding_state
from faasr_ai.agents.execute import execute_node


class GlobalState(TypedDict, total=False):
    # Input
    user_request: str
    
    # GitHub configuration
    gh_pat: str
    github_repository: str
    github_branch: str

    # Clarification outputs
    structured_request: Dict[str, Any]
    clarification_nl: str

    # Workflow outputs
    workflow_tasks: List[Dict[str, Any]]
    workflow_errors: List[str]
    
    # FaaSr workflow
    faasr_workflow: Dict[str, Any]
    
    # Reflection outputs
    validation_result: Dict[str, Any]
    reflection_count: int
    max_reflections: int
    has_issues: bool
    
    # Coding outputs
    notebook_path: str
    coding_errors: List[str]
    generated_functions: Dict[str, str]
    generation_count: int
    
    # Execute outputs
    execute: Dict[str, Any]


def clarification_node(state: GlobalState, llm_call: Callable[[str], str]) -> GlobalState:
    """
    Runs the Clarification Agent sub-graph and writes results into GlobalState.
    """
    logger.debug("Entering clarification_node")
    user_request = (state.get("user_request") or "").strip()
    if not user_request:
        state["structured_request"] = {}
        state["clarification_nl"] = ""
        state["workflow_errors"] = ["user_request is empty"]
        return state

    clarifier = build_clarification_graph(llm_call)
    clar_state = make_initial_clarification_state(user_request)
    clar_final = clarifier.invoke(clar_state)

    state["structured_request"] = clar_final.get("final_structured_request", {}) or {}
    state["clarification_nl"] = clar_final.get("enhanced_natural_language_request", "") or ""
    logger.debug("Exiting clarification_node")
    return state


def workflow_generation_node(state: GlobalState, llm_call: Callable[[str], str]) -> GlobalState:
    """
    Calls workflow agent and stores workflow tasks/errors in GlobalState.
    """
    logger.debug("Entering workflow_generation_node")
    user_request = state.get("clarification_nl") or ""
    init_wf_state = make_initial_workflow_state(user_request)

    wf_state = workflow_agent_node(init_wf_state, llm_call)

    # propagate outputs
    state["workflow_tasks"] = wf_state.get("workflow_tasks", []) or []
    state["workflow_errors"] = wf_state.get("workflow_errors", []) or []
    
    # Initialize reflection state
    state["reflection_count"] = 0
    state["max_reflections"] = state.get("max_reflections", 5)  # default 5 iterations
    state["has_issues"] = False
    
    logger.debug("Exiting workflow_generation_node")
    return state


def reflection_node(state: GlobalState, llm_call: Callable[[str], str], input_fn=None) -> GlobalState:
    """
    Runs the Reflection Agent sub-graph to validate and refine workflow.
    """
    logger.debug("Entering reflection_node")
    workflow_tasks = state.get("workflow_tasks", [])
    user_request = state.get("clarification_nl", "")
    max_reflections = state.get("max_reflections", 5)
    
    if not workflow_tasks:
        state["validation_result"] = {
            "has_issues": True,
            "issues": [{
                "task_id": "general",
                "severity": "critical",
                "category": "completeness",
                "description": "No workflow tasks to validate"
            }],
            "suggestions": [],
            "overall_quality": "poor"
        }
        state["has_issues"] = True
        logger.debug("No workflow tasks - setting has_issues=True")
        return state
    
    # Build and run reflection sub-graph
    reflector = build_reflection_graph(llm_call, input_fn=input_fn)
    reflection_state = make_initial_reflection_state(
        workflow_tasks=workflow_tasks,
        user_request=user_request,
        workflows_folder="workflows/",
        max_reflections=max_reflections
    )
    reflection_final = reflector.invoke(reflection_state)
    
    # Update global state with reflection results
    state["workflow_tasks"] = reflection_final.get("workflow_tasks", [])
    state["faasr_workflow"] = reflection_final.get("faasr_workflow", {}) or {}
    state["validation_result"] = reflection_final.get("validation_result", {})
    state["reflection_count"] = reflection_final.get("reflection_count", 0)
    state["has_issues"] = reflection_final.get("has_issues", False)
    
    logger.debug(f"Exiting reflection_node - has_issues={state['has_issues']}")
    return state


def coding_generation_node(state: GlobalState, llm_call: Callable[..., Any]) -> GlobalState:
    """
    Runs the Coding Agent to generate & execute task code and export a notebook.
    Only called if reflection says no issues (by router).
    """
    logger.debug("Entering coding_generation_node")
    # Windows: avoid zmq/proactor warnings when running Jupyter kernels
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    tasks = state.get("workflow_tasks", []) or []
    user_request = state.get("clarification_nl", "") or state.get("user_request", "") or ""

    # Initialize coding state
    coding_state = make_initial_coding_state(
        workflow_tasks=tasks,
        user_request=user_request,
        data_folder="tutorial",
        functions_folder="functions"
    )

    # Run async coding node inside sync orchestrator
    coding_result = asyncio.run(
        coding_node(
            coding_state,
            llm_call=llm_call,
            max_attempts=3,
            timeout_s=120
        )
    )

    # Propagate outputs into global state
    state["notebook_path"] = coding_result.get("notebook_path", "")
    state["coding_errors"] = coding_result.get("coding_errors", []) or []
    state["generated_functions"] = coding_result.get("generated_functions", {}) or {}
    state["generation_count"] = coding_result.get("generation_count", 0)

    logger.debug(f"Exiting coding_generation_node - coding_errors={state['coding_errors']}")
    return state


def execute_wrapper_node(state: GlobalState) -> GlobalState:
    """
    Wrapper for execute_node to handle GitHub operations.
    """
    
    # execute_node already handles state extraction and environment variables
    result = execute_node(state)
    
    execute_result = result.get('execute', {})
    logger.debug(f"Exiting execute_wrapper_node - execute.ok={execute_result.get('ok')}")
    if not execute_result.get('ok'):
        logger.debug(f"Execute error: {execute_result.get('error')}")
    else:
        logger.debug(f"Execute success - uploaded {len(execute_result.get('uploaded_paths', []))} files")
    
    return result


def router_after_reflection(state: GlobalState, mode: str) -> str:
    """Route based on validation results and execution mode."""
    logger.debug(f"router_after_reflection called in mode: {mode}")
    has_issues = state.get("has_issues", False)
    logger.debug(f"router_after_reflection - has_issues={has_issues}")
    if has_issues:
        logger.debug("Reflection agent could not resolve all issues within the maximum allowed revisions (MAX_LOOP).")
        logger.debug("The program will now terminate. Please review the workflow logic and your request.")
        logger.debug("router_after_reflection - routing to END")
        return "END"
        
    if mode == "workflow_only":
        logger.debug("router_after_reflection - workflow_only mode: routing to END")
        return "END"
        
    logger.debug("router_after_reflection - routing to coding")
    return "coding"



def router_after_coding(state: GlobalState, mode: str) -> str:
    """Route based on coding results and execution mode."""
    logger.debug(f"router_after_coding called in mode: {mode}")
    if state.get("coding_errors"):
        logger.debug("ERROR: THERE ARE CODING ERRORS. WILL NOT EXECUTE")
        return "END"
        
    if mode in ["workflow_and_code", "code_only"]:
        logger.debug(f"router_after_coding - {mode} mode: routing to END")
        return "END"
        
    logger.debug("router_after_coding - routing to execute")
    return "execute"



def router_after_execute(state: GlobalState) -> str:
    """Always end after execution."""
    logger.debug("router_after_execute called - routing to END")
    return "END"


def build_orchestrator_graph(llm_call: Callable[[str], str], mode: str = "full", input_fn=None):
    """
    Build the complete orchestration graph:
    clarification -> workflow_generation -> reflection -> coding -> execute -> END
    
    With conditional routing at reflection and coding stages.
    """
    logger.debug(f"Building orchestrator graph for mode: {mode}")
    g = StateGraph(GlobalState)

    # Add all nodes
    g.add_node("clarification", lambda s: clarification_node(s, llm_call))
    g.add_node("workflow_generation", lambda s: workflow_generation_node(s, llm_call))
    g.add_node("reflection", lambda s: reflection_node(s, llm_call, input_fn=input_fn))
    g.add_node("coding", lambda s: coding_generation_node(s, llm_call))
    g.add_node("execute", execute_wrapper_node)


    # Set entry point
    if mode in ["full", "workflow_only", "workflow_and_code"]:
        g.set_entry_point("clarification")
    elif mode in ["code_only", "code_and_deploy"]:
        g.set_entry_point("coding")
    elif mode == "deploy_only":
        g.set_entry_point("execute")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Linear flow until reflection
    g.add_edge("clarification", "workflow_generation")
    g.add_edge("workflow_generation", "reflection")

    # Conditional routing after reflection
    g.add_conditional_edges(
        "reflection",
        lambda state: router_after_reflection(state, mode),
        {
            "coding": "coding",
            "END": END,
        },
    )
    
    # Conditional routing after coding
    g.add_conditional_edges(
        "coding",
        lambda state: router_after_coding(state, mode),
        {
            "execute": "execute",
            "END": END,
        },
    )

    # Always end after execute
    g.add_conditional_edges(
        "execute",
        router_after_execute,
        {"END": END},
    )

    logger.debug("Orchestrator graph compiled")
    return g.compile()