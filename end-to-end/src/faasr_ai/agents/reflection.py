from __future__ import annotations

from typing import Any, Callable, Dict, List, TypedDict, Optional
import json
import os
import logging
from dotenv import load_dotenv

from faasr_ai.utils.faasr_workflow_converter import tasks_to_faasr_workflow
from faasr_ai.prompts.reflection_prompts import VALIDATION_PROMPT_TEMPLATE, REVISION_PROMPT_TEMPLATE
from faasr_ai.agents.coding import _ensure_dir

logger = logging.getLogger(__name__)


# -------------------------
# State (for integration)
# -------------------------
class ReflectionState(TypedDict, total=False):
    """State for the reflection agent."""
    # Input from workflow agent
    workflow_tasks: List[Dict[str, Any]]
    user_request: str
    
    # Reflection outputs
    validation_result: Dict[str, Any]  # Contains issues and suggestions
    has_issues: bool
    reflection_count: int
    max_reflections: int
    
    # For applying fixes
    suggestions: List[str]
    reflection_prompt: str
    reflection_raw: str
    
    # Human validation
    human_approved: bool
    human_feedback: Optional[str]

    # FaaSr workflow output
    faasr_workflow: Dict[str, Any]
    
    output_folder: str


# -------------------------
# Config
# -------------------------
DEFAULT_MAX_REFLECTIONS = 3

# -------------------------
# Validation Logic
# -------------------------
def validate_workflow(
    llm_call: Callable[[str], str],
    workflow_tasks: List[Dict[str, Any]],
    user_request: str,
) -> Dict[str, Any]:
    """
    Use LLM to validate workflow and identify issues.
    
    Returns:
        Dict with keys: has_issues, issues, suggestions, overall_quality
    """
    tasks_json = json.dumps(workflow_tasks, indent=2, ensure_ascii=False)
    
    prompt = VALIDATION_PROMPT_TEMPLATE.format(
        user_request=user_request,
        tasks_json=tasks_json,
    )
    
    raw = llm_call(prompt)
    
    # Remove markdown code fences if present
    clean_raw = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        validation_result = json.loads(clean_raw)
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        validation_result = {
            "has_issues": True,
            "issues": [{
                "task_id": "general",
                "severity": "critical",
                "category": "parsing",
                "description": "Failed to parse validation response"
            }],
            "suggestions": ["Re-run validation"],
            "overall_quality": "poor"
        }
    
    return validation_result


def revise_workflow(
    llm_call: Callable[[str], str],
    workflow_tasks: List[Dict[str, Any]],
    user_request: str,
    validation_result: Dict[str, Any],
    human_feedback: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Use LLM to revise workflow based on validation feedback and optional human feedback.
    
    Returns:
        Revised list of workflow tasks
    """
    tasks_json = json.dumps(workflow_tasks, indent=2, ensure_ascii=False)
    issues_json = json.dumps(validation_result.get("issues", []), indent=2, ensure_ascii=False)
    suggestions_json = json.dumps(validation_result.get("suggestions", []), indent=2, ensure_ascii=False)
    
    prompt = REVISION_PROMPT_TEMPLATE.format(
        user_request=user_request,
        tasks_json=tasks_json,
        issues_json=issues_json,
        suggestions_json=suggestions_json,
    )

    if human_feedback:
        prompt += f"\n\nAdditional feedback from the user:\n{human_feedback}"
    
    raw = llm_call(prompt)
    
    # Remove markdown code fences if present
    clean_raw = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        revised_tasks = json.loads(clean_raw)
        if not isinstance(revised_tasks, list):
            raise ValueError("Revised workflow is not a JSON array")
        return revised_tasks
    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"Failed to parse revised workflow: {e}")
        return workflow_tasks


# -------------------------
# Node Implementations
# -------------------------
def reflection_validation_node(
    state: ReflectionState,
    llm_call: Callable[[str], str]
) -> ReflectionState:
    """
    Validate the workflow and identify issues.
    """
    workflow_tasks = state.get("workflow_tasks", [])
    user_request = state.get("user_request", "")
    
    if not workflow_tasks:
        state["validation_result"] = {
            "has_issues": True,
            "issues": [{
                "task_id": "general",
                "severity": "critical",
                "category": "completeness",
                "description": "No workflow tasks to validate"
            }],
            "suggestions": ["Generate workflow tasks first"],
            "overall_quality": "poor"
        }
        state["has_issues"] = True
        return state
    
    print(f"\n[Reflection Agent] Validating workflow with {len(workflow_tasks)} tasks...")
    
    validation_result = validate_workflow(llm_call, workflow_tasks, user_request)
    
    state["validation_result"] = validation_result
    state["has_issues"] = validation_result.get("has_issues", False)
    state["suggestions"] = validation_result.get("suggestions", [])
    
    # Display validation results
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
            task_id = issue.get("task_id", "general")
            print(f"  {i}. [{severity.upper()}] Task {task_id} ({category}): {description}")
    
    suggestions = validation_result.get("suggestions", [])
    if suggestions:
        print(f"\nSuggestions ({len(suggestions)}):")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    
    return state


def human_validation_node(state: ReflectionState, input_fn=None) -> ReflectionState:
    """
    Present the workflow to the user and ask for approval.
    Sets human_approved and optionally human_feedback in state.
    input_fn: callable(prompt) -> str; defaults to built-in input().
    """
    if input_fn is None:
        input_fn = input

    workflow_tasks = state.get("workflow_tasks", [])

    print("\n=== Workflow Tasks (Please Review) ===")
    print(json.dumps(workflow_tasks, indent=2, ensure_ascii=False))

    answer = input_fn("\nDo you approve this workflow? [y/n]: ").strip().lower()

    if answer in ["y", "yes"]:
        state["human_approved"] = True
        state["human_feedback"] = None
        print("Workflow approved by user.")
    else:
        feedback = input_fn("Please provide feedback for revision (or press Enter to skip): ").strip()
        state["human_approved"] = False
        state["human_feedback"] = feedback or None
        print("Workflow rejected by user.")

    return state


def reflection_revision_node(
    state: ReflectionState,
    llm_call: Callable[[str], str]
) -> ReflectionState:
    """
    Revise the workflow based on validation feedback and optional human feedback.
    """
    workflow_tasks = state.get("workflow_tasks", [])
    user_request = state.get("user_request", "")
    validation_result = state.get("validation_result", {})
    human_feedback = state.get("human_feedback")

    print(f"\n[Reflection Agent] Revising workflow based on feedback...")
    if human_feedback:
        print(f"  Incorporating user feedback: {human_feedback}")

    revised_tasks = revise_workflow(
        llm_call,
        workflow_tasks,
        user_request,
        validation_result,
        human_feedback=human_feedback,
    )
    
    state["workflow_tasks"] = revised_tasks
    state["human_feedback"] = None  # Clear after use

    reflection_count = state.get("reflection_count", 0)
    state["reflection_count"] = reflection_count + 1
    
    print(f"✓ Workflow revised (iteration {state['reflection_count']})")
    
    return state


def reflection_check_node(state: ReflectionState) -> ReflectionState:
    """
    Check if we should continue reflecting or stop.
    """
    if "reflection_count" not in state:
        state["reflection_count"] = 0
    
    if "max_reflections" not in state:
        state["max_reflections"] = DEFAULT_MAX_REFLECTIONS

    state["human_approved"] = False
    state["human_feedback"] = None
    
    return state


def faasr_conversion_node(state: ReflectionState) -> ReflectionState:
    """
    Convert validated workflow tasks to FaaSr format.
    This runs after all reflections are complete.
    """
    workflow_tasks = state.get("workflow_tasks", [])
    
    if not workflow_tasks:
        print("\n⚠️  No workflow tasks to convert to FaaSr format")
        state["faasr_workflow"] = {}
        return state
    
    print(f"\n[FaaSr Conversion] Converting {len(workflow_tasks)} tasks to FaaSr format...")
    
    gh_username = os.getenv("FAASR_GH_USERNAME", "YOUR_USERNAME")
    gh_repo = os.getenv("FAASR_ACTION_REPO", "tutorial")
    gh_branch = os.getenv("GITHUB_REF_NAME", "main")
    workflow_name = os.getenv("FAASR_WORKFLOW_NAME", "tutorial")

    s3_endpoint = os.getenv("FAASR_S3_ENDPOINT", "https://play.min.io")
    s3_bucket = os.getenv("FAASR_S3_BUCKET", "faasr")
    s3_region = os.getenv("FAASR_S3_REGION", "us-east-1")

    function_repo_default = f"{gh_username}/{gh_repo}/functions"

    try:
        faasr_wf = tasks_to_faasr_workflow(
            workflow_tasks,
            github_username=gh_username,
            action_repo_name=gh_repo,
            branch_name=gh_branch,
            s3_endpoint=s3_endpoint,
            s3_bucket=s3_bucket,
            s3_region=s3_region,
            workflow_name=workflow_name,
            function_repo_default=function_repo_default,
        )
        
        state["faasr_workflow"] = faasr_wf
    
        print(f"✓ FaaSr workflow generated successfully")
        print("\n=== FaaSr Workflow Format ===")
        print(json.dumps(state.get("faasr_workflow", []), indent=2, ensure_ascii=False))
        
        load_dotenv()
        workflow_name = os.getenv("FAASR_WORKFLOW_NAME", "turtorial").strip()
        
        _ensure_dir(state.get("workflows_folder", "workflows/"))
        out_path = os.path.join(state.get("workflows_folder", "workflows/"), f"{workflow_name}.json")
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(faasr_wf, f, indent=2, ensure_ascii=False)
            
        print("\nSaved workflow file to: ", out_path)
        
    except Exception as e:
        print(f"❌ Error converting to FaaSr format: {e}")
        state["faasr_workflow"] = {}
    
    return state


# -------------------------
# Routers
# -------------------------
def router_after_validation(state: ReflectionState) -> str:
    """
    Route based on LLM validation results.
    - "revise" if there are issues and we haven't exceeded max reflections
    - "human_validate" if no LLM issues found
    """
    has_issues = state.get("has_issues", False)
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", DEFAULT_MAX_REFLECTIONS)
    
    if not has_issues:
        print(f"\n✓ LLM validation passed. Requesting human review...")
        return "human_validate"
    
    if reflection_count >= max_reflections:
        print(f"\n⚠️  Max reflections ({max_reflections}) reached. Requesting human review anyway...")
        validation_result = state.get("validation_result", {})
        issues = validation_result.get("issues", [])
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        if critical_issues:
            print(f"WARNING: {len(critical_issues)} critical issue(s) remain unresolved!")
        return "human_validate"
    
    print(f"\n→ Issues found. Proceeding to revision (iteration {reflection_count + 1}/{max_reflections})...")
    return "revise"


def router_after_human_validation(state: ReflectionState) -> str:
    """
    Route based on human approval.
    - "done" if approved
    - "revise" if rejected
    """
    if state.get("human_approved"):
        return "done"
    return "revise"


# -------------------------
# Build Graph
# -------------------------
def build_reflection_graph(llm_call: Callable[[str], str], input_fn=None):
    """
    Build the reflection agent graph.
    
    The graph structure:
    check → validate → human_validate → (revise → validate) → faasr_conversion → done
    """
    from langgraph.graph import StateGraph, END
    
    g = StateGraph(ReflectionState)
    
    g.add_node("check", reflection_check_node)
    g.add_node("validate", lambda s: reflection_validation_node(s, llm_call))
    g.add_node("human_validate", lambda s: human_validation_node(s, input_fn=input_fn))
    g.add_node("revise", lambda s: reflection_revision_node(s, llm_call))
    g.add_node("faasr_conversion", faasr_conversion_node)
    
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
            "done": "faasr_conversion",
        },
    )
    
    g.add_edge("revise", "validate")
    g.add_edge("faasr_conversion", END)
    
    return g.compile()


# -------------------------
# Initialize Reflection State
# -------------------------
def make_initial_reflection_state(
    workflow_tasks: List[Dict[str, Any]],
    user_request: str,
    workflows_folder: str,
    max_reflections: int = DEFAULT_MAX_REFLECTIONS,
) -> ReflectionState:
    return {
        "workflow_tasks": workflow_tasks,
        "user_request": user_request,
        "validation_result": {},
        "has_issues": False,
        "reflection_count": 0,
        "max_reflections": max_reflections,
        "suggestions": [],
        "reflection_prompt": "",
        "reflection_raw": "",
        "human_approved": False,
        "human_feedback": None,
        "faasr_workflow": {},
        "output_folder": workflows_folder,
    }