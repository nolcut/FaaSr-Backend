from __future__ import annotations

from typing import Any, Callable, Dict, List, TypedDict
import json
import os
import re
import logging

from faasr_ai.utils.task_type import TaskType
from faasr_ai.prompts.workflow_prompts import WORKFLOW_PROMPT

logger = logging.getLogger(__name__)


# -------------------------
# State (for integration)
# -------------------------
class WorkflowState(TypedDict, total=False):
    # Input artifact from clarification agent
    user_request: str

    # Output artifact
    workflow_tasks: List[Dict[str, Any]]

    # Debug/metadata
    workflow_prompt: str
    workflow_errors: List[str]


# -------------------------
# Config (tune via orchestrator if needed)
# -------------------------
DEFAULT_MAX_TASKS = 8

# -------------------------
# Helpers
# -------------------------
def build_task_type_desc() -> str:
    """
    Build a bullet list of available task types from TaskType Enum.
    """
    lines: List[str] = []
    for tt in TaskType:
        name = tt.type_name
        desc = getattr(tt.value, "desc", "")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)



def generate_workflow_tasks(
    llm_call: Callable[[str], str],
    structured_request: Dict[str, Any],
    *,
    max_tasks: int = DEFAULT_MAX_TASKS,
) -> List[Dict[str, Any]]:
    
    task_type_desc = build_task_type_desc()
    context_json = json.dumps(structured_request, indent=2, ensure_ascii=False)

    prompt = WORKFLOW_PROMPT.format(
        context_json=context_json,
        max_tasks=max_tasks,
    )

    raw = llm_call(prompt)

    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    tasks = json.loads(text)
    if not isinstance(tasks, list):
        raise ValueError("Model output was not a JSON array.")
    return tasks


def workflow_agent_node(state: WorkflowState, llm_call: Callable[[str], str]) -> WorkflowState:
    user_request = state.get("user_request") or ""
    if not isinstance(user_request, str) or not user_request:
        state["workflow_tasks"] = []
        state["workflow_errors"] = ["structured_request missing or invalid; expected a non-empty string."]
        return state

    try:
        tasks = generate_workflow_tasks(llm_call, user_request, max_tasks=DEFAULT_MAX_TASKS)
        state["workflow_tasks"] = tasks
        
        logger.debug("Workflow agent output (workflow_tasks): %s", json.dumps(state.get("workflow_tasks", []), indent=2, ensure_ascii=False))
        
        state["workflow_errors"] = []
    except Exception as e:
        state["workflow_tasks"] = []
        state["workflow_errors"] = [str(e)]

    return state

def make_initial_workflow_state(user_request: str) -> WorkflowState:
    return {
        "user_request": user_request,
        "workflow_tasks": [],
        "workflow_prompt": "",
        "workflow_errors": []
    }