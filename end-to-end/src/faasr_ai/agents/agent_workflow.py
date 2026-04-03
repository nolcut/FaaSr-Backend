# src/faasr_ai/agents/agent_workflow.py
from __future__ import annotations

import json
import os
import re
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

try:
    import readline  # noqa: F401 — enables backspace/arrow keys in input()
except ImportError:
    pass  # Windows: readline not available

from faasr_ai.prompts.agent_workflow_prompts import AGENT_WORKFLOW_GENERATION_PROMPT

logger = logging.getLogger(__name__)

MAX_GENERATION_ATTEMPTS = 3


class AgentWorkflowState(TypedDict, total=False):
    clarified_description: str
    faasr_json: Dict[str, Any]
    generation_errors: List[str]
    user_approved: bool
    review_feedback: str
    global_input_files: List[str]
    global_input_folders: List[str]


def _load_example_json() -> str:
    """Load ecample.json as the few-shot example."""
    example_path = Path(__file__).resolve().parent.parent.parent.parent / "example.json"
    if example_path.exists():
        return example_path.read_text()
    return "{}"


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _validate_schema(faasr_json: dict) -> list[str]:
    """Validate against FaaSr.schema.json. Returns list of error messages (empty = valid)."""
    try:
        import jsonschema
    except ImportError:
        return []  # skip validation if jsonschema not installed

    schema_path = (
        Path(__file__).resolve().parents[4] / "FaaSr_py" / "FaaSr.schema.json"
    )
    if not schema_path.exists():
        return []

    schema = json.loads(schema_path.read_text())
    try:
        validator = jsonschema.Draft202012Validator(schema)
    except AttributeError:
        validator = jsonschema.Draft7Validator(schema)
    errors = list(validator.iter_errors(faasr_json))
    return [f"{list(e.absolute_path)}: {e.message}" for e in errors]


def _validate_dag(faasr_json: dict) -> list[str]:
    """Run DAG validation (cycle/reachability/rank checks). Returns list of error messages."""
    from FaaSr_py import graph_functions as faasr_gf
    try:
        faasr_gf.check_dag(faasr_json)
        return []
    except SystemExit:
        return ["DAG validation failed: workflow has a cycle, unreachable action, or rank violation"]


def _unique_workflow_name(name: str, workflows_dir: Path) -> str:
    """If WorkflowName.json already exists in workflows_dir, append -2, -3, etc."""
    if not (workflows_dir / f"{name}.json").exists():
        return name
    i = 2
    while (workflows_dir / f"{name}-{i}.json").exists():
        i += 1
    return f"{name}-{i}"


def generate_agent_workflow_node(
    state: AgentWorkflowState, llm_call: Callable[[str, Optional[str]], str]
) -> AgentWorkflowState:
    """Generate a schema-valid FaaSr JSON."""
    description = state.get("clarified_description", "")
    feedback = state.get("review_feedback", "")

    if feedback:
        description = (
            f"{description}\n\n"
            f"Additional feedback from the user:\n{feedback}"
        )

    example_json = _load_example_json()

    gi_files = state.get("global_input_files", []) or []
    gi_folders = state.get("global_input_folders", []) or []
    if gi_files or gi_folders:
        lines = ["The following pre-existing data is available to all agents via the global registry:"]
        for f in gi_files:
            lines.append(f"  - File pattern: {f}")
        for d in gi_folders:
            lines.append(f"  - Folder: {d}")
        lines.append("Agents should expect this data to be discoverable from the registry at runtime.")
        global_inputs_context = "\n".join(lines)
    else:
        global_inputs_context = "No pre-existing global input data specified."

    base_prompt = AGENT_WORKFLOW_GENERATION_PROMPT.format(
        description=description,
        global_inputs_context=global_inputs_context,
        gh_username=os.getenv("FAASR_GH_USERNAME", ""),
        action_repo=os.getenv("FAASR_ACTION_REPO", ""),
        s3_endpoint=os.getenv("FAASR_S3_ENDPOINT", ""),
        s3_bucket=os.getenv("FAASR_S3_BUCKET", ""),
        s3_region=os.getenv("FAASR_S3_REGION", ""),
        example_json=example_json,
    )

    validation_errors: list[str] = []
    faasr_json: Optional[dict] = None

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        prompt = base_prompt
        if validation_errors:
            error_block = "\n".join(f"  - {e}" for e in validation_errors)
            prompt += (
                f"\n\nAttempt {attempt}/{MAX_GENERATION_ATTEMPTS}. "
                f"Previous attempt had these schema violations — fix ALL of them:\n{error_block}"
            )
            logger.debug(f"[Retry {attempt}/{MAX_GENERATION_ATTEMPTS}] Fixing schema violations...")

        raw = llm_call(prompt, "You are a workflow architect. Output only valid JSON.")
        faasr_json = _extract_json(raw)

        debug_dir = Path(__file__).resolve().parents[3] / "workflows" / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        if faasr_json is None:
            debug_path = debug_dir / f"attempt-{attempt}-raw.txt"
            debug_path.write_text(raw)
            logger.debug(f"[Attempt {attempt}] Failed to parse JSON — saved raw response to {debug_path}")
            validation_errors = ["Response was not valid JSON"]
            continue

        debug_name = faasr_json.get("WorkflowName", "workflow")
        debug_path = debug_dir / f"{debug_name}-attempt-{attempt}.json"
        debug_path.write_text(json.dumps(faasr_json, indent=2))
        logger.debug(f"[Attempt {attempt}] Saved to {debug_path}")

        validation_errors = _validate_schema(faasr_json)
        if validation_errors:
            logger.debug(f"[Attempt {attempt}] Schema violations: {'; '.join(validation_errors[:3])}")
            continue

        validation_errors = _validate_dag(faasr_json)
        if not validation_errors:
            logger.debug(f"[Attempt {attempt}] Schema and DAG validation passed.")
            break

        logger.debug(f"[Attempt {attempt}] DAG violations: {'; '.join(validation_errors[:3])}")

    if validation_errors:
        state["generation_errors"] = [
            f"Validation failed after {MAX_GENERATION_ATTEMPTS} attempts:\n"
            + "\n".join(f"  - {e}" for e in validation_errors)
        ]
        state["faasr_json"] = {}
        return state

    # Check for name collision in workflows/ dir
    workflows_dir = Path(__file__).resolve().parents[3] / "workflows"
    original_name = faasr_json.get("WorkflowName", "workflow")
    unique_name = _unique_workflow_name(original_name, workflows_dir)
    if unique_name != original_name:
        faasr_json["WorkflowName"] = unique_name
        logger.debug(f"Workflow name '{original_name}' already exists — using '{unique_name}'")

    state["faasr_json"] = faasr_json
    state["generation_errors"] = []
    return state


def review_summary_node(state: AgentWorkflowState) -> AgentWorkflowState:
    """Print a readable summary of the workflow and ask the user for approval."""
    faasr = state.get("faasr_json", {})

    if not faasr:
        errors = state.get("generation_errors", [])
        logger.debug("No workflow was generated.")
        for e in errors:
            logger.debug(f"  Error: {e}")
        state["user_approved"] = False
        return state

    print("\n" + "=" * 60)
    print("  WORKFLOW SUMMARY")
    print("=" * 60)
    print(f"  Name:        {faasr.get('WorkflowName', 'N/A')}")
    print(f"  Entry Point: {faasr.get('FunctionInvoke', 'N/A')}")
    print(f"  Data Store:  {faasr.get('DefaultDataStore', 'N/A')}")
    print("-" * 60)

    action_list = faasr.get("ActionList", {})
    for i, (name, action) in enumerate(action_list.items(), 1):
        action_type = action.get("Type", "N/A")
        invoke_next = action.get("InvokeNext", [])
        arrow = " -> " + ", ".join(str(n) for n in invoke_next) if invoke_next else " -> [END]"
        print(f"\n  [{i}] {name}{arrow}")
        print(f"      Type: {action_type}")
        if action_type in ("Python", "R"):
            print(f"      FunctionName: {action.get('FunctionName', 'N/A')}")
        else:
            prompt_text = action.get("Arguments", {}).get("prompt", "")
            print(f"      Prompt: {prompt_text}")

    logger.debug("=" * 60)

    while True:
        response = input("\nApprove this workflow? (yes/no): ").strip().lower()
        if response in ("yes", "y"):
            state["user_approved"] = True
            state["review_feedback"] = ""
            break
        elif response in ("no", "n"):
            feedback = input("Provide feedback for regeneration: ").strip()
            state["user_approved"] = False
            state["review_feedback"] = feedback
            break
        else:
            logger.debug("Please enter 'yes' or 'no'.")

    return state
