from __future__ import annotations

from typing import Any, Callable, Dict, List, TypedDict
from datetime import datetime
import os
import textwrap
import logging
from dotenv import load_dotenv

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

from faasr_ai.prompts.coding_prompts import CODEGEN_PROMPT, DEBUG_PROMPT
from faasr_ai.utils.faasr_function_converter import convert_to_faasr_function

logger = logging.getLogger(__name__)


class CodingState(TypedDict, total=False):
    workflow_tasks: List[Dict[str, Any]]
    user_request: str

    generated_functions: Dict[str, str]   # task_id -> cell source (one function per cell)
    notebook_path: str

    coding_errors: List[str]
    generation_count: int
    data_folder = str
    functions_folder = str

# used for naming nb files
def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# double check that path exists
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    
def make_initial_coding_state(
    workflow_tasks: List[Dict[str, Any]],
    user_request: str,
    data_folder: str,
    functions_folder: str
) -> Dict[str, Any]:
    """
    Initialize a clean CodingState for the coding agent / LangGraph node.
    """
    return {
        # Inputs
        "workflow_tasks": workflow_tasks,
        "user_request": user_request,

        # Outputs (empty at start)
        "generated_functions": {},
        "notebook_path": "",

        # Debug / metadata
        "coding_errors": [],
        "generation_count": 0,
        "data_folder": data_folder,
        "functions_folder": functions_folder
    }

def _summarize_cell_outputs(cell: Dict[str, Any]) -> str:
    outs: List[str] = []
    for o in cell.get("outputs", []):
        ot = o.get("output_type")
        if ot == "stream":
            outs.append(o.get("text", ""))
        elif ot == "error":
            ename = o.get("ename", "")
            evalue = o.get("evalue", "")
            tb = "\n".join(o.get("traceback", [])[:10])
            outs.append(f"{ename}: {evalue}\n{tb}".strip())
        elif ot in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "text/plain" in data:
                outs.append(str(data["text/plain"]))
    return "\n".join([x for x in outs if x]).strip()

async def coding_node(
    state: CodingState,
    llm_call: Callable[..., Any],
    *,
    max_attempts: int = 5,
    timeout_s: int = 120
) -> Dict[str, Any]:
    """
    LangGraph:
      - For each task: LLM generates a self-contained code cell (one function)
      - Execute it
      - On failure: LLM revises code using error/output and retry
      - Commit only successful cells
      - Export notebook to output/ with datetime filename
    """
    print("\n=== Generating Code ===")
    
    tasks = state.get("workflow_tasks", [])
    user_request = state.get("user_request", "")

    generated_functions: Dict[str, str] = dict(state.get("generated_functions", {}))
    coding_errors: List[str] = list(state.get("coding_errors", []))
    generation_count: int = int(state.get("generation_count", 0))

    nb = new_notebook()
    nb.cells.append(new_markdown_cell(f"# Generated Workflow Notebook\n\nUser request:\n{user_request}\n"))

    client = NotebookClient(nb, timeout=timeout_s)
    client.create_kernel_manager()
    client.start_new_kernel()
    client.start_new_kernel_client()

    load_dotenv()
    folder = os.getenv("FAASR_ACTION_REPO", "turtorial").strip()
    
    try:
        for task in tasks:
            tid = str(task.get("task_id", ""))
            function_name = f"task_{tid}"
            
            print(f"\nWorking on task: {function_name}")

            task_json = str(task)

            # 1) generate
            prompt = CODEGEN_PROMPT.format(
                function_name=function_name,
                user_request=user_request,
                task_json=task_json,
                folder = folder
            )
            code = llm_call(prompt)
            generation_count += 1

            success = False
            last_err = ""

            # 2) execute + debug loop
            for attempt in range(1, max_attempts + 1):
                print("Attempt: ", attempt)
                nb.cells.append(new_code_cell(code))
                idx = len(nb.cells) - 1

                try:
                    await client.async_execute_cell(nb.cells[idx], cell_index=idx)
                    success = True
                    generated_functions[tid] = code
                    break

                except BaseException as e:
                    logger.debug(f"Code: {code}")
                    last_err = f"{type(e).__name__}: {e}"
                    out_txt = _summarize_cell_outputs(nb.cells[idx])
                    logger.debug(f"Output: {out_txt}")

                    # remove failed cell (do not commit)
                    nb.cells.pop()

                    if attempt < max_attempts:
                        debug_prompt = DEBUG_PROMPT.format(
                            function_name=function_name,
                            user_request=user_request,
                            task_json=task_json,
                            code=code,
                            error_text=(f"{last_err}\n\nOutputs:\n{out_txt}".strip()),
                        )
                        code = llm_call(debug_prompt)
                        generation_count += 1
                    else:
                        coding_errors.append(
                            f"Task {tid} failed after {max_attempts} attempts. Last error: {last_err}"
                        )

            if not success:
                raise RuntimeError(f"Stopping: task {tid} could not be executed successfully.")
            
            print("\n=== Converting function to FaaSr format ===")
            faasr_function = convert_to_faasr_function(textwrap.dedent(code), task, folder, llm_call)
            print(faasr_function)
            
            _ensure_dir(state.get("functions_folder", "functions/"))
            out_path = os.path.join(state.get("functions_folder", "functions/"), f"{function_name}.py")
            
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(faasr_function)
            
            print(f"Saved code to: {out_path}")

        _ensure_dir(state.get("data_folder", "data/"))
        out_path = os.path.join(state.get("data_folder", "data/"), f"generated_workflow_{_timestamp()}.ipynb")
        nbformat.write(nb, out_path)

        return {
            "generated_functions": generated_functions,
            "coding_errors": coding_errors,
            "generation_count": generation_count,
            "notebook_path": out_path,
        }

    finally:
        try:
            await client.km.shutdown_kernel()
        except Exception:
            pass
