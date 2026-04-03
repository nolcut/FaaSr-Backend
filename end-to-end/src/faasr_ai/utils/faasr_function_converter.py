import ast
from typing import Any, Dict, List, Optional, Callable
import os
from faasr_ai.prompts.faasr_function_convert import FAASR_SYSTEM_PROMPT, USER_PROMPT

def convert_to_faasr_function(
    code: str,
    task: Dict[str, Any],
    folder: str,
    llm_call: Callable[[str, Optional[str]], str],
) -> str:
    inputs: List[str] = list(task.get("inputs") or [])
    outputs: List[str] = list(task.get("outputs") or [])

    prompt = USER_PROMPT.format(
        input_hint=", ".join(inputs) if inputs else "auto-detect from code",
        output_hint=", ".join(outputs) if outputs else "auto-detect from code",
        folder=folder,
        code=code,
    )

    converted = llm_call(prompt, FAASR_SYSTEM_PROMPT).strip()

    if converted.startswith("```"):
        lines = converted.splitlines()
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        converted = "\n".join(lines[start:end]).strip()

    return converted