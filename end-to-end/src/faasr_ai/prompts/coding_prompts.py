CODEGEN_PROMPT = """\
Write ONE Python code cell that defines EXACTLY ONE function implementing the workflow task, AND then calls the function.

Hard rules:
- Output ONLY Python code. No markdown. No backticks.
- The cell MUST be FULLY SELF-CONTAINED:
  - Include ALL required imports inside the cell (or inside the function).
  - Define ANY helper functions it uses inside the same cell.
  - Do NOT rely on variables, imports, or functions from any other cell.
- The cell MUST define EXACTLY ONE top-level function.
- The function name MUST be: {function_name}

Function signature rules (MUST follow exactly):
- The FIRST parameter MUST be: folder (string)
  - Default value MUST be: "{folder}"
- Then include input file parameters (in order):
  - For each item in task["inputs"] (a list of filenames), include a parameter named input1, input2, ...
  - Default values MUST be the exact filenames from task["inputs"].
- Then include output file parameters (in order):
  - For each item in task["outputs"] (a list of filenames), include a parameter named output1, output2, ...
  - Default values MUST be the exact filenames from task["outputs"].

I/O path rules (MUST follow exactly):
- The function MUST create folder if it does not exist.
- For EVERY read/write of task inputs and outputs, the code MUST join the folder with the filename using os.path.join:
  - Input path for input_i MUST be: os.path.join(folder, input_i)
  - Output path for output_j MUST be: os.path.join(folder, output_j)
- The function MUST read from the joined input paths (never from bare filenames).
- The function MUST write to the joined output paths (never to bare filenames).
- Do NOT assume filenames already include the folder; always join.

Behavior rules:
- The function MUST perform the task described in task["instruction"].
- DO NOT generate sample data if not explicitly asked.
- The function MUST write ALL declared outputs exactly (filenames matter).
- Inputs are local files already present by those names (if any), located under folder.
- Do NOT do any network calls.
- Do NOT require user interaction.
- Avoid printing large data; minimal logging is OK.
- ONLY include inputs and outputs that are specified in Task JSON in function arguments.

Execution rule (MUST):
- After defining the function, the cell MUST call it ONCE using ONLY its default arguments.
  - Example shape: {function_name}("{folder}", <default inputs...>, <default outputs...>)
- The call result MAY be ignored.
- Do NOT require a return statement unless the task explicitly specifies one.

User request:
{user_request}

Task JSON:
{task_json}

folder:
{folder}

Return ONLY the Python code cell content. REMEMBER: MUST call the function once inside the cell after function definition.
""".strip()


DEBUG_PROMPT = """\
The previous code cell failed when executed in a Jupyter kernel.

Hard rules:
- Output ONLY Python code. No markdown. No backticks.
- The cell MUST define exactly one function.
- The function name MUST be: {function_name}
- Keep the same I/O behavior required by the task.
- Fix the error and any obvious issues.
- Do not add extra unrelated features.

User request:
{user_request}

Task JSON:
{task_json}

Failed code:
{code}

Execution error / outputs:
{error_text}

Return ONLY the corrected code cell content.
""".strip()

CODEGEN_SYSTEM = "You are a coding agent that writes correct, runnable Python code for a Jupyter notebook."

DEBUG_SYSTEM = "You are a senior Python debugger. Fix the code so it runs correctly in a Jupyter notebook."