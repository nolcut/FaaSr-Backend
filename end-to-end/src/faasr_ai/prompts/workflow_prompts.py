WORKFLOW_PROMPT = """
# Context (structured request):
{context_json}

# Task:
Create a plan to achieve the goal. A plan consists of 1 to {max_tasks} tasks.

Rules:
- Output ONLY valid JSON (no markdown, no extra text).
- Top-level MUST be a JSON array.
- Each task MUST have: task_id, dependent_task_ids, instruction, inputs, outputs.
- task_id should be unique (use ordinal strings "1", "2", ...).
- dependent_task_ids must reference earlier tasks only.
- instruction should be one short phrase or sentence.
- inputs: array of file names that this task reads (can be empty for initial tasks). Use descriptive names like "dataset1.csv", "model.pkl", etc.
- outputs: array of file names that this task produces. Use descriptive names with appropriate extensions.
- File names MUST be static and generic — do NOT embed dates, timestamps, months, years, or any dynamic values in filenames (e.g. use "ndvi_clipped.tif" not "ndvi_clipped_2026_01.tif"). The workflow must be reusable across different runs.
- File names should be consistent across tasks - if task 1 outputs "dataset1.csv", task 2 should reference "dataset1.csv" in its inputs.

Return a JSON array like:
[
  {{
    "task_id": "1",
    "dependent_task_ids": [],
    "instruction": "...",
    "inputs": ["input_file.csv"],
    "outputs": ["output_file.csv"]
  }}
]
""".strip()
