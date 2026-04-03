VALIDATION_PROMPT_TEMPLATE = """
You are a workflow validation expert. Analyze the following workflow tasks for correctness and quality.

# User Request:
{user_request}

# Generated Workflow Tasks:
{tasks_json}

# Validation Criteria:

1. **Task Dependencies**:
   - Check if dependent_task_ids are valid and form a proper DAG (no cycles).
   - dependent_task_ids should be non-empty only when the current task requires one or more previous tasks to be completed first.
   - Do not include future tasks in dependent_task_ids. Dependencies must always point backward to prerequisite (already-defined) tasks, never forward to tasks that depend on the current task.
   - Ensure there is exactly one entrypoint task (a single root node with no dependencies).
   - Ensure all tasks are reachable from the entrypoint (no disconnected subgraphs or orphan tasks).

2. **Redundancy Check**:
   - Ensure there are no redundant or duplicate tasks.
   - Verify that each task serves a unique and necessary purpose in achieving the workflow goal.
   - Remove tasks that replicate functionality already covered by another task.

3. **File Consistency**:
   - Verify that outputs from one task correctly match the declared inputs of dependent tasks.
   - Ensure file names, formats, and paths are consistent across tasks.

4. **Completeness**:
   - Check whether the workflow fully addresses the user request.
   - Ensure no required intermediate steps are missing.

5. **Task Instructions**:
   - Verify instructions are clear, specific, and actionable.
   - Avoid ambiguous or underspecified execution steps.

6. **Input/Output Clarity**:
   - Ensure inputs and outputs are well-named, descriptive, and consistent.
   - Avoid redundant or unused inputs/outputs.


# Your Task:
Analyze the workflow and identify any issues or areas for improvement.

Return ONLY valid JSON in this exact format:
{{
  "has_issues": true/false,
  "issues": [
    {{
      "task_id": "ID or 'general'",
      "severity": "critical/warning/info",
      "category": "dependency/file_consistency/completeness/clarity",
      "description": "Clear description of the issue"
    }}
  ],
  "suggestions": [
    "Specific suggestion for improvement 1",
    "Specific suggestion for improvement 2"
  ],
  "overall_quality": "poor/fair/good/excellent"
}}

Critical issues MUST be fixed. Warnings should be addressed if possible.
If there are NO issues, set has_issues to false and return empty arrays for issues and suggestions.
""".strip()


REVISION_PROMPT_TEMPLATE = """
You are a workflow revision expert. Revise the workflow tasks based on the validation feedback.

# User Request:
{user_request}

# Current Workflow Tasks:
{tasks_json}

# Validation Issues Found:
{issues_json}

# Suggestions for Improvement:
{suggestions_json}

# Your Task:
Revise the workflow tasks to address ALL critical issues and as many warnings as possible.

Rules:
- Output ONLY valid JSON (no markdown, no extra text).
- Top-level MUST be a JSON array.
- Each task MUST have: task_id, dependent_task_ids, instruction, inputs, outputs.
- Fix all dependency issues
- Ensure file name consistency
- Improve clarity where suggested
- Maintain the same task structure unless changes are necessary

Return the complete revised workflow as a JSON array.
""".strip()
