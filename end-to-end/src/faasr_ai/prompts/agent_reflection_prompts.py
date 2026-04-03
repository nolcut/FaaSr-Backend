AGENT_VALIDATION_PROMPT_TEMPLATE = """
You are an agent workflow validation expert. Analyze the following FaaSr agentic workflow for correctness and quality.

# FaaSr Agent Runtime Semantics (read carefully before validating):
- Each agent action runs in an isolated container. It has NO prior context beyond its own prompt and the files it discovers from the S3 data registry at runtime.
- Agents write output files to a local staging directory. The FaaSr system AUTOMATICALLY uploads those files to S3 after the agent completes. Prompts that say "the system handles S3 storage automatically" are CORRECT — do not flag this as an error.
- InvokeNext defines which actions are triggered after an action completes. Multiple actions can list the same downstream action in their InvokeNext — FaaSr handles the invocation; do not flag this as a race condition unless you have specific evidence of a structural problem.
- Agents discover their input files by inspecting the S3 registry at runtime. Prompts should describe WHAT data to look for (structure, format, content), not WHERE it is stored.

# User's Clarified Description:
{clarified_description}

# Generated FaaSr Workflow JSON:
{faasr_json}

# Validation Criteria:

1. **Fidelity**:
   - Does the workflow accurately reflect the user's described goal, steps, and outputs?
   - Are the right number of actions present — not more, not fewer than the user described?
   - Do action names clearly correspond to the user's described steps?

2. **Prompt Quality**:
   - Is each action's Arguments.prompt detailed, self-contained, and actionable?
   - An agent executing a prompt has NO prior context beyond the prompt itself and files it discovers from the registry at runtime. Prompts must be fully self-explanatory.
   - Does each prompt clearly describe WHAT to do and what kind of data to produce?
   - Do NOT flag hardcoded external web URLs for robustness/fallback concerns — this is out of scope for workflow validation.
   - Do NOT flag prompts for missing I/O structure details, exact field names, JSON schemas, or ambiguous file format phrasing — agents inspect and parse actual files from the registry at runtime and can handle format discovery themselves.

3. **No S3 or Local File Path References**:
   - Prompts must NOT hardcode S3 keys, S3 folder paths, or local filesystem paths (e.g., /tmp/..., data/input.csv).
   - Agents discover their S3 input files automatically from the registry at runtime; output storage is handled by the system.
   - External web URLs (http:// or https://) for fetching data from public APIs or data sources are VALID and should NOT be flagged under this criterion.
   - Prompts should describe the TASK and expected data characteristics, not S3 storage locations.

3a. **No Dynamic Values in File Names**:
   - Output file names described in prompts MUST be static and generic. Prompts must NOT describe output files whose names contain dates, timestamps, months, years, invocation IDs, or any other run-specific values (e.g., "ndvi_raw_2024_01.tif" or "results_2026-03.csv"). Flag as CRITICAL if found.
   - Use generic descriptive names instead (e.g., "ndvi_raw.tif", "monthly_results.csv").

4. **Data Flow Coherence**:
   - Do predecessor actions produce data that successor actions can logically consume?
   - Are the described outputs of one action compatible with the described inputs of the next?
   - Does the InvokeNext DAG structure reflect the logical data dependencies?

5. **Coverage**:
   - Are all necessary steps present to accomplish the user's goal?
   - Are there any missing intermediate steps?
   - Are there any unnecessary or redundant actions that serve no unique purpose?
   - Are there any actions whose sole purpose is to upload or transfer files to S3? If so, flag as CRITICAL redundancy — FaaSr automatically uploads all staged files to S3 after each action completes. Such actions must be removed.

6. **Redundancy**:
   - No two actions should serve the same purpose or duplicate each other's functionality.

7. **Data Flow Completeness**:
   - Agents discover their input files from the S3 registry at runtime and can inspect file contents, formats, and structures dynamically. Do NOT flag prompts for not specifying exact field names, JSON schemas, file format choices, or I/O structure details.
   - Only flag a data flow issue when an upstream action clearly does NOT produce data that a downstream action logically needs (a genuine logical gap), not when a prompt omits format or schema specifications.

8. **Visualization Consistency**:
   - For actions that produce plots or figures, verify that all visual parameters within a single prompt are internally consistent (e.g., if facecolor='none' is specified, do not also specify a fill color for the same element). Flag contradictory parameter specifications as WARNING.

# Your Task:
Analyze the workflow and identify any issues or areas for improvement.

Return ONLY valid JSON in this exact format:
{{
  "has_issues": true/false,
  "issues": [
    {{
      "action_name": "ActionName or 'general'",
      "severity": "critical/warning/info",
      "category": "fidelity/prompt_quality/file_paths/data_flow/coverage/redundancy",
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


AGENT_REVISION_PROMPT_TEMPLATE = """
You are an agent workflow revision expert. Revise the FaaSr workflow JSON based on the validation feedback.

# FaaSr Agent Runtime Semantics (read carefully before revising):
- Agents write output files to a local staging directory. The FaaSr system AUTOMATICALLY uploads those files to S3. Prompts may correctly state this — do not remove or contradict it.
- InvokeNext fan-in (multiple predecessors invoking the same downstream action) is a valid FaaSr pattern. Do not restructure the DAG to avoid it unless there is a genuine logical problem.
- Agents discover S3 inputs from the registry at runtime — prompts should describe data by structure and content, not by S3 path or local filesystem path. External web URLs (http:// / https://) for fetching public data are valid and should not be removed.

# User's Clarified Description:
{clarified_description}

# Current FaaSr Workflow JSON:
{faasr_json}

# Validation Issues Found:
{issues_json}

# Suggestions for Improvement:
{suggestions_json}

# Your Task:
Revise the FaaSr workflow JSON to address ALL critical issues and as many warnings as possible.

Rules:
- Output ONLY the complete revised FaaSr JSON (no markdown fences, no explanation, no commentary).
- Preserve ALL top-level keys: WorkflowName, FunctionInvoke, ActionList, ActionContainers, ComputeServers, DataStores, PyPIPackageDownloads, DefaultDataStore, LoggingDataStore, FaaSrLog, InvocationIDFromDate.
- EVERY action in ActionList MUST keep its original "Type" (Python, R, or Agent) unchanged — do not change action types.
- Agent actions MUST keep "Arguments": {{"prompt": "<string>"}} — improve prompts where flagged. Python/R actions MUST keep "FunctionName".
- EVERY action MUST keep "FaaSServer" and "InvokeNext".
- ActionContainers, PyPIPackageDownloads, and FunctionGitRepo (if present) MUST stay in sync with ActionList keys.
- Do NOT hardcode file names, paths, or folders in prompts — agents discover I/O from the registry.
- Output file names in prompts MUST be static and generic — do NOT embed dates, timestamps, months, years, invocation IDs, or any run-specific values in file names (e.g. use "ndvi_raw.tif" not "ndvi_raw_2024_01.tif"). The workflow must produce the same file names on every run.
- Fix all dependency/DAG issues in InvokeNext.
- FunctionInvoke must name the action with no predecessors (entry point).
- Improve prompt clarity and detail where flagged, without adding file path references.

Return the complete revised FaaSr workflow as a single JSON object.
""".strip()
