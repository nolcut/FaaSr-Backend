# src/faasr_ai/prompts/agent_workflow_prompts.py
from __future__ import annotations

AGENT_WORKFLOW_GENERATION_PROMPT = """
You are an expert at designing FaaSr agentic workflows. Given a user's detailed workflow
description, generate a complete FaaSr workflow JSON where every action has Type "Agent".

Each agent action receives a natural-language prompt in Arguments.prompt. The prompt must be
detailed and self-contained; the agent executing it has no prior context beyond the prompt
itself and the files it discovers at runtime via the data registry.

IMPORTANT: Do NOT hardcode specific file names, folder names, or paths in prompts. Agents
discover their input files automatically from the registry at runtime and output storage is
handled by the system. Prompts should describe WHAT to do (the task, data transformation, and
expected outputs); not WHERE specific files are. Any output file names mentioned in prompts
must be static and generic — do NOT embed dates, timestamps, invocation IDs, or any other
run-specific values in file names. Workflows must produce the same file names on every run.

## User Description:
{description}

## Global Input Data:
{global_inputs_context}

## Infrastructure Configuration:
- GitHub Username: {gh_username}
- Action Repository: {action_repo}
- S3 Endpoint: {s3_endpoint}
- S3 Bucket: {s3_bucket}
- S3 Region: {s3_region}
- Agent Container Image: ghcr.io/nolcut/full-agent-gh:agents

## Example workflow (structural reference only — do NOT copy its domain, topic, steps, or approach):
{example_json}

## Rules:
1. WorkflowName must match pattern ^[a-zA-Z][a-zA-Z0-9-]*$ (PascalCase with hyphens, no spaces)
2. Action names must match the same pattern ^[a-zA-Z][a-zA-Z0-9-]*$
3. EVERY action in ActionList MUST include "Type": "Agent" explicitly — omitting this field causes schema validation failure
4. Agent actions MUST NOT include "FunctionName" — that field is only for Python/R actions
5. EVERY action MUST include "Arguments": {{"prompt": "<detailed non-empty string>"}}
6. EVERY action MUST include "FaaSServer": "GH" and "InvokeNext": [...]
7. FunctionInvoke must name the first action in the DAG (entry point with no predecessors)
8. InvokeNext defines DAG edges — terminal action(s) must have InvokeNext: []
9. Each prompt must describe the task and expected outputs clearly — do NOT reference any file names, folder names, or paths. Agents discover input files from the registry at runtime and the system handles output storage automatically.
10. Data flows between actions via the S3 registry automatically — describe what kind of data each action produces and consumes, not where it is stored
10a. Output file names produced by agents MUST be static and generic — do NOT embed dates, timestamps, months, years, invocation IDs, or any other dynamic/run-specific values in filenames (e.g. use "ndvi_raw.tif" not "ndvi_raw_2024_01.tif", use "results.csv" not "results_2026-03.csv"). Workflows must produce the same file names on every run.
11. Include ActionContainers mapping every action to "ghcr.io/nolcut/full-agent-gh:agents"
12. Include ComputeServers with "GH" configured for GitHubActions using the provided username and repo
13. Include DataStores with "S3" configured using the provided endpoint, bucket, and region with "Writable": "TRUE"
14. Include PyPIPackageDownloads listing required Python packages for each action
15. Include InvocationIDFromDate: "%Y-%m-%d-%H-%M-%S"
16. Include DefaultDataStore: "S3", LoggingDataStore: "S3", FaaSrLog: "FaaSrLog"
17. Output ONLY valid JSON. No markdown fences, no explanation, no commentary.
18. Do NOT include GlobalInputFiles or GlobalInputFolders in your JSON output -- these are injected by the system. If global input data is listed above, agent prompts should describe that pre-existing data is available via the registry at the start of the workflow.
19. Do NOT generate an action whose sole purpose is to upload files to S3. FaaSr automatically uploads every file written to the local staging directory to S3 after each action completes. A separate "upload" or "transfer to S3" action is always redundant and must not be included.
20. When prompts specify plot or visualization parameters (e.g., matplotlib edgecolor, facecolor, fill), ensure all related parameters are consistent and non-contradictory within the same prompt (e.g., do not specify both "no fill" and a fill color for the same element).
""".strip()


