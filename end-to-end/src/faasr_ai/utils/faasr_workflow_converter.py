from __future__ import annotations

from typing import Any, Dict, List


def _action_id(task_id: str) -> str:
    """Convert task_id to action_id format."""
    return f"task-{task_id}"


def _build_invoke_next(tasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    For each task_id, compute next task_ids (reverse dependency edges).
    """
    next_map: Dict[str, List[str]] = {t["task_id"]: [] for t in tasks}
    
    for t in tasks:
        tid = t["task_id"]
        deps = t.get("dependent_task_ids") or []
        for d in deps:
            if d in next_map:
                next_map[d].append(tid)
    
    return next_map


def tasks_to_faasr_workflow(
    tasks: List[Dict[str, Any]],
    github_username: str,
    action_repo_name: str,
    branch_name: str = "main",
    s3_endpoint: str = "https://play.min.io",
    s3_bucket: str = "faasr",
    s3_region: str = "us-east-1",
    workflow_name: str = "workflow",
    action_container_image: str = "ghcr.io/faasr/github-actions-python:latest",
    function_repo_default: str = "laurakuo1006/tutorial",
) -> Dict[str, Any]:
    """
    Convert a task DAG into a FaaSr workflow dict.

    Simple format conversion - no validation or topological sorting.
    Tasks are processed in the order they appear in the input list.
    """
    if not tasks:
        raise ValueError("Tasks list cannot be empty")

    next_map = _build_invoke_next(tasks)

    # Entry action = first task with no dependencies
    entry_task = next((t for t in tasks if not (t.get("dependent_task_ids") or [])), tasks[0])
    entry_task_id = entry_task["task_id"]
    entry_action_id = _action_id(entry_task_id)

    action_list: Dict[str, Any] = {}
    function_git_repo: Dict[str, str] = {}
    action_containers: Dict[str, str] = {}

    for t in tasks:
        tid = t["task_id"]
        aid = _action_id(tid)
        function_name = f"task_{tid}"
        
        function_git_repo[function_name] = function_repo_default
        action_containers[aid] = action_container_image

        # Expand inputs/outputs into input1/input2/... and output1/output2/...
        args: Dict[str, Any] = {
            # requested: set folder to action repo name
            "folder": action_repo_name
        }

        inputs = t.get("inputs", []) or []
        for i, val in enumerate(inputs, start=1):
            args[f"input{i}"] = val

        outputs = t.get("outputs", []) or []
        for i, val in enumerate(outputs, start=1):
            args[f"output{i}"] = val

        action_list[aid] = {
            "Arguments": args,
            "InvokeNext": [_action_id(x) for x in next_map.get(tid, [])],
            "FaaSServer": "GH",
            "Type": "Python",
            "FunctionName": function_name,
        }

    return {
        "ActionList": action_list,
        "ComputeServers": {
            "GH": {
                "FaaSType": "GitHubActions",
                "UserName": github_username,
                "UseSecretStore": True,
                "ActionRepoName": action_repo_name,
                "Branch": branch_name,
            }
        },
        "DataStores": {
            "S3": {
                "Endpoint": s3_endpoint,
                "Bucket": s3_bucket,
                "Region": s3_region,
            }
        },
        "ActionContainers": action_containers,
        "FunctionInvoke": entry_action_id,
        "DefaultDataStore": "S3",
        "FunctionGitRepo": function_git_repo,
        "LoggingDataStore": "S3",
        "FaaSrLog": "FaaSrLog",
        "WorkflowName": workflow_name,
    }