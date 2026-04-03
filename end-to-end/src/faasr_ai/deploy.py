# src/faasr_ai/deploy.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from github import Github

from faasr_ai.agents.execute import upsert_file, run_cmd
from faasr_ai.tui.sync_tracker import run_sync_tracker, print_executive_summary


def prompt_deploy_choice() -> str:
    """Prompt the user for a deployment option."""
    print("\nDeployment options:")
    print("  1. Synchronous  (deploy and track execution live)")
    print("  2. Asynchronous (deploy and return immediately)")
    print("  3. Don't deploy (save workflow JSON only)")

    while True:
        choice = input("\nChoose (1/2/3): ").strip()
        if choice == "1":
            return "sync"
        if choice == "2":
            return "async"
        if choice == "3":
            return "none"
        print("Please enter 1, 2, or 3.")


def _save_workflow_json(faasr_json: dict, repo_root: Path) -> Path:
    """Save the workflow JSON locally under workflows/."""
    workflow_name = faasr_json.get("WorkflowName", "workflow")
    workflows_dir = repo_root / "workflows"
    workflows_dir.mkdir(exist_ok=True)
    out_path = workflows_dir / f"{workflow_name}.json"
    with open(out_path, "w") as f:
        json.dump(faasr_json, f, indent=2)
    print(f"\nWorkflow saved to {out_path}")
    return out_path


def _upload_workflow_to_github(workflow_path: Path) -> None:
    """Upload the workflow JSON to the GitHub Actions repo."""
    load_dotenv()

    token = os.getenv("GH_PAT", "").strip()
    gh_username = os.getenv("FAASR_GH_USERNAME", "").strip()
    gh_action_repo = os.getenv("FAASR_ACTION_REPO", "").strip()
    branch = os.getenv("FAASR_ACTION_BRANCH", "main").strip()
    repo_name = f"{gh_username}/{gh_action_repo}"

    if not token or not repo_name:
        print("Missing GH_PAT or GitHub repo config. Skipping upload.")
        return

    gh = Github(token)
    repo = gh.get_repo(repo_name)

    repo_path = f"workflows/{workflow_path.name}"
    print(f"\nUploading {repo_path} to {repo_name}...")
    upsert_file(repo, repo_path, workflow_path, branch=branch)
    print("Upload complete.")


def _register_and_invoke(workflow_path: Path, repo_root: Path) -> None:
    """Register and invoke the workflow using shell scripts."""
    load_dotenv()

    token = os.getenv("GH_PAT", "").strip()
    gh_username = os.getenv("FAASR_GH_USERNAME", "").strip()
    gh_action_repo = os.getenv("FAASR_ACTION_REPO", "").strip()
    branch = os.getenv("FAASR_ACTION_BRANCH", "main").strip()
    repo_name = f"{gh_username}/{gh_action_repo}"

    extra_env = {
        "GH_PAT": token,
        "GITHUB_REPOSITORY": repo_name,
        "GITHUB_REF_NAME": branch,
    }

    register_script = repo_root / "register_workflow.sh"
    rel_workflow = str(workflow_path.relative_to(repo_root))

    if register_script.exists():
        print("\nRegistering workflow...")
        run_cmd(
            ["bash", str(register_script), "--workflow-file", rel_workflow, "-c"],
            cwd=repo_root,
            extra_env=extra_env,
        )
    else:
        print(f"Warning: {register_script} not found, skipping registration.")


def _invoke_sync(workflow_path: Path, repo_root: Path, faasr_json: dict | None = None, cache_statuses: dict | None = None) -> "WorkflowRunner | None":
    """Invoke the workflow and monitor it with the Rich TUI. Returns the runner if successful."""
    from framework.workflow_runner import WorkflowRunner
    from framework.s3_client import FaaSrS3Client

    import sys
    rel_workflow = str(workflow_path.relative_to(repo_root))
    original_argv = sys.argv
    sys.argv = ["invoke_workflow", "--workflow-file", rel_workflow]

    # invoke_workflow.main() reads GITHUB_REPOSITORY as owner/repo — set it if missing
    gh_username = os.getenv("FAASR_GH_USERNAME", "").strip()
    gh_action_repo = os.getenv("FAASR_ACTION_REPO", "").strip()
    if not os.getenv("GITHUB_REPOSITORY") and gh_username and gh_action_repo:
        os.environ["GITHUB_REPOSITORY"] = f"{gh_username}/{gh_action_repo}"
    if not os.getenv("GITHUB_REF_NAME"):
        os.environ["GITHUB_REF_NAME"] = os.getenv("FAASR_ACTION_BRANCH", "main")

    runner = None
    s3_client = None
    try:
        runner = WorkflowRunner.trigger_workflow(
            workflow_file=rel_workflow,
            stream_logs=False,
        )

        # Create S3 client for code artifact display if possible
        try:
            s3_client = FaaSrS3Client(
                workflow_data=runner._faasr_payload,
                access_key=os.getenv("S3_AccessKey", ""),
                secret_key=os.getenv("S3_SecretKey", ""),
            )
        except Exception:
            s3_client = None

        run_sync_tracker(runner, s3_client, faasr_json, cache_statuses=cache_statuses)
        if runner.monitoring_complete:
            runner.cleanup()
        return runner
    except Exception as e:
        if runner:
            runner.cleanup()
        print(f"Error during workflow execution: {e}")
        return None
    finally:
        sys.argv = original_argv


def _invoke_async(workflow_path: Path, repo_root: Path) -> None:
    """Invoke the workflow asynchronously (fire-and-forget)."""
    load_dotenv()

    token = os.getenv("GH_PAT", "").strip()
    gh_username = os.getenv("FAASR_GH_USERNAME", "").strip()
    gh_action_repo = os.getenv("FAASR_ACTION_REPO", "").strip()
    branch = os.getenv("FAASR_ACTION_BRANCH", "main").strip()
    repo_name = f"{gh_username}/{gh_action_repo}"

    extra_env = {
        "GH_PAT": token,
        "GITHUB_REPOSITORY": repo_name,
        "GITHUB_REF_NAME": branch,
    }

    invoke_script = repo_root / "invoke_workflow.sh"
    rel_workflow = str(workflow_path.relative_to(repo_root))

    if invoke_script.exists():
        print("\nInvoking workflow...")
        run_cmd(
            ["bash", str(invoke_script), "--workflow-file", rel_workflow],
            cwd=repo_root,
            extra_env=extra_env,
        )
        print("Workflow invoked. Check GitHub Actions for progress.")
    else:
        print(f"Warning: {invoke_script} not found.")


def handle_deploy(choice: str, faasr_json: dict, cache_statuses: dict | None = None) -> "WorkflowRunner | None":
    """Handle deployment based on user choice (sync/async/none). Returns runner if sync, None otherwise."""
    from framework.workflow_runner import WorkflowRunner

    # Determine repo root (faasr-agent-system/)
    repo_root = Path(__file__).resolve().parents[2]

    workflow_path = _save_workflow_json(faasr_json, repo_root)

    if choice == "none":
        return None

    _upload_workflow_to_github(workflow_path)
    _register_and_invoke(workflow_path, repo_root)

    if choice == "sync":
        return _invoke_sync(workflow_path, repo_root, faasr_json, cache_statuses=cache_statuses)
    elif choice == "async":
        _invoke_async(workflow_path, repo_root)
        return None

    return None
