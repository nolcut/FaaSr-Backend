# src/faasr_ai/agents/execute.py
from __future__ import annotations

import os
import subprocess
import logging
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from github import Github, GithubException

logger = logging.getLogger(__name__)


def get_file_sha(repo, file_path: str, branch: str = "main") -> str | None:
    """Returns the blob SHA if the file exists on GitHub, otherwise None."""
    try:
        contents = repo.get_contents(file_path, ref=branch)
        return contents.sha
    except GithubException as e:
        if getattr(e, "status", None) == 404:
            return None
        raise


def upsert_file(repo, repo_path: str, local_path: Path, branch: str = "main") -> None:
    """Create the file if it doesn't exist; otherwise update it."""
    sha = get_file_sha(repo, repo_path, branch=branch)

    content = local_path.read_text(encoding="utf-8")
    commit_message = f"update {repo_path}"

    if sha is None:
        repo.create_file(repo_path, commit_message, content, branch=branch)
    else:
        repo.update_file(repo_path, commit_message, content, sha, branch=branch)


def upload_tree(repo, local_root: Path, repo_root: str, branch: str) -> list[str]:
    """Upload/update all files under local_root into repo_root on GitHub."""
    touched: list[str] = []
    if not local_root.exists():
        return touched

    for file_path in local_root.rglob("*"):
        if file_path.is_file():
            rel = file_path.relative_to(local_root).as_posix()
            repo_path = f"{repo_root}/{rel}"
            upsert_file(repo, repo_path, file_path, branch=branch)
            touched.append(repo_path)

    return touched


def run_cmd(cmd: list[str], *, cwd: Path | None = None, extra_env: dict | None = None) -> str:
    """Run a command, print output in real-time, and return combined stdout/stderr."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    
    logger.debug(f"Running command: {' '.join(cmd)}")
    logger.debug(f"Working directory: {cwd}")
    
    # Run with real-time output
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        env=env
    )
    
    # Collect output while printing it
    output_lines = []
    for line in p.stdout:
        logger.debug(line.rstrip())  # Log in real-time
        output_lines.append(line)
    
    p.wait()  # Wait for process to complete
    
    output = ''.join(output_lines)
    
    logger.debug(f"Return code: {p.returncode}")
    
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")
    
    return output


def execute_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node:
    - uploads functions/* and workflows/* to GitHub Actions repo
    - registers workflow
    - invokes workflow

    Env required:
      GH_PAT, GITHUB_REPOSITORY
    Optional:
      GITHUB_BRANCH (default main)

    Writes results under state["execute"].
    """
    load_dotenv()

    gh_username = os.getenv("FAASR_GH_USERNAME", "").strip()
    gh_action_repo = os.getenv("FAASR_ACTION_REPO", "").strip()
    token = os.getenv("GH_PAT", "").strip()
    repo_name = f"{gh_username}/{gh_action_repo}"
    branch = os.getenv("FAASR_ACTION_BRANCH", "main").strip()

    if not token:
        state["execute"] = {"ok": False, "error": "GH_PAT is empty"}
        return state
    if not repo_name:
        state["execute"] = {"ok": False, "error": "GITHUB_REPOSITORY is empty"}
        return state

    # Determine repo root (assuming this file is in src/faasr_ai/agents/)
    repo_root = Path(__file__).resolve().parents[3]

    # 1) Upload/update to GitHub repo
    gh = Github(token)
    repo = gh.get_repo(repo_name)

    uploaded: list[str] = []
    logger.debug(f"Uploading functions to {repo_name}/functions")
    uploaded += upload_tree(repo, repo_root / "functions", "functions", branch)
    logger.debug(f"Uploading workflow file to {repo_name}/workflows")
    uploaded += upload_tree(repo, repo_root / "workflows", "workflows", branch)

    # 2) Register + Invoke workflow (local scripts)
    register_script = repo_root / "register_workflow.sh"
    invoke_script = repo_root / "invoke_workflow.sh"
    workflow_file = "workflows/tutorial.json"

    if not register_script.exists():
        state["execute"] = {"ok": False, "error": f"{register_script} not found"}
        return state
    if not invoke_script.exists():
        state["execute"] = {"ok": False, "error": f"{invoke_script} not found"}
        return state
    if not (repo_root / workflow_file).exists():
        state["execute"] = {"ok": False, "error": f"{workflow_file} not found"}
        return state

    extra_env = {
        "GH_PAT": token,
        "GITHUB_REPOSITORY": repo_name,
        "GITHUB_REF_NAME": branch,
    }

    logger.debug("Registering Workflow")
    register_out = run_cmd(
        ["bash", str(register_script), "--workflow-file", workflow_file, "-c"],
        cwd=repo_root,
        extra_env=extra_env,
    )

    logger.debug("Invoking Workflow")
    invoke_out = run_cmd(
        ["bash", str(invoke_script), "--workflow-file", workflow_file],
        cwd=repo_root,
        extra_env=extra_env,
    )

    state["execute"] = {
        "ok": True,
        "uploaded_paths": uploaded,
        "register_output": register_out,
        "invoke_output": invoke_out,
        "repo": repo_name,
        "branch": branch,
        "workflow_file": workflow_file,
    }
    return state