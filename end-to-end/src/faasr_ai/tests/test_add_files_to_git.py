import os
from pathlib import Path

from github import Github, GithubException
from dotenv import load_dotenv


def get_file_sha(repo, file_path: str, branch: str = "main") -> str | None:
    """
    Returns the blob SHA if the file exists on GitHub, otherwise None.
    """
    try:
        contents = repo.get_contents(file_path, ref=branch)
        return contents.sha
    except GithubException as e:
        if getattr(e, "status", None) == 404:
            return None
        raise


def upsert_file(repo, repo_path: str, local_path: Path, branch: str = "main") -> None:
    """
    Create the file if it doesn't exist; otherwise update it.
    """
    sha = get_file_sha(repo, repo_path, branch=branch)

    with open(local_path, "r", encoding="utf-8") as f:
        content = f.read()

    commit_message = f"update {repo_path}"

    if sha is None:
        repo.create_file(repo_path, commit_message, content, branch=branch)
        print(f"Created {repo_path}")
    else:
        repo.update_file(repo_path, commit_message, content, sha, branch=branch)
        print(f"Updated {repo_path}")


def main():
    load_dotenv()

    token = os.getenv("GH_PAT", "").strip()
    if not token:
        print("No Github PAT (GH_PAT is empty)")
        return

    gh = Github(token)

    repo_name = os.getenv("GITHUB_REPOSITORY", "").strip()
    if not repo_name:
        print("No Github repo (GITHUB_REPOSITORY is empty)")
        return

    branch = os.getenv("GITHUB_BRANCH", "main").strip() or "main"
    repo = gh.get_repo(repo_name)

    # Upload/update functions/*
    functions_folder = Path("functions")
    if functions_folder.exists():
        for file_path in functions_folder.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(functions_folder).as_posix()
                repo_path = f"functions/{rel}"
                print("Checking:", repo_path)
                upsert_file(repo, repo_path, file_path, branch=branch)

    # Upload/update workflows/*
    workflows_folder = Path("workflows")
    if workflows_folder.exists():
        for file_path in workflows_folder.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(workflows_folder).as_posix()
                repo_path = f"workflows/{rel}"
                print("Checking:", repo_path)
                upsert_file(repo, repo_path, file_path, branch=branch)


if __name__ == "__main__":
    main()
