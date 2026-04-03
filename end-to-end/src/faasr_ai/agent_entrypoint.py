# src/faasr_ai/agent_entrypoint.py
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

# Set FAASR_DEBUG=1 in your environment to enable verbose debug logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("FAASR_DEBUG") else logging.WARNING,
    format="%(name)s [%(levelname)s] %(message)s",
)

from faasr_ai.credentials import ensure_credentials
from faasr_ai.deploy import handle_deploy, prompt_deploy_choice
from faasr_ai.entrypoint import make_llm_call
from faasr_ai.agent_orchestrator import build_agent_orchestrator
from faasr_ai.vim_editor import open_vim_for_input


VIM_HEADER = """\
# Describe your workflow below.
# Each line starting with # is a comment and will be ignored.
#
# Be as detailed as possible: what data sources to use,
# what processing steps to perform, and what outputs to produce.
#
# The system will clarify any ambiguities before generating the workflow.
#
# Save and quit (:wq) when done.

"""


def main():
    print("Welcome to the FaaSr Workflow Generator.\n")

    # Step 1: Ensure credentials are configured
    ensure_credentials()

    # Step 2: Load existing or create new workflow
    mode, value = _prompt_load_or_create()

    def _t(var: str, default: float) -> float:
        return float(os.getenv(var, str(default)))

    llm_call_clarify  = make_llm_call("anthropic", _t("ANTHROPIC_CLARIFICATION_TEMPERATURE", 0.3))

    if mode == "workflow":
        faasr_json = value
        description = ""
    else:
        if mode == "prompt":
            prompt_path, initial_text = value
            print("Opening vim for your workflow description...")
            description = open_vim_for_input(initial_content=initial_text)
            if not description.strip():
                print("No description provided. Exiting.")
                return
            # Save edits back to the prompt file
            prompt_path.write_text(description + "\n")
        else:
            # Open vim for the workflow description
            print("Opening vim for your workflow description...")
            description = open_vim_for_input(header=VIM_HEADER)
            if not description.strip():
                print("No description provided. Exiting.")
                return

        # Step 3-5: Run the orchestrator (clarify -> generate -> review)
        llm_call_workflow = make_llm_call("anthropic", _t("ANTHROPIC_WORKFLOW_TEMPERATURE",      0.2))
        llm_call_reflect  = make_llm_call("anthropic", _t("ANTHROPIC_REFLECTION_TEMPERATURE",    0.1))

        app = build_agent_orchestrator(llm_call_clarify, llm_call_workflow, llm_call_reflect)
        final_state = app.invoke({
            "user_description": description,
        })

        faasr_json = final_state.get("faasr_json", {})
        if not faasr_json:
            errors = final_state.get("generation_errors", [])
            print("\nFailed to generate workflow.")
            if errors:
                for e in errors:
                    print(f"  - {e}")
            return

    # Step 6: Show cache status and allow invalidation before deploy
    cache_manager = _try_build_cache_manager(faasr_json)
    cache_statuses = None
    if cache_manager is not None:
        cache_statuses = _prompt_cache_status(cache_manager, faasr_json)

    # Step 7: Deploy
    deploy_choice = prompt_deploy_choice()
    runner = handle_deploy(deploy_choice, faasr_json, cache_statuses=cache_statuses)

    # Step 8: Oversight (if sync deployment completed)
    if runner is not None:
        from framework.utils.enums import FunctionStatus
        statuses = runner.get_function_statuses()
        failed = [n for n, s in statuses.items() if s == FunctionStatus.FAILED]
        if failed:
            print(f"\nActions failed: {', '.join(failed)}")
            choice = input("Press Enter to launch oversight agent (or q to quit): ").strip().lower()
            if choice == "q":
                print("\nDone.")
                return

        try:
            from faasr_ai.agents.agent_oversight import OversightAgent
            from framework.s3_client import FaaSrS3Client

            print("\n" + "=" * 60)
            print("WORKFLOW OVERSIGHT")
            print("=" * 60)

            while True:
                s3_client = FaaSrS3Client(
                    workflow_data=runner._faasr_payload,
                    access_key=os.getenv("S3_AccessKey", ""),
                    secret_key=os.getenv("S3_SecretKey", ""),
                )
                oversight_cache = _try_build_cache_manager(faasr_json, s3_client=s3_client)
                agent = OversightAgent(runner, faasr_json, llm_call_clarify, s3_client, user_prompt=description, cache_manager=oversight_cache, cache_snapshot=cache_statuses)
                rerun_json = agent.run()

                if rerun_json is None:
                    break

                faasr_json = rerun_json
                # Recompute cache statuses for the updated workflow
                rerun_cache_statuses = None
                rerun_cache_manager = _try_build_cache_manager(faasr_json, s3_client=s3_client)
                if rerun_cache_manager is not None:
                    rerun_cache_statuses = _prompt_cache_status(rerun_cache_manager, faasr_json)
                runner = handle_deploy("sync", faasr_json, cache_statuses=rerun_cache_statuses)
                cache_statuses = rerun_cache_statuses
                if runner is None:
                    print("Rerun failed. Exiting oversight.")
                    break

        except Exception as e:
            print(f"Error launching oversight agent: {e}")

    print("\nDone.")


def _prompt_load_or_create() -> tuple[str, Any]:
    """Prompt the user to choose how to start.

    Returns:
        ("workflow", dict)  — load existing workflow JSON, skip orchestrator
        ("new", None)       — open vim for description, run orchestrator
        ("prompt", (Path, str)) — load description from prompts/ file, run orchestrator
    """
    print("  1. Load existing workflow")
    print("  2. Create new workflow")
    print("  3. Load prompt from file")

    while True:
        try:
            choice = input("\nChoose (1/2/3): ").strip()
        except (EOFError, KeyboardInterrupt):
            return ("new", None)
        if choice in ("1", "2", "3"):
            break
        print("Please enter 1, 2, or 3.")

    if choice == "2":
        return ("new", None)

    repo_root = Path(__file__).resolve().parents[2]

    if choice == "3":
        prompts_dir = repo_root / "prompts"
        prompt_files = sorted(prompts_dir.iterdir()) if prompts_dir.exists() else []
        prompt_files = [f for f in prompt_files if f.is_file() and not f.name.startswith(".")]

        if not prompt_files:
            print("No prompt files found in prompts/.")
            return ("new", None)

        print("\nAvailable prompts:")
        for i, path in enumerate(prompt_files):
            print(f"  {i}. {path.name}")

        while True:
            try:
                raw = input("\nChoose prompt number: ").strip()
            except (EOFError, KeyboardInterrupt):
                return ("new", None)
            try:
                idx = int(raw)
                if 0 <= idx < len(prompt_files):
                    break
            except ValueError:
                pass
            print(f"Please enter a number between 0 and {len(prompt_files) - 1}.")

        chosen = prompt_files[idx]
        description = chosen.read_text().strip()
        return ("prompt", (chosen, description))

    # choice == "1": load existing workflow JSON
    workflows_dir = repo_root / "workflows"
    json_files = sorted(workflows_dir.glob("*.json")) if workflows_dir.exists() else []

    if not json_files:
        print("No saved workflows found.")
        return ("new", None)

    print("\nAvailable workflows:")
    for i, path in enumerate(json_files):
        print(f"  {i}. {path.stem}")

    while True:
        try:
            raw = input("\nChoose workflow number: ").strip()
        except (EOFError, KeyboardInterrupt):
            return ("new", None)
        try:
            idx = int(raw)
            if 0 <= idx < len(json_files):
                break
        except ValueError:
            pass
        print(f"Please enter a number between 0 and {len(json_files) - 1}.")

    chosen = json_files[idx]
    with open(chosen) as f:
        data = json.load(f)
    print(f"\nLoaded workflow: {chosen.name}")
    return ("workflow", data)


def _try_build_cache_manager(faasr_json: dict, s3_client=None):
    """Build a CacheManager if S3 credentials are available. Returns None on failure."""
    try:
        from faasr_ai.utils.cache_manager import CacheManager
        from framework.s3_client import FaaSrS3Client

        if s3_client is None:
            s3_client = FaaSrS3Client(
                workflow_data=faasr_json,
                access_key=os.getenv("S3_AccessKey", ""),
                secret_key=os.getenv("S3_SecretKey", ""),
            )
        workflow_name = faasr_json.get("WorkflowName", "workflow")
        return CacheManager(s3_client, workflow_name)
    except Exception:
        return None


def _prompt_cache_status(cache_manager, faasr_json: dict) -> dict | None:
    """Show per-action cache status and offer invalidation before deploy.

    Returns the final cache statuses dict, or None if unavailable.
    """
    try:
        from faasr_ai.utils.cache_manager import CacheStatus
        statuses = cache_manager.check_cache(faasr_json)
    except Exception:
        return None

    if not statuses:
        return None

    any_hit = any(e.status == CacheStatus.HIT for e in statuses.values())

    print("\nCache Status:")
    for action_name, entry in statuses.items():
        key_short = entry.cache_key[:12] if entry.cache_key else "?"
        if entry.status == CacheStatus.HIT:
            print(f"  {action_name}: HIT  (key={key_short}...)")
        elif entry.status == CacheStatus.INVALID:
            print(f"  {action_name}: INVALID")
        else:
            print(f"  {action_name}: MISS")

    if not any_hit:
        return statuses

    print("\nCached actions will re-execute existing code (skipping LLM generation).")
    while True:
        try:
            raw = input("(i)nvalidate <action> or Enter to continue: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            break

        parts = raw.split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() in ("i", "invalidate"):
            action = parts[1].strip()
        elif parts[0].lower() not in ("i", "invalidate"):
            action = parts[0]
        else:
            print("Usage: invalidate <action_name>")
            continue

        if action not in faasr_json.get("ActionList", {}):
            print(f"Unknown action: {action}")
            continue

        try:
            invalidated = cache_manager.invalidate(action, faasr_json)
            print(f"Invalidated: {', '.join(invalidated)}")
            # Refresh display
            try:
                statuses = cache_manager.check_cache(faasr_json)
                print("\nUpdated Cache Status:")
                for name, entry in statuses.items():
                    key_short = entry.cache_key[:12] if entry.cache_key else "?"
                    label = entry.status.value
                    suffix = f"  (key={key_short}...)" if entry.status.value == "HIT" else ""
                    print(f"  {name}: {label}{suffix}")
            except Exception:
                pass
        except Exception as e:
            print(f"Invalidation failed: {e}")

    return statuses


if __name__ == "__main__":
    main()
