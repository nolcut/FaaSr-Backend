# src/faasr_ai/agents/agent_oversight.py
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    tool,
)

from framework.s3_client import FaaSrS3Client
from framework.workflow_runner import WorkflowRunner
from framework.utils.enums import FunctionStatus
from faasr_ai.prompts.oversight_prompts import PROMPT_SUGGESTION_PROMPT

logger = logging.getLogger(__name__)


class OversightAgent:
    """
    Post-run oversight agent for a FaaSr workflow.

    Provides a conversational interface to explore workflow results:
    - View and download S3 artifacts
    - Ask questions about the workflow run
    - Get context about function statuses and logs
    """

    def __init__(
        self,
        runner: WorkflowRunner,
        faasr_json: dict[str, Any],
        llm_call: Callable[[str, Optional[str]], str],
        s3_client: FaaSrS3Client,
        user_prompt: str = "",
        cache_manager=None,
        cache_snapshot: dict | None = None,
    ):
        self.runner = runner
        self.faasr_json = faasr_json
        self.llm_call = llm_call
        self.s3_client = s3_client
        self.user_prompt = user_prompt
        self._cache_manager = cache_manager
        self._cache_snapshot = cache_snapshot
        self._context_cache: Optional[str] = None
        self._artifacts_cache: Optional[list[str]] = None
        self._prompts_changed: bool = False

    def list_artifacts(self) -> list[str]:
        """List S3 output artifacts for all actions in the workflow.

        Each action stores outputs at {action_name}_outputs/{filename}.
        """
        if self._artifacts_cache is not None:
            return self._artifacts_cache

        action_names = list(self.faasr_json.get("ActionList", {}).keys())
        if not action_names:
            return []

        keys: list[str] = []
        workflow_name = self.runner.workflow_name
        invocation_id = self.runner.invocation_id
        for action in action_names:
            try:
                found = self.s3_client.list_objects(prefix=f"{workflow_name}/{invocation_id}/{action}_outputs/")
                keys.extend(found)
            except Exception:
                pass

        self._artifacts_cache = keys
        return keys

    def download_artifact(self, key: str, dest_dir: str | None = None) -> str:
        """Download an artifact from S3 to local file system."""
        try:
            if dest_dir is None:
                dest_dir = f"downloads/{self.runner.workflow_name}"
            dest_path = Path(dest_dir) / Path(key).name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_object(key, str(dest_path))
            return str(dest_path)
        except Exception as e:
            logger.debug(f"Error downloading artifact: {e}")
            return ""

    def _get_agent_prompts(self) -> dict[str, str]:
        """Extract current prompts from Agent-type actions in the workflow."""
        prompts = {}
        for action_name, action_def in self.faasr_json.get("ActionList", {}).items():
            if action_def.get("Type") == "Agent":
                prompt = action_def.get("Arguments", {}).get("prompt")
                if prompt:
                    prompts[action_name] = prompt
        return prompts

    def suggest_prompt_changes(self) -> dict[str, Any]:
        """Ask the LLM to suggest improvements to agent prompts based on run output.

        Returns dict like {action_name: {suggested, reason}} or empty if no suggestions.
        """
        context = self._build_context()
        agent_prompts = self._get_agent_prompts()

        if not agent_prompts:
            logger.debug("No agent prompts found in this workflow.")
            return {}

        current_prompts_str = "\n".join([
            f"[{name}] {prompt}"
            for name, prompt in agent_prompts.items()
        ])

        prompt_text = PROMPT_SUGGESTION_PROMPT.format(
            context=context,
            current_prompts=current_prompts_str
        )

        try:
            response = self.llm_call(prompt_text, None)
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            suggestions = json.loads(text.strip())
            return suggestions
        except json.JSONDecodeError:
            logger.debug("Error parsing LLM suggestions (invalid JSON). Try again.")
            return {}
        except Exception as e:
            logger.debug(f"Error getting suggestions: {e}")
            return {}

    def _build_context(self) -> str:
        """Build knowledge base context string from run data."""
        if self._context_cache is not None:
            return self._context_cache

        lines = []
        lines.append("=" * 60)
        lines.append("WORKFLOW RUN CONTEXT")
        lines.append("=" * 60)

        # Original user prompt
        if self.user_prompt:
            lines.append("\nOriginal Prompt:")
            lines.append(self.user_prompt)

        # Invocation ID (not in the workflow JSON)
        lines.append(f"\nInvocation ID: {self.runner.invocation_id}")

        # Function statuses
        lines.append("\nFunction Statuses:")
        statuses = self.runner.get_function_statuses()
        for fn_name, status in statuses.items():
            lines.append(f"  {fn_name}: {status.value}")

        # Log excerpts — up to 15,000 chars total, divided fairly across actions, newest lines first
        MAX_LOG_CHARS = 15_000
        action_names = [fn for fn in statuses]
        per_action_budget = MAX_LOG_CHARS // max(len(action_names), 1)

        lines.append("\nFunction Logs:")
        for fn_name in action_names:
            status = statuses[fn_name]
            try:
                raw_logs = self.runner.get_function_logs_content(fn_name)
            except Exception:
                raw_logs = ""
            if raw_logs:
                # Take from the end (newest first) up to per_action_budget chars
                excerpt = raw_logs[-per_action_budget:]
                # Trim to a clean line boundary if we truncated
                if len(raw_logs) > per_action_budget:
                    first_newline = excerpt.find("\n")
                    if first_newline != -1:
                        excerpt = excerpt[first_newline + 1:]
                log_section = excerpt
            else:
                log_section = "(no logs)"
            lines.append(f"\n  [{fn_name}] ({status.value}):")
            for log_line in log_section.split("\n"):
                lines.append(f"    {log_line}")

        # Full workflow JSON
        lines.append("\nWorkflow Definition:")
        lines.append(json.dumps(self.faasr_json, indent=2))

        # S3 artifacts
        lines.append("\nS3 Output Artifacts:")
        artifacts = self.list_artifacts()
        if artifacts:
            for i, artifact in enumerate(artifacts):
                lines.append(f"  [{i}] {artifact}")
        else:
            lines.append("  (none found)")

        context = "\n".join(lines)
        self._context_cache = context
        return context

    def export_static_workflow(self, export_dir: str = "exports") -> None:
        """Export agent-generated code as a static FaaSr workflow.

        Downloads generated .py files and I/O manifests from the cache, wraps each
        with faasr_get_file/faasr_put_file calls, and generates a static
        workflow JSON. Fully deterministic — no LLM calls.
        """
        from FaaSr_py.helpers.cache_keys import compute_cache_keys

        workflow_name = self.runner.workflow_name
        action_list = self.faasr_json.get("ActionList", {})
        statuses = self.runner.get_function_statuses()

        cache_keys = compute_cache_keys(self.faasr_json)

        gh_username = os.getenv("FAASR_GH_USERNAME", "YOUR_USERNAME")
        action_repo = os.getenv("FAASR_ACTION_REPO", "YOUR_ACTION_REPO")

        out_dir = Path(export_dir) / workflow_name
        out_dir.mkdir(parents=True, exist_ok=True)

        exported_actions = []
        skipped_actions = []
        extracted_packages: dict[str, list[str]] = {}

        for action_name, action_def in action_list.items():
            if action_def.get("Type") != "Agent":
                continue

            status = statuses.get(action_name)
            from framework.utils.enums import FunctionStatus
            if status != FunctionStatus.COMPLETED:
                skipped_actions.append((action_name, f"status={status.value if status else 'unknown'}"))
                continue

            cache_key = cache_keys.get(action_name)
            if not cache_key:
                skipped_actions.append((action_name, "no cache key"))
                continue

            cache_prefix = f"{workflow_name}/_cache/{action_name}/{cache_key}"
            code_s3_key = f"{cache_prefix}/code_raw.py"
            manifest_s3_key = f"{cache_prefix}/manifest.json"

            # Download code file
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tf:
                    tmp_code = tf.name
                self.s3_client.download_object(code_s3_key, tmp_code)
                with open(tmp_code, "r") as f:
                    original_code = f.read()
                Path(tmp_code).unlink(missing_ok=True)
            except Exception as e:
                skipped_actions.append((action_name, f"no code file in cache: {e}"))
                continue

            # Download manifest (optional — degrade gracefully if missing)
            manifest = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tf:
                    tmp_manifest = tf.name
                self.s3_client.download_object(manifest_s3_key, tmp_manifest)
                with open(tmp_manifest, "r") as f:
                    manifest = json.load(f)
                Path(tmp_manifest).unlink(missing_ok=True)
            except Exception:
                manifest = None

            # Collect packages: manifest's declared list + faasr_install() calls in code
            packages: list[str] = list(manifest.get("packages", [])) if manifest else []
            packages += self._extract_faasr_install_packages(original_code)
            # Deduplicate while preserving order
            seen_pkgs: set[str] = set()
            unique_packages = [p for p in packages if not (p in seen_pkgs or seen_pkgs.add(p))]
            if unique_packages:
                extracted_packages[action_name] = unique_packages

            # Build wrapped function
            wrapped = self._wrap_code(action_name, original_code, manifest, f"{workflow_name}-static", action_list)
            out_path = out_dir / f"{action_name}.py"
            out_path.write_text(wrapped)
            exported_actions.append(action_name)

            if manifest is None:
                print(f"  {action_name}.py — warning: no manifest found, I/O wrapping may be incomplete")

        # Generate static workflow JSON
        static_json = self._build_static_workflow_json(gh_username, action_repo, extracted_packages)
        workflow_name_static = f"{workflow_name}-static"

        json_path = out_dir / f"{workflow_name_static}.json"
        with open(json_path, "w") as f:
            json.dump(static_json, f, indent=2)

        # Print summary
        print(f"\nExported static workflow to {out_dir}/\n")
        print("Files generated:")
        for name in exported_actions:
            print(f"  {out_dir}/{name}.py")
        if skipped_actions:
            print("\nSkipped (not completed):")
            for name, reason in skipped_actions:
                print(f"  {name}: {reason}")
        print(f"  {out_dir}/{workflow_name_static}.json")
        print(f"""
To deploy (run these commands in a shell/terminal, not in this interface):
  1. Upload function files to your action repo:
       git clone https://github.com/{gh_username}/{action_repo}.git
       mkdir -p {action_repo}/functions
       cp {out_dir}/*.py {action_repo}/functions/
       cd {action_repo} && git add -A && git commit -m "Add exported functions" && git push

  2. Upload the workflow JSON to your action repo:
       cp {out_dir}/{workflow_name_static}.json {action_repo}/workflows/
       cd {action_repo} && git add -A && git commit -m "Add static workflow" && git push

  3. Register and invoke via GitHub Actions.
""")

    # Stdlib modules injected into the coding agent exec namespace.
    # Static exports must import these explicitly since exec injection is gone.
    _CODING_AGENT_IMPORTS = [
        "import json",
        "import os",
        "import sys",
        "import csv",
        "import math",
        "import datetime",
        "import re",
        "import pathlib",
        "from pathlib import Path",
    ]

    def _wrap_code(
        self,
        action_name: str,
        original_code: str,
        manifest: Optional[dict],
        workflow_name: str,
        action_list: dict,
    ) -> str:
        """Wrap agent-generated code with faasr_get_file/faasr_put_file calls."""
        # Separate imports from non-import lines. Hoist all imports to module
        # level to prevent UnboundLocalError (Python treats any name assigned
        # anywhere in a function — including via import — as local throughout).
        # Also strip faasr_install() calls: packages are now declared in
        # PyPIPackageDownloads and pre-installed by the FaaSr invoke helper.
        code_imports: list[str] = []
        body_lines: list[str] = []
        for line in original_code.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                code_imports.append(stripped)
            elif stripped.startswith("faasr_install("):
                pass  # removed — packages handled via PyPIPackageDownloads
            else:
                body_lines.append(line)

        # Build deduplicated module-level imports: coding-agent stdlib first,
        # then any additional imports from the generated code.
        seen: set[str] = set()
        module_imports: list[str] = []
        for imp in self._CODING_AGENT_IMPORTS + code_imports:
            key = imp.strip()
            if key not in seen:
                seen.add(key)
                module_imports.append(key)

        lines = module_imports + ["", "", f"def {action_name}():"]
        lines += [
            '    os.makedirs("/tmp/agent/input", exist_ok=True)',
            '    os.makedirs("/tmp/agent/output", exist_ok=True)',
            '    input_dir = "/tmp/agent/input"',
            '    output_dir = "/tmp/agent/output"',
        ]

        if manifest:
            inputs = manifest.get("inputs", [])
            outputs = manifest.get("outputs", [])
        else:
            inputs = []
            outputs = []

        # faasr_get_file calls for each input
        if inputs:
            lines.append("")
            lines.append("    # Download inputs")
        for inp in inputs:
            producing_action = self._remap_input_folder(inp["remote_folder"], action_list)
            if producing_action:
                remote_folder = f'f"{{faasr_workflow_name()}}/{{faasr_invocation_id()}}/{producing_action}"'
            else:
                remote_folder = f'"{inp["remote_folder"]}"'
            lines += [
                "    faasr_get_file(",
                f'        local_file="{inp["local_file"]}",',
                f'        remote_file="{inp["remote_file"]}",',
                '        local_folder="/tmp/agent/input",',
                f'        remote_folder={remote_folder},',
                "    )",
            ]

        # Inject non-import body lines (indented 4 spaces)
        lines.append("")
        lines.append("    # --- Generated code ---")
        for code_line in body_lines:
            lines.append("    " + code_line if code_line.strip() else "")
        lines.append("    # --- End generated code ---")

        # faasr_put_file calls for each output
        if outputs:
            lines.append("")
            lines.append("    # Upload outputs")
        for out in outputs:
            local_file = out
            filename = Path(local_file).name
            subfolder = str(Path(local_file).parent)
            if subfolder == ".":
                local_folder = "/tmp/agent/output"
                remote_folder_expr = f'f"{{faasr_workflow_name()}}/{{faasr_invocation_id()}}/{action_name}"'
            else:
                local_folder = f"/tmp/agent/output/{subfolder}"
                remote_folder_expr = f'f"{{faasr_workflow_name()}}/{{faasr_invocation_id()}}/{action_name}/{subfolder}"'
            lines += [
                "    faasr_put_file(",
                f'        local_file="{filename}",',
                f'        remote_file="{filename}",',
                f'        local_folder="{local_folder}",',
                f'        remote_folder={remote_folder_expr},',
                "    )",
            ]

        source = "\n".join(lines) + "\n"

        # Strip unused imports (e.g. stdlib modules injected but not referenced)
        try:
            import autoflake
            source = autoflake.fix_code(source, remove_all_unused_imports=True)
        except ImportError:
            pass  # autoflake not installed — skip cleanup

        return source

    @staticmethod
    def _extract_faasr_install_packages(code: str) -> list[str]:
        """Parse faasr_install("pkg") / faasr_install('pkg') calls from generated code."""
        import re
        return re.findall(r'faasr_install\(["\']([^"\']+)["\']\)', code)

    def _remap_input_folder(self, remote_folder: str, action_list: dict) -> Optional[str]:
        """Return the producing action name (plus any subfolder) if from a sibling action, else None."""
        for other_action in action_list:
            tag = f"{other_action}_outputs"
            idx = remote_folder.find(tag)
            if idx != -1:
                tail = remote_folder[idx + len(tag):]
                return other_action + tail
        return None

    def _build_static_workflow_json(
        self,
        gh_username: str,
        action_repo: str,
        pypi_packages: Optional[dict[str, list[str]]] = None,
    ) -> dict:
        """Build the static workflow JSON from the agent workflow JSON."""
        fj = self.faasr_json
        wf_name = fj.get("WorkflowName", "workflow")

        action_list_static = {}
        for action_name, action_def in fj.get("ActionList", {}).items():
            action_list_static[action_name] = {
                "FaaSServer": action_def.get("FaaSServer", "GH"),
                "Type": "Python",
                "FunctionName": action_name,
                "InvokeNext": action_def.get("InvokeNext", []),
                "Arguments": {},
            }

        # Use packages extracted from actual generated code; fall back to
        # whatever the agent workflow declared if nothing was extracted.
        packages = pypi_packages if pypi_packages else fj.get("PyPIPackageDownloads", {})

        result: dict = {
            "FunctionInvoke": fj.get("FunctionInvoke"),
            "WorkflowName": f"{wf_name}-static",
            "DefaultDataStore": fj.get("DefaultDataStore", "S3"),
            "LoggingDataStore": fj.get("LoggingDataStore", "S3"),
            "FaaSrLog": fj.get("FaaSrLog", "FaaSrLog"),
            "ActionList": action_list_static,
            "FunctionGitRepo": {
                name: f"{gh_username}/{action_repo}"
                for name in fj.get("ActionList", {})
            },
            "ActionContainers": {
                name: "ghcr.io/faasr/github-actions-python:latest"
                for name in fj.get("ActionList", {})
            },
            "ComputeServers": fj.get("ComputeServers", {}),
            "DataStores": fj.get("DataStores", {}),
        }
        if packages:
            result["PyPIPackageDownloads"] = packages
        return result

    def run(self) -> Optional[dict]:
        """Interactive REPL for post-run oversight using Claude Agents SDK.

        Returns:
            None if user quits normally.
            Modified faasr_json dict if the user requests a rerun.
        """
        return asyncio.run(self._run_async())

    async def _run_async(self) -> Optional[dict]:
        context = self._build_context()
        print("\n" + context + "\n")

        print("=" * 60)
        print("WORKFLOW OVERSIGHT")
        print("=" * 60)
        print("\nCommands:")
        print("  (q)uit              — exit oversight")
        print("  (a)rtifacts         — list S3 artifacts")
        print("  (d)ownload <n>      — download artifact by index (comma-separated for multiple)")
        print("  (s)uggest           — suggest prompt improvements from LLM")
        print("  (r)erun             — rerun workflow (with any accepted changes)")
        print("  (e)xport            — export agent code as static FaaSr workflow")
        print("  (c)ache             — show cache status for each action")
        print("  (i)nvalidate <act>  — invalidate action cache and all downstream")
        print("  Or type a question  — ask about the workflow run")
        print()

        # Shared mutable state so MCP tools can signal rerun back to the loop.
        state: dict[str, Any] = {"rerun": False}

        mcp_server = self._build_mcp_server()

        system_prompt = (
            "You are an expert oversight agent for a FaaSr agentic workflow that just completed.\n"
            "You have full context on the run: function statuses, logs, S3 artifacts, and the "
            "workflow definition. Use your tools to investigate issues, answer questions, and "
            "help the user improve the workflow.\n\n"
            "Available actions via tools:\n"
            "  - list_artifacts / download_artifact: explore S3 outputs\n"
            "  - get_function_logs / get_execution_status: investigate failures\n"
            "  - read_workflow: inspect workflow definition\n"
            "  - read_function_code / read_function_manifest: view cached code and I/O\n"
            "  - get_cache_status / invalidate_cache: manage the code cache\n"
            "  - suggest_prompt_improvements / apply_prompt_change: improve agent prompts\n"
            "  - export_static_workflow: export generated code as a deployable static workflow\n\n"
            "Format responses for a CLI (plain text, no markdown).\n\n"
            f"## Run Context\n{context}"
        )

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            mcp_servers={"faasr": mcp_server},
            allowed_tools=[
                "mcp__faasr__list_artifacts",
                "mcp__faasr__download_artifact",
                "mcp__faasr__get_function_logs",
                "mcp__faasr__get_execution_status",
                "mcp__faasr__read_workflow",
                "mcp__faasr__read_function_code",
                "mcp__faasr__read_function_manifest",
                "mcp__faasr__get_cache_status",
                "mcp__faasr__invalidate_cache",
                "mcp__faasr__suggest_prompt_improvements",
                "mcp__faasr__apply_prompt_change",
                "mcp__faasr__export_static_workflow",
            ],
            permission_mode="bypassPermissions",
            max_turns=25,
        )

        async with ClaudeSDKClient(options) as client:
            while True:
                try:
                    user_input = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if not user_input:
                    continue

                cmd = user_input.lower()
                parts = user_input.split(maxsplit=1)
                cmd0 = parts[0].lower()

                # Deterministic commands

                if cmd in ("q", "quit", "exit"):
                    break

                elif cmd in ("a", "artifacts"):
                    artifacts = self.list_artifacts()
                    if not artifacts:
                        print("(no artifacts found)")
                    else:
                        for i, a in enumerate(artifacts):
                            print(f"  [{i}] {a}")

                elif cmd0 in ("d", "download"):
                    artifacts = self.list_artifacts()
                    if not artifacts:
                        print("(no artifacts found)")
                    elif len(parts) < 2:
                        print("Usage: download <n>  (comma-separated for multiple)")
                    else:
                        for token in parts[1].split(","):
                            token = token.strip()
                            try:
                                idx = int(token)
                            except ValueError:
                                print(f"  Invalid index: {token}")
                                continue
                            if not (0 <= idx < len(artifacts)):
                                print(f"  Index {idx} out of range (0-{len(artifacts)-1})")
                                continue
                            local_path = self.download_artifact(artifacts[idx])
                            if local_path:
                                print(f"  Downloaded: {local_path}")
                            else:
                                print(f"  Failed to download: {artifacts[idx]}")

                elif cmd in ("s", "suggest"):
                    print("Requesting prompt suggestions from LLM...")
                    suggestions = self.suggest_prompt_changes()
                    if not suggestions:
                        print("No suggestions generated.")
                    else:
                        for action_name, suggestion in suggestions.items():
                            print(f"\n[{action_name}]")
                            print(f"  Reason: {suggestion.get('reason', '')}")
                            print(f"  Suggested prompt:\n    {suggestion.get('suggested', '')}")
                            try:
                                apply = input(f"  Apply to {action_name}? (y/n): ").strip().lower()
                            except (EOFError, KeyboardInterrupt):
                                print()
                                break
                            if apply == "y":
                                action_list = self.faasr_json.get("ActionList", {})
                                if action_name in action_list and action_list[action_name].get("Type") == "Agent":
                                    action_list[action_name].setdefault("Arguments", {})["prompt"] = suggestion["suggested"]
                                    self._prompts_changed = True
                                    print(f"  Applied. Use 'rerun' to redeploy.")
                                else:
                                    print(f"  Skipped: '{action_name}' is not an Agent-type action.")

                elif cmd in ("r", "rerun"):
                    state["rerun"] = True
                    print("Rerunning workflow" + (" with updated prompts..." if self._prompts_changed else "..."))
                    break

                elif cmd in ("e", "export"):
                    self.export_static_workflow()

                elif cmd in ("c", "cache"):
                    if self._cache_manager is None:
                        print("Cache manager not available.")
                    else:
                        try:
                            from faasr_ai.utils.cache_manager import CacheStatus
                            statuses = self._cache_manager.check_cache(self.faasr_json)
                            if not statuses:
                                print("No cached actions found.")
                            else:
                                for name, entry in statuses.items():
                                    key_short = entry.cache_key[:12] if entry.cache_key else "?"
                                    suffix = f"  (key={key_short}...)" if entry.status == CacheStatus.HIT else ""
                                    print(f"  {name}: {entry.status.value}{suffix}")
                        except Exception as e:
                            print(f"Error: {e}")

                elif cmd0 in ("i", "invalidate"):
                    if self._cache_manager is None:
                        print("Cache manager not available.")
                    elif len(parts) < 2:
                        print("Usage: invalidate <action_name>")
                    else:
                        action = parts[1].strip()
                        if action not in self.faasr_json.get("ActionList", {}):
                            print(f"Unknown action: {action}")
                        else:
                            try:
                                invalidated = self._cache_manager.invalidate(action, self.faasr_json)
                                print(f"Invalidated: {', '.join(invalidated)}")
                            except Exception as e:
                                print(f"Error: {e}")

                # Free form to agent
                else:
                    await client.query(user_input)
                    async for message in client.receive_messages():
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    print(block.text, end="", flush=True)
                        elif isinstance(message, ResultMessage):
                            print()
                            break

                if state["rerun"]:
                    break

        return self.faasr_json if state["rerun"] else None

    def _build_mcp_server(self):
        """Build the in-process MCP server with all oversight tools."""

        agent = self

        @tool("list_artifacts", "List S3 output artifacts for all actions in the workflow.", {})
        async def list_artifacts(args: dict) -> dict:
            artifacts = agent.list_artifacts()
            if not artifacts:
                return {"content": [{"type": "text", "text": "(no artifacts found)"}]}
            lines = [f"[{i}] {a}" for i, a in enumerate(artifacts)]
            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        @tool(
            "download_artifact",
            "Download an S3 artifact by its index (from list_artifacts) to a local downloads/ directory.",
            {"index": int},
        )
        async def download_artifact(args: dict) -> dict:
            artifacts = agent.list_artifacts()
            idx = args["index"]
            if not (0 <= idx < len(artifacts)):
                return {"content": [{"type": "text", "text": f"Invalid index {idx}. Valid range: 0-{len(artifacts)-1}"}]}
            key = artifacts[idx]
            local_path = agent.download_artifact(key)
            if local_path:
                return {"content": [{"type": "text", "text": f"Downloaded to: {local_path}"}]}
            return {"content": [{"type": "text", "text": f"Failed to download: {key}"}]}

        @tool(
            "get_function_logs",
            "Get execution logs for a specific function from the workflow runner.",
            {"function_name": str},
        )
        async def get_function_logs(args: dict) -> dict:
            fn = args["function_name"]
            try:
                logs = agent.runner.get_function_logs_content(fn)
                return {"content": [{"type": "text", "text": logs or "(no logs)"}]}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"Error fetching logs: {e}"}]}

        @tool(
            "get_execution_status",
            "Get execution status for all functions in the workflow (COMPLETED, FAILED, PENDING, etc.).",
            {},
        )
        async def get_execution_status(args: dict) -> dict:
            statuses = agent.runner.get_function_statuses()
            lines = [f"  {fn}: {s.value}" for fn, s in statuses.items()]
            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        @tool("read_workflow", "Return the full workflow JSON definition.", {})
        async def read_workflow(args: dict) -> dict:
            return {"content": [{"type": "text", "text": json.dumps(agent.faasr_json, indent=2)}]}

        @tool(
            "read_function_code",
            "Download and return cached source code for a workflow action from S3.",
            {"action_name": str},
        )
        async def read_function_code(args: dict) -> dict:
            from FaaSr_py.helpers.cache_keys import compute_cache_keys
            action = args["action_name"]
            try:
                cache_keys = compute_cache_keys(agent.faasr_json)
                cache_key = cache_keys.get(action)
                if not cache_key:
                    return {"content": [{"type": "text", "text": f"Action '{action}' not found in workflow."}]}
                wf_name = agent.runner.workflow_name
                s3_key = f"{wf_name}/_cache/{action}/{cache_key}/code_raw.py"
                with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tf:
                    tmp = tf.name
                agent.s3_client.download_object(s3_key, tmp)
                code = Path(tmp).read_text()
                Path(tmp).unlink(missing_ok=True)
                return {"content": [{"type": "text", "text": code}]}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"Error: {e}"}]}

        @tool(
            "read_function_manifest",
            "Download and return cached manifest (inputs, outputs, packages) for a workflow action.",
            {"action_name": str},
        )
        async def read_function_manifest(args: dict) -> dict:
            from FaaSr_py.helpers.cache_keys import compute_cache_keys
            action = args["action_name"]
            try:
                cache_keys = compute_cache_keys(agent.faasr_json)
                cache_key = cache_keys.get(action)
                if not cache_key:
                    return {"content": [{"type": "text", "text": f"Action '{action}' not found in workflow."}]}
                wf_name = agent.runner.workflow_name
                s3_key = f"{wf_name}/_cache/{action}/{cache_key}/manifest.json"
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tf:
                    tmp = tf.name
                agent.s3_client.download_object(s3_key, tmp)
                content = Path(tmp).read_text()
                Path(tmp).unlink(missing_ok=True)
                return {"content": [{"type": "text", "text": content}]}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"Error: {e}"}]}

        @tool("get_cache_status", "Show pre-run cache hit/miss/invalid status for each action.", {})
        async def get_cache_status(args: dict) -> dict:
            # Use the pre-run snapshot so we report what the cache status was
            # before execution, not after (when all actions have written code).
            if agent._cache_snapshot is not None:
                from faasr_ai.utils.cache_manager import CacheStatus
                statuses = agent._cache_snapshot
            elif agent._cache_manager is not None:
                from faasr_ai.utils.cache_manager import CacheStatus
                statuses = agent._cache_manager.check_cache(agent.faasr_json)
            else:
                return {"content": [{"type": "text", "text": "Cache manager not available."}]}
            try:
                if not statuses:
                    return {"content": [{"type": "text", "text": "No cached actions found."}]}
                lines = []
                for name, entry in statuses.items():
                    key_short = entry.cache_key[:12] if entry.cache_key else "?"
                    suffix = f"  (key={key_short}...)" if entry.status == CacheStatus.HIT else ""
                    lines.append(f"  {name}: {entry.status.value}{suffix}")
                return {"content": [{"type": "text", "text": "\n".join(lines)}]}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"Error: {e}"}]}

        @tool(
            "invalidate_cache",
            "Invalidate an action's cache and all downstream actions.",
            {"action_name": str},
        )
        async def invalidate_cache(args: dict) -> dict:
            action = args["action_name"]
            if agent._cache_manager is None:
                return {"content": [{"type": "text", "text": "Cache manager not available."}]}
            if action not in agent.faasr_json.get("ActionList", {}):
                return {"content": [{"type": "text", "text": f"Unknown action: {action}"}]}
            try:
                invalidated = agent._cache_manager.invalidate(action, agent.faasr_json)
                return {"content": [{"type": "text", "text": f"Invalidated: {', '.join(invalidated)}"}]}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"Error: {e}"}]}

        @tool(
            "suggest_prompt_improvements",
            "Ask the LLM to analyze the run and suggest improvements to agent prompts. Returns JSON with suggestions.",
            {},
        )
        async def suggest_prompt_improvements(args: dict) -> dict:
            suggestions = agent.suggest_prompt_changes()
            if not suggestions:
                return {"content": [{"type": "text", "text": "No suggestions generated."}]}
            return {"content": [{"type": "text", "text": json.dumps(suggestions, indent=2)}]}

        @tool(
            "apply_prompt_change",
            "Apply a new prompt to an Agent-type action in the workflow (in preparation for rerun).",
            {"action_name": str, "new_prompt": str},
        )
        async def apply_prompt_change(args: dict) -> dict:
            action = args["action_name"]
            prompt = args["new_prompt"]
            action_list = agent.faasr_json.get("ActionList", {})
            if action not in action_list:
                return {"content": [{"type": "text", "text": f"Unknown action: {action}"}]}
            if action_list[action].get("Type") != "Agent":
                return {"content": [{"type": "text", "text": f"Action '{action}' is not an Agent type."}]}
            action_list[action].setdefault("Arguments", {})["prompt"] = prompt
            agent._prompts_changed = True
            return {"content": [{"type": "text", "text": f"Prompt updated for '{action}'. Use request_rerun to redeploy."}]}

        @tool(
            "export_static_workflow",
            "Export agent-generated code as a deployable static FaaSr workflow to the exports/ directory.",
            {},
        )
        async def export_static_workflow_tool(args: dict) -> dict:
            try:
                agent.export_static_workflow()
                return {"content": [{"type": "text", "text": "Export complete. See exports/ directory."}]}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"Export failed: {e}"}]}

        return create_sdk_mcp_server(
            "faasr-oversight",
            tools=[
                list_artifacts,
                download_artifact,
                get_function_logs,
                get_execution_status,
                read_workflow,
                read_function_code,
                read_function_manifest,
                get_cache_status,
                invalidate_cache,
                suggest_prompt_improvements,
                apply_prompt_change,
                export_static_workflow_tool,
            ],
        )
