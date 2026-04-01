import json

from FaaSr_py.client.agent_prompts import AVAILABLE_PACKAGES


def build_bridge_system_prompt(context: dict) -> str:
    """
    Build the system prompt for a Bridge coding agent.

    Unlike the regular coding agent, the Bridge has no user-supplied prompt.
    Instead, the task is auto-generated from:
      - upstream file metadata (schemas, samples, descriptions)
      - downstream action code or prompt

    context keys used:
        registry_entries    - upstream registry entries
        file_metadata       - {uri: {local_path, sidecar, sample}}
        downstream_specs    - list of {action_name, action_type, code_or_prompt, function_name}
        input_dir           - local input directory
        output_dir          - local output directory
        eval_feedback       - evaluator feedback from previous attempt (if loop_back)
        exception           - exception traceback from previous attempt (if loop_back)
        loop_count          - retry counter
    """
    registry_entries = context.get("registry_entries", [])
    file_metadata = context.get("file_metadata", {})
    downstream_specs = context.get("downstream_specs", [])
    input_dir = context.get("input_dir", "/tmp/agent/input")
    output_dir = context.get("output_dir", "/tmp/agent/output")
    eval_feedback = context.get("eval_feedback", "")
    exception = context.get("exception", "")
    loop_count = context.get("loop_count", 0)

    upstream_section = _build_upstream_section(registry_entries, file_metadata)
    downstream_section = _build_downstream_section(downstream_specs)

    retry_block = ""
    if loop_count > 0:
        retry_block = f"\n\nPREVIOUS ATTEMPT FAILED (attempt {loop_count}).\n"
        if exception:
            retry_block += f"Traceback:\n{exception}\n"
        if eval_feedback:
            retry_block += f"Evaluator feedback: {eval_feedback}\n"
        retry_block += (
            "You MUST fix the issue above. Do not repeat the same mistake.\n"
            "If the failure was a missing package, call faasr_install(\"package_name\") "
            "at the top of your code BEFORE any import that uses it.\n"
        )

    return f"""You are a data transformation agent. Your job is to transform upstream data files
into the format expected by a downstream action.

CRITICAL OUTPUT RULES:
- Generate ONLY pure Python code — no markdown, no triple backticks, no ```python tags
- Start immediately with import statements or code, no pretext

PACKAGE INSTALLATION RULES (IMPORTANT):
For every package NOT in the pre-installed list below, you MUST call faasr_install() on its own
line immediately before the import. Do this unconditionally — even if you think the package
might already be installed. Never import a non-pre-installed package without calling
faasr_install() first. Example:

    faasr_install("geopandas")
    import geopandas as gpd
    faasr_install("rioxarray")
    import rioxarray

{retry_block}

CRITICAL RUNTIME RULES:
- DO NOT use 'faasr' as a variable name or import any 'faasr' module — 'faasr' does not exist in this environment
- Use ONLY the provided functions (faasr_log, faasr_invocation_id, faasr_rank) for meta-context
- DO NOT perform any S3 operations — no faasr_put_file, no faasr_get_file
- Read inputs from: {input_dir}
- Write ALL outputs to: {output_dir}
- Use the input_dir and output_dir variables injected into the runtime
- Output filenames MUST match exactly what the downstream action expects — use the filenames
  you infer from the downstream code or prompt. They must be static string literals.

AVAILABLE FUNCTIONS (injected into runtime, do not import):
- faasr_log(log_message): Append a message to the local log file
- faasr_invocation_id(): Returns the current invocation ID string
- faasr_rank(): Returns a dict with "rank" and "max_rank"
- faasr_install(package_name): Install a Python package at runtime via pip (call before importing)

{AVAILABLE_PACKAGES}

{upstream_section}

{downstream_section}

YOUR TASK:
Transform the upstream data files into the format and filenames expected by the downstream action(s).
Read from {input_dir}, write transformed outputs to {output_dir}.
Infer the expected input filenames and formats from the downstream code or prompt above.
If multiple downstream actions expect different files, produce all of them.
Log key transformation steps with faasr_log."""


def _build_upstream_section(registry_entries: list, file_metadata: dict) -> str:
    if not file_metadata:
        return "UPSTREAM DATA: (none)"

    uri_descriptions = {
        e.get("file_uri", ""): e.get("description", "") for e in registry_entries
    }
    uri_producers = {
        e.get("file_uri", ""): e.get("produced_by", "") for e in registry_entries
    }

    parts = []
    max_schema_chars = 1000
    for uri, meta in file_metadata.items():
        local_path = meta.get("local_path", "")
        sidecar = meta.get("sidecar", {})
        sample = meta.get("sample", "")
        description = uri_descriptions.get(uri, "")
        producer = uri_producers.get(uri, "")

        sidecar_str = json.dumps(sidecar, indent=2) if sidecar else "(no schema)"
        if len(sidecar_str) > max_schema_chars:
            sidecar_str = sidecar_str[:max_schema_chars] + "\n... (schema truncated)"

        part = f"File: {uri}\n  Local path: {local_path}\n"
        if producer:
            part += f"  Produced by: {producer}\n"
        if description:
            part += f"  Description: {description}\n"
        part += f"  Schema: {sidecar_str}\n  Sample:\n{sample}"
        parts.append(part)

    return "UPSTREAM DATA AVAILABLE:\n" + "\n\n".join(parts)


def _build_downstream_section(downstream_specs: list) -> str:
    if not downstream_specs:
        return "DOWNSTREAM ACTION: (none — produce no output)"

    parts = []
    for spec in downstream_specs:
        action_name = spec.get("action_name", "")
        action_type = spec.get("action_type", "")
        code_or_prompt = spec.get("code_or_prompt", "")
        function_name = spec.get("function_name")

        header = f"DOWNSTREAM ACTION: {action_name} (Type: {action_type})"
        if function_name:
            header += f", FunctionName: {function_name}"

        if action_type in ("Python", "R"):
            label = "Source code"
        else:
            label = "Agent prompt"

        content = code_or_prompt.strip() if code_or_prompt.strip() else "(not available)"
        parts.append(f"{header}\n{label}:\n{content}")

    return "\n\n".join(parts)
