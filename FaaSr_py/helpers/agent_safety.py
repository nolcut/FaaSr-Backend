import json
import logging
import re
import tempfile
from pathlib import Path

from FaaSr_py.s3_api import faasr_put_file
from FaaSr_py.s3_api.registry import (
    _build_registry_entry,
    _generate_sidecar,
    faasr_registry_add,
    faasr_registry_query,
)

logger = logging.getLogger(__name__)


def check_agent_put_file_safety(faasr_payload, args: dict, existing_keys_snapshot: frozenset):
    """
    Reject agent put_file if the target file was registered by an upstream action
    or existed on S3 before this agent run started.

    Allow an action to upload files it produced itself (logs, outputs).

    Raises:
        RuntimeError if the target file is immutable
    """
    target_uri = re.sub(
        r"/+", "/",
        f"{args.get('remote_folder', '.')}/{args.get('remote_file', '')}"
    ).lstrip("/")

    remote_folder = args.get("remote_folder", "").rstrip("/")
    current_action = None
    if remote_folder:
        parts = remote_folder.split("/")
        for part in reversed(parts):
            if part.endswith("_logs") or part.endswith("_outputs"):
                current_action = part.replace("_logs", "").replace("_outputs", "")
                break

    for entry in faasr_registry_query(faasr_payload):
        if entry.get("file_uri", "").lstrip("/") == target_uri:
            produced_by = entry["produced_by"]
            if current_action and produced_by == current_action:
                continue
            raise RuntimeError(
                f"Cannot overwrite file produced by upstream action "
                f"'{produced_by}': {target_uri}"
            )
    if target_uri in existing_keys_snapshot:
        raise RuntimeError(f"Cannot overwrite pre-existing file: {target_uri}")


def handle_agent_post_put(faasr_payload, args: dict):
    """
    After a successful agent put_file:
    - Generate and upload a sidecar schema for JSON files
    - Add entry to registry
    """
    remote_file = args.get("remote_file", "")
    function_invoke = faasr_payload.get("FunctionInvoke", "")
    if (
        remote_file == "manifest.json"
        or remote_file.endswith("_coding_agent.log")
        or remote_file == f"{function_invoke}.py"
    ):
        return

    local_path = str(Path(args.get("local_folder", ".")) / args.get("local_file", ""))
    schema_uri = ""

    if local_path.endswith(".json"):
        sidecar = _generate_sidecar(local_path)
        if sidecar:
            schema_uri = _upload_sidecar(faasr_payload, args, sidecar)

    entry = _build_registry_entry(
        faasr_payload, args, schema_uri=schema_uri, description=args.get("description", "")
    )
    faasr_registry_add(faasr_payload, entry)


def _upload_sidecar(faasr_payload, args: dict, sidecar: dict) -> str:
    """
    Write sidecar JSON to a temp file and upload it alongside the main file.
    Returns the sidecar's file_uri.
    """
    remote_folder = args.get("remote_folder", ".")
    remote_file = args.get("remote_file", "")
    sidecar_remote_file = f"{remote_file}.schema.json"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".schema.json", delete=False) as tmp:
        json.dump(sidecar, tmp, indent=2)
        tmp_path = tmp.name

    try:
        faasr_put_file(
            faasr_payload=faasr_payload,
            local_file=Path(tmp_path).name,
            remote_file=sidecar_remote_file,
            local_folder=str(Path(tmp_path).parent),
            remote_folder=remote_folder,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return re.sub(r"/+", "/", f"{remote_folder}/{sidecar_remote_file}").lstrip("/")
