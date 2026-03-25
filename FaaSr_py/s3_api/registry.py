import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import boto3
import botocore

from FaaSr_py.config.debug_config import global_config

logger = logging.getLogger(__name__)

REGISTRY_PREFIX = "registry/"


def _action_registry_key(invocation_id: str, action_name: str) -> str:
    return f"{REGISTRY_PREFIX}{invocation_id}/{action_name}.json"


def _get_s3_client(faasr_payload, server_name=""):
    """Build a boto3 S3 client — same pattern as get_file.py."""
    if not server_name:
        server_name = faasr_payload.get("DefaultDataStore", "")
    if server_name not in faasr_payload.get("DataStores", {}):
        raise RuntimeError(f"Invalid data store name: {server_name}")
    target_s3 = faasr_payload["DataStores"][server_name]
    if target_s3.get("Anonymous", False):
        kwargs = dict(
            region_name=target_s3.get("Region", ""),
            config=botocore.config.Config(signature_version=botocore.UNSIGNED),
        )
        if target_s3.get("Endpoint"):
            kwargs["endpoint_url"] = target_s3["Endpoint"]
        return boto3.client("s3", **kwargs), target_s3
    kwargs = dict(
        aws_access_key_id=target_s3["AccessKey"],
        aws_secret_access_key=target_s3["SecretKey"],
        region_name=target_s3.get("Region", ""),
    )
    if target_s3.get("Endpoint"):
        kwargs["endpoint_url"] = target_s3["Endpoint"]
    return boto3.client("s3", **kwargs), target_s3


def _read_action_registry(faasr_payload, action_name: str) -> list:
    """Read a single action's registry file. Returns [] if not found."""
    invocation_id = faasr_payload.get("InvocationID", "")
    key = _action_registry_key(invocation_id, action_name)
    if global_config.USE_LOCAL_FILE_SYSTEM:
        path = Path(global_config.LOCAL_FILE_SYSTEM_DIR) / key
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())
        except Exception:
            return []
    try:
        s3_client, target_s3 = _get_s3_client(faasr_payload)
        response = s3_client.get_object(Bucket=target_s3["Bucket"], Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except Exception as e:
        err_code = ""
        if hasattr(e, "response") and isinstance(e.response, dict):
            err_code = e.response.get("Error", {}).get("Code", "")
        if err_code in ("NoSuchKey", "404"):
            return []
        logger.warning(f"Could not read registry for {action_name}: {e}")
        return []


def _write_action_registry(faasr_payload, action_name: str, entries: list):
    """Write entries to an action-specific registry file."""
    invocation_id = faasr_payload.get("InvocationID", "")
    key = _action_registry_key(invocation_id, action_name)
    body = json.dumps(entries, indent=2).encode("utf-8")
    if global_config.USE_LOCAL_FILE_SYSTEM:
        path = Path(global_config.LOCAL_FILE_SYSTEM_DIR) / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return
    s3_client, target_s3 = _get_s3_client(faasr_payload)
    s3_client.put_object(Bucket=target_s3["Bucket"], Key=key, Body=body)


def _list_registry_action_names(faasr_payload) -> list:
    """Return all action names that have a registry file for the current invocation."""
    invocation_id = faasr_payload.get("InvocationID", "")
    prefix = f"{REGISTRY_PREFIX}{invocation_id}/" if invocation_id else REGISTRY_PREFIX
    if global_config.USE_LOCAL_FILE_SYSTEM:
        base = Path(global_config.LOCAL_FILE_SYSTEM_DIR) / prefix.rstrip("/")
        if not base.exists():
            return []
        return [p.stem for p in base.glob("*.json") if p.stem != "global"]
    try:
        s3_client, target_s3 = _get_s3_client(faasr_payload)
        paginator = s3_client.get_paginator("list_objects_v2")
        names = []
        for page in paginator.paginate(Bucket=target_s3["Bucket"], Prefix=prefix):
            for obj in page.get("Contents", []):
                name = Path(obj["Key"]).stem
                if name != "global":
                    names.append(name)
        return names
    except Exception as e:
        logger.warning(f"Could not list registry files: {e}")
        return []


def _global_registry_key(invocation_id: str) -> str:
    return f"registry/{invocation_id}/global.json"


def _read_global_registry(faasr_payload) -> list:
    """Read the global (immutable) registry for the current invocation. Returns [] if not found."""
    invocation_id = faasr_payload.get("InvocationID", "")
    if not invocation_id:
        return []
    key = _global_registry_key(invocation_id)
    if global_config.USE_LOCAL_FILE_SYSTEM:
        path = Path(global_config.LOCAL_FILE_SYSTEM_DIR) / key
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())
        except Exception:
            return []
    try:
        s3_client, target_s3 = _get_s3_client(faasr_payload)
        response = s3_client.get_object(Bucket=target_s3["Bucket"], Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except Exception as e:
        err_code = ""
        if hasattr(e, "response") and isinstance(e.response, dict):
            err_code = e.response.get("Error", {}).get("Code", "")
        if err_code in ("NoSuchKey", "404"):
            return []
        logger.warning(f"Could not read global registry: {e}")
        return []


def _write_global_registry(faasr_payload, invocation_id: str, entries: list):
    """Write the global (immutable) registry for this invocation."""
    key = _global_registry_key(invocation_id)
    body = json.dumps(entries, indent=2).encode("utf-8")
    if global_config.USE_LOCAL_FILE_SYSTEM:
        path = Path(global_config.LOCAL_FILE_SYSTEM_DIR) / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return
    s3_client, target_s3 = _get_s3_client(faasr_payload)
    s3_client.put_object(Bucket=target_s3["Bucket"], Key=key, Body=body)


def _list_files_in_folder(faasr_payload, folder: str) -> list:
    """Return all file URIs under a folder prefix (recursive), preserving original paths."""
    if global_config.USE_LOCAL_FILE_SYSTEM:
        base = Path(global_config.LOCAL_FILE_SYSTEM_DIR) / folder
        if not base.exists():
            return []
        return [
            str(p.relative_to(Path(global_config.LOCAL_FILE_SYSTEM_DIR))).replace("\\", "/")
            for p in base.rglob("*") if p.is_file()
        ]
    try:
        s3_client, target_s3 = _get_s3_client(faasr_payload)
        prefix = folder if folder.endswith("/") else folder + "/"
        paginator = s3_client.get_paginator("list_objects_v2")
        uris = []
        for page in paginator.paginate(Bucket=target_s3["Bucket"], Prefix=prefix):
            for obj in page.get("Contents", []):
                uris.append(obj["Key"])
        return uris
    except Exception as e:
        logger.warning(f"Could not list files in folder {folder}: {e}")
        return []


def _make_global_entry(uri: str, invocation_id: str) -> dict:
    """Build a registry entry for a global (immutable) file."""
    return {
        "file_uri": uri,
        "name": Path(uri).name,
        "description": "",
        "schema_uri": "",
        "produced_by": "global",
        "run_id": invocation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def init_immutable_registry(faasr_payload):
    """
    Populate the global read-only registry from GlobalInputFiles and GlobalInputFolders.

    Called once at the entry action. Writes to registry/{InvocationID}/global.json so
    each workflow run has its own isolated global registry.
    """
    global_files = faasr_payload.get("GlobalInputFiles", []) or []
    global_folders = faasr_payload.get("GlobalInputFolders", []) or []
    if not global_files and not global_folders:
        return

    invocation_id = faasr_payload.get("InvocationID", "")
    entries = []

    for uri in global_files:
        parts = uri.rsplit("/", 1)
        folder, file = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
        if faasr_file_exists(faasr_payload, folder, file):
            entries.append(_make_global_entry(uri, invocation_id))
        else:
            logger.warning(f"GlobalInputFile not found, skipping: {uri}")

    for folder in global_folders:
        for uri in _list_files_in_folder(faasr_payload, folder):
            entries.append(_make_global_entry(uri, invocation_id))

    if entries:
        _write_global_registry(faasr_payload, invocation_id, entries)
        logger.info(f"Initialized global registry with {len(entries)} entries for invocation {invocation_id}")


def faasr_registry_query(faasr_payload, action_name=None) -> list:
    """
    Combine all per-action registries and the global registry in memory.

    Arguments:
        faasr_payload: FaaSr payload dict
        action_name: str -- if provided, excludes entries produced by this action
                           (returns only upstream files)
    """
    combined = []
    for name in _list_registry_action_names(faasr_payload):
        if action_name and name == action_name:
            continue
        combined.extend(_read_action_registry(faasr_payload, name))
    combined.extend(_read_global_registry(faasr_payload))
    return combined


def faasr_registry_add(faasr_payload, entry: dict):
    """
    Add or replace an entry in the producing action's registry file.

    Arguments:
        faasr_payload: FaaSr payload dict
        entry: dict with keys file_uri, name, description, schema_uri,
               produced_by, run_id, timestamp
    """
    action_name = entry.get("produced_by") or faasr_payload.get("FunctionInvoke", "unknown")
    entries = _read_action_registry(faasr_payload, action_name)
    entries = [e for e in entries if e.get("file_uri") != entry.get("file_uri")]
    entries.append(entry)
    _write_action_registry(faasr_payload, action_name, entries)


def faasr_registry_remove(faasr_payload, file_uri: str):
    """
    Remove an entry by file_uri from the current action's own registry file.

    Arguments:
        faasr_payload: FaaSr payload dict
        file_uri: str -- normalized file URI to remove
    """
    action_name = faasr_payload.get("FunctionInvoke", "unknown")
    entries = _read_action_registry(faasr_payload, action_name)
    updated = [e for e in entries if e.get("file_uri") != file_uri]
    if len(updated) < len(entries):
        _write_action_registry(faasr_payload, action_name, updated)


def faasr_snapshot_existing_keys(faasr_payload) -> frozenset:
    """
    Return a frozenset of all normalized keys currently in the S3 bucket (or local FS).
    Call once at server startup; use the snapshot for all subsequent pre-existing file checks.
    """
    if global_config.USE_LOCAL_FILE_SYSTEM:
        base = Path(global_config.LOCAL_FILE_SYSTEM_DIR)
        if not base.exists():
            return frozenset()
        return frozenset(
            str(p.relative_to(base)).replace("\\", "/")
            for p in base.rglob("*") if p.is_file()
        )
    try:
        s3_client, target_s3 = _get_s3_client(faasr_payload)
        bucket = target_s3["Bucket"]
        keys = []
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"].lstrip("/"))
        return frozenset(keys)
    except Exception as e:
        logger.warning(f"Could not snapshot existing S3 keys: {e}")
        return frozenset()


def faasr_file_exists(faasr_payload, remote_folder: str, remote_file: str) -> bool:
    """
    Return True if the file already exists on S3 (or local FS).

    Arguments:
        faasr_payload: FaaSr payload dict
        remote_folder: str -- remote folder path
        remote_file: str -- remote file name
    """
    key = re.sub(r"/+", "/", f"{remote_folder}/{remote_file}").lstrip("/")
    if global_config.USE_LOCAL_FILE_SYSTEM:
        return (Path(global_config.LOCAL_FILE_SYSTEM_DIR) / key).exists()
    try:
        s3_client, target_s3 = _get_s3_client(faasr_payload)
        s3_client.head_object(Bucket=target_s3["Bucket"], Key=key)
        return True
    except Exception:
        return False


def _generate_sidecar(local_path: str) -> dict:
    """
    Generate a sidecar schema dict for a JSON file using genson.
    Returns only the 'properties' field of the generated schema.

    Returns {} for non-JSON or unreadable files.
    """
    if not str(local_path).endswith(".json"):
        return {}
    try:
        from genson import SchemaBuilder
        with open(local_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        builder = SchemaBuilder()
        builder.add_object(data)
        schema = builder.to_schema()
        properties = schema.get("properties", {})
        if not properties:
            return {}
        return {"file_type": "json", "properties": properties}
    except Exception as e:
        logger.warning(f"Could not generate sidecar for {local_path}: {e}")
        return {}


def _build_registry_entry(faasr_payload, args: dict, schema_uri: str = "", description: str = "") -> dict:
    """
    Build a registry entry dict from put_file arguments.

    Arguments:
        faasr_payload: FaaSr payload dict
        args: put_file Arguments dict (local_file, remote_file, remote_folder, ...)
        schema_uri: str -- sidecar S3 URI if generated, else ""
        description: str -- natural language description of the data
    """
    remote_folder = args.get("remote_folder", ".")
    remote_file = args.get("remote_file", "")
    file_uri = re.sub(r"/+", "/", f"{remote_folder}/{remote_file}").lstrip("/")

    return {
        "file_uri": file_uri,
        "name": remote_file,
        "description": description,
        "schema_uri": schema_uri,
        "produced_by": faasr_payload.get("FunctionInvoke", ""),
        "run_id": faasr_payload.get("InvocationID", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
