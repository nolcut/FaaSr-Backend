import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import boto3
import botocore

from FaaSr_py.config.debug_config import global_config

logger = logging.getLogger(__name__)

REGISTRY_KEY = "registry.json"


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


def _read_registry(faasr_payload) -> list:
    """Read registry.json from S3 or local FS. Returns [] if not found."""
    if global_config.USE_LOCAL_FILE_SYSTEM:
        registry_path = Path(global_config.LOCAL_FILE_SYSTEM_DIR) / REGISTRY_KEY
        if not registry_path.exists():
            return []
        try:
            return json.loads(registry_path.read_text())
        except Exception:
            return []

    try:
        s3_client, target_s3 = _get_s3_client(faasr_payload)
        response = s3_client.get_object(Bucket=target_s3["Bucket"], Key=REGISTRY_KEY)
        return json.loads(response["Body"].read().decode("utf-8"))
    except Exception as e:
        err_code = ""
        if hasattr(e, "response") and isinstance(e.response, dict):
            err_code = e.response.get("Error", {}).get("Code", "")
        if err_code in ("NoSuchKey", "404"):
            return []
        logger.warning(f"Could not read registry: {e}")
        return []


def _write_registry(faasr_payload, entries: list):
    """Write registry.json to S3 or local FS."""
    body = json.dumps(entries, indent=2).encode("utf-8")
    if global_config.USE_LOCAL_FILE_SYSTEM:
        registry_path = Path(global_config.LOCAL_FILE_SYSTEM_DIR) / REGISTRY_KEY
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_bytes(body)
        return
    s3_client, target_s3 = _get_s3_client(faasr_payload)
    s3_client.put_object(Bucket=target_s3["Bucket"], Key=REGISTRY_KEY, Body=body)


def faasr_registry_query(faasr_payload, action_name=None) -> list:
    """
    Return registry entries available to the current action.

    Arguments:
        faasr_payload: FaaSr payload dict
        action_name: str -- if provided, excludes entries produced by this action
                           (returns only upstream files)
    """
    entries = _read_registry(faasr_payload)
    if action_name:
        entries = [e for e in entries if e.get("produced_by") != action_name]
    return entries


def faasr_registry_add(faasr_payload, entry: dict):
    """
    Add or replace a registry entry (keyed on file_uri).

    Arguments:
        faasr_payload: FaaSr payload dict
        entry: dict with keys file_uri, name, description, schema_uri,
               produced_by, run_id, timestamp
    """
    entries = _read_registry(faasr_payload)
    entries = [e for e in entries if e.get("file_uri") != entry.get("file_uri")]
    entries.append(entry)
    _write_registry(faasr_payload, entries)


def faasr_registry_remove(faasr_payload, file_uri: str):
    """
    Remove the registry entry matching file_uri.

    Arguments:
        faasr_payload: FaaSr payload dict
        file_uri: str -- normalized file URI to remove
    """
    entries = _read_registry(faasr_payload)
    updated = [e for e in entries if e.get("file_uri") != file_uri]
    if len(updated) < len(entries):
        _write_registry(faasr_payload, updated)


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
