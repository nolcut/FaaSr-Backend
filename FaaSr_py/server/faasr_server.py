import json
import logging
import re
import sys
import tempfile
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from FaaSr_py.config.debug_config import global_config
from FaaSr_py.helpers.rank import faasr_rank
from FaaSr_py.helpers.s3_helper_functions import flush_s3_log
from FaaSr_py.s3_api import (
    faasr_delete_file,
    faasr_get_file,
    faasr_get_folder_list,
    faasr_get_s3_creds,
    faasr_log,
    faasr_put_file,
    faasr_registry_add,
    faasr_registry_query,
    faasr_registry_remove,
)
from FaaSr_py.s3_api.registry import _build_registry_entry, _generate_sidecar, faasr_snapshot_existing_keys

logger = logging.getLogger(__name__)
faasr_api = FastAPI()
valid_functions = {
    "faasr_get_file",
    "faasr_put_file",
    "faasr_delete_file",
    "faasr_get_folder_list",
    "faasr_log",
    "faasr_rank",
}

# ensure agent does not make too many S3 calls
agent_request_count = 0
AGENT_MAX_REQUESTS = 40

class Request(BaseModel):
    ProcedureID: str
    Arguments: dict | None = None
    IsAgentRequest: bool = False


class Response(BaseModel):
    Success: bool
    Data: dict | None = None
    Message: str | None = None


class Return(BaseModel):
    FunctionResult: bool | None = None


class Result(BaseModel):
    FunctionResult: bool | None = None
    Error: bool | None = None
    Message: str | None = None
    Traceback: str | None = None


class Exit(BaseModel):
    Error: bool | None = None
    Message: str | None = None
    Traceback: str | None = None


def register_request_handler(faasr_payload):
    """ "
    Setup FastAPI request handlers for FaaSr functions

    Arguments:
        faasr_payload: FaaSr payload dict
    """
    return_val = None
    message = None
    traceback = None
    error = False
    agent_request_count = 0  # Local counter for agent requests
    existing_keys_snapshot = faasr_snapshot_existing_keys(faasr_payload)  # frozen at startup

    @faasr_api.post("/faasr-action")
    def faasr_request_handler(request: Request):
        """
        Handler for FaaSr function requests

        Enforces agent constraints if IsAgentRequest is True:
        - Agents cannot delete files
        - Agents have limited request count
        - Agents cannot overwrite files registered by upstream actions
        """
        nonlocal error, agent_request_count
        logger.info(f"Processing request: {request.ProcedureID} (Agent: {request.IsAgentRequest})")

        # Check agent constraints
        if request.IsAgentRequest:
            agent_request_count += 1

            # Enforce agent request limit
            if agent_request_count > AGENT_MAX_REQUESTS:
                error_msg = f"Agent request limit exceeded ({agent_request_count}/{AGENT_MAX_REQUESTS})"
                logger.error(error_msg)
                return Response(Success=False, Message=error_msg)

            # Agents cannot delete files
            if request.ProcedureID == "faasr_delete_file":
                error_msg = "Agents are not allowed to delete files"
                logger.error(error_msg)
                return Response(Success=False, Message=error_msg)

        args = request.Arguments or {}
        return_obj = Response(Success=True, Data={})
        try:
            match request.ProcedureID:
                case "faasr_log":
                    faasr_log(faasr_payload=faasr_payload, **args)
                case "faasr_put_file":
                    if request.IsAgentRequest:
                        try:
                            _check_agent_put_file_safety(faasr_payload, args, existing_keys_snapshot)
                        except RuntimeError as e:
                            return Response(Success=False, Message=str(e))
                    faasr_put_file(faasr_payload=faasr_payload, **args)
                    if request.IsAgentRequest:
                        _handle_agent_post_put(faasr_payload, args)
                case "faasr_get_file":
                    faasr_get_file(faasr_payload=faasr_payload, **args)
                case "faasr_delete_file":
                    faasr_delete_file(faasr_payload=faasr_payload, **args)
                    # Auto-remove registry entry on any delete
                    file_uri = re.sub(
                        r"/+", "/",
                        f"{args.get('remote_folder', '.')}/{args.get('remote_file', '')}"
                    ).lstrip("/")
                    faasr_registry_remove(faasr_payload, file_uri=file_uri)
                case "faasr_get_folder_list":
                    return_obj.Data["folder_list"] = faasr_get_folder_list(
                        faasr_payload=faasr_payload, **args
                    )
                case "faasr_invocation_id":
                    return_obj.Data["invocation_id"] = faasr_payload.get("InvocationID", "")
                case "faasr_rank":
                    return_obj.Data = faasr_rank(faasr_payload=faasr_payload)
                case "faasr_get_s3_creds":
                    if request.IsAgentRequest:
                        error_msg = "Agents are not allowed to access S3 credentials"
                        logger.error(error_msg)
                        return Response(Success=False, Message=error_msg)
                    return_obj.Data["s3_creds"] = faasr_get_s3_creds(
                        faasr_payload=faasr_payload, **args
                    )
                case _:
                    logging.error(
                        f"{request.ProcedureID} is not a valid FaaSr function call"
                    )
                    error = True
                    sys.exit(1)
        except Exception as e:
            err_msg = f"ERROR - failed to invoke {request.ProcedureID} - {e}"
            logger.error(err_msg)
            error = True
            sys.exit(1)
        # flush log after every function, since we don't know when user function will end
        flush_s3_log()
        return return_obj

    @faasr_api.post("/faasr-return")
    def faasr_return_handler(return_obj: Return):
        """
        Handler for FaaSr function return values
        """
        nonlocal return_val
        return_val = return_obj.FunctionResult
        flush_s3_log()
        return Response(Success=True)

    @faasr_api.post("/faasr-exit")
    def faasr_get_exit_handler(exit_obj: Exit):
        """
        Handler for FaaSr function exit values
        """
        nonlocal error, message, traceback
        if exit_obj.Error:
            error = True
            message = exit_obj.Message
            traceback = exit_obj.Traceback
        flush_s3_log()
        return Response(Success=True)

    @faasr_api.get("/faasr-get-return")
    def faasr_get_return_handler():
        """
        Handler to get the return value from the FaaSr function
        """
        flush_s3_log()
        return Result(
            FunctionResult=return_val,
            Error=error,
            Message=message,
            Traceback=traceback,
        )


@faasr_api.get("/faasr-echo")
def faasr_echo(message: str):
    """
    Echo to poll server
    """
    return {"message": message}


def _check_agent_put_file_safety(faasr_payload, args, existing_keys_snapshot: frozenset):
    """
    Reject agent put_file if the target file was registered by an upstream action
    or existed on S3 before this agent run started (snapshot taken at server startup).

    Arguments:
        faasr_payload: FaaSr payload dict
        args: Arguments for put_file
        existing_keys_snapshot: frozenset of normalized S3 keys at agent startup

    Raises:
        RuntimeError if the target file is immutable
    """
    target_uri = re.sub(
        r"/+", "/",
        f"{args.get('remote_folder', '.')}/{args.get('remote_file', '')}"
    ).lstrip("/")
    for entry in faasr_registry_query(faasr_payload):
        if entry.get("file_uri", "").lstrip("/") == target_uri:
            raise RuntimeError(
                f"Cannot overwrite file produced by upstream action "
                f"'{entry['produced_by']}': {target_uri}"
            )
    if target_uri in existing_keys_snapshot:
        raise RuntimeError(f"Cannot overwrite pre-existing file: {target_uri}")


def _handle_agent_post_put(faasr_payload, args):
    """
    After a successful agent put_file:
    - Generate and upload a sidecar schema for JSON files
    - Add entry to registry

    Arguments:
        faasr_payload: FaaSr payload dict
        args: put_file Arguments dict
    """
    local_path = str(Path(args.get("local_folder", ".")) / args.get("local_file", ""))
    schema_uri = ""

    if local_path.endswith(".json"):
        sidecar = _generate_sidecar(local_path)
        if sidecar:
            schema_uri = _upload_sidecar(faasr_payload, args, sidecar)

    entry = _build_registry_entry(faasr_payload, args, schema_uri=schema_uri, description=args.get("description", ""))
    faasr_registry_add(faasr_payload, entry)


def _upload_sidecar(faasr_payload, args, sidecar: dict) -> str:
    """
    Write sidecar JSON to a temp file and upload it alongside the main file.
    Returns the sidecar's file_uri.

    Arguments:
        faasr_payload: FaaSr payload dict
        args: original put_file args (to derive remote location)
        sidecar: dict from _generate_sidecar
    """
    remote_folder = args.get("remote_folder", ".")
    remote_file = args.get("remote_file", "")
    sidecar_remote_file = f"{remote_file}.schema.json"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".schema.json", delete=False
    ) as tmp:
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


def wait_for_server_start(port):
    """
    Polls the server until it's ready to accept requests
    Arguments:
        port: int -- port the server is running on
    """
    while True:
        try:
            r = requests.get(
                f"http://127.0.0.1:{port}/faasr-echo", params={"message": "echo"}
            )
            message = r.json()["message"]
            if message == "echo":
                break
        except Exception:
            continue


# starts a server listening on localhost
def run_server(faasr_payload, port, start_time):
    """
    Starts a FastAPI server to handle FaaSr requests

    Arguments:
        faasr_payload: FaaSr payload dict
        port: int -- port to run the server on
    """
    # since server runs as a seperate process, we need to re-add the s3 logger handler
    global_config.add_s3_log_handler(faasr_payload, start_time)

    register_request_handler(faasr_payload)
    config = uvicorn.Config(faasr_api, host="127.0.0.1", port=port)
    server = uvicorn.Server(config)
    server.run()
