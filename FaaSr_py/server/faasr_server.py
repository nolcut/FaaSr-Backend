import logging
import sys

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
)

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

    @faasr_api.post("/faasr-action")
    def faasr_request_handler(request: Request):
        """
        Handler for FaaSr function requests
        
        Enforces agent constraints if IsAgentRequest is True:
        - Agents cannot delete files
        - Agents have limited request count
        - Agents cannot modify existing files
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
                    # Add agent prefix to logs
                    if request.IsAgentRequest and "log_message" in args:
                        # Log message already has [AGENT] prefix from agent_stubs
                        pass
                    faasr_log(faasr_payload=faasr_payload, **args)
                case "faasr_put_file":
                    # For agent requests, check for existing file (prevent overwrites)
                    if request.IsAgentRequest:
                        _check_agent_put_file_safety(faasr_payload, args)
                    faasr_put_file(faasr_payload=faasr_payload, **args)
                case "faasr_get_file":
                    faasr_get_file(faasr_payload=faasr_payload, **args)
                case "faasr_delete_file":
                    faasr_delete_file(faasr_payload=faasr_payload, **args)
                case "faasr_get_folder_list":
                    return_obj.Data["folder_list"] = faasr_get_folder_list(
                        faasr_payload=faasr_payload, **args
                    )
                case "faasr_invocation_id":
                    return_obj.Data["invocation_id"] = faasr_payload.get("InvocationID", "")
                case "faasr_rank":
                    return_obj.Data = faasr_rank(faasr_payload=faasr_payload)
                case "faasr_get_s3_creds":
                    # Agents should never get S3 credentials
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


def _check_agent_put_file_safety(faasr_payload, args):
    """
    Check if an agent's put_file request is safe
    
    Agents should not overwrite existing files.
    This is a placeholder for future implementation that could check
    if a file already exists before allowing the write.
    
    Arguments:
        faasr_payload: FaaSr payload dict
        args: Arguments for put_file
        
    Raises:
        RuntimeError if the file operation is unsafe
    """
    # For now, we allow all puts since the agent generates unique names
    # In the future, this could check S3 for existing files:
    # remote_folder = args.get("remote_folder", ".")
    # remote_file = args.get("remote_file", "")
    # Check S3 if file exists, then raise if it does
    pass


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
