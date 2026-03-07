import logging
import sys

import requests

logger = logging.getLogger(__name__)


def agent_put_file(
    local_file, remote_file, server_name="", local_folder=".", remote_folder="."
):
    """
    Agent-safe version of faasr_put_file

    Restrictions are enforced by the RPC server.
    """
    request_json = {
        "ProcedureID": "faasr_put_file",
        "Arguments": {
            "local_file": str(local_file),
            "remote_file": str(remote_file),
            "server_name": server_name,
            "local_folder": str(local_folder),
            "remote_folder": str(remote_folder),
        },
        "IsAgentRequest": True,  # Flag for server-side constraint checking
    }

    r = requests.post("http://127.0.0.1:8000/faasr-action", json=request_json)
    try:
        response = r.json()
        if response.get("Success", False):
            logger.info(f"Agent uploaded: {remote_file}")
            return True
        else:
            err_msg = response.get("Message", "Agent put_file request failed")
            logger.error(err_msg)
            print(f'{{"agent_put_file": "{err_msg}"}}')
            sys.exit(1)
    except Exception as e:
        err_msg = f"Failed to parse response from FaaSr RPC: {e}"
        logger.error(err_msg)
        print(f'{{"agent_put_file": "{err_msg}"}}')
        sys.exit(1)


def agent_get_file(
    local_file, remote_file, server_name="", local_folder=".", remote_folder="."
):
    """
    Agent-safe version of faasr_get_file

    Restrictions are enforced by the RPC server.
    """
    request_json = {
        "ProcedureID": "faasr_get_file",
        "Arguments": {
            "local_file": str(local_file),
            "remote_file": str(remote_file),
            "server_name": server_name,
            "local_folder": str(local_folder),
            "remote_folder": str(remote_folder),
        },
        "IsAgentRequest": True,
    }

    r = requests.post("http://127.0.0.1:8000/faasr-action", json=request_json)
    try:
        response = r.json()
        if response.get("Success", False):
            logger.info(f"Agent downloaded: {remote_file}")
            return True
        else:
            err_msg = response.get("Message", "Agent get_file request failed")
            logger.error(err_msg)
            print(f'{{"agent_get_file": "{err_msg}"}}')
            sys.exit(1)
    except Exception as e:
        err_msg = f"Failed to parse response from FaaSr RPC: {e}"
        logger.error(err_msg)
        print(f'{{"agent_get_file": "{err_msg}"}}')
        sys.exit(1)


def agent_delete_file(remote_file, server_name="", remote_folder=""):
    """
    Agent version of delete_file - DISABLED for security

    Agents are never allowed to delete files
    """
    err_msg = "Agents are not allowed to delete files"
    logger.error(err_msg)
    print(err_msg)
    sys.exit(1)


def agent_get_folder_list(server_name="", prefix=""):
    """
    Agent-safe version of faasr_get_folder_list

    Restrictions are enforced by the RPC server.
    """
    request_json = {
        "ProcedureID": "faasr_get_folder_list",
        "Arguments": {"server_name": server_name, "prefix": str(prefix)},
        "IsAgentRequest": True,
    }

    r = requests.post("http://127.0.0.1:8000/faasr-action", json=request_json)
    try:
        response = r.json()
        if response.get("Success", False):
            return response["Data"]["folder_list"]
        else:
            err_msg = response.get("Message", "Agent get_folder_list request failed")
            logger.error(err_msg)
            print(err_msg)
            sys.exit(1)
    except Exception as e:
        err_msg = f"Failed to get folder list from server: {e}"
        logger.error(err_msg)
        print(err_msg)
        sys.exit(1)


def agent_log(log_message):
    """
    Agent-safe version of faasr_log

    Restrictions:
    - Limited to reasonable number of requests
    """
    if not log_message:
        err_msg = "agent_log called with empty log_message"
        logger.error(err_msg)
        print(err_msg)
        sys.exit(1)

    request_json = {
        "ProcedureID": "faasr_log",
        "Arguments": {"log_message": f"[AGENT] {log_message}"},
        "IsAgentRequest": True,
    }

    r = requests.post("http://127.0.0.1:8000/faasr-action", json=request_json)
    try:
        response = r.json()
        if response.get("Success", False):
            return True
        else:
            err_msg = "Agent faasr_log request failed"
            logger.error(err_msg)
            print(err_msg)
            sys.exit(1)
    except Exception as e:
        err_msg = f"Failed to send log message to server: {e}"
        logger.error(err_msg)
        print(err_msg)
        sys.exit(1)


def agent_invocation_id():
    """
    Agent-safe version of faasr_invocation_id

    Returns the invocation ID for the current function
    """
    request_json = {
        "ProcedureID": "faasr_invocation_id",
        "Arguments": {},
        "IsAgentRequest": True,
    }

    r = requests.post("http://127.0.0.1:8000/faasr-action", json=request_json)
    try:
        response = r.json()
        if response.get("Success", False):
            return response["Data"]["invocation_id"]
        else:
            err_msg = "Failed to get invocation ID"
            logger.error(err_msg)
            print(err_msg)
            sys.exit(1)
    except Exception as e:
        err_msg = f"Failed to get invocation ID from server: {e}"
        logger.error(err_msg)
        print(err_msg)
        sys.exit(1)


def agent_rank():
    """
    Agent-safe version of faasr_rank

    Returns rank information for the current function
    """
    request_json = {
        "ProcedureID": "faasr_rank",
        "Arguments": {},
        "IsAgentRequest": True,
    }

    r = requests.post("http://127.0.0.1:8000/faasr-action", json=request_json)
    try:
        response = r.json()
        if response.get("Success", False):
            return response["Data"]
        else:
            err_msg = "Failed to get rank information"
            logger.error(err_msg)
            print(err_msg)
            sys.exit(1)
    except Exception as e:
        err_msg = f"Failed to get rank from server: {e}"
        logger.error(err_msg)
        print(err_msg)
        sys.exit(1)


def agent_get_s3_creds():
    """
    Agent version of get_s3_creds - DISABLED for security

    Agents should NEVER be exposed to S3 credentials.
    All S3 operations are routed through the FaaSr RPC server.
    """
    err_msg = "Agents are not allowed to access S3 credentials"
    logger.error(err_msg)
    print(err_msg)
    sys.exit(1)
