import logging
import sys
import threading

from FaaSr_py.helpers.agent_safety import check_agent_put_file_safety, handle_agent_post_put
from FaaSr_py.s3_api import faasr_put_file

logger = logging.getLogger(__name__)

AGENT_MAX_REQUESTS = 100


class AgentS3Ops:
    """
    Direct S3 operations for the trusted agents (I/O and evaluator).

    Enforces the same safety constraints the RPC server previously handled:
    - Prevents overwriting upstream or pre-existing files
    - Limits total S3 requests to AGENT_MAX_REQUESTS
    - Blocks credential access

    The coding agent subprocess never calls this class — it only reads from
    input_dir and writes to output_dir using local stubs.
    """

    def __init__(self, faasr_payload: dict, existing_keys_snapshot: frozenset):
        self._faasr = faasr_payload
        self._snapshot = existing_keys_snapshot
        self._count = 0
        self._lock = threading.Lock()

    def _check_limit(self):
        with self._lock:
            if self._count >= AGENT_MAX_REQUESTS:
                raise RuntimeError(
                    f"Agent request limit exceeded ({self._count}/{AGENT_MAX_REQUESTS})"
                )
            self._count += 1

    def agent_put_file(
        self,
        local_file: str,
        remote_file: str,
        server_name: str = "",
        local_folder: str = ".",
        remote_folder: str = ".",
        description: str = "",
    ) -> bool:
        self._check_limit()
        args = {
            "local_file": str(local_file),
            "remote_file": str(remote_file),
            "server_name": server_name,
            "local_folder": str(local_folder),
            "remote_folder": str(remote_folder),
            "description": str(description),
        }
        check_agent_put_file_safety(self._faasr, args, self._snapshot)
        put_file_args = {k: v for k, v in args.items() if k != "description"}
        faasr_put_file(faasr_payload=self._faasr, **put_file_args)
        handle_agent_post_put(self._faasr, args)
        logger.info(f"Agent uploaded: {remote_file}")
        return True
