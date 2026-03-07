import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentConstraints:
    """Configuration for agent security constraints"""

    # Maximum number of S3 operations allowed
    max_s3_requests: int = 40
    
    # Track current operation count
    current_requests: int = 0
    
    # Allowed S3 operations for agents
    allowed_operations: set = None
    
    # Files that should never be modified
    protected_files: set = None

    def __post_init__(self):
        """Initialize constraint sets"""
        if self.allowed_operations is None:
            self.allowed_operations = {
                "faasr_put_file",
                "faasr_get_file",
                "faasr_get_folder_list",
                "faasr_log",
                "faasr_invocation_id",
                "faasr_rank",
            }
        
        if self.protected_files is None:
            # Files that should never be overwritten
            self.protected_files = {
                "function_completions",
                ".done",
                "credentials",
                "secrets",
            }


class AgentRequestValidator:
    """Validates and tracks agent S3 requests"""

    def __init__(self, constraints: Optional[AgentConstraints] = None):
        """
        Initialize validator

        Arguments:
            constraints: AgentConstraints instance or None for defaults
        """
        self.constraints = constraints or AgentConstraints()

    def check_request_limit(self) -> bool:
        """
        Check if agent has exceeded request limit

        Returns:
            True if within limit, False otherwise
        """
        if self.constraints.current_requests >= self.constraints.max_s3_requests:
            logger.warning(
                f"Agent request limit exceeded: "
                f"{self.constraints.current_requests}/{self.constraints.max_s3_requests}"
            )
            return False
        return True

    def increment_request_count(self):
        """Increment the request counter"""
        self.constraints.current_requests += 1
        logger.debug(
            f"Agent request {self.constraints.current_requests}/"
            f"{self.constraints.max_s3_requests}"
        )

    def validate_operation(self, operation: str) -> bool:
        """
        Validate that operation is allowed for agents

        Arguments:
            operation: Name of the S3 operation

        Returns:
            True if allowed, False otherwise
        """
        if operation not in self.constraints.allowed_operations:
            logger.error(f"Operation {operation} not allowed for agents")
            return False
        return True

    def validate_file_safety(self, file_path: str, operation: str) -> bool:
        """
        Check if file path is safe to modify

        Arguments:
            file_path: Path to the file
            operation: Operation being performed (put, delete, etc.)

        Returns:
            True if safe, False otherwise
        """
        # Check for protected file patterns
        file_lower = file_path.lower()
        for protected in self.constraints.protected_files:
            if protected.lower() in file_lower:
                logger.error(
                    f"Cannot {operation} protected file: {file_path}"
                )
                return False

        # For put operations, check if file already exists
        # This is enforced in the actual server-side check
        return True

    def validate_put_request(self, remote_file: str) -> bool:
        """
        Validate a put_file request

        Arguments:
            remote_file: Remote file path

        Returns:
            True if valid, False otherwise
        """
        if not self.check_request_limit():
            return False

        if not self.validate_operation("faasr_put_file"):
            return False

        # Agents cannot use delete_file
        return True

    def validate_get_request(self, remote_file: str) -> bool:
        """
        Validate a get_file request

        Arguments:
            remote_file: Remote file path

        Returns:
            True if valid, False otherwise
        """
        if not self.check_request_limit():
            return False

        if not self.validate_operation("faasr_get_file"):
            return False

        return True

    def validate_delete_request(self) -> bool:
        """
        Validate a delete_file request

        Returns:
            Always False - agents cannot delete files
        """
        logger.error("Agents are not allowed to delete files")
        return False

    def validate_folder_list_request(self) -> bool:
        """
        Validate a get_folder_list request

        Returns:
            True if valid, False otherwise
        """
        if not self.check_request_limit():
            return False

        if not self.validate_operation("faasr_get_folder_list"):
            return False

        return True