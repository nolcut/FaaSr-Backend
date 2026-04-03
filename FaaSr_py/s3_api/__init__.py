from .delete_file import faasr_delete_file
from .get_file import faasr_get_file
from .get_folder_list import faasr_get_folder_list
from .get_s3_creds import faasr_get_s3_creds
from .log import faasr_log
from .put_file import faasr_put_file
from .registry import faasr_registry_add, faasr_registry_query, faasr_registry_remove
import os

# Set environment variables for compatibility with OSN and other S3-compatible services
# These disable the new checksum behaviors introduced in boto3 1.36.0+ that cause 
# problems with some S3-compatible storage providers
os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "when_required"
os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "when_required"

__all__ = [
    "faasr_log",
    "faasr_put_file",
    "faasr_get_file",
    "faasr_delete_file",
    "faasr_get_folder_list",
    "faasr_get_s3_creds",
    "faasr_registry_query",
    "faasr_registry_add",
    "faasr_registry_remove",
]
