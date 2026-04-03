from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from framework.s3_client import FaaSrS3Client
from FaaSr_py.helpers.cache_keys import compute_cache_keys, get_downstream_actions
from FaaSr_py.helpers.graph_functions import build_adjacency_graph

logger = logging.getLogger(__name__)


class CacheStatus(Enum):
    HIT = "HIT"
    MISS = "MISS"
    INVALID = "INVALID"


class CacheEntry:
    """Per-action cache check result."""

    def __init__(self, action_name: str, status: CacheStatus, cache_key: str):
        self.action_name = action_name
        self.status = status
        self.cache_key = cache_key


def _cache_prefix(workflow_name: str, action_name: str) -> str:
    return f"{workflow_name}/_cache/{action_name}"


class CacheManager:
    """Middleware-side cache manager for checking and invalidating action caches.

    Cache artifacts are written by the runtime (agent_func_entry.py).
    This class only reads cache state and manages invalidation.
    """

    def __init__(self, s3_client: FaaSrS3Client, workflow_name: str):
        self._s3 = s3_client
        self._workflow_name = workflow_name

    def check_cache(self, faasr_json: dict[str, Any]) -> dict[str, CacheEntry]:
        """Check cache status for all actions in the workflow.

        Returns a dict mapping action_name -> CacheEntry.
        """
        cache_keys = compute_cache_keys(faasr_json)
        results: dict[str, CacheEntry] = {}

        for action_name, cache_key in cache_keys.items():
            action_def = faasr_json.get("ActionList", {}).get(action_name, {})
            if action_def.get("Type") != "Agent":
                continue

            prefix = _cache_prefix(self._workflow_name, action_name)

            # Check for invalidation sentinel
            invalid_key = f"{prefix}/.invalid"
            if self._s3.object_exists(invalid_key):
                results[action_name] = CacheEntry(action_name, CacheStatus.INVALID, cache_key)
                continue

            # Check for cached code
            code_key = f"{prefix}/{cache_key}/code_raw.py"
            if self._s3.object_exists(code_key):
                results[action_name] = CacheEntry(action_name, CacheStatus.HIT, cache_key)
            else:
                results[action_name] = CacheEntry(action_name, CacheStatus.MISS, cache_key)

        return results

    def invalidate(self, action_name: str, faasr_json: dict[str, Any]) -> list[str]:
        """Invalidate an action and all its downstream actions.

        Uploads .invalid sentinels to S3 for the target action and all
        downstream actions reachable via InvokeNext.

        Returns list of invalidated action names.
        """
        adj_graph, _ = build_adjacency_graph(faasr_json)
        downstream = get_downstream_actions(adj_graph, action_name)
        to_invalidate = {action_name} | downstream

        invalidated = []
        for name in sorted(to_invalidate):
            prefix = _cache_prefix(self._workflow_name, name)
            invalid_key = f"{prefix}/.invalid"
            try:
                self._s3.upload_object(invalid_key, b"")
                invalidated.append(name)
                logger.info(f"Invalidated cache for action: {name}")
            except Exception as e:
                logger.warning(f"Failed to invalidate cache for {name}: {e}")

        return invalidated
