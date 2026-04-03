import hashlib
import json
import re
from collections import defaultdict, deque

from FaaSr_py.helpers.graph_functions import build_adjacency_graph


def compute_action_spec_hash(action_name: str, action_def: dict) -> str:
    """Compute a SHA-256 hash of a single action's definition.

    The hash covers the action type, prompt/function name, and arguments,
    producing a stable fingerprint that changes when the action's behavior
    would change.
    """
    parts = [
        action_name,
        action_def.get("Type", ""),
    ]

    # For Agent actions, the prompt is the primary behavioral input
    args = action_def.get("Arguments", {})
    if action_def.get("Type") == "Agent":
        parts.append(args.get("prompt", ""))
    else:
        parts.append(action_def.get("FunctionName", ""))

    # Include sorted arguments (excluding prompt which is already included)
    sorted_args = sorted(
        (k, str(v)) for k, v in args.items() if k != "prompt"
    )
    parts.append(json.dumps(sorted_args, sort_keys=True))

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


def compute_cache_keys(faasr_json: dict) -> dict[str, str]:
    """Compute cache keys for all actions in the workflow.

    Each action's cache key depends on its own spec hash plus the cache keys
    of all its upstream (predecessor) actions. This means any change to an
    upstream action cascades to invalidate all downstream cache keys.

    Returns a dict mapping action_name -> cache_key (hex string).
    """
    action_list = faasr_json.get("ActionList", {})
    if not action_list:
        return {}

    # Build adjacency graph: predecessor -> [successors]
    adj_graph, _ = build_adjacency_graph(faasr_json)

    # Build reverse graph: action -> set of predecessors
    reverse_graph: dict[str, set[str]] = defaultdict(set)
    for pred, successors in adj_graph.items():
        for succ in successors:
            # Strip rank suffix e.g. "ActionName(2)" -> "ActionName"
            succ_name = re.split(r"[()]", succ)[0]
            reverse_graph[succ_name].add(pred)

    # Topological sort (Kahn's algorithm)
    in_degree: dict[str, int] = {name: 0 for name in action_list}
    for name in action_list:
        in_degree[name] = len(reverse_graph.get(name, set()))

    queue = deque(name for name, deg in in_degree.items() if deg == 0)
    topo_order: list[str] = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in adj_graph.get(node, []):
            succ_name = re.split(r"[()]", succ)[0]
            if succ_name in in_degree:
                in_degree[succ_name] -= 1
                if in_degree[succ_name] == 0:
                    queue.append(succ_name)

    # Compute cache keys in topological order (roots first)
    cache_keys: dict[str, str] = {}
    for action_name in topo_order:
        action_def = action_list.get(action_name, {})
        spec_hash = compute_action_spec_hash(action_name, action_def)

        predecessors = reverse_graph.get(action_name, set())
        if not predecessors:
            cache_keys[action_name] = spec_hash
        else:
            upstream_keys = sorted(cache_keys.get(p, "") for p in predecessors)
            combined = spec_hash + "|" + "|".join(upstream_keys)
            cache_keys[action_name] = hashlib.sha256(combined.encode()).hexdigest()

    # Handle any actions not reached by topo sort (disconnected)
    for action_name in action_list:
        if action_name not in cache_keys:
            action_def = action_list[action_name]
            cache_keys[action_name] = compute_action_spec_hash(action_name, action_def)

    return cache_keys


def get_downstream_actions(adj_graph: dict, action_name: str) -> set[str]:
    """BFS to find all downstream actions reachable from action_name.

    Does not include action_name itself in the result.
    """
    visited: set[str] = set()
    queue = deque(adj_graph.get(action_name, []))

    while queue:
        node = queue.popleft()
        node_name = re.split(r"[()]", node)[0]
        if node_name in visited:
            continue
        visited.add(node_name)
        queue.extend(adj_graph.get(node_name, []))

    return visited
