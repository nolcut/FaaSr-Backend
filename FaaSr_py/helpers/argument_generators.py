import itertools
import logging
import os

from FaaSr_py.builtin_generators import BUILTIN_GENERATORS
from FaaSr_py.helpers.faasr_start_invoke_helper import faasr_get_github_raw
from FaaSr_py.helpers.py_func_helper import faasr_import_function

logger = logging.getLogger(__name__)


def resolve_argument_generators(faasr, action_name: str, rank: int, max_rank: int) -> dict:
    """
    Resolves ArgumentGenerator entries for an action into concrete per-rank argument values.

    Each generator is a Python generator function with signature generate(n, **args) that
    yields exactly n values. resolve_argument_generators calls generate(max_rank, **args)
    and advances to the rank-th yielded value (1-indexed).

    Arguments:
        faasr: FaaSrPayload instance
        action_name: name of the action being dispatched
        rank: 1-indexed rank of the current replica
        max_rank: total number of replicas

    Returns:
        dict mapping argument name -> resolved value (empty if no ArgumentGenerator defined)
    """
    action = faasr["ActionList"][action_name]
    spec = action.get("ArgumentGenerator") or {}
    if not spec:
        return {}

    resolved = {}
    for arg_name, gen in spec.items():
        name = gen["Name"]
        gen_args = gen.get("Arguments") or {}
        generate = _load_generator(faasr, name)
        gen_iter = generate(max_rank, **gen_args)
        resolved[arg_name] = next(itertools.islice(gen_iter, rank - 1, None))

    return resolved


def _load_generator(faasr, name: str):
    """
    Returns a generate callable for the given generator name.
    Checks built-ins first, then fetches from the Generators URL map.
    """
    if name in BUILTIN_GENERATORS:
        return BUILTIN_GENERATORS[name]

    generators_map = faasr.get("Generators") or {}
    if name not in generators_map:
        raise RuntimeError(
            f"Generator '{name}' not found in built-ins or Generators map. "
            f"Available built-ins: {sorted(BUILTIN_GENERATORS)}"
        )

    path = generators_map[name]
    if isinstance(path, list):
        path = path[0]

    token = os.getenv("GH_PAT")
    target_dir = f"/tmp/generators/{faasr['InvocationID']}"
    os.makedirs(target_dir, exist_ok=True)

    file_name = os.path.basename(path)
    dest = os.path.join(target_dir, file_name)

    if not os.path.exists(dest):
        logger.info(f"Fetching custom generator '{name}' from {path}")
        content = faasr_get_github_raw(token, path)
        with open(dest, "w") as f:
            f.write(content)

    from pathlib import Path
    generate = faasr_import_function(Path(dest), "generate")
    if generate is None:
        raise RuntimeError(
            f"Could not find a callable named 'generate' in generator file for '{name}' ({path})"
        )
    return generate
