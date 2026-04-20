from FaaSr_py.builtin_generators.linspace import generate as _linspace
from FaaSr_py.builtin_generators.logspace import generate as _logspace
from FaaSr_py.builtin_generators.from_list import generate as _from_list
from FaaSr_py.builtin_generators.range_step import generate as _range_step
from FaaSr_py.builtin_generators.random_uniform import generate as _random_uniform
from FaaSr_py.builtin_generators.random_choice import generate as _random_choice

BUILTIN_GENERATORS = {
    "linspace": _linspace,
    "logspace": _logspace,
    "from_list": _from_list,
    "range_step": _range_step,
    "random_uniform": _random_uniform,
    "random_choice": _random_choice,
}
