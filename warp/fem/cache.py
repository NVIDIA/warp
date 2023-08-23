from typing import Callable, Optional

import warp as wp

from warp.fem.operator import Integrand

import re

_kernel_cache = dict()
_struct_cache = dict()
_func_cache = dict()


_key_re = re.compile("[^0-9a-zA-Z_]+")


def get_func(func, suffix=""):
    key = f"{func.__name__}_{suffix}"
    key = _key_re.sub("", key)

    if key not in _func_cache:
        _func_cache[key] = wp.Function(
            func=func,
            key=key,
            namespace="",
            module=wp.get_module(
                func.__module__,
            ),
        )

    return _func_cache[key]


def get_kernel(func, suffix=""):
    module = wp.get_module(func.__module__)
    key = func.__name__ + "_" + suffix
    key = _key_re.sub("", key)

    if key not in _kernel_cache:
        _kernel_cache[key] = wp.Kernel(func=func, key=key, module=module)
    return _kernel_cache[key]


def get_struct(Fields):
    module = wp.get_module(Fields.__module__)
    key = _key_re.sub("", Fields.__qualname__)

    if key not in _struct_cache:
        _struct_cache[key] = wp.codegen.Struct(
            cls=Fields,
            key=key,
            module=module,
        )

    return _struct_cache[key]


def get_integrand_function(
    integrand: Integrand,
    suffix: str,
    annotations=None,
    code_transformers=[],
):
    key = integrand.name + suffix
    key = _key_re.sub("", key)

    if key not in _func_cache:
        _func_cache[key] = wp.Function(
            func=integrand.func,
            key=key,
            namespace="",
            module=integrand.module,
            overloaded_annotations=annotations,
            code_transformers=code_transformers,
        )

    return _func_cache[key]


def get_integrand_kernel(
    integrand: Integrand,
    suffix: str,
    kernel_fn: Optional[Callable] = None,
    code_transformers=[],
):
    module = wp.get_module(f"{integrand.module.name}.{integrand.name}")
    module.options = integrand.module.options
    key = integrand.name + "_" + suffix
    key = _key_re.sub("", key)

    if key not in _kernel_cache:
        if kernel_fn is None:
            return None

        _kernel_cache[key] = wp.Kernel(func=kernel_fn, key=key, module=module, code_transformers=code_transformers)
    return _kernel_cache[key]
