# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Deprecated compatibility namespace for FFI-based JAX integration.

.. deprecated:: 1.14.0
    Use top-level :mod:`warp` JAX APIs instead. This namespace will be removed in Warp 1.18.
"""

# isort: skip_file

from ._deprecation import (
    warn_deprecated_jax_experimental_graph_cache_getter as _warn_deprecated_jax_experimental_graph_cache_getter,
)
from ._deprecation import (
    warn_deprecated_jax_experimental_graph_cache_setter as _warn_deprecated_jax_experimental_graph_cache_setter,
)
from ._deprecation import (
    warn_deprecated_jax_experimental_namespace as _warn_deprecated_jax_experimental_namespace,
)

_warn_deprecated_jax_experimental_namespace("warp.jax_experimental.ffi", "top-level `warp` JAX APIs")

from warp._src.jax import ffi as _ffi
from warp._src.jax.ffi import GraphMode as GraphMode
from warp._src.jax.ffi import JaxCallableGraphMode as JaxCallableGraphMode
from warp._src.jax.ffi import JaxModulePreloadMode as JaxModulePreloadMode
from warp._src.jax.ffi import ModulePreloadMode as ModulePreloadMode
from warp._src.jax.ffi import clear_jax_callable_graph_cache as clear_jax_callable_graph_cache
from warp._src.jax.ffi import jax_callable as _jax_callable
from warp._src.jax.ffi import jax_kernel as jax_kernel
from warp._src.jax.ffi import register_ffi_callback as register_ffi_callback


def jax_callable(
    func,
    num_outputs: int = 1,
    graph_mode: JaxCallableGraphMode = JaxCallableGraphMode.JAX,
    vmap_method: str | None = "broadcast_all",
    output_dims=None,
    in_out_argnames=None,
    stage_in_argnames=None,
    stage_out_argnames=None,
    graph_cache_max: int | None = None,
    module_preload_mode: JaxModulePreloadMode = JaxModulePreloadMode.CURRENT_DEVICE,
    has_side_effect: bool = False,
):
    if graph_cache_max is None:
        graph_cache_max = _ffi.get_jax_callable_default_graph_cache_max()

    return _jax_callable(
        func,
        num_outputs=num_outputs,
        graph_mode=graph_mode,
        vmap_method=vmap_method,
        output_dims=output_dims,
        in_out_argnames=in_out_argnames,
        stage_in_argnames=stage_in_argnames,
        stage_out_argnames=stage_out_argnames,
        graph_cache_max=graph_cache_max,
        module_preload_mode=module_preload_mode,
        has_side_effect=has_side_effect,
    )


def get_jax_callable_default_graph_cache_max():
    _warn_deprecated_jax_experimental_graph_cache_getter(
        "warp.jax_experimental.ffi.get_jax_callable_default_graph_cache_max"
    )
    return _ffi.get_jax_callable_default_graph_cache_max()


def set_jax_callable_default_graph_cache_max(cache_max: int | None):
    _warn_deprecated_jax_experimental_graph_cache_setter(
        "warp.jax_experimental.ffi.set_jax_callable_default_graph_cache_max"
    )
    _ffi.set_jax_callable_default_graph_cache_max(cache_max)
