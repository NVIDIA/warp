# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Deprecated alias for :mod:`warp.jax.ffi`.

.. deprecated:: 1.14.0
    Use :mod:`warp.jax.ffi` instead. This namespace will be removed in Warp 1.16.
"""

# isort: skip_file

from ._deprecation import warn_deprecated_jax_experimental_namespace as _warn_deprecated_jax_experimental_namespace

_warn_deprecated_jax_experimental_namespace("warp.jax_experimental.ffi", "warp.jax.ffi")

from warp.jax.ffi import GraphMode as GraphMode
from warp.jax.ffi import clear_jax_callable_graph_cache as clear_jax_callable_graph_cache
from warp.jax.ffi import (
    get_jax_callable_default_graph_cache_max as get_jax_callable_default_graph_cache_max,
)
from warp.jax.ffi import jax_callable as jax_callable
from warp.jax.ffi import jax_kernel as jax_kernel
from warp.jax.ffi import register_ffi_callback as register_ffi_callback
from warp.jax.ffi import (
    set_jax_callable_default_graph_cache_max as set_jax_callable_default_graph_cache_max,
)
