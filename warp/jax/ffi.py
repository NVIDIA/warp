# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current FFI-based implementation of JAX integration.

This module provides the Foreign Function Interface (FFI) implementation that supports
JAX 0.4.25 and later, including JAX 0.8.0+. It is the default implementation as of
Warp 1.10.

For low-level use cases, :func:`register_ffi_callback` provides direct FFI callback
registration for functions that do not use Warp-style type annotations.
"""

# isort: skip_file

from warp._src.jax.ffi import GraphMode as GraphMode
from warp._src.jax.ffi import clear_jax_callable_graph_cache as clear_jax_callable_graph_cache
from warp._src.jax.ffi import (
    get_jax_callable_default_graph_cache_max as get_jax_callable_default_graph_cache_max,
)
from warp._src.jax.ffi import jax_callable as jax_callable
from warp._src.jax.ffi import jax_kernel as jax_kernel
from warp._src.jax.ffi import register_ffi_callback as register_ffi_callback
from warp._src.jax.ffi import (
    set_jax_callable_default_graph_cache_max as set_jax_callable_default_graph_cache_max,
)
