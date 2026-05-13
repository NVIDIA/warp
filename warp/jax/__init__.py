# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX integration for Warp.

This module provides helpers for converting arrays between Warp and JAX, along with
JAX primitives for calling Warp kernels and Warp-backed Python functions from JAX.

The :func:`jax_kernel <warp.jax.ffi.jax_kernel>` function wraps individual Warp
kernels, while :func:`jax_callable <warp.jax.ffi.jax_callable>` wraps Python
functions that launch one or more Warp kernels. Both support automatic
differentiation, custom launch dimensions, and CUDA graph capture.

Usage:
    This module must be explicitly imported::

        import warp.jax

See Also:
    :ref:`jax-ffi` in the user guide for detailed examples and usage patterns.
"""

# isort: skip_file

from warp._src.jax import device_from_jax as device_from_jax
from warp._src.jax import device_to_jax as device_to_jax
from warp._src.jax import dtype_from_jax as dtype_from_jax
from warp._src.jax import dtype_to_jax as dtype_to_jax
from warp._src.jax import from_jax as from_jax
from warp._src.jax import to_jax as to_jax
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
