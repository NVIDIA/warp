# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Deprecated alias for :mod:`warp.jax`.

.. deprecated:: 1.14.0
    Use :mod:`warp.jax` instead. This namespace will be removed in Warp 1.16.
"""

# isort: skip_file

from ._deprecation import warn_deprecated_jax_experimental_namespace as _warn_deprecated_jax_experimental_namespace

_warn_deprecated_jax_experimental_namespace("warp.jax_experimental", "warp.jax")

from warp.jax import GraphMode as GraphMode
from warp.jax import clear_jax_callable_graph_cache as clear_jax_callable_graph_cache
from warp.jax import device_from_jax as device_from_jax
from warp.jax import device_to_jax as device_to_jax
from warp.jax import dtype_from_jax as dtype_from_jax
from warp.jax import dtype_to_jax as dtype_to_jax
from warp.jax import from_jax as from_jax
from warp.jax import (
    get_jax_callable_default_graph_cache_max as get_jax_callable_default_graph_cache_max,
)
from warp.jax import jax_callable as jax_callable
from warp.jax import jax_kernel as jax_kernel
from warp.jax import register_ffi_callback as register_ffi_callback
from warp.jax import (
    set_jax_callable_default_graph_cache_max as set_jax_callable_default_graph_cache_max,
)
from warp.jax import to_jax as to_jax
