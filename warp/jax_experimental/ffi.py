# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Current FFI-based implementation of JAX integration.

This module provides the Foreign Function Interface (FFI) implementation that supports
JAX 0.4.25 and later, including JAX 0.8.0+. It is the default implementation as of
Warp 1.10.

For low-level use cases, :func:`register_ffi_callback` provides direct FFI callback
registration for functions that don't use Warp-style type annotations.
"""

# isort: skip_file

from warp._src.jax_experimental.ffi import GraphMode as GraphMode
from warp._src.jax_experimental.ffi import jax_callable as jax_callable
from warp._src.jax_experimental.ffi import jax_kernel as jax_kernel
from warp._src.jax_experimental.ffi import register_ffi_callback as register_ffi_callback

from warp._src.jax_experimental.ffi import clear_jax_callable_graph_cache as clear_jax_callable_graph_cache
from warp._src.jax_experimental.ffi import (
    get_jax_callable_default_graph_cache_max as get_jax_callable_default_graph_cache_max,
)
from warp._src.jax_experimental.ffi import (
    set_jax_callable_default_graph_cache_max as set_jax_callable_default_graph_cache_max,
)


# TODO: Remove after cleaning up the public API.

from warp._src.jax_experimental import ffi as _ffi


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_ffi, "warp.jax_experimental", name)
