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

"""Experimental JAX integration for calling Warp kernels from JAX.

This module enables using Warp kernels as JAX primitives, allowing them to be
called inside jitted JAX functions. The
:func:`jax_kernel <warp.jax_experimental.ffi.jax_kernel>` function wraps
individual Warp kernels, while
:func:`jax_callable <warp.jax_experimental.ffi.jax_callable>` wraps Python
functions that launch multiple kernels. Both support automatic differentiation,
custom launch dimensions, and CUDA graph capture.

.. caution::
    This module is experimental and less stable than the core Warp API. The interface
    may change as new functionality is added and to accommodate changes in upcoming
    JAX library versions.

Usage:
    This module must be explicitly imported::

        import warp.jax_experimental

See Also:
    :ref:`jax-ffi` in the user guide for detailed examples and usage patterns.
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

from warp._src import jax_experimental as _jax_experimental


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_jax_experimental, "warp", name)
