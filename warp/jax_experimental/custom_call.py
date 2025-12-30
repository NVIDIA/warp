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

"""Deprecated JAX integration using custom calls.

.. deprecated:: 1.10.0
    This module is deprecated. Use :mod:`warp.jax_experimental.ffi` instead, which
    provides the current implementation of JAX integration and supports JAX 0.4.25
    and later, including JAX 0.8.0+.

This module contains the legacy custom call-based implementation that is only compatible
with JAX 0.4.25 - 0.7.x and is retained for backward compatibility.
"""

# TODO: Remove after cleaning up the public API.
from warp._src.jax_experimental import custom_call as _custom_call
from warp._src.jax_experimental.custom_call import jax_kernel as jax_kernel


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_custom_call, "warp.jax_experimental", name)
