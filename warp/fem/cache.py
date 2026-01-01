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

"""Caching utilities for specializing and caching kernels and functions.

This module provides decorators that create specialized, cached versions of kernels
and functions based on a suffix parameter. The suffix typically encodes type or domain
information, allowing a single function definition to be compiled into multiple optimized
versions for different argument types or problem domains.
"""

# isort: skip_file

from warp._src.fem.cache import dynamic_kernel as dynamic_kernel
from warp._src.fem.cache import dynamic_func as dynamic_func


# TODO: Remove after cleaning up the public API.

from warp._src.fem import cache as _cache


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_cache, "warp.fem", name)
