# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
