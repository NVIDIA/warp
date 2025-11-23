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

"""Shared utilities for matrix tests."""

import numpy as np

import warp as wp

# Standardized type lists
np_signed_int_types = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
]

np_unsigned_int_types = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

np_int_types = np_signed_int_types + np_unsigned_int_types
np_float_types = [np.float16, np.float32, np.float64]
np_scalar_types = np_int_types + np_float_types


def randvals(rng, shape, dtype):
    """Generate random values appropriate for the given dtype.

    Args:
        rng: NumPy random number generator.
        shape: Shape of the array to generate.
        dtype: NumPy dtype for the generated values.

    Returns:
        NumPy array of random values with appropriate range for the dtype.
    """
    if dtype in np_float_types:
        return rng.standard_normal(size=shape).astype(dtype)
    elif dtype in [np.int8, np.uint8]:
        return rng.integers(1, high=3, size=shape, dtype=dtype)
    return rng.integers(1, high=5, size=shape, dtype=dtype)


def getkernel(kernel_cache, func, suffix=""):
    """Get or create a cached kernel.

    Args:
        kernel_cache: Dictionary to cache kernels (module-local).
        func: Kernel function to wrap.
        suffix: Optional suffix for the kernel key.

    Returns:
        Cached or newly created wp.Kernel.
    """
    key = func.__name__ + "_" + suffix
    if key not in kernel_cache:
        kernel_cache[key] = wp.Kernel(func=func, key=key)
    return kernel_cache[key]


def get_select_kernel(kernel_cache, dtype):
    """Create a kernel that selects a single element from an array.

    Args:
        kernel_cache: Dictionary to cache kernels (module-local).
        dtype: Warp dtype for the array element.

    Returns:
        Cached or newly created selection kernel.
    """

    def output_select_kernel_fn(
        input: wp.array(dtype=dtype),
        index: int,
        out: wp.array(dtype=dtype),
    ):
        out[0] = input[index]

    return getkernel(kernel_cache, output_select_kernel_fn, suffix=dtype.__name__)
