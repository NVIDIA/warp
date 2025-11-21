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

# Standardized type lists
np_signed_int_types = [
    np.int8,  # smallest - edge case testing
    np.int32,  # most common
    np.int64,  # largest - edge case testing
]

np_unsigned_int_types = [
    np.uint8,  # smallest - edge case testing
    np.uint32,  # most common
    np.uint64,  # largest - edge case testing
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
