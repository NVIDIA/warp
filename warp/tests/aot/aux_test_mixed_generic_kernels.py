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

"""Auxiliary module for testing AOT compilation with mixed generic kernels.

This module contains:
- A generic kernel WITH overloads that should compile successfully
- A generic kernel WITHOUT overloads that should trigger a warning but not prevent compilation
"""

from typing import Any

import warp as wp


@wp.kernel
def scale_with_overloads(x: wp.array(dtype=Any), s: Any):
    """Generic kernel with overloads - should compile successfully."""
    i = wp.tid()
    x[i] = s * x[i]


# Define overloads for scale_with_overloads
scale_f32 = wp.overload(scale_with_overloads, [wp.array(dtype=wp.float32), wp.float32])
scale_f64 = wp.overload(scale_with_overloads, [wp.array(dtype=wp.float64), wp.float64])


@wp.kernel
def multiply_without_overloads(x: wp.array(dtype=Any), y: wp.array(dtype=Any), result: wp.array(dtype=Any)):
    """Generic kernel without overloads - should trigger a warning."""
    i = wp.tid()
    result[i] = x[i] * y[i]
