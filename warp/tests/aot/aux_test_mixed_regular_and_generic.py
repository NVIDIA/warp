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

"""Auxiliary module for testing AOT compilation with mixed regular and generic kernels.

This module contains:
- A regular (non-generic) kernel that should compile successfully
- A generic kernel without overloads that should trigger a warning but not prevent compilation
"""

from typing import Any

import warp as wp


@wp.kernel
def regular_add(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.float32)):
    """Regular non-generic kernel that adds two arrays."""
    i = wp.tid()
    c[i] = a[i] + b[i]


@wp.kernel
def generic_scale(x: wp.array(dtype=Any), s: Any):
    """Generic kernel without overloads - should trigger a warning."""
    i = wp.tid()
    x[i] = s * x[i]
