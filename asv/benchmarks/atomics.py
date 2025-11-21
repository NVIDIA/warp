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

"""Benchmarks for atomic operations under high thread contention.

All threads write to a single output location (index 0) to maximize contention
and measure worst-case atomic operation performance.
"""

from typing import Any

import numpy as np

import warp as wp

# Map string parameter names to warp dtypes
DTYPE_MAP = {
    "float32": wp.float32,
    "int32": wp.int32,
}

NUM_ELEMENTS = 32 * 1024 * 1024


@wp.kernel
def max_kernel(
    vals: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
):
    tid = wp.tid()
    val = vals[tid]
    wp.atomic_max(out, 0, val)  # All threads contend on out[0]


@wp.kernel
def min_kernel(
    vals: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
):
    tid = wp.tid()
    val = vals[tid]
    wp.atomic_min(out, 0, val)  # All threads contend on out[0]


class AtomicMax:
    """Benchmark wp.atomic_max() with high thread contention.

    Uses 4x larger arrays (128M elements) to reduce measurement variation,
    as atomic_max showed ~10% variation with the default 32M elements.
    """

    params = ["float32", "int32"]
    param_names = ["dtype"]

    repeat = 50
    number = 15

    # Use 4x more elements to reduce measurement variation
    num_elements = 4 * NUM_ELEMENTS

    def setup_cache(self):
        rng = np.random.default_rng(42)
        # Generate vals_np for each dtype in DTYPE_MAP
        vals_np_dict = {}
        for dtype_str_key, dtype in DTYPE_MAP.items():
            if dtype == wp.float32:
                vals_np = rng.random(self.num_elements).astype(np.float32)
            elif dtype == wp.int32:
                vals_np = rng.integers(0, 2**31 - 1, size=self.num_elements, dtype=np.int32)
            else:
                vals_np = None
            vals_np_dict[dtype_str_key] = vals_np

        return vals_np_dict

    def setup(self, vals_np_dict, dtype_str):
        wp.init()
        self.device = wp.get_device("cuda:0")

        dtype = DTYPE_MAP[dtype_str]

        self.vals = wp.array(vals_np_dict[dtype_str], dtype=dtype, device=self.device)
        self.out = wp.zeros(shape=(1,), dtype=dtype, device=self.device)

        self.cmd = wp.launch(
            max_kernel,
            (self.num_elements,),
            inputs=[self.vals],
            outputs=[self.out],
            device=self.device,
            record_cmd=True,
        )

        # Launch once to compile
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self, vals_np_dict, dtype_str):
        self.out.zero_()
        self.cmd.launch()
        wp.synchronize_device(self.device)


class AtomicMin:
    """Benchmark wp.atomic_min() with high thread contention.

    Uses standard array size (32M elements) as measurements are already stable.
    """

    params = ["float32", "int32"]
    param_names = ["dtype"]

    repeat = 100
    number = 25

    def setup_cache(self):
        rng = np.random.default_rng(42)
        # Generate vals_np for each dtype in DTYPE_MAP
        vals_np_dict = {}
        for dtype_str_key, dtype in DTYPE_MAP.items():
            if dtype == wp.float32:
                vals_np = rng.random(NUM_ELEMENTS).astype(np.float32)
            elif dtype == wp.int32:
                vals_np = rng.integers(0, 2**31 - 1, size=NUM_ELEMENTS, dtype=np.int32)
            else:
                vals_np = None
            vals_np_dict[dtype_str_key] = vals_np

        return vals_np_dict

    def setup(self, vals_np_dict, dtype_str):
        wp.init()
        self.device = wp.get_device("cuda:0")

        dtype = DTYPE_MAP[dtype_str]

        self.vals = wp.array(vals_np_dict[dtype_str], dtype=dtype, device=self.device)
        self.out = wp.zeros(shape=(1,), dtype=dtype, device=self.device)

        self.cmd = wp.launch(
            min_kernel,
            (NUM_ELEMENTS,),
            inputs=[self.vals],
            outputs=[self.out],
            device=self.device,
            record_cmd=True,
        )

        # Launch once to compile
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self, vals_np_dict, dtype_str):
        self.out.zero_()
        self.cmd.launch()
        wp.synchronize_device(self.device)
