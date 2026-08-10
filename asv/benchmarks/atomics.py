# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for atomic operations under high thread contention.

All threads write to a single output location (index 0) to maximize contention
and measure worst-case atomic operation performance.
"""

import math
from typing import Any

import numpy as np

import warp as wp

# Map string parameter names to warp dtypes
DTYPE_MAP = {
    "float32": wp.float32,
    "int32": wp.int32,
}

# Identity element of each reduction. out[0] must start here so that a thread holding a
# more extreme value actually takes the update path: seeding the minimum with zero
# against non-negative inputs leaves out[0] at zero and the contended update never fires.
MAX_INIT = {"float32": -math.inf, "int32": -(2**31)}
MIN_INIT = {"float32": math.inf, "int32": 2**31 - 1}

NUM_ELEMENTS = 32 * 1024 * 1024

# asv re-runs setup() before every timed call, so uploading the input array there would
# dominate wall time (repeat * number uploads per process) without being measured. The
# device-side state is immutable across iterations apart from the output, which
# time_cuda() resets itself, so it is safe to build it once per process and reuse it.
_LAUNCH_CACHE = {}


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


def _get_launch(kernel, vals_np, dtype, device, init_value):
    """Return the cached ``(vals, out, cmd)`` triple for a kernel and input array.

    The launch is recorded and compiled on the first call for a given key and reused
    afterwards, keeping setup() out of the benchmark's wall-clock budget.
    """

    key = (kernel.key, dtype, len(vals_np), device.alias)

    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        vals = wp.array(vals_np, dtype=dtype, device=device)
        out = wp.full(shape=(1,), value=init_value, dtype=dtype, device=device)

        cmd = wp.launch(
            kernel,
            (len(vals_np),),
            inputs=[vals],
            outputs=[out],
            device=device,
            record_cmd=True,
        )

        # Launch once to compile
        cmd.launch()
        wp.synchronize_device(device)

        entry = _LAUNCH_CACHE[key] = (vals, out, cmd)

    return entry


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

        self.init_value = MAX_INIT[dtype_str]
        self.vals, self.out, self.cmd = _get_launch(
            max_kernel, vals_np_dict[dtype_str], dtype, self.device, self.init_value
        )

    def time_cuda(self, vals_np_dict, dtype_str):
        self.out.fill_(self.init_value)
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

        self.init_value = MIN_INIT[dtype_str]
        self.vals, self.out, self.cmd = _get_launch(
            min_kernel, vals_np_dict[dtype_str], dtype, self.device, self.init_value
        )

    def time_cuda(self, vals_np_dict, dtype_str):
        self.out.fill_(self.init_value)
        self.cmd.launch()
        wp.synchronize_device(self.device)
