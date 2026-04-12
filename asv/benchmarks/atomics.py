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

"""Benchmarks for atomic operations and deterministic mode overhead.

All threads write to a single output location (index 0) to maximize contention
and measure worst-case atomic operation performance.
"""

from typing import Any

import numpy as np

import warp as wp

wp.set_module_options({"enable_backward": False})

# Map string parameter names to warp dtypes
DTYPE_MAP = {
    "float32": wp.float32,
    "int32": wp.int32,
}

NUM_ELEMENTS = 32 * 1024 * 1024
DETERMINISTIC_NUM_ELEMENTS = 1 * 1024 * 1024
COUNTER_NUM_ELEMENTS = 4 * 1024 * 1024
DETERMINISTIC_BENCHMARK_SIZES = [64 * 1024, 256 * 1024, 1024 * 1024]


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


@wp.kernel
def scatter_add_kernel(
    vals: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    wp.atomic_add(out, indices[tid], vals[tid])


@wp.kernel(deterministic=True, deterministic_max_records=1)
def scatter_add_kernel_deterministic(
    vals: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    wp.atomic_add(out, indices[tid], vals[tid])


@wp.kernel
def counter_kernel(
    vals: wp.array(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    slot = wp.atomic_add(counter, 0, 1)
    out[slot] = vals[tid]


@wp.kernel(deterministic=True, deterministic_max_records=1)
def counter_kernel_deterministic(
    vals: wp.array(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    slot = wp.atomic_add(counter, 0, 1)
    out[slot] = vals[tid]


@wp.kernel
def zero_float_array_kernel(out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    out[tid] = 0.0


@wp.kernel
def zero_int_array_kernel(out: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    out[tid] = 0


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


class AtomicAddDeterminismOverhead:
    """Benchmark the overhead of deterministic accumulation atomics.

    The benchmark compares the normal atomic-add path against deterministic
    scatter-sort-reduce for the same kernel using CUDA graph replay. A small
    size sweep exposes where deterministic execution crosses over. Two
    destination counts are used:

    - ``1``: worst-case contention, where every thread targets the same output.
    - ``65536``: lower contention, closer to a scatter workload.
    """

    params = (["normal", "deterministic"], [1, 65536], DETERMINISTIC_BENCHMARK_SIZES)
    param_names = ["mode", "num_outputs", "num_elements"]

    repeat = 10
    number = 5

    def setup_cache(self):
        rng = np.random.default_rng(123)
        vals_np = {n: rng.random(n, dtype=np.float32) for n in DETERMINISTIC_BENCHMARK_SIZES}
        indices_np = {}
        for n in DETERMINISTIC_BENCHMARK_SIZES:
            indices_np[n] = {
                1: np.zeros(n, dtype=np.int32),
                65536: rng.integers(0, 65536, size=n, dtype=np.int32),
            }
        return vals_np, indices_np

    def setup(self, cache, mode, num_outputs, num_elements):
        wp.init()
        self.device = wp.get_device("cuda:0")

        vals_np, indices_np = cache
        self.vals = wp.array(vals_np[num_elements], dtype=wp.float32, device=self.device)
        self.indices = wp.array(indices_np[num_elements][num_outputs], dtype=wp.int32, device=self.device)
        self.out = wp.zeros(shape=(num_outputs,), dtype=wp.float32, device=self.device)

        self.kernel = scatter_add_kernel_deterministic if mode == "deterministic" else scatter_add_kernel
        wp.launch(
            zero_float_array_kernel,
            dim=num_outputs,
            inputs=[self.out],
            device=self.device,
        )
        wp.launch(
            self.kernel,
            (num_elements,),
            inputs=[self.vals, self.indices],
            outputs=[self.out],
            device=self.device,
        )
        wp.synchronize_device(self.device)

        with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
            wp.launch(
                zero_float_array_kernel,
                dim=num_outputs,
                inputs=[self.out],
                device=self.device,
            )
            wp.launch(
                self.kernel,
                (num_elements,),
                inputs=[self.vals, self.indices],
                outputs=[self.out],
                device=self.device,
            )

        self.graph = capture.graph

        for _ in range(5):
            wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)

    def time_cuda(self, cache, mode, num_outputs, num_elements):
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)


class AtomicCounterDeterminismOverhead:
    """Benchmark the overhead of deterministic counter/allocator atomics.

    The timed path uses CUDA graph replay and includes resetting the output
    state inside the captured graph so the benchmark isolates device work.
    """

    params = (["normal", "deterministic"], DETERMINISTIC_BENCHMARK_SIZES)
    param_names = ["mode", "num_elements"]

    repeat = 10
    number = 5

    def setup_cache(self):
        rng = np.random.default_rng(321)
        return {n: rng.random(n, dtype=np.float32) for n in DETERMINISTIC_BENCHMARK_SIZES}

    def setup(self, vals_np, mode, num_elements):
        wp.init()
        self.device = wp.get_device("cuda:0")

        self.vals = wp.array(vals_np[num_elements], dtype=wp.float32, device=self.device)
        self.counter = wp.zeros(shape=(1,), dtype=wp.int32, device=self.device)
        self.out = wp.zeros(shape=(num_elements,), dtype=wp.float32, device=self.device)

        self.kernel = counter_kernel_deterministic if mode == "deterministic" else counter_kernel
        wp.launch(
            zero_int_array_kernel,
            dim=1,
            inputs=[self.counter],
            device=self.device,
        )
        wp.launch(
            zero_float_array_kernel,
            dim=num_elements,
            inputs=[self.out],
            device=self.device,
        )
        wp.launch(
            self.kernel,
            (num_elements,),
            inputs=[self.vals, self.counter],
            outputs=[self.out],
            device=self.device,
        )
        wp.synchronize_device(self.device)

        with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
            wp.launch(
                zero_int_array_kernel,
                dim=1,
                inputs=[self.counter],
                device=self.device,
            )
            wp.launch(
                zero_float_array_kernel,
                dim=num_elements,
                inputs=[self.out],
                device=self.device,
            )
            wp.launch(
                self.kernel,
                (num_elements,),
                inputs=[self.vals, self.counter],
                outputs=[self.out],
                device=self.device,
            )

        self.graph = capture.graph

        for _ in range(5):
            wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)

    def time_cuda(self, vals_np, mode, num_elements):
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
