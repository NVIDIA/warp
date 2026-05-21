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

"""Benchmarks for deterministic-mode overhead on atomic accumulation and counter kernels.

Compares the normal atomic path against the deterministic scatter-sort-reduce
(accumulation) and two-pass counter (allocator) paths using CUDA graph replay.
"""

import inspect

import numpy as np

import warp as wp

DETERMINISTIC_BENCHMARK_SIZES = [64 * 1024, 256 * 1024, 1024 * 1024]
DETERMINISM_SUPPORTED = "module_options" in inspect.signature(wp.kernel).parameters
DETERMINISTIC_BENCHMARK_MODES = ("normal", "deterministic")
DETERMINISTIC_KERNEL_OPTIONS = {"enable_backward": False}
if DETERMINISM_SUPPORTED:
    DETERMINISTIC_KERNEL_OPTIONS = {
        "enable_backward": False,
        "module": "unique",
        "module_options": {
            "deterministic": "run_to_run",
            "deterministic_max_records": 1,
        },
    }


@wp.kernel(enable_backward=False)
def scatter_add_kernel(
    vals: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    wp.atomic_add(out, indices[tid], vals[tid])


@wp.kernel(**DETERMINISTIC_KERNEL_OPTIONS)
def scatter_add_kernel_deterministic(
    vals: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    wp.atomic_add(out, indices[tid], vals[tid])


@wp.kernel(enable_backward=False)
def counter_kernel(
    vals: wp.array(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    slot = wp.atomic_add(counter, 0, 1)
    out[slot] = vals[tid]


@wp.kernel(**DETERMINISTIC_KERNEL_OPTIONS)
def counter_kernel_deterministic(
    vals: wp.array(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    slot = wp.atomic_add(counter, 0, 1)
    out[slot] = vals[tid]


@wp.kernel(enable_backward=False)
def zero_float_array_kernel(out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    out[tid] = 0.0


@wp.kernel(enable_backward=False)
def zero_int_array_kernel(out: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    out[tid] = 0


class AtomicAddDeterminismOverhead:
    """Benchmark the overhead of deterministic accumulation atomics.

    The benchmark compares the normal atomic-add path against deterministic
    scatter-sort-reduce for the same kernel using CUDA graph replay. A small
    size sweep exposes where deterministic execution crosses over. Two
    destination counts are used:

    - ``1``: worst-case contention, where every thread targets the same output.
    - ``65536``: lower contention, closer to a scatter workload.
    """

    params = (DETERMINISTIC_BENCHMARK_MODES, (1, 65536), tuple(DETERMINISTIC_BENCHMARK_SIZES))
    param_names = ("mode", "num_outputs", "num_elements")

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

        if mode == "deterministic" and not DETERMINISM_SUPPORTED:
            raise NotImplementedError("deterministic kernel options are not supported by this Warp version")

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

    params = (DETERMINISTIC_BENCHMARK_MODES, tuple(DETERMINISTIC_BENCHMARK_SIZES))
    param_names = ("mode", "num_elements")

    repeat = 10
    number = 5

    def setup_cache(self):
        rng = np.random.default_rng(321)
        return {n: rng.random(n, dtype=np.float32) for n in DETERMINISTIC_BENCHMARK_SIZES}

    def setup(self, vals_np, mode, num_elements):
        wp.init()
        self.device = wp.get_device("cuda:0")

        if mode == "deterministic" and not DETERMINISM_SUPPORTED:
            raise NotImplementedError("deterministic kernel options are not supported by this Warp version")

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
