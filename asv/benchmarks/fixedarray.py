# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark fixed-size scratch arrays created inside CUDA kernels.

Each thread creates a fixed-size array with ``wp.zeros()`` or ``wp.empty()``,
then either writes every element and reads them all back (``fill``), or
writes and reads a single element at a runtime index (``write_one``).
"""

import numpy as np

import warp as wp

from .benchmarks_utils import CudaEventTimer, setup_once

NUM_THREADS = 1 << 18


# The two factories differ only in the built-in that creates the scratch array. They
# can't share a kernel through a variable holding the built-in: the kernel's module
# hash doesn't cover it, so both would load the same cached module.
def _create_zeros_kernels(size):
    @wp.kernel(enable_backward=False, module="unique")
    def fill(output: wp.array[float]):
        i = wp.tid()
        scratch = wp.zeros(shape=size, dtype=float)
        for j in range(size):
            scratch[j] = float(j + i % 8)
        total = float(0.0)
        for j in range(size):
            total += scratch[(j * 7 + i) % size]
        output[i] = total

    @wp.kernel(enable_backward=False, module="unique")
    def write_one(output: wp.array[float]):
        i = wp.tid()
        scratch = wp.zeros(shape=size, dtype=float)
        j = (i * 7919) % size
        scratch[j] = float(i)
        output[i] = scratch[j]

    return {"fill": fill, "write_one": write_one}


def _create_empty_kernels(size):
    @wp.kernel(enable_backward=False, module="unique")
    def fill(output: wp.array[float]):
        i = wp.tid()
        scratch = wp.empty(shape=size, dtype=float)
        for j in range(size):
            scratch[j] = float(j + i % 8)
        total = float(0.0)
        for j in range(size):
            total += scratch[(j * 7 + i) % size]
        output[i] = total

    @wp.kernel(enable_backward=False, module="unique")
    def write_one(output: wp.array[float]):
        i = wp.tid()
        scratch = wp.empty(shape=size, dtype=float)
        j = (i * 7919) % size
        scratch[j] = float(i)
        output[i] = scratch[j]

    return {"fill": fill, "write_one": write_one}


_KERNEL_FACTORIES = {
    "empty": _create_empty_kernels,
    "zeros": _create_zeros_kernels,
}


def _expected_output(size, access):
    indices = np.arange(NUM_THREADS)
    if access == "fill":
        # Exact in single precision: every partial sum stays below 2**24.
        return (size * (size - 1) // 2 + size * (indices % 8)).astype(np.float32)
    return indices.astype(np.float32)


class FixedArrayCUDA:
    """Measure in-kernel fixed-size scratch arrays on CUDA."""

    timer = CudaEventTimer()
    # Eight samples keep the confidence interval tight across independent runs.
    repeat = 8
    number = 1
    params = (["empty", "zeros"], [4, 64, 1024], ["fill", "write_one"])
    param_names = ["alloc", "size", "access"]

    @setup_once
    def setup(self, alloc, size, access):
        if alloc not in wp._src.context.builtin_functions:
            raise NotImplementedError(f"the wp.{alloc}() built-in is not available in this Warp version")

        device = wp.get_device("cuda:0")
        kernel = _KERNEL_FACTORIES[alloc](size)[access]
        self.output = wp.empty(NUM_THREADS, dtype=float, device=device)
        self.command = wp.launch(
            kernel,
            dim=NUM_THREADS,
            outputs=[self.output],
            device=device,
            record_cmd=True,
        )
        for _ in range(5):
            self.command.launch()

        np.testing.assert_array_equal(self.output.numpy(), _expected_output(size, access))

    def time_cuda(self, alloc, size, access):
        self.command.launch()
