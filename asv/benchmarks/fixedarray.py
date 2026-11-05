# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark zero initialization of fixed-size scratch arrays."""

import numpy as np

import warp as wp

from .benchmarks_utils import CudaEventTimer, setup_once

NUM_THREADS = 1 << 18
SCRATCH_SIZE = 1024


@wp.kernel(enable_backward=False)
def fixedarray_zeros_kernel(output: wp.array[float]):
    i = wp.tid()
    scratch = wp.zeros(shape=SCRATCH_SIZE, dtype=float)
    # Keep the scratch array live while leaving most elements zero.
    scratch[i % SCRATCH_SIZE] = 1.0
    output[i] = scratch[0] + scratch[SCRATCH_SIZE - 1]


class FixedArrayZerosCUDA:
    """Measure in-kernel fixed-array zero initialization on CUDA."""

    timer = CudaEventTimer()
    # Eight samples keep the confidence interval tight across independent runs.
    repeat = 8
    number = 1

    @setup_once
    def setup(self):
        device = wp.get_device("cuda:0")
        self.output = wp.empty(NUM_THREADS, dtype=float, device=device)
        self.command = wp.launch(
            fixedarray_zeros_kernel,
            dim=NUM_THREADS,
            outputs=[self.output],
            device=device,
            record_cmd=True,
        )
        for _ in range(5):
            self.command.launch()

        indices = np.arange(NUM_THREADS) % SCRATCH_SIZE
        expected = ((indices == 0) | (indices == SCRATCH_SIZE - 1)).astype(np.float32)
        np.testing.assert_array_equal(self.output.numpy(), expected)

    def time_zeros(self):
        self.command.launch()
