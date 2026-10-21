# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark CPU loop vectorization over contiguous arrays."""

import numpy as np

import warp as wp

from .benchmarks_utils import setup_once

# Keep the arrays (384 KB) cache-resident so the timing measures vectorized
# arithmetic rather than memory placement, which varies between processes on
# CI runners. Repeating the row sweep restores the arithmetic per launch.
LOOP_ROWS = 128
LOOP_COLUMNS = 1 << 8
LOOP_REPEATS = 16
# Larger counts stop LLVM from fully unrolling this loop, which leaves the
# column loop non-innermost and prevents its vectorization.
ARITHMETIC_ITERATIONS = 16


@wp.kernel(enable_backward=False)
def linear_array_loop_kernel(a: wp.array[float], b: wp.array[float], output: wp.array[float]):
    row = wp.tid()
    begin = row * LOOP_COLUMNS
    for _ in range(LOOP_REPEATS):
        for column in range(LOOP_COLUMNS):
            i = begin + column
            value = a[i]
            factor = b[i]
            for _ in range(ARITHMETIC_ITERATIONS):
                value = value * factor + a[i]
            output[i] = value


class CpuVectorization:
    """Measure CPU loop vectorization."""

    repeat = 15
    number = 10

    @setup_once
    def setup(self):
        wp.init()
        device = wp.get_device("cpu")

        element_count = LOOP_ROWS * LOOP_COLUMNS
        loop_values = np.linspace(-1.0, 1.0, element_count, dtype=np.float32)
        loop_factors = np.linspace(0.25, 1.75, element_count, dtype=np.float32)
        self.loop_a = wp.array(loop_values, dtype=float, device=device)
        self.loop_b = wp.array(loop_factors, dtype=float, device=device)
        self.loop_output = wp.empty_like(self.loop_a)
        self.loop_command = wp.launch(
            linear_array_loop_kernel,
            dim=LOOP_ROWS,
            inputs=[self.loop_a, self.loop_b],
            outputs=[self.loop_output],
            device=device,
            block_dim=1,
            record_cmd=True,
        )

        for _ in range(5):
            self.loop_command.launch()

        loop_expected = loop_values.copy()
        for _ in range(ARITHMETIC_ITERATIONS):
            loop_expected = loop_expected * loop_factors + loop_values
        np.testing.assert_allclose(self.loop_output.numpy(), loop_expected, rtol=1.0e-5, atol=1.0e-5)

    def time_linear_array_loop_arithmetic(self):
        self.loop_command.launch()
