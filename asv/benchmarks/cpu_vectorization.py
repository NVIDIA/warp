# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark CPU loop and SLP vectorization."""

import numpy as np

import warp as wp

from .benchmarks_utils import setup_once

VEC4_ELEMENTS = 1 << 16

# Keep the arrays (384 KB) cache-resident so the timing measures vectorized
# arithmetic rather than memory placement, which varies between processes on
# CI runners. Repeating the row sweep restores the arithmetic per launch.
LOOP_ROWS = 128
LOOP_COLUMNS = 1 << 8
LOOP_REPEATS = 16
# The three loop arrays share one page-aligned allocation. Each array is a
# multiple of 4 KiB, so this gap puts them 1 KiB and 2 KiB apart modulo 4 KiB.
LOOP_ARRAY_GAP_BYTES = 1024
# Larger counts stop LLVM from fully unrolling this loop, which leaves the
# column loop non-innermost and prevents its vectorization.
ARITHMETIC_ITERATIONS = 16


@wp.kernel(enable_backward=False)
def vec4_component_kernel(a: wp.array[wp.vec4f], b: wp.array[wp.vec4f], output: wp.array[wp.vec4f]):
    i = wp.tid()
    value = a[i]
    factor = b[i]
    for _ in range(ARITHMETIC_ITERATIONS):
        value = wp.cw_mul(value, factor) + a[i]
    output[i] = value


# Scalar arrays keep aggregate ABI lowering from introducing vector operations
# before LLVM can vectorize the loop across tasks.
@wp.kernel(enable_backward=False)
def scalar_task_loop_kernel(a: wp.array[float], b: wp.array[float], output: wp.array[float]):
    i = wp.tid()
    value = a[i]
    factor = b[i]
    for _ in range(ARITHMETIC_ITERATIONS):
        value = value * factor + a[i]
    output[i] = value


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
    """Measure CPU loop and SLP vectorization."""

    repeat = 15
    number = 10

    @setup_once
    def setup(self):
        wp.init()
        device = wp.get_device("cpu")

        vec4_values = np.linspace(-1.0, 1.0, VEC4_ELEMENTS * 4, dtype=np.float32).reshape(VEC4_ELEMENTS, 4)
        vec4_factors = np.tile(np.array((1.25, -0.5, 2.0, 0.75), dtype=np.float32), (VEC4_ELEMENTS, 1))
        self.vec4_a = wp.array(vec4_values, dtype=wp.vec4f, device=device)
        self.vec4_b = wp.array(vec4_factors, dtype=wp.vec4f, device=device)
        self.vec4_output = wp.empty_like(self.vec4_a)
        self.vec4_command = wp.launch(
            vec4_component_kernel,
            dim=VEC4_ELEMENTS,
            inputs=[self.vec4_a, self.vec4_b],
            outputs=[self.vec4_output],
            device=device,
            block_dim=1,
            record_cmd=True,
        )

        self.scalar_a = wp.array(vec4_values.reshape(-1), dtype=float, device=device)
        self.scalar_b = wp.array(vec4_factors.reshape(-1), dtype=float, device=device)
        self.scalar_output = wp.empty_like(self.scalar_a)
        self.scalar_command = wp.launch(
            scalar_task_loop_kernel,
            dim=VEC4_ELEMENTS * 4,
            inputs=[self.scalar_a, self.scalar_b],
            outputs=[self.scalar_output],
            device=device,
            block_dim=1,
            record_cmd=True,
        )

        element_count = LOOP_ROWS * LOOP_COLUMNS
        loop_values = np.linspace(-1.0, 1.0, element_count, dtype=np.float32)
        loop_factors = np.linspace(0.25, 1.75, element_count, dtype=np.float32)
        # Separately allocated arrays land at offsets modulo 4 KiB that differ
        # between processes. At some relative offsets, loads from `a` and `b`
        # alias earlier stores to `output`, which slows each process by a
        # different amount. Slicing the arrays from one page-aligned
        # allocation fixes their placement.
        float_size = np.dtype(np.float32).itemsize
        page_elements = 4096 // float_size
        stride = element_count + LOOP_ARRAY_GAP_BYTES // float_size
        self.loop_storage = wp.empty(3 * stride + page_elements, dtype=float, device=device)
        start = (-self.loop_storage.ptr % 4096) // float_size
        self.loop_a, self.loop_b, self.loop_output = (
            self.loop_storage[start + k * stride : start + k * stride + element_count] for k in range(3)
        )
        self.loop_a.assign(loop_values)
        self.loop_b.assign(loop_factors)
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
            self.vec4_command.launch()
            self.scalar_command.launch()
            self.loop_command.launch()

        vec4_expected = vec4_values.copy()
        loop_expected = loop_values.copy()
        for _ in range(ARITHMETIC_ITERATIONS):
            vec4_expected = vec4_expected * vec4_factors + vec4_values
            loop_expected = loop_expected * loop_factors + loop_values
        np.testing.assert_allclose(self.vec4_output.numpy(), vec4_expected, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(self.scalar_output.numpy(), vec4_expected.reshape(-1), rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(self.loop_output.numpy(), loop_expected, rtol=1.0e-5, atol=1.0e-5)

    def time_vec4_component_arithmetic(self):
        """Measure the full CPU pipeline; this case alone does not isolate SLP."""
        self.vec4_command.launch()

    def time_scalar_task_loop_arithmetic(self):
        """Measure arithmetic across CPU tasks with scalar array elements."""
        self.scalar_command.launch()

    def time_linear_array_loop_arithmetic(self):
        self.loop_command.launch()
