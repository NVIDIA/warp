# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for signed integer floor division."""

import numpy as np

import warp as wp

from .benchmarks_utils import setup_once

DIVISIONS_PER_THREAD = 8
CPU_NUM_ELEMENTS = 1 << 20
CUDA_NUM_ELEMENTS = 1 << 20
DIVISOR_KINDS = ("runtime", "constant")


@wp.kernel(enable_backward=False)
def floordiv_runtime_kernel(
    values: wp.array[wp.int32],
    divisors: wp.array[wp.int32],
    output: wp.array[wp.int32],
):
    tid = wp.tid()
    value = values[tid]
    divisor = divisors[tid]
    total = wp.int32(0)
    for offset in range(DIVISIONS_PER_THREAD):
        total += (value + offset) // divisor
    output[tid] = total


@wp.kernel(enable_backward=False)
def floordiv_constant_kernel(
    values: wp.array[wp.int32],
    output: wp.array[wp.int32],
):
    tid = wp.tid()
    value = values[tid]
    total = wp.int32(0)
    for offset in range(DIVISIONS_PER_THREAD):
        total += (value + offset) // 32
    output[tid] = total


class SignedFloorDivCPU:
    """Measure mixed-sign ``int32`` floor division on CPU."""

    params = DIVISOR_KINDS
    param_names = ("divisor_kind",)
    repeat = 15  # Samples have low variance at 3-24 ms each.
    number = 1

    @setup_once
    def setup(self, divisor_kind):
        wp.init()
        self.device = wp.get_device("cpu")

        indices = np.arange(CPU_NUM_ELEMENTS, dtype=np.int64)
        values_np = ((indices * 104729 + 12345) % 200000000 + 1).astype(np.int32)
        divisors_np = (indices % 97 + 2).astype(np.int32)
        values_np[1::2] *= -1
        divisors_np[2::4] *= -1
        divisors_np[3::4] *= -1

        self.values = wp.array(values_np, dtype=wp.int32, device=self.device)
        self.output = wp.empty(CPU_NUM_ELEMENTS, dtype=wp.int32, device=self.device)

        if divisor_kind == "runtime":
            self.divisors = wp.array(divisors_np, dtype=wp.int32, device=self.device)
            self.command = wp.launch(
                floordiv_runtime_kernel,
                dim=CPU_NUM_ELEMENTS,
                inputs=[self.values, self.divisors],
                outputs=[self.output],
                device=self.device,
                record_cmd=True,
            )
        else:
            self.command = wp.launch(
                floordiv_constant_kernel,
                dim=CPU_NUM_ELEMENTS,
                inputs=[self.values],
                outputs=[self.output],
                device=self.device,
                record_cmd=True,
            )

        for _ in range(5):
            self.command.launch()

    def time_division(self, divisor_kind):
        self.command.launch()


class SignedFloorDivCUDA:
    """Measure mixed-sign ``int32`` floor division on CUDA."""

    params = DIVISOR_KINDS
    param_names = ("divisor_kind",)
    repeat = 31  # Robust 99% confidence intervals across independent CUDA runs.
    number = 16  # Amortize launch jitter for the 20-31 us kernels.

    @setup_once
    def setup(self, divisor_kind):
        wp.init()
        self.device = wp.get_device("cuda:0")

        indices = np.arange(CUDA_NUM_ELEMENTS, dtype=np.int64)
        values_np = ((indices * 104729 + 12345) % 200000000 + 1).astype(np.int32)
        divisors_np = (indices % 97 + 2).astype(np.int32)
        values_np[1::2] *= -1
        divisors_np[2::4] *= -1
        divisors_np[3::4] *= -1

        self.values = wp.array(values_np, dtype=wp.int32, device=self.device)
        self.output = wp.empty(CUDA_NUM_ELEMENTS, dtype=wp.int32, device=self.device)

        if divisor_kind == "runtime":
            self.divisors = wp.array(divisors_np, dtype=wp.int32, device=self.device)
            self.command = wp.launch(
                floordiv_runtime_kernel,
                dim=CUDA_NUM_ELEMENTS,
                inputs=[self.values, self.divisors],
                outputs=[self.output],
                device=self.device,
                record_cmd=True,
            )
        else:
            self.command = wp.launch(
                floordiv_constant_kernel,
                dim=CUDA_NUM_ELEMENTS,
                inputs=[self.values],
                outputs=[self.output],
                device=self.device,
                record_cmd=True,
            )

        for _ in range(5):
            self.command.launch()
        wp.synchronize_device(self.device)

    def time_division(self, divisor_kind):
        self.command.launch()
        wp.synchronize_device(self.device)
