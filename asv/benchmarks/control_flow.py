# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA benchmarks for divergent control flow in dynamic loops."""

import numpy as np

import warp as wp

from .benchmarks_utils import setup_once

TABLE_SIZE = 1024
NUM_THREADS = 1 << 15
NUM_STEPS = 128
LAUNCHES_PER_GRAPH = 32


@wp.kernel(enable_backward=False)
def divergent_loop(table: wp.array[wp.int32], steps: int, output: wp.array[wp.float32]):
    """Walk a small shared table with divergent, state-dependent updates."""
    i = wp.tid()
    index = i & (TABLE_SIZE - 1)
    value = float(0.0)
    active = bool(True)
    for _step in range(steps):
        if active:
            index = table[index & (TABLE_SIZE - 1)]
            value = value + float(index & 255)
            active = (index & 1) == 0
        else:
            index = table[(index + i) & (TABLE_SIZE - 1)]
            value = value * 0.125
            active = (index & 1) != 0
    output[i] = value


class DivergentLoopCUDA:
    """Measure branch-dependent table walks without geometry or large memory traffic."""

    # A 32-launch graph amortizes host jitter. Eight samples met a 2.5% budget
    # for 99% confidence interval widths across independent calibration runs.
    number = 1
    repeat = 8

    @setup_once
    def setup(self):
        """Compile, validate both update paths, and capture reusable launches."""
        self.device = wp.get_device("cuda:0")
        table = np.random.default_rng(42).integers(0, TABLE_SIZE, TABLE_SIZE, dtype=np.int32)
        self.table = wp.array(table, dtype=wp.int32, device=self.device)
        self.output = wp.empty(NUM_THREADS, dtype=wp.float32, device=self.device)
        self.command = wp.launch(
            divergent_loop,
            dim=NUM_THREADS,
            inputs=[self.table, NUM_STEPS],  # Pass the trip count as a runtime argument.
            outputs=[self.output],
            device=self.device,
            record_cmd=True,
        )

        # All paths depend only on the low index bits, so one table period
        # suffices for an independent NumPy reference for every output thread.
        threads = np.arange(TABLE_SIZE, dtype=np.int32)
        index = threads.copy()
        value = np.zeros(TABLE_SIZE, dtype=np.float32)
        active = np.ones(TABLE_SIZE, dtype=bool)
        for _ in range(NUM_STEPS):
            index = np.where(active, table[index], table[(index + threads) & (TABLE_SIZE - 1)])
            value = np.where(active, value + (index & 255).astype(np.float32), value * np.float32(0.125))
            active = np.where(active, (index & 1) == 0, (index & 1) != 0)
        for _ in range(3):
            self.command.launch()
        np.testing.assert_array_equal(self.output.numpy(), np.tile(value, NUM_THREADS // TABLE_SIZE))

        with wp.ScopedCapture(device=self.device) as capture:
            for _ in range(LAUNCHES_PER_GRAPH):
                self.command.launch()
        self.graph = capture.graph
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)

    def time_loop(self):
        """Time a batch of completed loop kernels while amortizing host launch jitter."""
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
