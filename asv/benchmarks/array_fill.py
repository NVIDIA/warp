# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for ``fill_()`` on strided views and indexed arrays."""

import numpy as np

import warp as wp

from .benchmarks_utils import setup_once

LAYOUTS = ("strided", "indexed")
NUM_ELEMENTS = (4096, 1 << 20)


def make_fill_target(layout, num_elements, device):
    """Return a non-contiguous view of every other ``float32`` in an array of ``2 * num_elements``."""
    if layout == "strided":
        return wp.zeros((num_elements, 2), dtype=wp.float32, device=device)[:, 1]

    data = wp.zeros(2 * num_elements, dtype=wp.float32, device=device)
    indices = wp.array(np.arange(0, 2 * num_elements, 2, dtype=np.int32), device=device)
    return wp.indexedarray(data, [indices])


class NonContiguousFillCPU:
    """Measure ``fill_()`` of a strided view or an indexed array on CPU."""

    params = (LAYOUTS, NUM_ELEMENTS)
    param_names = ("layout", "num_elements")
    repeat = 15
    number = 50  # Amortize timer overhead for the 3-13 us fills of 4096 elements.

    @setup_once
    def setup(self, layout, num_elements):
        wp.init()
        self.target = make_fill_target(layout, num_elements, wp.get_device("cpu"))

        for _ in range(5):
            self.target.fill_(1.0)

    def time_fill(self, layout, num_elements):
        self.target.fill_(1.0)


class NonContiguousFillCUDA:
    """Measure ``fill_()`` of a strided view or an indexed array on CUDA."""

    params = (LAYOUTS, NUM_ELEMENTS)
    param_names = ("layout", "num_elements")
    repeat = 31
    number = 50  # Amortize launch jitter for the 5-20 us fills.

    @setup_once
    def setup(self, layout, num_elements):
        wp.init()
        self.device = wp.get_device("cuda:0")
        self.target = make_fill_target(layout, num_elements, self.device)

        for _ in range(5):
            self.target.fill_(1.0)
        wp.synchronize_device(self.device)

    def time_fill(self, layout, num_elements):
        self.target.fill_(1.0)
        wp.synchronize_device(self.device)
