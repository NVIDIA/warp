# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import warp as wp

N = 8192


class ArrayEmpty:
    """Benchmark wp.empty()."""

    repeat = 1000  # Number of samples to run
    number = 1  # Number of timed calls per sample

    def setup(self):
        wp.init()
        self.alloc = None
        gc.disable()

    def teardown(self):
        gc.enable()
        self.alloc = None
        wp.synchronize_device("cuda:0")

    def time_empty(self):
        self.alloc = wp.empty(N, dtype=float, device="cuda:0")


class ArrayZeros:
    """Benchmark wp.zeros()."""

    repeat = 1000
    number = 1

    def setup(self):
        wp.init()
        self.alloc = None
        gc.disable()

    def teardown(self):
        gc.enable()
        self.alloc = None
        wp.synchronize_device("cuda:0")

    def time_zeros(self):
        self.alloc = wp.zeros(N, dtype=float, device="cuda:0")


class ArrayFree:
    """Benchmark array free including GPU work."""

    repeat = 100
    number = 1

    def setup(self):
        wp.init()
        self.test_array = wp.empty(N, dtype=float, device="cuda:0")
        self.allocs = [None] * 10
        for i in range(len(self.allocs)):
            self.allocs[i] = wp.empty(N, dtype=float, device="cuda:0")
        wp.synchronize_device("cuda:0")

    def time_ten_free_sync(self):
        for i in range(len(self.allocs)):
            self.allocs[i] = None
        wp.synchronize_device("cuda:0")
