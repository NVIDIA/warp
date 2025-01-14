# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import gc

import warp as wp

N = 8192


class ArrayAlloc:
    repeat = 1000
    number = 1
    rounds = 4

    def setup(self):
        wp.init()
        self.allocs = [None] * 10
        gc.disable()

    def teardown(self):
        gc.enable()
        self.allocs = [None] * 10
        wp.synchronize_device("cuda:0")

    def time_ten_empty(self):
        for i in range(len(self.allocs)):
            self.allocs[i] = wp.empty(N, dtype=float, device="cuda:0")

    def time_ten_empty_sync(self):
        for i in range(len(self.allocs)):
            self.allocs[i] = wp.empty(N, dtype=float, device="cuda:0")
        wp.synchronize_device("cuda:0")


class ArrayFree:
    repeat = 1000
    number = 1
    rounds = 4

    def setup(self):
        wp.init()
        self.test_array = wp.empty(N, dtype=float, device="cuda:0")
        self.allocs = [None] * 10
        for i in range(len(self.allocs)):
            self.allocs[i] = wp.empty(N, dtype=float, device="cuda:0")
        wp.synchronize_device("cuda:0")
        gc.disable()

    def teardown(self):
        gc.enable()
        wp.synchronize_device("cuda:0")

    def time_ten_free(self):
        for i in range(len(self.allocs)):
            self.allocs[i] = None

    def time_ten_free_sync(self):
        for i in range(len(self.allocs)):
            self.allocs[i] = None
        wp.synchronize_device("cuda:0")


class ArrayZeros:
    repeat = 1000
    number = 1
    rounds = 4

    def setup(self):
        wp.init()
        self.allocs = [None] * 10
        gc.disable()

    def teardown(self):
        gc.enable()
        self.allocs = [None] * 10
        wp.synchronize_device("cuda:0")

    def time_ten_zeros(self):
        for i in range(len(self.allocs)):
            self.allocs[i] = wp.zeros(N, dtype=float, device="cuda:0")

    def time_ten_zeros_sync(self):
        for i in range(len(self.allocs)):
            self.allocs[i] = wp.zeros(N, dtype=float, device="cuda:0")
        wp.synchronize_device("cuda:0")
