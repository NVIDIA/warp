# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
