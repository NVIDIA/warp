# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import warp as wp


class HalfFloatConversion:
    """Benchmark half-float conversion via METH_FASTCALL and ctypes paths."""

    repeat = 200
    number = 1  # Each benchmark function loops internally to minimize call overhead.

    def setup(self):
        wp.init()
        self.core = wp._src.context.runtime.core
        # On older revisions without fastcall, core.ctypes doesn't exist.
        # Fall back to core itself so the ctypes benchmarks measure the baseline.
        self.ctypes = self.core.ctypes if hasattr(self.core, "ctypes") else self.core

    def time_float_to_half_bits_fastcall(self):
        fn = self.core.wp_float_to_half_bits
        for _ in range(10_000):
            fn(1.0)

    def time_float_to_half_bits_ctypes(self):
        fn = self.ctypes.wp_float_to_half_bits
        for _ in range(1_000):
            fn(1.0)

    def time_half_bits_to_float_fastcall(self):
        fn = self.core.wp_half_bits_to_float
        for _ in range(10_000):
            fn(0x3C00)

    def time_half_bits_to_float_ctypes(self):
        fn = self.ctypes.wp_half_bits_to_float
        for _ in range(1_000):
            fn(0x3C00)

    def time_round_trip_fastcall(self):
        to_half = self.core.wp_float_to_half_bits
        to_float = self.core.wp_half_bits_to_float
        for _ in range(10_000):
            to_float(to_half(1.0))

    def time_round_trip_ctypes(self):
        to_half = self.ctypes.wp_float_to_half_bits
        to_float = self.ctypes.wp_half_bits_to_float
        for _ in range(1_000):
            to_float(to_half(1.0))
