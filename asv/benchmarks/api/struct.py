# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


@wp.struct
class A:
    a: int
    b: wp.vec3
    c: wp.array(dtype=float)


@wp.struct
class B:
    a: A
    b: wp.mat33
    c: wp.array2d(dtype=int)


@wp.struct
class C:
    a: A
    b: B


@wp.struct
class D:
    c: C
    b: B
    a: A
    e: wp.float64


class StructSetup:
    number = 1000  # Number of measurements to make between a single setup and teardown

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()

        self._vec = wp.vec3(1, 2, 3)
        self._mat = wp.mat33(1, 2, 3, 4, 5, 6, 7, 8, 9)
        self._array = wp.zeros(10, dtype=float, device="cpu")
        self._array2d = wp.zeros((10, 10), dtype=int, device="cpu")

        self.a = A()
        self.d = D()

    def time_construct_simple(self):
        """
        Time constructing a simple struct.
        """
        A()

    def time_fill_simple(self):
        """
        Time filling a simple struct.
        """
        self.fill_A(self.a)

    def time_construct_complex(self):
        """
        Time constructing a complex struct.
        """
        D()

    def time_fill_complex(self):
        """
        Time filling a complex struct.
        """
        self.fill_D(self.d)

    def fill_A(self, a: A):
        a.a = 1
        a.b = self._vec
        a.c = self._array

    def fill_B(self, b: B):
        self.fill_A(b.a)
        b.b = self._mat
        b.c = self._array2d

    def fill_C(self, c: C):
        self.fill_A(c.a)
        self.fill_B(c.b)

    def fill_D(self, d: D):
        self.fill_C(d.c)
        self.fill_B(d.b)
        self.fill_A(d.a)
        d.e = 1.0
