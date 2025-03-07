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

# This is what we are actually testing.
from __future__ import annotations

import unittest

import warp as wp
from warp.tests.unittest_utils import *


@wp.struct
class FooData:
    x: float
    y: float


class Foo:
    Data = FooData

    @wp.func
    def compute():
        pass


@wp.kernel
def kernel_1(
    out: wp.array(dtype=float),
):
    tid = wp.tid()


@wp.kernel
def kernel_2(
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    out[tid] = 1.23


def create_kernel_3(foo: Foo):
    def fn(
        data: foo.Data,
        out: wp.array(dtype=float),
    ):
        tid = wp.tid()

        # Referencing a variable in a type hint like `foo.Data` isn't officially
        # accepted by Python but it's still being used in some places (e.g.: `warp.fem`)
        # where it works only because the variable being referenced within the function,
        # which causes it to be promoted to a closure variable. Without that,
        # it wouldn't be possible to resolve `foo` and to evaluate the `foo.Data`
        # string to its corresponding type.
        foo.compute()

        out[tid] = data.x + data.y

    return wp.Kernel(func=fn)


def test_future_annotations(test, device):
    foo = Foo()
    foo_data = FooData()
    foo_data.x = 1.23
    foo_data.y = 2.34

    out = wp.empty(1, dtype=float)

    kernel_3 = create_kernel_3(foo)

    wp.launch(kernel_1, dim=out.shape, outputs=(out,))
    wp.launch(kernel_2, dim=out.shape, outputs=(out,))
    wp.launch(kernel_3, dim=out.shape, inputs=(foo_data,), outputs=(out,))


class TestFutureAnnotations(unittest.TestCase):
    pass


add_function_test(TestFutureAnnotations, "test_future_annotations", test_future_annotations)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
