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

import enum
import unittest

import warp as wp
from warp.tests.unittest_utils import *


class MyIntEnum(enum.IntEnum):
    A = 1
    B = 2
    C = 3


class MyIntFlag(enum.IntFlag):
    A = 1
    B = enum.auto()
    C = enum.auto()


def test_intenum_ints(test, device):
    @wp.kernel
    def expect_intenum_ints():
        wp.expect_eq(MyIntEnum.A, 1)
        wp.expect_eq(MyIntEnum.B, 2)
        wp.expect_eq(MyIntEnum.C, 3)
        wp.expect_eq(MyIntEnum.A + MyIntEnum.B, MyIntEnum.C)

    wp.launch(expect_intenum_ints, dim=1, device=device)


def test_intflag_ints(test, device):
    @wp.kernel
    def expect_intflag_ints():
        wp.expect_eq(MyIntFlag.A, 1)
        wp.expect_eq(MyIntFlag.B, 2)
        wp.expect_eq(MyIntFlag.C, 4)
        wp.expect_eq(MyIntFlag.A | MyIntFlag.B, 3)
        wp.expect_eq(MyIntFlag.A | MyIntFlag.B | MyIntFlag.C, 7)

    wp.launch(expect_intflag_ints, dim=1, device=device)


def test_alternative_accessors(test, device):
    @wp.kernel
    def expect_alternative_accessors():
        wp.expect_eq(int(MyIntEnum.A), 1)
        wp.expect_eq(int(MyIntEnum.B.value), 2)
        wp.expect_eq(MyIntEnum.C.value, 3)
        wp.expect_eq(MyIntEnum.A + int(MyIntEnum.B) + 0, MyIntEnum.C)
        wp.expect_eq(int(MyIntFlag.A), 1)
        wp.expect_eq(int(MyIntFlag.B.value), 2)
        wp.expect_eq(MyIntFlag.C.value, 4)
        wp.expect_eq(MyIntFlag.A | int(MyIntFlag.B), 3)
        wp.expect_eq(MyIntFlag.A | MyIntFlag.B.value | MyIntFlag.C, 7)

    wp.launch(expect_alternative_accessors, dim=1, device=device)


def test_static_accessors(test, device):
    @wp.kernel
    def expect_static_accessors():
        wp.expect_eq(wp.static(MyIntEnum.A), 1)
        wp.expect_eq(wp.static(int(MyIntEnum.A)), 1)
        wp.expect_eq(wp.static(MyIntEnum.A.value), 1)
        wp.expect_eq(wp.static(MyIntFlag.A), 1)
        wp.expect_eq(wp.static(int(MyIntFlag.A)), 1)
        wp.expect_eq(wp.static(MyIntFlag.A.value), 1)

    wp.launch(expect_static_accessors, dim=1, device=device)


def test_intflag_compare(test, device):
    @wp.kernel
    def compute_intflag_compare(ins: wp.array(dtype=wp.int32), outs: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        if ins[tid] & MyIntFlag.A:
            outs[tid] += MyIntFlag.A
        if ins[tid] & MyIntFlag.B:
            outs[tid] += MyIntFlag.B
        if ins[tid] & MyIntFlag.C:
            outs[tid] += MyIntFlag.C

    with wp.ScopedDevice(device):
        ins = wp.array(
            [
                0,
                MyIntFlag.A,
                MyIntFlag.B,
                MyIntFlag.C,
                MyIntFlag.A | MyIntFlag.B,
                MyIntFlag.A | MyIntFlag.B | MyIntFlag.C,
            ],
            dtype=wp.int32,
        )
        outs = wp.zeros(len(ins), dtype=wp.int32)
        wp.launch(compute_intflag_compare, dim=len(ins), inputs=[ins], outputs=[outs])
        outs = outs.numpy()
        test.assertEqual(outs[0], 0)
        test.assertEqual(outs[1], 1)
        test.assertEqual(outs[2], 2)
        test.assertEqual(outs[3], 4)
        test.assertEqual(outs[4], 3)
        test.assertEqual(outs[5], 7)


class TestEnum(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestEnum, "test_intenum_ints", test_intenum_ints, devices=devices)
add_function_test(TestEnum, "test_intflag_ints", test_intflag_ints, devices=devices)
add_function_test(TestEnum, "test_intflag_compare", test_intflag_compare, devices=devices)
add_function_test(TestEnum, "test_alternative_accessors", test_alternative_accessors, devices=devices)
add_function_test(TestEnum, "test_static_accessors", test_static_accessors, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
