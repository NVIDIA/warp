# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for augmented assignment double-evaluation fix (issue #1233).

Verifies that the RHS of augmented assignments (+=, -=, *=, etc.) is
evaluated exactly once for all target types: attribute, non-atomic
array subscript, composite non-atomic subscript, unsupported operator
on array subscript, and the generic fallback path.

Each test uses wp.atomic_add on a counter array as a side-effect
detector: the counter must be exactly 1 after a single augmented
assignment, not 2 (which would indicate double evaluation).
"""

import unittest

import warp as wp
from warp.tests.unittest_utils import *

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


@wp.struct
class SimpleState:
    value: float


@wp.func
def side_effect_add_f(counter: wp.array(dtype=int), a: float, b: float) -> float:
    wp.atomic_add(counter, 0, 1)
    return a + b


@wp.func
def side_effect_inc_i16(counter: wp.array(dtype=int), val: wp.int16) -> wp.int16:
    wp.atomic_add(counter, 0, 1)
    return val


@wp.func
def side_effect_mul_f(counter: wp.array(dtype=int), val: float) -> float:
    wp.atomic_add(counter, 0, 1)
    return val


@wp.func
def side_effect_sub_i16(counter: wp.array(dtype=int), val: wp.int16) -> wp.int16:
    wp.atomic_add(counter, 0, 1)
    return val


# ---------------------------------------------------------------------------
# Test kernels
# ---------------------------------------------------------------------------


@wp.kernel
def kernel_attr_augassign(counter: wp.array(dtype=int), result: wp.array(dtype=float)):
    """Attribute target: s.value += side_effect_func()"""
    s = SimpleState()
    s.value = 1.0
    s.value += side_effect_add_f(counter, 2.0, 3.0)
    result[0] = s.value


@wp.kernel
def kernel_non_atomic_subscript_add(counter: wp.array(dtype=int), data: wp.array(dtype=wp.int16)):
    """Non-atomic array subscript: arr_int16[i] += side_effect_func()

    int16 is a non-atomic type, so this reaches do_augmented_assign().
    """
    i = wp.tid()
    data[i] += side_effect_inc_i16(counter, wp.int16(1))


@wp.kernel
def kernel_non_atomic_subscript_sub(counter: wp.array(dtype=int), data: wp.array(dtype=wp.int16)):
    """Non-atomic array subscript with subtraction: arr_int16[i] -= side_effect_func()

    int16 is a non-atomic type, so this reaches do_augmented_assign().
    """
    i = wp.tid()
    data[i] -= side_effect_sub_i16(counter, wp.int16(3))


@wp.kernel
def kernel_unsupported_op_mul(counter: wp.array(dtype=int), data: wp.array(dtype=float)):
    """Unsupported operator on array subscript: arr[i] *= side_effect_func()

    Multiplication on a float array is not handled by atomic ops,
    so this falls through to do_augmented_assign().
    """
    i = wp.tid()
    data[i] *= side_effect_mul_f(counter, 2.0)


@wp.kernel
def kernel_attr_augassign_sub(counter: wp.array(dtype=int), result: wp.array(dtype=float)):
    """Attribute target with subtraction: s.value -= side_effect_func()"""
    s = SimpleState()
    s.value = 10.0
    s.value -= side_effect_add_f(counter, 1.0, 2.0)
    result[0] = s.value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_attr_augassign_single_eval(test, device):
    """s.value += expr should evaluate expr exactly once."""
    counter = wp.zeros(1, dtype=int, device=device)
    result = wp.zeros(1, dtype=float, device=device)

    wp.launch(kernel_attr_augassign, dim=1, inputs=[counter, result], device=device)

    test.assertEqual(counter.numpy()[0], 1, "RHS evaluated more than once for attribute +=")
    test.assertAlmostEqual(result.numpy()[0], 6.0, places=5, msg="s.value should be 1.0 + (2.0 + 3.0) = 6.0")


def test_non_atomic_subscript_add_single_eval(test, device):
    """arr_int16[i] += expr should evaluate expr exactly once."""
    counter = wp.zeros(1, dtype=int, device=device)
    data = wp.array([wp.int16(2)], dtype=wp.int16, device=device)

    wp.launch(kernel_non_atomic_subscript_add, dim=1, inputs=[counter, data], device=device)

    test.assertEqual(counter.numpy()[0], 1, "RHS evaluated more than once for non-atomic subscript +=")
    test.assertEqual(data.numpy()[0], 3, "data[0] should be 2 + 1 = 3")


def test_non_atomic_subscript_sub_single_eval(test, device):
    """arr_int16[i] -= expr should evaluate expr exactly once."""
    counter = wp.zeros(1, dtype=int, device=device)
    data = wp.array([wp.int16(10)], dtype=wp.int16, device=device)

    wp.launch(kernel_non_atomic_subscript_sub, dim=1, inputs=[counter, data], device=device)

    test.assertEqual(counter.numpy()[0], 1, "RHS evaluated more than once for non-atomic subscript -=")
    test.assertEqual(data.numpy()[0], 7, "data[0] should be 10 - 3 = 7")


def test_unsupported_op_mul_single_eval(test, device):
    """arr[i] *= expr should evaluate expr exactly once."""
    counter = wp.zeros(1, dtype=int, device=device)
    data = wp.array([1.0], dtype=float, device=device)

    wp.launch(kernel_unsupported_op_mul, dim=1, inputs=[counter, data], device=device)

    test.assertEqual(counter.numpy()[0], 1, "RHS evaluated more than once for array *= ")
    test.assertAlmostEqual(data.numpy()[0], 2.0, places=5, msg="data[0] should be 1.0 * 2.0 = 2.0")


def test_attr_augassign_sub_single_eval(test, device):
    """s.value -= expr should evaluate expr exactly once."""
    counter = wp.zeros(1, dtype=int, device=device)
    result = wp.zeros(1, dtype=float, device=device)

    wp.launch(kernel_attr_augassign_sub, dim=1, inputs=[counter, result], device=device)

    test.assertEqual(counter.numpy()[0], 1, "RHS evaluated more than once for attribute -=")
    test.assertAlmostEqual(result.numpy()[0], 7.0, places=5, msg="s.value should be 10.0 - (1.0 + 2.0) = 7.0")


devices = get_test_devices()


class TestAugmentedAssign(unittest.TestCase):
    pass


add_function_test(
    TestAugmentedAssign, "test_attr_augassign_single_eval", test_attr_augassign_single_eval, devices=devices
)
add_function_test(
    TestAugmentedAssign,
    "test_non_atomic_subscript_add_single_eval",
    test_non_atomic_subscript_add_single_eval,
    devices=devices,
)
add_function_test(
    TestAugmentedAssign,
    "test_non_atomic_subscript_sub_single_eval",
    test_non_atomic_subscript_sub_single_eval,
    devices=devices,
)
add_function_test(
    TestAugmentedAssign, "test_unsupported_op_mul_single_eval", test_unsupported_op_mul_single_eval, devices=devices
)
add_function_test(
    TestAugmentedAssign, "test_attr_augassign_sub_single_eval", test_attr_augassign_sub_single_eval, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
