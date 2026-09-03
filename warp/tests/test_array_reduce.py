# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *
from warp.utils import array_inner, array_sum


def make_test_array_sum(dtype):
    N = 1000

    def test_array_sum(test, device):
        rng = np.random.default_rng(123)

        cols = wp.types.type_size(dtype)

        values_np = rng.random(size=(N, cols))
        values = wp.array(values_np, device=device, dtype=dtype)

        vsum = array_sum(values)
        ref_vsum = values_np.sum(axis=0)

        assert_np_equal(vsum / N, ref_vsum / N, 0.0001)

    return test_array_sum


def make_test_array_sum_axis(dtype):
    I = 5
    J = 10
    K = 2

    N = I * J * K

    def test_array_sum(test, device):
        rng = np.random.default_rng(123)

        values_np = rng.random(size=(I, J, K))
        values = wp.array(values_np, shape=(I, J, K), device=device, dtype=dtype)

        for axis in range(3):
            vsum = array_sum(values, axis=axis)
            ref_vsum = values_np.sum(axis=axis)

            assert_np_equal(vsum.numpy() / N, ref_vsum / N, 0.0001)

    return test_array_sum


def test_array_sum_empty(test, device):
    values = wp.array([], device=device, dtype=wp.vec2)
    assert_np_equal(array_sum(values), np.zeros(2))

    values = wp.array([], shape=(0, 3), device=device, dtype=float)
    assert_np_equal(array_sum(values, axis=0).numpy(), np.zeros((1, 3)))


def make_test_array_inner(dtype):
    N = 1000

    def test_array_inner(test, device):
        rng = np.random.default_rng(123)

        cols = wp.types.type_size(dtype)

        a_np = rng.random(size=(N, cols))
        b_np = rng.random(size=(N, cols))

        a = wp.array(a_np, device=device, dtype=dtype)
        b = wp.array(b_np, device=device, dtype=dtype)

        ab = array_inner(a, b)
        ref_ab = np.dot(a_np.flatten(), b_np.flatten())

        test.assertAlmostEqual(ab / N, ref_ab / N, places=5)

    return test_array_inner


def make_test_array_inner_axis(dtype):
    I = 5
    J = 10
    K = 2

    N = I * J * K

    def test_array_inner(test, device):
        rng = np.random.default_rng(123)

        a_np = rng.random(size=(I, J, K))
        b_np = rng.random(size=(I, J, K))

        a = wp.array(a_np, shape=(I, J, K), device=device, dtype=dtype)
        b = wp.array(b_np, shape=(I, J, K), device=device, dtype=dtype)

        ab = array_inner(a, b, axis=0)
        ref_ab = np.einsum(a_np, [0, 1, 2], b_np, [0, 1, 2], [1, 2])
        assert_np_equal(ab.numpy() / N, ref_ab / N, 0.0001)

        ab = array_inner(a, b, axis=1)
        ref_ab = np.einsum(a_np, [0, 1, 2], b_np, [0, 1, 2], [0, 2])
        assert_np_equal(ab.numpy() / N, ref_ab / N, 0.0001)

        ab = array_inner(a, b, axis=2)
        ref_ab = np.einsum(a_np, [0, 1, 2], b_np, [0, 1, 2], [0, 1])
        assert_np_equal(ab.numpy() / N, ref_ab / N, 0.0001)

    return test_array_inner


def test_array_inner_empty(test, device):
    values = wp.array([], device=device, dtype=wp.vec2)
    test.assertEqual(array_inner(values, values), 0.0)

    values = wp.array([], shape=(0, 3), device=device, dtype=float)
    assert_np_equal(array_inner(values, values, axis=0).numpy(), np.zeros((1, 3)))


def test_array_reductions_negative_axis(test, device):
    """Verify array reductions accept negative axes across count modes."""
    values_np = np.arange(1.0, 25.0, dtype=np.float32).reshape(2, 3, 4)
    other_np = np.arange(25.0, 49.0, dtype=np.float32).reshape(2, 3, 4)
    values = wp.array(values_np, device=device)
    other = wp.array(other_np, device=device)

    for axis in range(-values.ndim, 0):
        with test.subTest(axis=axis, count="implicit"):
            expected_sum = values_np.sum(axis=axis, keepdims=True)
            expected_inner = (values_np * other_np).sum(axis=axis, keepdims=True)

            np.testing.assert_allclose(array_sum(values, axis=axis).numpy(), expected_sum)
            np.testing.assert_allclose(array_inner(values, other, axis=axis).numpy(), expected_inner)

        count = values.shape[axis] - 1
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(count)
        expected_sum = values_np[tuple(slices)].sum(axis=axis, keepdims=True)
        expected_inner = (values_np[tuple(slices)] * other_np[tuple(slices)]).sum(axis=axis, keepdims=True)
        sum_out = wp.full(expected_sum.shape, -1.0, dtype=wp.float32, device=device)
        inner_out = wp.full(expected_inner.shape, -1.0, dtype=wp.float32, device=device)

        with test.subTest(axis=axis, count=count):
            test.assertIs(array_sum(values, out=sum_out, value_count=count, axis=axis), sum_out)
            test.assertIs(array_inner(values, other, out=inner_out, count=count, axis=axis), inner_out)
            np.testing.assert_allclose(sum_out.numpy(), expected_sum)
            np.testing.assert_allclose(inner_out.numpy(), expected_inner)


def test_array_reductions_negative_axis_zero_count(test, device):
    """Verify array reductions return zeros for empty negative-axis reductions."""
    values = wp.ones((2, 3), dtype=wp.float32, device=device)
    expected = np.zeros((2, 1), dtype=np.float32)

    np.testing.assert_allclose(array_sum(values, value_count=0, axis=-1).numpy(), expected)
    np.testing.assert_allclose(array_inner(values, values, count=0, axis=-1).numpy(), expected)

    empty = wp.empty((2, 0, 3), dtype=wp.float32, device=device)
    expected_empty = np.zeros((2, 1, 3), dtype=np.float32)

    np.testing.assert_allclose(array_sum(empty, axis=-2).numpy(), expected_empty)
    np.testing.assert_allclose(array_inner(empty, empty, axis=-2).numpy(), expected_empty)


def test_array_reductions_invalid_axis(test, device):
    """Verify array reductions reject axes outside the valid range."""
    values = wp.ones((2, 3), dtype=wp.float32, device=device)

    for axis in (-values.ndim - 1, values.ndim):
        for count in (None, 0):
            sum_kwargs = {"axis": axis}
            inner_kwargs = {"axis": axis}
            if count is not None:
                sum_kwargs["value_count"] = count
                inner_kwargs["count"] = count

            with test.subTest(operation="array_sum", axis=axis, count=count):
                with test.assertRaisesRegex(IndexError, rf"array_sum\(\) axis {axis} is out of bounds"):
                    array_sum(values, **sum_kwargs)

            with test.subTest(operation="array_inner", axis=axis, count=count):
                with test.assertRaisesRegex(IndexError, rf"array_inner\(\) axis {axis} is out of bounds"):
                    array_inner(values, values, **inner_kwargs)


devices = get_test_devices()


class TestArrayReduce(unittest.TestCase):
    pass


add_function_test(TestArrayReduce, "test_array_sum_double", make_test_array_sum(wp.float64), devices=devices)
add_function_test(TestArrayReduce, "test_array_sum_vec3", make_test_array_sum(wp.vec3), devices=devices)
add_function_test(TestArrayReduce, "test_array_sum_axis_float", make_test_array_sum_axis(wp.float32), devices=devices)
add_function_test(TestArrayReduce, "test_array_sum_empty", test_array_sum_empty, devices=devices)
add_function_test(TestArrayReduce, "test_array_inner_double", make_test_array_inner(wp.float64), devices=devices)
add_function_test(TestArrayReduce, "test_array_inner_vec3", make_test_array_inner(wp.vec3), devices=devices)
add_function_test(
    TestArrayReduce, "test_array_inner_axis_float", make_test_array_inner_axis(wp.float32), devices=devices
)
add_function_test(TestArrayReduce, "test_array_inner_empty", test_array_inner_empty, devices=devices)
add_function_test(
    TestArrayReduce, "test_array_reductions_negative_axis", test_array_reductions_negative_axis, devices=devices
)
add_function_test(
    TestArrayReduce,
    "test_array_reductions_negative_axis_zero_count",
    test_array_reductions_negative_axis_zero_count,
    devices=devices,
)
add_function_test(
    TestArrayReduce, "test_array_reductions_invalid_axis", test_array_reductions_invalid_axis, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
