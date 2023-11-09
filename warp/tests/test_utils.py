# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import contextlib
import io
import inspect
import unittest

from warp.tests.test_base import *


wp.init()


def test_warn(test, device):
    with contextlib.redirect_stdout(io.StringIO()) as f:
        frame_info = inspect.getframeinfo(inspect.currentframe())
        wp.utils.warn("hello, world!")

    expected = '{}:{}: UserWarning: hello, world!\n  wp.utils.warn("hello, world!")\n'.format(
        frame_info.filename,
        frame_info.lineno + 1,
    )
    test.assertEqual(f.getvalue(), expected)


def test_transform_expand(test, device):
    t = (1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0)
    test.assertEqual(
        wp.utils.transform_expand(t),
        wp.transformf(p=(1.0, 2.0, 3.0), q=(4.0, 3.0, 2.0, 1.0)),
    )


def test_array_scan(test, device):
    rng = np.random.default_rng(123)

    for dtype in (int, float):
        if dtype == int:
            values = rng.integers(-1e6, high=1e6, size=100000, dtype=dtype)
        else:
            values = rng.uniform(low=-1e6, high=1e6, size=100000)

        expected = np.cumsum(values)

        values = wp.array(values, dtype=dtype, device=device)
        result_inc = wp.zeros_like(values)
        result_exc = wp.zeros_like(values)

        wp.utils.array_scan(values, result_inc, True)
        wp.utils.array_scan(values, result_exc, False)

        tolerance = 0 if dtype == int else 1e-3

        result_inc = result_inc.numpy().squeeze()
        result_exc = result_exc.numpy().squeeze()
        error_inc = np.max(np.abs(result_inc - expected)) / abs(expected[-1])
        error_exc = max(np.max(np.abs(result_exc[1:] - expected[:-1])), abs(result_exc[0])) / abs(expected[-2])

        test.assertTrue(error_inc <= tolerance)
        test.assertTrue(error_exc <= tolerance)


def test_array_scan_error_devices_mismatch(test, device):
    values = wp.zeros(123, dtype=int, device="cpu")
    result = wp.zeros_like(values, device="cuda:0")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array storage devices do not match$",
    ):
        wp.utils.array_scan(values, result, True)


def test_array_scan_error_sizes_mismatch(test, device):
    values = wp.zeros(123, dtype=int, device="cpu")
    result = wp.zeros(234, dtype=int, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array storage sizes do not match$",
    ):
        wp.utils.array_scan(values, result, True)


def test_array_scan_error_dtypes_mismatch(test, device):
    values = wp.zeros(123, dtype=int, device="cpu")
    result = wp.zeros(123, dtype=float, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array data types do not match$",
    ):
        wp.utils.array_scan(values, result, True)


def test_array_scan_error_unsupported_dtype(test, device):
    values = wp.zeros(123, dtype=wp.vec3, device=device)
    result = wp.zeros(123, dtype=wp.vec3, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Unsupported data type$",
    ):
        wp.utils.array_scan(values, result, True)


def test_radix_sort_pairs(test, device):
    keys = wp.array((7, 2, 8, 4, 1, 6, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0), dtype=int, device=device)
    values = wp.array((1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0), dtype=int, device=device)
    wp.utils.radix_sort_pairs(keys, values, 8)
    assert_np_equal(keys.numpy()[:8], np.array((1, 2, 3, 4, 5, 6, 7, 8)))
    assert_np_equal(values.numpy()[:8], np.array((5, 2, 8, 4, 7, 6, 1, 3)))


def test_radix_sort_pairs_error_devices_mismatch(test, device):
    keys = wp.array((1, 2, 3), dtype=int, device="cpu")
    values = wp.array((1, 2, 3), dtype=int, device="cuda:0")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array storage devices do not match$",
    ):
        wp.utils.radix_sort_pairs(keys, values, 1)


def test_radix_sort_pairs_error_insufficient_storage(test, device):
    keys = wp.array((1, 2, 3), dtype=int, device="cpu")
    values = wp.array((1, 2, 3), dtype=int, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array storage must be large enough to contain 2\*count elements$",
    ):
        wp.utils.radix_sort_pairs(keys, values, 3)


def test_radix_sort_pairs_error_unsupported_dtype(test, device):
    keys = wp.array((1.0, 2.0, 3.0), dtype=float, device=device)
    values = wp.array((1.0, 2.0, 3.0), dtype=float, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Unsupported data type$",
    ):
        wp.utils.radix_sort_pairs(keys, values, 1)


def test_array_sum(test, device):
    for dtype in (wp.float32, wp.float64):
        values = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        test.assertEqual(wp.utils.array_sum(values), 6.0)

        values = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        result = wp.empty(shape=(1,), dtype=dtype, device=device)
        wp.utils.array_sum(values, out=result)
        test.assertEqual(result.numpy()[0], 6.0)


def test_array_sum_error_out_device_mismatch(test, device):
    values = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
    result = wp.empty(shape=(1,), dtype=wp.float32, device="cuda:0")
    with test.assertRaisesRegex(
        RuntimeError,
        r"out storage device should match values array$",
    ):
        wp.utils.array_sum(values, out=result)


def test_array_sum_error_out_dtype_mismatch(test, device):
    values = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
    result = wp.empty(shape=(1,), dtype=wp.float64, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have type float32$",
    ):
        wp.utils.array_sum(values, out=result)


def test_array_sum_error_out_shape_mismatch(test, device):
    values = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
    result = wp.empty(shape=(2,), dtype=wp.float32, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have shape \(1,\)$",
    ):
        wp.utils.array_sum(values, out=result)


def test_array_sum_error_unsupported_dtype(test, device):
    values = wp.array((1, 2, 3), dtype=int, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Unsupported data type$",
    ):
        wp.utils.array_sum(values)


def test_array_inner(test, device):
    for dtype in (wp.float32, wp.float64):
        a = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        b = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        test.assertEqual(wp.utils.array_inner(a, b), 14.0)

        a = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        b = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        result = wp.empty(shape=(1,), dtype=dtype, device=device)
        wp.utils.array_inner(a, b, out=result)
        test.assertEqual(result.numpy()[0], 14.0)


def test_array_inner_error_sizes_mismatch(test, device):
    a = wp.array((1.0, 2.0), dtype=wp.float32, device="cpu")
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array storage sizes do not match$",
    ):
        wp.utils.array_inner(a, b)


def test_array_inner_error_devices_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cuda:0")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array storage devices do not match$",
    ):
        wp.utils.array_inner(a, b)


def test_array_inner_error_dtypes_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float64, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array data types do not match$",
    ):
        wp.utils.array_inner(a, b)


def test_array_inner_error_out_device_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
    result = wp.empty(shape=(1,), dtype=wp.float32, device="cuda:0")
    with test.assertRaisesRegex(
        RuntimeError,
        r"out storage device should match values array$",
    ):
        wp.utils.array_inner(a, b, result)


def test_array_inner_error_out_dtype_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
    result = wp.empty(shape=(1,), dtype=wp.float64, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have type float32$",
    ):
        wp.utils.array_inner(a, b, result)


def test_array_inner_error_out_shape_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
    result = wp.empty(shape=(2,), dtype=wp.float32, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have shape \(1,\)$",
    ):
        wp.utils.array_inner(a, b, result)


def test_array_inner_error_unsupported_dtype(test, device):
    a = wp.array((1, 2, 3), dtype=int, device=device)
    b = wp.array((1, 2, 3), dtype=int, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Unsupported data type$",
    ):
        wp.utils.array_inner(a, b)


def test_array_cast(test, device):
    values = wp.array((1, 2, 3), dtype=int, device=device)
    result = wp.empty(3, dtype=float, device=device)
    wp.utils.array_cast(values, result)
    test.assertEqual(result.dtype, wp.float32)
    test.assertEqual(result.shape, (3,))
    assert_np_equal(result.numpy(), np.array((1.0, 2.0, 3.0), dtype=float))

    values = wp.array((1, 2, 3, 4), dtype=int, device=device)
    result = wp.empty((2, 2), dtype=float, device=device)
    wp.utils.array_cast(values, result)
    test.assertEqual(result.dtype, wp.float32)
    test.assertEqual(result.shape, (2, 2))
    assert_np_equal(result.numpy(), np.array(((1.0, 2.0), (3.0, 4.0)), dtype=float))

    values = wp.array(((1, 2), (3, 4)), dtype=wp.vec2, device=device)
    result = wp.zeros(2, dtype=float, device=device)
    wp.utils.array_cast(values, result, count=1)
    test.assertEqual(result.dtype, wp.float32)
    test.assertEqual(result.shape, (2,))
    assert_np_equal(result.numpy(), np.array((1.0, 2.0), dtype=float))

    values = wp.array(((1, 2), (3, 4)), dtype=int, device="cpu")
    result = wp.zeros((2, 2), dtype=int, device="cpu")
    wp.utils.array_cast(values, result)
    test.assertEqual(result.dtype, wp.int32)
    test.assertEqual(result.shape, (2, 2))
    assert_np_equal(result.numpy(), np.array(((1, 2), (3, 4)), dtype=int))


def test_array_cast_error_devices_mismatch(test, device):
    values = wp.array((1, 2, 3), dtype=int, device="cpu")
    result = wp.empty(3, dtype=float, device="cuda:0")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array storage devices do not match$",
    ):
        wp.utils.array_cast(values, result)


def test_array_cast_error_unsupported_partial_cast(test, device):
    values = wp.array(((1, 2), (3, 4)), dtype=int, device="cpu")
    result = wp.zeros((2, 2), dtype=float, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Partial cast is not supported for arrays with more than one dimension$",
    ):
        wp.utils.array_cast(values, result, count=1)


def test_mesh_adjacency(test, device):
    triangles = (
        (0, 3, 1),
        (0, 2, 3),
    )
    adj = wp.utils.MeshAdjacency(triangles, len(triangles))
    expected_edges = {
        (0, 3): (0, 3, 1, 2, 0, 1),
        (1, 3): (3, 1, 0, -1, 0, -1),
        (0, 1): (1, 0, 3, -1, 0, -1),
        (0, 2): (0, 2, 3, -1, 1, -1),
        (2, 3): (2, 3, 0, -1, 1, -1),
    }
    edges = {k: (e.v0, e.v1, e.o0, e.o1, e.f0, e.f1) for k, e in adj.edges.items()}
    test.assertDictEqual(edges, expected_edges)


def test_mesh_adjacency_error_manifold(test, device):
    triangles = (
        (0, 3, 1),
        (0, 2, 3),
        (3, 0, 1),
    )

    with contextlib.redirect_stdout(io.StringIO()) as f:
        wp.utils.MeshAdjacency(triangles, len(triangles))

    test.assertEqual(f.getvalue(), "Detected non-manifold edge\n")


def test_scoped_timer(test, device):
    with contextlib.redirect_stdout(io.StringIO()) as f:
        with wp.ScopedTimer("hello"):
            pass

    test.assertRegex(f.getvalue(), r"^hello took \d+\.\d+ ms$")

    with contextlib.redirect_stdout(io.StringIO()) as f:
        with wp.ScopedTimer("hello", detailed=True):
            pass

    test.assertRegex(f.getvalue(), r"^         2 function calls in \d+\.\d+ seconds")
    test.assertRegex(f.getvalue(), r"hello took \d+\.\d+ ms$")


def register(parent):
    devices = get_test_devices()

    class TestUtils(parent):
        pass

    add_function_test(TestUtils, "test_warn", test_warn)
    add_function_test(TestUtils, "test_transform_expand", test_transform_expand)
    add_function_test(TestUtils, "test_array_scan", test_array_scan, devices=devices)
    add_function_test(TestUtils, "test_array_scan_error_devices_mismatch", test_array_scan_error_devices_mismatch)
    add_function_test(TestUtils, "test_array_scan_error_sizes_mismatch", test_array_scan_error_sizes_mismatch)
    add_function_test(TestUtils, "test_array_scan_error_dtypes_mismatch", test_array_scan_error_dtypes_mismatch)
    add_function_test(
        TestUtils, "test_array_scan_error_unsupported_dtype", test_array_scan_error_unsupported_dtype, devices=devices
    )
    add_function_test(TestUtils, "test_radix_sort_pairs", test_radix_sort_pairs, devices=devices)
    add_function_test(
        TestUtils, "test_radix_sort_pairs_error_devices_mismatch", test_radix_sort_pairs_error_devices_mismatch
    )
    add_function_test(
        TestUtils, "test_radix_sort_pairs_error_insufficient_storage", test_radix_sort_pairs_error_insufficient_storage
    )
    add_function_test(
        TestUtils,
        "test_radix_sort_pairs_error_unsupported_dtype",
        test_radix_sort_pairs_error_unsupported_dtype,
        devices=devices,
    )
    add_function_test(TestUtils, "test_array_sum", test_array_sum, devices=devices)
    add_function_test(TestUtils, "test_array_sum_error_out_device_mismatch", test_array_sum_error_out_device_mismatch)
    add_function_test(TestUtils, "test_array_sum_error_out_dtype_mismatch", test_array_sum_error_out_dtype_mismatch)
    add_function_test(TestUtils, "test_array_sum_error_out_shape_mismatch", test_array_sum_error_out_shape_mismatch)
    add_function_test(
        TestUtils, "test_array_sum_error_unsupported_dtype", test_array_sum_error_unsupported_dtype, devices=devices
    )
    add_function_test(TestUtils, "test_array_inner", test_array_inner, devices=devices)
    add_function_test(TestUtils, "test_array_inner_error_sizes_mismatch", test_array_inner_error_sizes_mismatch)
    add_function_test(TestUtils, "test_array_inner_error_devices_mismatch", test_array_inner_error_devices_mismatch)
    add_function_test(TestUtils, "test_array_inner_error_dtypes_mismatch", test_array_inner_error_dtypes_mismatch)
    add_function_test(
        TestUtils, "test_array_inner_error_out_device_mismatch", test_array_inner_error_out_device_mismatch
    )
    add_function_test(TestUtils, "test_array_inner_error_out_dtype_mismatch", test_array_inner_error_out_dtype_mismatch)
    add_function_test(TestUtils, "test_array_inner_error_out_shape_mismatch", test_array_inner_error_out_shape_mismatch)
    add_function_test(
        TestUtils, "test_array_inner_error_unsupported_dtype", test_array_inner_error_unsupported_dtype, devices=devices
    )
    add_function_test(TestUtils, "test_array_cast", test_array_cast, devices=devices)
    add_function_test(TestUtils, "test_array_cast_error_devices_mismatch", test_array_cast_error_devices_mismatch)
    add_function_test(
        TestUtils, "test_array_cast_error_unsupported_partial_cast", test_array_cast_error_unsupported_partial_cast
    )
    add_function_test(TestUtils, "test_mesh_adjacency", test_mesh_adjacency)
    add_function_test(TestUtils, "test_mesh_adjacency_error_manifold", test_mesh_adjacency_error_manifold)
    add_function_test(TestUtils, "test_scoped_timer", test_scoped_timer)
    return TestUtils


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    _ = register(unittest.TestCase)
    unittest.main(verbosity=2)
