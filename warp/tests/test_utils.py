# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import inspect
import io
import unittest

from warp.tests.unittest_utils import *


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


def test_array_scan_empty(test, device):
    values = wp.array((), dtype=int, device=device)
    result = wp.array((), dtype=int, device=device)
    wp.utils.array_scan(values, result)


def test_array_scan_error_sizes_mismatch(test, device):
    values = wp.zeros(123, dtype=int, device=device)
    result = wp.zeros(234, dtype=int, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array storage sizes do not match$",
    ):
        wp.utils.array_scan(values, result, True)


def test_array_scan_error_dtypes_mismatch(test, device):
    values = wp.zeros(123, dtype=int, device=device)
    result = wp.zeros(123, dtype=float, device=device)
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
    keyTypes = [int, wp.float32, wp.int64]

    for keyType in keyTypes:
        keys = wp.array((7, 2, 8, 4, 1, 6, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0), dtype=keyType, device=device)
        values = wp.array((1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0), dtype=int, device=device)
        wp.utils.radix_sort_pairs(keys, values, 8)
        assert_np_equal(keys.numpy()[:8], np.array((1, 2, 3, 4, 5, 6, 7, 8)))
        assert_np_equal(values.numpy()[:8], np.array((5, 2, 8, 4, 7, 6, 1, 3)))


def test_segmented_sort_pairs(test, device):
    keyTypes = [int, wp.float32]

    for keyType in keyTypes:
        keys = wp.array((7, 2, 8, 4, 1, 6, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0), dtype=keyType, device=device)
        values = wp.array((1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0), dtype=int, device=device)
        wp.utils.segmented_sort_pairs(
            keys,
            values,
            8,
            wp.array((0, 4), dtype=int, device=device),
            wp.array((4, 8), dtype=int, device=device),
        )
        assert_np_equal(keys.numpy()[:8], np.array((2, 4, 7, 8, 1, 3, 5, 6)))
        assert_np_equal(values.numpy()[:8], np.array((2, 4, 1, 3, 5, 8, 7, 6)))


def test_radix_sort_pairs_empty(test, device):
    keyTypes = [int, wp.float32, wp.int64]

    for keyType in keyTypes:
        keys = wp.array((), dtype=keyType, device=device)
        values = wp.array((), dtype=int, device=device)
        wp.utils.radix_sort_pairs(keys, values, 0)


def test_segmented_sort_pairs_empty(test, device):
    keyTypes = [int, wp.float32]

    for keyType in keyTypes:
        keys = wp.array((), dtype=keyType, device=device)
        values = wp.array((), dtype=int, device=device)
        wp.utils.segmented_sort_pairs(
            keys, values, 0, wp.array((), dtype=int, device=device), wp.array((), dtype=int, device=device)
        )


def test_radix_sort_pairs_error_insufficient_storage(test, device):
    keyTypes = [int, wp.float32, wp.int64]

    for keyType in keyTypes:
        keys = wp.array((1, 2, 3), dtype=keyType, device=device)
        values = wp.array((1, 2, 3), dtype=int, device=device)
        with test.assertRaisesRegex(
            RuntimeError,
            r"Array storage must be large enough to contain 2\*count elements$",
        ):
            wp.utils.radix_sort_pairs(keys, values, 3)


def test_segmented_sort_pairs_error_insufficient_storage(test, device):
    keyTypes = [int, wp.float32]

    for keyType in keyTypes:
        keys = wp.array((1, 2, 3), dtype=keyType, device=device)
        values = wp.array((1, 2, 3), dtype=int, device=device)
        with test.assertRaisesRegex(
            RuntimeError,
            r"Array storage must be large enough to contain 2\*count elements$",
        ):
            wp.utils.segmented_sort_pairs(
                keys,
                values,
                3,
                wp.array((0,), dtype=int, device=device),
                wp.array((3,), dtype=int, device=device),
            )


def test_radix_sort_pairs_error_unsupported_dtype(test, device):
    keyTypes = [int, wp.float32, wp.int64]

    for keyType in keyTypes:
        keys = wp.array((1.0, 2.0, 3.0), dtype=keyType, device=device)
        values = wp.array((1.0, 2.0, 3.0), dtype=float, device=device)
        with test.assertRaisesRegex(
            RuntimeError,
            r"Unsupported data type$",
        ):
            wp.utils.radix_sort_pairs(keys, values, 1)


def test_segmented_sort_pairs_error_unsupported_dtype(test, device):
    keyTypes = [int, wp.float32]

    for keyType in keyTypes:
        keys = wp.array((1.0, 2.0, 3.0), dtype=keyType, device=device)
        values = wp.array((1.0, 2.0, 3.0), dtype=float, device=device)
        with test.assertRaisesRegex(
            RuntimeError,
            r"Unsupported data type$",
        ):
            wp.utils.segmented_sort_pairs(
                keys,
                values,
                1,
                wp.array((0,), dtype=int, device=device),
                wp.array((3,), dtype=int, device=device),
            )


def test_array_sum(test, device):
    for dtype in (wp.float32, wp.float64):
        with test.subTest(dtype=dtype):
            values = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
            test.assertEqual(wp.utils.array_sum(values), 6.0)

            values = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
            result = wp.empty(shape=(1,), dtype=dtype, device=device)
            wp.utils.array_sum(values, out=result)
            test.assertEqual(result.numpy()[0], 6.0)


def test_array_sum_error_out_dtype_mismatch(test, device):
    values = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    result = wp.empty(shape=(1,), dtype=wp.float64, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have type float32$",
    ):
        wp.utils.array_sum(values, out=result)


def test_array_sum_error_out_shape_mismatch(test, device):
    values = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    result = wp.empty(shape=(2,), dtype=wp.float32, device=device)
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
    a = wp.array((1.0, 2.0), dtype=wp.float32, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array storage sizes do not match$",
    ):
        wp.utils.array_inner(a, b)


def test_array_inner_error_dtypes_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float64, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array data types do not match$",
    ):
        wp.utils.array_inner(a, b)


def test_array_inner_error_out_dtype_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    result = wp.empty(shape=(1,), dtype=wp.float64, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have type float32$",
    ):
        wp.utils.array_inner(a, b, result)


def test_array_inner_error_out_shape_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    result = wp.empty(shape=(2,), dtype=wp.float32, device=device)
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

    values = wp.array(((1, 2), (3, 4)), dtype=int, device=device)
    result = wp.zeros((2, 2), dtype=int, device=device)
    wp.utils.array_cast(values, result)
    test.assertEqual(result.dtype, wp.int32)
    test.assertEqual(result.shape, (2, 2))
    assert_np_equal(result.numpy(), np.array(((1, 2), (3, 4)), dtype=int))


def test_array_cast_error_unsupported_partial_cast(test, device):
    values = wp.array(((1, 2), (3, 4)), dtype=int, device=device)
    result = wp.zeros((2, 2), dtype=float, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Partial cast is not supported for arrays with more than one dimension$",
    ):
        wp.utils.array_cast(values, result, count=1)


devices = get_test_devices()


class TestUtils(unittest.TestCase):
    def test_warn(self):
        # Multiple warnings get printed out each time.
        with contextlib.redirect_stdout(io.StringIO()) as f:
            wp.utils.warn("hello, world!")
            wp.utils.warn("hello, world!")

        expected = "Warp UserWarning: hello, world!\nWarp UserWarning: hello, world!\n"

        self.assertEqual(f.getvalue(), expected)

        # Test verbose warnings
        saved_verbosity = wp.config.verbose_warnings
        try:
            wp.config.verbose_warnings = True
            with contextlib.redirect_stdout(io.StringIO()) as f:
                frame_info = inspect.getframeinfo(inspect.currentframe())
                wp.utils.warn("hello, world!")
                wp.utils.warn("hello, world!")

            expected = (
                f"Warp UserWarning: hello, world! ({frame_info.filename}:{frame_info.lineno + 1})\n"
                '  wp.utils.warn("hello, world!")\n'
                f"Warp UserWarning: hello, world! ({frame_info.filename}:{frame_info.lineno + 2})\n"
                '  wp.utils.warn("hello, world!")\n'
            )

            self.assertEqual(f.getvalue(), expected)

        finally:
            # make sure to restore warning verbosity
            wp.config.verbose_warnings = saved_verbosity

        # Multiple similar deprecation warnings get printed out only once.
        with contextlib.redirect_stdout(io.StringIO()) as f:
            wp.utils.warn("hello, world!", category=DeprecationWarning)
            wp.utils.warn("hello, world!", category=DeprecationWarning)

        expected = "Warp DeprecationWarning: hello, world!\n"

        self.assertEqual(f.getvalue(), expected)

        # Multiple different deprecation warnings get printed out each time.
        with contextlib.redirect_stdout(io.StringIO()) as f:
            wp.utils.warn("foo", category=DeprecationWarning)
            wp.utils.warn("bar", category=DeprecationWarning)

        expected = "Warp DeprecationWarning: foo\nWarp DeprecationWarning: bar\n"

        self.assertEqual(f.getvalue(), expected)

    def test_transform_expand(self):
        t = (1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0)
        self.assertEqual(
            wp.utils.transform_expand(t),
            wp.transformf(p=(1.0, 2.0, 3.0), q=(4.0, 3.0, 2.0, 1.0)),
        )

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_scan_error_devices_mismatch(self):
        values = wp.zeros(123, dtype=int, device="cpu")
        result = wp.zeros_like(values, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"Array storage devices do not match$",
        ):
            wp.utils.array_scan(values, result, True)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_radix_sort_pairs_error_devices_mismatch(self):
        keys = wp.array((1, 2, 3), dtype=int, device="cpu")
        values = wp.array((1, 2, 3), dtype=int, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"Array storage devices do not match$",
        ):
            wp.utils.radix_sort_pairs(keys, values, 1)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_inner_error_out_device_mismatch(self):
        a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
        b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
        result = wp.empty(shape=(1,), dtype=wp.float32, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"out storage device should match values array$",
        ):
            wp.utils.array_inner(a, b, result)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_sum_error_out_device_mismatch(self):
        values = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
        result = wp.empty(shape=(1,), dtype=wp.float32, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"out storage device should match values array$",
        ):
            wp.utils.array_sum(values, out=result)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_inner_error_devices_mismatch(self):
        a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
        b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"Array storage devices do not match$",
        ):
            wp.utils.array_inner(a, b)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_cast_error_devices_mismatch(self):
        values = wp.array((1, 2, 3), dtype=int, device="cpu")
        result = wp.empty(3, dtype=float, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"Array storage devices do not match$",
        ):
            wp.utils.array_cast(values, result)

    def test_mesh_adjacency(self):
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
        self.assertDictEqual(edges, expected_edges)

    def test_mesh_adjacency_error_manifold(self):
        triangles = (
            (0, 3, 1),
            (0, 2, 3),
            (3, 0, 1),
        )

        with contextlib.redirect_stdout(io.StringIO()) as f:
            wp.utils.MeshAdjacency(triangles, len(triangles))

        self.assertEqual(f.getvalue(), "Detected non-manifold edge\n")

    def test_scoped_timer(self):
        with contextlib.redirect_stdout(io.StringIO()) as f:
            with wp.ScopedTimer("hello"):
                pass

        self.assertRegex(f.getvalue(), r"^hello took \d+\.\d+ ms$")

        with contextlib.redirect_stdout(io.StringIO()) as f:
            with wp.ScopedTimer("hello", detailed=True):
                pass

        self.assertRegex(f.getvalue(), r"^         4 function calls in \d+\.\d+ seconds")
        self.assertRegex(f.getvalue(), r"hello took \d+\.\d+ ms$")


add_function_test(TestUtils, "test_array_scan", test_array_scan, devices=devices)
add_function_test(TestUtils, "test_array_scan_empty", test_array_scan_empty, devices=devices)
add_function_test(
    TestUtils, "test_array_scan_error_sizes_mismatch", test_array_scan_error_sizes_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_scan_error_dtypes_mismatch", test_array_scan_error_dtypes_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_scan_error_unsupported_dtype", test_array_scan_error_unsupported_dtype, devices=devices
)
add_function_test(TestUtils, "test_radix_sort_pairs", test_radix_sort_pairs, devices=devices)
add_function_test(TestUtils, "test_radix_sort_pairs_empty", test_radix_sort_pairs, devices=devices)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_error_insufficient_storage",
    test_radix_sort_pairs_error_insufficient_storage,
    devices=devices,
)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_error_unsupported_dtype",
    test_radix_sort_pairs_error_unsupported_dtype,
    devices=devices,
)
add_function_test(TestUtils, "test_segmented_sort_pairs", test_segmented_sort_pairs, devices=devices)
add_function_test(TestUtils, "test_segmented_sort_pairs_empty", test_segmented_sort_pairs, devices=devices)
add_function_test(
    TestUtils,
    "test_segmented_sort_pairs_error_insufficient_storage",
    test_segmented_sort_pairs_error_insufficient_storage,
    devices=devices,
)
add_function_test(
    TestUtils,
    "test_segmented_sort_pairs_error_unsupported_dtype",
    test_segmented_sort_pairs_error_unsupported_dtype,
    devices=devices,
)
add_function_test(TestUtils, "test_array_sum", test_array_sum, devices=devices)
add_function_test(
    TestUtils, "test_array_sum_error_out_dtype_mismatch", test_array_sum_error_out_dtype_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_sum_error_out_shape_mismatch", test_array_sum_error_out_shape_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_sum_error_unsupported_dtype", test_array_sum_error_unsupported_dtype, devices=devices
)
add_function_test(TestUtils, "test_array_inner", test_array_inner, devices=devices)
add_function_test(
    TestUtils, "test_array_inner_error_sizes_mismatch", test_array_inner_error_sizes_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_inner_error_dtypes_mismatch", test_array_inner_error_dtypes_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_inner_error_out_dtype_mismatch", test_array_inner_error_out_dtype_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_inner_error_out_shape_mismatch", test_array_inner_error_out_shape_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_inner_error_unsupported_dtype", test_array_inner_error_unsupported_dtype, devices=devices
)
add_function_test(TestUtils, "test_array_cast", test_array_cast, devices=devices)
add_function_test(
    TestUtils,
    "test_array_cast_error_unsupported_partial_cast",
    test_array_cast_error_unsupported_partial_cast,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
