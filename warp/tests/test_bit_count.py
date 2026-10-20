# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_test_devices

integer_types = (wp.int8, wp.uint8, wp.int16, wp.uint16, wp.int32, wp.uint32, wp.int64, wp.uint64)


def test_bit_count(test, device, dtype):
    """Verify fixed-width bit counts for signed and unsigned integer kernels."""

    @wp.kernel(module="unique")
    def count_bits(values: wp.array[dtype], counts: wp.array[wp.int32]):
        i = wp.tid()
        counts[i] = wp.bit_count(values[i])

    np_dtype = np.dtype(wp.dtype_to_numpy(dtype))
    unsigned_dtype = np.dtype(f"uint{np_dtype.itemsize * 8}")
    width = np_dtype.itemsize * 8
    # Exhaust all 8- and 16-bit patterns; sample the much larger 32- and 64-bit domains.
    if width <= 16:
        patterns = np.arange(1 << width, dtype=unsigned_dtype)
    else:
        mask = (1 << width) - 1
        sign_bit = 1 << (width - 1)
        # An all-ones mask divided by three gives 0101...; doubling it gives 1010....
        alternating = mask // 3
        patterns = [0, 1, 45, mask, mask - 1, sign_bit, sign_bit - 1, alternating, alternating * 2]
        patterns += [1 << bit for bit in range(width)]
        patterns += [mask ^ (1 << bit) for bit in range(width)]

        rng = np.random.default_rng(2071)
        # Decode raw bytes to sample the full unsigned range, including values above the signed maximum.
        random_patterns = np.frombuffer(rng.bytes(256 * np_dtype.itemsize), dtype=unsigned_dtype)
        patterns = np.concatenate((np.array(patterns, dtype=unsigned_dtype), random_patterns))
    # View the same bits as signed values without overflowing a NumPy conversion.
    values = wp.array(patterns.view(np_dtype), dtype=dtype, device=device)
    counts = wp.empty(len(patterns), dtype=wp.int32, device=device)

    wp.launch(count_bits, dim=len(patterns), inputs=[values], outputs=[counts], device=device)

    # Count the unsigned patterns with Python, regardless of the signed view passed to Warp.
    expected = np.array([int(value).bit_count() for value in patterns], dtype=np.int32)
    np.testing.assert_array_equal(counts.numpy(), expected)


def test_unsupported_types_kernel_scope(test, device):
    """Verify kernel compilation rejects floating-point bit-count inputs."""

    @wp.kernel(module="unique")
    def count_float32(value: wp.float32, counts: wp.array[wp.int32]):
        counts[0] = wp.bit_count(value)

    counts = wp.empty(1, dtype=wp.int32, device=device)
    with test.assertRaisesRegex(RuntimeError, "Couldn't find function overload for 'bit_count'"):
        wp.launch(count_float32, dim=1, inputs=[wp.float32(45)], outputs=[counts], device=device)


class TestBitCount(unittest.TestCase):
    def test_python_scope(self):
        """Verify fixed-width bit counts for Warp and Python integers in Python scope."""
        for dtype in integer_types:
            width = np.dtype(wp.dtype_to_numpy(dtype)).itemsize * 8
            cases = ((0, 0), (45, 4), (-1, width), (-2, width - 1), (1 << (width - 1), 1))
            for value, expected in cases:
                with self.subTest(dtype=dtype.__name__, value=value):
                    self.assertEqual(wp.bit_count(dtype(value)), expected)
        self.assertEqual(wp.bit_count(45), 4)
        self.assertEqual(wp.bit_count(-1), 32)

    def test_unsupported_types_python_scope(self):
        """Verify Python-scope bit counts reject non-integer and vector inputs."""
        for dtype in (wp.float32, wp.float64, wp.bool, wp.vec2i):
            with self.subTest(dtype=dtype.__name__):
                with self.assertRaisesRegex(RuntimeError, "Couldn't find a function 'bit_count' compatible"):
                    wp.bit_count(dtype(0))


devices = get_test_devices()
for dtype in integer_types:
    add_function_test(TestBitCount, f"test_bit_count_{dtype.__name__}", test_bit_count, devices=devices, dtype=dtype)
add_function_test(
    TestBitCount, "test_unsupported_types_kernel_scope", test_unsupported_types_kernel_scope, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
