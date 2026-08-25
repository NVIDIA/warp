# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def simple_kernel(a: wp.array[float]):
    tid = wp.tid()
    a[tid] = float(tid)


@wp.kernel(launch_bounds=128)
def bounded_kernel(a: wp.array[float]):
    tid = wp.tid()
    a[tid] = float(tid)


@wp.kernel
def lazy_kernel(a: wp.array[float]):
    tid = wp.tid()
    a[tid] = float(tid) + 1.0


@wp.kernel
def generic_kernel(a: wp.array[Any]):
    tid = wp.tid()
    a[tid] = a[tid]


@wp.struct
class ExternalConstantParams:
    value: wp.float32


@wp.kernel(enable_backward=False, entry_point_abi="external_constant_params", module="unique")
def external_constant_params_kernel(params: ExternalConstantParams):
    value = params.value


def test_suggested_block_size_basic(test, device):
    block_size, min_grid_size = wp.get_suggested_block_size(simple_kernel, device)
    test.assertGreater(block_size, 0)
    test.assertEqual(block_size % 32, 0, "Block size must be a multiple of 32")
    test.assertLessEqual(block_size, 1024)
    test.assertGreater(min_grid_size, 0)

    # Calling again should return identical results
    block_size2, min_grid_size2 = wp.get_suggested_block_size(simple_kernel, device)
    test.assertEqual(block_size, block_size2)
    test.assertEqual(min_grid_size, min_grid_size2)


def test_suggested_block_size_launch_bounds(test, device):
    block_size, min_grid_size = wp.get_suggested_block_size(bounded_kernel, device)
    test.assertGreater(block_size, 0)
    test.assertLessEqual(block_size, 128)
    test.assertGreater(min_grid_size, 0)


def test_suggested_block_size_lazy_compile(test, device):
    block_size, min_grid_size = wp.get_suggested_block_size(lazy_kernel, device)
    test.assertGreater(block_size, 0)
    test.assertGreater(min_grid_size, 0)


def test_suggested_block_size_usable_in_launch(test, device):
    n = 256
    a = wp.zeros(n, dtype=float, device=device)

    block_size, _min_grid_size = wp.get_suggested_block_size(simple_kernel, device)
    wp.launch(simple_kernel, dim=n, inputs=[a], block_dim=block_size, device=device)

    np.testing.assert_allclose(a.numpy(), np.arange(n, dtype=np.float32))


def _constructor_kernel_func(a: wp.array[float]):
    tid = wp.tid()
    a[tid] = float(tid)


def test_suggested_block_size_kernel_constructor(test, device):
    """Kernel created via wp.Kernel() constructor works with get_suggested_block_size."""
    kernel = wp.Kernel(func=_constructor_kernel_func)
    block_size, min_grid_size = wp.get_suggested_block_size(kernel, device)
    test.assertGreater(block_size, 0)
    test.assertEqual(block_size % 32, 0, "Block size must be a multiple of 32")
    test.assertLessEqual(block_size, 1024)
    test.assertGreater(min_grid_size, 0)


def test_suggested_block_size_rejects_generic_parent(test, device):
    """Generic kernel parents produce an actionable overload-selection error."""
    with test.assertRaisesRegex(RuntimeError, "requires a concrete overload.*wp.overload"):
        wp.get_suggested_block_size(generic_kernel, device)


def test_suggested_block_size_accepts_external_constant_params(test, device):
    """Verify occupancy queries accept external constant-params kernels."""
    # This ABI cannot use wp.launch(), but occupancy only needs its raw CUDA
    # entry point and does not launch the kernel.
    block_size, min_grid_size = wp.get_suggested_block_size(external_constant_params_kernel, device)

    test.assertGreater(block_size, 0)
    test.assertGreater(min_grid_size, 0)


devices = get_selected_cuda_test_devices()


class TestOccupancy(unittest.TestCase):
    def test_suggested_block_size_cpu(self):
        """CPU fallback returns (1, 1)."""
        for kernel in (simple_kernel, generic_kernel):
            with self.subTest(kernel=kernel.key):
                result = wp.get_suggested_block_size(kernel, "cpu")
                self.assertEqual(result, (1, 1))


add_function_test(TestOccupancy, "test_suggested_block_size_basic", test_suggested_block_size_basic, devices=devices)
add_function_test(
    TestOccupancy,
    "test_suggested_block_size_launch_bounds",
    test_suggested_block_size_launch_bounds,
    devices=devices,
)
add_function_test(
    TestOccupancy,
    "test_suggested_block_size_lazy_compile",
    test_suggested_block_size_lazy_compile,
    devices=devices,
)
add_function_test(
    TestOccupancy,
    "test_suggested_block_size_usable_in_launch",
    test_suggested_block_size_usable_in_launch,
    devices=devices,
)
add_function_test(
    TestOccupancy,
    "test_suggested_block_size_kernel_constructor",
    test_suggested_block_size_kernel_constructor,
    devices=devices,
)
add_function_test(
    TestOccupancy,
    "test_suggested_block_size_rejects_generic_parent",
    test_suggested_block_size_rejects_generic_parent,
    devices=devices,
)
add_function_test(
    TestOccupancy,
    "test_suggested_block_size_accepts_external_constant_params",
    test_suggested_block_size_accepts_external_constant_params,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
