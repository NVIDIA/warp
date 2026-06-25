# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import math
import unittest
from typing import Any

import warp as wp
from warp._src.types import check_array_shape
from warp.tests.unittest_utils import *


@wp.kernel
def conditional_sum(result: wp.array(dtype=wp.uint64)):
    i, _j, _k = wp.tid()

    if i == 0:
        wp.atomic_add(result, 0, wp.uint64(1))


def test_large_launch_large_kernel(test, device):
    """Test tid() on kernel launch of 2**33 threads.

    The function conditional sum will add 1 to result for every thread that has an i index of 0.
    Due to the size of the grid, this test is not run on CPUs
    """
    test_result = wp.zeros(shape=(1,), dtype=wp.uint64, device=device)

    large_dim_length = 2**16
    half_result = large_dim_length * large_dim_length

    wp.launch(kernel=conditional_sum, dim=[2, large_dim_length, large_dim_length], inputs=[test_result], device=device)
    test.assertEqual(test_result.numpy()[0], half_result)


@wp.kernel(grid_stride=True)
def count_elements(result: wp.array(dtype=wp.uint64)):
    wp.atomic_add(result, 0, wp.uint64(1))


@wp.kernel(grid_stride=True)
def conditional_sum_grid_stride(result: wp.array(dtype=wp.uint64)):
    i, _j, _k = wp.tid()

    if i == 0:
        wp.atomic_add(result, 0, wp.uint64(1))


def test_large_launch_max_blocks(test, device):
    # Loop over 1000x1x1 elements using a grid of 256 threads
    test_result = wp.zeros(shape=(1,), dtype=wp.uint64, device=device)
    wp.launch(count_elements, (1000,), inputs=[test_result], max_blocks=1, device=device)
    test.assertEqual(test_result.numpy()[0], 1000)

    # Loop over 2x10x10 elements using a grid of 256 threads, using the tid() index to count half the elements
    test_result.zero_()
    wp.launch(
        conditional_sum_grid_stride,
        (
            2,
            50,
            10,
        ),
        inputs=[test_result],
        max_blocks=1,
        device=device,
    )
    test.assertEqual(test_result.numpy()[0], 500)


def test_large_launch_very_large_kernel(test, device):
    """Due to the size of the grid, this test is not run on CPUs"""

    # Dim is chosen to be larger than the maximum CUDA one-dimensional grid size (total threads)
    dim = (2**31 - 1) * 256 + 1
    test_result = wp.zeros(shape=(1,), dtype=wp.uint64, device=device)
    wp.launch(count_elements, (dim,), inputs=[test_result], device=device)
    test.assertEqual(test_result.numpy()[0], dim)


@wp.kernel
def check_array_equal_value_2d(data: wp.array2d(dtype=Any), expect: Any):
    i, j = wp.tid()
    wp.expect_eq(data[i, j], expect)


@wp.kernel
def check_array_equal_value_3d(data: wp.array3d(dtype=Any), expect: Any):
    i, j, k = wp.tid()
    wp.expect_eq(data[i, j, k], expect)


@wp.kernel
def check_array_equal_value_4d(data: wp.array4d(dtype=Any), expect: Any):
    i, j, k, l = wp.tid()
    wp.expect_eq(data[i, j, k, l], expect)


def test_large_arrays(test, device):
    # Exercises zero_/fill_/kernel-launch on arrays with >2**31 elements across
    # 2-D, 3-D, and 4-D shapes. int8 keeps each allocation near 2 GiB.
    ndim_kernels = (
        (2, check_array_equal_value_2d),
        (3, check_array_equal_value_3d),
        (4, check_array_equal_value_4d),
    )
    for ndim, kernel in ndim_kernels:
        dim_x = math.ceil((2**31) ** (1 / ndim))
        shape = (dim_x,) * ndim

        a1 = wp.zeros(shape, dtype=wp.int8, device=device)
        try:
            a1.fill_(127)
            wp.launch(kernel, shape, inputs=[a1, wp.int8(127)], device=device)

            a1.zero_()
            wp.launch(kernel, shape, inputs=[a1, wp.int8(0)], device=device)
        finally:
            try:
                wp.synchronize_device(device)
            finally:
                del a1
                gc.collect()


def test_large_array_excessive_zeros(test, device):
    # Tests the allocation of an array with length exceeding 2**31-1 in a dimension

    with test.assertRaisesRegex(
        ValueError, "Array shapes must not exceed the maximum representable value of a signed 32-bit integer"
    ):
        _ = wp.zeros((2**31), dtype=int, device=device)


# grid_stride=False opts these kernels into the lean 3D launch (grid-stride is the default), so
# max_blocks raises while oversized dims are serviced directly by the 3D grid.
@wp.kernel(grid_stride=False)
def conditional_sum_max_blocks(result: wp.array(dtype=wp.uint64)):
    i, _j, _k = wp.tid()
    if i == 0:
        wp.atomic_add(result, 0, wp.uint64(1))


@wp.kernel(grid_stride=False)
def conditional_sum_large_dim(result: wp.array(dtype=wp.uint64)):
    i, _j, _k = wp.tid()
    if i == 0:
        wp.atomic_add(result, 0, wp.uint64(1))


def test_max_blocks_requires_grid_stride(test, device):
    # max_blocks > 0 can't be honored without a grid-stride loop, so a non-grid_stride CUDA kernel
    # raises and points at the grid_stride opt-in.
    test_result = wp.zeros(shape=(1,), dtype=wp.uint64, device=device)

    with test.assertRaisesRegex(RuntimeError, "grid_stride=True"):
        wp.launch(
            kernel=conditional_sum_max_blocks,
            dim=(2, 50, 50),
            inputs=[test_result],
            max_blocks=4,
            device=device,
        )


# grid_stride=False so the CPU no-op and block_dim normalization checks exercise the lean launch path.
@wp.kernel(grid_stride=False)
def count_all_threads(result: wp.array(dtype=wp.uint64)):
    wp.atomic_add(result, 0, wp.uint64(1))


def test_block_dim_zero_normalized(test, device):
    # A non-positive block_dim is clamped to 256 (matching the native launcher) rather than raising a
    # divide-by-zero in the Python block-count check.
    test_result = wp.zeros(shape=(1,), dtype=wp.uint64, device=device)
    wp.launch(count_all_threads, dim=1000, inputs=[test_result], block_dim=0, device=device)
    wp.synchronize_device(device)
    test.assertEqual(test_result.numpy()[0], 1000)


@wp.kernel(grid_stride=False)
def conditional_sum_set_dim(result: wp.array(dtype=wp.uint64)):
    i, _j, _k = wp.tid()
    if i == 0:
        wp.atomic_add(result, 0, wp.uint64(1))


def test_set_dim_lean_3d(test, device):
    # A recorded lean (grid_stride=False) launch runs on a 3D grid, so set_dim() can resize it past
    # the old 1D gridDim.x limit (2**31-1 blocks) -- the 3D grid covers it directly, no fallback.
    test_result = wp.zeros(shape=(1,), dtype=wp.uint64, device=device)

    cmd = wp.launch(
        conditional_sum_set_dim,
        dim=1,
        inputs=[test_result],
        device=device,
        record_cmd=True,
        block_dim=1,
    )

    # Resizing past the old 1D gridDim.x limit is fine on the lean 3D grid (block_dim=1 so the block
    # count equals dim, and 2**31 exceeds 2**31-1).
    cmd.set_dim(2**31)
    cmd.launch()
    wp.synchronize_device(device)
    test.assertEqual(test_result.numpy()[0], 1)  # only the i==0 work item increments

    # A grid-stride kernel resizes past the limit just as well.
    cmd_gs = wp.launch(
        conditional_sum_grid_stride,
        dim=1,
        inputs=[test_result],
        device=device,
        record_cmd=True,
        block_dim=1,
    )
    cmd_gs.set_dim(2**31)


def test_set_dim_zero_lean(test, device):
    # set_dim(0) on a recorded lean (grid_stride=False) launch resizes to an empty launch. The native
    # 3D launcher must clamp grid.z to 1 (like grid.x/grid.y) so cuLaunchKernel isn't invoked with
    # grid_z=0; the launch is then a no-op rather than a crash.
    test_result = wp.zeros(shape=(1,), dtype=wp.uint64, device=device)
    cmd = wp.launch(count_all_threads, dim=4, inputs=[test_result], device=device, record_cmd=True, block_dim=1)
    cmd.set_dim(0)
    cmd.launch()
    wp.synchronize_device(device)
    test.assertEqual(test_result.numpy()[0], 0)  # empty launch ran nothing


def test_default_grid_stride(test, device):
    # The hard default is grid-stride, so an un-annotated kernel handles max_blocks normally. Setting
    # wp.config.default_grid_stride = False opts un-annotated kernels into the 1D launch, where
    # max_blocks then raises (the global default drives each kernel's effective grid_stride at build).
    result = wp.zeros(1, dtype=wp.uint64, device=device)

    @wp.kernel(module="unique")
    def count_default(r: wp.array(dtype=wp.uint64)):
        wp.atomic_add(r, 0, wp.uint64(1))

    wp.launch(count_default, dim=1000, inputs=[result], max_blocks=1, device=device)
    wp.synchronize_device(device)
    test.assertEqual(result.numpy()[0], 1000)

    saved = wp.config.default_grid_stride
    try:
        wp.config.default_grid_stride = False

        @wp.kernel(module="unique")
        def count_opted_into_1d(r: wp.array(dtype=wp.uint64)):
            wp.atomic_add(r, 0, wp.uint64(1))

        result.zero_()
        with test.assertRaisesRegex(RuntimeError, "grid_stride=True"):
            wp.launch(count_opted_into_1d, dim=1000, inputs=[result], max_blocks=1, device=device)
    finally:
        wp.config.default_grid_stride = saved


def test_large_dim_lean_3d(test, device):
    # A lean (grid_stride=False) kernel launches on a 3D grid, so a block count exceeding the 1D
    # gridDim.x limit (2**31-1) is serviced directly -- no grid-stride loop, fallback, or warning.
    # block_dim=1 makes the block count equal dim, so 2**31 blocks exceeds the 1D limit.
    test_result = wp.zeros(shape=(1,), dtype=wp.uint64, device=device)

    wp.launch(
        kernel=conditional_sum_large_dim,
        dim=2**31,
        inputs=[test_result],
        block_dim=1,
        device=device,
    )
    wp.synchronize_device(device)
    test.assertEqual(test_result.numpy()[0], 1)


devices = get_test_devices()


class TestLarge(unittest.TestCase):
    def test_max_blocks_ignored_on_cpu(self):
        # max_blocks is a documented no-op on CPU, so a lean (grid_stride=False) kernel launched with
        # max_blocks must not raise there (the max_blocks raise is CUDA-only).
        test_result = wp.zeros(shape=(1,), dtype=wp.uint64, device="cpu")
        wp.launch(count_all_threads, dim=1000, inputs=[test_result], max_blocks=4, device="cpu")
        self.assertEqual(test_result.numpy()[0], 1000)

    def test_large_array_excessive_numpy(self):
        # Shape-validation is pure Python; no ndarray or device allocation
        # needed to exercise the 2**31-element boundary check. The user-facing
        # wp.array(ndarray) path reaches this same validator via
        # wp.array.__init__ -> _init_from_data -> _init_new -> check_array_shape;
        # the separate test below covers the integration path.
        with self.assertRaisesRegex(
            ValueError,
            "Array shapes must not exceed the maximum representable value of a signed 32-bit integer",
        ):
            check_array_shape((2**31,))

    def test_large_array_excessive_ndarray(self):
        # Exercise the wp.array(ndarray) -> _init_from_data -> _init_new path
        # at the 2**31 boundary without allocating a real 2**31-element buffer.
        # np.broadcast_to returns a zero-stride view over a single-element
        # source, and np.asarray preserves the view (no copy when dtypes
        # match). _init_new calls check_array_shape before any device
        # allocation, so the ValueError fires for free.
        large_view = np.broadcast_to(np.zeros(1, dtype=np.int8), (2**31,))
        with self.assertRaisesRegex(
            ValueError,
            "Array shapes must not exceed the maximum representable value of a signed 32-bit integer",
        ):
            _ = wp.array(large_view, dtype=wp.int8, device="cpu")


add_function_test(
    TestLarge,
    "test_large_launch_large_kernel",
    test_large_launch_large_kernel,
    devices=get_selected_cuda_test_devices(),
)

add_function_test(TestLarge, "test_large_launch_max_blocks", test_large_launch_max_blocks, devices=devices)
add_function_test(
    TestLarge,
    "test_large_launch_very_large_kernel",
    test_large_launch_very_large_kernel,
    devices=get_selected_cuda_test_devices(),
)

add_function_test(TestLarge, "test_large_arrays", test_large_arrays, devices=devices)
add_function_test(TestLarge, "test_large_array_excessive_zeros", test_large_array_excessive_zeros, devices=devices)

add_function_test(
    TestLarge,
    "test_max_blocks_requires_grid_stride",
    test_max_blocks_requires_grid_stride,
    devices=get_selected_cuda_test_devices(mode="basic"),
)
add_function_test(
    TestLarge,
    "test_large_dim_lean_3d",
    test_large_dim_lean_3d,
    devices=get_selected_cuda_test_devices(mode="basic"),
)
add_function_test(
    TestLarge,
    "test_set_dim_lean_3d",
    test_set_dim_lean_3d,
    devices=get_selected_cuda_test_devices(mode="basic"),
)
add_function_test(
    TestLarge,
    "test_set_dim_zero_lean",
    test_set_dim_zero_lean,
    devices=get_selected_cuda_test_devices(mode="basic"),
)
add_function_test(
    TestLarge,
    "test_default_grid_stride",
    test_default_grid_stride,
    devices=get_selected_cuda_test_devices(mode="basic"),
)
add_function_test(
    TestLarge,
    "test_block_dim_zero_normalized",
    test_block_dim_zero_normalized,
    devices=get_selected_cuda_test_devices(mode="basic"),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
