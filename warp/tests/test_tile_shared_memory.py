# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


# checks that we can configure shared memory to the expected size
def test_tile_shared_mem_size(test, device):
    DIM_M = 32
    DIM_N = 32

    BLOCK_DIM = 256

    @wp.kernel
    def compute(out: wp.array2d(dtype=float)):
        a = wp.tile_ones(DIM_M, DIM_N, dtype=float, storage="shared")
        b = wp.tile_ones(DIM_M, DIM_N, dtype=float, storage="shared") * 2.0

        c = a + b
        wp.tile_store(out, 0, 0, c)

    out = wp.empty((DIM_M, DIM_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)

    # check output
    assert_np_equal(out.numpy(), np.ones((DIM_M, DIM_N)) * 3.0)

    # check required shared memory
    expected_forward_bytes = DIM_M * DIM_N * 4 * 2
    expected_backward_bytes = expected_forward_bytes * 2

    # check shared memory for kernel on the device
    module_exec = compute.module.load(device, BLOCK_DIM)
    hooks = module_exec.get_kernel_hooks(compute)

    assert hooks.forward_smem_bytes == expected_forward_bytes
    assert hooks.backward_smem_bytes == expected_backward_bytes


# checks that we can configure shared memory > 48kb default
def test_tile_shared_mem_large(test, device):
    # set dimensions that require 64kb for the forward kernel
    DIM_M = 64
    DIM_N = 128

    BLOCK_DIM = 256

    # we disable backward kernel gen since 128k is not supported on most architectures
    @wp.kernel(enable_backward=False)
    def compute(out: wp.array2d(dtype=float)):
        a = wp.tile_ones(DIM_M, DIM_N, dtype=float, storage="shared")
        b = wp.tile_ones(DIM_M, DIM_N, dtype=float, storage="shared") * 2.0

        c = a + b
        wp.tile_store(out, 0, 0, c)

    out = wp.empty((DIM_M, DIM_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)

    # check output
    assert_np_equal(out.numpy(), np.ones((DIM_M, DIM_N)) * 3.0)

    # check required shared memory
    expected_forward_bytes = DIM_M * DIM_N * 4 * 2
    expected_backward_bytes = expected_forward_bytes * 2

    assert expected_forward_bytes == 2**16

    # check shared memory for kernel on the device
    module_exec = compute.module.load(device, BLOCK_DIM)
    hooks = module_exec.get_kernel_hooks(compute)

    assert hooks.forward_smem_bytes == expected_forward_bytes
    assert hooks.backward_smem_bytes == expected_backward_bytes


# checks that we can configure dynamic shared memory during graph capture
def test_tile_shared_mem_graph(test, device):
    DIM_M = 32
    DIM_N = 32

    BLOCK_DIM = 256

    @wp.kernel
    def compute(out: wp.array2d(dtype=float)):
        a = wp.tile_ones(DIM_M, DIM_N, dtype=float, storage="shared")
        b = wp.tile_ones(DIM_M, DIM_N, dtype=float, storage="shared") * 2.0

        c = a + b
        wp.tile_store(out, 0, 0, c)

    out = wp.empty((DIM_M, DIM_N), dtype=float, device=device)

    wp.load_module(device=device)

    wp.capture_begin(device, force_module_load=False)
    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)
    graph = wp.capture_end(device)

    wp.capture_launch(graph)

    # check output
    assert_np_equal(out.numpy(), np.ones((DIM_M, DIM_N)) * 3.0)

    # check required shared memory
    expected_forward_bytes = DIM_M * DIM_N * 4 * 2
    expected_backward_bytes = expected_forward_bytes * 2

    # check shared memory for kernel on the device
    module_exec = compute.module.load(device, BLOCK_DIM)
    hooks = module_exec.get_kernel_hooks(compute)

    assert hooks.forward_smem_bytes == expected_forward_bytes
    assert hooks.backward_smem_bytes == expected_backward_bytes


# checks that stack allocations work for user functions
def test_tile_shared_mem_func(test, device):
    DIM_M = 32
    DIM_N = 32

    BLOCK_DIM = 256

    @wp.func
    def add_tile_small():
        a = wp.tile_ones(16, 16, dtype=float, storage="shared")
        b = wp.tile_ones(16, 16, dtype=float, storage="shared") * 2.0

        return a + b

    @wp.func
    def add_tile_big():
        a = wp.tile_ones(64, 64, dtype=float, storage="shared")
        b = wp.tile_ones(64, 64, dtype=float, storage="shared") * 2.0

        return a + b

    @wp.kernel
    def compute(out: wp.array2d(dtype=float)):
        s = add_tile_small()
        b = add_tile_big()

        wp.tile_store(out, 0, 0, b)

    out = wp.empty((DIM_M, DIM_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)

    # check shared memory for kernel on the device
    module_exec = compute.module.load(device, BLOCK_DIM)
    hooks = module_exec.get_kernel_hooks(compute)

    # ensure that total required dynamic shared is the larger of the two tiles
    expected_required_shared = 64 * 64 * 4 * 2

    assert hooks.forward_smem_bytes == expected_required_shared
    assert hooks.backward_smem_bytes == expected_required_shared * 2


devices = get_cuda_test_devices()


class TestTileSharedMemory(unittest.TestCase):
    pass


add_function_test(
    TestTileSharedMemory, "test_tile_shared_mem_size", test_tile_shared_mem_size, devices=devices, check_output=False
)
add_function_test(
    TestTileSharedMemory, "test_tile_shared_mem_large", test_tile_shared_mem_large, devices=devices, check_output=False
)
add_function_test(TestTileSharedMemory, "test_tile_shared_mem_graph", test_tile_shared_mem_graph, devices=devices)
add_function_test(TestTileSharedMemory, "test_tile_shared_mem_func", test_tile_shared_mem_func, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
