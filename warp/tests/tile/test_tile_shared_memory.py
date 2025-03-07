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
        a = wp.tile_ones(shape=(DIM_M, DIM_N), dtype=float, storage="shared")
        b = wp.tile_ones(shape=(DIM_M, DIM_N), dtype=float, storage="shared") * 2.0

        c = a + b
        wp.tile_store(out, c)

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
        a = wp.tile_ones(shape=(DIM_M, DIM_N), dtype=float, storage="shared")
        b = wp.tile_ones(shape=(DIM_M, DIM_N), dtype=float, storage="shared") * 2.0

        c = a + b
        wp.tile_store(out, c)

    out = wp.empty((DIM_M, DIM_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)

    # check output
    assert_np_equal(out.numpy(), np.ones((DIM_M, DIM_N)) * 3.0)

    # check required shared memory
    expected_forward_bytes = DIM_M * DIM_N * 4 * 2
    expected_backward_bytes = 0

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
        a = wp.tile_ones(shape=(DIM_M, DIM_N), dtype=float, storage="shared")
        b = wp.tile_ones(shape=(DIM_M, DIM_N), dtype=float, storage="shared") * 2.0

        c = a + b
        wp.tile_store(out, c)

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
    DIM_M = 64
    DIM_N = 64

    SMALL_DIM_M = 64 // 4
    SMALL_DIM_N = 64 // 4

    BLOCK_DIM = 256

    @wp.func
    def add_tile_small():
        a = wp.tile_ones(shape=(SMALL_DIM_M, SMALL_DIM_N), dtype=float, storage="shared")
        b = wp.tile_ones(shape=(SMALL_DIM_M, SMALL_DIM_N), dtype=float, storage="shared") * 2.0

        return a + b

    @wp.func
    def add_tile_big():
        a = wp.tile_ones(shape=(DIM_M, DIM_N), dtype=float, storage="shared")
        b = wp.tile_ones(shape=(DIM_M, DIM_N), dtype=float, storage="shared") * 2.0

        return a + b

    @wp.kernel
    def compute(out: wp.array2d(dtype=float)):
        s = add_tile_small()
        b = add_tile_big()

        wp.tile_store(out, b)

    out = wp.empty((DIM_M, DIM_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)

    # check shared memory for kernel on the device
    module_exec = compute.module.load(device, BLOCK_DIM)
    hooks = module_exec.get_kernel_hooks(compute)

    # ensure that total required dynamic shared is the larger of the two tiles
    expected_required_shared = 64 * 64 * 4 * 2

    assert hooks.forward_smem_bytes == expected_required_shared
    assert hooks.backward_smem_bytes == expected_required_shared * 2


def round_up(a, b):
    return b * ((a + b - 1) // b)


# checks that using non-16B aligned sizes work
def test_tile_shared_non_aligned(test, device):
    # Tile size = 4 (float) * 1 * 3 = 12B % 16 != 0
    DIM_M = 1
    DIM_N = 3

    BLOCK_DIM = 256

    @wp.func
    def foo():
        a = wp.tile_ones(shape=(DIM_M, DIM_N), dtype=float, storage="shared") * 2.0
        b = wp.tile_ones(shape=(DIM_M, DIM_N), dtype=float, storage="shared") * 3.0
        return a + b

    @wp.kernel
    def compute(out: wp.array2d(dtype=float)):
        # This test the logic in the stack allocator, which should increment and
        # decrement the stack pointer each time foo() is called
        # Failing to do so correct will make b out of bounds and corrupt the results
        for _ in range(4096):
            foo()
        b = wp.tile_ones(shape=(DIM_M, DIM_N), dtype=float, storage="shared")
        wp.tile_store(out, b)

    out = wp.empty((DIM_M, DIM_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)

    assert_np_equal(out.numpy(), np.ones((DIM_M, DIM_N), dtype=float))

    # check shared memory for kernel on the device
    module_exec = compute.module.load(device, BLOCK_DIM)
    hooks = module_exec.get_kernel_hooks(compute)

    # ensure that total required dynamic shared is the larger of the two tiles
    expected_required_shared = 3 * round_up(DIM_M * DIM_N * 4, 16)

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
add_function_test(TestTileSharedMemory, "test_tile_shared_non_aligned", test_tile_shared_non_aligned, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
