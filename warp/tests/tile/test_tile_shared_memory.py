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

    @wp.kernel(module="unique")
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
    @wp.kernel(enable_backward=False, module="unique")
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

    @wp.kernel(module="unique")
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

    @wp.kernel(module="unique")
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

    @wp.kernel(module="unique")
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


def test_tile_shared_vec_accumulation(test, device):
    BLOCK_DIM = 256

    @wp.kernel(module="unique")
    def compute(indices: wp.array(dtype=int), vecs: wp.array(dtype=wp.vec3), output: wp.array2d(dtype=float)):
        i, j = wp.tid()

        idx_tile = wp.tile_load(indices, shape=BLOCK_DIM, offset=i * BLOCK_DIM)
        idx = idx_tile[j]

        s = wp.tile_zeros(shape=(1, 3), dtype=float)

        s[0, 0] += vecs[idx].x
        s[0, 1] += vecs[idx].y
        s[0, 2] += vecs[idx].z

        wp.tile_store(output, s, offset=(i, 0))

    N = BLOCK_DIM * 3

    basis_vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    vecs = wp.array(basis_vecs, dtype=wp.vec3, requires_grad=True, device=device)

    rng = np.random.default_rng(42)
    indices_np = rng.integers(0, 3, size=N)

    indices = wp.array(indices_np, dtype=int, requires_grad=True, device=device)

    output = wp.zeros(shape=(3, 3), dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch_tiled(compute, dim=3, inputs=[indices, vecs, output], block_dim=BLOCK_DIM, device=device)

    output.grad = wp.ones_like(output)

    tape.backward()

    n0 = np.count_nonzero(indices_np == 0)
    n1 = np.count_nonzero(indices_np == 1)
    n2 = np.count_nonzero(indices_np == 2)
    true_grads = np.array([[n0, n0, n0], [n1, n1, n1], [n2, n2, n2]])

    indices_np = indices_np.reshape((3, BLOCK_DIM))

    def compute_row(idx):
        n0 = np.count_nonzero(indices_np[idx, :] == 0)
        n1 = np.count_nonzero(indices_np[idx, :] == 1)
        n2 = np.count_nonzero(indices_np[idx, :] == 2)
        return np.array([1, 0, 0]) * n0 + np.array([0, 1, 0]) * n1 + np.array([0, 0, 1]) * n2

    row_0 = compute_row(0)
    row_1 = compute_row(1)
    row_2 = compute_row(2)

    true_vecs = np.stack([row_0, row_1, row_2])

    assert_np_equal(output.numpy(), true_vecs)
    assert_np_equal(vecs.grad.numpy(), true_grads)


def test_tile_shared_simple_reduction_add(test, device):
    BLOCK_DIM = 256

    @wp.kernel(module="unique")
    def compute(x: wp.array(dtype=float), y: wp.array(dtype=float)):
        i, j = wp.tid()

        t = wp.tile_load(x, shape=BLOCK_DIM, offset=BLOCK_DIM * i)

        k = BLOCK_DIM // 2
        while k > 0:
            if j < k:
                t[j] += t[j + k]
            k //= 2

        wp.tile_store(y, wp.tile_view(t, offset=(0,), shape=(1,)), i)

    N = BLOCK_DIM * 4
    x_np = np.arange(N, dtype=np.float32)
    x = wp.array(x_np, dtype=float, device=device)
    y = wp.zeros(4, dtype=float, device=device)

    wp.launch_tiled(compute, dim=4, inputs=[x], outputs=[y], block_dim=BLOCK_DIM, device=device)

    assert_np_equal(np.sum(y.numpy()), np.sum(x_np))


def test_tile_shared_simple_reduction_sub(test, device):
    BLOCK_DIM = 256

    @wp.kernel(module="unique")
    def compute(x: wp.array(dtype=float), y: wp.array(dtype=float)):
        i, j = wp.tid()

        t = wp.tile_load(x, shape=BLOCK_DIM, offset=BLOCK_DIM * i)

        k = BLOCK_DIM // 2
        while k > 0:
            if j < k:
                t[j] -= t[j + k]
            k //= 2

        wp.tile_store(y, wp.tile_view(t, offset=(0,), shape=(1,)), i)

    N = BLOCK_DIM * 4
    x_np = np.arange(N, dtype=np.float32)
    x = wp.array(x_np, dtype=float, device=device)
    y = wp.zeros(4, dtype=float, device=device)

    wp.launch_tiled(compute, dim=4, inputs=[x], outputs=[y], block_dim=BLOCK_DIM, device=device)

    assert_np_equal(np.sum(y.numpy()), 0.0)


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
add_function_test(
    TestTileSharedMemory, "test_tile_shared_vec_accumulation", test_tile_shared_vec_accumulation, devices=devices
)
add_function_test(
    TestTileSharedMemory,
    "test_tile_shared_simple_reduction_add",
    test_tile_shared_simple_reduction_add,
    devices=devices,
)
add_function_test(
    TestTileSharedMemory,
    "test_tile_shared_simple_reduction_sub",
    test_tile_shared_simple_reduction_sub,
    devices=devices,
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
