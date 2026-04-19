# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

    # preload the unique module
    wp.load_module(compute.module, device=device, block_dim=BLOCK_DIM)

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)

    wp.capture_launch(capture.graph)

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


def test_tile_scatter_add_basic(test, device):
    """Each thread adds its index + 1 to a distinct slot; verify values."""
    TILE_SIZE = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(out: wp.array(dtype=float)):
        _tile, i = wp.tid()
        t = wp.tile_zeros(shape=TILE_SIZE, dtype=float, storage="shared")
        wp.tile_scatter_add(t, i, float(i + 1), True)
        out[i] = wp.tile_extract(t, i)

    out = wp.zeros(TILE_SIZE, dtype=float, device=device)
    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=TILE_SIZE, device=device)

    assert_np_equal(out.numpy(), np.arange(1, TILE_SIZE + 1, dtype=np.float32))


def test_tile_scatter_add_conflicting(test, device):
    """All threads add 1.0 to the same index; verify the sum equals block_dim."""
    TILE_SIZE = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(out: wp.array(dtype=float)):
        _tile, i = wp.tid()
        t = wp.tile_zeros(shape=TILE_SIZE, dtype=float, storage="shared")
        wp.tile_scatter_add(t, 0, 1.0, True)
        val = wp.tile_extract(t, 0)
        if i == 0:
            out[0] = val

    out = wp.zeros(1, dtype=float, device=device)
    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=TILE_SIZE, device=device)

    assert_np_equal(out.numpy()[0], float(TILE_SIZE))


def test_tile_scatter_add_partial(test, device):
    """Only even-indexed threads add; odd slots stay zero."""
    TILE_SIZE = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(out: wp.array(dtype=float)):
        _tile, i = wp.tid()
        t = wp.tile_zeros(shape=TILE_SIZE, dtype=float, storage="shared")
        wp.tile_scatter_add(t, i, float(i + 1), (i % 2) == 0)
        out[i] = wp.tile_extract(t, i)

    out = wp.zeros(TILE_SIZE, dtype=float, device=device)
    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=TILE_SIZE, device=device)

    result = out.numpy()
    for i in range(TILE_SIZE):
        if i % 2 == 0:
            assert_np_equal(result[i], float(i + 1))
        else:
            assert_np_equal(result[i], 0.0)


def test_tile_scatter_add_2d(test, device):
    """Scatter-add with a 2D shared tile."""
    ROWS = 8
    COLS = 8
    BLOCK_DIM = ROWS * COLS

    @wp.kernel(enable_backward=False, module="unique")
    def compute(out: wp.array2d(dtype=float)):
        _tile, idx = wp.tid()
        row = idx // COLS
        col = idx % COLS
        t = wp.tile_zeros(shape=(ROWS, COLS), dtype=float, storage="shared")
        wp.tile_scatter_add(t, row, col, float(idx + 1), True)
        out[row, col] = wp.tile_extract(t, row, col)

    out = wp.zeros((ROWS, COLS), dtype=float, device=device)
    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)

    expected = np.arange(1, BLOCK_DIM + 1, dtype=np.float32).reshape(ROWS, COLS)
    assert_np_equal(out.numpy(), expected)


def test_tile_scatter_add_grad_basic(test, device):
    """Gradient flows through tile_scatter_add: output = input * 2 via shared tile."""
    TILE_SIZE = 64

    @wp.kernel(module="unique")
    def compute(inp: wp.array(dtype=float), out: wp.array(dtype=float)):
        _tile, i = wp.tid()
        val = inp[i] * 2.0
        t = wp.tile_zeros(shape=TILE_SIZE, dtype=float, storage="shared")
        wp.tile_scatter_add(t, i, val, True)
        out[i] = wp.tile_extract(t, i)

    inp = wp.array(np.ones(TILE_SIZE, dtype=np.float32), requires_grad=True, device=device)
    out = wp.zeros(TILE_SIZE, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=TILE_SIZE, device=device)

    out.grad = wp.ones_like(out, device=device)
    tape.backward()

    assert_np_equal(out.numpy(), np.full(TILE_SIZE, 2.0))
    assert_np_equal(inp.grad.numpy(), np.full(TILE_SIZE, 2.0))


def test_tile_scatter_add_grad_partial(test, device):
    """has_value gates the adjoint: only participating threads receive gradients."""
    TILE_SIZE = 64

    @wp.kernel(module="unique")
    def compute(inp: wp.array(dtype=float), out: wp.array(dtype=float)):
        _tile, i = wp.tid()
        val = inp[i] * 2.0
        t = wp.tile_zeros(shape=TILE_SIZE, dtype=float, storage="shared")
        wp.tile_scatter_add(t, i, val, (i % 2) == 0)
        out[i] = wp.tile_extract(t, i)

    inp = wp.array(np.ones(TILE_SIZE, dtype=np.float32), requires_grad=True, device=device)
    out = wp.zeros(TILE_SIZE, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=TILE_SIZE, device=device)

    out.grad = wp.ones_like(out, device=device)
    tape.backward()

    expected_grad = np.zeros(TILE_SIZE, dtype=np.float32)
    expected_grad[0::2] = 2.0
    assert_np_equal(inp.grad.numpy(), expected_grad)


def test_tile_scatter_add_grad_conflicting(test, device):
    """Gradient fans out correctly when multiple threads scatter-add to the same index."""
    TILE_SIZE = 64

    @wp.kernel(module="unique")
    def compute(inp: wp.array(dtype=float), out: wp.array(dtype=float)):
        _tile, i = wp.tid()
        val = inp[i]
        t = wp.tile_zeros(shape=1, dtype=float, storage="shared")
        wp.tile_scatter_add(t, 0, val, True)
        result = wp.tile_extract(t, 0)
        if i == 0:
            out[0] = result

    inp = wp.array(np.ones(TILE_SIZE, dtype=np.float32), requires_grad=True, device=device)
    out = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=TILE_SIZE, device=device)

    out.grad = wp.ones_like(out, device=device)
    tape.backward()

    # Forward: out[0] = sum of all inp[i] = 64.0
    assert_np_equal(out.numpy()[0], float(TILE_SIZE))
    # Backward: d(out[0])/d(inp[i]) = 1.0 for all i
    assert_np_equal(inp.grad.numpy(), np.ones(TILE_SIZE, dtype=np.float32))


# ---- Non-atomic scatter-add tests (atomic=False) ----


def test_tile_scatter_add_non_atomic_1d(test, device):
    """Non-atomic scatter-add with unique indices per thread (1D)."""
    TILE_SIZE = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(out: wp.array(dtype=float)):
        _tile, i = wp.tid()
        t = wp.tile_zeros(shape=TILE_SIZE, dtype=float, storage="shared")
        wp.tile_scatter_add(t, i, float(i + 1), True, atomic=False)
        out[i] = wp.tile_extract(t, i)

    out = wp.zeros(TILE_SIZE, dtype=float, device=device)
    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=TILE_SIZE, device=device)

    assert_np_equal(out.numpy(), np.arange(1, TILE_SIZE + 1, dtype=np.float32))


def test_tile_scatter_add_non_atomic_2d(test, device):
    """Non-atomic scatter-add with unique (row, col) per thread (2D)."""
    ROWS = 4
    COLS = 16
    TILE_SIZE = ROWS * COLS

    @wp.kernel(enable_backward=False, module="unique")
    def compute(out: wp.array2d(dtype=float)):
        _tile, i = wp.tid()
        row = i // COLS
        col = i % COLS
        t = wp.tile_zeros(shape=(ROWS, COLS), dtype=float, storage="shared")
        wp.tile_scatter_add(t, row, col, float(i + 1), True, atomic=False)
        out[row, col] = wp.tile_extract(t, row, col)

    out = wp.zeros((ROWS, COLS), dtype=float, device=device)
    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=TILE_SIZE, device=device)

    expected = np.arange(1, TILE_SIZE + 1, dtype=np.float32).reshape(ROWS, COLS)
    assert_np_equal(out.numpy(), expected)


def test_tile_scatter_add_non_atomic_grad(test, device):
    """Gradient flows correctly through non-atomic tile_scatter_add."""
    TILE_SIZE = 64

    @wp.kernel(module="unique")
    def compute(inp: wp.array(dtype=float), out: wp.array(dtype=float)):
        _tile, i = wp.tid()
        val = inp[i] * 2.0
        t = wp.tile_zeros(shape=TILE_SIZE, dtype=float, storage="shared")
        wp.tile_scatter_add(t, i, val, True, atomic=False)
        out[i] = wp.tile_extract(t, i)

    inp = wp.array(np.ones(TILE_SIZE, dtype=np.float32), requires_grad=True, device=device)
    out = wp.zeros(TILE_SIZE, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=TILE_SIZE, device=device)

    out.grad = wp.ones_like(out, device=device)
    tape.backward()

    assert_np_equal(out.numpy(), np.full(TILE_SIZE, 2.0))
    assert_np_equal(inp.grad.numpy(), np.full(TILE_SIZE, 2.0))


def test_tile_shared_coalesced_mat33(test, device):
    """Shared tile load/store of mat33 exercises the coalesced byte-copy path (sizeof(mat33) = 36 > 16)."""
    TILE_SIZE = 8
    BLOCK_DIM = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        inp: wp.array(dtype=wp.mat33),
        out: wp.array(dtype=wp.mat33),
    ):
        i = wp.tid()
        t = wp.tile_load(inp, shape=TILE_SIZE, offset=0, storage="shared")
        wp.tile_store(out, t, offset=0)

    inp_np = np.arange(TILE_SIZE * 9, dtype=np.float32).reshape(TILE_SIZE, 3, 3)
    inp = wp.array(inp_np, dtype=wp.mat33, device=device)
    out = wp.zeros(TILE_SIZE, dtype=wp.mat33, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    np.testing.assert_allclose(out.numpy(), inp_np)


def test_tile_shared_coalesced_mat44(test, device):
    """Shared tile load/store of mat44 exercises the coalesced byte-copy path (sizeof(mat44) = 64 > 16)."""
    TILE_SIZE = 4
    BLOCK_DIM = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        inp: wp.array(dtype=wp.mat44),
        out: wp.array(dtype=wp.mat44),
    ):
        i = wp.tid()
        t = wp.tile_load(inp, shape=TILE_SIZE, offset=0, storage="shared")
        wp.tile_store(out, t, offset=0)

    inp_np = np.arange(TILE_SIZE * 16, dtype=np.float32).reshape(TILE_SIZE, 4, 4)
    inp = wp.array(inp_np, dtype=wp.mat44, device=device)
    out = wp.zeros(TILE_SIZE, dtype=wp.mat44, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    np.testing.assert_allclose(out.numpy(), inp_np)


def test_tile_scatter_masked_basic(test, device):
    """Each thread writes its index; verify all values are visible after the call."""
    TILE_SIZE = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(out: wp.array(dtype=int)):
        _tile, i = wp.tid()
        t = wp.tile_zeros(shape=TILE_SIZE, dtype=int, storage="shared")
        wp.tile_scatter_masked(t, i, i + 1, True)
        out[i] = wp.tile_extract(t, i)

    out = wp.zeros(TILE_SIZE, dtype=int, device=device)
    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=TILE_SIZE, device=device)

    np.testing.assert_array_equal(out.numpy(), np.arange(1, TILE_SIZE + 1))


def test_tile_scatter_masked_partial(test, device):
    """Only even-indexed threads write; odd slots stay zero."""
    TILE_SIZE = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(out: wp.array(dtype=int)):
        _tile, i = wp.tid()
        t = wp.tile_zeros(shape=TILE_SIZE, dtype=int, storage="shared")
        wp.tile_scatter_masked(t, i, i + 1, (i % 2) == 0)
        out[i] = wp.tile_extract(t, i)

    out = wp.zeros(TILE_SIZE, dtype=int, device=device)
    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=TILE_SIZE, device=device)

    result = out.numpy()
    for i in range(TILE_SIZE):
        if i % 2 == 0:
            test.assertEqual(result[i], i + 1)
        else:
            test.assertEqual(result[i], 0)


def test_tile_scatter_masked_cross_thread(test, device):
    """Each thread reads a neighbor's slot, verifying the sync barrier works."""
    TILE_SIZE = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(out: wp.array(dtype=int)):
        _tile, i = wp.tid()
        t = wp.tile_zeros(shape=TILE_SIZE, dtype=int, storage="shared")
        wp.tile_scatter_masked(t, i, i * 10, True)
        neighbor = (i + 1) % TILE_SIZE
        out[i] = wp.tile_extract(t, neighbor)

    out = wp.zeros(TILE_SIZE, dtype=int, device=device)
    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=TILE_SIZE, device=device)

    expected = np.array([((i + 1) % TILE_SIZE) * 10 for i in range(TILE_SIZE)], dtype=np.int32)
    np.testing.assert_array_equal(out.numpy(), expected)


def test_tile_scatter_masked_2d(test, device):
    """tile_scatter_masked works with a 2-D shared tile."""
    ROWS = 8
    COLS = 8
    BLOCK_DIM = ROWS * COLS

    @wp.kernel(enable_backward=False, module="unique")
    def compute(out: wp.array2d(dtype=int)):
        _tile, idx = wp.tid()
        row = idx // COLS
        col = idx % COLS
        t = wp.tile_zeros(shape=(ROWS, COLS), dtype=int, storage="shared")
        wp.tile_scatter_masked(t, row, col, idx + 1, True)
        out[row, col] = wp.tile_extract(t, row, col)

    out = wp.zeros((ROWS, COLS), dtype=int, device=device)
    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)

    expected = np.arange(1, BLOCK_DIM + 1, dtype=np.int32).reshape(ROWS, COLS)
    np.testing.assert_array_equal(out.numpy(), expected)


def test_tile_scatter_masked_3d(test, device):
    """tile_scatter_masked works with a 3-D shared tile."""
    D0 = 4
    D1 = 4
    D2 = 4
    BLOCK_DIM = D0 * D1 * D2

    @wp.kernel(enable_backward=False, module="unique")
    def compute(out: wp.array3d(dtype=int)):
        _tile, idx = wp.tid()
        i = idx // (D1 * D2)
        j = (idx // D2) % D1
        k = idx % D2
        t = wp.tile_zeros(shape=(D0, D1, D2), dtype=int, storage="shared")
        wp.tile_scatter_masked(t, i, j, k, idx + 1, True)
        out[i, j, k] = wp.tile_extract(t, i, j, k)

    out = wp.zeros((D0, D1, D2), dtype=int, device=device)
    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)

    expected = np.arange(1, BLOCK_DIM + 1, dtype=np.int32).reshape(D0, D1, D2)
    np.testing.assert_array_equal(out.numpy(), expected)


def test_tile_scatter_masked_4d(test, device):
    """tile_scatter_masked works with a 4-D shared tile."""
    D0 = 2
    D1 = 2
    D2 = 2
    D3 = 4
    BLOCK_DIM = D0 * D1 * D2 * D3

    @wp.kernel(enable_backward=False, module="unique")
    def compute(out: wp.array4d(dtype=int)):
        _tile, idx = wp.tid()
        i = idx // (D1 * D2 * D3)
        j = (idx // (D2 * D3)) % D1
        k = (idx // D3) % D2
        l = idx % D3
        t = wp.tile_zeros(shape=(D0, D1, D2, D3), dtype=int, storage="shared")
        wp.tile_scatter_masked(t, i, j, k, l, idx + 1, True)
        out[i, j, k, l] = wp.tile_extract(t, i, j, k, l)

    out = wp.zeros((D0, D1, D2, D3), dtype=int, device=device)
    wp.launch_tiled(compute, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)

    expected = np.arange(1, BLOCK_DIM + 1, dtype=np.int32).reshape(D0, D1, D2, D3)
    np.testing.assert_array_equal(out.numpy(), expected)


def test_tile_scatter_masked_grad_basic(test, device):
    """Gradient flows through tile_scatter_masked: output = input * 2 via shared tile."""
    TILE_SIZE = 64

    @wp.kernel(module="unique")
    def compute(inp: wp.array(dtype=float), out: wp.array(dtype=float)):
        _tile, i = wp.tid()
        val = inp[i] * 2.0
        t = wp.tile_zeros(shape=TILE_SIZE, dtype=float, storage="shared")
        wp.tile_scatter_masked(t, i, val, True)
        out[i] = wp.tile_extract(t, i)

    inp = wp.array(np.ones(TILE_SIZE, dtype=np.float32), requires_grad=True, device=device)
    out = wp.zeros(TILE_SIZE, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=TILE_SIZE, device=device)

    out.grad = wp.ones_like(out, device=device)
    tape.backward()

    np.testing.assert_allclose(out.numpy(), np.full(TILE_SIZE, 2.0))
    np.testing.assert_allclose(inp.grad.numpy(), np.full(TILE_SIZE, 2.0))


def test_tile_scatter_masked_grad_partial(test, device):
    """has_value gates the adjoint: only writing threads receive gradients."""
    TILE_SIZE = 64

    @wp.kernel(module="unique")
    def compute(inp: wp.array(dtype=float), out: wp.array(dtype=float)):
        _tile, i = wp.tid()
        val = inp[i] * 2.0
        t = wp.tile_zeros(shape=TILE_SIZE, dtype=float, storage="shared")
        wp.tile_scatter_masked(t, i, val, (i % 2) == 0)
        out[i] = wp.tile_extract(t, i)

    inp = wp.array(np.ones(TILE_SIZE, dtype=np.float32), requires_grad=True, device=device)
    out = wp.zeros(TILE_SIZE, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=TILE_SIZE, device=device)

    out.grad = wp.ones_like(out, device=device)
    tape.backward()

    expected_grad = np.zeros(TILE_SIZE, dtype=np.float32)
    expected_grad[0::2] = 2.0
    np.testing.assert_allclose(inp.grad.numpy(), expected_grad)


def test_tile_scatter_masked_grad_cross_thread(test, device):
    """Gradient flows correctly when threads read each other's slots."""
    TILE_SIZE = 64

    @wp.kernel(module="unique")
    def compute(inp: wp.array(dtype=float), out: wp.array(dtype=float)):
        _tile, i = wp.tid()
        val = inp[i] * float(i + 1)
        t = wp.tile_zeros(shape=TILE_SIZE, dtype=float, storage="shared")
        wp.tile_scatter_masked(t, i, val, True)
        neighbor = (i + 1) % TILE_SIZE
        out[i] = wp.tile_extract(t, neighbor)

    inp = wp.array(np.ones(TILE_SIZE, dtype=np.float32), requires_grad=True, device=device)
    out = wp.zeros(TILE_SIZE, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=TILE_SIZE, device=device)

    out.grad = wp.ones_like(out, device=device)
    tape.backward()

    expected_fwd = np.array([((i + 1) % TILE_SIZE + 1) for i in range(TILE_SIZE)], dtype=np.float32)
    np.testing.assert_allclose(out.numpy(), expected_fwd)

    expected_grad = np.arange(1, TILE_SIZE + 1, dtype=np.float32)
    np.testing.assert_allclose(inp.grad.numpy(), expected_grad)


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
add_function_test(TestTileSharedMemory, "test_tile_scatter_add_basic", test_tile_scatter_add_basic, devices=devices)
add_function_test(
    TestTileSharedMemory, "test_tile_scatter_add_conflicting", test_tile_scatter_add_conflicting, devices=devices
)
add_function_test(TestTileSharedMemory, "test_tile_scatter_add_partial", test_tile_scatter_add_partial, devices=devices)
add_function_test(TestTileSharedMemory, "test_tile_scatter_add_2d", test_tile_scatter_add_2d, devices=devices)
add_function_test(
    TestTileSharedMemory, "test_tile_scatter_add_grad_basic", test_tile_scatter_add_grad_basic, devices=devices
)
add_function_test(
    TestTileSharedMemory, "test_tile_scatter_add_grad_partial", test_tile_scatter_add_grad_partial, devices=devices
)
add_function_test(
    TestTileSharedMemory,
    "test_tile_scatter_add_grad_conflicting",
    test_tile_scatter_add_grad_conflicting,
    devices=devices,
)
add_function_test(
    TestTileSharedMemory,
    "test_tile_scatter_add_non_atomic_1d",
    test_tile_scatter_add_non_atomic_1d,
    devices=devices,
)
add_function_test(
    TestTileSharedMemory,
    "test_tile_scatter_add_non_atomic_2d",
    test_tile_scatter_add_non_atomic_2d,
    devices=devices,
)
add_function_test(
    TestTileSharedMemory,
    "test_tile_scatter_add_non_atomic_grad",
    test_tile_scatter_add_non_atomic_grad,
    devices=devices,
)
add_function_test(
    TestTileSharedMemory,
    "test_tile_shared_coalesced_mat33",
    test_tile_shared_coalesced_mat33,
    devices=devices,
)
add_function_test(
    TestTileSharedMemory,
    "test_tile_shared_coalesced_mat44",
    test_tile_shared_coalesced_mat44,
    devices=devices,
)
add_function_test(
    TestTileSharedMemory, "test_tile_scatter_masked_basic", test_tile_scatter_masked_basic, devices=devices
)
add_function_test(
    TestTileSharedMemory, "test_tile_scatter_masked_partial", test_tile_scatter_masked_partial, devices=devices
)
add_function_test(
    TestTileSharedMemory,
    "test_tile_scatter_masked_cross_thread",
    test_tile_scatter_masked_cross_thread,
    devices=devices,
)
add_function_test(TestTileSharedMemory, "test_tile_scatter_masked_2d", test_tile_scatter_masked_2d, devices=devices)
add_function_test(TestTileSharedMemory, "test_tile_scatter_masked_3d", test_tile_scatter_masked_3d, devices=devices)
add_function_test(TestTileSharedMemory, "test_tile_scatter_masked_4d", test_tile_scatter_masked_4d, devices=devices)
add_function_test(
    TestTileSharedMemory, "test_tile_scatter_masked_grad_basic", test_tile_scatter_masked_grad_basic, devices=devices
)
add_function_test(
    TestTileSharedMemory,
    "test_tile_scatter_masked_grad_partial",
    test_tile_scatter_masked_grad_partial,
    devices=devices,
)
add_function_test(
    TestTileSharedMemory,
    "test_tile_scatter_masked_grad_cross_thread",
    test_tile_scatter_masked_grad_cross_thread,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
