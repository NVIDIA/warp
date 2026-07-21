# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests indexed tile memory operations with 32-thread blocks.

Covers indexed loads, indexed stores, indexed atomic adds, out-of-bounds
indices, and non-owning index tiles. The module includes forward and backward
tests. Add tests here when indexed tile addressing is the behavior under test.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_M = wp.constant(16)
TILE_N = wp.constant(8)

HALF_M = wp.constant(TILE_M // 2)
HALF_N = wp.constant(TILE_N // 2)
TWO_M = wp.constant(TILE_M * 2)
TWO_N = wp.constant(TILE_N * 2)

OOB_ROWS = wp.constant(4)
OOB_COLS = wp.constant(2)


@wp.kernel
def tile_load_indexed(x: wp.array2d[float], y: wp.array2d[float], z: wp.array2d[float]):
    i, j = wp.tid()

    evens_M = wp.tile_arange(HALF_M, dtype=int, storage="shared") * 2
    t0 = wp.tile_load_indexed(
        x, indices=evens_M, shape=(HALF_M, TILE_N), offset=(i * TILE_M, j * TILE_N), axis=0, storage="register"
    )
    wp.tile_store(y, t0, offset=(i * HALF_M, j * TILE_N))

    evens_N = wp.tile_arange(HALF_N, dtype=int, storage="shared") * 2
    t1 = wp.tile_load_indexed(
        x, indices=evens_N, shape=(TILE_M, HALF_N), offset=(i * TILE_M, j * TILE_N), axis=1, storage="shared"
    )
    wp.tile_store(z, t1, offset=(i * TILE_M, j * HALF_N))


def test_tile_load_indexed(test, device):
    M = TILE_M * 2
    N = TILE_N * 2

    arr = np.arange(M * N, dtype=float).reshape(M, N)

    x = wp.array(arr, dtype=float, requires_grad=True, device=device)
    y = wp.zeros((M // 2, N), dtype=float, requires_grad=True, device=device)
    z = wp.zeros((M, N // 2), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_load_indexed, dim=[2, 2], inputs=[x], outputs=[y, z], block_dim=32, device=device)

    y.grad = wp.ones_like(y)
    z.grad = wp.ones_like(z)

    tape.backward()

    x_grad_np = np.ones(arr.shape, dtype=float)
    x_grad_np[0::2, 0::2] += 1
    x_grad_np[1::2, 1::2] -= 1

    assert_np_equal(y.numpy(), arr[np.arange(0, arr.shape[0], 2, dtype=int)])
    assert_np_equal(z.numpy(), arr[:, np.arange(0, arr.shape[1], 2, dtype=int)])
    assert_np_equal(x.grad.numpy(), x_grad_np)


@wp.kernel(enable_backward=False)
def tile_load_indexed_non_owning(a: wp.array[float], b: wp.array[float]):
    indices = wp.tile_arange(HALF_M, dtype=int, storage="shared") * 2
    indices_non_owning = wp.tile_reshape(indices, shape=(HALF_M,))

    t = wp.tile_load_indexed(a, indices=indices_non_owning, shape=(HALF_M,), axis=0, storage="shared")
    wp.tile_store(b, t)


def test_tile_load_indexed_non_owning(test, device):
    a_np = np.arange(TILE_M)

    a = wp.array(a_np, dtype=float, device=device)
    b = wp.empty(shape=(HALF_M,), dtype=float, device=device)

    wp.launch_tiled(tile_load_indexed_non_owning, dim=1, inputs=[a], outputs=[b], block_dim=32, device=device)

    assert_np_equal(b.numpy(), a_np[::2])


@wp.kernel
def tile_load_indexed_oob_kernel(src: wp.array2d[float], idx: wp.array1d[int], out: wp.array2d[float]):
    indices = wp.tile_load(idx, shape=(OOB_ROWS,), storage="shared")
    t = wp.tile_load_indexed(src, indices=indices, shape=(OOB_ROWS, OOB_COLS), axis=0, storage="register")
    wp.tile_store(out, t)


def test_tile_load_indexed_oob(test, device):
    """A gather index outside the axis (negative or >= its length) predicates the element to zero
    on both the forward and backward passes, instead of reading/writing out of bounds."""
    # src is a view onto rows 1: of buf, so src's row -1 aliases buf[0]. Seeding buf[0] with a
    # sentinel makes an out-of-bounds negative-index read observable (999, not 0) rather than
    # reading undefined memory before the array.
    buf = np.zeros((OOB_ROWS + 1, OOB_COLS), dtype=np.float32)
    buf[0] = 999.0
    buf[1:] = np.arange(1, OOB_ROWS * OOB_COLS + 1, dtype=np.float32).reshape(OOB_ROWS, OOB_COLS)

    src = wp.array(buf, dtype=float, requires_grad=True, device=device)
    src_view = src[1:]  # src_view[k] == buf[k + 1]
    idx = wp.array([0, -1, 2, OOB_ROWS], dtype=int, device=device)  # -1 and OOB_ROWS are out of bounds
    out = wp.zeros((OOB_ROWS, OOB_COLS), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_load_indexed_oob_kernel, dim=1, inputs=[src_view, idx], outputs=[out], block_dim=32, device=device
        )

    expected = np.zeros((OOB_ROWS, OOB_COLS), dtype=np.float32)
    expected[0] = buf[1]  # idx 0 -> src_view[0]
    expected[2] = buf[3]  # idx 2 -> src_view[2]
    # idx -1 (below row 0) and idx OOB_ROWS (past the last row) both predicate to zero
    assert_np_equal(out.numpy(), expected)

    out.grad = wp.ones_like(out)
    tape.backward()

    # Backward shares the same bounds check: only in-bounds rows accumulate gradient, and the
    # sentinel row before src_view[0] is never written.
    expected_grad = np.zeros((OOB_ROWS + 1, OOB_COLS), dtype=np.float32)
    expected_grad[1] = 1.0  # src_view[0] gathered by idx 0
    expected_grad[3] = 1.0  # src_view[2] gathered by idx 2
    assert_np_equal(src.grad.numpy(), expected_grad)


@wp.func
def add_one(x: int):
    return x + 1


@wp.kernel
def tile_store_indexed(x: wp.array2d[float], y: wp.array2d[float], z: wp.array2d[float]):
    i, j = wp.tid()

    t = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N), storage="register")

    evens_M = wp.tile_arange(TILE_M, dtype=int, storage="shared") * 2
    odds_M = wp.tile_map(add_one, evens_M)

    wp.tile_store_indexed(y, indices=odds_M, t=t, offset=(i * TWO_M, j * TILE_N), axis=0)

    evens_N = wp.tile_arange(TILE_N, dtype=int, storage="shared") * 2
    odds_N = wp.tile_map(add_one, evens_N)

    wp.tile_store_indexed(z, indices=odds_N, t=t, offset=(i * TILE_M, j * TWO_N), axis=1)


def test_tile_store_indexed(test, device):
    M = TILE_M * 2
    N = TILE_N * 2

    arr = np.arange(M * N, dtype=float).reshape(M, N)

    x = wp.array(arr, dtype=float, requires_grad=True, device=device)
    y = wp.zeros((M * 2, N), dtype=float, requires_grad=True, device=device)
    z = wp.zeros((M, N * 2), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_store_indexed, dim=[2, 2], inputs=[x], outputs=[y, z], block_dim=32, device=device)

    y.grad = wp.ones_like(y)
    z.grad = wp.ones_like(z)

    tape.backward()

    y_np = np.zeros((M * 2, N))
    y_np[1::2, :] = arr

    z_np = np.zeros((M, N * 2))
    z_np[:, 1::2] = arr

    x_grad_np = np.ones((M, N)) * 2

    assert_np_equal(y.numpy(), y_np)
    assert_np_equal(z.numpy(), z_np)
    assert_np_equal(x.grad.numpy(), x_grad_np)


@wp.kernel
def tile_atomic_add_indexed(x: wp.array2d[float], y: wp.array2d[float]):
    i, j = wp.tid()

    t = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N), storage="register")

    ones = wp.tile_ones(TILE_M, dtype=int, storage="shared")

    wp.tile_atomic_add_indexed(y, indices=ones, t=t, offset=(i * TILE_M, j * TILE_N), axis=0)


def test_tile_atomic_add_indexed(test, device):
    M = TILE_M * 2
    N = TILE_N * 2

    arr = np.arange(M * N, dtype=float).reshape(M, N)

    x = wp.array(arr, dtype=float, requires_grad=True, device=device)
    y = wp.zeros((M, N), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_atomic_add_indexed, dim=[2, 2], inputs=[x], outputs=[y], block_dim=32, device=device)

    y.grad = wp.ones_like(y)

    tape.backward()

    y_np = np.zeros((M, N), dtype=float)
    y_np[1] = np.sum(arr[0:TILE_M], axis=0)
    y_np[TILE_M + 1] = np.sum(arr[TILE_M:], axis=0)

    x_grad_np = np.ones((M, N))

    assert_np_equal(y.numpy(), y_np)
    assert_np_equal(x.grad.numpy(), x_grad_np)


devices = get_test_devices()


class TestTileLoadIndexed(unittest.TestCase):
    pass


add_function_test(TestTileLoadIndexed, "test_tile_load_indexed", test_tile_load_indexed, devices=devices)
add_function_test(
    TestTileLoadIndexed,
    "test_tile_load_indexed_non_owning",
    test_tile_load_indexed_non_owning,
    devices=devices,
)
add_function_test(
    TestTileLoadIndexed,
    "test_tile_load_indexed_oob",
    test_tile_load_indexed_oob,
    devices=devices,
)
add_function_test(TestTileLoadIndexed, "test_tile_store_indexed", test_tile_store_indexed, devices=devices)
add_function_test(
    TestTileLoadIndexed,
    "test_tile_atomic_add_indexed",
    test_tile_atomic_add_indexed,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
