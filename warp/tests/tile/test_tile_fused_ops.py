# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for fused tile operations: tile_axpy and tile_dot."""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

BLOCK_DIM = 64
TILE_M = 4
TILE_N = 4


# ---- tile_axpy forward tests ----


def test_tile_axpy_shared_register(test, device):
    """Fused dest += src * alpha with shared dest, register src."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        dest_in: wp.array2d[float],
        src_in: wp.array2d[float],
        alpha: float,
        out: wp.array2d[float],
    ):
        i = wp.tid()
        dest = wp.tile_load(dest_in, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        src = wp.tile_load(src_in, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        wp.tile_axpy(alpha, src, dest)
        wp.tile_store(out, dest)

    dest_np = np.ones((TILE_M, TILE_N), dtype=np.float32) * 2.0
    src_np = np.ones((TILE_M, TILE_N), dtype=np.float32) * 3.0
    alpha = 5.0

    dest_in = wp.array(dest_np, device=device)
    src_in = wp.array(src_np, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[dest_in, src_in, alpha, out], block_dim=BLOCK_DIM, device=device)

    expected = dest_np + src_np * alpha  # 2 + 3*5 = 17
    assert_np_equal(out.numpy(), expected)


def test_tile_axpy_shared_shared(test, device):
    """Fused dest += src * alpha with both tiles in shared memory."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        dest_in: wp.array2d[float],
        src_in: wp.array2d[float],
        alpha: float,
        out: wp.array2d[float],
    ):
        i = wp.tid()
        dest = wp.tile_load(dest_in, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        src = wp.tile_load(src_in, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        wp.tile_axpy(alpha, src, dest)
        wp.tile_store(out, dest)

    dest_np = np.full((TILE_M, TILE_N), 1.0, dtype=np.float32)
    src_np = np.full((TILE_M, TILE_N), 4.0, dtype=np.float32)
    alpha = -2.0

    dest_in = wp.array(dest_np, device=device)
    src_in = wp.array(src_np, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[dest_in, src_in, alpha, out], block_dim=BLOCK_DIM, device=device)

    expected = dest_np + src_np * alpha  # 1 + 4*(-2) = -7
    assert_np_equal(out.numpy(), expected)


def test_tile_axpy_1d(test, device):
    """tile_axpy with 1-D tiles."""
    N = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        dest_in: wp.array[float],
        src_in: wp.array[float],
        alpha: float,
        out: wp.array[float],
    ):
        i = wp.tid()
        dest = wp.tile_load(dest_in, shape=N, offset=0, storage="shared")
        src = wp.tile_load(src_in, shape=N, offset=0, storage="register")
        wp.tile_axpy(alpha, src, dest)
        wp.tile_store(out, dest, offset=0)

    dest_np = np.arange(N, dtype=np.float32)
    src_np = np.ones(N, dtype=np.float32) * 2.0
    alpha = 3.0

    dest_in = wp.array(dest_np, device=device)
    src_in = wp.array(src_np, device=device)
    out = wp.zeros(N, dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[dest_in, src_in, alpha, out], block_dim=BLOCK_DIM, device=device)

    expected = dest_np + src_np * alpha
    assert_np_equal(out.numpy(), expected)


def test_tile_axpy_register_register(test, device):
    """Fused dest += src * alpha with both tiles in register storage."""
    N = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        dest_in: wp.array[float],
        src_in: wp.array[float],
        alpha: float,
        out: wp.array[float],
    ):
        i = wp.tid()
        dest = wp.tile_load(dest_in, shape=N, offset=0, storage="register")
        src = wp.tile_load(src_in, shape=N, offset=0, storage="register")
        wp.tile_axpy(alpha, src, dest)
        wp.tile_store(out, dest, offset=0)

    dest_np = np.arange(N, dtype=np.float32) * 0.5
    src_np = np.ones(N, dtype=np.float32) * 3.0
    alpha = 2.0

    dest_in = wp.array(dest_np, device=device)
    src_in = wp.array(src_np, device=device)
    out = wp.zeros(N, dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[dest_in, src_in, alpha, out], block_dim=BLOCK_DIM, device=device)

    expected = dest_np + src_np * alpha
    assert_np_equal(out.numpy(), expected)


def test_tile_axpy_zero_alpha(test, device):
    """tile_axpy with alpha=0 should leave dest unchanged."""
    N = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        dest_in: wp.array[float],
        src_in: wp.array[float],
        alpha: float,
        out: wp.array[float],
    ):
        i = wp.tid()
        dest = wp.tile_load(dest_in, shape=N, offset=0, storage="shared")
        src = wp.tile_load(src_in, shape=N, offset=0, storage="register")
        wp.tile_axpy(alpha, src, dest)
        wp.tile_store(out, dest, offset=0)

    dest_np = np.arange(N, dtype=np.float32)
    src_np = np.ones(N, dtype=np.float32) * 999.0
    alpha = 0.0

    dest_in = wp.array(dest_np, device=device)
    src_in = wp.array(src_np, device=device)
    out = wp.zeros(N, dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[dest_in, src_in, alpha, out], block_dim=BLOCK_DIM, device=device)

    assert_np_equal(out.numpy(), dest_np)


# ---- tile_axpy gradient tests ----


def test_tile_axpy_grad(test, device):
    """Gradient flows through tile_axpy for src and alpha."""
    N = 16

    @wp.kernel(module="unique")
    def compute(
        dest_in: wp.array[float],
        src_in: wp.array[float],
        alpha_in: wp.array[float],
        out: wp.array[float],
    ):
        i = wp.tid()
        dest = wp.tile_load(dest_in, shape=N, offset=0, storage="shared")
        src = wp.tile_load(src_in, shape=N, offset=0, storage="register")

        alpha_tile = wp.tile_load(alpha_in, shape=1, offset=0, storage="shared")
        alpha = wp.tile_extract(alpha_tile, 0)

        wp.tile_axpy(alpha, src, dest)
        wp.tile_store(out, dest, offset=0)

    dest_np = np.ones(N, dtype=np.float32) * 2.0
    src_np = np.ones(N, dtype=np.float32) * 3.0
    alpha_val = 5.0

    dest_in = wp.array(dest_np, requires_grad=True, device=device)
    src_in = wp.array(src_np, requires_grad=True, device=device)
    alpha_in = wp.array([alpha_val], dtype=float, requires_grad=True, device=device)
    out = wp.zeros(N, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[dest_in, src_in, alpha_in, out], block_dim=BLOCK_DIM, device=device)

    out.grad = wp.ones_like(out, device=device)
    tape.backward()

    # Forward: out = dest + src * alpha = 2 + 3*5 = 17
    assert_np_equal(out.numpy(), np.full(N, 17.0))

    # d(out)/d(dest) = 1
    assert_np_equal(dest_in.grad.numpy(), np.ones(N, dtype=np.float32))

    # d(out)/d(src) = alpha = 5
    assert_np_equal(src_in.grad.numpy(), np.full(N, alpha_val, dtype=np.float32))

    # d(out)/d(alpha) = sum(src) = 3*16 = 48
    assert_np_equal(alpha_in.grad.numpy(), np.array([src_np.sum()]))


# ---- tile_dot forward tests ----


def test_tile_dot_basic(test, device):
    """Basic dot product of two tiles returns correct scalar."""
    N = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        a_in: wp.array[float],
        b_in: wp.array[float],
        out: wp.array[float],
    ):
        i = wp.tid()
        a = wp.tile_load(a_in, shape=N, offset=0, storage="register")
        b = wp.tile_load(b_in, shape=N, offset=0, storage="shared")
        result = wp.tile_dot(a, b)
        if i == 0:
            out[0] = result

    a_np = np.ones(N, dtype=np.float32) * 2.0
    b_np = np.ones(N, dtype=np.float32) * 3.0

    a_in = wp.array(a_np, device=device)
    b_in = wp.array(b_np, device=device)
    out = wp.zeros(1, dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[a_in, b_in, out], block_dim=BLOCK_DIM, device=device)

    expected = np.dot(a_np, b_np)  # 2*3*64 = 384
    test.assertAlmostEqual(out.numpy()[0], expected, places=4)


def test_tile_dot_nonuniform(test, device):
    """Dot product with non-uniform values."""
    N = 32

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        a_in: wp.array[float],
        b_in: wp.array[float],
        out: wp.array[float],
    ):
        i = wp.tid()
        a = wp.tile_load(a_in, shape=N, offset=0, storage="register")
        b = wp.tile_load(b_in, shape=N, offset=0, storage="register")
        result = wp.tile_dot(a, b)
        if i == 0:
            out[0] = result

    a_np = np.arange(N, dtype=np.float32)
    b_np = np.arange(N, dtype=np.float32) * 0.5

    a_in = wp.array(a_np, device=device)
    b_in = wp.array(b_np, device=device)
    out = wp.zeros(1, dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[a_in, b_in, out], block_dim=BLOCK_DIM, device=device)

    expected = np.dot(a_np, b_np)
    test.assertAlmostEqual(out.numpy()[0], expected, places=4)


def test_tile_dot_2d(test, device):
    """Dot product of 2-D tiles (flattened element-wise)."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        a_in: wp.array2d[float],
        b_in: wp.array2d[float],
        out: wp.array[float],
    ):
        i = wp.tid()
        a = wp.tile_load(a_in, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        b = wp.tile_load(b_in, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        result = wp.tile_dot(a, b)
        if i == 0:
            out[0] = result

    a_np = np.ones((TILE_M, TILE_N), dtype=np.float32) * 2.0
    b_np = np.ones((TILE_M, TILE_N), dtype=np.float32) * 3.0

    a_in = wp.array(a_np, device=device)
    b_in = wp.array(b_np, device=device)
    out = wp.zeros(1, dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[a_in, b_in, out], block_dim=BLOCK_DIM, device=device)

    expected = np.sum(a_np * b_np)  # 2*3*16 = 96
    test.assertAlmostEqual(out.numpy()[0], expected, places=4)


def test_tile_dot_shared_shared(test, device):
    """Dot product with both tiles in shared storage."""
    N = 64

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        a_in: wp.array[float],
        b_in: wp.array[float],
        out: wp.array[float],
    ):
        i = wp.tid()
        a = wp.tile_load(a_in, shape=N, offset=0, storage="shared")
        b = wp.tile_load(b_in, shape=N, offset=0, storage="shared")
        result = wp.tile_dot(a, b)
        if i == 0:
            out[0] = result

    a_np = np.arange(N, dtype=np.float32) - 32.0  # mixed positive/negative
    b_np = np.ones(N, dtype=np.float32) * 2.0

    a_in = wp.array(a_np, device=device)
    b_in = wp.array(b_np, device=device)
    out = wp.zeros(1, dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[a_in, b_in, out], block_dim=BLOCK_DIM, device=device)

    expected = np.dot(a_np, b_np)
    test.assertAlmostEqual(out.numpy()[0], expected, places=4)


# ---- tile_dot gradient tests ----


def test_tile_dot_grad_shared(test, device):
    """Gradient flows through tile_dot with shared-storage operands."""
    N = 16

    @wp.kernel(module="unique")
    def compute(
        a_in: wp.array[float],
        b_in: wp.array[float],
        out: wp.array[float],
    ):
        i = wp.tid()
        a = wp.tile_load(a_in, shape=N, offset=0, storage="shared")
        b = wp.tile_load(b_in, shape=N, offset=0, storage="shared")
        result = wp.tile_dot(a, b)
        if i == 0:
            out[0] = result

    a_np = np.arange(1, N + 1, dtype=np.float32)
    b_np = np.arange(N, 0, -1, dtype=np.float32)

    a_in = wp.array(a_np, requires_grad=True, device=device)
    b_in = wp.array(b_np, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[a_in, b_in, out], block_dim=BLOCK_DIM, device=device)

    out.grad = wp.ones_like(out, device=device)
    tape.backward()

    expected_fwd = np.dot(a_np, b_np)
    test.assertAlmostEqual(out.numpy()[0], expected_fwd, places=4)

    assert_np_equal(a_in.grad.numpy(), b_np, tol=1e-4)
    assert_np_equal(b_in.grad.numpy(), a_np, tol=1e-4)


def test_tile_dot_grad(test, device):
    """Gradient flows correctly through tile_dot."""
    N = 16

    @wp.kernel(module="unique")
    def compute(
        a_in: wp.array[float],
        b_in: wp.array[float],
        out: wp.array[float],
    ):
        i = wp.tid()
        a = wp.tile_load(a_in, shape=N, offset=0, storage="register")
        b = wp.tile_load(b_in, shape=N, offset=0, storage="register")
        result = wp.tile_dot(a, b)
        if i == 0:
            out[0] = result

    a_np = np.arange(1, N + 1, dtype=np.float32)
    b_np = np.arange(N, 0, -1, dtype=np.float32)

    a_in = wp.array(a_np, requires_grad=True, device=device)
    b_in = wp.array(b_np, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[a_in, b_in, out], block_dim=BLOCK_DIM, device=device)

    out.grad = wp.ones_like(out, device=device)
    tape.backward()

    # Forward: dot(a, b)
    expected_fwd = np.dot(a_np, b_np)
    test.assertAlmostEqual(out.numpy()[0], expected_fwd, places=4)

    # d(dot)/d(a) = b, d(dot)/d(b) = a
    assert_np_equal(a_in.grad.numpy(), b_np, tol=1e-4)
    assert_np_equal(b_in.grad.numpy(), a_np, tol=1e-4)


def test_tile_dot_vec3(test, device):
    """tile_dot on vec3 tiles returns a scalar (full contraction via tensordot)."""
    N = 16

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        a_in: wp.array[wp.vec3],
        b_in: wp.array[wp.vec3],
        out: wp.array[float],
    ):
        i = wp.tid()
        a = wp.tile_load(a_in, shape=N, offset=0, storage="register")
        b = wp.tile_load(b_in, shape=N, offset=0, storage="shared")
        result = wp.tile_dot(a, b)
        if i == 0:
            out[0] = result

    a_np = np.arange(N * 3, dtype=np.float32).reshape(N, 3)
    b_np = np.ones((N, 3), dtype=np.float32) * 2.0

    a_in = wp.array(a_np, dtype=wp.vec3, device=device)
    b_in = wp.array(b_np, dtype=wp.vec3, device=device)
    out = wp.zeros(1, dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[a_in, b_in, out], block_dim=BLOCK_DIM, device=device)

    # Full contraction: sum of all element-wise products
    expected = np.sum(a_np * b_np)
    test.assertAlmostEqual(out.numpy()[0], expected, places=2)


def test_tile_dot_mat33(test, device):
    """tile_dot on mat33 tiles returns a scalar (full contraction via tensordot)."""
    N = 8

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        a_in: wp.array[wp.mat33],
        b_in: wp.array[wp.mat33],
        out: wp.array[float],
    ):
        i = wp.tid()
        a = wp.tile_load(a_in, shape=N, offset=0, storage="register")
        b = wp.tile_load(b_in, shape=N, offset=0, storage="shared")
        result = wp.tile_dot(a, b)
        if i == 0:
            out[0] = result

    a_np = np.arange(N * 9, dtype=np.float32).reshape(N, 3, 3)
    b_np = np.ones((N, 3, 3), dtype=np.float32) * 0.5

    a_in = wp.array(a_np, dtype=wp.mat33, device=device)
    b_in = wp.array(b_np, dtype=wp.mat33, device=device)
    out = wp.zeros(1, dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[a_in, b_in, out], block_dim=BLOCK_DIM, device=device)

    # Full contraction: sum of all element-wise products
    expected = np.sum(a_np * b_np)
    test.assertAlmostEqual(out.numpy()[0], expected, places=1)


def test_tile_axpy_vec3(test, device):
    """tile_axpy with vec3 tiles and scalar alpha."""
    N = 16

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        dest_in: wp.array[wp.vec3],
        src_in: wp.array[wp.vec3],
        alpha: float,
        out: wp.array[wp.vec3],
    ):
        i = wp.tid()
        dest = wp.tile_load(dest_in, shape=N, offset=0, storage="shared")
        src = wp.tile_load(src_in, shape=N, offset=0, storage="register")
        wp.tile_axpy(alpha, src, dest)
        wp.tile_store(out, dest, offset=0)

    dest_np = np.ones((N, 3), dtype=np.float32) * 2.0
    src_np = np.arange(N * 3, dtype=np.float32).reshape(N, 3)
    alpha = 3.0

    dest_in = wp.array(dest_np, dtype=wp.vec3, device=device)
    src_in = wp.array(src_np, dtype=wp.vec3, device=device)
    out = wp.zeros(N, dtype=wp.vec3, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[dest_in, src_in, alpha, out], block_dim=BLOCK_DIM, device=device)

    expected = dest_np + src_np * alpha
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)


def test_tile_axpy_broadcast_dest(test, device):
    """tile_axpy onto a broadcast shared tile (non-unique layout) uses atomics."""
    ROWS = 4
    COLS = 4

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        bias_in: wp.array[float],
        src_in: wp.array2d[float],
        alpha: float,
        out: wp.array2d[float],
    ):
        i = wp.tid()
        bias = wp.tile_load(bias_in, shape=COLS, offset=0, storage="shared")
        dest = wp.tile_broadcast(bias, shape=(ROWS, COLS))
        src = wp.tile_load(src_in, shape=(ROWS, COLS), offset=(0, 0), storage="register")
        wp.tile_axpy(alpha, src, dest)
        wp.tile_store(out, dest)

    bias_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    src_np = np.ones((ROWS, COLS), dtype=np.float32)
    alpha = 1.0

    bias_in = wp.array(bias_np, device=device)
    src_in = wp.array(src_np, device=device)
    out = wp.zeros((ROWS, COLS), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[bias_in, src_in, alpha, out], block_dim=BLOCK_DIM, device=device)

    # Each column of the broadcast tile maps to the same shared memory location,
    # so axpy accumulates ROWS contributions per element: bias + ROWS * alpha * 1.0
    expected = np.broadcast_to(bias_np + ROWS * alpha, (ROWS, COLS))
    assert_np_equal(out.numpy(), expected)


devices = get_test_devices()


class TestTileFusedOps(unittest.TestCase):
    pass


add_function_test(TestTileFusedOps, "test_tile_axpy_broadcast_dest", test_tile_axpy_broadcast_dest, devices=devices)
add_function_test(TestTileFusedOps, "test_tile_axpy_shared_register", test_tile_axpy_shared_register, devices=devices)
add_function_test(TestTileFusedOps, "test_tile_axpy_shared_shared", test_tile_axpy_shared_shared, devices=devices)
add_function_test(
    TestTileFusedOps, "test_tile_axpy_register_register", test_tile_axpy_register_register, devices=devices
)
add_function_test(TestTileFusedOps, "test_tile_axpy_zero_alpha", test_tile_axpy_zero_alpha, devices=devices)
add_function_test(TestTileFusedOps, "test_tile_axpy_1d", test_tile_axpy_1d, devices=devices)
add_function_test(TestTileFusedOps, "test_tile_axpy_grad", test_tile_axpy_grad, devices=devices)
add_function_test(TestTileFusedOps, "test_tile_dot_basic", test_tile_dot_basic, devices=devices)
add_function_test(TestTileFusedOps, "test_tile_dot_nonuniform", test_tile_dot_nonuniform, devices=devices)
add_function_test(TestTileFusedOps, "test_tile_dot_shared_shared", test_tile_dot_shared_shared, devices=devices)
add_function_test(TestTileFusedOps, "test_tile_dot_2d", test_tile_dot_2d, devices=devices)
add_function_test(TestTileFusedOps, "test_tile_dot_grad", test_tile_dot_grad, devices=devices)
add_function_test(TestTileFusedOps, "test_tile_dot_grad_shared", test_tile_dot_grad_shared, devices=devices)
add_function_test(TestTileFusedOps, "test_tile_dot_vec3", test_tile_dot_vec3, devices=devices)
add_function_test(TestTileFusedOps, "test_tile_dot_mat33", test_tile_dot_mat33, devices=devices)
add_function_test(TestTileFusedOps, "test_tile_axpy_vec3", test_tile_axpy_vec3, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
