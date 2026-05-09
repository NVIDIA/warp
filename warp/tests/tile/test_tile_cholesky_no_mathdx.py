# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile Cholesky tests with mathdx solver disabled (cooperative scalar fallback path).

Setting ``enable_mathdx_solver=False`` at module scope routes
``tile_cholesky`` and ``tile_cholesky_inplace`` (and after Stage 4, the
Cholesky adjoint) through the cooperative scalar implementation in
``tile_cholesky.h`` on GPU (or the CPU sequential branch on CPU),
exercising the path that runs whenever Warp is built without libmathdx or
when a user disables the option per-module.

Mirrors ``test_tile_matmul_no_mathdx.py`` / ``test_tile_solve_no_mathdx.py``.

This file covers the cholesky-factorization forward path only. Adjoint
coverage is added in the cooperative-adjoint commit.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# Disable mathdx solver ops (Cholesky and triangular solves) for all kernels
# defined in this module. Triangular-solve coverage with the solver disabled
# lives in test_tile_solve_no_mathdx.py; this file focuses on the Cholesky
# factorization paths.
wp.set_module_options({"enable_mathdx_solver": False})

TILE_DIM = 32
N = 8


# -----------------------------------------------------------------------------
# Lower Cholesky factorization
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def tile_cholesky_lower_kernel(gA: wp.array2d[wp.float64], gL: wp.array2d[wp.float64]):
    A = wp.tile_load(gA, shape=(N, N))
    L = wp.tile_cholesky(A)
    wp.tile_store(gL, L)


@wp.kernel(enable_backward=False)
def tile_cholesky_lower_inplace_kernel(gA: wp.array2d[wp.float64]):
    A = wp.tile_load(gA, shape=(N, N))
    wp.tile_cholesky_inplace(A)
    wp.tile_store(gA, A)


# -----------------------------------------------------------------------------
# Upper Cholesky factorization
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def tile_cholesky_upper_kernel(gA: wp.array2d[wp.float64], gU: wp.array2d[wp.float64]):
    A = wp.tile_load(gA, shape=(N, N))
    U = wp.tile_cholesky(A, fill_mode="upper")
    wp.tile_store(gU, U)


@wp.kernel(enable_backward=False)
def tile_cholesky_upper_inplace_kernel(gA: wp.array2d[wp.float64]):
    A = wp.tile_load(gA, shape=(N, N))
    wp.tile_cholesky_inplace(A, fill_mode="upper")
    wp.tile_store(gA, A)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _spd(n, seed=0):
    """Return an SPD matrix and its lower Cholesky factor."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    A = A @ A.T + n * np.eye(n)
    L = np.linalg.cholesky(A)
    return A, L


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_cholesky_lower(test, device):
    A_np, L_ref = _spd(N, seed=1)

    A = wp.array(A_np, dtype=wp.float64, device=device)
    L = wp.zeros((N, N), dtype=wp.float64, device=device)

    wp.launch_tiled(tile_cholesky_lower_kernel, dim=[1], inputs=[A, L], block_dim=TILE_DIM, device=device)
    assert_np_equal(L.numpy(), L_ref, tol=1e-10)


def test_cholesky_lower_inplace(test, device):
    A_np, L_ref = _spd(N, seed=2)

    A = wp.array(A_np.copy(), dtype=wp.float64, device=device)
    wp.launch_tiled(tile_cholesky_lower_inplace_kernel, dim=[1], inputs=[A], block_dim=TILE_DIM, device=device)
    # The lower-triangular factor lives in the lower triangle (upper zeroed).
    A_out = A.numpy()
    assert_np_equal(np.tril(A_out), L_ref, tol=1e-10)
    assert_np_equal(np.triu(A_out, k=1), np.zeros((N, N)), tol=0.0)


def test_cholesky_upper(test, device):
    A_np, L_ref = _spd(N, seed=3)
    U_ref = L_ref.T  # A = U^T U with U = L^T

    A = wp.array(A_np, dtype=wp.float64, device=device)
    U = wp.zeros((N, N), dtype=wp.float64, device=device)

    wp.launch_tiled(tile_cholesky_upper_kernel, dim=[1], inputs=[A, U], block_dim=TILE_DIM, device=device)
    assert_np_equal(U.numpy(), U_ref, tol=1e-10)


def test_cholesky_upper_inplace(test, device):
    A_np, L_ref = _spd(N, seed=4)
    U_ref = L_ref.T

    A = wp.array(A_np.copy(), dtype=wp.float64, device=device)
    wp.launch_tiled(tile_cholesky_upper_inplace_kernel, dim=[1], inputs=[A], block_dim=TILE_DIM, device=device)
    A_out = A.numpy()
    assert_np_equal(np.triu(A_out), U_ref, tol=1e-10)


# -----------------------------------------------------------------------------
# Cholesky adjoint -- exercises cooperative_scalar_cholesky_adj via the
# `enable_mathdx_solver=False` module setting. Kernels here are the same
# shape as the lower/upper backward kernels in test_tile_cholesky.py.
# -----------------------------------------------------------------------------


@wp.kernel
def tile_cholesky_lower_backward_kernel(gA: wp.array2d[wp.float64], gL: wp.array2d[wp.float64]):
    A = wp.tile_load(gA, shape=(N, N), storage="shared")
    L = wp.tile_cholesky(A)
    wp.tile_store(gL, L)


@wp.kernel
def tile_cholesky_upper_backward_kernel(gA: wp.array2d[wp.float64], gU: wp.array2d[wp.float64]):
    A = wp.tile_load(gA, shape=(N, N), storage="shared")
    U = wp.tile_cholesky(A, fill_mode="upper")
    wp.tile_store(gU, U)


def _cholesky_adjoint_numpy_lower(L, adj_L):
    P = L.T @ adj_L
    P = np.tril(P)
    P[np.diag_indices_from(P)] *= 0.5
    S = P + P.T
    X = np.linalg.solve(L.T, S)
    B = np.linalg.solve(L.T, X.T)
    grad_A = np.tril(B)
    grad_A[np.diag_indices_from(grad_A)] *= 0.5
    return grad_A


def _cholesky_adjoint_numpy_upper(U, adj_U):
    P = adj_U @ U.T
    P = np.triu(P)
    P[np.diag_indices_from(P)] *= 0.5
    S = P + P.T
    X = np.linalg.solve(U, S)
    B = np.linalg.solve(U, X.T)
    grad_A = np.triu(B)
    grad_A[np.diag_indices_from(grad_A)] *= 0.5
    return grad_A


def test_cholesky_lower_backward(test, device):
    A_np, L_ref = _spd(N, seed=20)
    rng = np.random.default_rng(21)
    adj_L = rng.standard_normal((N, N))

    A = wp.array(A_np, dtype=wp.float64, requires_grad=True, device=device)
    L = wp.zeros((N, N), dtype=wp.float64, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_cholesky_lower_backward_kernel, dim=[1], inputs=[A, L], block_dim=TILE_DIM, device=device)
    tape.backward(grads={L: wp.array(adj_L, dtype=wp.float64, device=device)})

    grad_A = A.grad.numpy()
    grad_A_ref = _cholesky_adjoint_numpy_lower(L_ref, adj_L)
    assert_np_equal(L.numpy(), L_ref, tol=1e-10)
    assert_np_equal(np.tril(grad_A), grad_A_ref, tol=1e-8)
    assert_np_equal(np.triu(grad_A, k=1), np.zeros((N, N)), tol=0.0)


def test_cholesky_upper_backward(test, device):
    A_np, L_ref = _spd(N, seed=22)
    U_ref = L_ref.T
    rng = np.random.default_rng(23)
    adj_U = rng.standard_normal((N, N))

    A = wp.array(A_np, dtype=wp.float64, requires_grad=True, device=device)
    U = wp.zeros((N, N), dtype=wp.float64, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_cholesky_upper_backward_kernel, dim=[1], inputs=[A, U], block_dim=TILE_DIM, device=device)
    tape.backward(grads={U: wp.array(adj_U, dtype=wp.float64, device=device)})

    grad_A = A.grad.numpy()
    grad_A_ref = _cholesky_adjoint_numpy_upper(U_ref, adj_U)
    assert_np_equal(U.numpy(), U_ref, tol=1e-10)
    assert_np_equal(np.triu(grad_A), grad_A_ref, tol=1e-8)
    assert_np_equal(np.tril(grad_A, k=-1), np.zeros((N, N)), tol=0.0)


# -----------------------------------------------------------------------------
# Larger-N adjoint smoke -- exercises the cooperative_scalar_cholesky_adj
# shared-mem scratch budget at a non-trivial tile size. Two __shared__ T W[n*n]
# buffers in float64 total 16 KiB at n=32 and 64 KiB at n=64 (before counting
# the shared input/output tiles). Catch budget failures in CI rather than at
# runtime.
# -----------------------------------------------------------------------------

N32 = 32


@wp.kernel
def tile_cholesky_lower_backward_n32_kernel(gA: wp.array2d[wp.float64], gL: wp.array2d[wp.float64]):
    A = wp.tile_load(gA, shape=(N32, N32), storage="shared")
    L = wp.tile_cholesky(A)
    wp.tile_store(gL, L)


def test_cholesky_lower_backward_n32(test, device):
    A_np, L_ref = _spd(N32, seed=40)
    rng = np.random.default_rng(41)
    adj_L = rng.standard_normal((N32, N32))

    A = wp.array(A_np, dtype=wp.float64, requires_grad=True, device=device)
    L = wp.zeros((N32, N32), dtype=wp.float64, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_cholesky_lower_backward_n32_kernel, dim=[1], inputs=[A, L], block_dim=TILE_DIM, device=device
        )
    tape.backward(grads={L: wp.array(adj_L, dtype=wp.float64, device=device)})

    grad_A = A.grad.numpy()
    grad_A_ref = _cholesky_adjoint_numpy_lower(L_ref, adj_L)
    assert_np_equal(L.numpy(), L_ref, tol=1e-10)
    assert_np_equal(np.tril(grad_A), grad_A_ref, tol=1e-7)
    assert_np_equal(np.triu(grad_A, k=1), np.zeros((N32, N32)), tol=0.0)


# block_dim == 1 GPU smoke: thread-strided loops collapse to sequential, but
# the GPU codegen path must still compile and run correctly.
def test_cholesky_lower_block_dim_1(test, device):
    A_np, L_ref = _spd(N, seed=42)

    A = wp.array(A_np, dtype=wp.float64, device=device)
    L = wp.zeros((N, N), dtype=wp.float64, device=device)

    wp.launch_tiled(tile_cholesky_lower_kernel, dim=[1], inputs=[A, L], block_dim=1, device=device)
    assert_np_equal(L.numpy(), L_ref, tol=1e-10)


# -----------------------------------------------------------------------------
# Suite registration
# -----------------------------------------------------------------------------


class TestTileCholeskyNoMathdx(unittest.TestCase):
    pass


_devices = get_test_devices()

for name, func in [
    ("test_cholesky_lower", test_cholesky_lower),
    ("test_cholesky_lower_inplace", test_cholesky_lower_inplace),
    ("test_cholesky_upper", test_cholesky_upper),
    ("test_cholesky_upper_inplace", test_cholesky_upper_inplace),
    ("test_cholesky_lower_backward", test_cholesky_lower_backward),
    ("test_cholesky_upper_backward", test_cholesky_upper_backward),
    ("test_cholesky_lower_backward_n32", test_cholesky_lower_backward_n32),
    ("test_cholesky_lower_block_dim_1", test_cholesky_lower_block_dim_1),
]:
    add_function_test(TestTileCholeskyNoMathdx, name, func, devices=_devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
