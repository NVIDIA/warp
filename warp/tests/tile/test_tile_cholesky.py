# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.init()  # For wp._src.context.runtime.core.wp_is_mathdx_enabled()

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_DIM = 32

# Forward-only kernels skip adjoint codegen
wp.get_module("test_cholesky_fwd").options["enable_backward"] = False


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky(
    gA: wp.array2d(dtype=wp.float64),
    gD: wp.array1d(dtype=wp.float64),
    gL: wp.array2d(dtype=wp.float64),
    gy: wp.array1d(dtype=wp.float64),
    gx: wp.array1d(dtype=wp.float64),
):
    # Load A, D & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    d = wp.tile_load(gD, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    # Ensure tile_diag_add() and tile_cholesky_solve() work with transposed matrices
    a_t = wp.tile_transpose(a)
    # Compute L st LL^T = A^T + diag(D)
    b = wp.tile_diag_add(a_t, d)
    l = wp.tile_cholesky(b)
    # Solve for y in LL^T x = y
    x = wp.tile_cholesky_solve(l, y)
    # Store L & y
    wp.tile_store(gL, l)
    wp.tile_store(gx, x)


def test_tile_cholesky_cholesky(test, device):
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64)
    D_h = 8.0 * np.ones(TILE_M, dtype=np.float64)
    L_h = np.zeros_like(A_h)
    Y_h = np.arange(TILE_M, dtype=np.float64)
    X_h = np.zeros_like(Y_h)

    A_np = A_h.T + np.diag(D_h)
    L_np = np.linalg.cholesky(A_np)
    X_np = np.linalg.solve(A_np, Y_h)

    A_wp = wp.array(A_h, requires_grad=True, dtype=wp.float64, device=device)
    D_wp = wp.array(D_h, requires_grad=True, dtype=wp.float64, device=device)
    L_wp = wp.array(L_h, requires_grad=True, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, requires_grad=True, dtype=wp.float64, device=device)
    X_wp = wp.array(X_h, requires_grad=True, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky, dim=[1, 1], inputs=[A_wp, D_wp, L_wp, Y_wp, X_wp], block_dim=TILE_DIM, device=device
    )

    np.testing.assert_allclose(X_wp.numpy(), X_np)
    np.testing.assert_allclose(L_wp.numpy(), L_np)

    # TODO: implement and test backward pass


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_inplace(
    gA: wp.array2d(dtype=wp.float64),
    gy: wp.array1d(dtype=wp.float64),
):
    # Load A & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    # Compute L st LL^T = A inplace
    wp.tile_cholesky_inplace(a)
    # Solve for y in LL^T x = y inplace
    wp.tile_cholesky_solve_inplace(a, y)
    # Store L & y
    wp.tile_store(gA, a)
    wp.tile_store(gy, y)


def test_tile_cholesky_cholesky_inplace(test, device):
    rng = np.random.default_rng(42)
    L_h = np.tril(rng.random((TILE_M, TILE_M)))  # Lower triangular matrix
    A_h = L_h @ L_h.T
    Y_h = np.arange(TILE_M, dtype=np.float64)

    Y_sol_np = np.linalg.solve(A_h, Y_h)

    A_wp = wp.array(A_h, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)

    wp.launch_tiled(tile_math_cholesky_inplace, dim=[1, 1], inputs=[A_wp, Y_wp], block_dim=TILE_DIM, device=device)

    np.testing.assert_allclose(Y_wp.numpy(), Y_sol_np)
    np.testing.assert_allclose(A_wp.numpy(), L_h)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_multiple_rhs(
    gA: wp.array2d(dtype=wp.float64),
    gD: wp.array1d(dtype=wp.float64),
    gL: wp.array2d(dtype=wp.float64),
    gy: wp.array2d(dtype=wp.float64),
    gx: wp.array2d(dtype=wp.float64),
    gz: wp.array2d(dtype=wp.float64),
):
    # Load A, D & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    d = wp.tile_load(gD, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=(TILE_M, TILE_M), storage="shared")
    # Ensure tile_diag_add() and tile_cholesky_solve() work with transposed matrices
    a_t = wp.tile_transpose(a)
    # Compute L st LL^T = A.T + diag(D)
    b = wp.tile_diag_add(a_t, d)
    l = wp.tile_cholesky(b)
    # Solve for y in LL^T x = y.T
    y_t = wp.tile_transpose(y)
    x = wp.tile_cholesky_solve(l, y_t)
    # Ensure matmul receives correct layout information
    z = wp.tile_matmul(x, x)
    # Store L & y
    wp.tile_store(gL, l)
    wp.tile_store(gx, x)
    wp.tile_store(gz, z)


def test_tile_cholesky_cholesky_multiple_rhs(test, device):
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64)
    D_h = 8.0 * np.ones(TILE_M, dtype=np.float64)
    L_h = np.zeros_like(A_h)
    Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
    X_h = np.zeros_like(Y_h)
    Z_h = np.zeros_like(Y_h)

    A_np = A_h.T + np.diag(D_h)
    L_np = np.linalg.cholesky(A_np)
    X_np = np.linalg.solve(A_np, Y_h.T)
    Z_np = X_np @ X_np

    A_wp = wp.array(A_h, requires_grad=True, dtype=wp.float64, device=device)
    D_wp = wp.array(D_h, requires_grad=True, dtype=wp.float64, device=device)
    L_wp = wp.array(L_h, requires_grad=True, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, requires_grad=True, dtype=wp.float64, device=device)
    X_wp = wp.array(X_h, requires_grad=True, dtype=wp.float64, device=device)
    Z_wp = wp.array(Z_h, requires_grad=True, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky_multiple_rhs,
        dim=[1, 1],
        inputs=[A_wp, D_wp, L_wp, Y_wp, X_wp, Z_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    np.testing.assert_allclose(L_wp.numpy(), L_np)
    np.testing.assert_allclose(X_wp.numpy(), X_np)
    np.testing.assert_allclose(Z_wp.numpy(), Z_np)

    # TODO: implement and test backward pass


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_multiple_rhs_inplace(
    gA: wp.array2d(dtype=wp.float64),
    gy: wp.array2d(dtype=wp.float64),
    gz: wp.array2d(dtype=wp.float64),
):
    # Load A & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=(TILE_M, TILE_M), storage="shared")
    # Compute L st LL^T = A inplace
    wp.tile_cholesky_inplace(a)
    # Solve for y in LL^T x = y.T inplace
    y_t = wp.tile_transpose(y)
    wp.tile_cholesky_solve_inplace(a, y_t)
    y = wp.tile_transpose(y_t)
    # Ensure matmul receives correct layout information
    z = wp.tile_matmul(y, y)
    # Store L & y
    wp.tile_store(gA, a)
    wp.tile_store(gy, y)
    wp.tile_store(gz, z)


def test_tile_cholesky_cholesky_multiple_rhs_inplace(test, device):
    rng = np.random.default_rng(42)
    L_h = np.tril(rng.random((TILE_M, TILE_M)))  # Lower triangular matrix
    A_h = L_h @ L_h.T
    Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
    Z_h = np.zeros_like(Y_h)

    Y_sol_np = np.linalg.solve(A_h, Y_h.T).T
    Z_np = Y_sol_np @ Y_sol_np

    A_wp = wp.array(A_h, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
    Z_wp = wp.array(Z_h, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky_multiple_rhs_inplace,
        dim=[1, 1],
        inputs=[A_wp, Y_wp, Z_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    np.testing.assert_allclose(A_wp.numpy(), L_h)
    np.testing.assert_allclose(Y_wp.numpy(), Y_sol_np)
    np.testing.assert_allclose(Z_wp.numpy(), Z_np)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_forward_substitution(
    gL: wp.array2d(dtype=wp.float64), gx: wp.array1d(dtype=wp.float64), gz: wp.array1d(dtype=wp.float64)
):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=TILE_M, storage="shared")
    # Solve for z in Lz = x
    # Transpose because we loaded an upper triangular matrix
    z = wp.tile_lower_solve(wp.tile_transpose(L), x)
    # Store z
    wp.tile_store(gz, z)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_forward_substitution_inplace(gL: wp.array2d(dtype=wp.float64), gx: wp.array1d(dtype=wp.float64)):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=TILE_M, storage="shared")
    # Solve for z in Lz = x inplace
    # Transpose because we loaded an upper triangular matrix
    wp.tile_lower_solve_inplace(wp.tile_transpose(L), x)
    # Store z
    wp.tile_store(gx, x)


def test_tile_cholesky_forward_substitution(test, device):
    # Create test data
    rng = np.random.default_rng(42)
    L_h = np.triu(rng.random((TILE_M, TILE_M)))  # Upper triangular matrix
    x_h = rng.random(TILE_M)
    z_h = np.zeros_like(x_h)

    # Compute reference solution using numpy
    z_np = np.linalg.solve(L_h.T, x_h)

    # Create Warp arrays
    L_wp = wp.array(L_h, requires_grad=True, dtype=wp.float64, device=device)
    x_wp = wp.array(x_h, requires_grad=True, dtype=wp.float64, device=device)
    z_wp = wp.array(z_h, requires_grad=True, dtype=wp.float64, device=device)

    # Run kernel
    wp.launch_tiled(
        tile_math_forward_substitution, dim=[1, 1], inputs=[L_wp, x_wp, z_wp], block_dim=TILE_DIM, device=device
    )

    # Verify results
    np.testing.assert_allclose(z_wp.numpy(), z_np)

    # TODO: implement and test backward pass

    # Run inplace kernel
    wp.launch_tiled(
        tile_math_forward_substitution_inplace, dim=[1, 1], inputs=[L_wp, x_wp], block_dim=TILE_DIM, device=device
    )

    # Verify results
    np.testing.assert_allclose(x_wp.numpy(), z_np)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_back_substitution(
    gL: wp.array2d(dtype=wp.float64), gx: wp.array1d(dtype=wp.float64), gz: wp.array1d(dtype=wp.float64)
):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=TILE_M, storage="shared")
    # Solve for z in L^T z = x
    # Transpose because we loaded a lower triangular matrix
    z = wp.tile_upper_solve(wp.tile_transpose(L), x)
    # Store z
    wp.tile_store(gz, z)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_back_substitution_inplace(gL: wp.array2d(dtype=wp.float64), gx: wp.array1d(dtype=wp.float64)):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=TILE_M, storage="shared")
    # Solve for z in L^T z = x inplace
    # Transpose because we loaded a lower triangular matrix
    wp.tile_upper_solve_inplace(wp.tile_transpose(L), x)
    # Store z
    wp.tile_store(gx, x)


def test_tile_cholesky_back_substitution(test, device):
    # Create test data
    rng = np.random.default_rng(42)
    L_h = np.tril(rng.random((TILE_M, TILE_M)))  # Lower triangular matrix
    x_h = rng.random(TILE_M)
    z_h = np.zeros_like(x_h)

    # Compute reference solution using numpy
    z_np = np.linalg.solve(L_h.T, x_h)

    # Create Warp arrays
    L_wp = wp.array(L_h, requires_grad=True, dtype=wp.float64, device=device)
    x_wp = wp.array(x_h, requires_grad=True, dtype=wp.float64, device=device)
    z_wp = wp.array(z_h, requires_grad=True, dtype=wp.float64, device=device)

    # Run kernel
    wp.launch_tiled(
        tile_math_back_substitution, dim=[1, 1], inputs=[L_wp, x_wp, z_wp], block_dim=TILE_DIM, device=device
    )

    # Verify results
    np.testing.assert_allclose(z_wp.numpy(), z_np)

    # TODO: implement and test backward pass

    # Run inplace kernel
    wp.launch_tiled(
        tile_math_back_substitution_inplace, dim=[1, 1], inputs=[L_wp, x_wp], block_dim=TILE_DIM, device=device
    )

    # Verify results
    np.testing.assert_allclose(x_wp.numpy(), z_np)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_forward_substitution_multiple_rhs(
    gL: wp.array2d(dtype=wp.float64),
    gx: wp.array2d(dtype=wp.float64),
    gz: wp.array2d(dtype=wp.float64),
    gc: wp.array2d(dtype=wp.float64),
):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=(TILE_M, TILE_M), storage="shared")
    # Solve for z in Lz = x.T
    x_t = wp.tile_transpose(x)
    z = wp.tile_lower_solve(L, x_t)
    # Ensure matmul receives correct layout information
    c = wp.tile_matmul(z, z)
    # Store z and c
    wp.tile_store(gz, z)
    wp.tile_store(gc, c)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_forward_substitution_multiple_rhs_inplace(
    gL: wp.array2d(dtype=wp.float64),
    gx: wp.array2d(dtype=wp.float64),
    gc: wp.array2d(dtype=wp.float64),
):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=(TILE_M, TILE_M), storage="shared")
    # Solve for z in Lz = x.T inplace
    x_t = wp.tile_transpose(x)
    wp.tile_lower_solve_inplace(L, x_t)
    # Ensure matmul receives correct layout information
    c = wp.tile_matmul(x_t, x_t)
    # Store x and c
    wp.tile_store(gx, x_t)
    wp.tile_store(gc, c)


def test_tile_cholesky_forward_substitution_multiple_rhs(test, device):
    # Create test data
    rng = np.random.default_rng(42)
    L_h = np.tril(rng.random((TILE_M, TILE_M)))  # Lower triangular matrix
    x_h = rng.random((TILE_M, TILE_M))  # Multiple right-hand sides
    z_h = np.zeros_like(x_h)
    c_h = np.zeros_like(x_h)

    # Compute reference solution using numpy
    z_np = np.linalg.solve(L_h, x_h.T)
    c_np = z_np @ z_np

    # Create Warp arrays
    L_wp = wp.array(L_h, requires_grad=True, dtype=wp.float64, device=device)
    x_wp = wp.array(x_h, requires_grad=True, dtype=wp.float64, device=device)
    z_wp = wp.array(z_h, requires_grad=True, dtype=wp.float64, device=device)
    c_wp = wp.array(c_h, requires_grad=True, dtype=wp.float64, device=device)

    # Run kernel
    wp.launch_tiled(
        tile_math_forward_substitution_multiple_rhs,
        dim=[1, 1],
        inputs=[L_wp, x_wp, z_wp, c_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    # Verify results
    np.testing.assert_allclose(z_wp.numpy(), z_np)
    np.testing.assert_allclose(c_wp.numpy(), c_np)

    # TODO: implement and test backward pass

    # Run inplace kernel
    wp.launch_tiled(
        tile_math_forward_substitution_multiple_rhs_inplace,
        dim=[1, 1],
        inputs=[L_wp, x_wp, c_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    # Verify results
    np.testing.assert_allclose(x_wp.numpy(), z_np)
    np.testing.assert_allclose(c_wp.numpy(), c_np)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_back_substitution_multiple_rhs(
    gL: wp.array2d(dtype=wp.float64),
    gx: wp.array2d(dtype=wp.float64),
    gz: wp.array2d(dtype=wp.float64),
    gc: wp.array2d(dtype=wp.float64),
):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=(TILE_M, TILE_M), storage="shared")
    # Solve for z in L^T z = x.T
    x_t = wp.tile_transpose(x)
    z = wp.tile_upper_solve(wp.tile_transpose(L), x_t)
    # Ensure matmul receives correct layout information
    c = wp.tile_matmul(z, z)
    # Store z and c
    wp.tile_store(gz, z)
    wp.tile_store(gc, c)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_back_substitution_multiple_rhs_inplace(
    gL: wp.array2d(dtype=wp.float64),
    gx: wp.array2d(dtype=wp.float64),
    gc: wp.array2d(dtype=wp.float64),
):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=(TILE_M, TILE_M), storage="shared")
    # Solve for z in L^T z = x.T inplace
    x_t = wp.tile_transpose(x)
    wp.tile_upper_solve_inplace(wp.tile_transpose(L), x_t)
    # Ensure matmul receives correct layout information
    c = wp.tile_matmul(x_t, x_t)
    # Store x and c
    wp.tile_store(gx, x_t)
    wp.tile_store(gc, c)


def test_tile_cholesky_back_substitution_multiple_rhs(test, device):
    # Create test data
    rng = np.random.default_rng(42)
    L_h = np.tril(rng.random((TILE_M, TILE_M)))  # Lower triangular matrix
    x_h = rng.random((TILE_M, TILE_M))  # Multiple right-hand sides
    z_h = np.zeros_like(x_h)
    c_h = np.zeros_like(x_h)

    # Compute reference solution using numpy
    z_np = np.linalg.solve(L_h.T, x_h.T)
    c_np = z_np @ z_np

    # Create Warp arrays
    L_wp = wp.array(L_h, requires_grad=True, dtype=wp.float64, device=device)
    x_wp = wp.array(x_h, requires_grad=True, dtype=wp.float64, device=device)
    z_wp = wp.array(z_h, requires_grad=True, dtype=wp.float64, device=device)
    c_wp = wp.array(c_h, requires_grad=True, dtype=wp.float64, device=device)

    # Run kernel
    wp.launch_tiled(
        tile_math_back_substitution_multiple_rhs,
        dim=[1, 1],
        inputs=[L_wp, x_wp, z_wp, c_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    # Verify results
    np.testing.assert_allclose(z_wp.numpy(), z_np)
    np.testing.assert_allclose(c_wp.numpy(), c_np)

    # TODO: implement and test backward pass

    # Run inplace kernel
    wp.launch_tiled(
        tile_math_back_substitution_multiple_rhs_inplace,
        dim=[1, 1],
        inputs=[L_wp, x_wp, c_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    # Verify results
    np.testing.assert_allclose(x_wp.numpy(), z_np)
    np.testing.assert_allclose(c_wp.numpy(), c_np)


@wp.kernel()
def tile_cholesky_lower_backward_kernel(
    gA: wp.array2d(dtype=Any),
    gL: wp.array2d(dtype=Any),
):
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    l = wp.tile_cholesky(a)
    wp.tile_store(gL, l)


wp.overload(
    tile_cholesky_lower_backward_kernel, {"gA": wp.array2d(dtype=wp.float32), "gL": wp.array2d(dtype=wp.float32)}
)
wp.overload(
    tile_cholesky_lower_backward_kernel, {"gA": wp.array2d(dtype=wp.float64), "gL": wp.array2d(dtype=wp.float64)}
)


def cholesky_adjoint_numpy_lower(L, adj_L):
    """Analytic adjoint of Cholesky factorization (lower-triangle parameterization)."""
    P = L.T @ adj_L
    P = np.tril(P)
    P[np.diag_indices_from(P)] *= 0.5
    S = P + P.T
    X = np.linalg.solve(L.T, S)
    B = np.linalg.solve(L.T, X.T)
    grad_A = np.tril(B)
    grad_A[np.diag_indices_from(grad_A)] *= 0.5
    return grad_A


def _cholesky_lower_backward(A_np, adj_L, device, wp_dtype=wp.float64):
    """Run tile Cholesky forward+backward, return (L_wp, grad_A) as NumPy arrays."""
    np_dtype = wp.dtype_to_numpy(wp_dtype)
    A_wp = wp.array(A_np.astype(np_dtype), dtype=wp_dtype, requires_grad=True, device=device)
    L_wp = wp.zeros((TILE_M, TILE_M), dtype=wp_dtype, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_cholesky_lower_backward_kernel,
            dim=[1, 1],
            inputs=[A_wp, L_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    tape.backward(grads={L_wp: wp.array(adj_L.astype(np_dtype), dtype=wp_dtype, device=device)})
    return L_wp.numpy(), A_wp.grad.numpy()


def test_tile_cholesky_lower_backward(dtype):
    def test(test, device):
        np_dtype = wp.dtype_to_numpy(dtype)
        fwd_atol = 1e-10 if dtype == wp.float64 else 1e-4
        bwd_atol = 1e-8 if dtype == wp.float64 else 1e-3
        zero_atol = 0.0 if dtype == wp.float64 else 1e-6

        def check(A_np, adj_L):
            L_np = np.linalg.cholesky(A_np)
            L_wp, grad_A = _cholesky_lower_backward(A_np, adj_L, device, wp_dtype=dtype)
            grad_A_ref = cholesky_adjoint_numpy_lower(L_np, adj_L)
            np.testing.assert_allclose(L_wp, L_np.astype(np_dtype), atol=fwd_atol)
            np.testing.assert_allclose(np.tril(grad_A), grad_A_ref.astype(np_dtype), atol=bwd_atol)
            np.testing.assert_allclose(np.triu(grad_A, k=1), 0.0, atol=zero_atol)

        # Random SPD
        rng = np.random.default_rng(42)
        M = rng.random((TILE_M, TILE_M))
        check(M.T @ M + 8.0 * np.eye(TILE_M), rng.standard_normal((TILE_M, TILE_M)))

        # Identity - closed-form adjoint (independent of Murray formula):
        # Linearize A = LL^T at L = I: dA = dL + dL^T, so adj_A = tril(adj_L) with halved diagonal.
        rng = np.random.default_rng(100)
        adj_L = rng.standard_normal((TILE_M, TILE_M))
        L_wp, grad_A = _cholesky_lower_backward(np.eye(TILE_M), adj_L, device, wp_dtype=dtype)
        grad_A_ref = np.tril(adj_L)
        grad_A_ref[np.diag_indices_from(grad_A_ref)] *= 0.5
        np.testing.assert_allclose(L_wp, np.eye(TILE_M, dtype=np_dtype), atol=fwd_atol)
        np.testing.assert_allclose(np.tril(grad_A), grad_A_ref.astype(np_dtype), atol=bwd_atol)
        np.testing.assert_allclose(np.triu(grad_A, k=1), 0.0, atol=zero_atol)

        # Diagonal - closed-form adjoint (independent of Murray formula):
        # Linearize A = LL^T at L = diag(s): dA = diag(s) dL^T + dL diag(s),
        # so adj_A = tril(adj_L) / s[newaxis,:] with halved diagonal.
        rng = np.random.default_rng(200)
        d = rng.uniform(1.0, 10.0, TILE_M)
        adj_L = rng.standard_normal((TILE_M, TILE_M))
        s = np.sqrt(d)
        L_wp, grad_A = _cholesky_lower_backward(np.diag(d), adj_L, device, wp_dtype=dtype)
        grad_A_ref = np.tril(adj_L) / s[np.newaxis, :]
        grad_A_ref[np.diag_indices_from(grad_A_ref)] *= 0.5
        np.testing.assert_allclose(L_wp, np.diag(s).astype(np_dtype), atol=fwd_atol)
        np.testing.assert_allclose(np.tril(grad_A), grad_A_ref.astype(np_dtype), atol=bwd_atol)
        np.testing.assert_allclose(np.triu(grad_A, k=1), 0.0, atol=zero_atol)

    return test


@wp.kernel()
def tile_cholesky_upper_backward_kernel(
    gA: wp.array2d(dtype=Any),
    gU: wp.array2d(dtype=Any),
):
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    u = wp.tile_cholesky(a, fill_mode="upper")
    wp.tile_store(gU, u)


wp.overload(
    tile_cholesky_upper_backward_kernel, {"gA": wp.array2d(dtype=wp.float32), "gU": wp.array2d(dtype=wp.float32)}
)
wp.overload(
    tile_cholesky_upper_backward_kernel, {"gA": wp.array2d(dtype=wp.float64), "gU": wp.array2d(dtype=wp.float64)}
)


def cholesky_adjoint_numpy_upper(U, adj_U):
    """Analytic adjoint of Cholesky factorization (upper-triangle parameterization)."""
    P = adj_U @ U.T
    P = np.triu(P)
    P[np.diag_indices_from(P)] *= 0.5
    S = P + P.T
    X = np.linalg.solve(U, S)
    B = np.linalg.solve(U, X.T)
    grad_A = np.triu(B)
    grad_A[np.diag_indices_from(grad_A)] *= 0.5
    return grad_A


def _cholesky_upper_backward(A_np, adj_U, device, wp_dtype=wp.float64):
    """Run tile Cholesky(upper) forward+backward, return (U_wp, grad_A) as NumPy arrays."""
    np_dtype = wp.dtype_to_numpy(wp_dtype)
    A_wp = wp.array(A_np.astype(np_dtype), dtype=wp_dtype, requires_grad=True, device=device)
    U_wp = wp.zeros((TILE_M, TILE_M), dtype=wp_dtype, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_cholesky_upper_backward_kernel,
            dim=[1, 1],
            inputs=[A_wp, U_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    tape.backward(grads={U_wp: wp.array(adj_U.astype(np_dtype), dtype=wp_dtype, device=device)})
    return U_wp.numpy(), A_wp.grad.numpy()


def test_tile_cholesky_upper_backward(dtype):
    def test(test, device):
        np_dtype = wp.dtype_to_numpy(dtype)
        fwd_atol = 1e-10 if dtype == wp.float64 else 1e-4
        bwd_atol = 1e-8 if dtype == wp.float64 else 1e-3
        zero_atol = 0.0 if dtype == wp.float64 else 1e-6

        def check(A_np, adj_U):
            U_np = np.linalg.cholesky(A_np).T
            U_wp, grad_A = _cholesky_upper_backward(A_np, adj_U, device, wp_dtype=dtype)
            grad_A_ref = cholesky_adjoint_numpy_upper(U_np, adj_U)
            np.testing.assert_allclose(U_wp, U_np.astype(np_dtype), atol=fwd_atol)
            np.testing.assert_allclose(np.triu(grad_A), grad_A_ref.astype(np_dtype), atol=bwd_atol)
            np.testing.assert_allclose(np.tril(grad_A, k=-1), 0.0, atol=zero_atol)

        # Random SPD
        rng = np.random.default_rng(42)
        M = rng.random((TILE_M, TILE_M))
        check(M.T @ M + 8.0 * np.eye(TILE_M), rng.standard_normal((TILE_M, TILE_M)))

        # Identity - closed-form adjoint (independent of Murray formula):
        # Linearize A = U^TU at U = I: dA = dU^T + dU, so adj_A = triu(adj_U) with halved diagonal.
        rng = np.random.default_rng(100)
        adj_U = rng.standard_normal((TILE_M, TILE_M))
        U_wp, grad_A = _cholesky_upper_backward(np.eye(TILE_M), adj_U, device, wp_dtype=dtype)
        grad_A_ref = np.triu(adj_U)
        grad_A_ref[np.diag_indices_from(grad_A_ref)] *= 0.5
        np.testing.assert_allclose(U_wp, np.eye(TILE_M, dtype=np_dtype), atol=fwd_atol)
        np.testing.assert_allclose(np.triu(grad_A), grad_A_ref.astype(np_dtype), atol=bwd_atol)
        np.testing.assert_allclose(np.tril(grad_A, k=-1), 0.0, atol=zero_atol)

        # Diagonal - closed-form adjoint (independent of Murray formula):
        # Linearize A = U^TU at U = diag(s): dA = dU^T diag(s) + diag(s) dU,
        # so adj_A = triu(adj_U) / s[:,newaxis] with halved diagonal.
        rng = np.random.default_rng(200)
        d = rng.uniform(1.0, 10.0, TILE_M)
        adj_U = rng.standard_normal((TILE_M, TILE_M))
        s = np.sqrt(d)
        U_wp, grad_A = _cholesky_upper_backward(np.diag(d), adj_U, device, wp_dtype=dtype)
        grad_A_ref = np.triu(adj_U) / s[:, np.newaxis]
        grad_A_ref[np.diag_indices_from(grad_A_ref)] *= 0.5
        np.testing.assert_allclose(U_wp, np.diag(s).astype(np_dtype), atol=fwd_atol)
        np.testing.assert_allclose(np.triu(grad_A), grad_A_ref.astype(np_dtype), atol=bwd_atol)
        np.testing.assert_allclose(np.tril(grad_A, k=-1), 0.0, atol=zero_atol)

    return test


# tests a complex composition of most libmathdx calls
def test_tile_cholesky_block_cholesky(test, device):
    BLOCK_SIZE = wp.constant(TILE_M // 2)

    @wp.kernel(enable_backward=False, module="unique")
    def block_cholesky_kernel(
        A: wp.array2d(dtype=float),
        L: wp.array2d(dtype=float),
    ):
        """Compute the Cholesky factorization of a symmetric positive definite matrix ``A`` in blocks.

        It returns a lower-triangular matrix ``L`` such that ``A = L L^T``.
        """

        # Process the matrix in blocks along its leading dimension.
        for k in range(0, TILE_M, BLOCK_SIZE):
            end = k + BLOCK_SIZE

            # Load current diagonal block A[k:end, k:end]
            # and update with contributions from previously computed blocks.
            A_kk_tile = wp.tile_load(A, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(k, k), storage="shared")

            for j in range(0, k, BLOCK_SIZE):
                L_block = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(k, j))
                L_block_T = wp.tile_transpose(L_block)
                wp.tile_matmul(L_block, L_block_T, A_kk_tile, alpha=-1.0)

            # Compute the Cholesky factorization for the block
            # print(A_kk_tile)
            L_kk_tile = wp.tile_cholesky(A_kk_tile)
            wp.tile_store(L, L_kk_tile, offset=(k, k))

            # Process the blocks below the current block
            for i in range(end, TILE_M, BLOCK_SIZE):
                A_ik_tile = wp.tile_load(A, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(i, k), storage="shared")

                for j in range(0, k, BLOCK_SIZE):
                    L_tile = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(i, j))
                    L_2_tile = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(k, j))
                    L_T_tile = wp.tile_transpose(L_2_tile)
                    wp.tile_matmul(L_tile, L_T_tile, A_ik_tile, alpha=-1.0)

                A_ik_T_tile = wp.tile_transpose(A_ik_tile)
                sol_T_tile = wp.tile_lower_solve(L_kk_tile, A_ik_T_tile)
                sol_tile = wp.tile_transpose(sol_T_tile)

                wp.tile_store(L, sol_tile, offset=(i, k))

    @wp.kernel(enable_backward=False, module="unique")
    def block_cholesky_solve_kernel(
        L: wp.array2d(dtype=float),
        b: wp.array2d(dtype=float),
        scratch: wp.array2d(dtype=float),
        x: wp.array2d(dtype=float),
    ):
        """Solve ``A x = b`` given the Cholesky factor ``L (A = L L^T)`` using blocked forward and backward substitution."""

        # Forward substitution: solve L y = b
        for i in range(0, TILE_M, BLOCK_SIZE):
            i_end = i + BLOCK_SIZE
            rhs_tile = wp.tile_load(b, shape=(BLOCK_SIZE, 1), offset=(i, 0))
            for j in range(0, i, BLOCK_SIZE):
                L_block = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(i, j))
                y_block = wp.tile_load(scratch, shape=(BLOCK_SIZE, 1), offset=(j, 0))
                wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)
            L_tile = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(i, i))
            y_tile = wp.tile_lower_solve(L_tile, rhs_tile)
            wp.tile_store(scratch, y_tile, offset=(i, 0))

        # Backward substitution: solve L^T x = y
        for i in range(TILE_M - BLOCK_SIZE, -1, -BLOCK_SIZE):
            i_start = i
            i_end = i_start + BLOCK_SIZE
            rhs_tile = wp.tile_load(scratch, shape=(BLOCK_SIZE, 1), offset=(i_start, 0))
            for j in range(i_end, TILE_M, BLOCK_SIZE):
                L_tile = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(j, i_start))
                L_T_tile = wp.tile_transpose(L_tile)
                x_tile = wp.tile_load(x, shape=(BLOCK_SIZE, 1), offset=(j, 0))
                wp.tile_matmul(L_T_tile, x_tile, rhs_tile, alpha=-1.0)
            L_tile = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(i_start, i_start))
            x_tile = wp.tile_upper_solve(wp.tile_transpose(L_tile), rhs_tile)
            wp.tile_store(x, x_tile, offset=(i_start, 0))

    # check block cholesky decomposition

    rng = np.random.default_rng(42)

    M = np.array(rng.random((TILE_M, TILE_M)), dtype=float)

    A_np = M.T @ M + np.eye(TILE_M, TILE_M)
    L_np = np.linalg.cholesky(A_np)

    A_wp = wp.array(A_np, dtype=float, device=device)
    L_wp = wp.zeros_like(A_wp)

    wp.launch_tiled(block_cholesky_kernel, dim=1, inputs=[A_wp], outputs=[L_wp], block_dim=TILE_DIM, device=device)

    # check block cholesky solve

    assert_np_equal(L_wp.numpy(), L_np, tol=1e-6)

    b_np = np.array(rng.random((TILE_M, 1)), dtype=float)
    b_wp = wp.array(b_np, dtype=float, device=device)

    scratch = wp.zeros_like(b_wp)

    x_np = np.linalg.solve(L_np.T, np.linalg.solve(L_np, b_np))
    x_wp = wp.zeros_like(b_wp)

    wp.launch_tiled(
        block_cholesky_solve_kernel,
        dim=1,
        inputs=[L_wp, b_wp, scratch],
        outputs=[x_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(x_wp.numpy(), x_np, tol=1e-6)


@wp.kernel(module="test_cholesky_fwd")
def test_tile_lower_solve(L: wp.array2d(dtype=float), y: wp.array(dtype=float), x: wp.array(dtype=float)):
    L_tile = wp.tile_load(L, shape=(TILE_M, TILE_M))
    y_tile = wp.tile_load(x, shape=(TILE_M,))
    sol = wp.tile_lower_solve(L_tile, y_tile)
    wp.tile_store(x, sol)


@wp.kernel(module="test_cholesky_fwd")
def test_tile_upper_solve(L: wp.array2d(dtype=float), y: wp.array(dtype=float), x: wp.array(dtype=float)):
    L_tile = wp.tile_load(L, shape=(TILE_M, TILE_M))
    y_tile = wp.tile_load(x, shape=(TILE_M,))
    sol = wp.tile_upper_solve(L_tile, y_tile)
    wp.tile_store(x, sol)


def test_tile_cholesky_singular_matrices(test, device):
    if not wp._src.context.runtime.core.wp_is_mathdx_enabled():
        test.skipTest("MathDx is not enabled")

    rng = np.random.default_rng(42)
    L_np = np.tril(rng.random((TILE_M, TILE_M)))  # Lower triangular matrix
    L_np[-1, -1] = 0.0  # Make it singular
    y_np = rng.random(TILE_M)

    L_wp = wp.array2d(L_np, dtype=float, device=device)
    y_wp = wp.array(y_np, dtype=float, device=device)
    x_wp = wp.zeros_like(y_wp)

    wp.launch_tiled(
        test_tile_lower_solve, dim=1, inputs=[L_wp, y_wp], outputs=[x_wp], block_dim=TILE_DIM, device=device
    )

    test.assertTrue(np.isnan(x_wp.numpy()).any())

    L_np = np.triu(rng.random((TILE_M, TILE_M)))  # Upper triangular matrix
    L_np[-1, -1] = 0.0  # Make it singular

    L_wp = wp.array2d(L_np, dtype=float, device=device)
    y_wp = wp.array(y_np, dtype=float, device=device)
    x_wp = wp.zeros_like(y_wp)

    wp.launch_tiled(
        test_tile_upper_solve, dim=1, inputs=[L_wp, y_wp], outputs=[x_wp], block_dim=TILE_DIM, device=device
    )

    test.assertTrue(np.isnan(x_wp.numpy()).any())


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_upper(
    gA: wp.array2d(dtype=wp.float64),
    gD: wp.array1d(dtype=wp.float64),
    gU: wp.array2d(dtype=wp.float64),
    gy: wp.array1d(dtype=wp.float64),
    gx: wp.array1d(dtype=wp.float64),
):
    # Load A, D & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    d = wp.tile_load(gD, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    # Ensure tile_diag_add() works with transposed matrices
    a_t = wp.tile_transpose(a)
    # Compute U st U^T U = A^T + diag(D)
    b = wp.tile_diag_add(a_t, d)
    u = wp.tile_cholesky(b, fill_mode="upper")
    # Solve for x in U^T U x = y
    x = wp.tile_cholesky_solve(u, y, fill_mode="upper")
    # Store U & x
    wp.tile_store(gU, u)
    wp.tile_store(gx, x)


def _test_tile_cholesky_upper_out_of_place(test, device, kernel, multiple_rhs):
    """Shared test logic for tile_cholesky(fill_mode="upper") (out-of-place) with vector or matrix RHS."""
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64)
    D_h = 8.0 * np.ones(TILE_M, dtype=np.float64)

    A_np = A_h.T + np.diag(D_h)
    U_np = np.linalg.cholesky(A_np).T

    if multiple_rhs:
        Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
        X_np = np.linalg.solve(A_np, Y_h.T)
        Z_np = X_np @ X_np

        A_wp = wp.array(A_h, dtype=wp.float64, device=device)
        D_wp = wp.array(D_h, dtype=wp.float64, device=device)
        U_wp = wp.zeros((TILE_M, TILE_M), dtype=wp.float64, device=device)
        Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
        X_wp = wp.zeros_like(Y_wp)
        Z_wp = wp.zeros_like(Y_wp)

        wp.launch_tiled(
            kernel, dim=[1, 1], inputs=[A_wp, D_wp, U_wp, Y_wp, X_wp, Z_wp], block_dim=TILE_DIM, device=device
        )

        np.testing.assert_allclose(U_wp.numpy(), U_np)
        np.testing.assert_allclose(X_wp.numpy(), X_np)
        np.testing.assert_allclose(Z_wp.numpy(), Z_np)
    else:
        Y_h = np.arange(TILE_M, dtype=np.float64)
        X_np = np.linalg.solve(A_np, Y_h)

        A_wp = wp.array(A_h, dtype=wp.float64, device=device)
        D_wp = wp.array(D_h, dtype=wp.float64, device=device)
        U_wp = wp.zeros((TILE_M, TILE_M), dtype=wp.float64, device=device)
        Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
        X_wp = wp.zeros_like(Y_wp)

        wp.launch_tiled(kernel, dim=[1, 1], inputs=[A_wp, D_wp, U_wp, Y_wp, X_wp], block_dim=TILE_DIM, device=device)

        np.testing.assert_allclose(U_wp.numpy(), U_np)
        np.testing.assert_allclose(X_wp.numpy(), X_np)


def test_tile_cholesky_upper(test, device):
    _test_tile_cholesky_upper_out_of_place(test, device, tile_math_cholesky_upper, multiple_rhs=False)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_upper_inplace(
    gA: wp.array2d(dtype=wp.float64),
    gy: wp.array1d(dtype=wp.float64),
):
    # Load A & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    # Compute U st U^T U = A inplace
    wp.tile_cholesky_inplace(a, fill_mode="upper")
    # Solve for x in U^T U x = y inplace
    wp.tile_cholesky_solve_inplace(a, y, fill_mode="upper")
    # Store U & y
    wp.tile_store(gA, a)
    wp.tile_store(gy, y)


def _test_tile_cholesky_upper_inplace(test, device, kernel, multiple_rhs):
    """Shared test logic for tile_cholesky_inplace(fill_mode="upper") with vector or matrix RHS."""
    rng = np.random.default_rng(42)
    L_h = np.tril(rng.random((TILE_M, TILE_M)))
    U_h = L_h.T
    A_h = U_h.T @ U_h

    if multiple_rhs:
        Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
        Y_sol_np = np.linalg.solve(A_h, Y_h.T)
        Z_np = Y_sol_np @ Y_sol_np

        A_wp = wp.array(A_h, dtype=wp.float64, device=device)
        Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
        Z_wp = wp.zeros_like(Y_wp)

        wp.launch_tiled(kernel, dim=[1, 1], inputs=[A_wp, Y_wp, Z_wp], block_dim=TILE_DIM, device=device)

        np.testing.assert_allclose(A_wp.numpy(), U_h)
        np.testing.assert_allclose(Y_wp.numpy(), Y_sol_np)
        np.testing.assert_allclose(Z_wp.numpy(), Z_np)
    else:
        Y_h = np.arange(TILE_M, dtype=np.float64)
        Y_sol_np = np.linalg.solve(A_h, Y_h)

        A_wp = wp.array(A_h, dtype=wp.float64, device=device)
        Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)

        wp.launch_tiled(kernel, dim=[1, 1], inputs=[A_wp, Y_wp], block_dim=TILE_DIM, device=device)

        np.testing.assert_allclose(Y_wp.numpy(), Y_sol_np)
        np.testing.assert_allclose(A_wp.numpy(), U_h)


def test_tile_cholesky_upper_inplace(test, device):
    _test_tile_cholesky_upper_inplace(test, device, tile_math_cholesky_upper_inplace, multiple_rhs=False)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_upper_multiple_rhs(
    gA: wp.array2d(dtype=wp.float64),
    gD: wp.array1d(dtype=wp.float64),
    gU: wp.array2d(dtype=wp.float64),
    gy: wp.array2d(dtype=wp.float64),
    gx: wp.array2d(dtype=wp.float64),
    gz: wp.array2d(dtype=wp.float64),
):
    # Load A, D & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    d = wp.tile_load(gD, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=(TILE_M, TILE_M), storage="shared")
    # Compute U st U^T U = A.T + diag(D)
    a_t = wp.tile_transpose(a)
    b = wp.tile_diag_add(a_t, d)
    u = wp.tile_cholesky(b, fill_mode="upper")
    # Solve for x in U^T U x = y.T
    y_t = wp.tile_transpose(y)
    x = wp.tile_cholesky_solve(u, y_t, fill_mode="upper")
    # Ensure matmul receives correct layout information
    z = wp.tile_matmul(x, x)
    # Store U, x & z
    wp.tile_store(gU, u)
    wp.tile_store(gx, x)
    wp.tile_store(gz, z)


def test_tile_cholesky_upper_multiple_rhs(test, device):
    _test_tile_cholesky_upper_out_of_place(test, device, tile_math_cholesky_upper_multiple_rhs, multiple_rhs=True)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_upper_multiple_rhs_inplace(
    gA: wp.array2d(dtype=wp.float64),
    gy: wp.array2d(dtype=wp.float64),
    gz: wp.array2d(dtype=wp.float64),
):
    # Load A & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=(TILE_M, TILE_M), storage="shared")
    # Compute U st U^T U = A inplace
    wp.tile_cholesky_inplace(a, fill_mode="upper")
    # Solve for x in U^T U x = y.T inplace
    y_t = wp.tile_transpose(y)
    wp.tile_cholesky_solve_inplace(a, y_t, fill_mode="upper")
    # Ensure matmul receives correct layout information
    z = wp.tile_matmul(y_t, y_t)
    # Store U, y & z
    wp.tile_store(gA, a)
    wp.tile_store(gy, y_t)
    wp.tile_store(gz, z)


def test_tile_cholesky_upper_multiple_rhs_inplace(test, device):
    _test_tile_cholesky_upper_inplace(test, device, tile_math_cholesky_upper_multiple_rhs_inplace, multiple_rhs=True)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_solve_upper(
    gU: wp.array2d(dtype=wp.float64),
    gy: wp.array1d(dtype=wp.float64),
    gx: wp.array1d(dtype=wp.float64),
):
    U = wp.tile_load(gU, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    x = wp.tile_cholesky_solve(U, y, fill_mode="upper")
    wp.tile_store(gx, x)


def test_tile_cholesky_solve_upper(test, device):
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64) + 8.0 * np.eye(TILE_M, dtype=np.float64)
    U_np = np.linalg.cholesky(A_h).T  # Upper triangular

    Y_h = np.arange(TILE_M, dtype=np.float64)
    X_np = np.linalg.solve(A_h, Y_h)

    U_wp = wp.array(U_np, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
    X_wp = wp.zeros(TILE_M, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky_solve_upper, dim=[1, 1], inputs=[U_wp, Y_wp, X_wp], block_dim=TILE_DIM, device=device
    )

    np.testing.assert_allclose(X_wp.numpy(), X_np)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_solve_upper_inplace(
    gU: wp.array2d(dtype=wp.float64),
    gy: wp.array1d(dtype=wp.float64),
):
    U = wp.tile_load(gU, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    wp.tile_cholesky_solve_inplace(U, y, fill_mode="upper")
    wp.tile_store(gy, y)


def test_tile_cholesky_solve_upper_inplace(test, device):
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64) + 8.0 * np.eye(TILE_M, dtype=np.float64)
    U_np = np.linalg.cholesky(A_h).T

    Y_h = np.arange(TILE_M, dtype=np.float64)
    X_np = np.linalg.solve(A_h, Y_h)

    U_wp = wp.array(U_np, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky_solve_upper_inplace, dim=[1, 1], inputs=[U_wp, Y_wp], block_dim=TILE_DIM, device=device
    )

    np.testing.assert_allclose(Y_wp.numpy(), X_np)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_solve_upper_multiple_rhs(
    gU: wp.array2d(dtype=wp.float64),
    gY: wp.array2d(dtype=wp.float64),
    gX: wp.array2d(dtype=wp.float64),
):
    U = wp.tile_load(gU, shape=(TILE_M, TILE_M), storage="shared")
    Y = wp.tile_load(gY, shape=(TILE_M, TILE_M), storage="shared")
    X = wp.tile_cholesky_solve(U, Y, fill_mode="upper")
    wp.tile_store(gX, X)


def test_tile_cholesky_solve_upper_multiple_rhs(test, device):
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64) + 8.0 * np.eye(TILE_M, dtype=np.float64)
    U_np = np.linalg.cholesky(A_h).T

    Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
    X_np = np.linalg.solve(A_h, Y_h)

    U_wp = wp.array(U_np, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
    X_wp = wp.zeros((TILE_M, TILE_M), dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky_solve_upper_multiple_rhs,
        dim=[1, 1],
        inputs=[U_wp, Y_wp, X_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    np.testing.assert_allclose(X_wp.numpy(), X_np)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_solve_upper_multiple_rhs_inplace(
    gU: wp.array2d(dtype=wp.float64),
    gY: wp.array2d(dtype=wp.float64),
):
    U = wp.tile_load(gU, shape=(TILE_M, TILE_M), storage="shared")
    Y = wp.tile_load(gY, shape=(TILE_M, TILE_M), storage="shared")
    wp.tile_cholesky_solve_inplace(U, Y, fill_mode="upper")
    wp.tile_store(gY, Y)


def test_tile_cholesky_solve_upper_multiple_rhs_inplace(test, device):
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64) + 8.0 * np.eye(TILE_M, dtype=np.float64)
    U_np = np.linalg.cholesky(A_h).T

    Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
    X_np = np.linalg.solve(A_h, Y_h)

    U_wp = wp.array(U_np, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky_solve_upper_multiple_rhs_inplace,
        dim=[1, 1],
        inputs=[U_wp, Y_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    np.testing.assert_allclose(Y_wp.numpy(), X_np)


all_devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


@unittest.skipUnless(
    not wp._src.context.runtime.core.wp_is_mathdx_enabled()
    or (
        wp._src.context.runtime.core.wp_is_mathdx_enabled()
        and wp._src.context.runtime.core.wp_cuda_toolkit_version() >= 12060
    ),
    "MathDx is not enabled or is enabled but CUDA toolkit version is less than 12.6",
)
class TestTileCholesky(unittest.TestCase):
    pass


add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_cholesky",
    test_tile_cholesky_cholesky,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_cholesky_inplace",
    test_tile_cholesky_cholesky_inplace,
    devices=all_devices,
    check_output=False,
)

add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_cholesky_multiple_rhs",
    test_tile_cholesky_cholesky_multiple_rhs,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_cholesky_multiple_rhs_inplace",
    test_tile_cholesky_cholesky_multiple_rhs_inplace,
    devices=all_devices,
    check_output=False,
)


add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_forward_substitution",
    test_tile_cholesky_forward_substitution,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_back_substitution",
    test_tile_cholesky_back_substitution,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_forward_substitution_multiple_rhs",
    test_tile_cholesky_forward_substitution_multiple_rhs,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_back_substitution_multiple_rhs",
    test_tile_cholesky_back_substitution_multiple_rhs,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_block_cholesky",
    test_tile_cholesky_block_cholesky,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_singular_matrices",
    test_tile_cholesky_singular_matrices,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_upper",
    test_tile_cholesky_upper,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_upper_inplace",
    test_tile_cholesky_upper_inplace,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_upper_multiple_rhs",
    test_tile_cholesky_upper_multiple_rhs,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_upper_multiple_rhs_inplace",
    test_tile_cholesky_upper_multiple_rhs_inplace,
    devices=all_devices,
    check_output=False,
)

add_function_test(
    TestTileCholesky, "test_tile_cholesky_solve_upper", test_tile_cholesky_solve_upper, devices=all_devices
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_solve_upper_inplace",
    test_tile_cholesky_solve_upper_inplace,
    devices=all_devices,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_solve_upper_multiple_rhs",
    test_tile_cholesky_solve_upper_multiple_rhs,
    devices=all_devices,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_solve_upper_multiple_rhs_inplace",
    test_tile_cholesky_solve_upper_multiple_rhs_inplace,
    devices=all_devices,
)


add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_lower_backward_fp32",
    test_tile_cholesky_lower_backward(wp.float32),
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_lower_backward_fp64",
    test_tile_cholesky_lower_backward(wp.float64),
    devices=all_devices,
    check_output=False,
)

add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_upper_backward_fp32",
    test_tile_cholesky_upper_backward(wp.float32),
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_upper_backward_fp64",
    test_tile_cholesky_upper_backward(wp.float64),
    devices=all_devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
