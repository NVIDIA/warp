# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile triangular-solve tests (mathdx / cuSolverDx path).

Tests ``tile_lower_solve``, ``tile_upper_solve``, and their inplace variants
with vector and matrix right-hand sides, plus singular-matrix edge cases.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.init()  # For wp._src.context.runtime.core.wp_is_mathdx_enabled()

TILE_M = wp.constant(8)
TILE_NRHS = wp.constant(4)

# num threads per-tile
TILE_DIM = 32

# Forward-only kernels skip adjoint codegen
wp.get_module("test_solve_fwd").options["enable_backward"] = False


@wp.kernel(module="test_solve_fwd")
def tile_math_forward_substitution(gL: wp.array2d[wp.float64], gx: wp.array1d[wp.float64], gz: wp.array1d[wp.float64]):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=TILE_M, storage="shared")
    # Solve for z in Lz = x
    # Transpose because we loaded an upper triangular matrix
    z = wp.tile_lower_solve(wp.tile_transpose(L), x)
    # Store z
    wp.tile_store(gz, z)


@wp.kernel(module="test_solve_fwd")
def tile_math_forward_substitution_inplace(gL: wp.array2d[wp.float64], gx: wp.array1d[wp.float64]):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=TILE_M, storage="shared")
    # Solve for z in Lz = x inplace
    # Transpose because we loaded an upper triangular matrix
    wp.tile_lower_solve_inplace(wp.tile_transpose(L), x)
    # Store z
    wp.tile_store(gx, x)


def test_tile_solve_forward_substitution(test, device):
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

    # Run inplace kernel
    wp.launch_tiled(
        tile_math_forward_substitution_inplace, dim=[1, 1], inputs=[L_wp, x_wp], block_dim=TILE_DIM, device=device
    )

    # Verify results
    np.testing.assert_allclose(x_wp.numpy(), z_np)


@wp.kernel(module="test_solve_fwd")
def tile_math_back_substitution(gL: wp.array2d[wp.float64], gx: wp.array1d[wp.float64], gz: wp.array1d[wp.float64]):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=TILE_M, storage="shared")
    # Solve for z in L^T z = x
    # Transpose because we loaded a lower triangular matrix
    z = wp.tile_upper_solve(wp.tile_transpose(L), x)
    # Store z
    wp.tile_store(gz, z)


@wp.kernel(module="test_solve_fwd")
def tile_math_back_substitution_inplace(gL: wp.array2d[wp.float64], gx: wp.array1d[wp.float64]):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=TILE_M, storage="shared")
    # Solve for z in L^T z = x inplace
    # Transpose because we loaded a lower triangular matrix
    wp.tile_upper_solve_inplace(wp.tile_transpose(L), x)
    # Store z
    wp.tile_store(gx, x)


def test_tile_solve_back_substitution(test, device):
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


@wp.kernel(module="test_solve_fwd")
def tile_math_forward_substitution_multiple_rhs(
    gL: wp.array2d[wp.float64],
    gx: wp.array2d[wp.float64],
    gz: wp.array2d[wp.float64],
    gc: wp.array2d[wp.float64],
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


@wp.kernel(module="test_solve_fwd")
def tile_math_forward_substitution_multiple_rhs_inplace(
    gL: wp.array2d[wp.float64],
    gx: wp.array2d[wp.float64],
    gc: wp.array2d[wp.float64],
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


def test_tile_solve_forward_substitution_multiple_rhs(test, device):
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


@wp.kernel(module="test_solve_fwd")
def tile_math_back_substitution_multiple_rhs(
    gL: wp.array2d[wp.float64],
    gx: wp.array2d[wp.float64],
    gz: wp.array2d[wp.float64],
    gc: wp.array2d[wp.float64],
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


@wp.kernel(module="test_solve_fwd")
def tile_math_back_substitution_multiple_rhs_inplace(
    gL: wp.array2d[wp.float64],
    gx: wp.array2d[wp.float64],
    gc: wp.array2d[wp.float64],
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


def test_tile_solve_back_substitution_multiple_rhs(test, device):
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


@wp.kernel(module="test_solve_fwd")
def tile_lower_solve_kernel(L: wp.array2d[float], y: wp.array[float], x: wp.array[float]):
    L_tile = wp.tile_load(L, shape=(TILE_M, TILE_M))
    y_tile = wp.tile_load(y, shape=(TILE_M,))
    sol = wp.tile_lower_solve(L_tile, y_tile)
    wp.tile_store(x, sol)


@wp.kernel(module="test_solve_fwd")
def tile_upper_solve_kernel(L: wp.array2d[float], y: wp.array[float], x: wp.array[float]):
    L_tile = wp.tile_load(L, shape=(TILE_M, TILE_M))
    y_tile = wp.tile_load(y, shape=(TILE_M,))
    sol = wp.tile_upper_solve(L_tile, y_tile)
    wp.tile_store(x, sol)


def test_tile_solve_singular_matrices(test, device):
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
        tile_lower_solve_kernel, dim=1, inputs=[L_wp, y_wp], outputs=[x_wp], block_dim=TILE_DIM, device=device
    )

    test.assertFalse(np.isfinite(x_wp.numpy()).all())

    L_np = np.triu(rng.random((TILE_M, TILE_M)))  # Upper triangular matrix
    L_np[-1, -1] = 0.0  # Make it singular

    L_wp = wp.array2d(L_np, dtype=float, device=device)
    y_wp = wp.array(y_np, dtype=float, device=device)
    x_wp = wp.zeros_like(y_wp)

    wp.launch_tiled(
        tile_upper_solve_kernel, dim=1, inputs=[L_wp, y_wp], outputs=[x_wp], block_dim=TILE_DIM, device=device
    )

    test.assertFalse(np.isfinite(x_wp.numpy()).all())


@wp.kernel
def tile_math_lower_solve_backward(gL: wp.array2d[wp.float64], gx: wp.array1d[wp.float64], gz: wp.array1d[wp.float64]):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=TILE_M, storage="shared")
    # Solve for z in Lz = x
    z = wp.tile_lower_solve(L, x)
    # Store z
    wp.tile_store(gz, z)


@wp.kernel
def tile_math_lower_solve_backward_transposed(
    gU: wp.array2d[wp.float64], gx: wp.array1d[wp.float64], gz: wp.array1d[wp.float64]
):
    U = wp.tile_load(gU, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=TILE_M, storage="shared")
    z = wp.tile_lower_solve(wp.tile_transpose(U), x)
    wp.tile_store(gz, z)


@wp.kernel
def tile_math_lower_solve_backward_multiple_rhs(
    gL: wp.array2d[wp.float64], gx: wp.array2d[wp.float64], gz: wp.array2d[wp.float64]
):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=(TILE_M, TILE_NRHS), storage="shared")
    # Solve for z in Lz = x
    z = wp.tile_lower_solve(L, x)
    # Store z
    wp.tile_store(gz, z)


@wp.kernel
def tile_math_lower_solve_backward_multiple_rhs_colmajor(
    gL: wp.array2d[wp.float64], gx: wp.array2d[wp.float64], gz: wp.array2d[wp.float64]
):
    # Load L & x
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=(TILE_NRHS, TILE_M), storage="shared")
    # Solve for z in Lz = x.T
    z = wp.tile_lower_solve(L, wp.tile_transpose(x))
    # Store z
    wp.tile_store(gz, z)


@wp.kernel
def tile_math_lower_solve_backward_multiple_rhs_float32(
    gL: wp.array2d[float], gx: wp.array2d[float], gz: wp.array2d[float]
):
    L = wp.tile_load(gL, shape=(TILE_M, TILE_M), storage="shared")
    x = wp.tile_load(gx, shape=(TILE_M, TILE_NRHS), storage="shared")
    z = wp.tile_lower_solve(L, x)
    wp.tile_store(gz, z)


def lower_solve_adjoint_numpy(L, y, adj_z):
    """Reference gradients for the kernel z = tile_lower_solve(L, y).

    Backward of L z = y:
        w = solve(L^T, adj_z)
        grad_y = w
        grad_L = -tril(outer(w, z))   for a vector RHS
        grad_L = -tril(W @ Z^T)       for a matrix RHS

    Returns (grad_L, grad_y).
    """
    z = np.linalg.solve(L, y)
    w = np.linalg.solve(L.T, adj_z)
    if y.ndim == 1:
        grad_L = -np.tril(np.outer(w, z))
    else:
        grad_L = -np.tril(w @ z.T)
    return grad_L, w


def test_tile_solve_lower_backward(test, device):
    # Create test data: lower triangular, well-conditioned (diagonal bumped away from zero)
    rng = np.random.default_rng(42)
    L_h = np.tril(rng.random((TILE_M, TILE_M))) + 2.0 * np.eye(TILE_M)
    x_h = rng.standard_normal(TILE_M)
    adj_z = rng.standard_normal(TILE_M)

    # Compute reference solution and gradients using NumPy
    z_np = np.linalg.solve(L_h, x_h)
    grad_L_ref, grad_x_ref = lower_solve_adjoint_numpy(L_h, x_h, adj_z)

    # Create Warp arrays
    L_wp = wp.array(L_h, requires_grad=True, dtype=wp.float64, device=device)
    x_wp = wp.array(x_h, requires_grad=True, dtype=wp.float64, device=device)
    z_wp = wp.zeros(TILE_M, requires_grad=True, dtype=wp.float64, device=device)

    # Run forward + backward
    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_math_lower_solve_backward, dim=[1, 1], inputs=[L_wp, x_wp, z_wp], block_dim=TILE_DIM, device=device
        )
    tape.backward(grads={z_wp: wp.array(adj_z, dtype=wp.float64, device=device)})

    # Verify the forward solution and gradients on both inputs.
    np.testing.assert_allclose(z_wp.numpy(), z_np)
    np.testing.assert_allclose(L_wp.grad.numpy(), grad_L_ref)
    np.testing.assert_allclose(x_wp.grad.numpy(), grad_x_ref)
    # grad_L is lower triangular, so its strictly-upper part is zero.
    np.testing.assert_allclose(np.triu(L_wp.grad.numpy(), k=1), 0.0, atol=1e-12)


def test_tile_solve_lower_backward_transposed(test, device):
    rng = np.random.default_rng(43)
    U_h = np.triu(rng.random((TILE_M, TILE_M))) + 2.0 * np.eye(TILE_M)
    x_h = rng.standard_normal(TILE_M)
    adj_z = rng.standard_normal(TILE_M)

    z_ref = np.linalg.solve(U_h.T, x_h)
    grad_L_ref, grad_x_ref = lower_solve_adjoint_numpy(U_h.T, x_h, adj_z)
    grad_U_ref = grad_L_ref.T

    U_wp = wp.array(U_h, requires_grad=True, dtype=wp.float64, device=device)
    x_wp = wp.array(x_h, requires_grad=True, dtype=wp.float64, device=device)
    z_wp = wp.zeros(TILE_M, requires_grad=True, dtype=wp.float64, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_math_lower_solve_backward_transposed,
            dim=[1, 1],
            inputs=[U_wp, x_wp, z_wp],
            block_dim=TILE_DIM,
            device=device,
        )
    tape.backward(grads={z_wp: wp.array(adj_z, dtype=wp.float64, device=device)})

    np.testing.assert_allclose(z_wp.numpy(), z_ref)
    np.testing.assert_allclose(U_wp.grad.numpy(), grad_U_ref)
    np.testing.assert_allclose(x_wp.grad.numpy(), grad_x_ref)
    # grad_U is upper triangular, so its strictly-lower part is zero.
    np.testing.assert_allclose(np.tril(U_wp.grad.numpy(), k=-1), 0.0, atol=1e-12)


def test_tile_solve_lower_backward_multiple_rhs(test, device):
    # Create test data
    rng = np.random.default_rng(7)
    L_h = np.tril(rng.random((TILE_M, TILE_M))) + 2.0 * np.eye(TILE_M)
    x_h = rng.standard_normal((TILE_M, TILE_NRHS))  # Multiple right-hand sides
    adj_z = rng.standard_normal((TILE_M, TILE_NRHS))

    # Compute reference solution and gradients using NumPy
    z_np = np.linalg.solve(L_h, x_h)
    grad_L_ref, grad_x_ref = lower_solve_adjoint_numpy(L_h, x_h, adj_z)

    # Create Warp arrays
    L_wp = wp.array(L_h, requires_grad=True, dtype=wp.float64, device=device)
    x_wp = wp.array(x_h, requires_grad=True, dtype=wp.float64, device=device)
    z_wp = wp.zeros((TILE_M, TILE_NRHS), requires_grad=True, dtype=wp.float64, device=device)

    # Run forward + backward
    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_math_lower_solve_backward_multiple_rhs,
            dim=[1, 1],
            inputs=[L_wp, x_wp, z_wp],
            block_dim=TILE_DIM,
            device=device,
        )
    tape.backward(grads={z_wp: wp.array(adj_z, dtype=wp.float64, device=device)})

    # Verify the forward solution and gradients on both inputs.
    np.testing.assert_allclose(z_wp.numpy(), z_np)
    np.testing.assert_allclose(L_wp.grad.numpy(), grad_L_ref)
    np.testing.assert_allclose(x_wp.grad.numpy(), grad_x_ref)
    # grad_L is lower triangular, so its strictly-upper part is zero.
    np.testing.assert_allclose(np.triu(L_wp.grad.numpy(), k=1), 0.0, atol=1e-12)


def test_tile_solve_lower_backward_multiple_rhs_colmajor(test, device):
    # Create test data
    rng = np.random.default_rng(17)
    L_h = np.tril(rng.random((TILE_M, TILE_M))) + 2.0 * np.eye(TILE_M)
    x_h = rng.standard_normal((TILE_NRHS, TILE_M))
    y_h = x_h.T
    adj_z = rng.standard_normal((TILE_M, TILE_NRHS))

    # Compute reference solution and gradients using NumPy
    z_np = np.linalg.solve(L_h, y_h)
    grad_L_ref, grad_y_ref = lower_solve_adjoint_numpy(L_h, y_h, adj_z)

    # Create Warp arrays
    L_wp = wp.array(L_h, requires_grad=True, dtype=wp.float64, device=device)
    x_wp = wp.array(x_h, requires_grad=True, dtype=wp.float64, device=device)
    z_wp = wp.zeros((TILE_M, TILE_NRHS), requires_grad=True, dtype=wp.float64, device=device)

    # Run forward + backward
    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_math_lower_solve_backward_multiple_rhs_colmajor,
            dim=[1, 1],
            inputs=[L_wp, x_wp, z_wp],
            block_dim=TILE_DIM,
            device=device,
        )
    tape.backward(grads={z_wp: wp.array(adj_z, dtype=wp.float64, device=device)})

    # Verify the forward solution and gradients on both inputs.
    np.testing.assert_allclose(z_wp.numpy(), z_np)
    np.testing.assert_allclose(L_wp.grad.numpy(), grad_L_ref)
    np.testing.assert_allclose(x_wp.grad.numpy(), grad_y_ref.T)
    # grad_L is lower triangular, so its strictly-upper part is zero.
    np.testing.assert_allclose(np.triu(L_wp.grad.numpy(), k=1), 0.0, atol=1e-12)


def test_tile_solve_lower_backward_multiple_rhs_float32(test, device):
    rng = np.random.default_rng(23)
    L_h = (np.tril(rng.random((TILE_M, TILE_M))) + 2.0 * np.eye(TILE_M)).astype(np.float32)
    x_h = rng.standard_normal((TILE_M, TILE_NRHS)).astype(np.float32)
    adj_z = rng.standard_normal((TILE_M, TILE_NRHS)).astype(np.float32)

    z_ref = np.linalg.solve(L_h, x_h)
    grad_x_ref = np.linalg.solve(L_h.T, adj_z)
    grad_L_ref = -np.tril(grad_x_ref @ z_ref.T)

    L_wp = wp.array(L_h, requires_grad=True, dtype=float, device=device)
    x_wp = wp.array(x_h, requires_grad=True, dtype=float, device=device)
    z_wp = wp.zeros((TILE_M, TILE_NRHS), requires_grad=True, dtype=float, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_math_lower_solve_backward_multiple_rhs_float32,
            dim=[1, 1],
            inputs=[L_wp, x_wp, z_wp],
            block_dim=TILE_DIM,
            device=device,
        )
    tape.backward(grads={z_wp: wp.array(adj_z, dtype=float, device=device)})

    np.testing.assert_allclose(z_wp.numpy(), z_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(L_wp.grad.numpy(), grad_L_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(x_wp.grad.numpy(), grad_x_ref, rtol=1e-5, atol=1e-5)


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
class TestTileSolve(unittest.TestCase):
    pass


add_function_test(
    TestTileSolve,
    "test_tile_solve_forward_substitution",
    test_tile_solve_forward_substitution,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileSolve,
    "test_tile_solve_back_substitution",
    test_tile_solve_back_substitution,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileSolve,
    "test_tile_solve_forward_substitution_multiple_rhs",
    test_tile_solve_forward_substitution_multiple_rhs,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileSolve,
    "test_tile_solve_back_substitution_multiple_rhs",
    test_tile_solve_back_substitution_multiple_rhs,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileSolve,
    "test_tile_solve_singular_matrices",
    test_tile_solve_singular_matrices,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileSolve,
    "test_tile_solve_lower_backward",
    test_tile_solve_lower_backward,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileSolve,
    "test_tile_solve_lower_backward_transposed",
    test_tile_solve_lower_backward_transposed,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileSolve,
    "test_tile_solve_lower_backward_multiple_rhs",
    test_tile_solve_lower_backward_multiple_rhs,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileSolve,
    "test_tile_solve_lower_backward_multiple_rhs_colmajor",
    test_tile_solve_lower_backward_multiple_rhs_colmajor,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileSolve,
    "test_tile_solve_lower_backward_multiple_rhs_float32",
    test_tile_solve_lower_backward_multiple_rhs_float32,
    devices=cuda_devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
