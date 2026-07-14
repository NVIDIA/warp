# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile triangular-solve tests with mathdx solver disabled (cooperative scalar fallback path).

Setting ``enable_mathdx_solver=False`` at module scope routes
``tile_cholesky_solve``, ``tile_lower_solve``, and ``tile_upper_solve`` (and
their inplace variants) through the cooperative scalar substitution
primitives in ``tile_solve.h`` on GPU (or the CPU sequential branch on CPU),
exercising the path that runs whenever Warp is built without libmathdx or
when a user disables the option per-module.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# Disable mathdx solver ops (Cholesky and triangular solves) for all kernels
# defined in this module.
wp.set_module_options({"enable_mathdx_solver": False})

TILE_DIM = 32
N = 8
M_RHS = 4


# -----------------------------------------------------------------------------
# tile_lower_solve  (Lz = y)
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def tile_lower_solve_vec_kernel(gL: wp.array2d[wp.float64], gy: wp.array1d[wp.float64], gz: wp.array1d[wp.float64]):
    L = wp.tile_load(gL, shape=(N, N))
    y = wp.tile_load(gy, shape=N)
    z = wp.tile_lower_solve(L, y)
    wp.tile_store(gz, z)


@wp.kernel(enable_backward=False)
def tile_lower_solve_mat_kernel(gL: wp.array2d[wp.float64], gy: wp.array2d[wp.float64], gz: wp.array2d[wp.float64]):
    L = wp.tile_load(gL, shape=(N, N))
    y = wp.tile_load(gy, shape=(N, M_RHS))
    z = wp.tile_lower_solve(L, y)
    wp.tile_store(gz, z)


@wp.kernel(enable_backward=False)
def tile_lower_solve_inplace_vec_kernel(gL: wp.array2d[wp.float64], gy: wp.array1d[wp.float64]):
    L = wp.tile_load(gL, shape=(N, N))
    y = wp.tile_load(gy, shape=N)
    wp.tile_lower_solve_inplace(L, y)
    wp.tile_store(gy, y)


@wp.kernel(enable_backward=False)
def tile_lower_solve_inplace_mat_kernel(gL: wp.array2d[wp.float64], gy: wp.array2d[wp.float64]):
    L = wp.tile_load(gL, shape=(N, N))
    y = wp.tile_load(gy, shape=(N, M_RHS))
    wp.tile_lower_solve_inplace(L, y)
    wp.tile_store(gy, y)


# -----------------------------------------------------------------------------
# tile_upper_solve  (Ux = z)
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def tile_upper_solve_vec_kernel(gU: wp.array2d[wp.float64], gz: wp.array1d[wp.float64], gx: wp.array1d[wp.float64]):
    U = wp.tile_load(gU, shape=(N, N))
    z = wp.tile_load(gz, shape=N)
    x = wp.tile_upper_solve(U, z)
    wp.tile_store(gx, x)


@wp.kernel(enable_backward=False)
def tile_upper_solve_mat_kernel(gU: wp.array2d[wp.float64], gz: wp.array2d[wp.float64], gx: wp.array2d[wp.float64]):
    U = wp.tile_load(gU, shape=(N, N))
    z = wp.tile_load(gz, shape=(N, M_RHS))
    x = wp.tile_upper_solve(U, z)
    wp.tile_store(gx, x)


@wp.kernel(enable_backward=False)
def tile_upper_solve_inplace_vec_kernel(gU: wp.array2d[wp.float64], gz: wp.array1d[wp.float64]):
    U = wp.tile_load(gU, shape=(N, N))
    z = wp.tile_load(gz, shape=N)
    wp.tile_upper_solve_inplace(U, z)
    wp.tile_store(gz, z)


@wp.kernel(enable_backward=False)
def tile_upper_solve_inplace_mat_kernel(gU: wp.array2d[wp.float64], gz: wp.array2d[wp.float64]):
    U = wp.tile_load(gU, shape=(N, N))
    z = wp.tile_load(gz, shape=(N, M_RHS))
    wp.tile_upper_solve_inplace(U, z)
    wp.tile_store(gz, z)


# -----------------------------------------------------------------------------
# tile_cholesky_solve  (LL^T x = y, given L) -- exercises both forward and
# back substitution composed.
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def tile_cholesky_solve_vec_kernel(gL: wp.array2d[wp.float64], gy: wp.array1d[wp.float64], gx: wp.array1d[wp.float64]):
    L = wp.tile_load(gL, shape=(N, N))
    y = wp.tile_load(gy, shape=N)
    x = wp.tile_cholesky_solve(L, y)
    wp.tile_store(gx, x)


@wp.kernel(enable_backward=False)
def tile_cholesky_solve_mat_kernel(gL: wp.array2d[wp.float64], gy: wp.array2d[wp.float64], gx: wp.array2d[wp.float64]):
    L = wp.tile_load(gL, shape=(N, N))
    y = wp.tile_load(gy, shape=(N, M_RHS))
    x = wp.tile_cholesky_solve(L, y)
    wp.tile_store(gx, x)


@wp.kernel(enable_backward=False)
def tile_cholesky_solve_inplace_vec_kernel(gL: wp.array2d[wp.float64], gy: wp.array1d[wp.float64]):
    L = wp.tile_load(gL, shape=(N, N))
    y = wp.tile_load(gy, shape=N)
    wp.tile_cholesky_solve_inplace(L, y)
    wp.tile_store(gy, y)


@wp.kernel(enable_backward=False)
def tile_cholesky_solve_inplace_mat_kernel(gL: wp.array2d[wp.float64], gy: wp.array2d[wp.float64]):
    L = wp.tile_load(gL, shape=(N, N))
    y = wp.tile_load(gy, shape=(N, M_RHS))
    wp.tile_cholesky_solve_inplace(L, y)
    wp.tile_store(gy, y)


@wp.kernel
def tile_lower_solve_backward_vec_kernel(
    gL: wp.array2d[wp.float64], gy: wp.array1d[wp.float64], gz: wp.array1d[wp.float64]
):
    L = wp.tile_load(gL, shape=(N, N))
    y = wp.tile_load(gy, shape=N)
    z = wp.tile_lower_solve(L, y)
    wp.tile_store(gz, z)


@wp.kernel
def tile_lower_solve_backward_transposed_vec_kernel(
    gU: wp.array2d[wp.float64], gy: wp.array1d[wp.float64], gz: wp.array1d[wp.float64]
):
    U = wp.tile_load(gU, shape=(N, N))
    y = wp.tile_load(gy, shape=N)
    z = wp.tile_lower_solve(wp.tile_transpose(U), y)
    wp.tile_store(gz, z)


@wp.kernel
def tile_lower_solve_backward_mat_kernel(
    gL: wp.array2d[wp.float64], gy: wp.array2d[wp.float64], gz: wp.array2d[wp.float64]
):
    L = wp.tile_load(gL, shape=(N, N))
    y = wp.tile_load(gy, shape=(N, M_RHS))
    z = wp.tile_lower_solve(L, y)
    wp.tile_store(gz, z)


@wp.kernel
def tile_lower_solve_backward_mat_colmajor_kernel(
    gL: wp.array2d[wp.float64], gy: wp.array2d[wp.float64], gz: wp.array2d[wp.float64]
):
    L = wp.tile_load(gL, shape=(N, N))
    y = wp.tile_load(gy, shape=(M_RHS, N))
    z = wp.tile_lower_solve(L, wp.tile_transpose(y))
    wp.tile_store(gz, z)


@wp.kernel
def tile_lower_solve_backward_mat_float32_kernel(gL: wp.array2d[float], gy: wp.array2d[float], gz: wp.array2d[float]):
    L = wp.tile_load(gL, shape=(N, N))
    y = wp.tile_load(gy, shape=(N, M_RHS))
    z = wp.tile_lower_solve(L, y)
    wp.tile_store(gz, z)


# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------


def _spd_lower_factor(n, seed=0):
    """Return a lower-triangular L such that A = L L^T is SPD (well-conditioned)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    A = A @ A.T + n * np.eye(n)
    return np.linalg.cholesky(A)  # lower


def _make_arr(arr_np, dtype, device):
    return wp.array(arr_np.astype(np.float64), dtype=dtype, device=device)


# -----------------------------------------------------------------------------
# Test functions (parameterised over device)
# -----------------------------------------------------------------------------


def test_lower_solve_vector(test, device):
    L_np = _spd_lower_factor(N, seed=1)
    y_np = np.arange(1.0, N + 1.0)
    z_ref = np.linalg.solve(L_np, y_np)

    L = _make_arr(L_np, wp.float64, device)
    y = _make_arr(y_np, wp.float64, device)
    z = wp.zeros(N, dtype=wp.float64, device=device)

    wp.launch_tiled(tile_lower_solve_vec_kernel, dim=[1], inputs=[L, y, z], block_dim=TILE_DIM, device=device)
    assert_np_equal(z.numpy(), z_ref, tol=1e-10)


def test_lower_solve_matrix(test, device):
    L_np = _spd_lower_factor(N, seed=2)
    y_np = np.arange(1.0, N * M_RHS + 1.0).reshape(N, M_RHS)
    z_ref = np.linalg.solve(L_np, y_np)

    L = _make_arr(L_np, wp.float64, device)
    y = _make_arr(y_np, wp.float64, device)
    z = wp.zeros((N, M_RHS), dtype=wp.float64, device=device)

    wp.launch_tiled(tile_lower_solve_mat_kernel, dim=[1], inputs=[L, y, z], block_dim=TILE_DIM, device=device)
    assert_np_equal(z.numpy(), z_ref, tol=1e-10)


def test_lower_solve_inplace_vector(test, device):
    L_np = _spd_lower_factor(N, seed=3)
    y_np = np.arange(1.0, N + 1.0)
    z_ref = np.linalg.solve(L_np, y_np)

    L = _make_arr(L_np, wp.float64, device)
    y = _make_arr(y_np, wp.float64, device)

    wp.launch_tiled(tile_lower_solve_inplace_vec_kernel, dim=[1], inputs=[L, y], block_dim=TILE_DIM, device=device)
    assert_np_equal(y.numpy(), z_ref, tol=1e-10)


def test_lower_solve_inplace_matrix(test, device):
    L_np = _spd_lower_factor(N, seed=4)
    y_np = np.arange(1.0, N * M_RHS + 1.0).reshape(N, M_RHS)
    z_ref = np.linalg.solve(L_np, y_np)

    L = _make_arr(L_np, wp.float64, device)
    y = _make_arr(y_np, wp.float64, device)

    wp.launch_tiled(tile_lower_solve_inplace_mat_kernel, dim=[1], inputs=[L, y], block_dim=TILE_DIM, device=device)
    assert_np_equal(y.numpy(), z_ref, tol=1e-10)


def _lower_solve_adjoint_numpy(L, y, adj_z):
    """Reference gradients for the kernel z = tile_lower_solve(L, y).

    ``L`` is lower triangular and loaded directly (no transpose), so the
    gradient lands on L itself. Backward of L z = y:
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


def _lower_solve_backward(L_np, y_np, adj_z, device):
    """Run tile_lower_solve forward+backward; return (z, grad_L, grad_y) as NumPy."""
    L_wp = wp.array(L_np, dtype=wp.float64, requires_grad=True, device=device)
    y_wp = wp.array(y_np, dtype=wp.float64, requires_grad=True, device=device)
    z_wp = wp.zeros(y_np.shape, dtype=wp.float64, requires_grad=True, device=device)

    kernel = tile_lower_solve_backward_vec_kernel if y_np.ndim == 1 else tile_lower_solve_backward_mat_kernel

    with wp.Tape() as tape:
        wp.launch_tiled(kernel, dim=[1], inputs=[L_wp, y_wp, z_wp], block_dim=TILE_DIM, device=device)

    tape.backward(grads={z_wp: wp.array(adj_z, dtype=wp.float64, device=device)})
    return z_wp.numpy(), L_wp.grad.numpy(), y_wp.grad.numpy()


def _lower_solve_backward_colmajor(L_np, y_np, adj_z, device):
    """Run backward with logical ``y`` represented by a transposed tile view."""
    L_wp = wp.array(L_np, dtype=wp.float64, requires_grad=True, device=device)
    y_wp = wp.array(y_np.T.copy(), dtype=wp.float64, requires_grad=True, device=device)
    z_wp = wp.zeros(y_np.shape, dtype=wp.float64, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_lower_solve_backward_mat_colmajor_kernel,
            dim=[1],
            inputs=[L_wp, y_wp, z_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    tape.backward(grads={z_wp: wp.array(adj_z, dtype=wp.float64, device=device)})
    return z_wp.numpy(), L_wp.grad.numpy(), y_wp.grad.numpy()


def test_lower_solve_backward_vector(test, device):
    L_np = _spd_lower_factor(N, seed=10)
    rng = np.random.default_rng(11)
    y_np = rng.standard_normal(N)
    adj_z = rng.standard_normal(N)

    z, grad_L, grad_y = _lower_solve_backward(L_np, y_np, adj_z, device)
    z_ref = np.linalg.solve(L_np, y_np)
    grad_L_ref, grad_y_ref = _lower_solve_adjoint_numpy(L_np, y_np, adj_z)

    assert_np_equal(z, z_ref, tol=1e-10)
    assert_np_equal(grad_L, grad_L_ref, tol=1e-8)
    assert_np_equal(grad_y, grad_y_ref, tol=1e-8)
    # grad_L is lower triangular, so its strictly-upper part is zero.
    assert_np_equal(np.triu(grad_L, k=1), np.zeros((N, N)), tol=1e-12)


def test_lower_solve_backward_transposed_vector(test, device):
    U_np = _spd_lower_factor(N, seed=18).T
    rng = np.random.default_rng(19)
    y_np = rng.standard_normal(N)
    adj_z = rng.standard_normal(N)

    U_wp = wp.array(U_np, dtype=wp.float64, requires_grad=True, device=device)
    y_wp = wp.array(y_np, dtype=wp.float64, requires_grad=True, device=device)
    z_wp = wp.zeros(N, dtype=wp.float64, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_lower_solve_backward_transposed_vec_kernel,
            dim=[1],
            inputs=[U_wp, y_wp, z_wp],
            block_dim=TILE_DIM,
            device=device,
        )
    tape.backward(grads={z_wp: wp.array(adj_z, dtype=wp.float64, device=device)})

    z_ref = np.linalg.solve(U_np.T, y_np)
    grad_L_ref, grad_y_ref = _lower_solve_adjoint_numpy(U_np.T, y_np, adj_z)
    grad_U_ref = grad_L_ref.T

    assert_np_equal(z_wp.numpy(), z_ref, tol=1e-10)
    assert_np_equal(U_wp.grad.numpy(), grad_U_ref, tol=1e-8)
    assert_np_equal(y_wp.grad.numpy(), grad_y_ref, tol=1e-8)
    # grad_U is upper triangular, so its strictly-lower part is zero.
    assert_np_equal(np.tril(U_wp.grad.numpy(), k=-1), np.zeros((N, N)), tol=1e-12)


def test_lower_solve_backward_matrix(test, device):
    L_np = _spd_lower_factor(N, seed=12)
    rng = np.random.default_rng(13)
    y_np = rng.standard_normal((N, M_RHS))
    adj_z = rng.standard_normal((N, M_RHS))

    z, grad_L, grad_y = _lower_solve_backward(L_np, y_np, adj_z, device)
    z_ref = np.linalg.solve(L_np, y_np)
    grad_L_ref, grad_y_ref = _lower_solve_adjoint_numpy(L_np, y_np, adj_z)

    assert_np_equal(z, z_ref, tol=1e-10)
    assert_np_equal(grad_L, grad_L_ref, tol=1e-8)
    assert_np_equal(grad_y, grad_y_ref, tol=1e-8)
    # grad_L is lower triangular, so its strictly-upper part is zero.
    assert_np_equal(np.triu(grad_L, k=1), np.zeros((N, N)), tol=1e-12)


def test_lower_solve_backward_matrix_colmajor(test, device):
    L_np = _spd_lower_factor(N, seed=14)
    rng = np.random.default_rng(15)
    y_np = rng.standard_normal((N, M_RHS))
    adj_z = rng.standard_normal((N, M_RHS))

    z, grad_L, grad_y_base = _lower_solve_backward_colmajor(L_np, y_np, adj_z, device)
    z_ref = np.linalg.solve(L_np, y_np)
    grad_L_ref, grad_y_ref = _lower_solve_adjoint_numpy(L_np, y_np, adj_z)

    assert_np_equal(z, z_ref, tol=1e-10)
    assert_np_equal(grad_L, grad_L_ref, tol=1e-8)
    assert_np_equal(grad_y_base, grad_y_ref.T, tol=1e-8)
    # grad_L is lower triangular, so its strictly-upper part is zero.
    assert_np_equal(np.triu(grad_L, k=1), np.zeros((N, N)), tol=1e-12)


def test_lower_solve_backward_matrix_float32(test, device):
    L_np = _spd_lower_factor(N, seed=16).astype(np.float32)
    rng = np.random.default_rng(17)
    y_np = rng.standard_normal((N, M_RHS)).astype(np.float32)
    adj_z = rng.standard_normal((N, M_RHS)).astype(np.float32)

    L_wp = wp.array(L_np, dtype=float, requires_grad=True, device=device)
    y_wp = wp.array(y_np, dtype=float, requires_grad=True, device=device)
    z_wp = wp.zeros(y_np.shape, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_lower_solve_backward_mat_float32_kernel,
            dim=[1],
            inputs=[L_wp, y_wp, z_wp],
            block_dim=TILE_DIM,
            device=device,
        )
    tape.backward(grads={z_wp: wp.array(adj_z, dtype=float, device=device)})

    z_ref = np.linalg.solve(L_np, y_np)
    grad_L_ref, grad_y_ref = _lower_solve_adjoint_numpy(L_np, y_np, adj_z)
    np.testing.assert_allclose(z_wp.numpy(), z_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(L_wp.grad.numpy(), grad_L_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_wp.grad.numpy(), grad_y_ref, rtol=1e-5, atol=1e-5)


def test_upper_solve_vector(test, device):
    L_np = _spd_lower_factor(N, seed=5)
    U_np = L_np.T  # upper triangular
    z_np = np.arange(1.0, N + 1.0)
    x_ref = np.linalg.solve(U_np, z_np)

    U = _make_arr(U_np, wp.float64, device)
    z = _make_arr(z_np, wp.float64, device)
    x = wp.zeros(N, dtype=wp.float64, device=device)

    wp.launch_tiled(tile_upper_solve_vec_kernel, dim=[1], inputs=[U, z, x], block_dim=TILE_DIM, device=device)
    assert_np_equal(x.numpy(), x_ref, tol=1e-10)


def test_upper_solve_matrix(test, device):
    L_np = _spd_lower_factor(N, seed=6)
    U_np = L_np.T
    z_np = np.arange(1.0, N * M_RHS + 1.0).reshape(N, M_RHS)
    x_ref = np.linalg.solve(U_np, z_np)

    U = _make_arr(U_np, wp.float64, device)
    z = _make_arr(z_np, wp.float64, device)
    x = wp.zeros((N, M_RHS), dtype=wp.float64, device=device)

    wp.launch_tiled(tile_upper_solve_mat_kernel, dim=[1], inputs=[U, z, x], block_dim=TILE_DIM, device=device)
    assert_np_equal(x.numpy(), x_ref, tol=1e-10)


def test_upper_solve_inplace_vector(test, device):
    L_np = _spd_lower_factor(N, seed=7)
    U_np = L_np.T
    z_np = np.arange(1.0, N + 1.0)
    x_ref = np.linalg.solve(U_np, z_np)

    U = _make_arr(U_np, wp.float64, device)
    z = _make_arr(z_np, wp.float64, device)

    wp.launch_tiled(tile_upper_solve_inplace_vec_kernel, dim=[1], inputs=[U, z], block_dim=TILE_DIM, device=device)
    assert_np_equal(z.numpy(), x_ref, tol=1e-10)


def test_upper_solve_inplace_matrix(test, device):
    L_np = _spd_lower_factor(N, seed=8)
    U_np = L_np.T
    z_np = np.arange(1.0, N * M_RHS + 1.0).reshape(N, M_RHS)
    x_ref = np.linalg.solve(U_np, z_np)

    U = _make_arr(U_np, wp.float64, device)
    z = _make_arr(z_np, wp.float64, device)

    wp.launch_tiled(tile_upper_solve_inplace_mat_kernel, dim=[1], inputs=[U, z], block_dim=TILE_DIM, device=device)
    assert_np_equal(z.numpy(), x_ref, tol=1e-10)


def test_cholesky_solve_vector(test, device):
    L_np = _spd_lower_factor(N, seed=9)
    A_np = L_np @ L_np.T
    y_np = np.arange(1.0, N + 1.0)
    x_ref = np.linalg.solve(A_np, y_np)

    L = _make_arr(L_np, wp.float64, device)
    y = _make_arr(y_np, wp.float64, device)
    x = wp.zeros(N, dtype=wp.float64, device=device)

    wp.launch_tiled(tile_cholesky_solve_vec_kernel, dim=[1], inputs=[L, y, x], block_dim=TILE_DIM, device=device)
    assert_np_equal(x.numpy(), x_ref, tol=1e-10)


def test_cholesky_solve_matrix(test, device):
    L_np = _spd_lower_factor(N, seed=10)
    A_np = L_np @ L_np.T
    y_np = np.arange(1.0, N * M_RHS + 1.0).reshape(N, M_RHS)
    x_ref = np.linalg.solve(A_np, y_np)

    L = _make_arr(L_np, wp.float64, device)
    y = _make_arr(y_np, wp.float64, device)
    x = wp.zeros((N, M_RHS), dtype=wp.float64, device=device)

    wp.launch_tiled(tile_cholesky_solve_mat_kernel, dim=[1], inputs=[L, y, x], block_dim=TILE_DIM, device=device)
    assert_np_equal(x.numpy(), x_ref, tol=1e-10)


def test_cholesky_solve_inplace_vector(test, device):
    L_np = _spd_lower_factor(N, seed=11)
    A_np = L_np @ L_np.T
    y_np = np.arange(1.0, N + 1.0)
    x_ref = np.linalg.solve(A_np, y_np)

    L = _make_arr(L_np, wp.float64, device)
    y = _make_arr(y_np, wp.float64, device)

    wp.launch_tiled(tile_cholesky_solve_inplace_vec_kernel, dim=[1], inputs=[L, y], block_dim=TILE_DIM, device=device)
    assert_np_equal(y.numpy(), x_ref, tol=1e-10)


def test_cholesky_solve_inplace_matrix(test, device):
    L_np = _spd_lower_factor(N, seed=12)
    A_np = L_np @ L_np.T
    y_np = np.arange(1.0, N * M_RHS + 1.0).reshape(N, M_RHS)
    x_ref = np.linalg.solve(A_np, y_np)

    L = _make_arr(L_np, wp.float64, device)
    y = _make_arr(y_np, wp.float64, device)

    wp.launch_tiled(tile_cholesky_solve_inplace_mat_kernel, dim=[1], inputs=[L, y], block_dim=TILE_DIM, device=device)
    assert_np_equal(y.numpy(), x_ref, tol=1e-10)


# -----------------------------------------------------------------------------
# block_dim == 1 smoke (GPU codegen for single-thread-block must compile and
# produce correct results -- the cooperative scalar path's thread-strided loops
# should collapse to sequential, exercising a different codegen path from
# block_dim == TILE_DIM. CPU coverage is incidental via the per-device fanout
# of the regular tests.)
# -----------------------------------------------------------------------------


def test_lower_solve_block_dim_1(test, device):
    L_np = _spd_lower_factor(N, seed=30)
    y_np = np.arange(1.0, N + 1.0)
    z_ref = np.linalg.solve(L_np, y_np)

    L = _make_arr(L_np, wp.float64, device)
    y = _make_arr(y_np, wp.float64, device)
    z = wp.zeros(N, dtype=wp.float64, device=device)

    wp.launch_tiled(tile_lower_solve_vec_kernel, dim=[1], inputs=[L, y, z], block_dim=1, device=device)
    assert_np_equal(z.numpy(), z_ref, tol=1e-10)


def test_cholesky_solve_matrix_block_dim_1(test, device):
    L_np = _spd_lower_factor(N, seed=31)
    A_np = L_np @ L_np.T
    y_np = np.arange(1.0, N * M_RHS + 1.0).reshape(N, M_RHS)
    x_ref = np.linalg.solve(A_np, y_np)

    L = _make_arr(L_np, wp.float64, device)
    y = _make_arr(y_np, wp.float64, device)
    x = wp.zeros((N, M_RHS), dtype=wp.float64, device=device)

    wp.launch_tiled(tile_cholesky_solve_mat_kernel, dim=[1], inputs=[L, y, x], block_dim=1, device=device)
    assert_np_equal(x.numpy(), x_ref, tol=1e-10)


# -----------------------------------------------------------------------------
# Suite registration
# -----------------------------------------------------------------------------


class TestTileSolveNoMathDx(unittest.TestCase):
    pass


_devices = get_test_devices()

for name, func in [
    ("test_lower_solve_vector", test_lower_solve_vector),
    ("test_lower_solve_matrix", test_lower_solve_matrix),
    ("test_lower_solve_inplace_vector", test_lower_solve_inplace_vector),
    ("test_lower_solve_inplace_matrix", test_lower_solve_inplace_matrix),
    ("test_lower_solve_backward_vector", test_lower_solve_backward_vector),
    ("test_lower_solve_backward_transposed_vector", test_lower_solve_backward_transposed_vector),
    ("test_lower_solve_backward_matrix", test_lower_solve_backward_matrix),
    ("test_lower_solve_backward_matrix_colmajor", test_lower_solve_backward_matrix_colmajor),
    ("test_lower_solve_backward_matrix_float32", test_lower_solve_backward_matrix_float32),
    ("test_upper_solve_vector", test_upper_solve_vector),
    ("test_upper_solve_matrix", test_upper_solve_matrix),
    ("test_upper_solve_inplace_vector", test_upper_solve_inplace_vector),
    ("test_upper_solve_inplace_matrix", test_upper_solve_inplace_matrix),
    ("test_cholesky_solve_vector", test_cholesky_solve_vector),
    ("test_cholesky_solve_matrix", test_cholesky_solve_matrix),
    ("test_cholesky_solve_inplace_vector", test_cholesky_solve_inplace_vector),
    ("test_cholesky_solve_inplace_matrix", test_cholesky_solve_inplace_matrix),
    ("test_lower_solve_block_dim_1", test_lower_solve_block_dim_1),
    ("test_cholesky_solve_matrix_block_dim_1", test_cholesky_solve_matrix_block_dim_1),
]:
    add_function_test(TestTileSolveNoMathDx, name, func, devices=_devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
