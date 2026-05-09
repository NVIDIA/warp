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


class TestTileSolveNoMathdx(unittest.TestCase):
    pass


_devices = get_test_devices()

for name, func in [
    ("test_lower_solve_vector", test_lower_solve_vector),
    ("test_lower_solve_matrix", test_lower_solve_matrix),
    ("test_lower_solve_inplace_vector", test_lower_solve_inplace_vector),
    ("test_lower_solve_inplace_matrix", test_lower_solve_inplace_matrix),
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
    add_function_test(TestTileSolveNoMathdx, name, func, devices=_devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
