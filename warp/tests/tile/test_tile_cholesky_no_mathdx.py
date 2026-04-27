# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile Cholesky tests with mathdx Cholesky disabled (cooperative scalar fallback path).

Setting ``enable_mathdx_cholesky=False`` at module scope routes
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

wp.init()

# Disable mathdx Cholesky for all kernels defined in this module. trsm stays
# enabled (defaults to True) so tile_cholesky_solve in this file runs through
# libmathdx; trsm coverage is in test_tile_solve_no_mathdx.py.
wp.set_module_options({"enable_mathdx_cholesky": False, "enable_backward": False})

TILE_DIM = 32
N = 8


# -----------------------------------------------------------------------------
# Lower Cholesky factorization
# -----------------------------------------------------------------------------


@wp.kernel
def tile_cholesky_lower_kernel(gA: wp.array2d(dtype=wp.float64), gL: wp.array2d(dtype=wp.float64)):
    A = wp.tile_load(gA, shape=(N, N))
    L = wp.tile_cholesky(A)
    wp.tile_store(gL, L)


@wp.kernel
def tile_cholesky_lower_inplace_kernel(gA: wp.array2d(dtype=wp.float64)):
    A = wp.tile_load(gA, shape=(N, N))
    wp.tile_cholesky_inplace(A)
    wp.tile_store(gA, A)


# -----------------------------------------------------------------------------
# Upper Cholesky factorization
# -----------------------------------------------------------------------------


@wp.kernel
def tile_cholesky_upper_kernel(gA: wp.array2d(dtype=wp.float64), gU: wp.array2d(dtype=wp.float64)):
    A = wp.tile_load(gA, shape=(N, N))
    U = wp.tile_cholesky(A, fill_mode="upper")
    wp.tile_store(gU, U)


@wp.kernel
def tile_cholesky_upper_inplace_kernel(gA: wp.array2d(dtype=wp.float64)):
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
    np.testing.assert_allclose(L.numpy(), L_ref, rtol=1e-10)


def test_cholesky_lower_inplace(test, device):
    A_np, L_ref = _spd(N, seed=2)

    A = wp.array(A_np.copy(), dtype=wp.float64, device=device)
    wp.launch_tiled(tile_cholesky_lower_inplace_kernel, dim=[1], inputs=[A], block_dim=TILE_DIM, device=device)
    # The lower-triangular factor lives in the lower triangle (upper zeroed).
    A_out = A.numpy()
    np.testing.assert_allclose(np.tril(A_out), L_ref, rtol=1e-10)
    np.testing.assert_allclose(np.triu(A_out, k=1), np.zeros_like(A_out[np.triu_indices(N, k=1)]).reshape(-1)[0])


def test_cholesky_upper(test, device):
    A_np, L_ref = _spd(N, seed=3)
    U_ref = L_ref.T  # A = U^T U with U = L^T

    A = wp.array(A_np, dtype=wp.float64, device=device)
    U = wp.zeros((N, N), dtype=wp.float64, device=device)

    wp.launch_tiled(tile_cholesky_upper_kernel, dim=[1], inputs=[A, U], block_dim=TILE_DIM, device=device)
    np.testing.assert_allclose(U.numpy(), U_ref, rtol=1e-10)


def test_cholesky_upper_inplace(test, device):
    A_np, L_ref = _spd(N, seed=4)
    U_ref = L_ref.T

    A = wp.array(A_np.copy(), dtype=wp.float64, device=device)
    wp.launch_tiled(tile_cholesky_upper_inplace_kernel, dim=[1], inputs=[A], block_dim=TILE_DIM, device=device)
    A_out = A.numpy()
    np.testing.assert_allclose(np.triu(A_out), U_ref, rtol=1e-10)


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
]:
    add_function_test(TestTileCholeskyNoMathdx, name, func, devices=_devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
