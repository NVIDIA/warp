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

    # TODO: implement and test backward pass

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


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
