# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test native tile Cholesky factors and adjoints on CUDA and CPU.

Disable MathDx solvers to exercise the native implementations, including
cooperative CPU blocks, in-place calls, tile views, and shared diagonals.
"""

import unittest
from typing import TYPE_CHECKING

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

if TYPE_CHECKING:
    from warp._src.context import DeviceLike

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


# This kernel lives in its own module so that the block_dim == 1 launch
# specializes only it instead of recompiling every kernel in this module,
# including the unrelated N=32 Cholesky adjoint. The module-scope
# enable_mathdx_solver option does not carry over to a unique module, so it is
# repeated here. The launch is always a single block, so the grid-stride
# wrapper is not needed either.
@wp.kernel(enable_backward=False, grid_stride=False, module="unique", module_options={"enable_mathdx_solver": False})
def tile_cholesky_lower_isolated_kernel(gA: wp.array2d[wp.float64], gL: wp.array2d[wp.float64]):
    A = wp.tile_load(gA, shape=(N, N))
    L = wp.tile_cholesky(A)
    wp.tile_store(gL, L)


def test_cholesky_lower_block_dim_1(test, device):
    """Check the lower Cholesky factorization with a single-thread block.

    At ``block_dim == 1`` the thread-strided loops collapse to sequential, but
    the codegen path must still compile and run correctly on every device.
    """
    A_np, L_ref = _spd(N, seed=42)

    A = wp.array(A_np, dtype=wp.float64, device=device)
    L = wp.zeros((N, N), dtype=wp.float64, device=device)

    wp.launch_tiled(tile_cholesky_lower_isolated_kernel, dim=[1], inputs=[A, L], block_dim=1, device=device)
    assert_np_equal(L.numpy(), L_ref, tol=1e-10)


# -----------------------------------------------------------------------------
# Suite registration
# -----------------------------------------------------------------------------


class TestTileCholeskyNoMathDx(unittest.TestCase):
    pass


_devices = get_test_devices()

cholesky_tests = [
    ("test_cholesky_lower", test_cholesky_lower),
    ("test_cholesky_lower_inplace", test_cholesky_lower_inplace),
    ("test_cholesky_upper", test_cholesky_upper),
    ("test_cholesky_upper_inplace", test_cholesky_upper_inplace),
    ("test_cholesky_lower_backward", test_cholesky_lower_backward),
    ("test_cholesky_upper_backward", test_cholesky_upper_backward),
    ("test_cholesky_lower_backward_n32", test_cholesky_lower_backward_n32),
    ("test_cholesky_lower_block_dim_1", test_cholesky_lower_block_dim_1),
]

for name, func in cholesky_tests:
    add_function_test(TestTileCholeskyNoMathDx, name, func, devices=_devices, check_output=False)

for name, func in cholesky_tests:
    if name.endswith("block_dim_1"):
        continue
    add_function_test(
        TestTileCholeskyNoMathDx,
        f"{name}_cpu_blocks",
        func,
        devices=get_cpu_test_devices(),
        check_output=False,
        enable_cpu_blocks=True,
    )


def make_cholesky_storage_kernel(n, dtype, upper, inplace, view, backward=False):
    padded_n = n + 2 if inplace else n
    offset = 1 if inplace else 0

    @wp.kernel(module="unique", enable_backward=backward)
    def factor_kernel(matrices: wp.array3d[dtype], factors: wp.array3d[dtype], observed: wp.array3d[dtype]):
        batch = wp.tid()
        if wp.static(view):
            full = wp.tile_load(matrices[batch], shape=(padded_n, padded_n), storage="shared")
            matrix = wp.tile_transpose(wp.tile_view(full, offset=(offset, offset), shape=(n, n)))
        else:
            matrix = wp.tile_load(matrices[batch], shape=(n, n), storage="shared")
        if wp.static(inplace):
            if wp.static(upper):
                wp.tile_cholesky_inplace(matrix, fill_mode="upper")
            else:
                wp.tile_cholesky_inplace(matrix)
            wp.tile_store(factors[batch], matrix)
        else:
            if wp.static(upper):
                factor = wp.tile_cholesky(matrix, fill_mode="upper")
            else:
                factor = wp.tile_cholesky(matrix)
            wp.tile_store(factors[batch], factor)
        wp.tile_store(observed[batch], matrix)

    wp.set_module_options({"enable_backward": backward, "enable_mathdx_solver": False}, module=factor_kernel.module)
    return factor_kernel


def cholesky_storage_inputs(n, dtype, upper, view, inplace=False):
    rng = np.random.default_rng(45 + n)
    normal = rng.standard_normal((2, n, n))
    matrices = (normal @ normal.transpose(0, 2, 1) / n + np.eye(n)).astype(dtype)
    expected = np.linalg.cholesky(matrices.astype(np.float64))
    if upper:
        expected = expected.transpose(0, 2, 1)
    unused = np.tril_indices(n, -1) if upper else np.triu_indices(n, 1)
    matrices[:, unused[0], unused[1]] = np.nan
    if view and inplace:
        padded = np.full((2, n + 2, n + 2), np.nan, dtype=dtype)
        padded[:, 1:-1, 1:-1] = matrices.transpose(0, 2, 1)
    elif view:
        padded = matrices.transpose(0, 2, 1).copy()
    else:
        padded = matrices.copy()
    return padded, matrices, expected


def check_cholesky_storage_cases(test, device, cases):
    """Check Cholesky factors, in-place semantics, and input immutability for ``(dtype, n, block_dim, view, upper, inplace)`` cases."""
    tolerances = {wp.float32: (np.float32, 2e-6), wp.float64: (np.float64, 1e-12)}
    for dtype, n, block_dim, view, upper, inplace in cases:
        np_dtype, tol = tolerances[dtype]
        with test.subTest(dtype=dtype, n=n, block_dim=block_dim, upper=upper, inplace=inplace, view=view):
            padded, matrices, expected = cholesky_storage_inputs(n, np_dtype, upper, view, inplace)
            a = wp.array(padded, dtype=dtype, device=device)
            factors = wp.empty((2, n, n), dtype=dtype, device=device)
            observed = wp.empty_like(factors)
            factor_kernel = make_cholesky_storage_kernel(n, dtype, upper, inplace, view)
            wp.launch_tiled(factor_kernel, dim=2, inputs=[a, factors, observed], block_dim=block_dim, device=device)
            np.testing.assert_allclose(factors.numpy(), expected, rtol=tol, atol=tol)
            if inplace:
                np.testing.assert_allclose(observed.numpy(), expected, rtol=tol, atol=tol)
            else:
                np.testing.assert_array_equal(observed.numpy(), matrices)
            np.testing.assert_array_equal(a.numpy(), padded)


def test_cholesky_storage_and_blocks_cuda(test, device: "DeviceLike"):
    """Check FP64 CUDA Cholesky across triangles, in-place calls, views, and diagonal-sharing block-size boundaries."""
    cases = [
        (wp.float64, n, block_dim, view, upper, inplace)
        for n, block_dim, view in (
            (5, 128, True),  # padded transposed view, shared diagonal
            (32, 32, False),  # block_dim == warp size, no sharing
            (32, 128, False),  # multi-warp block, shared diagonal
            (64, 64, False),  # block_dim == n, no sharing
        )
        for upper in (False, True)
        for inplace in (False, True)
    ]
    # FP32 does not take the shared-diagonal path; keep a minimal regression check.
    cases += [(wp.float32, 32, 128, False, False, False), (wp.float32, 32, 128, False, True, True)]
    check_cholesky_storage_cases(test, device, cases)


def test_cholesky_storage_and_blocks_cpu(test, device: "DeviceLike"):
    """Check CPU Cholesky aliasing and padded views for both triangles."""
    check_cholesky_storage_cases(
        test,
        device,
        [
            (wp.float64, 5, 128, True, False, True),
            (wp.float64, 5, 128, True, True, True),
            (wp.float64, 17, 64, False, True, False),
        ],
    )


def test_cholesky_storage_gradients(test, device):
    """Check large-block Cholesky gradients through both triangles and transposed tile views."""
    rng = np.random.default_rng(46)
    for dtype, np_dtype, tol in ((wp.float32, np.float32, 2e-5), (wp.float64, np.float64, 2e-8)):
        for n, view in ((5, True), (17, False)):
            for upper in (False, True):
                with test.subTest(dtype=dtype, n=n, upper=upper, view=view):
                    padded, matrices, _ = cholesky_storage_inputs(n, np_dtype, upper, view)
                    a = wp.array(padded, dtype=dtype, device=device, requires_grad=True)
                    factors = wp.empty((2, n, n), dtype=dtype, device=device, requires_grad=True)
                    observed = wp.empty((2, n, n), dtype=dtype, device=device)
                    factor_kernel = make_cholesky_storage_kernel(n, dtype, upper, False, view, backward=True)
                    with wp.Tape() as tape:
                        wp.launch_tiled(
                            factor_kernel, dim=2, inputs=[a, factors, observed], block_dim=128, device=device
                        )
                    weights = rng.standard_normal((2, n, n)).astype(np_dtype)
                    seed = wp.array(weights, dtype=dtype, device=device)
                    tape.backward(grads={factors: seed})
                    if device.is_cuda and view:
                        tape.zero()
                        seed.assign(weights)
                        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                            tape.backward(grads={factors: seed})
                        wp.capture_launch(capture.graph)

                    def objective(triangle, weight, upper=upper):
                        stored = np.triu(triangle) if upper else np.tril(triangle)
                        symmetric = stored + stored.T - np.diag(np.diag(stored))
                        factor = np.linalg.cholesky(symmetric)
                        return np.sum((factor.T if upper else factor) * weight)

                    expected = np.zeros((2, n, n), dtype=np.float64)
                    rows, cols = np.triu_indices(n) if upper else np.tril_indices(n)
                    step = 1e-5
                    for batch in range(2):
                        for i, j in zip(rows, cols, strict=True):
                            plus = matrices[batch].astype(np.float64)
                            minus = plus.copy()
                            plus[i, j] += step
                            minus[i, j] -= step
                            expected[batch, i, j] = (
                                objective(plus, weights[batch]) - objective(minus, weights[batch])
                            ) / (2 * step)
                    if view:
                        expected = expected.transpose(0, 2, 1)
                    np.testing.assert_allclose(a.grad.numpy(), expected, rtol=tol, atol=tol)


add_function_test(
    TestTileCholeskyNoMathDx,
    "test_cholesky_storage_and_blocks",
    test_cholesky_storage_and_blocks_cuda,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestTileCholeskyNoMathDx,
    "test_cholesky_storage_and_blocks",
    test_cholesky_storage_and_blocks_cpu,
    devices=get_cpu_test_devices(),
)
add_function_test(
    TestTileCholeskyNoMathDx,
    "test_cholesky_storage_and_blocks_cpu_blocks",
    test_cholesky_storage_and_blocks_cpu,
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)
add_function_test(
    TestTileCholeskyNoMathDx,
    "test_cholesky_storage_gradients",
    test_cholesky_storage_gradients,
    devices=get_test_devices(),
)
add_function_test(
    TestTileCholeskyNoMathDx,
    "test_cholesky_storage_gradients_cpu_blocks",
    test_cholesky_storage_gradients,
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
