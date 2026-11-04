# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_cpu_test_devices, get_test_devices


def make_lu_kernel(n, dtype, mathdx, view=False):
    padded_n = n + 2

    @wp.kernel(module="unique", enable_backward=False)
    def factor(
        matrices: wp.array3d[dtype],
        factors: wp.array3d[dtype],
        pivots: wp.array2d[wp.int32],
        status: wp.array2d[wp.int32],
        original: wp.array3d[dtype],
    ):
        batch = wp.tid()
        if wp.static(view):
            full = wp.tile_load(matrices[batch], shape=(padded_n, padded_n), storage="shared")
            a = wp.tile_transpose(wp.tile_view(full, offset=(1, 1), shape=(n, n)))
        else:
            a = wp.tile_load(matrices[batch], shape=(n, n))
        lu, p, info = wp.tile_lu(a)
        wp.tile_store(factors[batch], lu)
        wp.tile_store(pivots[batch], p)
        wp.tile_store(status[batch], info)
        wp.tile_store(original[batch], a)

    wp.set_module_options({"enable_mathdx_solver": mathdx}, module=factor.module)
    return factor


def check_lu(test, device, mathdx):
    rng = np.random.default_rng(42)
    for dtype, np_dtype, tol in ((wp.float32, np.float32, 2e-5), (wp.float64, np.float64, 1e-12)):
        for n in (1, 2, 8, 17):
            with test.subTest(dtype=dtype, n=n, mathdx=mathdx):
                # General nonsymmetric inputs, a forced row permutation, and exact
                # singularities at the first and last pivots (including all-zero A).
                matrices = rng.standard_normal((7, n, n)).astype(np_dtype)
                matrices[1:3] += np.eye(n, dtype=np_dtype) * n
                matrices[2] = np.roll(matrices[2], 1, axis=0)
                matrices[3] = np.eye(n, dtype=np_dtype)[::-1]
                matrices[4] = np.eye(n, dtype=np_dtype)
                matrices[4, -1, -1] = 0
                matrices[5] = np.eye(n, dtype=np_dtype)
                matrices[5, 0, 0] = 0
                matrices[6] = 0
                a = wp.array(matrices, device=device, dtype=dtype)
                lu = wp.empty_like(a)
                original = wp.empty_like(a)
                p = wp.empty((len(matrices), n), device=device, dtype=wp.int32)
                info = wp.empty((len(matrices), 1), device=device, dtype=wp.int32)
                kernel = make_lu_kernel(n, dtype, mathdx)
                wp.launch_tiled(
                    kernel, dim=len(matrices), inputs=[a, lu, p, info, original], block_dim=32, device=device
                )

                np.testing.assert_array_equal(original.numpy(), matrices)
                np.testing.assert_array_equal(info.numpy()[:, 0], [0, 0, 0, 0, n, 1, 1])
                factors, pivots = lu.numpy(), p.numpy()
                for b, matrix in enumerate(matrices):
                    permuted = matrix.copy()
                    rhs = rng.standard_normal((n, 2)).astype(np_dtype)
                    permuted_rhs = rhs.copy()
                    for k, pivot in enumerate(pivots[b]):
                        test.assertGreaterEqual(pivot, k)
                        test.assertLess(pivot, n)
                        permuted[[k, pivot]] = permuted[[pivot, k]]
                        permuted_rhs[[k, pivot]] = permuted_rhs[[pivot, k]]
                    lower = np.tril(factors[b], -1) + np.eye(n, dtype=np_dtype)
                    upper = np.triu(factors[b])
                    np.testing.assert_allclose(lower @ upper, permuted, rtol=tol, atol=tol)
                    if b < 4:
                        solution = np.linalg.solve(upper, np.linalg.solve(lower, permuted_rhs))
                        np.testing.assert_allclose(matrix @ solution, rhs, rtol=tol, atol=tol)

    # A transposed tile view has both a column-major layout and a padded stride.
    n = 5
    matrices = rng.standard_normal((2, n + 2, n + 2))
    matrices[:, 1:-1, 1:-1] += n * np.eye(n)
    a = wp.array(matrices, dtype=wp.float64, device=device)
    lu = wp.empty((2, n, n), dtype=wp.float64, device=device)
    original = wp.empty_like(lu)
    p = wp.empty((2, n), dtype=wp.int32, device=device)
    info = wp.empty((2, 1), dtype=wp.int32, device=device)
    wp.launch_tiled(
        make_lu_kernel(n, wp.float64, mathdx, view=True),
        dim=2,
        inputs=[a, lu, p, info, original],
        block_dim=64,
        device=device,
    )
    expected = matrices[:, 1:-1, 1:-1].transpose(0, 2, 1)
    np.testing.assert_array_equal(original.numpy(), expected)
    np.testing.assert_array_equal(info.numpy(), 0)
    for b, packed in enumerate(lu.numpy()):
        permuted = expected[b].copy()
        for k, pivot in enumerate(p.numpy()[b]):
            permuted[[k, pivot]] = permuted[[pivot, k]]
        np.testing.assert_allclose((np.tril(packed, -1) + np.eye(n)) @ np.triu(packed), permuted, atol=1e-12)


def test_lu_native(test, device):
    check_lu(test, device, False)


def test_lu_mathdx(test, device):
    if not device.is_cuda or not wp._src.context.runtime.core.wp_is_mathdx_enabled():
        test.skipTest("Requires CUDA and MathDx")
    check_lu(test, device, True)


def make_invalid_kernel(shape, dtype):
    rank = len(shape)
    m, n = shape[0], shape[-1]

    @wp.kernel(module="unique", enable_backward=False)
    def invalid():
        if wp.static(rank == 1):
            a = wp.tile_zeros(shape=(m,), dtype=dtype)
        else:
            a = wp.tile_zeros(shape=(m, n), dtype=dtype)
        _lu, _p, _info = wp.tile_lu(a)

    return invalid


class TestTileLU(unittest.TestCase):
    def test_invalid_arguments(self):
        for shape, dtype, message in (
            ((3,), wp.float32, "must be a 2D tile"),
            ((2, 3), wp.float32, "must be square"),
            ((2, 2), wp.float16, "float32 or float64"),
        ):
            with self.subTest(shape=shape, dtype=dtype):
                with self.assertRaisesRegex((TypeError, ValueError, RuntimeError), message):
                    wp.launch_tiled(make_invalid_kernel(shape, dtype), dim=1, block_dim=32, device="cpu")


devices = get_test_devices()
add_function_test(TestTileLU, "test_lu_native", test_lu_native, devices=devices)
add_function_test(TestTileLU, "test_lu_mathdx", test_lu_mathdx, devices=devices)
add_function_test(
    TestTileLU,
    "test_lu_native_cpu_blocks",
    test_lu_native,
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
