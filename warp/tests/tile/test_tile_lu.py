# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any
from unittest import mock

import numpy as np

import warp as wp
from warp._src.context import builtin_functions
from warp.tests.unittest_utils import add_function_test, get_cpu_test_devices, get_cuda_test_devices, get_test_devices


def make_lu_kernel(n, dtype, mathdx, view=False, backward=False, use_helper=False):
    padded_n = n + 2

    @wp.func
    def factor_value(matrix: Any):
        return wp.tile_lu(matrix)

    @wp.kernel(module="unique", enable_backward=backward)
    def factor_kernel(
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
        if wp.static(use_helper):
            lu, p, info = factor_value(a)
        else:
            lu, p, info = wp.tile_lu(a)
        wp.tile_store(factors[batch], lu)
        wp.tile_store(pivots[batch], p)
        wp.tile_store(status[batch], info)
        wp.tile_store(original[batch], a)

    wp.set_module_options({"enable_mathdx_solver": mathdx, "enable_backward": backward}, module=factor_kernel.module)
    return factor_kernel


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
    """Validate native LU factors, pivots, singularities, and tile views."""
    check_lu(test, device, False)


def skip_unless_mathdx_links(test):
    """Skip a test that compiles and links cuSOLVERDx code when the build cannot."""
    if not wp._src.context.runtime.core.wp_is_mathdx_enabled():
        test.skipTest("Requires MathDx")
    if wp.config.llvm_cuda:
        test.skipTest("The Clang CUDA backend cannot link MathDx code")


def test_lu_mathdx(test, device):
    """Validate cuSOLVERDx LU factors, pivots, singularities, and tile views."""
    skip_unless_mathdx_links(test)
    check_lu(test, device, True)


def test_lu_backend_dispatch(test, device):
    """Verify that MathDx emits and links GETRF and both backward triangular solves."""
    if not wp._src.context.runtime.core.wp_is_mathdx_enabled():
        test.skipTest("Requires MathDx")
    for mathdx in (False, True):
        for backward in (False, True):
            factor_kernel = make_lu_kernel(8, wp.float32, mathdx, backward=backward)
            options = factor_kernel.module.resolve_options(wp.config, block_dim=32) | {
                "output_arch": device.get_cuda_compile_arch()
            }
            source, _, _, ltoirs, fatbins = factor_kernel.module._run_codegen(options, is_cpu=False)
            if mathdx:
                test.assertRegex(source, r"wp::tile_lu\(getrf_partial_pivot_8_8_1_\w+,")
                test.assertEqual(len(ltoirs), 3 if backward else 1)
                test.assertTrue(fatbins)
                if backward:
                    test.assertRegex(
                        source,
                        r"wp::adj_tile_lu\(getrf_partial_pivot_\w+, trsm_\w+_0_0_0, trsm_\w+_1_1_1,",
                    )
            else:
                test.assertNotIn("getrf_partial_pivot_", source)
                test.assertNotIn("trsm_", source)
                test.assertFalse(ltoirs)
                test.assertFalse(fatbins)


def check_lu_backward(test, device, mathdx):
    rng = np.random.default_rng(29)
    for dtype, np_dtype, tol in ((wp.float32, np.float32, 2e-5), (wp.float64, np.float64, 2e-9)):
        for n, view, use_helper in (
            (1, False, False),
            (3, False, True),
            (8, False, False),
            (17, False, False),
            (3, True, False),
        ):
            with test.subTest(dtype=dtype, n=n, view=view, use_helper=use_helper, mathdx=mathdx):
                if (
                    mathdx
                    and use_helper
                    and wp.config.mode == "debug"
                    and wp._src.context.runtime.toolkit_version >= (13, 0)
                ):
                    # CUDA 13 nvJitLink rejects debug builds that call MathDx from a Warp function
                    # ("Symbol name with unsupported characters"). Kernel-level calls still link.
                    test.skipTest("MathDx calls from Warp functions do not link in CUDA 13 debug builds")
                lower = np.eye(n) + np.tril(rng.uniform(-0.2, 0.2, (n, n)), -1)
                upper = np.triu(rng.uniform(-0.5, 0.5, (n, n))) + 2.0 * np.eye(n)
                base = lower @ upper
                # A cyclic permutation forces multiple, noncommuting row swaps.
                matrices = np.stack([base, np.roll(base, 1, axis=0)]).astype(np_dtype)
                if view:
                    padded = np.zeros((2, n + 2, n + 2), dtype=np_dtype)
                    padded[:, 1:-1, 1:-1] = matrices.transpose(0, 2, 1)
                else:
                    padded = matrices
                a = wp.array(padded, dtype=dtype, device=device, requires_grad=True)
                lu = wp.empty((2, n, n), dtype=dtype, device=device, requires_grad=True)
                original = wp.empty_like(lu)
                p = wp.empty((2, n), dtype=wp.int32, device=device)
                info = wp.empty((2, 1), dtype=wp.int32, device=device)
                factor_kernel = make_lu_kernel(n, dtype, mathdx, view, backward=True, use_helper=use_helper)
                block_dim = 64 if n == 17 or view else 32
                with wp.Tape() as tape:
                    wp.launch_tiled(
                        factor_kernel, dim=2, inputs=[a, lu, p, info, original], block_dim=block_dim, device=device
                    )
                pivots = p.numpy()
                np.testing.assert_array_equal(info.numpy(), 0)
                np.testing.assert_array_equal(pivots[0], np.arange(n))
                if n > 1:
                    test.assertGreaterEqual(np.count_nonzero(pivots[1] != np.arange(n)), 2)

                # Differentiate an arbitrary weighting of BOTH packed triangles.
                # The direct input use also checks that the adjoint accumulates.
                weights = rng.standard_normal((2, n, n)).astype(np_dtype)
                direct_weights = rng.standard_normal((2, n, n)).astype(np_dtype)
                seeds = {
                    lu: wp.array(weights, dtype=dtype, device=device),
                    original: wp.array(direct_weights, dtype=dtype, device=device),
                }
                tape.backward(grads=seeds)
                if device.is_cuda and n == 8:
                    tape.zero()
                    # Tape.zero() also zeros the installed seed arrays.
                    seeds[lu].assign(weights)
                    seeds[original].assign(direct_weights)
                    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                        tape.backward(grads=seeds)
                    wp.capture_launch(capture.graph)

                # Independent central differences of elimination with recorded pivots.
                # Use double precision for the numerical oracle even for FP32 kernels.
                def packed_objective(matrix, swaps, seed):
                    packed = matrix.copy()
                    for k, pivot in enumerate(swaps):
                        # The test inputs stay away from pivot-choice boundaries.
                        test.assertEqual(k + np.argmax(np.abs(packed[k:, k])), pivot)
                        packed[[k, pivot]] = packed[[pivot, k]]
                        packed[k + 1 :, k] /= packed[k, k]
                        packed[k + 1 :, k + 1 :] -= np.outer(packed[k + 1 :, k], packed[k, k + 1 :])
                    return np.sum(packed * seed)

                expected = np.empty_like(matrices, dtype=np.float64)
                step = 1.0e-5
                for batch in range(2):
                    for i in range(n):
                        for j in range(n):
                            plus = matrices[batch].astype(np.float64)
                            minus = plus.copy()
                            plus[i, j] += step
                            minus[i, j] -= step
                            expected[batch, i, j] = (
                                packed_objective(plus, pivots[batch], weights[batch])
                                - packed_objective(minus, pivots[batch], weights[batch])
                            ) / (2.0 * step)
                expected += direct_weights
                if view:
                    padded_expected = np.zeros_like(padded)
                    padded_expected[:, 1:-1, 1:-1] = expected.transpose(0, 2, 1)
                    expected = padded_expected
                np.testing.assert_allclose(a.grad.numpy(), expected, atol=tol, rtol=tol)
                test.assertIsNone(p.grad)
                test.assertIsNone(info.grad)


def test_lu_native_backward(test, device):
    """Check native packed-factor gradients through fixed row swaps and tile views."""
    check_lu_backward(test, device, False)


def test_lu_mathdx_backward(test, device):
    """Check cuSOLVERDx packed-factor gradients through fixed row swaps and tile views."""
    skip_unless_mathdx_links(test)
    check_lu_backward(test, device, True)


def make_invalid_kernel(shape, dtype):
    rank = len(shape)
    m, n = shape[0], shape[-1]

    @wp.kernel(module="unique", enable_backward=False)
    def invalid_kernel():
        if wp.static(rank == 1):
            a = wp.tile_zeros(shape=(m,), dtype=dtype)
        else:
            a = wp.tile_zeros(shape=(m, n), dtype=dtype)
        _lu, _p, _info = wp.tile_lu(a)

    return invalid_kernel


class TestTileLU(unittest.TestCase):
    def test_backward_cache_key(self):
        """Invalidate cached LU kernels when the builtin's adjoint support changes."""
        for use_helper in (False, True):
            with self.subTest(use_helper=use_helper):
                factor_kernel = make_lu_kernel(3, wp.float32, False, backward=True, use_helper=use_helper)
                with mock.patch.object(builtin_functions["tile_lu"], "is_differentiable", False):
                    old_hash = factor_kernel.module.hash_module()
                self.assertNotEqual(old_hash, factor_kernel.module.hash_module())

    def test_backward_cache_key_alias(self):
        """Invalidate cached LU kernels that call the builtin through a local alias."""

        @wp.kernel(module="unique", enable_backward=True)
        def alias_kernel(matrices: wp.array2d[float], factors: wp.array2d[float]):
            factor = wp.tile_lu
            lu, _p, _info = factor(wp.tile_load(matrices, shape=(3, 3)))
            wp.tile_store(factors, lu)

        with mock.patch.object(builtin_functions["tile_lu"], "is_differentiable", False):
            old_hash = alias_kernel.module.hash_module()
        self.assertNotEqual(old_hash, alias_kernel.module.hash_module())

    def test_invalid_arguments(self):
        """Reject unsupported tile ranks, shapes, and precisions before launch."""
        for shape, dtype, error_type, message in (
            ((3,), wp.float32, ValueError, "must be a 2D tile"),
            ((2, 3), wp.float32, ValueError, "must be square"),
            ((2, 2), wp.float16, TypeError, "float32 or float64"),
        ):
            with self.subTest(shape=shape, dtype=dtype):
                with self.assertRaisesRegex(error_type, message):
                    wp.launch_tiled(make_invalid_kernel(shape, dtype), dim=1, block_dim=32, device="cpu")


devices = get_test_devices()
add_function_test(TestTileLU, "test_lu_native", test_lu_native, devices=devices)
add_function_test(TestTileLU, "test_lu_mathdx", test_lu_mathdx, devices=get_cuda_test_devices())
add_function_test(TestTileLU, "test_lu_backend_dispatch", test_lu_backend_dispatch, devices=get_cuda_test_devices())
add_function_test(TestTileLU, "test_lu_native_backward", test_lu_native_backward, devices=devices)
add_function_test(TestTileLU, "test_lu_mathdx_backward", test_lu_mathdx_backward, devices=get_cuda_test_devices())
add_function_test(
    TestTileLU,
    "test_lu_native_backward_cpu_blocks",
    test_lu_native_backward,
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)
add_function_test(
    TestTileLU,
    "test_lu_native_cpu_blocks",
    test_lu_native,
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
