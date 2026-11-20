# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp._src.codegen import WarpCodegenError
from warp.tests.unittest_utils import add_function_test, get_cpu_test_devices, get_test_devices


def make_ldlt_kernel(n, dtype, upper=False, view=False, helper=False, backward=False, mathdx=True):
    @wp.func
    def factor_helper(matrix: wp.tile[dtype, n, n]):
        if wp.static(upper):
            return wp.tile_ldlt(matrix, fill_mode="upper")
        else:
            return wp.tile_ldlt(matrix)

    padded_n = n + 2

    @wp.kernel(module="unique", enable_backward=backward)
    def factor_kernel(
        a: wp.array3d[dtype], factors: wp.array3d[dtype], pivots: wp.array2d[int], status: wp.array2d[int]
    ):
        batch = wp.tid()
        if wp.static(view):
            full = wp.tile_load(a[batch], shape=(padded_n, padded_n), storage="shared")
            matrix = wp.tile_transpose(wp.tile_view(full, offset=(1, 1), shape=(n, n)))
        else:
            matrix = wp.tile_load(a[batch], shape=(n, n))
        if wp.static(helper):
            packed, p, info = factor_helper(matrix)
        elif wp.static(upper):
            packed, p, info = wp.tile_ldlt(matrix, fill_mode="upper")
        else:
            packed, p, info = wp.tile_ldlt(matrix)
        wp.tile_store(factors[batch], packed)
        wp.tile_store(pivots[batch], p)
        wp.tile_store(status[batch], info)

    wp.set_module_options({"enable_backward": backward, "enable_mathdx_solver": mathdx}, module=factor_kernel.module)
    return factor_kernel


def ldlt_matrices(n):
    rng = np.random.default_rng(51 + n)
    q, _ = np.linalg.qr(rng.standard_normal((7, n, n)))
    spectrum = np.linspace(0.5, 2, n) * np.where(np.arange(n) % 2, 1, -1)
    matrices = (q * spectrum) @ q.transpose(0, 2, 1)
    matrices = (matrices + matrices.transpose(0, 2, 1)) * 0.5
    matrices[0] = np.diag(spectrum)
    if n >= 2:
        matrices[1] = np.eye(n)
        matrices[1, :2, :2] = [[0, 1], [1, 0]]
        matrices[2] = np.eye(n)
        matrices[2, 0, 0] = matrices[2, -1, -1] = 0
        matrices[2, 0, -1] = matrices[2, -1, 0] = 1
        matrices[3] = matrices[2].copy()
        matrices[3, -1, -1] = 10
    if n >= 3:
        first, second = (32, 33) if n > 33 else (1, n - 1)
        matrices[4] = np.eye(n)
        matrices[4, 0, 0] = 0
        matrices[4, 0, first] = matrices[4, first, 0] = 1
        matrices[4, 0, second] = matrices[4, second, 0] = -1
        matrices[4, first, first] = matrices[4, second, second] = 10
        matrices[6] = matrices[4].copy()
        matrices[6, 0, second] = matrices[6, second, 0] = -np.nextafter(1.0, 2.0)
    return matrices


def ldlt_input(matrices, upper, view):
    n = matrices.shape[1]
    stored = matrices.copy()
    unused = np.tril_indices(n, -1) if upper else np.triu_indices(n, 1)
    stored[:, unused[0], unused[1]] = np.nan
    if view:
        padded = np.full((len(matrices), n + 2, n + 2), np.nan, dtype=matrices.dtype)
        padded[:, 1:-1, 1:-1] = stored.transpose(0, 2, 1)
        return padded
    return stored


def unpack_ldlt(test, packed, pivots, upper=False):
    packed = packed.T if upper else packed
    n = len(packed)
    lower = np.tril(packed, -1) + np.eye(n)
    diagonal = np.diag(np.diag(packed)).copy()
    permutation = np.arange(n)
    sizes = []
    k = 0
    while k < n:
        pivot = int(pivots[k])
        if pivot < 0:
            test.assertLess(k + 1, n)
            test.assertEqual(pivots[k + 1], pivot)
            target, pivot, size = k + 1, -pivot - 1, 2
            diagonal[k, k + 1] = diagonal[k + 1, k] = packed[k + 1, k]
            lower[k + 1, k] = 0
        else:
            target, size = k, 1
        test.assertGreaterEqual(pivot, target)
        test.assertLess(pivot, n)
        permutation[[target, pivot]] = permutation[[pivot, target]]
        sizes.append(size)
        k += size
    return lower, diagonal, permutation, sizes


def fixed_ldlt(matrix, permutation, sizes):
    """Factor a permuted matrix with fixed blocks using NumPy dense solves."""
    work = matrix[np.ix_(permutation, permutation)].copy()
    lower = np.eye(len(matrix))
    diagonal = np.zeros_like(work)
    k = 0
    for size in sizes:
        end = k + size
        block = work[k:end, k:end]
        diagonal[k:end, k:end] = block
        lower[end:, k:end] = np.linalg.solve(block, work[end:, k:end].T).T
        work[end:, end:] -= lower[end:, k:end] @ block @ lower[end:, k:end].T
        k = end
    packed = np.tril(lower, -1) + np.diag(np.diag(diagonal))
    k = 0
    for size in sizes:
        if size == 2:
            packed[k + 1, k] = diagonal[k + 1, k]
        k += size
    return packed


def test_ldlt_forward(test, device):
    """Check factors, symmetric swaps, block pivots, solves, views, and singular status."""
    for dtype, precision, tol in ((wp.float32, np.float32, 2e-5), (wp.float64, np.float64, 2e-12)):
        for n, block, view, helper in (
            (1, 32, False, False),
            (2, 32, False, False),
            (5, 128, True, False),
            (8, 32, False, True),
            (17, 128, False, False),
            (32, 128, False, False),
            (64, 128, False, False),
        ):
            for upper in (False, True):
                with test.subTest(dtype=dtype, n=n, upper=upper, view=view, helper=helper):
                    matrices = ldlt_matrices(n).astype(precision)
                    matrices = np.concatenate(
                        (
                            matrices,
                            np.zeros((1, n, n), dtype=precision),
                            np.diag([1] * (n - 1) + [0]).astype(precision)[None],
                        )
                    )
                    stored = ldlt_input(matrices, upper, view)
                    a = wp.array(stored, dtype=dtype, device=device)
                    factors = wp.empty((len(matrices), n, n), dtype=dtype, device=device)
                    pivots = wp.empty((len(matrices), n), dtype=wp.int32, device=device)
                    status = wp.empty((len(matrices), 1), dtype=wp.int32, device=device)
                    kernel = make_ldlt_kernel(n, dtype, upper, view, helper, mathdx=upper)
                    wp.launch_tiled(
                        kernel, dim=len(matrices), inputs=[a, factors, pivots, status], block_dim=block, device=device
                    )
                    expected_status = np.zeros((len(matrices), 1), dtype=np.int32)
                    expected_status[-2, 0], expected_status[-1, 0] = 1, n
                    np.testing.assert_array_equal(status.numpy(), expected_status)
                    np.testing.assert_array_equal(a.numpy(), stored)
                    values, swaps = factors.numpy().astype(np.float64), pivots.numpy()
                    if n >= 2:
                        np.testing.assert_array_equal(swaps[1, :2], [-2, -2])
                        np.testing.assert_array_equal(swaps[2, :2], [-n, -n])
                        test.assertEqual(swaps[3, 0], n - 1)
                    if n >= 3:
                        test.assertEqual(swaps[4, 0], 32 if n > 33 else 1)
                        # Preserve pivot distinctions smaller than FP32 resolution.
                        first, second = (32, 33) if n > 33 else (1, n - 1)
                        test.assertEqual(swaps[6, 0], second if dtype == wp.float64 else first)
                    for index, (matrix, packed, p) in enumerate(zip(matrices, values, swaps, strict=True)):
                        lower, diagonal, permutation, sizes = unpack_ldlt(test, packed, p, upper)
                        np.testing.assert_allclose(
                            lower @ diagonal @ lower.T, matrix[np.ix_(permutation, permutation)], rtol=tol, atol=tol
                        )
                        unused = np.tril(packed, -1) if upper else np.triu(packed, 1)
                        np.testing.assert_array_equal(unused, 0)
                        if index < len(matrices) - 2:
                            expected = fixed_ldlt(matrix.astype(np.float64), permutation, sizes)
                            np.testing.assert_allclose(packed.T if upper else packed, expected, rtol=tol, atol=tol)
                            rhs = np.arange(1, n + 1, dtype=np.float64)
                            answer = np.zeros(n)
                            answer[permutation] = np.linalg.solve(
                                lower.T, np.linalg.solve(diagonal, np.linalg.solve(lower, rhs[permutation]))
                            )
                            np.testing.assert_allclose(matrix @ answer, rhs, rtol=tol, atol=tol)


def test_ldlt_backward(test, device):
    """Check packed-factor gradients with fixed swaps and 2x2 blocks, including tile views."""
    rng = np.random.default_rng(52)
    for dtype, precision, tol in ((wp.float32, np.float32, 4e-5), (wp.float64, np.float64, 2e-8)):
        for n, view, helper in ((2, False, False), (5, True, False), (16, False, False), (17, False, True)):
            for upper in (False, True):
                with test.subTest(dtype=dtype, n=n, upper=upper, view=view, helper=helper):
                    matrices = ldlt_matrices(n)[[1, 2, 3, 5]].astype(precision)
                    stored = ldlt_input(matrices, upper, view)
                    a = wp.array(stored, dtype=dtype, device=device, requires_grad=True)
                    factors = wp.empty((len(matrices), n, n), dtype=dtype, device=device, requires_grad=True)
                    pivots = wp.empty((len(matrices), n), dtype=wp.int32, device=device)
                    status = wp.empty((len(matrices), 1), dtype=wp.int32, device=device)
                    kernel = make_ldlt_kernel(n, dtype, upper, view, helper, backward=True)
                    with wp.Tape() as tape:
                        wp.launch_tiled(
                            kernel, dim=len(matrices), inputs=[a, factors, pivots, status], block_dim=128, device=device
                        )
                    weights = rng.standard_normal((len(matrices), n, n)).astype(precision)
                    seed = wp.array(weights, dtype=dtype, device=device)
                    a.grad.fill_(0.25)
                    tape.backward(grads={factors: seed})
                    expected = np.zeros_like(matrices, dtype=np.float64)
                    for batch, (packed, p) in enumerate(zip(factors.numpy(), pivots.numpy(), strict=True)):
                        _, _, permutation, sizes = unpack_ldlt(test, packed, p, upper)
                        rows, cols = np.triu_indices(n) if upper else np.tril_indices(n)
                        weight = weights[batch].T if upper else weights[batch]
                        for i, j in zip(rows, cols, strict=True):
                            plus = matrices[batch].astype(np.float64)
                            minus = plus.copy()
                            step = 1e-5
                            plus[i, j] += step
                            minus[i, j] -= step
                            if i != j:
                                plus[j, i] += step
                                minus[j, i] -= step
                            difference = fixed_ldlt(plus, permutation, sizes) - fixed_ldlt(minus, permutation, sizes)
                            expected[batch, i, j] = np.sum(difference * weight) / (2 * step)
                    if view:
                        padded = np.zeros_like(stored, dtype=np.float64)
                        padded[:, 1:-1, 1:-1] = expected.transpose(0, 2, 1)
                        expected = padded
                    np.testing.assert_allclose(a.grad.numpy(), expected + 0.25, rtol=tol, atol=tol)
                    if device.is_cuda and view:
                        tape.zero()
                        seed.assign(weights)
                        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                            tape.backward(grads={factors: seed})
                        wp.capture_launch(capture.graph)
                        np.testing.assert_allclose(a.grad.numpy(), expected, rtol=tol, atol=tol)
                    test.assertIsNone(pivots.grad)
                    test.assertIsNone(status.grad)


def test_ldlt_scaling(test, device):
    """Preserve finite factors when pivot inverses overflow or determinants underflow."""
    templates = np.array(
        [
            [[0, 1, 0.2], [1, 0, -0.1], [0.2, -0.1, 1]],
            [[-1, 0.1, 0.2], [0.1, 2, 0.1], [0.2, 0.1, 1]],
        ]
    )
    # Cooperative CPU-block lanes keep the floating-point control state their fibers were
    # created with, so they can flush subnormals even when the launching thread does not.
    # Keep subnormal inputs to execution paths that use the launching thread's state.
    subnormal_inputs = not (device.is_cpu and wp.config.enable_cpu_blocks)
    for dtype, precision, all_scales, tol in (
        (wp.float32, np.float32, [1e-39, 1e-22, 1e30], 2e-5),
        (wp.float64, np.float64, [1e-310, 1e-160, 1e290], 2e-11),
    ):
        scales = [scale for scale in all_scales if subnormal_inputs or scale >= np.finfo(precision).tiny]
        matrices = (np.array(scales)[:, None, None, None] * templates).reshape(-1, 3, 3).astype(precision)
        repeated_scales = np.repeat(scales, len(templates))
        a = wp.array(matrices, dtype=dtype, device=device)
        factors = wp.empty_like(a)
        pivots = wp.empty((len(matrices), 3), dtype=wp.int32, device=device)
        status = wp.empty((len(matrices), 1), dtype=wp.int32, device=device)
        wp.launch_tiled(
            make_ldlt_kernel(3, dtype),
            dim=len(matrices),
            inputs=[a, factors, pivots, status],
            block_dim=128,
            device=device,
        )
        np.testing.assert_array_equal(status.numpy(), 0)
        for scale, original, packed, p in zip(repeated_scales, matrices, factors.numpy(), pivots.numpy(), strict=True):
            lower, diagonal, permutation, _ = unpack_ldlt(test, packed.astype(np.float64), p)
            test.assertTrue(np.isfinite(lower).all())
            np.testing.assert_allclose(
                lower @ (diagonal / scale) @ lower.T,
                original[np.ix_(permutation, permutation)] / scale,
                rtol=tol,
                atol=tol,
            )


def test_ldlt_mixed_scale(test, device):
    """Return finite factors when a 2-by-2 pivot's diagonal-to-off-diagonal ratio overflows."""
    # The pivot block [[0, b], [b, c]] has c / b beyond the dtype's range and an
    # unrepresentable inverse, while the exact multipliers stay finite.
    for dtype, precision, off_diagonal, diagonal, rtol in (
        (wp.float32, np.float32, 1e-19, 4e19, 1e-5),
        (wp.float64, np.float64, 1e-100, 2e208, 1e-12),
    ):
        b, c = off_diagonal, diagonal
        matrix = np.array([[0, b, 0.9 * b], [b, c, 1.6 * c], [0.9 * b, 1.6 * c, 2 * c]], dtype=precision)
        e, f, g = (float(value) for value in (matrix[2, 0], matrix[2, 1], matrix[2, 2]))
        b, c = float(matrix[1, 0]), float(matrix[1, 1])
        second = e / b
        first = (f - c * second) / b
        schur = g - (first * e + second * f)
        for upper in (False, True):
            with test.subTest(dtype=dtype.__name__, upper=upper):
                a = wp.array(ldlt_input(matrix[None], upper, False), dtype=dtype, device=device)
                factors = wp.empty_like(a)
                pivots = wp.empty((1, 3), dtype=wp.int32, device=device)
                status = wp.empty((1, 1), dtype=wp.int32, device=device)
                wp.launch_tiled(
                    make_ldlt_kernel(3, dtype, upper=upper),
                    dim=1,
                    inputs=[a, factors, pivots, status],
                    block_dim=64,
                    device=device,
                )
                np.testing.assert_array_equal(status.numpy(), 0)
                np.testing.assert_array_equal(pivots.numpy()[0], [-2, -2, 2])
                lower, block_diagonal, _, _ = unpack_ldlt(
                    test, factors.numpy()[0].astype(np.float64), pivots.numpy()[0], upper
                )
                np.testing.assert_allclose(lower[2, :2], [first, second], rtol=rtol)
                np.testing.assert_allclose(block_diagonal[2, 2], schur, rtol=rtol)


def test_ldlt_large_pivot_rhs(test, device):
    """Return finite 2-by-2 pivot gradients when the incoming adjoints are near the dtype's range."""
    # A 2-by-2 pivot block whose inverse overflows float32, plus an isolated third row.
    # The third row's multipliers are zero, so seeding them with w gives a gradient of
    # B^-1 w with respect to A[2, :2] and zero elsewhere. Forming w0 - (a / b) * w1 directly
    # overflows even though B^-1 w = (-1.2, 3.6) is small.
    a, b, c = 5e37, 1e38, -5e37
    w = np.array([3e38, -3e38])
    matrix = np.array([[a, b, 0], [b, c, 0], [0, 0, 1]])
    expected_solution = np.linalg.solve(matrix[:2, :2], w)
    for upper in (False, True):
        with test.subTest(upper=upper):
            stored = ldlt_input(matrix[None].astype(np.float32), upper, False)
            a_array = wp.array(stored, dtype=wp.float32, device=device, requires_grad=True)
            factors = wp.empty((1, 3, 3), dtype=wp.float32, device=device, requires_grad=True)
            pivots = wp.empty((1, 3), dtype=wp.int32, device=device)
            status = wp.empty((1, 1), dtype=wp.int32, device=device)
            with wp.Tape() as tape:
                wp.launch_tiled(
                    make_ldlt_kernel(3, wp.float32, upper=upper, backward=True),
                    dim=1,
                    inputs=[a_array, factors, pivots, status],
                    block_dim=64,
                    device=device,
                )
            np.testing.assert_array_equal(status.numpy(), 0)
            np.testing.assert_array_equal(pivots.numpy()[0], [-2, -2, 2])
            weights = np.zeros((1, 3, 3), dtype=np.float32)
            weights[0, 2, :2] = w
            seed = wp.array(weights.transpose(0, 2, 1) if upper else weights, dtype=wp.float32, device=device)
            tape.backward(grads={factors: seed})
            expected = np.zeros((3, 3))
            expected[2, :2] = expected_solution
            gradient = a_array.grad.numpy()[0]
            np.testing.assert_allclose(gradient.T if upper else gradient, expected, rtol=1e-5, atol=0)


class TestTileLDLT(unittest.TestCase):
    def test_invalid_arguments(self):
        """Reject nonsquare tiles, integer tiles, and invalid triangle selectors."""

        @wp.kernel(module="unique")
        def nonsquare_kernel():
            wp.tile_ldlt(wp.tile_zeros(shape=(2, 3), dtype=wp.float32))

        @wp.kernel(module="unique")
        def integer_kernel():
            wp.tile_ldlt(wp.tile_zeros(shape=(2, 2), dtype=wp.int32))

        @wp.kernel(module="unique")
        def half_kernel():
            wp.tile_ldlt(wp.tile_zeros(shape=(2, 2), dtype=wp.float16))

        @wp.kernel(module="unique")
        def triangle_kernel():
            wp.tile_ldlt(wp.tile_zeros(shape=(2, 2), dtype=wp.float32), fill_mode="invalid")

        for kernel, error, message in (
            (nonsquare_kernel, ValueError, "square"),
            (integer_kernel, WarpCodegenError, "function overload"),
            (half_kernel, TypeError, "float32 or float64"),
            (triangle_kernel, ValueError, "fill_mode"),
        ):
            with self.subTest(kernel=kernel.key), self.assertRaisesRegex(error, message):
                kernel.module.load("cpu", block_dim=32)


for name, function in (
    ("test_ldlt_forward", test_ldlt_forward),
    ("test_ldlt_backward", test_ldlt_backward),
    ("test_ldlt_scaling", test_ldlt_scaling),
    ("test_ldlt_mixed_scale", test_ldlt_mixed_scale),
    ("test_ldlt_large_pivot_rhs", test_ldlt_large_pivot_rhs),
):
    add_function_test(TestTileLDLT, name, function, devices=get_test_devices())
    add_function_test(
        TestTileLDLT, f"{name}_cpu_blocks", function, devices=get_cpu_test_devices(), enable_cpu_blocks=True
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
