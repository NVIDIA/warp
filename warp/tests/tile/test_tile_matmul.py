# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import re
import unittest
import warnings
from typing import Any
from unittest import mock

import numpy as np

import warp as wp
import warp._src.build as warp_build
from warp._src.build import _gemm_lto_alignment_unsupported, _gemm_operand_alignments
from warp.tests.unittest_utils import *

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_DIM = 64

# These tests retain their original 32-thread launches. Keep their kernels in one shared Warp module so the large
# default module above is not compiled for a second block dimension; using module="unique" would instead compile
# each closely related kernel separately.
MATMUL_32_BLOCK_DIM = 32


@wp.kernel
def tile_grouped_gemm(A: wp.array3d[float], B: wp.array3d[float], C: wp.array3d[float]):
    # output tile index
    i = wp.tid()

    a = wp.tile_load(A[i], shape=(TILE_M, TILE_K))
    b = wp.tile_load(B[i], shape=(TILE_K, TILE_N))

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

    wp.tile_matmul(a, b, sum)

    wp.tile_store(C[i], sum)


def test_tile_grouped_gemm(test, device):
    batch_count = 56

    M = TILE_M
    N = TILE_N
    K = TILE_K

    rng = np.random.default_rng(42)
    A = rng.random((batch_count, M, K), dtype=np.float32)
    B = rng.random((batch_count, K, N), dtype=np.float32)
    C = A @ B

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.zeros((batch_count, TILE_M, TILE_N), requires_grad=True, device=device)

    with wp.Tape():
        wp.launch_tiled(
            tile_grouped_gemm, dim=[batch_count], inputs=[A_wp, B_wp, C_wp], block_dim=TILE_DIM, device=device
        )

    # TODO: 32 mismatched elements
    assert_np_equal(C_wp.numpy(), C, 1e-6)


@wp.kernel
def tile_gemm(A: wp.array2d[Any], B: wp.array2d[Any], C: wp.array2d[Any]):
    # output tile index
    i, j = wp.tid()

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=A.dtype)

    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

    count = int(K / TILE_K)

    for k in range(0, count):
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))

        # sum += a*b
        wp.tile_matmul(a, b, sum)

    wp.tile_store(C, sum, offset=(i * TILE_M, j * TILE_N))


wp.overload(tile_gemm, {"A": wp.array2d[wp.float16], "B": wp.array2d[wp.float16], "C": wp.array2d[wp.float16]})
wp.overload(tile_gemm, {"A": wp.array2d[wp.float32], "B": wp.array2d[wp.float32], "C": wp.array2d[wp.float32]})
wp.overload(tile_gemm, {"A": wp.array2d[wp.float64], "B": wp.array2d[wp.float64], "C": wp.array2d[wp.float64]})


def test_tile_gemm(dtype):
    def test(test, device):
        M = TILE_M * 7
        K = TILE_K * 6
        N = TILE_N * 5

        rng = np.random.default_rng(42)
        A = rng.random((M, K), dtype=float).astype(wp.dtype_to_numpy(dtype))
        B = rng.random((K, N), dtype=float).astype(wp.dtype_to_numpy(dtype))
        C = np.zeros((M, N), dtype=float).astype(wp.dtype_to_numpy(dtype))

        A_wp = wp.array(A, requires_grad=True, device=device)
        B_wp = wp.array(B, requires_grad=True, device=device)
        C_wp = wp.array(C, requires_grad=True, device=device)

        with wp.Tape() as tape:
            wp.launch_tiled(
                tile_gemm,
                dim=(int(M / TILE_M), int(N / TILE_N)),
                inputs=[A_wp, B_wp, C_wp],
                block_dim=TILE_DIM,
                device=device,
            )

        assert_np_equal(C_wp.numpy(), A @ B, tol=1.0e-1)

        adj_C = np.ones_like(C)

        tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

        assert_np_equal(A_wp.grad.numpy(), adj_C @ B.T, tol=1.0e-1)
        assert_np_equal(B_wp.grad.numpy(), A.T @ adj_C, 1.0e-1)

    return test


@wp.kernel(module="test_tile_matmul_32")
def tile_matmul_mixed_precision_kernel(A: wp.array2d[wp.float16], B: wp.array2d[wp.float32], C: wp.array2d[wp.float64]):
    i, j = wp.tid()
    a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, j * TILE_K))
    b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(i * TILE_K, j * TILE_N))
    c = wp.tile_load(C, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    wp.tile_matmul(a, b, c, alpha=0.5, beta=-1.3)
    wp.tile_store(C, c, offset=(i * TILE_M, j * TILE_N))


def test_tile_matmul_mixed_precision(test, device):
    """Multiply mixed-precision tiles and propagate gradients for all operands."""

    rng = np.random.default_rng(42)

    A = rng.random((TILE_M, TILE_K), dtype=np.float64).astype(np.float16)
    B = rng.random((TILE_K, TILE_N), dtype=np.float32)
    C = rng.random((TILE_M, TILE_N), dtype=np.float64)

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.array(C, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_matmul_mixed_precision_kernel,
            dim=[1, 1],
            inputs=[A_wp, B_wp, C_wp],
            block_dim=MATMUL_32_BLOCK_DIM,
            device=device,
        )

    assert_np_equal(C_wp.numpy(), 0.5 * A @ B - 1.3 * C, tol=1e-2)

    adj_C = np.ones_like(C)

    tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

    assert_np_equal(A_wp.grad.numpy(), 0.5 * adj_C @ B.T, tol=1e-2)
    assert_np_equal(B_wp.grad.numpy(), 0.5 * A.T @ adj_C, tol=1e-2)
    assert_np_equal(C_wp.grad.numpy(), -1.3 * adj_C, tol=1e-2)


# Reassigning tile variables inside the dynamic loop is not differentiable.
@wp.kernel(module="test_tile_matmul_32", enable_backward=False)
def tile_pipelined_gemm_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    i, j = wp.tid()

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
    a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0), storage="register")
    b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, j * TILE_N), storage="register")

    count = int(A.shape[1] / TILE_K)
    for k in range(1, count):
        a_next = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K), storage="register")
        b_next = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N), storage="register")

        wp.tile_matmul(a, b, sum)
        a = a_next
        b = b_next

    wp.tile_matmul(a, b, sum)
    wp.tile_store(C, sum, offset=(i * TILE_M, j * TILE_N))


@wp.kernel(module="test_tile_matmul_32")
def tile_reassign_after_matmul_kernel(
    A: wp.array2d[float],
    B: wp.array2d[float],
    C_sum: wp.array2d[float],
    C_reassigned: wp.array2d[float],
    C_direct: wp.array2d[float],
):
    a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(0, 0), storage="register")
    b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, 0), storage="register")

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
    wp.tile_matmul(a, b, sum)
    wp.tile_store(C_sum, sum)

    a_next = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(TILE_M, 0), storage="register")
    a = a_next

    wp.tile_store(C_reassigned, a)
    wp.tile_store(C_direct, a_next)


def test_tile_matmul_pipelined_reassign(test, device):
    """Exercise register-tile reassignment in a forward-only pipelined GEMM.

    The dynamic loop reassigns tile variables after each multiplication. This
    path is intentionally forward-only because those loop-carried assignments
    are not differentiable.
    """

    M = TILE_M * 3
    K = TILE_K * 3
    N = TILE_N * 5

    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    A_wp = wp.array(A, device=device)
    B_wp = wp.array(B, device=device)
    C_wp = wp.array(C, device=device)

    wp.launch_tiled(
        tile_pipelined_gemm_kernel,
        dim=(int(M / TILE_M), int(N / TILE_N)),
        inputs=[A_wp, B_wp, C_wp],
        block_dim=MATMUL_32_BLOCK_DIM,
        device=device,
    )

    assert_np_equal(C_wp.numpy(), A @ B, tol=1.0e-4)


def test_tile_matmul_reassign_backward(test, device):
    """Propagate gradients through register-to-shared tile reassignment.

    Keep the reassignment outside a dynamic loop to isolate the supported
    assignment adjoint used by pipelined tile multiplication.
    """

    M = TILE_M * 2
    K = TILE_K
    N = TILE_N

    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)
    C_sum = np.zeros((TILE_M, TILE_N), dtype=np.float32)
    C_reassigned = np.zeros((TILE_M, TILE_K), dtype=np.float32)
    C_direct = np.zeros((TILE_M, TILE_K), dtype=np.float32)

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_sum_wp = wp.array(C_sum, requires_grad=True, device=device)
    C_reassigned_wp = wp.array(C_reassigned, requires_grad=True, device=device)
    C_direct_wp = wp.array(C_direct, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_reassign_after_matmul_kernel,
            dim=1,
            inputs=[A_wp, B_wp, C_sum_wp, C_reassigned_wp, C_direct_wp],
            block_dim=MATMUL_32_BLOCK_DIM,
            device=device,
        )

    assert_np_equal(C_sum_wp.numpy(), A[:TILE_M, :] @ B, tol=1.0e-4)
    assert_np_equal(C_reassigned_wp.numpy(), A[TILE_M:, :], tol=1.0e-4)
    assert_np_equal(C_direct_wp.numpy(), A[TILE_M:, :], tol=1.0e-4)

    adj_sum = np.ones_like(C_sum)
    adj_reassigned = np.ones_like(C_reassigned)
    adj_direct = np.ones_like(C_direct)
    tape.backward(
        grads={
            C_sum_wp: wp.array(adj_sum, device=device),
            C_reassigned_wp: wp.array(adj_reassigned, device=device),
            C_direct_wp: wp.array(adj_direct, device=device),
        }
    )

    expected_A_grad = np.zeros_like(A)
    expected_A_grad[:TILE_M, :] = adj_sum @ B.T
    expected_A_grad[TILE_M:, :] = adj_reassigned + adj_direct
    assert_np_equal(A_wp.grad.numpy(), expected_A_grad, tol=1.0e-4)
    assert_np_equal(B_wp.grad.numpy(), A[:TILE_M, :].T @ adj_sum, tol=1.0e-4)


@wp.kernel
def test_tile_transpose_matmul_kernel(input: wp.array2d[float], output: wp.array2d[float]):
    x = wp.tile_load(input, shape=(TILE_M, TILE_N))
    y = wp.tile_transpose(x)

    z = wp.tile_zeros(dtype=float, shape=(TILE_N, TILE_N))
    wp.tile_matmul(y, x, z)

    wp.tile_store(output, z)


def test_tile_transpose_matmul(test, device):
    rng = np.random.default_rng(42)
    input = wp.array(rng.random((TILE_M, TILE_N), dtype=np.float32), device=device)
    output = wp.zeros((TILE_N, TILE_N), dtype=float, device=device)

    wp.launch_tiled(
        test_tile_transpose_matmul_kernel, dim=[1], inputs=[input, output], block_dim=TILE_DIM, device=device
    )

    assert_np_equal(output.numpy(), input.numpy().T @ input.numpy(), 1e-6)


@wp.kernel
def test_tile_matmul_return_form_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    """Test the c = wp.tile_matmul(a, b) form which returns a fresh tile."""
    a = wp.tile_load(A, shape=(TILE_M, TILE_K))
    b = wp.tile_load(B, shape=(TILE_K, TILE_N))

    # Use the return form (not the accumulate form)
    # This tests that we don't read from the uninitialized output tile
    c = wp.tile_matmul(a, b)

    wp.tile_store(C, c)


def test_tile_matmul_return_form(test, device):
    """Test that c = wp.tile_matmul(a, b) works correctly with verify_fp.

    This specifically tests a fix where the return form was incorrectly
    reading from the uninitialized output tile (which could contain NaN
    when verify_fp is enabled, causing the result to be NaN).
    """
    # Enable verify_fp to trigger NaN initialization of tiles
    old_verify_fp = wp.config.verify_fp
    wp.config.verify_fp = True

    try:
        M = TILE_M
        K = TILE_K
        N = TILE_N

        rng = np.random.default_rng(42)
        A = rng.random((M, K), dtype=np.float32)
        B = rng.random((K, N), dtype=np.float32)
        expected = A @ B

        A_wp = wp.array(A, device=device)
        B_wp = wp.array(B, device=device)
        C_wp = wp.zeros((M, N), dtype=float, device=device)

        wp.launch_tiled(
            test_tile_matmul_return_form_kernel,
            dim=[1],
            inputs=[A_wp, B_wp, C_wp],
            block_dim=TILE_DIM,
            device=device,
        )

        result = C_wp.numpy()

        # Check that result doesn't contain NaN (which would happen with the bug)
        test.assertFalse(np.any(np.isnan(result)), "Result contains NaN values")

        # Check correctness
        assert_np_equal(result, expected, tol=1e-5)
    finally:
        wp.config.verify_fp = old_verify_fp


@wp.kernel(module="unique")
def f32_to_bf16_kernel(input: wp.array2d[wp.float32], output: wp.array2d[wp.bfloat16]):
    i, j = wp.tid()
    output[i, j] = wp.bfloat16(input[i, j])


@wp.kernel(module="unique")
def bf16_to_f32_kernel(input: wp.array2d[wp.bfloat16], output: wp.array2d[wp.float32]):
    i, j = wp.tid()
    output[i, j] = wp.float32(input[i, j])


# cuBLASDx restricts the accumulator dtype to float16, float32, or float64. The kernel goes
# in its own module with enable_backward=False so that backward LTOs (which would use
# bfloat16 accumulators for adjA, adjB) are not generated.
@wp.kernel(module="unique", module_options={"enable_backward": False})
def tile_gemm_bf16(A: wp.array2d[wp.bfloat16], B: wp.array2d[wp.bfloat16], C: wp.array2d[wp.float32]):
    i, j = wp.tid()

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

    K = A.shape[1]
    count = int(K / TILE_K)

    for k in range(0, count):
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))

        wp.tile_matmul(a, b, sum)

    wp.tile_store(C, sum, offset=(i * TILE_M, j * TILE_N))


def test_tile_gemm_bf16(test, device):
    M = TILE_M * 7
    K = TILE_K * 6
    N = TILE_N * 5

    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)

    # Convert float32 numpy data to bfloat16 on device via kernels
    A_f32 = wp.array(A, device=device)
    B_f32 = wp.array(B, device=device)
    A_wp = wp.zeros((M, K), dtype=wp.bfloat16, device=device)
    B_wp = wp.zeros((K, N), dtype=wp.bfloat16, device=device)
    C_wp = wp.zeros((M, N), dtype=wp.float32, device=device)

    wp.launch(f32_to_bf16_kernel, dim=(M, K), inputs=[A_f32, A_wp], device=device)
    wp.launch(f32_to_bf16_kernel, dim=(K, N), inputs=[B_f32, B_wp], device=device)

    wp.launch_tiled(
        tile_gemm_bf16,
        dim=(int(M / TILE_M), int(N / TILE_N)),
        inputs=[A_wp, B_wp, C_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    # Quantize reference inputs through bfloat16 to match what the kernel sees
    A_ref = wp.zeros((M, K), dtype=wp.float32, device=device)
    B_ref = wp.zeros((K, N), dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=(M, K), inputs=[A_wp, A_ref], device=device)
    wp.launch(bf16_to_f32_kernel, dim=(K, N), inputs=[B_wp, B_ref], device=device)

    np.testing.assert_allclose(C_wp.numpy(), A_ref.numpy() @ B_ref.numpy(), rtol=1.0e-2)


def test_tile_matmul_bf16_out_rejected(test, device):
    """Reject a ``bfloat16`` output accumulator in ``tile_matmul``.

    Require an output accumulator with a ``float16``, ``float32``, or ``float64``
    data type.
    """

    @wp.kernel(module="unique")
    def kernel_bf16_out(
        A: wp.array2d[wp.bfloat16],
        B: wp.array2d[wp.bfloat16],
        C: wp.array2d[wp.bfloat16],
    ):
        i, j = wp.tid()
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, j * TILE_N))
        c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.bfloat16)
        wp.tile_matmul(a, b, c)
        wp.tile_store(C, c, offset=(i * TILE_M, j * TILE_N))

    A = wp.zeros((TILE_M, TILE_K), dtype=wp.bfloat16, device=device)
    B = wp.zeros((TILE_K, TILE_N), dtype=wp.bfloat16, device=device)
    C = wp.zeros((TILE_M, TILE_N), dtype=wp.bfloat16, device=device)

    with test.assertRaisesRegex(TypeError, r"does not support a bfloat16 'out' tile"):
        wp.launch_tiled(kernel_bf16_out, dim=(1, 1), inputs=[A, B, C], block_dim=TILE_DIM, device=device)


def test_tile_matmul_bf16_out_rejected_return_form(test, device):
    """Reject a synthesized ``bfloat16`` output from ``tile_matmul``.

    The two-argument form ``c = wp.tile_matmul(a, b)`` synthesizes ``out`` with
    ``dtype=a.dtype``, so ``bfloat16`` inputs must reach the same rejection path.
    """

    @wp.kernel(module="unique")
    def kernel_bf16_return(
        A: wp.array2d[wp.bfloat16],
        B: wp.array2d[wp.bfloat16],
        C: wp.array2d[wp.bfloat16],
    ):
        i, j = wp.tid()
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, j * TILE_N))
        c = wp.tile_matmul(a, b)
        wp.tile_store(C, c, offset=(i * TILE_M, j * TILE_N))

    A = wp.zeros((TILE_M, TILE_K), dtype=wp.bfloat16, device=device)
    B = wp.zeros((TILE_K, TILE_N), dtype=wp.bfloat16, device=device)
    C = wp.zeros((TILE_M, TILE_N), dtype=wp.bfloat16, device=device)

    with test.assertRaisesRegex(TypeError, r"bfloat16 'out' tile"):
        wp.launch_tiled(kernel_bf16_return, dim=(1, 1), inputs=[A, B, C], block_dim=TILE_DIM, device=device)


def test_tile_matmul_bf16_a_with_backward_rejected(test, device):
    """Reject a ``bfloat16`` left operand during ``tile_matmul`` differentiation.

    The left operand is the accumulator for ``adjA``.
    """

    @wp.kernel(module="unique")
    def kernel_bf16_a(
        A: wp.array2d[wp.bfloat16],
        B: wp.array2d[wp.float32],
        C: wp.array2d[wp.float32],
    ):
        i, j = wp.tid()
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, j * TILE_N))
        c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
        wp.tile_matmul(a, b, c)
        wp.tile_store(C, c, offset=(i * TILE_M, j * TILE_N))

    A = wp.zeros((TILE_M, TILE_K), dtype=wp.bfloat16, device=device)
    B = wp.zeros((TILE_K, TILE_N), dtype=wp.float32, device=device)
    C = wp.zeros((TILE_M, TILE_N), dtype=wp.float32, device=device)

    with test.assertRaisesRegex(TypeError, r"bfloat16 'a' or 'b' tiles when the backward pass is enabled"):
        wp.launch_tiled(kernel_bf16_a, dim=(1, 1), inputs=[A, B, C], block_dim=TILE_DIM, device=device)


def test_tile_matmul_bf16_b_with_backward_rejected(test, device):
    """Reject a ``bfloat16`` right operand during ``tile_matmul`` differentiation.

    The right operand is the accumulator for ``adjB``.
    """

    @wp.kernel(module="unique")
    def kernel_bf16_b(
        A: wp.array2d[wp.float32],
        B: wp.array2d[wp.bfloat16],
        C: wp.array2d[wp.float32],
    ):
        i, j = wp.tid()
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, j * TILE_N))
        c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
        wp.tile_matmul(a, b, c)
        wp.tile_store(C, c, offset=(i * TILE_M, j * TILE_N))

    A = wp.zeros((TILE_M, TILE_K), dtype=wp.float32, device=device)
    B = wp.zeros((TILE_K, TILE_N), dtype=wp.bfloat16, device=device)
    C = wp.zeros((TILE_M, TILE_N), dtype=wp.float32, device=device)

    with test.assertRaisesRegex(TypeError, r"bfloat16 'a' or 'b' tiles when the backward pass is enabled"):
        wp.launch_tiled(kernel_bf16_b, dim=(1, 1), inputs=[A, B, C], block_dim=TILE_DIM, device=device)


def test_tile_matmul_complex_rejected(test, device):
    """Check that ``wp.tile_matmul(a, b, out)`` rejects complex (vec2) tiles with an actionable error.

    Complex GEMM is not implemented, so a clear ``TypeError`` should be raised rather than an opaque
    overload-resolution failure. All three complex element types (vec2h, vec2f, vec2d) are exercised
    since the error overloads match ``vector(length=2, dtype=Float)``.
    """
    for vec_dtype in (wp.vec2h, wp.vec2f, wp.vec2d):

        @wp.kernel(module="unique")
        def kernel_complex_out(A: wp.array2d[vec_dtype], B: wp.array2d[vec_dtype], C: wp.array2d[vec_dtype]):
            a = wp.tile_load(A, shape=(TILE_M, TILE_K))
            b = wp.tile_load(B, shape=(TILE_K, TILE_N))
            c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=vec_dtype)  # noqa: B023 (kernel is launched in the same iteration)
            wp.tile_matmul(a, b, c)
            wp.tile_store(C, c)

        A = wp.zeros((TILE_M, TILE_K), dtype=vec_dtype, device=device)
        B = wp.zeros((TILE_K, TILE_N), dtype=vec_dtype, device=device)
        C = wp.zeros((TILE_M, TILE_N), dtype=vec_dtype, device=device)

        with test.assertRaisesRegex(TypeError, r"does not support complex tiles"):
            wp.launch_tiled(kernel_complex_out, dim=[1], inputs=[A, B, C], block_dim=TILE_DIM, device=device)


def test_tile_matmul_complex_rejected_return_form(test, device):
    """Check that the returning form ``c = wp.tile_matmul(a, b)`` rejects complex (vec2) tiles the same way."""
    for vec_dtype in (wp.vec2h, wp.vec2f, wp.vec2d):

        @wp.kernel(module="unique")
        def kernel_complex_return(A: wp.array2d[vec_dtype], B: wp.array2d[vec_dtype], C: wp.array2d[vec_dtype]):
            a = wp.tile_load(A, shape=(TILE_M, TILE_K))
            b = wp.tile_load(B, shape=(TILE_K, TILE_N))
            c = wp.tile_matmul(a, b)
            wp.tile_store(C, c)

        A = wp.zeros((TILE_M, TILE_K), dtype=vec_dtype, device=device)
        B = wp.zeros((TILE_K, TILE_N), dtype=vec_dtype, device=device)
        C = wp.zeros((TILE_M, TILE_N), dtype=vec_dtype, device=device)

        with test.assertRaisesRegex(TypeError, r"does not support complex tiles"):
            wp.launch_tiled(kernel_complex_return, dim=[1], inputs=[A, B, C], block_dim=TILE_DIM, device=device)


@wp.kernel
def tile_matmul_reshaped_view_kernel(A: wp.array2d[wp.float16], B: wp.array2d[wp.float16], C: wp.array2d[wp.float16]):
    # a row-offset view of a tile whose rows are 8 bytes wide puts the base pointer at +8 bytes;
    # reshaping it to 16-element rows (32 bytes) must not let cuBLASDx assume a 16-byte base
    t = wp.tile_load(A, shape=(129, 4))
    v = wp.tile_view(t, offset=(1, 0), shape=(128, 4))
    r = wp.tile_reshape(v, shape=(32, 16))
    b = wp.tile_load(B, shape=(16, 32))
    c = wp.tile_matmul(r, b)
    wp.tile_store(C, c)


# Forward-only alignment fixtures skip adjoint codegen, which would add two unused GEMM LTOs per kernel
wp.get_module("test_tile_matmul_forward").options["enable_backward"] = False


@wp.kernel(module="test_tile_matmul_forward")
def tile_matmul_row_offset_views_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    # 16-byte-wide rows: row-offset views keep a 16-byte base, but they are still aliases, so
    # nothing is declared; used only for its LTO symbols
    a = wp.tile_view(wp.tile_load(A, shape=(25, 4)), offset=(1, 0), shape=(24, 4))
    b = wp.tile_view(wp.tile_load(B, shape=(5, 4)), offset=(1, 0), shape=(4, 4))
    c = wp.tile_view(wp.tile_zeros(shape=(25, 4), dtype=float), offset=(1, 0), shape=(24, 4))
    wp.tile_matmul(a, b, c)
    wp.tile_store(C, c)


@wp.kernel(module="test_tile_matmul_forward")
def tile_matmul_matvec_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    # single-column B and C: only A qualifies for 16-byte alignment
    a = wp.tile_load(A, shape=(32, 32))
    b = wp.tile_load(B, shape=(32, 1))
    c = wp.tile_matmul(a, b)
    wp.tile_store(C, c)


@wp.kernel
def tile_matmul_mixed_dtype_kernel(A: wp.array2d[wp.float16], B: wp.array2d[wp.float16], C: wp.array2d[float]):
    # fp16 operands with 12-element rows (24 bytes) and an fp32 accumulator with 48-byte rows
    a = wp.tile_load(A, shape=(16, 12))
    b = wp.tile_load(B, shape=(12, 12))
    c = wp.tile_zeros(shape=(16, 12), dtype=float)
    wp.tile_matmul(a, b, c)
    wp.tile_store(C, c)


@wp.kernel(module="unique", module_options={"enable_backward": False})
def tile_matmul_fp16_n16_kernel(A: wp.array2d[wp.float16], B: wp.array2d[wp.float16], C: wp.array2d[wp.float16]):
    # every operand qualifies for 16-byte alignment, but cuBLASDx rejects this fp16 layout when C
    # is declared 16-byte aligned on sm_90 and newer, so C must be declared at 8 bytes instead
    a = wp.tile_load(A, shape=(32, 16))
    b = wp.tile_load(B, shape=(16, 16))
    c = wp.tile_matmul(a, b)
    wp.tile_store(C, c)


@wp.func(module="test_tile_matmul_forward")
def tile_matmul_func_param(
    a: wp.tile[wp.float16, 32, 16], b: wp.tile[wp.float16, 16, 48], c: wp.tile[wp.float16, 32, 48]
):
    wp.tile_matmul(a, b, c)


@wp.kernel(module="test_tile_matmul_forward")
def tile_matmul_view_through_func_kernel(
    A: wp.array2d[wp.float16], B: wp.array2d[wp.float16], C: wp.array2d[wp.float16]
):
    # inside the function the parameter type is the annotation, so the dispatch cannot see that
    # the caller passed a reshaped row-offset view whose base pointer is only 8-byte aligned
    t = wp.tile_load(A, shape=(129, 4))
    v = wp.tile_reshape(wp.tile_view(t, offset=(1, 0), shape=(128, 4)), shape=(32, 16))
    b = wp.tile_load(B, shape=(16, 48), storage="shared")
    c = wp.tile_zeros(shape=(32, 48), dtype=wp.float16, storage="shared")
    tile_matmul_func_param(v, b, c)
    wp.tile_store(C, c)


def test_tile_matmul_alignment_matrix(test, device):
    """Compute correct cuBLASDx GEMMs across the operand classes the alignment declaration distinguishes.

    Each section declares a different alignment: an fp16 row-offset view whose base pointer is
    offset by 8 bytes even though reshaping gives it a dense 32-byte leading dimension, the same
    view passed through a ``@wp.func`` parameter, an fp16 layout cuBLASDx refuses to compile with
    a 16-byte-aligned C operand, a matrix-vector product where only A qualifies, and mixed
    fp16/fp32 element sizes. Operands that declare no alignment compile to the pre-existing LTO
    and are covered by the decision-matrix and symbol tests.
    """
    if not wp._src.context.runtime.core.wp_is_mathdx_enabled():
        test.skipTest("MathDx is not enabled")

    rng = np.random.default_rng(7)

    def run(kernel, a, b, c_shape, wpdt, block_dim=64):
        A = wp.array(a, device=device)
        B = wp.array(b, device=device)
        C = wp.zeros(c_shape, dtype=wpdt, device=device)
        wp.launch_tiled(kernel, dim=[1], inputs=[A, B, C], block_dim=block_dim, device=device)
        return C.numpy().astype(np.float64)

    # reshape of a row-offset fp16 view feeding A
    a = rng.standard_normal((129, 4)).astype(np.float16)
    b = rng.standard_normal((16, 32)).astype(np.float16)
    ref = a[1:].astype(np.float64).reshape(32, 16) @ b.astype(np.float64)
    np.testing.assert_allclose(
        run(tile_matmul_reshaped_view_kernel, a, b, (32, 32), wp.float16), ref, rtol=2e-2, atol=5e-2
    )

    # the same reshaped view, passed through a @wp.func parameter
    b = rng.standard_normal((16, 48)).astype(np.float16)
    ref = a[1:].astype(np.float64).reshape(32, 16) @ b.astype(np.float64)
    np.testing.assert_allclose(
        run(tile_matmul_view_through_func_kernel, a, b, (32, 48), wp.float16), ref, rtol=2e-2, atol=5e-2
    )

    # fp16 32x16x16 with every operand aligned at 128 threads: the 16-bit C is capped at 8 bytes,
    # which cuBLASDx accepts where a 16-byte C is rejected on sm_90 and newer
    a = rng.standard_normal((32, 16)).astype(np.float16)
    b = rng.standard_normal((16, 16)).astype(np.float16)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    np.testing.assert_allclose(
        run(tile_matmul_fp16_n16_kernel, a, b, (32, 16), wp.float16, block_dim=128), ref, rtol=2e-2, atol=5e-2
    )

    # matrix-vector product
    a = rng.standard_normal((32, 32)).astype(np.float32)
    b = rng.standard_normal((32, 1)).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    np.testing.assert_allclose(run(tile_matmul_matvec_kernel, a, b, (32, 1), wp.float32), ref, rtol=1e-5, atol=1e-4)

    # fp16 operands, fp32 accumulator (cuBLASDx fp16 inputs carry ~1e-3 relative error)
    a = rng.standard_normal((16, 12)).astype(np.float16)
    b = rng.standard_normal((12, 12)).astype(np.float16)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    np.testing.assert_allclose(
        run(tile_matmul_mixed_dtype_kernel, a, b, (16, 12), wp.float32), ref, rtol=1e-2, atol=2e-2
    )


def test_tile_matmul_lto_operators(test, device):
    """Encode per-operand alignment and the static block dimension in the cuBLASDx LTO symbols.

    Runs codegen for this module, whose kernels cover distinct alignment classes, and reads the
    GEMM LTO declarations from the generated source, so the check is independent of the kernel cache.
    """
    if not wp._src.context.runtime.core.wp_is_mathdx_enabled():
        test.skipTest("MathDx is not enabled")

    arch = wp.get_device(device).get_cuda_compile_arch()

    def gemm_symbols(module, block_dim):
        options = module.resolve_options(wp.config, block_dim=block_dim) | {"output_arch": arch}
        source, *_ = module._run_codegen(options, is_cpu=False)
        found = set(re.findall(r"void (dot_\w+)\(", source))
        test.assertTrue(found, "expected GEMM LTO declarations in the generated source")
        return found

    def check(symbols, shape, precisions, expected):
        # symbol layout: dot_{M}_{N}_{K}_{arch}_{threads}_{arrangements}_{precA}_{precB}_{precC}_{type}[_lds]_al{A}_{B}_{C}_sb{N}
        prefix, marker = f"dot_{shape}_", f"_{precisions}_"
        matching = {s for s in symbols if s.startswith(prefix) and marker in s}
        test.assertTrue(matching, f"no GEMM LTO for shape {shape} with precisions {precisions}")
        found = {s.rsplit("_al", 1)[1] for s in matching}
        if found == {"0_0_0_sb1"} and expected != "0_0_0":
            # cuBLASDx rejected the aligned variant on this platform and the fallback was used
            aligned = {s.rsplit("_al", 1)[0] + "_al" + expected for s in matching}
            test.assertTrue(aligned & _gemm_lto_alignment_unsupported, f"unexpected unaligned LTO for {shape}")
            return
        test.assertEqual(found, {expected + "_sb1"})

    symbols = gemm_symbols(tile_gemm.module, TILE_DIM)
    forward_symbols = gemm_symbols(tile_matmul_matvec_kernel.module, TILE_DIM)

    # tile_gemm: dense fp32 8x4x8 tiles that own their storage -> every operand 16-byte aligned
    check(symbols, "8_4_8", "5_5_5", "16_16_16")
    # the fp16 instantiation has 8-byte B and C rows, which keep the 2-byte default
    check(symbols, "8_4_8", "3_3_3", "16_2_2")
    # matvec: single-column B and C keep their 4-byte default while A qualifies
    check(forward_symbols, "32_1_32", "5_5_5", "16_4_4")
    # reshaped-view A (fp16) is never declared aligned even with 32-byte rows; B and C are owners,
    # and a 16-bit C is capped at 8 bytes
    check(symbols, "32_32_16", "3_3_3", "2_16_8")
    # its adjoints permute the operands: the view is C in adjA and the transposed A in adjB
    check(symbols, "32_16_32", "3_3_3", "16_16_2")
    check(symbols, "16_32_32", "3_3_3", "2_16_8")
    # row-offset views on every operand: nothing qualifies, operator left unset
    check(forward_symbols, "24_4_4", "5_5_5", "0_0_0")
    # mixed dtypes: 24-byte fp16 rows keep the 2-byte default, the 48-byte fp32 rows qualify,
    # and the adjoints carry each operand's element size along with its position
    check(symbols, "16_12_12", "3_3_5", "2_2_16")
    check(symbols, "16_12_12", "5_3_3", "16_2_2")
    check(symbols, "12_12_16", "3_5_3", "2_16_2")
    # inside a @wp.func every operand is a parameter whose alignment is unknown, so nothing is declared
    check(forward_symbols, "32_48_16", "3_3_3", "0_0_0")
    # every operand qualifies; the 16-bit C is capped at 8 bytes, which cuBLASDx accepts for this
    # layout where a 16-byte C is rejected (its own module, inspected at the 128-thread block size)
    check(gemm_symbols(tile_matmul_fp16_n16_kernel.module, 128), "32_16_16", "3_3_3", "16_16_8")


def test_tile_matmul_alignment_fallback(test, device):
    """Fall back to the unaligned GEMM LTO when cuBLASDx rejects the aligned one.

    cuBLASDx refuses some layouts only when an operand is declared 16-byte aligned, and which
    layouts depends on the architecture and block size, so the rejection is simulated by failing
    every native compile that requests a non-default alignment. Covers the warning, the error suppression
    on the probe, the once-per-process memo (including when the unaligned variant was already built for
    another GEMM in the same module), and the error when no variant compiles.
    """
    if not wp._src.context.runtime.core.wp_is_mathdx_enabled():
        test.skipTest("MathDx is not enabled")

    # a shape no other test compiles, so the persistent LTO cache cannot satisfy the aligned variant
    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
        a = wp.tile_load(A, shape=(24, 8))
        b = wp.tile_load(B, shape=(8, 20))
        c = wp.tile_matmul(a, b)
        wp.tile_store(C, c)

    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def kernel_again(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
        a = wp.tile_load(A, shape=(24, 8))
        b = wp.tile_load(B, shape=(8, 20))
        c = wp.tile_matmul(a, b)
        wp.tile_store(C, c)

    # one kernel whose first GEMM is unaligned from the start (every operand is a row-offset view),
    # followed by two owner-tile GEMMs of the same shape whose aligned variant is rejected; the
    # calls dispatch in source order, unlike kernels within a module
    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def kernel_view_then_owners(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
        av = wp.tile_view(wp.tile_load(A, shape=(25, 16)), offset=(1, 0), shape=(24, 16))
        bv = wp.tile_view(wp.tile_load(B, shape=(17, 20)), offset=(1, 0), shape=(16, 20))
        cv = wp.tile_view(wp.tile_zeros(shape=(25, 20), dtype=float), offset=(1, 0), shape=(24, 20))
        wp.tile_matmul(av, bv, cv)
        a = wp.tile_load(A, shape=(24, 16))
        b = wp.tile_load(B, shape=(16, 20))
        c = wp.tile_matmul(a, b)
        d = wp.tile_matmul(a, b)
        wp.tile_store(C, c + d + cv)

    core = wp._src.context.runtime.core
    real_compile = core.wp_cuda_compile_dot
    calls = []

    def reject_aligned(*args):
        # trailing arguments are (alignment_A, alignment_B, alignment_C, enable_static_block_dim, suppress_errors)
        alignment, suppress_errors = args[-5:-2], args[-1]
        calls.append((alignment, suppress_errors))
        if any(alignment):
            return False
        return real_compile(*args)

    def reject_all(*args):
        return False

    def codegen(module):
        options = module.resolve_options(wp.config, block_dim=TILE_DIM) | {
            "output_arch": wp.get_device(device).get_cuda_compile_arch()
        }
        # Warp's warnings go to stderr rather than warnings.catch_warnings(record=True)
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()) as stderr:
            warnings.simplefilter("always")
            source, *_ = module._run_codegen(options, is_cpu=False)
        return set(re.findall(r"void (dot_\w+)\(", source)), stderr.getvalue()

    saved = set(_gemm_lto_alignment_unsupported)
    try:
        with mock.patch.object(core, "wp_cuda_compile_dot", reject_aligned):
            symbols, log = codegen(kernel.module)
        test.assertTrue(symbols)
        for symbol in symbols:
            test.assertTrue(symbol.endswith("_al0_0_0_sb1"), symbol)
        # developers must be able to see that a fallback happened and for which GEMM
        test.assertIn("cuBLASDx rejected the 16-byte-aligned GEMM for tile_matmul (float32 24x20x8", log)
        # the aligned probe suppresses the native error print; the unaligned retry does not (it
        # may be served from the persistent LTO cache without a native call)
        test.assertEqual({suppress for alignment, suppress in calls if any(alignment)}, {1})
        test.assertEqual({suppress for alignment, suppress in calls if not any(alignment)} - {0}, set())
        # the rejected aligned variants are remembered so later builds skip the failing compile
        rejected = {s for s in _gemm_lto_alignment_unsupported if s.startswith("dot_24_20_8_")}
        test.assertTrue(rejected)

        calls.clear()
        with mock.patch.object(core, "wp_cuda_compile_dot", reject_aligned):
            symbols, log = codegen(kernel_again.module)
        for symbol in symbols:
            test.assertTrue(symbol.endswith("_al0_0_0_sb1"), symbol)
        test.assertFalse([alignment for alignment, _ in calls if any(alignment)], "aligned compile retried")
        test.assertNotIn("cuBLASDx rejected", log)

        # when the unaligned variant already exists in the module (built for the view GEMM), the
        # rejection is still recorded, so the third GEMM does not retry the aligned compile
        calls.clear()
        with mock.patch.object(core, "wp_cuda_compile_dot", reject_aligned):
            symbols, log = codegen(kernel_view_then_owners.module)
        test.assertEqual({s for s in symbols if s.startswith("dot_24_20_16_")}, symbols)
        for symbol in symbols:
            test.assertTrue(symbol.endswith("_al0_0_0_sb1"), symbol)
        test.assertEqual(len([alignment for alignment, _ in calls if any(alignment)]), 1, "aligned compile retried")
        test.assertEqual(log.count("cuBLASDx rejected"), 1)
        test.assertTrue({s for s in _gemm_lto_alignment_unsupported if s.startswith("dot_24_20_16_")})

        # when no variant compiles the error names the last attempted symbol; the persistent LTO
        # cache is bypassed so the unaligned variant is really recompiled
        _gemm_lto_alignment_unsupported.difference_update(rejected)
        with (
            mock.patch.object(core, "wp_cuda_compile_dot", reject_all),
            mock.patch.object(warp_build, "get_cached_lto", return_value=None),
        ):
            with test.assertRaisesRegex(RuntimeError, r"Failed to compile LTO 'dot_24_20_8_\w+_al0_0_0_sb1'"):
                codegen(kernel.module)
        # a failed fallback does not disable the aligned variant for the rest of the process
        test.assertFalse({s for s in _gemm_lto_alignment_unsupported if s.startswith("dot_24_20_8_")})
    finally:
        _gemm_lto_alignment_unsupported.clear()
        _gemm_lto_alignment_unsupported.update(saved)


class TestTileMatmul(unittest.TestCase):
    def test_gemm_operand_alignments(self):
        """Declare 16-byte cuBLASDx alignment per operand only for owning tiles whose leading dimension allows it.

        The alignment operator asserts each operand's base pointer is 16-byte aligned. That holds
        for tiles that own their shared storage when the operand's leading dimension spans a
        multiple of 16 bytes, judged with the operand's own element size. Aliasing operands (views,
        reshapes, transposes) and explicit-leading-dimension GEMMs never qualify, and when nothing
        qualifies the operator is left unset (all zeros).
        """
        own = (True, True, True)
        f32 = (4, 4, 4)
        # dense fp32 tiles with rows of 16, 8, 4 elements: 64, 32, 16 bytes -> all aligned
        self.assertEqual(_gemm_operand_alignments((16, 8, 4), (0, 0, 0), f32, own), (16, 16, 16))
        # a 6-element fp32 row (24 bytes) keeps only that operand at its 4-byte default
        self.assertEqual(_gemm_operand_alignments((8, 6, 8), (0, 0, 0), f32, own), (16, 4, 16))
        # single-column B and C (matvec): only A qualifies
        self.assertEqual(_gemm_operand_alignments((32, 1, 1), (0, 0, 0), f32, own), (16, 4, 4))
        # an aliasing operand is never declared aligned, even with qualifying rows
        self.assertEqual(_gemm_operand_alignments((16, 16, 16), (0, 0, 0), f32, (False, True, True)), (4, 16, 16))
        # nothing qualifies -> operator left unset
        self.assertEqual(_gemm_operand_alignments((6, 6, 6), (0, 0, 0), f32, own), (0, 0, 0))
        self.assertEqual(_gemm_operand_alignments((16, 16, 16), (0, 0, 0), f32, (False, False, False)), (0, 0, 0))
        # a capped 16-bit C still improves on its 2-byte default, so the operator is declared
        self.assertEqual(_gemm_operand_alignments((16, 16, 16), (0, 0, 0), (2, 2, 2), (False, False, True)), (2, 2, 8))
        # explicit leading dimensions describe a view -> operator left unset
        self.assertEqual(_gemm_operand_alignments((16, 16, 16), (32, 32, 32), f32, own), (0, 0, 0))
        # mixed dtypes are judged per operand: fp16 A/B with 12-element rows are 24 bytes (default 2),
        # while the fp32 C with 12-element rows is 48 bytes (aligned)
        self.assertEqual(_gemm_operand_alignments((12, 12, 12), (0, 0, 0), (2, 2, 4), own), (2, 2, 16))
        # fp16 needs 8-element rows, fp64 needs 2-element rows; a 16-bit C is capped at 8 bytes
        self.assertEqual(_gemm_operand_alignments((8, 8, 8), (0, 0, 0), (2, 2, 2), own), (16, 16, 8))
        self.assertEqual(_gemm_operand_alignments((8, 8, 8), (0, 0, 0), (2, 2, 4), own), (16, 16, 16))
        self.assertEqual(_gemm_operand_alignments((3, 4, 4), (0, 0, 0), (8, 8, 8), own), (8, 16, 16))


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()

# bfloat16 requires CC >= 8.0 (Ampere+)
bf16_devices = get_cpu_test_devices()
for cuda_device in cuda_devices:
    if cuda_device.arch >= 80:
        bf16_devices.append(cuda_device)

add_function_test(TestTileMatmul, "test_tile_gemm_fp16", test_tile_gemm(wp.float16), devices=devices)
add_function_test(
    TestTileMatmul, "test_tile_matmul_alignment_matrix", test_tile_matmul_alignment_matrix, devices=cuda_devices
)
add_function_test(
    TestTileMatmul, "test_tile_matmul_lto_operators", test_tile_matmul_lto_operators, devices=cuda_devices
)
add_function_test(
    TestTileMatmul, "test_tile_matmul_alignment_fallback", test_tile_matmul_alignment_fallback, devices=cuda_devices
)
add_function_test(TestTileMatmul, "test_tile_gemm_bf16", test_tile_gemm_bf16, devices=bf16_devices, check_output=False)
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_bf16_out_rejected",
    test_tile_matmul_bf16_out_rejected,
    devices=bf16_devices,
    check_output=False,
)
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_bf16_out_rejected_return_form",
    test_tile_matmul_bf16_out_rejected_return_form,
    devices=bf16_devices,
    check_output=False,
)
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_bf16_a_with_backward_rejected",
    test_tile_matmul_bf16_a_with_backward_rejected,
    devices=bf16_devices,
    check_output=False,
)
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_bf16_b_with_backward_rejected",
    test_tile_matmul_bf16_b_with_backward_rejected,
    devices=bf16_devices,
    check_output=False,
)
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_complex_rejected",
    test_tile_matmul_complex_rejected,
    devices=devices,
)
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_complex_rejected_return_form",
    test_tile_matmul_complex_rejected_return_form,
    devices=devices,
)
add_function_test(TestTileMatmul, "test_tile_gemm_fp32", test_tile_gemm(wp.float32), devices=devices)
add_function_test(TestTileMatmul, "test_tile_gemm_fp64", test_tile_gemm(wp.float64), devices=devices)
add_function_test(TestTileMatmul, "test_tile_grouped_gemm", test_tile_grouped_gemm, devices=devices)
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_mixed_precision",
    test_tile_matmul_mixed_precision,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_pipelined_reassign",
    test_tile_matmul_pipelined_reassign,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_reassign_backward",
    test_tile_matmul_reassign_backward,
    devices=devices,
    check_output=False,
)
add_function_test(TestTileMatmul, "test_tile_transpose_matmul", test_tile_transpose_matmul, devices=devices)
add_function_test(TestTileMatmul, "test_tile_matmul_return_form", test_tile_matmul_return_form, devices=devices)

cpu_devices = get_cpu_test_devices()
for name, func in (
    ("test_tile_gemm_fp32_cpu_blocks", test_tile_gemm(wp.float32)),
    ("test_tile_grouped_gemm_cpu_blocks", test_tile_grouped_gemm),
    ("test_tile_transpose_matmul_cpu_blocks", test_tile_transpose_matmul),
    ("test_tile_matmul_return_form_cpu_blocks", test_tile_matmul_return_form),
):
    add_function_test(TestTileMatmul, name, func, devices=cpu_devices, enable_cpu_blocks=True)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
