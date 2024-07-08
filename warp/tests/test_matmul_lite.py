# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.init()  # For wp.context.runtime.core.is_cutlass_enabled()


class gemm_test_bed_runner:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    def alloc(self, m, n, k, batch_count):
        rng = np.random.default_rng(42)
        low = -4.5
        high = 3.5
        if batch_count == 1:
            A = wp.array2d(
                np.ceil(rng.uniform(low=low, high=high, size=(m, k))),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            B = wp.array2d(
                np.ceil(rng.uniform(low=low, high=high, size=(k, n))),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            C = wp.array2d(
                np.ceil(rng.uniform(low=low, high=high, size=(m, n))),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            D = wp.array2d(np.zeros((m, n)), dtype=self.dtype, device=self.device, requires_grad=True)
        else:
            A = wp.array3d(
                np.ceil(rng.uniform(low=low, high=high, size=(batch_count, m, k))),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            B = wp.array3d(
                np.ceil(rng.uniform(low=low, high=high, size=(batch_count, k, n))),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            C = wp.array3d(
                np.ceil(rng.uniform(low=low, high=high, size=(batch_count, m, n))),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            D = wp.array3d(np.zeros((batch_count, m, n)), dtype=self.dtype, device=self.device, requires_grad=True)
        return A, B, C, D

    def run_and_verify(self, m, n, k, batch_count, alpha, beta):
        A, B, C, D = self.alloc(m, n, k, batch_count)
        ones = wp.zeros_like(D)
        ones.fill_(1.0)

        if batch_count == 1:
            tape = wp.Tape()
            with tape:
                wp.matmul(A, B, C, D, alpha, beta, False)
            tape.backward(grads={D: ones})

            D_np = alpha * (A.numpy() @ B.numpy()) + beta * C.numpy()
            assert_np_equal(D.numpy(), D_np)

            adj_A_np = alpha * np.matmul(ones.numpy(), B.numpy().transpose())
            adj_B_np = alpha * (A.numpy().transpose() @ ones.numpy())
            adj_C_np = beta * ones.numpy()

        else:
            tape = wp.Tape()
            with tape:
                wp.batched_matmul(A, B, C, D, alpha, beta, False)
            tape.backward(grads={D: ones})

            D_np = alpha * np.matmul(A.numpy(), B.numpy()) + beta * C.numpy()
            assert_np_equal(D.numpy(), D_np)

            adj_A_np = alpha * np.matmul(ones.numpy(), B.numpy().transpose((0, 2, 1)))
            adj_B_np = alpha * np.matmul(A.numpy().transpose((0, 2, 1)), ones.numpy())
            adj_C_np = beta * ones.numpy()

        assert_np_equal(A.grad.numpy(), adj_A_np)
        assert_np_equal(B.grad.numpy(), adj_B_np)
        assert_np_equal(C.grad.numpy(), adj_C_np)

    def run(self):
        Ms = [8]
        Ns = [16]
        Ks = [32]
        batch_counts = [1]
        betas = [1.0]
        alpha = 1.0

        for batch_count in batch_counts:
            for m in Ms:
                for n in Ns:
                    for k in Ks:
                        for beta in betas:
                            self.run_and_verify(m, n, k, batch_count, alpha, beta)


class gemm_test_bed_runner_transpose:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    def alloc(self, m, n, k, batch_count):
        rng = np.random.default_rng(42)
        low = -4.5
        high = 3.5
        if batch_count == 1:
            A = wp.array2d(
                np.ceil(rng.uniform(low=low, high=high, size=(m, k))),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            B = wp.array2d(
                np.ceil(rng.uniform(low=low, high=high, size=(k, n))),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            C = wp.array2d(
                np.ceil(rng.uniform(low=low, high=high, size=(m, n))),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            D = wp.array2d(np.zeros((m, n)), dtype=self.dtype, device=self.device, requires_grad=True)
            AT = wp.array2d(A.numpy().transpose([1, 0]), dtype=self.dtype, device=self.device, requires_grad=True)
            BT = wp.array2d(B.numpy().transpose([1, 0]), dtype=self.dtype, device=self.device, requires_grad=True)
        else:
            A = wp.array3d(
                np.ceil(rng.uniform(low=low, high=high, size=(batch_count, m, k))),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            B = wp.array3d(
                np.ceil(rng.uniform(low=low, high=high, size=(batch_count, k, n))),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            C = wp.array3d(
                np.ceil(rng.uniform(low=low, high=high, size=(batch_count, m, n))),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            D = wp.array3d(np.zeros((batch_count, m, n)), dtype=self.dtype, device=self.device, requires_grad=True)
            AT = wp.array3d(A.numpy().transpose([0, 2, 1]), dtype=self.dtype, device=self.device, requires_grad=True)
            BT = wp.array3d(B.numpy().transpose([0, 2, 1]), dtype=self.dtype, device=self.device, requires_grad=True)
        return A, B, C, D, AT, BT

    def run_and_verify(self, m, n, k, batch_count, alpha, beta):
        A, B, C1, D1, AT1, BT1 = self.alloc(m, n, k, batch_count)
        C2 = wp.clone(C1)
        C3 = wp.clone(C1)
        D2 = wp.clone(D1)
        D3 = wp.clone(D1)
        AT2 = wp.clone(AT1)
        BT2 = wp.clone(BT1)
        ones1 = wp.zeros_like(D1)
        ones1.fill_(1.0)
        ones2 = wp.zeros_like(D2)
        ones2.fill_(1.0)
        ones3 = wp.zeros_like(D3)
        ones3.fill_(1.0)

        if batch_count == 1:
            ATT1 = AT1.transpose([1, 0])
            BTT1 = BT1.transpose([1, 0])
            ATT2 = AT2.transpose([1, 0])
            BTT2 = BT2.transpose([1, 0])
            tape = wp.Tape()
            with tape:
                wp.matmul(A, BTT1, C1, D1, alpha, beta, False)
                wp.matmul(ATT1, B, C2, D2, alpha, beta, False)
                wp.matmul(ATT2, BTT2, C3, D3, alpha, beta, False)
            tape.backward(grads={D1: ones1, D2: ones2, D3: ones3})

            D_np = alpha * (A.numpy() @ B.numpy()) + beta * C1.numpy()
            assert_np_equal(D1.numpy(), D_np)
            assert_np_equal(D2.numpy(), D_np)
            assert_np_equal(D3.numpy(), D_np)

            adj_A_np = alpha * (ones1.numpy() @ B.numpy().transpose())
            adj_B_np = alpha * (A.numpy().transpose() @ ones1.numpy())
            adj_C_np = beta * ones1.numpy()

        else:
            ATT1 = AT1.transpose([0, 2, 1])
            BTT1 = BT1.transpose([0, 2, 1])
            ATT2 = AT2.transpose([0, 2, 1])
            BTT2 = BT2.transpose([0, 2, 1])
            tape = wp.Tape()
            with tape:
                wp.batched_matmul(A, BTT1, C1, D1, alpha, beta, False)
                wp.batched_matmul(ATT1, B, C2, D2, alpha, beta, False)
                wp.batched_matmul(ATT2, BTT2, C3, D3, alpha, beta, False)
            tape.backward(grads={D1: ones1, D2: ones2, D3: ones3})

            D_np = alpha * np.matmul(A.numpy(), B.numpy()) + beta * C1.numpy()
            assert_np_equal(D1.numpy(), D_np)
            assert_np_equal(D2.numpy(), D_np)
            assert_np_equal(D3.numpy(), D_np)

            adj_A_np = alpha * np.matmul(ones1.numpy(), B.numpy().transpose((0, 2, 1)))
            adj_B_np = alpha * np.matmul(A.numpy().transpose((0, 2, 1)), ones1.numpy())
            adj_C_np = beta * ones1.numpy()

        assert_np_equal(A.grad.numpy(), adj_A_np)
        assert_np_equal(ATT1.grad.numpy(), adj_A_np)
        assert_np_equal(ATT2.grad.numpy(), adj_A_np)
        assert_np_equal(B.grad.numpy(), adj_B_np)
        assert_np_equal(BTT1.grad.numpy(), adj_B_np)
        assert_np_equal(BTT2.grad.numpy(), adj_B_np)
        assert_np_equal(C1.grad.numpy(), adj_C_np)
        assert_np_equal(C2.grad.numpy(), adj_C_np)
        assert_np_equal(C3.grad.numpy(), adj_C_np)

    def run(self):
        m = 8
        n = 16
        k = 32
        batch_counts = [1, 4]
        beta = 1.0
        alpha = 1.0

        for batch_count in batch_counts:
            self.run_and_verify(m, n, k, batch_count, alpha, beta)


@unittest.skipUnless(wp.context.runtime.core.is_cutlass_enabled(), "Warp was not built with CUTLASS support")
def test_f32(test, device):
    gemm_test_bed_runner(wp.float32, device).run()
    gemm_test_bed_runner_transpose(wp.float32, device).run()


@wp.kernel
def matrix_sum_kernel(arr: wp.array2d(dtype=float), loss: wp.array(dtype=float)):
    i, j = wp.tid()
    wp.atomic_add(loss, 0, arr[i, j])


@unittest.skipUnless(wp.context.runtime.core.is_cutlass_enabled(), "Warp was not built with CUTLASS support")
def test_tape(test, device):
    rng = np.random.default_rng(42)
    low = -4.5
    high = 3.5
    m = 8
    n = 16
    k = 32
    A = wp.array2d(
        np.ceil(rng.uniform(low=low, high=high, size=(m, k))), dtype=float, device=device, requires_grad=True
    )
    B = wp.array2d(
        np.ceil(rng.uniform(low=low, high=high, size=(k, n))), dtype=float, device=device, requires_grad=True
    )
    C = wp.array2d(
        np.ceil(rng.uniform(low=low, high=high, size=(m, n))), dtype=float, device=device, requires_grad=True
    )
    D = wp.array2d(np.zeros((m, n)), dtype=float, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    # test tape
    tape = wp.Tape()
    with tape:
        wp.matmul(A, B, C, D)
        wp.launch(matrix_sum_kernel, dim=(m, n), inputs=[D, loss], device=device)

    tape.backward(loss=loss)
    A_grad = A.grad.numpy()
    tape.reset()

    # test adjoint
    D.grad = wp.ones((m, n), dtype=float, device=device)
    wp.adj_matmul(A, B, C, A.grad, B.grad, C.grad, D.grad)
    assert_np_equal(A_grad, A.grad.numpy())

    # test zero
    tape.zero()
    assert_array_equal(A.grad, wp.zeros_like(A))


@unittest.skipUnless(wp.context.runtime.core.is_cutlass_enabled(), "Warp was not built with CUTLASS support")
def test_operator(test, device):
    rng = np.random.default_rng(42)
    low = -4.5
    high = 3.5
    m = 8
    n = 16
    k = 32
    A = wp.array2d(
        np.ceil(rng.uniform(low=low, high=high, size=(m, k))), dtype=float, device=device, requires_grad=True
    )
    B = wp.array2d(
        np.ceil(rng.uniform(low=low, high=high, size=(k, n))), dtype=float, device=device, requires_grad=True
    )
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    # test tape
    tape = wp.Tape()
    with tape:
        D = A @ B
        wp.launch(matrix_sum_kernel, dim=(m, n), inputs=[D, loss], device=device)

    tape.backward(loss=loss)

    # test adjoint
    D.grad = wp.ones((m, n), dtype=float, device=device)
    B_transpose = wp.array2d(B.transpose().numpy(), dtype=float, device=device)

    adj_A = D.grad @ B_transpose
    assert_array_equal(adj_A, A.grad)

    # test zero
    tape.zero()
    assert_array_equal(A.grad, wp.zeros_like(A))


@unittest.skipUnless(wp.context.runtime.core.is_cutlass_enabled(), "Warp was not built with CUTLASS support")
def test_large_batch_count(test, device):
    rng = np.random.default_rng(42)
    low = -4.5
    high = 3.5
    m = 2
    n = 3
    k = 4
    batch_count = 65535 * 2 + int(65535 / 2)
    A = wp.array3d(
        np.ceil(rng.uniform(low=low, high=high, size=(batch_count, m, k))),
        dtype=float,
        device=device,
        requires_grad=True,
    )
    B = wp.array3d(
        np.ceil(rng.uniform(low=low, high=high, size=(batch_count, k, n))),
        dtype=float,
        device=device,
        requires_grad=True,
    )
    C = wp.array3d(
        np.ceil(rng.uniform(low=low, high=high, size=(batch_count, m, n))),
        dtype=float,
        device=device,
        requires_grad=True,
    )
    D = wp.array3d(np.zeros((batch_count, m, n)), dtype=float, device=device, requires_grad=True)
    ones = wp.zeros_like(D)
    ones.fill_(1.0)

    alpha = 1.0
    beta = 1.0

    tape = wp.Tape()
    with tape:
        wp.batched_matmul(A, B, C, D, alpha=alpha, beta=beta, allow_tf32x3_arith=False)
    tape.backward(grads={D: ones})

    D_np = alpha * np.matmul(A.numpy(), B.numpy()) + beta * C.numpy()
    assert_np_equal(D.numpy(), D_np)

    adj_A_np = alpha * np.matmul(ones.numpy(), B.numpy().transpose((0, 2, 1)))
    adj_B_np = alpha * np.matmul(A.numpy().transpose((0, 2, 1)), ones.numpy())
    adj_C_np = beta * ones.numpy()

    assert_np_equal(A.grad.numpy(), adj_A_np)
    assert_np_equal(B.grad.numpy(), adj_B_np)
    assert_np_equal(C.grad.numpy(), adj_C_np)


devices = get_test_devices()


class TestMatmulLite(unittest.TestCase):
    pass


add_function_test(TestMatmulLite, "test_f32", test_f32, devices=devices)
add_function_test(TestMatmulLite, "test_tape", test_tape, devices=devices)
add_function_test(TestMatmulLite, "test_operator", test_operator, devices=devices)
add_function_test(TestMatmulLite, "test_large_batch_count", test_large_batch_count, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
