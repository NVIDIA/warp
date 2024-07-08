# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest
from typing import Any

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

        np_dtype = wp.types.warp_type_to_np_dtype[self.dtype]

        if batch_count == 1:
            tape = wp.Tape()
            with tape:
                wp.matmul(A, B, C, D, alpha, beta, False)
            tape.backward(grads={D: ones})

            D_np = alpha * np.matmul(A.numpy(), B.numpy(), dtype=np_dtype) + beta * C.numpy()
            assert_np_equal(D.numpy(), D_np)

            adj_A_np = alpha * np.matmul(ones.numpy(), B.numpy().transpose(), dtype=np_dtype)
            adj_B_np = alpha * np.matmul(A.numpy().transpose(), ones.numpy(), dtype=np_dtype)
            adj_C_np = beta * ones.numpy()

        else:
            tape = wp.Tape()
            with tape:
                wp.batched_matmul(A, B, C, D, alpha, beta, False)
            tape.backward(grads={D: ones})

            D_np = alpha * np.matmul(A.numpy(), B.numpy(), dtype=np_dtype) + beta * C.numpy()
            assert_np_equal(D.numpy(), D_np)

            adj_A_np = alpha * np.matmul(ones.numpy(), B.numpy().transpose((0, 2, 1)), dtype=np_dtype)
            adj_B_np = alpha * np.matmul(A.numpy().transpose((0, 2, 1)), ones.numpy(), dtype=np_dtype)
            adj_C_np = beta * ones.numpy()

        assert_np_equal(A.grad.numpy(), adj_A_np)
        assert_np_equal(B.grad.numpy(), adj_B_np)
        assert_np_equal(C.grad.numpy(), adj_C_np)

    def run(self):
        Ms = [64, 128, 256]
        Ns = [64, 128, 256]
        Ks = [64, 128, 256]
        batch_counts = [1, 4]
        betas = [0.0, 1.0]
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

        np_dtype = wp.types.warp_type_to_np_dtype[self.dtype]

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

            D_np = alpha * np.matmul(A.numpy(), B.numpy(), dtype=np_dtype) + beta * C1.numpy()
            assert_np_equal(D1.numpy(), D_np)
            assert_np_equal(D2.numpy(), D_np)
            assert_np_equal(D3.numpy(), D_np)

            adj_A_np = alpha * np.matmul(ones1.numpy(), B.numpy().transpose(), dtype=np_dtype)
            adj_B_np = alpha * np.matmul(A.numpy().transpose(), ones1.numpy(), dtype=np_dtype)
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

            D_np = alpha * np.matmul(A.numpy(), B.numpy(), dtype=np_dtype) + beta * C1.numpy()
            assert_np_equal(D1.numpy(), D_np)
            assert_np_equal(D2.numpy(), D_np)
            assert_np_equal(D3.numpy(), D_np)

            adj_A_np = alpha * np.matmul(ones1.numpy(), B.numpy().transpose((0, 2, 1)), dtype=np_dtype)
            adj_B_np = alpha * np.matmul(A.numpy().transpose((0, 2, 1)), ones1.numpy(), dtype=np_dtype)
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
        m = 16
        n = 32
        k = 64
        batch_counts = [1, 4]
        beta = 1.0
        alpha = 1.0

        for batch_count in batch_counts:
            self.run_and_verify(m, n, k, batch_count, alpha, beta)


# NOTE: F16 tests are slow due to the performance of the reference numpy F16 matmuls performed on CPU.
def test_f16(test, device):
    gemm_test_bed_runner(wp.float16, device).run()
    gemm_test_bed_runner_transpose(wp.float16, device).run()


@unittest.skipUnless(wp.context.runtime.core.is_cutlass_enabled(), "Warp was not built with CUTLASS support")
def test_f32(test, device):
    gemm_test_bed_runner(wp.float32, device).run()
    gemm_test_bed_runner_transpose(wp.float32, device).run()


@unittest.skipUnless(wp.context.runtime.core.is_cutlass_enabled(), "Warp was not built with CUTLASS support")
def test_f64(test, device):
    gemm_test_bed_runner(wp.float64, device).run()
    gemm_test_bed_runner_transpose(wp.float64, device).run()


@wp.kernel
def matrix_sum_kernel(arr: wp.array2d(dtype=float), loss: wp.array(dtype=float)):
    i, j = wp.tid()
    wp.atomic_add(loss, 0, arr[i, j])


@unittest.skipUnless(wp.context.runtime.core.is_cutlass_enabled(), "Warp was not built with CUTLASS support")
def test_tape(test, device):
    rng = np.random.default_rng(42)
    low = -4.5
    high = 3.5
    m = 64
    n = 128
    k = 256
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
    m = 64
    n = 128
    k = 256
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


@unittest.skipUnless(wp.context.runtime.core.is_cutlass_enabled(), "Warp was not built with CUTLASS support")
def test_adjoint_accumulation(test, device):
    a_np = np.ones(shape=(2, 3))
    b_np = np.ones(shape=(3, 2))
    c_np = np.zeros(shape=(2, 2))
    d_np = np.zeros(shape=(2, 2))

    a_wp = wp.from_numpy(a_np, dtype=float, requires_grad=True, device=device)
    b_wp = wp.from_numpy(b_np, dtype=float, requires_grad=True, device=device)
    c_wp = wp.from_numpy(c_np, dtype=float, requires_grad=True, device=device)
    d1_wp = wp.from_numpy(d_np, dtype=float, requires_grad=True, device=device)
    d2_wp = wp.from_numpy(d_np, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()

    with tape:
        wp.matmul(a_wp, b_wp, c_wp, d1_wp, alpha=1.0, beta=1.0)
        wp.matmul(a_wp, b_wp, d1_wp, d2_wp, alpha=1.0, beta=1.0)

    d_grad = wp.zeros_like(d2_wp, device=device)
    d_grad.fill_(1.0)
    grads = {d2_wp: d_grad}
    tape.backward(grads=grads)

    assert_np_equal(a_wp.grad.numpy(), 4.0 * np.ones(shape=(2, 3)))
    assert_np_equal(b_wp.grad.numpy(), 4.0 * np.ones(shape=(3, 2)))
    assert_np_equal(c_wp.grad.numpy(), np.ones(shape=(2, 2)))


@unittest.skipUnless(wp.context.runtime.core.is_cutlass_enabled(), "Warp was not built with CUTLASS support")
def test_cuda_graph_capture(test, device):
    @wp.kernel
    def mat_sum(mat: wp.array2d(dtype=Any), loss: wp.array(dtype=Any)):
        i, j = wp.tid()
        e = mat[i, j]
        wp.atomic_add(loss, 0, e)

    for T in [wp.float16, wp.float32, wp.float64]:
        wp.overload(mat_sum, [wp.array2d(dtype=T), wp.array(dtype=T)])

    wp.load_module(device=device)
    wp.load_module(module="warp.utils", device=device)

    for dtype in [wp.float16, wp.float32, wp.float64]:
        m = 8
        n = 8
        k = 8

        A = wp.ones((m, n), dtype=dtype, device=device, requires_grad=True)
        B = wp.ones((n, k), dtype=dtype, device=device, requires_grad=True)
        C = wp.zeros((m, k), dtype=dtype, device=device, requires_grad=True)
        D = wp.zeros((m, k), dtype=dtype, device=device, requires_grad=True)

        loss = wp.zeros(1, dtype=dtype, device=device, requires_grad=True)

        wp.capture_begin(device, force_module_load=False)
        try:
            tape = wp.Tape()

            with tape:
                wp.matmul(A, B, C, D)
                wp.launch(mat_sum, dim=(m, k), inputs=[D, loss], device=device)

            tape.backward(loss=loss)
        finally:
            graph = wp.capture_end(device)

        wp.capture_launch(graph)

        assert_np_equal(A.grad.numpy(), 8.0 * np.ones((m, n), dtype=wp.types.warp_type_to_np_dtype[dtype]))


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()


class TestMatmul(unittest.TestCase):
    pass


# add_function_test(TestMatmul, "test_f16", test_f16, devices=devices)
add_function_test(TestMatmul, "test_f32", test_f32, devices=devices)
add_function_test(TestMatmul, "test_f64", test_f64, devices=devices)
add_function_test(TestMatmul, "test_tape", test_tape, devices=devices)
add_function_test(TestMatmul, "test_operator", test_operator, devices=devices)
add_function_test(TestMatmul, "test_large_batch_count", test_large_batch_count, devices=devices)
add_function_test(TestMatmul, "test_adjoint_accumulation", test_adjoint_accumulation, devices=devices)
add_function_test(TestMatmul, "test_cuda_graph_capture", test_cuda_graph_capture, devices=cuda_devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
