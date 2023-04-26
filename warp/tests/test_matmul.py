import numpy as np
import unittest

import warp as wp
from warp.tests.test_base import *

np.random.seed(0)

wp.init()
wp.config.mode = "debug"


class GemmTestbedRunner:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    def alloc(self, m, n, k, batch_count):
        low = -4.5
        high = 3.5
        if batch_count == 1:
            A = wp.array2d(
                np.ceil(np.random.uniform(low=low, high=high, size=(m, k))), dtype=self.dtype, device=self.device
            )
            B = wp.array2d(
                np.ceil(np.random.uniform(low=low, high=high, size=(k, n))), dtype=self.dtype, device=self.device
            )
            C = wp.array2d(
                np.ceil(np.random.uniform(low=low, high=high, size=(m, n))), dtype=self.dtype, device=self.device
            )
            D = wp.array2d(np.zeros((m, n)), dtype=self.dtype, device=self.device)
            adj_A = wp.array2d(np.zeros((m, k)), dtype=self.dtype, device=self.device)
            adj_B = wp.array2d(np.zeros((k, n)), dtype=self.dtype, device=self.device)
            adj_C = wp.array2d(np.zeros((m, n)), dtype=self.dtype, device=self.device)
            adj_D = wp.array2d(np.ones((m, n)), dtype=self.dtype, device=self.device)
        else:
            A = wp.array2d(
                np.ceil(np.random.uniform(low=low, high=high, size=(batch_count, m, k))),
                dtype=self.dtype,
                device=self.device,
            )
            B = wp.array2d(
                np.ceil(np.random.uniform(low=low, high=high, size=(batch_count, k, n))),
                dtype=self.dtype,
                device=self.device,
            )
            C = wp.array2d(
                np.ceil(np.random.uniform(low=low, high=high, size=(batch_count, m, n))),
                dtype=self.dtype,
                device=self.device,
            )
            D = wp.array2d(np.zeros((batch_count, m, n)), dtype=self.dtype, device=self.device)
            adj_A = wp.array2d(np.zeros((batch_count, m, k)), dtype=self.dtype, device=self.device)
            adj_B = wp.array2d(np.zeros((batch_count, k, n)), dtype=self.dtype, device=self.device)
            adj_C = wp.array2d(np.zeros((batch_count, m, n)), dtype=self.dtype, device=self.device)
            adj_D = wp.array2d(np.ones((batch_count, m, n)), dtype=self.dtype, device=self.device)
        return A, B, C, D, adj_A, adj_B, adj_C, adj_D

    def run_and_verify(self, m, n, k, batch_count, alpha, beta):
        A, B, C, D, adj_A, adj_B, adj_C, adj_D = self.alloc(m, n, k, batch_count)
        if batch_count == 1:
            wp.matmul(A, B, C, D, alpha, beta, False, self.device)
            D_np = alpha * (A.numpy() @ B.numpy()) + beta * C.numpy()
            assert np.array_equal(D_np, D.numpy())

            wp.adj_matmul(A, B, C, adj_A, adj_B, adj_C, adj_D, alpha, beta, False, self.device)
            adj_A_np = alpha * np.matmul(adj_D.numpy(), B.numpy().transpose())
            adj_B_np = alpha * (A.numpy().transpose() @ adj_D.numpy())
            adj_C_np = beta * adj_D.numpy()

            assert np.array_equal(adj_A_np, adj_A.numpy())
            assert np.array_equal(adj_B_np, adj_B.numpy())
            assert np.array_equal(adj_C_np, adj_C.numpy())
        else:
            wp.batched_matmul(A, B, C, D, alpha, beta, False, self.device)
            D_np = alpha * np.matmul(A.numpy(), B.numpy()) + beta * C.numpy()
            assert np.array_equal(D_np, D.numpy())

            wp.adj_batched_matmul(A, B, C, adj_A, adj_B, adj_C, adj_D, alpha, beta, False, self.device)
            adj_A_np = alpha * np.matmul(adj_D.numpy(), B.numpy().transpose((0, 2, 1)))
            adj_B_np = alpha * np.matmul(A.numpy().transpose((0, 2, 1)), adj_D.numpy())
            adj_C_np = beta * adj_D.numpy()
            assert np.array_equal(adj_A_np, adj_A.numpy())
            assert np.array_equal(adj_B_np, adj_B.numpy())
            assert np.array_equal(adj_C_np, adj_C.numpy())

    def run(self):
        Ms = [64, 128, 512]
        Ns = [64, 128, 512]
        Ks = [64, 128, 512]
        batch_counts = [1, 4]
        betas = [0.0, 1.0]
        alpha = 1.0

        for batch_count in batch_counts:
            for m in Ms:
                for n in Ns:
                    for k in Ks:
                        for beta in betas:
                            self.run_and_verify(m, n, k, batch_count, alpha, beta)


# NOTE: F16 tests are slow due to the performance of the reference numpy F16 matmuls performed on CPU.
def test_f16(test, device):
    GemmTestbedRunner(wp.float16, device).run()


def test_f32(test, device):
    GemmTestbedRunner(wp.float32, device).run()


def test_f64(test, device):
    GemmTestbedRunner(wp.float64, device).run()


@wp.kernel
def matrix_sum_kernel(arr: wp.array2d(dtype=float), loss: wp.array(dtype=float)):
    i, j = wp.tid()
    wp.atomic_add(loss, 0, arr[i, j])


def test_tape(test, device):
    low = -4.5
    high = 3.5
    m = 64
    n = 128
    k = 256
    A = wp.array2d(
        np.ceil(np.random.uniform(low=low, high=high, size=(m, k))), dtype=float, device=device, requires_grad=True
    )
    B = wp.array2d(
        np.ceil(np.random.uniform(low=low, high=high, size=(k, n))), dtype=float, device=device, requires_grad=True
    )
    C = wp.array2d(
        np.ceil(np.random.uniform(low=low, high=high, size=(m, n))), dtype=float, device=device, requires_grad=True
    )
    D = wp.array2d(np.zeros((m, n)), dtype=float, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    # test tape
    tape = wp.Tape()
    with tape:
        wp.matmul(A, B, C, D, device=device)
        wp.launch(matrix_sum_kernel, dim=(m, n), inputs=[D, loss], device=device)

    tape.backward(loss=loss)
    A_grad = A.grad.numpy()

    # test adjoint
    D.grad = wp.array2d(np.ones((m, n)), dtype=float, device=device)
    wp.adj_matmul(A, B, C, A.grad, B.grad, C.grad, D.grad)
    assert_np_equal(A_grad, A.grad.numpy())

    # test zero
    tape.zero()
    assert_array_equal(A.grad, wp.zeros_like(A))


def register(parent):
    # we test two cases
    # A: arrays are stored on host, multiplied on device
    # B: arrays are stored on device, multipled on device
    devices = [d for d in get_test_devices()]

    class TestMatmul(parent):
        pass

    if devices:
        # check if CUTLASS is available
        from warp.context import runtime

        if runtime.core.is_cutlass_enabled():
            # add_function_test(TestMatmul, "test_f16", test_f16, devices=devices)
            add_function_test(TestMatmul, "test_f32", test_f32, devices=devices)
            add_function_test(TestMatmul, "test_f64", test_f64, devices=devices)
        else:
            print(f"Skipping matmul tests because CUTLASS is not supported in this build")

    return TestMatmul


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)
