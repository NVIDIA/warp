import numpy as np
import unittest

import warp as wp
from warp.tests.test_base import *

np.random.seed(0)

wp.init()


class GemmTestbedRunner:
    def __init__(self, dtype):
        self.dtype = dtype

    def alloc(self, m, n, k, batch_count):
        low=-4.5
        high=3.5
        if batch_count == 1:
            A = wp.array2d(np.ceil(np.random.uniform(low=low, high=high, size=(m, k))), dtype=self.dtype)
            B = wp.array2d(np.ceil(np.random.uniform(low=low, high=high, size=(k, n))), dtype=self.dtype)
            C = wp.array2d(np.ceil(np.random.uniform(low=low, high=high, size=(m, n))), dtype=self.dtype)
            D = wp.array2d(np.zeros((m, n)), dtype=self.dtype)
        else:
            A = wp.array2d(np.ceil(np.random.uniform(low=low, high=high, size=(batch_count, m, k))), dtype=self.dtype)
            B = wp.array2d(np.ceil(np.random.uniform(low=low, high=high, size=(batch_count, k, n))), dtype=self.dtype)
            C = wp.array2d(np.ceil(np.random.uniform(low=low, high=high, size=(batch_count, m, n))), dtype=self.dtype)
            D = wp.array2d(np.zeros((batch_count, m, n)), dtype=self.dtype)
        return A, B, C, D

    def run_and_verify(self, m, n, k, batch_count, alpha, beta):
        A, B, C, D = self.alloc(m, n, k, batch_count)
        if batch_count == 1:
            wp.matmul(A, B, C, D, alpha, beta)
            D_np = alpha * (A.numpy() @ B.numpy()) + beta * C.numpy()
            assert np.array_equal(D_np, D.numpy())
        else:
            wp.batched_matmul(A, B, C, D, alpha, beta)
            D_np = alpha * np.matmul(A.numpy(), B.numpy()) + beta * C.numpy()
            assert np.array_equal(D_np, D.numpy())

    def run(self):
        Ms = [128, 256, 512, 1024]
        Ns = [128, 256, 512, 1024]
        Ks = [128, 256, 512, 1024]
        batch_counts = [1, 2, 4]
        betas = [0., 1.]
        alpha = 1.

        for batch_count in batch_counts:
            for m in Ms:
                for n in Ns:
                    for k in Ks:
                        for beta in betas:
                            self.run_and_verify(m, n, k, batch_count, alpha, beta)


# NOTE: F16 tests are slow due to the performance of the reference numpy F16 matmuls performed on CPU.
def test_f16(test, device):
    GemmTestbedRunner(wp.float16).run()


def test_f32(test, device):
    GemmTestbedRunner(wp.float32).run()


def test_f64(test, device):
    GemmTestbedRunner(wp.float64).run()


def register(parent):

    # Implementation currently only available on GPU
    devices = [d for d in wp.get_devices() if 'cuda' in d.alias]

    class TestMatmul(parent):
        pass

    add_function_test(TestMatmul, "test_f16", test_f16, devices=devices)
    add_function_test(TestMatmul, "test_f32", test_f32, devices=devices)
    add_function_test(TestMatmul, "test_f64", test_f64, devices=devices)

    return TestMatmul


if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)
