
import numpy as np
import math

import warp as wp
from warp.tests.test_base import *

import unittest

wp.init()


@wp.kernel
def eval_dense_gemm(m: int, n: int, p: int, t1: int, t2: int, A: wp.array(dtype=float), B: wp.array(dtype=float), C: wp.array(dtype=float)):
    wp.dense_gemm(m, n, p, t1, t2, A, B, C)

@wp.kernel
def eval_dense_cholesky(n: int, A: wp.array(dtype=float), regularization: float, L: wp.array(dtype=float)):
    wp.dense_chol(n, A, regularization, L)

@wp.kernel
def eval_dense_subs(n: int, L: wp.array(dtype=float), b: wp.array(dtype=float), x: wp.array(dtype=float)):
    wp.dense_subs(n, L, b, x)

# helper that propagates gradients back to A, treating L as a constant / temporary variable
# allows us to reuse the Cholesky decomposition from the forward pass
@wp.kernel
def eval_dense_solve(n: int, A: wp.array(dtype=float), L: wp.array(dtype=float), b: wp.array(dtype=float), x: wp.array(dtype=float)):
    wp.dense_solve(n, A, L, b, x)



def register(parent):

    devices = wp.get_devices()

    class TestDense(parent):
        pass

    # just testing compilation of the dense matrix routines
    # most are deprecated / WIP
    wp.force_load()

    return TestDense

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)