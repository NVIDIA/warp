# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def eval_dense_gemm(
    m: int,
    n: int,
    p: int,
    t1: int,
    t2: int,
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    C: wp.array(dtype=float),
):
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
def eval_dense_solve(
    n: int, A: wp.array(dtype=float), L: wp.array(dtype=float), b: wp.array(dtype=float), x: wp.array(dtype=float)
):
    wp.dense_solve(n, A, L, b, x)


def test_dense_compilation(test, device):
    # just testing compilation of the dense matrix routines
    # most are deprecated / WIP
    wp.load_module(device=device)


devices = get_test_devices()


class TestDense(unittest.TestCase):
    pass


add_function_test(TestDense, "test_dense_compilation", test_dense_compilation, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
