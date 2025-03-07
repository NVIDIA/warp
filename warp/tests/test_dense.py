# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
