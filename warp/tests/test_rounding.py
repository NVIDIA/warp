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

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

compare_to_numpy = False
print_results = False


@wp.kernel
def test_kernel(
    x: wp.array(dtype=float),
    x_round: wp.array(dtype=float),
    x_rint: wp.array(dtype=float),
    x_trunc: wp.array(dtype=float),
    x_cast: wp.array(dtype=float),
    x_floor: wp.array(dtype=float),
    x_ceil: wp.array(dtype=float),
    x_frac: wp.array(dtype=float),
):
    tid = wp.tid()

    x_round[tid] = wp.round(x[tid])
    x_rint[tid] = wp.rint(x[tid])
    x_trunc[tid] = wp.trunc(x[tid])
    x_cast[tid] = float(int(x[tid]))
    x_floor[tid] = wp.floor(x[tid])
    x_ceil[tid] = wp.ceil(x[tid])
    x_frac[tid] = wp.frac(x[tid])


def test_rounding(test, device):
    # fmt: off
    nx = np.array([
        4.9,  4.5,  4.1,  3.9,  3.5,  3.1,  2.9,  2.5,  2.1,  1.9,
        1.5,  1.1,  0.9,  0.5,  0.1, -0.1, -0.5, -0.9, -1.1, -1.5,
       -1.9, -2.1, -2.5, -2.9, -3.1, -3.5, -3.9, -4.1, -4.5, -4.9
    ], dtype=np.float32)
    # fmt: on

    x = wp.array(nx, device=device)
    N = len(x)

    x_round = wp.empty(N, dtype=float, device=device)
    x_rint = wp.empty(N, dtype=float, device=device)
    x_trunc = wp.empty(N, dtype=float, device=device)
    x_cast = wp.empty(N, dtype=float, device=device)
    x_floor = wp.empty(N, dtype=float, device=device)
    x_ceil = wp.empty(N, dtype=float, device=device)
    x_frac = wp.empty(N, dtype=float, device=device)

    wp.launch(
        kernel=test_kernel, dim=N, inputs=[x, x_round, x_rint, x_trunc, x_cast, x_floor, x_ceil, x_frac], device=device
    )

    wp.synchronize()

    nx_round = x_round.numpy().reshape(N)
    nx_rint = x_rint.numpy().reshape(N)
    nx_trunc = x_trunc.numpy().reshape(N)
    nx_cast = x_cast.numpy().reshape(N)
    nx_floor = x_floor.numpy().reshape(N)
    nx_ceil = x_ceil.numpy().reshape(N)
    nx_frac = x_frac.numpy().reshape(N)

    tab = np.stack([nx, nx_round, nx_rint, nx_trunc, nx_cast, nx_floor, nx_ceil, nx_frac], axis=1)

    golden = np.array(
        [
            [4.9, 5.0, 5.0, 4.0, 4.0, 4.0, 5.0, 0.9],
            [4.5, 5.0, 4.0, 4.0, 4.0, 4.0, 5.0, 0.5],
            [4.1, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 0.1],
            [3.9, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 0.9],
            [3.5, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 0.5],
            [3.1, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 0.1],
            [2.9, 3.0, 3.0, 2.0, 2.0, 2.0, 3.0, 0.9],
            [2.5, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.5],
            [2.1, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.1],
            [1.9, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 0.9],
            [1.5, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 0.5],
            [1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.1],
            [0.9, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.9],
            [0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1],
            [-0.1, -0.0, -0.0, -0.0, 0.0, -1.0, -0.0, -0.1],
            [-0.5, -1.0, -0.0, -0.0, 0.0, -1.0, -0.0, -0.5],
            [-0.9, -1.0, -1.0, -0.0, 0.0, -1.0, -0.0, -0.9],
            [-1.1, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, -0.1],
            [-1.5, -2.0, -2.0, -1.0, -1.0, -2.0, -1.0, -0.5],
            [-1.9, -2.0, -2.0, -1.0, -1.0, -2.0, -1.0, -0.9],
            [-2.1, -2.0, -2.0, -2.0, -2.0, -3.0, -2.0, -0.1],
            [-2.5, -3.0, -2.0, -2.0, -2.0, -3.0, -2.0, -0.5],
            [-2.9, -3.0, -3.0, -2.0, -2.0, -3.0, -2.0, -0.9],
            [-3.1, -3.0, -3.0, -3.0, -3.0, -4.0, -3.0, -0.1],
            [-3.5, -4.0, -4.0, -3.0, -3.0, -4.0, -3.0, -0.5],
            [-3.9, -4.0, -4.0, -3.0, -3.0, -4.0, -3.0, -0.9],
            [-4.1, -4.0, -4.0, -4.0, -4.0, -5.0, -4.0, -0.1],
            [-4.5, -5.0, -4.0, -4.0, -4.0, -5.0, -4.0, -0.5],
            [-4.9, -5.0, -5.0, -4.0, -4.0, -5.0, -4.0, -0.9],
        ],
        dtype=np.float32,
    )

    assert_np_equal(tab, golden, tol=1e-6)

    if print_results:
        np.set_printoptions(formatter={"float": lambda x: f"{x:6.1f}".replace(".0", ".")})

        print("----------------------------------------------")
        print(f"   {'x ':>5s} {'round':>5s} {'rint':>5s} {'trunc':>5s} {'cast':>5s} {'floor':>5s} {'ceil':>5s}")
        print(tab)
        print("----------------------------------------------")

    if compare_to_numpy:
        nx_round = np.round(nx)
        nx_rint = np.rint(nx)
        nx_trunc = np.trunc(nx)
        nx_fix = np.fix(nx)
        nx_floor = np.floor(nx)
        nx_ceil = np.ceil(nx)
        nx_frac = np.modf(nx)[0]

        tab = np.stack([nx, nx_round, nx_rint, nx_trunc, nx_fix, nx_floor, nx_ceil, nx_frac], axis=1)
        print(f"   {'x ':>5s} {'round':>5s} {'rint':>5s} {'trunc':>5s} {'fix':>5s} {'floor':>5s} {'ceil':>5s}")
        print(tab)
        print("----------------------------------------------")


class TestRounding(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestRounding, "test_rounding", test_rounding, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
