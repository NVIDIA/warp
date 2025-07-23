# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_DIM = 64


@wp.kernel
def tile_grouped_gemm(A: wp.array3d(dtype=float), B: wp.array3d(dtype=float), C: wp.array3d(dtype=float)):
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

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_grouped_gemm, dim=[batch_count], inputs=[A_wp, B_wp, C_wp], block_dim=TILE_DIM, device=device
        )

    # TODO: 32 mismatched elements
    assert_np_equal(C_wp.numpy(), C, 1e-6)


@wp.kernel
def tile_gemm(A: wp.array2d(dtype=Any), B: wp.array2d(dtype=Any), C: wp.array2d(dtype=Any)):
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


wp.overload(
    tile_gemm, {"A": wp.array2d(dtype=wp.float16), "B": wp.array2d(dtype=wp.float16), "C": wp.array2d(dtype=wp.float16)}
)
wp.overload(
    tile_gemm, {"A": wp.array2d(dtype=wp.float32), "B": wp.array2d(dtype=wp.float32), "C": wp.array2d(dtype=wp.float32)}
)
wp.overload(
    tile_gemm, {"A": wp.array2d(dtype=wp.float64), "B": wp.array2d(dtype=wp.float64), "C": wp.array2d(dtype=wp.float64)}
)


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


@wp.kernel
def test_tile_transpose_matmul_kernel(input: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):
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

    assert_np_equal(output.numpy(), input.numpy().T @ input.numpy())


class TestTileMatmul(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestTileMatmul, "test_tile_gemm_fp16", test_tile_gemm(wp.float16), devices=devices)
add_function_test(TestTileMatmul, "test_tile_gemm_fp32", test_tile_gemm(wp.float32), devices=devices)
add_function_test(TestTileMatmul, "test_tile_gemm_fp64", test_tile_gemm(wp.float64), devices=devices)
add_function_test(TestTileMatmul, "test_tile_grouped_gemm", test_tile_grouped_gemm, devices=devices)
add_function_test(TestTileMatmul, "test_tile_transpose_matmul", test_tile_transpose_matmul, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
