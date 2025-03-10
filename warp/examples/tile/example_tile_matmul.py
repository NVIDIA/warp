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

###########################################################################
# Example Tile MatMul
#
# Shows how to write a simple GEMM kernel using Warp tile primitives.
#
###########################################################################

import numpy as np

import warp as wp

# tile size
TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_THREADS = 64


@wp.kernel
def tile_gemm(A: wp.array2d(dtype=wp.float32), B: wp.array2d(dtype=wp.float16), C: wp.array2d(dtype=wp.float64)):
    # output tile index
    i, j = wp.tid()

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float64)

    _M = A.shape[0]
    _N = B.shape[1]
    K = A.shape[1]

    count = int(K / TILE_K)

    for k in range(0, count):
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))

        # sum += a*b
        wp.tile_matmul(a, b, sum)

    wp.tile_store(C, sum, offset=(i * TILE_M, j * TILE_N))


if __name__ == "__main__":
    # generate some tile aligned matrix dimensions
    M = TILE_M * 7
    K = TILE_K * 6
    N = TILE_N * 5

    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float64)

    A_wp = wp.array(A, requires_grad=True)
    B_wp = wp.array(B, requires_grad=True)
    C_wp = wp.array(C, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_gemm,
            dim=(M // TILE_M, N // TILE_N),
            inputs=[A_wp, B_wp],
            outputs=[C_wp],
            block_dim=TILE_THREADS,
        )

    assert np.allclose(C_wp.numpy(), A @ B, atol=1.0e-4)

    print("Example matrix multiplication passed")
