# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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

    sum = wp.tile_zeros(m=TILE_M, n=TILE_N, dtype=wp.float64)

    _M = A.shape[0]
    _N = B.shape[1]
    K = A.shape[1]

    count = int(K / TILE_K)

    for k in range(0, count):
        a = wp.tile_load(A, i, k, m=TILE_M, n=TILE_K)
        b = wp.tile_load(B, k, j, m=TILE_K, n=TILE_N)

        # sum += a*b
        wp.tile_matmul(a, b, sum)

    wp.tile_store(C, i, j, sum)


if __name__ == "__main__":
    wp.set_device("cuda:0")

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
            dim=(int(M / TILE_M), int(N / TILE_N)),
            inputs=[A_wp, B_wp],
            outputs=[C_wp],
            block_dim=TILE_THREADS,
        )

    assert np.allclose(C_wp.numpy(), A @ B)

    print("Example matrix multiplication passed")
