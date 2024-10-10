import numpy as np

import warp as wp

wp.init()
wp.build.clear_kernel_cache()

BLOCK_DIM = 32
M, N, K = 4, 8, 16


@wp.kernel
def matmul_tiled(ga: wp.array2d(dtype=wp.float32), gb: wp.array2d(dtype=wp.float16), gc: wp.array2d(dtype=wp.float64)):
    i, j, _ = wp.tid()
    a = wp.tile_load(ga, i, j, m=M, n=K)
    b = wp.tile_load(gb, i, j, m=K, n=N)
    c = wp.tile_zeros(m=M, n=N, dtype=wp.float64)
    wp.tile_matmul(a, b, c)
    wp.tile_store(gc, i, j, c)


A = np.ones((M, K), dtype=np.float32)
B = 3 * np.ones((K, N), dtype=np.float16)
C = np.zeros((M, N), dtype=np.float64)

A_wp = wp.array2d(A, dtype=wp.float32)
B_wp = wp.array2d(B, dtype=wp.float16)
C_wp = wp.array2d(C, dtype=wp.float64)

wp.launch(matmul_tiled, dim=[1, 1, BLOCK_DIM], inputs=[A_wp, B_wp, C_wp], block_dim=BLOCK_DIM)
wp.synchronize()

print("inputs:\n", A, "\n", B)
print("output (should be = 48 * np.ones(4, 8)):\n", C_wp)
