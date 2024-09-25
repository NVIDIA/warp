import numpy as np
import warp as wp

wp.init()
wp.build.clear_kernel_cache()

BLOCK_DIM = 32
M, N, K = 4, 8, 16

@wp.kernel
def matmul_tiled(ga: wp.array2d(dtype=wp.float64),
                 gb: wp.array2d(dtype=wp.float64),
                 gc: wp.array2d(dtype=wp.float64)):
    
    i, j, _ = wp.tid()
    a = wp.tile_load(ga, i, j, m=M, n=K)
    b = wp.tile_load(gb, i, j, m=K, n=N)
    c = wp.tile_zeros(m=M, n=N, dtype=wp.float64)
    wp.tile_matmul_dx(a, b, c)
    wp.tile_store(gc, i, j, c)


A = np.ones((M, K), dtype=np.float64)
B = 3 * np.ones((K, N), dtype=np.float64)
C = np.zeros((M, N), dtype=np.float64)

A_wp = wp.array2d(A, dtype=wp.float64)
B_wp = wp.array2d(B, dtype=wp.float64)
C_wp = wp.array2d(C, dtype=wp.float64)

wp.launch(matmul_tiled, dim=[1, 1, BLOCK_DIM], inputs=[A_wp, B_wp, C_wp], block_dim=BLOCK_DIM)
wp.synchronize()

print("inputs:\n", A, '\n', B)
print("output (should be = 48 * np.ones(4, 8)):\n", C_wp)
