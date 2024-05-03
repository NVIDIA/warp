import numpy as np
import warp as wp

wp.init()
wp.set_module_options({"enable_backwards": False})
wp.set_device("cuda:0")

@wp.kernel
def gemm(A: wp.array2d(dtype=float),
         B: wp.array2d(dtype=float),
         C: wp.array2d(dtype=float)):

    # output index
    i, j = wp.tid()

    sum = float(0.0)

    for k in range(0, A.shape[1]):
        sum += A[i, k]*B[k, j]

    C[i, j] = sum

TILE_M = wp.constant(16)
TILE_N = wp.constant(16)
TILE_K = wp.constant(8)

@wp.kernel
def gemm_tiled(A: wp.array2d(dtype=float),
               B: wp.array2d(dtype=float),
               C: wp.array2d(dtype=float)):

    # output tile index
    i, j = wp.tid()

    sum = wp.tile_zeros(m=TILE_M, n=TILE_N, dtype=wp.float32)

    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

    for k in range(0, K, TILE_K):

        a = wp.tile_load(A, i, j+k, m=TILE_M, n=TILE_K)
        b = wp.tile_load(B, i+k, j, m=TILE_K, n=TILE_N)

        # sum += a*b
        wp.tile_matmul(a, b, sum)

    wp.tile_store(C, i, j, sum)


M = 240
K = 80
N = 350

rng = np.random.default_rng(42)
A = rng.random((M, K), dtype=np.float32)
B = rng.random((K, N), dtype=np.float32)
C = np.zeros((M, N), dtype=np.float32)

A_wp = wp.array(A)
B_wp = wp.array(B)
C_wp = wp.array(C)

iters = 100

with wp.ScopedTimer("NumPy"):

    for i in range(iters):
        C = A@B

#wp.force_load()

with wp.ScopedTimer("Warp", cuda_flags=wp.TIMING_KERNEL):

    for i in range(iters):
        wp.launch(gemm, dim=(M, N), inputs=[A_wp, B_wp, C_wp])
    

print(np.allclose(C, C_wp.numpy(), rtol=1.e-4))


