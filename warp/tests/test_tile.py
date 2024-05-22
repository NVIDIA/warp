import numpy as np
import warp as wp

import torch

#wp.config.mode = "debug"

wp.init()
wp.set_module_options({"enable_backward": False})
wp.set_device("cuda:0")


wp.build.clear_kernel_cache()

TILE_M = 8
TILE_N = 4

@wp.kernel
def copy_tiled(A: wp.array2d(dtype=float),
               B: wp.array2d(dtype=float)):
    
    # tile index
    i, j = wp.tid() 
    
    a = wp.tile_load(A, i, j, m=TILE_M, n=TILE_N)
    wp.tile_store(B, i, j, a)


def test_copy_tiled():

    rng = np.random.default_rng(42)

    M = TILE_M*7
    N = TILE_N*5

    A = rng.random((M, N), dtype=np.float32)
    B = rng.random((M, N), dtype=np.float32)

    A_wp = wp.array(A)
    B_wp = wp.array(B)

    wp.launch(copy_tiled, dim=[int(M/TILE_M), int(N/TILE_N)], inputs=[A_wp, B_wp], tile_size=8)

    assert(np.allclose(A, B_wp.numpy(), rtol=1.e-4))
    
    print("Copy passed")


#test_copy_tiled()


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



TILE_M = wp.constant(64)
TILE_N = wp.constant(64)
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

    count = int(K / 8) # todo: must be the same as TILE_K

    for k in range(count):

        a = wp.tile_load(A, i, k, m=TILE_M, n=TILE_K)
        b = wp.tile_load(B, k, j, m=TILE_K, n=TILE_N)

        # sum += a*b
        wp.tile_matmul(a, b, sum)

    wp.tile_store(C, i, j, sum)


M = TILE_M*7
K = TILE_K*4
N = TILE_N*6

rng = np.random.default_rng(42)
A = rng.random((M, K), dtype=np.float32)
B = rng.random((K, N), dtype=np.float32)
C = np.zeros((M, N), dtype=np.float32)

A_wp = wp.array(A)
B_wp = wp.array(B)
C_wp = wp.array(C)

iters = 10

with wp.ScopedTimer("NumPy"):

    for i in range(iters):
        C = A@B

wp.force_load(device="cuda:0")

with wp.ScopedTimer("Warp", cuda_filter=wp.TIMING_KERNEL):

    for i in range(iters):
        wp.launch(gemm, dim=(M, N), inputs=[A_wp, B_wp, C_wp])


    print(np.allclose(C, C_wp.numpy(), rtol=1.e-4))

    for i in range(iters):
        wp.launch(gemm_tiled, dim=(int(M/TILE_M), int(N/TILE_N)), inputs=[A_wp, B_wp, C_wp], tile_size=128)
        wp.synchronize()


    print(np.allclose(C, C_wp.numpy(), rtol=1.e-4))


A_tc = torch.from_numpy(A).to("cuda:0")
B_tc = torch.from_numpy(B).to("cuda:0")
C_tc = torch.from_numpy(C).to("cuda:0")

for i in range(10):
    torch.matmul(A_tc, B_tc, out=C_tc)

with wp.ScopedTimer("Torch"):

    for i in range(iters):
        torch.matmul(A_tc, B_tc, out=C_tc)

    torch.cuda.synchronize()

    


