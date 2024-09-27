import numpy as np
import warp as wp

wp.clear_kernel_cache()

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_DIM = 64


@wp.kernel
def tile_grouped_gemm(A: wp.array3d(dtype=float), B: wp.array3d(dtype=float), C: wp.array3d(dtype=float)):
    # output tile index
    i = wp.tid()

    a = wp.tile_load(A[i], 0, 0, m=TILE_M, n=TILE_K)
    b = wp.tile_load(B[i], 0, 0, m=TILE_K, n=TILE_N)

    print(a)
    print(b)

    # sum = wp.tile_zeros(m=TILE_M, n=TILE_N, dtype=wp.float32)

    # wp.tile_matmul(a, b, sum)

    # print(sum)

    # wp.tile_store(C[i], 0, 0, sum)


batch_count = 1

M = TILE_M
N = TILE_N
K = TILE_K

device = "cuda:0"

rng = np.random.default_rng(42)
A = rng.random((batch_count, M, K), dtype=np.float32)
B = rng.random((batch_count, K, N), dtype=np.float32)
C = A @ B

A_wp = wp.array(A, requires_grad=True, device=device)
B_wp = wp.array(B, requires_grad=True, device=device)
C_wp = wp.zeros((batch_count, TILE_M, TILE_N), requires_grad=True, device=device)

with wp.Tape() as tape:
    wp.launch(tile_grouped_gemm, 
                dim=[batch_count, TILE_DIM], 
                inputs=[A_wp, B_wp, C_wp], 
                block_dim=TILE_DIM, 
                device=device)

wp.synchronize()

# TODO: 32 mismatched elements
#assert_np_equal(C_wp.numpy(), C)
#print(C_wp.numpy())

