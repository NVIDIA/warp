from itertools import product

import numpy as np
import torch as tc

import warp as wp

tc.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for matrix multiplications
tc.backends.cudnn.allow_tf32 = False  # Disable TF32 for cuDNN operations

wp.init()
wp.clear_kernel_cache()
wp.set_module_options({"fast_math": True, "enable_backward": False})


# returns a kernel to compute a GEMM given m,n,k tile sizes
def create_gemm_kernel(m, n, k):
    TILE_M = m
    TILE_N = n
    TILE_K = k

    @wp.kernel
    def gemm(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):
        i, j = wp.tid()
        sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

        count = A.shape[1] // TILE_K

        for k in range(count):
            a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
            b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))

            wp.tile_matmul(a, b, sum)

        wp.tile_store(output, sum, offset=(i * TILE_M, j * TILE_N))

    return gemm


def benchmark_torch(A, B, warm_up, iterations):
    # warm-up
    for _ in range(warm_up):
        tc.matmul(A, B)

    timers = {}
    tc.cuda.synchronize()

    with wp.ScopedTimer("torch", print=False, dict=timers, synchronize=True):
        for _ in range(iterations):
            tc.matmul(A, B)

        tc.cuda.synchronize()

    return timers["torch"][0]


def benchmark_warp(A, B, config, warm_up, iterations):
    TILE_M = config[0]
    TILE_N = config[1]
    TILE_K = config[2]
    BLOCK_DIM = config[3]

    mlp = create_gemm_kernel(TILE_M, TILE_N, TILE_K)

    M = A.shape[0]
    N = B.shape[1]

    output = wp.zeros((M, N), dtype=float)

    # create launch command
    cmd = wp.launch_tiled(
        kernel=mlp,
        dim=[M // TILE_M, N // TILE_N],
        inputs=[A, B, output],
        block_dim=BLOCK_DIM,
        record_cmd=True,
    )

    # warm-up
    for _ in range(warm_up):
        cmd.launch()

    # check output
    if warm_up > 0:
        assert np.allclose(output.numpy(), A.numpy() @ B.numpy(), atol=1e-3, rtol=1e-3)

    # benchmark
    timers = {}
    with wp.ScopedTimer("warp", print=False, dict=timers, synchronize=True):
        for _ in range(iterations):
            cmd.launch()

    return timers["warp"][0]


tile_m = [8, 16, 32, 64]
tile_n = [8, 16, 32, 64]
tile_k = [8, 16, 64]
block = [32, 64, 128]

M = 1024
N = 1024
K = 1024

A = tc.randn(M, K).cuda()
B = tc.randn(K, N).cuda()

iterations = 1000
warm_up = 10

time_torch = benchmark_torch(A, B, warm_up, iterations)
print(f"Torch: {time_torch}")

configs = list(product(tile_m, tile_n, tile_k, block))

wp.config.quiet = True

# header
print(
    "{:<{}} {:<{}} {:<{}} {:<{}} {:<{}} {:<{}}".format(
        "TILE_M", 12, "TILE_N", 12, "TILE_K", 12, "BLOCK", 12, "Time", 12, "Relative", 12
    )
)
for c in configs:
    time_warp = benchmark_warp(wp.from_torch(A), wp.from_torch(B), c, warm_up, iterations)
    print(
        "{:<{}} {:<{}} {:<{}} {:<{}} {:<{}} {:<{}}".format(
            c[0], 12, c[1], 12, c[2], 12, c[3], 12, time_warp, 12, time_warp / time_torch, 12
        )
    )
