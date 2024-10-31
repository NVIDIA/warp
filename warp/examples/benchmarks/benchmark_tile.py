# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch

import warp as wp

wp.init()
wp.set_module_options({"enable_backward": False, "fast_math": True})
wp.set_device("cuda:0")

wp.build.clear_kernel_cache()


@wp.kernel
def gemm(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    # output index
    i, j = wp.tid()

    sum = float(0.0)

    for k in range(0, A.shape[1]):
        sum += A[i, k] * B[k, j]

    C[i, j] = sum


TILE_M = wp.constant(64)
TILE_N = wp.constant(64)
TILE_K = wp.constant(8)


@wp.kernel
def gemm_tiled(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    # output tile index
    i, j = wp.tid()

    sum = wp.tile_zeros(m=TILE_M, n=TILE_N, dtype=wp.float32)

    _M = A.shape[0]
    _N = B.shape[1]
    K = A.shape[1]

    count = int(K / 8)  # TODO: code-gen bug if you use a constant before passing it to a kwd arg (in this case TILE_K)

    for k in range(count):
        a = wp.tile_load(A, i, k, m=TILE_M, n=TILE_K)
        b = wp.tile_load(B, k, j, m=TILE_K, n=TILE_N)

        # sum += a*b
        wp.tile_matmul(a, b, sum)

    wp.tile_store(C, i, j, sum)


def benchmark_numpy(A, B, C):
    timers = {}
    iters = 10

    # warm up
    for _i in range(10):
        _C = A @ B

    with wp.ScopedTimer("NumPy", dict=timers):
        for _i in range(iters):
            _C = A @ B

    return min(timers["NumPy"])


def benchmark_warp_simt(A, B, C):
    timers = {}
    iters = 10

    A_wp = wp.array(A)
    B_wp = wp.array(B)
    C_wp = wp.array(C)

    # warm up
    for _i in range(10):
        wp.launch(gemm, dim=(M, N), inputs=[A_wp, B_wp, C_wp])

    with wp.ScopedTimer("Warp (SIMT)", dict=timers, print=False, synchronize=True):
        for _i in range(iters):
            wp.launch(gemm, dim=(M, N), inputs=[A_wp, B_wp, C_wp])

    return min(timers["Warp (SIMT)"])


def benchmark_warp_tiled(A, B, C):
    timers = {}
    iters = 10

    # must match with the tile_matmul() partition size
    SUB_TILE_M = 4
    SUB_TILE_N = 4

    num_threads = int(TILE_M / SUB_TILE_M) * int(TILE_N / SUB_TILE_N)
    A_wp = wp.array(A)
    B_wp = wp.array(B)
    C_wp = wp.array(C)

    # warm up
    wp.capture_begin()

    for _i in range(iters):
        wp.launch(gemm_tiled, dim=(int(M / TILE_M), int(N / TILE_N)), inputs=[A_wp, B_wp, C_wp], tile_size=num_threads)

    graph = wp.capture_end()

    with wp.ScopedTimer("Warp (Tiled)", dict=timers, print=False, synchronize=True):
        # for i in range(iters):
        #    wp.launch(gemm_tiled, dim=(int(M/TILE_M), int(N/TILE_N)), inputs=[A_wp, B_wp, C_wp], tile_size=num_threads)
        wp.capture_launch(graph)

    return min(timers["Warp (Tiled)"])


def benchmark_torch(A, B, C):
    A_tc = torch.from_numpy(A).to("cuda:0")
    B_tc = torch.from_numpy(B).to("cuda:0")
    C_tc = torch.from_numpy(C).to("cuda:0")

    # warm-up
    for _i in range(10):
        torch.matmul(A_tc, B_tc, out=C_tc)

    timers = {}
    iters = 10

    torch.cuda.synchronize()

    with wp.ScopedTimer("Torch", dict=timers, print=False):
        for _i in range(iters):
            torch.matmul(A_tc, B_tc)  # , out=C_tc)

        torch.cuda.synchronize()

    return min(timers["Torch"])


results_torch = []
results_warp_simt = []
results_warp_tiled = []

print("{:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format("M", "N", "K", "Torch", "Warp (SIMT)", "Warp (Tiled)"))
print("--------------------------------------------------------")

for i in range(2, 33):
    # for i in range(8,9):

    M = i * 128
    N = M
    K = N

    # M = TILE_M*21
    # K = TILE_K*7
    # N = TILE_M*12

    rng = np.random.default_rng(42)

    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    results_torch.append(benchmark_torch(A, B, C))
    results_warp_simt.append(0.0)  # benchmark_warp_simt(A, B, C))
    results_warp_tiled.append(benchmark_warp_tiled(A, B, C))

    print(
        "{:>8d} {:>8d} {:>8d} {:>8f} {:>8f} {:>8f}".format(
            M, N, K, results_torch[-1], results_warp_simt[-1], results_warp_tiled[-1]
        )
    )
