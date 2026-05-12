# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare shared-memory tile GEMM with register-preload pipelining.

This benchmark exercises register-to-shared tile reassignment in pipelined
``wp.tile_matmul()`` kernels. The baseline keeps each ``tile_load``/``tile_matmul``
pair in one shared-memory pipeline. The pipelined variant preloads the next A/B
tiles into registers, computes the current tile pair, then reassigns those
register tiles into the next shared-memory ``tile_matmul``.

Usage::

    python warp/examples/benchmarks/benchmark_tile_matmul_pipelined.py
    python warp/examples/benchmarks/benchmark_tile_matmul_pipelined.py --size 256 --iterations 5
    python warp/examples/benchmarks/benchmark_tile_matmul_pipelined.py --repeats 10
    python warp/examples/benchmarks/benchmark_tile_matmul_pipelined.py --configs 16,16,16,32
"""

import argparse
from dataclasses import dataclass

import numpy as np

import warp as wp

DEFAULT_CONFIGS = (
    (16, 16, 16, 32),
    (32, 32, 16, 64),
    (32, 32, 32, 64),
    (64, 64, 16, 64),
    (64, 64, 32, 64),
)


@dataclass(frozen=True)
class TimingStats:
    median: float
    mean: float
    std: float
    minimum: float


def create_shared_gemm_kernel(tile_m, tile_n, tile_k):
    TILE_M = tile_m
    TILE_N = tile_n
    TILE_K = tile_k

    @wp.kernel
    def gemm_shared(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
        i, j = wp.tid()
        acc = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

        count = A.shape[1] // TILE_K
        for k in range(count):
            a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K), storage="shared")
            b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N), storage="shared")
            wp.tile_matmul(a, b, acc)

        wp.tile_store(C, acc, offset=(i * TILE_M, j * TILE_N))

    return gemm_shared


def create_pipelined_gemm_kernel(tile_m, tile_n, tile_k):
    TILE_M = tile_m
    TILE_N = tile_n
    TILE_K = tile_k

    @wp.kernel
    def gemm_pipelined(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
        i, j = wp.tid()
        acc = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0), storage="register")
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, j * TILE_N), storage="register")

        count = A.shape[1] // TILE_K
        for k in range(1, count):
            a_next = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K), storage="register")
            b_next = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N), storage="register")

            wp.tile_matmul(a, b, acc)
            a = a_next
            b = b_next

        wp.tile_matmul(a, b, acc)
        wp.tile_store(C, acc, offset=(i * TILE_M, j * TILE_N))

    return gemm_pipelined


def parse_config(config):
    values = config.split(",")
    if len(values) != 4:
        raise argparse.ArgumentTypeError("expected TILE_M,TILE_N,TILE_K,BLOCK_DIM")

    try:
        tile_m, tile_n, tile_k, block_dim = (int(value) for value in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("configuration entries must be integers") from exc

    if tile_m <= 0 or tile_n <= 0 or tile_k <= 0 or block_dim <= 0:
        raise argparse.ArgumentTypeError("configuration entries must be positive")

    return tile_m, tile_n, tile_k, block_dim


def validate_dimensions(m, n, k, configs):
    for tile_m, tile_n, tile_k, _block_dim in configs:
        if m % tile_m or n % tile_n or k % tile_k:
            raise ValueError(f"M={m}, N={n}, K={k} must be divisible by tile config {tile_m},{tile_n},{tile_k}")


def create_command(kernel, A, B, C, tile_m, tile_n, block_dim):
    M = A.shape[0]
    N = B.shape[1]
    return wp.launch_tiled(
        kernel=kernel,
        dim=[M // tile_m, N // tile_n],
        inputs=[A, B, C],
        block_dim=block_dim,
        record_cmd=True,
    )


def warm_up_command(cmd, warm_up, device):
    for _ in range(warm_up):
        cmd.launch()

    wp.synchronize_device(device)


def validate_command(cmd, C, expected, check_output, device):
    if check_output:
        cmd.launch()
        wp.synchronize_device(device)
        np.testing.assert_allclose(C.numpy(), expected, atol=1e-3, rtol=1e-3)


def time_command(cmd, iterations, device):
    wp.synchronize_device(device)
    with wp.ScopedTimer("benchmark", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        for _ in range(iterations):
            cmd.launch()

    return [result.elapsed for result in timer.timing_results]


def summarize_timing(timing_results):
    timings = np.asarray(timing_results, dtype=np.float64)
    timing_std = float(np.std(timings, ddof=1)) if len(timings) > 1 else 0.0
    return TimingStats(
        median=float(np.median(timings)),
        mean=float(np.mean(timings)),
        std=timing_std,
        minimum=float(np.min(timings)),
    )


def classify_ratio(ratio):
    if ratio < 0.95:
        return "pipeline faster"
    if ratio > 1.05:
        return "shared faster"
    return "similar"


def run_benchmark(args):
    configs = tuple(args.configs)
    validate_dimensions(args.m, args.n, args.k, configs)

    rng = np.random.default_rng(args.seed)
    A_np = rng.standard_normal((args.m, args.k), dtype=np.float32)
    B_np = rng.standard_normal((args.k, args.n), dtype=np.float32)
    expected = None if args.skip_check else A_np @ B_np

    device = wp.get_cuda_device()
    A = wp.array(A_np, device=device)
    B = wp.array(B_np, device=device)
    C_shared = wp.zeros((args.m, args.n), dtype=float, device=device)
    C_pipelined = wp.zeros_like(C_shared)

    print(f"Device: {device.name}")
    print(
        f"Config: M={args.m}, N={args.n}, K={args.k}, iterations={args.iterations}, "
        f"repeats={args.repeats}, warm_up={args.warm_up}, enable_mathdx_gemm=False"
    )
    print()

    columns = (
        ("TILE_M", 8),
        ("TILE_N", 8),
        ("TILE_K", 8),
        ("BLOCK", 8),
        ("Shared med", 12),
        ("Shared std", 12),
        ("Pipe med", 11),
        ("Pipe std", 10),
        ("Pipe/Shared", 13),
        ("Notes", 16),
    )
    header = "".join(f"{name:<{width}s}" for name, width in columns)
    print(header)
    print("-" * len(header))

    for tile_m, tile_n, tile_k, block_dim in configs:
        shared_kernel = create_shared_gemm_kernel(tile_m, tile_n, tile_k)
        pipelined_kernel = create_pipelined_gemm_kernel(tile_m, tile_n, tile_k)

        shared_cmd = create_command(
            shared_kernel,
            A,
            B,
            C_shared,
            tile_m,
            tile_n,
            block_dim,
        )
        pipelined_cmd = create_command(
            pipelined_kernel,
            A,
            B,
            C_pipelined,
            tile_m,
            tile_n,
            block_dim,
        )

        warm_up_command(shared_cmd, args.warm_up, device)
        warm_up_command(pipelined_cmd, args.warm_up, device)
        validate_command(shared_cmd, C_shared, expected, not args.skip_check, device)
        validate_command(pipelined_cmd, C_pipelined, expected, not args.skip_check, device)

        shared_timings = []
        pipelined_timings = []

        for repeat in range(args.repeats):
            if repeat % 2 == 0:
                first = (shared_cmd, shared_timings)
                second = (pipelined_cmd, pipelined_timings)
            else:
                first = (pipelined_cmd, pipelined_timings)
                second = (shared_cmd, shared_timings)

            first[1].extend(time_command(first[0], args.iterations, device))
            second[1].extend(time_command(second[0], args.iterations, device))

        shared_stats = summarize_timing(shared_timings)
        pipelined_stats = summarize_timing(pipelined_timings)
        ratio = pipelined_stats.median / shared_stats.median
        print(
            f"{tile_m:<8d}{tile_n:<8d}{tile_k:<8d}{block_dim:<8d}"
            f"{shared_stats.median:<12.6g}{shared_stats.std:<12.2g}"
            f"{pipelined_stats.median:<11.6g}{pipelined_stats.std:<10.2g}"
            f"{ratio:<13.4g}{classify_ratio(ratio):<16s}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark shared vs pipelined tile_matmul GEMM")
    parser.add_argument("--size", type=int, default=1024, help="Set M, N, and K to the same size (default: 1024)")
    parser.add_argument("--m", type=int, help="Rows in A/C. Overrides --size.")
    parser.add_argument("--n", type=int, help="Columns in B/C. Overrides --size.")
    parser.add_argument("--k", type=int, help="Columns in A and rows in B. Overrides --size.")
    parser.add_argument(
        "--configs",
        type=parse_config,
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Tile configurations as TILE_M,TILE_N,TILE_K,BLOCK_DIM",
    )
    parser.add_argument("--iterations", type=int, default=100, help="Timed iterations per kernel (default: 100)")
    parser.add_argument("--repeats", type=int, default=5, help="Alternating timing rounds per kernel (default: 5)")
    parser.add_argument("--warm-up", type=int, default=5, help="Warm-up launches per kernel (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--skip-check", action="store_true", help="Skip NumPy correctness checks")
    parser.add_argument("--clear-kernel-cache", action="store_true", help="Clear Warp's kernel cache before running")
    args = parser.parse_args()

    args.m = args.size if args.m is None else args.m
    args.n = args.size if args.n is None else args.n
    args.k = args.size if args.k is None else args.k

    if args.iterations <= 0 or args.repeats <= 0 or args.warm_up < 0:
        raise ValueError("--iterations and --repeats must be positive and --warm-up must be non-negative")

    wp.config.quiet = True
    wp.init()
    wp.set_module_options({"fast_math": True, "enable_backward": False, "enable_mathdx_gemm": False})

    if args.clear_kernel_cache:
        wp.clear_kernel_cache()

    if not wp.is_cuda_available():
        print("Error: This benchmark requires a CUDA device.")
        raise SystemExit(1)

    run_benchmark(args)


if __name__ == "__main__":
    main()
