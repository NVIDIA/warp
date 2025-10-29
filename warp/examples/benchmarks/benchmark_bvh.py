# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Compare BVH query performance between single-threaded and tiled (thread-block parallel) queries.

This benchmark compares AABB and ray queries for both single-threaded and tiled versions.
"""

from statistics import mean, stdev

import numpy as np

import warp as wp

# Benchmark configuration
NUM_QUERIES = 128  # Number of queries to run in parallel
BLOCK_DIMS = [32, 64, 128, 256]  # Block dimensions to test for tiled queries
NUM_BOUNDS = 10000  # Number of bounding boxes in the BVH
ITERATIONS = 100  # Number of benchmark iterations
WARM_UP = 5  # Number of warm-up iterations


# Single-threaded AABB query kernel
@wp.kernel
def bvh_query_aabb_kernel(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    hit_counts: wp.array(dtype=int),
):
    i = wp.tid()
    query = wp.bvh_query_aabb(bvh_id, query_lowers[i], query_uppers[i])
    bounds_nr = int(0)
    count = int(0)

    while wp.bvh_query_next(query, bounds_nr):
        count += 1

    hit_counts[i] = count


# Single-threaded ray query kernel
@wp.kernel
def bvh_query_ray_kernel(
    bvh_id: wp.uint64,
    query_starts: wp.array(dtype=wp.vec3),
    query_dirs: wp.array(dtype=wp.vec3),
    hit_counts: wp.array(dtype=int),
):
    i = wp.tid()
    query = wp.bvh_query_ray(bvh_id, query_starts[i], query_dirs[i])
    bounds_nr = int(0)
    count = int(0)

    while wp.bvh_query_next(query, bounds_nr):
        count += 1

    hit_counts[i] = count


# Tiled AABB query kernel
@wp.kernel
def tile_bvh_query_aabb_kernel(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    hit_counts: wp.array(dtype=int),
):
    i, _j = wp.tid()
    query = wp.tile_bvh_query_aabb(bvh_id, query_lowers[i], query_uppers[i])

    result_tile = wp.tile_bvh_query_next(query)

    # Continue querying while we have results
    # Each iteration, each thread in the block gets one result (or -1)
    while wp.tile_max(result_tile)[0] >= 0:
        # Each thread processes its result from the tile
        result_idx = wp.untile(result_tile)

        # Atomically increment the count for each valid hit
        if result_idx >= 0:
            wp.atomic_add(hit_counts, i, 1)

        result_tile = wp.tile_bvh_query_next(query)


# Tiled ray query kernel
@wp.kernel
def tile_bvh_query_ray_kernel(
    bvh_id: wp.uint64,
    query_starts: wp.array(dtype=wp.vec3),
    query_dirs: wp.array(dtype=wp.vec3),
    hit_counts: wp.array(dtype=int),
):
    i, _j = wp.tid()
    query = wp.tile_bvh_query_ray(bvh_id, query_starts[i], query_dirs[i])

    result_tile = wp.tile_bvh_query_next(query)

    # Continue querying while we have results
    # Each iteration, each thread in the block gets one result (or -1)
    while wp.tile_max(result_tile)[0] >= 0:
        # Each thread processes its result from the tile
        result_idx = wp.untile(result_tile)

        # Atomically increment the count for each valid hit
        if result_idx >= 0:
            wp.atomic_add(hit_counts, i, 1)

        result_tile = wp.tile_bvh_query_next(query)


def benchmark_bvh_aabb(bvh, query_lowers, query_uppers, hit_counts, warm_up, iterations):
    """Benchmark single-threaded AABB queries."""
    # Zero the hit counts
    hit_counts.zero_()

    # Warm-up
    for _ in range(warm_up):
        wp.launch(
            kernel=bvh_query_aabb_kernel,
            dim=NUM_QUERIES,
            inputs=[bvh.id, query_lowers, query_uppers, hit_counts],
            device="cuda",
        )

    # Benchmark
    with wp.ScopedTimer("bvh_aabb", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        for _ in range(iterations):
            wp.launch(
                kernel=bvh_query_aabb_kernel,
                dim=NUM_QUERIES,
                inputs=[bvh.id, query_lowers, query_uppers, hit_counts],
                device="cuda",
            )

    timing_results = [result.elapsed for result in timer.timing_results]
    return mean(timing_results), stdev(timing_results)


def benchmark_bvh_ray(bvh, query_starts, query_dirs, hit_counts, warm_up, iterations):
    """Benchmark single-threaded ray queries."""
    # Zero the hit counts
    hit_counts.zero_()

    # Warm-up
    for _ in range(warm_up):
        wp.launch(
            kernel=bvh_query_ray_kernel,
            dim=NUM_QUERIES,
            inputs=[bvh.id, query_starts, query_dirs, hit_counts],
            device="cuda",
        )

    # Benchmark
    with wp.ScopedTimer("bvh_ray", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        for _ in range(iterations):
            wp.launch(
                kernel=bvh_query_ray_kernel,
                dim=NUM_QUERIES,
                inputs=[bvh.id, query_starts, query_dirs, hit_counts],
                device="cuda",
            )

    timing_results = [result.elapsed for result in timer.timing_results]
    return mean(timing_results), stdev(timing_results)


def benchmark_tile_bvh_aabb(bvh, query_lowers, query_uppers, hit_counts, warm_up, iterations, block_dim):
    """Benchmark tiled AABB queries."""
    # Zero the hit counts
    hit_counts.zero_()

    # Create launch command for reuse
    cmd = wp.launch_tiled(
        kernel=tile_bvh_query_aabb_kernel,
        dim=NUM_QUERIES,
        inputs=[bvh.id, query_lowers, query_uppers, hit_counts],
        device="cuda",
        block_dim=block_dim,
        record_cmd=True,
    )

    # Warm-up
    for _ in range(warm_up):
        cmd.launch()

    # Benchmark
    with wp.ScopedTimer("tile_bvh_aabb", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        for _ in range(iterations):
            cmd.launch()

    timing_results = [result.elapsed for result in timer.timing_results]
    return mean(timing_results), stdev(timing_results)


def benchmark_tile_bvh_ray(bvh, query_starts, query_dirs, hit_counts, warm_up, iterations, block_dim):
    """Benchmark tiled ray queries."""
    # Zero the hit counts
    hit_counts.zero_()

    # Create launch command for reuse
    cmd = wp.launch_tiled(
        kernel=tile_bvh_query_ray_kernel,
        dim=NUM_QUERIES,
        inputs=[bvh.id, query_starts, query_dirs, hit_counts],
        device="cuda",
        block_dim=block_dim,
        record_cmd=True,
    )

    # Warm-up
    for _ in range(warm_up):
        cmd.launch()

    # Benchmark
    with wp.ScopedTimer("tile_bvh_ray", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        for _ in range(iterations):
            cmd.launch()

    timing_results = [result.elapsed for result in timer.timing_results]
    return mean(timing_results), stdev(timing_results)


if __name__ == "__main__":
    wp.init()
    wp.clear_kernel_cache()
    wp.set_module_options({"fast_math": True, "enable_backward": False})

    # Create random BVH data
    rng = np.random.default_rng(42)
    lowers = rng.random(size=(NUM_BOUNDS, 3)) * 100.0
    uppers = lowers + rng.random(size=(NUM_BOUNDS, 3)) * 10.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device="cuda")
    device_uppers = wp.array(uppers, dtype=wp.vec3, device="cuda")

    # Build BVH
    print(f"Building BVH with {NUM_BOUNDS} bounds...")
    bvh = wp.Bvh(device_lowers, device_uppers)

    # Generate query data
    # AABB queries: random boxes in the space
    query_lowers_np = rng.random(size=(NUM_QUERIES, 3)) * 80.0
    query_uppers_np = query_lowers_np + rng.random(size=(NUM_QUERIES, 3)) * 20.0

    query_lowers = wp.array(query_lowers_np, dtype=wp.vec3, device="cuda")
    query_uppers = wp.array(query_uppers_np, dtype=wp.vec3, device="cuda")

    # Ray queries: random rays through the space
    query_starts_np = rng.random(size=(NUM_QUERIES, 3)) * 50.0
    query_dirs_np = rng.random(size=(NUM_QUERIES, 3)) - 0.5
    # Normalize directions
    query_dirs_np = query_dirs_np / np.linalg.norm(query_dirs_np, axis=1, keepdims=True)

    query_starts = wp.array(query_starts_np, dtype=wp.vec3, device="cuda")
    query_dirs = wp.array(query_dirs_np, dtype=wp.vec3, device="cuda")

    # Output arrays
    hit_counts_single = wp.zeros(NUM_QUERIES, dtype=int, device="cuda")
    hit_counts_tiled = wp.zeros(NUM_QUERIES, dtype=int, device="cuda")

    print("\nBenchmark Configuration:")
    print(f"  Num Queries: {NUM_QUERIES}")
    print(f"  Num Bounds: {NUM_BOUNDS}")
    print(f"  Block Dims (tiled): {BLOCK_DIMS}")
    print(f"  Total threads (single): {NUM_QUERIES}")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Warm-up: {WARM_UP}")
    print()

    # Run single-threaded benchmarks once (same for all block dims)
    print("Benchmarking single-threaded AABB queries...")
    time_aabb_mean, time_aabb_std = benchmark_bvh_aabb(
        bvh, query_lowers, query_uppers, hit_counts_single, WARM_UP, ITERATIONS
    )

    print("Benchmarking single-threaded ray queries...")
    time_ray_mean, time_ray_std = benchmark_bvh_ray(
        bvh, query_starts, query_dirs, hit_counts_single, WARM_UP, ITERATIONS
    )

    # Store results for each block dimension
    aabb_results = {}
    ray_results = {}

    # Loop over all block dimensions
    for block_dim in BLOCK_DIMS:
        print(f"\n{'=' * 80}")
        print(f"Testing with BLOCK_DIM = {block_dim} (Total threads: {NUM_QUERIES * block_dim})")
        print(f"{'=' * 80}")

        # Benchmark AABB queries
        print("  Benchmarking tiled AABB queries...")
        time_tile_aabb_mean, time_tile_aabb_std = benchmark_tile_bvh_aabb(
            bvh, query_lowers, query_uppers, hit_counts_tiled, WARM_UP, ITERATIONS, block_dim
        )

        # Run validation pass for AABB
        hit_counts_single.zero_()
        wp.launch(
            kernel=bvh_query_aabb_kernel,
            dim=NUM_QUERIES,
            inputs=[bvh.id, query_lowers, query_uppers, hit_counts_single],
            device="cuda",
        )

        hit_counts_tiled.zero_()
        wp.launch_tiled(
            kernel=tile_bvh_query_aabb_kernel,
            dim=NUM_QUERIES,
            inputs=[bvh.id, query_lowers, query_uppers, hit_counts_tiled],
            device="cuda",
            block_dim=block_dim,
        )

        # Verify AABB results match
        single_hits = hit_counts_single.numpy()
        tiled_hits = hit_counts_tiled.numpy()
        if not np.allclose(single_hits, tiled_hits):
            print("  WARNING: Results don't match for AABB queries!")
            print(f"    Single hits: {single_hits}")
            print(f"    Tiled hits:  {tiled_hits}")
        else:
            print(f"  ✓ AABB results verified (avg hits: {single_hits.mean():.1f})")

        # Benchmark ray queries
        print("  Benchmarking tiled ray queries...")
        time_tile_ray_mean, time_tile_ray_std = benchmark_tile_bvh_ray(
            bvh, query_starts, query_dirs, hit_counts_tiled, WARM_UP, ITERATIONS, block_dim
        )

        # Run validation pass for ray
        hit_counts_single.zero_()
        wp.launch(
            kernel=bvh_query_ray_kernel,
            dim=NUM_QUERIES,
            inputs=[bvh.id, query_starts, query_dirs, hit_counts_single],
            device="cuda",
        )

        hit_counts_tiled.zero_()
        wp.launch_tiled(
            kernel=tile_bvh_query_ray_kernel,
            dim=NUM_QUERIES,
            inputs=[bvh.id, query_starts, query_dirs, hit_counts_tiled],
            device="cuda",
            block_dim=block_dim,
        )

        # Verify ray results match
        single_hits = hit_counts_single.numpy()
        tiled_hits = hit_counts_tiled.numpy()
        if not np.allclose(single_hits, tiled_hits):
            print("  WARNING: Results don't match for ray queries!")
            print(f"    Single hits: {single_hits}")
            print(f"    Tiled hits:  {tiled_hits}")
        else:
            print(f"  ✓ Ray results verified (avg hits: {single_hits.mean():.1f})")

        # Store results
        aabb_results[block_dim] = (time_tile_aabb_mean, time_tile_aabb_std)
        ray_results[block_dim] = (time_tile_ray_mean, time_tile_ray_std)

    # Print summary results
    print("\n" + "=" * 100)
    print(f"{'Query Type':<15s} {'Method':<20s} {'Time (ms)':<20s} {'Speedup':<15s} {'Threads':<15s}")
    print("=" * 100)

    # AABB results
    print(
        f"{'AABB':<15s} {'Single':<20s} {f'{time_aabb_mean:.6g}±{time_aabb_std:.2g}':<20s} {'1.00x':<15s} {NUM_QUERIES:<15d}"
    )
    for block_dim in BLOCK_DIMS:
        time_mean, time_std = aabb_results[block_dim]
        speedup = time_aabb_mean / time_mean
        total_threads = NUM_QUERIES * block_dim
        print(
            f"{'AABB':<15s} {f'Tiled (BD={block_dim})':<20s} {f'{time_mean:.6g}±{time_std:.2g}':<20s} "
            f"{f'{speedup:.2f}x':<15s} {total_threads:<15d}"
        )

    print()

    # Ray results
    print(
        f"{'Ray':<15s} {'Single':<20s} {f'{time_ray_mean:.6g}±{time_ray_std:.2g}':<20s} {'1.00x':<15s} {NUM_QUERIES:<15d}"
    )
    for block_dim in BLOCK_DIMS:
        time_mean, time_std = ray_results[block_dim]
        speedup = time_ray_mean / time_mean
        total_threads = NUM_QUERIES * block_dim
        print(
            f"{'Ray':<15s} {f'Tiled (BD={block_dim})':<20s} {f'{time_mean:.6g}±{time_std:.2g}':<20s} "
            f"{f'{speedup:.2f}x':<15s} {total_threads:<15d}"
        )

    print("=" * 100)
