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

"""Compare Mesh query performance between single-threaded and tiled (thread-block parallel) queries.

This benchmark compares AABB queries for both single-threaded and tiled versions.
"""

from statistics import mean, stdev

import numpy as np

import warp as wp

# Benchmark configuration
NUM_QUERIES = 128  # Number of queries to run in parallel
BLOCK_DIMS = [32, 64, 128, 256]  # Block dimensions to test for tiled queries
MESH_SIZE = 100  # Grid size for mesh (creates MESH_SIZE x MESH_SIZE x 2 triangles)
ITERATIONS = 100  # Number of benchmark iterations
WARM_UP = 5  # Number of warm-up iterations


# Single-threaded AABB query kernel
@wp.kernel
def mesh_query_aabb_kernel(
    mesh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    hit_counts: wp.array(dtype=int),
):
    i = wp.tid()
    query = wp.mesh_query_aabb(mesh_id, query_lowers[i], query_uppers[i])
    face_index = int(0)
    count = int(0)

    while wp.mesh_query_aabb_next(query, face_index):
        count += 1

    hit_counts[i] = count


# Tiled AABB query kernel
@wp.kernel
def tile_mesh_query_aabb_kernel(
    mesh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    hit_counts: wp.array(dtype=int),
):
    i, _j = wp.tid()
    query = wp.tile_mesh_query_aabb(mesh_id, query_lowers[i], query_uppers[i])

    result_tile = wp.tile_mesh_query_aabb_next(query)

    # Continue querying while we have results
    # Each iteration, each thread in the block gets one result (or -1)
    while wp.tile_max(result_tile)[0] >= 0:
        # Each thread processes its result from the tile
        result_idx = wp.untile(result_tile)

        # Atomically increment the count for each valid hit
        if result_idx >= 0:
            wp.atomic_add(hit_counts, i, 1)

        result_tile = wp.tile_mesh_query_aabb_next(query)


def benchmark_mesh_aabb(mesh, query_lowers, query_uppers, hit_counts, warm_up, iterations):
    """Benchmark single-threaded AABB queries."""
    # Zero the hit counts
    hit_counts.zero_()

    # Warm-up
    for _ in range(warm_up):
        wp.launch(
            kernel=mesh_query_aabb_kernel,
            dim=NUM_QUERIES,
            inputs=[mesh.id, query_lowers, query_uppers, hit_counts],
            device="cuda",
        )

    # Benchmark
    with wp.ScopedTimer("mesh_aabb", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        for _ in range(iterations):
            wp.launch(
                kernel=mesh_query_aabb_kernel,
                dim=NUM_QUERIES,
                inputs=[mesh.id, query_lowers, query_uppers, hit_counts],
                device="cuda",
            )

    timing_results = [result.elapsed for result in timer.timing_results]
    return mean(timing_results), stdev(timing_results)


def benchmark_tile_mesh_aabb(mesh, query_lowers, query_uppers, hit_counts, warm_up, iterations, block_dim):
    """Benchmark tiled AABB queries."""
    # Zero the hit counts
    hit_counts.zero_()

    # Create launch command for reuse
    cmd = wp.launch_tiled(
        kernel=tile_mesh_query_aabb_kernel,
        dim=NUM_QUERIES,
        inputs=[mesh.id, query_lowers, query_uppers, hit_counts],
        device="cuda",
        block_dim=block_dim,
        record_cmd=True,
    )

    # Warm-up
    for _ in range(warm_up):
        cmd.launch()

    # Benchmark
    with wp.ScopedTimer("tile_mesh_aabb", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        for _ in range(iterations):
            cmd.launch()

    timing_results = [result.elapsed for result in timer.timing_results]
    return mean(timing_results), stdev(timing_results)


def create_grid_mesh(size, device):
    """Create a grid mesh with (size x size) quads (2 triangles each)."""
    # Create vertices
    num_vertices = (size + 1) * (size + 1)
    vertices = np.zeros((num_vertices, 3), dtype=np.float32)

    for i in range(size + 1):
        for j in range(size + 1):
            idx = i * (size + 1) + j
            vertices[idx] = [i, j, 0]

    # Create triangles (2 per quad)
    num_triangles = size * size * 2
    indices = np.zeros((num_triangles, 3), dtype=np.int32)

    tri_idx = 0
    for i in range(size):
        for j in range(size):
            # Bottom-left corner of quad
            v0 = i * (size + 1) + j
            v1 = v0 + 1
            v2 = v0 + (size + 1)
            v3 = v2 + 1

            # First triangle
            indices[tri_idx] = [v0, v1, v2]
            tri_idx += 1

            # Second triangle
            indices[tri_idx] = [v1, v3, v2]
            tri_idx += 1

    # Convert to warp arrays
    points = wp.array(vertices, dtype=wp.vec3, device=device)
    indices_array = wp.array(indices.flatten(), dtype=int, device=device)

    return wp.Mesh(points=points, indices=indices_array)


if __name__ == "__main__":
    wp.init()
    wp.clear_kernel_cache()
    wp.set_module_options({"fast_math": True, "enable_backward": False})

    # Create mesh
    print(f"Creating mesh with {MESH_SIZE}x{MESH_SIZE} grid ({MESH_SIZE * MESH_SIZE * 2} triangles)...")
    mesh = create_grid_mesh(MESH_SIZE, "cuda")
    num_triangles = MESH_SIZE * MESH_SIZE * 2

    # Generate query data
    # AABB queries: random boxes in the mesh space
    # Make sure they intersect the Z=0 plane where the mesh is
    rng = np.random.default_rng(42)
    query_lowers_np = rng.random(size=(NUM_QUERIES, 3)) * (MESH_SIZE * 0.8)
    query_uppers_np = query_lowers_np + rng.random(size=(NUM_QUERIES, 3)) * (MESH_SIZE * 0.2)
    # Force Z coordinates to span across Z=0 so queries intersect the mesh
    query_lowers_np[:, 2] = -1.0
    query_uppers_np[:, 2] = 1.0

    query_lowers = wp.array(query_lowers_np, dtype=wp.vec3, device="cuda")
    query_uppers = wp.array(query_uppers_np, dtype=wp.vec3, device="cuda")

    # Output arrays
    hit_counts_single = wp.zeros(NUM_QUERIES, dtype=int, device="cuda")
    hit_counts_tiled = wp.zeros(NUM_QUERIES, dtype=int, device="cuda")

    print("\nBenchmark Configuration:")
    print(f"  Num Queries: {NUM_QUERIES}")
    print(f"  Num Triangles: {num_triangles}")
    print(f"  Block Dims (tiled): {BLOCK_DIMS}")
    print(f"  Total threads (single): {NUM_QUERIES}")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Warm-up: {WARM_UP}")
    print()

    # Run single-threaded benchmark once (same for all block dims)
    print("Benchmarking single-threaded AABB queries...")
    time_aabb_mean, time_aabb_std = benchmark_mesh_aabb(
        mesh, query_lowers, query_uppers, hit_counts_single, WARM_UP, ITERATIONS
    )

    # Store results for each block dimension
    aabb_results = {}

    # Loop over all block dimensions
    for block_dim in BLOCK_DIMS:
        print(f"\n{'=' * 80}")
        print(f"Testing with BLOCK_DIM = {block_dim} (Total threads: {NUM_QUERIES * block_dim})")
        print(f"{'=' * 80}")

        # Benchmark AABB queries
        print("  Benchmarking tiled AABB queries...")
        time_tile_aabb_mean, time_tile_aabb_std = benchmark_tile_mesh_aabb(
            mesh, query_lowers, query_uppers, hit_counts_tiled, WARM_UP, ITERATIONS, block_dim
        )

        # Run validation pass for AABB
        hit_counts_single.zero_()
        wp.launch(
            kernel=mesh_query_aabb_kernel,
            dim=NUM_QUERIES,
            inputs=[mesh.id, query_lowers, query_uppers, hit_counts_single],
            device="cuda",
        )

        hit_counts_tiled.zero_()
        wp.launch_tiled(
            kernel=tile_mesh_query_aabb_kernel,
            dim=NUM_QUERIES,
            inputs=[mesh.id, query_lowers, query_uppers, hit_counts_tiled],
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

        # Store results
        aabb_results[block_dim] = (time_tile_aabb_mean, time_tile_aabb_std)

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

    print("=" * 100)
