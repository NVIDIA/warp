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

"""Benchmark comparing NanoVDB Volume SDF vs Texture-based Sparse SDF.

This benchmark creates a signed distance field from a simple box mesh and compares:
1. NanoVDB Volume-based SDF sampling (sparse + coarse with fallback)
2. Sparse SDF sampling using 3D CUDA textures (coarse + subgrid approach)

The texture-based SDF is built using GPU kernels following the pattern from 
PhysX's SDFConstruction.cu:
1. Build dense SDF grid (one thread per voxel querying mesh distance)
2. Build background SDF by sampling at subgrid corners
3. Mark required subgrids (those intersecting narrow band)
4. Use exclusive scan to assign sequential addresses
5. Populate subgrid textures from dense SDF
"""

from statistics import mean, stdev

import numpy as np

import warp as wp

from volume_sdf import (
    create_box_mesh,
    create_volume_from_mesh,
    sample_volume_with_fallback,
    BACKGROUND_VALUE,
)
from texture_sdf import (
    build_dense_sdf,
    build_sparse_sdf_from_dense,
    create_sparse_sdf_textures,
    sample_sparse_sdf,
)


# ============================================================================
# Benchmark Configuration
# ============================================================================

NUM_QUERY_POINTS = 1024 * 1024  # 1M query points
SDF_RESOLUTION = 64  # Grid resolution
SUBGRID_SIZE = 8  # Cells per subgrid block (matches volume tile size)
NARROW_BAND_WORLD = 0.1  # Narrow band distance in world units
ITERATIONS = 100
WARM_UP = 10


# ============================================================================
# Main Benchmark
# ============================================================================


def run_benchmark():
    wp.init()
    wp.clear_kernel_cache()
    wp.set_module_options({"fast_math": True, "enable_backward": False})

    device = "cuda:0"
    print("=" * 80)
    print("Sparse SDF vs Volume SDF Benchmark")
    print("=" * 80)

    # Configuration
    box_center = (0.5, 0.5, 0.5)
    box_half_extents = (0.3, 0.3, 0.3)
    margin = 0.2
    narrow_band = (-NARROW_BAND_WORLD, NARROW_BAND_WORLD)

    print(f"\nConfiguration:")
    print(f"  SDF Resolution: {SDF_RESOLUTION}")
    print(f"  Subgrid Size: {SUBGRID_SIZE}")
    print(f"  Narrow Band: +/-{NARROW_BAND_WORLD} world units")
    print(f"  Query Points: {NUM_QUERY_POINTS:,}")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Warm-up: {WARM_UP}")

    # ========== Create Mesh ==========
    print("\nCreating mesh...")
    mesh = create_box_mesh(box_center, box_half_extents, device)
    
    # Compute mesh bounds for dense SDF
    points_np = mesh.points.numpy()
    min_ext = np.min(points_np, axis=0) - margin
    max_ext = np.max(points_np, axis=0) + margin

    # ========== Create Volume SDF ==========
    print("\nCreating NanoVDB volume from mesh...")
    sparse_volume, coarse_volume, vol_min_ext, vol_max_ext, vol_spacing = create_volume_from_mesh(
        mesh, narrow_band, margin=margin, max_dims=SDF_RESOLUTION, verbose=True
    )
    print(f"  Volume bounds: [{vol_min_ext}, {vol_max_ext}]")

    # Verify volume works
    print("\n  Verifying volume sampling...")
    test_pts = np.array([
        [0.5, 0.5, 0.5],  # Center (inside)
        [0.2, 0.5, 0.5],  # Surface
        [0.1, 0.5, 0.5],  # Outside
    ], dtype=np.float32)
    test_pts_arr = wp.array(test_pts, dtype=wp.vec3, device=device)
    test_results = wp.zeros(len(test_pts), dtype=float, device=device)
    wp.launch(
        sample_volume_with_fallback,
        dim=len(test_pts),
        inputs=[sparse_volume.id, coarse_volume.id, BACKGROUND_VALUE, test_pts_arr, test_results],
        device=device,
    )
    wp.synchronize()
    for pt, val in zip(test_pts, test_results.numpy()):
        print(f"    {pt}: SDF = {val:.4f}")

    # ========== Create Texture SDF (GPU-accelerated construction) ==========
    print("\nBuilding texture-based sparse SDF using GPU kernels...")
    
    # Step 1: Build dense SDF from mesh
    print("  Step 1: Building dense SDF from mesh...")
    import time
    t0 = time.perf_counter()
    
    dense_sdf, dense_x, dense_y, dense_z, cell_size = build_dense_sdf(
        mesh, min_ext, max_ext, SDF_RESOLUTION, device
    )
    wp.synchronize()
    dense_time = time.perf_counter() - t0
    print(f"  Dense SDF build time: {dense_time * 1000:.2f} ms")
    
    # Step 2: Build sparse representation from dense SDF
    print("  Step 2: Building sparse representation...")
    t0 = time.perf_counter()
    
    sparse_data = build_sparse_sdf_from_dense(
        dense_sdf, dense_x, dense_y, dense_z,
        cell_size, min_ext, max_ext,
        SUBGRID_SIZE, NARROW_BAND_WORLD, device
    )
    wp.synchronize()
    sparse_build_time = time.perf_counter() - t0
    print(f"  Sparse build time: {sparse_build_time * 1000:.2f} ms")
    print(f"  Total texture SDF construction: {(dense_time + sparse_build_time) * 1000:.2f} ms")

    print("\nCreating sparse SDF textures...")
    coarse_tex, subgrid_tex, subgrid_slots, sdf_params = create_sparse_sdf_textures(sparse_data, device)
    print("  Textures created successfully")

    # ========== Generate Query Points ==========
    print("\nGenerating query points...")
    rng = np.random.default_rng(42)
    query_points_np = rng.uniform(
        low=min_ext,
        high=max_ext,
        size=(NUM_QUERY_POINTS, 3),
    ).astype(np.float32)

    query_points = wp.array(query_points_np, dtype=wp.vec3, device=device)
    results_volume = wp.zeros(NUM_QUERY_POINTS, dtype=float, device=device)
    results_sparse = wp.zeros(NUM_QUERY_POINTS, dtype=float, device=device)

    # ========== Benchmark Volume SDF ==========
    print("\nWarming up Volume SDF...")
    for _ in range(WARM_UP):
        wp.launch(
            sample_volume_with_fallback,
            dim=NUM_QUERY_POINTS,
            inputs=[sparse_volume.id, coarse_volume.id, BACKGROUND_VALUE, query_points, results_volume],
            device=device,
        )
    wp.synchronize()

    print("Benchmarking Volume SDF (NanoVDB)...")
    with wp.ScopedTimer("volume_sdf", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        for _ in range(ITERATIONS):
            wp.launch(
                sample_volume_with_fallback,
                dim=NUM_QUERY_POINTS,
                inputs=[sparse_volume.id, coarse_volume.id, BACKGROUND_VALUE, query_points, results_volume],
                device=device,
            )

    volume_times = [result.elapsed for result in timer.timing_results]
    volume_mean = mean(volume_times)
    volume_std = stdev(volume_times) if len(volume_times) > 1 else 0.0

    # ========== Benchmark Texture SDF ==========
    print("\nWarming up Sparse SDF (Textures)...")
    for _ in range(WARM_UP):
        wp.launch(
            sample_sparse_sdf,
            dim=NUM_QUERY_POINTS,
            inputs=[sdf_params, coarse_tex, subgrid_tex, subgrid_slots, query_points, results_sparse],
            device=device,
        )
    wp.synchronize()

    print("Benchmarking Sparse SDF (3D Textures)...")
    with wp.ScopedTimer("sparse_sdf", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        for _ in range(ITERATIONS):
            wp.launch(
                sample_sparse_sdf,
                dim=NUM_QUERY_POINTS,
                inputs=[sdf_params, coarse_tex, subgrid_tex, subgrid_slots, query_points, results_sparse],
                device=device,
            )

    sparse_times = [result.elapsed for result in timer.timing_results]
    sparse_mean = mean(sparse_times)
    sparse_std = stdev(sparse_times) if len(sparse_times) > 1 else 0.0

    # ========== Validate Results Match ==========
    print("\nValidating results match...")
    vol_np = results_volume.numpy()
    sparse_np = results_sparse.numpy()

    max_diff = np.max(np.abs(vol_np - sparse_np))
    mean_diff = np.mean(np.abs(vol_np - sparse_np))
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if max_diff < 0.1:
        print("  [OK] Results match within tolerance!")
    else:
        print("  [WARN] Results differ - this may be expected due to interpolation differences")

    # ========== Print Results ==========
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\nVolume SDF (NanoVDB sparse + coarse):")
    print(f"  Time per iteration: {volume_mean * 1000:.3f} +/- {volume_std * 1000:.3f} ms")
    print(f"  Throughput: {NUM_QUERY_POINTS / volume_mean / 1e6:.2f} M samples/sec")
    print(f"  Result stats: min={vol_np.min():.4f}, max={vol_np.max():.4f}, mean={vol_np.mean():.4f}")

    print(f"\nSparse SDF (3D Textures):")
    print(f"  Time per iteration: {sparse_mean * 1000:.3f} +/- {sparse_std * 1000:.3f} ms")
    print(f"  Throughput: {NUM_QUERY_POINTS / sparse_mean / 1e6:.2f} M samples/sec")
    print(f"  Result stats: min={sparse_np.min():.4f}, max={sparse_np.max():.4f}, mean={sparse_np.mean():.4f}")

    if volume_mean > 0 and sparse_mean > 0:
        speedup = volume_mean / sparse_mean
        print(f"\nSpeedup (Texture vs Volume): {speedup:.2f}x")

    # Memory usage
    coarse_mem = np.prod(sparse_data["coarse_sdf"].shape) * 4
    subgrid_mem = np.prod(sparse_data["subgrid_data"].shape) * 4
    slots_mem = len(sparse_data["subgrid_start_slots"]) * 4
    dense_mem = dense_x * dense_y * dense_z * 4

    print(f"\nMemory Usage:")
    print(f"  Dense SDF (temporary): {dense_mem / 1024:.1f} KB")
    print(f"  Coarse texture: {coarse_mem / 1024:.1f} KB")
    print(f"  Subgrid texture: {subgrid_mem / 1024:.1f} KB")
    print(f"  Indirection slots: {slots_mem / 1024:.1f} KB")
    print(f"  Total (persistent): {(coarse_mem + subgrid_mem + slots_mem) / 1024:.1f} KB")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_benchmark()
