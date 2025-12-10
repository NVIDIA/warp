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

import time
from statistics import mean, stdev

import numpy as np
from texture_sdf import (
    QuantizationMode,
    build_dense_sdf,
    build_sparse_sdf_from_dense,
    create_sparse_sdf_textures,
    get_quantization_bytes,
    sample_sparse_sdf,
    sample_sparse_sdf_grad,
)
from volume_sdf import (
    create_box_mesh,
    create_volume_from_mesh,
    sample_volume_with_fallback,
    sample_volume_with_fallback_grad,
)

import warp as wp

# ============================================================================
# Benchmark Configuration
# ============================================================================

NUM_QUERY_POINTS = 1024 * 1024  # 1M query points
SDF_RESOLUTION = 256  # Grid resolution
SUBGRID_SIZE = 6  # Cells per subgrid block (optimal value from PhysX testing)
NARROW_BAND_WORLD = 0.1  # Narrow band distance in world units
ITERATIONS = 100
WARM_UP = 10

# Quantization mode for subgrid texture data:
#   QuantizationMode.FLOAT32 - Full precision (4 bytes per sample)
#   QuantizationMode.UINT16  - 16-bit compression (2 bytes per sample)
#   QuantizationMode.UINT8   - 8-bit compression (1 byte per sample)
QUANTIZATION_MODE = QuantizationMode.UINT16


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

    # Map quantization mode to readable name
    quant_names = {
        QuantizationMode.FLOAT32: "FLOAT32 (4 bytes)",
        QuantizationMode.UINT16: "UINT16 (2 bytes)",
        QuantizationMode.UINT8: "UINT8 (1 byte)",
    }

    print("\nConfiguration:")
    print(f"  SDF Resolution: {SDF_RESOLUTION}")
    print(f"  Subgrid Size: {SUBGRID_SIZE}")
    print(f"  Narrow Band: +/-{NARROW_BAND_WORLD} world units")
    print(f"  Quantization: {quant_names.get(QUANTIZATION_MODE, 'Unknown')}")
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
    sparse_volume, coarse_volume, sparse_max_value, vol_min_ext, vol_max_ext, _ = create_volume_from_mesh(
        mesh, narrow_band, margin=margin, max_dims=SDF_RESOLUTION, verbose=True
    )
    print(f"  Volume bounds: [{vol_min_ext}, {vol_max_ext}]")

    # Verify volume works
    print("\n  Verifying volume sampling...")
    test_pts = np.array(
        [
            [0.5, 0.5, 0.5],  # Center (inside)
            [0.2, 0.5, 0.5],  # Surface
            [0.1, 0.5, 0.5],  # Outside
        ],
        dtype=np.float32,
    )
    test_pts_arr = wp.array(test_pts, dtype=wp.vec3, device=device)
    test_results = wp.zeros(len(test_pts), dtype=float, device=device)
    sdf_lower = wp.vec3(vol_min_ext[0], vol_min_ext[1], vol_min_ext[2])
    sdf_upper = wp.vec3(vol_max_ext[0], vol_max_ext[1], vol_max_ext[2])
    sparse_threshold = sparse_max_value * 1.5  # Precompute threshold
    wp.launch(
        sample_volume_with_fallback,
        dim=len(test_pts),
        inputs=[sparse_volume.id, coarse_volume.id, sparse_threshold, sdf_lower, sdf_upper, test_pts_arr, test_results],
        device=device,
    )
    wp.synchronize()
    for pt, val in zip(test_pts, test_results.numpy()):
        print(f"    {pt}: SDF = {val:.4f}")

    # ========== Create Texture SDF (GPU-accelerated construction) ==========
    print("\nBuilding texture-based sparse SDF using GPU kernels...")

    # Step 1: Build dense SDF from mesh
    print("  Step 1: Building dense SDF from mesh...")
    t0 = time.perf_counter()

    dense_sdf, dense_x, dense_y, dense_z, cell_size = build_dense_sdf(mesh, min_ext, max_ext, SDF_RESOLUTION, device)
    wp.synchronize()
    dense_time = time.perf_counter() - t0
    print(f"  Dense SDF build time: {dense_time * 1000:.2f} ms")

    # Step 2: Build sparse representation from dense SDF
    print("  Step 2: Building sparse representation...")
    t0 = time.perf_counter()

    sparse_data = build_sparse_sdf_from_dense(
        dense_sdf,
        dense_x,
        dense_y,
        dense_z,
        cell_size,
        min_ext,
        max_ext,
        subgrid_size=SUBGRID_SIZE,
        narrow_band_thickness=NARROW_BAND_WORLD,
        quantization_mode=QUANTIZATION_MODE,
        device=device,
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
            inputs=[
                sparse_volume.id,
                coarse_volume.id,
                sparse_threshold,
                sdf_lower,
                sdf_upper,
                query_points,
                results_volume,
            ],
            device=device,
        )
    wp.synchronize()

    print("Benchmarking Volume SDF (NanoVDB)...")
    with wp.ScopedTimer("volume_sdf", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        for _ in range(ITERATIONS):
            wp.launch(
                sample_volume_with_fallback,
                dim=NUM_QUERY_POINTS,
                inputs=[
                    sparse_volume.id,
                    coarse_volume.id,
                    sparse_threshold,
                    sdf_lower,
                    sdf_upper,
                    query_points,
                    results_volume,
                ],
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

    # ========== Benchmark Gradient Evaluation ==========
    print("\n" + "-" * 40)
    print("GRADIENT EVALUATION BENCHMARKS")
    print("-" * 40)

    # Allocate gradient arrays
    gradients_volume = wp.zeros(NUM_QUERY_POINTS, dtype=wp.vec3, device=device)
    gradients_sparse = wp.zeros(NUM_QUERY_POINTS, dtype=wp.vec3, device=device)

    # ========== Benchmark Volume SDF Gradient ==========
    print("\nWarming up Volume SDF gradient...")
    for _ in range(WARM_UP):
        wp.launch(
            sample_volume_with_fallback_grad,
            dim=NUM_QUERY_POINTS,
            inputs=[
                sparse_volume.id,
                coarse_volume.id,
                sparse_threshold,
                sdf_lower,
                sdf_upper,
                query_points,
                results_volume,
                gradients_volume,
            ],
            device=device,
        )
    wp.synchronize()

    print("Benchmarking Volume SDF gradient (NanoVDB)...")
    with wp.ScopedTimer("volume_sdf_grad", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        for _ in range(ITERATIONS):
            wp.launch(
                sample_volume_with_fallback_grad,
                dim=NUM_QUERY_POINTS,
                inputs=[
                    sparse_volume.id,
                    coarse_volume.id,
                    sparse_threshold,
                    sdf_lower,
                    sdf_upper,
                    query_points,
                    results_volume,
                    gradients_volume,
                ],
                device=device,
            )

    volume_grad_times = [result.elapsed for result in timer.timing_results]
    volume_grad_mean = mean(volume_grad_times)
    volume_grad_std = stdev(volume_grad_times) if len(volume_grad_times) > 1 else 0.0

    # ========== Benchmark Texture SDF Gradient ==========
    print("\nWarming up Sparse SDF gradient (Textures)...")
    for _ in range(WARM_UP):
        wp.launch(
            sample_sparse_sdf_grad,
            dim=NUM_QUERY_POINTS,
            inputs=[sdf_params, coarse_tex, subgrid_tex, subgrid_slots, query_points, results_sparse, gradients_sparse],
            device=device,
        )
    wp.synchronize()

    print("Benchmarking Sparse SDF gradient (3D Textures with finite differences)...")
    with wp.ScopedTimer("sparse_sdf_grad", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        for _ in range(ITERATIONS):
            wp.launch(
                sample_sparse_sdf_grad,
                dim=NUM_QUERY_POINTS,
                inputs=[
                    sdf_params,
                    coarse_tex,
                    subgrid_tex,
                    subgrid_slots,
                    query_points,
                    results_sparse,
                    gradients_sparse,
                ],
                device=device,
            )

    sparse_grad_times = [result.elapsed for result in timer.timing_results]
    sparse_grad_mean = mean(sparse_grad_times)
    sparse_grad_std = stdev(sparse_grad_times) if len(sparse_grad_times) > 1 else 0.0

    # Validate gradient results
    print("\nValidating gradient results...")
    vol_grad_np = gradients_volume.numpy()
    sparse_grad_np = gradients_sparse.numpy()

    # Compute gradient magnitudes
    vol_grad_mag = np.linalg.norm(vol_grad_np, axis=1)
    sparse_grad_mag = np.linalg.norm(sparse_grad_np, axis=1)

    print(f"  Volume gradient magnitude: min={vol_grad_mag.min():.4f}, max={vol_grad_mag.max():.4f}")
    print(f"  Texture gradient magnitude: min={sparse_grad_mag.min():.4f}, max={sparse_grad_mag.max():.4f}")

    # Check angular difference between gradients (using normalized dot product)
    vol_grad_normalized = vol_grad_np / (np.linalg.norm(vol_grad_np, axis=1, keepdims=True) + 1e-10)
    sparse_grad_normalized = sparse_grad_np / (np.linalg.norm(sparse_grad_np, axis=1, keepdims=True) + 1e-10)
    dot_products = np.sum(vol_grad_normalized * sparse_grad_normalized, axis=1)
    angular_diff_deg = np.arccos(np.clip(dot_products, -1, 1)) * 180 / np.pi

    print(f"  Angular difference: mean={angular_diff_deg.mean():.2f} deg, max={angular_diff_deg.max():.2f} deg")

    # ========== Print Results ==========
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    print("\nVolume SDF (NanoVDB sparse + coarse):")
    print(f"  Time per iteration: {volume_mean * 1000:.3f} +/- {volume_std * 1000:.3f} ms")
    print(f"  Throughput: {NUM_QUERY_POINTS / volume_mean / 1e6:.2f} M samples/sec")
    print(f"  Result stats: min={vol_np.min():.4f}, max={vol_np.max():.4f}, mean={vol_np.mean():.4f}")

    print("\nSparse SDF (3D Textures):")
    print(f"  Time per iteration: {sparse_mean * 1000:.3f} +/- {sparse_std * 1000:.3f} ms")
    print(f"  Throughput: {NUM_QUERY_POINTS / sparse_mean / 1e6:.2f} M samples/sec")
    print(f"  Result stats: min={sparse_np.min():.4f}, max={sparse_np.max():.4f}, mean={sparse_np.mean():.4f}")

    if volume_mean > 0 and sparse_mean > 0:
        speedup = volume_mean / sparse_mean
        print(f"\nSpeedup (Texture vs Volume): {speedup:.2f}x")

    print("\n" + "-" * 40)
    print("GRADIENT EVALUATION RESULTS")
    print("-" * 40)

    print("\nVolume SDF Gradient (NanoVDB volume_sample_grad_f):")
    print(f"  Time per iteration: {volume_grad_mean * 1000:.3f} +/- {volume_grad_std * 1000:.3f} ms")
    print(f"  Throughput: {NUM_QUERY_POINTS / volume_grad_mean / 1e6:.2f} M samples/sec")
    grad_overhead_volume = (volume_grad_mean - volume_mean) / volume_mean * 100
    print(f"  Overhead vs value-only: {grad_overhead_volume:.1f}%")

    print("\nSparse SDF Gradient (3D Textures + finite differences):")
    print(f"  Time per iteration: {sparse_grad_mean * 1000:.3f} +/- {sparse_grad_std * 1000:.3f} ms")
    print(f"  Throughput: {NUM_QUERY_POINTS / sparse_grad_mean / 1e6:.2f} M samples/sec")
    grad_overhead_sparse = (sparse_grad_mean - sparse_mean) / sparse_mean * 100
    print(f"  Overhead vs value-only: {grad_overhead_sparse:.1f}%")

    if volume_grad_mean > 0 and sparse_grad_mean > 0:
        grad_speedup = volume_grad_mean / sparse_grad_mean
        print(f"\nGradient Speedup (Texture vs Volume): {grad_speedup:.2f}x")

    # Memory usage (accounting for quantization mode)
    # Note: textures are always float32 on GPU, but source data size varies
    bytes_per_sample = get_quantization_bytes(QUANTIZATION_MODE)
    coarse_mem = np.prod(sparse_data["coarse_sdf"].shape) * 4  # Coarse is always float32
    subgrid_mem_source = np.prod(sparse_data["subgrid_data"].shape) * bytes_per_sample
    subgrid_mem_texture = np.prod(sparse_data["subgrid_data"].shape) * 4  # GPU texture is float32
    slots_mem = len(sparse_data["subgrid_start_slots"]) * 4
    dense_mem = dense_x * dense_y * dense_z * 4
    texture_total_mem = coarse_mem + subgrid_mem_texture + slots_mem

    # Get volume memory usage
    sparse_volume_mem = sparse_volume.get_grid_info().size_in_bytes
    coarse_volume_mem = coarse_volume.get_grid_info().size_in_bytes
    volume_total_mem = sparse_volume_mem + coarse_volume_mem

    print("\nMemory Usage:")
    print("\n  NanoVDB Volumes:")
    print(f"    Sparse volume: {sparse_volume_mem / 1024:.1f} KB")
    print(f"    Coarse volume: {coarse_volume_mem / 1024:.1f} KB")
    print(f"    Total: {volume_total_mem / 1024:.1f} KB")

    print("\n  Texture SDF:")
    print(f"    Dense SDF (temporary): {dense_mem / 1024:.1f} KB")
    print(f"    Coarse texture: {coarse_mem / 1024:.1f} KB")
    print(f"    Subgrid texture (GPU): {subgrid_mem_texture / 1024:.1f} KB")
    if QUANTIZATION_MODE != QuantizationMode.FLOAT32:
        print(f"    Subgrid data (quantized): {subgrid_mem_source / 1024:.1f} KB ({bytes_per_sample} bytes/sample)")
    print(f"    Indirection slots: {slots_mem / 1024:.1f} KB")
    print(f"    Total (persistent): {texture_total_mem / 1024:.1f} KB")

    # Memory ratio comparison
    if texture_total_mem > 0 and volume_total_mem > 0:
        memory_ratio = volume_total_mem / texture_total_mem
        print(f"\n  Memory Ratio (Volume / Texture): {memory_ratio:.2f}x")
        if memory_ratio > 1:
            print(f"  Texture SDF uses {(1 - 1 / memory_ratio) * 100:.1f}% less memory")
        else:
            print(f"  Volume SDF uses {(1 - memory_ratio) * 100:.1f}% less memory")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_benchmark()
