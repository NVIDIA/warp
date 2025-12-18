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

import os
import time
from statistics import mean, stdev

import numpy as np
from texture_sdf import (
    QuantizationMode,
    create_sparse_sdf_from_mesh,
    get_quantization_bytes,
    sample_sparse_sdf,
    sample_sparse_sdf_grad,
)
from volume_sdf import (
    create_sphere_mesh,
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
MARGIN_FACTOR = 0.01  # Margin as fraction of AABB diagonal (1%)
NARROW_BAND_FACTOR = 0.01  # Narrow band as fraction of AABB diagonal (1%)
ITERATIONS = 100
WARM_UP = 10

# Quantization mode for subgrid texture data:
#   QuantizationMode.FLOAT32 - Full precision (4 bytes per sample)
#   QuantizationMode.UINT16  - 16-bit compression (2 bytes per sample)
#   QuantizationMode.UINT8   - 8-bit compression (1 byte per sample)
QUANTIZATION_MODE = QuantizationMode.UINT16

# Optional: Path to external mesh file (OBJ format)
# Set to None to use the default box mesh
# Example: MESH_FILE = r"C:\Documents\Meshes\RobotArmNanoVDB\Robot_Arm_Input.obj"
MESH_FILE = None
# MESH_FILE = r"C:\Documents\Meshes\RobotArmNanoVDB\Robot_Arm_Input.obj"


# ============================================================================
# Mesh Loading Utilities
# ============================================================================


def load_obj_mesh(filepath: str, device: str) -> wp.Mesh:
    """
    Load a mesh from an OBJ file.

    Args:
        filepath: Path to the OBJ file
        device: Warp device string

    Returns:
        wp.Mesh with support_winding_number=True
    """
    vertices = []
    faces = []

    with open(filepath) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == "v":
                # Vertex position
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # Face - handle various formats: v, v/vt, v/vt/vn, v//vn
                face_verts = []
                for part in parts[1:]:
                    # Extract vertex index (before any /)
                    v_idx = int(part.split("/")[0])
                    # OBJ indices are 1-based
                    face_verts.append(v_idx - 1)

                # Triangulate faces with more than 3 vertices (fan triangulation)
                for i in range(1, len(face_verts) - 1):
                    faces.extend([face_verts[0], face_verts[i], face_verts[i + 1]])

    vertices_np = np.array(vertices, dtype=np.float32)
    indices_np = np.array(faces, dtype=np.int32)

    print(f"  Loaded mesh: {len(vertices_np)} vertices, {len(indices_np) // 3} triangles")

    points = wp.array(vertices_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np, dtype=int, device=device)

    return wp.Mesh(points=points, indices=indices, support_winding_number=True)


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

    # Map quantization mode to readable name
    quant_names = {
        QuantizationMode.FLOAT32: "FLOAT32 (4 bytes)",
        QuantizationMode.UINT16: "UINT16 (2 bytes)",
        QuantizationMode.UINT8: "UINT8 (1 byte)",
    }

    print("\nConfiguration:")
    print(f"  SDF Resolution: {SDF_RESOLUTION}")
    print(f"  Subgrid Size: {SUBGRID_SIZE}")
    print(f"  Margin Factor: {MARGIN_FACTOR * 100:.1f}% of AABB diagonal")
    print(f"  Narrow Band Factor: {NARROW_BAND_FACTOR * 100:.1f}% of AABB diagonal")
    print(f"  Quantization: {quant_names.get(QUANTIZATION_MODE, 'Unknown')}")
    print(f"  Query Points: {NUM_QUERY_POINTS:,}")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Warm-up: {WARM_UP}")
    if MESH_FILE:
        print(f"  Mesh File: {MESH_FILE}")
    else:
        print("  Mesh: Default box mesh")

    # ========== Create Mesh ==========
    if MESH_FILE:
        if not os.path.exists(MESH_FILE):
            raise FileNotFoundError(f"Mesh file not found: {MESH_FILE}")
        print(f"\nLoading mesh from file: {MESH_FILE}")
        mesh = load_obj_mesh(MESH_FILE, device)
    else:
        print("\nCreating sphere mesh...")
        sphere_center = (0.5, 0.5, 0.5)
        sphere_radius = 0.3
        mesh = create_sphere_mesh(sphere_center, sphere_radius, device, subdivisions=32)
        print(f"  Sphere: center={sphere_center}, radius={sphere_radius}")

        # Alternative: Use box mesh instead
        # from volume_sdf import create_box_mesh
        # box_center = (0.5, 0.5, 0.5)
        # box_half_extents = (0.3, 0.3, 0.3)
        # mesh = create_box_mesh(box_center, box_half_extents, device)
        # print(f"  Box: center={box_center}, half_extents={box_half_extents}")

    # ========== Create Volume SDF ==========
    print("\nCreating NanoVDB volume from mesh...")
    sparse_volume, coarse_volume, sparse_max_value, vol_min_ext, vol_max_ext, _, vol_meta = create_volume_from_mesh(
        mesh, margin_factor=MARGIN_FACTOR, narrow_band_factor=NARROW_BAND_FACTOR, max_dims=SDF_RESOLUTION, verbose=True
    )
    print(f"  Volume bounds: [{vol_min_ext}, {vol_max_ext}]")
    print(f"  Computed narrow band: +/-{vol_meta['narrow_band_distance'][1]:.4f} world units")

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
    t0 = time.perf_counter()

    coarse_tex, subgrid_tex, subgrid_slots, sdf_params, tex_meta = create_sparse_sdf_from_mesh(
        mesh,
        margin_factor=MARGIN_FACTOR,
        narrow_band_factor=NARROW_BAND_FACTOR,
        resolution=SDF_RESOLUTION,
        subgrid_size=SUBGRID_SIZE,
        quantization_mode=QUANTIZATION_MODE,
        verbose=True,
    )
    wp.synchronize()
    tex_build_time = time.perf_counter() - t0
    print(f"  Total texture SDF construction: {tex_build_time * 1000:.2f} ms")
    print(f"  Texture bounds: [{tex_meta['min_ext']}, {tex_meta['max_ext']}]")

    # Use volume bounds for query points (should match texture bounds)
    min_ext = vol_min_ext
    max_ext = vol_max_ext

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

    # Compute coarse texture size from sdf_params
    coarse_tex_samples = (
        (sdf_params.coarse_size_x + 1) * (sdf_params.coarse_size_y + 1) * (sdf_params.coarse_size_z + 1)
    )
    coarse_mem = coarse_tex_samples * 4  # float32

    # Compute subgrid texture size from metadata
    num_subgrids = tex_meta["num_subgrids"]
    subgrid_samples = (SUBGRID_SIZE + 1) ** 3
    subgrid_mem_source = num_subgrids * subgrid_samples * bytes_per_sample
    subgrid_mem_texture = num_subgrids * subgrid_samples * 4  # GPU texture is float32

    # Slots memory
    total_coarse_cells = sdf_params.coarse_size_x * sdf_params.coarse_size_y * sdf_params.coarse_size_z
    slots_mem = total_coarse_cells * 4

    # Dense SDF (temporary) - approximate based on resolution
    cell_size = tex_meta["cell_size"]
    extent = tex_meta["max_ext"] - tex_meta["min_ext"]
    dense_dims = (extent / cell_size).astype(int) + 1
    dense_mem = int(np.prod(dense_dims)) * 4

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
    print(f"    Subgrid texture (GPU): {subgrid_mem_texture / 1024:.1f} KB ({num_subgrids} subgrids)")
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
