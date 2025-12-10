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

"""Unit tests for SDF consistency between Volume-based and Texture-based implementations.

This test ensures that the NanoVDB volume SDF and the texture-based sparse SDF
return consistent signed distance values when configured with matching resolutions.

Run with: python -m pytest test_sdf_consistency.py -v
Or directly: python test_sdf_consistency.py
"""

import sys

import numpy as np
from texture_sdf import (
    QuantizationMode,
    build_dense_sdf,
    build_sparse_sdf_from_dense,
    create_sparse_sdf_textures,
    sample_sparse_sdf,
)
from volume_sdf import (
    create_box_mesh,
    create_volume_from_mesh,
    get_distance_to_mesh,
    sample_volume_with_fallback,
)

import warp as wp

# ============================================================================
# Ground Truth Mesh Query Kernel
# ============================================================================


@wp.kernel
def compute_mesh_distances(
    mesh: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
):
    """Compute ground truth signed distances using mesh query."""
    tid = wp.tid()
    pos = query_points[tid]
    results[tid] = get_distance_to_mesh(mesh, pos, 10000.0)


# ============================================================================
# Test Configuration
# ============================================================================

# Use consistent resolution for both SDF types
# The subgrid size determines the coarse grid resolution:
#   coarse_dims = (resolution - 1) // subgrid_size
# For volume SDF, the coarse grid is always 8x8x8 (one tile)
# To match, we need: (resolution - 1) // subgrid_size + 1 = 8
# So: resolution = 7 * subgrid_size + 1 (e.g., subgrid_size=8 -> resolution=57)
# But we also want resolution divisible by 8 for volume SDF tiles.
# Best approach: use resolution that gives similar coarse grid density.

TEST_RESOLUTION = 64  # Grid resolution (should be divisible by 8)
TEST_SUBGRID_SIZE = 8  # Cells per subgrid (gives ~8x8x8 coarse grid for 64^3)
TEST_NARROW_BAND = 0.15  # Narrow band distance in world units
TEST_MARGIN = 0.2  # Margin around mesh bounds
NUM_TEST_POINTS = 10000  # Number of random test points

# Tolerance for SDF value comparison
# Some difference is expected due to:
# - Different interpolation methods (NanoVDB vs texture trilinear)
# - Slightly different grid alignments
# - Coarse grid resolution differences
TOLERANCE_MAX = 0.05  # Maximum allowed difference
TOLERANCE_MEAN = 0.02  # Mean difference threshold


# ============================================================================
# Test Fixtures
# ============================================================================


def setup_test_environment(device: str = "cuda:0"):
    """Set up test mesh and both SDF representations."""
    wp.init()

    # Create simple box mesh
    box_center = (0.5, 0.5, 0.5)
    box_half_extents = (0.25, 0.25, 0.25)

    mesh = create_box_mesh(box_center, box_half_extents, device)

    narrow_band = (-TEST_NARROW_BAND, TEST_NARROW_BAND)

    # Create Volume SDF first to get its bounds
    sparse_volume, coarse_volume, sparse_max_value, vol_min_ext, vol_max_ext, vol_spacing = create_volume_from_mesh(
        mesh, narrow_band, margin=TEST_MARGIN, max_dims=TEST_RESOLUTION, verbose=False
    )

    # Use the Volume SDF's bounds for the Texture SDF to ensure matching domains
    min_ext = vol_min_ext
    max_ext = vol_max_ext

    # Create Texture SDF with matching bounds
    dense_sdf, dense_x, dense_y, dense_z, cell_size = build_dense_sdf(mesh, min_ext, max_ext, TEST_RESOLUTION, device)

    sparse_data = build_sparse_sdf_from_dense(
        dense_sdf,
        dense_x,
        dense_y,
        dense_z,
        cell_size,
        min_ext,
        max_ext,
        subgrid_size=TEST_SUBGRID_SIZE,
        narrow_band_thickness=TEST_NARROW_BAND,
        quantization_mode=QuantizationMode.FLOAT32,
        device=device,
    )

    coarse_tex, subgrid_tex, subgrid_slots, sdf_params = create_sparse_sdf_textures(sparse_data, device)

    wp.synchronize()

    # Use slightly inset bounds for queries to avoid edge interpolation issues
    query_margin = np.max(vol_spacing) * 2  # Stay 2 voxels inside the boundary
    query_min = min_ext + query_margin
    query_max = max_ext - query_margin

    # Create bounding box vectors for volume sampling
    sdf_lower = wp.vec3(min_ext[0], min_ext[1], min_ext[2])
    sdf_upper = wp.vec3(max_ext[0], max_ext[1], max_ext[2])

    return {
        "mesh": mesh,
        "min_ext": query_min,  # Use inset bounds for queries
        "max_ext": query_max,
        "full_min_ext": min_ext,  # Full bounds for reference
        "full_max_ext": max_ext,
        "sdf_lower": sdf_lower,  # For volume sampling
        "sdf_upper": sdf_upper,
        "sparse_threshold": sparse_max_value * 1.5,  # Precomputed threshold for fallback
        "device": device,
        # Volume SDF
        "sparse_volume": sparse_volume,
        "coarse_volume": coarse_volume,
        # Texture SDF
        "coarse_tex": coarse_tex,
        "subgrid_tex": subgrid_tex,
        "subgrid_slots": subgrid_slots,
        "sdf_params": sdf_params,
    }


def sample_all_sdfs(env: dict, query_points: np.ndarray):
    """Sample both SDF representations and ground truth mesh query at the given points."""
    device = env["device"]
    n_points = len(query_points)

    query_points_wp = wp.array(query_points.astype(np.float32), dtype=wp.vec3, device=device)
    results_volume = wp.zeros(n_points, dtype=float, device=device)
    results_texture = wp.zeros(n_points, dtype=float, device=device)
    results_ground_truth = wp.zeros(n_points, dtype=float, device=device)

    # Sample Ground Truth (mesh query)
    wp.launch(
        compute_mesh_distances,
        dim=n_points,
        inputs=[env["mesh"].id, query_points_wp, results_ground_truth],
        device=device,
    )

    # Sample Volume SDF
    wp.launch(
        sample_volume_with_fallback,
        dim=n_points,
        inputs=[
            env["sparse_volume"].id,
            env["coarse_volume"].id,
            env["sparse_threshold"],
            env["sdf_lower"],
            env["sdf_upper"],
            query_points_wp,
            results_volume,
        ],
        device=device,
    )

    # Sample Texture SDF
    wp.launch(
        sample_sparse_sdf,
        dim=n_points,
        inputs=[
            env["sdf_params"],
            env["coarse_tex"],
            env["subgrid_tex"],
            env["subgrid_slots"],
            query_points_wp,
            results_texture,
        ],
        device=device,
    )

    wp.synchronize()

    return results_ground_truth.numpy(), results_volume.numpy(), results_texture.numpy()


# ============================================================================
# Test Cases
# ============================================================================


def test_sdf_consistency_random_points():
    """Test that both SDF implementations match ground truth mesh query for random points."""
    print("\n" + "=" * 70)
    print("Test: SDF Accuracy - Random Points vs Ground Truth")
    print("=" * 70)

    env = setup_test_environment()

    # Generate random test points within the SDF domain
    rng = np.random.default_rng(42)
    query_points = rng.uniform(
        low=env["min_ext"],
        high=env["max_ext"],
        size=(NUM_TEST_POINTS, 3),
    )

    gt_results, vol_results, tex_results = sample_all_sdfs(env, query_points)

    # Compute differences vs ground truth
    vol_diff = np.abs(vol_results - gt_results)
    tex_diff = np.abs(tex_results - gt_results)

    vol_max_diff = np.max(vol_diff)
    vol_mean_diff = np.mean(vol_diff)
    tex_max_diff = np.max(tex_diff)
    tex_mean_diff = np.mean(tex_diff)

    print(f"\nResults for {NUM_TEST_POINTS} random points:")
    print(f"  Ground Truth: min={gt_results.min():.4f}, max={gt_results.max():.4f}, mean={gt_results.mean():.4f}")
    print(f"  Volume SDF:   min={vol_results.min():.4f}, max={vol_results.max():.4f}, mean={vol_results.mean():.4f}")
    print(f"  Texture SDF:  min={tex_results.min():.4f}, max={tex_results.max():.4f}, mean={tex_results.mean():.4f}")
    print("\nVolume SDF vs Ground Truth:")
    print(f"  Max diff:  {vol_max_diff:.6f} (threshold: {TOLERANCE_MAX})")
    print(f"  Mean diff: {vol_mean_diff:.6f} (threshold: {TOLERANCE_MEAN})")
    print("\nTexture SDF vs Ground Truth:")
    print(f"  Max diff:  {tex_max_diff:.6f} (threshold: {TOLERANCE_MAX})")
    print(f"  Mean diff: {tex_mean_diff:.6f} (threshold: {TOLERANCE_MEAN})")

    # Assertions - both SDFs should match ground truth
    assert vol_max_diff < TOLERANCE_MAX, f"Volume max difference {vol_max_diff:.6f} exceeds tolerance {TOLERANCE_MAX}"
    assert vol_mean_diff < TOLERANCE_MEAN, (
        f"Volume mean difference {vol_mean_diff:.6f} exceeds tolerance {TOLERANCE_MEAN}"
    )
    assert tex_max_diff < TOLERANCE_MAX, f"Texture max difference {tex_max_diff:.6f} exceeds tolerance {TOLERANCE_MAX}"
    assert tex_mean_diff < TOLERANCE_MEAN, (
        f"Texture mean difference {tex_mean_diff:.6f} exceeds tolerance {TOLERANCE_MEAN}"
    )

    print("\n[PASS] Random points test passed!")
    return True


def test_sdf_consistency_surface_points():
    """Test accuracy near the mesh surface where precision matters most."""
    print("\n" + "=" * 70)
    print("Test: SDF Accuracy - Near-Surface Points vs Ground Truth")
    print("=" * 70)

    env = setup_test_environment()

    # Generate points specifically near the surface (within narrow band)
    rng = np.random.default_rng(123)

    # Box surface is at x,y,z = 0.25 to 0.75 (center 0.5, half-extent 0.25)
    # Generate points that are likely near the surface
    n_per_face = NUM_TEST_POINTS // 6
    surface_points = []

    # Near each face of the box
    for axis in range(3):
        for sign in [-1, 1]:
            pts = rng.uniform(
                low=env["min_ext"],
                high=env["max_ext"],
                size=(n_per_face, 3),
            )
            # Place points near the surface along this axis
            surface_coord = 0.5 + sign * 0.25  # Box surface location
            pts[:, axis] = surface_coord + rng.uniform(-0.1, 0.1, n_per_face)
            surface_points.append(pts)

    query_points = np.vstack(surface_points)

    gt_results, vol_results, tex_results = sample_all_sdfs(env, query_points)

    # Compute differences vs ground truth
    vol_diff = np.abs(vol_results - gt_results)
    tex_diff = np.abs(tex_results - gt_results)

    vol_max_diff = np.max(vol_diff)
    vol_mean_diff = np.mean(vol_diff)
    tex_max_diff = np.max(tex_diff)
    tex_mean_diff = np.mean(tex_diff)

    print(f"\nResults for {len(query_points)} near-surface points:")
    print(f"  Ground Truth: min={gt_results.min():.4f}, max={gt_results.max():.4f}")
    print(f"  Volume SDF:   min={vol_results.min():.4f}, max={vol_results.max():.4f}")
    print(f"  Texture SDF:  min={tex_results.min():.4f}, max={tex_results.max():.4f}")
    print("\nVolume SDF vs Ground Truth:")
    print(f"  Max diff:  {vol_max_diff:.6f} (threshold: {TOLERANCE_MAX})")
    print(f"  Mean diff: {vol_mean_diff:.6f} (threshold: {TOLERANCE_MEAN})")
    print("\nTexture SDF vs Ground Truth:")
    print(f"  Max diff:  {tex_max_diff:.6f} (threshold: {TOLERANCE_MAX})")
    print(f"  Mean diff: {tex_mean_diff:.6f} (threshold: {TOLERANCE_MEAN})")

    assert vol_max_diff < TOLERANCE_MAX, f"Volume max difference {vol_max_diff:.6f} exceeds tolerance {TOLERANCE_MAX}"
    assert vol_mean_diff < TOLERANCE_MEAN, (
        f"Volume mean difference {vol_mean_diff:.6f} exceeds tolerance {TOLERANCE_MEAN}"
    )
    assert tex_max_diff < TOLERANCE_MAX, f"Texture max difference {tex_max_diff:.6f} exceeds tolerance {TOLERANCE_MAX}"
    assert tex_mean_diff < TOLERANCE_MEAN, (
        f"Texture mean difference {tex_mean_diff:.6f} exceeds tolerance {TOLERANCE_MEAN}"
    )

    print("\n[PASS] Near-surface points test passed!")
    return True


def test_sdf_consistency_grid_points():
    """Test accuracy at regular grid points."""
    print("\n" + "=" * 70)
    print("Test: SDF Accuracy - Regular Grid Points vs Ground Truth")
    print("=" * 70)

    env = setup_test_environment()

    # Generate regular grid of test points
    n_per_dim = 20
    x = np.linspace(env["min_ext"][0], env["max_ext"][0], n_per_dim)
    y = np.linspace(env["min_ext"][1], env["max_ext"][1], n_per_dim)
    z = np.linspace(env["min_ext"][2], env["max_ext"][2], n_per_dim)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    query_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    gt_results, vol_results, tex_results = sample_all_sdfs(env, query_points)

    # Compute differences vs ground truth
    vol_diff = np.abs(vol_results - gt_results)
    tex_diff = np.abs(tex_results - gt_results)

    vol_max_diff = np.max(vol_diff)
    vol_mean_diff = np.mean(vol_diff)
    tex_max_diff = np.max(tex_diff)
    tex_mean_diff = np.mean(tex_diff)

    print(f"\nResults for {len(query_points)} grid points ({n_per_dim}^3):")
    print(f"  Ground Truth: min={gt_results.min():.4f}, max={gt_results.max():.4f}")
    print(f"  Volume SDF:   min={vol_results.min():.4f}, max={vol_results.max():.4f}")
    print(f"  Texture SDF:  min={tex_results.min():.4f}, max={tex_results.max():.4f}")
    print("\nVolume SDF vs Ground Truth:")
    print(f"  Max diff:  {vol_max_diff:.6f} (threshold: {TOLERANCE_MAX})")
    print(f"  Mean diff: {vol_mean_diff:.6f} (threshold: {TOLERANCE_MEAN})")
    print("\nTexture SDF vs Ground Truth:")
    print(f"  Max diff:  {tex_max_diff:.6f} (threshold: {TOLERANCE_MAX})")
    print(f"  Mean diff: {tex_mean_diff:.6f} (threshold: {TOLERANCE_MEAN})")

    assert vol_max_diff < TOLERANCE_MAX, f"Volume max difference {vol_max_diff:.6f} exceeds tolerance {TOLERANCE_MAX}"
    assert vol_mean_diff < TOLERANCE_MEAN, (
        f"Volume mean difference {vol_mean_diff:.6f} exceeds tolerance {TOLERANCE_MEAN}"
    )
    assert tex_max_diff < TOLERANCE_MAX, f"Texture max difference {tex_max_diff:.6f} exceeds tolerance {TOLERANCE_MAX}"
    assert tex_mean_diff < TOLERANCE_MEAN, (
        f"Texture mean difference {tex_mean_diff:.6f} exceeds tolerance {TOLERANCE_MEAN}"
    )

    print("\n[PASS] Grid points test passed!")
    return True


def test_sdf_consistency_inside_outside():
    """Test that both SDFs agree with ground truth on inside/outside classification."""
    print("\n" + "=" * 70)
    print("Test: SDF Accuracy - Inside/Outside Agreement vs Ground Truth")
    print("=" * 70)

    env = setup_test_environment()

    # Generate random test points
    rng = np.random.default_rng(456)
    query_points = rng.uniform(
        low=env["min_ext"],
        high=env["max_ext"],
        size=(NUM_TEST_POINTS, 3),
    )

    gt_results, vol_results, tex_results = sample_all_sdfs(env, query_points)

    # Check sign agreement with ground truth (inside = negative, outside = positive)
    gt_inside = gt_results < 0
    vol_inside = vol_results < 0
    tex_inside = tex_results < 0

    vol_agreement = np.sum(vol_inside == gt_inside) / len(query_points)
    tex_agreement = np.sum(tex_inside == gt_inside) / len(query_points)
    vol_disagreement = np.sum(vol_inside != gt_inside)
    tex_disagreement = np.sum(tex_inside != gt_inside)

    print(f"\nInside/Outside classification for {NUM_TEST_POINTS} points:")
    print(f"  Ground Truth: {np.sum(gt_inside)} inside, {np.sum(~gt_inside)} outside")
    print(f"  Volume SDF:   {np.sum(vol_inside)} inside, {np.sum(~vol_inside)} outside")
    print(f"  Texture SDF:  {np.sum(tex_inside)} inside, {np.sum(~tex_inside)} outside")
    print(f"\nVolume vs Ground Truth: {vol_agreement * 100:.2f}% ({vol_disagreement} disagreements)")
    print(f"Texture vs Ground Truth: {tex_agreement * 100:.2f}% ({tex_disagreement} disagreements)")

    # Check disagreements are all near zero (near surface)
    if vol_disagreement > 0:
        disagree_mask = vol_inside != gt_inside
        max_disagree_dist = np.max(np.abs(gt_results[disagree_mask]))
        print(f"  Volume max distance from surface in disagreements: {max_disagree_dist:.6f}")
        assert max_disagree_dist < TOLERANCE_MAX, (
            f"Volume inside/outside disagreement at distance {max_disagree_dist:.6f} from surface"
        )

    if tex_disagreement > 0:
        disagree_mask = tex_inside != gt_inside
        max_disagree_dist = np.max(np.abs(gt_results[disagree_mask]))
        print(f"  Texture max distance from surface in disagreements: {max_disagree_dist:.6f}")
        assert max_disagree_dist < TOLERANCE_MAX, (
            f"Texture inside/outside disagreement at distance {max_disagree_dist:.6f} from surface"
        )

    # At least 99% agreement expected
    assert vol_agreement > 0.99, f"Volume inside/outside agreement {vol_agreement * 100:.1f}% is below 99%"
    assert tex_agreement > 0.99, f"Texture inside/outside agreement {tex_agreement * 100:.1f}% is below 99%"

    print("\n[PASS] Inside/outside agreement test passed!")
    return True


def test_sdf_consistency_out_of_bounds():
    """Test that both SDFs handle out-of-bounds queries with extrapolation vs ground truth."""
    print("\n" + "=" * 70)
    print("Test: SDF Accuracy - Out-of-Bounds Queries vs Ground Truth")
    print("=" * 70)

    env = setup_test_environment()

    # Generate test points OUTSIDE the SDF domain
    rng = np.random.default_rng(999)

    # Get the full bounds
    full_min = env["full_min_ext"]
    full_max = env["full_max_ext"]

    # Generate points outside in various directions
    n_per_direction = NUM_TEST_POINTS // 6
    out_of_bounds_points = []

    for axis in range(3):
        for sign in [-1, 1]:
            pts = rng.uniform(low=full_min, high=full_max, size=(n_per_direction, 3))
            # Push points outside along this axis
            if sign < 0:
                pts[:, axis] = full_min[axis] - rng.uniform(0.1, 0.5, n_per_direction)
            else:
                pts[:, axis] = full_max[axis] + rng.uniform(0.1, 0.5, n_per_direction)
            out_of_bounds_points.append(pts)

    query_points = np.vstack(out_of_bounds_points)

    gt_results, vol_results, tex_results = sample_all_sdfs(env, query_points)

    # For out-of-bounds points, all should return positive values (outside the mesh)
    gt_positive = np.all(gt_results > 0)
    vol_positive = np.all(vol_results > 0)
    tex_positive = np.all(tex_results > 0)

    print(f"\nResults for {len(query_points)} out-of-bounds points:")
    print(f"  Ground Truth: min={gt_results.min():.4f}, max={gt_results.max():.4f}, mean={gt_results.mean():.4f}")
    print(f"  Volume SDF:   min={vol_results.min():.4f}, max={vol_results.max():.4f}, mean={vol_results.mean():.4f}")
    print(f"  Texture SDF:  min={tex_results.min():.4f}, max={tex_results.max():.4f}, mean={tex_results.mean():.4f}")
    print(f"  All positive: GT={gt_positive}, Volume={vol_positive}, Texture={tex_positive}")

    # Compute differences vs ground truth
    vol_diff = np.abs(vol_results - gt_results)
    tex_diff = np.abs(tex_results - gt_results)

    # For out-of-bounds, allow larger tolerance since extrapolation approximates the true distance
    oob_tolerance_max = TOLERANCE_MAX * 3
    oob_tolerance_mean = TOLERANCE_MEAN * 3

    vol_max_diff = np.max(vol_diff)
    vol_mean_diff = np.mean(vol_diff)
    tex_max_diff = np.max(tex_diff)
    tex_mean_diff = np.mean(tex_diff)

    print("\nVolume SDF vs Ground Truth:")
    print(f"  Max diff:  {vol_max_diff:.6f} (threshold: {oob_tolerance_max})")
    print(f"  Mean diff: {vol_mean_diff:.6f} (threshold: {oob_tolerance_mean})")
    print("\nTexture SDF vs Ground Truth:")
    print(f"  Max diff:  {tex_max_diff:.6f} (threshold: {oob_tolerance_max})")
    print(f"  Mean diff: {tex_mean_diff:.6f} (threshold: {oob_tolerance_mean})")

    # All out-of-bounds points should have positive SDF (outside the mesh)
    assert gt_positive, "Ground truth returned non-positive values for out-of-bounds points"
    assert vol_positive, "Volume SDF returned non-positive values for out-of-bounds points"
    assert tex_positive, "Texture SDF returned non-positive values for out-of-bounds points"

    assert vol_max_diff < oob_tolerance_max, (
        f"Volume max difference {vol_max_diff:.6f} exceeds tolerance {oob_tolerance_max}"
    )
    assert vol_mean_diff < oob_tolerance_mean, (
        f"Volume mean difference {vol_mean_diff:.6f} exceeds tolerance {oob_tolerance_mean}"
    )
    assert tex_max_diff < oob_tolerance_max, (
        f"Texture max difference {tex_max_diff:.6f} exceeds tolerance {oob_tolerance_max}"
    )
    assert tex_mean_diff < oob_tolerance_mean, (
        f"Texture mean difference {tex_mean_diff:.6f} exceeds tolerance {oob_tolerance_mean}"
    )

    print("\n[PASS] Out-of-bounds test passed!")
    return True


def test_sdf_consistency_quantized():
    """Test that quantized texture SDF matches ground truth reasonably."""
    print("\n" + "=" * 70)
    print("Test: SDF Accuracy - Quantized (UINT16) vs Ground Truth")
    print("=" * 70)

    device = "cuda:0"
    wp.init()

    # Create mesh
    box_center = (0.5, 0.5, 0.5)
    box_half_extents = (0.25, 0.25, 0.25)
    mesh = create_box_mesh(box_center, box_half_extents, device)

    narrow_band = (-TEST_NARROW_BAND, TEST_NARROW_BAND)

    # Create Volume SDF first to get its bounds (we don't actually use volume here,
    # but we use the same bounds for consistency)
    _, _, _, vol_min_ext, vol_max_ext, vol_spacing = create_volume_from_mesh(
        mesh, narrow_band, margin=TEST_MARGIN, max_dims=TEST_RESOLUTION, verbose=False
    )

    # Use the Volume SDF's bounds for the Texture SDF
    min_ext = vol_min_ext
    max_ext = vol_max_ext

    # Create Texture SDF with UINT16 quantization
    dense_sdf, dense_x, dense_y, dense_z, cell_size = build_dense_sdf(mesh, min_ext, max_ext, TEST_RESOLUTION, device)

    sparse_data = build_sparse_sdf_from_dense(
        dense_sdf,
        dense_x,
        dense_y,
        dense_z,
        cell_size,
        min_ext,
        max_ext,
        subgrid_size=TEST_SUBGRID_SIZE,
        narrow_band_thickness=TEST_NARROW_BAND,
        quantization_mode=QuantizationMode.UINT16,
        device=device,
    )

    coarse_tex, subgrid_tex, subgrid_slots, sdf_params = create_sparse_sdf_textures(sparse_data, device)

    wp.synchronize()

    # Generate test points with inset bounds to avoid edge issues
    query_margin = np.max(vol_spacing) * 2
    query_min = min_ext + query_margin
    query_max = max_ext - query_margin

    rng = np.random.default_rng(789)
    query_points = rng.uniform(low=query_min, high=query_max, size=(NUM_TEST_POINTS, 3))
    query_points_wp = wp.array(query_points.astype(np.float32), dtype=wp.vec3, device=device)

    results_ground_truth = wp.zeros(NUM_TEST_POINTS, dtype=float, device=device)
    results_texture = wp.zeros(NUM_TEST_POINTS, dtype=float, device=device)

    # Sample ground truth (mesh query)
    wp.launch(
        compute_mesh_distances,
        dim=NUM_TEST_POINTS,
        inputs=[mesh.id, query_points_wp, results_ground_truth],
        device=device,
    )

    # Sample quantized texture SDF
    wp.launch(
        sample_sparse_sdf,
        dim=NUM_TEST_POINTS,
        inputs=[sdf_params, coarse_tex, subgrid_tex, subgrid_slots, query_points_wp, results_texture],
        device=device,
    )

    wp.synchronize()

    gt_results = results_ground_truth.numpy()
    tex_results = results_texture.numpy()

    # Compute differences vs ground truth
    diff = np.abs(tex_results - gt_results)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Quantization tolerance is larger (16-bit has ~0.0015% precision)
    quant_tolerance_max = TOLERANCE_MAX * 1.5
    quant_tolerance_mean = TOLERANCE_MEAN * 1.5

    print(f"\nResults for {NUM_TEST_POINTS} points (UINT16 quantization):")
    print(f"  Ground Truth:  min={gt_results.min():.4f}, max={gt_results.max():.4f}")
    print(f"  Texture SDF:   min={tex_results.min():.4f}, max={tex_results.max():.4f}")
    print("\nTexture SDF (UINT16) vs Ground Truth:")
    print(f"  Max diff:  {max_diff:.6f} (threshold: {quant_tolerance_max})")
    print(f"  Mean diff: {mean_diff:.6f} (threshold: {quant_tolerance_mean})")

    assert max_diff < quant_tolerance_max, f"Max difference {max_diff:.6f} exceeds tolerance {quant_tolerance_max}"
    assert mean_diff < quant_tolerance_mean, f"Mean difference {mean_diff:.6f} exceeds tolerance {quant_tolerance_mean}"

    print("\n[PASS] Quantized SDF test passed!")
    return True


# ============================================================================
# Main Entry Point
# ============================================================================


def run_all_tests():
    """Run all consistency tests."""
    print("\n" + "=" * 70)
    print("SDF CONSISTENCY TEST SUITE")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Resolution: {TEST_RESOLUTION}")
    print(f"  Subgrid Size: {TEST_SUBGRID_SIZE}")
    print(f"  Narrow Band: +/-{TEST_NARROW_BAND}")
    print(f"  Test Points: {NUM_TEST_POINTS}")
    print(f"  Tolerance (max): {TOLERANCE_MAX}")
    print(f"  Tolerance (mean): {TOLERANCE_MEAN}")

    tests = [
        ("Random Points", test_sdf_consistency_random_points),
        ("Near-Surface Points", test_sdf_consistency_surface_points),
        ("Grid Points", test_sdf_consistency_grid_points),
        ("Inside/Outside Agreement", test_sdf_consistency_inside_outside),
        ("Out-of-Bounds Queries", test_sdf_consistency_out_of_bounds),
        ("Quantized (UINT16)", test_sdf_consistency_quantized),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, None))
        except AssertionError as e:
            results.append((name, False, str(e)))
            print(f"\n[FAIL] {name}: {e}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n[ERROR] {name}: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")
        if error:
            print(f"         {error}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
