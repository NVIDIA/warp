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
    QuantizationMode,
)


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
    sparse_volume, coarse_volume, vol_min_ext, vol_max_ext, vol_spacing = create_volume_from_mesh(
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


def sample_both_sdfs(env: dict, query_points: np.ndarray):
    """Sample both SDF representations at the given points."""
    device = env["device"]
    n_points = len(query_points)

    query_points_wp = wp.array(query_points.astype(np.float32), dtype=wp.vec3, device=device)
    results_volume = wp.zeros(n_points, dtype=float, device=device)
    results_texture = wp.zeros(n_points, dtype=float, device=device)

    # Sample Volume SDF
    wp.launch(
        sample_volume_with_fallback,
        dim=n_points,
        inputs=[
            env["sparse_volume"].id,
            env["coarse_volume"].id,
            BACKGROUND_VALUE,
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

    return results_volume.numpy(), results_texture.numpy()


# ============================================================================
# Test Cases
# ============================================================================


def test_sdf_consistency_random_points():
    """Test that both SDF implementations return similar values for random points."""
    print("\n" + "=" * 70)
    print("Test: SDF Consistency - Random Points")
    print("=" * 70)

    env = setup_test_environment()

    # Generate random test points within the SDF domain
    rng = np.random.default_rng(42)
    query_points = rng.uniform(
        low=env["min_ext"],
        high=env["max_ext"],
        size=(NUM_TEST_POINTS, 3),
    )

    vol_results, tex_results = sample_both_sdfs(env, query_points)

    # Compute differences
    diff = np.abs(vol_results - tex_results)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    median_diff = np.median(diff)

    print(f"\nResults for {NUM_TEST_POINTS} random points:")
    print(f"  Volume SDF:  min={vol_results.min():.4f}, max={vol_results.max():.4f}, mean={vol_results.mean():.4f}")
    print(f"  Texture SDF: min={tex_results.min():.4f}, max={tex_results.max():.4f}, mean={tex_results.mean():.4f}")
    print(f"\nDifferences:")
    print(f"  Max diff:    {max_diff:.6f} (threshold: {TOLERANCE_MAX})")
    print(f"  Mean diff:   {mean_diff:.6f} (threshold: {TOLERANCE_MEAN})")
    print(f"  Median diff: {median_diff:.6f}")

    # Assertions
    assert max_diff < TOLERANCE_MAX, f"Max difference {max_diff:.6f} exceeds tolerance {TOLERANCE_MAX}"
    assert mean_diff < TOLERANCE_MEAN, f"Mean difference {mean_diff:.6f} exceeds tolerance {TOLERANCE_MEAN}"

    print("\n[PASS] Random points test passed!")
    return True


def test_sdf_consistency_surface_points():
    """Test consistency near the mesh surface where precision matters most."""
    print("\n" + "=" * 70)
    print("Test: SDF Consistency - Near-Surface Points")
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

    vol_results, tex_results = sample_both_sdfs(env, query_points)

    # Compute differences
    diff = np.abs(vol_results - tex_results)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\nResults for {len(query_points)} near-surface points:")
    print(f"  Volume SDF:  min={vol_results.min():.4f}, max={vol_results.max():.4f}, mean={vol_results.mean():.4f}")
    print(f"  Texture SDF: min={tex_results.min():.4f}, max={tex_results.max():.4f}, mean={tex_results.mean():.4f}")
    print(f"\nDifferences:")
    print(f"  Max diff:  {max_diff:.6f} (threshold: {TOLERANCE_MAX})")
    print(f"  Mean diff: {mean_diff:.6f} (threshold: {TOLERANCE_MEAN})")

    assert max_diff < TOLERANCE_MAX, f"Max difference {max_diff:.6f} exceeds tolerance {TOLERANCE_MAX}"
    assert mean_diff < TOLERANCE_MEAN, f"Mean difference {mean_diff:.6f} exceeds tolerance {TOLERANCE_MEAN}"

    print("\n[PASS] Near-surface points test passed!")
    return True


def test_sdf_consistency_grid_points():
    """Test consistency at regular grid points."""
    print("\n" + "=" * 70)
    print("Test: SDF Consistency - Regular Grid Points")
    print("=" * 70)

    env = setup_test_environment()

    # Generate regular grid of test points
    n_per_dim = 20
    x = np.linspace(env["min_ext"][0], env["max_ext"][0], n_per_dim)
    y = np.linspace(env["min_ext"][1], env["max_ext"][1], n_per_dim)
    z = np.linspace(env["min_ext"][2], env["max_ext"][2], n_per_dim)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    query_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    vol_results, tex_results = sample_both_sdfs(env, query_points)

    # Compute differences
    diff = np.abs(vol_results - tex_results)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\nResults for {len(query_points)} grid points ({n_per_dim}^3):")
    print(f"  Volume SDF:  min={vol_results.min():.4f}, max={vol_results.max():.4f}, mean={vol_results.mean():.4f}")
    print(f"  Texture SDF: min={tex_results.min():.4f}, max={tex_results.max():.4f}, mean={tex_results.mean():.4f}")
    print(f"\nDifferences:")
    print(f"  Max diff:  {max_diff:.6f} (threshold: {TOLERANCE_MAX})")
    print(f"  Mean diff: {mean_diff:.6f} (threshold: {TOLERANCE_MEAN})")

    assert max_diff < TOLERANCE_MAX, f"Max difference {max_diff:.6f} exceeds tolerance {TOLERANCE_MAX}"
    assert mean_diff < TOLERANCE_MEAN, f"Mean difference {mean_diff:.6f} exceeds tolerance {TOLERANCE_MEAN}"

    print("\n[PASS] Grid points test passed!")
    return True


def test_sdf_consistency_inside_outside():
    """Test that both SDFs agree on inside/outside classification."""
    print("\n" + "=" * 70)
    print("Test: SDF Consistency - Inside/Outside Agreement")
    print("=" * 70)

    env = setup_test_environment()

    # Generate random test points
    rng = np.random.default_rng(456)
    query_points = rng.uniform(
        low=env["min_ext"],
        high=env["max_ext"],
        size=(NUM_TEST_POINTS, 3),
    )

    vol_results, tex_results = sample_both_sdfs(env, query_points)

    # Check sign agreement (inside = negative, outside = positive)
    vol_inside = vol_results < 0
    tex_inside = tex_results < 0

    agreement = np.sum(vol_inside == tex_inside) / len(query_points)
    disagreement_count = np.sum(vol_inside != tex_inside)

    print(f"\nInside/Outside classification for {NUM_TEST_POINTS} points:")
    print(f"  Volume: {np.sum(vol_inside)} inside, {np.sum(~vol_inside)} outside")
    print(f"  Texture: {np.sum(tex_inside)} inside, {np.sum(~tex_inside)} outside")
    print(f"  Agreement: {agreement * 100:.2f}% ({disagreement_count} disagreements)")

    # Allow small disagreement near surface (within tolerance of zero)
    # Check disagreements are all near zero
    if disagreement_count > 0:
        disagree_mask = vol_inside != tex_inside
        vol_disagree = np.abs(vol_results[disagree_mask])
        tex_disagree = np.abs(tex_results[disagree_mask])
        max_disagree_dist = max(np.max(vol_disagree), np.max(tex_disagree))
        print(f"  Max distance from surface in disagreements: {max_disagree_dist:.6f}")

        # Disagreements should only occur very close to the surface
        assert max_disagree_dist < TOLERANCE_MAX, (
            f"Inside/outside disagreement at distance {max_disagree_dist:.6f} from surface"
        )

    # At least 99% agreement expected
    assert agreement > 0.99, f"Inside/outside agreement {agreement * 100:.1f}% is below 99%"

    print("\n[PASS] Inside/outside agreement test passed!")
    return True


def test_sdf_consistency_out_of_bounds():
    """Test that both SDFs handle out-of-bounds queries with extrapolation."""
    print("\n" + "=" * 70)
    print("Test: SDF Consistency - Out-of-Bounds Queries")
    print("=" * 70)
    
    env = setup_test_environment()
    
    # Generate test points OUTSIDE the SDF domain
    rng = np.random.default_rng(999)
    
    # Get the full bounds
    full_min = env["full_min_ext"]
    full_max = env["full_max_ext"]
    extent = full_max - full_min
    
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
    
    vol_results, tex_results = sample_both_sdfs(env, query_points)
    
    # For out-of-bounds points, both should return positive values
    # (distance to surface + distance to boundary)
    vol_positive = np.all(vol_results > 0)
    tex_positive = np.all(tex_results > 0)
    
    print(f"\nResults for {len(query_points)} out-of-bounds points:")
    print(f"  Volume SDF:  min={vol_results.min():.4f}, max={vol_results.max():.4f}, mean={vol_results.mean():.4f}")
    print(f"  Texture SDF: min={tex_results.min():.4f}, max={tex_results.max():.4f}, mean={tex_results.mean():.4f}")
    print(f"  All Volume values positive: {vol_positive}")
    print(f"  All Texture values positive: {tex_positive}")
    
    # Compute differences
    diff = np.abs(vol_results - tex_results)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # For out-of-bounds, allow slightly larger tolerance since extrapolation 
    # can differ between implementations
    oob_tolerance_max = TOLERANCE_MAX * 2
    oob_tolerance_mean = TOLERANCE_MEAN * 2
    
    print(f"\nDifferences:")
    print(f"  Max diff:  {max_diff:.6f} (threshold: {oob_tolerance_max})")
    print(f"  Mean diff: {mean_diff:.6f} (threshold: {oob_tolerance_mean})")
    
    # All out-of-bounds points should have positive SDF (outside the mesh)
    assert vol_positive, "Volume SDF returned non-positive values for out-of-bounds points"
    assert tex_positive, "Texture SDF returned non-positive values for out-of-bounds points"
    
    assert max_diff < oob_tolerance_max, f"Max difference {max_diff:.6f} exceeds tolerance {oob_tolerance_max}"
    assert mean_diff < oob_tolerance_mean, f"Mean difference {mean_diff:.6f} exceeds tolerance {oob_tolerance_mean}"
    
    print("\n[PASS] Out-of-bounds test passed!")
    return True


def test_sdf_consistency_quantized():
    """Test that quantized texture SDF still matches volume SDF reasonably."""
    print("\n" + "=" * 70)
    print("Test: SDF Consistency - Quantized (UINT16) vs Volume")
    print("=" * 70)

    device = "cuda:0"
    wp.init()

    # Create mesh
    box_center = (0.5, 0.5, 0.5)
    box_half_extents = (0.25, 0.25, 0.25)
    mesh = create_box_mesh(box_center, box_half_extents, device)

    narrow_band = (-TEST_NARROW_BAND, TEST_NARROW_BAND)

    # Create Volume SDF first to get its bounds
    sparse_volume, coarse_volume, vol_min_ext, vol_max_ext, vol_spacing = create_volume_from_mesh(
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

    results_volume = wp.zeros(NUM_TEST_POINTS, dtype=float, device=device)
    results_texture = wp.zeros(NUM_TEST_POINTS, dtype=float, device=device)

    # Create bounding box vectors for volume sampling
    sdf_lower = wp.vec3(min_ext[0], min_ext[1], min_ext[2])
    sdf_upper = wp.vec3(max_ext[0], max_ext[1], max_ext[2])

    # Sample both
    wp.launch(
        sample_volume_with_fallback,
        dim=NUM_TEST_POINTS,
        inputs=[
            sparse_volume.id,
            coarse_volume.id,
            BACKGROUND_VALUE,
            sdf_lower,
            sdf_upper,
            query_points_wp,
            results_volume,
        ],
        device=device,
    )

    wp.launch(
        sample_sparse_sdf,
        dim=NUM_TEST_POINTS,
        inputs=[sdf_params, coarse_tex, subgrid_tex, subgrid_slots, query_points_wp, results_texture],
        device=device,
    )

    wp.synchronize()

    vol_results = results_volume.numpy()
    tex_results = results_texture.numpy()

    # Compute differences (allow slightly more tolerance for quantization)
    diff = np.abs(vol_results - tex_results)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Quantization tolerance is larger (16-bit has ~0.0015% precision)
    quant_tolerance_max = TOLERANCE_MAX * 1.5
    quant_tolerance_mean = TOLERANCE_MEAN * 1.5

    print(f"\nResults for {NUM_TEST_POINTS} points (UINT16 quantization):")
    print(f"  Volume SDF:  min={vol_results.min():.4f}, max={vol_results.max():.4f}")
    print(f"  Texture SDF: min={tex_results.min():.4f}, max={tex_results.max():.4f}")
    print(f"\nDifferences:")
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
    print(f"\nConfiguration:")
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
    import sys

    success = run_all_tests()
    sys.exit(0 if success else 1)
