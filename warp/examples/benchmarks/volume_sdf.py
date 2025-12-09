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

"""Volume-based SDF construction and sampling.

This module provides tools to create sparse NanoVDB volumes from triangle meshes
and sample signed distance fields. The implementation closely follows the patterns
from newton's sdf_utils.py and sdf_contact.py.

Key features:
- Sparse volume allocation using narrow band tiles
- Coarse background volume for far-field queries
- Extrapolated sampling with coarse fallback (matching sdf_contact.py)
"""

import numpy as np

import warp as wp

# Background value for unallocated voxels
BACKGROUND_VALUE = 1000.0


# ============================================================================
# Warp Functions for Mesh Distance Queries
# ============================================================================


@wp.func
def int_to_vec3f(x: wp.int32, y: wp.int32, z: wp.int32):
    """Convert integer coordinates to vec3f."""
    return wp.vec3f(float(x), float(y), float(z))


@wp.func
def get_distance_to_mesh(mesh: wp.uint64, point: wp.vec3, max_dist: wp.float32):
    """
    Compute signed distance from point to mesh surface.

    Uses winding number for inside/outside determination.
    Returns positive values outside, negative inside.
    """
    res = wp.mesh_query_point_sign_winding_number(mesh, point, max_dist)
    if res.result:
        closest = wp.mesh_eval_position(mesh, res.face, res.u, res.v)
        vec_to_surface = closest - point
        return res.sign * wp.length(vec_to_surface)
    return max_dist


# ============================================================================
# Warp Kernels for Volume Construction
# ============================================================================


@wp.kernel
def sdf_from_mesh_kernel(
    mesh: wp.uint64,
    sdf: wp.uint64,
    tile_points: wp.array(dtype=wp.vec3i),
    thickness: wp.float32,
):
    """
    Populate SDF volume from triangle mesh.

    Launch with dim=(num_tiles, 8, 8, 8) to process all voxels in allocated tiles.
    """
    tile_idx, local_x, local_y, local_z = wp.tid()

    tile_origin = tile_points[tile_idx]
    x_id = tile_origin[0] + local_x
    y_id = tile_origin[1] + local_y
    z_id = tile_origin[2] + local_z

    sample_pos = wp.volume_index_to_world(sdf, int_to_vec3f(x_id, y_id, z_id))
    signed_distance = get_distance_to_mesh(mesh, sample_pos, 10000.0)
    signed_distance -= thickness
    wp.volume_store(sdf, x_id, y_id, z_id, signed_distance)


@wp.kernel
def check_tile_occupied_mesh_kernel(
    mesh: wp.uint64,
    tile_points: wp.array(dtype=wp.vec3f),
    threshold: wp.vec2f,
    tile_occupied: wp.array(dtype=bool),
):
    """
    Check which tiles intersect the narrow band around the mesh surface.

    A tile is marked occupied if its center is within the threshold distance
    of the surface.
    """
    tid = wp.tid()
    sample_pos = tile_points[tid]

    signed_distance = get_distance_to_mesh(mesh, sample_pos, 10000.0)
    is_occupied = wp.bool(False)
    if wp.sign(signed_distance) > 0.0:
        is_occupied = signed_distance < threshold[1]
    else:
        is_occupied = signed_distance > threshold[0]
    tile_occupied[tid] = is_occupied


# ============================================================================
# Warp Kernels for Volume Sampling
# ============================================================================


@wp.kernel
def compute_tile_max_kernel(
    sdf: wp.uint64,
    tile_points: wp.array(dtype=wp.vec3i),
    tile_max_values: wp.array(dtype=float),
):
    """
    Compute the maximum SDF value in each tile.
    Launch with dim=(num_tiles, 8, 8, 8).
    """
    tile_idx, local_x, local_y, local_z = wp.tid()

    tile_origin = tile_points[tile_idx]
    x_id = tile_origin[0] + local_x
    y_id = tile_origin[1] + local_y
    z_id = tile_origin[2] + local_z

    # Read the SDF value at this voxel
    sdf_value = wp.volume_lookup_f(sdf, x_id, y_id, z_id)

    # Atomic max to find the maximum value in this tile
    wp.atomic_max(tile_max_values, tile_idx, sdf_value)


@wp.kernel
def sample_volume_with_fallback(
    sparse_volume: wp.uint64,
    coarse_volume: wp.uint64,
    sparse_max_value: float,
    sdf_lower: wp.vec3,
    sdf_upper: wp.vec3,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
):
    """
    Sample SDF with coarse fallback and boundary extrapolation.

    Matches the pattern from sdf_contact.py's sample_sdf_extrapolated:
    1. Point inside extent, in narrow band: Returns sparse grid value
    2. Point inside extent, outside narrow band: Returns coarse grid value
    3. Point outside extent: Returns SDF(clamped_boundary) + distance_to_boundary

    The sparse_max_value is the maximum SDF value stored in any sparse tile.
    If we sample a value greater than this (plus margin), it must be contaminated
    with background due to tile boundary interpolation, so we fall back to coarse.

    This ensures the SDF can be evaluated at ANY point in space with reasonable values.
    """
    tid = wp.tid()
    pos = query_points[tid]

    # Check if point is inside extent
    inside_extent = (
        pos[0] >= sdf_lower[0]
        and pos[0] <= sdf_upper[0]
        and pos[1] >= sdf_lower[1]
        and pos[1] <= sdf_upper[1]
        and pos[2] >= sdf_lower[2]
        and pos[2] <= sdf_upper[2]
    )

    if inside_extent:
        # Sample sparse grid
        sparse_idx = wp.volume_world_to_index(sparse_volume, pos)
        sparse_dist = wp.volume_sample_f(sparse_volume, sparse_idx, wp.Volume.LINEAR)

        # If sparse returns a value > max stored value + margin, it's contaminated
        # with background (1000) due to tile boundary interpolation. Use coarse.
        # Add 50% margin to account for interpolation between valid values
        threshold = sparse_max_value * 1.5

        if sparse_dist <= threshold:
            results[tid] = sparse_dist
        else:
            # Fallback to coarse grid
            coarse_idx = wp.volume_world_to_index(coarse_volume, pos)
            results[tid] = wp.volume_sample_f(coarse_volume, coarse_idx, wp.Volume.LINEAR)
    else:
        # Point is outside extent - project to boundary
        clamped_pos = wp.vec3(
            wp.clamp(pos[0], sdf_lower[0], sdf_upper[0]),
            wp.clamp(pos[1], sdf_lower[1], sdf_upper[1]),
            wp.clamp(pos[2], sdf_lower[2], sdf_upper[2]),
        )
        dist_to_boundary = wp.length(pos - clamped_pos)

        # Sample at the boundary point using coarse grid (more reliable for extrapolation)
        coarse_idx = wp.volume_world_to_index(coarse_volume, clamped_pos)
        boundary_dist = wp.volume_sample_f(coarse_volume, coarse_idx, wp.Volume.LINEAR)

        # Extrapolate: value at boundary + distance to boundary
        results[tid] = boundary_dist + dist_to_boundary


@wp.kernel
def sample_volume_simple(
    volume: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
):
    """Sample SDF from a single volume (no fallback)."""
    tid = wp.tid()
    pos = query_points[tid]
    idx = wp.volume_world_to_index(volume, pos)
    results[tid] = wp.volume_sample_f(volume, idx, wp.Volume.LINEAR)


# ============================================================================
# Host-side Volume Construction
# ============================================================================


def create_volume_from_mesh(
    mesh: wp.Mesh,
    narrow_band_distance: tuple[float, float],
    margin: float = 0.2,
    max_dims: int = 64,
    verbose: bool = False,
) -> tuple[wp.Volume, wp.Volume, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sparse and coarse SDF volumes from a mesh.

    This follows the pattern from newton's sdf_utils.py:
    - Sparse volume: High-resolution tiles allocated only in the narrow band
    - Coarse volume: Low-resolution (8x8x8) covering the entire extent

    Args:
        mesh: wp.Mesh with support_winding_number=True
        narrow_band_distance: Tuple of (inner, outer) distances, e.g., (-0.1, 0.1)
        margin: Margin to add around mesh bounding box
        max_dims: Maximum grid dimension
        verbose: Print debug info

    Returns:
        Tuple of (sparse_volume, coarse_volume, sparse_max_value, min_ext, max_ext, voxel_size)
        - sparse_max_value: Maximum SDF value stored in any sparse tile (for threshold)
    """
    device = mesh.device

    # Compute mesh bounds
    points_np = mesh.points.numpy()
    min_ext = np.min(points_np, axis=0) - margin
    max_ext = np.max(points_np, axis=0) + margin
    ext = max_ext - min_ext

    # Calculate voxel size
    max_extent = np.max(ext)
    target_voxel_size = max_extent / max_dims
    grid_tile_nums = (ext / target_voxel_size).astype(int) // 8
    grid_tile_nums = np.maximum(grid_tile_nums, 1)
    grid_dims = grid_tile_nums * 8
    actual_voxel_size = ext / (grid_dims - 1)

    if verbose:
        print(f"  Extent: {ext}")
        print(f"  Grid dims: {grid_dims}")
        print(f"  Voxel size: {actual_voxel_size}")

    # Generate all potential tiles
    tile_max = np.around((max_ext - min_ext) / actual_voxel_size).astype(np.int32) // 8
    tiles = np.array(
        [[i, j, k] for i in range(tile_max[0] + 1) for j in range(tile_max[1] + 1) for k in range(tile_max[2] + 1)],
        dtype=np.int32,
    )
    tile_points = tiles * 8

    # Compute tile centers in world space
    tile_center_points_world = (tile_points + 4) * actual_voxel_size + min_ext
    tile_center_points_world = wp.array(tile_center_points_world, dtype=wp.vec3f, device=device)
    tile_occupied = wp.zeros(len(tile_points), dtype=bool, device=device)

    # Check which tiles intersect narrow band
    tile_radius = np.linalg.norm(4 * actual_voxel_size)
    threshold = wp.vec2f(narrow_band_distance[0] - tile_radius, narrow_band_distance[1] + tile_radius)

    wp.launch(
        check_tile_occupied_mesh_kernel,
        dim=len(tile_points),
        inputs=[mesh.id, tile_center_points_world, threshold],
        outputs=[tile_occupied],
        device=device,
    )

    if verbose:
        occupancy = tile_occupied.numpy().sum() / len(tile_points)
        print(f"  Tile occupancy: {occupancy:.2%}")

    # Filter to occupied tiles
    tile_points = tile_points[tile_occupied.numpy()]
    tile_points_wp = wp.array(tile_points, dtype=wp.vec3i, device=device)

    # Allocate sparse volume
    sparse_volume = wp.Volume.allocate_by_tiles(
        tile_points=tile_points_wp,
        voxel_size=wp.vec3(actual_voxel_size),
        translation=wp.vec3(min_ext),
        bg_value=BACKGROUND_VALUE,
        device=device,
    )

    # Populate sparse volume
    num_allocated_tiles = len(tile_points)
    wp.launch(
        sdf_from_mesh_kernel,
        dim=(num_allocated_tiles, 8, 8, 8),
        inputs=[mesh.id, sparse_volume.id, tile_points_wp, 0.0],
        device=device,
    )

    # Compute max SDF value in sparse tiles (for threshold during sampling)
    tile_max_values = wp.zeros(num_allocated_tiles, dtype=float, device=device)
    # Initialize to large negative value for atomic_max
    tile_max_values.fill_(-1e10)

    wp.launch(
        compute_tile_max_kernel,
        dim=(num_allocated_tiles, 8, 8, 8),
        inputs=[sparse_volume.id, tile_points_wp, tile_max_values],
        device=device,
    )

    wp.synchronize()
    sparse_max_value = float(np.max(tile_max_values.numpy()))

    if verbose:
        print(f"  Sparse max SDF value: {sparse_max_value:.4f}")

    # Create coarse background volume (8x8x8 = one tile)
    coarse_voxel_size = ext / 7  # 8 voxels, 7 intervals
    coarse_tile_points = np.array([[0, 0, 0]], dtype=np.int32)
    coarse_tile_points_wp = wp.array(coarse_tile_points, dtype=wp.vec3i, device=device)

    coarse_volume = wp.Volume.allocate_by_tiles(
        tile_points=coarse_tile_points_wp,
        voxel_size=wp.vec3(coarse_voxel_size),
        translation=wp.vec3(min_ext),
        bg_value=BACKGROUND_VALUE,
        device=device,
    )

    # Populate coarse volume
    wp.launch(
        sdf_from_mesh_kernel,
        dim=(1, 8, 8, 8),
        inputs=[mesh.id, coarse_volume.id, coarse_tile_points_wp, 0.0],
        device=device,
    )

    wp.synchronize()

    if verbose:
        print(f"  Coarse voxel size: {coarse_voxel_size}")

    return sparse_volume, coarse_volume, sparse_max_value, min_ext, max_ext, actual_voxel_size


def create_box_mesh(center: tuple, half_extents: tuple, device: str) -> wp.Mesh:
    """
    Create a simple box mesh for testing.

    Args:
        center: Box center (x, y, z)
        half_extents: Half-sizes along each axis
        device: Warp device string

    Returns:
        wp.Mesh with support_winding_number=True
    """
    cx, cy, cz = center
    hx, hy, hz = half_extents

    # 8 vertices of the box
    vertices = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )

    # 12 triangles (counter-clockwise winding for outward normals)
    indices = np.array(
        [
            # Front face (Z-)
            0,
            2,
            1,
            0,
            3,
            2,
            # Back face (Z+)
            4,
            5,
            6,
            4,
            6,
            7,
            # Left face (X-)
            0,
            7,
            3,
            0,
            4,
            7,
            # Right face (X+)
            1,
            2,
            6,
            1,
            6,
            5,
            # Bottom face (Y-)
            0,
            1,
            5,
            0,
            5,
            4,
            # Top face (Y+)
            3,
            6,
            2,
            3,
            7,
            6,
        ],
        dtype=np.int32,
    )

    points = wp.array(vertices, dtype=wp.vec3, device=device)
    indices_arr = wp.array(indices, dtype=int, device=device)

    return wp.Mesh(points=points, indices=indices_arr, support_winding_number=True)
