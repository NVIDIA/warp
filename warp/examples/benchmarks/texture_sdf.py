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

"""Texture-based (tex3d) sparse SDF construction and sampling.

This module provides a GPU-accelerated sparse SDF implementation using 3D CUDA textures.
The construction follows the pattern from PhysX's SDFConstruction.cu:

1. Build dense SDF using mesh queries (one thread per voxel)
2. Build background/coarse SDF by sampling dense grid at subgrid corners
3. Mark required subgrids (those intersecting the narrow band)
4. Populate subgrid textures from dense SDF

The format uses:
- A coarse 3D texture for background/far-field sampling
- A packed subgrid 3D texture for narrow-band high-resolution sampling
- An indirection array mapping coarse cells to subgrid blocks
"""

from __future__ import annotations

import numpy as np
from volume_sdf import get_distance_to_mesh

import warp as wp

# ============================================================================
# Sparse SDF Data Structure
# ============================================================================


class QuantizationMode:
    """Quantization modes for subgrid SDF data."""

    FLOAT32 = 4  # No quantization, full precision
    UINT16 = 2  # 16-bit quantization
    UINT8 = 1  # 8-bit quantization


def get_quantization_bytes(mode: int) -> int:
    """
    Get the number of bytes per sample for a given quantization mode.

    Args:
        mode: QuantizationMode value (FLOAT32, UINT16, or UINT8)

    Returns:
        Number of bytes per sample (4, 2, or 1)

    Raises:
        ValueError: If mode is not a valid QuantizationMode
    """
    bytes_map = {
        QuantizationMode.FLOAT32: 4,
        QuantizationMode.UINT16: 2,
        QuantizationMode.UINT8: 1,
    }
    if mode not in bytes_map:
        raise ValueError(f"Invalid quantization mode: {mode}")
    return bytes_map[mode]


@wp.struct
class SparseSDF:
    """Parameters for sparse SDF sampling (textures passed separately)."""

    sdf_box_lower: wp.vec3
    sdf_box_upper: wp.vec3
    sdf_dx: float  # Voxel size (for gradient finite differences)
    inv_sdf_dx: float
    inv_2dx: float  # 0.5 / sdf_dx (for gradient central differences)
    coarse_size_x: int
    coarse_size_y: int
    coarse_size_z: int
    subgrid_size: int
    subgrid_size_f: float  # float(subgrid_size) - avoids int->float conversion
    subgrid_samples_f: float  # float(subgrid_size + 1) - samples per subgrid dimension
    fine_to_coarse: float
    # Precomputed inverse texture sizes (avoid divisions in sampling)
    inv_coarse_tex_size_x: float  # 1.0 / (coarse_size_x + 1)
    inv_coarse_tex_size_y: float  # 1.0 / (coarse_size_y + 1)
    inv_coarse_tex_size_z: float  # 1.0 / (coarse_size_z + 1)
    inv_subgrid_tex_size: float  # 1.0 / subgrid_tex_size
    # Quantization parameters for subgrid values
    subgrids_min_sdf_value: float
    subgrids_sdf_value_range: float  # max - min


# ============================================================================
# Dense SDF Construction Kernels
# ============================================================================


@wp.func
def idx3d(x: int, y: int, z: int, size_x: int, size_y: int) -> int:
    """Convert 3D coordinates to linear index."""
    return z * size_x * size_y + y * size_x + x


@wp.func
def id_to_xyz(idx: int, size_x: int, size_y: int) -> wp.vec3i:
    """Convert linear index to 3D coordinates."""
    z = idx // (size_x * size_y)
    rem = idx - z * size_x * size_y
    y = rem // size_x
    x = rem - y * size_x
    return wp.vec3i(x, y, z)


@wp.kernel
def build_dense_sdf_kernel(
    mesh: wp.uint64,
    sdf_data: wp.array(dtype=float),
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    size_x: int,
    size_y: int,
    size_z: int,
):
    """
    Build dense SDF grid by querying mesh distance at each voxel.

    Each thread computes the signed distance for one voxel.
    """
    tid = wp.tid()

    total_size = size_x * size_y * size_z
    if tid >= total_size:
        return

    # Convert linear index to 3D coordinates
    coords = id_to_xyz(tid, size_x, size_y)
    x = coords[0]
    y = coords[1]
    z = coords[2]

    # Compute world position (vertex-centered)
    pos = min_corner + wp.vec3(
        float(x) * cell_size[0],
        float(y) * cell_size[1],
        float(z) * cell_size[2],
    )

    # Query mesh for signed distance
    dist = get_distance_to_mesh(mesh, pos, 10000.0)

    sdf_data[tid] = dist


@wp.kernel
def build_background_sdf_kernel(
    dense_sdf: wp.array(dtype=float),
    background_sdf: wp.array(dtype=float),
    cells_per_subgrid: int,
    dense_size_x: int,
    dense_size_y: int,
    bg_size_x: int,
    bg_size_y: int,
    bg_size_z: int,
):
    """
    Populate background SDF by sampling dense SDF at subgrid corners.

    This matches SDFConstruction.cu's sdfPopulateBackgroundSDF kernel.
    """
    tid = wp.tid()

    total_bg = bg_size_x * bg_size_y * bg_size_z
    if tid >= total_bg:
        return

    # Convert to 3D coordinates
    coords = id_to_xyz(tid, bg_size_x, bg_size_y)
    x_block = coords[0]
    y_block = coords[1]
    z_block = coords[2]

    # Sample dense SDF at subgrid corner
    dense_x = x_block * cells_per_subgrid
    dense_y = y_block * cells_per_subgrid
    dense_z = z_block * cells_per_subgrid

    dense_idx = idx3d(dense_x, dense_y, dense_z, dense_size_x, dense_size_y)
    background_sdf[tid] = dense_sdf[dense_idx]


@wp.func
def sample_background_sdf_trilinear(
    background_sdf: wp.array(dtype=float),
    fx: float,
    fy: float,
    fz: float,
    bg_size_x: int,
    bg_size_y: int,
    bg_size_z: int,
) -> float:
    """
    Trilinear interpolation of background SDF at fractional coordinates.

    This matches PhysX's DenseSDF::sampleSDFDirect for error threshold calculation.
    """
    # Get integer and fractional parts
    x0 = int(wp.floor(fx))
    y0 = int(wp.floor(fy))
    z0 = int(wp.floor(fz))

    # Clamp to valid range
    x0 = wp.clamp(x0, 0, bg_size_x - 2)
    y0 = wp.clamp(y0, 0, bg_size_y - 2)
    z0 = wp.clamp(z0, 0, bg_size_z - 2)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Fractional position within cell
    tx = fx - float(x0)
    ty = fy - float(y0)
    tz = fz - float(z0)

    tx = wp.clamp(tx, 0.0, 1.0)
    ty = wp.clamp(ty, 0.0, 1.0)
    tz = wp.clamp(tz, 0.0, 1.0)

    # Sample 8 corners
    v000 = background_sdf[idx3d(x0, y0, z0, bg_size_x, bg_size_y)]
    v100 = background_sdf[idx3d(x1, y0, z0, bg_size_x, bg_size_y)]
    v010 = background_sdf[idx3d(x0, y1, z0, bg_size_x, bg_size_y)]
    v110 = background_sdf[idx3d(x1, y1, z0, bg_size_x, bg_size_y)]
    v001 = background_sdf[idx3d(x0, y0, z1, bg_size_x, bg_size_y)]
    v101 = background_sdf[idx3d(x1, y0, z1, bg_size_x, bg_size_y)]
    v011 = background_sdf[idx3d(x0, y1, z1, bg_size_x, bg_size_y)]
    v111 = background_sdf[idx3d(x1, y1, z1, bg_size_x, bg_size_y)]

    # Trilinear interpolation
    c00 = v000 * (1.0 - tx) + v100 * tx
    c10 = v010 * (1.0 - tx) + v110 * tx
    c01 = v001 * (1.0 - tx) + v101 * tx
    c11 = v011 * (1.0 - tx) + v111 * tx

    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty

    return c0 * (1.0 - tz) + c1 * tz


@wp.kernel
def mark_required_subgrids_kernel(
    dense_sdf: wp.array(dtype=float),
    background_sdf: wp.array(dtype=float),
    subgrid_required: wp.array(dtype=wp.int32),
    subgrid_min: wp.array(dtype=float),
    subgrid_max: wp.array(dtype=float),
    cells_per_subgrid: int,
    dense_size_x: int,
    dense_size_y: int,
    dense_size_z: int,
    num_subgrids_x: int,
    num_subgrids_y: int,
    num_subgrids_z: int,
    narrow_band_thickness: float,
    error_threshold: float,
):
    """
    Mark which subgrids are required (intersect narrow band and exceed error threshold).

    Each thread processes one subgrid and finds min/max SDF values.
    Matches PhysX's sdfMarkRequiredSdfSubgrids kernel.
    """
    tid = wp.tid()

    total_subgrids = num_subgrids_x * num_subgrids_y * num_subgrids_z
    if tid >= total_subgrids:
        return

    # Convert to 3D subgrid coordinates
    coords = id_to_xyz(tid, num_subgrids_x, num_subgrids_y)
    block_x = coords[0]
    block_y = coords[1]
    block_z = coords[2]

    # Background grid size
    bg_size_x = num_subgrids_x + 1
    bg_size_y = num_subgrids_y + 1
    bg_size_z = num_subgrids_z + 1

    # Scale factor from fine to coarse coordinates
    s = 1.0 / float(cells_per_subgrid)

    # Find min/max SDF values and max error in this subgrid
    sdf_min = float(1e10)
    sdf_max = float(-1e10)
    max_abs_error = float(0.0)

    samples_per_dim = cells_per_subgrid + 1
    for lz in range(samples_per_dim):
        for ly in range(samples_per_dim):
            for lx in range(samples_per_dim):
                gx = block_x * cells_per_subgrid + lx
                gy = block_y * cells_per_subgrid + ly
                gz = block_z * cells_per_subgrid + lz

                # Clamp to dense grid bounds
                gx = wp.min(gx, dense_size_x - 1)
                gy = wp.min(gy, dense_size_y - 1)
                gz = wp.min(gz, dense_size_z - 1)

                dense_idx = idx3d(gx, gy, gz, dense_size_x, dense_size_y)
                sdf_val = dense_sdf[dense_idx]

                sdf_min = wp.min(sdf_min, sdf_val)
                sdf_max = wp.max(sdf_max, sdf_val)

                # Compute error vs coarse SDF interpolation
                # Sample coarse at fractional position (block_x + lx*s, block_y + ly*s, block_z + lz*s)
                coarse_fx = float(block_x) + float(lx) * s
                coarse_fy = float(block_y) + float(ly) * s
                coarse_fz = float(block_z) + float(lz) * s

                coarse_val = sample_background_sdf_trilinear(
                    background_sdf, coarse_fx, coarse_fy, coarse_fz, bg_size_x, bg_size_y, bg_size_z
                )
                max_abs_error = wp.max(max_abs_error, wp.abs(sdf_val - coarse_val))

    # Check if range overlaps narrow band
    # rangesOverlap(sdf_min, sdf_max, -narrow_band, +narrow_band)
    overlaps_narrow_band = not (sdf_min > narrow_band_thickness or -narrow_band_thickness > sdf_max)

    # Subgrid is required if it overlaps the narrow band AND the coarse SDF has significant error
    subgrid_is_required = overlaps_narrow_band and (max_abs_error >= error_threshold)

    if subgrid_is_required:
        subgrid_required[tid] = 1
    else:
        subgrid_required[tid] = 0

    subgrid_min[tid] = sdf_min
    subgrid_max[tid] = sdf_max


@wp.kernel
def populate_subgrid_texture_float32_kernel(
    dense_sdf: wp.array(dtype=float),
    subgrid_required: wp.array(dtype=wp.int32),
    subgrid_addresses: wp.array(dtype=wp.int32),
    subgrid_start_slots: wp.array(dtype=wp.uint32),
    subgrid_texture: wp.array(dtype=float),
    cells_per_subgrid: int,
    dense_size_x: int,
    dense_size_y: int,
    num_subgrids_x: int,
    num_subgrids_y: int,
    num_subgrids_z: int,
    tex_blocks_per_dim: int,
    tex_size: int,
):
    """
    Populate subgrid texture from dense SDF (float32 version).

    Each thread processes one sample in the subgrid texture.
    """
    tid = wp.tid()

    total_subgrids = num_subgrids_x * num_subgrids_y * num_subgrids_z
    samples_per_dim = cells_per_subgrid + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim

    # Determine which subgrid and local sample this thread handles
    subgrid_idx = tid // samples_per_subgrid
    local_sample = tid - subgrid_idx * samples_per_subgrid

    if subgrid_idx >= total_subgrids:
        return

    if subgrid_required[subgrid_idx] == 0:
        return

    # Get subgrid 3D coordinates
    subgrid_coords = id_to_xyz(subgrid_idx, num_subgrids_x, num_subgrids_y)
    block_x = subgrid_coords[0]
    block_y = subgrid_coords[1]
    block_z = subgrid_coords[2]

    # Get local sample 3D coordinates
    local_coords = id_to_xyz(local_sample, samples_per_dim, samples_per_dim)
    lx = local_coords[0]
    ly = local_coords[1]
    lz = local_coords[2]

    # Read from dense SDF
    gx = wp.min(block_x * cells_per_subgrid + lx, dense_size_x - 1)
    gy = wp.min(block_y * cells_per_subgrid + ly, dense_size_y - 1)
    gz = block_z * cells_per_subgrid + lz

    dense_idx = idx3d(gx, gy, gz, dense_size_x, dense_size_y)
    sdf_val = dense_sdf[dense_idx]

    # Get texture block address
    address = subgrid_addresses[subgrid_idx]
    if address < 0:
        return

    # Decode texture block position (cubic texture)
    addr_coords = id_to_xyz(address, tex_blocks_per_dim, tex_blocks_per_dim)
    addr_x = addr_coords[0]
    addr_y = addr_coords[1]
    addr_z = addr_coords[2]

    # Write start slot (only first sample per subgrid)
    if local_sample == 0:
        start_slot = wp.uint32(addr_x) | (wp.uint32(addr_y) << wp.uint32(10)) | (wp.uint32(addr_z) << wp.uint32(20))
        subgrid_start_slots[subgrid_idx] = start_slot

    # Compute texture coordinates (cubic texture)
    tex_x = addr_x * samples_per_dim + lx
    tex_y = addr_y * samples_per_dim + ly
    tex_z = addr_z * samples_per_dim + lz

    tex_idx = idx3d(tex_x, tex_y, tex_z, tex_size, tex_size)
    subgrid_texture[tex_idx] = sdf_val


@wp.kernel
def populate_subgrid_texture_uint16_kernel(
    dense_sdf: wp.array(dtype=float),
    subgrid_required: wp.array(dtype=wp.int32),
    subgrid_addresses: wp.array(dtype=wp.int32),
    subgrid_start_slots: wp.array(dtype=wp.uint32),
    subgrid_texture: wp.array(dtype=wp.uint16),
    cells_per_subgrid: int,
    dense_size_x: int,
    dense_size_y: int,
    num_subgrids_x: int,
    num_subgrids_y: int,
    num_subgrids_z: int,
    tex_blocks_per_dim: int,
    tex_size: int,
    sdf_min: float,
    sdf_range_inv: float,
):
    """
    Populate subgrid texture from dense SDF (uint16 quantized version).

    Values are normalized to [0, 65535] range.
    """
    tid = wp.tid()

    total_subgrids = num_subgrids_x * num_subgrids_y * num_subgrids_z
    samples_per_dim = cells_per_subgrid + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim

    subgrid_idx = tid // samples_per_subgrid
    local_sample = tid - subgrid_idx * samples_per_subgrid

    if subgrid_idx >= total_subgrids:
        return

    if subgrid_required[subgrid_idx] == 0:
        return

    subgrid_coords = id_to_xyz(subgrid_idx, num_subgrids_x, num_subgrids_y)
    block_x = subgrid_coords[0]
    block_y = subgrid_coords[1]
    block_z = subgrid_coords[2]

    local_coords = id_to_xyz(local_sample, samples_per_dim, samples_per_dim)
    lx = local_coords[0]
    ly = local_coords[1]
    lz = local_coords[2]

    gx = wp.min(block_x * cells_per_subgrid + lx, dense_size_x - 1)
    gy = wp.min(block_y * cells_per_subgrid + ly, dense_size_y - 1)
    gz = block_z * cells_per_subgrid + lz

    dense_idx = idx3d(gx, gy, gz, dense_size_x, dense_size_y)
    sdf_val = dense_sdf[dense_idx]

    address = subgrid_addresses[subgrid_idx]
    if address < 0:
        return

    addr_coords = id_to_xyz(address, tex_blocks_per_dim, tex_blocks_per_dim)
    addr_x = addr_coords[0]
    addr_y = addr_coords[1]
    addr_z = addr_coords[2]

    if local_sample == 0:
        start_slot = wp.uint32(addr_x) | (wp.uint32(addr_y) << wp.uint32(10)) | (wp.uint32(addr_z) << wp.uint32(20))
        subgrid_start_slots[subgrid_idx] = start_slot

    tex_x = addr_x * samples_per_dim + lx
    tex_y = addr_y * samples_per_dim + ly
    tex_z = addr_z * samples_per_dim + lz

    # Normalize to [0, 1] then scale to [0, 65535]
    v_normalized = wp.clamp((sdf_val - sdf_min) * sdf_range_inv, 0.0, 1.0)
    quantized = wp.uint16(v_normalized * 65535.0)

    tex_idx = idx3d(tex_x, tex_y, tex_z, tex_size, tex_size)
    subgrid_texture[tex_idx] = quantized


@wp.kernel
def populate_subgrid_texture_uint8_kernel(
    dense_sdf: wp.array(dtype=float),
    subgrid_required: wp.array(dtype=wp.int32),
    subgrid_addresses: wp.array(dtype=wp.int32),
    subgrid_start_slots: wp.array(dtype=wp.uint32),
    subgrid_texture: wp.array(dtype=wp.uint8),
    cells_per_subgrid: int,
    dense_size_x: int,
    dense_size_y: int,
    num_subgrids_x: int,
    num_subgrids_y: int,
    num_subgrids_z: int,
    tex_blocks_per_dim: int,
    tex_size: int,
    sdf_min: float,
    sdf_range_inv: float,
):
    """
    Populate subgrid texture from dense SDF (uint8 quantized version).

    Values are normalized to [0, 255] range.
    """
    tid = wp.tid()

    total_subgrids = num_subgrids_x * num_subgrids_y * num_subgrids_z
    samples_per_dim = cells_per_subgrid + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim

    subgrid_idx = tid // samples_per_subgrid
    local_sample = tid - subgrid_idx * samples_per_subgrid

    if subgrid_idx >= total_subgrids:
        return

    if subgrid_required[subgrid_idx] == 0:
        return

    subgrid_coords = id_to_xyz(subgrid_idx, num_subgrids_x, num_subgrids_y)
    block_x = subgrid_coords[0]
    block_y = subgrid_coords[1]
    block_z = subgrid_coords[2]

    local_coords = id_to_xyz(local_sample, samples_per_dim, samples_per_dim)
    lx = local_coords[0]
    ly = local_coords[1]
    lz = local_coords[2]

    gx = wp.min(block_x * cells_per_subgrid + lx, dense_size_x - 1)
    gy = wp.min(block_y * cells_per_subgrid + ly, dense_size_y - 1)
    gz = block_z * cells_per_subgrid + lz

    dense_idx = idx3d(gx, gy, gz, dense_size_x, dense_size_y)
    sdf_val = dense_sdf[dense_idx]

    address = subgrid_addresses[subgrid_idx]
    if address < 0:
        return

    addr_coords = id_to_xyz(address, tex_blocks_per_dim, tex_blocks_per_dim)
    addr_x = addr_coords[0]
    addr_y = addr_coords[1]
    addr_z = addr_coords[2]

    if local_sample == 0:
        start_slot = wp.uint32(addr_x) | (wp.uint32(addr_y) << wp.uint32(10)) | (wp.uint32(addr_z) << wp.uint32(20))
        subgrid_start_slots[subgrid_idx] = start_slot

    tex_x = addr_x * samples_per_dim + lx
    tex_y = addr_y * samples_per_dim + ly
    tex_z = addr_z * samples_per_dim + lz

    # Normalize to [0, 1] then scale to [0, 255]
    v_normalized = wp.clamp((sdf_val - sdf_min) * sdf_range_inv, 0.0, 1.0)
    quantized = wp.uint8(v_normalized * 255.0)

    tex_idx = idx3d(tex_x, tex_y, tex_z, tex_size, tex_size)
    subgrid_texture[tex_idx] = quantized


# ============================================================================
# Texture Sampling Functions and Kernel
# ============================================================================


@wp.func
def apply_subgrid_start(start_slot: wp.uint32, local_f: wp.vec3, subgrid_samples_f: float) -> wp.vec3:
    """Apply subgrid block offset to local coordinates."""
    block_x = float(start_slot & wp.uint32(0x3FF))
    block_y = float((start_slot >> wp.uint32(10)) & wp.uint32(0x3FF))
    block_z = float((start_slot >> wp.uint32(20)) & wp.uint32(0x3FF))

    return wp.vec3(
        local_f[0] + block_x * subgrid_samples_f,
        local_f[1] + block_y * subgrid_samples_f,
        local_f[2] + block_z * subgrid_samples_f,
    )


@wp.func
def apply_subgrid_sdf_scale(raw_value: float, min_value: float, value_range: float) -> float:
    """
    Apply quantization scale to convert normalized [0,1] value back to SDF distance.

    Matches PhysX's applySubgridSdfScale.
    """
    return raw_value * value_range + min_value


@wp.kernel
def sample_sparse_sdf(
    sdf: SparseSDF,
    coarse_texture: wp.texture3d_t,
    subgrid_texture: wp.texture3d_t,
    subgrid_start_slots: wp.array(dtype=wp.uint32),
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
):
    """
    Sample SDF using sparse texture representation.

    Matches PhysX's PxSdfDistance + SparseSDFTexture::Sample.
    Supports both quantized and non-quantized subgrid textures.
    """
    tid = wp.tid()
    local_pos = query_points[tid]

    # Clamp to SDF box (matches PxSdfDistance)
    clamped = wp.vec3(
        wp.clamp(local_pos[0], sdf.sdf_box_lower[0], sdf.sdf_box_upper[0]),
        wp.clamp(local_pos[1], sdf.sdf_box_lower[1], sdf.sdf_box_upper[1]),
        wp.clamp(local_pos[2], sdf.sdf_box_lower[2], sdf.sdf_box_upper[2]),
    )
    diff_mag = wp.length(local_pos - clamped)

    # Convert to grid coordinates
    f = (clamped - sdf.sdf_box_lower) * sdf.inv_sdf_dx

    # Compute coarse cell indices
    x_base = wp.clamp(int(f[0] * sdf.fine_to_coarse), 0, sdf.coarse_size_x - 1)
    y_base = wp.clamp(int(f[1] * sdf.fine_to_coarse), 0, sdf.coarse_size_y - 1)
    z_base = wp.clamp(int(f[2] * sdf.fine_to_coarse), 0, sdf.coarse_size_z - 1)

    # Look up indirection slot: (z * sy + y) * sx + x
    slot_idx = (z_base * sdf.coarse_size_y + y_base) * sdf.coarse_size_x + x_base
    start_slot = subgrid_start_slots[slot_idx]

    sdf_val = float(0.0)

    if start_slot == wp.uint32(0xFFFFFFFF):
        # No subgrid - sample from coarse texture
        coarse_f = f * sdf.fine_to_coarse
        u = (coarse_f[0] + 0.5) * sdf.inv_coarse_tex_size_x
        v = (coarse_f[1] + 0.5) * sdf.inv_coarse_tex_size_y
        w = (coarse_f[2] + 0.5) * sdf.inv_coarse_tex_size_z
        sdf_val = wp.tex3d_float(coarse_texture, u, v, w)
    else:
        # Sample from subgrid texture (convert to float once)
        fx_base = float(x_base)
        fy_base = float(y_base)
        fz_base = float(z_base)
        local_x = wp.clamp(f[0] - fx_base * sdf.subgrid_size_f, 0.0, sdf.subgrid_samples_f)
        local_y = wp.clamp(f[1] - fy_base * sdf.subgrid_size_f, 0.0, sdf.subgrid_samples_f)
        local_z = wp.clamp(f[2] - fz_base * sdf.subgrid_size_f, 0.0, sdf.subgrid_samples_f)

        local_f = wp.vec3(local_x, local_y, local_z)
        tex_coords = apply_subgrid_start(start_slot, local_f, sdf.subgrid_samples_f)

        # Cubic texture - use same size for all dimensions
        u = (tex_coords[0] + 0.5) * sdf.inv_subgrid_tex_size
        v = (tex_coords[1] + 0.5) * sdf.inv_subgrid_tex_size
        w = (tex_coords[2] + 0.5) * sdf.inv_subgrid_tex_size

        raw_val = wp.tex3d_float(subgrid_texture, u, v, w)

        # Apply quantization scale (no-op for float32 where range=1.0, min=0.0)
        sdf_val = apply_subgrid_sdf_scale(raw_val, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)

    results[tid] = sdf_val + diff_mag


@wp.func
def sample_sparse_sdf_at(
    sdf: SparseSDF,
    coarse_texture: wp.texture3d_t,
    subgrid_texture: wp.texture3d_t,
    subgrid_start_slots: wp.array(dtype=wp.uint32),
    local_pos: wp.vec3,
) -> float:
    """
    Sample SDF at a specific position (functional version for gradient computation).

    Matches the logic of sample_sparse_sdf but returns the value directly.
    """
    # Clamp to SDF box (matches PxSdfDistance)
    clamped = wp.vec3(
        wp.clamp(local_pos[0], sdf.sdf_box_lower[0], sdf.sdf_box_upper[0]),
        wp.clamp(local_pos[1], sdf.sdf_box_lower[1], sdf.sdf_box_upper[1]),
        wp.clamp(local_pos[2], sdf.sdf_box_lower[2], sdf.sdf_box_upper[2]),
    )
    diff_mag = wp.length(local_pos - clamped)

    # Convert to grid coordinates
    f = (clamped - sdf.sdf_box_lower) * sdf.inv_sdf_dx

    # Compute coarse cell indices
    x_base = wp.clamp(int(f[0] * sdf.fine_to_coarse), 0, sdf.coarse_size_x - 1)
    y_base = wp.clamp(int(f[1] * sdf.fine_to_coarse), 0, sdf.coarse_size_y - 1)
    z_base = wp.clamp(int(f[2] * sdf.fine_to_coarse), 0, sdf.coarse_size_z - 1)

    # Look up indirection slot: (z * sy + y) * sx + x = 2 int muls instead of 3
    slot_idx = (z_base * sdf.coarse_size_y + y_base) * sdf.coarse_size_x + x_base
    start_slot = subgrid_start_slots[slot_idx]

    sdf_val = float(0.0)

    if start_slot == wp.uint32(0xFFFFFFFF):
        # No subgrid - sample from coarse texture
        coarse_f = f * sdf.fine_to_coarse
        u = (coarse_f[0] + 0.5) * sdf.inv_coarse_tex_size_x
        v = (coarse_f[1] + 0.5) * sdf.inv_coarse_tex_size_y
        w = (coarse_f[2] + 0.5) * sdf.inv_coarse_tex_size_z
        sdf_val = wp.tex3d_float(coarse_texture, u, v, w)
    else:
        # Sample from subgrid texture (use float x_base to avoid int->float)
        fx_base = float(x_base)
        fy_base = float(y_base)
        fz_base = float(z_base)
        local_x = wp.clamp(f[0] - fx_base * sdf.subgrid_size_f, 0.0, sdf.subgrid_samples_f)
        local_y = wp.clamp(f[1] - fy_base * sdf.subgrid_size_f, 0.0, sdf.subgrid_samples_f)
        local_z = wp.clamp(f[2] - fz_base * sdf.subgrid_size_f, 0.0, sdf.subgrid_samples_f)

        local_f = wp.vec3(local_x, local_y, local_z)
        tex_coords = apply_subgrid_start(start_slot, local_f, sdf.subgrid_samples_f)

        u = (tex_coords[0] + 0.5) * sdf.inv_subgrid_tex_size
        v = (tex_coords[1] + 0.5) * sdf.inv_subgrid_tex_size
        w = (tex_coords[2] + 0.5) * sdf.inv_subgrid_tex_size

        raw_val = wp.tex3d_float(subgrid_texture, u, v, w)
        sdf_val = apply_subgrid_sdf_scale(raw_val, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)

    return sdf_val + diff_mag


@wp.func
def sample_texture_at_grid_coords(
    sdf: SparseSDF,
    coarse_texture: wp.texture3d_t,
    subgrid_texture: wp.texture3d_t,
    subgrid_start_slots: wp.array(dtype=wp.uint32),
    f: wp.vec3,
) -> float:
    """
    Sample SDF directly at grid coordinates (fast path, no boundary checks).

    This is the inner sampling loop used by the fast gradient path.
    Assumes f is already in valid grid coordinate range.
    """
    # Compute coarse cell indices
    x_base = int(f[0] * sdf.fine_to_coarse)
    y_base = int(f[1] * sdf.fine_to_coarse)
    z_base = int(f[2] * sdf.fine_to_coarse)

    # Look up indirection slot: (z * sy + y) * sx + x
    slot_idx = (z_base * sdf.coarse_size_y + y_base) * sdf.coarse_size_x + x_base
    start_slot = subgrid_start_slots[slot_idx]

    sdf_val = float(0.0)

    if start_slot == wp.uint32(0xFFFFFFFF):
        # No subgrid - sample from coarse texture
        coarse_f = f * sdf.fine_to_coarse
        u = (coarse_f[0] + 0.5) * sdf.inv_coarse_tex_size_x
        v = (coarse_f[1] + 0.5) * sdf.inv_coarse_tex_size_y
        w = (coarse_f[2] + 0.5) * sdf.inv_coarse_tex_size_z
        sdf_val = wp.tex3d_float(coarse_texture, u, v, w)
    else:
        # Sample from subgrid texture (convert to float once)
        fx_base = float(x_base)
        fy_base = float(y_base)
        fz_base = float(z_base)
        local_x = f[0] - fx_base * sdf.subgrid_size_f
        local_y = f[1] - fy_base * sdf.subgrid_size_f
        local_z = f[2] - fz_base * sdf.subgrid_size_f

        local_f = wp.vec3(local_x, local_y, local_z)
        tex_coords = apply_subgrid_start(start_slot, local_f, sdf.subgrid_samples_f)

        u = (tex_coords[0] + 0.5) * sdf.inv_subgrid_tex_size
        v = (tex_coords[1] + 0.5) * sdf.inv_subgrid_tex_size
        w = (tex_coords[2] + 0.5) * sdf.inv_subgrid_tex_size

        raw_val = wp.tex3d_float(subgrid_texture, u, v, w)
        sdf_val = apply_subgrid_sdf_scale(raw_val, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)

    return sdf_val


@wp.func
def sample_with_precomputed_cell(
    sdf: SparseSDF,
    coarse_texture: wp.texture3d_t,
    subgrid_texture: wp.texture3d_t,
    start_slot: wp.uint32,
    subgrid_start_slots: wp.array(dtype=wp.uint32),
    x_base: int,
    y_base: int,
    z_base: int,
    f: wp.vec3,
) -> float:
    """
    Sample SDF with pre-computed cell indices (optimized for gradient).

    Handles both coarse (start_slot == 0xFFFFFFFF) and subgrid cases.
    This avoids recomputing the cell lookup for each of the 7 gradient samples.
    """

    if start_slot == wp.uint32(0xFFFFFFFE):
        # Compute coarse cell indices
        x_base = int(f[0] * sdf.fine_to_coarse)
        y_base = int(f[1] * sdf.fine_to_coarse)
        z_base = int(f[2] * sdf.fine_to_coarse)

        # Look up indirection slot: (z * sy + y) * sx + x
        slot_idx = (z_base * sdf.coarse_size_y + y_base) * sdf.coarse_size_x + x_base
        start_slot = subgrid_start_slots[slot_idx]

    if start_slot == wp.uint32(0xFFFFFFFF):
        # No subgrid - sample from coarse texture
        coarse_f = f * sdf.fine_to_coarse
        u = (coarse_f[0] + 0.5) * sdf.inv_coarse_tex_size_x
        v = (coarse_f[1] + 0.5) * sdf.inv_coarse_tex_size_y
        w = (coarse_f[2] + 0.5) * sdf.inv_coarse_tex_size_z
        return wp.tex3d_float(coarse_texture, u, v, w)
    else:
        # Sample from subgrid texture (convert to float once)
        fx_base = float(x_base)
        fy_base = float(y_base)
        fz_base = float(z_base)
        local_x = f[0] - fx_base * sdf.subgrid_size_f
        local_y = f[1] - fy_base * sdf.subgrid_size_f
        local_z = f[2] - fz_base * sdf.subgrid_size_f

        local_f = wp.vec3(local_x, local_y, local_z)
        tex_coords = apply_subgrid_start(start_slot, local_f, sdf.subgrid_samples_f)

        u = (tex_coords[0] + 0.5) * sdf.inv_subgrid_tex_size
        v = (tex_coords[1] + 0.5) * sdf.inv_subgrid_tex_size
        w = (tex_coords[2] + 0.5) * sdf.inv_subgrid_tex_size

        raw_val = wp.tex3d_float(subgrid_texture, u, v, w)
        return apply_subgrid_sdf_scale(raw_val, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)


@wp.kernel
def sample_sparse_sdf_grad(
    sdf: SparseSDF,
    coarse_texture: wp.texture3d_t,
    subgrid_texture: wp.texture3d_t,
    subgrid_start_slots: wp.array(dtype=wp.uint32),
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
    gradients: wp.array(dtype=wp.vec3),
):
    """
    Sample SDF and gradient using sparse texture representation.

    Gradient is computed using finite differences, matching the pattern from
    PhysX's PxVolumeGrad in sdfCollision.cuh.

    Optimizations:
    1. Fast path for interior points (skips boundary clamping/extrapolation)
    2. Shared subgrid lookup when all 7 samples fall in the same cell
    """
    tid = wp.tid()
    local_pos = query_points[tid]

    # Convert to grid coordinates (compute once, reuse for all samples)
    f = (local_pos - sdf.sdf_box_lower) * sdf.inv_sdf_dx

    # Grid dimensions in fine coordinates
    fine_dims_x = float(sdf.coarse_size_x * sdf.subgrid_size)
    fine_dims_y = float(sdf.coarse_size_y * sdf.subgrid_size)
    fine_dims_z = float(sdf.coarse_size_z * sdf.subgrid_size)

    # Check if point is well inside the grid (fast path condition from PhysX PxVolumeGrad)
    # We need 1 voxel margin on each side for the +/- 1 samples
    inside = (
        f[0] >= 1.0
        and f[0] <= fine_dims_x - 2.0
        and f[1] >= 1.0
        and f[1] <= fine_dims_y - 2.0
        and f[2] >= 1.0
        and f[2] <= fine_dims_z - 2.0
    )

    dist = float(0.0)
    grad = wp.vec3(0.0, 0.0, 0.0)

    factor = 0.5

    if inside:
        # FAST PATH: Point is well inside grid

        # Compute coarse cell indices for center point
        x_base = int(f[0] * sdf.fine_to_coarse)
        y_base = int(f[1] * sdf.fine_to_coarse)
        z_base = int(f[2] * sdf.fine_to_coarse)

        # Look up indirection slot once: (z * sy + y) * sx + x
        slot_idx = (z_base * sdf.coarse_size_y + y_base) * sdf.coarse_size_x + x_base
        start_slot = subgrid_start_slots[slot_idx]

        # Compute cell bounds for per-sample checks (convert to float once, reuse)
        fx_base = float(x_base)
        fy_base = float(y_base)
        fz_base = float(z_base)
        cell_lower_x = fx_base * sdf.subgrid_size_f
        cell_lower_y = fy_base * sdf.subgrid_size_f
        cell_lower_z = fz_base * sdf.subgrid_size_f
        cell_upper_x = cell_lower_x + sdf.subgrid_size_f
        cell_upper_y = cell_lower_y + sdf.subgrid_size_f
        cell_upper_z = cell_lower_z + sdf.subgrid_size_f

        # Check each sample point individually - use marker only when needed
        marker = wp.uint32(0xFFFFFFFE)

        # Center point always uses precomputed slot
        dist = sample_with_precomputed_cell(
            sdf, coarse_texture, subgrid_texture, start_slot, subgrid_start_slots, x_base, y_base, z_base, f
        )

        # +x: check if f[0]+1 is still in same cell
        slot_px = start_slot if f[0] + 1.0 <= cell_upper_x else marker
        dist_px = sample_with_precomputed_cell(
            sdf,
            coarse_texture,
            subgrid_texture,
            slot_px,
            subgrid_start_slots,
            x_base,
            y_base,
            z_base,
            wp.vec3(f[0] + 1.0, f[1], f[2]),
        )

        # -x: check if f[0]-1 is still in same cell
        slot_mx = start_slot if f[0] - 1.0 >= cell_lower_x else marker
        dist_mx = sample_with_precomputed_cell(
            sdf,
            coarse_texture,
            subgrid_texture,
            slot_mx,
            subgrid_start_slots,
            x_base,
            y_base,
            z_base,
            wp.vec3(f[0] - 1.0, f[1], f[2]),
        )

        # +y: check if f[1]+1 is still in same cell
        slot_py = start_slot if f[1] + 1.0 <= cell_upper_y else marker
        dist_py = sample_with_precomputed_cell(
            sdf,
            coarse_texture,
            subgrid_texture,
            slot_py,
            subgrid_start_slots,
            x_base,
            y_base,
            z_base,
            wp.vec3(f[0], f[1] + 1.0, f[2]),
        )

        # -y: check if f[1]-1 is still in same cell
        slot_my = start_slot if f[1] - 1.0 >= cell_lower_y else marker
        dist_my = sample_with_precomputed_cell(
            sdf,
            coarse_texture,
            subgrid_texture,
            slot_my,
            subgrid_start_slots,
            x_base,
            y_base,
            z_base,
            wp.vec3(f[0], f[1] - 1.0, f[2]),
        )

        # +z: check if f[2]+1 is still in same cell
        slot_pz = start_slot if f[2] + 1.0 <= cell_upper_z else marker
        dist_pz = sample_with_precomputed_cell(
            sdf,
            coarse_texture,
            subgrid_texture,
            slot_pz,
            subgrid_start_slots,
            x_base,
            y_base,
            z_base,
            wp.vec3(f[0], f[1], f[2] + 1.0),
        )

        # -z: check if f[2]-1 is still in same cell
        slot_mz = start_slot if f[2] - 1.0 >= cell_lower_z else marker
        dist_mz = sample_with_precomputed_cell(
            sdf,
            coarse_texture,
            subgrid_texture,
            slot_mz,
            subgrid_start_slots,
            x_base,
            y_base,
            z_base,
            wp.vec3(f[0], f[1], f[2] - 1.0),
        )

        factor = 0.5
    else:
        # SLOW PATH: Near boundary, use full sampling with clamping/extrapolation
        dist = sample_sparse_sdf_at(sdf, coarse_texture, subgrid_texture, subgrid_start_slots, local_pos)

        dx = sdf.sdf_dx

        dist_px = sample_sparse_sdf_at(
            sdf, coarse_texture, subgrid_texture, subgrid_start_slots, local_pos + wp.vec3(dx, 0.0, 0.0)
        )
        dist_mx = sample_sparse_sdf_at(
            sdf, coarse_texture, subgrid_texture, subgrid_start_slots, local_pos - wp.vec3(dx, 0.0, 0.0)
        )
        dist_py = sample_sparse_sdf_at(
            sdf, coarse_texture, subgrid_texture, subgrid_start_slots, local_pos + wp.vec3(0.0, dx, 0.0)
        )
        dist_my = sample_sparse_sdf_at(
            sdf, coarse_texture, subgrid_texture, subgrid_start_slots, local_pos - wp.vec3(0.0, dx, 0.0)
        )
        dist_pz = sample_sparse_sdf_at(
            sdf, coarse_texture, subgrid_texture, subgrid_start_slots, local_pos + wp.vec3(0.0, 0.0, dx)
        )
        dist_mz = sample_sparse_sdf_at(
            sdf, coarse_texture, subgrid_texture, subgrid_start_slots, local_pos - wp.vec3(0.0, 0.0, dx)
        )

        factor = sdf.inv_2dx

    grad = wp.vec3(
        (dist_px - dist_mx) * factor,
        (dist_py - dist_my) * factor,
        (dist_pz - dist_mz) * factor,
    )

    results[tid] = dist
    gradients[tid] = grad


# ============================================================================
# Host-side Construction Functions
# ============================================================================


def build_dense_sdf(
    mesh: wp.Mesh,
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    resolution: int,
    device: str = "cuda",
) -> tuple[wp.array, int, int, int, np.ndarray]:
    """
    Build a dense SDF grid from a mesh using GPU kernels.

    Args:
        mesh: Warp mesh with support_winding_number=True
        min_corner: Lower corner of SDF domain
        max_corner: Upper corner of SDF domain
        resolution: Maximum grid dimension
        device: Warp device

    Returns:
        Tuple of (sdf_array, size_x, size_y, size_z, cell_size)
    """
    ext = max_corner - min_corner
    max_ext = np.max(ext)
    cell_size_scalar = max_ext / resolution

    # Compute grid dimensions
    dims = np.ceil(ext / cell_size_scalar).astype(int) + 1
    size_x, size_y, size_z = int(dims[0]), int(dims[1]), int(dims[2])
    cell_size = ext / (dims - 1)

    print(f"  Dense SDF dims: {size_x}x{size_y}x{size_z}")
    print(f"  Cell size: {cell_size}")

    # Allocate dense SDF array
    total_voxels = size_x * size_y * size_z
    dense_sdf = wp.zeros(int(total_voxels), dtype=float, device=device)

    # Launch kernel to build dense SDF
    wp.launch(
        build_dense_sdf_kernel,
        dim=total_voxels,
        inputs=[
            mesh.id,
            dense_sdf,
            wp.vec3(min_corner[0], min_corner[1], min_corner[2]),
            wp.vec3(cell_size[0], cell_size[1], cell_size[2]),
            size_x,
            size_y,
            size_z,
        ],
        device=device,
    )
    wp.synchronize()

    return dense_sdf, size_x, size_y, size_z, cell_size


def build_sparse_sdf_from_dense(
    dense_sdf: wp.array,
    dense_size_x: int,
    dense_size_y: int,
    dense_size_z: int,
    cell_size: np.ndarray,
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    subgrid_size: int = 8,
    narrow_band_thickness: float = 0.1,
    error_threshold: float | None = None,
    quantization_mode: int = QuantizationMode.FLOAT32,
    device: str = "cuda",
) -> dict:
    """
    Build sparse SDF texture representation from dense SDF using GPU kernels.

    This follows the pattern from PhysX SDFConstruction.cu / PxgSDFBuilder.cpp:
    1. Populate background SDF by sampling at subgrid corners
    2. Mark required subgrids (those intersecting narrow band AND exceeding error threshold)
    3. Exclusive scan to assign sequential addresses
    4. Populate subgrid texture from dense SDF (with optional quantization)

    Args:
        dense_sdf: Dense SDF array from build_dense_sdf
        dense_size_x/y/z: Dense grid dimensions
        cell_size: Dense grid cell size
        min_corner: Lower corner of domain
        max_corner: Upper corner of domain
        subgrid_size: Cells per subgrid (typically 8)
        narrow_band_thickness: Distance threshold for subgrids
        error_threshold: Skip subgrids where coarse SDF error is below this (None = auto)
        quantization_mode: QuantizationMode.FLOAT32, UINT16, or UINT8
        device: Warp device

    Returns:
        Dictionary with all sparse SDF data
    """
    # Compute number of subgrids (w, h, d in PhysX code)
    # Note: dense grid is (width+1) x (height+1) x (depth+1) samples
    # So we have width/subgrid_size subgrids per dimension
    w = (dense_size_x - 1) // subgrid_size
    h = (dense_size_y - 1) // subgrid_size
    d = (dense_size_z - 1) // subgrid_size
    total_subgrids = w * h * d

    print(f"  Subgrid dims: {w}x{h}x{d} = {total_subgrids} subgrids")

    # Build background SDF (coarse grid) - samples at subgrid corners
    bg_size_x = w + 1
    bg_size_y = h + 1
    bg_size_z = d + 1
    total_bg = bg_size_x * bg_size_y * bg_size_z

    background_sdf = wp.zeros(total_bg, dtype=float, device=device)

    wp.launch(
        build_background_sdf_kernel,
        dim=total_bg,
        inputs=[
            dense_sdf,
            background_sdf,
            subgrid_size,
            dense_size_x,
            dense_size_y,
            bg_size_x,
            bg_size_y,
            bg_size_z,
        ],
        device=device,
    )

    # Compute error threshold if not provided (matches PhysX: 1e-6 * extents.magnitude())
    if error_threshold is None:
        extents = max_corner - min_corner
        error_threshold = float(1e-6 * np.linalg.norm(extents))

    # Mark required subgrids (with error threshold optimization)
    subgrid_required = wp.zeros(total_subgrids, dtype=wp.int32, device=device)
    subgrid_min = wp.zeros(total_subgrids, dtype=float, device=device)
    subgrid_max = wp.zeros(total_subgrids, dtype=float, device=device)

    wp.launch(
        mark_required_subgrids_kernel,
        dim=total_subgrids,
        inputs=[
            dense_sdf,
            background_sdf,
            subgrid_required,
            subgrid_min,
            subgrid_max,
            subgrid_size,
            dense_size_x,
            dense_size_y,
            dense_size_z,
            w,
            h,
            d,
            narrow_band_thickness,
            error_threshold,
        ],
        device=device,
    )
    wp.synchronize()

    # Exclusive scan to assign sequential addresses to required subgrids
    # This matches PhysX's scan.exclusiveScan(subgridAddressesD, stream)
    subgrid_addresses = wp.zeros(total_subgrids, dtype=wp.int32, device=device)
    wp._src.utils.array_scan(subgrid_required, subgrid_addresses, inclusive=False)
    wp.synchronize()

    # Count required subgrids and get global min/max for quantization
    required_np = subgrid_required.numpy()
    subgrid_min_np = subgrid_min.numpy()
    subgrid_max_np = subgrid_max.numpy()

    num_required = int(np.sum(required_np))
    print(f"  Required subgrids: {num_required} / {total_subgrids}")

    # Compute global min/max SDF values across all required subgrids (for quantization)
    required_mask = required_np > 0
    if np.any(required_mask):
        global_sdf_min = float(np.min(subgrid_min_np[required_mask]))
        global_sdf_max = float(np.max(subgrid_max_np[required_mask]))
    else:
        global_sdf_min = 0.0
        global_sdf_max = 1.0

    sdf_range = global_sdf_max - global_sdf_min
    if sdf_range < 1e-10:
        sdf_range = 1.0  # Avoid division by zero

    print(f"  Subgrid SDF range: [{global_sdf_min:.6f}, {global_sdf_max:.6f}]")

    if num_required == 0:
        # No subgrids needed
        subgrid_start_slots = np.full(total_subgrids, 0xFFFFFFFF, dtype=np.uint32)
        subgrid_texture_data = np.zeros((1, 1, 1), dtype=np.float32)
        tex_size = 1
        # For no subgrids, use identity scale
        final_sdf_min = 0.0
        final_sdf_range = 1.0
    else:
        # Compute CUBIC 3D texture block layout (matches PhysX cubic root approach)
        # Enforce cubic texture for correct sampling with single tex_size
        cubic_root = num_required ** (1.0 / 3.0)
        up = max(1, int(np.ceil(cubic_root)))

        # Make texture cubic by using 'up' for all dimensions
        tex_blocks_per_dim = up
        while tex_blocks_per_dim**3 < num_required:
            tex_blocks_per_dim += 1

        samples_per_dim = subgrid_size + 1
        tex_size = tex_blocks_per_dim * samples_per_dim

        print(
            f"  Subgrid texture layout: {tex_blocks_per_dim}x{tex_blocks_per_dim}x{tex_blocks_per_dim} blocks (cubic)"
        )
        print(f"  Subgrid texture size: {tex_size}x{tex_size}x{tex_size}")
        print(f"  Quantization mode: {['', 'UINT8', 'UINT16', '', 'FLOAT32'][quantization_mode]}")

        # Initialize start slots to 0xFFFFFFFF
        subgrid_start_slots = np.full(total_subgrids, 0xFFFFFFFF, dtype=np.uint32)
        subgrid_start_slots_gpu = wp.array(subgrid_start_slots, dtype=wp.uint32, device=device)

        # Allocate and populate subgrid texture based on quantization mode
        total_tex_samples = tex_size * tex_size * tex_size
        samples_per_subgrid = samples_per_dim**3
        total_work = total_subgrids * samples_per_subgrid

        sdf_range_inv = 1.0 / sdf_range

        if quantization_mode == QuantizationMode.FLOAT32:
            subgrid_texture_gpu = wp.zeros(total_tex_samples, dtype=float, device=device)
            wp.launch(
                populate_subgrid_texture_float32_kernel,
                dim=total_work,
                inputs=[
                    dense_sdf,
                    subgrid_required,
                    subgrid_addresses,
                    subgrid_start_slots_gpu,
                    subgrid_texture_gpu,
                    subgrid_size,
                    dense_size_x,
                    dense_size_y,
                    w,
                    h,
                    d,
                    tex_blocks_per_dim,
                    tex_size,
                ],
                device=device,
            )
            # For float32, we store raw values - no quantization scale needed
            final_sdf_min = 0.0
            final_sdf_range = 1.0
            subgrid_texture_data = subgrid_texture_gpu.numpy().reshape((tex_size, tex_size, tex_size))
            subgrid_texture_data = subgrid_texture_data.astype(np.float32)

        elif quantization_mode == QuantizationMode.UINT16:
            subgrid_texture_gpu = wp.zeros(total_tex_samples, dtype=wp.uint16, device=device)
            wp.launch(
                populate_subgrid_texture_uint16_kernel,
                dim=total_work,
                inputs=[
                    dense_sdf,
                    subgrid_required,
                    subgrid_addresses,
                    subgrid_start_slots_gpu,
                    subgrid_texture_gpu,
                    subgrid_size,
                    dense_size_x,
                    dense_size_y,
                    w,
                    h,
                    d,
                    tex_blocks_per_dim,
                    tex_size,
                    global_sdf_min,
                    sdf_range_inv,
                ],
                device=device,
            )
            final_sdf_min = global_sdf_min
            final_sdf_range = sdf_range
            # Convert uint16 to float32 normalized for texture
            uint16_data = subgrid_texture_gpu.numpy().reshape((tex_size, tex_size, tex_size))
            subgrid_texture_data = uint16_data.astype(np.float32) / 65535.0

        elif quantization_mode == QuantizationMode.UINT8:
            subgrid_texture_gpu = wp.zeros(total_tex_samples, dtype=wp.uint8, device=device)
            wp.launch(
                populate_subgrid_texture_uint8_kernel,
                dim=total_work,
                inputs=[
                    dense_sdf,
                    subgrid_required,
                    subgrid_addresses,
                    subgrid_start_slots_gpu,
                    subgrid_texture_gpu,
                    subgrid_size,
                    dense_size_x,
                    dense_size_y,
                    w,
                    h,
                    d,
                    tex_blocks_per_dim,
                    tex_size,
                    global_sdf_min,
                    sdf_range_inv,
                ],
                device=device,
            )
            final_sdf_min = global_sdf_min
            final_sdf_range = sdf_range
            # Convert uint8 to float32 normalized for texture
            uint8_data = subgrid_texture_gpu.numpy().reshape((tex_size, tex_size, tex_size))
            subgrid_texture_data = uint8_data.astype(np.float32) / 255.0

        else:
            raise ValueError(f"Unknown quantization mode: {quantization_mode}")

        wp.synchronize()

        # Copy start slots back
        subgrid_start_slots = subgrid_start_slots_gpu.numpy()

    # Convert background SDF to numpy for texture
    background_sdf_np = background_sdf.numpy().reshape((bg_size_z, bg_size_y, bg_size_x))

    return {
        "coarse_sdf": background_sdf_np.astype(np.float32),
        "subgrid_data": subgrid_texture_data.astype(np.float32),
        "subgrid_start_slots": subgrid_start_slots,
        "coarse_dims": (w, h, d),
        "subgrid_tex_size": tex_size,  # Cubic texture size
        "num_subgrids": num_required,
        "min_extents": min_corner,
        "max_extents": max_corner,
        "cell_size": cell_size,
        "subgrid_size": subgrid_size,
        "quantization_mode": quantization_mode,
        "subgrids_min_sdf_value": final_sdf_min,
        "subgrids_sdf_value_range": final_sdf_range,
    }


def create_sparse_sdf_textures(
    sparse_data: dict,
    device: str = "cuda",
) -> tuple[wp.Texture3D, wp.Texture3D, wp.array, SparseSDF]:
    """
    Create GPU textures and SparseSDF struct from sparse data.

    The subgrid texture contains normalized values [0, 1] for quantized modes,
    or raw SDF values for float32 mode. The SparseSDF struct contains the
    scale parameters needed to convert back to actual SDF distances.
    """
    coarse_tex = wp.Texture3D(
        sparse_data["coarse_sdf"],
        filter_mode=wp.Texture3D.LINEAR,
        address_mode=wp.Texture3D.CLAMP,
        device=device,
    )

    subgrid_tex = wp.Texture3D(
        sparse_data["subgrid_data"],
        filter_mode=wp.Texture3D.LINEAR,
        address_mode=wp.Texture3D.CLAMP,
        device=device,
    )

    subgrid_slots = wp.array(sparse_data["subgrid_start_slots"], dtype=wp.uint32, device=device)

    # Create SparseSDF struct
    avg_spacing = np.mean(sparse_data["cell_size"])
    coarse_x = sparse_data["coarse_dims"][0]
    coarse_y = sparse_data["coarse_dims"][1]
    coarse_z = sparse_data["coarse_dims"][2]
    subgrid_tex_size = float(sparse_data["subgrid_tex_size"])

    sdf_params = SparseSDF()
    sdf_params.sdf_box_lower = wp.vec3(
        sparse_data["min_extents"][0],
        sparse_data["min_extents"][1],
        sparse_data["min_extents"][2],
    )
    sdf_params.sdf_box_upper = wp.vec3(
        sparse_data["max_extents"][0],
        sparse_data["max_extents"][1],
        sparse_data["max_extents"][2],
    )
    sdf_params.sdf_dx = avg_spacing
    sdf_params.inv_sdf_dx = 1.0 / avg_spacing
    sdf_params.inv_2dx = 0.5 / avg_spacing  # For gradient central differences
    sdf_params.coarse_size_x = coarse_x
    sdf_params.coarse_size_y = coarse_y
    sdf_params.coarse_size_z = coarse_z
    sdf_params.subgrid_size = sparse_data["subgrid_size"]
    sdf_params.subgrid_size_f = float(sparse_data["subgrid_size"])
    sdf_params.subgrid_samples_f = float(sparse_data["subgrid_size"] + 1)
    sdf_params.fine_to_coarse = 1.0 / sparse_data["subgrid_size"]

    # Precomputed inverse texture sizes (avoid divisions in sampling)
    sdf_params.inv_coarse_tex_size_x = 1.0 / float(coarse_x + 1)
    sdf_params.inv_coarse_tex_size_y = 1.0 / float(coarse_y + 1)
    sdf_params.inv_coarse_tex_size_z = 1.0 / float(coarse_z + 1)
    sdf_params.inv_subgrid_tex_size = 1.0 / subgrid_tex_size

    # Quantization parameters for subgrid values
    sdf_params.subgrids_min_sdf_value = sparse_data["subgrids_min_sdf_value"]
    sdf_params.subgrids_sdf_value_range = sparse_data["subgrids_sdf_value_range"]

    return coarse_tex, subgrid_tex, subgrid_slots, sdf_params
