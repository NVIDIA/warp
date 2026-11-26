# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nonoverlapping adaptive integration cells resolving every active spline knot."""

import warp as wp
import warp.fem as fem

from .field import basis_key, find_key
from .keyed import coordinates, unique_keys


@wp.kernel
def _basis_support_kernel(keys: wp.array[wp.int64], d: int, out: wp.array[wp.int64]):
    i, k = wp.tid()
    c = coordinates(keys[i]) - wp.vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1)
    out[i * 8 + k] = wp.int64(0x7FFFFFFFFFFFFFFF)
    if wp.min(c) >= 0 and wp.max(c) < 1 << d:
        out[i * 8 + k] = basis_key(d, c[0], c[1], c[2])


@wp.kernel
def _normal_support_kernel(keys: wp.array[wp.int64], d: int, offset: int, out: wp.array[wp.int64]):
    i, k = wp.tid()
    idx = offset + i * 27 + k
    out[idx] = wp.int64(0x7FFFFFFFFFFFFFFF)
    if int(keys[i] >> wp.int64(48)) == d:
        c = coordinates(keys[i]) + wp.vec3i(k // 9 - 1, (k // 3) % 3 - 1, k % 3 - 1)
        if wp.min(c) >= 0 and wp.max(c) < 1 << d:
            out[idx] = basis_key(d, c[0], c[1], c[2])


@wp.kernel
def _parent_kernel(keys: wp.array[wp.int64], d: int, offset: int, out: wp.array[wp.int64]):
    i = wp.tid()
    c = coordinates(keys[i])
    out[offset + i] = basis_key(d, c[0] // 2, c[1] // 2, c[2] // 2)


@wp.kernel
def _children_kernel(parents: wp.array[wp.int64], refined: wp.array[wp.int64], d: int, out: wp.array[wp.int64]):
    i, k = wp.tid()
    c = coordinates(parents[i]) * 2 + wp.vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1)
    key = basis_key(d, c[0], c[1], c[2])
    out[i * 8 + k] = key
    if find_key(refined, key) >= 0:
        out[i * 8 + k] = wp.int64(0x7FFFFFFFFFFFFFFF)


@wp.kernel
def _pack_kernel(
    keys: wp.array[wp.int64], shift: int, offset: int, xyz: wp.array[wp.vec3i], levels: wp.array[wp.uint8]
):
    i = wp.tid()
    c = coordinates(keys[i])
    xyz[offset + i] = c * (1 << shift)
    levels[offset + i] = wp.uint8(shift)


@wp.kernel
def _reorder_kernel(xyz: wp.array[wp.vec3i], levels: wp.array[wp.uint8], volume: wp.uint64, out: wp.array[wp.uint8]):
    i = wp.tid()
    c = xyz[i]
    index = wp.volume_lookup_index(volume, c[0], c[1], c[2])
    out[index] = levels[i]


def integration_grid(tree, normals, depth, dtype=wp.float32):
    """Resolve basis knots through depth, and optionally the normal-field knots."""
    device = tree.basis[0].keys.device
    required = [None] * (depth + 1)
    refined = [None] * (depth + 1)
    empty = wp.empty(0, dtype=wp.int64, device=device)
    for d in range(depth, -1, -1):
        records = None
        if normals is not None:
            records = normals.levels[d] if normals.levels is not None else normals.records
        normal_count = len(records.keys) if records is not None else 0
        fine = required[d + 1].keys if d < depth else empty
        keys = wp.empty(2 * len(fine), dtype=wp.int64, device=device)
        if len(fine):
            wp.launch(_parent_kernel, len(fine), inputs=[fine, d, 0, keys], device=device)
        refined[d] = unique_keys(keys, len(fine))
        basis = tree.basis[d].keys
        count = 8 * len(basis) + 27 * normal_count + len(refined[d].keys)
        keys = wp.empty(2 * count, dtype=wp.int64, device=device)
        if len(basis):
            wp.launch(_basis_support_kernel, (len(basis), 8), inputs=[basis, d, keys], device=device)
        if normal_count:
            wp.launch(
                _normal_support_kernel,
                (normal_count, 27),
                inputs=[records.keys, d, 8 * len(basis), keys],
                device=device,
            )
        wp.copy(keys, refined[d].keys, dest_offset=count - len(refined[d].keys), count=len(refined[d].keys))
        required[d] = unique_keys(keys, count)
    leaves = []
    # The degree-1 complete root basis always resolves the entire reconstruction cube.
    for d in range(depth + 1):
        if d == 0:
            keys = wp.zeros(2, dtype=wp.int64, device=device)
            if len(refined[0].keys):
                leaves.append(empty)
            else:
                leaves.append(keys[:1])
        else:
            parents = refined[d - 1].keys
            n = 8 * len(parents)
            keys = wp.empty(2 * n, dtype=wp.int64, device=device)
            if n:
                wp.launch(
                    _children_kernel, (len(parents), 8), inputs=[parents, refined[d].keys, d, keys], device=device
                )
            leaves.append(unique_keys(keys, n).keys)
    count = sum(len(l) for l in leaves)
    xyz = wp.empty(count, dtype=wp.vec3i, device=device)
    levels = wp.empty(count, dtype=wp.uint8, device=device)
    offset = 0
    for d, keys in enumerate(leaves):
        if len(keys):
            wp.launch(_pack_kernel, len(keys), inputs=[keys, depth - d, offset, xyz, levels], device=device)
        offset += len(keys)
    h = 1.0 / (1 << depth)
    volume = wp.Volume.allocate_by_voxels(xyz, voxel_size=h, translation=(h / 2, h / 2, h / 2), device=device)
    reordered = wp.empty(count, dtype=wp.uint8, device=device)
    wp.launch(_reorder_kernel, count, inputs=[xyz, levels, volume.id, reordered], device=device)
    coarsest = min(d for d, keys in enumerate(leaves) if len(keys))
    return fem.AdaptiveNanogrid(
        volume, cell_level=reordered, level_count=depth - coarsest + 1, temporary_store=None, scalar_type=dtype
    )


@wp.kernel
def _level_cells_kernel(keys: wp.array[wp.int64], depth: int, xyz: wp.array[wp.vec3i], mask: wp.array[int]):
    i, k = wp.tid()
    c = coordinates(keys[i]) - wp.vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1)
    xyz[i * 8 + k] = c
    mask[i * 8 + k] = int(wp.min(c) >= 0 and wp.max(c) < 1 << depth)


def level_integration_grid(tree, depth, dtype=wp.float32):
    """Sparse cells touching the level's hats, for its remaining FEM forms.

    Normal and stiffness stencils integrate analytically. Point forms need only
    this level's support cells, not a covering multilevel partition of the cube.
    The original adaptive partition remains available for verification.
    """
    basis = tree.basis[depth].keys
    if not len(basis):
        # Sparse inputs may stop refining before the requested maximum depth.
        # Keep a valid covering grid for that empty terminal level.
        return integration_grid(tree, None, depth, dtype=dtype)
    device = basis.device
    xyz = wp.empty(8 * len(basis), dtype=wp.vec3i, device=device)
    mask = wp.empty(len(xyz), dtype=int, device=device)
    wp.launch(_level_cells_kernel, (len(basis), 8), inputs=[basis, depth, xyz, mask], device=device)
    h = 1.0 / (1 << depth)
    # NanoVDB already deduplicates voxels. Pass candidates and their validity
    # mask directly instead of sorting/compacting the same cells beforehand.
    volume = wp.Volume.allocate_by_voxels(
        xyz, voxel_size=h, translation=(h / 2, h / 2, h / 2), point_mask=mask, device=device
    )
    return fem.Nanogrid(volume, temporary_store=None, scalar_type=dtype)
