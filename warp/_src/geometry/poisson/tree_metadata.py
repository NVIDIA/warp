# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-demand GPU topology diagnostics, including pruned nodes and padding.

Reconstruction needs the active support sets in LogicalTree. This diagnostic
expands the retained input/density/normal broods as well, without importing any
reference topology or adding work to ordinary reconstruction calls.
"""

from dataclasses import dataclass

import warp as wp

from .field import find_key
from .keyed import coordinates, unique_keys
from .tree import _parent_keys_kernel as _parent_keys
from .tree import parent, tree_coordinates, tree_key


@wp.kernel
def _broods_kernel(keys: wp.array[wp.int64], depth: int, field_keys: int, offset: int, out: wp.array[wp.int64]):
    i, k = wp.tid()
    c = tree_coordinates(keys[i])
    valid = True
    if field_keys:
        c = coordinates(keys[i])
        valid = int(keys[i] >> wp.int64(48)) == depth
    out[offset + i * 8 + k] = wp.int64(0x7FFFFFFFFFFFFFFF)
    if valid:
        c = parent(c) * 2 + wp.vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1)
        out[offset + i * 8 + k] = tree_key(c)


@wp.kernel
def _global_keys_kernel(local: wp.array[wp.int64], depth: int, out: wp.array[wp.int64]):
    i = wp.tid()
    out[i] = tree_key(tree_coordinates(local[i]) + wp.vec3i(1 << (depth + 1)))


@wp.kernel
def _padding_kernel(keys: wp.array[wp.int64]):
    i = wp.tid()
    keys[i] = tree_key(wp.vec3i(i // 4, (i // 2) % 2, i % 2))


@wp.kernel
def _links_kernel(
    keys: wp.array[wp.int64],
    parent_keys: wp.array[wp.int64],
    child_keys: wp.array[wp.int64],
    active: wp.array[wp.int64],
    active_child: wp.array[wp.int64],
    depth: int,
    flags: wp.array[wp.uint8],
    ghost: wp.array[wp.uint8],
    parents: wp.array[int],
    children: wp.array2d[int],
):
    i = wp.tid()
    c = tree_coordinates(keys[i])
    parents[i] = find_key(parent_keys, tree_key(parent(c)))
    is_ghost = depth == -2
    bits = int(0)
    if depth >= 0:
        local = c - wp.vec3i(1 << (depth + 1))
        is_ghost = find_key(active, tree_key(local)) < 0
        if not is_ghost and wp.min(local) >= 0:
            if wp.max(local) < 1 << depth:
                bits |= 1 | 4  # SPACE and quadratic Dirichlet FEM validity
            if wp.max(local) <= 1 << depth:
                bits |= 2  # Degree-one Neumann FEM validity
    has_child = False
    child_is_active = depth < -1
    for k in range(8):
        q = c * 2 + wp.vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1)
        child = find_key(child_keys, tree_key(q))
        children[i, k] = child
        if child >= 0:
            has_child = True
            if depth >= -1:
                local = q - wp.vec3i(1 << (depth + 2))
                child_is_active = child_is_active or find_key(active_child, tree_key(local)) >= 0
    if has_child and not child_is_active:
        bits |= 64  # The reference stores the children's ghost flag here.
    flags[i] = wp.uint8(bits)
    ghost[i] = wp.uint8(is_ghost)


@dataclass
class MetadataLevel:
    local_depth: int
    keys: wp.array  # Global lattice coordinates, encoded with tree_key.
    flags: wp.array
    ghost: wp.array
    parents: wp.array  # Indices into the preceding level; -1 for the root.
    children: wp.array  # Next-level indices, in reference x-bit-first order.


@dataclass
class TreeMetadata:
    levels: list
    host_sync_seconds: float
    depth_offset: int = 2


def build_tree_metadata(tree, samples, density, normals):
    """Materialize all retained broods and links on GPU for the pinned profile.

    Global coordinates uniquely identify the two negative local-depth padding
    levels, where the reference's ordinary local-coordinate accessor returns -1.
    """
    depth = len(tree.levels) - 1
    device = samples.keys.device
    empty = wp.empty(0, dtype=wp.int64, device=device)
    raw = [None] * (depth + 1)
    sync = 0.0
    for d in range(depth, -1, -1):
        fine = raw[d + 1] if d < depth else empty
        scratch = wp.empty(2 * len(fine), dtype=wp.int64, device=device)
        wp.launch(_parent_keys, len(fine), inputs=[fine, scratch], device=device)
        parents = unique_keys(scratch, len(fine))
        sync += parents.sync_seconds
        sources = []
        if d == depth:
            sources.append(samples.keys)
        if d < len(density.levels):
            sources.append(density.levels[d].keys)
        sources.append(normals.levels[d].keys if normals.levels is not None else normals.records.keys)
        active = tree.levels[d].keys
        count = len(active) + 8 * (len(parents.keys) + sum(len(k) for k in sources))
        candidates = wp.empty(2 * count, dtype=wp.int64, device=device)
        wp.copy(candidates, active, count=len(active))
        offset = len(active)
        wp.launch(
            _broods_kernel, (len(parents.keys), 8), inputs=[parents.keys, d, 0, offset, candidates], device=device
        )
        offset += 8 * len(parents.keys)
        for keys in sources:
            wp.launch(_broods_kernel, (len(keys), 8), inputs=[keys, d, 1, offset, candidates], device=device)
            offset += 8 * len(keys)
        records = unique_keys(candidates, count)
        raw[d] = records.keys
        sync += records.sync_seconds
    keys = [wp.empty(n, dtype=wp.int64, device=device) for n in (1, 8)]
    for padding in keys:
        wp.launch(_padding_kernel, len(padding), inputs=[padding], device=device)
    for d, local in enumerate(raw):
        global_keys = wp.empty_like(local)
        wp.launch(_global_keys_kernel, len(local), inputs=[local, d, global_keys], device=device)
        keys.append(global_keys)
    levels = []
    for g, level_keys in enumerate(keys):
        d = g - 2
        count = len(level_keys)
        level = MetadataLevel(
            d,
            level_keys,
            wp.empty(count, dtype=wp.uint8, device=device),
            wp.empty(count, dtype=wp.uint8, device=device),
            wp.empty(count, dtype=int, device=device),
            wp.empty((count, 8), dtype=int, device=device),
        )
        wp.launch(
            _links_kernel,
            count,
            inputs=[
                level_keys,
                keys[g - 1] if g else empty,
                keys[g + 1] if g + 1 < len(keys) else empty,
                tree.levels[d].keys if d >= 0 else empty,
                tree.levels[d + 1].keys if -1 <= d < depth else empty,
                d,
                level.flags,
                level.ghost,
                level.parents,
                level.children,
            ],
            device=device,
        )
        levels.append(level)
    return TreeMetadata(levels, sync)
