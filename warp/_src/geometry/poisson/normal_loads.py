# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference multilevel normal constraints with verified tensor-product stencils.

The public Warp FEM quadrature route remains available via stencil_loads=False.

Coarse normal coefficients are prolonged only onto existing tree functions.
The pinned reference uses the degree-one test-space transfer for these degree-two
coefficients, an observable compatibility quirk.
Fine loads are restricted through the scalar test spaces. Summing the original
normal levels as a global continuous field misses these adaptive truncations.
"""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp
import warp.sparse as sparse
from warp.fem.linalg import array_axpy

from .field import basis_key, find_key
from .integration import integration_grid
from .keyed import coordinates
from .normal_field import assemble_normal_load, cache_normal_quadrature
from .normal_stencils import assemble_normal_stencil
from .spaces import make_level_space

wp.set_module_options({"fuse_fp": False})


@lru_cache(None)
def prolongation_kernel(dtype):
    vec4 = wp.types.vector(length=4, dtype=dtype)

    @wp.kernel
    def prolong_kernel(
        keys: wp.array[wp.int64],
        previous_keys: wp.array[wp.int64],
        previous: wp.array[vec4],
        source_keys: wp.array[wp.int64],
        source: wp.array[vec4],
        depth: int,
        out: wp.array[vec4],
    ):
        i = wp.tid()
        c = coordinates(keys[i])
        value = vec4(dtype(0.0))
        own = find_key(source_keys, keys[i])
        if own >= 0:
            value = source[own]
        if depth > 0:
            for corner in range(8):
                p = wp.vec3i(c[0] // 2, c[1] // 2, c[2] // 2)
                weight = dtype(1.0)
                for axis in range(3):
                    if (corner >> axis) & 1:
                        p[axis] += 1
                    weight *= dtype(1.0) - wp.abs(dtype(c[axis]) * dtype(0.5) - dtype(p[axis]))
                if weight > dtype(0.0):
                    j = find_key(previous_keys, basis_key(depth - 1, p[0], p[1], p[2]))
                    if j >= 0:
                        value += previous[j] * weight
        out[i] = value

    return prolong_kernel


def assemble_reference_normal_loads(
    tree,
    geometry,
    normals,
    max_depth,
    base_depth,
    transfers,
    operator_geometries=None,
    level_partitions=True,
    ordered_root=True,
    stencil_loads=True,
):
    device = normals.records.keys.device
    dtype = wp.float64 if wp.types.types_equal(normals.average.dtype, wp.vec3d) else wp.float32
    if base_depth == 0 and dtype == wp.float32 and ordered_root:
        from .ordered_normal_loads import (  # noqa: PLC0415
            ordered_normal_loads,
        )

        rounded = ordered_normal_loads(tree, normals, max_depth, base_depth)
        result = []
        for load in rounded:
            output = wp.empty(len(load), dtype=wp.float64, device=device)
            wp.utils.array_cast(load, output)
            result.append(output)
        return result
    vec4 = normals.records.values.dtype
    finest = len(tree.basis) - 1

    if operator_geometries is None:
        operator_geometries = {finest: geometry}

    def space(grid, d, resolution):
        return make_level_space(grid, tree.basis[d].keys, d, resolution)

    same = [None] * (max_depth + 1)
    from_child = [None] * (max_depth + 1)
    from_parent = [None] * (max_depth + 1)
    previous_keys = wp.empty(0, dtype=wp.int64, device=device)
    previous = wp.empty(0, dtype=vec4, device=device)
    has_cumulative = False
    for d in range(max_depth + 1):
        records = normals.levels[d] if normals.levels is not None else normals.records
        own = SimpleNamespace(records=records, average=normals.average)
        has_cumulative = has_cumulative or bool(len(records.keys))
        # The reference begins normal splatting at base_depth.
        if d >= base_depth and stencil_loads:
            same[d] = assemble_normal_stencil(tree.basis[d].keys, records, d, d)
            if d > base_depth:
                from_child[d - 1] = assemble_normal_stencil(tree.basis[d - 1].keys, records, d - 1, d)
        elif d >= base_depth and len(records.keys):
            grid, resolution = geometry, finest
            if level_partitions and d < finest:
                grid, resolution = integration_grid(tree, normals, d, dtype=dtype), d
            cache = cache_normal_quadrature(grid, own, d, first_depth=d)
            same[d] = assemble_normal_load(space(grid, d, resolution), normals, d, cache=cache)
            if d > base_depth:
                from_child[d - 1] = assemble_normal_load(space(grid, d - 1, resolution), normals, d, cache=cache)
            del cache
            del grid
        elif d >= base_depth:
            same[d] = wp.zeros(len(tree.basis[d].keys), dtype=wp.float64, device=device)
            if d > base_depth:
                from_child[d - 1] = wp.zeros(len(tree.basis[d - 1].keys), dtype=wp.float64, device=device)
        if d < max_depth:
            keys = tree.basis[d].keys
            cumulative = wp.empty(len(keys), dtype=vec4, device=device)
            wp.launch(
                prolongation_kernel(dtype),
                len(keys),
                inputs=[
                    keys,
                    previous_keys,
                    previous,
                    records.keys,
                    records.values,
                    d,
                    cumulative,
                ],
                device=device,
            )
            if d >= base_depth and has_cumulative:
                field = SimpleNamespace(
                    records=SimpleNamespace(keys=keys, values=cumulative),
                    average=normals.average,
                )
                grid, resolution = geometry, finest
                if stencil_loads:
                    from_parent[d + 1] = assemble_normal_stencil(tree.basis[d + 1].keys, field.records, d + 1, d)
                elif level_partitions:
                    if d + 1 not in operator_geometries:
                        operator_geometries[d + 1] = integration_grid(tree, None, d + 1, dtype=dtype)
                    grid, resolution = operator_geometries[d + 1], d + 1
                if not stencil_loads:
                    cache = cache_normal_quadrature(grid, field, d, first_depth=d)
                    from_parent[d + 1] = assemble_normal_load(space(grid, d + 1, resolution), field, d, cache=cache)
                    del cache
                del grid
            previous_keys, previous = keys, cumulative
    finer = wp.zeros_like(same[max_depth])
    loads = [None] * (max_depth + 1)
    for d in range(max_depth, base_depth - 1, -1):
        load = same[d]
        array_axpy(finer, load)
        if from_parent[d] is not None:
            array_axpy(from_parent[d], load)
        loads[d] = load
        if d > base_depth:
            next_finer = from_child[d - 1]
            transfer = sparse.bsr_copy(transfers[d - 1], scalar_type=wp.float64)
            sparse.bsr_mv(sparse.bsr_transposed(transfer), finer, next_finer, beta=1.0)
            finer = next_finer
    return loads
