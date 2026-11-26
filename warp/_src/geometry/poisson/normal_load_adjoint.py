# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional replay of the production multilevel normal-load assembly."""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp
import warp.sparse as sparse

from . import ordered_normal_loads as ordered
from .field import basis_key, find_key
from .keyed import coordinates
from .normal_loads import prolongation_kernel
from .normal_stencils import _integrals_kernel as _integrals
from .normal_stencils import load_kernel
from .solver_adjoint import functional_spmv

wp.set_module_options({"fuse_fp": False})


@lru_cache(None)
def precision_kernels(dtype):
    @wp.kernel
    def combine_kernel(a: wp.array[dtype], b: wp.array[dtype], out: wp.array[dtype]):
        i = wp.tid()
        out[i] = a[i] + b[i]

    @wp.kernel
    def cast_kernel(a: wp.array[dtype], out: wp.array[wp.float64]):
        i = wp.tid()
        out[i] = wp.float64(a[i])

    @wp.kernel
    def restrict_kernel(
        fine_keys: wp.array[wp.int64],
        fine: wp.array[dtype],
        coarse_keys: wp.array[wp.int64],
        depth: int,
        initial: wp.array[dtype],
        out: wp.array[dtype],
    ):
        i = wp.tid()
        c = coordinates(coarse_keys[i])
        value = initial[i]
        for k in range(27):
            off = wp.vec3i(k // 9 - 1, (k // 3) % 3 - 1, k % 3 - 1)
            n = 2 * c + off
            if wp.min(n) >= 0 and wp.max(n) <= 1 << depth:
                j = find_key(fine_keys, basis_key(depth, n[0], n[1], n[2]))
                if j >= 0:
                    weight = dtype(1)
                    for axis in range(3):
                        if off[axis] != 0:
                            weight *= dtype(0.5)
                    value += fine[j] * weight
        out[i] = value

    return SimpleNamespace(combine=combine_kernel, cast=cast_kernel, restrict=restrict_kernel)


def _sum(a, b):
    out = wp.empty_like(a, requires_grad=True)
    wp.launch(precision_kernels(a.dtype).combine, len(a), inputs=[a, b, out], device=a.device)
    return out


def _stencil(keys, records, test_depth, normal_depth):
    out = wp.zeros(len(keys), dtype=wp.float64, device=keys.device, requires_grad=True)
    if not len(records.keys):
        return out
    width, first = (6, -3) if test_depth < normal_depth else (4, -2)
    table = wp.empty(((1 << test_depth) + 1, width), dtype=wp.vec2d, device=keys.device)
    wp.launch(
        _integrals, table.shape, inputs=[test_depth, normal_depth, first, table], device=keys.device, record_tape=False
    )
    wp.launch(
        load_kernel(records.values.dtype),
        len(keys),
        inputs=[keys, records.keys, records.values, test_depth, normal_depth, first, width, table, out],
        device=keys.device,
    )
    return out


def _ordered(tree, normals, max_depth):
    device, dtype = normals.average.device, wp.float32
    same, parent, _, prolong, from_coarse = ordered.kernels(dtype)
    loads = [None] * (max_depth + 1)
    finer = wp.zeros(len(tree.basis[max_depth].keys), dtype=dtype, device=device)
    for d in range(max_depth, -1, -1):
        keys, own = tree.basis[d].keys, normals.levels[d]
        loads[d] = wp.empty_like(finer, requires_grad=True)
        wp.launch(same, len(keys), inputs=[keys, own.keys, own.values, finer, d, loads[d]], device=device)
        if d:
            coarse = tree.basis[d - 1].keys
            next_finer = wp.zeros(len(coarse), dtype=dtype, device=device, requires_grad=True)
            if len(own.keys):
                count = len(coarse) * 216
                orders = wp.empty(count * 2, dtype=int, device=device)
                indices = wp.empty_like(orders)
                segments = wp.empty(len(coarse) * 6 + 1, dtype=int, device=device)
                wp.launch(
                    ordered.parent_order_kernel,
                    (len(coarse), 216),
                    inputs=[coarse, orders, indices],
                    device=device,
                    record_tape=False,
                )
                wp.launch(
                    ordered.segment_offsets_kernel, len(segments), inputs=[segments], device=device, record_tape=False
                )
                wp.utils.segmented_sort_pairs(orders, indices, count, segments)
                wp.launch(
                    parent,
                    len(coarse),
                    inputs=[coarse, own.keys, own.values, indices, d - 1, next_finer],
                    device=device,
                )
            if d < max_depth:
                out = wp.empty_like(next_finer, requires_grad=True)
                wp.launch(
                    precision_kernels(dtype).restrict,
                    len(coarse),
                    inputs=[keys, finer, coarse, d, next_finer, out],
                    device=device,
                )
                next_finer = out
            finer = next_finer
    previous_keys = wp.empty(0, dtype=wp.int64, device=device)
    previous = wp.empty(0, dtype=normals.records.values.dtype, device=device)
    for d in range(max_depth):
        keys, own = tree.basis[d].keys, normals.levels[d]
        cumulative = wp.empty(len(keys), dtype=normals.records.values.dtype, device=device, requires_grad=True)
        wp.launch(
            prolong,
            len(keys),
            inputs=[keys, own.keys, own.values, previous_keys, previous, d, cumulative],
            device=device,
        )
        # from_coarse only adds to zero here; preserve its final addition order.
        contribution = wp.zeros(len(loads[d + 1]), dtype=dtype, device=device, requires_grad=True)
        wp.launch(
            from_coarse,
            len(contribution),
            inputs=[tree.basis[d + 1].keys, keys, cumulative, d + 1, contribution],
            device=device,
        )
        loads[d + 1] = _sum(loads[d + 1], contribution)
        previous_keys, previous = keys, cumulative
    result = []
    for load in loads:
        out = wp.empty(len(load), dtype=wp.float64, device=device, requires_grad=True)
        wp.launch(precision_kernels(dtype).cast, len(load), inputs=[load, out], device=device)
        result.append(out)
    return result


def prepare_normal_restrictions(transfers):
    """Build constant double restrictions before entering any replay tape."""
    return [sparse.bsr_transposed(sparse.bsr_copy(p, scalar_type=wp.float64)) for p in transfers]


def replay_normal_loads(tree, normals, max_depth, base_depth, transfers, *, restrictions):
    """Keep primal accumulation order, with immutable intermediates for backward.

    Supports the production stencil route, including the special ordered f32
    root. Tree, normal-record keys and transfer matrices are fixed constants.
    """
    dtype = wp.float64 if normals.average.dtype == wp.vec3d else wp.float32
    if base_depth == 0 and dtype == wp.float32:
        return _ordered(tree, normals, max_depth)
    device, vec4 = normals.average.device, normals.records.values.dtype
    same, child, parent = ([None] * (max_depth + 1) for _ in range(3))
    previous_keys = wp.empty(0, dtype=wp.int64, device=device)
    previous = wp.empty(0, dtype=vec4, device=device)
    has_cumulative = False
    for d in range(max_depth + 1):
        records = normals.levels[d]
        has_cumulative = has_cumulative or bool(len(records.keys))
        if d >= base_depth:
            same[d] = _stencil(tree.basis[d].keys, records, d, d)
            if d > base_depth:
                child[d - 1] = _stencil(tree.basis[d - 1].keys, records, d - 1, d)
        if d < max_depth:
            keys = tree.basis[d].keys
            cumulative = wp.empty(len(keys), dtype=vec4, device=device, requires_grad=True)
            wp.launch(
                prolongation_kernel(dtype),
                len(keys),
                inputs=[keys, previous_keys, previous, records.keys, records.values, d, cumulative],
                device=device,
            )
            if d >= base_depth and has_cumulative:
                parent[d + 1] = _stencil(
                    tree.basis[d + 1].keys, SimpleNamespace(keys=keys, values=cumulative), d + 1, d
                )
            previous_keys, previous = keys, cumulative
    finer = wp.zeros_like(same[max_depth])
    loads = [None] * (max_depth + 1)
    for d in range(max_depth, base_depth - 1, -1):
        load = _sum(same[d], finer)
        if parent[d] is not None:
            load = _sum(load, parent[d])
        loads[d] = load
        if d > base_depth:
            finer = functional_spmv(restrictions[d - 1], finer, child[d - 1], beta=1)
    return loads
