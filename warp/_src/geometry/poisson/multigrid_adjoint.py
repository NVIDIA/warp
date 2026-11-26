# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite multigrid replay with fixed Galerkin topology and root branches."""

import math
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import warp as wp
import warp.sparse as sparse

from .relaxation import row_schedule
from .relaxation_adjoint import differentiable_relaxation
from .solver import padded_root_system, root_row
from .solver import precision_kernels as solver_kernels
from .solver_adjoint import finite_root_cg, functional_spmv
from .solver_adjoint import precision_kernels as functional_kernels

wp.set_module_options({"fuse_fp": False})


@wp.func
def lookup(offsets: wp.array[int], columns: wp.array[int], row: int, col: int):
    first = int(offsets[row])
    end = int(offsets[row + 1])
    while first < end:
        mid = (first + end) // 2
        if columns[mid] < col:
            first = mid + 1
        else:
            end = mid
    if first < offsets[row + 1] and columns[first] == col:
        return first
    return -1


@lru_cache(None)
def precision_kernels(dtype):
    @wp.kernel
    def galerkin_backward_kernel(
        fine_offsets: wp.array[int],
        fine_columns: wp.array[int],
        p_offsets: wp.array[int],
        p_columns: wp.array[int],
        p_values: wp.array[dtype],
        coarse_offsets: wp.array[int],
        coarse_columns: wp.array[int],
        grad_coarse: wp.array[dtype],
        grad_fine: wp.array[dtype],
    ):
        i = wp.tid()
        for entry in range(fine_offsets[i], fine_offsets[i + 1]):
            j = fine_columns[entry]
            total = dtype(0)
            for a in range(p_offsets[i], p_offsets[i + 1]):
                for b in range(p_offsets[j], p_offsets[j + 1]):
                    index = lookup(coarse_offsets, coarse_columns, p_columns[a], p_columns[b])
                    if index >= 0:
                        total += p_values[a] * grad_coarse[index] * p_values[b]
            wp.atomic_add(grad_fine, entry, total)

    @wp.kernel
    def unpad_matrix_backward_kernel(
        keys: wp.array[wp.int64],
        offsets: wp.array[int],
        columns: wp.array[int],
        padded_offsets: wp.array[int],
        padded_columns: wp.array[int],
        grad_padded: wp.array[dtype],
        grad_values: wp.array[dtype],
    ):
        i = wp.tid()
        for entry in range(offsets[i], offsets[i + 1]):
            index = lookup(padded_offsets, padded_columns, root_row(keys[i]), root_row(keys[columns[entry]]))
            if index >= 0:
                wp.atomic_add(grad_values, entry, grad_padded[index])

    return SimpleNamespace(
        galerkin_backward=galerkin_backward_kernel, unpad_matrix_backward=unpad_matrix_backward_kernel
    )


@dataclass
class Hierarchy:
    matrices: list
    transfers: list
    restrictions: list
    keys: list
    schedules: list
    padded: object = None


def prepare_hierarchy(matrix, keys, transfers, *, add_dc_term=False):
    """Build primal Galerkin matrices outside a tape, retaining exact sparse ops.

    The caller's matrix must be tracked. All returned matrices own independent
    value/gradient buffers, and must remain immutable until backward finishes.
    """
    if not matrix.values.requires_grad:
        raise ValueError("the finest matrix must have tracked values")
    if len(transfers) != len(keys) - 1 or (add_dc_term and len(keys) != 1):
        raise ValueError("transfer count must match levels; padded constant mode requires base zero")
    base = len(keys) - 1
    matrices = [None] * base + [matrix]
    restrictions = [sparse.bsr_transposed(p) for p in transfers]
    for d in range(base, 0, -1):
        matrices[d - 1] = sparse.bsr_mm(sparse.bsr_mm(restrictions[d - 1], matrices[d]), transfers[d - 1])
        matrices[d - 1].values.requires_grad = True
    padded = None
    if add_dc_term:
        dummy = wp.zeros(len(keys[0]), dtype=matrix.values.dtype, device=keys[0].device)
        padded, _ = padded_root_system(matrix, dummy, keys[0])
        padded.values.requires_grad = True
    return Hierarchy(matrices, transfers, restrictions, keys, [row_schedule(k, d) for d, k in enumerate(keys)], padded)


def _record_matrix_vjps(hierarchy, tape):
    dtype, device = hierarchy.matrices[-1].values.dtype, hierarchy.keys[0].device
    kernels = precision_kernels(dtype)

    def record_galerkin(fine, coarse, transfer):
        def backward():
            wp.launch(
                kernels.galerkin_backward,
                fine.nrow,
                inputs=[
                    fine.offsets,
                    fine.columns,
                    transfer.offsets,
                    transfer.columns,
                    transfer.values,
                    coarse.offsets,
                    coarse.columns,
                    coarse.values.grad,
                    fine.values.grad,
                ],
                device=device,
                record_tape=False,
            )

        tape.record_func(backward, [fine.values, coarse.values])

    for d in range(len(hierarchy.keys) - 1, 0, -1):
        record_galerkin(hierarchy.matrices[d], hierarchy.matrices[d - 1], hierarchy.transfers[d - 1])
    if hierarchy.padded is not None:
        matrix, padded, keys = hierarchy.matrices[0], hierarchy.padded, hierarchy.keys[0]

        def backward():
            wp.launch(
                kernels.unpad_matrix_backward,
                len(keys),
                inputs=[
                    keys,
                    matrix.offsets,
                    matrix.columns,
                    padded.offsets,
                    padded.columns,
                    padded.values.grad,
                    matrix.values.grad,
                ],
                device=device,
                record_tape=False,
            )

        tape.record_func(backward, [matrix.values, padded.values])


def finite_multigrid(hierarchy, b, *, tape, iters, cycles, cg_accuracy):
    """Replay every finite V-cycle, GS ordering, outer skip and root CG step.

    Must run within the supplied tape. ``prepare_hierarchy`` runs before it.
    Returns the tracked finest solution and the discrete root branch trace.
    """
    _record_matrix_vjps(hierarchy, tape)
    matrices, keys = hierarchy.matrices, hierarchy.keys
    base, dtype, device = len(keys) - 1, b.dtype, b.device
    rhs = [wp.zeros(len(k), dtype=dtype, device=device) for k in keys]
    rhs[base] = b
    x = [wp.zeros_like(v) for v in rhs]
    padded = hierarchy.padded
    root_matrix = padded if padded is not None else matrices[0]
    if padded is not None:
        root_b = wp.zeros(64, dtype=dtype, device=device, requires_grad=True)
        wp.launch(solver_kernels(dtype)._root_embed, len(keys[0]), inputs=[keys[0], b, root_b], device=device)
        root_x = wp.zeros_like(root_b)
    trace = []
    for cycle in range(cycles):
        for d in range(base, 0, -1):
            x[d] = differentiable_relaxation(matrices[d], rhs[d], x[d], hierarchy.schedules[d], tape=tape, iters=iters)
            residual = functional_spmv(matrices[d], x[d], rhs[d], alpha=-1, beta=1)
            rhs[d - 1] = functional_spmv(hierarchy.restrictions[d - 1], residual)
            x[d - 1] = wp.zeros_like(rhs[d - 1])
        if padded is None:
            root_b, root_x = rhs[0], x[0]
        residual = root_b
        if base == 0 and cycle > 0:
            residual = functional_spmv(root_matrix, root_x, root_b, alpha=-1, beta=1)
            if padded is not None:
                shifted = wp.empty_like(residual, requires_grad=True)
                wp.launch(
                    functional_kernels(dtype).add_mean, 1, inputs=[root_x, residual, dtype(-1), shifted], device=device
                )
                residual = shifted
        norm_squared = wp.utils.array_inner(residual, residual)
        info = {"iterations": 0, "outer_skip": True}
        if norm_squared > cg_accuracy * cg_accuracy:
            root_x, info = finite_root_cg(
                root_matrix,
                root_b,
                root_x,
                maxiter=len(keys[0]),
                tol=0 if base == 0 else cg_accuracy,
                atol=cg_accuracy * math.sqrt(norm_squared) if base == 0 else 0,
                add_dc_term=padded is not None,
            )
            info["outer_skip"] = False
        trace.append(info)
        if padded is None:
            x[0] = root_x
        for d in range(1, base + 1):
            x[d] = functional_spmv(hierarchy.transfers[d - 1], x[d - 1], x[d], beta=1)
            x[d - 1] = wp.zeros_like(x[d - 1])
            x[d] = differentiable_relaxation(
                matrices[d], rhs[d], x[d], hierarchy.schedules[d], tape=tape, iters=iters, forward=False
            )
    if padded is not None:
        x[0] = wp.empty_like(b, requires_grad=True)
        wp.launch(solver_kernels(dtype)._root_extract, len(keys[0]), inputs=[keys[0], root_x, x[0]], device=device)
    return x[base], trace
