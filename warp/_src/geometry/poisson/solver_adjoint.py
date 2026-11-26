# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional sparse primitives and finite root CG for the opt-in backward path.

These are isolated solver building blocks, not a reconstruction backward API.
"""

import math
from functools import lru_cache
from types import SimpleNamespace

import warp as wp

wp.set_module_options({"fuse_fp": True})


@lru_cache(None)
def precision_kernels(dtype):
    @wp.kernel
    def copy_kernel(src: wp.array[dtype], dst: wp.array[dtype]):
        i = wp.tid()
        dst[i] = src[i]

    @wp.kernel
    def matvec_kernel(
        offsets: wp.array[int],
        counts: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        x: wp.array[dtype],
        y: wp.array[dtype],
        alpha: dtype,
        beta: dtype,
        out: wp.array[dtype],
    ):
        i = wp.tid()
        v = dtype(0)
        if alpha != dtype(0):
            end = offsets[i + 1]
            if counts.shape[0] != 0:
                end = offsets[i] + counts[i]
            for k in range(offsets[i], end):
                v += values[k] * x[columns[k]]
            v *= alpha
        if beta != dtype(0):
            v += beta * y[i]
        out[i] = v

    @wp.kernel(module="unique")
    def tiled_matvec_kernel(
        offsets: wp.array[int],
        counts: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        x: wp.array[dtype],
        y: wp.array[dtype],
        alpha: dtype,
        beta: dtype,
        out: wp.array[dtype],
    ):
        row, lane = wp.tid()
        if beta == dtype(0):
            total = wp.tile_zeros(shape=(1,), dtype=dtype)
        else:
            total = beta * wp.tile_load(y, 1, row)
        if alpha != dtype(0):
            begin = offsets[row]
            count = offsets[row + 1] - begin
            if counts.shape[0]:
                count = counts[row]
            lane_sum = dtype(0)
            for column in range(lane, count, 64):
                entry = begin + column
                lane_sum += values[entry] * x[columns[entry]]
            lane_sum *= alpha
            total += wp.tile_sum(wp.tile(lane_sum))
        wp.tile_store(out, total, row)

    @wp.kernel(module="unique")
    def dot_kernel(a: wp.array[dtype], b: wp.array[dtype], out: wp.array[dtype]):
        _column, _block, _lane = wp.tid()
        av = wp.tile_load(a, shape=512)
        bv = wp.tile_load(b, shape=512)
        wp.tile_store(out, wp.tile_sum(wp.tile_map(wp.mul, av, bv)))

    @wp.kernel
    def tolerance_kernel(norm: wp.array[dtype], rtol: dtype, atol: dtype, out: wp.array[dtype]):
        a = wp.max(rtol * wp.sqrt(norm[0]), atol)
        out[0] = a * a

    @wp.kernel
    def step_kernel(
        x: wp.array[dtype],
        r: wp.array[dtype],
        p: wp.array[dtype],
        ap: wp.array[dtype],
        rr: wp.array[dtype],
        pap: wp.array[dtype],
        tol: wp.array[dtype],
        next_x: wp.array[dtype],
        next_r: wp.array[dtype],
    ):
        i = wp.tid()
        alpha = dtype(0)
        if rr[0] > tol[0]:
            alpha = rr[0] / pap[0]
        next_x[i] = x[i] + alpha * p[i]
        next_r[i] = r[i] - alpha * ap[i]

    @wp.kernel
    def direction_kernel(
        r: wp.array[dtype],
        p: wp.array[dtype],
        old_rr: wp.array[dtype],
        rr: wp.array[dtype],
        tol: wp.array[dtype],
        out: wp.array[dtype],
    ):
        i = wp.tid()
        beta = dtype(0)
        if rr[0] > tol[0]:
            beta = rr[0] / old_rr[0]
        out[i] = r[i] + beta * p[i]

    @wp.kernel(module="unique")
    def add_mean_kernel(x: wp.array[dtype], y: wp.array[dtype], alpha: dtype, out: wp.array[dtype]):
        # Match the production root operator's scalar, unfused accumulation.
        mean = dtype(0)
        for i in range(x.shape[0]):
            mean += x[i]
        mean /= dtype(x.shape[0])
        for i in range(y.shape[0]):
            out[i] = y[i] + alpha * mean

    wp.set_module_options({"fuse_fp": False}, module=add_mean_kernel.module)
    return SimpleNamespace(
        copy=copy_kernel,
        matvec=matvec_kernel,
        tiled_matvec=tiled_matvec_kernel,
        dot=dot_kernel,
        tolerance=tolerance_kernel,
        step=step_kernel,
        direction=direction_kernel,
        add_mean=add_mean_kernel,
    )


def functional_spmv(matrix, x, y=None, *, alpha=1.0, beta=0.0):
    """Scalar CSR ``alpha*A*x + beta*y`` with immutable inputs and a new output.

    Matches Warp's serial or 64-lane tiled reduction using its nnz-capacity
    heuristic. Coarse Galerkin capacity can select tiles even when actual rows
    are short. CSR topology is fixed; tracked values, x and y receive gradients.
    """
    dtype, device = x.dtype, x.device
    if (
        dtype not in (wp.float32, wp.float64)
        or matrix.values.dtype != dtype
        or x.shape != (matrix.ncol,)
        or matrix.values.device != device
    ):
        raise ValueError("require a scalar float32/float64 matrix and matching vector/device")
    if y is None:
        if beta != 0:
            raise ValueError("nonzero beta requires y")
        y = wp.empty(0, dtype=dtype, device=device)
    elif y.dtype != dtype or y.device != device or y.shape != (matrix.nrow,):
        raise ValueError("y must match the matrix rows, precision, and device")
    counts = matrix.row_counts
    if counts is None:
        counts = wp.empty(0, dtype=int, device=device)
    out = wp.empty(matrix.nrow, dtype=dtype, device=device, requires_grad=True)
    inputs = [matrix.offsets, counts, matrix.columns, matrix.values, x, y, dtype(alpha), dtype(beta), out]
    if device.is_cuda and matrix.nnz > 128 * matrix.nrow:
        wp.launch(precision_kernels(dtype).tiled_matvec, (matrix.nrow, 64), inputs=inputs, block_dim=64, device=device)
    else:
        wp.launch(precision_kernels(dtype).matvec, len(out), inputs=inputs, device=device)
    return out


def finite_root_cg(matrix, b, initial, *, maxiter, tol=0.0, atol=0.0, add_dc_term=False):
    """Differentiate unpreconditioned finite CG for at most 64 root rows.

    Match the pinned Warp CG arithmetic and per-iteration stopping decisions,
    holding those decisions fixed during backward. Return (tracked solution,
    diagnostics). Caller inputs and topology must remain immutable through
    backward. The optional constant mode is ``mean(x)`` over all matrix rows;
    pass the production 64-row padded matrix when that operator is required.
    This function does not implement reconstruction's outer pre-CG skip test.
    """
    if (
        b.dtype not in (wp.float32, wp.float64)
        or initial.dtype != b.dtype
        or b.shape != initial.shape
        or len(b) < 1
        or len(b) > 64
        or matrix.nrow != len(b)
        or matrix.ncol != len(b)
        or b.device != initial.device
    ):
        raise ValueError("root CG requires 1..64 matching scalar square rows")
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if any(not math.isfinite(v) or v < 0 for v in [tol, atol]):
        raise ValueError("tolerances must be finite and nonnegative")
    dtype, device = b.dtype, b.device
    if device.is_capturing:
        raise ValueError("finite root CG checks convergence on the host and cannot be captured")
    kernels = precision_kernels(dtype)

    def dot_kernel(a, c):
        out = wp.empty(1, dtype=dtype, device=device, requires_grad=True)
        wp.launch(kernels.dot, (1, 1, 512), inputs=[a, c, out], block_dim=512, device=device)
        return out

    def operator(x, y=None, alpha=1.0, beta=0.0):
        value = functional_spmv(matrix, x, y, alpha=alpha, beta=beta)
        if add_dc_term:
            out = wp.empty_like(value, requires_grad=True)
            wp.launch(kernels.add_mean, 1, inputs=[x, value, dtype(alpha), out], device=device)
            value = out
        return value

    norm = dot_kernel(b, b)
    threshold = wp.empty(1, dtype=dtype, device=device)
    wp.launch(kernels.tolerance, 1, inputs=[norm, dtype(tol), dtype(atol), threshold], device=device, record_tape=False)
    threshold_host = float(threshold.numpy()[0])
    x = wp.empty_like(initial, requires_grad=True)
    wp.launch(kernels.copy, len(initial), inputs=[initial, x], device=device)
    r = operator(x, b, alpha=-1.0, beta=1.0)
    rr = dot_kernel(r, r)
    p = r
    steps = 0
    residual = float(rr.numpy()[0])
    while residual > threshold_host and steps < maxiter:
        ap = operator(p)
        pap = dot_kernel(p, ap)
        next_x = wp.empty_like(x, requires_grad=True)
        next_r = wp.empty_like(r, requires_grad=True)
        wp.launch(kernels.step, len(b), inputs=[x, r, p, ap, rr, pap, threshold, next_x, next_r], device=device)
        next_rr = dot_kernel(next_r, next_r)
        next_p = wp.empty_like(p, requires_grad=True)
        wp.launch(kernels.direction, len(b), inputs=[next_r, p, rr, next_rr, threshold, next_p], device=device)
        x, r, p, rr = next_x, next_r, next_p, next_rr
        steps += 1
        residual = float(rr.numpy()[0])
    return x, {
        "iterations": steps,
        "residual_norm": math.sqrt(residual),
        "absolute_tolerance": math.sqrt(threshold_host),
    }
