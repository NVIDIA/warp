# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example: Conjugate Gradient with Warp + STF (STF-driven loop, array deps)
#
# Solves the same SPD system as :mod:`example_stf_cg`. The iteration loop
# is driven by CUDASTF's own ``while_loop`` (a ``cudaGraphCondTypeWhile``
# conditional graph node). Storage is split along its natural lifetime,
# so the example composes well with surrounding Warp code:
#
# * Persistent state -- ``x``, ``r``, ``p``, ``rs_old`` and ``iter_count``
#   -- lives in ``wp.array`` objects allocated up-front. They are visible
#   to the rest of the program (host reads, downstream Warp kernels) and
#   are ordered through STF with ``ctx.dep(array)`` ordering tokens.
# * Per-iteration scratch -- ``Ap``, ``pAp``, ``alpha`` and ``rs_new`` --
#   is declared with ``ctx.logical_data_empty(..., no_export=True)``
#   *inside* the while body. STF owns the storage, the underlying
#   allocation is hoisted out of the conditional body (no
#   ``cudaGraphNodeTypeMemAlloc`` inside the loop) and reused across
#   iterations.
#
# Per-iteration STF DAG (-> = depends on):
#
#     spmv  (Ap = A p)
#       \\
#       dot_pAp -> divide_alpha -+-> axpy_x   (x += alpha p)
#                                |
#                                +-> axpy_r   (r -= alpha Ap)
#                                       |
#                                       v
#                                  dot_rs_new -> update_p (p = r + beta p; rs_old = rs_new)
#                                                       |
#                                                       v
#                                                  update_cond  (drives loop.cond_handle)
###########################################################################

import numpy as np

import warp as wp
from warp import stf_experimental as wp_stf

N = 1 << 14
MAX_ITERS = 256
TOL = 1.0e-5


# ---------------------------------------------------------------------------
# Kernels (identical to example_stf_cg.py / example_stf_cg_stackable.py)
# ---------------------------------------------------------------------------


@wp.kernel
def spmv_csr(
    values: wp.array[float],
    columns: wp.array[int],
    offsets: wp.array[int],
    x: wp.array[float],
    y: wp.array[float],
):
    row = wp.tid()
    s = float(0.0)
    for j in range(offsets[row], offsets[row + 1]):
        s += values[j] * x[columns[j]]
    y[row] = s


@wp.kernel
def axpby(
    out: wp.array[float],
    a: float,
    x: wp.array[float],
    b: float,
    y: wp.array[float],
):
    """out = a*x + b*y (host scalars)."""
    i = wp.tid()
    out[i] = a * x[i] + b * y[i]


@wp.kernel
def axpy_dev_scalar(
    y: wp.array[float],
    sign: float,
    alpha: wp.array[float],
    x: wp.array[float],
):
    """y += sign * alpha[0] * x (alpha is a 1-element device array)."""
    i = wp.tid()
    y[i] = y[i] + sign * alpha[0] * x[i]


@wp.kernel
def update_p(
    p: wp.array[float],
    r: wp.array[float],
    rs_new: wp.array[float],
    rs_old: wp.array[float],
):
    """p = r + (rs_new[0] / rs_old[0]) * p."""
    i = wp.tid()
    beta = rs_new[0] / rs_old[0]
    p[i] = r[i] + beta * p[i]


@wp.kernel
def zero_scalar(s: wp.array[float]):
    s[0] = 0.0


@wp.kernel
def dot_kernel(a: wp.array[float], b: wp.array[float], out: wp.array[float]):
    """Atomic accumulation of <a, b> into out[0]. Caller must zero out[0]."""
    i = wp.tid()
    wp.atomic_add(out, 0, a[i] * b[i])


@wp.kernel
def divide_scalar(
    out: wp.array[float],
    num: wp.array[float],
    denom: wp.array[float],
):
    out[0] = num[0] / denom[0]


@wp.kernel
def copy_scalar(dst: wp.array[float], src: wp.array[float]):
    dst[0] = src[0]


# Native helper: drive a ``cudaGraphConditionalHandle`` directly from
# within a Warp kernel (Warp does not yet expose this as a builtin).
_set_cond_snippet = """
cudaGraphSetConditional((cudaGraphConditionalHandle)handle, (unsigned int)value);
"""


@wp.func_native(_set_cond_snippet)
def stf_set_cond(handle: wp.uint64, value: wp.int32): ...


@wp.kernel
def update_cond(
    cond_handle: wp.uint64,
    iter_count: wp.array[int],
    rs_new: wp.array[float],
    tol2: float,
    max_iters: int,
):
    """Increment iter_count and drive the while-loop's conditional handle.

    Sets the conditional to 1 (continue) iff ``rs_new > tol^2`` AND
    ``iter < max_iters``, else 0 (stop).
    """
    iter_count[0] = iter_count[0] + 1
    keep = wp.int32(0)
    if rs_new[0] > tol2 and iter_count[0] < max_iters:
        keep = wp.int32(1)
    stf_set_cond(cond_handle, keep)


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------


def build_tridiag_csr(n: int, device):
    """Tridiagonal SPD matrix with diagonal 4 and off-diagonals -1."""
    cols: list[int] = []
    vals: list[float] = []
    offsets = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        offsets[i] = len(vals)
        if i > 0:
            cols.append(i - 1)
            vals.append(-1.0)
        cols.append(i)
        vals.append(4.0)
        if i < n - 1:
            cols.append(i + 1)
            vals.append(-1.0)
    offsets[n] = len(vals)

    return (
        wp.array(np.asarray(vals, dtype=np.float32), dtype=wp.float32, device=device),
        wp.array(np.asarray(cols, dtype=np.int32), dtype=wp.int32, device=device),
        wp.array(offsets, dtype=wp.int32, device=device),
    )


def host_residual_norm(A_csr, x: wp.array, b: wp.array) -> tuple[float, float]:
    """Compute ||b - A x|| and ||b|| on the host for verification."""
    values, columns, offsets = (a.numpy() for a in A_csr)
    x_h = x.numpy().astype(np.float64)
    b_h = b.numpy().astype(np.float64)
    Ax = np.zeros_like(b_h)
    for i in range(len(b_h)):
        s = 0.0
        for j in range(offsets[i], offsets[i + 1]):
            s += values[j] * x_h[columns[j]]
        Ax[i] = s
    return float(np.linalg.norm(b_h - Ax)), float(np.linalg.norm(b_h))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda:0", help="Override the default Warp device.")
    parser.add_argument("--stage-path", type=str, default=None, help="Unused; accepted for test harness compatibility.")
    args = parser.parse_known_args()[0]

    wp.init()

    if not (wp.is_cuda_available() and wp.is_conditional_graph_supported() and wp_stf.is_available()):
        raise RuntimeError(
            "This example requires CUDA, conditional CUDA graphs, and cuda.stf. Install "
            "cuda-stf[cu12] or cuda-stf[cu13] for your CUDA toolkit."
        )

    device = wp.get_device(args.device)

    # ---- Warp-owned device storage for the persistent CG state ----
    A_csr = build_tridiag_csr(N, device)
    A_values, A_columns, A_offsets = A_csr
    b = wp.array(np.ones(N, dtype=np.float32), dtype=wp.float32, device=device)
    x = wp.zeros(N, dtype=wp.float32, device=device)
    r = wp.empty(N, dtype=wp.float32, device=device)
    p = wp.empty(N, dtype=wp.float32, device=device)
    rs_old = wp.empty(1, dtype=wp.float32, device=device)
    iter_count = wp.zeros(1, dtype=wp.int32, device=device)

    # ---- build the STF DAG: setup + while_loop, ordered via array-backed deps ----
    with wp.ScopedDevice(device), wp_stf.context() as ctx:  # stf.stackable_context() under the hood
        with ctx.graph_scope():
            # ---- one-shot setup: r = b - A x; p = r; rs_old = <r, r> ----

            # r = A x
            with ctx.task(
                ctx.dep(A_values).read(),
                ctx.dep(A_columns).read(),
                ctx.dep(A_offsets).read(),
                ctx.dep(x).read(),
                ctx.dep(r).write(),
            ) as (s,):
                wp.launch(
                    spmv_csr,
                    dim=N,
                    inputs=[A_values, A_columns, A_offsets, x, r],
                    stream=s,
                )

            # r = b - r
            with ctx.task(ctx.dep(r).rw(), ctx.dep(b).read()) as (s,):
                wp.launch(axpby, dim=N, inputs=[r, 1.0, b, -1.0, r], stream=s)

            # p = r
            with ctx.task(ctx.dep(p).write(), ctx.dep(r).read()) as (s,):
                wp.launch(axpby, dim=N, inputs=[p, 1.0, r, 0.0, r], stream=s)

            # rs_old = <r, r>
            with ctx.task(ctx.dep(r).read(), ctx.dep(rs_old).write()) as (s,):
                wp.launch(zero_scalar, dim=1, inputs=[rs_old], stream=s)
                wp.launch(dot_kernel, dim=N, inputs=[r, r, rs_old], stream=s)

            # ---- while body: one CG iteration as an STF DAG ----
            with ctx.while_loop() as loop:
                # Per-iteration scratch: STF owns the storage and uses
                # ``no_export=True`` so the underlying allocation is
                # hoisted out of the conditional body and reused across
                # iterations. Declaring these inside the loop matches
                # how the values are actually used: written, consumed,
                # and discarded within a single iteration.
                l_Ap = ctx.logical_data_empty((N,), dtype=np.float32, name="Ap", no_export=True)
                l_pAp = ctx.logical_data_empty((1,), dtype=np.float32, name="pAp", no_export=True)
                l_alpha = ctx.logical_data_empty((1,), dtype=np.float32, name="alpha", no_export=True)
                l_rs_new = ctx.logical_data_empty((1,), dtype=np.float32, name="rs_new", no_export=True)

                # Ap = A p
                with ctx.task(
                    ctx.dep(A_values).read(),
                    ctx.dep(A_columns).read(),
                    ctx.dep(A_offsets).read(),
                    ctx.dep(p).read(),
                    l_Ap.write(),
                ) as (s, Ap):
                    wp.launch(
                        spmv_csr,
                        dim=N,
                        inputs=[A_values, A_columns, A_offsets, p, Ap],
                        stream=s,
                    )

                # pAp = <p, Ap>
                with ctx.task(ctx.dep(p).read(), l_Ap.read(), l_pAp.write()) as (s, Ap, pAp):
                    wp.launch(zero_scalar, dim=1, inputs=[pAp], stream=s)
                    wp.launch(dot_kernel, dim=N, inputs=[p, Ap, pAp], stream=s)

                # alpha = rs_old / pAp
                with ctx.task(ctx.dep(rs_old).read(), l_pAp.read(), l_alpha.write()) as (s, pAp, alpha):
                    wp.launch(divide_scalar, dim=1, inputs=[alpha, rs_old, pAp], stream=s)

                # x += alpha p   --+
                #                  | sibling tasks: STF schedules them on
                # r -= alpha Ap  --+ disjoint streams (disjoint deps)
                with ctx.task(ctx.dep(x).rw(), l_alpha.read(), ctx.dep(p).read()) as (s, alpha):
                    wp.launch(axpy_dev_scalar, dim=N, inputs=[x, 1.0, alpha, p], stream=s)

                with ctx.task(ctx.dep(r).rw(), l_alpha.read(), l_Ap.read()) as (s, alpha, Ap):
                    wp.launch(axpy_dev_scalar, dim=N, inputs=[r, -1.0, alpha, Ap], stream=s)

                # rs_new = <r, r>
                with ctx.task(ctx.dep(r).read(), l_rs_new.write()) as (s, rs_new):
                    wp.launch(zero_scalar, dim=1, inputs=[rs_new], stream=s)
                    wp.launch(dot_kernel, dim=N, inputs=[r, r, rs_new], stream=s)

                # p = r + (rs_new / rs_old) p; rs_old <- rs_new
                with ctx.task(
                    ctx.dep(p).rw(),
                    ctx.dep(r).read(),
                    l_rs_new.read(),
                    ctx.dep(rs_old).rw(),
                ) as (s, rs_new):
                    wp.launch(update_p, dim=N, inputs=[p, r, rs_new, rs_old], stream=s)
                    wp.launch(copy_scalar, dim=1, inputs=[rs_old, rs_new], stream=s)

                # Bump iter_count, evaluate the stop conditions, and drive
                # the while-loop's conditional handle
                with ctx.task(ctx.dep(iter_count).rw(), l_rs_new.read()) as (s, rs_new):
                    wp.launch(
                        update_cond,
                        dim=1,
                        inputs=[wp.uint64(loop.cond_handle), iter_count, rs_new, TOL * TOL, MAX_ITERS],
                        stream=s,
                    )

    iters = int(iter_count.numpy()[0])
    res, bn = host_residual_norm(A_csr, x, b)
    rel = res / bn
    print(f"CG (STF while_loop, array deps): N={N}, iters={iters}/{MAX_ITERS}, ||b - A x||/||b|| = {rel:.3e}")
    assert rel < TOL, f"CG did not converge: relative residual {rel:.3e} >= {TOL}"
