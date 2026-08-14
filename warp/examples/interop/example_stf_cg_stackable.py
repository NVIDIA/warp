# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example: Conjugate Gradient with Warp + STF (STF-driven loop)
#
# Same problem as :mod:`example_stf_cg`, same kernels, same per-iteration
# DAG -- but the iteration loop is driven by CUDASTF's own ``while_loop``
# (a ``cudaGraphCondTypeWhile`` conditional graph node) instead of
# :func:`warp.capture_while`. The whole solve -- pre-loop setup, the
# while body, and finalize -- is described declaratively as one stackable
# STF context, and ``ctx.finalize()`` instantiates and launches the
# resulting CUDA graph in a single host call.
#
# Differences from :mod:`example_stf_cg`:
#
# * ``stf.stackable_context()`` (self-owned) replaces
#   ``wp.ScopedCapture`` + ``wp_stf.context(stream=...)``.
# * Persistent state (``x``, ``r``, ``p``, ``rs_old``, ``iter``, ``cond``)
#   is ``ctx.logical_data(host_arr)`` -- STF owns the storage and writes
#   back to the host buffers at ``finalize()``.
# * Intra-iteration scratch (``Ap``, ``pAp``, ``alpha``, ``rs_new``) is
#   created *inside* the while body as
#   ``ctx.logical_data_empty(shape, dtype, no_export=True)``. The
#   stackable backend hoists the underlying allocation to the parent
#   scope and rebinds it per iteration, so no
#   ``cudaGraphNodeTypeMemAlloc`` node lands in the conditional body.
# * Read-only inputs (``A_values``, ``A_columns``, ``A_offsets``, ``b``)
#   are flagged with ``set_read_only()`` so STF lets concurrent reads of
#   them across sibling tasks run in parallel without serialization.
# * The convergence check (``update_cond``) lives inside the while body
#   as a single Warp kernel that AND-s the residual and iteration-cap
#   conditions and *directly* drives the while-loop's
#   ``cudaGraphConditionalHandle`` via ``cudaGraphSetConditional``. There
#   is no auxiliary ``cond`` logical_data and no separate
#   ``loop.continue_while(...)`` compare node.
###########################################################################

import numpy as np

import warp as wp
from warp import stf_experimental as wp_stf

N = 1 << 14
MAX_ITERS = 256
TOL = 1.0e-5


# ---------------------------------------------------------------------------
# Kernels (identical to example_stf_cg.py)
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


# Native helper: directly drive a ``cudaGraphConditionalHandle`` from inside
# a Warp kernel. This is what lets ``update_cond`` set the while-loop's
# conditional handle in-place, with no auxiliary ``cond`` logical_data and
# no separate ``loop.continue_while(...)`` scalar-compare node.
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
    ``iter < max_iters``, else 0 (stop). Folds both the convergence test
    and the iteration-cap into a single device-side kernel.
    """
    iter_count[0] = iter_count[0] + 1
    keep = wp.int32(0)
    if rs_new[0] > tol2 and iter_count[0] < max_iters:
        keep = wp.int32(1)
    stf_set_cond(cond_handle, keep)


# ---------------------------------------------------------------------------
# Problem setup (identical to example_stf_cg.py)
# ---------------------------------------------------------------------------


def build_tridiag_csr_host(n: int):
    """Tridiagonal SPD matrix with diagonal 4 and off-diagonals -1.

    Returns the CSR triple as host numpy arrays so they can be handed to
    ``ctx.logical_data(...)`` directly -- STF moves them to the device
    at first task access.
    """
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
        np.asarray(vals, dtype=np.float32),
        np.asarray(cols, dtype=np.int32),
        offsets,
    )


def host_residual_norm_csr(values, columns, offsets, x, b) -> tuple[float, float]:
    """Compute ||b - A x|| and ||b|| on host arrays."""
    x_h = x.astype(np.float64)
    b_h = b.astype(np.float64)
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

    # ---- host-side problem and state buffers ----
    A_values_h, A_columns_h, A_offsets_h = build_tridiag_csr_host(N)
    b_h = np.ones(N, dtype=np.float32)
    x_h = np.zeros(N, dtype=np.float32)
    r_h = np.zeros(N, dtype=np.float32)
    p_h = np.zeros(N, dtype=np.float32)
    rs_old_h = np.zeros(1, dtype=np.float32)

    iter_h = np.zeros(1, dtype=np.int32)

    # ---- build the STF DAG: setup + while_loop + (no post-loop) ----
    with wp.ScopedDevice(device), wp_stf.context() as ctx:  # stf.stackable_context() under the hood
        # Persistent / cross-iteration state
        l_values = ctx.logical_data(A_values_h, name="A_values")
        l_columns = ctx.logical_data(A_columns_h, name="A_columns")
        l_offsets = ctx.logical_data(A_offsets_h, name="A_offsets")
        l_b = ctx.logical_data(b_h, name="b")
        l_x = ctx.logical_data(x_h, name="x")
        l_r = ctx.logical_data(r_h, name="r")
        l_p = ctx.logical_data(p_h, name="p")
        l_rs_old = ctx.logical_data(rs_old_h, name="rs_old")

        # Loop control
        l_iter = ctx.logical_data(iter_h, name="iter")

        # Inputs that never get written to: flag them so STF can run the
        # multiple sibling tasks that read them concurrently (otherwise
        # the runtime conservatively serializes accesses to the same data
        # across nested scopes).
        for ld in (l_values, l_columns, l_offsets, l_b):
            ld.set_read_only()

        with ctx.graph_scope():
            # ---- one-shot setup: r = b - A x; p = r; rs_old = <r, r> ----

            # r = A x  (spmv)
            with ctx.task(
                l_values.read(),
                l_columns.read(),
                l_offsets.read(),
                l_x.read(),
                l_r.write(),
            ) as (s, vals, cols, offs, x_arr, r_arr):
                wp.launch(
                    spmv_csr,
                    dim=N,
                    inputs=[vals, cols, offs, x_arr, r_arr],
                    stream=s,
                )

            # r = b - r
            with ctx.task(l_r.rw(), l_b.read()) as (s, r_arr, b_arr):
                wp.launch(axpby, dim=N, inputs=[r_arr, 1.0, b_arr, -1.0, r_arr], stream=s)

            # p = r
            with ctx.task(l_p.write(), l_r.read()) as (s, p_arr, r_arr):
                wp.launch(axpby, dim=N, inputs=[p_arr, 1.0, r_arr, 0.0, r_arr], stream=s)

            # rs_old = <r, r>
            with ctx.task(l_r.read(), l_rs_old.write()) as (s, r_arr, rso):
                wp.launch(zero_scalar, dim=1, inputs=[rso], stream=s)
                wp.launch(dot_kernel, dim=N, inputs=[r_arr, r_arr, rso], stream=s)

            # ---- while body: one CG iteration as an STF DAG ----
            with ctx.while_loop() as loop:
                # Per-iteration scratch: STF owns the storage and uses
                # ``no_export=True`` to keep the buffer local to this
                # while-body scope, so the underlying allocation is hoisted
                # out of the conditional body (no cudaGraphNodeTypeMemAlloc
                # inside the loop) and reused across iterations.
                l_Ap = ctx.logical_data_empty((N,), dtype=np.float32, name="Ap", no_export=True)
                l_pAp = ctx.logical_data_empty((1,), dtype=np.float32, name="pAp", no_export=True)
                l_alpha = ctx.logical_data_empty((1,), dtype=np.float32, name="alpha", no_export=True)
                l_rs_new = ctx.logical_data_empty((1,), dtype=np.float32, name="rs_new", no_export=True)

                # Ap = A p
                with ctx.task(
                    l_values.read(),
                    l_columns.read(),
                    l_offsets.read(),
                    l_p.read(),
                    l_Ap.write(),
                ) as (s, vals, cols, offs, p_arr, Ap_arr):
                    wp.launch(
                        spmv_csr,
                        dim=N,
                        inputs=[vals, cols, offs, p_arr, Ap_arr],
                        stream=s,
                    )

                # pAp = <p, Ap>
                with ctx.task(l_p.read(), l_Ap.read(), l_pAp.write()) as (s, p_arr, Ap_arr, pAp_arr):
                    wp.launch(zero_scalar, dim=1, inputs=[pAp_arr], stream=s)
                    wp.launch(dot_kernel, dim=N, inputs=[p_arr, Ap_arr, pAp_arr], stream=s)

                # alpha = rs_old / pAp
                with ctx.task(l_rs_old.read(), l_pAp.read(), l_alpha.write()) as (s, rso, pAp_arr, alpha_arr):
                    wp.launch(divide_scalar, dim=1, inputs=[alpha_arr, rso, pAp_arr], stream=s)

                # x += alpha p   --+
                #                  | sibling tasks: STF schedules them on
                # r -= alpha Ap  --+ disjoint streams (disjoint logical data)
                with ctx.task(l_x.rw(), l_alpha.read(), l_p.read()) as (s, x_arr, alpha_arr, p_arr):
                    wp.launch(
                        axpy_dev_scalar,
                        dim=N,
                        inputs=[x_arr, 1.0, alpha_arr, p_arr],
                        stream=s,
                    )

                with ctx.task(l_r.rw(), l_alpha.read(), l_Ap.read()) as (s, r_arr, alpha_arr, Ap_arr):
                    wp.launch(
                        axpy_dev_scalar,
                        dim=N,
                        inputs=[r_arr, -1.0, alpha_arr, Ap_arr],
                        stream=s,
                    )

                # rs_new = <r, r>
                with ctx.task(l_r.read(), l_rs_new.write()) as (s, r_arr, rsn):
                    wp.launch(zero_scalar, dim=1, inputs=[rsn], stream=s)
                    wp.launch(dot_kernel, dim=N, inputs=[r_arr, r_arr, rsn], stream=s)

                # p = r + (rs_new / rs_old) p; rs_old <- rs_new
                with ctx.task(
                    l_p.rw(),
                    l_r.read(),
                    l_rs_new.read(),
                    l_rs_old.rw(),
                ) as (s, p_arr, r_arr, rsn, rso):
                    wp.launch(update_p, dim=N, inputs=[p_arr, r_arr, rsn, rso], stream=s)
                    wp.launch(copy_scalar, dim=1, inputs=[rso, rsn], stream=s)

                # Bump iter_count, evaluate the stop conditions, and drive
                # the while-loop's conditional handle
                with ctx.task(l_iter.rw(), l_rs_new.read()) as (s, it_arr, rsn):
                    wp.launch(
                        update_cond,
                        dim=1,
                        inputs=[wp.uint64(loop.cond_handle), it_arr, rsn, TOL * TOL, MAX_ITERS],
                        stream=s,
                    )

    # finalize() instantiates and launches the whole graph; on return,
    # logical-data backing buffers carry the final values.
    iters = int(iter_h[0])
    res, bn = host_residual_norm_csr(A_values_h, A_columns_h, A_offsets_h, x_h, b_h)
    rel = res / bn
    print(f"CG (STF while_loop): N={N}, iters={iters}/{MAX_ITERS}, ||b - A x||/||b|| = {rel:.3e}")
    assert rel < TOL, f"CG did not converge: relative residual {rel:.3e} >= {TOL}"
