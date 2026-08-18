# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example: Conjugate Gradient with Warp + STF (inner pattern)
#
# Solves a symmetric positive-definite system A x = b using the conjugate
# gradient method, recorded as a single Warp CUDA graph with a conditional
# ``while`` body. Inside the body, each CG iteration is expressed as a
# small CUDASTF DAG so the kernels with no data dependency on each other
# -- the ``x += alpha p`` and ``r -= alpha A p`` updates -- run on
# sibling streams.
#
# Convergence is decided entirely on the device: the body updates a
# ``cond`` flag from ``<r, r>`` and an iteration counter, and Warp's
# :func:`capture_while` re-launches the body until ``cond`` is zero, so
# the host launches the graph exactly once per solve regardless of the
# iteration count.
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
###########################################################################

import numpy as np

import warp as wp
from warp import stf_experimental as wp_stf

N = 1 << 14
MAX_ITERS = 256
TOL = 1.0e-5


# ---------------------------------------------------------------------------
# Kernels
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
    """p = r + (rs_new[0] / rs_old[0]) * p; rs_old <- rs_new."""
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


@wp.kernel
def update_cond(
    cond: wp.array[int],
    iter_count: wp.array[int],
    rs_new: wp.array[float],
    tol2: float,
    max_iters: int,
):
    """Increment iter_count; set cond=1 iff we should keep iterating."""
    iter_count[0] = iter_count[0] + 1
    keep = int(0)
    if rs_new[0] > tol2 and iter_count[0] < max_iters:
        keep = 1
    cond[0] = keep


@wp.kernel
def reset_solver(cond: wp.array[int], iter_count: wp.array[int]):
    cond[0] = 1
    iter_count[0] = 0


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------


def build_tridiag_csr(n: int, device):
    """Tridiagonal SPD matrix with diagonal 4 and off-diagonals -1.

    Diagonally dominant, so CG converges quickly regardless of n -- a
    convenient shape for an example.
    """
    cols = []
    vals = []
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

    A_csr = build_tridiag_csr(N, device)
    A_values, A_columns, A_offsets = A_csr
    b = wp.array(np.ones(N, dtype=np.float32), dtype=wp.float32, device=device)
    x = wp.zeros(N, dtype=wp.float32, device=device)

    # Pre-allocate every buffer the captured body touches: a conditional
    # graph body cannot contain cudaGraphNodeTypeMemAlloc / MemFree nodes,
    # so all Warp memory pool activity must happen before capture starts.
    r = wp.empty(N, dtype=wp.float32, device=device)
    p = wp.empty(N, dtype=wp.float32, device=device)
    Ap = wp.empty(N, dtype=wp.float32, device=device)
    pAp = wp.empty(1, dtype=wp.float32, device=device)
    rs_old = wp.empty(1, dtype=wp.float32, device=device)
    rs_new = wp.empty(1, dtype=wp.float32, device=device)
    alpha = wp.empty(1, dtype=wp.float32, device=device)

    iter_count = wp.zeros(1, dtype=wp.int32, device=device)
    cond = wp.full((1,), 1, dtype=wp.int32, device=device)

    with wp.ScopedDevice(device):
        # ---- one-shot setup on the device default stream: r = b - A x; p = r; rs_old = <r, r>. ----
        wp.launch(spmv_csr, dim=N, inputs=[A_values, A_columns, A_offsets, x, r], device=device)
        wp.launch(axpby, dim=N, inputs=[r, 1.0, b, -1.0, r], device=device)
        wp.launch(axpby, dim=N, inputs=[p, 1.0, r, 0.0, r], device=device)
        wp.launch(zero_scalar, dim=1, inputs=[rs_old], device=device)
        wp.launch(dot_kernel, dim=N, inputs=[r, r, rs_old], device=device)
        wp.synchronize_device(device)

        capture_stream = wp.Stream(device)

        # Force kernel JIT before capture starts so module-load events do not
        # leak into the captured graph.
        wp.load_module(device=device)

        # ---- captured while body: one CG iteration as an STF DAG ----
        def cg_iteration():
            with wp_stf.context(stream=capture_stream) as ctx:
                # Ap = A p
                with ctx.task(
                    ctx.dep(A_values).read(),
                    ctx.dep(A_columns).read(),
                    ctx.dep(A_offsets).read(),
                    ctx.dep(p).read(),
                    ctx.dep(Ap).write(),
                ) as (s,):
                    wp.launch(
                        spmv_csr,
                        dim=N,
                        inputs=[A_values, A_columns, A_offsets, p, Ap],
                        stream=s,
                    )

                # pAp = <p, Ap>
                with ctx.task(ctx.dep(p).read(), ctx.dep(Ap).read(), ctx.dep(pAp).write()) as (s,):
                    wp.launch(zero_scalar, dim=1, inputs=[pAp], stream=s)
                    wp.launch(dot_kernel, dim=N, inputs=[p, Ap, pAp], stream=s)

                # alpha = rs_old / pAp
                with ctx.task(ctx.dep(rs_old).read(), ctx.dep(pAp).read(), ctx.dep(alpha).write()) as (s,):
                    wp.launch(divide_scalar, dim=1, inputs=[alpha, rs_old, pAp], stream=s)

                # x += alpha p   --+
                #                  | STF schedules these two on sibling streams
                # r -= alpha Ap  --+   because they touch disjoint deps.
                with ctx.task(ctx.dep(x).rw(), ctx.dep(alpha).read(), ctx.dep(p).read()) as (s,):
                    wp.launch(axpy_dev_scalar, dim=N, inputs=[x, 1.0, alpha, p], stream=s)

                with ctx.task(ctx.dep(r).rw(), ctx.dep(alpha).read(), ctx.dep(Ap).read()) as (s,):
                    wp.launch(axpy_dev_scalar, dim=N, inputs=[r, -1.0, alpha, Ap], stream=s)

                # rs_new = <r, r>
                with ctx.task(ctx.dep(r).read(), ctx.dep(rs_new).write()) as (s,):
                    wp.launch(zero_scalar, dim=1, inputs=[rs_new], stream=s)
                    wp.launch(dot_kernel, dim=N, inputs=[r, r, rs_new], stream=s)

                # p = r + (rs_new/rs_old) p; rs_old <- rs_new
                with ctx.task(
                    ctx.dep(p).rw(),
                    ctx.dep(r).read(),
                    ctx.dep(rs_new).read(),
                    ctx.dep(rs_old).rw(),
                ) as (s,):
                    wp.launch(update_p, dim=N, inputs=[p, r, rs_new, rs_old], stream=s)
                    wp.launch(copy_scalar, dim=1, inputs=[rs_old, rs_new], stream=s)

            # Back on capture_stream: update iteration count and convergence flag.
            wp.launch(
                update_cond,
                dim=1,
                inputs=[cond, iter_count, rs_new, TOL * TOL, MAX_ITERS],
                device=device,
                stream=capture_stream,
            )

        # ---- capture the iterative phase as one CUDA graph with a conditional body ----
        with wp.ScopedStream(capture_stream):
            with wp.ScopedCapture(
                device=device,
                stream=capture_stream,
                force_module_load=False,
                capture_mode=wp.CaptureMode.RELAXED,
            ) as capture:
                wp.capture_while(cond, cg_iteration)

        # One launch -> the loop runs to convergence on the device.
        wp.capture_launch(capture.graph, stream=capture_stream)
        wp.synchronize_stream(capture_stream)

        iters = int(iter_count.numpy()[0])
        res, bn = host_residual_norm(A_csr, x, b)
        rel = res / bn
        print(f"CG: N={N}, iters={iters}/{MAX_ITERS}, ||b - A x||/||b|| = {rel:.3e}")
        assert rel < TOL, f"CG did not converge: relative residual {rel:.3e} >= {TOL}"
