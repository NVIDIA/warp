# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate emitter drive levels toward the design flux spec.

Physics: emitter i produces flux r_i^2 + 0.1. Each emitter has a hard
regulatory limit FCAP; the plant controller clamps any flux above it, so the
deliverable flux is min(r^2 + 0.1, FCAP). We calibrate r to match the design
spec as closely as the clamp allows (some spec values sit above FCAP; those
emitters should just saturate).

Run with --small for the laptop config (256 emitters); default is the
production config (32768 emitters).
"""

import argparse

import numpy as np

import warp as wp

FCAP = 1.72


@wp.kernel
def flux_kernel(r: wp.array(dtype=float), flux: wp.array(dtype=float)):
    i = wp.tid()
    flux[i] = r[i] * r[i] + 0.1


@wp.kernel
def clamp_kernel(flux: wp.array(dtype=float), cap: float):
    i = wp.tid()
    f = flux[i]
    if f > cap:
        flux[i] = cap


@wp.kernel
def mse_kernel(
    flux: wp.array(dtype=float),
    spec: wp.array(dtype=float),
    inv_n: float,
    loss: wp.array(dtype=float),
):
    i = wp.tid()
    d = flux[i] - spec[i]
    wp.atomic_add(loss, 0, d * d * inv_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="laptop config (256 emitters)")
    args = ap.parse_args()
    n = 256 if args.small else 32768

    rng = np.random.default_rng(12)
    r_design = rng.normal(1.0, 0.1, n).astype(np.float32)
    spec = wp.array((r_design.astype(np.float64) ** 2 + 0.1).astype(np.float32), dtype=float)

    # warm start from last month's calibration
    r = wp.array((r_design + rng.normal(0.0, 0.03, n)).astype(np.float32), dtype=float, requires_grad=True)
    flux = wp.zeros(n, dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    lr = 2.0
    for it in range(300):
        loss.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(flux_kernel, dim=n, inputs=[r], outputs=[flux])
            wp.launch(clamp_kernel, dim=n, inputs=[flux, FCAP])
            wp.launch(mse_kernel, dim=n, inputs=[flux, spec, 1.0 / n], outputs=[loss])
        tape.backward(loss=loss)
        g = r.grad.numpy().copy()
        tape.zero()
        r = wp.array(r.numpy() - lr * g, dtype=float, requires_grad=True)
        if it % 60 == 0 or it == 299:
            print(f"iter {it:4d}  loss={loss.numpy()[0]:.6e}  |grad|={np.linalg.norm(g):.3e}")


if __name__ == "__main__":
    main()
