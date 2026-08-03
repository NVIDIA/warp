# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable statics: tune per-node loads p so the structure's
equilibrium displacement matches a target profile.

Equilibrium satisfies u[i] = 0.45*(u[i-1] + u[i+1]) + 0.1*p[i] (periodic
ring). We solve it with fixed-point (Jacobi) iteration inside the tape and
backprop through the solver. SOLVER_ITERS = 15 keeps the taped iteration
count (and memory) modest.

QA acceptance: the final design is re-checked with the production solver
(400 iterations, fully converged).
"""

import numpy as np

import warp as wp

N = 32
NN = wp.constant(N)
INV_N = wp.constant(1.0 / N)
SOLVER_ITERS = 15
QA_ITERS = 400


@wp.kernel
def jacobi_step(u: wp.array(dtype=float), p: wp.array(dtype=float), un: wp.array(dtype=float)):
    i = wp.tid()
    im = (i - 1 + NN) % NN
    ip = (i + 1) % NN
    un[i] = 0.45 * (u[im] + u[ip]) + 0.1 * p[i]


@wp.kernel
def mse(u: wp.array(dtype=float), target: wp.array(dtype=float), loss: wp.array(dtype=float)):
    i = wp.tid()
    d = u[i] - target[i]
    wp.atomic_add(loss, 0, d * d * INV_N)


def solve(p, iters):
    """Fixed-point solve with one distinct buffer per iteration (tape-safe)."""
    bufs = [wp.zeros(N, dtype=float, requires_grad=True) for _ in range(iters + 1)]
    for t in range(iters):
        wp.launch(jacobi_step, dim=N, inputs=[bufs[t], p], outputs=[bufs[t + 1]])
    return bufs[iters]


def solve_np(p_np, iters):
    u = np.zeros(N, dtype=np.float64)
    for _ in range(iters):
        u = 0.45 * (np.roll(u, 1) + np.roll(u, -1)) + 0.1 * p_np
    return u


def main():
    rng = np.random.default_rng(21)
    p_true = rng.normal(0.0, 1.0, N).astype(np.float32)
    target_np = solve_np(p_true, QA_ITERS).astype(np.float32)
    target = wp.array(target_np, dtype=float)

    p = wp.array(np.zeros(N, dtype=np.float32), dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    lr = 30.0
    for it in range(2000):
        loss.zero_()
        tape = wp.Tape()
        with tape:
            u = solve(p, SOLVER_ITERS)
            wp.launch(mse, dim=N, inputs=[u, target], outputs=[loss])
        tape.backward(loss=loss)
        g = p.grad.numpy().copy()
        tape.zero()
        p = wp.array(p.numpy() - lr * g, dtype=float, requires_grad=True)
        if it % 400 == 0 or it == 1999:
            print(f"iter {it:5d}  training loss = {loss.numpy()[0]:.6e}")

    # QA acceptance check with the production solver
    u_qa = solve_np(p.numpy().astype(np.float64), QA_ITERS)
    qa_loss = float(np.mean((u_qa - target_np.astype(np.float64)) ** 2))
    print(f"QA loss (converged solver): {qa_loss:.6e}   (acceptance threshold 1e-6)")
    print(f"max |u_qa - target| = {np.abs(u_qa - target_np).max():.4f}")


if __name__ == "__main__":
    main()
