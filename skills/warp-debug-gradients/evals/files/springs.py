# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-gate gradient check for our spring-chain energy calibration.

We calibrate 2D node positions p so the total elastic energy of the chain
matches a reference energy E_ref measured on hardware. Each node also carries
a constant reference potential (BASELINE) that is part of the certified energy
bookkeeping and must stay in the total.

Our release checklist requires a finite-difference check of dLoss/dp before we
ship — and it fails with complete nonsense (run this file to see it).
"""

import numpy as np

import warp as wp

N = 1024  # nodes; springs connect i to i+1
K = wp.constant(50.0)
BASELINE = wp.constant(1.0e4)
L0 = wp.constant(1.0)
EPS2 = wp.constant(1.0e-12)


@wp.func
def safe_len(dx: float, dy: float) -> float:
    return wp.sqrt(dx * dx + dy * dy + EPS2)


# Custom gradient: keep the adjoint finite for degenerate (zero-length) springs.
@wp.func_grad(safe_len)
def adj_safe_len(dx: float, dy: float, adj_ret: float):
    l = wp.sqrt(dx * dx + dy * dy + EPS2)
    # chain rule through u = dx^2 + dy^2 + eps, l = sqrt(u)
    dudx = 2.0 * dx
    dudy = 2.0 * dy
    dldu = 1.0 / l
    wp.adjoint[dx] += dudx * dldu * adj_ret
    wp.adjoint[dy] += dudy * dldu * adj_ret


@wp.kernel
def total_energy(p: wp.array(dtype=wp.vec2), energy: wp.array(dtype=float)):
    i = wp.tid()  # spring i connects node i and i+1
    d = p[i + 1] - p[i]
    l = safe_len(d[0], d[1])
    e = BASELINE + 0.5 * K * (l - L0) * (l - L0)
    wp.atomic_add(energy, 0, e)


@wp.kernel
def energy_loss(energy: wp.array(dtype=float), e_ref: float, loss: wp.array(dtype=float)):
    d = energy[0] - e_ref
    loss[0] = d * d


def make_positions(rng):
    # chain along x with spacing ~1.3 (springs stretched ~30%) plus jitter
    xs = np.cumsum(np.full(N, 1.3)) + rng.normal(0.0, 0.02, N)
    ys = rng.normal(0.0, 0.05, N)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def main():
    rng = np.random.default_rng(9)
    p_np = make_positions(rng)
    p = wp.array(p_np, dtype=wp.vec2, requires_grad=True)
    energy = wp.zeros(1, dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    # certified reference energy: all springs at rest length + baselines
    e_ref = float((N - 1) * 1.0e4)

    tape = wp.Tape()
    with tape:
        wp.launch(total_energy, dim=N - 1, inputs=[p], outputs=[energy])
        wp.launch(energy_loss, dim=1, inputs=[energy, e_ref], outputs=[loss])
    tape.backward(loss=loss)

    print(f"E = {energy.numpy()[0]:.6e}   E_ref = {e_ref:.6e}   loss = {loss.numpy()[0]:.6e}")
    g = p.grad.numpy()
    print(f"|dLoss/dp| max = {np.abs(g).max():.6e}, sample dLoss/dp[10] = {g[10]}")

    # release-gate FD check on a few coordinates
    def loss_of(pv):
        e2 = wp.zeros(1, dtype=float)
        l2 = wp.zeros(1, dtype=float)
        wp.launch(total_energy, dim=N - 1, inputs=[wp.array(pv, dtype=wp.vec2)], outputs=[e2])
        wp.launch(energy_loss, dim=1, inputs=[e2, e_ref], outputs=[l2])
        return float(l2.numpy()[0])

    eps = 1.0e-4
    print("\nFD check (central differences, eps=1e-4):")
    for idx, comp in [(10, 0), (10, 1), (500, 0), (900, 1)]:
        pp = p_np.copy()
        pp[idx, comp] += eps
        lp = loss_of(pp)
        pp[idx, comp] -= 2 * eps
        lm = loss_of(pp)
        fd = (lp - lm) / (2 * eps)
        print(f"  p[{idx}][{comp}]: AD = {g[idx][comp]:+.6e}   FD = {fd:+.6e}")


if __name__ == "__main__":
    main()
