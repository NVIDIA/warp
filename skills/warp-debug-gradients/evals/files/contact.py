# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate contact stiffness so the model's mean penalty energy matches a
reference measurement.

Penalty energy per contact: softplus(k * d), the standard smooth barrier.
softplus overflows for large arguments, so we guard it with wp.where — for
z > 20 softplus(z) is z to float precision anyway.
"""

import numpy as np

import warp as wp

N = 256
INV_N = wp.constant(1.0 / N)


@wp.func
def softplus(z: float) -> float:
    # guard the overflow: for z > 20, log(1+exp(z)) == z in float32
    return wp.where(z > 20.0, z, wp.log(1.0 + wp.exp(z)))


@wp.kernel
def mean_energy(
    k: wp.array(dtype=float),
    d: wp.array(dtype=float),
    energy: wp.array(dtype=float),
):
    i = wp.tid()
    e = softplus(k[0] * d[i])
    wp.atomic_add(energy, 0, e * INV_N)


@wp.kernel
def energy_loss(energy: wp.array(dtype=float), e_ref: float, loss: wp.array(dtype=float)):
    diff = energy[0] - e_ref
    loss[0] = diff * diff


def mean_energy_np(k, d):
    z = (k * d).astype(np.float64)
    return float(np.mean(np.where(z > 20.0, z, np.log1p(np.exp(np.minimum(z, 20.0))))))


def main():
    rng = np.random.default_rng(15)
    # penetration depths: mostly shallow, a few deep contacts (production scene)
    d_np = np.concatenate([rng.uniform(0.01, 0.4, N - 8), rng.uniform(1.2, 1.5, 8)]).astype(np.float32)
    k_true = 65.0
    e_ref = mean_energy_np(k_true, d_np)

    d = wp.array(d_np, dtype=float)
    k = wp.array([80.0], dtype=float, requires_grad=True)
    energy = wp.zeros(1, dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    lr = 2.0
    for it in range(100):
        energy.zero_()
        loss.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(mean_energy, dim=N, inputs=[k, d], outputs=[energy])
            wp.launch(energy_loss, dim=1, inputs=[energy, e_ref], outputs=[loss])
        tape.backward(loss=loss)
        g = k.grad.numpy().copy()
        tape.zero()
        k = wp.array(k.numpy() - lr * g, dtype=float, requires_grad=True)
        if it % 20 == 0 or it == 99:
            print(f"iter {it:3d}  loss={loss.numpy()[0]:.6e}  k={k.numpy()[0]:.4f}  dL/dk={g[0]:+.6e}")

    print(f"expected k -> {k_true}")


if __name__ == "__main__":
    main()
