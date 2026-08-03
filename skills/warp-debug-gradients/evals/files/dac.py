# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate per-channel gains for a transmitter driving an 8-bit DAC.

The physical output is quantized by the DAC: out = round(gain * x * 256) / 256.
We calibrate the gains so the DAC output matches reference measurements.
"""

import numpy as np

import warp as wp

N = 64
LEVELS = wp.constant(256.0)
INV_N = wp.constant(1.0 / N)


@wp.kernel
def dac_loss(
    gains: wp.array(dtype=float),
    x: wp.array(dtype=float),
    target: wp.array(dtype=float),
    loss: wp.array(dtype=float),
):
    i = wp.tid()
    out = wp.round(gains[i] * x[i] * LEVELS) / LEVELS
    d = out - target[i]
    wp.atomic_add(loss, 0, d * d * INV_N)


def main():
    rng = np.random.default_rng(6)
    x_np = rng.uniform(0.3, 0.9, N).astype(np.float32)
    g_true = rng.uniform(0.8, 1.2, N).astype(np.float32)
    target_np = (np.round(g_true * x_np * 256.0) / 256.0).astype(np.float32)

    x = wp.array(x_np, dtype=float)
    target = wp.array(target_np, dtype=float)
    gains = wp.array(np.ones(N, dtype=np.float32), dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    lr = 5.0
    for it in range(200):
        loss.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(dac_loss, dim=N, inputs=[gains, x, target], outputs=[loss])
        tape.backward(loss=loss)
        g = gains.grad.numpy().copy()
        tape.zero()
        gains = wp.array(gains.numpy() - lr * g, dtype=float, requires_grad=True)
        if it % 40 == 0 or it == 199:
            print(f"iter {it:4d}  loss={loss.numpy()[0]:.6e}  |grad|={np.linalg.norm(g):.3e}")


if __name__ == "__main__":
    main()
