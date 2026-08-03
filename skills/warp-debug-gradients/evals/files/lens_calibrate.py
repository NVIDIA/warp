# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate a radial lens-distortion coefficient k from feature detections.

Model: r_observed = r2 * (1 + k * r2). Detections are noisy, so each forward
pass applies a fresh sensor-noise augmentation (standard data-augmentation
practice to avoid overfitting the detector noise).
"""

import numpy as np

import warp as wp

N = 256
JITTER = 0.05

_eval_count = 0  # decorrelates the augmentation noise across forward passes


@wp.kernel
def distortion_loss(
    k: wp.array(dtype=float),
    r2: wp.array(dtype=float),
    obs: wp.array(dtype=float),
    seed: int,
    loss: wp.array(dtype=float),
):
    i = wp.tid()
    state = wp.rand_init(seed, i)
    jitter = JITTER * wp.randn(state)
    pred = r2[i] * (1.0 + k[0] * r2[i])
    d = pred - (obs[i] + jitter)
    wp.atomic_add(loss, 0, d * d / float(N))


def forward(k, r2, obs, loss):
    """One augmented forward pass of the calibration objective."""
    global _eval_count
    _eval_count += 1
    loss.zero_()
    wp.launch(distortion_loss, dim=N, inputs=[k, r2, obs, _eval_count], outputs=[loss])


def main():
    rng = np.random.default_rng(7)
    r2_np = rng.uniform(0.1, 2.0, N).astype(np.float32)
    obs_np = (r2_np * (1.0 + 0.3 * r2_np)).astype(np.float32)  # true k = 0.3

    r2 = wp.array(r2_np, dtype=float)
    obs = wp.array(obs_np, dtype=float)
    k = wp.array([0.0], dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    lr = 0.05
    for it in range(300):
        tape = wp.Tape()
        with tape:
            forward(k, r2, obs, loss)
        tape.backward(loss=loss)
        g = k.grad.numpy().copy()
        k = wp.array(k.numpy() - lr * g, dtype=float, requires_grad=True)
        if it % 50 == 0 or it == 299:
            print(f"iter {it:4d}  loss={loss.numpy()[0]:.6e}  k={k.numpy()[0]:+.5f}")

    print("loss should reach ~1e-6 for a well-calibrated k, but it plateaus")


if __name__ == "__main__":
    main()
