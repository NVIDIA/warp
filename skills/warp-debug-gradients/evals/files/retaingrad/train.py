# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate the sensor strip's per-channel gains and biases.

A reference measurement (``target``) is produced by a strip with known
calibration; training recovers the gains/biases from a perturbed initial
guess using tape gradients and plain gradient descent.
"""

import model
import numpy as np

import warp as wp

N_CHANNELS = 32
KAPPA = 0.05
LEARNING_RATE = 0.4
TRAIN_ITERS = 40
SEED = 7


def make_data(rng: np.random.Generator):
    signal = rng.uniform(0.5, 1.5, size=N_CHANNELS).astype(np.float32)

    gain_true = rng.uniform(0.8, 1.2, size=N_CHANNELS).astype(np.float32)
    bias_true = rng.uniform(-0.2, 0.2, size=N_CHANNELS).astype(np.float32)

    # reference response of the correctly calibrated strip
    x = gain_true * signal + bias_true
    target = np.log1p(np.exp(x)).astype(np.float32)
    target[1:-1] += KAPPA * (signal[:-2] + signal[2:])

    # perturbed initial guess for the calibration
    gain0 = gain_true + rng.normal(0.0, 0.25, size=N_CHANNELS).astype(np.float32)
    bias0 = bias_true + rng.normal(0.0, 0.25, size=N_CHANNELS).astype(np.float32)

    return signal, target, gain0, bias0


def main():
    rng = np.random.default_rng(SEED)
    signal_np, target_np, gain_np, bias_np = make_data(rng)

    signal = wp.array(signal_np, dtype=float)
    target = wp.array(target_np, dtype=float)

    for it in range(TRAIN_ITERS):
        gain = wp.array(gain_np, dtype=float, requires_grad=True)
        bias = wp.array(bias_np, dtype=float, requires_grad=True)

        tape = wp.Tape()
        with tape:
            _response, loss = model.forward(gain, bias, signal, KAPPA, target)
        tape.backward(loss)

        gain_grad = gain.grad.numpy()
        bias_grad = bias.grad.numpy()

        if it % 5 == 0 or it == TRAIN_ITERS - 1:
            gnorm = np.sqrt(np.sum(gain_grad**2) + np.sum(bias_grad**2))
            print(f"iter {it:02d}  loss {loss.numpy()[0]:12.8f}  grad_norm {gnorm:10.6f}")

        gain_np = gain_np - LEARNING_RATE * gain_grad
        bias_np = bias_np - LEARNING_RATE * bias_grad


if __name__ == "__main__":
    main()
