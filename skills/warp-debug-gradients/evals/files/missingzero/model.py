# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Per-channel sensor calibration model.
#
# Each channel i applies a soft-cubic gain response to its excitation and
# adds a per-channel offset:
#
#   pred_i = (w_i + 0.1 * w_i^3) * x_i + b_i
#
# The loss is the sum of squared residuals against measured readings.

import numpy as np

import warp as wp


@wp.kernel
def calibration_loss(
    w: wp.array[float],
    b: wp.array[float],
    x: wp.array[float],
    y: wp.array[float],
    loss: wp.array[float],
):
    i = wp.tid()
    g = w[i]
    pred = (g + 0.1 * g * g * g) * x[i] + b[i]
    r = pred - y[i]
    wp.atomic_add(loss, 0, r * r)


def forward(w, b, x, y, loss):
    """Evaluate the calibration loss for the current parameters.

    Returns the single-element ``loss`` array.
    """
    loss.zero_()
    wp.launch(calibration_loss, dim=x.shape[0], inputs=[w, b, x, y], outputs=[loss])
    return loss


def make_problem(n: int, seed: int, device=None):
    """Build a synthetic calibration problem.

    Returns ``(w, b, x, y, loss)`` where ``w`` and ``b`` are the trainable
    parameters (initialized to zero) and ``y`` are readings generated from
    hidden ground-truth gains/offsets.
    """
    rng = np.random.default_rng(seed)
    true_w = rng.uniform(0.5, 1.5, n)
    true_b = rng.uniform(-0.5, 0.5, n)
    x = rng.uniform(0.5, 2.0, n)
    y = (true_w + 0.1 * true_w**3) * x + true_b

    w = wp.zeros(n, dtype=float, device=device, requires_grad=True)
    b = wp.zeros(n, dtype=float, device=device, requires_grad=True)
    x_wp = wp.array(x.astype(np.float32), dtype=float, device=device)
    y_wp = wp.array(y.astype(np.float32), dtype=float, device=device)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    return w, b, x_wp, y_wp, loss
