# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable model of a 1D photodiode sensor strip.

Each channel converts a raw optical signal into a response through a
per-channel gain/bias followed by a softplus saturation. Interior channels
additionally pick up crosstalk that leaks in from the raw signal of their
immediate neighbors. The gains and biases are the trainable calibration
parameters.
"""

import warp as wp


@wp.func
def softplus(x: float) -> float:
    return wp.log(1.0 + wp.exp(x))


@wp.kernel
def direct_response(
    gain: wp.array(dtype=float),
    bias: wp.array(dtype=float),
    signal: wp.array(dtype=float),
    response: wp.array(dtype=float),
):
    i = wp.tid()
    response[i] = softplus(gain[i] * signal[i] + bias[i])


@wp.kernel
def add_crosstalk(
    gain: wp.array(dtype=float),
    bias: wp.array(dtype=float),
    signal: wp.array(dtype=float),
    kappa: float,
    response: wp.array(dtype=float),
):
    # interior channels only; the strip endpoints have no two-sided neighbors
    i = wp.tid() + 1

    # recompute the direct term here rather than reading ``response``
    # (reading and writing the same array within one kernel is not
    # differentiable in Warp)
    direct = softplus(gain[i] * signal[i] + bias[i])
    leak = kappa * (signal[i - 1] + signal[i + 1])
    response[i] = direct + leak


@wp.kernel
def sse_loss(
    response: wp.array(dtype=float),
    target: wp.array(dtype=float),
    loss: wp.array(dtype=float),
):
    i = wp.tid()
    d = response[i] - target[i]
    wp.atomic_add(loss, 0, d * d)


def forward(
    gain: wp.array,
    bias: wp.array,
    signal: wp.array,
    kappa: float,
    target: wp.array,
):
    """Run the sensor model and return ``(response, loss)`` arrays."""
    n = gain.shape[0]

    # keep grads for inspection after backward
    response = wp.zeros(n, dtype=float, requires_grad=True, retain_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    wp.launch(direct_response, dim=n, inputs=[gain, bias, signal], outputs=[response])
    wp.launch(add_crosstalk, dim=n - 2, inputs=[gain, bias, signal, kappa], outputs=[response])
    wp.launch(sse_loss, dim=n, inputs=[response, target], outputs=[loss])

    return response, loss
