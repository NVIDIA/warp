# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Beer-Lambert attenuation model for a bank of detector channels.

An excitation signal passes through a stack of absorbing layers. Each layer
scales every channel by exp(-mu * thickness), where ``mu`` holds the
per-channel absorption coefficients.
"""

import warp as wp

NUM_LAYERS = 4
LAYER_THICKNESS = 0.25


@wp.kernel
def absorb_layer(mu: wp.array[float], thickness: float, signal: wp.array[float]):
    i = wp.tid()
    signal[i] *= wp.exp(-mu[i] * thickness)


@wp.kernel
def residual_loss(signal: wp.array[float], target: wp.array[float], loss: wp.array[float]):
    i = wp.tid()
    d = signal[i] - target[i]
    wp.atomic_add(loss, 0, d * d)


def run_stack(mu: wp.array, signal: wp.array, target: wp.array) -> wp.array:
    """Propagate ``signal`` through the layer stack and return the scalar loss array."""
    n = signal.shape[0]

    for _ in range(NUM_LAYERS):
        wp.launch(absorb_layer, dim=n, inputs=[mu, LAYER_THICKNESS], outputs=[signal])

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    wp.launch(residual_loss, dim=n, inputs=[signal, target], outputs=[loss])
    return loss
