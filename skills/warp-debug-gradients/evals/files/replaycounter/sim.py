# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Threshold-gated particle compaction model.

Each particle produces a response from a trainable per-particle gain.
Particles whose response exceeds a fixed activation threshold are appended to
a compact list of active values via an atomic cursor, and a scoring kernel
compares the active values against a target level.
"""

import warp as wp

ACTIVATION_THRESHOLD = 1.9


@wp.func
def claim_slot(cursor: wp.array[int], slot_of: wp.array[int], pid: int):
    """Reserve the next free slot in the compact list for particle ``pid``."""
    slot = wp.atomic_add(cursor, 0, 1)
    # record which slot each particle landed in
    slot_of[pid] = slot
    return slot


@wp.kernel
def compute_response(
    gains: wp.array[float],
    positions: wp.array[float],
    response: wp.array[float],
):
    i = wp.tid()
    g = gains[i]
    x = positions[i]
    response[i] = g * g * x + wp.sin(2.0 * g)


@wp.kernel
def compact_active(
    response: wp.array[float],
    cursor: wp.array[int],
    slot_of: wp.array[int],
    active_values: wp.array[float],
):
    i = wp.tid()
    r = response[i]
    if r > ACTIVATION_THRESHOLD:
        slot = claim_slot(cursor, slot_of, i)
        active_values[slot] = r


@wp.kernel
def score_active(
    active_values: wp.array[float],
    num_active: wp.array[int],
    level: float,
    loss: wp.array[float],
):
    j = wp.tid()
    if j < num_active[0]:
        d = active_values[j] - level
        wp.atomic_add(loss, 0, d * d)


@wp.kernel
def gain_penalty(
    gains: wp.array[float],
    beta: float,
    loss: wp.array[float],
):
    i = wp.tid()
    d = gains[i] - 1.0
    wp.atomic_add(loss, 0, beta * d * d)


def run_episode(gains: wp.array, positions: wp.array, level: float, beta: float) -> wp.array:
    """Run one forward pass and return the scalar loss array.

    Launches must be recorded on an active tape by the caller if gradients
    are required.
    """
    n = gains.shape[0]

    cursor = wp.zeros(1, dtype=int)
    slot_of = wp.zeros(n, dtype=int)
    response = wp.zeros(n, dtype=float, requires_grad=True)
    active_values = wp.zeros(n, dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    wp.launch(compute_response, dim=n, inputs=[gains, positions], outputs=[response])
    wp.launch(compact_active, dim=n, inputs=[response, cursor, slot_of], outputs=[active_values])
    wp.launch(score_active, dim=n, inputs=[active_values, cursor, level], outputs=[loss])
    wp.launch(gain_penalty, dim=n, inputs=[gains, beta], outputs=[loss])

    return loss
