# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layered filter transmission model.

Each sensor's ray crosses a run of filter panes. Every pane has a trainable
absorption coefficient, and attenuates the ray by a factor that depends on
the coefficient and the pane thickness. Sensors sit at different depths in
the stack, so each ray crosses a different number of panes; the crossings
are stored in a flat list with per-sensor offsets.
"""

import warp as wp


@wp.kernel
def compute_transmission(
    absorb: wp.array[float],
    pane: wp.array[int],
    offsets: wp.array[int],
    thickness: wp.array[float],
    trans: wp.array[float],
):
    i = wp.tid()
    start = offsets[i]
    end = offsets[i + 1]

    t = float(1.0)
    for k in range(start, end):
        t = t * (1.0 - absorb[pane[k]] * thickness[k])

    trans[i] = t


@wp.kernel
def compute_loss(
    trans: wp.array[float],
    target: wp.array[float],
    loss: wp.array[float],
):
    i = wp.tid()
    d = trans[i] - target[i]
    wp.atomic_add(loss, 0, d * d)


def run_model(
    absorb: wp.array,
    pane: wp.array,
    offsets: wp.array,
    thickness: wp.array,
    target: wp.array,
) -> wp.array:
    """Run one forward pass and return the scalar loss array.

    Launches must be recorded on an active tape by the caller if gradients
    are required.
    """
    n = target.shape[0]

    trans = wp.zeros(n, dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    wp.launch(compute_transmission, dim=n, inputs=[absorb, pane, offsets, thickness], outputs=[trans])
    wp.launch(compute_loss, dim=n, inputs=[trans, target], outputs=[loss])

    return loss
