# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Iterative relaxation solver for a chain of particles under gravity.

Each relaxation sweep moves every particle toward the rest length of the
springs connecting it to its neighbors and applies a small gravity
displacement. The endpoints of the chain are pinned.
"""

import warp as wp


@wp.kernel
def relax_sweep(
    q: wp.array[wp.vec3],
    rest_length: float,
    stiffness: float,
    gravity: float,
    q_next: wp.array[wp.vec3],
):
    i = wp.tid()
    n = q.shape[0]

    p = q[i]
    correction = wp.vec3(0.0, -gravity, 0.0)

    if i > 0:
        d = p - q[i - 1]
        stretch = wp.length(d) - rest_length
        correction -= stiffness * stretch * wp.normalize(d)

    if i < n - 1:
        d = p - q[i + 1]
        stretch = wp.length(d) - rest_length
        correction -= stiffness * stretch * wp.normalize(d)

    # the endpoints of the chain are pinned
    if i == 0 or i == n - 1:
        correction = wp.vec3(0.0, 0.0, 0.0)

    q_next[i] = p + correction


def relax(q: wp.array, num_sweeps: int, rest_length: float, stiffness: float, gravity: float) -> wp.array:
    """Run ``num_sweeps`` relaxation sweeps, returning the new positions.

    Each sweep writes into a freshly allocated output array so the tape sees
    distinct buffers for every sweep.
    """
    for _ in range(num_sweeps):
        q_next = wp.empty_like(q)
        wp.launch(relax_sweep, dim=q.shape[0], inputs=[q, rest_length, stiffness, gravity], outputs=[q_next])
        q = q_next
    return q
