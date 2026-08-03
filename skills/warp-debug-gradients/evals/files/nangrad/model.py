# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""2D multilateration: recover transmitter positions from measured ranges to fixed anchors."""

import numpy as np

import warp as wp

NUM_POINTS = 6
NUM_ANCHORS = 4
DOMAIN = 2.0


@wp.kernel
def range_loss_kernel(
    points: wp.array[wp.vec2],
    anchors: wp.array[wp.vec2],
    ranges: wp.array2d[float],
    loss: wp.array[float],
):
    i, j = wp.tid()
    v = points[i] - anchors[j]
    d = wp.sqrt(wp.dot(v, v))
    r = d - ranges[i, j]
    wp.atomic_add(loss, 0, r * r)


def anchor_positions():
    return np.array(
        [[0.0, 0.0], [DOMAIN, 0.0], [0.0, DOMAIN], [DOMAIN, DOMAIN]],
        dtype=np.float32,
    )


def true_positions(seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.25, DOMAIN - 0.25, size=(NUM_POINTS, 2)).astype(np.float32)


def measured_ranges(seed=42):
    p = true_positions(seed)
    a = anchor_positions()
    diff = p[:, None, :] - a[None, :, :]
    return np.linalg.norm(diff, axis=2).astype(np.float32)


def initial_guess():
    # Start each point at one of the anchors.
    a = anchor_positions()
    return a[np.arange(NUM_POINTS) % NUM_ANCHORS].copy()


def compute_loss(points, anchors, ranges, loss):
    wp.launch(
        range_loss_kernel,
        dim=(NUM_POINTS, NUM_ANCHORS),
        inputs=[points, anchors, ranges],
        outputs=[loss],
    )
