# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import warp as wp

NUM_PARTICLES = 24
NUM_STEPS = 24
DT = 2.0e-2
SPRING_K = 60.0
REST_LEN = 0.25
DRAG = 1.5
GRAVITY_Y = -1.0
SPACING = 0.3


@wp.kernel
def step_kernel(
    q: wp.array[wp.vec2],
    qd: wp.array[wp.vec2],
    anchors: wp.array[wp.vec2],
    q_out: wp.array[wp.vec2],
    qd_out: wp.array[wp.vec2],
):
    i = wp.tid()

    d = q[i] - anchors[i]
    stretch = wp.length(d) - REST_LEN
    f = -SPRING_K * stretch * wp.normalize(d)

    v = qd[i]
    f -= DRAG * wp.length(v) * v
    f += wp.vec2(0.0, GRAVITY_Y)

    v_new = v + DT * f
    qd_out[i] = v_new
    q_out[i] = q[i] + DT * v_new


@wp.kernel
def reset_state(rest: wp.array[wp.vec2], q: wp.array[wp.vec2], qd: wp.array[wp.vec2]):
    i = wp.tid()
    q[i] = rest[i]
    qd[i] = wp.vec2(0.0, 0.0)


@wp.kernel
def speed_metrics(qd: wp.array[wp.vec2], speeds: wp.array[float]):
    i = wp.tid()
    speeds[i] = wp.length(qd[i])


@wp.kernel
def loss_kernel(q: wp.array[wp.vec2], target: wp.array[wp.vec2], loss: wp.array[float]):
    i = wp.tid()
    d = q[i] - target[i]
    wp.atomic_add(loss, 0, wp.dot(d, d))


def rest_positions():
    x = np.arange(NUM_PARTICLES, dtype=np.float32) * SPACING
    return np.stack([x, np.zeros(NUM_PARTICLES, dtype=np.float32)], axis=1)


def anchor_positions():
    rest = rest_positions()
    rest[:, 1] += REST_LEN
    return rest


def allocate_states(num_steps, requires_grad=True):
    states_q = [wp.zeros(NUM_PARTICLES, dtype=wp.vec2, requires_grad=requires_grad) for _ in range(num_steps)]
    states_qd = [wp.zeros(NUM_PARTICLES, dtype=wp.vec2, requires_grad=requires_grad) for _ in range(num_steps)]
    return states_q, states_qd


def simulate(q0, qd0, anchors, states_q, states_qd, num_steps):
    wp.launch(step_kernel, dim=NUM_PARTICLES, inputs=[q0, qd0, anchors], outputs=[states_q[0], states_qd[0]])
    for k in range(1, num_steps):
        wp.launch(
            step_kernel,
            dim=NUM_PARTICLES,
            inputs=[states_q[k - 1], states_qd[k - 1], anchors],
            outputs=[states_q[k], states_qd[k]],
        )


def rollout(q0, qd0, anchors, states_q, states_qd, num_steps, target, loss):
    simulate(q0, qd0, anchors, states_q, states_qd, num_steps)
    wp.launch(loss_kernel, dim=NUM_PARTICLES, inputs=[states_q[num_steps - 1], target], outputs=[loss])


def reset_states(rest, states_q, states_qd):
    for qk, qdk in zip(states_q, states_qd, strict=False):
        wp.launch(reset_state, dim=NUM_PARTICLES, inputs=[rest], outputs=[qk, qdk])


def true_control(seed=7, scale=0.6):
    rng = np.random.default_rng(seed)
    return (scale * rng.standard_normal((NUM_PARTICLES, 2))).astype(np.float32)


def make_target(num_steps, seed=7):
    q0 = wp.array(rest_positions(), dtype=wp.vec2)
    qd0 = wp.array(true_control(seed=seed), dtype=wp.vec2)
    anchors = wp.array(anchor_positions(), dtype=wp.vec2)
    states_q, states_qd = allocate_states(num_steps, requires_grad=False)
    simulate(q0, qd0, anchors, states_q, states_qd, num_steps)
    return states_q[num_steps - 1].numpy()
