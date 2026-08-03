# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import warp as wp

NUM_PARTICLES = 8
NUM_STEPS = 160
DT = 5.0e-4
SPRING_K = 2.0e3
SPRING_K3 = 5.0e5
REST_LEN = 0.1
MASS = 1.0e-2
DAMPING = 5.0e-2


@wp.kernel
def step_kernel(
    x: wp.array[float],
    v: wp.array[float],
    x_out: wp.array[float],
    v_out: wp.array[float],
):
    i = wp.tid()

    x_left = 0.0
    if i > 0:
        x_left = x[i - 1]

    s = x[i] - x_left - REST_LEN
    f = -(SPRING_K * s + SPRING_K3 * s * s * s)

    if i + 1 < NUM_PARTICLES:
        s = x[i + 1] - x[i] - REST_LEN
        f += SPRING_K * s + SPRING_K3 * s * s * s

    f -= DAMPING * v[i]

    v_new = v[i] + DT * f / MASS
    v_out[i] = v_new
    x_out[i] = x[i] + DT * v_new


@wp.kernel
def loss_kernel(x: wp.array[float], target: wp.array[float], loss: wp.array[float]):
    i = wp.tid()
    d = x[i] - target[i]
    wp.atomic_add(loss, 0, d * d)


def rest_positions():
    return np.arange(1, NUM_PARTICLES + 1, dtype=np.float32) * REST_LEN


def allocate_states(num_steps):
    states_x = [wp.zeros(NUM_PARTICLES, dtype=float, requires_grad=True) for _ in range(num_steps)]
    states_v = [wp.zeros(NUM_PARTICLES, dtype=float, requires_grad=True) for _ in range(num_steps)]
    return states_x, states_v


def simulate(x_init, v0, states_x, states_v, num_steps):
    wp.launch(step_kernel, dim=NUM_PARTICLES, inputs=[x_init, v0], outputs=[states_x[0], states_v[0]])
    for k in range(1, num_steps):
        wp.launch(
            step_kernel,
            dim=NUM_PARTICLES,
            inputs=[states_x[k - 1], states_v[k - 1]],
            outputs=[states_x[k], states_v[k]],
        )


def rollout(x_init, v0, states_x, states_v, num_steps, target, loss):
    simulate(x_init, v0, states_x, states_v, num_steps)
    wp.launch(loss_kernel, dim=NUM_PARTICLES, inputs=[states_x[num_steps - 1], target], outputs=[loss])


def true_initial_velocity(seed=42, scale=1.0, noise=0.001):
    rng = np.random.default_rng(seed)
    profile = np.sin(np.pi * np.arange(1, NUM_PARTICLES + 1) / (NUM_PARTICLES + 1))
    return (scale * profile + noise * rng.standard_normal(NUM_PARTICLES)).astype(np.float32)


def make_target(num_steps, seed=42):
    x_init = wp.array(rest_positions(), dtype=float)
    v0 = wp.array(true_initial_velocity(seed=seed), dtype=float)
    states_x = [wp.zeros(NUM_PARTICLES, dtype=float) for _ in range(num_steps)]
    states_v = [wp.zeros(NUM_PARTICLES, dtype=float) for _ in range(num_steps)]
    simulate(x_init, v0, states_x, states_v, num_steps)
    return states_x[num_steps - 1].numpy()
