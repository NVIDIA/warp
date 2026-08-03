# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-contact energy model with a smoothed friction term.

Each contact point ``i`` has a scalar displacement ``u_i`` driven by a
parameter ``theta_i`` through a fixed gain. The energy of a contact is

    E_i(u) = mu * smooth_abs(u) + 0.5 * k_i * u^2 - f_i * u

i.e. a smoothed Coulomb friction penalty, an elastic term, and the work done
by the external force. ``smooth_abs`` carries a hand-written gradient so the
derivative stays finite at zero displacement.
"""

import warp as wp

EPS = 1.0e-4


@wp.func
def smooth_abs(x: float, eps: float) -> float:
    return wp.sqrt(x * x + eps)


@wp.func_grad(smooth_abs)
def adj_smooth_abs(x: float, eps: float, adj_ret: float):
    # d/dx sqrt(x^2 + eps) = x / sqrt(x^2 + eps), finite everywhere
    wp.adjoint[x] = adj_ret * x / wp.sqrt(x * x + eps)


@wp.kernel
def apply_gain(
    theta: wp.array(dtype=float),
    gain: wp.array(dtype=float),
    disp: wp.array(dtype=float),
):
    i = wp.tid()
    disp[i] = theta[i] * gain[i]


@wp.kernel
def contact_energy(
    disp: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    force: wp.array(dtype=float),
    mu: float,
    loss: wp.array(dtype=float),
):
    i = wp.tid()
    u = disp[i]
    e = mu * smooth_abs(u, EPS) + 0.5 * stiffness[i] * u * u - force[i] * u
    wp.atomic_add(loss, 0, e)
