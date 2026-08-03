# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory optimization for a damped oscillator with gradient checkpointing.

On the real problem (millions of particles) taping the whole horizon OOMs, so
we checkpoint the state every S steps during an untaped forward pass, then
during backward we replay one segment at a time under a short tape and chain
the state adjoints across segments (standard gradient checkpointing). This
repro keeps that exact structure at toy size.

Objective: drive the oscillator to x = 1 at the final step with small controls:
    L = (x_T - 1)^2 + 1e-3 * sum_t u_t^2
"""

import numpy as np

import warp as wp

T = 64  # horizon
S = 16  # checkpoint interval
DT = 0.1
K = 1.5  # stiffness
C = 0.4  # damping
TARGET = 1.0


@wp.kernel
def step_kernel(
    x: wp.array(dtype=float),
    v: wp.array(dtype=float),
    u: wp.array(dtype=float),
    t: int,
    xo: wp.array(dtype=float),
    vo: wp.array(dtype=float),
):
    vn = v[0] + DT * (-K * x[0] - C * v[0] + u[t])
    vo[0] = vn
    xo[0] = x[0] + DT * vn


@wp.kernel
def loss_kernel(x: wp.array(dtype=float), u: wp.array(dtype=float), loss: wp.array(dtype=float)):
    d = x[0] - TARGET
    reg = float(0.0)
    for t in range(T):
        reg += u[t] * u[t]
    loss[0] = d * d + 1.0e-3 * reg


def _state(val=0.0):
    return wp.array([val], dtype=float, requires_grad=True)


def step(x, v, u, t):
    xo, vo = _state(), _state()
    wp.launch(step_kernel, dim=1, inputs=[x, v, u, t], outputs=[xo, vo])
    return xo, vo


def forward_with_checkpoints(u):
    """Untaped forward pass; saves a state checkpoint every S steps."""
    x, v = _state(0.0), _state(0.0)
    ckpts = []
    for t in range(T):
        if t % S == 0:
            ckpts.append((x.numpy().copy(), v.numpy().copy()))
        x, v = step(x, v, u, t)
    return x, ckpts


def replay_segment(u, ckpt, t0, n_steps):
    """Re-run steps [t0, t0+n_steps) under a tape from checkpoint state."""
    x0 = wp.array(ckpt[0], dtype=float, requires_grad=True)
    v0 = wp.array(ckpt[1], dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
        x, v = x0, v0
        for t in range(t0, t0 + n_steps):
            x, v = step(x, v, u, t)
    return tape, x0, v0, x, v


def checkpointed_backward(u, ckpts):
    """Chain adjoints backward across segments; accumulates into u.grad."""
    n_seg = T // S
    x_adj = None
    v_adj = None
    for j in range(n_seg - 1, -1, -1):
        tape, x0, v0, x_end, v_end = replay_segment(u, ckpts[j], j * S, S)
        if j == n_seg - 1:
            loss = _state(0.0)
            with tape:
                wp.launch(loss_kernel, dim=1, inputs=[x_end, u], outputs=[loss])
            tape.backward(loss=loss)
        else:
            # Seed the segment-end adjoint with the sensitivity carried back
            # from the later segments. The objective only reads position, so
            # the position adjoint is what propagates between segments.
            x_end.grad.assign(x_adj)
            tape.backward()
        x_adj = x0.grad.numpy().copy()
        v_adj = v0.grad.numpy().copy()


def main():
    rng = np.random.default_rng(1)
    u = wp.array(np.zeros(T, dtype=np.float32), dtype=float, requires_grad=True)

    lr = 2.0
    for it in range(200):
        u.grad.zero_()
        x_final, ckpts = forward_with_checkpoints(u)
        checkpointed_backward(u, ckpts)
        g = u.grad.numpy().copy()
        u = wp.array(u.numpy() - lr * g, dtype=float, requires_grad=True)
        if it % 40 == 0 or it == 199:
            xf = x_final.numpy()[0]
            print(f"iter {it:4d}  x_T={xf:+.5f}  (target {TARGET})  |u|={np.linalg.norm(u.numpy()):.4f}")

    x_final, _ = forward_with_checkpoints(u)
    print(f"final x_T = {x_final.numpy()[0]:+.5f}, should be within ~0.02 of {TARGET}")


if __name__ == "__main__":
    main()
