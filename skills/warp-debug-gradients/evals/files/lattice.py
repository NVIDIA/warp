# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover the initial state of a coupled logistic lattice from its final state.

Forward model: x_{t+1}[i] = (1-c) * f(x_t[i]) + c/2 * (f(x_t[i-1]) + f(x_t[i+1]))
with the logistic map f(x) = r x (1 - x), periodic neighbors.
We observe the lattice after T steps and optimize the initial state x0.
"""

import numpy as np

import warp as wp

N = 8
T = 60
R = wp.constant(3.9)
CPL = wp.constant(0.1)
NN = wp.constant(N)
INV_N = wp.constant(1.0 / N)


@wp.func
def lmap(x: float) -> float:
    return R * x * (1.0 - x)


@wp.kernel
def step_kernel(xin: wp.array(dtype=float), xout: wp.array(dtype=float)):
    i = wp.tid()
    im = (i - 1 + NN) % NN
    ip = (i + 1) % NN
    xout[i] = (1.0 - CPL) * lmap(xin[i]) + 0.5 * CPL * (lmap(xin[im]) + lmap(xin[ip]))


@wp.kernel
def copy_kernel(src: wp.array(dtype=float), dst: wp.array(dtype=float)):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def mse_kernel(x: wp.array(dtype=float), target: wp.array(dtype=float), loss: wp.array(dtype=float)):
    i = wp.tid()
    d = x[i] - target[i]
    wp.atomic_add(loss, 0, d * d * INV_N)


def rollout_np(x0, steps):
    x = x0.astype(np.float32).copy()
    for _ in range(steps):
        f = (3.9 * x * (1.0 - x)).astype(np.float32)
        x = ((1.0 - 0.1) * f + 0.05 * (np.roll(f, 1) + np.roll(f, -1))).astype(np.float32)
    return x


def main():
    rng = np.random.default_rng(4)
    x0_true = rng.uniform(0.2, 0.8, N).astype(np.float32)
    target = wp.array(rollout_np(x0_true, T), dtype=float)

    x0 = wp.array((x0_true + 0.05).clip(0.05, 0.95).astype(np.float32), dtype=float, requires_grad=True)
    # ping-pong state buffers (only two needed since each step reads one, writes the other)
    xa = wp.zeros(N, dtype=float, requires_grad=True)
    xb = wp.zeros(N, dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    lr = 1.0e-6
    for it in range(20):
        loss.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(copy_kernel, dim=N, inputs=[x0], outputs=[xa])
            cur, nxt = xa, xb
            for _ in range(T):
                wp.launch(step_kernel, dim=N, inputs=[cur], outputs=[nxt])
                cur, nxt = nxt, cur
            wp.launch(mse_kernel, dim=N, inputs=[cur, target], outputs=[loss])
        tape.backward(loss=loss)
        g = x0.grad.numpy().copy()
        tape.zero()
        x0 = wp.array(x0.numpy() - lr * g, dtype=float, requires_grad=True)
        print(f"iter {it:3d}  loss={loss.numpy()[0]:.6e}  |grad|={np.linalg.norm(g):.3e}")


if __name__ == "__main__":
    main()
