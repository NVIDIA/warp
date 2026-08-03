# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robust linear regression with a Huber loss, fit with gradient descent."""

import numpy as np

import warp as wp

N = 64
DELTA = wp.constant(1.0)
INV_N = wp.constant(1.0 / N)


@wp.func
def huber(x: float) -> float:
    ax = wp.abs(x)
    quad = 0.5 * x * x
    lin = DELTA * (ax - 0.5 * DELTA)
    return wp.where(ax <= DELTA, quad, lin)


# Custom gradient: wp.abs() has kinks at 0 and +-DELTA, so we provide the
# analytic Huber derivative ourselves instead of relying on autodiff of abs().
@wp.func_grad(huber)
def adj_huber(x: float, adj_ret: float):
    ax = wp.abs(x)
    g = wp.where(ax <= DELTA, x, x / DELTA)
    wp.adjoint[x] += g * adj_ret


@wp.kernel
def huber_loss(
    params: wp.array(dtype=float),
    ts: wp.array(dtype=float),
    targets: wp.array(dtype=float),
    loss: wp.array(dtype=float),
):
    i = wp.tid()
    pred = params[0] + params[1] * ts[i]
    r = pred - targets[i]
    wp.atomic_add(loss, 0, huber(r) * INV_N)


def make_data(rng):
    t = np.linspace(0.0, 1.0, N).astype(np.float32)
    y = (0.5 + 2.0 * t + rng.normal(0.0, 0.05, N)).astype(np.float32)
    # ~20% of the sensor readings are saturated outliers
    idx = rng.choice(N, N // 5, replace=False)
    y[idx] += 8.0
    return t, y


def main():
    rng = np.random.default_rng(3)
    t_np, y_np = make_data(rng)
    ts = wp.array(t_np, dtype=float)
    targets = wp.array(y_np, dtype=float)
    params = wp.array([0.0, 0.0], dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    lr = 0.5
    for it in range(400):
        loss.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(huber_loss, dim=N, inputs=[params, ts, targets], outputs=[loss])
        tape.backward(loss=loss)
        g = params.grad.numpy().copy()
        p = params.numpy()
        params = wp.array(p - lr * g, dtype=float, requires_grad=True)
        if it % 50 == 0 or it == 399:
            print(f"iter {it:4d}  loss={loss.numpy()[0]:.4f}  a={p[0]:+.4f}  b={p[1]:+.4f}")

    print("true inlier model: a=+0.5  b=+2.0")


if __name__ == "__main__":
    main()
