# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deconvolution: recover a sharp signal p such that blur(blur(p)) matches an
observed doubly-blurred signal. Gradient descent on MSE.
"""

import numpy as np

import warp as wp

N = 256
INV_N = wp.constant(1.0 / N)
NM1 = wp.constant(N - 1)


@wp.kernel
def blur(src: wp.array(dtype=float), dst: wp.array(dtype=float)):
    i = wp.tid()
    im = wp.max(i - 1, 0)
    ip = wp.min(i + 1, NM1)
    dst[i] = 0.25 * src[im] + 0.5 * src[i] + 0.25 * src[ip]


@wp.kernel
def mse(pred: wp.array(dtype=float), obs: wp.array(dtype=float), loss: wp.array(dtype=float)):
    i = wp.tid()
    d = pred[i] - obs[i]
    wp.atomic_add(loss, 0, d * d * INV_N)


def apply_blur(src, dst):
    wp.launch(blur, dim=N, inputs=[src], outputs=[dst])
    return dst


def main():
    rng = np.random.default_rng(11)
    p_true = rng.normal(0.0, 1.0, N).astype(np.float32)

    # build the observation: two blur passes of the true signal
    tmp_a = wp.array(p_true, dtype=float)
    tmp_b = wp.zeros(N, dtype=float)
    tmp_c = wp.zeros(N, dtype=float)
    apply_blur(tmp_a, tmp_b)
    apply_blur(tmp_b, tmp_c)
    obs = wp.array(tmp_c.numpy(), dtype=float)

    p = wp.array(np.zeros(N, dtype=np.float32), dtype=float, requires_grad=True)
    stage1 = wp.zeros(N, dtype=float)
    stage2 = wp.zeros(N, dtype=float, requires_grad=True)
    # keep stage2's gradient alive after backward so the dashboard can plot it
    stage2.retain_grad = True
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    lr = 5.0
    for it in range(500):
        loss.zero_()
        tape = wp.Tape()
        with tape:
            apply_blur(p, stage1)
            apply_blur(stage1, stage2)
            wp.launch(blur, dim=N, inputs=[stage1], outputs=[stage2])
            wp.launch(mse, dim=N, inputs=[stage2, obs], outputs=[loss])

        tape.backward(loss=loss)
        g = p.grad.numpy().copy()
        tape.zero()
        p = wp.array(p.numpy() - lr * g, dtype=float, requires_grad=True)
        if it % 100 == 0 or it == 499:
            print(f"iter {it:4d}  loss={loss.numpy()[0]:.6e}  |grad|={np.linalg.norm(g):.3e}")

    err = np.abs(p.numpy() - p_true).max()
    print(f"recovery max error vs true signal: {err:.4f}")


if __name__ == "__main__":
    main()
