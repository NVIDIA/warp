# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable heightfield fit with a live preview.

We optimize a height profile h so its smoothed version matches a target
profile. Every iteration we also present the smoothed field to the viewer;
the viewer API registered a zero-copy handle to our field buffer at startup,
and expects tone-mapped values in [-1, 1] at present time.
"""

import numpy as np

import warp as wp

N = 128
INV_N = wp.constant(1.0 / N)
NM1 = wp.constant(N - 1)


@wp.kernel
def smooth(h: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    im = wp.max(i - 1, 0)
    ip = wp.min(i + 1, NM1)
    y[i] = 0.25 * h[im] + 0.5 * h[i] + 0.25 * h[ip]


@wp.kernel
def mse(y: wp.array(dtype=float), target: wp.array(dtype=float), loss: wp.array(dtype=float)):
    i = wp.tid()
    d = y[i] - target[i]
    wp.atomic_add(loss, 0, d * d * INV_N)


@wp.kernel
def tonemap(img: wp.array(dtype=float)):
    i = wp.tid()
    v = img[i]
    img[i] = v / (1.0 + wp.abs(v))


def present(preview):
    """Tone-map the registered preview buffer and hand it to the viewer."""
    wp.launch(tonemap, dim=N, inputs=[preview])
    # viewer.blit(preview)  # (viewer omitted in this repro)


def main():
    i = np.arange(N, dtype=np.float32)
    target_np = (0.8 * np.sin(2.0 * np.pi * i / N)).astype(np.float32)

    target = wp.array(target_np, dtype=float)
    h = wp.array(np.zeros(N, dtype=np.float32), dtype=float, requires_grad=True)
    y = wp.zeros(N, dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    # zero-copy handle the viewer API holds onto (registered once at startup)
    preview = wp.array(ptr=y.ptr, dtype=float, shape=y.shape, device=y.device)

    lr = 30.0
    for it in range(300):
        loss.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(smooth, dim=N, inputs=[h], outputs=[y])
            wp.launch(mse, dim=N, inputs=[y, target], outputs=[loss])

        present(preview)

        tape.backward(loss=loss)
        g = h.grad.numpy().copy()
        h = wp.array(h.numpy() - lr * g, dtype=float, requires_grad=True)
        if it % 50 == 0 or it == 299:
            print(f"iter {it:4d}  loss={loss.numpy()[0]:.6e}  max|h|={np.abs(h.numpy()).max():.3f}")

    # the smoothed field should match the target almost exactly
    wp.launch(smooth, dim=N, inputs=[h], outputs=[y])
    err = np.abs(y.numpy() - target_np).max()
    print(f"final max |smooth(h) - target| = {err:.4f}  (target amplitude 0.8)")


if __name__ == "__main__":
    main()
