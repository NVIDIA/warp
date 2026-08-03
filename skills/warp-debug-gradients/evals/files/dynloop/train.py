# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit per-pane absorption coefficients to measured sensor readings.

The scene is a stack of filter panes with photodiodes embedded at different
depths: sensor i sits behind the first ``counts[i]`` panes, so its ray
crosses panes 0..counts[i]-1 in order. The forward model attenuates each ray
by the panes it crosses (see model.py) and scores the transmitted
intensities against the measured readings. Gradients w.r.t. the absorption
coefficients are computed with wp.Tape and used for gradient descent.
"""

import model
import numpy as np

import warp as wp

NUM_SENSORS = 16
NUM_PANES = 12
TRAIN_ITERS = 15
LEARNING_RATE = 0.4
SEED = 11


def make_scene(rng: np.random.Generator):
    """Build the per-sensor pane crossings and measured readings."""
    counts = rng.integers(4, NUM_PANES + 1, size=NUM_SENSORS)
    counts[0] = NUM_PANES  # the deepest sensor sits behind the full stack
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)

    pane_thickness = rng.uniform(0.35, 0.85, size=NUM_PANES).astype(np.float32)

    # sensor i crosses panes 0..counts[i]-1, in order
    pane = np.concatenate([np.arange(c) for c in counts]).astype(np.int32)
    thickness = pane_thickness[pane]

    # "measured" readings: reference coefficient set + sensor noise
    true_absorb = rng.uniform(0.5, 1.0, size=NUM_PANES)
    target = np.empty(NUM_SENSORS, dtype=np.float32)
    for i in range(NUM_SENSORS):
        f = 1.0 - true_absorb[pane[offsets[i] : offsets[i + 1]]] * thickness[offsets[i] : offsets[i + 1]]
        target[i] = np.prod(f)
    target *= (1.0 + 0.05 * rng.standard_normal(NUM_SENSORS)).astype(np.float32)

    absorb0 = rng.uniform(0.2, 0.5, size=NUM_PANES).astype(np.float32)
    return absorb0, pane, offsets, thickness, target


def main():
    rng = np.random.default_rng(SEED)
    absorb_np, pane_np, offsets_np, thickness_np, target_np = make_scene(rng)

    pane = wp.array(pane_np, dtype=int)
    offsets = wp.array(offsets_np, dtype=int)
    thickness = wp.array(thickness_np, dtype=float)
    target = wp.array(target_np, dtype=float)

    for it in range(TRAIN_ITERS):
        absorb = wp.array(absorb_np, dtype=float, requires_grad=True)

        tape = wp.Tape()
        with tape:
            loss = model.run_model(absorb, pane, offsets, thickness, target)
        tape.backward(loss)

        grad = absorb.grad.numpy()
        grad_norm = np.linalg.norm(grad)
        print(f"iter {it:02d}  loss {loss.numpy()[0]:10.6f}  grad_norm {grad_norm:10.4f}")

        absorb_np = absorb_np - LEARNING_RATE * grad

    print("final coefficients:", np.array2string(absorb_np, precision=4))


if __name__ == "__main__":
    main()
