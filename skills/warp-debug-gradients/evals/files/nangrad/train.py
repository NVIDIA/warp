# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import model
import numpy as np

import warp as wp

learning_rate = 0.05
num_iterations = 30
seed = 42


def train(lr, iters, verbose=True):
    points = wp.array(model.initial_guess(), dtype=wp.vec2, requires_grad=True)
    anchors = wp.array(model.anchor_positions(), dtype=wp.vec2)
    ranges = wp.array(model.measured_ranges(seed), dtype=float)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    history = []
    for it in range(iters):
        loss.zero_()

        tape = wp.Tape()
        with tape:
            model.compute_loss(points, anchors, ranges, loss)

        tape.backward(loss)

        grad = points.grad.numpy().copy()
        loss_val = float(loss.numpy()[0])
        grad_norm = float(np.linalg.norm(grad))
        history.append((loss_val, grad_norm))

        if verbose:
            print(f"iter {it:3d}  loss {loss_val:.6e}  grad_norm {grad_norm:.6e}")

        tape.zero()
        tape.reset()

        points.assign(points.numpy() - lr * grad)

    return history


if __name__ == "__main__":
    wp.init()
    train(learning_rate, num_iterations)
