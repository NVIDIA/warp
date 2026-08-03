# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Differentiable field-fitting pipeline.
#
# Three stages:
#   encode:  params_a + coords -> features
#   mix:     features + params_b -> field
#   readout: field vs. target -> residual, scalar loss
#
# All stages are separate kernels so the tape records one launch per stage.

import numpy as np

import warp as wp


@wp.kernel
def encode(params_a: wp.array[float], coords: wp.array[float], features: wp.array[float]):
    i = wp.tid()
    features[i] = wp.tanh(params_a[i]) * coords[i] + 0.25 * wp.sin(2.0 * params_a[i])


@wp.kernel
def mix(features: wp.array[float], params_b: wp.array[float], field: wp.array[float]):
    i = wp.tid()
    field[i] = features[i] * params_b[i] + 0.1 * wp.sin(params_b[i])


@wp.kernel
def readout(
    field: wp.array[float],
    target: wp.array[float],
    residual: wp.array[float],
    loss: wp.array[float],
):
    i = wp.tid()
    r = field[i] - target[i]
    residual[i] = r
    wp.atomic_add(loss, 0, r * r)


def allocate_buffers(n: int, device=None) -> dict:
    """Allocate all intermediate buffers used by the pipeline."""
    field = wp.zeros(n, dtype=float, device=device, requires_grad=True)
    residual = wp.zeros_like(field)
    features = wp.zeros(n, dtype=float, device=device)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    return {"features": features, "field": field, "residual": residual, "loss": loss}


def forward(params_a, params_b, coords, target, buffers):
    """Run the full pipeline; returns the scalar loss array (buffers['loss'])."""
    n = coords.shape[0]
    buffers["loss"].zero_()
    wp.launch(encode, dim=n, inputs=[params_a, coords], outputs=[buffers["features"]])
    wp.launch(mix, dim=n, inputs=[buffers["features"], params_b], outputs=[buffers["field"]])
    wp.launch(
        readout,
        dim=n,
        inputs=[buffers["field"], target],
        outputs=[buffers["residual"], buffers["loss"]],
    )
    return buffers["loss"]


def make_problem(n: int, seed: int = 7, device=None):
    """Build a synthetic fitting problem with a known ground-truth parameter set."""
    rng = np.random.default_rng(seed)

    coords_np = np.linspace(-1.0, 1.0, n).astype(np.float32)
    true_a = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    true_b = rng.uniform(0.5, 1.5, n).astype(np.float32)

    # Generate the target by evaluating the model at the true parameters.
    feat = np.tanh(true_a) * coords_np + 0.25 * np.sin(2.0 * true_a)
    target_np = feat * true_b + 0.1 * np.sin(true_b)

    coords = wp.array(coords_np, dtype=float, device=device)
    target = wp.array(target_np, dtype=float, device=device)
    params_a = wp.array(rng.uniform(-0.5, 0.5, n).astype(np.float32), dtype=float, device=device, requires_grad=True)
    params_b = wp.array(np.ones(n, dtype=np.float32), dtype=float, device=device, requires_grad=True)

    return params_a, params_b, coords, target
