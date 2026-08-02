# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable loss independent of stencil-bank analysis."""

from __future__ import annotations

import numpy as np

import warp as wp

_BLOCK_DIM = 64


@wp.kernel(module="stencil_bank.loss")
def stencil_loss_kernel(values: wp.array[wp.float32], target: wp.array[wp.float32], loss: wp.array[wp.float32]) -> None:
    """Accumulate the mean squared residual of one input element."""
    index = wp.tid()
    residual = values[index] - target[index]
    wp.atomic_add(loss, 0, residual * residual / wp.float32(values.shape[0]))


def stencil_loss(values: np.ndarray, target: np.ndarray, device: str = "cuda:0") -> tuple[float, np.ndarray]:
    """Return mean squared error and its gradient with respect to ``values``."""
    source_values = np.asarray(values)
    target_values = np.asarray(target)
    if source_values.ndim != 1 or target_values.shape != source_values.shape or source_values.size == 0:
        raise ValueError("values and target must be matching nonempty one-dimensional arrays")
    if source_values.dtype != np.float32 or target_values.dtype != np.float32:
        raise TypeError("values and target must use float32")
    source = wp.array(np.ascontiguousarray(source_values), dtype=wp.float32, device=device, requires_grad=True)
    expected = wp.array(np.ascontiguousarray(target_values), dtype=wp.float32, device=device, requires_grad=True)
    result = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            stencil_loss_kernel,
            dim=source_values.size,
            inputs=[source, expected],
            outputs=[result],
            device=device,
            block_dim=_BLOCK_DIM,
        )
    tape.backward(loss=result)
    return float(result.numpy()[0]), tape.gradients[source].numpy()
