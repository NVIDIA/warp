# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradient-based calibration against a measured target.

This path differentiates through operators in :mod:`field_solver.kernels`, the
same module :mod:`field_solver.inspect_field` uses forward only.
"""

from __future__ import annotations

import numpy as np

import warp as wp

from .kernels import STENCIL_TAPS, apply_stencil, residual_against

BLOCK_DIM = 256


def residual_and_gradient(
    field: np.ndarray,
    target: np.ndarray,
    device: str = "cuda:0",
) -> tuple[float, np.ndarray]:
    """Return the residual against ``target`` and its gradient w.r.t. ``field``.

    Args:
        field: One-dimensional float32 field being calibrated.
        target: Measured field of the same shape.
        device: Warp device to run on.

    Returns:
        Tuple of scalar residual and the gradient array.
    """
    field_values = np.ascontiguousarray(field)
    target_values = np.ascontiguousarray(target)
    if field_values.ndim != 1 or target_values.shape != field_values.shape or field_values.size == 0:
        raise ValueError("field and target must be matching nonempty one-dimensional arrays")
    if field_values.dtype != np.float32 or target_values.dtype != np.float32:
        raise TypeError("field and target must use float32")

    size = field_values.size
    source = wp.array(field_values, dtype=wp.float32, device=device, requires_grad=True)
    measured = wp.array(target_values, dtype=wp.float32, device=device)
    filtered = wp.zeros(size, dtype=wp.float32, device=device, requires_grad=True)
    residual = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    taps = wp.array(
        np.full(STENCIL_TAPS, 1.0 / STENCIL_TAPS, dtype=np.float32),
        dtype=wp.float32,
        device=device,
        requires_grad=True,
    )

    launch = {"device": device, "block_dim": BLOCK_DIM}
    tape = wp.Tape()
    with tape:
        wp.launch(apply_stencil, dim=size, inputs=[source, taps], outputs=[filtered], **launch)
        wp.launch(residual_against, dim=size, inputs=[filtered, measured], outputs=[residual], **launch)
    tape.backward(loss=residual)

    return float(residual.numpy()[0]), tape.gradients[source].numpy()
