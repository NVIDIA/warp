# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibration entry point, checked against finite differences.

The analytic gradient comes from a ``wp.Tape`` through ``field_solver.kernels``.
If adjoint code stops being generated for that module the gradient silently
becomes zero, so this compares it against a numerical estimate and fails loudly.
"""

from __future__ import annotations

import sys

import numpy as np
from field_solver.calibrate import residual_and_gradient

import warp as wp

STEP = np.float32(1.0e-2)
TOLERANCE = 2.0e-2


def main() -> int:
    """Compare analytic and numerical gradients; return a process exit code."""
    if wp.get_cuda_device_count() == 0:
        raise RuntimeError("This program requires a CUDA device, but Warp sees none.")

    size = 64
    positions = np.arange(size, dtype=np.float32)
    field = np.sin(positions * np.float32(0.15)).astype(np.float32)
    target = np.cos(positions * np.float32(0.10)).astype(np.float32)

    residual, analytic = residual_and_gradient(field, target)

    probes = (3, 17, 31, 48)
    numerical = np.zeros(len(probes), dtype=np.float32)
    for slot, index in enumerate(probes):
        forward = field.copy()
        forward[index] += STEP
        backward = field.copy()
        backward[index] -= STEP
        high, _ = residual_and_gradient(forward, target)
        low, _ = residual_and_gradient(backward, target)
        numerical[slot] = (high - low) / (2.0 * STEP)

    sampled = analytic[list(probes)]
    print(f"residual: {residual:.6f}")
    print(f"analytic:  {np.array2string(sampled, precision=6)}")
    print(f"numerical: {np.array2string(numerical, precision=6)}")

    if not np.any(np.abs(sampled) > 0.0):
        print("FAIL: analytic gradient is identically zero -- adjoint code was not generated", file=sys.stderr)
        return 1
    if not np.allclose(sampled, numerical, rtol=TOLERANCE, atol=TOLERANCE):
        print("FAIL: analytic gradient does not match the numerical estimate", file=sys.stderr)
        return 1

    print("gradient check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
