# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the dense stage once and print its three reported totals."""

from __future__ import annotations

from dense_stage import kernels

import warp as wp

SAMPLES = 4096
DEVICE = "cuda:0"


def main() -> None:
    """Run every kernel once and print the report."""
    if wp.get_cuda_device_count() == 0:
        raise RuntimeError("This program requires a CUDA device, but Warp sees none.")

    size = SAMPLES
    launch = {"device": DEVICE, "block_dim": kernels.BLOCK_DIM}

    lattice = wp.zeros(size, dtype=wp.float32, device=DEVICE)
    snapshot = wp.zeros(size, dtype=wp.float32, device=DEVICE)
    lowered = wp.zeros(size, dtype=wp.float32, device=DEVICE)
    raised = wp.zeros(size, dtype=wp.float32, device=DEVICE)
    residual = wp.zeros(size, dtype=wp.float32, device=DEVICE)
    blurred = wp.zeros(size, dtype=wp.float32, device=DEVICE)
    sharpened = wp.zeros(size, dtype=wp.float32, device=DEVICE)
    marks = wp.zeros(size, dtype=wp.float32, device=DEVICE)
    moments = wp.zeros(size, dtype=wp.float32, device=DEVICE)
    normalised = wp.zeros(size, dtype=wp.float32, device=DEVICE)
    report = wp.zeros(3, dtype=wp.float32, device=DEVICE)

    wp.launch(kernels.seed_lattice, dim=size, inputs=[lattice, wp.float32(0.02)], **launch)
    wp.launch(kernels.stage_snapshot, dim=size, inputs=[lattice, snapshot], **launch)
    wp.launch(kernels.relax_interior, dim=size, inputs=[lattice, wp.float32(0.4)], **launch)
    wp.launch(kernels.edge_pass_lower, dim=size, inputs=[lattice, lowered, wp.float32(0.5)], **launch)
    wp.launch(kernels.edge_pass_upper, dim=size, inputs=[lowered, raised, wp.float32(0.5)], **launch)
    wp.launch(kernels.restore_residual, dim=size, inputs=[snapshot, raised, residual], **launch)
    wp.launch(kernels.weighted_blur, dim=size, inputs=[raised, blurred, 4], **launch)
    wp.launch(kernels.sharpen, dim=size, inputs=[raised, blurred, sharpened], **launch)
    wp.launch(kernels.local_extrema, dim=size, inputs=[sharpened, marks], **launch)
    wp.launch(kernels.running_moment, dim=size, inputs=[sharpened, moments, 3], **launch)
    wp.launch(kernels.normalise_range, dim=size, inputs=[moments, normalised, wp.float32(2.0)], **launch)
    wp.launch(kernels.accumulate_report, dim=size, inputs=[normalised, residual, marks, report], **launch)

    energy, drift, density = (float(value) for value in report.numpy())
    print(f"energy:  {energy:.6f}")
    print(f"drift:   {drift:.6f}")
    print(f"density: {density:.6f}")


if __name__ == "__main__":
    main()
