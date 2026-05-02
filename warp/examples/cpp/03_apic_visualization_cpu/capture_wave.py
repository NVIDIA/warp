# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Capture a wave simulation as an APIC graph for C++ execution on CPU.

This script captures a full frame of wave simulation (multiple substeps) as a
single graph using Warp's APIC mode on the CPU device. The resulting .wrp file
can be loaded and executed from C++ without requiring a Python runtime or GPU.

Usage:
    python capture_wave.py
    uv run capture_wave.py

Output:
    generated/wave_sim.wrp          - Serialized graph
    generated/wave_sim_modules/     - Compiled CPU modules (.o files)
"""

import os

import warp as wp


@wp.func
def sample_height(heights: wp.array(dtype=float), x: int, y: int, width: int, height: int):
    """Sample height field with boundary clamping."""
    x = wp.clamp(x, 0, width - 1)
    y = wp.clamp(y, 0, height - 1)
    return heights[y * width + x]


@wp.kernel
def wave_displace(
    hprevious: wp.array(dtype=float),
    hcurrent: wp.array(dtype=float),
    mouse_pos: wp.array(dtype=wp.vec2),
    width: int,
    height: int,
    radius: float,
    magnitude: float,
):
    """Apply displacement at mouse position - creates ripples where user clicks."""
    tid = wp.tid()
    x = tid % width
    y = tid // width

    center = mouse_pos[0]
    dx = float(x) - center[0]
    dy = float(y) - center[1]
    dist_sq = dx * dx + dy * dy

    if dist_sq < radius * radius:
        falloff = 1.0 - dist_sq / (radius * radius)
        h = magnitude * falloff

        hcurrent[tid] = hcurrent[tid] + h
        hprevious[tid] = hprevious[tid] + h


@wp.kernel
def wave_solve(
    hprevious: wp.array(dtype=float),
    hcurrent: wp.array(dtype=float),
    width: int,
    height: int,
    inv_cell: float,
    k_speed: float,
    k_damp: float,
    dt: float,
):
    """Integrate wave equation using finite differences."""
    tid = wp.tid()
    x = tid % width
    y = tid // width

    h = sample_height(hcurrent, x, y, width, height)
    h_xp = sample_height(hcurrent, x + 1, y, width, height)
    h_xm = sample_height(hcurrent, x - 1, y, width, height)
    h_yp = sample_height(hcurrent, x, y + 1, width, height)
    h_ym = sample_height(hcurrent, x, y - 1, width, height)

    laplacian = (h_xp + h_xm + h_yp + h_ym - 4.0 * h) * inv_cell * inv_cell

    h0 = sample_height(hprevious, x, y, width, height)

    h_new = 2.0 * h - h0 + dt * dt * (k_speed * laplacian - k_damp * (h - h0))

    hprevious[tid] = h_new


def capture_wave_graph(
    width: int = 128,
    height: int = 128,
    substeps: int = 16,
    output_path: str = "generated/wave_sim",
):
    """Capture the wave simulation as an APIC graph on CPU."""
    device = "cpu"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    grid0 = wp.zeros(width * height, dtype=float, device=device)
    grid1 = wp.zeros(width * height, dtype=float, device=device)
    mouse_pos = wp.zeros(1, dtype=wp.vec2, device=device)

    grid_size = 0.1
    dt = 1.0 / 60.0 / substeps
    radius = 5.0
    magnitude = 0.3
    k_speed = 60.0
    k_damp = 1000.0

    print("Capturing wave simulation graph (CPU)...")
    print(f"  Grid: {width}x{height}")
    print(f"  Substeps: {substeps}")
    print(f"  dt: {dt:.6f}")

    wp.capture_begin(device=device, apic=True)

    for s in range(substeps):
        if s == 0:
            wp.launch(
                wave_displace,
                dim=width * height,
                inputs=[grid0, grid1, mouse_pos, width, height, radius, magnitude],
                device=device,
            )

        wp.launch(
            wave_solve,
            dim=width * height,
            inputs=[grid0, grid1, width, height, 1.0 / grid_size, k_speed, k_damp, dt],
            device=device,
        )

        grid0, grid1 = grid1, grid0

    graph = wp.capture_end(device=device)

    wp.capture_save(
        graph,
        output_path,
        inputs={
            "heights": grid1,
            "heights_prev": grid0,
            "mouse_pos": mouse_pos,
        },
        outputs={
            "heights_out": grid1,
            "heights_prev_out": grid0,
        },
    )

    print(f"\nSaved graph to {output_path}.wrp")
    print(f"  - {substeps + 1} kernel launches captured")
    print("  - Parameters:")
    print(f"      heights:      {width * height * 4} bytes (float[{width * height}])")
    print(f"      heights_prev: {width * height * 4} bytes (float[{width * height}])")
    print("      mouse_pos:   8 bytes (vec2)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capture wave simulation as APIC graph (CPU)")
    parser.add_argument("--width", type=int, default=128, help="Grid width")
    parser.add_argument("--height", type=int, default=128, help="Grid height")
    parser.add_argument("--substeps", type=int, default=16, help="Substeps per frame")
    parser.add_argument("--output", type=str, default="generated/wave_sim", help="Output path")

    args = parser.parse_args()

    capture_wave_graph(
        width=args.width,
        height=args.height,
        substeps=args.substeps,
        output_path=args.output,
    )
