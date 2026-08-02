# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run every pipeline stage once and print a compact report.

Each stage owns its Warp module and its own stable block dimension.
"""

from __future__ import annotations

import numpy as np
from stage_pipeline import geometry, reduction, spectral, statistics, transport

import warp as wp

SAMPLES = 2048
POINTS = 512
DEVICE = "cuda:0"


def _signal() -> np.ndarray:
    positions = np.arange(SAMPLES, dtype=np.float32)
    return (np.sin(positions * np.float32(0.03)) + np.float32(0.25) * np.cos(positions * np.float32(0.11))).astype(
        np.float32
    )


def run_spectral(signal: wp.array) -> float:
    """Run the spectral stage and return its summary mean."""
    bands = spectral.BANDS
    energies = wp.zeros(bands, dtype=wp.float32, device=DEVICE)
    phases = wp.zeros(bands, dtype=wp.float32, device=DEVICE)
    whitened = wp.zeros(bands, dtype=wp.float32, device=DEVICE)
    contrast = wp.zeros(bands, dtype=wp.float32, device=DEVICE)
    rolloff = wp.zeros(bands, dtype=wp.float32, device=DEVICE)
    flatness = wp.zeros(bands, dtype=wp.float32, device=DEVICE)
    slope = wp.zeros(bands, dtype=wp.float32, device=DEVICE)
    peaks = wp.zeros(bands, dtype=wp.float32, device=DEVICE)
    summary = wp.zeros(bands, dtype=wp.float32, device=DEVICE)

    launch = {"device": DEVICE, "block_dim": spectral.BLOCK_DIM}
    wp.launch(spectral.band_energy, dim=bands, inputs=[signal, energies, SAMPLES], **launch)
    wp.launch(spectral.band_phase, dim=bands, inputs=[signal, phases, SAMPLES], **launch)
    wp.launch(spectral.whiten_bands, dim=bands, inputs=[energies, whitened, wp.float32(1.0e-4)], **launch)
    wp.launch(spectral.band_contrast, dim=bands, inputs=[whitened, contrast], **launch)
    wp.launch(spectral.band_rolloff, dim=bands, inputs=[whitened, rolloff], **launch)
    wp.launch(spectral.band_flatness, dim=bands, inputs=[whitened, flatness], **launch)
    wp.launch(spectral.band_slope, dim=bands, inputs=[whitened, slope], **launch)
    wp.launch(spectral.band_peaks, dim=bands, inputs=[contrast, peaks, wp.float32(0.1)], **launch)
    wp.launch(spectral.band_summary, dim=bands, inputs=[whitened, flatness, slope, summary], **launch)
    return float(summary.numpy().mean())


def run_geometry() -> float:
    """Run the geometry stage and return the mean local density."""
    angles = np.linspace(0.0, 6.0, POINTS, dtype=np.float32)
    cloud = np.stack([np.cos(angles), np.sin(angles), angles * np.float32(0.1)], axis=1).astype(np.float32)
    points = wp.array(cloud, dtype=wp.vec3, device=DEVICE)
    centred = wp.zeros(POINTS, dtype=wp.vec3, device=DEVICE)
    normals = wp.zeros(POINTS, dtype=wp.vec3, device=DEVICE)
    radii = wp.zeros(POINTS, dtype=wp.float32, device=DEVICE)
    density = wp.zeros(POINTS, dtype=wp.float32, device=DEVICE)

    launch = {"device": DEVICE, "block_dim": geometry.BLOCK_DIM}
    wp.launch(geometry.centre_points, dim=POINTS, inputs=[points, centred, wp.vec3(0.0, 0.0, 0.5)], **launch)
    wp.launch(geometry.radial_extent, dim=POINTS, inputs=[centred, radii], **launch)
    wp.launch(geometry.local_density, dim=POINTS, inputs=[centred, density, wp.float32(0.5)], **launch)
    wp.launch(geometry.orient_normals, dim=POINTS, inputs=[centred, normals, wp.vec3(0.0, 0.0, 1.0)], **launch)
    return float(density.numpy().mean())


def run_statistics(signal: wp.array) -> float:
    """Run the statistics stage and return the outlier fraction."""
    means = wp.zeros(SAMPLES, dtype=wp.float32, device=DEVICE)
    spread = wp.zeros(SAMPLES, dtype=wp.float32, device=DEVICE)
    scores = wp.zeros(SAMPLES, dtype=wp.float32, device=DEVICE)
    flags = wp.zeros(SAMPLES, dtype=wp.float32, device=DEVICE)

    launch = {"device": DEVICE, "block_dim": statistics.BLOCK_DIM}
    wp.launch(statistics.rolling_mean, dim=SAMPLES, inputs=[signal, means, 4], **launch)
    wp.launch(statistics.rolling_spread, dim=SAMPLES, inputs=[signal, means, spread, 4], **launch)
    wp.launch(statistics.robust_score, dim=SAMPLES, inputs=[signal, means, spread, scores], **launch)
    wp.launch(statistics.flag_outliers, dim=SAMPLES, inputs=[scores, flags, wp.float32(1.0)], **launch)
    return float(flags.numpy().mean())


def run_transport(signal: wp.array) -> float:
    """Run the transport stage and return the accumulated flux."""
    velocity = wp.array(np.full(SAMPLES, 0.25, dtype=np.float32), dtype=wp.float32, device=DEVICE)
    advected = wp.zeros(SAMPLES, dtype=wp.float32, device=DEVICE)
    relaxed = wp.zeros(SAMPLES, dtype=wp.float32, device=DEVICE)
    limited = wp.zeros(SAMPLES, dtype=wp.float32, device=DEVICE)
    total = wp.zeros(1, dtype=wp.float32, device=DEVICE)

    launch = {"device": DEVICE, "block_dim": transport.BLOCK_DIM}
    wp.launch(transport.upwind_advect, dim=SAMPLES, inputs=[signal, velocity, advected, wp.float32(0.1)], **launch)
    wp.launch(transport.relax_field, dim=SAMPLES, inputs=[advected, relaxed, wp.float32(0.3)], **launch)
    wp.launch(transport.clamp_flux, dim=SAMPLES, inputs=[relaxed, limited, wp.float32(1.5)], **launch)
    wp.launch(transport.accumulate_flux, dim=SAMPLES, inputs=[limited, total], **launch)
    return float(total.numpy()[0])


def run_reduction(signal: wp.array) -> float:
    """Run the reduction stage and return the histogram entropy."""
    buckets = reduction.BUCKETS
    counts = wp.zeros(buckets, dtype=wp.float32, device=DEVICE)
    shares = wp.zeros(buckets, dtype=wp.float32, device=DEVICE)
    cumulative = wp.zeros(buckets, dtype=wp.float32, device=DEVICE)
    entropy = wp.zeros(1, dtype=wp.float32, device=DEVICE)

    launch = {"device": DEVICE, "block_dim": reduction.BLOCK_DIM}
    wp.launch(
        reduction.build_histogram,
        dim=SAMPLES,
        inputs=[signal, counts, wp.float32(-2.0), wp.float32(2.0)],
        **launch,
    )
    wp.launch(reduction.normalise_histogram, dim=buckets, inputs=[counts, shares, wp.float32(SAMPLES)], **launch)
    wp.launch(reduction.cumulative_share, dim=buckets, inputs=[shares, cumulative], **launch)
    wp.launch(reduction.histogram_entropy, dim=buckets, inputs=[shares, entropy], **launch)
    return float(entropy.numpy()[0])


def main() -> None:
    """Run every stage and print its headline number."""
    if wp.get_cuda_device_count() == 0:
        raise RuntimeError("This program requires a CUDA device, but Warp sees none.")

    signal = wp.array(_signal(), dtype=wp.float32, device=DEVICE)
    print(f"spectral:   {run_spectral(signal):.6f}")
    print(f"geometry:   {run_geometry():.6f}")
    print(f"statistics: {run_statistics(signal):.6f}")
    print(f"transport:  {run_transport(signal):.6f}")
    print(f"reduction:  {run_reduction(signal):.6f}")


if __name__ == "__main__":
    main()
