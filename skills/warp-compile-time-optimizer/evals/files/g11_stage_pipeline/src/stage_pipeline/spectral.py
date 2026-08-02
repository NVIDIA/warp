# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spectral stage: the largest of the five modules.

Nothing in this pipeline differentiates, so adjoint code generation is already
off for every stage.
"""

from __future__ import annotations

import warp as wp

# This module is always launched at BLOCK_DIM, so declare it as the module
# default too: a preload that does not name a block dimension then builds the
# variant this stage actually uses instead of a second one.
wp.set_module_options({"enable_backward": False, "block_dim": 128})

BLOCK_DIM = 128
BANDS = 12


@wp.func
def window(position: wp.float32, span: wp.float32) -> wp.float32:
    """Hann-style taper over ``span`` samples."""
    phase = wp.float32(2.0) * wp.pi * position / span
    return wp.float32(0.5) * (wp.float32(1.0) - wp.cos(phase))


@wp.kernel
def band_energy(signal: wp.array[wp.float32], energies: wp.array[wp.float32], span: wp.int32) -> None:
    """Accumulate windowed energy per frequency band."""
    band = wp.tid()
    total = wp.float32(0.0)
    frequency = wp.float32(band + 1) * wp.float32(0.01)
    for offset in range(span):
        sample = signal[offset] * window(wp.float32(offset), wp.float32(span))
        total += sample * wp.cos(frequency * wp.float32(offset))
    energies[band] = total * total / wp.float32(span)


@wp.kernel
def band_phase(signal: wp.array[wp.float32], phases: wp.array[wp.float32], span: wp.int32) -> None:
    """Estimate the dominant phase per band."""
    band = wp.tid()
    frequency = wp.float32(band + 1) * wp.float32(0.01)
    real = wp.float32(0.0)
    imaginary = wp.float32(0.0)
    for offset in range(span):
        sample = signal[offset] * window(wp.float32(offset), wp.float32(span))
        real += sample * wp.cos(frequency * wp.float32(offset))
        imaginary += sample * wp.sin(frequency * wp.float32(offset))
    phases[band] = wp.atan2(imaginary, real)


@wp.kernel
def whiten_bands(energies: wp.array[wp.float32], whitened: wp.array[wp.float32], floor: wp.float32) -> None:
    """Flatten the band energies onto a logarithmic scale."""
    band = wp.tid()
    whitened[band] = wp.log(wp.max(energies[band], floor)) - wp.log(floor)


@wp.kernel
def band_contrast(whitened: wp.array[wp.float32], contrast: wp.array[wp.float32]) -> None:
    """Difference each band against its neighbours."""
    band = wp.tid()
    last = whitened.shape[0] - 1
    lower = wp.max(band - 1, 0)
    upper = wp.min(band + 1, last)
    contrast[band] = whitened[band] - wp.float32(0.5) * (whitened[lower] + whitened[upper])


@wp.kernel
def band_rolloff(whitened: wp.array[wp.float32], rolloff: wp.array[wp.float32]) -> None:
    """Cumulative share of energy up to each band."""
    band = wp.tid()
    total = wp.float32(0.0)
    running = wp.float32(0.0)
    for other in range(whitened.shape[0]):
        total += whitened[other]
        if other <= band:
            running += whitened[other]
    rolloff[band] = running / wp.max(total, wp.float32(1.0e-6))


@wp.kernel
def band_flatness(whitened: wp.array[wp.float32], flatness: wp.array[wp.float32]) -> None:
    """Ratio of geometric to arithmetic mean over a sliding window."""
    band = wp.tid()
    last = whitened.shape[0] - 1
    logs = wp.float32(0.0)
    total = wp.float32(0.0)
    count = wp.int32(0)
    for offset in range(-2, 3):
        other = wp.clamp(band + offset, 0, last)
        value = wp.max(whitened[other], wp.float32(1.0e-6))
        logs += wp.log(value)
        total += value
        count += 1
    geometric = wp.exp(logs / wp.float32(count))
    flatness[band] = geometric / (total / wp.float32(count))


@wp.kernel
def band_slope(whitened: wp.array[wp.float32], slope: wp.array[wp.float32]) -> None:
    """Least-squares slope of the band curve in a local window."""
    band = wp.tid()
    last = whitened.shape[0] - 1
    sum_x = wp.float32(0.0)
    sum_y = wp.float32(0.0)
    sum_xy = wp.float32(0.0)
    sum_xx = wp.float32(0.0)
    count = wp.float32(0.0)
    for offset in range(-3, 4):
        other = wp.clamp(band + offset, 0, last)
        x = wp.float32(offset)
        y = whitened[other]
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_xx += x * x
        count += wp.float32(1.0)
    denominator = count * sum_xx - sum_x * sum_x
    slope[band] = (count * sum_xy - sum_x * sum_y) / wp.max(denominator, wp.float32(1.0e-6))


@wp.kernel
def band_peaks(contrast: wp.array[wp.float32], peaks: wp.array[wp.float32], threshold: wp.float32) -> None:
    """Mark bands that stand above both neighbours."""
    band = wp.tid()
    last = contrast.shape[0] - 1
    lower = wp.max(band - 1, 0)
    upper = wp.min(band + 1, last)
    value = contrast[band]
    if value > contrast[lower] and value > contrast[upper] and value > threshold:
        peaks[band] = wp.float32(1.0)
    else:
        peaks[band] = wp.float32(0.0)


@wp.kernel
def band_summary(
    whitened: wp.array[wp.float32],
    flatness: wp.array[wp.float32],
    slope: wp.array[wp.float32],
    summary: wp.array[wp.float32],
) -> None:
    """Fold the per-band descriptors into one score."""
    band = wp.tid()
    summary[band] = whitened[band] * wp.float32(0.5) + flatness[band] * wp.float32(0.3) + slope[band] * wp.float32(0.2)
