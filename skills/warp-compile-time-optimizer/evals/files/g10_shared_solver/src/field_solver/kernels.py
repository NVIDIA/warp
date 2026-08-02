# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scalar-field operators shared by the inspection and calibration paths.

Every kernel here lives in one Warp module, so any module option set on this
file applies to both consumers. ``inspect_field`` runs these forward only;
``calibrate`` differentiates through a subset of them with a ``wp.Tape``.
"""

from __future__ import annotations

import warp as wp

STENCIL_TAPS = 9


@wp.func
def taper(position: wp.float32, width: wp.float32) -> wp.float32:
    """Smooth roll-off used to keep operators well behaved near the border."""
    scaled = position / width
    return wp.float32(1.0) / (wp.float32(1.0) + scaled * scaled)


@wp.func
def blend(low: wp.float32, high: wp.float32, weight: wp.float32) -> wp.float32:
    """Convex combination of two samples."""
    return low * (wp.float32(1.0) - weight) + high * weight


@wp.kernel
def seed_field(field: wp.array[wp.float32], frequency: wp.float32) -> None:
    """Fill the field with a smooth deterministic pattern."""
    index = wp.tid()
    position = wp.float32(index)
    value = wp.sin(position * frequency) + wp.float32(0.5) * wp.cos(position * frequency * wp.float32(2.0))
    field[index] = value * taper(position, wp.float32(field.shape[0]))


@wp.kernel
def smooth_field(source: wp.array[wp.float32], destination: wp.array[wp.float32], radius: wp.int32) -> None:
    """Box-smooth the field over ``radius`` neighbours on each side."""
    index = wp.tid()
    count = wp.int32(0)
    total = wp.float32(0.0)
    for offset in range(-radius, radius + 1):
        neighbour = index + offset
        if neighbour >= 0 and neighbour < source.shape[0]:
            total += source[neighbour]
            count += 1
    destination[index] = total / wp.float32(count)


@wp.kernel
def gradient_field(source: wp.array[wp.float32], destination: wp.array[wp.float32], spacing: wp.float32) -> None:
    """Central-difference first derivative with one-sided edges."""
    index = wp.tid()
    last = source.shape[0] - 1
    if index == 0:
        destination[index] = (source[1] - source[0]) / spacing
    elif index == last:
        destination[index] = (source[last] - source[last - 1]) / spacing
    else:
        destination[index] = (source[index + 1] - source[index - 1]) / (wp.float32(2.0) * spacing)


@wp.kernel
def curvature_field(source: wp.array[wp.float32], destination: wp.array[wp.float32], spacing: wp.float32) -> None:
    """Second derivative by the standard three-point stencil."""
    index = wp.tid()
    last = source.shape[0] - 1
    if index == 0 or index == last:
        destination[index] = wp.float32(0.0)
    else:
        centre = wp.float32(2.0) * source[index]
        destination[index] = (source[index + 1] - centre + source[index - 1]) / (spacing * spacing)


@wp.kernel
def advect_field(
    source: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
    destination: wp.array[wp.float32],
    step: wp.float32,
) -> None:
    """Semi-Lagrangian advection with linear interpolation."""
    index = wp.tid()
    origin = wp.float32(index) - velocity[index] * step
    lower = wp.int32(wp.floor(origin))
    weight = origin - wp.float32(lower)
    lower = wp.clamp(lower, 0, source.shape[0] - 1)
    upper = wp.clamp(lower + 1, 0, source.shape[0] - 1)
    destination[index] = blend(source[lower], source[upper], weight)


@wp.kernel
def diffuse_field(
    source: wp.array[wp.float32],
    destination: wp.array[wp.float32],
    rate: wp.float32,
    spacing: wp.float32,
) -> None:
    """One explicit diffusion step."""
    index = wp.tid()
    last = source.shape[0] - 1
    if index == 0 or index == last:
        destination[index] = source[index]
    else:
        centre = wp.float32(2.0) * source[index]
        laplacian = (source[index + 1] - centre + source[index - 1]) / (spacing * spacing)
        destination[index] = source[index] + rate * laplacian


@wp.kernel
def apply_stencil(source: wp.array[wp.float32], taps: wp.array[wp.float32], destination: wp.array[wp.float32]) -> None:
    """Convolve the field with a symmetric tap bank."""
    index = wp.tid()
    half = taps.shape[0] // 2
    total = wp.float32(0.0)
    for tap in range(taps.shape[0]):
        neighbour = index + tap - half
        if neighbour >= 0 and neighbour < source.shape[0]:
            total += taps[tap] * source[neighbour]
    destination[index] = total


@wp.kernel
def rescale_field(field: wp.array[wp.float32], gain: wp.float32, bias: wp.float32) -> None:
    """Affine rescale applied in place."""
    index = wp.tid()
    field[index] = field[index] * gain + bias


@wp.kernel
def clamp_field(source: wp.array[wp.float32], destination: wp.array[wp.float32], limit: wp.float32) -> None:
    """Soft-clamp the field into ``[-limit, limit]``."""
    index = wp.tid()
    value = source[index]
    destination[index] = limit * wp.tanh(value / limit)


@wp.kernel
def energy_of_field(field: wp.array[wp.float32], energy: wp.array[wp.float32]) -> None:
    """Accumulate the mean squared magnitude of the field."""
    index = wp.tid()
    value = field[index]
    wp.atomic_add(energy, 0, value * value / wp.float32(field.shape[0]))


@wp.kernel
def residual_against(
    field: wp.array[wp.float32],
    target: wp.array[wp.float32],
    residual: wp.array[wp.float32],
) -> None:
    """Accumulate the mean squared residual between field and target."""
    index = wp.tid()
    difference = field[index] - target[index]
    wp.atomic_add(residual, 0, difference * difference / wp.float32(field.shape[0]))
