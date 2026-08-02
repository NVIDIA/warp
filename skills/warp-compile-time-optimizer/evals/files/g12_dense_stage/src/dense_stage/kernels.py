# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every kernel this application runs, in one module.

Nothing here differentiates, so adjoint code generation is off. All of these
kernels are launched on every run, and the block dimension never varies.
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False, "block_dim": 256})

BLOCK_DIM = 256
LOBES = 6


@wp.func
def falloff(distance: wp.float32, scale: wp.float32) -> wp.float32:
    """Smooth inverse-square falloff."""
    ratio = distance / scale
    return wp.float32(1.0) / (wp.float32(1.0) + ratio * ratio)


@wp.kernel
def seed_lattice(lattice: wp.array[wp.float32], frequency: wp.float32) -> None:
    """Fill the lattice with a deterministic multi-lobe pattern."""
    index = wp.tid()
    position = wp.float32(index)
    total = wp.float32(0.0)
    for lobe in range(LOBES):
        weight = wp.float32(1.0) / wp.float32(lobe + 1)
        total += weight * wp.sin(position * frequency * wp.float32(lobe + 1))
    lattice[index] = total


@wp.kernel
def stage_snapshot(lattice: wp.array[wp.float32], snapshot: wp.array[wp.float32]) -> None:
    """Retain the pre-relaxation lattice.

    ``relax_interior`` writes back into ``lattice``, and ``restore_residual``
    needs the original values, so this copy is load bearing despite looking
    like a no-op.
    """
    index = wp.tid()
    snapshot[index] = lattice[index]


@wp.kernel
def relax_interior(lattice: wp.array[wp.float32], weight: wp.float32) -> None:
    """Relax interior samples in place toward their neighbour average."""
    index = wp.tid()
    last = lattice.shape[0] - 1
    if index > 0 and index < last:
        average = wp.float32(0.5) * (lattice[index - 1] + lattice[index + 1])
        lattice[index] = lattice[index] * (wp.float32(1.0) - weight) + average * weight


@wp.kernel
def edge_pass_lower(lattice: wp.array[wp.float32], result: wp.array[wp.float32], slope: wp.float32) -> None:
    """Extrapolate across the lower boundary.

    Deliberately mirrors ``edge_pass_upper``; the two differ in which end they
    anchor to and are not interchangeable.
    """
    index = wp.tid()
    if index == 0:
        result[index] = lattice[1] - slope * (lattice[2] - lattice[1])
    else:
        result[index] = lattice[index]


@wp.kernel
def edge_pass_upper(lattice: wp.array[wp.float32], result: wp.array[wp.float32], slope: wp.float32) -> None:
    """Extrapolate across the upper boundary."""
    index = wp.tid()
    last = lattice.shape[0] - 1
    if index == last:
        result[index] = lattice[last - 1] - slope * (lattice[last - 2] - lattice[last - 1])
    else:
        result[index] = lattice[index]


@wp.kernel
def restore_residual(
    snapshot: wp.array[wp.float32],
    relaxed: wp.array[wp.float32],
    residual: wp.array[wp.float32],
) -> None:
    """Difference the relaxed lattice against the retained snapshot."""
    index = wp.tid()
    residual[index] = snapshot[index] - relaxed[index]


@wp.kernel
def weighted_blur(source: wp.array[wp.float32], result: wp.array[wp.float32], radius: wp.int32) -> None:
    """Distance-weighted blur over a centred window."""
    index = wp.tid()
    total = wp.float32(0.0)
    weights = wp.float32(0.0)
    for offset in range(-radius, radius + 1):
        other = index + offset
        if other >= 0 and other < source.shape[0]:
            weight = falloff(wp.float32(offset), wp.float32(radius))
            total += source[other] * weight
            weights += weight
    result[index] = total / wp.max(weights, wp.float32(1.0e-6))


@wp.kernel
def sharpen(source: wp.array[wp.float32], blurred: wp.array[wp.float32], result: wp.array[wp.float32]) -> None:
    """Unsharp mask against the blurred lattice."""
    index = wp.tid()
    result[index] = source[index] + (source[index] - blurred[index])


@wp.kernel
def local_extrema(source: wp.array[wp.float32], marks: wp.array[wp.float32]) -> None:
    """Mark local maxima and minima."""
    index = wp.tid()
    last = source.shape[0] - 1
    lower = source[wp.max(index - 1, 0)]
    upper = source[wp.min(index + 1, last)]
    here = source[index]
    if (here > lower and here > upper) or (here < lower and here < upper):
        marks[index] = wp.float32(1.0)
    else:
        marks[index] = wp.float32(0.0)


@wp.kernel
def running_moment(source: wp.array[wp.float32], moments: wp.array[wp.float32], order: wp.int32) -> None:
    """Windowed raw moment of the requested order."""
    index = wp.tid()
    total = wp.float32(0.0)
    count = wp.int32(0)
    for offset in range(-4, 5):
        other = index + offset
        if other >= 0 and other < source.shape[0]:
            term = wp.float32(1.0)
            for _power in range(order):
                term *= source[other]
            total += term
            count += 1
    moments[index] = total / wp.float32(count)


@wp.kernel
def normalise_range(source: wp.array[wp.float32], result: wp.array[wp.float32], span: wp.float32) -> None:
    """Map the lattice into a unit range."""
    index = wp.tid()
    result[index] = wp.tanh(source[index] / wp.max(span, wp.float32(1.0e-6)))


@wp.kernel
def accumulate_report(
    normalised: wp.array[wp.float32],
    residual: wp.array[wp.float32],
    marks: wp.array[wp.float32],
    report: wp.array[wp.float32],
) -> None:
    """Fold the pass into three reported totals."""
    index = wp.tid()
    samples = wp.float32(normalised.shape[0])
    wp.atomic_add(report, 0, normalised[index] * normalised[index] / samples)
    wp.atomic_add(report, 1, wp.abs(residual[index]) / samples)
    wp.atomic_add(report, 2, marks[index] / samples)
