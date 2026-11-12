# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING

import numpy as np

import warp as wp
from warp._src.utils import array_scan

if TYPE_CHECKING:
    from warp._src.context import DeviceLike


_SF_PHI = wp.constant(wp.float64(1.0 / math.sqrt(2.0)))
"""First Super-Fibonacci spiral constant, ``1 / sqrt(2)``."""

_SF_PSI = wp.constant(wp.float64(1.0 / 1.533751168755204288118041))
"""Second Super-Fibonacci spiral constant, the reciprocal of the plastic-like root."""


@wp.func
def super_fibonacci(i: int, n: int) -> wp.quat:
    """Return the ``i``-th of ``n`` rotations spread near-uniformly over SO(3).

    The rotations follow the Super-Fibonacci spiral of Alexa, "Super-Fibonacci Spirals: Fast,
    Low-Discrepancy Sampling of SO(3)", CVPR 2022. Together, the ``n`` rotations form a
    low-discrepancy sample of SO(3), so ``n`` rotations evaluated in parallel, one per thread,
    cover the rotations evenly without any random state. The sequence is deterministic: the
    same ``i`` and ``n`` always give the same rotation.

    This function can be called from kernels and other Warp functions.

    Args:
        i: Sample index, in ``[0, n)``.
        n: Total number of samples. The spread is near-uniform over the ``n`` samples taken
            together, so changing ``n`` changes every sample.

    Returns:
        The rotation as a unit quaternion.
    """
    # Both spiral angles grow linearly with ``i``, and only their fractional turn
    # matters. They are therefore wrapped into ``[0, 1)`` in double precision
    # before reaching sin/cos: in float32 the angle for ``i`` in the tens of
    # thousands is large enough that argument reduction consumes the fractional
    # bits the spiral depends on, and the samples stop being low-discrepancy.
    s = wp.float64(i) + wp.float64(0.5)
    a = s * _SF_PHI
    b = s * _SF_PSI
    alpha = wp.float32(a - wp.floor(a)) * 2.0 * wp.pi
    beta = wp.float32(b - wp.floor(b)) * 2.0 * wp.pi

    t = wp.float32(s / wp.float64(n))
    r = wp.sqrt(t)
    R = wp.sqrt(1.0 - t)
    return wp.quat(r * wp.sin(alpha), r * wp.cos(alpha), R * wp.sin(beta), R * wp.cos(beta))


@wp.func
def radical_inverse(base: int, i: int) -> wp.float32:
    # Van der Corput radical inverse of ``i`` in ``base``: reflect its base-``base`` digits
    # about the radix point into ``[0, 1)``. The building block of the Halton sequence.
    digit = 1.0 / wp.float32(base)
    scale = digit
    result = wp.float32(0.0)
    rem = i
    while rem > 0:
        result += wp.float32(rem % base) * scale
        rem = rem // base
        scale *= digit
    return result


@wp.func
def small_rotation(i: int, radius: wp.float32) -> wp.quat:
    # The ``i``-th deterministic rotation probing a neighborhood of angular radius ``radius``
    # around the identity. The rotation vector is a 3D Halton point (bases 2, 3, 5) mapped
    # uniformly into the ball: being low-discrepancy it covers the ball evenly instead of
    # clumping like an axis/angle split, and the radial map ``r = radius * u^(1/3)`` samples
    # the whole ball rather than concentrating near the center, where the larger corrective
    # steps of the local search would otherwise be missed. The Halton sequence is progressive, so
    # the number of samples need not be known.
    idx = i + 1  # Halton point 0 is the origin (a zero rotation); skip it.
    radial = wp.pow(radical_inverse(2, idx), 1.0 / 3.0)
    cos_theta = 1.0 - 2.0 * radical_inverse(3, idx)
    sin_theta = wp.sqrt(wp.max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * wp.pi * radical_inverse(5, idx)
    axis = wp.vec3(sin_theta * wp.cos(phi), sin_theta * wp.sin(phi), cos_theta)

    angle = radius * radial
    half = 0.5 * angle
    s = wp.sin(half)
    return wp.quat(axis[0] * s, axis[1] * s, axis[2] * s, wp.cos(half))


class OBBMeasureType(enum.IntEnum):
    """Objective minimized when searching for an oriented bounding box.

    ``IntEnum`` members are integers, so a value can be passed straight into a
    kernel launch (Warp kernels cannot take a ``str`` parameter).
    """

    VOLUME = 0
    """Minimize the volume of the bounding box."""
    SURFACE_AREA = 1
    """Minimize the surface area of the bounding box."""
    MEAN_WIDTH = 2
    """Minimize the mean width of the bounding box, which is half the sum of its side lengths."""


_OBB_POINT_CHUNKS = 256
"""Maximum threads each candidate OBB orientation splits its point loop across."""

_OBB_MAX_LAUNCH_THREADS = 1 << 22
"""Soft cap on the threads of one bounds launch; the point chunks shrink to fit it when many shapes are batched."""

_OBB_MAX_SCRATCH_ROWS = 1 << 25
"""Largest ``num_shapes * num_candidates`` accepted, which bounds the per-orientation scratch memory."""

_OBB_REFINE_RADIUS0 = 0.35
"""Initial angular radius (radians) of the local search around the best orientation."""

_OBB_REFINE_SHRINK = 0.72
"""Factor by which the refine radius shrinks each round, focusing the local search."""

_OBB_REFINE_STARTS = 4
"""Default number of distinct basins refined in parallel."""

_OBB_REFINE_ITERATIONS = 12
"""Default number of coarse-to-fine refinement rounds."""

_OBB_REFINE_SAMPLES = 128
"""Default number of orientations probed per refinement round around each start."""

_OBB_SPIN_SAMPLES = 64
"""Default number of spin angles probed about each box axis in each spin round."""

_OBB_SPIN_HALF_RANGES = (math.pi / 4, 0.02, 0.002)
"""Half-width (radians) of the spin angles probed in each spin round, coarse to fine."""

_OBB_START_SEPARATION = 0.3
"""Minimum angular separation (radians), modulo the symmetries of a box, between two refinement starts."""

_INT32_MAX = 2**31 - 1


@wp.func
def search_stride(count: int, max_search_points: int) -> int:
    # Stride of the deterministic subsample that scores orientations: every ``stride``-th point,
    # so that at most ``max_search_points`` of the ``count`` points are visited. Written to avoid
    # overflow when ``max_search_points`` is the int32 maximum, which means "use every point".
    stride = int(1)
    if count > max_search_points:
        stride = (count - 1) // max_search_points + 1
    return stride


@wp.kernel(enable_backward=False)
def oriented_bounding_box_samples_kernel(num_samples: int, rotations: wp.array2d[wp.quat]):
    # Fill the spiral portion of every shape's candidate list. Any extra candidates are
    # written into the slots past ``num_samples`` by the kernels below.
    shape, i = wp.tid()
    rotations[shape, i] = super_fibonacci(i, num_samples)


@wp.kernel(enable_backward=False)
def oriented_bounding_box_identity_kernel(slot: int, rotations: wp.array2d[wp.quat]):
    # The spiral never contains the identity exactly, so the axis-aligned box is
    # not otherwise among the candidates.
    shape = wp.tid()
    rotations[shape, slot] = wp.quat_identity()


@wp.kernel(enable_backward=False)
def oriented_bounding_box_single_shape_offsets_kernel(num_points: int, offsets: wp.array[wp.int32]):
    # Offsets for a single shape that spans the whole array, written on device so that the
    # one-shape entry point stays free of host-to-device copies and remains capturable.
    offsets[0] = 0
    offsets[1] = num_points


@wp.kernel(enable_backward=False)
def oriented_bounding_box_search_count_kernel(
    offsets: wp.array[wp.int32],
    max_search_points: int,
    counts: wp.array[wp.int32],
):
    # How many points of each shape the strided subsample keeps.
    shape = wp.tid()
    count = offsets[shape + 1] - offsets[shape]
    stride = search_stride(count, max_search_points)
    counts[shape] = (count + stride - 1) // stride


@wp.kernel(enable_backward=False)
def oriented_bounding_box_subsample_kernel(
    points: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    max_search_points: int,
    search_offsets: wp.array[wp.int32],
    num_chunks: int,
    out_points: wp.array[wp.vec3],
):
    # Gather every ``stride``-th point of each shape into a compact array. The orientation that
    # minimizes the box is a property of the cloud's shape, which a strided subsample captures;
    # scoring the candidates on it instead of every point makes the search cost independent of
    # the (possibly millions of) input points, and the compact copy keeps the many reads of the
    # scoring passes contiguous. The winner's bounds are later recomputed exactly over the full
    # set, so the returned box is unchanged.
    shape, chunk = wp.tid()
    begin = offsets[shape]
    count = offsets[shape + 1] - begin
    stride = search_stride(count, max_search_points)
    num_search = (count + stride - 1) // stride
    out_begin = search_offsets[shape]
    for k in range(chunk, num_search, num_chunks):
        out_points[out_begin + k] = points[begin + k * stride]


@wp.kernel(enable_backward=False)
def point_sum_kernel(
    points: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    max_search_points: int,
    num_chunks: int,
    out_sum: wp.array[wp.vec3],
):
    shape, chunk = wp.tid()
    begin = offsets[shape]
    count = offsets[shape + 1] - begin
    stride = search_stride(count, max_search_points)
    num_search = (count + stride - 1) // stride

    total = wp.vec3(0.0)
    for k in range(chunk, num_search, num_chunks):
        total += points[begin + k * stride]
    wp.atomic_add(out_sum, shape, total)


@wp.kernel(enable_backward=False)
def point_covariance_kernel(
    points: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    max_search_points: int,
    num_chunks: int,
    point_sum: wp.array[wp.vec3],
    out_covariance: wp.array[wp.mat33],
):
    # Scatter matrix about the centroid. Left unnormalized: scaling by 1/n does
    # not change the eigenvectors, which is all the caller wants.
    shape, chunk = wp.tid()
    begin = offsets[shape]
    count = offsets[shape + 1] - begin
    stride = search_stride(count, max_search_points)
    num_search = (count + stride - 1) // stride
    centroid = point_sum[shape] / float(num_search)

    total = wp.mat33(0.0)
    for k in range(chunk, num_search, num_chunks):
        d = points[begin + k * stride] - centroid
        total += wp.outer(d, d)
    wp.atomic_add(out_covariance, shape, total)


@wp.kernel(enable_backward=False)
def oriented_bounding_box_pca_kernel(
    covariance: wp.array[wp.mat33],
    slot: int,
    rotations: wp.array2d[wp.quat],
):
    # The principal axes of the point set are the eigenvectors of its covariance
    # matrix, which is a good starting guess for elongated shapes: the spiral
    # resolves orientation only to its sample spacing, and a few degrees of error
    # costs a lot of volume when one axis is much longer than the others.
    # Eigenvalues are unused: the box is the same whichever order the axes come in.
    shape = wp.tid()
    Q, _eigenvalues = wp.eig3(covariance[shape])

    # ``eig3`` returns orthonormal columns but does not promise a right-handed
    # frame, and ``quat_from_matrix`` is only defined for a pure rotation.
    if wp.determinant(Q) < 0.0:
        Q = wp.mat33(
            -Q[0, 0], Q[0, 1], Q[0, 2],
            -Q[1, 0], Q[1, 1], Q[1, 2],
            -Q[2, 0], Q[2, 1], Q[2, 2],
        )  # fmt: skip

    # Q's columns are the principal axes in world space, so Q maps box-local to
    # world; the search below wants the world-to-box-local direction.
    rotations[shape, slot] = wp.quat_inverse(wp.quat_from_matrix(Q))


@wp.kernel(enable_backward=False)
def oriented_bounding_box_bounds_kernel(
    points: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    max_search_points: int,
    rotations: wp.array2d[wp.quat],
    num_chunks: int,
    min_bounds: wp.array2d[wp.vec3],
    max_bounds: wp.array2d[wp.vec3],
):
    # Parallel over (shape, orientation, point chunk). Parallelizing over orientations
    # alone caps the launch at one thread per candidate no matter how many points
    # there are, which leaves most of the GPU idle; splitting the point loop as
    # well makes the available parallelism scale with the point count.
    shape, candidate, chunk = wp.tid()

    begin = offsets[shape]
    count = offsets[shape + 1] - begin

    # Score over a strided subsample of this shape's points, in place: no copy is made, and
    # each shape picks its own stride, so shapes of different sizes can share one launch
    # without the host knowing their sizes.
    stride = search_stride(count, max_search_points)
    num_search = (count + stride - 1) // stride

    # Chunks past the end of a small shape have nothing to do.
    if chunk >= num_search:
        return

    rot = rotations[shape, candidate]

    # Axis-aligned bounds of this thread's slice of points in the rotated frame.
    # A grid-stride slice keeps neighboring threads on neighboring points, so the
    # reads coalesce.
    lo = wp.vec3(wp.inf, wp.inf, wp.inf)
    hi = wp.vec3(-wp.inf, -wp.inf, -wp.inf)

    for k in range(chunk, num_search, num_chunks):
        rotated = wp.quat_rotate(rot, points[begin + k * stride])
        lo = wp.min(lo, rotated)
        hi = wp.max(hi, rotated)

    # Combine the per-chunk bounds. Unlike a sum, min/max are exact in floating
    # point, so this reduction is deterministic regardless of the order in which
    # the atomics land.
    wp.atomic_min(min_bounds, shape, candidate, lo)
    wp.atomic_max(max_bounds, shape, candidate, hi)


@wp.kernel(enable_backward=False)
def oriented_bounding_box_measure_kernel(
    rotations: wp.array2d[wp.quat],
    min_bounds: wp.array2d[wp.vec3],
    max_bounds: wp.array2d[wp.vec3],
    measure_type: int,
    measures: wp.array2d[wp.float32],
    transforms: wp.array2d[wp.transform],
    extents: wp.array2d[wp.vec3],
):
    # One thread per (shape, candidate orientation), scoring the box found above.
    shape, i = wp.tid()

    rot = rotations[shape, i]
    lo = min_bounds[shape, i]
    hi = max_bounds[shape, i]

    # Full side lengths of the box and the measure being minimized. Default to the
    # volume; the other branches override it (keeps `measure` defined on every path
    # for codegen).
    dims = hi - lo
    measure = dims[0] * dims[1] * dims[2]
    if measure_type == wp.static(int(OBBMeasureType.SURFACE_AREA)):
        measure = 2.0 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[0] * dims[2])
    elif measure_type == wp.static(int(OBBMeasureType.MEAN_WIDTH)):
        measure = 0.5 * (dims[0] + dims[1] + dims[2])

    # Box center in the rotated frame, mapped back into world space. The stored
    # transform takes the box's local axis-aligned frame to world coordinates.
    center = (hi + lo) * 0.5
    world_center = wp.quat_rotate_inv(rot, center)

    measures[shape, i] = measure
    extents[shape, i] = dims
    transforms[shape, i] = wp.transform(world_center, wp.quat_inverse(rot))


@wp.kernel(enable_backward=False)
def oriented_bounding_box_refine_generate_kernel(
    champion: wp.array2d[wp.quat],
    radius: float,
    batch: int,
    rot_batch: wp.array2d[wp.quat],
):
    # Each thread proposes one orientation in a shrinking neighborhood of one of its shape's
    # current champions (``batch`` consecutive columns per start).
    shape, i = wp.tid()
    start = i // batch
    rot_batch[shape, i] = wp.normalize(small_rotation(i % batch, radius) * champion[shape, start])


@wp.kernel(enable_backward=False)
def oriented_bounding_box_champion_kernel(
    rotations: wp.array2d[wp.quat],
    measures: wp.array2d[wp.float32],
    batch: int,
    champion: wp.array2d[wp.quat],
    champion_measure: wp.array2d[wp.float32],
):
    # One thread per (shape, start): argmin over that start's scored batch, committing its best
    # if it improves on the start's champion.
    shape, start = wp.tid()
    base = start * batch

    best = int(0)
    best_measure = measures[shape, base]
    for i in range(1, batch):
        if measures[shape, base + i] < best_measure:
            best_measure = measures[shape, base + i]
            best = i

    if best_measure < champion_measure[shape, start]:
        champion_measure[shape, start] = best_measure
        champion[shape, start] = rotations[shape, base + best]


@wp.func
def box_symmetry(index: int) -> wp.quat:
    # The ``index``-th of the 24 rotations that map an axis-aligned box onto itself, up to sign:
    # the identity and three half turns about the axes, six quarter turns about the axes, six half
    # turns about edge directions and eight third turns about the body diagonals.
    root_half = wp.sqrt(0.5)
    q = wp.vec4(0.0, 0.0, 0.0, 0.0)
    if index == 0:
        q[3] = 1.0
    elif index < 4:
        q[index - 1] = 1.0
    elif index < 10:
        q[(index - 4) // 2] = root_half if (index & 1) == 0 else -root_half
        q[3] = root_half
    elif index < 16:
        pair = (index - 10) // 2
        q[pair // 2] = root_half
        q[wp.min(pair + 1, 2)] = root_half if (index & 1) == 0 else -root_half
    else:
        bits = index - 16
        q = wp.vec4(
            0.5 if (bits & 1) == 0 else -0.5,
            0.5 if (bits & 2) == 0 else -0.5,
            0.5 if (bits & 4) == 0 else -0.5,
            0.5,
        )
    return wp.quat(q[0], q[1], q[2], q[3])


@wp.kernel(enable_backward=False)
def oriented_bounding_box_block_kernel(
    rotations: wp.array2d[wp.quat],
    picked: wp.array2d[wp.quat],
    pick_index: int,
    min_cos_half_angle: float,
    work: wp.array2d[wp.float32],
):
    # Exclude every candidate within the separation angle of the latest pick, modulo the box
    # symmetries, so that the next pick comes from a different basin.
    shape, i = wp.tid()
    q = rotations[shape, i]
    p = picked[shape, pick_index]
    for g in range(24):
        if wp.abs(wp.dot(q, box_symmetry(g) * p)) > min_cos_half_angle:
            work[shape, i] = wp.float32(math.inf)


@wp.kernel(enable_backward=False)
def oriented_bounding_box_select_kernel(
    rotations: wp.array2d[wp.quat],
    measures: wp.array2d[wp.float32],
    work: wp.array2d[wp.float32],
    pick_index: int,
    champion: wp.array2d[wp.quat],
    champion_measure: wp.array2d[wp.float32],
):
    shape = wp.tid()
    best = int(0)
    best_work = work[shape, 0]
    for i in range(1, work.shape[1]):
        if work[shape, i] < best_work:
            best_work = work[shape, i]
            best = i
    # The first pick always commits the argmin, even if every measure is non-finite, so the champion is
    # never left uninitialized.
    if best_work < wp.float32(math.inf) or pick_index == 0:
        champion[shape, pick_index] = rotations[shape, best]
        champion_measure[shape, pick_index] = measures[shape, best]
        work[shape, best] = wp.float32(math.inf)
    else:  # fewer distinct basins than starts: repeat the best
        champion[shape, pick_index] = champion[shape, 0]
        champion_measure[shape, pick_index] = champion_measure[shape, 0]


@wp.kernel(enable_backward=False)
def oriented_bounding_box_final_kernel(
    champion: wp.array2d[wp.quat],
    champion_measure: wp.array2d[wp.float32],
    out: wp.array2d[wp.quat],
):
    shape = wp.tid()
    best = int(0)
    for i in range(1, champion.shape[1]):
        if champion_measure[shape, i] < champion_measure[shape, best]:
            best = i
    out[shape, 0] = champion[shape, best]


@wp.kernel(enable_backward=False)
def oriented_bounding_box_spin_generate_kernel(
    champion: wp.array2d[wp.quat],
    half_range: float,
    samples_per_axis: int,
    rot_batch: wp.array2d[wp.quat],
):
    # One thread per (shape, start, axis, angle): spin the start's champion about one of its own box axes.
    shape, i = wp.tid()
    per_start = 3 * samples_per_axis
    start = i // per_start
    r = i % per_start
    axis_index = r // samples_per_axis
    j = r % samples_per_axis
    angle = half_range * (2.0 * (float(j) + 0.5) / float(samples_per_axis) - 1.0)
    axis = wp.vec3(0.0, 0.0, 0.0)
    axis[axis_index] = 1.0
    rot_batch[shape, i] = wp.normalize(wp.quat_from_axis_angle(axis, angle) * champion[shape, start])


def oriented_bounding_boxes(
    points: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    measure_type: OBBMeasureType = OBBMeasureType.VOLUME,
    num_samples: int = 4096,
    *,
    include_axis_aligned: bool = True,
    include_pca: bool = True,
    max_search_points: int | None = None,
    num_refine_starts: int = _OBB_REFINE_STARTS,
    num_refine_iterations: int = _OBB_REFINE_ITERATIONS,
    num_refine_samples: int = _OBB_REFINE_SAMPLES,
    num_spin_samples: int = _OBB_SPIN_SAMPLES,
    max_points_per_shape: int | None = None,
    validate: bool = True,
    device: DeviceLike | None = None,
) -> tuple[wp.array, wp.array, wp.array]:
    """Approximate an oriented bounding box (OBB) of each of several point sets in one pass.

    Equivalent to calling :func:`oriented_bounding_box` on every point set, but all point sets
    share each kernel launch, so the cost of many small point sets is no longer dominated by
    launch overhead. The point sets are stored back to back in ``points``; ``offsets`` marks
    where each one starts and ends. See :func:`oriented_bounding_box` for the search that is
    run on each point set and for the meaning of the shared search parameters.

    Args:
        points: Array of point positions of type :class:`warp.vec3`, holding every point set one
            after another.
        offsets: Array of ``num_shapes + 1`` ``int32`` offsets into ``points``. Point set ``i``
            is ``points[offsets[i]:offsets[i + 1]]``, so the offsets must be non-decreasing, and
            every point set must contain at least one point.
        measure_type: Quantity to minimize, as an :class:`OBBMeasureType` member.
        num_samples: Number of candidate orientations to sample from the spiral for each point set.
        include_axis_aligned: Whether to also evaluate the identity rotation.
        include_pca: Whether to also evaluate the principal axes of each point set.
        max_search_points: Upper bound on how many points of each point set are used to *score*
            candidate orientations, or ``None`` (the default) to score over every point. A cap bounds
            the cost of fitting very large point sets but can change which orientation is chosen. Each
            box is re-measured over all of its points either way.
        num_refine_starts: Number of distinct candidate orientations that are refined, each from a
            different basin of the search space. Refining several avoids committing to a sampled
            orientation that is not near the best one.
        num_refine_iterations: Number of coarse-to-fine refinement rounds after the initial search.
        num_refine_samples: Number of orientations probed per refinement round around each start.
        num_spin_samples: Number of angles probed about each of the box's three axes in each of three
            final spin rounds (coarse to fine) around each start. Spinning about an axis keeps a box face
            that is flush with a face of the point set's convex hull flush, so this settles the box onto
            the best such position. Pass ``0`` to skip the spin rounds.
        max_points_per_shape: Optional upper bound on the size of any one point set, used only
            to size the launches. Supplying it avoids over-provisioning threads when every
            point set is small. Results do not depend on it.
        validate: Whether to read ``offsets`` back to the host to check it. This synchronizes
            the device, so pass ``False`` when the offsets are known to be valid, which also
            makes the call capturable in a CUDA graph. Validation is skipped during graph
            capture regardless. Without validation, a point set that is empty yields a box with
            non-finite extents instead of an error.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        A tuple ``(transform, extents, measure)`` of length-``num_shapes`` device arrays, where
        entry ``i`` describes point set ``i`` exactly as for :func:`oriented_bounding_box`.

    Note:
        The cost of the search is proportional to the total number of points. Replacing each point set
        with its convex hull vertices, or with any other subset that is known to contain every hull
        vertex, leaves every box unchanged and lowers the cost in proportion to the number
        of points removed; see :func:`oriented_bounding_box`.

        Scratch memory grows with
        ``num_shapes * max(num_samples, num_refine_starts * num_refine_samples, 3 * num_refine_starts * num_spin_samples)``.
        For a very large number of shapes, call this function on groups of them.

    Raises:
        ValueError: If a search parameter is out of range as for :func:`oriented_bounding_box`, if
            ``offsets`` is not on the device of ``points``, if ``offsets`` is not ``num_shapes + 1``
            non-decreasing offsets within ``points`` with no empty point set, if
            ``max_points_per_shape`` is not positive, or if the batch is too large for the scratch memory.
        TypeError: If ``offsets`` does not hold ``int32`` values.
    """
    _validate_search_parameters(
        num_samples, num_refine_starts, num_refine_iterations, num_refine_samples, num_spin_samples, max_search_points
    )
    if max_points_per_shape is not None and max_points_per_shape < 1:
        raise ValueError(f"`max_points_per_shape` must be a positive integer or None, but got {max_points_per_shape}.")
    if offsets.dtype != wp.int32:
        raise TypeError(f"`offsets` must be an array of `int32`, but got {offsets.dtype}.")
    if offsets.ndim != 1 or offsets.shape[0] < 2:
        raise ValueError("`offsets` must be a 1D array of at least two offsets, one more than the number of shapes.")

    num_points = points.shape[0]
    device = wp.get_device(device) if device is not None else points.device
    if device != points.device:
        raise ValueError(f"`device` ({device}) must match the device of `points` ({points.device}).")
    if offsets.device != device:
        raise ValueError(f"`offsets` is on device {offsets.device}, but must be on the device of `points` ({device}).")

    max_shape_points = num_points
    if validate and not device.is_capturing:
        host_offsets = offsets.numpy()
        sizes = np.diff(host_offsets.astype(np.int64))
        if host_offsets[0] < 0 or host_offsets[-1] > num_points or np.any(sizes < 0):
            raise ValueError(
                "`offsets` must be non-decreasing and lie within `points`, "
                f"but got offsets from {host_offsets[0]} to {host_offsets[-1]} for {num_points} points."
            )
        if np.any(sizes == 0):
            raise ValueError(
                f"Every point set must contain at least one point, but point set {int(np.argmax(sizes == 0))} is empty."
            )
        # The exact largest point set is known now, which sizes the launches tightly.
        max_shape_points = int(sizes.max())

    return _fit_oriented_bounding_boxes(
        points,
        offsets,
        offsets.shape[0] - 1,
        max_points_per_shape if max_points_per_shape is not None else max_shape_points,
        measure_type,
        num_samples,
        include_axis_aligned,
        include_pca,
        max_search_points,
        num_refine_starts,
        num_refine_iterations,
        num_refine_samples,
        num_spin_samples,
        device,
    )


def oriented_bounding_box(
    points: wp.array[wp.vec3],
    measure_type: OBBMeasureType = OBBMeasureType.VOLUME,
    num_samples: int = 4096,
    *,
    include_axis_aligned: bool = True,
    include_pca: bool = True,
    max_search_points: int | None = None,
    num_refine_starts: int = _OBB_REFINE_STARTS,
    num_refine_iterations: int = _OBB_REFINE_ITERATIONS,
    num_refine_samples: int = _OBB_REFINE_SAMPLES,
    num_spin_samples: int = _OBB_SPIN_SAMPLES,
    device: DeviceLike | None = None,
) -> tuple[wp.array, wp.array, wp.array]:
    """Approximate an oriented bounding box (OBB) of a point set by sampling orientations.

    Candidate orientations are drawn from a Super-Fibonacci spiral, which spreads
    ``num_samples`` rotations near-uniformly over SO(3) (Alexa, "Super-Fibonacci
    Spirals: Fast, Low-Discrepancy Sampling of SO(3)", CVPR 2022). For each
    orientation the axis-aligned bounding box of the rotated points is evaluated, and
    the orientation whose box minimizes ``measure_type`` is returned. The result is an
    approximation whose quality improves with ``num_samples``; it is not guaranteed to
    be the globally optimal OBB.

    Two further candidates are appended by default. ``include_axis_aligned`` adds the
    identity rotation, which the spiral never contains exactly and which guarantees the
    result is no worse than the axis-aligned bounding box when every point is scored (with
    ``max_search_points`` the candidates are compared on the subsample only, so a box that wins
    there can be larger than the axis-aligned box over all points). ``include_pca`` adds the
    principal axes of the point set, obtained from the eigenvectors of its covariance
    matrix. The spiral resolves orientation only to roughly its sample spacing, and on
    a strongly elongated shape a few degrees of error costs a large amount of volume,
    so the principal axes are often a much better answer there than any sampled
    rotation. For a near-isotropic point set the covariance eigenvectors are close to
    degenerate and that candidate carries little information, leaving accuracy governed
    by ``num_samples`` alone. Both extras are cheap relative to ``num_samples``
    candidates.

    After the initial search the best ``num_refine_starts`` orientations from distinct basins
    are each refined by ``num_refine_iterations`` rounds of coarse-to-fine local search: each
    round scores a batch of orientations in a shrinking angular neighborhood of the current
    best of that start and keeps any improvement. Three spin rounds then rotate each start about
    each of its box axes, which converges onto a face-flush box that the isotropic rounds reach only
    slowly. The best refined orientation is returned. Because the spiral
    resolves orientation only to its sample spacing, this reaches the quality of a much
    larger ``num_samples`` at a fraction of the cost. The orientation search scores every point by
    default. For a very large point set, ``max_search_points`` caps the search at a strided
    subsample, which bounds its cost but can change which orientation is chosen; the returned box
    is always re-measured over every point, so it stays exact and encloses all of them.

    The cost of the search is proportional to the number of points. A bounding box is determined by
    the convex hull of the points, so replacing a very large input with its convex hull vertices, or
    with any other subset that is known to contain every hull vertex, leaves the box for every orientation unchanged and
    lowers the cost in proportion to the number of points removed. Unlike ``max_search_points``, this
    does not change which orientation is searched. The PCA candidate is the one exception, since it
    comes from the covariance of whichever points are given.

    To fit many point sets at once, see :func:`oriented_bounding_boxes`.

    Args:
        points: Array of point positions of type :class:`warp.vec3`.
        measure_type: Quantity to minimize, as an :class:`OBBMeasureType` member:
            :attr:`OBBMeasureType.VOLUME` minimizes the box volume;
            :attr:`OBBMeasureType.SURFACE_AREA` minimizes the box surface area;
            :attr:`OBBMeasureType.MEAN_WIDTH` minimizes the box mean width, half the sum of its side
            lengths.
        num_samples: Number of candidate orientations to sample from the spiral. Larger
            values give a tighter box at higher cost.
        include_axis_aligned: Whether to also evaluate the identity rotation.
        include_pca: Whether to also evaluate the principal axes of ``points``.
        max_search_points: Upper bound on how many points are used to *score* candidate
            orientations, or ``None`` (the default) to score over every point. When ``points`` is
            larger than a given cap, candidates are scored over a deterministic strided subsample
            of about that size, which bounds the cost of the search but can change which
            orientation is chosen. The winning orientation's box is recomputed over the full point
            set, so the returned box is exact and encloses every point either way. For very large
            inputs, prefer filtering the points as described above, which does not change the search.
        num_refine_starts: Number of distinct candidate orientations that are refined, each from a
            different basin of the search space. Refining several avoids committing to a sampled
            orientation that is not near the best one.
        num_refine_iterations: Number of coarse-to-fine refinement rounds after the initial search.
            Pass ``0`` together with ``num_spin_samples=0`` to return the best sampled orientation directly.
        num_refine_samples: Number of orientations probed per refinement round around each start.
        num_spin_samples: Number of angles probed about each of the box's three axes in each of three
            final spin rounds (coarse to fine) around each start. Spinning about an axis keeps a box face
            that is flush with a face of the point set's convex hull flush, so this settles the box onto
            the best such position. Pass ``0`` to skip the spin rounds.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        A tuple ``(transform, extents, measure)`` of length-1 device arrays, left on
        the device so the whole search is capturable in a CUDA graph. ``transform`` is
        a :class:`warp.array` of :class:`warp.transform` mapping the box's local frame
        (axis-aligned and centered at the origin) into world space, ``extents`` is a
        :class:`warp.array` of :class:`warp.vec3` of the box's full side lengths, and
        ``measure`` is a :class:`warp.array` of ``float32`` holding the achieved value
        of ``measure_type``. Read an element with ``transform.numpy()[0]`` (which
        synchronizes) when a host-side value is needed.

    Note:
        The search, refinement, and winner selection all run on the device, so this
        function does not synchronize and can be captured in a CUDA graph. Refinement is
        deterministic (its neighborhood is generated without host randomness).

        With ``include_pca`` enabled the covariance matrix is accumulated with
        floating-point atomics, so the principal axes -- and therefore the result,
        when they win -- can vary in the last bits between runs. Pass
        ``include_pca=False`` for a bitwise reproducible result.

    Raises:
        ValueError: If ``num_samples``, ``num_refine_starts`` or ``num_refine_samples`` is not a positive
            integer, if ``num_spin_samples`` or ``num_refine_iterations`` is negative, if
            ``max_search_points`` is not positive, if ``points`` is empty, or if the search needs too much
            scratch memory.
    """
    _validate_search_parameters(
        num_samples, num_refine_starts, num_refine_iterations, num_refine_samples, num_spin_samples, max_search_points
    )

    num_points = points.shape[0]
    if num_points == 0:
        raise ValueError("`points` must contain at least one point, but got an empty array.")

    device = wp.get_device(device) if device is not None else points.device
    if device != points.device:
        raise ValueError(f"`device` ({device}) must match the device of `points` ({points.device}).")

    offsets = wp.empty(2, dtype=wp.int32, device=device)
    wp.launch(
        oriented_bounding_box_single_shape_offsets_kernel, dim=1, inputs=[num_points], outputs=[offsets], device=device
    )

    return _fit_oriented_bounding_boxes(
        points,
        offsets,
        1,
        num_points,
        measure_type,
        num_samples,
        include_axis_aligned,
        include_pca,
        max_search_points,
        num_refine_starts,
        num_refine_iterations,
        num_refine_samples,
        num_spin_samples,
        device,
    )


def _validate_search_parameters(
    num_samples, num_refine_starts, num_refine_iterations, num_refine_samples, num_spin_samples, max_search_points
):
    for name, value in (
        ("num_samples", num_samples),
        ("num_refine_starts", num_refine_starts),
        ("num_refine_iterations", num_refine_iterations),
        ("num_refine_samples", num_refine_samples),
        ("num_spin_samples", num_spin_samples),
        ("max_search_points", max_search_points),
    ):
        if value is not None and not isinstance(value, (int, np.integer)):
            raise ValueError(f"`{name}` must be an integer, but got {value!r}.")
    if num_samples < 1:
        raise ValueError(f"`num_samples` must be a positive integer, but got {num_samples}.")
    if num_refine_starts < 1:
        raise ValueError(f"`num_refine_starts` must be a positive integer, but got {num_refine_starts}.")
    if num_refine_iterations < 0:
        raise ValueError(f"`num_refine_iterations` must be a non-negative integer, but got {num_refine_iterations}.")
    if num_refine_samples < 1:
        raise ValueError(f"`num_refine_samples` must be a positive integer, but got {num_refine_samples}.")
    if num_spin_samples < 0:
        raise ValueError(f"`num_spin_samples` must be a non-negative integer, but got {num_spin_samples}.")
    if max_search_points is not None and max_search_points < 1:
        raise ValueError(f"`max_search_points` must be a positive integer or None, but got {max_search_points}.")


def _num_chunks(num_shapes, num_rows, num_points_hint):
    """Threads per orientation: enough to fill the device, but few enough to keep one launch small.

    The candidate count alone is far too little parallelism to fill a GPU, so the point loop is
    split as well. The cap keeps the atomic contention on each orientation's bounds low, and the
    thread budget keeps a launch over many shapes within the size of a launch grid.
    """
    budget = max(1, _OBB_MAX_LAUNCH_THREADS // (num_shapes * num_rows))
    return max(1, min(_OBB_POINT_CHUNKS, num_points_hint, budget))


def _fit_oriented_bounding_boxes(
    points,
    offsets,
    num_shapes,
    max_shape_points,
    measure_type,
    num_samples,
    include_axis_aligned,
    include_pca,
    max_search_points,
    num_refine_starts,
    num_refine_iterations,
    num_refine_samples,
    num_spin_samples,
    device,
):
    measure_code = int(OBBMeasureType(measure_type))
    # ``None`` means "score over every point", which a stride of 1 selects for any shape size.
    search_cap = max_search_points if max_search_points is not None else _INT32_MAX
    # Host-side bound on how many points a single shape contributes to scoring, used only to size launches.
    search_hint = min(max_shape_points, search_cap)

    # Points used to *score* orientations. When some shape is larger than ``search_cap``, gather a
    # compact strided subsample of every shape so that the search cost is independent of the point
    # count; otherwise every shape is scored as is. The winner's box is recomputed over the full
    # set at the end, so the returned box is exact either way. Only the sizes the host can bound
    # are used to allocate: each shape keeps at most ``search_cap`` points.
    if search_hint < max_shape_points:
        search_counts = wp.zeros(num_shapes + 1, dtype=wp.int32, device=device)
        search_offsets = wp.empty(num_shapes + 1, dtype=wp.int32, device=device)
        wp.launch(
            oriented_bounding_box_search_count_kernel,
            dim=num_shapes,
            inputs=[offsets, search_cap],
            outputs=[search_counts],
            device=device,
        )
        array_scan(search_counts, search_offsets, inclusive=False)
        search_points = wp.empty(min(points.shape[0], num_shapes * search_cap), dtype=wp.vec3, device=device)
        copy_chunks = max(1, min(search_hint, _OBB_MAX_LAUNCH_THREADS // num_shapes))
        wp.launch(
            oriented_bounding_box_subsample_kernel,
            dim=(num_shapes, copy_chunks),
            inputs=[points, offsets, search_cap, search_offsets, copy_chunks],
            outputs=[search_points],
            device=device,
        )
        search_cap = _INT32_MAX
    else:
        search_points, search_offsets = points, offsets

    # Candidate orientations: the spiral first, then any extras in the trailing slots.
    num_candidates = num_samples + int(include_axis_aligned) + int(include_pca)
    if (
        num_shapes
        * max(num_candidates, num_refine_starts * num_refine_samples, num_refine_starts * 3 * num_spin_samples)
        > _OBB_MAX_SCRATCH_ROWS
    ):
        raise ValueError(
            f"Fitting {num_shapes} shapes with {num_candidates} candidate orientations each needs too much "
            "scratch memory; fit the shapes in smaller groups or reduce `num_samples`, `num_refine_starts`, "
            "`num_refine_samples` or `num_spin_samples`."
        )
    rotations = wp.empty((num_shapes, num_candidates), dtype=wp.quat, device=device)

    wp.launch(
        oriented_bounding_box_samples_kernel,
        dim=(num_shapes, num_samples),
        inputs=[num_samples],
        outputs=[rotations],
        device=device,
    )

    slot = num_samples
    if include_axis_aligned:
        wp.launch(
            oriented_bounding_box_identity_kernel, dim=num_shapes, inputs=[slot], outputs=[rotations], device=device
        )
        slot += 1

    if include_pca:
        pca_chunks = _num_chunks(num_shapes, 1, search_hint)
        point_sum = wp.zeros(num_shapes, dtype=wp.vec3, device=device)
        covariance = wp.zeros(num_shapes, dtype=wp.mat33, device=device)
        wp.launch(
            point_sum_kernel,
            dim=(num_shapes, pca_chunks),
            inputs=[search_points, search_offsets, search_cap, pca_chunks],
            outputs=[point_sum],
            device=device,
        )
        wp.launch(
            point_covariance_kernel,
            dim=(num_shapes, pca_chunks),
            inputs=[search_points, search_offsets, search_cap, pca_chunks, point_sum],
            outputs=[covariance],
            device=device,
        )
        wp.launch(
            oriented_bounding_box_pca_kernel,
            dim=num_shapes,
            inputs=[covariance, slot],
            outputs=[rotations],
            device=device,
        )
        slot += 1

    measures, _, _ = _score_orientations(
        search_points,
        search_offsets,
        search_cap,
        rotations,
        _num_chunks(num_shapes, num_candidates, search_hint),
        measure_code,
        device,
    )

    # Refinement starts: the best-scoring candidates, skipping any within ``_OBB_START_SEPARATION`` of an
    # earlier pick (modulo the box symmetries). The single best sample is often not in the basin of the
    # best orientation, so several distinct basins are refined and the best result is kept. Everything
    # stays on the device so the search remains capturable in a CUDA graph.
    num_starts = num_refine_starts
    champion = wp.empty((num_shapes, num_starts), dtype=wp.quat, device=device)
    champion_measure = wp.empty((num_shapes, num_starts), dtype=wp.float32, device=device)
    work = wp.clone(measures)
    for k in range(num_starts):
        if k > 0:
            wp.launch(
                oriented_bounding_box_block_kernel,
                dim=(num_shapes, num_candidates),
                inputs=[rotations, champion, k - 1, math.cos(0.5 * _OBB_START_SEPARATION)],
                outputs=[work],
                device=device,
            )
        wp.launch(
            oriented_bounding_box_select_kernel,
            dim=num_shapes,
            inputs=[rotations, measures, work, k],
            outputs=[champion, champion_measure],
            device=device,
        )

    # Coarse-to-fine refinement: each round scores a batch of orientations in a shrinking neighborhood of
    # every start and keeps any improvement. A fixed number of fixed-size rounds keeps it capturable.
    if num_refine_iterations > 0:
        rot_batch = wp.empty((num_shapes, num_starts * num_refine_samples), dtype=wp.quat, device=device)
        batch_chunks = _num_chunks(num_shapes, num_starts * num_refine_samples, search_hint)
        radius = _OBB_REFINE_RADIUS0
        for _ in range(num_refine_iterations):
            wp.launch(
                oriented_bounding_box_refine_generate_kernel,
                dim=(num_shapes, num_starts * num_refine_samples),
                inputs=[champion, radius, num_refine_samples],
                outputs=[rot_batch],
                device=device,
            )
            batch_measures, _, _ = _score_orientations(
                search_points, search_offsets, search_cap, rot_batch, batch_chunks, measure_code, device
            )
            wp.launch(
                oriented_bounding_box_champion_kernel,
                dim=(num_shapes, num_starts),
                inputs=[rot_batch, batch_measures, num_refine_samples],
                outputs=[champion, champion_measure],
                device=device,
            )
            radius *= _OBB_REFINE_SHRINK

    # Spin rounds: rotate each start's champion about each of its own box axes, over shrinking angular
    # ranges. A spin about a box axis leaves that axis, and any hull face it is flush with, in place, so
    # this walks along the floor of the narrow valley that isotropic proposals rarely hit.
    if num_spin_samples > 0:
        spin_batch = 3 * num_spin_samples
        spin_rotations = wp.empty((num_shapes, num_starts * spin_batch), dtype=wp.quat, device=device)
        spin_chunks = _num_chunks(num_shapes, num_starts * spin_batch, search_hint)
        for half_range in _OBB_SPIN_HALF_RANGES:
            wp.launch(
                oriented_bounding_box_spin_generate_kernel,
                dim=(num_shapes, num_starts * spin_batch),
                inputs=[champion, half_range, num_spin_samples],
                outputs=[spin_rotations],
                device=device,
            )
            spin_measures, _, _ = _score_orientations(
                search_points, search_offsets, search_cap, spin_rotations, spin_chunks, measure_code, device
            )
            wp.launch(
                oriented_bounding_box_champion_kernel,
                dim=(num_shapes, num_starts),
                inputs=[spin_rotations, spin_measures, spin_batch],
                outputs=[champion, champion_measure],
                device=device,
            )

    best_champion = wp.empty((num_shapes, 1), dtype=wp.quat, device=device)
    wp.launch(
        oriented_bounding_box_final_kernel,
        dim=num_shapes,
        inputs=[champion, champion_measure],
        outputs=[best_champion],
        device=device,
    )
    champion = best_champion

    # Re-measure the winning orientation over every point of each shape. This makes the returned box
    # exact when the search used a subsample or a refined orientation, and unifies every path.
    out_measure, out_transform, out_extents = _score_orientations(
        points,
        offsets,
        _INT32_MAX,
        champion,
        _num_chunks(num_shapes, 1, max_shape_points),
        measure_code,
        device,
    )

    return out_transform.flatten(), out_extents.flatten(), out_measure.flatten()


def _score_orientations(points, offsets, max_search_points, rotations, num_chunks, measure_code, device):
    """Score every shape's candidate orientations: their box bounds, measure, transform, extents.

    ``rotations`` has shape ``(num_shapes, count)``. Returns ``(measures, transforms, extents)``
    device arrays of that same shape.
    """
    num_shapes, count = rotations.shape
    min_bounds = wp.full((num_shapes, count), wp.vec3(math.inf, math.inf, math.inf), dtype=wp.vec3, device=device)
    max_bounds = wp.full((num_shapes, count), wp.vec3(-math.inf, -math.inf, -math.inf), dtype=wp.vec3, device=device)
    wp.launch(
        oriented_bounding_box_bounds_kernel,
        dim=(num_shapes, count, num_chunks),
        inputs=[points, offsets, max_search_points, rotations, num_chunks],
        outputs=[min_bounds, max_bounds],
        device=device,
    )

    measures = wp.empty((num_shapes, count), dtype=wp.float32, device=device)
    transforms = wp.empty((num_shapes, count), dtype=wp.transform, device=device)
    extents = wp.empty((num_shapes, count), dtype=wp.vec3, device=device)
    wp.launch(
        oriented_bounding_box_measure_kernel,
        dim=(num_shapes, count),
        inputs=[rotations, min_bounds, max_bounds, measure_code],
        outputs=[measures, transforms, extents],
        device=device,
    )
    return measures, transforms, extents


# register API functions so they appear in the documentation

wp._src.context.register_api_function(
    super_fibonacci,
    module="warp.geometry",
)
