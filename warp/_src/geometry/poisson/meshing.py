# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional GPU marching cubes. Importing reconstruction does not load this module.

The native sparse path prunes aligned cells using polynomial convex-hull bounds.
Explicit world bounds select uniform resampling in bounded slabs. Neither path
needs a signed-distance or Lipschitz assumption. All backends use Warp's
fixed marching-cubes case table; selection/connectivity carry no derivative.
"""

import math
import operator

import warp as wp
from warp._src.geometry.marching_cubes import IsoSurfaceMarchingCubes
from warp._src.geometry.sparse_marching_cubes import sparse_marching_cubes_from_cells
from warp.utils import array_scan

from .field import LevelField

# Match Warp geometry's grid-position arithmetic, including fused operations.
wp.set_module_options({"fuse_fp": True})


@wp.func
def _corner(c: int):
    # Warp MC ordering: (000,100,110,010,001,101,111,011).
    x = int(c == 1 or c == 2 or c == 5 or c == 6)
    y = int(c == 2 or c == 3 or c == 6 or c == 7)
    return wp.vec3i(x, y, c // 4)


@wp.kernel
def _grid_points_kernel(
    ny: int, nz: int, start: int, lower: wp.vec3, upper: wp.vec3, step: wp.vec3, points: wp.array[wp.vec3]
):
    i = wp.tid()
    cell = wp.vec3(float(start + i // (ny * nz)), float((i // nz) % ny), float(i % nz))
    points[i] = wp.min(wp.max(lower + wp.cw_mul(step, cell), lower), upper)


@wp.kernel
def _mark_cells_kernel(
    values: wp.array[float], ny: int, nz: int, start: int, mask: wp.array[wp.uint32], status: wp.array[int]
):
    i = wp.tid()
    x, y, z = i // ((ny - 1) * (nz - 1)), (i // (nz - 1)) % (ny - 1), i % (nz - 1)
    code = int(0)
    for c in range(8):
        q = wp.vec3i(x, y, z) + _corner(c)
        v = values[(q[0] * ny + q[1]) * nz + q[2]]
        if not wp.isfinite(v):
            wp.atomic_or(status, 1, 1)
        if v >= 0.0:
            code = code | (1 << c)
    if code != 0 and code != 255:
        index = ((start + x) * (ny - 1) + y) * (nz - 1) + z
        wp.atomic_or(mask, index // 32, wp.uint32(1) << wp.uint32(index % 32))


@wp.kernel
def _word_counts_kernel(mask: wp.array[wp.uint32], counts: wp.array[int]):
    i = wp.tid()
    counts[i] = wp.bit_count(mask[i])


@wp.kernel
def _total_kernel(offsets: wp.array[int], status: wp.array[int]):
    status[0] = offsets[offsets.shape[0] - 1]


@wp.kernel
def _emit_cells_kernel(mask: wp.array[wp.uint32], offsets: wp.array[int], ny: int, nz: int, cells: wp.array[wp.vec3i]):
    word = wp.tid()
    bits = mask[word]
    out = offsets[word] - wp.bit_count(bits)
    for b in range(32):
        if bits & (wp.uint32(1) << wp.uint32(b)):
            i = word * 32 + b
            cells[out] = wp.vec3i(i // ((ny - 1) * (nz - 1)), (i // (nz - 1)) % (ny - 1), i % (nz - 1))
            out += 1


@wp.kernel
def _cell_points_kernel(
    cells: wp.array[wp.vec3i], lower: wp.vec3, upper: wp.vec3, step: wp.vec3, points: wp.array[wp.vec3]
):
    i = wp.tid()
    c = cells[i // 8] + _corner(i % 8)
    p = lower + wp.cw_mul(step, wp.vec3(float(c[0]), float(c[1]), float(c[2])))
    points[i] = wp.min(wp.max(p, lower), upper)


@wp.kernel
def _model_vertices_kernel(vertices: wp.array[wp.vec3], lower: wp.vec3, step: wp.vec3, out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = lower + wp.cw_mul(step, vertices[i])


@wp.kernel
def _check_values_kernel(values: wp.array[float], status: wp.array[int]):
    i = wp.tid()
    if not wp.isfinite(values[i]):
        wp.atomic_or(status, 1, 1)


@wp.kernel
def _center_kernel(values: wp.array[float], iso: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = values[i] - iso[0]


@wp.kernel
def _brood_points_kernel(cells: wp.array[wp.vec3i], width: float, points: wp.array[wp.vec3]):
    index = wp.tid()
    local = index % 27
    node = cells[8 * (index // 27)] + wp.vec3i(local // 9, (local // 3) % 3, local % 3)
    points[index] = wp.vec3(node) * width


@wp.func
def _candidate_value(values: wp.array[float], cell: int, corner: int, shared: bool):
    if shared:
        child = cell % 8
        node = wp.vec3i(child // 4, (child // 2) % 2, child % 2) + _corner(corner)
        return values[(cell // 8) * 27 + (node[0] * 3 + node[1]) * 3 + node[2]]
    return values[8 * cell + corner]


@wp.kernel
def _crossing_counts_kernel(values: wp.array[float], shared: bool, counts: wp.array[int], status: wp.array[int]):
    index = wp.tid()
    code = int(0)
    for corner in range(8):
        value = _candidate_value(values, index, corner, shared)
        if not wp.isfinite(value):
            wp.atomic_or(status, 1, 1)
        if value >= 0.0:
            code |= 1 << corner
    counts[index] = int(code != 0 and code != 255)


@wp.kernel
def _compact_crossings_kernel(
    cells: wp.array[wp.vec3i],
    values: wp.array[float],
    shared: bool,
    counts: wp.array[int],
    offsets: wp.array[int],
    out_cells: wp.array[wp.vec3i],
    out_values: wp.array[float],
):
    index = wp.tid()
    if counts[index]:
        target = offsets[index] - 1
        out_cells[target] = cells[index]
        for corner in range(8):
            out_values[target * 8 + corner] = _candidate_value(values, index, corner, shared)


@wp.kernel
def _cube_vertices_kernel(vertices: wp.array[wp.vec3], transforms: wp.array[wp.mat44], out: wp.array[wp.vec3]):
    index = wp.tid()
    out[index] = wp.transform_point(transforms[0], vertices[index])


def extract_native_mesh(reconstruction, *, resolution, backend, iso, differentiable, statistics=None):
    """Extract in exact dyadic cube coordinates and transform vertices to the model."""
    from .field_adjoint import center_values, cube_field, evaluate_cube  # noqa: PLC0415
    from .meshing_bounds import candidate_cells, sample_cube_kernel  # noqa: PLC0415

    field = reconstruction.field
    if differentiable:
        cube = field.cube
        transforms = field.transforms
        if field.ready is not None:
            wp.get_stream(field.coefficients.device).wait_event(field.ready)
    else:
        cube = cube_field(field.keys, field.coefficients, field.cumulative, field.depth)
        transforms = wp.array(
            [reconstruction.metadata["cube_to_model"]], dtype=wp.mat44, device=field.coefficients.device
        )
    if resolution is None:
        depth = cube.depth
    else:
        try:
            shape = (operator.index(resolution),) * 3
        except TypeError:
            try:
                shape = tuple(operator.index(value) for value in resolution)
            except (TypeError, ValueError):
                raise ValueError("native resolution must be 2**depth + 1, or three equal node counts") from None
        if (
            isinstance(resolution, bool)
            or len(shape) != 3
            or len(set(shape)) != 1
            or shape[0] < 2
            or (shape[0] - 1) & (shape[0] - 2)
        ):
            raise ValueError("native resolution must be 2**depth + 1, or three equal node counts")
        depth = (shape[0] - 1).bit_length() - 1
    if depth > 15:
        raise ValueError("native extraction depth must be at most 15")
    device = cube.coefficients.device
    width = 1.0 / (1 << depth)

    def sample(points, *, tracked=False):
        if tracked:
            # Query positions are fixed; their gradient storage is private to
            # the cube evaluator's VJP. Only samples and the final model frame
            # participate in the reconstruction's derivative.
            points.requires_grad = True
            values, _, _ = evaluate_cube(cube, points, tape=field.tape)
            return center_values(values, iso, tape=field.tape)
        centered = wp.empty(len(points), dtype=float, device=device)
        wp.launch(
            sample_cube_kernel,
            len(points),
            inputs=[cube.keys, cube.coefficients, cube.cumulative, points, cube.depth, iso, centered],
            device=device,
            record_tape=False,
        )
        return centered

    if backend == "dense":
        size = (1 << depth) + 1
        if 3 * size**3 > 2**31 - 1:
            raise ValueError("dense MC edge indexing exceeds int32; use the sparse method")
        points = wp.empty(size**3, dtype=wp.vec3, device=device)
        wp.launch(
            _grid_points_kernel,
            len(points),
            inputs=[size, size, 0, wp.vec3(0), wp.vec3(1), wp.vec3(width), points],
            device=device,
            record_tape=False,
        )
        values = sample(points, tracked=differentiable)
        status = wp.zeros(2, dtype=int, device=device)
        wp.launch(_check_values_kernel, len(values), inputs=[values, status], device=device, record_tape=False)
        if status.numpy()[1]:
            raise ValueError("field returned nonfinite values inside extraction bounds")
        vertices, indices = IsoSurfaceMarchingCubes.extract(
            values.reshape((size,) * 3), threshold=0.0, lower=wp.vec3(0), upper=wp.vec3(1)
        )
    else:
        cells = candidate_cells(cube, depth, iso, statistics=statistics)
        if len(cells) > (2**31 - 1) // 8:
            raise ValueError("candidate corner count exceeds int32 indexing")
        count = 0
        shared = depth > 0
        if len(cells):
            points = wp.empty(len(cells) // 8 * 27 if shared else 8, dtype=wp.vec3, device=device)
            if shared:
                wp.launch(
                    _brood_points_kernel, len(points), inputs=[cells, width, points], device=device, record_tape=False
                )
            else:
                wp.launch(
                    _cell_points_kernel,
                    len(points),
                    inputs=[cells, wp.vec3(0), wp.vec3(1), wp.vec3(width), points],
                    device=device,
                    record_tape=False,
                )
            values = sample(points)
            counts = wp.empty(len(cells), dtype=int, device=device)
            offsets = wp.empty_like(counts)
            status = wp.zeros(2, dtype=int, device=device)
            wp.launch(
                _crossing_counts_kernel,
                len(cells),
                inputs=[values, shared, counts, status],
                device=device,
                record_tape=False,
            )
            array_scan(counts, offsets, inclusive=True)
            wp.launch(_total_kernel, 1, inputs=[offsets, status], device=device, record_tape=False)
            count, invalid = map(int, status.numpy())
            if invalid:
                raise ValueError("field returned nonfinite values inside extraction bounds")
        if statistics is not None:
            statistics["crossing_cells"] = count
        if not count:
            return wp.empty(0, dtype=wp.vec3, device=device, requires_grad=differentiable), wp.empty(
                0, dtype=int, device=device
            )
        crossing_cells = wp.empty(count, dtype=wp.vec3i, device=device)
        crossing_values = wp.empty(count * 8, dtype=float, device=device)
        wp.launch(
            _compact_crossings_kernel,
            len(cells),
            inputs=[cells, values, shared, counts, offsets, crossing_cells, crossing_values],
            device=device,
            record_tape=False,
        )
        if differentiable:
            points = wp.empty(count * 8, dtype=wp.vec3, device=device)
            wp.launch(
                _cell_points_kernel,
                len(points),
                inputs=[crossing_cells, wp.vec3(0), wp.vec3(1), wp.vec3(width), points],
                device=device,
                record_tape=False,
            )
            crossing_values = sample(points, tracked=True)
        vertices, indices = sparse_marching_cubes_from_cells(
            crossing_cells, crossing_values.reshape((count, 8)), cell_width=width, threshold=0.0, device=device
        )
    model_vertices = wp.empty_like(vertices, requires_grad=differentiable)
    wp.launch(_cube_vertices_kernel, len(vertices), inputs=[vertices, transforms, model_vertices], device=device)
    return model_vertices, indices


def _grid(resolution, bounds):
    if isinstance(resolution, int) and not isinstance(resolution, bool):
        resolution = (resolution,) * 3
    try:
        shape = tuple(operator.index(v) for v in resolution)
    except (TypeError, ValueError):
        raise ValueError("resolution must be an integer or three integers >= 2") from None
    if len(shape) != 3 or any(isinstance(v, bool) for v in resolution) or min(shape) < 2:
        raise ValueError("resolution must be an integer or three integers >= 2")
    try:
        lo, hi = (tuple(float(v) for v in row) for row in bounds)
    except (TypeError, ValueError):
        raise ValueError("bounds must contain two finite 3D corners") from None
    if (
        len(lo) != 3
        or len(hi) != 3
        or not all(math.isfinite(v) for v in (*lo, *hi))
        or any(a >= b for a, b in zip(lo, hi, strict=True))
    ):
        raise ValueError("bounds must be finite with lower < upper on every axis")
    if math.prod(n - 1 for n in shape) > 2**31 - 1:
        raise ValueError("extraction cell count exceeds int32 indexing")
    if math.prod(shape) > 2**31 - 1:
        raise ValueError("grid node count exceeds int32 sampling indices")
    lower, upper = wp.vec3(*lo), wp.vec3(*hi)
    # Match Warp MC's float32 bounds/spacing arithmetic.
    step = wp.cw_div(upper - lower, wp.vec3(*(n - 1 for n in shape)))
    if not all(math.isfinite(float(step[a])) and step[a] > 0 for a in range(3)):
        raise ValueError("bounds/spacing must be representable in float32")
    return shape, lower, upper, step


def extract_isosurface(
    field,
    *,
    resolution,
    bounds,
    backend="sparse",
    device=None,
    slab_size=16,
    classification_field=None,
):
    """Extract the zero set of a batched GPU callback ``field(points) -> values``.

    ``points`` are float32 ``wp.vec3`` model positions; returned values must be
    finite 1D float32 arrays on the same device. Subtract any (possibly tracked)
    isovalue in the callback. Vertices/flat triangle indices remain on device.

    ``sparse`` exhaustively classifies cells using streamed slabs and a bitmask,
    then calls Warp's explicit-cell sparse MC. It is conservative relative to
    the matched dense grid, including disconnected surfaces and non-SDF fields.
    ``dense`` materializes the full scalar grid for Warp's dense MC.

    Warp tracks vertex interpolation when the callback returns tracked values
    inside a Tape. Grid/cell selection, bounds and connectivity are fixed.
    Callbacks must be deterministic and must not mutate shared output buffers:
    backward needs each invocation's values. GPU count reads inside Warp's
    extractors remain; this wrapper never downloads field values or meshes.

    For ``sparse``, an optional ``classification_field`` callback can evaluate
    the same values without recording gradients for fixed cell selection.
    Only retained corners use ``field`` in that case. The classification callback
    must agree with ``field`` at every grid node; a surrogate can omit crossings.
    """
    if backend not in ("dense", "sparse"):
        raise ValueError("backend must be dense or sparse")
    if classification_field is not None and backend != "sparse":
        raise ValueError("classification_field is only used by the sparse backend")
    shape, lower, upper, step = _grid(resolution, bounds)
    if not isinstance(slab_size, int) or isinstance(slab_size, bool) or slab_size < 1:
        raise ValueError("slab_size must be a positive integer")
    device = wp.get_device(device)

    def sample(points, *, classification=False):
        callback = classification_field if classification and classification_field is not None else field
        values = callback(points)
        if (
            not isinstance(values, wp.array)
            or values.dtype != wp.float32
            or values.shape != (len(points),)
            or values.device != device
            or not values.is_contiguous
        ):
            raise ValueError("field must return contiguous float32 values, one per point, on the query device")
        return values

    with wp.ScopedDevice(device):
        if backend == "dense":
            if 3 * math.prod(shape) > 2**31 - 1:
                raise ValueError("dense MC edge indexing exceeds int32; use the sparse backend")
            points = wp.empty(math.prod(shape), dtype=wp.vec3, device=device)
            wp.launch(
                _grid_points_kernel,
                len(points),
                inputs=[shape[1], shape[2], 0, lower, upper, step, points],
                device=device,
            )
            values = sample(points)
            status = wp.zeros(2, dtype=int, device=device)
            wp.launch(_check_values_kernel, len(values), inputs=[values, status], device=device, record_tape=False)
            if status.numpy()[1]:
                raise ValueError("field returned nonfinite values inside extraction bounds")
            vertices, indices = IsoSurfaceMarchingCubes.extract(
                values.reshape(shape), threshold=0.0, lower=lower, upper=upper
            )
        else:
            nx, ny, nz = shape
            words = (math.prod(n - 1 for n in shape) + 31) // 32
            mask = wp.zeros(words, dtype=wp.uint32, device=device)
            status = wp.zeros(2, dtype=int, device=device)
            tracked = False
            for start in range(0, nx - 1, slab_size):
                width = min(slab_size, nx - 1 - start)
                points = wp.empty((width + 1) * ny * nz, dtype=wp.vec3, device=device)
                wp.launch(
                    _grid_points_kernel,
                    len(points),
                    inputs=[ny, nz, start, lower, upper, step, points],
                    device=device,
                    record_tape=False,
                )
                values = sample(points, classification=True)
                tracked = tracked or values.requires_grad
                wp.launch(
                    _mark_cells_kernel,
                    width * (ny - 1) * (nz - 1),
                    inputs=[values, ny, nz, start, mask, status],
                    device=device,
                    record_tape=False,
                )
            counts = wp.empty(words, dtype=int, device=device)
            offsets = wp.empty_like(counts)
            wp.launch(_word_counts_kernel, words, inputs=[mask, counts], device=device, record_tape=False)
            array_scan(counts, offsets, inclusive=True)
            wp.launch(_total_kernel, 1, inputs=[offsets, status], device=device, record_tape=False)
            count, invalid = map(int, status.numpy())
            if invalid:
                raise ValueError("field returned nonfinite values inside extraction bounds")
            if count > (2**31 - 1) // 8:
                raise ValueError("retained corner count exceeds int32 indexing")
            if count:
                cells = wp.empty(count, dtype=wp.vec3i, device=device)
                wp.launch(
                    _emit_cells_kernel, words, inputs=[mask, offsets, ny, nz, cells], device=device, record_tape=False
                )
                del mask, counts, offsets, points, values
                points = wp.empty(count * 8, dtype=wp.vec3, device=device)
                wp.launch(
                    _cell_points_kernel,
                    len(points),
                    inputs=[cells, lower, upper, step, points],
                    device=device,
                    record_tape=False,
                )
                values = sample(points)
                grid_vertices, indices = sparse_marching_cubes_from_cells(
                    cells, values.reshape((count, 8)), cell_width=1.0, threshold=0.0, device=device
                )
                vertices = wp.empty_like(grid_vertices, requires_grad=grid_vertices.requires_grad)
                wp.launch(
                    _model_vertices_kernel, len(vertices), inputs=[grid_vertices, lower, step, vertices], device=device
                )
            else:
                vertices = wp.empty(0, dtype=wp.vec3, device=device, requires_grad=tracked)
                indices = wp.empty(0, dtype=wp.int32, device=device)
    return vertices, indices


def extract_mesh(reconstruction, *, resolution=None, bounds=None, iso_value=None, differentiable=False, **options):
    """Mesh a Reconstruction or LevelField without changing reconstruction cost.

    Warp MC currently takes float32 fields. Float64 inputs fail explicitly.
    An optional one-element device ``iso_value`` array retains threshold gradients.
    With differentiable=True, a result from reconstruct_differentiable retains
    gradients through mesh vertices to input positions, normals and isovalue.
    Use the same tape as reconstruction. A direct LevelField without cumulative
    coefficients also supports tracked coefficients and an optional isovalue.
    Cell selection, bounds, connectivity and topology remain fixed.
    """
    field = getattr(reconstruction, "field", reconstruction)
    full_backward = not isinstance(field, LevelField)
    if full_backward:
        from .field_adjoint import (  # noqa: PLC0415
            DifferentiableField,
        )

        if not isinstance(field, DifferentiableField):
            raise TypeError("expected a LevelField or differentiable reconstruction field")
        if not differentiable:
            raise ValueError("a differentiable reconstruction requires differentiable=True for meshing")
    if field.coefficients.dtype != wp.float32:
        raise ValueError("Warp mesh extraction currently requires float32 fields")
    if differentiable and not full_backward and (field.cumulative is not None or not field.coefficients.requires_grad):
        raise ValueError(
            "differentiable meshing requires a direct LevelField with tracked coefficients; "
            "use reconstruct_differentiable for the reconstruction backward pass"
        )
    device = field.coefficients.device
    if "device" in options and wp.get_device(options["device"]) != device:
        raise ValueError("mesh device must match the field device")
    options["device"] = device
    if iso_value is None:
        iso_value = field.iso_value
    if isinstance(iso_value, wp.array):
        if iso_value.dtype != wp.float32 or iso_value.shape != (1,) or iso_value.device != device:
            raise ValueError("iso_value must be one float32 value on the field device")
        if iso_value.requires_grad and not differentiable:
            raise ValueError("tracked iso_value requires differentiable=True")
        iso = iso_value
    else:
        if not math.isfinite(iso_value):
            raise ValueError("iso_value must be finite")
        iso = wp.full(1, float(iso_value), dtype=wp.float32, device=device)

    if bounds is None:
        slab_size = options.get("slab_size", 16)
        if not isinstance(slab_size, int) or isinstance(slab_size, bool) or slab_size < 1:
            raise ValueError("slab_size must be a positive integer")
        return extract_native_mesh(
            reconstruction,
            resolution=resolution,
            backend=options.get("backend", "sparse"),
            iso=iso,
            differentiable=differentiable,
        )
    if resolution is None:
        raise ValueError("explicit bounds require an explicit resolution")

    def sample(points):
        if full_backward:
            from .field_adjoint import (  # noqa: PLC0415
                center_values,
            )

            values, _, _ = field.evaluate(points)
            return center_values(values, iso, tape=field.tape)
        else:
            values, _, _ = field.evaluate(points, requires_grad=differentiable)
        out = wp.empty_like(values, requires_grad=differentiable)
        wp.launch(_center_kernel, len(values), inputs=[values, iso, out], device=device)
        return out

    if full_backward and options.get("backend", "sparse") == "sparse":

        def classify(points):
            values, _, _ = field.evaluate(points, requires_grad=False)
            out = wp.empty_like(values, requires_grad=False)
            wp.launch(_center_kernel, len(values), inputs=[values, iso, out], device=device, record_tape=False)
            return out

        options.setdefault("classification_field", classify)
    vertices, indices = extract_isosurface(sample, resolution=resolution, bounds=bounds, **options)
    if differentiable and not len(vertices):
        vertices.requires_grad = True
    return vertices, indices
