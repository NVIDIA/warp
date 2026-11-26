# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public contracts for adaptive Poisson reconstruction and field queries."""

from __future__ import annotations

from typing import Literal

import warp as wp
from warp._src import context


def _validate_vectors(array, name, *, device=None, dtype=None):
    """Check the layout and precision required by reconstruction arrays."""
    if not isinstance(array, wp.array):
        raise TypeError(f"{name} must be a Warp array")
    if dtype is None:
        if not any(wp.types.types_equal(array.dtype, vector) for vector in (wp.vec3, wp.vec3d)):
            raise TypeError(f"{name} must contain float32 or float64 3D vectors")
    elif not wp.types.types_equal(array.dtype, dtype):
        raise TypeError(f"{name} must match the reconstruction precision")
    if array.ndim != 1 or not array.is_contiguous:
        raise ValueError(f"{name} must be a contiguous one-dimensional array")
    if device is not None and array.device != device:
        raise ValueError(f"{name} must be on the reconstruction device")


class PoissonReconstruction:
    """Own a reconstructed implicit field and its optional derivative checkpoints.

    Create instances with :func:`warp.geometry.poisson_reconstruct`. The surface
    is the level set ``value == iso_value``; field values are not signed distances.
    Keep inputs and field arrays immutable until all queries and backward passes
    finish. Only first-order derivatives with fixed discrete assignments are
    supported. The result retains the reconstruction tape when one was supplied.
    """

    def __init__(self, result, tape=None):
        self._result = result
        self._tape = tape
        self._dtype = result.field.coefficients.dtype
        self._vector_dtype = wp.vec3d if self._dtype == wp.float64 else wp.vec3
        self._device = result.field.coefficients.device

    @property
    def iso_value(self) -> float:
        """Return the cached host isovalue without synchronizing the device."""
        return self._result.metadata["iso_value"]

    @property
    def iso_value_array(self) -> wp.array:
        """Return the one-element device isovalue, tracked when a tape was supplied."""
        if self._tape is not None:
            return self._result.iso_value
        return self._result.iso_value_array

    def evaluate(self, queries: wp.array, *, centered: bool = False):
        """Evaluate the field and its spatial gradient at a batch of input-coordinate points.

        Args:
            queries: Contiguous 3D vectors on the field device, in the same precision.
                Empty batches are supported. Nonfinite or out-of-domain points are invalid.
            centered: Subtract the isovalue from returned values, so the surface is zero.
                This does not change the returned spatial gradients.

        Returns:
            Values, input-coordinate spatial gradients, and an int32 validity mask.
            Invalid points have NaN values/gradients and a zero mask; valid points have
            a mask of one. Differentiable results record query and input derivatives
            on their reconstruction tape. Untracked results use the independent fast
            batch evaluator and must be queried outside an active tape.
        """
        _validate_vectors(queries, "queries", device=self._device, dtype=self._vector_dtype)
        if not isinstance(centered, bool):
            raise TypeError("centered must be a bool")
        active_tape = context.runtime.tape
        if active_tape is not None and active_tape is not self._tape:
            raise ValueError("evaluate must use the reconstruction tape; use poisson_evaluate for query-only gradients")
        if self._tape is not None:
            return self._result.evaluate(queries, centered=centered)
        if not centered:
            return self._result.evaluate(queries)
        values, gradients, valid = self._result.evaluate(queries)
        if centered:
            from warp._src.geometry.poisson.field_adjoint import (  # noqa: PLC0415
                precision_kernels,
            )

            centered_values = wp.empty_like(values)
            wp.launch(
                precision_kernels(self._dtype).center,
                len(values),
                inputs=[values, self.iso_value_array],
                outputs=[centered_values],
                device=values.device,
            )
            values = centered_values
        return values, gradients, valid

    def device_field(self):
        """Return a zero-copy device handle for :func:`warp.geometry.poisson_evaluate`.

        Returns:
            A :class:`warp.geometry.PoissonField` or :class:`warp.geometry.PoissonField64`
            sharing the result's arrays. Keep this result alive through kernel execution
            and backward. Query-only derivatives are supported even for an untracked
            reconstruction. For a tracked reconstruction, record kernels on its tape;
            creating the handle on the query stream establishes device readiness.
        """
        return self._result.device_field()


def poisson_reconstruct(
    positions: wp.array,
    normals: wp.array,
    *,
    depth: int = 6,
    full_depth: int = 3,
    base_depth: int | None = None,
    solve_depth: int | None = None,
    kernel_depth: int | None = None,
    samples_per_node: float = 1.5,
    scale: float = 1.1,
    point_weight: float = 2.0,
    iterations: int = 8,
    base_v_cycles: int = 1,
    cg_accuracy: float = 1e-3,
    tape: wp.Tape | None = None,
) -> PoissonReconstruction:
    """Reconstruct an adaptive Poisson implicit field from oriented CUDA points.

    Use zero ``point_weight`` for Poisson reconstruction and a positive weight for
    screened reconstruction with approximate interpolation. The implementation uses
    degree-1 basis functions and Neumann boundaries, with a cascadic multilevel solve,
    conjugate gradients at the base level, and ordered Gauss-Seidel relaxation.
    Exact point interpolation, confidence weighting, and other basis degrees are not
    supported. Finite positions are required; invalid or zero normals are omitted.

    This host operation reads adaptive counts and solver status, and cannot run inside
    CUDA graph capture. Call outside an active tape. Supplying ``tape`` opts into a
    checkpointed first-order derivative of the finite algorithm with respect to tracked
    positions/normals. Tree assignments, support, bins and stopping decisions are fixed
    during backward; scalar options are constants. Without a tape no reconstruction
    derivative checkpoints are allocated.

    Args:
        positions: Nonempty contiguous array of float32 or float64 3D vectors on CUDA.
        normals: Oriented normals with the same shape, precision and device as positions.
        depth: Maximum octree depth, between 1 and 15.
        full_depth: Complete-tree depth. Require ``0 <= full_depth <= depth`` and
            ``depth - full_depth <= 7``.
        base_depth: Base solve depth, at most full_depth. None selects full_depth.
        solve_depth: Finest solved level, between full_depth and depth. None selects depth.
        kernel_depth: Density-estimation depth, between zero and depth. None selects
            ``max(depth - 2, 0)``.
        samples_per_node: Finite positive density target used to adapt sampling depth.
        scale: Finite bounding-cube expansion factor, strictly greater than one.
        point_weight: Finite nonnegative screening weight; zero disables screening.
        iterations: Positive relaxation iteration count per solved level.
        base_v_cycles: Positive number of base multigrid cycles.
        cg_accuracy: Finite positive stopping accuracy for the base conjugate-gradient solve.
        tape: Optional tape to retain first-order reconstruction derivatives. At least one
            input must have requires_grad=True. Keep all inputs immutable through backward.

    Returns:
        An owning implicit-field result with host/device isovalues, batched evaluation
        and a device-query handle. Mesh extraction is a separate, optional operation.
    """
    _validate_vectors(positions, "positions")
    _validate_vectors(normals, "normals", device=positions.device, dtype=positions.dtype)
    if not positions.device.is_cuda:
        raise ValueError("Poisson reconstruction requires CUDA input arrays")
    if positions.shape != normals.shape or not len(positions):
        raise ValueError("positions and normals must have matching nonempty shapes")
    if positions.device.is_capturing:
        raise ValueError("Poisson reconstruction cannot run inside CUDA graph capture")
    if context.runtime.tape is not None:
        raise ValueError("call poisson_reconstruct outside an active tape and supply tape explicitly")
    if tape is not None and not isinstance(tape, wp.Tape):
        raise TypeError("tape must be a Warp Tape or None")
    for name, value in (
        ("depth", depth),
        ("full_depth", full_depth),
        ("base_depth", base_depth),
        ("solve_depth", solve_depth),
        ("kernel_depth", kernel_depth),
        ("iterations", iterations),
        ("base_v_cycles", base_v_cycles),
    ):
        if value is None and name in ("base_depth", "solve_depth", "kernel_depth"):
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    options = {
        "depth": depth,
        "full_depth": full_depth,
        "base_depth": base_depth,
        "solve_depth": solve_depth,
        "kernel_depth": kernel_depth,
        "samples_per_node": samples_per_node,
        "scale": scale,
        "point_weight": point_weight,
        "iters": iterations,
        "base_v_cycles": base_v_cycles,
        "cg_accuracy": cg_accuracy,
    }
    if tape is None:
        from warp._src.geometry.poisson.reconstruction import (  # noqa: PLC0415
            reconstruct,
        )

        result = reconstruct(positions, normals, **options)
    else:
        from warp._src.geometry.poisson.reconstruction_adjoint import (  # noqa: PLC0415
            reconstruct_differentiable,
        )

        result = reconstruct_differentiable(positions, normals, tape=tape, **options)
    return PoissonReconstruction(result, tape)


def poisson_extract_mesh(
    result: PoissonReconstruction,
    *,
    resolution=None,
    bounds=None,
    method: Literal["sparse", "dense"] = "sparse",
    iso_value=None,
    slab_size: int = 16,
) -> tuple[wp.array, wp.array]:
    """Extract a triangle mesh from a float32 Poisson reconstruction using Warp marching cubes.

    Args:
        result: Reconstruction whose field will be sampled. Float64 extraction is unsupported.
        resolution: Optional node count. With native bounds, use ``2**depth + 1``
            (or three equal counts), with depth at most 15. Omit to match the
            reconstruction depth. With explicit bounds, an integer or three counts
            of at least two is required.
        bounds: Optional lower and upper 3D corners in input coordinates. Omit to
            align exactly with the reconstruction cube's dyadic grid; vertices are
            transformed back using the reconstruction's retained normalization.
            Explicit bounds select uniform resampling inside the field's domain.
        method: "sparse" conservatively bounds the multilevel polynomial field,
            refines uncertain cells, and sends target-depth crossing cells to sparse
            marching cubes. "dense" samples the same aligned grid in full. With
            explicit bounds, "sparse" classifies the resampling grid in slabs.
        iso_value: Optional finite scalar or one-element float32 device array overriding the
            reconstruction isovalue. Tracked overrides require a differentiable reconstruction.
        slab_size: Positive number of grid-cell slabs classified at once when using
            the sparse method with explicit bounds. Native extraction does not use slabs.

    Returns:
        A tuple ``(vertices, indices)`` of device arrays: float32 3D vertices in input
        coordinates and flat int32 triangle indices, with three indices per triangle.
        Pass these arrays to ``wp.Mesh(points=vertices, indices=indices)`` when a Warp
        query mesh is needed. For a differentiable result, call inside its reconstruction
        tape to record vertex losses. Cell classification and connectivity are fixed
        in backward; gradients propagate through vertex interpolation, field samples,
        the native normalization frame and the finite solve. Explicit bounds are fixed.
        Extraction performs device count reads and is not graph-capturable.
    """
    if not isinstance(result, PoissonReconstruction):
        raise TypeError("result must be a PoissonReconstruction")
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    if method not in ("sparse", "dense"):
        raise ValueError("method must be 'sparse' or 'dense'")
    if result._result.field.coefficients.device.is_capturing:
        raise ValueError("Poisson mesh extraction cannot run inside CUDA graph capture")
    if context.runtime.tape is not result._tape:
        raise ValueError(
            "extract the mesh inside its reconstruction tape, or outside all tapes for an untracked result"
        )
    from warp._src.geometry.poisson.meshing import (  # noqa: PLC0415
        extract_mesh,
    )

    return extract_mesh(
        result._result,
        resolution=resolution,
        bounds=bounds,
        backend=method,
        iso_value=result.iso_value_array if iso_value is None else iso_value,
        differentiable=result._tape is not None,
        slab_size=slab_size,
    )
