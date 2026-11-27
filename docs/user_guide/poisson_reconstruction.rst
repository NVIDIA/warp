Poisson surface reconstruction
==============================

.. currentmodule:: warp.geometry

:func:`poisson_reconstruct` builds an adaptive implicit surface from oriented
points. Inputs are contiguous one-dimensional arrays of ``wp.vec3`` or
``wp.vec3d`` on CUDA, with matching shape, precision and device. Use a positive
``point_weight`` for screened Poisson reconstruction or zero for unscreened
Poisson reconstruction. Mesh extraction is optional::

    import warp as wp
    import warp.geometry as geometry

    result = geometry.poisson_reconstruct(positions, normals, depth=8)
    values, gradients, valid = result.evaluate(queries)
    vertices, indices = geometry.poisson_extract_mesh(result)
    mesh = wp.Mesh(points=vertices, indices=indices)  # Optional spatial-query mesh.

``values`` are implicit function values; the surface is ``value == result.iso_value``.
The returned gradients are spatial gradients in the original input coordinates.
Use ``centered=True`` to subtract the isovalue from batched values. Invalid queries
(nonfinite or outside the normalization cube) return NaN values and gradients
and zero validity. This field is not a signed-distance function.

The bounding cube is centered on the finite input bounding box and expands its
longest side by ``scale``. By default extraction uses this exact cube, with nodes
at ``i / 2**depth`` in cube coordinates: 257 nodes per axis at depth 8, or 513
at depth 9. Vertices are transformed back using the retained cube-to-input map.
The default sparse method bounds the multilevel field on its hierarchy, refines
only regions that may contain a crossing, and passes crossing cells to
:func:`sparse_marching_cubes_from_cells`. Dense extraction samples the same aligned
grid in full and uses :class:`IsoSurfaceMarchingCubes`.
Select ``method="sparse"`` or ``method="dense"``. Both return device arrays
``(vertices, indices)``: float32 3D positions and flat int32 triangle indices.
Mesh extraction currently supports float32 fields only.

Override native resolution with ``resolution=2**extraction_depth + 1`` (maximum
depth 15). At every intermediate depth, convex-hull bounds account for the
coarse cumulative polynomial and finer corrections. A cell is discarded only
when its complete field range excludes the isovalue; uncertain cells subdivide.
Target-depth cells use exact corner sign tests. All extracted cells have equal
width and share the same sampling lattice, including across adaptive interfaces.
The certificate includes floating-point margins and the evaluator's boundary
conventions. It preserves crossings extractable by matching dense marching cubes;
features smaller than the requested MC grid can still be missed by either method.

For an independent resampling grid, supply both ``resolution`` and ``bounds``.
Bounds must lie inside the normalization cube; resolution may then be any integer
or three integer node counts of at least two. This explicit resampling route
classifies every cell in slabs and does not use the native hierarchy certificates.

The generic :func:`sparse_marching_cubes` can skip regions using Lipschitz pruning
when the caller has a proven bound. Poisson reconstruction supplies no such
bound, and its implicit field is not a signed-distance function. That generic
route remains separate from :func:`poisson_extract_mesh`.

Single-point device queries
---------------------------

Use :meth:`PoissonReconstruction.device_field` to obtain a zero-copy field handle
for a custom kernel or a composed device function. Keep the reconstruction alive
and its arrays immutable until kernel execution and backward finish::

    @wp.kernel
    def query_kernel(
        field: geometry.PoissonField,
        points: wp.array[wp.vec3],
        values: wp.array[float],
    ):
        i = wp.tid()
        value, gradient, valid = geometry.poisson_evaluate(field, points[i])
        if valid:
            values[i] = value

    values = wp.zeros(len(queries), device=queries.device)
    wp.launch(query_kernel, len(queries),
              inputs=[result.device_field(), queries], outputs=[values],
              device=queries.device)

The float64 overload takes :class:`PoissonField64` and ``wp.vec3d``. The fast
batched evaluator remains a separate optimized path. Both return the same values,
gradients and validity convention. Device queries support first-order derivatives
of values and spatial gradients, including query-position derivatives when the
reconstruction itself is untracked.

Optional differentiation
------------------------

Supply a tape explicitly to retain reconstruction derivatives. Reconstruction
must run outside an active tape, and at least one input must have
``requires_grad=True``. Record queries, mesh extraction and downstream losses on
that same tape::

    tape = wp.Tape()
    result = geometry.poisson_reconstruct(positions, normals, depth=6, tape=tape)
    with tape:
        values, gradients, valid = result.evaluate(queries, centered=True)
        vertices, indices = geometry.poisson_extract_mesh(result)
        # Launch a loss kernel using values, gradients or vertices here.
    # tape.backward(loss)

Derivatives follow the finite numerical algorithm, including its finite solver
iterations, normalization, screening and isovalue. Octree topology, sample grouping,
support decisions, solver stopping decisions and marching-cubes connectivity are
held fixed. Native extraction differentiates the cube-to-input normalization map;
explicit resampling bounds are constant. Derivatives are local to those discrete decisions; second-order
derivatives are unsupported. The host ``iso_value`` is a cached constant; use
``iso_value_array`` for the tracked device scalar. Without a tape, reconstruction
does not allocate derivative checkpoints.

Reconstruction and mesh extraction read adaptive counts on the host and cannot be
captured in a CUDA graph. Device queries can be captured after module loading.
For tracked results, acquire the field handle on the stream that will launch the
query to establish readiness. Synchronize downstream work before backward on a
different stream, and do not share gradient buffers across concurrent backward passes.

Numerical scope
---------------

The discretization uses degree-1 B-spline basis functions with Neumann boundaries
and approximate screening. ``warp.fem`` supplies adaptive-grid topology, basis and
quadrature machinery; specialized kernels assemble sparse operators and transfer
between levels. A cascadic multilevel solve uses conjugate gradients at the root,
multigrid cycles through ``base_depth``, and ordered Gauss-Seidel relaxation at finer
levels. No FFT or periodic grid is used. Exact point interpolation, confidence
weighting and higher basis degrees are unsupported.

Run ``python -m warp.examples.geometry.example_poisson_reconstruction`` for a small
oriented-ellipsoid example with optional USD output.
