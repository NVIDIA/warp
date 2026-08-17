# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod

import warp as wp


def validate_field(field: wp.array) -> None:
    """Validate that ``field`` is a non-empty 3D array of ``wp.float32`` values.

    Raises:
        ValueError: If ``field`` is not a 3D array or is empty.
        TypeError: If the ``field`` data type is not ``wp.float32``.
    """
    if len(field.shape) != 3:
        raise ValueError(f"Expected a 3D array for 'field', but got an array with shape {field.shape}.")

    if field.size == 0:
        raise ValueError("The 'field' array cannot be empty.")

    if field.dtype != wp.float32:
        raise TypeError(f"Expected a dtype of wp.float32 for 'field', but got {field.dtype}.")


def resolve_domain_bounds(
    shape: tuple[int, int, int],
    domain_bounds_lower_corner: wp.vec3 | tuple[float, float, float] | None,
    domain_bounds_upper_corner: wp.vec3 | tuple[float, float, float] | None,
) -> tuple[wp.vec3, wp.vec3]:
    """Apply the default bounds policy and compute the grid spacing.

    Args:
        shape: The grid dimensions, as numbers of nodes ``(nx, ny, nz)``.
        domain_bounds_lower_corner: The 3D coordinate that the grid's corner
            at index (0,0,0) maps to, or ``None`` for the default policy.
        domain_bounds_upper_corner: The 3D coordinate that the grid's corner
            at index (nx-1, ny-1, nz-1) maps to, or ``None`` for the default
            policy.

    Returns:
        A tuple ``(lower_corner, grid_delta)`` where ``grid_delta`` is the
        per-axis cell size.
    """
    # Parse out dimensions, being careful to distinguish between nodes and cells
    nnode_x, nnode_y, nnode_z = shape[0], shape[1], shape[2]
    ncell_x, ncell_y, ncell_z = nnode_x - 1, nnode_y - 1, nnode_z - 1

    # Apply default policies for bounds
    if domain_bounds_lower_corner is None:
        domain_bounds_lower_corner = wp.vec3((0.0, 0.0, 0.0))
    if domain_bounds_upper_corner is None:
        # The default convention is to treat the nodes of the grid as having integer coordinates at 0,1,2,...
        # This means the upper-rightmost node of the grid has coordinates (nnode_x-1, nnode_y-1, nnode_z-1)
        # (which happens to be the same as the number cells, although it may be more confusing to think of it that way)
        domain_bounds_upper_corner = wp.vec3((float(nnode_x - 1), float(nnode_y - 1), float(nnode_z - 1)))

    # quietly allow tuples as input too, although this technically violates
    # the type hinting
    domain_bounds_lower_corner = wp.vec3(domain_bounds_lower_corner)
    domain_bounds_upper_corner = wp.vec3(domain_bounds_upper_corner)

    # Compute the grid spacing
    domain_width = domain_bounds_upper_corner - domain_bounds_lower_corner
    grid_delta = wp.cw_div(domain_width, wp.vec3(ncell_x, ncell_y, ncell_z))

    return domain_bounds_lower_corner, grid_delta


class IsoSurfaceBase(ABC):
    """Abstract base class for isosurface extraction from dense 3D scalar fields.

    Concrete backends such as :class:`warp.MarchingCubes` and
    :class:`warp.SurfaceNets` implement this interface so that extraction
    algorithms can be swapped behind a single API: construct an instance with
    the grid dimensions, then call :meth:`~.surface` repeatedly to extract
    meshes from fields of that size, reading the results from the
    :attr:`verts` and :attr:`indices` attributes. For a stateless one-shot
    extraction, use the :meth:`~.extract` class method instead.

    All backends share the same conventions: the input is a dense 3D
    ``wp.float32`` field sampled at grid nodes, the output faces are wound
    counter-clockwise when viewed from outside for a signed distance field
    (negative inside), and the ``indices`` array is a flat ``wp.int32`` array
    listing the vertices of each face in order. Backends produce triangles,
    where each group of three consecutive entries forms one face, unless they
    support another face type and were configured to use it (see the
    ``topology`` parameter of :class:`warp.SurfaceNets`, which can produce
    quads instead).

    Args:
        nx: Number of grid nodes in the x-direction.
        ny: Number of grid nodes in the y-direction.
        nz: Number of grid nodes in the z-direction.
        domain_bounds_lower_corner: The 3D coordinate that the grid's corner
          at index (0,0,0) maps to. Defaults to ``(0.0, 0.0, 0.0)`` if
          ``None``.
        domain_bounds_upper_corner: The 3D coordinate that the grid's corner
          at index (nx-1, ny-1, nz-1) maps to. Defaults to align with the
          grid's maximal indices if ``None``.

    Attributes:
        nx (int): The number of grid nodes in the x-direction.
        ny (int): The number of grid nodes in the y-direction.
        nz (int): The number of grid nodes in the z-direction.
        domain_bounds_lower_corner (warp.vec3f | tuple | None): The lower bound
          for the mesh coordinate scaling.
        domain_bounds_upper_corner (warp.vec3f | tuple | None): The upper bound
          for the mesh coordinate scaling.
        verts (warp.array | None): An array of vertex positions of type
          :class:`warp.vec3f` for the output mesh.
          This is populated by calling the :meth:`~.surface` method.
        indices (warp.array | None): An array of face vertex indices of type
          :class:`warp.int32` for the output mesh.
          This is populated by calling the :meth:`~.surface` method.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        *,
        domain_bounds_lower_corner: wp.vec3 | tuple[float, float, float] | None = None,
        domain_bounds_upper_corner: wp.vec3 | tuple[float, float, float] | None = None,
    ):
        # Input domain sizes, as number of nodes in the grid (note this is 1 more than the number of cubes)
        self.nx = nx
        self.ny = ny
        self.nz = nz

        # Geometry of the extraction domain
        # (or None, to implicitly use a domain with integer-coordinate nodes)
        self.domain_bounds_lower_corner = domain_bounds_lower_corner
        self.domain_bounds_upper_corner = domain_bounds_upper_corner

        # Output arrays
        self.verts: wp.array(dtype=wp.vec3f) | None = None
        self.indices: wp.array(dtype=wp.int32) | None = None

    def resize(self, nx: int, ny: int, nz: int) -> None:
        """Update the grid dimensions for the context.

        This allows the instance to be reused for scalar fields of a different
        resolution. The new dimensions take effect on the next call to
        :meth:`~.surface`.

        Args:
          nx: New number of nodes in the x-direction.
          ny: New number of nodes in the y-direction.
          nz: New number of nodes in the z-direction.
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def _check_field_shape(self, field: wp.array) -> None:
        """Check that a field matches the configured grid dimensions.

        Raises:
            ValueError: If the shape of ``field`` does not match the
                configured grid dimensions of the instance.
        """
        # nx, ny, nz is the number of nodes, which should agree with the size of the field
        if field.shape != (self.nx, self.ny, self.nz):
            raise ValueError(
                f"Field shape {field.shape} does not match context grid dimensions {(self.nx, self.ny, self.nz)}."
            )

    @abstractmethod
    def surface(self, field: wp.array(dtype=float, ndim=3), threshold: float) -> None:
        """Compute a 2D surface mesh of a given isosurface from a 3D scalar field.

        The resulting mesh data is stored in the :attr:`verts` and
        :attr:`indices` attributes, replacing any previous outputs.

        Args:
          field: A 3D scalar field whose shape must match the grid dimensions
            (nx, ny, nz) of the instance.
          threshold: The field value defining the isosurface to extract.

        Raises:
          ValueError: If the shape of ``field`` does not match the configured
            grid dimensions of the instance.
        """

    @classmethod
    @abstractmethod
    def extract(
        cls,
        field: wp.array3d(dtype=wp.float32),
        threshold: float = 0.0,
        *,
        domain_bounds_lower_corner: wp.vec3 | tuple[float, float, float] | None = None,
        domain_bounds_upper_corner: wp.vec3 | tuple[float, float, float] | None = None,
    ) -> tuple[wp.array(dtype=wp.vec3), wp.array(dtype=wp.int32)]:
        """Extract a mesh from a 3D scalar field in a single stateless call.

        Args:
            field: A 3D array representing the scalar values on a regular grid.
            threshold: The field value defining the isosurface to extract.
            domain_bounds_lower_corner: The 3D coordinate that the grid's corner
                at index (0,0,0) maps to. Defaults to ``(0.0, 0.0, 0.0)``
                if ``None``.
            domain_bounds_upper_corner: The 3D coordinate that the grid's corner
                at index (nx-1, ny-1, nz-1) maps to. Defaults to align with the
                grid's maximal indices if ``None``.

        Returns:
            A tuple ``(vertices, indices)``, where ``indices`` is a flat
            ``wp.int32`` array in which each group of three consecutive
            entries forms one triangle by referencing vertices in the
            ``vertices`` array. Backends that support another face type accept
            a keyword argument to select it, and then write that type of face
            to ``indices`` instead (see the ``topology`` parameter of
            :class:`warp.SurfaceNets`).

        Raises:
            ValueError: If ``field`` is not a 3D array or is empty.
            TypeError: If the ``field`` data type is not ``wp.float32``.
        """
