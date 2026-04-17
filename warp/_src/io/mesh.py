# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mesh file I/O utilities for OBJ, STL, and PLY formats."""

from __future__ import annotations

# Standard library
import dataclasses
import os
from typing import TYPE_CHECKING

# Third-party
import numpy as np

# Warp
import warp as wp
from warp._src.context import DeviceLike, get_device

if TYPE_CHECKING:
    from warp._src.types import BvhConstructor


@dataclasses.dataclass
class MeshData:
    """Raw mesh data parsed from a file.

    Attributes:
        points: Vertex positions as NumPy array, shape (N, 3), dtype float32.
        indices: Triangle indices as NumPy array, shape (M * 3,), dtype int32.
        normals: Vertex normals if available, shape (N, 3) or None.
        uvs: Texture coordinates if available, shape (N, 2) or None.
        colors: Vertex colors if available, shape (N, 3) or (N, 4) or None.
    """

    points: np.ndarray
    indices: np.ndarray
    normals: np.ndarray | None = None
    uvs: np.ndarray | None = None
    colors: np.ndarray | None = None

    def to_warp_mesh(
        self,
        device: DeviceLike | None = None,
        **kwargs,
    ) -> wp.Mesh:
        """Convert to a wp.Mesh.

        Note: device is passed to wp.array() calls, NOT to wp.Mesh constructor.
        The wp.Mesh constructor inherits device from the points array.

        Args:
            device: Device on which to create the mesh arrays. If None, uses
                the current default device.
            **kwargs: Additional arguments passed to wp.Mesh constructor
                (e.g., support_winding_number, bvh_constructor, bvh_leaf_size).

        Returns:
            A wp.Mesh with BVH built and ready for queries.
        """
        return wp.Mesh(
            points=wp.array(self.points, dtype=wp.vec3, device=device),
            indices=wp.array(self.indices, dtype=wp.int32, device=device),
            **kwargs,
        )


def load_mesh(
    filename: str,
    device: DeviceLike | None = None,
    *,
    file_format: str | None = None,
    flip_winding: bool = False,
    max_file_size_mb: float | None = 500.0,
    stl_merge_tolerance: float = 1e-6,
    support_winding_number: bool = False,
    bvh_constructor: BvhConstructor | str | None = None,
    bvh_leaf_size: int | None = None,
) -> wp.Mesh:
    """Load a triangle mesh from a file.

    Supports OBJ, STL (binary/ASCII), and PLY (binary/ASCII) formats.
    The format is detected automatically from the file extension.

    Args:
        filename: Path to the mesh file. Both absolute and relative paths
            are accepted (relative paths are from current working directory).
        device: Device on which to create the mesh arrays. If None, calls
            wp.get_device() to use the current default device.
        file_format: Explicit format override (``\"obj\"``, ``\"stl\"``, ``\"ply\"``).
            Use this when the file extension is missing or incorrect.
            If None, format is inferred from the filename extension.
        flip_winding: If True, reverse triangle winding order. Use this
            when the mesh has inverted normals (e.g., clockwise vs
            counter-clockwise convention). Useful when signed distance
            queries give incorrect results.
        max_file_size_mb: Maximum file size in MB before raising ValueError.
            Set to None to disable the check. Default is 500 MB.
        stl_merge_tolerance: Vertex merge tolerance for STL files.
            STL files duplicate vertices per-triangle; this tolerance
            controls merging (in world units).
        support_winding_number: Passed to wp.Mesh constructor.
        bvh_constructor: Passed to wp.Mesh constructor. When None,
            the optimal constructor is chosen automatically (``\"sah\"`` for CPU,
            ``\"lbvh\"`` for CUDA).
        bvh_leaf_size: Passed to wp.Mesh constructor.

    Returns:
        A wp.Mesh with BVH built and ready for queries.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not recognized, or if the file
            size exceeds max_file_size_mb.
        RuntimeError: If the file cannot be parsed or contains no vertices/faces.

    Example:
        >>> import warp as wp
        >>> mesh = wp.load_mesh("bunny.obj")
        >>> mesh = wp.load_mesh("part.stl", device="cuda:0")
        >>> mesh = wp.load_mesh("mesh.unknown", file_format="ply")
    """
    # Resolve device: None means use the current default device
    if device is None:
        device = get_device()

    # Delegate parsing to read_mesh (handles validation and format detection)
    data = read_mesh(
        filename,
        file_format=file_format,
        flip_winding=flip_winding,
        max_file_size_mb=max_file_size_mb,
        stl_merge_tolerance=stl_merge_tolerance,
    )

    # Convert to wp.Mesh with device and BVH parameters
    return data.to_warp_mesh(
        device=device,
        support_winding_number=support_winding_number,
        bvh_constructor=bvh_constructor,
        bvh_leaf_size=bvh_leaf_size,
    )


def read_mesh(
    filename: str,
    *,
    file_format: str | None = None,
    flip_winding: bool = False,
    max_file_size_mb: float | None = 500.0,
    stl_merge_tolerance: float = 1e-6,
) -> MeshData:
    """Read mesh data from a file without constructing a wp.Mesh.

    Use this when you need access to normals, UVs, or colors,
    or when you want to modify the data before creating a wp.Mesh.

    Args:
        filename: Path to the mesh file.
        file_format: Explicit format override (``\"obj\"``, ``\"stl\"``, ``\"ply\"``).
            If None, format is inferred from the filename extension.
        flip_winding: If True, reverse triangle winding order.
        max_file_size_mb: Maximum file size in MB before raising ValueError.
        stl_merge_tolerance: Vertex merge tolerance for STL files.

    Returns:
        MeshData containing points, indices, and optional normals/uvs/colors.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not recognized.
        RuntimeError: If the file cannot be parsed.

    Example:
        >>> import warp as wp
        >>> data = wp.read_mesh("bunny.obj")
        >>> print(f"Loaded {data.points.shape[0]} vertices")
        >>> # Modify points if needed
        >>> mesh = data.to_warp_mesh()
    """
    # File existence check
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Mesh file not found: '{filename}'")

    # File size check
    file_size = os.path.getsize(filename)
    if max_file_size_mb is not None:
        max_bytes = max_file_size_mb * 1024 * 1024
        if file_size > max_bytes:
            raise ValueError(
                f"File too large: {file_size / (1024 * 1024):.1f} MB exceeds limit of {max_file_size_mb} MB"
            )

    # Format detection
    if file_format is None:
        _, ext = os.path.splitext(filename)
        ext = ext.lower().lstrip(".")
        if not ext:
            raise ValueError(
                "Cannot detect format from file extension. Use the 'file_format' parameter to specify explicitly."
            )
        file_format = ext

    # Dispatch to format-specific parser
    from warp._src.io.obj import read_obj  # noqa: PLC0415
    from warp._src.io.ply import read_ply  # noqa: PLC0415
    from warp._src.io.stl import read_stl  # noqa: PLC0415

    if file_format == "obj":
        return read_obj(filename, flip_winding=flip_winding)
    elif file_format == "stl":
        return read_stl(filename, flip_winding=flip_winding, merge_tolerance=stl_merge_tolerance)
    elif file_format == "ply":
        return read_ply(filename, flip_winding=flip_winding)
    else:
        raise ValueError(f"Unsupported format: '{file_format}'. Supported formats are: obj, stl, ply")


def save_mesh(
    mesh: wp.Mesh,
    filename: str,
    *,
    binary: bool = True,
) -> None:
    """Save a triangle mesh to a file.

    Supports OBJ, STL, and PLY formats. The format is detected
    automatically from the file extension.

    Args:
        mesh: The wp.Mesh to save.
        filename: Output file path (extension determines format).
        binary: If True (default), write binary for STL/PLY.
            OBJ is always ASCII.

    Raises:
        ValueError: If the file format is not recognized.

    Example:
        >>> import warp as wp
        >>> mesh = wp.load_mesh("bunny.obj")
        >>> wp.save_mesh(mesh, "output.stl")
        >>> wp.save_mesh(mesh, "output.ply", binary=False)
    """
    _, ext = os.path.splitext(filename)
    ext = ext.lower().lstrip(".")

    if not ext:
        raise ValueError(
            "Cannot detect format from file extension. Provide a filename with .obj, .stl, or .ply extension."
        )

    # Get mesh data as numpy arrays
    points = mesh.points.numpy().astype(np.float32)
    indices = mesh.indices.numpy().astype(np.int32)

    from warp._src.io.obj import write_obj  # noqa: PLC0415
    from warp._src.io.ply import write_ply  # noqa: PLC0415
    from warp._src.io.stl import write_stl  # noqa: PLC0415

    if ext == "obj":
        write_obj(points, indices, filename)
    elif ext == "stl":
        write_stl(points, indices, filename, binary=binary)
    elif ext == "ply":
        write_ply(points, indices, filename, binary=binary)
    else:
        raise ValueError(f"Unsupported format: '{ext}'. Supported formats are: obj, stl, ply")


def _flip_winding_order(indices: np.ndarray) -> np.ndarray:
    """Reverse triangle winding order.

    Transforms [v0, v1, v2, v3, v4, v5, ...] to [v2, v1, v0, v5, v4, v3, ...]

    Args:
        indices: Flattened triangle indices array.

    Returns:
        Indices with reversed winding order.
    """
    # Reshape to (num_tris, 3), reverse each triangle, flatten back
    return indices.reshape(-1, 3)[:, [2, 1, 0]].reshape(-1)


def _apply_flip_winding(
    indices: np.ndarray,
    normals: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply winding order flip, also flipping normals if present.

    Flipping triangle winding order requires flipping vertex normals to
    maintain correct surface orientation.

    Args:
        indices: Triangle indices array.
        normals: Vertex normals array or None.

    Returns:
        Tuple of (flipped_indices, flipped_normals).
    """
    indices = _flip_winding_order(indices)
    if normals is not None:
        normals = -normals  # Negate all normal vectors
    return indices, normals
