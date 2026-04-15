# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""STL (Stereolithography) file format parser."""

from __future__ import annotations

# Standard library
import os
import struct

# Third-party
import numpy as np

from warp._src.io.mesh import MeshData, _apply_flip_winding


def _detect_stl_format(filename: str) -> str:
    """Detect STL format by attempting to parse.

    Returns "binary" or "ascii". Raises RuntimeError if neither works.

    Args:
        filename: Path to the STL file.

    Returns:
        "binary" or "ascii".
    """
    # Check file size first
    file_size = os.path.getsize(filename)

    # Binary STL must be at least 84 bytes (80 header + 4 count)
    if file_size < 84:
        # Must be ASCII
        try:
            _read_ascii_stl(filename)
            return "ascii"
        except (UnicodeDecodeError, ValueError, OSError):
            pass

    # Try binary first (fast)
    try:
        with open(filename, "rb") as f:
            f.read(80)  # Skip header
            num_tris = struct.unpack("<I", f.read(4))[0]
            # Check if file size matches expected size
            expected_size = 84 + num_tris * 50
            if file_size == expected_size:
                # Try to read a triangle to verify
                f.read(12)  # normal
                for _ in range(3):
                    f.read(12)  # vertex
                f.read(2)  # attribute byte count
                return "binary"
    except (struct.error, OSError, EOFError):
        pass

    # Try ASCII
    try:
        _read_ascii_stl(filename)
        return "ascii"
    except (UnicodeDecodeError, ValueError, OSError):
        raise RuntimeError(f"Unable to parse STL file: '{filename}'") from None


def _deduplicate_stl_vertices(
    raw_points: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate STL vertices using spatial hashing.

    More memory-efficient than np.unique() for large meshes.
    Uses a dictionary to track first occurrence of each grid-rounded position.

    Args:
        raw_points: Raw vertex positions, shape (N, 3).
        tolerance: Distance tolerance for merging vertices.

    Returns:
        Tuple of (unique_points, unique_indices).
    """
    if tolerance <= 0.0:
        return raw_points, np.arange(len(raw_points), dtype=np.int32)

    # Scale points to integer grid
    scale = 1.0 / tolerance
    grid_points = np.round(raw_points * scale).astype(np.int64)

    # Use dictionary for first occurrence lookup
    seen = {}
    unique_indices = []
    unique_points = []

    for i, gp in enumerate(grid_points):
        key = tuple(gp)
        if key not in seen:
            seen[key] = len(unique_points)
            unique_points.append(raw_points[i])
        unique_indices.append(seen[key])

    return (
        np.array(unique_points, dtype=np.float32),
        np.array(unique_indices, dtype=np.int32),
    )


def _read_binary_stl(filename: str, merge_tolerance: float) -> MeshData:
    """Read binary STL with vertex deduplication using spatial hashing.

    Args:
        filename: Path to the binary STL file.
        merge_tolerance: Distance tolerance for vertex merging.

    Returns:
        MeshData containing points and indices. Note: STL face normals are
        not included since they are per-face, not per-vertex.
    """
    with open(filename, "rb") as f:
        f.read(80)  # Skip header
        num_tris = struct.unpack("<I", f.read(4))[0]

        if num_tris == 0:
            raise RuntimeError(f"No triangles found in STL file: '{filename}'")

        # Pre-allocate: worst case = 3 * num_tris unique vertices
        raw_points = np.empty((num_tris * 3, 3), dtype=np.float32)
        # STL face normals are read but not used (per-face, not per-vertex)
        face_normals = np.empty((num_tris, 3), dtype=np.float32)

        for i in range(num_tris):
            normal = struct.unpack("<3f", f.read(12))
            face_normals[i] = normal
            for j in range(3):
                raw_points[i * 3 + j] = struct.unpack("<3f", f.read(12))
            f.read(2)  # attribute byte count

    # Vertex deduplication using spatial hash
    points, indices = _deduplicate_stl_vertices(raw_points, merge_tolerance)

    # Note: STL normals are per-face, not per-vertex. After deduplication,
    # the number of vertices differs from the number of faces, so we cannot
    # return face_normals in the normals field (which is documented as shape (N, 3)).
    return MeshData(
        points=points,
        indices=indices,
    )


def _read_ascii_stl(filename: str) -> MeshData:
    """Read ASCII STL format.

    Args:
        filename: Path to the ASCII STL file.

    Returns:
        MeshData containing points and indices.
    """
    points = []
    indices = []
    vertex_offset = 0
    in_solid = False
    in_facet = False

    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # noqa: PLW2901
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            keyword = parts[0].lower()

            if keyword == "solid":
                in_solid = True
            elif keyword == "endsolid":
                in_solid = False
            elif keyword == "facet" and in_solid:
                in_facet = True
            elif keyword == "endfacet":
                in_facet = False
            elif keyword == "vertex" and in_facet:
                if len(parts) >= 4:
                    points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    if len(points) % 3 == 0:
                        indices.extend([vertex_offset, vertex_offset + 1, vertex_offset + 2])
                        vertex_offset += 3

    if not points:
        raise RuntimeError(f"No vertices found in STL file: '{filename}'")

    return MeshData(
        points=np.array(points, dtype=np.float32),
        indices=np.array(indices, dtype=np.int32),
    )


def read_stl(filename: str, flip_winding: bool = False, merge_tolerance: float = 1e-6) -> MeshData:
    """Read a mesh from an STL file (binary or ASCII).

    STL files store each triangle independently, so vertices are duplicated
    across triangles. This function deduplicates vertices within the
    merge_tolerance.

    Args:
        filename: Path to the STL file.
        flip_winding: If True, reverse triangle winding order.
        merge_tolerance: Distance tolerance for vertex deduplication.

    Returns:
        MeshData containing points and indices.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be parsed or contains no triangles.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"STL file not found: '{filename}'")

    # Auto-detect format
    format_type = _detect_stl_format(filename)

    if format_type == "binary":
        data = _read_binary_stl(filename, merge_tolerance)
    else:
        data = _read_ascii_stl(filename)
        # Apply same deduplication as binary format for consistency
        if merge_tolerance > 0:
            points, indices = _deduplicate_stl_vertices(data.points, merge_tolerance)
            data = MeshData(points=points, indices=indices)

    if flip_winding:
        data.indices, data.normals = _apply_flip_winding(data.indices, data.normals)

    return data


def write_stl(
    points: np.ndarray,
    indices: np.ndarray,
    filename: str,
    binary: bool = True,
) -> None:
    """Write a mesh to an STL file.

    Args:
        points: Vertex positions, shape (N, 3).
        indices: Triangle indices, shape (M * 3,).
        filename: Output file path.
        binary: If True, write binary STL (default). If False, write ASCII.
    """
    indices_reshaped = indices.reshape(-1, 3)

    if binary:
        with open(filename, "wb") as f:
            # Write header (80 bytes)
            f.write(b"\x00" * 80)

            # Write number of triangles
            num_tris = len(indices_reshaped)
            f.write(struct.pack("<I", num_tris))

            # Write triangles
            for face in indices_reshaped:
                v0 = points[face[0]]
                v1 = points[face[1]]
                v2 = points[face[2]]

                # Compute face normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm_length = np.linalg.norm(normal)
                if norm_length > 0:
                    normal = normal / norm_length
                else:
                    normal = np.array([0.0, 0.0, 1.0])

                # Write normal
                f.write(struct.pack("<3f", normal[0], normal[1], normal[2]))

                # Write vertices
                for v in (v0, v1, v2):
                    f.write(struct.pack("<3f", v[0], v[1], v[2]))

                # Write attribute byte count (usually 0)
                f.write(struct.pack("<H", 0))
    else:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("solid mesh\n")
            for face in indices_reshaped:
                v0 = points[face[0]]
                v1 = points[face[1]]
                v2 = points[face[2]]

                # Compute face normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm_length = np.linalg.norm(normal)
                if norm_length > 0:
                    normal = normal / norm_length
                else:
                    normal = np.array([0.0, 0.0, 1.0])

                f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
                f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
                f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write("endsolid mesh\n")
