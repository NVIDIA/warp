# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PLY (Polygon File Format) parser."""

from __future__ import annotations

# Standard library
import os
import struct

# Third-party
import numpy as np

from warp._src.io.mesh import MeshData, _apply_flip_winding


def _read_ply_header(filename: str) -> dict:
    """Read PLY header and return metadata.

    Args:
        filename: Path to the PLY file.

    Returns:
        Dictionary with header information including format, vertex_count,
        face_count, vertex_properties, and has_colors.
    """
    with open(filename, "rb") as f:
        # Read magic number
        magic = f.readline().decode("ascii").strip()
        if magic != "ply":
            raise RuntimeError(f"Invalid PLY file: '{filename}' (missing magic number)")

        header_info = {
            "format": None,
            "vertex_count": 0,
            "face_count": 0,
            "vertex_properties": [],
            "has_colors": False,
        }

        current_element = None

        while True:
            line = f.readline().decode("ascii").strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            keyword = parts[0]

            if keyword == "end_header":
                break
            elif keyword == "format":
                header_info["format"] = parts[1]
            elif keyword == "element":
                current_element = parts[1]
                if parts[1] == "vertex":
                    header_info["vertex_count"] = int(parts[2])
                elif parts[1] == "face":
                    header_info["face_count"] = int(parts[2])
            elif keyword == "property":
                # Only add properties for the vertex element
                if current_element == "vertex" and len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    header_info["vertex_properties"].append((prop_type, prop_name))
                    if prop_name in ("red", "green", "blue", "alpha"):
                        header_info["has_colors"] = True

    return header_info


def _read_ply_ascii(filename: str, header: dict) -> MeshData:
    """Read ASCII PLY file.

    Args:
        filename: Path to the PLY file.
        header: Parsed header information.

    Returns:
        MeshData containing points, indices, and optional normals/colors.
    """
    points = []
    normals = []
    colors = []
    indices = []

    with open(filename, "rb") as f:
        # Skip to end of header
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break

        # Read vertices
        for _ in range(header["vertex_count"]):
            line = f.readline().decode("ascii").strip()
            values = line.split()
            points.append([float(values[0]), float(values[1]), float(values[2])])

            # Check for normals (nx, ny, nz typically come after x, y, z)
            idx = 3
            for _prop_type, prop_name in header["vertex_properties"][3:]:
                if idx < len(values):
                    if prop_name == "nx" or prop_name == "ny" or prop_name == "nz":
                        if not normals:
                            normals = [[0.0, 0.0, 0.0] for _ in range(len(points))]
                        if prop_name == "nx":
                            normals[-1][0] = float(values[idx])
                        elif prop_name == "ny":
                            normals[-1][1] = float(values[idx])
                        elif prop_name == "nz":
                            normals[-1][2] = float(values[idx])
                    elif prop_name in ("red", "green", "blue", "alpha"):
                        if not colors:
                            colors = [[0, 0, 0] for _ in range(len(points))]
                        color_idx = {"red": 0, "green": 1, "blue": 2}.get(prop_name)
                        if color_idx is not None:
                            colors[-1][color_idx] = int(values[idx])
                idx += 1

        # Read faces
        for _ in range(header["face_count"]):
            line = f.readline().decode("ascii").strip()
            values = list(map(int, line.split()))
            num_verts = values[0]
            face_verts = values[1:]

            # Fan triangulation for n-gons
            if num_verts >= 3:
                for i in range(1, num_verts - 1):
                    indices.extend([face_verts[0], face_verts[i], face_verts[i + 1]])

    if not points:
        raise RuntimeError(f"No vertices found in PLY file: '{filename}'")

    return MeshData(
        points=np.array(points, dtype=np.float32),
        indices=np.array(indices, dtype=np.int32),
        normals=np.array(normals, dtype=np.float32) if normals else None,
        colors=np.array(colors, dtype=np.uint8) if colors else None,
    )


def _read_ply_binary(filename: str, header: dict) -> MeshData:
    """Read binary PLY file (little-endian or big-endian).

    Args:
        filename: Path to the PLY file.
        header: Parsed header information.

    Returns:
        MeshData containing points, indices, and optional normals/colors.
    """
    is_little_endian = header["format"] == "binary_little_endian"
    endian = "<" if is_little_endian else ">"

    # Type mapping for struct
    type_map = {
        "char": "b",
        "uchar": "B",
        "short": "h",
        "ushort": "H",
        "int": "i",
        "int8": "b",
        "int32": "i",
        "uint": "I",
        "uint8": "B",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
    }

    # Build vertex format string
    vertex_format = ""
    vertex_size = 0
    prop_offsets = {}
    prop_index = 0  # Tuple index in unpacked data, not byte offset

    for prop_type, prop_name in header["vertex_properties"]:
        if prop_type in type_map:
            fmt_char = type_map[prop_type]
            vertex_format += fmt_char
            prop_offsets[prop_name] = prop_index
            prop_index += 1
            vertex_size += struct.calcsize(fmt_char)

    with open(filename, "rb") as f:
        # Skip to end of header
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break

        # Pre-allocate arrays
        vertex_count = header["vertex_count"]
        points = np.empty((vertex_count, 3), dtype=np.float32)
        normals = None
        colors = None

        # Check if we have normals
        has_normals = any(name in prop_offsets for name in ["nx", "ny", "nz"])
        if has_normals:
            normals = np.empty((vertex_count, 3), dtype=np.float32)

        # Check if we have colors
        if header["has_colors"]:
            colors = np.empty((vertex_count, 3), dtype=np.uint8)

        # Read vertices
        vertex_struct = struct.Struct(endian + vertex_format)
        for i in range(vertex_count):
            data = vertex_struct.unpack(f.read(vertex_size))
            # Only access properties that exist in the header
            points[i] = [data[prop_offsets["x"]], data[prop_offsets["y"]], data[prop_offsets["z"]]]

            if has_normals and normals is not None:
                nx_idx = prop_offsets.get("nx")
                ny_idx = prop_offsets.get("ny")
                nz_idx = prop_offsets.get("nz")
                if nx_idx is not None and ny_idx is not None and nz_idx is not None:
                    normals[i] = [data[nx_idx], data[ny_idx], data[nz_idx]]

            if header["has_colors"] and colors is not None:
                red_idx = prop_offsets.get("red")
                green_idx = prop_offsets.get("green")
                blue_idx = prop_offsets.get("blue")
                if red_idx is not None and green_idx is not None and blue_idx is not None:
                    colors[i] = [data[red_idx], data[green_idx], data[blue_idx]]

        # Read faces
        indices = []
        face_count = header["face_count"]

        # For faces, format is: uchar count + vertex_indices
        for _ in range(face_count):
            count_byte = f.read(1)
            if not count_byte:
                break
            num_verts = struct.unpack("B", count_byte)[0]

            # Read vertex indices (typically int32/uint)
            idx_format = endian + "i"
            idx_size = struct.calcsize(idx_format)
            face_verts = []
            for _ in range(num_verts):
                idx = struct.unpack(idx_format, f.read(idx_size))[0]
                face_verts.append(idx)

            # Fan triangulation
            if num_verts >= 3:
                for i in range(1, num_verts - 1):
                    indices.extend([face_verts[0], face_verts[i], face_verts[i + 1]])

    return MeshData(
        points=points,
        indices=np.array(indices, dtype=np.int32),
        normals=normals,
        colors=colors,
    )


def read_ply(filename: str, flip_winding: bool = False) -> MeshData:
    """Read a mesh from a PLY file (binary or ASCII).

    Supports ASCII, binary little-endian, and binary big-endian formats.
    Automatically detects vertex properties including normals and colors.

    Args:
        filename: Path to the PLY file.
        flip_winding: If True, reverse triangle winding order.

    Returns:
        MeshData containing points, indices, and optional normals/colors.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be parsed or contains no vertices.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"PLY file not found: '{filename}'")

    header = _read_ply_header(filename)

    if header["vertex_count"] == 0:
        raise RuntimeError(f"No vertices found in PLY file: '{filename}'")

    # Dispatch to format-specific reader
    if header["format"] == "ascii":
        data = _read_ply_ascii(filename, header)
    elif header["format"] in ("binary_little_endian", "binary_big_endian"):
        data = _read_ply_binary(filename, header)
    else:
        raise RuntimeError(f"Unsupported PLY format: '{header['format']}'")

    if flip_winding:
        data.indices, data.normals = _apply_flip_winding(data.indices, data.normals)

    return data


def write_ply(
    points: np.ndarray,
    indices: np.ndarray,
    filename: str,
    binary: bool = True,
    normals: np.ndarray | None = None,
    colors: np.ndarray | None = None,
) -> None:
    """Write a mesh to a PLY file.

    Args:
        points: Vertex positions, shape (N, 3).
        indices: Triangle indices, shape (M * 3,).
        filename: Output file path.
        binary: If True, write binary PLY (default). If False, write ASCII.
        normals: Optional vertex normals, shape (N, 3).
        colors: Optional vertex colors, shape (N, 3) or (N, 4).
    """
    indices_reshaped = indices.reshape(-1, 3)
    num_verts = len(points)
    num_faces = len(indices_reshaped)

    has_normals = normals is not None
    has_colors = colors is not None

    if binary:
        with open(filename, "wb") as f:
            # Write header
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")

            # Vertex properties
            f.write(f"element vertex {num_verts}\n".encode())
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            if has_normals:
                f.write(b"property float nx\n")
                f.write(b"property float ny\n")
                f.write(b"property float nz\n")
            if has_colors:
                f.write(b"property uchar red\n")
                f.write(b"property uchar green\n")
                f.write(b"property uchar blue\n")

            # Face properties
            f.write(f"element face {num_faces}\n".encode())
            f.write(b"property list uchar int vertex_indices\n")
            f.write(b"end_header\n")

            # Write vertices
            vertex_format = "<3f"
            if has_normals:
                vertex_format += "<3f"
            if has_colors:
                vertex_format += "<3B"
            vertex_struct = struct.Struct(vertex_format)

            for i in range(num_verts):
                p = points[i]
                data = [p[0], p[1], p[2]]
                if has_normals:
                    n = normals[i]
                    data.extend([n[0], n[1], n[2]])
                if has_colors:
                    c = colors[i]
                    data.extend([int(c[0]), int(c[1]), int(c[2])])
                f.write(vertex_struct.pack(*data))

            # Write faces
            for face in indices_reshaped:
                f.write(struct.pack("<B", 3))  # 3 vertices per face
                f.write(struct.pack("<3i", face[0], face[1], face[2]))
    else:
        with open(filename, "w") as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")

            # Vertex properties
            f.write(f"element vertex {num_verts}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_normals:
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")
            if has_colors:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

            # Face properties
            f.write(f"element face {num_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Write vertices
            for i in range(num_verts):
                p = points[i]
                line = f"{p[0]} {p[1]} {p[2]}"
                if has_normals:
                    n = normals[i]
                    line += f" {n[0]} {n[1]} {n[2]}"
                if has_colors:
                    c = colors[i]
                    line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
                f.write(line + "\n")

            # Write faces
            for face in indices_reshaped:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
