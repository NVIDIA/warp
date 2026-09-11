# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for :mod:`warp.geometry` tests.

Mesh builders return ``(points, indices)`` as NumPy arrays of shape
``(num_vertices, 3)`` and ``(3 * num_triangles,)``. All meshes are generated
procedurally so the tests stay deterministic and need no asset files.
"""

from __future__ import annotations

import math

import numpy as np


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    points = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )  # fmt: skip
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int32,
    )  # fmt: skip
    return points, faces


def _subdivide(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split every triangle into four, sharing the new edge midpoints. Preserves winding."""
    verts = list(points)
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            cache[key] = len(verts)
            verts.append(0.5 * (points[a] + points[b]))
        return cache[key]

    new_faces = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])

    return np.array(verts, dtype=np.float64), np.array(new_faces, dtype=np.int32)


def icosphere(subdivisions: int = 2, radius: float = 1.0, center=(0.0, 0.0, 0.0)):
    """Build a closed, outward-oriented genus-0 sphere approximation."""
    points, faces = _icosahedron()
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    for _ in range(subdivisions):
        points, faces = _subdivide(points, faces)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    points = points * radius + np.asarray(center, dtype=np.float64)
    return points.astype(np.float32), faces.flatten().astype(np.int32)
