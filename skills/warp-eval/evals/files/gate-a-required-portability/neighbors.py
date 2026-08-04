# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

Point3 = tuple[float, float, float]


def neighbors_within_cutoff(
    points: Sequence[Point3],
    cutoff: float,
) -> list[list[int]]:
    """Return symmetric neighbor lists without imposing an ordering contract."""
    cutoff_squared = cutoff * cutoff
    neighbors = [[] for _ in points]

    for i, point_i in enumerate(points):
        for j in range(i + 1, len(points)):
            point_j = points[j]
            dx = point_i[0] - point_j[0]
            dy = point_i[1] - point_j[1]
            dz = point_i[2] - point_j[2]
            if dx * dx + dy * dy + dz * dz <= cutoff_squared:
                neighbors[i].append(j)
                neighbors[j].append(i)

    return neighbors
