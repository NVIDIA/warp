# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

Point2 = tuple[float, float]


def points_in_radius(
    points: Sequence[Point2],
    center: Point2,
    radius: float,
) -> list[int]:
    """Return indices of points inside a circular region."""
    radius_squared = radius * radius
    selected = []

    for index, point in enumerate(points):
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        if dx * dx + dy * dy <= radius_squared:
            selected.append(index)

    return selected
