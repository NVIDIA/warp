# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

Aabb = tuple[float, float, float, float]


def overlapping_pairs(boxes: Sequence[Aabb]) -> list[tuple[int, int]]:
    """Return all overlapping 2D axis-aligned bounding-box pairs."""
    pairs = []

    for i, (min_x_i, min_y_i, max_x_i, max_y_i) in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            min_x_j, min_y_j, max_x_j, max_y_j = boxes[j]
            separated = max_x_i < min_x_j or max_x_j < min_x_i or max_y_i < min_y_j or max_y_j < min_y_i
            if not separated:
                pairs.append((i, j))

    return pairs
