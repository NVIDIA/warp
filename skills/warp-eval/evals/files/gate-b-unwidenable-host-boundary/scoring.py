# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def nearest_segment_score(
    point: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
) -> float:
    """Return the inverse distance to the closest line segment."""
    best_squared = np.inf

    for start, end in zip(segment_starts, segment_ends, strict=True):
        edge = end - start
        edge_squared = float(np.dot(edge, edge))
        if edge_squared == 0.0:
            projection = start
        else:
            t = float(np.dot(point - start, edge) / edge_squared)
            projection = start + np.clip(t, 0.0, 1.0) * edge
        delta = point - projection
        best_squared = min(best_squared, float(np.dot(delta, delta)))

    return 1.0 / (1.0 + np.sqrt(best_squared))
