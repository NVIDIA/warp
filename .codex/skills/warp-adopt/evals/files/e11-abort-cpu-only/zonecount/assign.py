# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-zone event aggregation for the zonecount appliance.

The hot path: for every event, find the first zone whose polygon contains it.
Irregular, branch-heavy, point-in-polygon per candidate — and entirely serial.
"""

import numpy as np


def point_in_polygon(px, py, xs, ys):
    inside = False
    j = len(xs) - 1
    for i in range(len(xs)):
        if (ys[i] > py) != (ys[j] > py):
            x_cross = (xs[j] - xs[i]) * (py - ys[i]) / (ys[j] - ys[i]) + xs[i]
            if px < x_cross:
                inside = not inside
        j = i
    return inside


def assign_zones(events, zones):
    """events: (n, 2) float array. zones: list of (xs, ys) rings."""
    out = np.full(len(events), -1, dtype=np.int32)
    for e in range(len(events)):
        px, py = events[e]
        for z, (xs, ys) in enumerate(zones):
            if point_in_polygon(px, py, xs, ys):
                out[e] = z
                break
    return out


def counts_per_zone(events, zones):
    assigned = assign_zones(events, zones)
    return np.bincount(assigned[assigned >= 0], minlength=len(zones))
