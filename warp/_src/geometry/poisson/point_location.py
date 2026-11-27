# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit float64 containing-cell coordinates for sparse and adaptive grids.

Use a containing-cell lookup in the geometry precision for interior constraints.
Feed the result through PicQuadrature's public (cell, coordinates) constructor.
"""

import warp as wp
import warp.fem as fem


@wp.kernel
def _locate_kernel(
    positions: wp.array[wp.vec3d],
    grid: wp.uint64,
    levels: wp.array[wp.uint8],
    cells: wp.array[int],
    coordinates: wp.array[wp.vec3d],
):
    i = wp.tid()
    p = wp.volume_world_to_index(grid, positions[i]) + wp.vec3d(wp.float64(0.5))
    fine = wp.vec3i(int(wp.floor(p[0])), int(wp.floor(p[1])), int(wp.floor(p[2])))
    cells[i] = -1
    coordinates[i] = wp.vec3d(wp.float64(0.0))
    for shift in range(8):
        c = wp.vec3i(
            (fine[0] >> shift) << shift,
            (fine[1] >> shift) << shift,
            (fine[2] >> shift) << shift,
        )
        cell = wp.volume_lookup_index(grid, c[0], c[1], c[2])
        if cell >= 0:
            if int(levels[cell]) == shift:
                cells[i] = cell
                coordinates[i] = (p - wp.vec3d(c)) / wp.float64(1 << shift)
                break


def locate_constraints(geometry, positions):
    cells = wp.empty(len(positions), dtype=int, device=positions.device)
    coordinates = wp.empty(len(positions), dtype=wp.vec3d, device=positions.device)
    args = geometry.cell_arg_value(positions.device)
    levels = (
        args.cell_level
        if isinstance(geometry, fem.AdaptiveNanogrid)
        else wp.zeros(geometry.cell_count(), dtype=wp.uint8, device=positions.device)
    )
    wp.launch(
        _locate_kernel,
        len(positions),
        inputs=[positions, geometry.cell_grid.id, levels, cells, coordinates],
        device=positions.device,
    )
    return cells, coordinates
