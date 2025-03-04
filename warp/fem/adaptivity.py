# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import numpy as np

import warp as wp
from warp.fem import cache
from warp.fem.domain import Cells
from warp.fem.field import GeometryField
from warp.fem.geometry import AdaptiveNanogrid
from warp.fem.integrate import interpolate
from warp.fem.operator import integrand, lookup
from warp.fem.types import NULL_ELEMENT_INDEX, Domain, Field, Sample


def adaptive_nanogrid_from_hierarchy(
    grids: List[wp.Volume], grading: Optional[str] = None, temporary_store: Optional[cache.TemporaryStore] = None
) -> AdaptiveNanogrid:
    """
    Constructs a :class:`warp.fem.AdaptiveNanogrid` from a non-overlapping grid hierarchy.

    Warning: The result is undefined if there are partial overlaps between levels, that is if a cell at level `l` is only partially covered by cells at levels `l-1` or lower.

    Args:
        grids: List of sparse Volumes, from finest to coarsest
        grading: Supplementary grading condition, may be ``None``, "face" or "vertex"; see :func:`enforce_nanogrid_grading`
        temporary_store: Storage for temporary allocations
    """
    if not grids:
        raise ValueError("No grids to build from!")

    level_count = len(grids)

    device = grids[0].device

    # Concatenate voxels for each grid
    voxel_counts = [grid.get_voxel_count() for grid in grids]

    voxel_offsets = np.cumsum(np.array([0] + voxel_counts))
    merged_ijks = cache.borrow_temporary(temporary_store, dtype=wp.vec3i, shape=int(voxel_offsets[-1]), device=device)
    for l in range(level_count):
        voxel_count = voxel_counts[l]
        grid_voxels = cache.borrow_temporary(temporary_store, shape=(voxel_count,), dtype=wp.vec3i, device=device)
        grids[l].get_voxels(out=grid_voxels.array)

        wp.launch(
            _fill_hierarchy_merged_ijk,
            dim=voxel_count,
            device=device,
            inputs=[l, voxel_offsets[l], grid_voxels.array, merged_ijks.array],
        )

    # Allocate merged grid
    grid_info = grids[0].get_grid_info()
    cell_grid = wp.Volume.allocate_by_voxels(
        merged_ijks.array,
        transform=grid_info.transform_matrix,
        translation=grid_info.translation,
        device=device,
    )

    # Get unique voxel and corresponding level
    cell_count = cell_grid.get_voxel_count()
    cell_ijk = cache.borrow_temporary(temporary_store, shape=(cell_count,), dtype=wp.vec3i, device=device)
    cell_level = wp.array(shape=(cell_count,), dtype=wp.uint8, device=device)

    cell_grid.get_voxels(out=cell_ijk.array)

    cell_grid_ids = wp.array([grid.id for grid in grids], dtype=wp.uint64, device=device)
    wp.launch(
        _fill_hierarchy_cell_level,
        device=device,
        dim=cell_count,
        inputs=[level_count, cell_grid_ids, cell_ijk.array, cell_level],
    )

    cell_grid, cell_level = enforce_nanogrid_grading(
        cell_grid, cell_level, level_count=level_count, grading=grading, temporary_store=temporary_store
    )

    return AdaptiveNanogrid(cell_grid, cell_level=cell_level, level_count=level_count, temporary_store=temporary_store)


def adaptive_nanogrid_from_field(
    coarse_grid: wp.Volume,
    level_count: int,
    refinement_field: GeometryField,
    samples_per_voxel: int = 64,
    grading: Optional[str] = None,
    temporary_store: Optional[cache.TemporaryStore] = None,
) -> AdaptiveNanogrid:
    """
    Constructs a :class:`warp.fem.AdaptiveNanogrid` from a coarse grid and a refinement field.

    Args:
        coarse_grid: Base grid from which to start refining. No voxels will be added outside of the base grid.
        level_count: Maximum number of refinement levels
        refinement_field: Scalar field used as a refinement oracle. If the returned value is negative, the corresponding voxel will be carved out.
            Positive values indicate the desired refinement with 0.0 corresponding to the finest level and 1.0 to the coarsest level.
        samples_per_voxel: How many samples to use for evaluating the refinement field within each voxel
        grading: Supplementary grading condition, may be ``None``, "face" or "vertex"; see :func:`enforce_nanogrid_grading`
        temporary_store: Storage for temporary allocations
    """

    device = coarse_grid.device
    cell_count = coarse_grid.get_voxel_count()

    cell_ijk = cache.borrow_temporary(temporary_store, shape=(cell_count,), dtype=wp.vec3i, device=device)
    cell_level = cache.borrow_temporary(temporary_store, shape=(cell_count,), dtype=wp.uint8, device=device)

    cell_level.array.fill_(level_count - 1)
    coarse_grid.get_voxels(out=cell_ijk.array)

    domain = Cells(refinement_field.geometry)

    fine_count = cache.borrow_temporary(temporary_store, dtype=int, shape=1, device=device)
    fine_count.array.zero_()

    for _ in range(level_count):
        cell_count = cell_ijk.array.shape[0]
        cell_refinement = cache.borrow_temporary(temporary_store, shape=(cell_count,), dtype=wp.int8, device=device)

        with wp.ScopedDevice(device):
            interpolate(
                _count_refined_voxels,
                domain=domain,
                dim=cell_count,
                fields={"field": refinement_field},
                values={
                    "sample_count": samples_per_voxel,
                    "level_count": level_count,
                    "coarse_grid": coarse_grid.id,
                    "coarse_ijk": cell_ijk.array,
                    "coarse_level": cell_level.array,
                    "coarse_refinement": cell_refinement.array,
                    "fine_count": fine_count.array,
                },
            )

        fine_shape = int(fine_count.array.numpy()[0])
        fine_ijk = cache.borrow_temporary(temporary_store, shape=fine_shape, dtype=wp.vec3i, device=device)
        fine_level = cache.borrow_temporary(temporary_store, shape=fine_shape, dtype=wp.uint8, device=device)

        wp.launch(
            _fill_refined_voxels,
            dim=cell_count,
            device=device,
            inputs=[
                cell_ijk.array,
                cell_level.array,
                cell_refinement.array,
                fine_count.array,
                fine_ijk.array,
                fine_level.array,
            ],
        )

        # Fine is the new coarse
        cell_ijk = fine_ijk
        cell_level = fine_level

    wp.launch(_adjust_refined_ijk, dim=fine_shape, device=device, inputs=[cell_ijk.array, cell_level.array])

    # We now have our refined voxels, allocate the grid
    coarse_info = coarse_grid.get_grid_info()
    fine_scale = 1.0 / (1 << (level_count - 1))
    fine_transform = coarse_info.transform_matrix * fine_scale
    fine_translation = coarse_info.translation + (fine_scale - 1.0) * 0.5 * wp.vec3(coarse_grid.get_voxel_size())
    fine_grid = wp.Volume.allocate_by_voxels(
        cell_ijk.array, translation=fine_translation, transform=fine_transform, device=device
    )

    # Reorder cell_levels (voxels will have moved)
    fine_count = fine_grid.get_voxel_count()
    fine_level = wp.array(dtype=wp.uint8, shape=fine_count, device=device)
    wp.launch(
        _fill_refined_level,
        dim=fine_count,
        device=device,
        inputs=[fine_grid.id, cell_ijk.array, cell_level.array, fine_level],
    )

    fine_grid, fine_level = enforce_nanogrid_grading(
        fine_grid, fine_level, level_count=level_count, grading=grading, temporary_store=temporary_store
    )

    return AdaptiveNanogrid(fine_grid, cell_level=fine_level, level_count=level_count, temporary_store=temporary_store)


def enforce_nanogrid_grading(
    cell_grid: wp.Volume,
    cell_level: wp.array,
    level_count: int,
    grading: Optional[str] = None,
    temporary_store: Optional[cache.TemporaryStore] = None,
) -> Tuple[wp.Volume, wp.array]:
    """
    Refines an adaptive grid such that if satisfies a grading condition.

    Arguments are similar to the :class:`warp.fem.AdaptiveNanogrid` constructor, with the
    addition of the `grading` condition which can be:
         - "face": two cells sharing a common face must have a level difference of at most 1
         - "vertex": two cells sharing a common vertex must have a level difference of at most 1
         - "none" or ``None``: no grading condition

    Returns the refined grid and levels
    """

    if not grading or grading == "none" or level_count <= 2:
        # skip
        return cell_grid, cell_level

    device = cell_grid.device

    grid_info = cell_grid.get_grid_info()

    fine_count = cache.borrow_temporary(temporary_store, shape=(1,), dtype=int, device=device)
    grading_kernel = _count_ungraded_faces if grading == "face" else _count_ungraded_vertices

    for _ in range(level_count - 2):
        cell_count = cell_grid.get_voxel_count()
        cell_ijk = cache.borrow_temporary(temporary_store, shape=(cell_count,), dtype=wp.vec3i, device=device)
        cell_grid.get_voxels(out=cell_ijk.array)

        refinement = cache.borrow_temporary(temporary_store, shape=(cell_count,), dtype=int, device=device)
        refinement.array.zero_()

        wp.launch(
            grading_kernel,
            dim=cell_count,
            device=device,
            inputs=[cell_grid.id, cell_ijk.array, cell_level, level_count, refinement.array],
        )

        fine_count.array.fill_(cell_count)
        wp.launch(
            _count_graded_cells,
            dim=cell_count,
            device=device,
            inputs=[
                refinement.array,
                fine_count.array,
            ],
        )

        # Add new coordinates
        fine_shape = int(fine_count.array.numpy()[0])
        if fine_shape == cell_count:
            break

        fine_ijk = cache.borrow_temporary(temporary_store, shape=fine_shape, dtype=wp.vec3i, device=device)
        fine_level = cache.borrow_temporary(temporary_store, shape=fine_shape, dtype=wp.uint8, device=device)

        wp.launch(
            _fill_graded_cells,
            dim=cell_count,
            device=device,
            inputs=[
                cell_ijk.array,
                cell_level,
                refinement.array,
                fine_count.array,
                fine_ijk.array,
                fine_level.array,
            ],
        )

        # Rebuild grid and levels
        cell_grid = wp.Volume.allocate_by_voxels(
            fine_ijk.array, translation=grid_info.translation, transform=grid_info.transform_matrix, device=device
        )
        cell_level = wp.empty(fine_shape, dtype=wp.uint8, device=device)
        wp.launch(
            _fill_refined_level,
            dim=fine_shape,
            device=device,
            inputs=[cell_grid.id, fine_ijk.array, fine_level.array, cell_level],
        )

    return cell_grid, cell_level


@wp.kernel
def _count_ungraded_faces(
    cell_grid: wp.uint64,
    cell_ijk: wp.array(dtype=wp.vec3i),
    cell_level: wp.array(dtype=wp.uint8),
    level_count: int,
    refinement: wp.array(dtype=wp.int32),
):
    cell = wp.tid()

    ijk = cell_ijk[cell]
    level = int(cell_level[cell])

    for axis in range(3):
        for j in range(2):
            nijk = ijk
            nijk[axis] += (2 * j - 1) << level

            nidx = AdaptiveNanogrid.find_cell(cell_grid, nijk, level_count, cell_level)
            if nidx != -1:
                n_level = cell_level[nidx]
                if n_level > (level + 1):
                    wp.atomic_add(refinement, nidx, 1)


@wp.kernel
def _count_ungraded_vertices(
    cell_grid: wp.uint64,
    cell_ijk: wp.array(dtype=wp.vec3i),
    cell_level: wp.array(dtype=wp.uint8),
    level_count: int,
    refinement: wp.array(dtype=wp.int32),
):
    cell = wp.tid()

    ijk = cell_ijk[cell]
    level = int(cell_level[cell])

    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                nijk = ijk + AdaptiveNanogrid.fine_ijk(wp.vec3i(i, j, k), level)

                nidx = AdaptiveNanogrid.find_cell(cell_grid, nijk, level_count, cell_level)
                if nidx != -1:
                    n_level = cell_level[nidx]
                    if n_level > (level + 1):
                        wp.atomic_add(refinement, nidx, 1)


@wp.kernel
def _count_graded_cells(
    refinement: wp.array(dtype=wp.int32),
    fine_count: wp.array(dtype=int),
):
    cell = wp.tid()
    if refinement[cell] > 0:
        wp.atomic_add(fine_count, 0, 7)


@wp.kernel
def _fill_graded_cells(
    coarse_ijk: wp.array(dtype=wp.vec3i),
    coarse_level: wp.array(dtype=wp.uint8),
    coarse_refinement: wp.array(dtype=wp.int32),
    fine_count: wp.array(dtype=int),
    fine_ijk: wp.array(dtype=wp.vec3i),
    fine_level: wp.array(dtype=wp.uint8),
):
    cell = wp.tid()
    ijk = coarse_ijk[cell]
    level = int(coarse_level[cell])
    refinement = wp.min(1, coarse_refinement[cell])

    count = wp.where(refinement > 0, 8, 1)
    offset = wp.atomic_sub(fine_count, 0, count) - count

    f_level = level - refinement
    for k in range(count):
        f_ijk = ijk + AdaptiveNanogrid.fine_ijk(wp.vec3i(k >> 2, (k & 2) >> 1, k & 1), f_level)
        fine_ijk[offset + k] = f_ijk
        fine_level[offset + k] = wp.uint8(f_level)


@integrand
def _sample_refinement(
    rng: wp.uint32,
    sample_count: int,
    ijk: wp.vec3i,
    domain: Domain,
    field: Field,
    coarse_grid: wp.uint64,
    cur_level: int,
    level_count: int,
):
    min_level = level_count

    scale = 1.0 / float(1 << (level_count - 1 - cur_level))
    uvw = wp.vec3(ijk) * scale + wp.vec3(0.5 * (scale - 1.0))

    for _ in range(sample_count):
        trial_uvw = uvw + wp.sample_unit_cube(rng) * scale
        pos = wp.volume_index_to_world(coarse_grid, trial_uvw)
        field_s = lookup(domain, pos)
        if field_s.element_index != NULL_ELEMENT_INDEX:
            sampled_level = wp.min(level_count - 1, int(wp.floor(field(field_s) * float(level_count))))
            if sampled_level >= 0:
                min_level = wp.min(sampled_level, min_level)

    return wp.where(min_level < level_count, cur_level - wp.clamp(min_level, 0, cur_level), -1)


@integrand
def _count_refined_voxels(
    s: Sample,
    domain: Domain,
    field: Field,
    sample_count: int,
    level_count: int,
    coarse_grid: wp.uint64,
    coarse_ijk: wp.array(dtype=wp.vec3i),
    coarse_level: wp.array(dtype=wp.uint8),
    coarse_refinement: wp.array(dtype=wp.int8),
    fine_count: wp.array(dtype=int),
):
    cell = s.qp_index

    ijk = coarse_ijk[cell]
    cur_level = int(coarse_level[cell])

    seed = (cur_level << 24) ^ (ijk[0] << 12) ^ (ijk[1] << 6) ^ ijk[2]
    rng = wp.rand_init(seed)

    refinement = _sample_refinement(rng, sample_count, ijk, domain, field, coarse_grid, cur_level, level_count)

    coarse_refinement[cell] = wp.int8(refinement)
    if refinement >= 0:
        wp.atomic_add(fine_count, 0, wp.where(refinement > 0, 8, 1))


@wp.kernel
def _fill_refined_voxels(
    coarse_ijk: wp.array(dtype=wp.vec3i),
    coarse_level: wp.array(dtype=wp.uint8),
    coarse_refinement: wp.array(dtype=wp.int8),
    fine_count: wp.array(dtype=int),
    fine_ijk: wp.array(dtype=wp.vec3i),
    fine_level: wp.array(dtype=wp.uint8),
):
    cell = wp.tid()
    ijk = coarse_ijk[cell]
    level = int(coarse_level[cell])
    refinement = wp.min(1, int(coarse_refinement[cell]))

    if refinement >= 0:
        count = wp.where(refinement > 0, 8, 1)
        offset = wp.atomic_sub(fine_count, 0, count) - count

        f_level = level - refinement
        for k in range(count):
            f_ijk = AdaptiveNanogrid.fine_ijk(ijk, refinement) + wp.vec3i(k >> 2, (k & 2) >> 1, k & 1)
            fine_ijk[offset + k] = f_ijk
            fine_level[offset + k] = wp.uint8(f_level)


@wp.kernel
def _adjust_refined_ijk(
    cell_ijk: wp.array(dtype=wp.vec3i),
    cell_level: wp.array(dtype=wp.uint8),
):
    cell = wp.tid()
    cell_ijk[cell] = AdaptiveNanogrid.fine_ijk(cell_ijk[cell], int(cell_level[cell]))


@wp.kernel
def _fill_refined_level(
    fine_grid: wp.uint64,
    cell_ijk: wp.array(dtype=wp.vec3i),
    cell_level: wp.array(dtype=wp.uint8),
    fine_level: wp.array(dtype=wp.uint8),
):
    cell = wp.tid()
    ijk = cell_ijk[cell]
    level = cell_level[cell]

    fine_level[wp.volume_lookup_index(fine_grid, ijk[0], ijk[1], ijk[2])] = level


@wp.kernel
def _fill_hierarchy_merged_ijk(
    level: int, level_offset: int, level_ijk: wp.array(dtype=wp.vec3i), merged_ijk: wp.array(dtype=wp.vec3i)
):
    cell = wp.tid()
    merged_ijk[cell + level_offset] = AdaptiveNanogrid.fine_ijk(level_ijk[cell], level)


@wp.kernel
def _fill_hierarchy_cell_level(
    level_count: int,
    grids: wp.array(dtype=wp.uint64),
    cells_ijk: wp.array(dtype=wp.vec3i),
    levels: wp.array(dtype=wp.uint8),
):
    cell = wp.tid()
    ijk = cells_ijk[cell]

    for k in range(level_count):
        idx = wp.volume_lookup_index(grids[k], ijk[0], ijk[1], ijk[2])
        if idx != -1:
            levels[cell] = wp.uint8(k)
            break
        ijk = AdaptiveNanogrid.coarse_ijk(ijk, 1)
