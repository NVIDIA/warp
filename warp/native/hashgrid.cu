// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "cuda_util.h"
#include "hashgrid.h"
#include "sort.h"

extern CUcontext get_current_context();

namespace wp {

template <typename Type> __global__ void compute_cell_indices(HashGrid_t<Type> grid, wp::array_t<vec_t<3, Type>> points)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < points.shape[0]) {
        grid.point_cells[tid] = hash_grid_index(grid, wp::index(points, tid));
        grid.point_ids[tid] = tid;
    }
}

template <typename Type>
__global__ void compute_point_keys(HashGrid_t<Type> grid, wp::array_t<vec_t<3, Type>> points, wp::array_t<int> groups)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < points.shape[0]) {
        const int cell = hash_grid_index(grid, wp::index(points, tid));
        grid.point_keys[tid] = hash_grid_point_key(cell, wp::index(groups, tid));
        grid.point_ids[tid] = tid;
    }
}

__global__ void extract_cells_from_keys(int* point_cells, const uint64_t* point_keys, int num_points)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_points)
        point_cells[tid] = (int)(point_keys[tid] >> 32);
}

__global__ void compute_cell_offsets(int* cell_starts, int* cell_ends, const int* point_cells, int num_points)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // compute cell start / end
    if (tid < num_points) {
        // scan the particle-cell array to find the start and end
        const int c = point_cells[tid];

        if (tid == 0)
            cell_starts[c] = 0;
        else {
            const int p = point_cells[tid - 1];

            if (c != p) {
                cell_starts[c] = tid;
                cell_ends[p] = tid;
            }
        }

        if (tid == num_points - 1) {
            cell_ends[c] = tid + 1;
        }
    }
}

template <typename Type>
void hash_grid_rebuild_device(
    const wp::HashGrid_t<Type>& grid, const wp::array_t<vec_t<3, Type>>& points, const wp::array_t<int>* groups
)
{
    ContextGuard guard(grid.context);

    const int num_points = points.shape[0];
    const int num_cells = hash_grid_num_cells(grid);

    if (groups) {
        // sort composite (cell, group) keys so each cell's points are contiguous per group
        wp_launch_device(WP_CURRENT_CONTEXT, (wp::compute_point_keys<Type>), num_points, (grid, points, *groups));

        radix_sort_pairs_device(WP_CURRENT_CONTEXT, grid.point_keys, grid.point_ids, num_points);

        wp_launch_device(
            WP_CURRENT_CONTEXT, wp::extract_cells_from_keys, num_points, (grid.point_cells, grid.point_keys, num_points)
        );
    } else {
        wp_launch_device(WP_CURRENT_CONTEXT, (wp::compute_cell_indices<Type>), num_points, (grid, points));

        radix_sort_pairs_device(WP_CURRENT_CONTEXT, grid.point_cells, grid.point_ids, num_points);
    }

    wp_memset_device(WP_CURRENT_CONTEXT, grid.cell_starts, 0, sizeof(int) * num_cells);
    wp_memset_device(WP_CURRENT_CONTEXT, grid.cell_ends, 0, sizeof(int) * num_cells);

    wp_launch_device(
        WP_CURRENT_CONTEXT, wp::compute_cell_offsets, num_points,
        (grid.cell_starts, grid.cell_ends, grid.point_cells, num_points)
    );
}

// Explicit template instantiations
template void
hash_grid_rebuild_device<half>(const HashGrid_t<half>&, const array_t<vec_t<3, half>>&, const array_t<int>*);
template void
hash_grid_rebuild_device<float>(const HashGrid_t<float>&, const array_t<vec_t<3, float>>&, const array_t<int>*);
template void
hash_grid_rebuild_device<double>(const HashGrid_t<double>&, const array_t<vec_t<3, double>>&, const array_t<int>*);


}  // namespace wp
