/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "warp.h"
#include "cuda_util.h"
#include "hashgrid.h"
#include "sort.h"

namespace wp
{

__global__ void compute_cell_indices(HashGrid grid, wp::array_t<wp::vec3> points)
{
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < points.shape[0])
    {
        const vec3& point = wp::index(points, tid);
        grid.point_cells[tid] = hash_grid_index(grid, point);
        grid.point_ids[tid] = tid;
    }
}

__global__ void compute_cell_offsets(int* cell_starts, int* cell_ends, const int* point_cells, int num_points)
{
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;

    // compute cell start / end
    if (tid < num_points)
    {
		// scan the particle-cell array to find the start and end
		const int c = point_cells[tid];
		
		if (tid == 0)
			cell_starts[c] = 0;
		else
		{
			const int p = point_cells[tid-1];

			if (c != p)
			{
				cell_starts[c] = tid;
                cell_ends[p] = tid;
			}
		}

        if (tid == num_points - 1)
        {
            cell_ends[c] = tid + 1;
        }
	}    
}

void hash_grid_rebuild_device(const wp::HashGrid& grid, const wp::array_t<wp::vec3>& points)
{
    ContextGuard guard(grid.context);

    int num_points = points.shape[0];

    wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_cell_indices, num_points, (grid, points));

    radix_sort_pairs_device(WP_CURRENT_CONTEXT, grid.point_cells, grid.point_ids, num_points);

    const int num_cells = grid.dim_x * grid.dim_y * grid.dim_z;
    
    wp_memset_device(WP_CURRENT_CONTEXT, grid.cell_starts, 0, sizeof(int) * num_cells);
    wp_memset_device(WP_CURRENT_CONTEXT, grid.cell_ends, 0, sizeof(int) * num_cells);

    wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_cell_offsets, num_points, (grid.cell_starts, grid.cell_ends, grid.point_cells, num_points));
}


} // namespace wp
