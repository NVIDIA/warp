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
#include "string.h"

using namespace wp;

#include <map>

namespace 
{
    // host-side copy of mesh descriptors, maps GPU mesh address (id) to a CPU desc
    std::map<uint64_t, HashGrid> g_hash_grid_descriptors;

} // anonymous namespace


namespace wp
{

bool hash_grid_get_descriptor(uint64_t id, HashGrid& grid)
{
    const auto& iter = g_hash_grid_descriptors.find(id);
    if (iter == g_hash_grid_descriptors.end())
        return false;
    else
        grid = iter->second;
        return true;
}

void hash_grid_add_descriptor(uint64_t id, const HashGrid& grid)
{
    g_hash_grid_descriptors[id] = grid;
}

void hash_grid_rem_descriptor(uint64_t id)
{
    g_hash_grid_descriptors.erase(id);

}

// implemented in hashgrid.cu
void hash_grid_rebuild_device(const HashGrid& grid, const wp::array_t<wp::vec3>& points);

} // namespace wp


// host methods
uint64_t wp_hash_grid_create_host(int dim_x, int dim_y, int dim_z)
{
    HashGrid* grid = new HashGrid();
    memset(grid, 0, sizeof(HashGrid));
    
    grid->dim_x = dim_x;
    grid->dim_y = dim_y;
    grid->dim_z = dim_z;

    const int num_cells = dim_x*dim_y*dim_z;   
    grid->cell_starts = (int*)wp_alloc_host(num_cells*sizeof(int));
    grid->cell_ends = (int*)wp_alloc_host(num_cells*sizeof(int));

    return (uint64_t)(grid);
}

void wp_hash_grid_destroy_host(uint64_t id)
{
    HashGrid* grid = (HashGrid*)(id);

    wp_free_host(grid->point_ids);
    wp_free_host(grid->point_cells);
    wp_free_host(grid->cell_starts);
    wp_free_host(grid->cell_ends);

    delete grid;
}

void wp_hash_grid_reserve_host(uint64_t id, int num_points)
{
    HashGrid* grid = (HashGrid*)(id);

    if (num_points > grid->max_points)
    {
        wp_free_host(grid->point_cells);
        wp_free_host(grid->point_ids);
        
        const int num_to_alloc = num_points*3/2;
        grid->point_cells = (int*)wp_alloc_host(2*num_to_alloc*sizeof(int));  // *2 for auxiliary radix buffers
        grid->point_ids = (int*)wp_alloc_host(2*num_to_alloc*sizeof(int));    // *2 for auxiliary radix buffers

        grid->max_points = num_to_alloc;
    }

    grid->num_points = num_points;
}

void wp_hash_grid_update_host(uint64_t id, float cell_width, const wp::array_t<wp::vec3>* points)
{
    // Python enforces this, but let's be defensive anyways
    if (!points || points->ndim != 1)
    {
        fprintf(stderr, "Warp error: Invalid points array passed to %s\n", __FUNCTION__);
        return;
    }

    if (!id)
    {
        fprintf(stderr, "Warp error: Invalid grid passed to %s\n", __FUNCTION__);
        return;
    }

    HashGrid* grid = (HashGrid*)(id);
    int num_points = points->shape[0];

    wp_hash_grid_reserve_host(id, num_points);

    grid->cell_width = cell_width;
    grid->cell_width_inv = 1.0f / cell_width;

    // calculate cell for each position
    for (int i=0; i < num_points; ++i)
    {
        const vec3& point = wp::index(*points, i);
        grid->point_cells[i] = hash_grid_index(*grid, point);
        grid->point_ids[i] = i;
    }
    
    // sort indices
    radix_sort_pairs_host(grid->point_cells, grid->point_ids, num_points);

    const int num_cells = grid->dim_x * grid->dim_y * grid->dim_z;
    memset(grid->cell_starts, 0, sizeof(int) * num_cells);
    memset(grid->cell_ends, 0, sizeof(int) * num_cells);

    // compute cell start / end
    for (int i=0; i < num_points; ++i)
    {
		// scan the particle-cell array to find the start and end
		const int c = grid->point_cells[i];
		
		if (i == 0)
			grid->cell_starts[c] = 0;
		else
		{
			const int p = grid->point_cells[i-1];

			if (c != p)
			{
				grid->cell_starts[c] = i;
                grid->cell_ends[p] = i;
			}
		}

        if (i == num_points - 1)
        {
            grid->cell_ends[c] = i + 1;
        }
	}
}

// device methods
uint64_t wp_hash_grid_create_device(void* context, int dim_x, int dim_y, int dim_z)
{
    ContextGuard guard(context);

    HashGrid grid;
    memset(&grid, 0, sizeof(HashGrid));

    grid.context = context ? context : wp_cuda_context_get_current();

    grid.dim_x = dim_x;
    grid.dim_y = dim_y;
    grid.dim_z = dim_z;

    const int num_cells = dim_x*dim_y*dim_z;   
    grid.cell_starts = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, num_cells*sizeof(int));
    grid.cell_ends = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, num_cells*sizeof(int));

    // upload to device
    HashGrid* grid_device = (HashGrid*)(wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(HashGrid)));
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, grid_device, &grid, sizeof(HashGrid));

    uint64_t grid_id = (uint64_t)(grid_device);
    hash_grid_add_descriptor(grid_id, grid);

    return grid_id;
}

void wp_hash_grid_destroy_device(uint64_t id)
{
    HashGrid grid;
    if (hash_grid_get_descriptor(id, grid))
    {
        ContextGuard guard(grid.context);

        wp_free_device(WP_CURRENT_CONTEXT, grid.point_ids);
        wp_free_device(WP_CURRENT_CONTEXT, grid.point_cells);
        wp_free_device(WP_CURRENT_CONTEXT, grid.cell_starts);
        wp_free_device(WP_CURRENT_CONTEXT, grid.cell_ends);

        wp_free_device(WP_CURRENT_CONTEXT, (HashGrid*)id);
        
        hash_grid_rem_descriptor(id);
    }
}


void wp_hash_grid_reserve_device(uint64_t id, int num_points)
{
    HashGrid grid;

    if (hash_grid_get_descriptor(id, grid))
    {
        if (num_points > grid.max_points)
        {
            ContextGuard guard(grid.context);

            wp_free_device(WP_CURRENT_CONTEXT, grid.point_cells);
            wp_free_device(WP_CURRENT_CONTEXT, grid.point_ids);
            
            const int num_to_alloc = num_points*3/2;
            grid.point_cells = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, 2*num_to_alloc*sizeof(int));  // *2 for auxiliary radix buffers
            grid.point_ids = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, 2*num_to_alloc*sizeof(int));    // *2 for auxiliary radix buffers
            grid.max_points = num_to_alloc;

            // ensure we pre-size our sort routine to avoid
            // allocations during graph capture
            radix_sort_reserve(WP_CURRENT_CONTEXT, num_to_alloc);

            // update device side grid descriptor, todo: this is
            // slightly redundant since it is performed again
            // inside wp_hash_grid_update_device(), but since
            // reserve can be called from Python we need to make 
            // sure it is consistent
            wp_memcpy_h2d(WP_CURRENT_CONTEXT, (HashGrid*)id, &grid, sizeof(HashGrid));

            // update host side grid descriptor
            hash_grid_add_descriptor(id, grid);
        }
    }
}

void wp_hash_grid_update_device(uint64_t id, float cell_width, const wp::array_t<wp::vec3>* points)
{
    // Python enforces this, but let's be defensive anyways
    if (!points || points->ndim != 1)
    {
        fprintf(stderr, "Warp error: Invalid points array passed to %s\n", __FUNCTION__);
        return;
    }

    int num_points = points->shape[0];

    // ensure we have enough memory reserved for update
    // this must be done before retrieving the descriptor
    // below since it may update it
    wp_hash_grid_reserve_device(id, num_points);

    // host grid must be static so that we can
    // perform host->device memcpy from this variable
    // and have it safely recorded inside CUDA graphs
    static HashGrid grid;

    if (hash_grid_get_descriptor(id, grid))
    {
        ContextGuard guard(grid.context);

        grid.num_points = num_points;
        grid.cell_width = cell_width;
        grid.cell_width_inv = 1.0f / cell_width;

        hash_grid_rebuild_device(grid, *points);

        // update device side grid descriptor
        wp_memcpy_h2d(WP_CURRENT_CONTEXT, (HashGrid*)id, &grid, sizeof(HashGrid));

        // update host side grid descriptor
        hash_grid_add_descriptor(id, grid);
    }
}

#if !WP_ENABLE_CUDA

namespace wp
{

void hash_grid_rebuild_device(const HashGrid& grid, const wp::array_t<wp::vec3>& points)
{

}

} // namespace wp

#endif // !WP_ENABLE_CUDA