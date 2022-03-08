/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "warp.h"
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
void hash_grid_rebuild_device(const HashGrid& grid, const wp::vec3* points, int num_points);

} // namespace wp


// host methods
uint64_t hash_grid_create_host(int dim_x, int dim_y, int dim_z)
{
    HashGrid* grid = new HashGrid();
    memset(grid, 0, sizeof(HashGrid));
    
    grid->dim_x = dim_x;
    grid->dim_y = dim_y;
    grid->dim_z = dim_z;

    const int num_cells = dim_x*dim_y*dim_z;   
    grid->cell_starts = (int*)alloc_host(num_cells*sizeof(int));
    grid->cell_ends = (int*)alloc_host(num_cells*sizeof(int));

    return (uint64_t)(grid);
}

void hash_grid_destroy_host(uint64_t id)
{
    HashGrid* grid = (HashGrid*)(id);

    free_host(grid->point_ids);
    free_host(grid->point_cells);
    free_host(grid->cell_starts);

    delete grid;
}

void hash_grid_resize_host(HashGrid& grid, int num_points)
{
    if (num_points > grid.max_points)
    {
        free_host(grid.point_cells);
        free_host(grid.point_ids);
        
        const int num_to_alloc = num_points*3/2;
        grid.point_cells = (int*)alloc_host(2*num_to_alloc*sizeof(int));  // *2 for auxilliary radix buffers
        grid.point_ids = (int*)alloc_host(2*num_to_alloc*sizeof(int));    // *2 for auxilliary radix buffers

        grid.max_points = num_to_alloc;
    }

    grid.num_points = num_points;
}

void hash_grid_update_host(uint64_t id, float cell_width, const wp::vec3* points, int num_points)
{
    HashGrid* grid = (HashGrid*)(id);

    hash_grid_resize_host(*grid, num_points);

    grid->cell_width = cell_width;
    grid->cell_width_inv = 1.0f / cell_width;

    // calculate cell for each position
    for (int i=0; i < num_points; ++i)
    {
        grid->point_cells[i] = hash_grid_index(*grid, points[i]);
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
uint64_t hash_grid_create_device(int dim_x, int dim_y, int dim_z)
{
    HashGrid grid;
    memset(&grid, 0, sizeof(HashGrid));
    
    grid.dim_x = dim_x;
    grid.dim_y = dim_y;
    grid.dim_z = dim_z;

    const int num_cells = dim_x*dim_y*dim_z;   
    grid.cell_starts = (int*)alloc_device(num_cells*sizeof(int));
    grid.cell_ends = (int*)alloc_device(num_cells*sizeof(int));

    // upload to device
    HashGrid* grid_device = (HashGrid*)(alloc_device(sizeof(HashGrid)));
    memcpy_h2d(grid_device, &grid, sizeof(HashGrid));

    uint64_t grid_id = (uint64_t)(grid_device);
    hash_grid_add_descriptor(grid_id, grid);

    return grid_id;
}

void hash_grid_destroy_device(uint64_t id)
{
    HashGrid grid;
    if (hash_grid_get_descriptor(id, grid))
    {
        free_device(grid.point_ids);
        free_device(grid.point_cells);
        free_device(grid.cell_starts);

        free_device((HashGrid*)id);
        
        hash_grid_rem_descriptor(id);
    }
}


void hash_grid_update_device(uint64_t id, float cell_width, const wp::vec3* points, int num_points)
{
    // host grid must be static so that we can
    // perform host->device memcpy from this var
    // and have it safely recorded inside CUDA graphs
    static HashGrid grid;

    if (hash_grid_get_descriptor(id, grid))
    {
        if (num_points > grid.max_points)
        {
            free_device(grid.point_cells);
            free_device(grid.point_ids);
            
            const int num_to_alloc = num_points*3/2;
            grid.point_cells = (int*)alloc_device(2*num_to_alloc*sizeof(int));  // *2 for auxilliary radix buffers
            grid.point_ids = (int*)alloc_device(2*num_to_alloc*sizeof(int));    // *2 for auxilliary radix buffers

            grid.max_points = num_to_alloc;
        }

        grid.num_points = num_points;

        grid.cell_width = cell_width;
        grid.cell_width_inv = 1.0f / cell_width;

        hash_grid_rebuild_device(grid, points, num_points);

        // update device side grid descriptor
        memcpy_h2d((HashGrid*)id, &grid, sizeof(HashGrid));

        // update host side grid descriptor
        hash_grid_add_descriptor(id, grid);
    }
}

#if __APPLE__

namespace wp
{

void hash_grid_rebuild_device(const HashGrid& grid, const wp::vec3* points, int num_points)
{

}

} // namespace wp

#endif // __APPLE__