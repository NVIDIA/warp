#include "warp.h"
#include "hashgrid.h"
#include "sort.h"
#include "string.h"

uint64_t hash_grid_create_host(int dim_x, int dim_y, int dim_z)
{
    wp::HashGrid* grid = new wp::HashGrid();
    memset(grid, 0, sizeof(wp::HashGrid));
    
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
    wp::HashGrid* grid = (wp::HashGrid*)(id);

    free_host(grid->point_ids);
    free_host(grid->point_cells);
    free_host(grid->cell_starts);

    delete grid;
}

void hash_grid_resize_host(wp::HashGrid& grid, int num_points)
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
    wp::HashGrid* grid = (wp::HashGrid*)(id);

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
    radix_sort_pairs(grid->point_cells, grid->point_ids, num_points);

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


