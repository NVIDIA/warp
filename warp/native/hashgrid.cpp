#include "warp.h"
#include "hashgrid.h"

uint64_t hash_grid_create_host(int dim_x, int dim_y, int dim_z)
{
    wp::HashGrid* grid = new wp::HashGrid();
    
    grid->cell_indices = (int*)alloc_host(dim_x*dim_y*dim_z*sizeof(int));
    grid->cell_starts = (int*)alloc_host(dim_x*dim_y*dim_z*sizeof(int));

    return (uint64_t)(grid);
}

void hash_grid_destroy_host(uint64_t id)
{
    wp::HashGrid* grid = (wp::HashGrid*)(id);

    free_host(grid->cell_indices);
    free_host(grid->cell_starts);

    delete grid;
}

void hash_grid_update_host(uint64_t id, float cell_width, const wp::vec3* positions)
{
    // calculate cell for each position

    // sort indices

    // compute cell start / end

    

}


