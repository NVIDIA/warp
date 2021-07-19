#pragma once

#include "builtin.h"

namespace wp
{

struct HashGrid
{
    float cell_width;
    float cell_width_inv;

    int* cell_starts;
    int* cell_indices;

    int dim_x;
    int dim_y;
    int dim_z;    
};

struct HashGridQuery
{
    HashGridQuery(const HashGrid& grid) : grid(grid) {} 

    int x_start;
    int y_start;
    int z_start;

    int x_end;
    int y_end;
    int z_end;

    int x;
    int y;
    int z;

    int cell_index;     // offset in the current cell (index into cell_indices)
    int cell_end;       // index following the end of this cell 
    
    const HashGrid& grid;
};

// convert a virtual (world) cell coordinate to a physical one
CUDA_CALLABLE inline int hash_grid_index(const HashGrid& grid, int x, int y, int z)
{
    assert(x >= 0);
    assert(y >= 0);
    assert(z >= 0);

    // compute physical cell
    int cx = x & (grid.dim_x-1);
    int cy = y & (grid.dim_y-1);
    int cz = z & (grid.dim_z-1);

    return cz*(grid.dim_x*grid.dim_y) + cy*grid.dim_x + cx;

}

CUDA_CALLABLE inline HashGridQuery hash_grid_query(uint64_t id, wp::vec3 pos, float radius)
{
    const HashGrid&  grid = (const HashGrid&)(id);
    
    HashGridQuery query(grid);

    // offset to ensure positive coordinates, todo: expose?
    const int origin = 65536;

    // convert coordinate to grid
    query.x_start = int((pos.x-radius)*grid.cell_width_inv) + origin;
    query.y_start = int((pos.y-radius)*grid.cell_width_inv) + origin;
    query.z_start = int((pos.z-radius)*grid.cell_width_inv) + origin;

    query.x_end = int((pos.x+radius)*grid.cell_width_inv) + origin;
    query.y_end = int((pos.y+radius)*grid.cell_width_inv) + origin;
    query.z_end = int((pos.z+radius)*grid.cell_width_inv) + origin;

    query.x = query.x_start;
    query.y = query.y_start;
    query.z = query.z_start;

    const int cell = hash_grid_index(query.grid, query.x, query.y, query.z);

    query.cell_index = query.grid.cell_starts[cell] - 1;    // -1 so that the iterator returns the first item 
    query.cell_end = query.grid.cell_starts[cell+1];
}


CUDA_CALLABLE inline bool hash_grid_query_next(HashGridQuery& query, int& index)
{
    query.cell_index++;

    if (query.cell_index >= query.cell_end)
    {
        query.x++;
        if (query.x > query.x_end)
        {
            query.x = query.x_start;
            query.y++;
        }

        if (query.y > query.y_end)
        {
            query.y = query.y_start;
            query.z++;
        }

        if (query.z > query.z_end)
        {
            // finished lookup grid
            return false;
        }

        // update cell pointers
        const int cell = hash_grid_index(query.grid, query.x, query.y, query.z);

        query.cell_index = query.grid.cell_starts[cell];
        query.cell_end = query.grid.cell_starts[cell+1];
    }

    // write output index
    index = query.grid.cell_indices[query.cell_index];

    return true;
}

// CUDA_CALLABLE inline int hash_grid_query_current(HashGridQuery& query)
// {
//     return query.grid.cell_indices[query.cell_index];
// }

// query = hash_grid_query(grid, p);

// while (hash_grid_query_next(query)):
// {
//     index = hash_grid_query_current(query)

//     # compute position

//     # compute kernel

//     # apply force

//     # bye-bye
// }


uint64_t hash_grid_create_host(int dim_x, int dim_y, int dim_z);
void hash_grid_destroy_host(uint64_t id);
void hash_grid_update_host(uint64_t id, float cell_width, const wp::vec3* positions);

uint64_t hash_grid_create_device(int dim_x, int dim_y, int dim_z);
void hash_grid_destroy_device(uint64_t id);
void hash_grid_update_device(uint64_t id, float cell_width, const wp::vec3* positions);

} // namespace wp
