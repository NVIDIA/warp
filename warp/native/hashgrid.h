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

#pragma once

namespace wp
{

struct HashGrid
{
    float cell_width;
    float cell_width_inv;

    int* point_cells{nullptr};   // cell id of a point
    int* point_ids{nullptr};     // index to original point
    
    int* cell_starts{nullptr};   // start index of a range of indices belonging to a cell, dim_x*dim_y*dim_z in length
    int* cell_ends{nullptr};     // end index of a range of indices belonging to a cell, dim_x*dim_y*dim_z in length

    int dim_x;
    int dim_y;
    int dim_z;

    int num_points;
    int max_points;

    void* context{nullptr};
};

// convert a virtual (world) cell coordinate to a physical one
CUDA_CALLABLE inline int hash_grid_index(const HashGrid& grid, int x, int y, int z)
{
    // offset to ensure positive coordinates (means grid dim should be less than 4096^3)
    const int origin = 1<<20;

    x += origin;
    y += origin;
    z += origin;

    assert(0 < x);
    assert(0 < y);
    assert(0 < z);

    // clamp in case any particles fall outside the guard region (-10^20 cell index)
    x = max(0, x);
    y = max(0, y);
    z = max(0, z);

    // compute physical cell (assume pow2 grid dims)
    // int cx = x & (grid.dim_x-1);
    // int cy = y & (grid.dim_y-1);
    // int cz = z & (grid.dim_z-1);

    // compute physical cell (arbitrary grid dims)
    int cx = x%grid.dim_x;
    int cy = y%grid.dim_y;
    int cz = z%grid.dim_z;

    return cz*(grid.dim_x*grid.dim_y) + cy*grid.dim_x + cx;
}

CUDA_CALLABLE inline int hash_grid_index(const HashGrid& grid, const vec3& p)
{
    return hash_grid_index(grid, 
                           int(p[0]*grid.cell_width_inv), 
                           int(p[1]*grid.cell_width_inv),
                           int(p[2]*grid.cell_width_inv));
}

// stores state required to traverse neighboring cells of a point
struct hash_grid_query_t
{
    CUDA_CALLABLE hash_grid_query_t()
        : x_start(0),
          y_start(0),
          z_start(0),
          x_end(0),
          y_end(0),
          z_end(0),
          x(0),
          y(0),
          z(0),
          cell(0),
          cell_index(0),
          cell_end(0),
          current(0),
          grid()
    {}

    // Required for adjoint computations.
    CUDA_CALLABLE inline hash_grid_query_t& operator+=(const hash_grid_query_t& other)
    {
        return *this;
    }

    int x_start;
    int y_start;
    int z_start;

    int x_end;
    int y_end;
    int z_end;

    int x;
    int y;
    int z;

    int cell;
    int cell_index;     // offset in the current cell (index into cell_indices)
    int cell_end;       // index following the end of this cell 
    
    int current;        // index of the current iterator value

    HashGrid grid;
};


CUDA_CALLABLE inline hash_grid_query_t hash_grid_query(uint64_t id, wp::vec3 pos, float radius)
{
    hash_grid_query_t query;

    query.grid = *(const HashGrid*)(id);

    // convert coordinate to grid
    query.x_start = int((pos[0]-radius)*query.grid.cell_width_inv);
    query.y_start = int((pos[1]-radius)*query.grid.cell_width_inv);
    query.z_start = int((pos[2]-radius)*query.grid.cell_width_inv);

    // do not want to visit any cells more than once, so limit large radius offset to one pass over each dimension
    query.x_end = min(int((pos[0]+radius)*query.grid.cell_width_inv), query.x_start + query.grid.dim_x-1);
    query.y_end = min(int((pos[1]+radius)*query.grid.cell_width_inv), query.y_start + query.grid.dim_y-1);
    query.z_end = min(int((pos[2]+radius)*query.grid.cell_width_inv), query.z_start + query.grid.dim_z-1);

    query.x = query.x_start;
    query.y = query.y_start;
    query.z = query.z_start;

    const int cell = hash_grid_index(query.grid, query.x, query.y, query.z);
    query.cell_index = query.grid.cell_starts[cell];
    query.cell_end = query.grid.cell_ends[cell];

    return query;
}


CUDA_CALLABLE inline bool hash_grid_query_next(hash_grid_query_t& query, int& index)
{
    const HashGrid& grid = query.grid;
    if (!grid.point_cells)
        return false;

    while (1)
    {
        if (query.cell_index < query.cell_end)
        {
            // write output index
            index = grid.point_ids[query.cell_index++];            
            return true;
        }
        else
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
            const int cell = hash_grid_index(grid, query.x, query.y, query.z);

            query.cell_index = grid.cell_starts[cell];
            query.cell_end = grid.cell_ends[cell];        
        }
    }
}

CUDA_CALLABLE inline int iter_next(hash_grid_query_t& query)
{
    return query.current;
}

CUDA_CALLABLE inline bool iter_cmp(hash_grid_query_t& query)
{
    bool finished = hash_grid_query_next(query, query.current);
    return finished;
}

CUDA_CALLABLE inline hash_grid_query_t iter_reverse(const hash_grid_query_t& query)
{
    // can't reverse grid queries, users should not rely on neighbor ordering
    return query;
}

CUDA_CALLABLE inline void adj_iter_reverse(const hash_grid_query_t& query, hash_grid_query_t& adj_query, hash_grid_query_t& adj_ret)
{
}



CUDA_CALLABLE inline int hash_grid_point_id(uint64_t id, int& index)
{
    const HashGrid* grid = (const HashGrid*)(id);
    if (grid->point_ids == nullptr)
        return -1;
    return grid->point_ids[index];
}

CUDA_CALLABLE inline void adj_hash_grid_query(uint64_t id, wp::vec3 pos, float radius, uint64_t& adj_id, wp::vec3& adj_pos, float& adj_radius, hash_grid_query_t& adj_res) {}
CUDA_CALLABLE inline void adj_hash_grid_query_next(hash_grid_query_t& query, int& index, hash_grid_query_t& adj_query, int& adj_index, bool& adj_res) {}
CUDA_CALLABLE inline void adj_hash_grid_point_id(uint64_t id, int& index, uint64_t & adj_id, int& adj_index, int& adj_res) {}


} // namespace wp
