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

namespace wp {

// Note: Field order is important! Type-independent fields come first so that
// point_ids is at a consistent offset regardless of Type. This allows
// hash_grid_point_id to work without knowing the grid's scalar type.
template <typename Type> struct HashGrid_t {
    int* point_cells { nullptr };  // cell id of a point
    int* point_ids { nullptr };  // index to original point

    int* cell_starts { nullptr };  // start index of a range of indices belonging to a cell, dim_x*dim_y*dim_z in length
    int* cell_ends { nullptr };  // end index of a range of indices belonging to a cell, dim_x*dim_y*dim_z in length

    int dim_x;
    int dim_y;
    int dim_z;

    int num_points;
    int max_points;

    void* context { nullptr };

    // Type-dependent fields at end (different sizes for half/float/double)
    Type cell_width;
    Type cell_width_inv;
};

// Type aliases for backward compatibility and convenience
using HashGrid = HashGrid_t<float>;
using HashGridH = HashGrid_t<half>;
using HashGridD = HashGrid_t<double>;

// convert a virtual (world) cell coordinate to a physical one
template <typename Type> CUDA_CALLABLE inline int hash_grid_index(const HashGrid_t<Type>& grid, int x, int y, int z)
{
    // offset to ensure positive coordinates (means grid dim should be less than 4096^3)
    const int origin = 1 << 20;

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
    int cx = x % grid.dim_x;
    int cy = y % grid.dim_y;
    int cz = z % grid.dim_z;

    return cz * (grid.dim_x * grid.dim_y) + cy * grid.dim_x + cx;
}

template <typename Type> CUDA_CALLABLE inline int hash_grid_index(const HashGrid_t<Type>& grid, const vec_t<3, Type>& p)
{
    return hash_grid_index(
        grid, int(p[0] * grid.cell_width_inv), int(p[1] * grid.cell_width_inv), int(p[2] * grid.cell_width_inv)
    );
}

// stores state required to traverse neighboring cells of a point
template <typename Type> struct hash_grid_query_t {
    CUDA_CALLABLE hash_grid_query_t()
        : x_start(0)
        , y_start(0)
        , z_start(0)
        , x_end(0)
        , y_end(0)
        , z_end(0)
        , x(0)
        , y(0)
        , z(0)
        , cell(0)
        , cell_index(0)
        , cell_end(0)
        , current(0)
        , grid()
    {
    }

    // Required for adjoint computations.
    CUDA_CALLABLE inline hash_grid_query_t& operator+=(const hash_grid_query_t& other) { return *this; }

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
    int cell_index;  // offset in the current cell (index into cell_indices)
    int cell_end;  // index following the end of this cell

    int current;  // index of the current iterator value

    HashGrid_t<Type> grid;
};

// Type aliases for query structs
using hash_grid_query_f = hash_grid_query_t<float>;
using hash_grid_query_h = hash_grid_query_t<half>;
using hash_grid_query_d = hash_grid_query_t<double>;


template <typename Type>
CUDA_CALLABLE inline hash_grid_query_t<Type> hash_grid_query(uint64_t id, vec_t<3, Type> pos, Type radius)
{
    hash_grid_query_t<Type> query;

    query.grid = *(const HashGrid_t<Type>*)(id);

    // convert coordinate to grid cell indices
    Type cell_width_inv = query.grid.cell_width_inv;

    query.x_start = int((pos[0] - radius) * cell_width_inv);
    query.y_start = int((pos[1] - radius) * cell_width_inv);
    query.z_start = int((pos[2] - radius) * cell_width_inv);

    // do not want to visit any cells more than once, so limit large radius offset to one pass over each dimension
    query.x_end = min(int((pos[0] + radius) * cell_width_inv), query.x_start + query.grid.dim_x - 1);
    query.y_end = min(int((pos[1] + radius) * cell_width_inv), query.y_start + query.grid.dim_y - 1);
    query.z_end = min(int((pos[2] + radius) * cell_width_inv), query.z_start + query.grid.dim_z - 1);

    query.x = query.x_start;
    query.y = query.y_start;
    query.z = query.z_start;

    const int cell = hash_grid_index(query.grid, query.x, query.y, query.z);
    query.cell_index = query.grid.cell_starts[cell];
    query.cell_end = query.grid.cell_ends[cell];

    return query;
}


template <typename Type> CUDA_CALLABLE inline bool hash_grid_query_next(hash_grid_query_t<Type>& query, int& index)
{
    const HashGrid_t<Type>& grid = query.grid;
    if (!grid.point_cells)
        return false;

    while (1) {
        if (query.cell_index < query.cell_end) {
            // write output index
            index = grid.point_ids[query.cell_index++];
            return true;
        } else {
            query.x++;
            if (query.x > query.x_end) {
                query.x = query.x_start;
                query.y++;
            }

            if (query.y > query.y_end) {
                query.y = query.y_start;
                query.z++;
            }

            if (query.z > query.z_end) {
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

template <typename Type> CUDA_CALLABLE inline int iter_next(hash_grid_query_t<Type>& query) { return query.current; }

template <typename Type> CUDA_CALLABLE inline bool iter_cmp(hash_grid_query_t<Type>& query)
{
    bool finished = hash_grid_query_next(query, query.current);
    return finished;
}

template <typename Type> CUDA_CALLABLE inline hash_grid_query_t<Type> iter_reverse(const hash_grid_query_t<Type>& query)
{
    // can't reverse grid queries, users should not rely on neighbor ordering
    return query;
}

template <typename Type>
CUDA_CALLABLE inline void adj_iter_reverse(
    const hash_grid_query_t<Type>& query, hash_grid_query_t<Type>& adj_query, hash_grid_query_t<Type>& adj_ret
)
{
}


// hash_grid_point_id is not templated because it only accesses point_ids (int*)
// which is at the same offset in all HashGrid_t<Type> instantiations
CUDA_CALLABLE inline int hash_grid_point_id(uint64_t id, int& index)
{
    const HashGrid* grid = (const HashGrid*)(id);
    if (grid->point_ids == nullptr)
        return -1;
    return grid->point_ids[index];
}

template <typename Type>
CUDA_CALLABLE inline void adj_hash_grid_query(
    uint64_t id,
    vec_t<3, Type> pos,
    Type radius,
    uint64_t& adj_id,
    vec_t<3, Type>& adj_pos,
    Type& adj_radius,
    hash_grid_query_t<Type>& adj_res
)
{
}

template <typename Type>
CUDA_CALLABLE inline void adj_hash_grid_query_next(
    hash_grid_query_t<Type>& query, int& index, hash_grid_query_t<Type>& adj_query, int& adj_index, bool& adj_res
)
{
}

CUDA_CALLABLE inline void
adj_hash_grid_point_id(uint64_t id, int& index, uint64_t& adj_id, int& adj_index, int& adj_res)
{
}


}  // namespace wp
