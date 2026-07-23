// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace wp {

enum HashGridTypeId {
    HASH_GRID_TYPE_FLOAT16 = 0,
    HASH_GRID_TYPE_FLOAT32 = 1,
    HASH_GRID_TYPE_FLOAT64 = 2,
};

// Note: Field order is important! Type-independent fields come first so that
// point_ids is at a consistent offset regardless of Type. This allows
// hash_grid_point_id to work without knowing the grid's scalar type.
template <typename Type> struct HashGrid_t {
    int* point_cells = nullptr;  // cell id of a point
    int* point_ids = nullptr;  // index to original point
    uint64_t* point_keys = nullptr;  // sorted (cell, group) keys, allocated only for grouped builds

    int* cell_starts = nullptr;  // start index of a range of indices belonging to a cell, dim_x*dim_y*dim_z in length
    int* cell_ends = nullptr;  // end index of a range of indices belonging to a cell, dim_x*dim_y*dim_z in length

    int dim_x = 0;
    int dim_y = 0;
    int dim_z = 0;

    int num_points = 0;
    int max_points = 0;
    int max_keys = 0;  // capacity of point_keys
    int has_groups = 0;  // whether the most recent build was grouped

    void* context = nullptr;

    // Type-dependent fields at end (different sizes for half/float/double)
    Type cell_width = {};
    Type cell_width_inv = {};
};

// Type aliases for backward compatibility and convenience
using HashGrid = HashGrid_t<float>;
using HashGridH = HashGrid_t<half>;
using HashGridD = HashGrid_t<double>;

template <typename Type> CUDA_CALLABLE inline int hash_grid_num_cells(const HashGrid_t<Type>& grid)
{
    // dimensions are validated against overflow at grid creation (HashGrid._validate_cell_count)
    return grid.dim_x * grid.dim_y * grid.dim_z;
}

template <typename Type> CUDA_CALLABLE inline bool hash_grid_has_groups(const HashGrid_t<Type>& grid)
{
    return grid.point_keys != nullptr && grid.has_groups != 0;
}

// composite sort key: spatial cell in the high bits, group id bit pattern in the low bits
CUDA_CALLABLE inline uint64_t hash_grid_point_key(int cell, int group)
{
    return ((uint64_t)cell << 32) | (uint64_t)(uint32_t)group;
}

// first index in [lo, hi) whose key is >= key
CUDA_CALLABLE inline int hash_grid_lower_bound(const uint64_t* keys, int lo, int hi, uint64_t key)
{
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (keys[mid] < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

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
    // Use floor() to round toward negative infinity, not int() which truncates toward zero.
    // Without floor(), negative fractional coordinates map to the wrong cell
    // (e.g., -0.3 with cell_width=1.0: int(-0.3)=0 instead of floor(-0.3)=-1).
    return hash_grid_index(
        grid, int(floor(p[0] * grid.cell_width_inv)), int(floor(p[1] * grid.cell_width_inv)),
        int(floor(p[2] * grid.cell_width_inv))
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
        , group(0)
        , filter_by_group(false)
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

    int group;  // group id to filter by when filter_by_group is set
    bool filter_by_group;

    HashGrid_t<Type> grid;
};

// Type aliases for query structs
using hash_grid_query_f = hash_grid_query_t<float>;
using hash_grid_query_h = hash_grid_query_t<half>;
using hash_grid_query_d = hash_grid_query_t<double>;


template <typename Type> CUDA_CALLABLE inline void hash_grid_query_set_cell(hash_grid_query_t<Type>& query)
{
    const int cell = hash_grid_index(query.grid, query.x, query.y, query.z);
    int start = query.grid.cell_starts[cell];
    int end = query.grid.cell_ends[cell];

    if (query.filter_by_group && hash_grid_has_groups(query.grid)) {
        // the group's points form a contiguous key-sorted sub-range of the cell;
        // key + 1 stays bounded by `end` even when the group bits are 0xffffffff
        const uint64_t key = hash_grid_point_key(cell, query.group);
        start = hash_grid_lower_bound(query.grid.point_keys, start, end, key);
        end = hash_grid_lower_bound(query.grid.point_keys, start, end, key + 1);
    }

    query.cell_index = start;
    query.cell_end = end;
}

template <typename Type>
CUDA_CALLABLE inline hash_grid_query_t<Type>
hash_grid_query_impl(uint64_t id, vec_t<3, Type> pos, Type radius, int group, bool filter_by_group)
{
    hash_grid_query_t<Type> query;

    query.grid = *(const HashGrid_t<Type>*)(id);
    query.group = group;
    query.filter_by_group = filter_by_group;

    // Convert coordinate to grid cell indices using floor() (see hash_grid_index above)
    Type cell_width_inv = query.grid.cell_width_inv;

    query.x_start = int(floor((pos[0] - radius) * cell_width_inv));
    query.y_start = int(floor((pos[1] - radius) * cell_width_inv));
    query.z_start = int(floor((pos[2] - radius) * cell_width_inv));

    // do not want to visit any cells more than once, so limit large radius offset to one pass over each dimension
    query.x_end = min(int(floor((pos[0] + radius) * cell_width_inv)), query.x_start + query.grid.dim_x - 1);
    query.y_end = min(int(floor((pos[1] + radius) * cell_width_inv)), query.y_start + query.grid.dim_y - 1);
    query.z_end = min(int(floor((pos[2] + radius) * cell_width_inv)), query.z_start + query.grid.dim_z - 1);

    query.x = query.x_start;
    query.y = query.y_start;
    query.z = query.z_start;

    hash_grid_query_set_cell(query);

    return query;
}

template <typename Type>
CUDA_CALLABLE inline hash_grid_query_t<Type> hash_grid_query(uint64_t id, vec_t<3, Type> pos, Type radius, int group)
{
    return hash_grid_query_impl(id, pos, radius, group, true);
}

template <typename Type>
CUDA_CALLABLE inline hash_grid_query_t<Type> hash_grid_query(uint64_t id, vec_t<3, Type> pos, Type radius)
{
    return hash_grid_query_impl(id, pos, radius, 0, false);
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
            hash_grid_query_set_cell(query);
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

// hash_grid_point_id is not templated because it only accesses point_ids (int*)
// which is at the same offset in all HashGrid_t<Type> instantiations
CUDA_CALLABLE inline int hash_grid_point_id(uint64_t id, int& index)
{
    const HashGrid* grid = (const HashGrid*)(id);
    if (grid->point_ids == nullptr)
        return -1;
    return grid->point_ids[index];
}


}  // namespace wp
