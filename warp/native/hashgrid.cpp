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

#include <cstddef>

using namespace wp;

// Verify that type-independent fields are at the same offset in all HashGrid_t instantiations.
// This is required for hash_grid_point_id() to work without knowing the grid's scalar type.
static_assert(
    offsetof(HashGrid, point_cells) == offsetof(HashGridH, point_cells),
    "HashGrid point_cells offset mismatch between float and half"
);
static_assert(
    offsetof(HashGrid, point_cells) == offsetof(HashGridD, point_cells),
    "HashGrid point_cells offset mismatch between float and double"
);
static_assert(
    offsetof(HashGrid, point_ids) == offsetof(HashGridH, point_ids),
    "HashGrid point_ids offset mismatch between float and half"
);
static_assert(
    offsetof(HashGrid, point_ids) == offsetof(HashGridD, point_ids),
    "HashGrid point_ids offset mismatch between float and double"
);

#include <map>

namespace {
// host-side copy of hash grid descriptors, maps GPU hash grid address (id) to a CPU desc
// separate maps for each precision type
std::map<uint64_t, HashGrid> g_hash_grid_descriptors_f;
std::map<uint64_t, HashGridH> g_hash_grid_descriptors_h;
std::map<uint64_t, HashGridD> g_hash_grid_descriptors_d;

}  // anonymous namespace


namespace wp {

// Templated descriptor access functions
template <typename Type> std::map<uint64_t, HashGrid_t<Type>>& get_descriptor_map();

template <> std::map<uint64_t, HashGrid>& get_descriptor_map<float>() { return g_hash_grid_descriptors_f; }

template <> std::map<uint64_t, HashGridH>& get_descriptor_map<half>() { return g_hash_grid_descriptors_h; }

template <> std::map<uint64_t, HashGridD>& get_descriptor_map<double>() { return g_hash_grid_descriptors_d; }

template <typename Type> bool hash_grid_get_descriptor(uint64_t id, HashGrid_t<Type>& grid)
{
    auto& descriptors = get_descriptor_map<Type>();
    const auto& iter = descriptors.find(id);
    if (iter == descriptors.end())
        return false;
    else
        grid = iter->second;
    return true;
}

template <typename Type> void hash_grid_add_descriptor(uint64_t id, const HashGrid_t<Type>& grid)
{
    auto& descriptors = get_descriptor_map<Type>();
    descriptors[id] = grid;
}

template <typename Type> void hash_grid_rem_descriptor(uint64_t id)
{
    auto& descriptors = get_descriptor_map<Type>();
    descriptors.erase(id);
}

// implemented in hashgrid.cu
template <typename Type>
void hash_grid_rebuild_device(const HashGrid_t<Type>& grid, const wp::array_t<vec_t<3, Type>>& points);

}  // namespace wp


// =============================================================================
// Host methods - templated implementation
// =============================================================================

template <typename Type> uint64_t hash_grid_create_host_impl(int dim_x, int dim_y, int dim_z)
{
    HashGrid_t<Type>* grid = new HashGrid_t<Type>();
    memset(grid, 0, sizeof(HashGrid_t<Type>));

    grid->dim_x = dim_x;
    grid->dim_y = dim_y;
    grid->dim_z = dim_z;

    const int num_cells = dim_x * dim_y * dim_z;
    grid->cell_starts = (int*)wp_alloc_host(num_cells * sizeof(int));
    grid->cell_ends = (int*)wp_alloc_host(num_cells * sizeof(int));

    return (uint64_t)(grid);
}

template <typename Type> void hash_grid_destroy_host_impl(uint64_t id)
{
    HashGrid_t<Type>* grid = (HashGrid_t<Type>*)(id);

    wp_free_host(grid->point_ids);
    wp_free_host(grid->point_cells);
    wp_free_host(grid->cell_starts);
    wp_free_host(grid->cell_ends);

    delete grid;
}

template <typename Type> void hash_grid_reserve_host_impl(uint64_t id, int num_points)
{
    HashGrid_t<Type>* grid = (HashGrid_t<Type>*)(id);

    if (num_points > grid->max_points) {
        wp_free_host(grid->point_cells);
        wp_free_host(grid->point_ids);

        const int num_to_alloc = num_points * 3 / 2;
        grid->point_cells = (int*)wp_alloc_host(2 * num_to_alloc * sizeof(int));  // *2 for auxiliary radix buffers
        grid->point_ids = (int*)wp_alloc_host(2 * num_to_alloc * sizeof(int));  // *2 for auxiliary radix buffers

        grid->max_points = num_to_alloc;
    }

    grid->num_points = num_points;
}

template <typename Type>
void hash_grid_update_host_impl(uint64_t id, Type cell_width, const wp::array_t<vec_t<3, Type>>* points)
{
    // Python enforces this, but let's be defensive anyways
    if (!points || points->ndim != 1) {
        fprintf(stderr, "Warp error: Invalid points array passed to %s\n", __FUNCTION__);
        return;
    }

    if (!id) {
        fprintf(stderr, "Warp error: Invalid grid passed to %s\n", __FUNCTION__);
        return;
    }

    HashGrid_t<Type>* grid = (HashGrid_t<Type>*)(id);
    int num_points = points->shape[0];

    hash_grid_reserve_host_impl<Type>(id, num_points);

    grid->cell_width = cell_width;
    grid->cell_width_inv = Type(1) / cell_width;

    // calculate cell for each position
    for (int i = 0; i < num_points; ++i) {
        const vec_t<3, Type>& point = wp::index(*points, i);
        grid->point_cells[i] = hash_grid_index(*grid, point);
        grid->point_ids[i] = i;
    }

    // sort indices
    radix_sort_pairs_host(grid->point_cells, grid->point_ids, num_points);

    const int num_cells = grid->dim_x * grid->dim_y * grid->dim_z;
    memset(grid->cell_starts, 0, sizeof(int) * num_cells);
    memset(grid->cell_ends, 0, sizeof(int) * num_cells);

    // compute cell start / end
    for (int i = 0; i < num_points; ++i) {
        // scan the particle-cell array to find the start and end
        const int c = grid->point_cells[i];

        if (i == 0)
            grid->cell_starts[c] = 0;
        else {
            const int p = grid->point_cells[i - 1];

            if (c != p) {
                grid->cell_starts[c] = i;
                grid->cell_ends[p] = i;
            }
        }

        if (i == num_points - 1) {
            grid->cell_ends[c] = i + 1;
        }
    }
}

// =============================================================================
// Device methods - templated implementation
// =============================================================================

template <typename Type> uint64_t hash_grid_create_device_impl(void* context, int dim_x, int dim_y, int dim_z)
{
    ContextGuard guard(context);

    HashGrid_t<Type> grid;
    memset(&grid, 0, sizeof(HashGrid_t<Type>));

    grid.context = context ? context : wp_cuda_context_get_current();

    grid.dim_x = dim_x;
    grid.dim_y = dim_y;
    grid.dim_z = dim_z;

    const int num_cells = dim_x * dim_y * dim_z;
    grid.cell_starts = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, num_cells * sizeof(int));
    grid.cell_ends = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, num_cells * sizeof(int));

    // upload to device
    HashGrid_t<Type>* grid_device = (HashGrid_t<Type>*)(wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(HashGrid_t<Type>)));
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, grid_device, &grid, sizeof(HashGrid_t<Type>));

    uint64_t grid_id = (uint64_t)(grid_device);
    hash_grid_add_descriptor(grid_id, grid);

    return grid_id;
}

template <typename Type> void hash_grid_destroy_device_impl(uint64_t id)
{
    HashGrid_t<Type> grid;
    if (hash_grid_get_descriptor(id, grid)) {
        ContextGuard guard(grid.context);

        wp_free_device(WP_CURRENT_CONTEXT, grid.point_ids);
        wp_free_device(WP_CURRENT_CONTEXT, grid.point_cells);
        wp_free_device(WP_CURRENT_CONTEXT, grid.cell_starts);
        wp_free_device(WP_CURRENT_CONTEXT, grid.cell_ends);

        wp_free_device(WP_CURRENT_CONTEXT, (HashGrid_t<Type>*)id);

        hash_grid_rem_descriptor<Type>(id);
    }
}


template <typename Type> void hash_grid_reserve_device_impl(uint64_t id, int num_points)
{
    HashGrid_t<Type> grid;

    if (hash_grid_get_descriptor(id, grid)) {
        if (num_points > grid.max_points) {
            ContextGuard guard(grid.context);

            wp_free_device(WP_CURRENT_CONTEXT, grid.point_cells);
            wp_free_device(WP_CURRENT_CONTEXT, grid.point_ids);

            const int num_to_alloc = num_points * 3 / 2;
            grid.point_cells = (int*)wp_alloc_device(
                WP_CURRENT_CONTEXT, 2 * num_to_alloc * sizeof(int)
            );  // *2 for auxiliary radix buffers
            grid.point_ids = (int*)wp_alloc_device(
                WP_CURRENT_CONTEXT, 2 * num_to_alloc * sizeof(int)
            );  // *2 for auxiliary radix buffers
            grid.max_points = num_to_alloc;

            // ensure we pre-size our sort routine to avoid
            // allocations during graph capture
            radix_sort_reserve(WP_CURRENT_CONTEXT, num_to_alloc);

            // update device side grid descriptor, todo: this is
            // slightly redundant since it is performed again
            // inside hash_grid_update_device_impl(), but since
            // reserve can be called from Python we need to make
            // sure it is consistent
            wp_memcpy_h2d(WP_CURRENT_CONTEXT, (HashGrid_t<Type>*)id, &grid, sizeof(HashGrid_t<Type>));

            // update host side grid descriptor
            hash_grid_add_descriptor(id, grid);
        }
    }
}

// Note: For device update, we use a static local variable for CUDA graph compatibility.
// We need separate static variables for each type to avoid aliasing issues.
template <typename Type>
void hash_grid_update_device_impl(uint64_t id, Type cell_width, const wp::array_t<vec_t<3, Type>>* points)
{
    // Python enforces this, but let's be defensive anyways
    if (!points || points->ndim != 1) {
        fprintf(stderr, "Warp error: Invalid points array passed to %s\n", __FUNCTION__);
        return;
    }

    int num_points = points->shape[0];

    // ensure we have enough memory reserved for update
    // this must be done before retrieving the descriptor
    // below since it may update it
    hash_grid_reserve_device_impl<Type>(id, num_points);

    // host grid must be static so that we can
    // perform host->device memcpy from this variable
    // and have it safely recorded inside CUDA graphs
    static HashGrid_t<Type> grid;

    if (hash_grid_get_descriptor(id, grid)) {
        ContextGuard guard(grid.context);

        grid.num_points = num_points;
        grid.cell_width = cell_width;
        grid.cell_width_inv = Type(1) / cell_width;

        hash_grid_rebuild_device(grid, *points);

        // update device side grid descriptor
        wp_memcpy_h2d(WP_CURRENT_CONTEXT, (HashGrid_t<Type>*)id, &grid, sizeof(HashGrid_t<Type>));

        // update host side grid descriptor
        hash_grid_add_descriptor(id, grid);
    }
}

// =============================================================================
// Exported API
// =============================================================================

enum HashGridTypeId {
    HASH_GRID_TYPE_FLOAT16 = 0,
    HASH_GRID_TYPE_FLOAT32 = 1,
    HASH_GRID_TYPE_FLOAT64 = 2,
};

uint64_t wp_hash_grid_create_host(int type, int dim_x, int dim_y, int dim_z)
{
    switch (type) {
    case HASH_GRID_TYPE_FLOAT16:
        return hash_grid_create_host_impl<half>(dim_x, dim_y, dim_z);
    case HASH_GRID_TYPE_FLOAT32:
        return hash_grid_create_host_impl<float>(dim_x, dim_y, dim_z);
    case HASH_GRID_TYPE_FLOAT64:
        return hash_grid_create_host_impl<double>(dim_x, dim_y, dim_z);
    default:
        fprintf(stderr, "Warp error: Invalid hash grid type %d\n", type);
        return 0;
    }
}

void wp_hash_grid_destroy_host(uint64_t id, int type)
{
    switch (type) {
    case HASH_GRID_TYPE_FLOAT16:
        hash_grid_destroy_host_impl<half>(id);
        break;
    case HASH_GRID_TYPE_FLOAT32:
        hash_grid_destroy_host_impl<float>(id);
        break;
    case HASH_GRID_TYPE_FLOAT64:
        hash_grid_destroy_host_impl<double>(id);
        break;
    default:
        fprintf(stderr, "Warp error: Invalid hash grid type %d\n", type);
    }
}

void wp_hash_grid_update_host(uint64_t id, int type, double cell_width, const void* points)
{
    switch (type) {
    case HASH_GRID_TYPE_FLOAT16:
        hash_grid_update_host_impl<half>(id, half(cell_width), (const wp::array_t<wp::vec3h>*)points);
        break;
    case HASH_GRID_TYPE_FLOAT32:
        hash_grid_update_host_impl<float>(id, float(cell_width), (const wp::array_t<wp::vec3f>*)points);
        break;
    case HASH_GRID_TYPE_FLOAT64:
        hash_grid_update_host_impl<double>(id, cell_width, (const wp::array_t<wp::vec3d>*)points);
        break;
    default:
        fprintf(stderr, "Warp error: Invalid hash grid type %d\n", type);
    }
}

void wp_hash_grid_reserve_host(uint64_t id, int type, int num_points)
{
    switch (type) {
    case HASH_GRID_TYPE_FLOAT16:
        hash_grid_reserve_host_impl<half>(id, num_points);
        break;
    case HASH_GRID_TYPE_FLOAT32:
        hash_grid_reserve_host_impl<float>(id, num_points);
        break;
    case HASH_GRID_TYPE_FLOAT64:
        hash_grid_reserve_host_impl<double>(id, num_points);
        break;
    default:
        fprintf(stderr, "Warp error: Invalid hash grid type %d\n", type);
    }
}

uint64_t wp_hash_grid_create_device(void* context, int type, int dim_x, int dim_y, int dim_z)
{
    switch (type) {
    case HASH_GRID_TYPE_FLOAT16:
        return hash_grid_create_device_impl<half>(context, dim_x, dim_y, dim_z);
    case HASH_GRID_TYPE_FLOAT32:
        return hash_grid_create_device_impl<float>(context, dim_x, dim_y, dim_z);
    case HASH_GRID_TYPE_FLOAT64:
        return hash_grid_create_device_impl<double>(context, dim_x, dim_y, dim_z);
    default:
        fprintf(stderr, "Warp error: Invalid hash grid type %d\n", type);
        return 0;
    }
}

void wp_hash_grid_destroy_device(uint64_t id, int type)
{
    switch (type) {
    case HASH_GRID_TYPE_FLOAT16:
        hash_grid_destroy_device_impl<half>(id);
        break;
    case HASH_GRID_TYPE_FLOAT32:
        hash_grid_destroy_device_impl<float>(id);
        break;
    case HASH_GRID_TYPE_FLOAT64:
        hash_grid_destroy_device_impl<double>(id);
        break;
    default:
        fprintf(stderr, "Warp error: Invalid hash grid type %d\n", type);
    }
}

void wp_hash_grid_update_device(uint64_t id, int type, double cell_width, const void* points)
{
    switch (type) {
    case HASH_GRID_TYPE_FLOAT16:
        hash_grid_update_device_impl<half>(id, half(cell_width), (const wp::array_t<wp::vec3h>*)points);
        break;
    case HASH_GRID_TYPE_FLOAT32:
        hash_grid_update_device_impl<float>(id, float(cell_width), (const wp::array_t<wp::vec3f>*)points);
        break;
    case HASH_GRID_TYPE_FLOAT64:
        hash_grid_update_device_impl<double>(id, cell_width, (const wp::array_t<wp::vec3d>*)points);
        break;
    default:
        fprintf(stderr, "Warp error: Invalid hash grid type %d\n", type);
    }
}

void wp_hash_grid_reserve_device(uint64_t id, int type, int num_points)
{
    switch (type) {
    case HASH_GRID_TYPE_FLOAT16:
        hash_grid_reserve_device_impl<half>(id, num_points);
        break;
    case HASH_GRID_TYPE_FLOAT32:
        hash_grid_reserve_device_impl<float>(id, num_points);
        break;
    case HASH_GRID_TYPE_FLOAT64:
        hash_grid_reserve_device_impl<double>(id, num_points);
        break;
    default:
        fprintf(stderr, "Warp error: Invalid hash grid type %d\n", type);
    }
}

// =============================================================================
// Stub implementations when CUDA is disabled
// =============================================================================

#if !WP_ENABLE_CUDA

namespace wp {

template <typename Type>
void hash_grid_rebuild_device(const HashGrid_t<Type>& grid, const wp::array_t<vec_t<3, Type>>& points)
{
}

// Explicit instantiations
template void hash_grid_rebuild_device<half>(const HashGrid_t<half>&, const wp::array_t<vec_t<3, half>>&);
template void hash_grid_rebuild_device<float>(const HashGrid_t<float>&, const wp::array_t<vec_t<3, float>>&);
template void hash_grid_rebuild_device<double>(const HashGrid_t<double>&, const wp::array_t<vec_t<3, double>>&);

}  // namespace wp

#endif  // !WP_ENABLE_CUDA
