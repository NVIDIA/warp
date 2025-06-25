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

#include "cuda_util.h"
#include "volume_builder.h"
#include "volume_impl.h"
#include "warp.h"

#include <map>

using namespace wp;

namespace
{

struct VolumeDesc
{
    // NanoVDB buffer either in device or host memory
    void* buffer;
    uint64_t size_in_bytes;
    bool owner; // whether the buffer should be deallocated when the volume is destroyed

    pnanovdb_grid_t grid_data;
    pnanovdb_tree_t tree_data;

    // Host-accessible version of the blind metadata (copy if GPU, alias if CPU)
    pnanovdb_gridblindmetadata_t* blind_metadata;

    // CUDA context for this volume (NULL if CPU)
    void* context;

    pnanovdb_buf_t as_pnano() const { return pnanovdb_make_buf(static_cast<uint32_t*>(buffer), size_in_bytes); }
};

// Host-side volume descriptors. Maps each CPU/GPU volume buffer address (id) to a CPU desc
std::map<uint64_t, VolumeDesc> g_volume_descriptors;

bool volume_get_descriptor(uint64_t id, const VolumeDesc*& volumeDesc)
{
    if (id == 0)
        return false;

    const auto& iter = g_volume_descriptors.find(id);
    if (iter == g_volume_descriptors.end())
        return false;
    else
        volumeDesc = &iter->second;
    return true;
}

bool volume_exists(const void* id)
{
    const VolumeDesc* volume;
    return volume_get_descriptor((uint64_t)id, volume);
}

void volume_add_descriptor(uint64_t id, VolumeDesc&& volumeDesc) { g_volume_descriptors[id] = std::move(volumeDesc); }

void volume_rem_descriptor(uint64_t id) { g_volume_descriptors.erase(id); }

void volume_set_map(nanovdb::Map& map, const float transform[9], const float translation[3])
{
    // Need to transpose as Map::set is transposing again
    const mat_t<3, 3, double> transpose(transform[0], transform[3], transform[6], transform[1], transform[4],
                                        transform[7], transform[2], transform[5], transform[8]);
    const mat_t<3, 3, double> inv = inverse(transpose);

    map.set(transpose.data, inv.data, translation);
}

} // anonymous namespace

// NB: buf must be a host pointer
uint64_t wp_volume_create_host(void* buf, uint64_t size, bool copy, bool owner)
{
    if (buf == nullptr || (size > 0 && size < sizeof(pnanovdb_grid_t) + sizeof(pnanovdb_tree_t)))
        return 0; // This cannot be a valid NanoVDB grid with data

    if (!copy && volume_exists(buf))
    {
        // descriptor already created for this volume
        return 0;
    }

    VolumeDesc volume;
    volume.context = NULL;

    wp_memcpy_h2h(&volume.grid_data, buf, sizeof(pnanovdb_grid_t));
    wp_memcpy_h2h(&volume.tree_data, (pnanovdb_grid_t*)buf + 1, sizeof(pnanovdb_tree_t));

    if (volume.grid_data.magic != PNANOVDB_MAGIC_NUMBER && volume.grid_data.magic != PNANOVDB_MAGIC_GRID)
        return 0;

    if (size == 0)
    {
        size = volume.grid_data.grid_size;
    }

    // Copy or alias buffer
    volume.size_in_bytes = size;
    if (copy)
    {
        volume.buffer = wp_alloc_host(size);
        wp_memcpy_h2h(volume.buffer, buf, size);
        volume.owner = true;
    }
    else
    {
        volume.buffer = buf;
        volume.owner = owner;
    }

    // Alias blind metadata
    volume.blind_metadata = reinterpret_cast<pnanovdb_gridblindmetadata_t*>(static_cast<uint8_t*>(volume.buffer) +
                                                                            volume.grid_data.blind_metadata_offset);

    uint64_t id = (uint64_t)volume.buffer;

    volume_add_descriptor(id, std::move(volume));

    return id;
}

// NB: buf must be a pointer on the same device
uint64_t wp_volume_create_device(void* context, void* buf, uint64_t size, bool copy, bool owner)
{
    if (buf == nullptr || (size > 0 && size < sizeof(pnanovdb_grid_t) + sizeof(pnanovdb_tree_t)))
        return 0; // This cannot be a valid NanoVDB grid with data

    if (!copy && volume_exists(buf))
    {
        // descriptor already created for this volume
        return 0;
    }

    ContextGuard guard(context);

    VolumeDesc volume;
    volume.context = context ? context : wp_cuda_context_get_current();

    wp_memcpy_d2h(WP_CURRENT_CONTEXT, &volume.grid_data, buf, sizeof(pnanovdb_grid_t));
    wp_memcpy_d2h(WP_CURRENT_CONTEXT, &volume.tree_data, (pnanovdb_grid_t*)buf + 1, sizeof(pnanovdb_tree_t));
    // no sync needed since the above copies are to pageable memory

    if (volume.grid_data.magic != PNANOVDB_MAGIC_NUMBER && volume.grid_data.magic != PNANOVDB_MAGIC_GRID)
        return 0;

    if (size == 0)
    {
        size = volume.grid_data.grid_size;
    }

    // Copy or alias data buffer
    volume.size_in_bytes = size;
    if (copy)
    {
        volume.buffer = wp_alloc_device(WP_CURRENT_CONTEXT, size);
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, volume.buffer, buf, size);
        volume.owner = true;
    }
    else
    {
        volume.buffer = buf;
        volume.owner = owner;
    }

    // Make blind metadata accessible on host
    const uint64_t blindmetadata_size = volume.grid_data.blind_metadata_count * sizeof(pnanovdb_gridblindmetadata_t);
    volume.blind_metadata = static_cast<pnanovdb_gridblindmetadata_t*>(wp_alloc_pinned(blindmetadata_size));
    wp_memcpy_d2h(WP_CURRENT_CONTEXT, volume.blind_metadata,
                  static_cast<uint8_t*>(volume.buffer) + volume.grid_data.blind_metadata_offset, blindmetadata_size);

    uint64_t id = (uint64_t)volume.buffer;
    volume_add_descriptor(id, std::move(volume));

    return id;
}

void wp_volume_get_buffer_info(uint64_t id, void** buf, uint64_t* size)
{
    *buf = 0;
    *size = 0;

    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume))
    {
        *buf = volume->buffer;
        *size = volume->size_in_bytes;
    }
}

void wp_volume_get_voxel_size(uint64_t id, float* dx, float* dy, float* dz)
{
    *dx = *dx = *dz = 0.0f;

    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume))
    {
        *dx = (float)volume->grid_data.voxel_size[0];
        *dy = (float)volume->grid_data.voxel_size[1];
        *dz = (float)volume->grid_data.voxel_size[2];
    }
}

void wp_volume_get_tile_and_voxel_count(uint64_t id, uint32_t& tile_count, uint64_t& voxel_count)
{
    tile_count = 0;
    voxel_count = 0;

    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume))
    {
        tile_count = volume->tree_data.node_count_leaf;

        const uint32_t grid_type = volume->grid_data.grid_type;

        switch (grid_type)
        {
        case PNANOVDB_GRID_TYPE_ONINDEX:
        case PNANOVDB_GRID_TYPE_ONINDEXMASK:
            // number of indexable voxels is number of active voxels
            voxel_count = volume->tree_data.voxel_count;
            break;
        default:
            // all leaf voxels are indexable
            voxel_count = uint64_t(tile_count) * PNANOVDB_LEAF_TABLE_COUNT;
        }
    }
}

const char* wp_volume_get_grid_info(uint64_t id, uint64_t* grid_size, uint32_t* grid_index, uint32_t* grid_count,
                                    float translation[3], float transform[9], char type_str[16])
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume))
    {
        const pnanovdb_grid_t& grid_data = volume->grid_data;
        *grid_count = grid_data.grid_count;
        *grid_index = grid_data.grid_index;
        *grid_size = grid_data.grid_size;

        memcpy(translation, grid_data.map.vecf, sizeof(grid_data.map.vecf));
        memcpy(transform, grid_data.map.matf, sizeof(grid_data.map.matf));

        nanovdb::toStr(type_str, static_cast<nanovdb::GridType>(grid_data.grid_type));
        return (const char*)grid_data.grid_name;
    }

    *grid_size = 0;
    *grid_index = 0;
    *grid_count = 0;
    type_str[0] = 0;

    return nullptr;
}

uint32_t wp_volume_get_blind_data_count(uint64_t id)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume))
    {
        return volume->grid_data.blind_metadata_count;
    }
    return 0;
}

const char* wp_volume_get_blind_data_info(uint64_t id, uint32_t data_index, void** buf, uint64_t* value_count,
                                          uint32_t* value_size, char type_str[16])
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume) && data_index < volume->grid_data.blind_metadata_count)
    {
        const pnanovdb_gridblindmetadata_t& metadata = volume->blind_metadata[data_index];
        *value_count = metadata.value_count;
        *value_size = metadata.value_size;

        nanovdb::toStr(type_str, static_cast<nanovdb::GridType>(metadata.data_type));
        *buf = static_cast<uint8_t*>(volume->buffer) + volume->grid_data.blind_metadata_offset +
               data_index * sizeof(pnanovdb_gridblindmetadata_t) + metadata.data_offset;
        return (const char*)metadata.name;
    }
    *buf = nullptr;
    *value_count = 0;
    *value_size = 0;
    type_str[0] = 0;
    return nullptr;
}

void wp_volume_get_tiles_host(uint64_t id, void* buf)
{
    static constexpr uint32_t MASK = (1u << 3u) - 1u; // mask for bit operations

    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume))
    {
        const uint32_t leaf_count = volume->tree_data.node_count_leaf;

        pnanovdb_coord_t* leaf_coords = static_cast<pnanovdb_coord_t*>(buf);

        const uint64_t first_leaf =
            (uint64_t)volume->buffer + sizeof(pnanovdb_grid_t) + volume->tree_data.node_offset_leaf;
        const uint32_t leaf_stride = PNANOVDB_GRID_TYPE_GET(volume->grid_data.grid_type, leaf_size);

        const pnanovdb_buf_t pnano_buf = volume->as_pnano();

        for (uint32_t i = 0; i < leaf_count; ++i)
        {
            pnanovdb_leaf_handle_t leaf = volume::get_leaf(pnano_buf, i);
            leaf_coords[i] = volume::leaf_origin(pnano_buf, leaf);
        }
    }
}

void wp_volume_get_voxels_host(uint64_t id, void* buf)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume))
    {
        uint32_t leaf_count;
        uint64_t voxel_count;
        wp_volume_get_tile_and_voxel_count(id, leaf_count, voxel_count);

        pnanovdb_coord_t* voxel_coords = static_cast<pnanovdb_coord_t*>(buf);

        const pnanovdb_buf_t pnano_buf = volume->as_pnano();
        for (uint32_t i = 0; i < leaf_count; ++i)
        {
            pnanovdb_leaf_handle_t leaf = volume::get_leaf(pnano_buf, i);
            pnanovdb_coord_t leaf_coords = volume::leaf_origin(pnano_buf, leaf);

            for (uint32_t n = 0; n < 512; ++n)
            {
                pnanovdb_coord_t loc_ijk = volume::leaf_offset_to_local_coord(n);
                pnanovdb_coord_t ijk = {
                    loc_ijk.x + leaf_coords.x,
                    loc_ijk.y + leaf_coords.y,
                    loc_ijk.z + leaf_coords.z,
                };

                const uint64_t index = volume::leaf_voxel_index(pnano_buf, i, ijk);
                if (index < voxel_count)
                {
                    voxel_coords[index] = ijk;
                }
            }
        }
    }
}

void wp_volume_destroy_host(uint64_t id)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume))
    {
        if (volume->owner)
        {
            wp_free_host(volume->buffer);
        }
        volume_rem_descriptor(id);
    }
}

void wp_volume_destroy_device(uint64_t id)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume))
    {
        ContextGuard guard(volume->context);
        if (volume->owner)
        {
            wp_free_device(WP_CURRENT_CONTEXT, volume->buffer);
        }
        wp_free_pinned(volume->blind_metadata);
        volume_rem_descriptor(id);
    }
}

#if WP_ENABLE_CUDA

uint64_t wp_volume_from_tiles_device(void* context, void* points, int num_points, float transform[9], float translation[3],
                                     bool points_in_world_space, const void* value_ptr, uint32_t value_size,
                                     const char* value_type)
{
    char gridTypeStr[12];

#define EXPAND_BUILDER_TYPE(type)                                                                                      \
    nanovdb::toStr(gridTypeStr, nanovdb::toGridType<type>());                                                          \
    if (strncmp(gridTypeStr, value_type, sizeof(gridTypeStr)) == 0)                                                    \
    {                                                                                                                  \
        BuildGridParams<type> params;                                                                                  \
        memcpy(&params.background_value, value_ptr, value_size);                                                       \
        volume_set_map(params.map, transform, translation);                                                            \
        size_t gridSize;                                                                                               \
        nanovdb::Grid<nanovdb::NanoTree<type>>* grid;                                                                  \
        build_grid_from_points(grid, gridSize, points, num_points, points_in_world_space, params);                     \
        return wp_volume_create_device(context, grid, gridSize, false, true);                                             \
    }

    WP_VOLUME_BUILDER_INSTANTIATE_TYPES
#undef EXPAND_BUILDER_TYPE

    return 0;
}

uint64_t wp_volume_index_from_tiles_device(void* context, void* points, int num_points, float transform[9],
                                           float translation[3], bool points_in_world_space)
{
    nanovdb::IndexGrid* grid;
    size_t gridSize;
    BuildGridParams<nanovdb::ValueIndex> params;
    volume_set_map(params.map, transform, translation);

    build_grid_from_points(grid, gridSize, points, num_points, points_in_world_space, params);

    return wp_volume_create_device(context, grid, gridSize, false, true);
}

uint64_t wp_volume_from_active_voxels_device(void* context, void* points, int num_points, float transform[9],
                                             float translation[3], bool points_in_world_space)
{
    nanovdb::OnIndexGrid* grid;
    size_t gridSize;
    BuildGridParams<nanovdb::ValueOnIndex> params;
    volume_set_map(params.map, transform, translation);

    build_grid_from_points(grid, gridSize, points, num_points, points_in_world_space, params);

    return wp_volume_create_device(context, grid, gridSize, false, true);
}

void launch_get_leaf_coords(void* context, const uint32_t leaf_count, pnanovdb_coord_t* leaf_coords,
                            pnanovdb_buf_t buf);
void launch_get_voxel_coords(void* context, const uint32_t leaf_count, const uint32_t voxel_count,
                             pnanovdb_coord_t* voxel_coords, pnanovdb_buf_t buf);

void wp_volume_get_tiles_device(uint64_t id, void* buf)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume))
    {
        const uint32_t leaf_count = volume->tree_data.node_count_leaf;

        pnanovdb_coord_t* leaf_coords = static_cast<pnanovdb_coord_t*>(buf);
        launch_get_leaf_coords(volume->context, leaf_count, leaf_coords, volume->as_pnano());
    }
}

void wp_volume_get_voxels_device(uint64_t id, void* buf)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume))
    {
        uint32_t leaf_count;
        uint64_t voxel_count;
        wp_volume_get_tile_and_voxel_count(id, leaf_count, voxel_count);

        pnanovdb_coord_t* voxel_coords = static_cast<pnanovdb_coord_t*>(buf);
        launch_get_voxel_coords(volume->context, leaf_count, voxel_count, voxel_coords, volume->as_pnano());
    }
}

#else
// stubs for non-CUDA platforms
uint64_t wp_volume_from_tiles_device(void* context, void* points, int num_points, float transform[9],
                                     float translation[3], bool points_in_world_space, const void* value_ptr, uint32_t value_size,
                                     const char* value_type)
{
    return 0;
}

uint64_t wp_volume_index_from_tiles_device(void* context, void* points, int num_points, float transform[9],
                                           float translation[3], bool points_in_world_space)
{
    return 0;
}

uint64_t wp_volume_from_active_voxels_device(void* context, void* points, int num_points, float transform[9],
                                             float translation[3], bool points_in_world_space)
{
    return 0;
}

void wp_volume_get_tiles_device(uint64_t id, void* buf) {}

void wp_volume_get_voxels_device(uint64_t id, void* buf) {}

#endif
