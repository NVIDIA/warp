// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "volume.h"

#include <climits>
#include <cstring>

#include <nanovdb/NanoVDB.h>

#define WP_VOLUME_BUILDER_INSTANTIATE_TYPES                                                                            \
    EXPAND_BUILDER_TYPE(int32_t)                                                                                       \
    EXPAND_BUILDER_TYPE(uint32_t)                                                                                      \
    EXPAND_BUILDER_TYPE(int64_t)                                                                                       \
    EXPAND_BUILDER_TYPE(float)                                                                                         \
    EXPAND_BUILDER_TYPE(double)                                                                                        \
    EXPAND_BUILDER_TYPE(nanovdb::Vec3f)                                                                                \
    EXPAND_BUILDER_TYPE(nanovdb::Vec3d)                                                                                \
    EXPAND_BUILDER_TYPE(nanovdb::Vec4f)

template <typename BuildT> struct BuildGridParams {
    nanovdb::Map map;
    BuildT background_value { 0 };
    char name[256] = "";
};

template <> struct BuildGridParams<nanovdb::ValueIndex> {
    nanovdb::Map map;
    nanovdb::ValueIndex background_value;
    char name[256] = "";
};

template <> struct BuildGridParams<nanovdb::ValueOnIndex> {
    nanovdb::Map map;
    double voxel_size = 1.0;
    char name[256] = "";
};

template <typename BuildT>
void allocate_grid_from_tiles(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const BuildGridParams<BuildT>& params
);

template <typename BuildT>
void allocate_grid_from_tiles_host(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const BuildGridParams<BuildT>& params
);

void allocate_grid_from_active_voxels(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const BuildGridParams<nanovdb::ValueOnIndex>& params
);

void allocate_grid_from_active_voxels_host(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const BuildGridParams<nanovdb::ValueOnIndex>& params
);

enum VolumeRebuildStatus : uint32_t {
    WP_VOLUME_REBUILD_SUCCESS = 0u,
    WP_VOLUME_REBUILD_LEAF_CAPACITY_EXCEEDED = 1u << 0u,
    WP_VOLUME_REBUILD_LOWER_CAPACITY_EXCEEDED = 1u << 1u,
    WP_VOLUME_REBUILD_UPPER_CAPACITY_EXCEEDED = 1u << 2u,
    WP_VOLUME_REBUILD_VOXEL_CAPACITY_EXCEEDED = 1u << 3u,
    WP_VOLUME_REBUILD_INVALID_INPUT = 1u << 4u,
};

struct VolumeRebuildCapacities {
    uint32_t leaf_count = 0;
    uint32_t lower_count = 0;
    uint32_t upper_count = 0;
    uint64_t voxel_count = 0;
};

namespace wp::volume_builder::internal {

static constexpr uint64_t REBUILD_INVALID_KEY = ~uint64_t(0);

enum RebuildCountSlot : uint32_t {
    REBUILD_COUNT_LEAF = 0,
    REBUILD_COUNT_LOWER = 1,
    REBUILD_COUNT_UPPER = 2,
    REBUILD_COUNT_VOXEL = 3,
    REBUILD_COUNT_SLOT_COUNT = 4,
};

struct RebuildGridDataBase {
    pnanovdb_buf_t buf;
    uint64_t size;
    uint64_t grid, tree, root, upper, lower, leaf;
    uint32_t grid_type;
    uint32_t grid_class;
    uint32_t value_size;
    uint32_t background_value[8];

    VolumeRebuildCapacities capacities;
    nanovdb::Map map;

    CUDA_CALLABLE pnanovdb_address_t address(uint64_t byte_offset) const
    {
        pnanovdb_address_t addr;
        addr.byte_offset = byte_offset;
        return addr;
    }

    CUDA_CALLABLE pnanovdb_grid_handle_t getGrid() const { return { address(grid) }; }
    CUDA_CALLABLE pnanovdb_tree_handle_t getTree() const { return { address(tree) }; }
    CUDA_CALLABLE pnanovdb_root_handle_t getRoot() const { return { address(root) }; }
    CUDA_CALLABLE pnanovdb_upper_handle_t getUpper(uint32_t i) const
    {
        return { address(upper + uint64_t(i) * PNANOVDB_GRID_TYPE_GET(grid_type, upper_size)) };
    }
    CUDA_CALLABLE pnanovdb_lower_handle_t getLower(uint32_t i) const
    {
        return { address(lower + uint64_t(i) * PNANOVDB_GRID_TYPE_GET(grid_type, lower_size)) };
    }
    CUDA_CALLABLE pnanovdb_leaf_handle_t getLeaf(uint32_t i) const
    {
        return { address(leaf + uint64_t(i) * PNANOVDB_GRID_TYPE_GET(grid_type, leaf_size)) };
    }
};

CUDA_CALLABLE inline pnanovdb_coord_t rebuild_make_pnano_coord(const nanovdb::Coord& coord)
{
    return { coord[0], coord[1], coord[2] };
}

CUDA_CALLABLE inline nanovdb::Coord rebuild_make_coord(const pnanovdb_coord_t& coord)
{
    return nanovdb::Coord(coord.x, coord.y, coord.z);
}

template <typename DataT> CUDA_CALLABLE inline uint8_t* rebuild_byte_ptr(const DataT& data, pnanovdb_address_t address)
{
    return reinterpret_cast<uint8_t*>(data.buf.data) + address.byte_offset;
}

template <typename DataT>
CUDA_CALLABLE inline void rebuild_zero_bytes(const DataT& data, pnanovdb_address_t address, uint32_t byte_count)
{
    uint8_t* dst = rebuild_byte_ptr(data, address);
    for (uint32_t i = 0; i < byte_count; ++i) {
        dst[i] = 0u;
    }
}

template <typename DataT>
CUDA_CALLABLE inline void
rebuild_write_bytes(const DataT& data, pnanovdb_address_t address, const uint32_t* words, uint32_t byte_count)
{
    uint8_t* dst = rebuild_byte_ptr(data, address);
    const uint8_t* src = reinterpret_cast<const uint8_t*>(words);
    for (uint32_t i = 0; i < byte_count; ++i) {
        dst[i] = src[i];
    }
}

template <typename DataT>
CUDA_CALLABLE inline void rebuild_write_background(const DataT& data, pnanovdb_address_t address)
{
    rebuild_write_bytes(data, address, data.background_value, data.value_size);
}

CUDA_CALLABLE inline bool rebuild_is_index_grid(uint32_t grid_type)
{
    return grid_type == PNANOVDB_GRID_TYPE_INDEX || grid_type == PNANOVDB_GRID_TYPE_ONINDEX;
}

CUDA_CALLABLE inline bool rebuild_is_onindex_grid(uint32_t grid_type)
{
    return grid_type == PNANOVDB_GRID_TYPE_ONINDEX;
}

CUDA_CALLABLE inline bool rebuild_is_regular_value_grid(uint32_t grid_type)
{
    return !rebuild_is_index_grid(grid_type);
}

template <typename DataT>
CUDA_CALLABLE inline pnanovdb_address_t
rebuild_leaf_index_offset_address(const DataT& data, pnanovdb_leaf_handle_t leaf)
{
    return pnanovdb_address_offset(leaf.address, PNANOVDB_GRID_TYPE_GET(data.grid_type, leaf_off_table));
}

template <typename DataT>
CUDA_CALLABLE inline pnanovdb_address_t
rebuild_leaf_index_prefix_address(const DataT& data, pnanovdb_leaf_handle_t leaf)
{
    return pnanovdb_address_offset(leaf.address, PNANOVDB_GRID_TYPE_GET(data.grid_type, leaf_off_table) + 8u);
}

template <typename DataT>
CUDA_CALLABLE inline void
rebuild_clear_mask_words(const DataT& data, pnanovdb_address_t node, uint32_t mask_offset, uint32_t word_count)
{
    for (uint32_t i = 0; i < word_count; ++i) {
        pnanovdb_write_uint64(data.buf, pnanovdb_address_offset(node, mask_offset + 8u * i), 0u);
    }
}

CUDA_CALLABLE inline pnanovdb_coord_t rebuild_invalid_bbox_min() { return { INT_MAX, INT_MAX, INT_MAX }; }

CUDA_CALLABLE inline pnanovdb_coord_t rebuild_invalid_bbox_max() { return { INT_MIN, INT_MIN, INT_MIN }; }

template <typename DataT>
CUDA_CALLABLE inline void
rebuild_set_bbox(const DataT& data, pnanovdb_address_t node, pnanovdb_coord_t min, pnanovdb_coord_t max)
{
    pnanovdb_write_coord(data.buf, pnanovdb_address_offset(node, 0u), PNANOVDB_REF(min));
    pnanovdb_write_coord(data.buf, pnanovdb_address_offset(node, 12u), PNANOVDB_REF(max));
}

template <typename DataT> CUDA_CALLABLE inline void rebuild_set_invalid_bbox(const DataT& data, pnanovdb_address_t node)
{
    rebuild_set_bbox(data, node, rebuild_invalid_bbox_min(), rebuild_invalid_bbox_max());
}

CUDA_CALLABLE inline uint32_t rebuild_find_lowest_on(uint64_t word)
{
    if (word == 0u) {
        return 64u;
    }
#if defined(__CUDA_ARCH__)
    return uint32_t(__ffsll(word) - 1);
#else
    for (uint32_t i = 0; i < 64u; ++i) {
        if (word & (uint64_t(1) << i)) {
            return i;
        }
    }
    return 64u;
#endif
}

CUDA_CALLABLE inline uint32_t rebuild_find_highest_on(uint64_t word)
{
    if (word == 0u) {
        return 64u;
    }
#if defined(__CUDA_ARCH__)
    return 63u - uint32_t(__clzll(word));
#else
    for (uint32_t i = 64u; i > 0u; --i) {
        if (word & (uint64_t(1) << (i - 1u))) {
            return i - 1u;
        }
    }
    return 64u;
#endif
}

CUDA_CALLABLE inline pnanovdb_coord_t rebuild_leaf_bbox_max(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf)
{
    const pnanovdb_coord_t min = pnanovdb_leaf_get_bbox_min(buf, leaf);
    const uint32_t bbox_dif_and_flags = pnanovdb_leaf_get_bbox_dif_and_flags(buf, leaf);
    return {
        min.x + int32_t(bbox_dif_and_flags & 0xffu),
        min.y + int32_t((bbox_dif_and_flags >> 8u) & 0xffu),
        min.z + int32_t((bbox_dif_and_flags >> 16u) & 0xffu),
    };
}

template <typename DataT>
CUDA_CALLABLE inline void rebuild_update_leaf_bbox(const DataT& data, pnanovdb_leaf_handle_t leaf)
{
    uint64_t union_word = 0u;
    uint32_t x_min = 8u;
    uint32_t x_max = 8u;
    for (uint32_t i = 0; i < 8u; ++i) {
        const uint64_t word = pnanovdb_read_uint64(
            data.buf, pnanovdb_address_offset(leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK + 8u * i)
        );
        if (word) {
            union_word |= word;
            if (x_min == 8u) {
                x_min = i;
            }
            x_max = i;
        }
    }

    const uint32_t bbox_dif_and_flags = pnanovdb_leaf_get_bbox_dif_and_flags(data.buf, leaf);
    uint32_t flags = bbox_dif_and_flags >> 24u;
    if (!union_word) {
        pnanovdb_leaf_set_bbox_dif_and_flags(data.buf, leaf, (flags & ~2u) << 24u);
        return;
    }

    pnanovdb_coord_t min = pnanovdb_leaf_get_bbox_min(data.buf, leaf);
    min.x = (min.x & ~7) + int32_t(x_min);
    min.y = (min.y & ~7) + int32_t(rebuild_find_lowest_on(union_word) >> 3u);
    const uint32_t word32 = uint32_t(union_word) | uint32_t(union_word >> 32u);
    const uint32_t word16 = (word32 & 0xffffu) | (word32 >> 16u);
    const uint32_t byte = (word16 & 0xffu) | (word16 >> 8u);
    min.z = (min.z & ~7) + int32_t(rebuild_find_lowest_on(byte));
    pnanovdb_leaf_set_bbox_min(data.buf, leaf, PNANOVDB_REF(min));

    const uint32_t x_dif = x_max - x_min;
    const uint32_t y_dif = (rebuild_find_highest_on(union_word) >> 3u) - (rebuild_find_lowest_on(union_word) >> 3u);
    const uint32_t z_dif = rebuild_find_highest_on(byte) - rebuild_find_lowest_on(byte);
    flags |= 2u;
    pnanovdb_leaf_set_bbox_dif_and_flags(data.buf, leaf, x_dif | (y_dif << 8u) | (z_dif << 16u) | (flags << 24u));
}

CUDA_CALLABLE inline uint64_t rebuild_upper_key_from_coord(const nanovdb::Coord& ijk)
{
    // Match NanoVDB's upper-root-tile key ordering.
    static constexpr int64_t kOffset = int64_t(1) << 31;
    return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12))
        | (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21)
        | (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42);
}

CUDA_CALLABLE inline nanovdb::Coord rebuild_upper_key_to_coord(uint64_t key)
{
    // Match NanoVDB's upper-node origin decoding.
    static constexpr int64_t kOffset = int64_t(1) << 31;
    static constexpr uint64_t mask = (uint64_t(1) << 21) - 1u;
    return nanovdb::Coord(
        int(int64_t(((key >> 42) & mask) << 12) - kOffset), int(int64_t(((key >> 21) & mask) << 12) - kOffset),
        int(int64_t((key & mask) << 12) - kOffset)
    );
}

CUDA_CALLABLE inline uint64_t rebuild_hierarchy_key(uint32_t upper_id, const nanovdb::Coord& ijk)
{
    const pnanovdb_coord_t coord = rebuild_make_pnano_coord(ijk);
    return (uint64_t(upper_id) << 36) | (uint64_t(pnanovdb_upper_coord_to_offset(PNANOVDB_REF(coord))) << 21)
        | (uint64_t(pnanovdb_lower_coord_to_offset(PNANOVDB_REF(coord))) << 9)
        | uint64_t(pnanovdb_leaf_coord_to_offset(PNANOVDB_REF(coord)));
}

template <typename BuildT> size_t rebuildable_grid_size(const VolumeRebuildCapacities& capacities)
{
    uint64_t offset = 0;
    offset += nanovdb::NanoGrid<BuildT>::memUsage();
    offset += nanovdb::NanoTree<BuildT>::memUsage();
    offset += nanovdb::NanoRoot<BuildT>::memUsage(capacities.upper_count);
    offset += nanovdb::NanoUpper<BuildT>::memUsage() * uint64_t(capacities.upper_count);
    offset += nanovdb::NanoLower<BuildT>::memUsage() * uint64_t(capacities.lower_count);
    offset += nanovdb::NanoLeaf<BuildT>::DataType::memUsage() * uint64_t(capacities.leaf_count);
    return size_t(offset);
}

template <typename BuildT, typename DataT>
DataT make_rebuild_data_base(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>* grid,
    size_t grid_size,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params
)
{
    DataT data = {};
    data.buf
        = pnanovdb_make_buf(reinterpret_cast<uint32_t*>(grid), (grid_size + sizeof(uint32_t) - 1u) / sizeof(uint32_t));
    data.size = grid_size;
    data.grid = 0;
    data.tree = nanovdb::NanoGrid<BuildT>::memUsage();
    data.root = data.tree + nanovdb::NanoTree<BuildT>::memUsage();
    data.upper = data.root + nanovdb::NanoRoot<BuildT>::memUsage(capacities.upper_count);
    data.lower = data.upper + nanovdb::NanoUpper<BuildT>::memUsage() * uint64_t(capacities.upper_count);
    data.leaf = data.lower + nanovdb::NanoLower<BuildT>::memUsage() * uint64_t(capacities.lower_count);
    data.grid_type = uint32_t(nanovdb::toGridType<BuildT>());
    data.grid_class = nanovdb::BuildTraits<BuildT>::is_index ? uint32_t(nanovdb::GridClass::IndexGrid)
                                                             : uint32_t(nanovdb::GridClass::Unknown);
    data.value_size = nanovdb::BuildTraits<BuildT>::is_index ? uint32_t(sizeof(uint64_t)) : uint32_t(sizeof(BuildT));
    data.capacities = capacities;
    data.map = params.map;
    if constexpr (!nanovdb::BuildTraits<BuildT>::is_index) {
        static_assert(sizeof(BuildT) <= sizeof(data.background_value));
        std::memcpy(data.background_value, &params.background_value, sizeof(BuildT));
    }
    return data;
}

}  // namespace wp::volume_builder::internal

template <typename BuildT>
void allocate_rebuildable_grid_from_tiles(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params,
    uint32_t* status
);

template <typename BuildT>
void allocate_rebuildable_grid_from_tiles_host(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params,
    uint32_t* status
);

template <typename BuildT>
void rebuild_grid_from_tiles(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>* grid,
    size_t grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params,
    uint32_t* status
);

template <typename BuildT>
void rebuild_grid_from_tiles_host(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>* grid,
    size_t grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params,
    uint32_t* status
);

void allocate_rebuildable_grid_from_active_voxels(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<nanovdb::ValueOnIndex>& params,
    uint32_t* status
);

void allocate_rebuildable_grid_from_active_voxels_host(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<nanovdb::ValueOnIndex>& params,
    uint32_t* status
);

void rebuild_grid_from_active_voxels(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>* grid,
    size_t grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<nanovdb::ValueOnIndex>& params,
    uint32_t* status
);

void rebuild_grid_from_active_voxels_host(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>* grid,
    size_t grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<nanovdb::ValueOnIndex>& params,
    uint32_t* status
);
