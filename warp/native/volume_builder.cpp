// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "volume_builder.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using namespace wp::volume_builder::internal;

struct HostRebuildScratch {
    bool active_voxel_grid = false;
    std::vector<uint64_t> leaf_keys;
    std::vector<uint64_t> lower_keys;
    std::vector<uint64_t> upper_keys;
    std::vector<uint64_t> voxel_keys;
    uint32_t counts[REBUILD_COUNT_SLOT_COUNT] = {};
};

struct HostRebuildGridData : RebuildGridDataBase {
    const HostRebuildScratch* scratch = nullptr;
};

void rebuild_set_mask_on(
    const HostRebuildGridData& data, pnanovdb_address_t node, uint32_t mask_offset, uint32_t bit_index
)
{
    const pnanovdb_address_t word_address = pnanovdb_address_offset(node, mask_offset + 8u * (bit_index >> 6u));
    const uint64_t word = pnanovdb_read_uint64(data.buf, word_address);
    pnanovdb_write_uint64(data.buf, word_address, word | (uint64_t(1) << (bit_index & 63u)));
}

void rebuild_expand_bbox(
    const HostRebuildGridData& data, pnanovdb_address_t node, pnanovdb_coord_t min, pnanovdb_coord_t max
)
{
    pnanovdb_coord_t bbox_min = pnanovdb_read_coord(data.buf, pnanovdb_address_offset(node, 0u));
    pnanovdb_coord_t bbox_max = pnanovdb_read_coord(data.buf, pnanovdb_address_offset(node, 12u));
    bbox_min.x = std::min(bbox_min.x, min.x);
    bbox_min.y = std::min(bbox_min.y, min.y);
    bbox_min.z = std::min(bbox_min.z, min.z);
    bbox_max.x = std::max(bbox_max.x, max.x);
    bbox_max.y = std::max(bbox_max.y, max.y);
    bbox_max.z = std::max(bbox_max.z, max.z);
    rebuild_set_bbox(data, node, bbox_min, bbox_max);
}

nanovdb::Coord
rebuild_point_to_coord(const void* points, size_t tid, bool points_in_world_space, const nanovdb::Map& map)
{
    if (points_in_world_space) {
        return map.applyInverseMapF(reinterpret_cast<const nanovdb::Vec3f*>(points)[tid]).round();
    }

    return reinterpret_cast<const nanovdb::Coord*>(points)[tid];
}

void rebuild_sort_unique(std::vector<uint64_t>& keys)
{
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
}

bool rebuild_count_points(
    HostRebuildScratch& scratch,
    const void* points,
    size_t num_points,
    const int32_t* point_mask,
    bool points_in_world_space,
    bool active_voxel_grid,
    const nanovdb::Map& map
)
{
    if (num_points == 0 || num_points > size_t(std::numeric_limits<int>::max())) {
        return false;
    }

    scratch = {};
    scratch.active_voxel_grid = active_voxel_grid;

    std::vector<uint64_t> keys;
    keys.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        if (point_mask && point_mask[i] == 0) {
            continue;
        }

        keys.push_back(rebuild_upper_key_from_coord(rebuild_point_to_coord(points, i, points_in_world_space, map)));
    }
    if (keys.empty()) {
        return true;
    }
    scratch.upper_keys = keys;
    rebuild_sort_unique(scratch.upper_keys);

    keys.clear();
    for (size_t i = 0; i < num_points; ++i) {
        if (point_mask && point_mask[i] == 0) {
            continue;
        }

        const nanovdb::Coord ijk = rebuild_point_to_coord(points, i, points_in_world_space, map);
        const uint64_t upper_key = rebuild_upper_key_from_coord(ijk);
        const auto iter = std::lower_bound(scratch.upper_keys.begin(), scratch.upper_keys.end(), upper_key);
        if (iter == scratch.upper_keys.end() || *iter != upper_key) {
            keys.push_back(REBUILD_INVALID_KEY);
        } else {
            keys.push_back(rebuild_hierarchy_key(uint32_t(iter - scratch.upper_keys.begin()), ijk));
        }
    }
    std::sort(keys.begin(), keys.end());
    while (!keys.empty() && keys.back() == REBUILD_INVALID_KEY) {
        keys.pop_back();
    }
    if (keys.empty()) {
        return false;
    }

    if (active_voxel_grid) {
        scratch.voxel_keys = keys;
        rebuild_sort_unique(scratch.voxel_keys);
    }

    scratch.leaf_keys.resize(keys.size());
    scratch.lower_keys.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        scratch.leaf_keys[i] = keys[i] >> 9u;
        scratch.lower_keys[i] = keys[i] >> 21u;
    }
    rebuild_sort_unique(scratch.leaf_keys);
    rebuild_sort_unique(scratch.lower_keys);

    scratch.counts[REBUILD_COUNT_LEAF] = uint32_t(scratch.leaf_keys.size());
    scratch.counts[REBUILD_COUNT_LOWER] = uint32_t(scratch.lower_keys.size());
    scratch.counts[REBUILD_COUNT_UPPER] = uint32_t(scratch.upper_keys.size());
    scratch.counts[REBUILD_COUNT_VOXEL] = active_voxel_grid
        ? uint32_t(scratch.voxel_keys.size())
        : uint32_t(scratch.leaf_keys.size() * PNANOVDB_LEAF_TABLE_COUNT);
    return true;
}

VolumeRebuildCapacities rebuild_exact_capacities(const HostRebuildScratch& scratch)
{
    VolumeRebuildCapacities capacities;
    capacities.leaf_count = scratch.counts[REBUILD_COUNT_LEAF];
    capacities.lower_count = scratch.counts[REBUILD_COUNT_LOWER];
    capacities.upper_count = scratch.counts[REBUILD_COUNT_UPPER];
    capacities.voxel_count = scratch.active_voxel_grid ? scratch.counts[REBUILD_COUNT_VOXEL]
                                                       : uint64_t(capacities.leaf_count) * PNANOVDB_LEAF_TABLE_COUNT;
    return capacities;
}

uint32_t rebuild_capacity_status(const HostRebuildScratch& scratch, const VolumeRebuildCapacities& capacities)
{
    uint32_t status = WP_VOLUME_REBUILD_SUCCESS;
    if (scratch.counts[REBUILD_COUNT_LEAF] > capacities.leaf_count) {
        status |= WP_VOLUME_REBUILD_LEAF_CAPACITY_EXCEEDED;
    }
    if (scratch.counts[REBUILD_COUNT_LOWER] > capacities.lower_count) {
        status |= WP_VOLUME_REBUILD_LOWER_CAPACITY_EXCEEDED;
    }
    if (scratch.counts[REBUILD_COUNT_UPPER] > capacities.upper_count) {
        status |= WP_VOLUME_REBUILD_UPPER_CAPACITY_EXCEEDED;
    }
    if (uint64_t(scratch.counts[REBUILD_COUNT_VOXEL]) > capacities.voxel_count) {
        status |= WP_VOLUME_REBUILD_VOXEL_CAPACITY_EXCEEDED;
    }
    return status;
}

template <typename BuildT>
HostRebuildGridData host_make_rebuild_data(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>* grid,
    size_t grid_size,
    const HostRebuildScratch& scratch,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params
)
{
    HostRebuildGridData data = make_rebuild_data_base<BuildT, HostRebuildGridData>(grid, grid_size, capacities, params);
    data.scratch = &scratch;
    return data;
}

void rebuild_init_grid_tree_root(HostRebuildGridData& data)
{
    const uint32_t leaf_count = data.scratch->counts[REBUILD_COUNT_LEAF];
    const uint32_t lower_count = data.scratch->counts[REBUILD_COUNT_LOWER];
    const uint32_t upper_count = data.scratch->counts[REBUILD_COUNT_UPPER];
    const uint32_t voxel_count = data.scratch->counts[REBUILD_COUNT_VOXEL];

    const pnanovdb_grid_handle_t grid = data.getGrid();
    const pnanovdb_tree_handle_t tree = data.getTree();
    const pnanovdb_root_handle_t root = data.getRoot();

    rebuild_set_invalid_bbox(data, root.address);
    pnanovdb_root_set_tile_count(data.buf, root, upper_count);
    if (rebuild_is_regular_value_grid(data.grid_type)) {
        rebuild_write_background(
            data, pnanovdb_address_offset(root.address, PNANOVDB_GRID_TYPE_GET(data.grid_type, root_off_background))
        );
    }

    pnanovdb_tree_set_node_offset_root(data.buf, tree, data.root - data.tree);
    pnanovdb_tree_set_node_offset_upper(data.buf, tree, data.upper - data.tree);
    pnanovdb_tree_set_node_offset_lower(data.buf, tree, data.lower - data.tree);
    pnanovdb_tree_set_node_offset_leaf(data.buf, tree, data.leaf - data.tree);
    pnanovdb_tree_set_node_count_upper(data.buf, tree, upper_count);
    pnanovdb_tree_set_tile_count_upper(data.buf, tree, upper_count);
    pnanovdb_tree_set_node_count_lower(data.buf, tree, lower_count);
    pnanovdb_tree_set_tile_count_lower(data.buf, tree, lower_count);
    pnanovdb_tree_set_node_count_leaf(data.buf, tree, leaf_count);
    pnanovdb_tree_set_tile_count_leaf(data.buf, tree, leaf_count);
    pnanovdb_tree_set_voxel_count(data.buf, tree, voxel_count);

    pnanovdb_grid_set_magic(data.buf, grid, NANOVDB_MAGIC_GRID);
    pnanovdb_grid_set_checksum(data.buf, grid, ~uint64_t(0));
    pnanovdb_grid_set_version(
        data.buf, grid,
        pnanovdb_make_version(
            PNANOVDB_MAJOR_VERSION_NUMBER, PNANOVDB_MINOR_VERSION_NUMBER, PNANOVDB_PATCH_VERSION_NUMBER
        )
    );
    pnanovdb_grid_set_flags(
        data.buf, grid, uint32_t(nanovdb::GridFlags::HasBBox) | uint32_t(nanovdb::GridFlags::IsBreadthFirst)
    );
    pnanovdb_grid_set_grid_index(data.buf, grid, 0u);
    pnanovdb_grid_set_grid_count(data.buf, grid, 1u);
    pnanovdb_grid_set_grid_size(data.buf, grid, data.size);
    pnanovdb_grid_set_grid_name(data.buf, grid, 0u, 0u);
    *reinterpret_cast<nanovdb::Map*>(
        rebuild_byte_ptr(data, pnanovdb_address_offset(grid.address, PNANOVDB_GRID_OFF_MAP))
    ) = data.map;
    const nanovdb::Vec3d voxel_size = data.map.getVoxelSize();
    pnanovdb_grid_set_voxel_size(data.buf, grid, 0u, voxel_size[0]);
    pnanovdb_grid_set_voxel_size(data.buf, grid, 1u, voxel_size[1]);
    pnanovdb_grid_set_voxel_size(data.buf, grid, 2u, voxel_size[2]);
    pnanovdb_grid_set_grid_class(data.buf, grid, data.grid_class);
    pnanovdb_grid_set_grid_type(data.buf, grid, data.grid_type);
    pnanovdb_grid_set_blind_metadata_offset(data.buf, grid, data.size);
    pnanovdb_grid_set_blind_metadata_count(data.buf, grid, 0u);
    pnanovdb_write_uint32(data.buf, data.address(652u), 0u);
    pnanovdb_write_uint64(
        data.buf, data.address(656u), rebuild_is_index_grid(data.grid_type) ? uint64_t(voxel_count) + 1u : 0u
    );
    pnanovdb_write_uint64(data.buf, data.address(664u), 0u);
}

void rebuild_build_upper_nodes(HostRebuildGridData& data)
{
    const uint32_t upper_count = data.scratch->counts[REBUILD_COUNT_UPPER];
    for (uint32_t tid = 0; tid < upper_count; ++tid) {
        const pnanovdb_root_handle_t root = data.getRoot();
        const pnanovdb_upper_handle_t upper = data.getUpper(tid);
        const nanovdb::Coord ijk = rebuild_upper_key_to_coord(data.scratch->upper_keys[tid]);
        const pnanovdb_coord_t coord = rebuild_make_pnano_coord(ijk);
        const pnanovdb_root_tile_handle_t tile = pnanovdb_root_get_tile(data.grid_type, root, tid);

        pnanovdb_root_tile_set_key(data.buf, tile, pnanovdb_coord_to_key(PNANOVDB_REF(coord)));
        pnanovdb_root_tile_set_state(data.buf, tile, 0u);
        pnanovdb_root_tile_set_child(
            data.buf, tile, int64_t(upper.address.byte_offset) - int64_t(root.address.byte_offset)
        );

        rebuild_set_bbox(data, upper.address, coord, rebuild_invalid_bbox_max());
        pnanovdb_write_uint64(data.buf, pnanovdb_address_offset(upper.address, PNANOVDB_UPPER_OFF_FLAGS), 0u);
        rebuild_clear_mask_words(data, upper.address, PNANOVDB_UPPER_OFF_VALUE_MASK, PNANOVDB_UPPER_TABLE_COUNT / 64u);
        rebuild_clear_mask_words(data, upper.address, PNANOVDB_UPPER_OFF_CHILD_MASK, PNANOVDB_UPPER_TABLE_COUNT / 64u);
    }
}

void rebuild_set_upper_background_values(HostRebuildGridData& data)
{
    if (!rebuild_is_regular_value_grid(data.grid_type)) {
        return;
    }

    const uint32_t upper_count = data.scratch->counts[REBUILD_COUNT_UPPER];
    for (uint32_t upper_id = 0; upper_id < upper_count; ++upper_id) {
        const pnanovdb_upper_handle_t upper = data.getUpper(upper_id);
        for (uint32_t n = 0; n < PNANOVDB_UPPER_TABLE_COUNT; ++n) {
            rebuild_write_background(data, pnanovdb_upper_get_table_address(data.grid_type, data.buf, upper, n));
        }
    }
}

void rebuild_build_lower_nodes(HostRebuildGridData& data)
{
    const uint32_t lower_count = data.scratch->counts[REBUILD_COUNT_LOWER];
    const uint32_t upper_count = data.scratch->counts[REBUILD_COUNT_UPPER];
    for (uint32_t tid = 0; tid < lower_count; ++tid) {
        const uint64_t lower_key = data.scratch->lower_keys[tid];
        const uint32_t upper_id = uint32_t(lower_key >> 15u);
        if (upper_id >= upper_count) {
            continue;
        }

        const pnanovdb_upper_handle_t upper = data.getUpper(upper_id);
        const pnanovdb_lower_handle_t lower = data.getLower(tid);
        const uint32_t upper_offset = uint32_t(lower_key & 32767u);
        rebuild_set_mask_on(data, upper.address, PNANOVDB_UPPER_OFF_CHILD_MASK, upper_offset);
        pnanovdb_upper_set_table_child(
            data.grid_type, data.buf, upper, upper_offset,
            int64_t(lower.address.byte_offset) - int64_t(upper.address.byte_offset)
        );

        const pnanovdb_coord_t origin = rebuild_make_pnano_coord(
            rebuild_make_coord(pnanovdb_upper_get_bbox_min(data.buf, upper))
            + nanovdb::Coord(
                int32_t((upper_offset >> 10u) & 31u) << 7, int32_t((upper_offset >> 5u) & 31u) << 7,
                int32_t(upper_offset & 31u) << 7
            )
        );
        rebuild_set_bbox(data, lower.address, origin, rebuild_invalid_bbox_max());
        pnanovdb_write_uint64(data.buf, pnanovdb_address_offset(lower.address, PNANOVDB_LOWER_OFF_FLAGS), 0u);
        rebuild_clear_mask_words(data, lower.address, PNANOVDB_LOWER_OFF_VALUE_MASK, PNANOVDB_LOWER_TABLE_COUNT / 64u);
        rebuild_clear_mask_words(data, lower.address, PNANOVDB_LOWER_OFF_CHILD_MASK, PNANOVDB_LOWER_TABLE_COUNT / 64u);
    }
}

void rebuild_set_lower_background_values(HostRebuildGridData& data)
{
    if (!rebuild_is_regular_value_grid(data.grid_type)) {
        return;
    }

    const uint32_t lower_count = data.scratch->counts[REBUILD_COUNT_LOWER];
    for (uint32_t lower_id = 0; lower_id < lower_count; ++lower_id) {
        const pnanovdb_lower_handle_t lower = data.getLower(lower_id);
        for (uint32_t n = 0; n < PNANOVDB_LOWER_TABLE_COUNT; ++n) {
            rebuild_write_background(data, pnanovdb_lower_get_table_address(data.grid_type, data.buf, lower, n));
        }
    }
}

void rebuild_build_leaf_nodes(HostRebuildGridData& data, bool active_voxel_grid)
{
    const uint32_t leaf_count = data.scratch->counts[REBUILD_COUNT_LEAF];
    for (uint32_t tid = 0; tid < leaf_count; ++tid) {
        const uint64_t leaf_key = data.scratch->leaf_keys[tid];
        const uint64_t lower_key = leaf_key >> 12u;
        const auto lower_iter
            = std::lower_bound(data.scratch->lower_keys.begin(), data.scratch->lower_keys.end(), lower_key);
        if (lower_iter == data.scratch->lower_keys.end() || *lower_iter != lower_key) {
            continue;
        }

        const uint32_t lower_id = uint32_t(lower_iter - data.scratch->lower_keys.begin());
        const pnanovdb_lower_handle_t lower = data.getLower(lower_id);
        const pnanovdb_leaf_handle_t leaf = data.getLeaf(tid);
        const uint32_t lower_offset = uint32_t(leaf_key & 4095u);
        rebuild_set_mask_on(data, lower.address, PNANOVDB_LOWER_OFF_CHILD_MASK, lower_offset);
        pnanovdb_lower_set_table_child(
            data.grid_type, data.buf, lower, lower_offset,
            int64_t(leaf.address.byte_offset) - int64_t(lower.address.byte_offset)
        );

        const pnanovdb_coord_t lower_origin = pnanovdb_lower_get_bbox_min(data.buf, lower);
        pnanovdb_coord_t leaf_origin = {
            lower_origin.x + (int32_t((lower_offset >> 8u) & 15u) << 3),
            lower_origin.y + (int32_t((lower_offset >> 4u) & 15u) << 3),
            lower_origin.z + (int32_t(lower_offset & 15u) << 3),
        };
        pnanovdb_leaf_set_bbox_min(data.buf, leaf, PNANOVDB_REF(leaf_origin));
        pnanovdb_leaf_set_bbox_dif_and_flags(data.buf, leaf, 2u << 24u);
        rebuild_clear_mask_words(data, leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK, PNANOVDB_LEAF_TABLE_COUNT / 64u);

        if (data.grid_type == PNANOVDB_GRID_TYPE_INDEX) {
            pnanovdb_write_uint64(
                data.buf, rebuild_leaf_index_offset_address(data, leaf), uint64_t(tid) * PNANOVDB_LEAF_TABLE_COUNT + 1u
            );
            pnanovdb_write_uint64(data.buf, rebuild_leaf_index_prefix_address(data, leaf), 0u);
            for (uint32_t word = 0; word < PNANOVDB_LEAF_TABLE_COUNT / 64u; ++word) {
                pnanovdb_write_uint64(
                    data.buf, pnanovdb_address_offset(leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK + 8u * word),
                    ~uint64_t(0)
                );
            }
            pnanovdb_leaf_set_bbox_dif_and_flags(data.buf, leaf, 0x00070707u | (2u << 24u));
        } else if (rebuild_is_onindex_grid(data.grid_type)) {
            pnanovdb_write_uint64(data.buf, rebuild_leaf_index_offset_address(data, leaf), 1u);
            pnanovdb_write_uint64(data.buf, rebuild_leaf_index_prefix_address(data, leaf), 0u);
        } else if (!active_voxel_grid) {
            for (uint32_t word = 0; word < PNANOVDB_LEAF_TABLE_COUNT / 64u; ++word) {
                pnanovdb_write_uint64(
                    data.buf, pnanovdb_address_offset(leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK + 8u * word),
                    ~uint64_t(0)
                );
            }
            pnanovdb_leaf_set_bbox_dif_and_flags(data.buf, leaf, 0x00070707u | (2u << 24u));
        }
    }
}

void rebuild_set_leaf_values(HostRebuildGridData& data)
{
    if (!rebuild_is_regular_value_grid(data.grid_type)) {
        return;
    }

    const uint32_t leaf_count = data.scratch->counts[REBUILD_COUNT_LEAF];
    for (uint32_t leaf_id = 0; leaf_id < leaf_count; ++leaf_id) {
        const pnanovdb_leaf_handle_t leaf = data.getLeaf(leaf_id);
        for (uint32_t n = 0; n < PNANOVDB_LEAF_TABLE_COUNT; ++n) {
            rebuild_write_background(data, pnanovdb_leaf_get_table_address(data.grid_type, data.buf, leaf, n));
        }
    }
}

void rebuild_set_active_voxels(HostRebuildGridData& data, std::vector<uint32_t>& leaf_active_counts)
{
    const uint32_t leaf_count = data.scratch->counts[REBUILD_COUNT_LEAF];
    leaf_active_counts.assign(leaf_count, 0u);
    for (uint64_t voxel_key : data.scratch->voxel_keys) {
        const uint64_t leaf_key = voxel_key >> 9u;
        const auto leaf_iter
            = std::lower_bound(data.scratch->leaf_keys.begin(), data.scratch->leaf_keys.end(), leaf_key);
        if (leaf_iter == data.scratch->leaf_keys.end() || *leaf_iter != leaf_key) {
            continue;
        }

        const uint32_t leaf_id = uint32_t(leaf_iter - data.scratch->leaf_keys.begin());
        const pnanovdb_leaf_handle_t leaf = data.getLeaf(leaf_id);
        const uint32_t offset = uint32_t(voxel_key & 511u);
        rebuild_set_mask_on(data, leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK, offset);
        ++leaf_active_counts[leaf_id];
    }
}

void rebuild_finalize_onindex_leaves(HostRebuildGridData& data, const std::vector<uint32_t>& leaf_active_counts)
{
    const uint32_t leaf_count = data.scratch->counts[REBUILD_COUNT_LEAF];
    uint64_t active_prefix = 0u;
    for (uint32_t tid = 0; tid < leaf_count; ++tid) {
        const pnanovdb_leaf_handle_t leaf = data.getLeaf(tid);
        pnanovdb_write_uint64(data.buf, rebuild_leaf_index_offset_address(data, leaf), 1u + active_prefix);
        active_prefix += leaf_active_counts[tid];

        uint64_t prefix = nanovdb::util::countOn(
            pnanovdb_read_uint64(data.buf, pnanovdb_address_offset(leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK))
        );
        uint64_t sum = prefix;
        for (int n = 9; n < 55; n += 9) {
            sum += nanovdb::util::countOn(pnanovdb_read_uint64(
                data.buf, pnanovdb_address_offset(leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK + 8u * uint32_t(n / 9))
            ));
            prefix |= sum << n;
        }
        pnanovdb_write_uint64(data.buf, rebuild_leaf_index_prefix_address(data, leaf), prefix);
        rebuild_update_leaf_bbox(data, leaf);
    }
}

void rebuild_reset_bboxes(HostRebuildGridData& data)
{
    rebuild_set_invalid_bbox(data, data.getRoot().address);
    for (uint32_t i = 0; i < data.scratch->counts[REBUILD_COUNT_LOWER]; ++i) {
        rebuild_set_invalid_bbox(data, data.getLower(i).address);
    }
    for (uint32_t i = 0; i < data.scratch->counts[REBUILD_COUNT_UPPER]; ++i) {
        rebuild_set_invalid_bbox(data, data.getUpper(i).address);
    }
}

void rebuild_propagate_leaf_bboxes(HostRebuildGridData& data)
{
    const uint32_t lower_count = data.scratch->counts[REBUILD_COUNT_LOWER];
    for (uint32_t tid = 0; tid < data.scratch->counts[REBUILD_COUNT_LEAF]; ++tid) {
        const pnanovdb_leaf_handle_t leaf = data.getLeaf(tid);
        if (!rebuild_is_onindex_grid(data.grid_type)) {
            rebuild_update_leaf_bbox(data, leaf);
        }

        const uint64_t lower_key = data.scratch->leaf_keys[tid] >> 12u;
        const auto lower_iter
            = std::lower_bound(data.scratch->lower_keys.begin(), data.scratch->lower_keys.end(), lower_key);
        if (lower_iter != data.scratch->lower_keys.end() && *lower_iter == lower_key) {
            const uint32_t lower_id = uint32_t(lower_iter - data.scratch->lower_keys.begin());
            rebuild_expand_bbox(
                data, data.getLower(lower_id).address, pnanovdb_leaf_get_bbox_min(data.buf, leaf),
                rebuild_leaf_bbox_max(data.buf, leaf)
            );
        }
    }
}

void rebuild_propagate_lower_bboxes(HostRebuildGridData& data)
{
    const uint32_t upper_count = data.scratch->counts[REBUILD_COUNT_UPPER];
    for (uint32_t tid = 0; tid < data.scratch->counts[REBUILD_COUNT_LOWER]; ++tid) {
        const uint64_t upper_key = data.scratch->lower_keys[tid] >> 15u;
        if (upper_key >= upper_count) {
            continue;
        }

        const pnanovdb_lower_handle_t lower = data.getLower(tid);
        rebuild_expand_bbox(
            data, data.getUpper(uint32_t(upper_key)).address, pnanovdb_lower_get_bbox_min(data.buf, lower),
            pnanovdb_lower_get_bbox_max(data.buf, lower)
        );
    }
}

void rebuild_propagate_upper_bboxes(HostRebuildGridData& data)
{
    for (uint32_t tid = 0; tid < data.scratch->counts[REBUILD_COUNT_UPPER]; ++tid) {
        const pnanovdb_upper_handle_t upper = data.getUpper(tid);
        rebuild_expand_bbox(
            data, data.getRoot().address, pnanovdb_upper_get_bbox_min(data.buf, upper),
            pnanovdb_upper_get_bbox_max(data.buf, upper)
        );
    }
}

void rebuild_finalize_world_bbox(HostRebuildGridData& data)
{
    const pnanovdb_grid_handle_t grid = data.getGrid();
    nanovdb::Vec3dBBox world_bbox;
    if (data.scratch->counts[REBUILD_COUNT_UPPER] > 0) {
        const pnanovdb_root_handle_t root = data.getRoot();
        const pnanovdb_coord_t root_min = pnanovdb_root_get_bbox_min(data.buf, root);
        const pnanovdb_coord_t root_max = pnanovdb_root_get_bbox_max(data.buf, root);

        const nanovdb::Vec3d index_min(root_min.x, root_min.y, root_min.z);
        const nanovdb::Vec3d index_max(root_max.x + 1.0, root_max.y + 1.0, root_max.z + 1.0);
        world_bbox = nanovdb::Vec3dBBox(data.map.applyMap(index_min), data.map.applyMap(index_min));
        world_bbox.expand(data.map.applyMap(nanovdb::Vec3d(index_min[0], index_min[1], index_max[2])));
        world_bbox.expand(data.map.applyMap(nanovdb::Vec3d(index_min[0], index_max[1], index_min[2])));
        world_bbox.expand(data.map.applyMap(nanovdb::Vec3d(index_max[0], index_min[1], index_min[2])));
        world_bbox.expand(data.map.applyMap(nanovdb::Vec3d(index_max[0], index_max[1], index_min[2])));
        world_bbox.expand(data.map.applyMap(nanovdb::Vec3d(index_max[0], index_min[1], index_max[2])));
        world_bbox.expand(data.map.applyMap(nanovdb::Vec3d(index_min[0], index_max[1], index_max[2])));
        world_bbox.expand(data.map.applyMap(index_max));
    }
    pnanovdb_grid_set_world_bbox(data.buf, grid, 0u, world_bbox[0][0]);
    pnanovdb_grid_set_world_bbox(data.buf, grid, 1u, world_bbox[0][1]);
    pnanovdb_grid_set_world_bbox(data.buf, grid, 2u, world_bbox[0][2]);
    pnanovdb_grid_set_world_bbox(data.buf, grid, 3u, world_bbox[1][0]);
    pnanovdb_grid_set_world_bbox(data.buf, grid, 4u, world_bbox[1][1]);
    pnanovdb_grid_set_world_bbox(data.buf, grid, 5u, world_bbox[1][2]);
}

template <typename BuildT>
void rebuild_populate_grid(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>* grid,
    size_t grid_size,
    const HostRebuildScratch& scratch,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params
)
{
    std::memset(grid, 0, grid_size);
    HostRebuildGridData data = host_make_rebuild_data(grid, grid_size, scratch, capacities, params);

    rebuild_init_grid_tree_root(data);
    rebuild_build_upper_nodes(data);
    rebuild_set_upper_background_values(data);
    rebuild_build_lower_nodes(data);
    rebuild_set_lower_background_values(data);
    rebuild_build_leaf_nodes(data, scratch.active_voxel_grid);

    if (scratch.active_voxel_grid && data.grid_type == PNANOVDB_GRID_TYPE_ONINDEX) {
        std::vector<uint32_t> leaf_active_counts;
        rebuild_set_active_voxels(data, leaf_active_counts);
        rebuild_finalize_onindex_leaves(data, leaf_active_counts);
    } else {
        rebuild_set_leaf_values(data);
    }

    rebuild_reset_bboxes(data);
    rebuild_propagate_leaf_bboxes(data);
    rebuild_propagate_lower_bboxes(data);
    rebuild_propagate_upper_bboxes(data);
    rebuild_finalize_world_bbox(data);
}

template <typename BuildT>
void allocate_exact_grid_from_points_host_impl(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    const int32_t* point_mask,
    bool points_in_world_space,
    bool active_voxel_grid,
    const BuildGridParams<BuildT>& params
)
{
    HostRebuildScratch scratch;
    if (!rebuild_count_points(
            scratch, points, num_points, point_mask, points_in_world_space, active_voxel_grid, params.map
        )) {
        out_grid = nullptr;
        out_grid_size = 0u;
        return;
    }

    const VolumeRebuildCapacities capacities = rebuild_exact_capacities(scratch);
    out_grid_size = rebuildable_grid_size<BuildT>(capacities);
    out_grid = static_cast<nanovdb::Grid<nanovdb::NanoTree<BuildT>>*>(
        wp_alloc_host(out_grid_size, "(native:volume_builder)")
    );
    rebuild_populate_grid(out_grid, out_grid_size, scratch, capacities, params);
}

template <typename BuildT>
void rebuild_grid_from_points_host_impl(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>* grid,
    size_t grid_size,
    const void* points,
    size_t num_points,
    const int32_t* point_mask,
    bool points_in_world_space,
    bool active_voxel_grid,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params,
    uint32_t* status
)
{
    if (status) {
        *status = WP_VOLUME_REBUILD_SUCCESS;
    }

    if (capacities.leaf_count == 0 || capacities.lower_count == 0 || capacities.upper_count == 0) {
        if (status) {
            *status = WP_VOLUME_REBUILD_INVALID_INPUT;
        }
        return;
    }

    HostRebuildScratch scratch;
    if (!rebuild_count_points(
            scratch, points, num_points, point_mask, points_in_world_space, active_voxel_grid, params.map
        )) {
        if (status) {
            *status = WP_VOLUME_REBUILD_INVALID_INPUT;
        }
        return;
    }

    const uint32_t capacity_status = rebuild_capacity_status(scratch, capacities);
    if (capacity_status != WP_VOLUME_REBUILD_SUCCESS) {
        if (status) {
            *status = capacity_status;
        }
        return;
    }

    rebuild_populate_grid(grid, grid_size, scratch, capacities, params);
}

template <typename BuildT>
void allocate_rebuildable_grid_from_points_host_impl(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    const int32_t* point_mask,
    bool points_in_world_space,
    bool active_voxel_grid,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params,
    uint32_t* status
)
{
    HostRebuildScratch scratch;
    if (!rebuild_count_points(
            scratch, points, num_points, point_mask, points_in_world_space, active_voxel_grid, params.map
        )) {
        if (status) {
            *status = WP_VOLUME_REBUILD_INVALID_INPUT;
        }
        out_grid = nullptr;
        out_grid_size = 0u;
        return;
    }

    const uint32_t capacity_status = rebuild_capacity_status(scratch, capacities);
    out_grid_size = rebuildable_grid_size<BuildT>(capacities);
    out_grid = static_cast<nanovdb::Grid<nanovdb::NanoTree<BuildT>>*>(
        wp_alloc_host(out_grid_size, "(native:volume_builder)")
    );
    if (capacity_status == WP_VOLUME_REBUILD_SUCCESS) {
        rebuild_populate_grid(out_grid, out_grid_size, scratch, capacities, params);
    } else {
        HostRebuildScratch empty_scratch;
        empty_scratch.active_voxel_grid = scratch.active_voxel_grid;
        rebuild_populate_grid(out_grid, out_grid_size, empty_scratch, capacities, params);
    }

    if (status) {
        *status = capacity_status;
    }
}

}  // namespace

template <typename BuildT>
void allocate_grid_from_tiles_host(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const BuildGridParams<BuildT>& params
)
{
    allocate_exact_grid_from_points_host_impl(
        out_grid, out_grid_size, points, num_points, point_mask, points_in_world_space, false, params
    );
}

#define EXPAND_BUILDER_TYPE(type)                                                                                      \
    template void allocate_grid_from_tiles_host(                                                                       \
        nanovdb::Grid<nanovdb::NanoTree<type>>*&, size_t&, const void*, size_t, bool, const int32_t*,                  \
        const BuildGridParams<type>&                                                                                    \
    );

WP_VOLUME_BUILDER_INSTANTIATE_TYPES
#undef EXPAND_BUILDER_TYPE

template void allocate_grid_from_tiles_host(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueIndex>>*&,
    size_t&,
    const void*,
    size_t,
    bool,
    const int32_t*,
    const BuildGridParams<nanovdb::ValueIndex>&
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
)
{
    rebuild_grid_from_points_host_impl(
        grid, grid_size, points, num_points, point_mask, points_in_world_space, false, capacities, params, status
    );
}

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
)
{
    allocate_rebuildable_grid_from_points_host_impl(
        out_grid, out_grid_size, points, num_points, point_mask, points_in_world_space, false, capacities, params,
        status
    );
}

#define EXPAND_BUILDER_TYPE(type)                                                                                      \
    template void allocate_rebuildable_grid_from_tiles_host(                                                           \
        nanovdb::Grid<nanovdb::NanoTree<type>>*&, size_t&, const void*, size_t, bool, const int32_t*,                  \
        const VolumeRebuildCapacities&, const BuildGridParams<type>&, uint32_t*                                        \
    );                                                                                                                 \
    template void rebuild_grid_from_tiles_host(                                                                        \
        nanovdb::Grid<nanovdb::NanoTree<type>>*, size_t, const void*, size_t, bool, const int32_t*,                    \
        const VolumeRebuildCapacities&, const BuildGridParams<type>&, uint32_t*                                        \
    );

WP_VOLUME_BUILDER_INSTANTIATE_TYPES
#undef EXPAND_BUILDER_TYPE

template void allocate_rebuildable_grid_from_tiles_host(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueIndex>>*&,
    size_t&,
    const void*,
    size_t,
    bool,
    const int32_t*,
    const VolumeRebuildCapacities&,
    const BuildGridParams<nanovdb::ValueIndex>&,
    uint32_t*
);

template void rebuild_grid_from_tiles_host(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueIndex>>*,
    size_t,
    const void*,
    size_t,
    bool,
    const int32_t*,
    const VolumeRebuildCapacities&,
    const BuildGridParams<nanovdb::ValueIndex>&,
    uint32_t*
);

void allocate_grid_from_active_voxels_host(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const BuildGridParams<nanovdb::ValueOnIndex>& params
)
{
    allocate_exact_grid_from_points_host_impl(
        out_grid, out_grid_size, points, num_points, point_mask, points_in_world_space, true, params
    );
}

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
)
{
    rebuild_grid_from_points_host_impl(
        grid, grid_size, points, num_points, point_mask, points_in_world_space, true, capacities, params, status
    );
}

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
)
{
    allocate_rebuildable_grid_from_points_host_impl(
        out_grid, out_grid_size, points, num_points, point_mask, points_in_world_space, true, capacities, params, status
    );
}
