// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "cuda_util.h"
#include "temp_buffer.h"
#include "volume_builder.h"

#include <algorithm>
#include <limits>

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace {

static constexpr unsigned REBUILD_NUM_THREADS = 128;

inline unsigned rebuild_num_blocks(uint64_t n) { return unsigned((n + REBUILD_NUM_THREADS - 1) / REBUILD_NUM_THREADS); }

using namespace wp::volume_builder::internal;

struct RebuildGridData : RebuildGridDataBase {
    uint64_t* leaf_keys;
    uint64_t* lower_keys;
    uint64_t* upper_keys;
    uint64_t* voxel_keys;
    uint32_t* counts;
    uint32_t* leaf_active_counts;
    uint64_t* leaf_active_prefix;
    uint32_t* status;
};

__device__ inline int32_t* rebuild_int_ptr(const RebuildGridData& data, pnanovdb_address_t address)
{
    return reinterpret_cast<int32_t*>(rebuild_byte_ptr(data, address));
}

__device__ inline unsigned long long int* rebuild_u64_ptr(const RebuildGridData& data, pnanovdb_address_t address)
{
    return reinterpret_cast<unsigned long long int*>(rebuild_byte_ptr(data, address));
}

__device__ inline void rebuild_set_mask_on_atomic(
    const RebuildGridData& data, pnanovdb_address_t node, uint32_t mask_offset, uint32_t bit_index
)
{
    const pnanovdb_address_t word_address = pnanovdb_address_offset(node, mask_offset + 8u * (bit_index >> 6u));
    atomicOr(rebuild_u64_ptr(data, word_address), 1ull << (bit_index & 63u));
}

__device__ inline void rebuild_expand_bbox_atomic(
    const RebuildGridData& data, pnanovdb_address_t node, pnanovdb_coord_t min, pnanovdb_coord_t max
)
{
    atomicMin(rebuild_int_ptr(data, pnanovdb_address_offset(node, 0u)), min.x);
    atomicMin(rebuild_int_ptr(data, pnanovdb_address_offset(node, 4u)), min.y);
    atomicMin(rebuild_int_ptr(data, pnanovdb_address_offset(node, 8u)), min.z);
    atomicMax(rebuild_int_ptr(data, pnanovdb_address_offset(node, 12u)), max.x);
    atomicMax(rebuild_int_ptr(data, pnanovdb_address_offset(node, 16u)), max.y);
    atomicMax(rebuild_int_ptr(data, pnanovdb_address_offset(node, 20u)), max.z);
}

__device__ inline void rebuild_set_status(uint32_t* status, uint32_t bits)
{
    if (status) {
        atomicOr(status, bits);
    }
}

__global__ void rebuild_write_status(uint32_t* status, uint32_t bits)
{
    if (status) {
        *status = bits;
    }
}

struct RebuildKeyScratch {
    size_t num_points = 0;
    int point_count = 0;
    bool active_voxel_grid = false;
    uint64_t* keys_a = nullptr;
    uint64_t* keys_b = nullptr;
    uint64_t* leaf_keys = nullptr;
    uint64_t* lower_keys = nullptr;
    uint64_t* upper_keys = nullptr;
    uint64_t* voxel_keys = nullptr;
    uint64_t* parent_keys = nullptr;
    uint32_t* run_counts = nullptr;
    uint32_t* counts = nullptr;
};

void rebuild_free_key_scratch(RebuildKeyScratch& scratch)
{
    if (scratch.keys_a)
        wp_free_device(WP_CURRENT_CONTEXT, scratch.keys_a);
    if (scratch.keys_b)
        wp_free_device(WP_CURRENT_CONTEXT, scratch.keys_b);
    if (scratch.leaf_keys)
        wp_free_device(WP_CURRENT_CONTEXT, scratch.leaf_keys);
    if (scratch.lower_keys)
        wp_free_device(WP_CURRENT_CONTEXT, scratch.lower_keys);
    if (scratch.upper_keys)
        wp_free_device(WP_CURRENT_CONTEXT, scratch.upper_keys);
    if (scratch.voxel_keys)
        wp_free_device(WP_CURRENT_CONTEXT, scratch.voxel_keys);
    if (scratch.parent_keys)
        wp_free_device(WP_CURRENT_CONTEXT, scratch.parent_keys);
    if (scratch.run_counts)
        wp_free_device(WP_CURRENT_CONTEXT, scratch.run_counts);
    if (scratch.counts)
        wp_free_device(WP_CURRENT_CONTEXT, scratch.counts);

    scratch = {};
}

bool rebuild_alloc_key_scratch(
    RebuildKeyScratch& scratch, size_t num_points, bool active_voxel_grid, uint32_t* status, cudaStream_t stream
)
{
    if (status) {
        check_cuda(cudaMemsetAsync(status, 0, sizeof(uint32_t), stream));
    }

    if (num_points == 0 || num_points > size_t(std::numeric_limits<int>::max())) {
        if (status) {
            rebuild_write_status<<<1, 1, 0, stream>>>(status, WP_VOLUME_REBUILD_INVALID_INPUT);
            check_cuda(cudaGetLastError());
        }
        return false;
    }

    scratch.num_points = num_points;
    scratch.point_count = int(num_points);
    scratch.active_voxel_grid = active_voxel_grid;
    scratch.keys_a = static_cast<uint64_t*>(
        wp_alloc_device(WP_CURRENT_CONTEXT, num_points * sizeof(uint64_t), "(native:volume_builder)")
    );
    scratch.keys_b = static_cast<uint64_t*>(
        wp_alloc_device(WP_CURRENT_CONTEXT, num_points * sizeof(uint64_t), "(native:volume_builder)")
    );
    scratch.leaf_keys = static_cast<uint64_t*>(
        wp_alloc_device(WP_CURRENT_CONTEXT, num_points * sizeof(uint64_t), "(native:volume_builder)")
    );
    scratch.lower_keys = static_cast<uint64_t*>(
        wp_alloc_device(WP_CURRENT_CONTEXT, num_points * sizeof(uint64_t), "(native:volume_builder)")
    );
    scratch.upper_keys = static_cast<uint64_t*>(
        wp_alloc_device(WP_CURRENT_CONTEXT, num_points * sizeof(uint64_t), "(native:volume_builder)")
    );
    scratch.voxel_keys = active_voxel_grid
        ? static_cast<uint64_t*>(
              wp_alloc_device(WP_CURRENT_CONTEXT, num_points * sizeof(uint64_t), "(native:volume_builder)")
          )
        : nullptr;
    scratch.parent_keys = static_cast<uint64_t*>(
        wp_alloc_device(WP_CURRENT_CONTEXT, num_points * sizeof(uint64_t), "(native:volume_builder)")
    );
    scratch.run_counts = static_cast<uint32_t*>(
        wp_alloc_device(WP_CURRENT_CONTEXT, num_points * sizeof(uint32_t), "(native:volume_builder)")
    );
    scratch.counts = static_cast<uint32_t*>(
        wp_alloc_device(WP_CURRENT_CONTEXT, REBUILD_COUNT_SLOT_COUNT * sizeof(uint32_t), "(native:volume_builder)")
    );

    check_cuda(cudaMemsetAsync(scratch.counts, 0, REBUILD_COUNT_SLOT_COUNT * sizeof(uint32_t), stream));
    return true;
}

template <typename PtrT> struct RebuildPointerTraits {
    using element_type = typename nanovdb::util::remove_reference<decltype(*nanovdb::util::declval<PtrT>())>::type;
};

template <typename T> struct RebuildPointerTraits<T*> {
    using element_type = T;
};

template <typename PtrT>
__device__ nanovdb::Coord rebuild_point_to_coord(const PtrT points, size_t tid, const nanovdb::Map& map)
{
    using Vec3T = typename nanovdb::util::remove_const<typename RebuildPointerTraits<PtrT>::element_type>::type;
    if constexpr (nanovdb::util::is_same<Vec3T, nanovdb::Coord>::value) {
        return points[tid];
    } else if constexpr (nanovdb::util::is_same<Vec3T, nanovdb::Vec3f>::value) {
        return map.applyInverseMapF(points[tid]).round();
    } else {
        return map.applyInverseMap(points[tid]).round();
    }
}

__device__ inline int32_t rebuild_find_key_u64(const uint64_t* keys, uint32_t count, uint64_t key)
{
    uint32_t first = 0;
    uint32_t last = count;
    while (first < last) {
        const uint32_t mid = first + ((last - first) >> 1u);
        const uint64_t mid_key = keys[mid];
        if (mid_key < key) {
            first = mid + 1u;
        } else {
            last = mid;
        }
    }
    return first < count && keys[first] == key ? int32_t(first) : -1;
}

__device__ inline bool rebuild_point_is_masked_out(const int32_t* point_mask, size_t tid)
{
    return point_mask && point_mask[tid] == 0;
}

template <typename PtrT>
__global__ void rebuild_emit_upper_keys(
    size_t point_count, const PtrT points, const int32_t* point_mask, nanovdb::Map map, uint64_t* keys
)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= point_count)
        return;

    if (rebuild_point_is_masked_out(point_mask, tid)) {
        keys[tid] = REBUILD_INVALID_KEY;
        return;
    }

    keys[tid] = rebuild_upper_key_from_coord(rebuild_point_to_coord(points, tid, map));
}

template <typename PtrT>
__global__ void rebuild_emit_hierarchy_keys(
    size_t point_count,
    const PtrT points,
    const int32_t* point_mask,
    nanovdb::Map map,
    const uint64_t* upper_keys,
    const uint32_t* counts,
    uint64_t* keys,
    uint32_t* status
)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= point_count)
        return;

    if (rebuild_point_is_masked_out(point_mask, tid)) {
        keys[tid] = REBUILD_INVALID_KEY;
        return;
    }

    const nanovdb::Coord ijk = rebuild_point_to_coord(points, tid, map);
    const uint64_t upper_key = rebuild_upper_key_from_coord(ijk);
    const int32_t upper_id = rebuild_find_key_u64(upper_keys, counts[REBUILD_COUNT_UPPER], upper_key);
    if (upper_id < 0 || upper_id >= (1 << 28)) {
        keys[tid] = REBUILD_INVALID_KEY;
        rebuild_set_status(status, WP_VOLUME_REBUILD_INVALID_INPUT);
        return;
    }

    keys[tid] = rebuild_hierarchy_key(uint32_t(upper_id), ijk);
}

template <uint32_t Shift>
__global__ void rebuild_shift_keys(const uint64_t* keys_in, uint64_t* keys_out, size_t key_count)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= key_count)
        return;

    const uint64_t key = keys_in[tid];
    keys_out[tid] = key == REBUILD_INVALID_KEY ? REBUILD_INVALID_KEY : key >> Shift;
}

__global__ void rebuild_finalize_counts(
    const uint64_t* leaf_keys,
    const uint64_t* lower_keys,
    const uint64_t* upper_keys,
    const uint64_t* voxel_keys,
    uint32_t* counts,
    uint32_t* status,
    VolumeRebuildCapacities capacities,
    bool active_voxel_grid,
    bool enforce_capacities
)
{
    uint32_t leaf_count = counts[REBUILD_COUNT_LEAF];
    uint32_t lower_count = counts[REBUILD_COUNT_LOWER];
    uint32_t upper_count = counts[REBUILD_COUNT_UPPER];
    uint32_t voxel_count = counts[REBUILD_COUNT_VOXEL];

    if (leaf_count > 0 && leaf_keys && leaf_keys[leaf_count - 1] == REBUILD_INVALID_KEY)
        --leaf_count;
    if (lower_count > 0 && lower_keys && lower_keys[lower_count - 1] == REBUILD_INVALID_KEY)
        --lower_count;
    if (upper_count > 0 && upper_keys && upper_keys[upper_count - 1] == REBUILD_INVALID_KEY)
        --upper_count;
    if (voxel_count > 0 && voxel_keys && voxel_keys[voxel_count - 1] == REBUILD_INVALID_KEY)
        --voxel_count;

    if (enforce_capacities && leaf_count > capacities.leaf_count) {
        rebuild_set_status(status, WP_VOLUME_REBUILD_LEAF_CAPACITY_EXCEEDED);
        leaf_count = capacities.leaf_count;
    }
    if (enforce_capacities && lower_count > capacities.lower_count) {
        rebuild_set_status(status, WP_VOLUME_REBUILD_LOWER_CAPACITY_EXCEEDED);
        lower_count = capacities.lower_count;
    }
    if (enforce_capacities && upper_count > capacities.upper_count) {
        rebuild_set_status(status, WP_VOLUME_REBUILD_UPPER_CAPACITY_EXCEEDED);
        upper_count = capacities.upper_count;
    }
    if (enforce_capacities && uint64_t(voxel_count) > capacities.voxel_count) {
        rebuild_set_status(status, WP_VOLUME_REBUILD_VOXEL_CAPACITY_EXCEEDED);
        voxel_count = uint32_t(capacities.voxel_count);
    }

    counts[REBUILD_COUNT_LEAF] = leaf_count;
    counts[REBUILD_COUNT_LOWER] = lower_count;
    counts[REBUILD_COUNT_UPPER] = upper_count;
    counts[REBUILD_COUNT_VOXEL] = active_voxel_grid ? voxel_count : leaf_count * PNANOVDB_LEAF_TABLE_COUNT;
}

__global__ void rebuild_init_grid_tree_root(RebuildGridData data)
{
    const uint32_t leaf_count = data.counts[REBUILD_COUNT_LEAF];
    const uint32_t lower_count = data.counts[REBUILD_COUNT_LOWER];
    const uint32_t upper_count = data.counts[REBUILD_COUNT_UPPER];
    const uint32_t voxel_count = data.counts[REBUILD_COUNT_VOXEL];

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
    rebuild_zero_bytes(
        data, pnanovdb_address_offset(root.address, PNANOVDB_GRID_TYPE_GET(data.grid_type, root_off_min)),
        PNANOVDB_GRID_TYPE_GET(data.grid_type, root_off_max) - PNANOVDB_GRID_TYPE_GET(data.grid_type, root_off_min)
    );
    rebuild_zero_bytes(
        data, pnanovdb_address_offset(root.address, PNANOVDB_GRID_TYPE_GET(data.grid_type, root_off_max)),
        PNANOVDB_GRID_TYPE_GET(data.grid_type, root_off_ave) - PNANOVDB_GRID_TYPE_GET(data.grid_type, root_off_max)
    );
    rebuild_zero_bytes(
        data, pnanovdb_address_offset(root.address, PNANOVDB_GRID_TYPE_GET(data.grid_type, root_off_ave)),
        PNANOVDB_GRID_TYPE_GET(data.grid_type, root_off_stddev) - PNANOVDB_GRID_TYPE_GET(data.grid_type, root_off_ave)
    );
    rebuild_zero_bytes(
        data, pnanovdb_address_offset(root.address, PNANOVDB_GRID_TYPE_GET(data.grid_type, root_off_stddev)),
        PNANOVDB_GRID_TYPE_GET(data.grid_type, root_size) - PNANOVDB_GRID_TYPE_GET(data.grid_type, root_off_stddev)
    );

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

__global__ void rebuild_build_upper_nodes(RebuildGridData data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t upper_count = data.counts[REBUILD_COUNT_UPPER];
    if (tid >= data.capacities.upper_count || tid >= upper_count)
        return;

    const pnanovdb_root_handle_t root = data.getRoot();
    const pnanovdb_upper_handle_t upper = data.getUpper(tid);
    const nanovdb::Coord ijk = rebuild_upper_key_to_coord(data.upper_keys[tid]);
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
    rebuild_zero_bytes(
        data, pnanovdb_upper_get_min_address(data.grid_type, data.buf, upper),
        PNANOVDB_GRID_TYPE_GET(data.grid_type, upper_off_max) - PNANOVDB_GRID_TYPE_GET(data.grid_type, upper_off_min)
    );
    rebuild_zero_bytes(
        data, pnanovdb_upper_get_max_address(data.grid_type, data.buf, upper),
        PNANOVDB_GRID_TYPE_GET(data.grid_type, upper_off_ave) - PNANOVDB_GRID_TYPE_GET(data.grid_type, upper_off_max)
    );
    rebuild_zero_bytes(
        data, pnanovdb_upper_get_ave_address(data.grid_type, data.buf, upper),
        PNANOVDB_GRID_TYPE_GET(data.grid_type, upper_off_stddev) - PNANOVDB_GRID_TYPE_GET(data.grid_type, upper_off_ave)
    );
    rebuild_zero_bytes(
        data, pnanovdb_upper_get_stddev_address(data.grid_type, data.buf, upper),
        PNANOVDB_GRID_TYPE_GET(data.grid_type, upper_off_table)
            - PNANOVDB_GRID_TYPE_GET(data.grid_type, upper_off_stddev)
    );
}

__global__ void rebuild_set_upper_background_values(RebuildGridData data)
{
    if (!rebuild_is_regular_value_grid(data.grid_type))
        return;

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t upper_count = data.counts[REBUILD_COUNT_UPPER];
    const uint64_t value_count = uint64_t(data.capacities.upper_count) << 15u;
    if (tid >= value_count || (tid >> 15u) >= upper_count)
        return;

    const pnanovdb_upper_handle_t upper = data.getUpper(uint32_t(tid >> 15u));
    rebuild_write_background(
        data, pnanovdb_upper_get_table_address(data.grid_type, data.buf, upper, uint32_t(tid & 32767u))
    );
}

__global__ void rebuild_build_lower_nodes(RebuildGridData data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lower_count = data.counts[REBUILD_COUNT_LOWER];
    const uint32_t upper_count = data.counts[REBUILD_COUNT_UPPER];
    if (tid >= data.capacities.lower_count || tid >= lower_count)
        return;

    const uint64_t lower_key = data.lower_keys[tid];
    const uint32_t upper_id = uint32_t(lower_key >> 15u);
    if (upper_id >= upper_count) {
        rebuild_set_status(data.status, WP_VOLUME_REBUILD_UPPER_CAPACITY_EXCEEDED);
        return;
    }

    const pnanovdb_upper_handle_t upper = data.getUpper(upper_id);
    const pnanovdb_lower_handle_t lower = data.getLower(tid);
    const uint32_t upper_offset = uint32_t(lower_key & 32767u);
    rebuild_set_mask_on_atomic(data, upper.address, PNANOVDB_UPPER_OFF_CHILD_MASK, upper_offset);
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
    rebuild_zero_bytes(
        data, pnanovdb_lower_get_min_address(data.grid_type, data.buf, lower),
        PNANOVDB_GRID_TYPE_GET(data.grid_type, lower_off_max) - PNANOVDB_GRID_TYPE_GET(data.grid_type, lower_off_min)
    );
    rebuild_zero_bytes(
        data, pnanovdb_lower_get_max_address(data.grid_type, data.buf, lower),
        PNANOVDB_GRID_TYPE_GET(data.grid_type, lower_off_ave) - PNANOVDB_GRID_TYPE_GET(data.grid_type, lower_off_max)
    );
    rebuild_zero_bytes(
        data, pnanovdb_lower_get_ave_address(data.grid_type, data.buf, lower),
        PNANOVDB_GRID_TYPE_GET(data.grid_type, lower_off_stddev) - PNANOVDB_GRID_TYPE_GET(data.grid_type, lower_off_ave)
    );
    rebuild_zero_bytes(
        data, pnanovdb_lower_get_stddev_address(data.grid_type, data.buf, lower),
        PNANOVDB_GRID_TYPE_GET(data.grid_type, lower_off_table)
            - PNANOVDB_GRID_TYPE_GET(data.grid_type, lower_off_stddev)
    );
}

__global__ void rebuild_set_lower_background_values(RebuildGridData data)
{
    if (!rebuild_is_regular_value_grid(data.grid_type))
        return;

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lower_count = data.counts[REBUILD_COUNT_LOWER];
    const uint64_t value_count = uint64_t(data.capacities.lower_count) << 12u;
    if (tid >= value_count || (tid >> 12u) >= lower_count)
        return;

    const pnanovdb_lower_handle_t lower = data.getLower(uint32_t(tid >> 12u));
    rebuild_write_background(
        data, pnanovdb_lower_get_table_address(data.grid_type, data.buf, lower, uint32_t(tid & 4095u))
    );
}

__global__ void rebuild_build_leaf_nodes(RebuildGridData data, bool active_voxel_grid)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t leaf_count = data.counts[REBUILD_COUNT_LEAF];
    const uint32_t lower_count = data.counts[REBUILD_COUNT_LOWER];
    if (tid >= data.capacities.leaf_count || tid >= leaf_count)
        return;

    const uint64_t leaf_key = data.leaf_keys[tid];
    const uint64_t lower_key = leaf_key >> 12u;
    const int32_t lower_id = rebuild_find_key_u64(data.lower_keys, lower_count, lower_key);
    if (lower_id < 0) {
        rebuild_set_status(data.status, WP_VOLUME_REBUILD_LOWER_CAPACITY_EXCEEDED);
        return;
    }

    const pnanovdb_lower_handle_t lower = data.getLower(uint32_t(lower_id));
    const pnanovdb_leaf_handle_t leaf = data.getLeaf(tid);
    const uint32_t lower_offset = uint32_t(leaf_key & 4095u);
    rebuild_set_mask_on_atomic(data, lower.address, PNANOVDB_LOWER_OFF_CHILD_MASK, lower_offset);
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
                data.buf, pnanovdb_address_offset(leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK + 8u * word), ~uint64_t(0)
            );
        }
        pnanovdb_leaf_set_bbox_dif_and_flags(data.buf, leaf, 0x00070707u | (2u << 24u));
    } else if (rebuild_is_onindex_grid(data.grid_type)) {
        pnanovdb_write_uint64(data.buf, rebuild_leaf_index_offset_address(data, leaf), 1u);
        pnanovdb_write_uint64(data.buf, rebuild_leaf_index_prefix_address(data, leaf), 0u);
    } else {
        rebuild_zero_bytes(
            data, pnanovdb_leaf_get_min_address(data.grid_type, data.buf, leaf),
            PNANOVDB_GRID_TYPE_GET(data.grid_type, leaf_off_max) - PNANOVDB_GRID_TYPE_GET(data.grid_type, leaf_off_min)
        );
        rebuild_zero_bytes(
            data, pnanovdb_leaf_get_max_address(data.grid_type, data.buf, leaf),
            PNANOVDB_GRID_TYPE_GET(data.grid_type, leaf_off_ave) - PNANOVDB_GRID_TYPE_GET(data.grid_type, leaf_off_max)
        );
        rebuild_zero_bytes(
            data, pnanovdb_leaf_get_ave_address(data.grid_type, data.buf, leaf),
            PNANOVDB_GRID_TYPE_GET(data.grid_type, leaf_off_stddev)
                - PNANOVDB_GRID_TYPE_GET(data.grid_type, leaf_off_ave)
        );
        rebuild_zero_bytes(
            data, pnanovdb_leaf_get_stddev_address(data.grid_type, data.buf, leaf),
            PNANOVDB_GRID_TYPE_GET(data.grid_type, leaf_off_table)
                - PNANOVDB_GRID_TYPE_GET(data.grid_type, leaf_off_stddev)
        );
        if (!active_voxel_grid) {
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

__global__ void rebuild_set_leaf_values(RebuildGridData data)
{
    if (!rebuild_is_regular_value_grid(data.grid_type))
        return;

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t leaf_count = data.counts[REBUILD_COUNT_LEAF];
    const uint64_t value_count = uint64_t(data.capacities.leaf_count) << 9u;
    if (tid >= value_count || (tid >> 9u) >= leaf_count)
        return;

    const pnanovdb_leaf_handle_t leaf = data.getLeaf(uint32_t(tid >> 9u));
    rebuild_write_background(
        data, pnanovdb_leaf_get_table_address(data.grid_type, data.buf, leaf, uint32_t(tid & 511u))
    );
}

__global__ void rebuild_set_active_voxels(RebuildGridData data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t voxel_count = data.counts[REBUILD_COUNT_VOXEL];
    const uint32_t leaf_count = data.counts[REBUILD_COUNT_LEAF];
    if (tid >= data.capacities.voxel_count || tid >= voxel_count)
        return;

    const uint64_t voxel_key = data.voxel_keys[tid];
    const uint64_t leaf_key = voxel_key >> 9u;
    const int32_t leaf_id = rebuild_find_key_u64(data.leaf_keys, leaf_count, leaf_key);
    if (leaf_id < 0) {
        rebuild_set_status(data.status, WP_VOLUME_REBUILD_LEAF_CAPACITY_EXCEEDED);
        return;
    }

    const pnanovdb_leaf_handle_t leaf = data.getLeaf(uint32_t(leaf_id));
    const uint32_t offset = uint32_t(voxel_key & 511u);
    rebuild_set_mask_on_atomic(data, leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK, offset);
    atomicAdd(data.leaf_active_counts + leaf_id, 1u);
}

__global__ void rebuild_finalize_onindex_leaves(RebuildGridData data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t leaf_count = data.counts[REBUILD_COUNT_LEAF];
    if (tid >= data.capacities.leaf_count || tid >= leaf_count)
        return;

    const pnanovdb_leaf_handle_t leaf = data.getLeaf(tid);
    pnanovdb_write_uint64(data.buf, rebuild_leaf_index_offset_address(data, leaf), 1u + data.leaf_active_prefix[tid]);

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

__global__ void rebuild_reset_bboxes(RebuildGridData data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        rebuild_set_invalid_bbox(data, data.getRoot().address);
    }
    if (tid < data.counts[REBUILD_COUNT_LOWER]) {
        rebuild_set_invalid_bbox(data, data.getLower(tid).address);
    }
    if (tid < data.counts[REBUILD_COUNT_UPPER]) {
        rebuild_set_invalid_bbox(data, data.getUpper(tid).address);
    }
}

__global__ void rebuild_propagate_leaf_bboxes(RebuildGridData data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t leaf_count = data.counts[REBUILD_COUNT_LEAF];
    const uint32_t lower_count = data.counts[REBUILD_COUNT_LOWER];
    if (tid >= data.capacities.leaf_count || tid >= leaf_count)
        return;

    const pnanovdb_leaf_handle_t leaf = data.getLeaf(tid);
    if (!rebuild_is_onindex_grid(data.grid_type)) {
        rebuild_update_leaf_bbox(data, leaf);
    }

    const uint64_t lower_key = data.leaf_keys[tid] >> 12u;
    const int32_t lower_id = rebuild_find_key_u64(data.lower_keys, lower_count, lower_key);
    if (lower_id >= 0) {
        rebuild_expand_bbox_atomic(
            data, data.getLower(uint32_t(lower_id)).address, pnanovdb_leaf_get_bbox_min(data.buf, leaf),
            rebuild_leaf_bbox_max(data.buf, leaf)
        );
    }
}

__global__ void rebuild_propagate_lower_bboxes(RebuildGridData data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lower_count = data.counts[REBUILD_COUNT_LOWER];
    const uint32_t upper_count = data.counts[REBUILD_COUNT_UPPER];
    if (tid >= data.capacities.lower_count || tid >= lower_count)
        return;

    const uint32_t upper_id = uint32_t(data.lower_keys[tid] >> 15u);
    if (upper_id < upper_count) {
        const pnanovdb_lower_handle_t lower = data.getLower(tid);
        rebuild_expand_bbox_atomic(
            data, data.getUpper(upper_id).address, pnanovdb_lower_get_bbox_min(data.buf, lower),
            pnanovdb_lower_get_bbox_max(data.buf, lower)
        );
    }
}

__global__ void rebuild_propagate_upper_bboxes(RebuildGridData data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t upper_count = data.counts[REBUILD_COUNT_UPPER];
    if (tid >= data.capacities.upper_count || tid >= upper_count)
        return;

    const pnanovdb_upper_handle_t upper = data.getUpper(tid);
    rebuild_expand_bbox_atomic(
        data, data.getRoot().address, pnanovdb_upper_get_bbox_min(data.buf, upper),
        pnanovdb_upper_get_bbox_max(data.buf, upper)
    );
}

__global__ void rebuild_finalize_world_bbox(RebuildGridData data)
{
    nanovdb::Vec3dBBox world_bbox;
    if (data.counts[REBUILD_COUNT_UPPER] > 0) {
        const pnanovdb_root_handle_t root = data.getRoot();
        const pnanovdb_coord_t root_min = pnanovdb_root_get_bbox_min(data.buf, root);
        const pnanovdb_coord_t root_max = pnanovdb_root_get_bbox_max(data.buf, root);
        const nanovdb::CoordBBox index_bbox(rebuild_make_coord(root_min), rebuild_make_coord(root_max));
        world_bbox = index_bbox.transform(data.map);
    }
    const pnanovdb_grid_handle_t grid = data.getGrid();
    pnanovdb_grid_set_world_bbox(data.buf, grid, 0u, world_bbox[0][0]);
    pnanovdb_grid_set_world_bbox(data.buf, grid, 1u, world_bbox[0][1]);
    pnanovdb_grid_set_world_bbox(data.buf, grid, 2u, world_bbox[0][2]);
    pnanovdb_grid_set_world_bbox(data.buf, grid, 3u, world_bbox[1][0]);
    pnanovdb_grid_set_world_bbox(data.buf, grid, 4u, world_bbox[1][1]);
    pnanovdb_grid_set_world_bbox(data.buf, grid, 5u, world_bbox[1][2]);
}

template <typename BuildT>
RebuildGridData make_rebuild_data(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>* grid,
    size_t grid_size,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params,
    uint64_t* leaf_keys,
    uint64_t* lower_keys,
    uint64_t* upper_keys,
    uint64_t* voxel_keys,
    uint32_t* counts,
    uint32_t* leaf_active_counts,
    uint64_t* leaf_active_prefix,
    uint32_t* status
)
{
    RebuildGridData data = make_rebuild_data_base<BuildT, RebuildGridData>(grid, grid_size, capacities, params);
    data.leaf_keys = leaf_keys;
    data.lower_keys = lower_keys;
    data.upper_keys = upper_keys;
    data.voxel_keys = voxel_keys;
    data.counts = counts;
    data.leaf_active_counts = leaf_active_counts;
    data.leaf_active_prefix = leaf_active_prefix;
    data.status = status;
    return data;
}

void rebuild_sort_keys(uint64_t* keys_in, uint64_t* keys_out, int count, cudaStream_t stream)
{
    size_t temp_size = 0;
    check_cuda(cub::DeviceRadixSort::SortKeys(nullptr, temp_size, keys_in, keys_out, count, 0, 64, stream));
    ScopedTemporary<> temp(WP_CURRENT_CONTEXT, temp_size);
    check_cuda(cub::DeviceRadixSort::SortKeys(temp.buffer(), temp_size, keys_in, keys_out, count, 0, 64, stream));
}

template <typename InputIt>
void rebuild_encode_runs(
    InputIt keys_in, uint64_t* unique_keys, uint32_t* run_counts, uint32_t* run_count, int count, cudaStream_t stream
)
{
    size_t temp_size = 0;
    check_cuda(
        cub::DeviceRunLengthEncode::Encode(
            nullptr, temp_size, keys_in, unique_keys, run_counts, run_count, count, stream
        )
    );
    ScopedTemporary<> temp(WP_CURRENT_CONTEXT, temp_size);
    check_cuda(
        cub::DeviceRunLengthEncode::Encode(
            temp.buffer(), temp_size, keys_in, unique_keys, run_counts, run_count, count, stream
        )
    );
}

template <typename T> void rebuild_exclusive_sum(T* in, T* out, int count, cudaStream_t stream)
{
    size_t temp_size = 0;
    check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, temp_size, in, out, count, stream));
    ScopedTemporary<> temp(WP_CURRENT_CONTEXT, temp_size);
    check_cuda(cub::DeviceScan::ExclusiveSum(temp.buffer(), temp_size, in, out, count, stream));
}

void rebuild_exclusive_sum_u32_to_u64(uint32_t* in, uint64_t* out, int count, cudaStream_t stream)
{
    size_t temp_size = 0;
    check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, temp_size, in, out, count, stream));
    ScopedTemporary<> temp(WP_CURRENT_CONTEXT, temp_size);
    check_cuda(cub::DeviceScan::ExclusiveSum(temp.buffer(), temp_size, in, out, count, stream));
}

template <typename PtrT>
bool rebuild_count_points(
    RebuildKeyScratch& scratch,
    const PtrT points,
    size_t num_points,
    const int32_t* point_mask,
    bool active_voxel_grid,
    const VolumeRebuildCapacities& capacities,
    const nanovdb::Map& map,
    bool enforce_capacities,
    uint32_t* status,
    cudaStream_t stream
)
{
    if (!rebuild_alloc_key_scratch(scratch, num_points, active_voxel_grid, status, stream)) {
        return false;
    }

    rebuild_emit_upper_keys<<<rebuild_num_blocks(num_points), REBUILD_NUM_THREADS, 0, stream>>>(
        num_points, points, point_mask, map, scratch.keys_a
    );
    check_cuda(cudaGetLastError());

    rebuild_sort_keys(scratch.keys_a, scratch.keys_b, scratch.point_count, stream);
    rebuild_encode_runs(
        scratch.keys_b, scratch.upper_keys, scratch.run_counts, scratch.counts + REBUILD_COUNT_UPPER,
        scratch.point_count, stream
    );

    rebuild_emit_hierarchy_keys<<<rebuild_num_blocks(num_points), REBUILD_NUM_THREADS, 0, stream>>>(
        num_points, points, point_mask, map, scratch.upper_keys, scratch.counts, scratch.keys_a, status
    );
    check_cuda(cudaGetLastError());

    rebuild_sort_keys(scratch.keys_a, scratch.keys_b, scratch.point_count, stream);
    if (active_voxel_grid) {
        rebuild_encode_runs(
            scratch.keys_b, scratch.voxel_keys, scratch.run_counts, scratch.counts + REBUILD_COUNT_VOXEL,
            scratch.point_count, stream
        );
    }

    rebuild_shift_keys<9u><<<rebuild_num_blocks(num_points), REBUILD_NUM_THREADS, 0, stream>>>(
        scratch.keys_b, scratch.parent_keys, num_points
    );
    rebuild_encode_runs(
        scratch.parent_keys, scratch.leaf_keys, scratch.run_counts, scratch.counts + REBUILD_COUNT_LEAF,
        scratch.point_count, stream
    );

    rebuild_shift_keys<21u><<<rebuild_num_blocks(num_points), REBUILD_NUM_THREADS, 0, stream>>>(
        scratch.keys_b, scratch.parent_keys, num_points
    );
    rebuild_encode_runs(
        scratch.parent_keys, scratch.lower_keys, scratch.run_counts, scratch.counts + REBUILD_COUNT_LOWER,
        scratch.point_count, stream
    );
    check_cuda(cudaGetLastError());

    rebuild_finalize_counts<<<1, 1, 0, stream>>>(
        scratch.leaf_keys, scratch.lower_keys, scratch.upper_keys, scratch.voxel_keys, scratch.counts, status,
        capacities, active_voxel_grid, enforce_capacities
    );
    check_cuda(cudaGetLastError());

    return true;
}

VolumeRebuildCapacities rebuild_copy_exact_capacities(const RebuildKeyScratch& scratch, cudaStream_t stream)
{
    uint32_t counts[REBUILD_COUNT_SLOT_COUNT] = {};
    wp_memcpy_d2h(WP_CURRENT_CONTEXT, counts, scratch.counts, sizeof(counts));
    wp_cuda_stream_synchronize(stream);

    VolumeRebuildCapacities capacities;
    capacities.leaf_count = counts[REBUILD_COUNT_LEAF];
    capacities.lower_count = counts[REBUILD_COUNT_LOWER];
    capacities.upper_count = counts[REBUILD_COUNT_UPPER];
    capacities.voxel_count = scratch.active_voxel_grid ? counts[REBUILD_COUNT_VOXEL]
                                                       : uint64_t(capacities.leaf_count) * PNANOVDB_LEAF_TABLE_COUNT;
    return capacities;
}

template <typename BuildT>
void rebuild_populate_grid_from_scratch(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>* grid,
    size_t grid_size,
    const RebuildKeyScratch& scratch,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params,
    uint32_t* status,
    cudaStream_t stream
)
{
    uint32_t* leaf_active_counts = nullptr;
    uint64_t* leaf_active_prefix = nullptr;
    if (scratch.active_voxel_grid && capacities.leaf_count > 0) {
        leaf_active_counts = static_cast<uint32_t*>(
            wp_alloc_device(WP_CURRENT_CONTEXT, capacities.leaf_count * sizeof(uint32_t), "(native:volume_builder)")
        );
        leaf_active_prefix = static_cast<uint64_t*>(
            wp_alloc_device(WP_CURRENT_CONTEXT, capacities.leaf_count * sizeof(uint64_t), "(native:volume_builder)")
        );
        check_cuda(cudaMemsetAsync(leaf_active_counts, 0, capacities.leaf_count * sizeof(uint32_t), stream));
    }

    RebuildGridData data = make_rebuild_data(
        grid, grid_size, capacities, params, scratch.leaf_keys, scratch.lower_keys, scratch.upper_keys,
        scratch.voxel_keys, scratch.counts, leaf_active_counts, leaf_active_prefix, status
    );

    rebuild_init_grid_tree_root<<<1, 1, 0, stream>>>(data);
    if (capacities.upper_count > 0) {
        rebuild_build_upper_nodes<<<rebuild_num_blocks(capacities.upper_count), REBUILD_NUM_THREADS, 0, stream>>>(data);
        rebuild_set_upper_background_values<<<
            rebuild_num_blocks(uint64_t(capacities.upper_count) << 15u), REBUILD_NUM_THREADS, 0, stream>>>(data);
    }
    if (capacities.lower_count > 0) {
        rebuild_build_lower_nodes<<<rebuild_num_blocks(capacities.lower_count), REBUILD_NUM_THREADS, 0, stream>>>(data);
        rebuild_set_lower_background_values<<<
            rebuild_num_blocks(uint64_t(capacities.lower_count) << 12u), REBUILD_NUM_THREADS, 0, stream>>>(data);
    }
    if (capacities.leaf_count > 0) {
        rebuild_build_leaf_nodes<<<rebuild_num_blocks(capacities.leaf_count), REBUILD_NUM_THREADS, 0, stream>>>(
            data, scratch.active_voxel_grid
        );
    }
    check_cuda(cudaGetLastError());

    if (scratch.active_voxel_grid && data.grid_type == PNANOVDB_GRID_TYPE_ONINDEX && capacities.leaf_count > 0) {
        if (capacities.voxel_count > 0) {
            rebuild_set_active_voxels<<<rebuild_num_blocks(capacities.voxel_count), REBUILD_NUM_THREADS, 0, stream>>>(
                data
            );
        }
        rebuild_exclusive_sum_u32_to_u64(leaf_active_counts, leaf_active_prefix, int(capacities.leaf_count), stream);
        rebuild_finalize_onindex_leaves<<<rebuild_num_blocks(capacities.leaf_count), REBUILD_NUM_THREADS, 0, stream>>>(
            data
        );
    } else if (capacities.leaf_count > 0) {
        rebuild_set_leaf_values<<<
            rebuild_num_blocks(uint64_t(capacities.leaf_count) << 9u), REBUILD_NUM_THREADS, 0, stream>>>(data);
    }

    const uint32_t bbox_capacity
        = std::max(std::max(capacities.leaf_count, capacities.lower_count), capacities.upper_count);
    rebuild_reset_bboxes<<<rebuild_num_blocks(std::max(1u, bbox_capacity)), REBUILD_NUM_THREADS, 0, stream>>>(data);
    if (capacities.leaf_count > 0) {
        rebuild_propagate_leaf_bboxes<<<rebuild_num_blocks(capacities.leaf_count), REBUILD_NUM_THREADS, 0, stream>>>(
            data
        );
    }
    if (capacities.lower_count > 0) {
        rebuild_propagate_lower_bboxes<<<rebuild_num_blocks(capacities.lower_count), REBUILD_NUM_THREADS, 0, stream>>>(
            data
        );
    }
    if (capacities.upper_count > 0) {
        rebuild_propagate_upper_bboxes<<<rebuild_num_blocks(capacities.upper_count), REBUILD_NUM_THREADS, 0, stream>>>(
            data
        );
    }
    rebuild_finalize_world_bbox<<<1, 1, 0, stream>>>(data);
    check_cuda(cudaGetLastError());

    if (leaf_active_counts)
        wp_free_device(WP_CURRENT_CONTEXT, leaf_active_counts);
    if (leaf_active_prefix)
        wp_free_device(WP_CURRENT_CONTEXT, leaf_active_prefix);
}

template <typename BuildT, typename PtrT>
void rebuild_grid_from_points_impl(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>* grid,
    size_t grid_size,
    const PtrT points,
    size_t num_points,
    const int32_t* point_mask,
    bool active_voxel_grid,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params,
    uint32_t* status
)
{
    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());
    if (capacities.leaf_count == 0 || capacities.lower_count == 0 || capacities.upper_count == 0) {
        if (status) {
            check_cuda(cudaMemsetAsync(status, 0, sizeof(uint32_t), stream));
            rebuild_write_status<<<1, 1, 0, stream>>>(status, WP_VOLUME_REBUILD_INVALID_INPUT);
            check_cuda(cudaGetLastError());
        }
        return;
    }

    RebuildKeyScratch scratch;
    if (!rebuild_count_points(
            scratch, points, num_points, point_mask, active_voxel_grid, capacities, params.map, true, status, stream
        )) {
        return;
    }

    rebuild_populate_grid_from_scratch(grid, grid_size, scratch, capacities, params, status, stream);
    rebuild_free_key_scratch(scratch);
}

template <typename BuildT, typename PtrT>
void allocate_exact_grid_from_points_impl(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>*& out_grid,
    size_t& out_grid_size,
    const PtrT points,
    size_t num_points,
    const int32_t* point_mask,
    bool active_voxel_grid,
    const BuildGridParams<BuildT>& params
)
{
    out_grid = nullptr;
    out_grid_size = 0;

    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());
    RebuildKeyScratch scratch;
    VolumeRebuildCapacities empty_capacities;
    if (!rebuild_count_points(
            scratch, points, num_points, point_mask, active_voxel_grid, empty_capacities, params.map, false, nullptr,
            stream
        )) {
        return;
    }

    const VolumeRebuildCapacities capacities = rebuild_copy_exact_capacities(scratch, stream);
    out_grid_size = rebuildable_grid_size<BuildT>(capacities);
    out_grid = static_cast<nanovdb::Grid<nanovdb::NanoTree<BuildT>>*>(
        wp_alloc_device(WP_CURRENT_CONTEXT, out_grid_size, "(native:volume_builder)")
    );
    rebuild_populate_grid_from_scratch(out_grid, out_grid_size, scratch, capacities, params, nullptr, stream);
    rebuild_free_key_scratch(scratch);
}

}  // namespace

template <typename BuildT>
void allocate_grid_from_tiles(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const BuildGridParams<BuildT>& params
)
{
    if (points_in_world_space) {
        allocate_exact_grid_from_points_impl(
            out_grid, out_grid_size, static_cast<const nanovdb::Vec3f*>(points), num_points, point_mask, false, params
        );
    } else {
        allocate_exact_grid_from_points_impl(
            out_grid, out_grid_size, static_cast<const nanovdb::Coord*>(points), num_points, point_mask, false, params
        );
    }
}

void allocate_grid_from_active_voxels(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const int32_t* point_mask,
    const BuildGridParams<nanovdb::ValueOnIndex>& params
)
{
    if (points_in_world_space) {
        allocate_exact_grid_from_points_impl(
            out_grid, out_grid_size, static_cast<const nanovdb::Vec3f*>(points), num_points, point_mask, true, params
        );
    } else {
        allocate_exact_grid_from_points_impl(
            out_grid, out_grid_size, static_cast<const nanovdb::Coord*>(points), num_points, point_mask, true, params
        );
    }
}

#define EXPAND_BUILDER_TYPE(type)                                                                                      \
    template void allocate_grid_from_tiles(                                                                            \
        nanovdb::Grid<nanovdb::NanoTree<type>>*&, size_t&, const void*, size_t, bool, const int32_t*,                  \
        const BuildGridParams<type>&                                                                                    \
    );

WP_VOLUME_BUILDER_INSTANTIATE_TYPES
#undef EXPAND_BUILDER_TYPE

template void allocate_grid_from_tiles(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueIndex>>*&,
    size_t&,
    const void*,
    size_t,
    bool,
    const int32_t*,
    const BuildGridParams<nanovdb::ValueIndex>&
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
)
{
    if (points_in_world_space) {
        rebuild_grid_from_points_impl(
            grid, grid_size, static_cast<const nanovdb::Vec3f*>(points), num_points, point_mask, false, capacities,
            params, status
        );
    } else {
        rebuild_grid_from_points_impl(
            grid, grid_size, static_cast<const nanovdb::Coord*>(points), num_points, point_mask, false, capacities,
            params, status
        );
    }
}

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
)
{
    out_grid_size = rebuildable_grid_size<BuildT>(capacities);
    out_grid = static_cast<nanovdb::Grid<nanovdb::NanoTree<BuildT>>*>(
        wp_alloc_device(WP_CURRENT_CONTEXT, out_grid_size, "(native:volume_builder)")
    );
    rebuild_grid_from_tiles(
        out_grid, out_grid_size, points, num_points, points_in_world_space, point_mask, capacities, params, status
    );
}

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
)
{
    if (points_in_world_space) {
        rebuild_grid_from_points_impl(
            grid, grid_size, static_cast<const nanovdb::Vec3f*>(points), num_points, point_mask, true, capacities,
            params, status
        );
    } else {
        rebuild_grid_from_points_impl(
            grid, grid_size, static_cast<const nanovdb::Coord*>(points), num_points, point_mask, true, capacities,
            params, status
        );
    }
}

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
)
{
    out_grid_size = rebuildable_grid_size<nanovdb::ValueOnIndex>(capacities);
    out_grid = static_cast<nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>*>(
        wp_alloc_device(WP_CURRENT_CONTEXT, out_grid_size, "(native:volume_builder)")
    );
    rebuild_grid_from_active_voxels(
        out_grid, out_grid_size, points, num_points, points_in_world_space, point_mask, capacities, params, status
    );
}

#define EXPAND_BUILDER_TYPE(type)                                                                                      \
    template void allocate_rebuildable_grid_from_tiles(                                                                \
        nanovdb::Grid<nanovdb::NanoTree<type>>*&, size_t&, const void*, size_t, bool, const int32_t*,                  \
        const VolumeRebuildCapacities&, const BuildGridParams<type>&, uint32_t*                                        \
    );                                                                                                                 \
    template void rebuild_grid_from_tiles(                                                                             \
        nanovdb::Grid<nanovdb::NanoTree<type>>*, size_t, const void*, size_t, bool, const int32_t*,                    \
        const VolumeRebuildCapacities&, const BuildGridParams<type>&, uint32_t*                                        \
    );

WP_VOLUME_BUILDER_INSTANTIATE_TYPES
#undef EXPAND_BUILDER_TYPE

template void allocate_rebuildable_grid_from_tiles(
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
template void rebuild_grid_from_tiles(
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
