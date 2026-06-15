// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "cuda_util.h"
#include "volume_builder.h"

#include <algorithm>
#include <limits>

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace {

template <typename Node>
__device__ std::enable_if_t<!nanovdb::BuildTraits<typename Node::BuildType>::is_index>
setBackgroundValue(Node& node, unsigned tile_id, const typename Node::BuildType background_value)
{
    node.setValue(tile_id, background_value);
}

template <typename Node>
__device__ std::enable_if_t<nanovdb::BuildTraits<typename Node::BuildType>::is_index>
setBackgroundValue(Node& node, unsigned tile_id, const typename Node::BuildType background_value)
{
}

template <typename Node>
__device__ std::enable_if_t<!nanovdb::BuildTraits<typename Node::BuildType>::is_index>
setBackgroundValue(Node& node, const typename Node::BuildType background_value)
{
    node.mBackground = background_value;
}

template <typename Node>
__device__ std::enable_if_t<nanovdb::BuildTraits<typename Node::BuildType>::is_index>
setBackgroundValue(Node& node, const typename Node::BuildType background_value)
{
}

static constexpr uint64_t REBUILD_INVALID_KEY = ~uint64_t(0);
static constexpr unsigned REBUILD_NUM_THREADS = 128;

inline unsigned rebuild_num_blocks(uint64_t n) { return unsigned((n + REBUILD_NUM_THREADS - 1) / REBUILD_NUM_THREADS); }

enum RebuildCountSlot : uint32_t {
    REBUILD_COUNT_LEAF = 0,
    REBUILD_COUNT_LOWER = 1,
    REBUILD_COUNT_UPPER = 2,
    REBUILD_COUNT_VOXEL = 3,
    REBUILD_COUNT_SLOT_COUNT = 4,
};

template <typename BuildT> struct RebuildGridData {
    void* buffer;
    uint64_t size;
    uint64_t grid, tree, root, upper, lower, leaf;

    VolumeRebuildCapacities capacities;
    uint64_t* leaf_keys;
    uint64_t* lower_keys;
    uint64_t* upper_keys;
    uint64_t* voxel_keys;
    uint32_t* counts;
    uint32_t* leaf_active_counts;
    uint64_t* leaf_active_prefix;
    uint32_t* status;

    nanovdb::Map map;
    typename nanovdb::NanoTree<BuildT>::BuildType background_value;

    __hostdev__ nanovdb::Grid<nanovdb::NanoTree<BuildT>>& getGrid() const
    {
        return *nanovdb::util::PtrAdd<nanovdb::Grid<nanovdb::NanoTree<BuildT>>>(buffer, grid);
    }
    __hostdev__ nanovdb::NanoTree<BuildT>& getTree() const
    {
        return *nanovdb::util::PtrAdd<nanovdb::NanoTree<BuildT>>(buffer, tree);
    }
    __hostdev__ nanovdb::NanoRoot<BuildT>& getRoot() const
    {
        return *nanovdb::util::PtrAdd<nanovdb::NanoRoot<BuildT>>(buffer, root);
    }
    __hostdev__ nanovdb::NanoUpper<BuildT>& getUpper(uint32_t i) const
    {
        return *(nanovdb::util::PtrAdd<nanovdb::NanoUpper<BuildT>>(buffer, upper) + i);
    }
    __hostdev__ nanovdb::NanoLower<BuildT>& getLower(uint32_t i) const
    {
        return *(nanovdb::util::PtrAdd<nanovdb::NanoLower<BuildT>>(buffer, lower) + i);
    }
    __hostdev__ nanovdb::NanoLeaf<BuildT>& getLeaf(uint32_t i) const
    {
        return *(nanovdb::util::PtrAdd<nanovdb::NanoLeaf<BuildT>>(buffer, leaf) + i);
    }
};

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

__hostdev__ inline uint64_t rebuild_upper_key_from_coord(const nanovdb::Coord& ijk)
{
    // Match NanoVDB's upper-root-tile key ordering.
    static constexpr int64_t kOffset = int64_t(1) << 31;
    return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12))
        | (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21)
        | (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42);
}

__hostdev__ inline nanovdb::Coord rebuild_upper_key_to_coord(uint64_t key)
{
    // Match NanoVDB's upper-node origin decoding.
    static constexpr int64_t kOffset = int64_t(1) << 31;
    static constexpr uint64_t mask = (uint64_t(1) << 21) - 1u;
    return nanovdb::Coord(
        int(int64_t(((key >> 42) & mask) << 12) - kOffset), int(int64_t(((key >> 21) & mask) << 12) - kOffset),
        int(int64_t((key & mask) << 12) - kOffset)
    );
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

template <typename BuildT>
__hostdev__ inline uint64_t rebuild_hierarchy_key(uint32_t upper_id, const nanovdb::Coord& ijk)
{
    return (uint64_t(upper_id) << 36) | (uint64_t(nanovdb::NanoUpper<BuildT>::CoordToOffset(ijk)) << 21)
        | (uint64_t(nanovdb::NanoLower<BuildT>::CoordToOffset(ijk)) << 9)
        | uint64_t(nanovdb::NanoLeaf<BuildT>::CoordToOffset(ijk));
}

template <typename PtrT>
__global__ void rebuild_emit_upper_keys(size_t point_count, const PtrT points, nanovdb::Map map, uint64_t* keys)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= point_count)
        return;

    keys[tid] = rebuild_upper_key_from_coord(rebuild_point_to_coord(points, tid, map));
}

template <typename BuildT, typename PtrT>
__global__ void rebuild_emit_hierarchy_keys(
    size_t point_count,
    const PtrT points,
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

    const nanovdb::Coord ijk = rebuild_point_to_coord(points, tid, map);
    const uint64_t upper_key = rebuild_upper_key_from_coord(ijk);
    const int32_t upper_id = rebuild_find_key_u64(upper_keys, counts[REBUILD_COUNT_UPPER], upper_key);
    if (upper_id < 0 || upper_id >= (1 << 28)) {
        keys[tid] = REBUILD_INVALID_KEY;
        rebuild_set_status(status, WP_VOLUME_REBUILD_INVALID_INPUT);
        return;
    }

    keys[tid] = rebuild_hierarchy_key<BuildT>(uint32_t(upper_id), ijk);
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

template <typename BuildT> __device__ int32_t rebuild_find_key(const uint64_t* keys, uint32_t count, uint64_t key)
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

template <typename BuildT>
__global__ void rebuild_init_grid_tree_root(RebuildGridData<BuildT> data, bool active_voxel_grid)
{
    const uint32_t leaf_count = data.counts[REBUILD_COUNT_LEAF];
    const uint32_t lower_count = data.counts[REBUILD_COUNT_LOWER];
    const uint32_t upper_count = data.counts[REBUILD_COUNT_UPPER];
    const uint32_t voxel_count = data.counts[REBUILD_COUNT_VOXEL];

    auto& root = data.getRoot();
    root.mBBox = typename nanovdb::NanoRoot<BuildT>::BBoxType();
    root.mTableSize = upper_count;
    setBackgroundValue(root, data.background_value);
    root.mMinimum = root.mMaximum = typename nanovdb::NanoRoot<BuildT>::ValueType(0);
    root.mAverage = root.mStdDevi = typename nanovdb::NanoRoot<BuildT>::FloatType(0);

    auto& tree = data.getTree();
    tree.setRoot(&root);
    tree.setFirstNode(&data.getUpper(0));
    tree.setFirstNode(&data.getLower(0));
    tree.setFirstNode(&data.getLeaf(0));
    tree.mNodeCount[2] = tree.mTileCount[2] = upper_count;
    tree.mNodeCount[1] = tree.mTileCount[1] = lower_count;
    tree.mNodeCount[0] = tree.mTileCount[0] = leaf_count;
    tree.mVoxelCount = voxel_count;

    auto& grid = data.getGrid();
    grid.init(
        { nanovdb::GridFlags::HasBBox, nanovdb::GridFlags::IsBreadthFirst }, data.size, data.map,
        nanovdb::toGridType<BuildT>()
    );
    grid.mChecksum = ~uint64_t(0);
    grid.mBlindMetadataCount = 0;
    grid.mBlindMetadataOffset = 0;
    grid.mGridName[0] = '\0';
    if constexpr (nanovdb::BuildTraits<BuildT>::is_index) {
        grid.mGridClass = nanovdb::GridClass::IndexGrid;
        grid.mData1 = uint64_t(voxel_count) + 1u;
    }
    grid.mWorldBBox = root.mBBox.transform(grid.map());
}

template <typename BuildT> __global__ void rebuild_build_upper_nodes(RebuildGridData<BuildT> data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t upper_count = data.counts[REBUILD_COUNT_UPPER];
    if (tid >= data.capacities.upper_count || tid >= upper_count)
        return;

    auto& root = data.getRoot();
    auto& upper = data.getUpper(tid);
    const nanovdb::Coord ijk = rebuild_upper_key_to_coord(data.upper_keys[tid]);

    root.tile(tid)->setChild(ijk, &upper, &root);
    upper.mBBox = nanovdb::CoordBBox();
    upper.mBBox[0] = ijk;
    upper.mFlags = 0;
    upper.mValueMask.setOff();
    upper.mChildMask.setOff();
    upper.mMinimum = upper.mMaximum = typename nanovdb::NanoUpper<BuildT>::ValueType(0);
    upper.mAverage = upper.mStdDevi = typename nanovdb::NanoUpper<BuildT>::FloatType(0);
}

template <typename BuildT> __global__ void rebuild_set_upper_background_values(RebuildGridData<BuildT> data)
{
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t upper_count = data.counts[REBUILD_COUNT_UPPER];
    const uint64_t value_count = uint64_t(data.capacities.upper_count) << 15u;
    if (tid >= value_count || (tid >> 15u) >= upper_count)
        return;

    auto& upper = data.getUpper(uint32_t(tid >> 15u));
    setBackgroundValue(upper, uint32_t(tid & 32767u), data.background_value);
}

template <typename BuildT> __global__ void rebuild_build_lower_nodes(RebuildGridData<BuildT> data)
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

    auto& upper = data.getUpper(upper_id);
    const uint32_t upper_offset = uint32_t(lower_key & 32767u);
    upper.mChildMask.setOnAtomic(upper_offset);

    auto& lower = data.getLower(tid);
    upper.setChild(upper_offset, &lower);
    lower.mBBox = nanovdb::CoordBBox();
    lower.mBBox[0] = upper.offsetToGlobalCoord(upper_offset);
    lower.mFlags = 0;
    lower.mValueMask.setOff();
    lower.mChildMask.setOff();
    lower.mMinimum = lower.mMaximum = typename nanovdb::NanoLower<BuildT>::ValueType(0);
    lower.mAverage = lower.mStdDevi = typename nanovdb::NanoLower<BuildT>::FloatType(0);
}

template <typename BuildT> __global__ void rebuild_set_lower_background_values(RebuildGridData<BuildT> data)
{
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lower_count = data.counts[REBUILD_COUNT_LOWER];
    const uint64_t value_count = uint64_t(data.capacities.lower_count) << 12u;
    if (tid >= value_count || (tid >> 12u) >= lower_count)
        return;

    auto& lower = data.getLower(uint32_t(tid >> 12u));
    setBackgroundValue(lower, uint32_t(tid & 4095u), data.background_value);
}

template <typename BuildT>
__global__ void rebuild_build_leaf_nodes(RebuildGridData<BuildT> data, bool active_voxel_grid)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t leaf_count = data.counts[REBUILD_COUNT_LEAF];
    const uint32_t lower_count = data.counts[REBUILD_COUNT_LOWER];
    if (tid >= data.capacities.leaf_count || tid >= leaf_count)
        return;

    const uint64_t leaf_key = data.leaf_keys[tid];
    const uint64_t lower_key = leaf_key >> 12u;
    const int32_t lower_id = rebuild_find_key<BuildT>(data.lower_keys, lower_count, lower_key);
    if (lower_id < 0) {
        rebuild_set_status(data.status, WP_VOLUME_REBUILD_LOWER_CAPACITY_EXCEEDED);
        return;
    }

    auto& lower = data.getLower(uint32_t(lower_id));
    const uint32_t lower_offset = uint32_t(leaf_key & 4095u);
    lower.mChildMask.setOnAtomic(lower_offset);

    auto& leaf = data.getLeaf(tid);
    lower.setChild(lower_offset, &leaf);
    leaf.mBBoxMin = lower.offsetToGlobalCoord(lower_offset);
    leaf.mBBoxDif[0] = leaf.mBBoxDif[1] = leaf.mBBoxDif[2] = 0;
    leaf.mFlags = uint8_t(2u);
    leaf.mValueMask.setOff();

    if constexpr (nanovdb::BuildTraits<BuildT>::is_offindex) {
        leaf.mOffset = uint64_t(tid) * PNANOVDB_LEAF_TABLE_COUNT + 1u;
        leaf.mPrefixSum = 0u;
        leaf.mValueMask.setOn();
        leaf.mBBoxDif[0] = leaf.mBBoxDif[1] = leaf.mBBoxDif[2] = 7u;
    } else if constexpr (nanovdb::BuildTraits<BuildT>::is_onindex) {
        leaf.mOffset = 1u;
        leaf.mPrefixSum = 0u;
    } else if constexpr (!nanovdb::BuildTraits<BuildT>::is_special) {
        leaf.mAverage = leaf.mStdDevi = typename nanovdb::NanoLeaf<BuildT>::FloatType(0);
        leaf.mMinimum = leaf.mMaximum = typename nanovdb::NanoLeaf<BuildT>::ValueType(0);
        if (!active_voxel_grid) {
            leaf.mValueMask.setOn();
            leaf.mBBoxDif[0] = leaf.mBBoxDif[1] = leaf.mBBoxDif[2] = 7u;
        }
    }
}

template <typename BuildT> __global__ void rebuild_set_leaf_values(RebuildGridData<BuildT> data)
{
    if constexpr (!nanovdb::BuildTraits<BuildT>::is_special) {
        const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t leaf_count = data.counts[REBUILD_COUNT_LEAF];
        const uint64_t value_count = uint64_t(data.capacities.leaf_count) << 9u;
        if (tid >= value_count || (tid >> 9u) >= leaf_count)
            return;

        auto& leaf = data.getLeaf(uint32_t(tid >> 9u));
        leaf.mValues[tid & 511u] = data.background_value;
    }
}

template <typename BuildT> __global__ void rebuild_set_active_voxels(RebuildGridData<BuildT> data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t voxel_count = data.counts[REBUILD_COUNT_VOXEL];
    const uint32_t leaf_count = data.counts[REBUILD_COUNT_LEAF];
    if (tid >= data.capacities.voxel_count || tid >= voxel_count)
        return;

    const uint64_t voxel_key = data.voxel_keys[tid];
    const uint64_t leaf_key = voxel_key >> 9u;
    const int32_t leaf_id = rebuild_find_key<BuildT>(data.leaf_keys, leaf_count, leaf_key);
    if (leaf_id < 0) {
        rebuild_set_status(data.status, WP_VOLUME_REBUILD_LEAF_CAPACITY_EXCEEDED);
        return;
    }

    auto& leaf = data.getLeaf(uint32_t(leaf_id));
    const uint32_t offset = uint32_t(voxel_key & 511u);
    leaf.mValueMask.setOnAtomic(offset);
    atomicAdd(data.leaf_active_counts + leaf_id, 1u);
}

template <typename BuildT> __global__ void rebuild_finalize_onindex_leaves(RebuildGridData<BuildT> data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t leaf_count = data.counts[REBUILD_COUNT_LEAF];
    if (tid >= data.capacities.leaf_count || tid >= leaf_count)
        return;

    auto& leaf = data.getLeaf(tid);
    leaf.mOffset = 1u + data.leaf_active_prefix[tid];

    const uint64_t* words = leaf.mValueMask.words();
    uint64_t prefix = nanovdb::util::countOn(words[0]);
    uint64_t sum = prefix;
    for (int n = 9; n < 55; n += 9) {
        sum += nanovdb::util::countOn(words[n / 9]);
        prefix |= sum << n;
    }
    leaf.mPrefixSum = prefix;
    leaf.updateBBox();
}

template <typename BuildT> __global__ void rebuild_reset_bboxes(RebuildGridData<BuildT> data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        data.getRoot().mBBox = typename nanovdb::NanoRoot<BuildT>::BBoxType();
    }
    if (tid < data.counts[REBUILD_COUNT_LOWER]) {
        data.getLower(tid).mBBox = nanovdb::CoordBBox();
    }
    if (tid < data.counts[REBUILD_COUNT_UPPER]) {
        data.getUpper(tid).mBBox = nanovdb::CoordBBox();
    }
}

template <typename BuildT> __global__ void rebuild_propagate_leaf_bboxes(RebuildGridData<BuildT> data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t leaf_count = data.counts[REBUILD_COUNT_LEAF];
    const uint32_t lower_count = data.counts[REBUILD_COUNT_LOWER];
    if (tid >= data.capacities.leaf_count || tid >= leaf_count)
        return;

    auto& leaf = data.getLeaf(tid);
    if constexpr (!nanovdb::BuildTraits<BuildT>::is_onindex) {
        leaf.updateBBox();
    }

    const uint64_t lower_key = data.leaf_keys[tid] >> 12u;
    const int32_t lower_id = rebuild_find_key<BuildT>(data.lower_keys, lower_count, lower_key);
    if (lower_id >= 0) {
        data.getLower(uint32_t(lower_id)).mBBox.expandAtomic(leaf.bbox());
    }
}

template <typename BuildT> __global__ void rebuild_propagate_lower_bboxes(RebuildGridData<BuildT> data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lower_count = data.counts[REBUILD_COUNT_LOWER];
    const uint32_t upper_count = data.counts[REBUILD_COUNT_UPPER];
    if (tid >= data.capacities.lower_count || tid >= lower_count)
        return;

    const uint32_t upper_id = uint32_t(data.lower_keys[tid] >> 15u);
    if (upper_id < upper_count) {
        data.getUpper(upper_id).mBBox.expandAtomic(data.getLower(tid).bbox());
    }
}

template <typename BuildT> __global__ void rebuild_propagate_upper_bboxes(RebuildGridData<BuildT> data)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t upper_count = data.counts[REBUILD_COUNT_UPPER];
    if (tid >= data.capacities.upper_count || tid >= upper_count)
        return;

    data.getRoot().mBBox.expandAtomic(data.getUpper(tid).bbox());
}

template <typename BuildT> __global__ void rebuild_finalize_world_bbox(RebuildGridData<BuildT> data)
{
    data.getGrid().mWorldBBox = data.getRoot().mBBox.transform(data.map);
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

template <typename BuildT>
RebuildGridData<BuildT> make_rebuild_data(
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
    RebuildGridData<BuildT> data;
    data.buffer = grid;
    data.size = grid_size;
    data.grid = 0;
    data.tree = nanovdb::NanoGrid<BuildT>::memUsage();
    data.root = data.tree + nanovdb::NanoTree<BuildT>::memUsage();
    data.upper = data.root + nanovdb::NanoRoot<BuildT>::memUsage(capacities.upper_count);
    data.lower = data.upper + nanovdb::NanoUpper<BuildT>::memUsage() * uint64_t(capacities.upper_count);
    data.leaf = data.lower + nanovdb::NanoLower<BuildT>::memUsage() * uint64_t(capacities.lower_count);
    data.capacities = capacities;
    data.leaf_keys = leaf_keys;
    data.lower_keys = lower_keys;
    data.upper_keys = upper_keys;
    data.voxel_keys = voxel_keys;
    data.counts = counts;
    data.leaf_active_counts = leaf_active_counts;
    data.leaf_active_prefix = leaf_active_prefix;
    data.status = status;
    data.map = params.map;
    if constexpr (nanovdb::util::is_same<BuildT, nanovdb::ValueOnIndex>::value) {
        data.background_value = BuildT();
    } else {
        data.background_value = params.background_value;
    }
    return data;
}

template <typename Func> void rebuild_cub_temp_call(size_t& temp_size, Func&& func)
{
    void* temp = nullptr;
    check_cuda(func(temp, temp_size));
    if (temp_size > 0) {
        temp = wp_alloc_device(WP_CURRENT_CONTEXT, temp_size, "(native:volume_builder)");
        check_cuda(func(temp, temp_size));
        wp_free_device(WP_CURRENT_CONTEXT, temp);
    }
}

void rebuild_sort_keys(uint64_t* keys_in, uint64_t* keys_out, int count, cudaStream_t stream)
{
    size_t temp_size = 0;
    rebuild_cub_temp_call(temp_size, [&](void* temp, size_t& size) {
        return cub::DeviceRadixSort::SortKeys(temp, size, keys_in, keys_out, count, 0, 64, stream);
    });
}

template <typename InputIt>
void rebuild_encode_runs(
    InputIt keys_in, uint64_t* unique_keys, uint32_t* run_counts, uint32_t* run_count, int count, cudaStream_t stream
)
{
    size_t temp_size = 0;
    rebuild_cub_temp_call(temp_size, [&](void* temp, size_t& size) {
        return cub::DeviceRunLengthEncode::Encode(
            temp, size, keys_in, unique_keys, run_counts, run_count, count, stream
        );
    });
}

template <typename T> void rebuild_exclusive_sum(T* in, T* out, int count, cudaStream_t stream)
{
    size_t temp_size = 0;
    rebuild_cub_temp_call(temp_size, [&](void* temp, size_t& size) {
        return cub::DeviceScan::ExclusiveSum(temp, size, in, out, count, stream);
    });
}

void rebuild_exclusive_sum_u32_to_u64(uint32_t* in, uint64_t* out, int count, cudaStream_t stream)
{
    size_t temp_size = 0;
    rebuild_cub_temp_call(temp_size, [&](void* temp, size_t& size) {
        return cub::DeviceScan::ExclusiveSum(temp, size, in, out, count, stream);
    });
}

template <typename BuildT, typename PtrT>
bool rebuild_count_points(
    RebuildKeyScratch& scratch,
    const PtrT points,
    size_t num_points,
    bool active_voxel_grid,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params,
    bool enforce_capacities,
    uint32_t* status,
    cudaStream_t stream
)
{
    if (!rebuild_alloc_key_scratch(scratch, num_points, active_voxel_grid, status, stream)) {
        return false;
    }

    rebuild_emit_upper_keys<<<rebuild_num_blocks(num_points), REBUILD_NUM_THREADS, 0, stream>>>(
        num_points, points, params.map, scratch.keys_a
    );
    check_cuda(cudaGetLastError());

    rebuild_sort_keys(scratch.keys_a, scratch.keys_b, scratch.point_count, stream);
    rebuild_encode_runs(
        scratch.keys_b, scratch.upper_keys, scratch.run_counts, scratch.counts + REBUILD_COUNT_UPPER,
        scratch.point_count, stream
    );

    rebuild_emit_hierarchy_keys<BuildT><<<rebuild_num_blocks(num_points), REBUILD_NUM_THREADS, 0, stream>>>(
        num_points, points, params.map, scratch.upper_keys, scratch.counts, scratch.keys_a, status
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
    if (scratch.active_voxel_grid) {
        leaf_active_counts = static_cast<uint32_t*>(
            wp_alloc_device(WP_CURRENT_CONTEXT, capacities.leaf_count * sizeof(uint32_t), "(native:volume_builder)")
        );
        leaf_active_prefix = static_cast<uint64_t*>(
            wp_alloc_device(WP_CURRENT_CONTEXT, capacities.leaf_count * sizeof(uint64_t), "(native:volume_builder)")
        );
        check_cuda(cudaMemsetAsync(leaf_active_counts, 0, capacities.leaf_count * sizeof(uint32_t), stream));
    }

    RebuildGridData<BuildT> data = make_rebuild_data(
        grid, grid_size, capacities, params, scratch.leaf_keys, scratch.lower_keys, scratch.upper_keys,
        scratch.voxel_keys, scratch.counts, leaf_active_counts, leaf_active_prefix, status
    );

    rebuild_init_grid_tree_root<<<1, 1, 0, stream>>>(data, scratch.active_voxel_grid);
    rebuild_build_upper_nodes<<<rebuild_num_blocks(capacities.upper_count), REBUILD_NUM_THREADS, 0, stream>>>(data);
    rebuild_set_upper_background_values<<<
        rebuild_num_blocks(uint64_t(capacities.upper_count) << 15u), REBUILD_NUM_THREADS, 0, stream>>>(data);
    rebuild_build_lower_nodes<<<rebuild_num_blocks(capacities.lower_count), REBUILD_NUM_THREADS, 0, stream>>>(data);
    rebuild_set_lower_background_values<<<
        rebuild_num_blocks(uint64_t(capacities.lower_count) << 12u), REBUILD_NUM_THREADS, 0, stream>>>(data);
    rebuild_build_leaf_nodes<<<rebuild_num_blocks(capacities.leaf_count), REBUILD_NUM_THREADS, 0, stream>>>(
        data, scratch.active_voxel_grid
    );
    check_cuda(cudaGetLastError());

    if (scratch.active_voxel_grid) {
        if constexpr (nanovdb::BuildTraits<BuildT>::is_onindex) {
            rebuild_set_active_voxels<<<rebuild_num_blocks(capacities.voxel_count), REBUILD_NUM_THREADS, 0, stream>>>(
                data
            );
            rebuild_exclusive_sum_u32_to_u64(
                leaf_active_counts, leaf_active_prefix, int(capacities.leaf_count), stream
            );
            rebuild_finalize_onindex_leaves<<<
                rebuild_num_blocks(capacities.leaf_count), REBUILD_NUM_THREADS, 0, stream>>>(data);
        }
    } else {
        rebuild_set_leaf_values<<<
            rebuild_num_blocks(uint64_t(capacities.leaf_count) << 9u), REBUILD_NUM_THREADS, 0, stream>>>(data);
    }

    const uint32_t bbox_capacity
        = std::max(std::max(capacities.leaf_count, capacities.lower_count), capacities.upper_count);
    rebuild_reset_bboxes<<<rebuild_num_blocks(bbox_capacity), REBUILD_NUM_THREADS, 0, stream>>>(data);
    rebuild_propagate_leaf_bboxes<<<rebuild_num_blocks(capacities.leaf_count), REBUILD_NUM_THREADS, 0, stream>>>(data);
    rebuild_propagate_lower_bboxes<<<rebuild_num_blocks(capacities.lower_count), REBUILD_NUM_THREADS, 0, stream>>>(
        data
    );
    rebuild_propagate_upper_bboxes<<<rebuild_num_blocks(capacities.upper_count), REBUILD_NUM_THREADS, 0, stream>>>(
        data
    );
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
            scratch, points, num_points, active_voxel_grid, capacities, params, true, status, stream
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
            scratch, points, num_points, active_voxel_grid, empty_capacities, params, false, nullptr, stream
        )) {
        return;
    }

    const VolumeRebuildCapacities capacities = rebuild_copy_exact_capacities(scratch, stream);
    if (capacities.leaf_count == 0 || capacities.lower_count == 0 || capacities.upper_count == 0) {
        rebuild_free_key_scratch(scratch);
        return;
    }

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
    const BuildGridParams<BuildT>& params
)
{
    if (points_in_world_space) {
        allocate_exact_grid_from_points_impl(
            out_grid, out_grid_size, static_cast<const nanovdb::Vec3f*>(points), num_points, false, params
        );
    } else {
        allocate_exact_grid_from_points_impl(
            out_grid, out_grid_size, static_cast<const nanovdb::Coord*>(points), num_points, false, params
        );
    }
}

void allocate_grid_from_active_voxels(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const BuildGridParams<nanovdb::ValueOnIndex>& params
)
{
    if (points_in_world_space) {
        allocate_exact_grid_from_points_impl(
            out_grid, out_grid_size, static_cast<const nanovdb::Vec3f*>(points), num_points, true, params
        );
    } else {
        allocate_exact_grid_from_points_impl(
            out_grid, out_grid_size, static_cast<const nanovdb::Coord*>(points), num_points, true, params
        );
    }
}

#define EXPAND_BUILDER_TYPE(type)                                                                                      \
    template void allocate_grid_from_tiles(                                                                            \
        nanovdb::Grid<nanovdb::NanoTree<type>>*&, size_t&, const void*, size_t, bool, const BuildGridParams<type>&      \
    );

WP_VOLUME_BUILDER_INSTANTIATE_TYPES
#undef EXPAND_BUILDER_TYPE

template void allocate_grid_from_tiles(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueIndex>>*&,
    size_t&,
    const void*,
    size_t,
    bool,
    const BuildGridParams<nanovdb::ValueIndex>&
);

template <typename BuildT>
void rebuild_grid_from_tiles(
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>* grid,
    size_t grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params,
    uint32_t* status
)
{
    if (points_in_world_space) {
        rebuild_grid_from_points_impl(
            grid, grid_size, static_cast<const nanovdb::Vec3f*>(points), num_points, false, capacities, params, status
        );
    } else {
        rebuild_grid_from_points_impl(
            grid, grid_size, static_cast<const nanovdb::Coord*>(points), num_points, false, capacities, params, status
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
        out_grid, out_grid_size, points, num_points, points_in_world_space, capacities, params, status
    );
}

void rebuild_grid_from_active_voxels(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>* grid,
    size_t grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<nanovdb::ValueOnIndex>& params,
    uint32_t* status
)
{
    if (points_in_world_space) {
        rebuild_grid_from_points_impl(
            grid, grid_size, static_cast<const nanovdb::Vec3f*>(points), num_points, true, capacities, params, status
        );
    } else {
        rebuild_grid_from_points_impl(
            grid, grid_size, static_cast<const nanovdb::Coord*>(points), num_points, true, capacities, params, status
        );
    }
}

void allocate_rebuildable_grid_from_active_voxels(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
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
        out_grid, out_grid_size, points, num_points, points_in_world_space, capacities, params, status
    );
}

#define EXPAND_BUILDER_TYPE(type)                                                                                      \
    template void allocate_rebuildable_grid_from_tiles(                                                                \
        nanovdb::Grid<nanovdb::NanoTree<type>>*&, size_t&, const void*, size_t, bool,                                  \
        const VolumeRebuildCapacities&, const BuildGridParams<type>&, uint32_t*                                        \
    );                                                                                                                 \
    template void rebuild_grid_from_tiles(                                                                             \
        nanovdb::Grid<nanovdb::NanoTree<type>>*, size_t, const void*, size_t, bool,                                    \
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
    const VolumeRebuildCapacities&,
    const BuildGridParams<nanovdb::ValueIndex>&,
    uint32_t*
);
