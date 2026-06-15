// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanovdb/NanoVDB.h>

#define WP_VOLUME_BUILDER_INSTANTIATE_TYPES                                                                            \
    EXPAND_BUILDER_TYPE(int32_t)                                                                                       \
    EXPAND_BUILDER_TYPE(float)                                                                                         \
    EXPAND_BUILDER_TYPE(nanovdb::Vec3f)                                                                                \
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
    const BuildGridParams<BuildT>& params
);

void allocate_grid_from_active_voxels(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
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
);

void allocate_rebuildable_grid_from_active_voxels(
    nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>>*& out_grid,
    size_t& out_grid_size,
    const void* points,
    size_t num_points,
    bool points_in_world_space,
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
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<nanovdb::ValueOnIndex>& params,
    uint32_t* status
);
