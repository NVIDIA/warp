// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "cuda_util.h"
#include "volume_impl.h"

extern CUcontext get_current_context();

__global__ void
volume_get_leaf_coords(const uint32_t leaf_count, pnanovdb_coord_t* leaf_coords, const pnanovdb_buf_t buf)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t actual_leaf_count = pnanovdb_tree_get_node_count_leaf(buf, wp::volume::get_tree(buf));

    if (tid < leaf_count && tid < actual_leaf_count) {
        pnanovdb_leaf_handle_t leaf = wp::volume::get_leaf(buf, tid);
        leaf_coords[tid] = wp::volume::leaf_origin(buf, leaf);
    }
}

__global__ void
volume_get_voxel_coords(const uint32_t voxel_count, pnanovdb_coord_t* voxel_coords, const pnanovdb_buf_t buf)
{
    const uint32_t leaf_index = blockIdx.x;
    const pnanovdb_tree_handle_t tree = wp::volume::get_tree(buf);
    const uint32_t actual_leaf_count = pnanovdb_tree_get_node_count_leaf(buf, tree);
    const uint64_t actual_voxel_count = pnanovdb_tree_get_voxel_count(buf, tree);

    if (leaf_index >= actual_leaf_count)
        return;

    pnanovdb_leaf_handle_t leaf = wp::volume::get_leaf(buf, leaf_index);
    pnanovdb_coord_t leaf_coords = wp::volume::leaf_origin(buf, leaf);

    pnanovdb_coord_t ijk = {
        int32_t(threadIdx.x) + leaf_coords.x,
        int32_t(threadIdx.y) + leaf_coords.y,
        int32_t(threadIdx.z) + leaf_coords.z,
    };

    const uint64_t index = wp::volume::leaf_voxel_index(buf, leaf_index, ijk);
    if (index < voxel_count && index < actual_voxel_count) {
        voxel_coords[index] = ijk;
    }
}

void launch_get_leaf_coords(void* context, const uint32_t leaf_count, pnanovdb_coord_t* leaf_coords, pnanovdb_buf_t buf)
{
    ContextGuard guard(context);
    wp_launch_device(WP_CURRENT_CONTEXT, volume_get_leaf_coords, leaf_count, (leaf_count, leaf_coords, buf));
}

void launch_get_voxel_coords(
    void* context,
    const uint32_t leaf_count,
    const uint32_t voxel_count,
    pnanovdb_coord_t* voxel_coords,
    pnanovdb_buf_t buf
)
{
    ContextGuard guard(context);
    cudaStream_t stream = (cudaStream_t)wp_cuda_stream_get_current();
    volume_get_voxel_coords<<<leaf_count, dim3(8, 8, 8), 0, stream>>>(voxel_count, voxel_coords, buf);
}
