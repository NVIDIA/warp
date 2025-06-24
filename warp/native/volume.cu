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
#include "volume_impl.h"
#include "warp.h"


__global__ void volume_get_leaf_coords(const uint32_t leaf_count, pnanovdb_coord_t *leaf_coords,
                                       const pnanovdb_buf_t buf)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < leaf_count)
    {
        pnanovdb_leaf_handle_t leaf = wp::volume::get_leaf(buf, tid);
        leaf_coords[tid] = wp::volume::leaf_origin(buf, leaf);
    }
}

__global__ void volume_get_voxel_coords(const uint32_t voxel_count, pnanovdb_coord_t *voxel_coords,
                                        const pnanovdb_buf_t buf)
{
    const uint32_t leaf_index = blockIdx.x;
    pnanovdb_leaf_handle_t leaf = wp::volume::get_leaf(buf, leaf_index);
    pnanovdb_coord_t leaf_coords = wp::volume::leaf_origin(buf, leaf);

    pnanovdb_coord_t ijk = {
        int32_t(threadIdx.x) + leaf_coords.x,
        int32_t(threadIdx.y) + leaf_coords.y,
        int32_t(threadIdx.z) + leaf_coords.z,
    };

    const uint64_t index = wp::volume::leaf_voxel_index(buf, leaf_index, ijk);
    if (index < voxel_count)
    {
        voxel_coords[index] = ijk;
    }
}

void launch_get_leaf_coords(void *context, const uint32_t leaf_count, pnanovdb_coord_t *leaf_coords, pnanovdb_buf_t buf)
{
    ContextGuard guard(context);
    wp_launch_device(WP_CURRENT_CONTEXT, volume_get_leaf_coords, leaf_count, (leaf_count, leaf_coords, buf));
}

void launch_get_voxel_coords(void *context, const uint32_t leaf_count, const uint32_t voxel_count,
                             pnanovdb_coord_t *voxel_coords, pnanovdb_buf_t buf)
{
    ContextGuard guard(context);
    cudaStream_t stream = (cudaStream_t)wp_cuda_stream_get_current();
    volume_get_voxel_coords<<<leaf_count, dim3(8, 8, 8), 0, stream>>>(voxel_count, voxel_coords, buf);
}
