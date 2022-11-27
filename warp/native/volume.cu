/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "volume.h"
#include "warp.h"
#include "cuda_util.h"


__global__ void volume_get_leaf_coords(const uint32_t leaf_count, pnanovdb_coord_t *leaf_coords, const uint64_t first_leaf, const uint32_t leaf_stride)
{ 
    static constexpr uint32_t MASK = (1u << 3u) - 1u; // mask for bit operations

    const int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < leaf_count) {
        leaf_coords[tid] = ((pnanovdb_leaf_t*)(first_leaf + leaf_stride * tid))->bbox_min;
        leaf_coords[tid].x &= ~MASK;
        leaf_coords[tid].y &= ~MASK;
        leaf_coords[tid].z &= ~MASK;
    }
}

void launch_get_leaf_coords(void* context, const uint32_t leaf_count, pnanovdb_coord_t *leaf_coords, const uint64_t first_leaf, const uint32_t leaf_stride)
{
    ContextGuard guard(context);
    wp_launch_device(WP_CURRENT_CONTEXT, volume_get_leaf_coords, leaf_count, (leaf_count, leaf_coords, first_leaf, leaf_stride));
}
