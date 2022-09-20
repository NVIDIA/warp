/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "warp.h"
#include "cuda_util.h"
#include "bvh.h"
#include "wpbvh.h"


namespace wp
{

__global__ void set_bounds_from_lowers_and_uppers(int n, const vec3* lowers, const vec3* uppers, bounds3* b)
{
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < n)
    {
        b[tid] = bounds3(lowers[tid], uppers[tid]);
    }
}

} // namespace wp


void bvh_refit_device(uint64_t id)
{

    // recompute triangle bounds
    wp::Bvh bvh;
    if (mesh_get_descriptor(id, m))
    {
        ContextGuard guard(m.context);

        wp_launch_device(WP_CURRENT_CONTEXT, wp::set_bounds_from_lowers_and_uppers, bvh.num_bounds, (bvh.num_bounds, bvh.lowers, bvh.uppers, bvh.bounds));

        bvh_refit_device(bvh.internal_bvh, bvh.bounds);
    }

}

