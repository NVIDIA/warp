/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "warp.h"
#include "mesh.h"
#include "bvh.h"

namespace wp
{

__global__ void compute_triangle_bounds(int n, const vec3* points, const int* indices, bounds3* b)
{
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < n)
    {
        // if leaf then update bounds
        int i = indices[tid*3+0];
        int j = indices[tid*3+1];
        int k = indices[tid*3+2];

        vec3 p = points[i];
        vec3 q = points[j];
        vec3 r = points[k];

        vec3 lower = min(min(p, q), r);
        vec3 upper = max(max(p, q), r);

        b[tid] = bounds3(lower, upper);
    }
}

} // namespace wp

void mesh_refit_device(uint64_t id)
{

    // recompute triangle bounds
    wp::Mesh m;
    if (mesh_get_descriptor(id, m))
    {
        wp_launch_device(wp::compute_triangle_bounds, m.num_tris, (m.num_tris, m.points, m.indices, m.bounds));

        bvh_refit_device(m.bvh, m.bounds);
    }

}

