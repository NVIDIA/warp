/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "warp.h"
#include "cuda_util.h"
#include "mesh.h"
#include "bvh.h"
#include "scan.h"

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

__global__ void compute_mesh_edge_lengths(int n, const vec3* points, const int* indices, float* edge_lengths)
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


        edge_lengths[tid] = length(p-q) + length(p-r) + length(q-r);
    }
}

__global__ void compute_average_mesh_edge_length(int n, float* sum_edge_lengths, Mesh* m)
{
    m->average_edge_length = sum_edge_lengths[n - 1] / (3*n);
}

__global__ void bvh_refit_with_solid_angle_kernel(int n, const int* __restrict__ parents, int* __restrict__ child_count, BVHPackedNodeHalf* __restrict__ lowers, BVHPackedNodeHalf* __restrict__ uppers, const vec3* points, const int* indices, SolidAngleProps* solid_angle_props)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        bool leaf = lowers[index].b;

        if (leaf)
        {
            // update the leaf node
            const int leaf_index = lowers[index].i;        
            precompute_triangle_solid_angle_props(points[indices[leaf_index*3+0]], points[indices[leaf_index*3+1]], points[indices[leaf_index*3+2]], solid_angle_props[index]);        

            make_node(lowers+index, solid_angle_props[index].box.lower, leaf_index, true);
            make_node(uppers+index, solid_angle_props[index].box.upper, 0, false);
        }
        else
        {
            // only keep leaf threads
            return;
        }

        // update hierarchy
        for (;;)
        {
            int parent = parents[index];
            
            // reached root
            if (parent == -1)
                return;

            // ensure all writes are visible
            __threadfence();
         
            int finished = atomicAdd(&child_count[parent], 1);

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the the next parent in the hierarchy
            if (finished == 1)
            {
                //printf("Compute non-leaf at %d\n", index);
                const int left_child = lowers[parent].i;
                const int right_child = uppers[parent].i;

                vec3 left_lower = vec3(lowers[left_child].x,
                                       lowers[left_child].y, 
                                       lowers[left_child].z);

                vec3 left_upper = vec3(uppers[left_child].x,
                                       uppers[left_child].y, 
                                       uppers[left_child].z);

                vec3 right_lower = vec3(lowers[right_child].x,
                                       lowers[right_child].y,
                                       lowers[right_child].z);


                vec3 right_upper = vec3(uppers[right_child].x, 
                                       uppers[right_child].y, 
                                       uppers[right_child].z);

                // union of child bounds
                vec3 lower = min(left_lower, right_lower);
                vec3 upper = max(left_upper, right_upper);
                
                // write new BVH nodes
                make_node(lowers+parent, lower, left_child, false);
                make_node(uppers+parent, upper, right_child, false);

                // combine
                SolidAngleProps* left_child_data = &solid_angle_props[left_child];
                SolidAngleProps* right_child_data = (left_child != right_child) ? &solid_angle_props[right_child] : NULL;
        
                combine_precomputed_solid_angle_props(solid_angle_props[parent], left_child_data, right_child_data);

                // move onto processing the parent
                index = parent;
            }
            else
            {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }		
    }
}


void bvh_refit_with_solid_angle_device(BVH& bvh, Mesh& mesh)
{
    ContextGuard guard(bvh.context);

    // clear child counters
    memset_device(WP_CURRENT_CONTEXT, bvh.node_counts, 0, sizeof(int)*bvh.max_nodes);

    wp_launch_device(WP_CURRENT_CONTEXT, bvh_refit_with_solid_angle_kernel, bvh.max_nodes, (bvh.max_nodes, bvh.node_parents, bvh.node_counts, bvh.node_lowers, bvh.node_uppers, mesh.points, mesh.indices, mesh.solid_angle_props));
}
} // namespace wp

void mesh_refit_device(uint64_t id)
{

    // recompute triangle bounds
    wp::Mesh m;
    if (mesh_get_descriptor(id, m))
    {
        ContextGuard guard(m.context);

        // we compute mesh the average edge length
        // for use in mesh_query_point_sign_normal()
        // since it relies on an epsilon for welding
        
        // re-use bounds memory temporarily for computing edge lengths
        float* length_tmp_ptr = (float*)m.bounds;
        wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_mesh_edge_lengths, m.num_tris, (m.num_tris, m.points, m.indices, length_tmp_ptr));
        
        scan_device(length_tmp_ptr, length_tmp_ptr, m.num_tris, true);
            
        wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_average_mesh_edge_length, 1, (m.num_tris, length_tmp_ptr, (wp::Mesh*)id));
        wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_triangle_bounds, m.num_tris, (m.num_tris, m.points, m.indices, m.bounds));

        if (m.solid_angle_props) 
        {
            bvh_refit_with_solid_angle_device(m.bvh, m);            
        }
        else 
        {
            bvh_refit_device(m.bvh, m.bounds);
        }
    }

}
