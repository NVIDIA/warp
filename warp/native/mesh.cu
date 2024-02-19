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

__global__ void compute_triangle_bounds(int n, const vec3* points, const int* indices, vec3* lowers, vec3* uppers)
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

        lowers[tid] = lower;
        uppers[tid] = upper;
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

    wp_launch_device(WP_CURRENT_CONTEXT, bvh_refit_with_solid_angle_kernel, bvh.num_items, (bvh.num_items, bvh.node_parents, bvh.node_counts, bvh.node_lowers, bvh.node_uppers, mesh.points, mesh.indices, mesh.solid_angle_props));
}

} // namespace wp


uint64_t mesh_create_device(void* context, wp::array_t<wp::vec3> points, wp::array_t<wp::vec3> velocities, wp::array_t<int> indices, int num_points, int num_tris, int support_winding_number)
{
    ContextGuard guard(context);

    wp::Mesh mesh(points, velocities, indices, num_points, num_tris);

    mesh.context = context ? context : cuda_context_get_current();

    {
        // // todo: BVH creation only on CPU at the moment so temporarily bring all the data back to host
        // vec3* points_host = (vec3*)alloc_host(sizeof(vec3)*num_points);
        // int* indices_host = (int*)alloc_host(sizeof(int)*num_tris*3);
        // bounds3* bounds_host = (bounds3*)alloc_host(sizeof(bounds3)*num_tris);

        // memcpy_d2h(WP_CURRENT_CONTEXT, points_host, points, sizeof(vec3)*num_points);
        // memcpy_d2h(WP_CURRENT_CONTEXT, indices_host, indices, sizeof(int)*num_tris*3);
        // cuda_context_synchronize(WP_CURRENT_CONTEXT);

        // float sum = 0.0;
        // for (int i=0; i < num_tris; ++i)
        // {
        //     bounds_host[i] = bounds3();
        //     wp::vec3 p0 = points_host[indices_host[i*3+0]];
        //     wp::vec3 p1 = points_host[indices_host[i*3+1]];
        //     wp::vec3 p2 = points_host[indices_host[i*3+2]];
        //     bounds_host[i].add_point(p0);
        //     bounds_host[i].add_point(p1);
        //     bounds_host[i].add_point(p2);
        //     sum += length(p0-p1) + length(p0-p2) + length(p2-p1);
        // }
        // mesh.average_edge_length = sum / (num_tris*3);

        // BVH bvh_host = bvh_create(bounds_host, num_tris);
        // BVH bvh_device = bvh_clone(WP_CURRENT_CONTEXT, bvh_host);

        // bvh_destroy_host(bvh_host);

        // create lower upper arrays expected by GPU BVH builder
        mesh.lowers = (wp::vec3*)alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::vec3)*num_tris);
        mesh.uppers = (wp::vec3*)alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::vec3)*num_tris);

        wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_triangle_bounds, num_tris, (num_tris, points.data, indices.data, mesh.lowers, mesh.uppers));

        uint64_t bvh_id = bvh_create_device(mesh.context, mesh.lowers, mesh.uppers, num_tris);
        wp::bvh_get_descriptor(bvh_id, mesh.bvh);

        if (support_winding_number)
        {
            int num_bvh_nodes = 2*num_tris;
            mesh.solid_angle_props = (wp::SolidAngleProps*)alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::SolidAngleProps)*num_bvh_nodes);
        }        
    }

    wp::Mesh* mesh_device = (wp::Mesh*)alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::Mesh));
    memcpy_h2d(WP_CURRENT_CONTEXT, mesh_device, &mesh, sizeof(wp::Mesh));
    
    // save descriptor
    uint64_t mesh_id = (uint64_t)mesh_device;
    mesh_add_descriptor(mesh_id, mesh);

    if (support_winding_number) 
        mesh_refit_device(mesh_id);

    return mesh_id;
}

void mesh_destroy_device(uint64_t id)
{
    wp::Mesh mesh;
    if (wp::mesh_get_descriptor(id, mesh))
    {
        ContextGuard guard(mesh.context);

        wp::bvh_destroy_device(mesh.bvh);

        free_device(WP_CURRENT_CONTEXT, mesh.lowers);
        free_device(WP_CURRENT_CONTEXT, mesh.uppers);
        free_device(WP_CURRENT_CONTEXT, (wp::Mesh*)id);

        if (mesh.solid_angle_props) {
            free_device(WP_CURRENT_CONTEXT, mesh.solid_angle_props);
        }
        wp::mesh_rem_descriptor(id);
    }
}

void mesh_update_stats(uint64_t id)
{
    
}

void mesh_refit_device(uint64_t id)
{
    // recompute triangle bounds
    wp::Mesh m;
    if (mesh_get_descriptor(id, m))
    {
        ContextGuard guard(m.context);

        wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_triangle_bounds, m.num_tris, (m.num_tris, m.points, m.indices, m.lowers, m.uppers));

        if (m.solid_angle_props) 
        {
            // we compute mesh the average edge length
            // for use in mesh_query_point_sign_normal()
            // since it relies on an epsilon for welding

            // reuse bounds memory temporarily for computing edge lengths
            float* length_tmp_ptr = (float*)m.lowers;
            wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_mesh_edge_lengths, m.num_tris, (m.num_tris, m.points, m.indices, length_tmp_ptr));
            
            scan_device(length_tmp_ptr, length_tmp_ptr, m.num_tris, true);
                
            wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_average_mesh_edge_length, 1, (m.num_tris, length_tmp_ptr, (wp::Mesh*)id));
            wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_triangle_bounds, m.num_tris, (m.num_tris, m.points, m.indices, m.lowers, m.uppers));

            // update solid angle data
            bvh_refit_with_solid_angle_device(m.bvh, m);            
        }
        else 
        {
            bvh_refit_device(m.bvh);
        }
    }
}

