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

__global__ void bvh_refit_with_solid_angle_kernel(int n, const int* __restrict__ parents, 
    int* __restrict__ child_count, BVHPackedNodeHalf* __restrict__ node_lowers, BVHPackedNodeHalf* __restrict__ node_uppers, 
    const vec3* points, const int* indices, const int* primitive_indices, SolidAngleProps* solid_angle_props)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        bool leaf = node_lowers[index].b;
        int parent = parents[index];

        if (leaf)
        {
            BVHPackedNodeHalf& lower = node_lowers[index];
            BVHPackedNodeHalf& upper = node_uppers[index];

            // update the leaf node
            bool true_leaf = true;

            if (parent != -1)
            {
                true_leaf = !node_lowers[parent].b;
            }

            if (true_leaf)
            {
                SolidAngleProps node_solid_angle_props;

                const int start = lower.i;
                const int end = upper.i;

                // loops through primitives in the leaf
                for (int primitive_counter = start; primitive_counter < end; primitive_counter++)
                {
                    int primitive_index = primitive_indices[primitive_counter];
                    if (primitive_counter == start)
                    {
                        precompute_triangle_solid_angle_props(points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
                            points[indices[primitive_index * 3 + 2]], node_solid_angle_props);
                    }
                    else
                    {
                        SolidAngleProps triangle_solid_angle_props;
                        precompute_triangle_solid_angle_props(points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
                            points[indices[primitive_index * 3 + 2]], triangle_solid_angle_props);
                        node_solid_angle_props = combine_precomputed_solid_angle_props(&node_solid_angle_props, &triangle_solid_angle_props);
                    }
                }

                (vec3&)lower = node_solid_angle_props.box.lower;
                (vec3&)upper = node_solid_angle_props.box.upper;
                solid_angle_props[index] = node_solid_angle_props;
            }
        }

        else
        {
            // only keep leaf threads
            return;
        }

        // update hierarchy
        for (;;)
        {
            parent = parents[index];
            
            // reached root
            if (parent == -1)
                return;

            // ensure all writes are visible
            __threadfence();
         
            int finished = atomicAdd(&child_count[parent], 1);

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the next parent in the hierarchy
            if (finished == 1)
            {
                BVHPackedNodeHalf& parent_lower = node_lowers[parent];
                BVHPackedNodeHalf& parent_upper = node_uppers[parent];
                if (parent_lower.b)
                    // a packed leaf node can still be a parent in LBVH, we need to recompute its bounds
                    // since we've lost its left and right child node index in the muting process
                {
                    int parent_parent = parents[parent];;
                    // only need to compute bound when this is a valid leaf node
                    bool true_leaf = true;

                    if (parent_parent != -1)
                    {
                        true_leaf = !node_lowers[parent_parent].b;
                    }

                    if (true_leaf)
                    {
                        SolidAngleProps node_solid_angle_props;
                        const int start = parent_lower.i;
                        const int end = parent_upper.i;
                        // loops through primitives in the leaf
                        for (int primitive_counter = start; primitive_counter < end; primitive_counter++)
                        {
                            int primitive_index = primitive_indices[primitive_counter];
                            if (primitive_counter == start)
                            {
                                precompute_triangle_solid_angle_props(points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
                                    points[indices[primitive_index * 3 + 2]], node_solid_angle_props);
                            }
                            else
                            {
                                SolidAngleProps triangle_solid_angle_props;
                                precompute_triangle_solid_angle_props(points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
                                    points[indices[primitive_index * 3 + 2]], triangle_solid_angle_props);
                                node_solid_angle_props = combine_precomputed_solid_angle_props(&node_solid_angle_props, &triangle_solid_angle_props);
                            }
                        }

                        (vec3&)parent_lower = node_solid_angle_props.box.lower;
                        (vec3&)parent_upper = node_solid_angle_props.box.upper;
                        solid_angle_props[parent] = node_solid_angle_props;
                    }
                }
                else
                {
                    //printf("Compute non-leaf at %d\n", index);
                    const int left_child = node_lowers[parent].i;
                    const int right_child = node_uppers[parent].i;

                    vec3 left_lower = (vec3&)(node_lowers[left_child]);
                    vec3 left_upper = (vec3&)(node_uppers[left_child]);
                    vec3 right_lower = (vec3&)(node_lowers[right_child]);
                    vec3 right_upper = (vec3&)(node_uppers[right_child]);

                    // union of child bounds
                    vec3 lower = min(left_lower, right_lower);
                    vec3 upper = max(left_upper, right_upper);

                    // write new BVH nodes
                    (vec3&)parent_lower = lower;
                    (vec3&)parent_upper = upper;

                    // combine
                    SolidAngleProps* left_child_data = &solid_angle_props[left_child];
                    SolidAngleProps* right_child_data = (left_child != right_child) ? &solid_angle_props[right_child] : NULL;

                    combine_precomputed_solid_angle_props(solid_angle_props[parent], left_child_data, right_child_data);
                }
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
    memset_device(WP_CURRENT_CONTEXT, bvh.node_counts, 0, sizeof(int) * bvh.max_nodes);
    wp_launch_device(WP_CURRENT_CONTEXT, bvh_refit_with_solid_angle_kernel, bvh.num_leaf_nodes, 
        (bvh.num_leaf_nodes, bvh.node_parents, bvh.node_counts, bvh.node_lowers, bvh.node_uppers, mesh.points, mesh.indices, bvh.primitive_indices, mesh.solid_angle_props));
}

} // namespace wp


uint64_t mesh_create_device(void* context, wp::array_t<wp::vec3> points, wp::array_t<wp::vec3> velocities, wp::array_t<int> indices, int num_points, int num_tris, int support_winding_number, int constructor_type)
{
    ContextGuard guard(context);

    wp::Mesh mesh(points, velocities, indices, num_points, num_tris);

    mesh.context = context ? context : cuda_context_get_current();

    // create lower upper arrays expected by GPU BVH builder
    mesh.lowers = (wp::vec3*)alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::vec3)*num_tris);
    mesh.uppers = (wp::vec3*)alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::vec3)*num_tris);

    if (support_winding_number)
    {
        int num_bvh_nodes = 2 * num_tris;
        mesh.solid_angle_props = (wp::SolidAngleProps*)alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::SolidAngleProps) * num_bvh_nodes);
    }

    wp::Mesh* mesh_device = (wp::Mesh*)alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::Mesh));
    memcpy_h2d(WP_CURRENT_CONTEXT, mesh_device, &mesh, sizeof(wp::Mesh));

    // save descriptor
    uint64_t mesh_id = (uint64_t)mesh_device;

    // we compute mesh the average edge length
    // for use in mesh_query_point_sign_normal()
    // since it relies on an epsilon for welding
    // reuse bounds memory temporarily for computing edge lengths
    float* length_tmp_ptr = (float*)mesh.lowers;
    wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_mesh_edge_lengths, mesh.num_tris, (mesh.num_tris, mesh.points, mesh.indices, length_tmp_ptr));
    scan_device(length_tmp_ptr, length_tmp_ptr, mesh.num_tris, true);
    wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_average_mesh_edge_length, 1, (mesh.num_tris, length_tmp_ptr, mesh_device));

    // compute triangle bound and construct BVH
    wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_triangle_bounds, mesh.num_tris, (mesh.num_tris, mesh.points, mesh.indices, mesh.lowers, mesh.uppers));
    wp::bvh_create_device(mesh.context, mesh.lowers, mesh.uppers, num_tris, constructor_type, mesh.bvh);

    // we need to overwrite mesh.bvh because it is not initialized when we construct it on device
    memcpy_h2d(WP_CURRENT_CONTEXT, &(mesh_device->bvh), &mesh.bvh, sizeof(wp::BVH));

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

        // we compute mesh the average edge length
        // for use in mesh_query_point_sign_normal()
        // since it relies on an epsilon for welding
        
        // reuse bounds memory temporarily for computing edge lengths
        float* length_tmp_ptr = (float*)m.lowers;
        wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_mesh_edge_lengths, m.num_tris, (m.num_tris, m.points, m.indices, length_tmp_ptr));

        scan_device(length_tmp_ptr, length_tmp_ptr, m.num_tris, true);

        wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_average_mesh_edge_length, 1, (m.num_tris, length_tmp_ptr, (wp::Mesh*)id));
        wp_launch_device(WP_CURRENT_CONTEXT, wp::compute_triangle_bounds, m.num_tris, (m.num_tris, m.points, m.indices, m.lowers, m.uppers));

        if (m.solid_angle_props) 
        {
            // update solid angle data
            bvh_refit_with_solid_angle_device(m.bvh, m);            
        }
        else 
        {
            bvh_refit_device(m.bvh);
        }
    }
}

void mesh_set_points_device(uint64_t id, wp::array_t<wp::vec3> points)
{
    wp::Mesh m;
    if (mesh_get_descriptor(id, m))
    {
        if (points.ndim != 1 || points.shape[0] != m.points.shape[0])
        {
            fprintf(stderr, "The new points input for mesh_set_points_device does not match the shape of the original points!\n");
            return;
        }

        m.points = points;

        wp::Mesh* mesh_device = (wp::Mesh*)id;
        memcpy_h2d(WP_CURRENT_CONTEXT, mesh_device, &m, sizeof(wp::Mesh));

        // update the cpu copy as well
        mesh_set_descriptor(id, m);

        mesh_refit_device(id);
    }
    else 
    {
        fprintf(stderr, "The mesh id provided to mesh_set_points_device is not valid!\n");
        return;
    }
}

void mesh_set_velocities_device(uint64_t id, wp::array_t<wp::vec3> velocities)
{
    wp::Mesh m;
    if (mesh_get_descriptor(id, m))
    {
        if (velocities.ndim != 1 || velocities.shape[0] != m.velocities.shape[0])
        {
            fprintf(stderr, "The new velocities input for mesh_set_velocities_device does not match the shape of the original velocities\n");
            return;
        }

        m.velocities = velocities;

        wp::Mesh* mesh_device = (wp::Mesh*)id;
        memcpy_h2d(WP_CURRENT_CONTEXT, mesh_device, &m, sizeof(wp::Mesh));
        mesh_set_descriptor(id, m);
    }
    else 
    {
        fprintf(stderr, "The mesh id provided to mesh_set_velocities_device is not valid!\n");
        return;
    }
}
