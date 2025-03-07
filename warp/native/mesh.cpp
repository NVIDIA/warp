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

#include "mesh.h"
#include "bvh.h"
#include "warp.h"
#include "cuda_util.h"

using namespace wp;

#include <map>

namespace 
{
    // host-side copy of mesh descriptors, maps GPU mesh address (id) to a CPU desc
    std::map<uint64_t, Mesh> g_mesh_descriptors;

} // anonymous namespace


namespace wp
{

bool mesh_get_descriptor(uint64_t id, Mesh& mesh)
{
    const auto& iter = g_mesh_descriptors.find(id);
    if (iter == g_mesh_descriptors.end())
        return false;
    else
        mesh = iter->second;
        return true;
}

bool mesh_set_descriptor(uint64_t id, const Mesh& mesh)
{
    const auto& iter = g_mesh_descriptors.find(id);
    if (iter == g_mesh_descriptors.end())
        return false;
    else
        iter->second = mesh;
    return true;
}

void mesh_add_descriptor(uint64_t id, const Mesh& mesh)
{
    g_mesh_descriptors[id] = mesh;
}

void mesh_rem_descriptor(uint64_t id)
{
    g_mesh_descriptors.erase(id);

}

} // namespace wp

void bvh_refit_with_solid_angle_recursive_host(BVH& bvh, int index, Mesh& mesh)
{
    BVHPackedNodeHalf& lower = bvh.node_lowers[index];
    BVHPackedNodeHalf& upper = bvh.node_uppers[index];

    if (lower.b)
    {
        // Leaf, compute properties
         const int start = lower.i;
         const int end = upper.i;
         // loops through primitives in the leaf
         for (int primitive_counter = start; primitive_counter < end; primitive_counter++)
         {
             int primitive_index = mesh.bvh.primitive_indices[primitive_counter];
             if (primitive_counter == start)
             {
                 precompute_triangle_solid_angle_props(mesh.points[mesh.indices[primitive_index * 3 + 0]], mesh.points[mesh.indices[primitive_index * 3 + 1]],
                     mesh.points[mesh.indices[primitive_index * 3 + 2]], mesh.solid_angle_props[index]);
             }
             else
             {
                 SolidAngleProps triangle_solid_angle_props;
                 precompute_triangle_solid_angle_props(mesh.points[mesh.indices[primitive_index * 3 + 0]], mesh.points[mesh.indices[primitive_index * 3 + 1]],
                     mesh.points[mesh.indices[primitive_index * 3 + 2]], triangle_solid_angle_props);
                 mesh.solid_angle_props[index] = combine_precomputed_solid_angle_props(&mesh.solid_angle_props[index], &triangle_solid_angle_props);
             }
         }

         (vec3&)lower = mesh.solid_angle_props[index].box.lower;
         (vec3&)upper = mesh.solid_angle_props[index].box.upper;
    }
    else
    {
        int left_index = lower.i;
        int right_index = upper.i;

        bvh_refit_with_solid_angle_recursive_host(bvh, left_index, mesh);
        bvh_refit_with_solid_angle_recursive_host(bvh, right_index, mesh);

        // combine
        SolidAngleProps* left_child_data = &mesh.solid_angle_props[left_index];
        SolidAngleProps* right_child_data = (left_index != right_index) ? &mesh.solid_angle_props[right_index] : NULL;
        
        combine_precomputed_solid_angle_props(mesh.solid_angle_props[index], left_child_data, right_child_data);

        // compute union of children
        const vec3& left_lower = (vec3&)bvh.node_lowers[left_index];
        const vec3& left_upper = (vec3&)bvh.node_uppers[left_index];

        const vec3& right_lower = (vec3&)bvh.node_lowers[right_index];
        const vec3& right_upper = (vec3&)bvh.node_uppers[right_index];

        // union of child bounds
        vec3 new_lower = min(left_lower, right_lower);
        vec3 new_upper = max(left_upper, right_upper);
        
        // write new BVH nodes
        (vec3&)lower = new_lower;
        (vec3&)upper = new_upper;        
    }
}

void bvh_refit_with_solid_angle_host(BVH& bvh, Mesh& mesh)
{
    bvh_refit_with_solid_angle_recursive_host(bvh, 0, mesh);
}

uint64_t mesh_create_host(array_t<wp::vec3> points, array_t<wp::vec3> velocities, array_t<int> indices, int num_points, int num_tris, int support_winding_number, int constructor_type)
{
    Mesh* m = new Mesh(points, velocities, indices, num_points, num_tris);

    m->lowers = new vec3[num_tris];
    m->uppers = new vec3[num_tris];

    float sum = 0.0;
    for (int i=0; i < num_tris; ++i)
    {
        wp::vec3& p0 = points[indices[i*3+0]];
        wp::vec3& p1 = points[indices[i*3+1]];
        wp::vec3& p2 = points[indices[i*3+2]];
        
        // compute triangle bounds
        bounds3 b;       
        b.add_point(p0);
        b.add_point(p1);
        b.add_point(p2);

        m->lowers[i] = b.lower;
        m->uppers[i] = b.upper;

        // compute edge lengths
        sum += length(p0-p1) + length(p0-p2) + length(p2-p1);
    }
    m->average_edge_length = sum / (num_tris*3);

    wp::bvh_create_host(m->lowers, m->uppers, num_tris, constructor_type, m->bvh);
    
    if (support_winding_number) 
    {
        // Let's first compute the sold
        int num_bvh_nodes = 2*num_tris-1;
        m->solid_angle_props = new SolidAngleProps[num_bvh_nodes];
        bvh_refit_with_solid_angle_host(m->bvh, *m);
    }

    return (uint64_t)m;
}


void mesh_destroy_host(uint64_t id)
{
    Mesh* m = (Mesh*)(id);

    delete[] m->lowers;
    delete[] m->uppers;

    if (m->solid_angle_props) {
        delete [] m->solid_angle_props;
    }
    bvh_destroy_host(m->bvh);

    delete m;
}

void mesh_refit_host(uint64_t id)
{
    Mesh* m = (Mesh*)(id);

    float sum = 0.0;
    for (int i=0; i < m->num_tris; ++i)
    {
        wp::vec3 p0 = m->points.data[m->indices.data[i*3+0]];
        wp::vec3 p1 = m->points.data[m->indices.data[i*3+1]];
        wp::vec3 p2 = m->points.data[m->indices.data[i*3+2]];

        // compute triangle bounds
        bounds3 b;       
        b.add_point(p0);
        b.add_point(p1);
        b.add_point(p2);

        m->lowers[i] = b.lower;
        m->uppers[i] = b.upper;

        sum += length(p0-p1) + length(p0-p2) + length(p2-p1);
    }
    m->average_edge_length = sum / (m->num_tris*3);

    if (m->solid_angle_props)
    {
        // If solid angle were used, use refit solid angle
        bvh_refit_with_solid_angle_host(m->bvh, *m);
    }
    else
    {
        bvh_refit_host(m->bvh);
    }
}

void mesh_set_points_host(uint64_t id, wp::array_t<wp::vec3> points)
{
    Mesh* m = (Mesh*)(id);
    if (points.ndim != 1 || points.shape[0] != m->points.shape[0])
    {
        fprintf(stderr, "The new points input for mesh_set_points_host does not match the shape of the original points!\n");
        return;
    }

    m->points = points;

    mesh_refit_host(id);
}

void mesh_set_velocities_host(uint64_t id, wp::array_t<wp::vec3> velocities)
{
    Mesh* m = (Mesh*)(id);
    if (velocities.ndim != 1 || velocities.shape[0] != m->velocities.shape[0])
    {
        fprintf(stderr, "The new velocities input for mesh_set_velocities_host does not match the shape of the original velocities!\n");
        return;
    }
    m->velocities = velocities;
}

// stubs for non-CUDA platforms
#if !WP_ENABLE_CUDA


WP_API uint64_t mesh_create_device(void* context, wp::array_t<wp::vec3> points, wp::array_t<wp::vec3> velocities, wp::array_t<int> tris, int num_points, int num_tris, int support_winding_number, int constructor_type) { return 0; }
WP_API void mesh_destroy_device(uint64_t id) {}
WP_API void mesh_refit_device(uint64_t id) {}
WP_API void mesh_set_points_device(uint64_t id, wp::array_t<wp::vec3> points) {};
WP_API void mesh_set_velocities_device(uint64_t id, wp::array_t<wp::vec3> points) {};


#endif // !WP_ENABLE_CUDA