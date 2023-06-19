/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
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
        const int leaf_index = lower.i;

        precompute_triangle_solid_angle_props(mesh.points[mesh.indices[leaf_index*3+0]], mesh.points[mesh.indices[leaf_index*3+1]], mesh.points[mesh.indices[leaf_index*3+2]], mesh.solid_angle_props[index]);        
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

uint64_t mesh_create_host(array_t<wp::vec3> points, array_t<wp::vec3> velocities, array_t<int> indices, int num_points, int num_tris, int support_winding_number)
{
    Mesh* m = new Mesh(points, velocities, indices, num_points, num_tris);

    m->bounds = new bounds3[num_tris];

    float sum = 0.0;
    for (int i=0; i < num_tris; ++i)
    {
        wp::vec3& p0 = points[indices[i*3+0]];
        wp::vec3& p1 = points[indices[i*3+1]];
        wp::vec3& p2 = points[indices[i*3+2]];
        m->bounds[i].add_point(p0);
        m->bounds[i].add_point(p1);
        m->bounds[i].add_point(p2);
        sum += length(p0-p1) + length(p0-p2) + length(p2-p1);
    }
    m->average_edge_length = sum / (num_tris*3);

    m->bvh = bvh_create(m->bounds, num_tris);

    if (support_winding_number) 
    {
        // Let's first compute the sold
        int num_bvh_nodes = 2*num_tris-1;
        m->solid_angle_props = new SolidAngleProps[num_bvh_nodes];
        bvh_refit_with_solid_angle_host(m->bvh, *m);
    }

    return (uint64_t)m;
}

uint64_t mesh_create_device(void* context, array_t<wp::vec3> points, array_t<wp::vec3> velocities, array_t<int> indices, int num_points, int num_tris, int support_winding_number)
{
    ContextGuard guard(context);

    Mesh mesh(points, velocities, indices, num_points, num_tris);

    mesh.context = context ? context : cuda_context_get_current();

    // mesh.points = array_t<vec3>(points, num_points, points_grad);
    // mesh.velocities = array_t<vec3>(velocities, num_points, velocities_grad);
    // mesh.indices = array_t<int>(indices, num_tris, 3);

    // mesh.num_points = num_points;
    // mesh.num_tris = num_tris;

    {
        // todo: BVH creation only on CPU at the moment so temporarily bring all the data back to host
        vec3* points_host = (vec3*)alloc_host(sizeof(vec3)*num_points);
        int* indices_host = (int*)alloc_host(sizeof(int)*num_tris*3);
        bounds3* bounds_host = (bounds3*)alloc_host(sizeof(bounds3)*num_tris);

        memcpy_d2h(WP_CURRENT_CONTEXT, points_host, points, sizeof(vec3)*num_points);
        memcpy_d2h(WP_CURRENT_CONTEXT, indices_host, indices, sizeof(int)*num_tris*3);
        cuda_context_synchronize(WP_CURRENT_CONTEXT);

        float sum = 0.0;
        for (int i=0; i < num_tris; ++i)
        {
            bounds_host[i] = bounds3();
            wp::vec3 p0 = points_host[indices_host[i*3+0]];
            wp::vec3 p1 = points_host[indices_host[i*3+1]];
            wp::vec3 p2 = points_host[indices_host[i*3+2]];
            bounds_host[i].add_point(p0);
            bounds_host[i].add_point(p1);
            bounds_host[i].add_point(p2);
            sum += length(p0-p1) + length(p0-p2) + length(p2-p1);
        }
        mesh.average_edge_length = sum / (num_tris*3);

        BVH bvh_host = bvh_create(bounds_host, num_tris);
        BVH bvh_device = bvh_clone(WP_CURRENT_CONTEXT, bvh_host);

        bvh_destroy_host(bvh_host);

        // save gpu-side copy of bounds
        mesh.bounds = (bounds3*)alloc_device(WP_CURRENT_CONTEXT, sizeof(bounds3)*num_tris);
        memcpy_h2d(WP_CURRENT_CONTEXT, mesh.bounds, bounds_host, sizeof(bounds3)*num_tris);

        free_host(points_host);
        free_host(indices_host);
        free_host(bounds_host);

        mesh.bvh = bvh_device;

        if (support_winding_number)
        {
            int num_bvh_nodes = 2*num_tris-1;
            mesh.solid_angle_props = (SolidAngleProps*)alloc_device(WP_CURRENT_CONTEXT, sizeof(SolidAngleProps)*num_bvh_nodes);
        }        
    }

    Mesh* mesh_device = (Mesh*)alloc_device(WP_CURRENT_CONTEXT, sizeof(Mesh));
    memcpy_h2d(WP_CURRENT_CONTEXT, mesh_device, &mesh, sizeof(Mesh));
    
    // save descriptor
    uint64_t mesh_id = (uint64_t)mesh_device;
    mesh_add_descriptor(mesh_id, mesh);

    if (support_winding_number) 
    {
        mesh_refit_device(mesh_id);
    }
    return mesh_id;
}

void mesh_destroy_host(uint64_t id)
{
    Mesh* m = (Mesh*)(id);

    delete[] m->bounds;
    if (m->solid_angle_props) {
        delete [] m->solid_angle_props;
    }
    bvh_destroy_host(m->bvh);

    delete m;
}

void mesh_destroy_device(uint64_t id)
{
    Mesh mesh;
    if (mesh_get_descriptor(id, mesh))
    {
        ContextGuard guard(mesh.context);

        bvh_destroy_device(mesh.bvh);

        free_device(WP_CURRENT_CONTEXT, mesh.bounds);
        free_device(WP_CURRENT_CONTEXT, (Mesh*)id);

        if (mesh.solid_angle_props) {
            free_device(WP_CURRENT_CONTEXT, mesh.solid_angle_props);
        }
        mesh_rem_descriptor(id);
    }
}

void mesh_refit_host(uint64_t id)
{
    Mesh* m = (Mesh*)(id);

    float sum = 0.0;
    for (int i=0; i < m->num_tris; ++i)
    {
        m->bounds[i] = bounds3();
        wp::vec3 p0 = m->points.data[m->indices.data[i*3+0]];
        wp::vec3 p1 = m->points.data[m->indices.data[i*3+1]];
        wp::vec3 p2 = m->points.data[m->indices.data[i*3+2]];
        m->bounds[i].add_point(p0);
        m->bounds[i].add_point(p1);
        m->bounds[i].add_point(p2);
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
        bvh_refit_host(m->bvh, m->bounds);
    }
}


// stubs for non-CUDA platforms
#if !WP_ENABLE_CUDA

void mesh_refit_device(uint64_t id)
{
}


#endif // !WP_ENABLE_CUDA