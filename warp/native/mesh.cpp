/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "mesh.h"
#include "bvh.h"

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

uint64_t mesh_create_host(vec3* points, vec3* velocities, int* indices, int num_points, int num_tris)
{
    Mesh* m = new Mesh();

    m->points = points;
    m->velocities = velocities;
    m->indices = indices;

    m->num_points = num_points;
    m->num_tris = num_tris;

    m->bounds = new bounds3[num_tris];

    for (int i=0; i < num_tris; ++i)
    {
        m->bounds[i].add_point(points[indices[i*3+0]]);
        m->bounds[i].add_point(points[indices[i*3+1]]);
        m->bounds[i].add_point(points[indices[i*3+2]]);
    }

    m->bvh = bvh_create(m->bounds, num_tris);

    return (uint64_t)m;
}

uint64_t mesh_create_device(vec3* points, vec3* velocities, int* indices, int num_points, int num_tris)
{
    Mesh mesh;

    mesh.points = points;
    mesh.velocities = velocities;
    mesh.indices = indices;

    mesh.num_points = num_points;
    mesh.num_tris = num_tris;

    {
        // todo: BVH creation only on CPU at the moment so temporarily bring all the data back to host
        vec3* points_host = (vec3*)alloc_host(sizeof(vec3)*num_points);
        int* indices_host = (int*)alloc_host(sizeof(int)*num_tris*3);
        bounds3* bounds_host = (bounds3*)alloc_host(sizeof(bounds3)*num_tris);

        memcpy_d2h(points_host, points, sizeof(vec3)*num_points);
        memcpy_d2h(indices_host, indices, sizeof(int)*num_tris*3);
        synchronize();

        for (int i=0; i < num_tris; ++i)
        {
            bounds_host[i] = bounds3();
            bounds_host[i].add_point(points_host[indices_host[i*3+0]]);
            bounds_host[i].add_point(points_host[indices_host[i*3+1]]);
            bounds_host[i].add_point(points_host[indices_host[i*3+2]]);
        }

        BVH bvh_host = bvh_create(bounds_host, num_tris);
        BVH bvh_device = bvh_clone(bvh_host);

        bvh_destroy_host(bvh_host);

        // save gpu-side copy of bounds
        mesh.bounds = (bounds3*)alloc_device(sizeof(bounds3)*num_tris);
        memcpy_h2d(mesh.bounds, bounds_host, sizeof(bounds3)*num_tris);

        free_host(points_host);
        free_host(indices_host);
        free_host(bounds_host);

        mesh.bvh = bvh_device;
    }

    Mesh* mesh_device = (Mesh*)alloc_device(sizeof(Mesh));
    memcpy_h2d(mesh_device, &mesh, sizeof(Mesh));
    
    // save descriptor
    uint64_t mesh_id = (uint64_t)mesh_device;
    mesh_add_descriptor(mesh_id, mesh);

    return mesh_id;
}

void mesh_destroy_host(uint64_t id)
{
    Mesh* m = (Mesh*)(id);

    delete[] m->bounds;
    bvh_destroy_host(m->bvh);

    delete m;
}

void mesh_destroy_device(uint64_t id)
{
    Mesh mesh;
    if (mesh_get_descriptor(id, mesh))
    {    
        bvh_destroy_device(mesh.bvh);
        free_device(mesh.bounds);
        free_device((Mesh*)id);

        mesh_rem_descriptor(id);
    }
}

void mesh_refit_host(uint64_t id)
{
    Mesh* m = (Mesh*)(id);

    for (int i=0; i < m->num_tris; ++i)
    {
        m->bounds[i] = bounds3();
        m->bounds[i].add_point(m->points[m->indices[i*3+0]]);
        m->bounds[i].add_point(m->points[m->indices[i*3+1]]);
        m->bounds[i].add_point(m->points[m->indices[i*3+2]]);
    }

    bvh_refit_host(m->bvh, m->bounds);
}


// stubs for non-CUDA platforms
#if WP_DISABLE_CUDA

void mesh_refit_device(uint64_t id)
{
}


#endif // WP_DISABLE_CUDA