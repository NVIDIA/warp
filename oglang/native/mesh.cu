#include "mesh.h"
#include "bvh.h"

using namespace og;

#include <map>

namespace
{
    // host-side copy of mesh descriptors, maps GPU mesh address (id) to a CPU desc
    std::map<uint64_t, Mesh> g_mesh_descriptors;


    bool get_descriptor(uint64_t id, Mesh& mesh)
    {
        const auto& iter = g_mesh_descriptors.find(id);
        if (iter == g_mesh_descriptors.end())
            return false;
        else
            mesh = iter->second;
            return true;
    }

    void add_descriptor(uint64_t id, const Mesh& mesh)
    {
        g_mesh_descriptors[id] = mesh;
    }

    void rem_descriptor(uint64_t id)
    {
        g_mesh_descriptors.erase(id);

    }

} // anonymous namespace




uint64_t mesh_create_host(vec3* points, int* indices, int num_points, int num_tris)
{
    Mesh* m = new Mesh();

    m->points = points;
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

uint64_t mesh_create_device(vec3* points, int* indices, int num_points, int num_tris)
{
    Mesh mesh;

    mesh.points = points;
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
    add_descriptor(mesh_id, mesh);

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
    if (get_descriptor(id, mesh))
    {    
        bvh_destroy_device(mesh.bvh);
        free_device(mesh.bounds);
        free_device((Mesh*)id);

        rem_descriptor(id);
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


void mesh_refit_device(uint64_t id)
{
    // recompute triangle bounds
    Mesh m;
    if (get_descriptor(id, m))
    {
        const int num_threads_per_block = 256;
        const int num_blocks = (m.num_tris + num_threads_per_block - 1)/num_threads_per_block;

        compute_triangle_bounds<<<num_blocks, num_threads_per_block>>>(m.num_tris, m.points, m.indices, m.bounds);

        bvh_refit_device(m.bvh, m.bounds);
    }
}
