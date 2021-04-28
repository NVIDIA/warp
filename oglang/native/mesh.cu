#include "mesh.h"

struct Mesh
{

};


uint64_t mesh_create(const float3* points, int* tris)
{
    Mesh* m = new Mesh();

    m->points = points;
    m->tris = tris;

    m->bvh = bvh_create();

    return (uint64_t)(m);
}

void mesh_destroy(uint64_t id)
{
    Mesh* m = (Mesh*)(id);

    bvh_destroy(m->bvh)
}
void mesh_update(uint64_t m, const float3* points, int* tris, bool refit)
{

}




float3 mesh_query_point(uint64_t id, float3 point, float max_dist)
{
    const Mesh& m = mesh_get(id)

    

    

}

float mesh_query_ray()
{

}