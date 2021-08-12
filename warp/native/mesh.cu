#include "warp.h"
#include "mesh.h"
#include "bvh.h"

using namespace wp;

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
    if (mesh_get_descriptor(id, m))
    {
        launch_device(compute_triangle_bounds, m.num_tris, (cudaStream_t)cuda_get_stream(), (m.num_tris, m.points, m.indices, m.bounds));

        bvh_refit_device(m.bvh, m.bounds);
    }

}
