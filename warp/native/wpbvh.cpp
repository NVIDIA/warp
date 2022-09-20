/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "wpbvh.h"
#include "bvh.h"
#include "warp.h"
#include "cuda_util.h"

using namespace wp;

#include <map>

namespace 
{
    // host-side copy of bvh descriptors, maps GPU bvh address (id) to a CPU desc
    std::map<uint64_t, Bvh> g_bvh_descriptors;

} // anonymous namespace


namespace wp
{

bool bvh_get_descriptor(uint64_t id, Bvh& bvh)
{
    const auto& iter = g_bvh_descriptors.find(id);
    if (iter == g_bvh_descriptors.end())
        return false;
    else
        bvh = iter->second;
        return true;
}

void bvh_add_descriptor(uint64_t id, const Bvh& bvh)
{
    g_bvh_descriptors[id] = bvh;
    
}

void bvh_rem_descriptor(uint64_t id)
{
    g_bvh_descriptors.erase(id);

}

} // namespace wp

uint64_t bvh_create_host(vec3* lowers, vec3* uppers, int num_bounds)
{
    Bvh* bvh = new Bvh();

    bvh->context = NULL;

    bvh->lowers = lowers;
    bvh->uppers = uppers;
    bvh->num_bounds = num_bounds;

    bvh->bounds = new bounds3[num_bounds];  

    for (int i=0; i < num_bounds; ++i)
    {
        bvh->bounds[i].lower = lowers[i];
        bvh->bounds[i].upper = uppers[i];
    }

    bvh->internal_bvh = bvh_create(bvh->bounds, num_bounds);

    return (uint64_t)bvh;
}

uint64_t bvh_create_device(void* context, vec3* lowers, vec3* uppers, int num_bounds)
{
    ContextGuard guard(context);

    Bvh bvh;

    bvh.context = context ? context : cuda_context_get_current();

    bvh.num_bounds = num_bounds;

    {
        // todo: BVH creation only on CPU at the moment so temporarily bring all the data back to host
        bounds3* bounds_host = (bounds3*)alloc_host(sizeof(bounds3)*num_bounds);

        vec3* lowers_host = (vec3*)alloc_host(sizeof(vec3)*num_bounds);
        vec3* uppers_host = (vec3*)alloc_host(sizeof(vec3)*num_bounds);
        bounds3* bounds_host = (bounds3*)alloc_host(sizeof(bounds3)*num_bounds);

        memcpy_d2h(WP_CURRENT_CONTEXT, lowers_host, lowers, sizeof(vec3)*num_bounds);
        memcpy_d2h(WP_CURRENT_CONTEXT, uppers_host, uppers, sizeof(vec3)*num_bounds);
        cuda_context_synchronize(WP_CURRENT_CONTEXT);

        for (int i=0; i < num_bounds; ++i)
        {
            bounds_host[i] = bounds3();
            bounds_host[i].lower = lowers_host[i];
            bounds_host[i].upper = uppers_host[i];
        }

        BVH bvh_host = bvh_create(bounds_host, num_bounds);
        BVH bvh_device = bvh_clone(WP_CURRENT_CONTEXT, bvh_host);

        bvh_destroy_host(bvh_host);

        // save gpu-side copy of bounds
        bvh.bounds = (bounds3*)alloc_device(WP_CURRENT_CONTEXT, sizeof(bounds3)*num_bounds);
        memcpy_h2d(WP_CURRENT_CONTEXT, bvh.bounds, bounds_host, sizeof(bounds3)*num_bounds);

        free_host(lowers_host);
        free_host(uppers_host);
        free_host(bounds_host);

        bvh.internal_bvh = bvh_device;
    }

    Bvh* bvh_device = (Bvh*)alloc_device(WP_CURRENT_CONTEXT, sizeof(Bvh));
    memcpy_h2d(WP_CURRENT_CONTEXT, bvh_device, &bvh, sizeof(Bvh));
    
    // save descriptor
    uint64_t bvh_id = (uint64_t)bvh_device;
    bvh_add_descriptor(bvh_id, bvh);

    return bvh_id;
}

void bvh_destroy_host(uint64_t id)
{
    Bvh* bvh = (Bvh*)(id);

    delete[] bvh->bounds;
    bvh_destroy_host(bvh->internal_bvh);

    delete bvh;
}

void mesh_destroy_device(uint64_t id)
{
    Bvh bvh;
    if (bvh_get_descriptor(id, bvh))
    {
        ContextGuard guard(bvh.context);

        bvh_destroy_device(bvh.internal_bvh);

        free_device(WP_CURRENT_CONTEXT, bvh.bounds);
        free_device(WP_CURRENT_CONTEXT, (Bvh*)id);

        mesh_rem_descriptor(id);
    }
}

void mesh_refit_host(uint64_t id)
{
    Bvh* bvh = (Bvh*)(id);

    for (int i=0; i < bvh->num_bounds; ++i)
    {
        bvh->bounds[i] = bounds3();
        bvh->bounds[i].lower = bvh->lowers[i];
        bvh->bounds[i].upper = bvh->uppers[i];
    }

    bvh_refit_host(bvh->internal_bvh, bvh->bounds);
}


// stubs for non-CUDA platforms
#if WP_DISABLE_CUDA

void mesh_refit_device(uint64_t id)
{
}


#endif // WP_DISABLE_CUDA