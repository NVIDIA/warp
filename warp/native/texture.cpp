#include "warp.h"

#include "cuda_util.h"
#include "texture.h"

#include <map>

using namespace wp;

namespace {
// Track descriptors on host for id->desc mapping, similar to BVH/Mesh
std::map<uint64_t, Texture2D> g_texture_descriptors;
}

namespace wp {

bool texture_get_descriptor(uint64_t id, Texture2D& desc)
{
    const auto iter = g_texture_descriptors.find(id);
    if (iter == g_texture_descriptors.end())
        return false;
    else {
        desc = iter->second;
        return true;
    }
}

void texture_add_descriptor(uint64_t id, const Texture2D& desc) { g_texture_descriptors[id] = desc; }

void texture_rem_descriptor(uint64_t id) { g_texture_descriptors.erase(id); }

}  // namespace wp


WP_API uint64_t wp_texture_create_device(
    void* context,
    void* data_ptr,
    uint64_t size,
    int width,
    int height,
    int channels,
    int format,
    int normalized_coords,
    int address_mode,
    int filter_mode
)
{
#if WP_ENABLE_CUDA
    ContextGuard guard(context);

    cudaChannelFormatDesc ch_desc {};
    ch_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    cudaArray_t cu_array = nullptr;
    {
        auto err = cudaMallocArray(&cu_array, &ch_desc, width, height, cudaArrayDefault);
        if (!check_cuda(err) || !cu_array)
            return 0;
    }

    size_t row_size = (size_t)width * (size_t)channels * sizeof(unsigned char);
    size_t total_bytes = row_size * (size_t)height;
    if (size * sizeof(unsigned char) < total_bytes) {
        cudaFreeArray(cu_array);
        return 0;
    }
    {
        auto err = cudaMemcpy2DToArray(
            cu_array, 0, 0, data_ptr, (size_t)width * (size_t)channels * sizeof(unsigned char),
            (size_t)width * (size_t)channels * sizeof(unsigned char), (size_t)height, cudaMemcpyDefault
        );
        check_cuda(err);
    }

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array;

    cudaTextureDesc tex_desc = {};
    tex_desc.normalizedCoords = 1;
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;

    cudaTextureObject_t tex_obj = 0;
    {
        auto err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);
        if (!check_cuda(err)) {
            cudaFreeArray(cu_array);
            return 0;
        }
    }

    Texture2D desc {};
    desc.handle = (unsigned long long)tex_obj;
    desc.width = width;
    desc.height = height;
    desc.num_channels = channels;
    desc.context = context ? context : wp_cuda_context_get_current();
    desc.array_handle = (uint64)cu_array;

    // allocate device-side copy of descriptor so kernels can deref the id
    Texture2D* texture_device_ptr = (Texture2D*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(Texture2D));
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, texture_device_ptr, &desc, sizeof(Texture2D));
    uint64_t id = (uint64_t)texture_device_ptr;

    texture_add_descriptor(id, desc);

    return id;
#else
    (void)context;
    (void)data_ptr;
    (void)size;
    (void)width;
    (void)height;
    (void)channels;
    (void)format;
    (void)normalized_coords;
    (void)address_mode;
    (void)filter_mode;
    return 0;
#endif
}

WP_API void wp_texture_destroy_device(uint64_t id)
{
#if WP_ENABLE_CUDA
    Texture2D desc;
    if (texture_get_descriptor(id, desc)) {
        ContextGuard guard(desc.context);

        cudaDestroyTextureObject((cudaTextureObject_t)desc.handle);
        cudaFreeArray((cudaArray_t)desc.array_handle);

        texture_rem_descriptor(id);

        wp_free_device(WP_CURRENT_CONTEXT, (void*)id);
    }
#else
    (void)id;
#endif
}
