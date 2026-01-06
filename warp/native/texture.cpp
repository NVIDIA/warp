#include "warp.h"

#include "cuda_util.h"
#include "texture.h"

using namespace wp;

#include <map>

namespace {
// host-side copy of Texture descriptors, maps GPU texture address (id) to a CPU desc
std::map<uint64_t, TextureDesc> g_texture_descriptors;
}

namespace wp {

bool texture_get_descriptor(uint64_t id, TextureDesc& desc)
{
    const auto& iter = g_texture_descriptors.find(id);
    if (iter == g_texture_descriptors.end())
        return false;
    else
        desc = iter->second;
    return true;
}

void texture_add_descriptor(uint64_t id, const TextureDesc& desc) { g_texture_descriptors[id] = desc; }

void texture_rem_descriptor(uint64_t id) { g_texture_descriptors.erase(id); }

size_t texture_format_element_size(int format)
{
    switch (format) {
    case TEX_FORMAT_UINT8:
        return sizeof(uint8_t);
    default:
        return sizeof(float);
    }
}

}  // namespace wp

WP_API uint64_t wp_texture_create_device(
    void* context,
    void* data_ptr,
    int type,
    int width,
    int height,
    int depth,
    int channels,
    int format,
    int normalized_coords,
    int address_mode,
    int filter_mode
)
{
#ifdef WP_ENABLE_CUDA
    ContextGuard guard(context);

    cudaChannelFormatKind kind;
    int bits;
    switch (format) {
    case wp::TEX_FORMAT_UINT8:
        kind = cudaChannelFormatKindUnsigned;
        bits = 8;
        break;
    default:
        kind = cudaChannelFormatKindFloat;
        bits = 32;
        break;
    }

    cudaChannelFormatDesc ch_desc;
    switch (channels) {
    case 1:
        ch_desc = cudaCreateChannelDesc(bits, 0, 0, 0, kind);
        break;
    case 2:
        ch_desc = cudaCreateChannelDesc(bits, bits, 0, 0, kind);
        break;
    default:
        ch_desc = cudaCreateChannelDesc(bits, bits, bits, bits, kind);
        break;
    }

    size_t elem_size = wp::texture_format_element_size(format);
    cudaArray_t cu_array = nullptr;

    if (type == wp::TEX_TYPE_3D) {
        cudaExtent extent = make_cudaExtent((size_t)width, (size_t)height, (size_t)depth);
        cudaError_t err = cudaMalloc3DArray(&cu_array, &ch_desc, extent, cudaArrayDefault);
        if (!check_cuda(err) || !cu_array)
            return 0;

        cudaMemcpy3DParms copy_params = {};
        copy_params.dstArray = cu_array;
        copy_params.extent = extent;
        copy_params.kind = cudaMemcpyDefault;
        copy_params.srcPtr = make_cudaPitchedPtr(
            data_ptr, (size_t)width * (size_t)channels * elem_size, (size_t)width, (size_t)height
        );

        cudaError_t err2 = cudaMemcpy3D(&copy_params);
        if (!check_cuda(err2)) {
            cudaFreeArray(cu_array);
            return 0;
        }
    } else {
        cudaError_t err = cudaMallocArray(&cu_array, &ch_desc, (size_t)width, (size_t)height, cudaArrayDefault);
        if (!check_cuda(err) || !cu_array)
            return 0;

        size_t row_size = (size_t)width * (size_t)channels * elem_size;
        cudaError_t err2
            = cudaMemcpy2DToArray(cu_array, 0, 0, data_ptr, row_size, row_size, (size_t)height, cudaMemcpyDefault);

        if (!check_cuda(err2)) {
            cudaFreeArray(cu_array);
            return 0;
        }
    }

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array;

    cudaTextureDesc tex_desc = {};
    tex_desc.normalizedCoords = normalized_coords;
    cudaTextureAddressMode addr_mode;
    switch (address_mode) {
    case wp::TEX_ADDRESS_WRAP:
        addr_mode = cudaAddressModeWrap;
        break;
    case wp::TEX_ADDRESS_BORDER:
        addr_mode = cudaAddressModeBorder;
        break;
    default:
        addr_mode = cudaAddressModeClamp;
        break;
    }
    tex_desc.addressMode[0] = addr_mode;
    tex_desc.addressMode[1] = addr_mode;
    tex_desc.addressMode[2] = addr_mode;
    tex_desc.filterMode = (filter_mode == wp::TEX_FILTER_LINEAR) ? cudaFilterModeLinear : cudaFilterModePoint;

    // float -> element type; integer -> normalized float
    tex_desc.readMode = (format == wp::TEX_FORMAT_FLOAT32) ? cudaReadModeElementType : cudaReadModeNormalizedFloat;

    cudaTextureObject_t tex_obj = 0;
    cudaError_t err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);
    if (!check_cuda(err)) {
        cudaFreeArray(cu_array);
        return 0;
    }

    wp::TextureDesc desc {};
    desc.type = type;
    desc.handle = (unsigned long long)tex_obj;
    desc.width = width;
    desc.height = height;
    desc.depth = (type == wp::TEX_TYPE_3D) ? depth : 1;
    desc.num_channels = channels;
    desc.format = format;
    desc.filter_mode = filter_mode;
    desc.address_mode = address_mode;
    desc.is_cpu = false;
    desc.array_handle = (wp::uint64)cu_array;
    desc.data = nullptr;

    wp::TextureDesc* desc_ptr = (wp::TextureDesc*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::TextureDesc));
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, desc_ptr, &desc, sizeof(wp::TextureDesc));

    uint64_t id = (uint64_t)desc_ptr;
    wp::texture_add_descriptor(id, desc);

    return id;
#else
    return 0;
#endif
}

WP_API uint64_t wp_texture_create_host(
    void* data_ptr,
    int type,
    int width,
    int height,
    int depth,
    int channels,
    int format,
    int address_mode,
    int filter_mode
)
{
    size_t elem_size = wp::texture_format_element_size(format);

    size_t num_texels
        = (type == wp::TEX_TYPE_3D) ? (size_t)width * (size_t)height * (size_t)depth : (size_t)width * (size_t)height;
    size_t total_size = num_texels * (size_t)channels * elem_size;

    void* data_copy = malloc(total_size);
    if (!data_copy)
        return 0;

    memcpy(data_copy, data_ptr, total_size);

    wp::TextureDesc* desc = new wp::TextureDesc();
    desc->type = type;
    desc->width = width;
    desc->height = height;
    desc->depth = (type == wp::TEX_TYPE_3D) ? depth : 1;
    desc->num_channels = channels;
    desc->format = format;
    desc->filter_mode = filter_mode;
    desc->address_mode = address_mode;
    desc->is_cpu = true;
    desc->handle = 0;
    desc->array_handle = 0;
    desc->data = data_copy;

    uint64_t id = (uint64_t)desc;
    wp::texture_add_descriptor(id, *desc);

    return id;
}

WP_API void wp_texture_destroy_host(uint64_t id)
{
    wp::TextureDesc desc;
    if (wp::texture_get_descriptor(id, desc)) {
        if (desc.data)
            free(desc.data);
        wp::texture_rem_descriptor(id);
        delete (wp::TextureDesc*)id;
    }
}

WP_API void wp_texture_destroy_device(uint64_t id)
{
#ifdef WP_ENABLE_CUDA
    wp::TextureDesc desc;
    if (wp::texture_get_descriptor(id, desc)) {
        cudaDestroyTextureObject((cudaTextureObject_t)desc.handle);
        cudaFreeArray((cudaArray_t)desc.array_handle);

        wp::texture_rem_descriptor(id);
        wp_free_device(WP_CURRENT_CONTEXT, (void*)id);
    }
#endif
}
