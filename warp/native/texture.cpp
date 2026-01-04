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
    case TEX_FORMAT_UINT16:
        return sizeof(uint16_t);
    default:
        return sizeof(float);
    }
}

}  // namespace wp

#ifdef __CUDA_ARCH__

namespace {
// Helpers are only needed for the host-side creation of cuda arrays/texture objects.
inline cudaChannelFormatDesc get_channel_format_desc(int format, int channels)
{
    cudaChannelFormatKind kind;
    int bits;

    switch (format) {
    case wp::TEX_FORMAT_UINT8:
        kind = cudaChannelFormatKindUnsigned;
        bits = 8;
        break;
    case wp::TEX_FORMAT_UINT16:
        kind = cudaChannelFormatKindUnsigned;
        bits = 16;
        break;
    default:
        kind = cudaChannelFormatKindFloat;
        bits = 32;
        break;
    }

    switch (channels) {
    case 1:
        return cudaCreateChannelDesc(bits, 0, 0, 0, kind);
    case 2:
        return cudaCreateChannelDesc(bits, bits, 0, 0, kind);
    default:
        return cudaCreateChannelDesc(bits, bits, bits, bits, kind);
    }
}

inline cudaTextureAddressMode get_address_mode(int mode)
{
    switch (mode) {
    case wp::TEX_ADDRESS_WRAP:
        return cudaAddressModeWrap;
    case wp::TEX_ADDRESS_BORDER:
        return cudaAddressModeBorder;
    default:
        return cudaAddressModeClamp;
    }
}

inline cudaTextureFilterMode get_filter_mode(int mode)
{
    return mode == wp::TEX_FILTER_LINEAR ? cudaFilterModeLinear : cudaFilterModePoint;
}
}  // namespace

#endif  // __CUDA_ARCH__

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
#ifdef __CUDA_ARCH__
    ContextGuard guard(context);

    int actual_channels = (channels == 3) ? 4 : channels;
    cudaChannelFormatDesc ch_desc = get_channel_format_desc(format, actual_channels);
    size_t elem_size = wp::texture_format_element_size(format);

    cudaArray_t cu_array = nullptr;
    void* temp_data = nullptr;

    if (type == wp::TEX_TYPE_3D) {
        cudaExtent extent = make_cudaExtent((size_t)width, (size_t)height, (size_t)depth);
        cudaError_t err = cudaMalloc3DArray(&cu_array, &ch_desc, extent, cudaArrayDefault);
        if (!check_cuda(err) || !cu_array)
            return 0;

        size_t num_texels = (size_t)width * (size_t)height * (size_t)depth;

        cudaMemcpy3DParms copy_params = {};
        copy_params.dstArray = cu_array;
        copy_params.extent = extent;
        copy_params.kind = cudaMemcpyDefault;

        if (channels == 3) {
            size_t total_size = num_texels * (size_t)4 * elem_size;
            temp_data = malloc(total_size);
            if (!temp_data) {
                cudaFreeArray(cu_array);
                return 0;
            }

            for (size_t i = 0; i < num_texels; ++i) {
                const uint8_t* src = (const uint8_t*)data_ptr + i * (size_t)3 * elem_size;
                uint8_t* dst = (uint8_t*)temp_data + i * (size_t)4 * elem_size;
                memcpy(dst, src, (size_t)3 * elem_size);
                if (format == wp::TEX_FORMAT_FLOAT32)
                    ((float*)dst)[3] = 1.0f;
                else if (format == wp::TEX_FORMAT_UINT16)
                    ((uint16_t*)dst)[3] = 65535;
                else
                    dst[3] = 255;
            }

            copy_params.srcPtr
                = make_cudaPitchedPtr(temp_data, (size_t)width * (size_t)4 * elem_size, (size_t)width, (size_t)height);
        } else {
            copy_params.srcPtr = make_cudaPitchedPtr(
                data_ptr, (size_t)width * (size_t)actual_channels * elem_size, (size_t)width, (size_t)height
            );
        }

        cudaError_t err2 = cudaMemcpy3D(&copy_params);
        if (temp_data)
            free(temp_data);
        if (!check_cuda(err2)) {
            cudaFreeArray(cu_array);
            return 0;
        }
    } else {
        cudaError_t err = cudaMallocArray(&cu_array, &ch_desc, (size_t)width, (size_t)height, cudaArrayDefault);
        if (!check_cuda(err) || !cu_array)
            return 0;

        size_t row_size_dst = (size_t)width * (size_t)actual_channels * elem_size;

        if (channels == 3) {
            size_t row_size_src = (size_t)width * (size_t)3 * elem_size;
            size_t total_size = row_size_dst * (size_t)height;

            temp_data = malloc(total_size);
            if (!temp_data) {
                cudaFreeArray(cu_array);
                return 0;
            }

            for (int y = 0; y < height; ++y) {
                const uint8_t* src = (const uint8_t*)data_ptr + (size_t)y * row_size_src;
                uint8_t* dst = (uint8_t*)temp_data + (size_t)y * row_size_dst;
                for (int x = 0; x < width; ++x) {
                    memcpy(
                        dst + (size_t)x * (size_t)4 * elem_size, src + (size_t)x * (size_t)3 * elem_size,
                        (size_t)3 * elem_size
                    );
                    if (format == wp::TEX_FORMAT_FLOAT32)
                        ((float*)dst)[(size_t)x * 4 + 3] = 1.0f;
                    else if (format == wp::TEX_FORMAT_UINT16)
                        ((uint16_t*)dst)[(size_t)x * 4 + 3] = 65535;
                    else
                        dst[(size_t)x * (size_t)4 * elem_size + 3] = 255;
                }
            }

            cudaError_t err2 = cudaMemcpy2DToArray(
                cu_array, 0, 0, temp_data, row_size_dst, row_size_dst, (size_t)height, cudaMemcpyHostToDevice
            );

            free(temp_data);
            temp_data = nullptr;

            if (!check_cuda(err2)) {
                cudaFreeArray(cu_array);
                return 0;
            }
        } else {
            cudaError_t err2 = cudaMemcpy2DToArray(
                cu_array, 0, 0, data_ptr, row_size_dst, row_size_dst, (size_t)height, cudaMemcpyDefault
            );

            if (!check_cuda(err2)) {
                cudaFreeArray(cu_array);
                return 0;
            }
        }
    }

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array;

    cudaTextureDesc tex_desc = {};
    tex_desc.normalizedCoords = normalized_coords;
    tex_desc.addressMode[0] = get_address_mode(address_mode);
    tex_desc.addressMode[1] = get_address_mode(address_mode);
    tex_desc.addressMode[2] = get_address_mode(address_mode);
    tex_desc.filterMode = get_filter_mode(filter_mode);

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
    desc.num_channels = actual_channels;
    desc.format = format;
    desc.is_cpu = false;
    desc.context = context ? context : wp::wp_cuda_context_get_current();
    desc.array_handle = (wp::uint64)cu_array;
    desc.data = nullptr;

    wp::TextureDesc* desc_ptr = (wp::TextureDesc*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::TextureDesc));
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, desc_ptr, &desc, sizeof(wp::TextureDesc));

    uint64_t id = (uint64_t)desc_ptr;
    wp::texture_add_descriptor(id, desc);

    return id;
#else
    (void)context;
    (void)data_ptr;
    (void)type;
    (void)width;
    (void)height;
    (void)depth;
    (void)channels;
    (void)format;
    (void)normalized_coords;
    (void)address_mode;
    (void)filter_mode;
    return 0;
#endif
}

WP_API uint64_t
wp_texture_create_host(void* data_ptr, int type, int width, int height, int depth, int channels, int format)
{
    size_t elem_size = wp::texture_format_element_size(format);
    int actual_channels = (channels == 3) ? 4 : channels;

    size_t num_texels
        = (type == wp::TEX_TYPE_3D) ? (size_t)width * (size_t)height * (size_t)depth : (size_t)width * (size_t)height;
    size_t total_size = num_texels * (size_t)actual_channels * elem_size;

    void* data_copy = malloc(total_size);
    if (!data_copy)
        return 0;

    if (channels == 3) {
        for (size_t i = 0; i < num_texels; ++i) {
            const uint8_t* src = (const uint8_t*)data_ptr + i * (size_t)3 * elem_size;
            uint8_t* dst = (uint8_t*)data_copy + i * (size_t)4 * elem_size;
            memcpy(dst, src, (size_t)3 * elem_size);
            if (format == wp::TEX_FORMAT_FLOAT32) {
                ((float*)dst)[3] = 1.0f;
            } else if (format == wp::TEX_FORMAT_UINT16) {
                ((uint16_t*)dst)[3] = 65535;
            } else {
                dst[3] = 255;
            }
        }
    } else {
        memcpy(data_copy, data_ptr, total_size);
    }

    wp::TextureDesc* desc = new wp::TextureDesc();
    desc->type = type;
    desc->width = width;
    desc->height = height;
    desc->depth = (type == wp::TEX_TYPE_3D) ? depth : 1;
    desc->num_channels = actual_channels;
    desc->format = format;
    desc->is_cpu = true;
    desc->context = nullptr;
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
#ifdef __CUDA_ARCH__
    wp::TextureDesc desc;
    if (wp::texture_get_descriptor(id, desc)) {
        ContextGuard guard(desc.context);

        cudaDestroyTextureObject((cudaTextureObject_t)desc.handle);
        cudaFreeArray((cudaArray_t)desc.array_handle);

        wp::texture_rem_descriptor(id);
        wp_free_device(WP_CURRENT_CONTEXT, (void*)id);
    }
#else
    (void)id;
#endif
}