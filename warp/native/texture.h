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

#pragma once

#include "builtin.h"

#include <cstdlib>
#include <cstring>
#include <map>

#if WP_ENABLE_CUDA
// Needed for ContextGuard, check_cuda, WP_CURRENT_CONTEXT, wp_alloc_device/wp_memcpy_h2d/wp_free_device, etc.
#include "cuda_util.h"

// Use CUDA runtime types in host compilation units.
#include <cuda_runtime.h>
#endif

namespace wp {

enum TextureType {
    TEX_TYPE_2D = 0,
    TEX_TYPE_3D = 1,
};

enum TextureFilterMode {
    TEX_FILTER_POINT = 0,
    TEX_FILTER_LINEAR = 1,
};

enum TextureAddressMode {
    TEX_ADDRESS_CLAMP = 0,
    TEX_ADDRESS_WRAP = 1,
    TEX_ADDRESS_BORDER = 2,
};

enum TextureFormat {
    TEX_FORMAT_FLOAT32 = 0,
    TEX_FORMAT_UINT8 = 1,
    TEX_FORMAT_UINT16 = 2,
};

struct TextureDesc {
    int type;
    int width;
    int height;
    int depth;
    int num_channels;
    int format;
    bool is_cpu;
    void* context;  // Warp CUDA context (or nullptr for CPU)
    unsigned long long handle;  // cudaTextureObject_t stored as 64-bit integer
    uint64 array_handle;  // cudaArray_t stored as uint64
    void* data;  // host copy for CPU textures
};

// NOTE: id is a pointer to TextureDesc on CPU OR a pointer to device memory holding TextureDesc on GPU.
// Do not dereference device pointers on host.
CUDA_CALLABLE inline TextureDesc texture_get_desc(uint64 id)
{
#if defined(__CUDA_ARCH__)
    return *(TextureDesc*)(id);
#else
    // Host should not dereference a device pointer; return a default.
    TextureDesc d {};
    return d;
#endif
}

template <typename T> CUDA_CALLABLE inline float normalize_value(T val);

template <> CUDA_CALLABLE inline float normalize_value<uint8>(uint8 val) { return float(val) / 255.0f; }

template <> CUDA_CALLABLE inline float normalize_value<uint16>(uint16 val) { return float(val) / 65535.0f; }

template <> CUDA_CALLABLE inline float normalize_value<float>(float val) { return val; }

CUDA_CALLABLE inline int wrap_coord(int c, int size, int address_mode)
{
    if (address_mode == TEX_ADDRESS_CLAMP) {
        return c < 0 ? 0 : (c >= size ? size - 1 : c);
    } else if (address_mode == TEX_ADDRESS_WRAP) {
        c = c % size;
        return c < 0 ? c + size : c;
    } else {
        return (c < 0 || c >= size) ? -1 : c;
    }
}

template <typename T> CUDA_CALLABLE inline vec4 sample_cpu_2d(const TextureDesc* t, float u, float v)
{
    float fx = u * float(t->width) - 0.5f;
    float fy = v * float(t->height) - 0.5f;
    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    float wx = fx - float(x0);
    float wy = fy - float(y0);

    auto fetch = [&](int x, int y) -> vec4 {
        x = wrap_coord(x, t->width, TEX_ADDRESS_CLAMP);
        y = wrap_coord(y, t->height, TEX_ADDRESS_CLAMP);
        const T* ptr = (const T*)t->data + (y * t->width + x) * t->num_channels;
        vec4 c(0.0f, 0.0f, 0.0f, 1.0f);
        for (int i = 0; i < t->num_channels; ++i)
            c[i] = normalize_value<T>(ptr[i]);
        if (t->num_channels < 4)
            c[3] = 1.0f;
        return c;
    };

    vec4 c00 = fetch(x0, y0);
    vec4 c10 = fetch(x0 + 1, y0);
    vec4 c01 = fetch(x0, y0 + 1);
    vec4 c11 = fetch(x0 + 1, y0 + 1);

    vec4 c0 = c00 * (1.0f - wx) + c10 * wx;
    vec4 c1 = c01 * (1.0f - wx) + c11 * wx;
    return c0 * (1.0f - wy) + c1 * wy;
}

template <typename T> CUDA_CALLABLE inline vec4 sample_cpu_3d(const TextureDesc* t, float u, float v, float w)
{
    float fx = u * float(t->width) - 0.5f;
    float fy = v * float(t->height) - 0.5f;
    float fz = w * float(t->depth) - 0.5f;
    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    int z0 = int(floor(fz));
    float wx = fx - float(x0);
    float wy = fy - float(y0);
    float wz = fz - float(z0);

    auto fetch = [&](int x, int y, int z) -> vec4 {
        x = wrap_coord(x, t->width, TEX_ADDRESS_CLAMP);
        y = wrap_coord(y, t->height, TEX_ADDRESS_CLAMP);
        z = wrap_coord(z, t->depth, TEX_ADDRESS_CLAMP);
        const T* ptr = (const T*)t->data + ((z * t->height + y) * t->width + x) * t->num_channels;
        vec4 c(0.0f, 0.0f, 0.0f, 1.0f);
        for (int i = 0; i < t->num_channels; ++i)
            c[i] = normalize_value<T>(ptr[i]);
        if (t->num_channels < 4)
            c[3] = 1.0f;
        return c;
    };

    vec4 c000 = fetch(x0, y0, z0);
    vec4 c100 = fetch(x0 + 1, y0, z0);
    vec4 c010 = fetch(x0, y0 + 1, z0);
    vec4 c110 = fetch(x0 + 1, y0 + 1, z0);
    vec4 c001 = fetch(x0, y0, z0 + 1);
    vec4 c101 = fetch(x0 + 1, y0, z0 + 1);
    vec4 c011 = fetch(x0, y0 + 1, z0 + 1);
    vec4 c111 = fetch(x0 + 1, y0 + 1, z0 + 1);

    vec4 c00 = c000 * (1.0f - wx) + c100 * wx;
    vec4 c10 = c010 * (1.0f - wx) + c110 * wx;
    vec4 c01 = c001 * (1.0f - wx) + c101 * wx;
    vec4 c11 = c011 * (1.0f - wx) + c111 * wx;

    vec4 c0 = c00 * (1.0f - wy) + c10 * wy;
    vec4 c1 = c01 * (1.0f - wy) + c11 * wy;
    return c0 * (1.0f - wz) + c1 * wz;
}

CUDA_CALLABLE inline vec4 texture_sample_cpu_2d(const TextureDesc* t, vec2 uv)
{
    switch (t->format) {
    case TEX_FORMAT_UINT8:
        return sample_cpu_2d<uint8>(t, uv[0], uv[1]);
    case TEX_FORMAT_UINT16:
        return sample_cpu_2d<uint16>(t, uv[0], uv[1]);
    default:
        return sample_cpu_2d<float>(t, uv[0], uv[1]);
    }
}

CUDA_CALLABLE inline vec4 texture_sample_cpu_3d(const TextureDesc* t, vec3 uvw)
{
    switch (t->format) {
    case TEX_FORMAT_UINT8:
        return sample_cpu_3d<uint8>(t, uvw[0], uvw[1], uvw[2]);
    case TEX_FORMAT_UINT16:
        return sample_cpu_3d<uint16>(t, uvw[0], uvw[1], uvw[2]);
    default:
        return sample_cpu_3d<float>(t, uvw[0], uvw[1], uvw[2]);
    }
}

// 2D sampling
CUDA_CALLABLE inline vec4 texture2d_sample_v4(uint64 id, vec2 uv)
{
    const TextureDesc* t = (const TextureDesc*)(id);
    if (!t)
        return vec4(0.0f, 0.0f, 0.0f, 0.0f);

    if (t->is_cpu) {
        return texture_sample_cpu_2d(t, uv);
    }

#if defined(__CUDA_ARCH__)
    // In device code, cudaTextureObject_t is available from CUDA headers included by NVCC.
    cudaTextureObject_t tex = (cudaTextureObject_t)(t->handle);

    if (t->num_channels == 1) {
        float c = tex2D<float>(tex, uv[0], uv[1]);
        return vec4(c, c, c, 1.0f);
    } else if (t->num_channels == 2) {
        float2 c = tex2D<float2>(tex, uv[0], uv[1]);
        return vec4(c.x, c.y, 0.0f, 1.0f);
    } else {
        float4 c = tex2D<float4>(tex, uv[0], uv[1]);
        return vec4(c.x, c.y, c.z, c.w);
    }
#else
    return vec4(0.0f, 0.0f, 0.0f, 0.0f);
#endif
}

CUDA_CALLABLE inline vec3 texture2d_sample_v3(uint64 id, vec2 uv)
{
    const vec4 c = texture2d_sample_v4(id, uv);
    return vec3(c[0], c[1], c[2]);
}

CUDA_CALLABLE inline vec2 texture2d_sample_v2(uint64 id, vec2 uv)
{
    const vec4 c = texture2d_sample_v4(id, uv);
    return vec2(c[0], c[1]);
}

CUDA_CALLABLE inline float texture2d_sample_f(uint64 id, vec2 uv)
{
    const vec4 c = texture2d_sample_v4(id, uv);
    return c[0];
}

// 3D sampling
CUDA_CALLABLE inline vec4 texture3d_sample_v4(uint64 id, vec3 uvw)
{
    const TextureDesc* t = (const TextureDesc*)(id);
    if (!t)
        return vec4(0.0f, 0.0f, 0.0f, 0.0f);

    if (t->is_cpu) {
        return texture_sample_cpu_3d(t, uvw);
    }

#if defined(__CUDA_ARCH__)
    cudaTextureObject_t tex = (cudaTextureObject_t)(t->handle);

    if (t->num_channels == 1) {
        float c = tex3D<float>(tex, uvw[0], uvw[1], uvw[2]);
        return vec4(c, c, c, 1.0f);
    } else if (t->num_channels == 2) {
        float2 c = tex3D<float2>(tex, uvw[0], uvw[1], uvw[2]);
        return vec4(c.x, c.y, 0.0f, 1.0f);
    } else {
        float4 c = tex3D<float4>(tex, uvw[0], uvw[1], uvw[2]);
        return vec4(c.x, c.y, c.z, c.w);
    }
#else
    return vec4(0.0f, 0.0f, 0.0f, 0.0f);
#endif
}

CUDA_CALLABLE inline vec3 texture3d_sample_v3(uint64 id, vec3 uvw)
{
    const vec4 c = texture3d_sample_v4(id, uvw);
    return vec3(c[0], c[1], c[2]);
}

CUDA_CALLABLE inline vec2 texture3d_sample_v2(uint64 id, vec3 uvw)
{
    const vec4 c = texture3d_sample_v4(id, uvw);
    return vec2(c[0], c[1]);
}

CUDA_CALLABLE inline float texture3d_sample_f(uint64 id, vec3 uvw)
{
    const vec4 c = texture3d_sample_v4(id, uvw);
    return c[0];
}

// Adjoint functions (not differentiable)
CUDA_CALLABLE inline void adj_texture2d_sample_v4(uint64, vec2, uint64&, vec2&, vec4&) { }
CUDA_CALLABLE inline void adj_texture2d_sample_v3(uint64, vec2, uint64&, vec2&, vec3&) { }
CUDA_CALLABLE inline void adj_texture2d_sample_v2(uint64, vec2, uint64&, vec2&, vec2&) { }
CUDA_CALLABLE inline void adj_texture2d_sample_f(uint64, vec2, uint64&, vec2&, float&) { }

CUDA_CALLABLE inline void adj_texture3d_sample_v4(uint64, vec3, uint64&, vec3&, vec4&) { }
CUDA_CALLABLE inline void adj_texture3d_sample_v3(uint64, vec3, uint64&, vec3&, vec3&) { }
CUDA_CALLABLE inline void adj_texture3d_sample_v2(uint64, vec3, uint64&, vec3&, vec2&) { }
CUDA_CALLABLE inline void adj_texture3d_sample_f(uint64, vec3, uint64&, vec3&, float&) { }

CUDA_CALLABLE inline void adj_texture_sample_v4(uint64, vec2, uint64&, vec2&, vec4&) { }
CUDA_CALLABLE inline void adj_texture_sample_v3(uint64, vec2, uint64&, vec2&, vec3&) { }
CUDA_CALLABLE inline void adj_texture_sample_f(uint64, vec2, uint64&, vec2&, float&) { }

// Descriptor management (header-only implementation using inline + static)
namespace detail {
inline std::map<uint64_t, TextureDesc>& get_texture_descriptors()
{
    static std::map<uint64_t, TextureDesc> descriptors;
    return descriptors;
}
}

inline bool texture_get_descriptor(uint64_t id, TextureDesc& desc)
{
    auto& descriptors = detail::get_texture_descriptors();
    const auto iter = descriptors.find(id);
    if (iter == descriptors.end())
        return false;
    desc = iter->second;
    return true;
}

inline void texture_add_descriptor(uint64_t id, const TextureDesc& desc)
{
    detail::get_texture_descriptors()[id] = desc;
}

inline void texture_rem_descriptor(uint64_t id) { detail::get_texture_descriptors().erase(id); }

inline size_t texture_format_element_size(int format)
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

// Host texture creation/destruction (implemented inline)
inline uint64_t
wp_texture_create_host(void* data_ptr, int type, int width, int height, int depth, int channels, int format)
{
    using namespace wp;

    size_t elem_size = texture_format_element_size(format);
    int actual_channels = (channels == 3) ? 4 : channels;

    size_t num_texels
        = (type == wp::TEX_TYPE_3D) ? (size_t)width * (size_t)height * (size_t)depth : (size_t)width * (size_t)height;
    size_t total_size = num_texels * (size_t)actual_channels * elem_size;

    void* data_copy = malloc(total_size);
    if (!data_copy)
        return 0;

    if (channels == 3) {
        for (size_t i = 0; i < num_texels; ++i) {
            const uint8_t* src = (const uint8_t*)data_ptr + i * 3 * elem_size;
            uint8_t* dst = (uint8_t*)data_copy + i * 4 * elem_size;
            memcpy(dst, src, 3 * elem_size);
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

    TextureDesc* desc = new TextureDesc();
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
    texture_add_descriptor(id, *desc);

    return id;
}

inline void wp_texture_destroy_host(uint64_t id)
{
    using namespace wp;
    TextureDesc desc;
    if (texture_get_descriptor(id, desc)) {
        if (desc.data)
            free(desc.data);
        texture_rem_descriptor(id);
        delete (TextureDesc*)id;
    }
}

#if WP_ENABLE_CUDA

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

#endif  // WP_ENABLE_CUDA

// IMPORTANT: Warp's public header (warp.h) declares these APIs with C linkage.
// Define them with matching linkage here.
extern "C" {

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
#if WP_ENABLE_CUDA
    wp::ContextGuard guard(context);

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

    wp::TextureDesc* desc_ptr = (wp::TextureDesc*)wp::wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::TextureDesc));
    wp::wp_memcpy_h2d(WP_CURRENT_CONTEXT, desc_ptr, &desc, sizeof(wp::TextureDesc));

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

WP_API void wp_texture_destroy_device(uint64_t id)
{
#if WP_ENABLE_CUDA
    wp::TextureDesc desc;
    if (wp::texture_get_descriptor(id, desc)) {
        wp::ContextGuard guard(desc.context);

        cudaDestroyTextureObject((cudaTextureObject_t)desc.handle);
        cudaFreeArray((cudaArray_t)desc.array_handle);

        wp::texture_rem_descriptor(id);
        wp::wp_free_device(WP_CURRENT_CONTEXT, (void*)id);
    }
#else
    (void)id;
#endif
}

}  // extern "C"

// Device texture creation/destruction (CUDA implementation)
// Note: These are declared here but implemented elsewhere if needed.
