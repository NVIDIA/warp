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
};

struct TextureDesc {
    int type;
    int width;
    int height;
    int depth;
    int num_channels;
    int format;
    int filter_mode;
    int address_mode;
    bool is_cpu;
    void* context;
    unsigned long long handle;
    uint64 array_handle;
    void* data;
};

CUDA_CALLABLE inline TextureDesc texture_get(uint64_t id) { return *(TextureDesc*)(id); }

template <typename T> CUDA_CALLABLE inline float normalize_value(T val);

template <> CUDA_CALLABLE inline float normalize_value<uint8>(uint8 val) { return float(val) / 255.0f; }

template <> CUDA_CALLABLE inline float normalize_value<float>(float val) { return val; }

CUDA_CALLABLE inline int wrap_coord(int c, int size, int address_mode)
{
    if (address_mode == TEX_ADDRESS_CLAMP) {
        return c < 0 ? 0 : (c >= size ? size - 1 : c);
    } else if (address_mode == TEX_ADDRESS_WRAP) {
        c = c % size;
        return c < 0 ? c + size : c;
    } else {
        // TEX_ADDRESS_BORDER: return -1 for out-of-bounds (caller handles border value)
        return (c < 0 || c >= size) ? -1 : c;
    }
}

template <typename T> CUDA_CALLABLE inline vec4 fetch_texel_2d(const TextureDesc* t, int x, int y)
{
    x = wrap_coord(x, t->width, t->address_mode);
    y = wrap_coord(y, t->height, t->address_mode);

    // Border addressing returns 0 for out-of-bounds
    if (x < 0 || y < 0) {
        return vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const T* ptr = (const T*)t->data + (y * t->width + x) * t->num_channels;
    vec4 c(0.0f, 0.0f, 0.0f, 1.0f);
    for (int i = 0; i < t->num_channels; ++i)
        c[i] = normalize_value<T>(ptr[i]);
    if (t->num_channels < 4)
        c[3] = 1.0f;
    return c;
}

template <typename T> CUDA_CALLABLE inline vec4 fetch_texel_3d(const TextureDesc* t, int x, int y, int z)
{
    x = wrap_coord(x, t->width, t->address_mode);
    y = wrap_coord(y, t->height, t->address_mode);
    z = wrap_coord(z, t->depth, t->address_mode);

    // Border addressing returns 0 for out-of-bounds
    if (x < 0 || y < 0 || z < 0) {
        return vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const T* ptr = (const T*)t->data + ((z * t->height + y) * t->width + x) * t->num_channels;
    vec4 c(0.0f, 0.0f, 0.0f, 1.0f);
    for (int i = 0; i < t->num_channels; ++i)
        c[i] = normalize_value<T>(ptr[i]);
    if (t->num_channels < 4)
        c[3] = 1.0f;
    return c;
}

template <typename T> CUDA_CALLABLE inline vec4 sample_cpu_2d(const TextureDesc* t, float u, float v)
{
    float fx = u * float(t->width) - 0.5f;
    float fy = v * float(t->height) - 0.5f;

    if (t->filter_mode == TEX_FILTER_POINT) {
        // Nearest neighbor: round to nearest texel
        int x = int(floor(fx + 0.5f));
        int y = int(floor(fy + 0.5f));
        return fetch_texel_2d<T>(t, x, y);
    }

    // Bilinear filtering
    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    float wx = fx - float(x0);
    float wy = fy - float(y0);

    vec4 c00 = fetch_texel_2d<T>(t, x0, y0);
    vec4 c10 = fetch_texel_2d<T>(t, x0 + 1, y0);
    vec4 c01 = fetch_texel_2d<T>(t, x0, y0 + 1);
    vec4 c11 = fetch_texel_2d<T>(t, x0 + 1, y0 + 1);

    vec4 c0 = c00 * (1.0f - wx) + c10 * wx;
    vec4 c1 = c01 * (1.0f - wx) + c11 * wx;
    return c0 * (1.0f - wy) + c1 * wy;
}

template <typename T> CUDA_CALLABLE inline vec4 sample_cpu_3d(const TextureDesc* t, float u, float v, float w)
{
    float fx = u * float(t->width) - 0.5f;
    float fy = v * float(t->height) - 0.5f;
    float fz = w * float(t->depth) - 0.5f;

    if (t->filter_mode == TEX_FILTER_POINT) {
        // Nearest neighbor: round to nearest texel
        int x = int(floor(fx + 0.5f));
        int y = int(floor(fy + 0.5f));
        int z = int(floor(fz + 0.5f));
        return fetch_texel_3d<T>(t, x, y, z);
    }

    // Trilinear filtering
    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    int z0 = int(floor(fz));
    float wx = fx - float(x0);
    float wy = fy - float(y0);
    float wz = fz - float(z0);

    vec4 c000 = fetch_texel_3d<T>(t, x0, y0, z0);
    vec4 c100 = fetch_texel_3d<T>(t, x0 + 1, y0, z0);
    vec4 c010 = fetch_texel_3d<T>(t, x0, y0 + 1, z0);
    vec4 c110 = fetch_texel_3d<T>(t, x0 + 1, y0 + 1, z0);
    vec4 c001 = fetch_texel_3d<T>(t, x0, y0, z0 + 1);
    vec4 c101 = fetch_texel_3d<T>(t, x0 + 1, y0, z0 + 1);
    vec4 c011 = fetch_texel_3d<T>(t, x0, y0 + 1, z0 + 1);
    vec4 c111 = fetch_texel_3d<T>(t, x0 + 1, y0 + 1, z0 + 1);

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
    default:
        return sample_cpu_2d<float>(t, uv[0], uv[1]);
    }
}

CUDA_CALLABLE inline vec4 texture_sample_cpu_3d(const TextureDesc* t, vec3 uvw)
{
    switch (t->format) {
    case TEX_FORMAT_UINT8:
        return sample_cpu_3d<uint8>(t, uvw[0], uvw[1], uvw[2]);
    default:
        return sample_cpu_3d<float>(t, uvw[0], uvw[1], uvw[2]);
    }
}

CUDA_CALLABLE inline vec4 texture2d_sample_v4(uint64 id, vec2 uv)
{
    const TextureDesc t = texture_get(id);
    if (t.is_cpu) {
        return texture_sample_cpu_2d(&t, uv);
    }

#if defined(__CUDA_ARCH__)
    cudaTextureObject_t tex = (cudaTextureObject_t)(t.handle);

    if (t.num_channels == 1) {
        float c = tex2D<float>(tex, uv[0], uv[1]);
        return vec4(c, c, c, 1.0f);
    } else if (t.num_channels == 2) {
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

CUDA_CALLABLE inline vec4 texture3d_sample_v4(uint64 id, vec3 uvw)
{
    const TextureDesc t = texture_get(id);
    if (t.is_cpu) {
        return texture_sample_cpu_3d(&t, uvw);
    }

#if defined(__CUDA_ARCH__)
    cudaTextureObject_t tex = (cudaTextureObject_t)(t.handle);

    if (t.num_channels == 1) {
        float c = tex3D<float>(tex, uvw[0], uvw[1], uvw[2]);
        return vec4(c, c, c, 1.0f);
    } else if (t.num_channels == 2) {
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

// Adjoint functions (no-op for textures)
CUDA_CALLABLE inline void adj_texture2d_sample_v4(uint64, vec2, uint64&, vec2&, vec4&) { }
CUDA_CALLABLE inline void adj_texture2d_sample_v2(uint64, vec2, uint64&, vec2&, vec2&) { }
CUDA_CALLABLE inline void adj_texture2d_sample_f(uint64, vec2, uint64&, vec2&, float&) { }

CUDA_CALLABLE inline void adj_texture3d_sample_v4(uint64, vec3, uint64&, vec3&, vec4&) { }
CUDA_CALLABLE inline void adj_texture3d_sample_v2(uint64, vec3, uint64&, vec3&, vec2&) { }
CUDA_CALLABLE inline void adj_texture3d_sample_f(uint64, vec3, uint64&, vec3&, float&) { }

// Host-side functions (declared here, defined in texture.cpp)
bool texture_get_descriptor(uint64_t id, TextureDesc& desc);
void texture_add_descriptor(uint64_t id, const TextureDesc& desc);
void texture_rem_descriptor(uint64_t id);
size_t texture_format_element_size(int format);

}  // namespace wp
