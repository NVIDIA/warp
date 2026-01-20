/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Data type constants (must match texture.cpp and Python side)
#define WP_TEXTURE_DTYPE_UINT8  0
#define WP_TEXTURE_DTYPE_UINT16 1
#define WP_TEXTURE_DTYPE_FLOAT32 2

// Filter mode constants
#define WP_TEXTURE_FILTER_CLOSEST 0
#define WP_TEXTURE_FILTER_LINEAR 1

// Address mode constants
#define WP_TEXTURE_ADDRESS_WRAP   0
#define WP_TEXTURE_ADDRESS_CLAMP  1
#define WP_TEXTURE_ADDRESS_MIRROR 2
#define WP_TEXTURE_ADDRESS_BORDER 3

// CPU texture descriptor - mirrors the struct in texture.cpp
// This is what the tex handle points to on CPU
struct cpu_texture2d_data {
    void* data;
    int32 width;
    int32 height;
    int32 num_channels;
    int32 dtype;
    int32 filter_mode;
    int32 address_mode_u;  // Per-axis address mode for U
    int32 address_mode_v;  // Per-axis address mode for V
    bool use_normalized_coords;  // If true, coords in [0,1]; if false, in texel space
};

struct cpu_texture3d_data {
    void* data;
    int32 width;
    int32 height;
    int32 depth;
    int32 num_channels;
    int32 dtype;
    int32 filter_mode;
    int32 address_mode_u;  // Per-axis address mode for U
    int32 address_mode_v;  // Per-axis address mode for V
    int32 address_mode_w;  // Per-axis address mode for W
    bool use_normalized_coords;  // If true, coords in [0,1]; if false, in texel space
};

// Texture descriptor passed to kernels
// Contains the CUDA texture object handle (GPU) or pointer to cpu_texture*_data (CPU)
struct texture2d_t {
    uint64 tex;  // CUtexObject handle (GPU) or cpu_texture2d_data* (CPU)
    int32 width;
    int32 height;
    int32 num_channels;

    CUDA_CALLABLE inline texture2d_t()
        : tex(0)
        , width(0)
        , height(0)
        , num_channels(0)
    {
    }

    CUDA_CALLABLE inline texture2d_t(uint64 tex, int32 width, int32 height, int32 num_channels)
        : tex(tex)
        , width(width)
        , height(height)
        , num_channels(num_channels)
    {
    }
};

struct texture3d_t {
    uint64 tex;  // CUtexObject handle (GPU) or cpu_texture3d_data* (CPU)
    int32 width;
    int32 height;
    int32 depth;
    int32 num_channels;

    CUDA_CALLABLE inline texture3d_t()
        : tex(0)
        , width(0)
        , height(0)
        , depth(0)
        , num_channels(0)
    {
    }

    CUDA_CALLABLE inline texture3d_t(uint64 tex, int32 width, int32 height, int32 depth, int32 num_channels)
        : tex(tex)
        , width(width)
        , height(height)
        , depth(depth)
        , num_channels(num_channels)
    {
    }
};

// ============================================================================
// CPU Software Texture Sampling Helpers
// ============================================================================

// Apply address mode to get a valid texture coordinate
inline float cpu_apply_address_mode_1d(float coord, int size, int address_mode)
{
    // Normalized coordinates to texel space
    float texel = coord * size - 0.5f;
    float fsize = (float)size;

    switch (address_mode) {
    case WP_TEXTURE_ADDRESS_WRAP:
        texel = texel - floor(texel / fsize) * fsize;
        if (texel < 0)
            texel += fsize;
        break;
    case WP_TEXTURE_ADDRESS_CLAMP:
        if (texel < 0.0f)
            texel = 0.0f;
        if (texel > fsize - 1.0f)
            texel = fsize - 1.0f;
        break;
    case WP_TEXTURE_ADDRESS_MIRROR: {
        float period = 2.0f * fsize;
        texel = texel - floor(texel / period) * period;
        if (texel < 0)
            texel += period;
        if (texel >= fsize)
            texel = period - texel - 1.0f;
    } break;
    case WP_TEXTURE_ADDRESS_BORDER:
    default:
        // Border mode: values outside [0, size-1] return 0
        // This is handled by bounds checking in the fetch function
        break;
    }
    return texel;
}

// Clamp integer index to valid range
inline int cpu_clamp_index(int idx, int size) { return (idx < 0) ? 0 : ((idx >= size) ? size - 1 : idx); }

// Apply address mode to an integer texel index
// This is used for neighbor indices in linear filtering to properly handle wrap/mirror
inline int cpu_apply_address_mode_index(int idx, int size, int address_mode)
{
    switch (address_mode) {
    case WP_TEXTURE_ADDRESS_WRAP: {
        int m = idx % size;
        if (m < 0)
            m += size;
        return m;
    }
    case WP_TEXTURE_ADDRESS_CLAMP:
        return cpu_clamp_index(idx, size);
    case WP_TEXTURE_ADDRESS_MIRROR: {
        int period = 2 * size;
        int m = idx % period;
        if (m < 0)
            m += period;
        if (m >= size)
            return period - m - 1;
        return m;
    }
    case WP_TEXTURE_ADDRESS_BORDER:
    default:
        return idx;  // border handled by fetch bounds checks
    }
}

// Check if index is within bounds (for border mode)
inline bool cpu_in_bounds_2d(int x, int y, int w, int h) { return x >= 0 && x < w && y >= 0 && y < h; }

inline bool cpu_in_bounds_3d(int x, int y, int z, int w, int h, int d)
{
    return x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d;
}

// Fetch a single texel value (normalized to [0,1] for uint types, as-is for float)
inline float cpu_fetch_texel_2d(const cpu_texture2d_data* tex, int x, int y, int channel)
{
    // Border mode and invalid channels return 0
    if (!cpu_in_bounds_2d(x, y, tex->width, tex->height) || channel < 0 || channel >= tex->num_channels) {
        return 0.0f;
    }

    int idx = (y * tex->width + x) * tex->num_channels + channel;

    switch (tex->dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return ((const uint8_t*)tex->data)[idx] / 255.0f;
    case WP_TEXTURE_DTYPE_UINT16:
        return ((const uint16_t*)tex->data)[idx] / 65535.0f;
    case WP_TEXTURE_DTYPE_FLOAT32:
    default:
        return ((const float*)tex->data)[idx];
    }
}

inline float cpu_fetch_texel_3d(const cpu_texture3d_data* tex, int x, int y, int z, int channel)
{
    // Border mode and invalid channels return 0
    if (!cpu_in_bounds_3d(x, y, z, tex->width, tex->height, tex->depth) || channel < 0
        || channel >= tex->num_channels) {
        return 0.0f;
    }

    int idx = ((z * tex->height + y) * tex->width + x) * tex->num_channels + channel;

    switch (tex->dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return ((const uint8_t*)tex->data)[idx] / 255.0f;
    case WP_TEXTURE_DTYPE_UINT16:
        return ((const uint16_t*)tex->data)[idx] / 65535.0f;
    case WP_TEXTURE_DTYPE_FLOAT32:
    default:
        return ((const float*)tex->data)[idx];
    }
}

// Sample a single channel with bilinear interpolation (2D)
inline float cpu_sample_2d_channel(const cpu_texture2d_data* tex, float u, float v, int channel)
{
    // Convert to texel space if using normalized coordinates
    float coord_u = tex->use_normalized_coords ? u : (u / (float)tex->width);
    float coord_v = tex->use_normalized_coords ? v : (v / (float)tex->height);

    float tx = cpu_apply_address_mode_1d(coord_u, tex->width, tex->address_mode_u);
    float ty = cpu_apply_address_mode_1d(coord_v, tex->height, tex->address_mode_v);

    if (tex->filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        // Nearest neighbor
        int x = (int)floor(tx + 0.5f);
        int y = (int)floor(ty + 0.5f);

        if (tex->address_mode_u != WP_TEXTURE_ADDRESS_BORDER) {
            x = cpu_clamp_index(x, tex->width);
        }
        if (tex->address_mode_v != WP_TEXTURE_ADDRESS_BORDER) {
            y = cpu_clamp_index(y, tex->height);
        }

        return cpu_fetch_texel_2d(tex, x, y, channel);
    } else {
        // Bilinear interpolation
        int x0 = (int)floor(tx);
        int y0 = (int)floor(ty);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float fx = tx - x0;
        float fy = ty - y0;

        // Apply address mode to neighbor indices (properly handles wrap/mirror at edges)
        x0 = cpu_apply_address_mode_index(x0, tex->width, tex->address_mode_u);
        x1 = cpu_apply_address_mode_index(x1, tex->width, tex->address_mode_u);
        y0 = cpu_apply_address_mode_index(y0, tex->height, tex->address_mode_v);
        y1 = cpu_apply_address_mode_index(y1, tex->height, tex->address_mode_v);

        float v00 = cpu_fetch_texel_2d(tex, x0, y0, channel);
        float v10 = cpu_fetch_texel_2d(tex, x1, y0, channel);
        float v01 = cpu_fetch_texel_2d(tex, x0, y1, channel);
        float v11 = cpu_fetch_texel_2d(tex, x1, y1, channel);

        // Bilinear interpolation
        float v0 = v00 * (1.0f - fx) + v10 * fx;
        float v1 = v01 * (1.0f - fx) + v11 * fx;
        return v0 * (1.0f - fy) + v1 * fy;
    }
}

// Sample a single channel with trilinear interpolation (3D)
inline float cpu_sample_3d_channel(const cpu_texture3d_data* tex, float u, float v, float w, int channel)
{
    // Convert to texel space if using normalized coordinates
    float coord_u = tex->use_normalized_coords ? u : (u / (float)tex->width);
    float coord_v = tex->use_normalized_coords ? v : (v / (float)tex->height);
    float coord_w = tex->use_normalized_coords ? w : (w / (float)tex->depth);

    float tx = cpu_apply_address_mode_1d(coord_u, tex->width, tex->address_mode_u);
    float ty = cpu_apply_address_mode_1d(coord_v, tex->height, tex->address_mode_v);
    float tz = cpu_apply_address_mode_1d(coord_w, tex->depth, tex->address_mode_w);

    if (tex->filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        // Nearest neighbor
        int x = (int)floor(tx + 0.5f);
        int y = (int)floor(ty + 0.5f);
        int z = (int)floor(tz + 0.5f);

        if (tex->address_mode_u != WP_TEXTURE_ADDRESS_BORDER) {
            x = cpu_clamp_index(x, tex->width);
        }
        if (tex->address_mode_v != WP_TEXTURE_ADDRESS_BORDER) {
            y = cpu_clamp_index(y, tex->height);
        }
        if (tex->address_mode_w != WP_TEXTURE_ADDRESS_BORDER) {
            z = cpu_clamp_index(z, tex->depth);
        }

        return cpu_fetch_texel_3d(tex, x, y, z, channel);
    } else {
        // Trilinear interpolation
        int x0 = (int)floor(tx);
        int y0 = (int)floor(ty);
        int z0 = (int)floor(tz);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        int z1 = z0 + 1;

        float fx = tx - x0;
        float fy = ty - y0;
        float fz = tz - z0;

        // Apply address mode to neighbor indices (properly handles wrap/mirror at edges)
        x0 = cpu_apply_address_mode_index(x0, tex->width, tex->address_mode_u);
        x1 = cpu_apply_address_mode_index(x1, tex->width, tex->address_mode_u);
        y0 = cpu_apply_address_mode_index(y0, tex->height, tex->address_mode_v);
        y1 = cpu_apply_address_mode_index(y1, tex->height, tex->address_mode_v);
        z0 = cpu_apply_address_mode_index(z0, tex->depth, tex->address_mode_w);
        z1 = cpu_apply_address_mode_index(z1, tex->depth, tex->address_mode_w);

        // Fetch 8 corner values
        float v000 = cpu_fetch_texel_3d(tex, x0, y0, z0, channel);
        float v100 = cpu_fetch_texel_3d(tex, x1, y0, z0, channel);
        float v010 = cpu_fetch_texel_3d(tex, x0, y1, z0, channel);
        float v110 = cpu_fetch_texel_3d(tex, x1, y1, z0, channel);
        float v001 = cpu_fetch_texel_3d(tex, x0, y0, z1, channel);
        float v101 = cpu_fetch_texel_3d(tex, x1, y0, z1, channel);
        float v011 = cpu_fetch_texel_3d(tex, x0, y1, z1, channel);
        float v111 = cpu_fetch_texel_3d(tex, x1, y1, z1, channel);

        // Trilinear interpolation
        float v00 = v000 * (1.0f - fx) + v100 * fx;
        float v10 = v010 * (1.0f - fx) + v110 * fx;
        float v01 = v001 * (1.0f - fx) + v101 * fx;
        float v11 = v011 * (1.0f - fx) + v111 * fx;

        float v0 = v00 * (1.0f - fy) + v10 * fy;
        float v1 = v01 * (1.0f - fy) + v11 * fy;

        return v0 * (1.0f - fz) + v1 * fz;
    }
}

// ============================================================================
// Texture Sampling Functions
// ============================================================================

// Helper to convert CUDA types to Warp types
template <typename T> struct texture_sample_helper;

template <> struct texture_sample_helper<float> {
    static CUDA_CALLABLE float sample_2d(const texture2d_t& tex, float u, float v)
    {
#if defined(__CUDA_ARCH__)
        return tex2D<float>(tex.tex, u, v);
#else
        if (tex.tex == 0)
            return 0.0f;
        const cpu_texture2d_data* cpu_tex = (const cpu_texture2d_data*)tex.tex;
        return cpu_sample_2d_channel(cpu_tex, u, v, 0);
#endif
    }

    static CUDA_CALLABLE float sample_3d(const texture3d_t& tex, float u, float v, float w)
    {
#if defined(__CUDA_ARCH__)
        return tex3D<float>(tex.tex, u, v, w);
#else
        if (tex.tex == 0)
            return 0.0f;
        const cpu_texture3d_data* cpu_tex = (const cpu_texture3d_data*)tex.tex;
        return cpu_sample_3d_channel(cpu_tex, u, v, w, 0);
#endif
    }

    static CUDA_CALLABLE float zero() { return 0.0f; }
};

template <> struct texture_sample_helper<vec2f> {
    static CUDA_CALLABLE vec2f sample_2d(const texture2d_t& tex, float u, float v)
    {
#if defined(__CUDA_ARCH__)
        float2 val = tex2D<float2>(tex.tex, u, v);
        return vec2f(val.x, val.y);
#else
        if (tex.tex == 0)
            return vec2f(0.0f, 0.0f);
        const cpu_texture2d_data* cpu_tex = (const cpu_texture2d_data*)tex.tex;
        return vec2f(cpu_sample_2d_channel(cpu_tex, u, v, 0), cpu_sample_2d_channel(cpu_tex, u, v, 1));
#endif
    }

    static CUDA_CALLABLE vec2f sample_3d(const texture3d_t& tex, float u, float v, float w)
    {
#if defined(__CUDA_ARCH__)
        float2 val = tex3D<float2>(tex.tex, u, v, w);
        return vec2f(val.x, val.y);
#else
        if (tex.tex == 0)
            return vec2f(0.0f, 0.0f);
        const cpu_texture3d_data* cpu_tex = (const cpu_texture3d_data*)tex.tex;
        return vec2f(cpu_sample_3d_channel(cpu_tex, u, v, w, 0), cpu_sample_3d_channel(cpu_tex, u, v, w, 1));
#endif
    }

    static CUDA_CALLABLE vec2f zero() { return vec2f(0.0f, 0.0f); }
};

template <> struct texture_sample_helper<vec4f> {
    static CUDA_CALLABLE vec4f sample_2d(const texture2d_t& tex, float u, float v)
    {
#if defined(__CUDA_ARCH__)
        float4 val = tex2D<float4>(tex.tex, u, v);
        return vec4f(val.x, val.y, val.z, val.w);
#else
        if (tex.tex == 0)
            return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
        const cpu_texture2d_data* cpu_tex = (const cpu_texture2d_data*)tex.tex;
        return vec4f(
            cpu_sample_2d_channel(cpu_tex, u, v, 0), cpu_sample_2d_channel(cpu_tex, u, v, 1),
            cpu_sample_2d_channel(cpu_tex, u, v, 2), cpu_sample_2d_channel(cpu_tex, u, v, 3)
        );
#endif
    }

    static CUDA_CALLABLE vec4f sample_3d(const texture3d_t& tex, float u, float v, float w)
    {
#if defined(__CUDA_ARCH__)
        float4 val = tex3D<float4>(tex.tex, u, v, w);
        return vec4f(val.x, val.y, val.z, val.w);
#else
        if (tex.tex == 0)
            return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
        const cpu_texture3d_data* cpu_tex = (const cpu_texture3d_data*)tex.tex;
        return vec4f(
            cpu_sample_3d_channel(cpu_tex, u, v, w, 0), cpu_sample_3d_channel(cpu_tex, u, v, w, 1),
            cpu_sample_3d_channel(cpu_tex, u, v, w, 2), cpu_sample_3d_channel(cpu_tex, u, v, w, 3)
        );
#endif
    }

    static CUDA_CALLABLE vec4f zero() { return vec4f(0.0f, 0.0f, 0.0f, 0.0f); }
};

// 2D texture sampling with vec2 coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture2d_t& tex, const vec2f& uv)
{
    return texture_sample_helper<T>::sample_2d(tex, uv[0], uv[1]);
}

// 2D texture sampling with separate u, v coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture2d_t& tex, float u, float v)
{
    return texture_sample_helper<T>::sample_2d(tex, u, v);
}

// 3D texture sampling with vec3 coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture3d_t& tex, const vec3f& uvw)
{
    return texture_sample_helper<T>::sample_3d(tex, uvw[0], uvw[1], uvw[2]);
}

// 3D texture sampling with separate u, v, w coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture3d_t& tex, float u, float v, float w)
{
    return texture_sample_helper<T>::sample_3d(tex, u, v, w);
}

// Adjoint stubs for texture sampling (non-differentiable for now)
template <typename T>
CUDA_CALLABLE void
adj_texture_sample(const texture2d_t& tex, const vec2f& uv, texture2d_t& adj_tex, vec2f& adj_uv, const T& adj_ret)
{
    // Texture sampling is not differentiable in this implementation
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture2d_t& tex, float u, float v, texture2d_t& adj_tex, float& adj_u, float& adj_v, const T& adj_ret
)
{
    // Texture sampling is not differentiable in this implementation
}

template <typename T>
CUDA_CALLABLE void
adj_texture_sample(const texture3d_t& tex, const vec3f& uvw, texture3d_t& adj_tex, vec3f& adj_uvw, const T& adj_ret)
{
    // Texture sampling is not differentiable in this implementation
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture3d_t& tex,
    float u,
    float v,
    float w,
    texture3d_t& adj_tex,
    float& adj_u,
    float& adj_v,
    float& adj_w,
    const T& adj_ret
)
{
    // Texture sampling is not differentiable in this implementation
}

// Type aliases for code generation
using Texture2D = texture2d_t;
using Texture3D = texture3d_t;

// ============================================================================
// Adjoint Support for Texture Types (required when textures are array dtypes)
// Textures are not differentiable, so these are essentially no-ops
// ============================================================================

// 2D Texture operations
CUDA_CALLABLE inline texture2d_t add(const texture2d_t& a, const texture2d_t& b)
{
    // Textures are not addable; return first argument unchanged
    return a;
}

CUDA_CALLABLE inline texture2d_t& operator+=(texture2d_t& a, const texture2d_t& b)
{
    // No-op: textures have no gradients to accumulate
    return a;
}

CUDA_CALLABLE inline void adj_atomic_add(texture2d_t* p, const texture2d_t& t)
{
    // No-op: textures are not differentiable
}

// 3D Texture operations
CUDA_CALLABLE inline texture3d_t add(const texture3d_t& a, const texture3d_t& b)
{
    // Textures are not addable; return first argument unchanged
    return a;
}

CUDA_CALLABLE inline texture3d_t& operator+=(texture3d_t& a, const texture3d_t& b)
{
    // No-op: textures have no gradients to accumulate
    return a;
}

CUDA_CALLABLE inline void adj_atomic_add(texture3d_t* p, const texture3d_t& t)
{
    // No-op: textures are not differentiable
}

}  // namespace wp
