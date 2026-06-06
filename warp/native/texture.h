// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "builtin.h"

namespace wp {

// Data type constants (must match texture.cpp and Python side)
#define WP_TEXTURE_DTYPE_UINT8   0
#define WP_TEXTURE_DTYPE_UINT16  1
#define WP_TEXTURE_DTYPE_FLOAT32 2
#define WP_TEXTURE_DTYPE_INT8    3
#define WP_TEXTURE_DTYPE_INT16   4
#define WP_TEXTURE_DTYPE_FLOAT16 5
#define WP_TEXTURE_DTYPE_UINT32  6
#define WP_TEXTURE_DTYPE_INT32   7

// Filter mode constants
#define WP_TEXTURE_FILTER_CLOSEST 0
#define WP_TEXTURE_FILTER_LINEAR 1

// Address mode constants
#define WP_TEXTURE_ADDRESS_WRAP   0
#define WP_TEXTURE_ADDRESS_CLAMP  1
#define WP_TEXTURE_ADDRESS_MIRROR 2
#define WP_TEXTURE_ADDRESS_BORDER 3

// Maximum number of mipmap levels supported per texture.
// CUDA textures are capped at 16 levels (16384x16384 base resolution).
#define WP_TEXTURE_MAX_MIP_LEVELS 16

// Sentinel value for the ``lod`` argument of ``texture_sample`` meaning
// "no level-of-detail requested". When callers omit ``lod`` the builtin's
// default (``-1.0``) routes sampling through the original non-LOD code path,
// preserving the behavior of single-mip textures.
#define WP_TEXTURE_LOD_DISABLED (-1.0f)

// Helper function to get bytes per channel from dtype
inline int get_texture_bytes_per_channel(int dtype)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
    case WP_TEXTURE_DTYPE_INT8:
        return 1;
    case WP_TEXTURE_DTYPE_UINT16:
    case WP_TEXTURE_DTYPE_INT16:
    case WP_TEXTURE_DTYPE_FLOAT16:
        return 2;
    case WP_TEXTURE_DTYPE_UINT32:
    case WP_TEXTURE_DTYPE_INT32:
    case WP_TEXTURE_DTYPE_FLOAT32:
        return 4;
    default:
        return 0;
    }
}

// Texture class for CPU or other hardware without texture units.
//
// Stores the full mipmap chain as a single contiguous buffer with per-level pointers.
// For a non-mipmapped texture (``num_mip_levels == 1``), ``data == mip_data[0]``.
struct Texture {

    // Unified constructor: allocates a contiguous buffer holding all mipmap levels
    // and initializes per-level pointers. Per-level shapes come from ``mip_widths``,
    // ``mip_heights`` and ``mip_depths`` (height/depth ignored for lower-dim textures).
    // If ``num_mip_levels == 1`` the texture is non-mipmapped and base dimensions are used.
    Texture(
        int32 ndim,
        int32 num_mip_levels,
        const int32* mip_widths,
        const int32* mip_heights,
        const int32* mip_depths,
        int32 num_channels,
        int32 dtype,
        int32 filter_mode,
        int32 mip_filter_mode,
        int32 address_mode_u,
        int32 address_mode_v,
        int32 address_mode_w,
        bool use_normalized_coords
    )
        : data(nullptr)
        , width(mip_widths[0])
        , height(ndim > 1 ? mip_heights[0] : 0)
        , depth(ndim > 2 ? mip_depths[0] : 0)
        , num_channels(num_channels)
        , dtype(dtype)
        , filter_mode(filter_mode)
        , mip_filter_mode(mip_filter_mode)
        , num_mip_levels(num_mip_levels)
        , address_mode_u(address_mode_u)
        , address_mode_v(address_mode_v)
        , address_mode_w(address_mode_w)
        , use_normalized_coords(use_normalized_coords)
        , mip_data {}
        , mip_offsets {}
        , mip_widths_arr {}
        , mip_heights_arr {}
        , mip_depths_arr {}
    {
        const int bytes_per_channel = get_texture_bytes_per_channel(dtype);

        size_t total_bytes = 0;
        for (int level = 0; level < num_mip_levels; ++level) {
            mip_widths_arr[level] = mip_widths[level];
            mip_heights_arr[level] = ndim > 1 ? mip_heights[level] : 1;
            mip_depths_arr[level] = ndim > 2 ? mip_depths[level] : 1;

            const size_t level_bytes = size_t(mip_widths_arr[level]) * size_t(mip_heights_arr[level])
                * size_t(mip_depths_arr[level]) * size_t(num_channels) * size_t(bytes_per_channel);

            mip_offsets[level] = total_bytes;
            total_bytes += level_bytes;
        }

        uint8* buffer = new uint8[total_bytes];
        data = buffer;

        for (int level = 0; level < num_mip_levels; ++level) {
            mip_data[level] = buffer + mip_offsets[level];
        }
    }

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    ~Texture() { delete[] static_cast<uint8*>(data); }

    void* data = nullptr;  // base-level pointer, equals mip_data[0]
    int32 width = 0;  // base-level width
    int32 height = 0;  // base-level height
    int32 depth = 0;  // base-level depth
    int32 num_channels = 0;
    int32 dtype = 0;
    int32 filter_mode = 0;
    int32 mip_filter_mode = 0;  // mipmap level filter mode (nearest or linear between levels)
    int32 num_mip_levels = 0;
    int32 address_mode_u = 0;  // Per-axis address mode for U
    int32 address_mode_v = 0;  // Per-axis address mode for V
    int32 address_mode_w = 0;  // Per-axis address mode for W
    bool use_normalized_coords = false;  // If true, coords in [0,1]; if false, in texel space

    // Per-level metadata (owned by the texture; indexed [0, num_mip_levels))
    void* mip_data[WP_TEXTURE_MAX_MIP_LEVELS] = {};
    size_t mip_offsets[WP_TEXTURE_MAX_MIP_LEVELS] = {};
    int32 mip_widths_arr[WP_TEXTURE_MAX_MIP_LEVELS] = {};
    int32 mip_heights_arr[WP_TEXTURE_MAX_MIP_LEVELS] = {};
    int32 mip_depths_arr[WP_TEXTURE_MAX_MIP_LEVELS] = {};
};

// Texture descriptor passed to kernels
// Contains the CUDA texture object handle (GPU) or pointer to cpu_texture*_data (CPU)
struct texture1d_t {
    uint64 tex;  // CUtexObject handle (GPU) or Texture* (CPU)
    int32 width;
    int32 num_channels;

    CUDA_CALLABLE inline texture1d_t()
        : tex(0)
        , width(0)
        , num_channels(0)
    {
    }

    CUDA_CALLABLE inline texture1d_t(uint64 tex, int32 width, int32 num_channels)
        : tex(tex)
        , width(width)
        , num_channels(num_channels)
    {
    }
};

struct texture2d_t {
    uint64 tex;  // CUtexObject handle (GPU) or Texture* (CPU)
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
    uint64 tex;  // CUtexObject handle (GPU) or Texture* (CPU)
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

struct cuda_array_desc_t {
    int32 ndim;
    int32 shape[3];
    int32 num_channels;
    int32 dtype;
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
inline bool cpu_in_bounds_1d(int x, int w) { return x >= 0 && x < w; }

inline bool cpu_in_bounds_2d(int x, int y, int w, int h) { return x >= 0 && x < w && y >= 0 && y < h; }

inline bool cpu_in_bounds_3d(int x, int y, int z, int w, int h, int d)
{
    return x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d;
}

// Clamp a LOD value to the texture's valid mip-level range.
inline float cpu_clamp_lod(const Texture* tex, float lod)
{
    if (lod < 0.0f)
        return 0.0f;
    float max_lod = (float)(tex->num_mip_levels - 1);
    return lod > max_lod ? max_lod : lod;
}

// Convert IEEE 754 half-precision bits to float (for CPU float16 texture support)
inline float cpu_half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t result;
    if (exp == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            // Denormalized: convert to normalized float
            exp = 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x3FF;
            result = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)mantissa << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        result = sign | 0x7F800000u | ((uint32_t)mantissa << 13);
    } else {
        result = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)mantissa << 13);
    }

    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = result;
    return conv.f;
}

// Decode a raw texel at ``idx`` into a normalized float.
// Unsigned integers are normalized to [0, 1], signed integers to [-1, 1],
// float types are returned as-is.
inline float cpu_decode_texel(const void* level_data, int dtype, int idx)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return ((const uint8_t*)level_data)[idx] / 255.0f;
    case WP_TEXTURE_DTYPE_UINT16:
        return ((const uint16_t*)level_data)[idx] / 65535.0f;
    case WP_TEXTURE_DTYPE_UINT32:
        return ((const uint32_t*)level_data)[idx] / 4294967295.0f;
    case WP_TEXTURE_DTYPE_INT8: {
        float v = ((const int8_t*)level_data)[idx] / 127.0f;
        return v < -1.0f ? -1.0f : v;
    }
    case WP_TEXTURE_DTYPE_INT16: {
        float v = ((const int16_t*)level_data)[idx] / 32767.0f;
        return v < -1.0f ? -1.0f : v;
    }
    case WP_TEXTURE_DTYPE_INT32: {
        float v = ((const int32_t*)level_data)[idx] / 2147483647.0f;
        return v < -1.0f ? -1.0f : v;
    }
    case WP_TEXTURE_DTYPE_FLOAT16:
        return cpu_half_to_float(((const uint16_t*)level_data)[idx]);
    case WP_TEXTURE_DTYPE_FLOAT32:
    default:
        return ((const float*)level_data)[idx];
    }
}

// Fetch a single texel value from the given mip level.
inline float cpu_fetch_texel_1d(const Texture* tex, int level, int x, int channel)
{
    const int w = tex->mip_widths_arr[level];
    if (!cpu_in_bounds_1d(x, w) || channel < 0 || channel >= tex->num_channels) {
        return 0.0f;
    }
    int idx = x * tex->num_channels + channel;
    return cpu_decode_texel(tex->mip_data[level], tex->dtype, idx);
}

inline float cpu_fetch_texel_2d(const Texture* tex, int level, int x, int y, int channel)
{
    const int w = tex->mip_widths_arr[level];
    const int h = tex->mip_heights_arr[level];
    if (!cpu_in_bounds_2d(x, y, w, h) || channel < 0 || channel >= tex->num_channels) {
        return 0.0f;
    }
    int idx = (y * w + x) * tex->num_channels + channel;
    return cpu_decode_texel(tex->mip_data[level], tex->dtype, idx);
}

inline float cpu_fetch_texel_3d(const Texture* tex, int level, int x, int y, int z, int channel)
{
    const int w = tex->mip_widths_arr[level];
    const int h = tex->mip_heights_arr[level];
    const int d = tex->mip_depths_arr[level];
    if (!cpu_in_bounds_3d(x, y, z, w, h, d) || channel < 0 || channel >= tex->num_channels) {
        return 0.0f;
    }
    int idx = ((z * h + y) * w + x) * tex->num_channels + channel;
    return cpu_decode_texel(tex->mip_data[level], tex->dtype, idx);
}

// Sample a single channel with linear interpolation (1D) at a specific mip level.
inline float cpu_sample_1d_channel_at_level(const Texture* tex, int level, float u, int channel)
{
    const int w = tex->mip_widths_arr[level];

    // Convert to texel space if using normalized coordinates
    float coord_u = tex->use_normalized_coords ? u : (u / (float)tex->width);

    float tx = cpu_apply_address_mode_1d(coord_u, w, tex->address_mode_u);

    if (tex->filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        // Nearest neighbor
        int x = (int)floor(tx + 0.5f);

        if (tex->address_mode_u != WP_TEXTURE_ADDRESS_BORDER) {
            x = cpu_clamp_index(x, w);
        }

        return cpu_fetch_texel_1d(tex, level, x, channel);
    } else {
        // Linear interpolation
        int x0 = (int)floor(tx);
        int x1 = x0 + 1;

        float fx = tx - x0;

        // Apply address mode to neighbor indices (properly handles wrap/mirror at edges)
        x0 = cpu_apply_address_mode_index(x0, w, tex->address_mode_u);
        x1 = cpu_apply_address_mode_index(x1, w, tex->address_mode_u);

        float v0 = cpu_fetch_texel_1d(tex, level, x0, channel);
        float v1 = cpu_fetch_texel_1d(tex, level, x1, channel);

        return v0 * (1.0f - fx) + v1 * fx;
    }
}

// Sample a single channel with bilinear interpolation (2D) at a specific mip level.
inline float cpu_sample_2d_channel_at_level(const Texture* tex, int level, float u, float v, int channel)
{
    const int w = tex->mip_widths_arr[level];
    const int h = tex->mip_heights_arr[level];

    float coord_u = tex->use_normalized_coords ? u : (u / (float)tex->width);
    float coord_v = tex->use_normalized_coords ? v : (v / (float)tex->height);

    float tx = cpu_apply_address_mode_1d(coord_u, w, tex->address_mode_u);
    float ty = cpu_apply_address_mode_1d(coord_v, h, tex->address_mode_v);

    if (tex->filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        int x = (int)floor(tx + 0.5f);
        int y = (int)floor(ty + 0.5f);

        if (tex->address_mode_u != WP_TEXTURE_ADDRESS_BORDER) {
            x = cpu_clamp_index(x, w);
        }
        if (tex->address_mode_v != WP_TEXTURE_ADDRESS_BORDER) {
            y = cpu_clamp_index(y, h);
        }

        return cpu_fetch_texel_2d(tex, level, x, y, channel);
    } else {
        int x0 = (int)floor(tx);
        int y0 = (int)floor(ty);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float fx = tx - x0;
        float fy = ty - y0;

        x0 = cpu_apply_address_mode_index(x0, w, tex->address_mode_u);
        x1 = cpu_apply_address_mode_index(x1, w, tex->address_mode_u);
        y0 = cpu_apply_address_mode_index(y0, h, tex->address_mode_v);
        y1 = cpu_apply_address_mode_index(y1, h, tex->address_mode_v);

        float v00 = cpu_fetch_texel_2d(tex, level, x0, y0, channel);
        float v10 = cpu_fetch_texel_2d(tex, level, x1, y0, channel);
        float v01 = cpu_fetch_texel_2d(tex, level, x0, y1, channel);
        float v11 = cpu_fetch_texel_2d(tex, level, x1, y1, channel);

        float v0 = v00 * (1.0f - fx) + v10 * fx;
        float v1 = v01 * (1.0f - fx) + v11 * fx;
        return v0 * (1.0f - fy) + v1 * fy;
    }
}

// Sample a single channel with trilinear interpolation (3D) at a specific mip level.
inline float cpu_sample_3d_channel_at_level(const Texture* tex, int level, float u, float v, float w_coord, int channel)
{
    const int w = tex->mip_widths_arr[level];
    const int h = tex->mip_heights_arr[level];
    const int d = tex->mip_depths_arr[level];

    float coord_u = tex->use_normalized_coords ? u : (u / (float)tex->width);
    float coord_v = tex->use_normalized_coords ? v : (v / (float)tex->height);
    float coord_w = tex->use_normalized_coords ? w_coord : (w_coord / (float)tex->depth);

    float tx = cpu_apply_address_mode_1d(coord_u, w, tex->address_mode_u);
    float ty = cpu_apply_address_mode_1d(coord_v, h, tex->address_mode_v);
    float tz = cpu_apply_address_mode_1d(coord_w, d, tex->address_mode_w);

    if (tex->filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        int x = (int)floor(tx + 0.5f);
        int y = (int)floor(ty + 0.5f);
        int z = (int)floor(tz + 0.5f);

        if (tex->address_mode_u != WP_TEXTURE_ADDRESS_BORDER) {
            x = cpu_clamp_index(x, w);
        }
        if (tex->address_mode_v != WP_TEXTURE_ADDRESS_BORDER) {
            y = cpu_clamp_index(y, h);
        }
        if (tex->address_mode_w != WP_TEXTURE_ADDRESS_BORDER) {
            z = cpu_clamp_index(z, d);
        }

        return cpu_fetch_texel_3d(tex, level, x, y, z, channel);
    } else {
        int x0 = (int)floor(tx);
        int y0 = (int)floor(ty);
        int z0 = (int)floor(tz);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        int z1 = z0 + 1;

        float fx = tx - x0;
        float fy = ty - y0;
        float fz = tz - z0;

        x0 = cpu_apply_address_mode_index(x0, w, tex->address_mode_u);
        x1 = cpu_apply_address_mode_index(x1, w, tex->address_mode_u);
        y0 = cpu_apply_address_mode_index(y0, h, tex->address_mode_v);
        y1 = cpu_apply_address_mode_index(y1, h, tex->address_mode_v);
        z0 = cpu_apply_address_mode_index(z0, d, tex->address_mode_w);
        z1 = cpu_apply_address_mode_index(z1, d, tex->address_mode_w);

        float v000 = cpu_fetch_texel_3d(tex, level, x0, y0, z0, channel);
        float v100 = cpu_fetch_texel_3d(tex, level, x1, y0, z0, channel);
        float v010 = cpu_fetch_texel_3d(tex, level, x0, y1, z0, channel);
        float v110 = cpu_fetch_texel_3d(tex, level, x1, y1, z0, channel);
        float v001 = cpu_fetch_texel_3d(tex, level, x0, y0, z1, channel);
        float v101 = cpu_fetch_texel_3d(tex, level, x1, y0, z1, channel);
        float v011 = cpu_fetch_texel_3d(tex, level, x0, y1, z1, channel);
        float v111 = cpu_fetch_texel_3d(tex, level, x1, y1, z1, channel);

        float v00 = v000 * (1.0f - fx) + v100 * fx;
        float v10 = v010 * (1.0f - fx) + v110 * fx;
        float v01 = v001 * (1.0f - fx) + v101 * fx;
        float v11 = v011 * (1.0f - fx) + v111 * fx;

        float v0 = v00 * (1.0f - fy) + v10 * fy;
        float v1 = v01 * (1.0f - fy) + v11 * fy;

        return v0 * (1.0f - fz) + v1 * fz;
    }
}

// Sample a single channel across mipmap levels using the texture's mip filter mode.
inline float cpu_sample_1d_channel(const Texture* tex, float u, int channel, float lod)
{
    float clamped_lod = cpu_clamp_lod(tex, lod);
    int level0 = (int)floor(clamped_lod);
    if (tex->num_mip_levels <= 1 || tex->mip_filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        int nearest = (int)floor(clamped_lod + 0.5f);
        if (nearest >= tex->num_mip_levels)
            nearest = tex->num_mip_levels - 1;
        return cpu_sample_1d_channel_at_level(tex, nearest, u, channel);
    }
    int level1 = level0 + 1;
    if (level1 >= tex->num_mip_levels)
        level1 = tex->num_mip_levels - 1;
    float fl = clamped_lod - (float)level0;
    float v0 = cpu_sample_1d_channel_at_level(tex, level0, u, channel);
    float v1 = cpu_sample_1d_channel_at_level(tex, level1, u, channel);
    return v0 * (1.0f - fl) + v1 * fl;
}

inline float cpu_sample_2d_channel(const Texture* tex, float u, float v, int channel, float lod)
{
    float clamped_lod = cpu_clamp_lod(tex, lod);
    int level0 = (int)floor(clamped_lod);
    if (tex->num_mip_levels <= 1 || tex->mip_filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        int nearest = (int)floor(clamped_lod + 0.5f);
        if (nearest >= tex->num_mip_levels)
            nearest = tex->num_mip_levels - 1;
        return cpu_sample_2d_channel_at_level(tex, nearest, u, v, channel);
    }
    int level1 = level0 + 1;
    if (level1 >= tex->num_mip_levels)
        level1 = tex->num_mip_levels - 1;
    float fl = clamped_lod - (float)level0;
    float v0 = cpu_sample_2d_channel_at_level(tex, level0, u, v, channel);
    float v1 = cpu_sample_2d_channel_at_level(tex, level1, u, v, channel);
    return v0 * (1.0f - fl) + v1 * fl;
}

inline float cpu_sample_3d_channel(const Texture* tex, float u, float v, float w, int channel, float lod)
{
    float clamped_lod = cpu_clamp_lod(tex, lod);
    int level0 = (int)floor(clamped_lod);
    if (tex->num_mip_levels <= 1 || tex->mip_filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        int nearest = (int)floor(clamped_lod + 0.5f);
        if (nearest >= tex->num_mip_levels)
            nearest = tex->num_mip_levels - 1;
        return cpu_sample_3d_channel_at_level(tex, nearest, u, v, w, channel);
    }
    int level1 = level0 + 1;
    if (level1 >= tex->num_mip_levels)
        level1 = tex->num_mip_levels - 1;
    float fl = clamped_lod - (float)level0;
    float v0 = cpu_sample_3d_channel_at_level(tex, level0, u, v, w, channel);
    float v1 = cpu_sample_3d_channel_at_level(tex, level1, u, v, w, channel);
    return v0 * (1.0f - fl) + v1 * fl;
}

// ============================================================================
// Texture Sampling Functions
// ============================================================================

// TODO: Implement texture fetch functions for Clang CUDA JIT (currently stubbed, texture API is not supported)
#if defined(__clang__) && defined(__CUDA__)
template <typename T> __device__ T tex1D(unsigned long long texObj, float x)
{
    T v {};
    return v;
}
template <typename T> __device__ T tex2D(unsigned long long texObj, float x, float y)
{
    T v {};
    return v;
}
template <typename T> __device__ T tex3D(unsigned long long texObj, float x, float y, float z)
{
    T v {};
    return v;
}
template <typename T> __device__ T tex1DLod(unsigned long long texObj, float x, float lod)
{
    T v {};
    return v;
}
template <typename T> __device__ T tex2DLod(unsigned long long texObj, float x, float y, float lod)
{
    T v {};
    return v;
}
template <typename T> __device__ T tex3DLod(unsigned long long texObj, float x, float y, float z, float lod)
{
    T v {};
    return v;
}
#endif

// Helper to convert CUDA types to Warp types
//
// Each ``sample_*`` helper accepts a single ``lod`` argument. When ``lod`` equals
// :c:macro:`WP_TEXTURE_LOD_DISABLED`, the helper dispatches to the original non-LOD
// texture fetch (``tex1D`` / ``tex2D`` / ``tex3D`` on GPU, base mip level on CPU)
// so that callers that never opt into LOD preserve the pre-mipmap behavior exactly.
// Any other ``lod`` value routes through the mipmap-aware path (``tex1DLod`` / ...).
template <typename T> struct texture_sample_helper;

template <> struct texture_sample_helper<float> {
    static CUDA_CALLABLE float sample_1d(const texture1d_t& tex, float u, float lod)
    {
#if defined(__CUDA_ARCH__)
        if (lod < 0.0f)
            return tex1D<float>(tex.tex, u);
        return tex1DLod<float>(tex.tex, u, lod);
#else
        if (tex.tex == 0)
            return 0.0f;
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 0);
        return cpu_sample_1d_channel(cpu_tex, u, 0, lod);
#endif
    }

    static CUDA_CALLABLE float sample_2d(const texture2d_t& tex, float u, float v, float lod)
    {
#if defined(__CUDA_ARCH__)
        if (lod < 0.0f)
            return tex2D<float>(tex.tex, u, v);
        return tex2DLod<float>(tex.tex, u, v, lod);
#else
        if (tex.tex == 0)
            return 0.0f;
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 0);
        return cpu_sample_2d_channel(cpu_tex, u, v, 0, lod);
#endif
    }

    static CUDA_CALLABLE float sample_3d(const texture3d_t& tex, float u, float v, float w, float lod)
    {
#if defined(__CUDA_ARCH__)
        if (lod < 0.0f)
            return tex3D<float>(tex.tex, u, v, w);
        return tex3DLod<float>(tex.tex, u, v, w, lod);
#else
        if (tex.tex == 0)
            return 0.0f;
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 0);
        return cpu_sample_3d_channel(cpu_tex, u, v, w, 0, lod);
#endif
    }

    static CUDA_CALLABLE float zero() { return 0.0f; }
};

template <> struct texture_sample_helper<vec2f> {
    static CUDA_CALLABLE vec2f sample_1d(const texture1d_t& tex, float u, float lod)
    {
#if defined(__CUDA_ARCH__)
        float2 val = (lod < 0.0f) ? tex1D<float2>(tex.tex, u) : tex1DLod<float2>(tex.tex, u, lod);
        return vec2f(val.x, val.y);
#else
        if (tex.tex == 0)
            return vec2f(0.0f, 0.0f);
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return vec2f(
                cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 0), cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 1)
            );
        return vec2f(cpu_sample_1d_channel(cpu_tex, u, 0, lod), cpu_sample_1d_channel(cpu_tex, u, 1, lod));
#endif
    }

    static CUDA_CALLABLE vec2f sample_2d(const texture2d_t& tex, float u, float v, float lod)
    {
#if defined(__CUDA_ARCH__)
        float2 val = (lod < 0.0f) ? tex2D<float2>(tex.tex, u, v) : tex2DLod<float2>(tex.tex, u, v, lod);
        return vec2f(val.x, val.y);
#else
        if (tex.tex == 0)
            return vec2f(0.0f, 0.0f);
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return vec2f(
                cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 0), cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 1)
            );
        return vec2f(cpu_sample_2d_channel(cpu_tex, u, v, 0, lod), cpu_sample_2d_channel(cpu_tex, u, v, 1, lod));
#endif
    }

    static CUDA_CALLABLE vec2f sample_3d(const texture3d_t& tex, float u, float v, float w, float lod)
    {
#if defined(__CUDA_ARCH__)
        float2 val = (lod < 0.0f) ? tex3D<float2>(tex.tex, u, v, w) : tex3DLod<float2>(tex.tex, u, v, w, lod);
        return vec2f(val.x, val.y);
#else
        if (tex.tex == 0)
            return vec2f(0.0f, 0.0f);
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return vec2f(
                cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 0),
                cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 1)
            );
        return vec2f(cpu_sample_3d_channel(cpu_tex, u, v, w, 0, lod), cpu_sample_3d_channel(cpu_tex, u, v, w, 1, lod));
#endif
    }

    static CUDA_CALLABLE vec2f zero() { return vec2f(0.0f, 0.0f); }
};

template <> struct texture_sample_helper<vec4f> {
    static CUDA_CALLABLE vec4f sample_1d(const texture1d_t& tex, float u, float lod)
    {
#if defined(__CUDA_ARCH__)
        float4 val = (lod < 0.0f) ? tex1D<float4>(tex.tex, u) : tex1DLod<float4>(tex.tex, u, lod);
        return vec4f(val.x, val.y, val.z, val.w);
#else
        if (tex.tex == 0)
            return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return vec4f(
                cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 0), cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 1),
                cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 2), cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 3)
            );
        return vec4f(
            cpu_sample_1d_channel(cpu_tex, u, 0, lod), cpu_sample_1d_channel(cpu_tex, u, 1, lod),
            cpu_sample_1d_channel(cpu_tex, u, 2, lod), cpu_sample_1d_channel(cpu_tex, u, 3, lod)
        );
#endif
    }

    static CUDA_CALLABLE vec4f sample_2d(const texture2d_t& tex, float u, float v, float lod)
    {
#if defined(__CUDA_ARCH__)
        float4 val = (lod < 0.0f) ? tex2D<float4>(tex.tex, u, v) : tex2DLod<float4>(tex.tex, u, v, lod);
        return vec4f(val.x, val.y, val.z, val.w);
#else
        if (tex.tex == 0)
            return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return vec4f(
                cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 0),
                cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 1),
                cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 2), cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 3)
            );
        return vec4f(
            cpu_sample_2d_channel(cpu_tex, u, v, 0, lod), cpu_sample_2d_channel(cpu_tex, u, v, 1, lod),
            cpu_sample_2d_channel(cpu_tex, u, v, 2, lod), cpu_sample_2d_channel(cpu_tex, u, v, 3, lod)
        );
#endif
    }

    static CUDA_CALLABLE vec4f sample_3d(const texture3d_t& tex, float u, float v, float w, float lod)
    {
#if defined(__CUDA_ARCH__)
        float4 val = (lod < 0.0f) ? tex3D<float4>(tex.tex, u, v, w) : tex3DLod<float4>(tex.tex, u, v, w, lod);
        return vec4f(val.x, val.y, val.z, val.w);
#else
        if (tex.tex == 0)
            return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return vec4f(
                cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 0),
                cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 1),
                cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 2),
                cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 3)
            );
        return vec4f(
            cpu_sample_3d_channel(cpu_tex, u, v, w, 0, lod), cpu_sample_3d_channel(cpu_tex, u, v, w, 1, lod),
            cpu_sample_3d_channel(cpu_tex, u, v, w, 2, lod), cpu_sample_3d_channel(cpu_tex, u, v, w, 3, lod)
        );
#endif
    }

    static CUDA_CALLABLE vec4f zero() { return vec4f(0.0f, 0.0f, 0.0f, 0.0f); }
};

// 1D texture sampling with scalar coordinate
template <typename T> CUDA_CALLABLE T texture_sample(const texture1d_t& tex, float u, float lod)
{
    return texture_sample_helper<T>::sample_1d(tex, u, lod);
}

// 2D texture sampling with vec2 coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture2d_t& tex, const vec2f& uv, float lod)
{
    return texture_sample_helper<T>::sample_2d(tex, uv[0], uv[1], lod);
}

// 2D texture sampling with separate u, v coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture2d_t& tex, float u, float v, float lod)
{
    return texture_sample_helper<T>::sample_2d(tex, u, v, lod);
}

// 3D texture sampling with vec3 coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture3d_t& tex, const vec3f& uvw, float lod)
{
    return texture_sample_helper<T>::sample_3d(tex, uvw[0], uvw[1], uvw[2], lod);
}

// 3D texture sampling with separate u, v, w coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture3d_t& tex, float u, float v, float w, float lod)
{
    return texture_sample_helper<T>::sample_3d(tex, u, v, w, lod);
}

// Adjoint stubs for texture sampling
template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture1d_t& tex, float u, float lod, texture1d_t& adj_tex, float& adj_u, float& adj_lod, const T& adj_ret
)
{
    // MISSINGADJOINT: differentiable for linear interpolation;
    // route adj_ret to neighboring texels and mip levels (weighted by interpolation factors)
    // and to adj_u/adj_lod (via the interpolation derivatives along U and LOD).
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture2d_t& tex,
    const vec2f& uv,
    float lod,
    texture2d_t& adj_tex,
    vec2f& adj_uv,
    float& adj_lod,
    const T& adj_ret
)
{
    // MISSINGADJOINT: differentiable for linear interpolation;
    // route adj_ret to the four neighboring texels and mip levels (weighted by interpolation factors)
    // and to adj_uv/adj_lod (via the interpolation derivatives along U, V, and LOD).
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture2d_t& tex,
    float u,
    float v,
    float lod,
    texture2d_t& adj_tex,
    float& adj_u,
    float& adj_v,
    float& adj_lod,
    const T& adj_ret
)
{
    // MISSINGADJOINT: differentiable for linear interpolation;
    // route adj_ret to the four neighboring texels and mip levels (weighted by interpolation factors)
    // and to adj_u/adj_v/adj_lod (via the interpolation derivatives along each axis and LOD).
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture3d_t& tex,
    const vec3f& uvw,
    float lod,
    texture3d_t& adj_tex,
    vec3f& adj_uvw,
    float& adj_lod,
    const T& adj_ret
)
{
    // MISSINGADJOINT: differentiable for linear interpolation;
    // route adj_ret to the eight neighboring texels and mip levels (weighted by interpolation factors)
    // and to adj_uvw/adj_lod (via the interpolation derivatives along U, V, W, and LOD).
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture3d_t& tex,
    float u,
    float v,
    float w,
    float lod,
    texture3d_t& adj_tex,
    float& adj_u,
    float& adj_v,
    float& adj_w,
    float& adj_lod,
    const T& adj_ret
)
{
    // MISSINGADJOINT: differentiable for linear interpolation;
    // route adj_ret to the eight neighboring texels and mip levels (weighted by interpolation factors)
    // and to adj_u/adj_v/adj_w/adj_lod (via the interpolation derivatives along each axis and LOD).
}

// Type aliases for code generation
using Texture1D = texture1d_t;
using Texture2D = texture2d_t;
using Texture3D = texture3d_t;

// ============================================================================
// Adjoint Support for Texture Types (required when textures are array dtypes)
// Textures are not differentiable, so these are essentially no-ops
// ============================================================================

// 1D Texture operations
CUDA_CALLABLE inline texture1d_t add(const texture1d_t& a, const texture1d_t& b)
{
    // Textures are not addable; return first argument unchanged
    return a;
}

CUDA_CALLABLE inline texture1d_t& operator+=(texture1d_t& a, const texture1d_t& b)
{
    // No-op: textures have no gradients to accumulate
    return a;
}

CUDA_CALLABLE inline void adj_atomic_add(texture1d_t* p, const texture1d_t& t)
{
    // No-op: textures are not differentiable
}

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
