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
CUDA_CALLABLE inline uint32 texture_metadata(
    int32 num_channels,
    int32 dtype,
    int32 filter_mode,
    int32 mip_filter_mode,
    int32 num_mip_levels,
    int32 use_normalized_coords,
    int32 address_mode_u = WP_TEXTURE_ADDRESS_CLAMP,
    int32 address_mode_v = WP_TEXTURE_ADDRESS_CLAMP,
    int32 address_mode_w = WP_TEXTURE_ADDRESS_CLAMP
)
{
    return (uint32)num_channels | ((uint32)dtype << 3) | ((uint32)filter_mode << 6) | ((uint32)mip_filter_mode << 7)
        | ((uint32)num_mip_levels << 8) | ((uint32)use_normalized_coords << 13) | ((uint32)address_mode_u << 14)
        | ((uint32)address_mode_v << 16) | ((uint32)address_mode_w << 18);
}

#define WP_TEXTURE_METADATA_ACCESSORS                                                                                  \
    CUDA_CALLABLE inline int32 get_num_channels() const { return (int32)(metadata & 0x7u); }                           \
    CUDA_CALLABLE inline int32 get_dtype() const { return (int32)((metadata >> 3) & 0x7u); }                           \
    CUDA_CALLABLE inline int32 get_filter_mode() const { return (int32)((metadata >> 6) & 0x1u); }                     \
    CUDA_CALLABLE inline int32 get_mip_filter_mode() const { return (int32)((metadata >> 7) & 0x1u); }                 \
    CUDA_CALLABLE inline int32 get_num_mip_levels() const { return (int32)((metadata >> 8) & 0x1fu); }                 \
    CUDA_CALLABLE inline int32 get_use_normalized_coords() const { return (int32)((metadata >> 13) & 0x1u); } \
    CUDA_CALLABLE inline int32 get_address_mode_u() const { return (int32)((metadata >> 14) & 0x3u); } \
    CUDA_CALLABLE inline int32 get_address_mode_v() const { return (int32)((metadata >> 16) & 0x3u); } \
    CUDA_CALLABLE inline int32 get_address_mode_w() const { return (int32)((metadata >> 18) & 0x3u); }

struct texture1d_t {
    uint64 tex;  // CUtexObject handle (GPU) or Texture* (CPU)
    int32 width;
    uint32 metadata;
    WP_TEXTURE_METADATA_ACCESSORS

    CUDA_CALLABLE inline texture1d_t()
        : tex(0)
        , width(0)
        , metadata(texture_metadata(0, 0, 0, 0, 1, 1))
    {
    }

    CUDA_CALLABLE inline texture1d_t(
        uint64 tex,
        int32 width,
        int32 num_channels,
        int32 dtype,
        int32 filter_mode,
        int32 mip_filter_mode,
        int32 num_mip_levels,
        int32 use_normalized_coords,
        int32 address_mode_u = WP_TEXTURE_ADDRESS_CLAMP,
        int32 address_mode_v = WP_TEXTURE_ADDRESS_CLAMP,
        int32 address_mode_w = WP_TEXTURE_ADDRESS_CLAMP
    )
        : tex(tex)
        , width(width)
        , metadata(texture_metadata(
              num_channels,
              dtype,
              filter_mode,
              mip_filter_mode,
              num_mip_levels,
              use_normalized_coords,
              address_mode_u,
              address_mode_v,
              address_mode_w
          ))
    {
    }
};

struct texture2d_t {
    uint64 tex;  // CUtexObject handle (GPU) or Texture* (CPU)
    int32 width;
    int32 height;
    uint32 metadata;
    WP_TEXTURE_METADATA_ACCESSORS

    CUDA_CALLABLE inline texture2d_t()
        : tex(0)
        , width(0)
        , height(0)
        , metadata(texture_metadata(0, 0, 0, 0, 1, 1))
    {
    }

    CUDA_CALLABLE inline texture2d_t(
        uint64 tex,
        int32 width,
        int32 height,
        int32 num_channels,
        int32 dtype,
        int32 filter_mode,
        int32 mip_filter_mode,
        int32 num_mip_levels,
        int32 use_normalized_coords,
        int32 address_mode_u = WP_TEXTURE_ADDRESS_CLAMP,
        int32 address_mode_v = WP_TEXTURE_ADDRESS_CLAMP,
        int32 address_mode_w = WP_TEXTURE_ADDRESS_CLAMP
    )
        : tex(tex)
        , width(width)
        , height(height)
        , metadata(texture_metadata(
              num_channels,
              dtype,
              filter_mode,
              mip_filter_mode,
              num_mip_levels,
              use_normalized_coords,
              address_mode_u,
              address_mode_v,
              address_mode_w
          ))
    {
    }
};

struct texture3d_t {
    uint64 tex;  // CUtexObject handle (GPU) or Texture* (CPU)
    int32 width;
    int32 height;
    int32 depth;
    uint32 metadata;
    WP_TEXTURE_METADATA_ACCESSORS

    CUDA_CALLABLE inline texture3d_t()
        : tex(0)
        , width(0)
        , height(0)
        , depth(0)
        , metadata(texture_metadata(0, 0, 0, 0, 1, 1))
    {
    }

    CUDA_CALLABLE inline texture3d_t(
        uint64 tex,
        int32 width,
        int32 height,
        int32 depth,
        int32 num_channels,
        int32 dtype,
        int32 filter_mode,
        int32 mip_filter_mode,
        int32 num_mip_levels,
        int32 use_normalized_coords,
        int32 address_mode_u = WP_TEXTURE_ADDRESS_CLAMP,
        int32 address_mode_v = WP_TEXTURE_ADDRESS_CLAMP,
        int32 address_mode_w = WP_TEXTURE_ADDRESS_CLAMP
    )
        : tex(tex)
        , width(width)
        , height(height)
        , depth(depth)
        , metadata(texture_metadata(
              num_channels,
              dtype,
              filter_mode,
              mip_filter_mode,
              num_mip_levels,
              use_normalized_coords,
              address_mode_u,
              address_mode_v,
              address_mode_w
          ))
    {
    }
};

static_assert(sizeof(texture1d_t) == 16, "texture1d_t metadata must remain packed");
static_assert(sizeof(texture2d_t) == 24, "texture2d_t metadata must remain packed");
static_assert(sizeof(texture3d_t) == 24, "texture3d_t metadata must remain packed");

#undef WP_TEXTURE_METADATA_ACCESSORS

struct cuda_array_desc_t {
    int32 ndim;
    int32 shape[3];
    int32 num_channels;
    int32 dtype;
};

// ============================================================================
// CPU Software Texture Sampling Helpers
// ============================================================================

#if !defined(__CUDA_ARCH__)

// Reduce periodic coordinates before division or scaling can discard the
// fractional position (or overflow). Unnormalized CPU textures also support
// periodic addressing, unlike CUDA textures.
inline float cpu_texture_normalized_coord(float coord, int normalized, int base_dim, int address_mode)
{
    if (address_mode == WP_TEXTURE_ADDRESS_WRAP || address_mode == WP_TEXTURE_ADDRESS_MIRROR) {
        const float period
            = (normalized ? 1.0f : float(base_dim)) * (address_mode == WP_TEXTURE_ADDRESS_MIRROR ? 2.0f : 1.0f);
        coord = fmodf(coord, period);
        if (coord < 0.0f)
            coord += period;
    }
    return normalized ? coord : coord / float(base_dim);
}

// Convert reduced normalized coordinates to bounded texel coordinates.
// Apply periodic addressing to integer indices after rounding, so CLOSEST
// filtering chooses the same side of a MIRROR boundary as the CUDA sampler.
inline float cpu_apply_address_mode_1d(float coord, int size, int address_mode)
{
    float texel = coord * float(size) - 0.5f;
    if (address_mode == WP_TEXTURE_ADDRESS_CLAMP)
        texel = max(0.0f, min(texel, float(size - 1)));
    else if (address_mode != WP_TEXTURE_ADDRESS_WRAP && address_mode != WP_TEXTURE_ADDRESS_MIRROR)
        // Keep both neighboring indices outside the texture without converting
        // an unbounded float to int, including when finite coord * size overflows.
        texel = max(-2.0f, min(texel, float(size)));
    return texel;
}

// Clamp integer index to valid range
inline int64 cpu_clamp_index(int64 idx, int size) { return (idx < 0) ? 0 : ((idx >= size) ? size - 1 : idx); }

// Apply address mode to an integer texel index
// This is used for neighbor indices in linear filtering to properly handle wrap/mirror
inline int64 cpu_apply_address_mode_index(int64 idx, int size, int address_mode)
{
    switch (address_mode) {
    case WP_TEXTURE_ADDRESS_WRAP: {
        int64 m = idx % size;
        if (m < 0)
            m += size;
        return m;
    }
    case WP_TEXTURE_ADDRESS_CLAMP:
        return cpu_clamp_index(idx, size);
    case WP_TEXTURE_ADDRESS_MIRROR: {
        int64 period = 2 * int64(size);
        int64 m = idx % period;
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

// Nearest filtering rounds the original coordinate down to a texel index.
// Avoid the linear sampler's subtract/add-half round trip, which can erase a
// small negative coordinate before BORDER addressing. Wide arithmetic also
// preserves its sign through normalization, including float32 subnormals.
inline int64 cpu_nearest_texel_index(float coord, int normalized, int base_dim, int level_dim, int address_mode)
{
    double reduced = double(coord);
    if (address_mode == WP_TEXTURE_ADDRESS_WRAP || address_mode == WP_TEXTURE_ADDRESS_MIRROR) {
        const double period
            = (normalized ? 1.0 : double(base_dim)) * (address_mode == WP_TEXTURE_ADDRESS_MIRROR ? 2.0 : 1.0);
        // Keep negative remainders: adding the period can round them up to its
        // endpoint. Integer addressing below handles either sign directly.
        reduced = fmod(reduced, period);
    }
    double texel = normalized ? reduced * double(level_dim) : reduced * double(level_dim) / double(base_dim);
    if (address_mode != WP_TEXTURE_ADDRESS_WRAP && address_mode != WP_TEXTURE_ADDRESS_MIRROR)
        texel = max(-1.0, min(texel, double(level_dim)));
    return cpu_apply_address_mode_index(int64(floor(texel)), level_dim, address_mode);
}

// Check if index is within bounds (for border mode)
inline bool cpu_in_bounds_1d(int64 x, int w) { return x >= 0 && x < w; }

inline bool cpu_in_bounds_2d(int64 x, int64 y, int w, int h) { return x >= 0 && x < w && y >= 0 && y < h; }

inline bool cpu_in_bounds_3d(int64 x, int64 y, int64 z, int w, int h, int d)
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
inline float cpu_decode_texel(const void* level_data, int dtype, size_t idx)
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
inline float cpu_fetch_texel_1d(const Texture* tex, int level, int64 x, int channel)
{
    const int w = tex->mip_widths_arr[level];
    if (!cpu_in_bounds_1d(x, w) || channel < 0 || channel >= tex->num_channels) {
        return 0.0f;
    }
    size_t idx = size_t(x) * tex->num_channels + channel;
    return cpu_decode_texel(tex->mip_data[level], tex->dtype, idx);
}

inline float cpu_fetch_texel_2d(const Texture* tex, int level, int64 x, int64 y, int channel)
{
    const int w = tex->mip_widths_arr[level];
    const int h = tex->mip_heights_arr[level];
    if (!cpu_in_bounds_2d(x, y, w, h) || channel < 0 || channel >= tex->num_channels) {
        return 0.0f;
    }
    size_t idx = (size_t(y) * w + x) * tex->num_channels + channel;
    return cpu_decode_texel(tex->mip_data[level], tex->dtype, idx);
}

inline float cpu_fetch_texel_3d(const Texture* tex, int level, int64 x, int64 y, int64 z, int channel)
{
    const int w = tex->mip_widths_arr[level];
    const int h = tex->mip_heights_arr[level];
    const int d = tex->mip_depths_arr[level];
    if (!cpu_in_bounds_3d(x, y, z, w, h, d) || channel < 0 || channel >= tex->num_channels) {
        return 0.0f;
    }
    size_t idx = ((size_t(z) * h + y) * w + x) * tex->num_channels + channel;
    return cpu_decode_texel(tex->mip_data[level], tex->dtype, idx);
}

// Sample a single channel with linear interpolation (1D) at a specific mip level.
inline float cpu_sample_1d_channel_at_level(const Texture* tex, int level, float u, int channel)
{
    const int w = tex->mip_widths_arr[level];

    if (tex->filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        const int64 x = cpu_nearest_texel_index(u, tex->use_normalized_coords, tex->width, w, tex->address_mode_u);
        return cpu_fetch_texel_1d(tex, level, x, channel);
    } else {
        // Linear interpolation
        float coord_u = cpu_texture_normalized_coord(u, tex->use_normalized_coords, tex->width, tex->address_mode_u);
        float tx = cpu_apply_address_mode_1d(coord_u, w, tex->address_mode_u);
        int64 x0 = (int64)floor(tx);
        int64 x1 = x0 + 1;

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

    if (tex->filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        const int64 x = cpu_nearest_texel_index(u, tex->use_normalized_coords, tex->width, w, tex->address_mode_u);
        const int64 y = cpu_nearest_texel_index(v, tex->use_normalized_coords, tex->height, h, tex->address_mode_v);
        return cpu_fetch_texel_2d(tex, level, x, y, channel);
    } else {
        float coord_u = cpu_texture_normalized_coord(u, tex->use_normalized_coords, tex->width, tex->address_mode_u);
        float coord_v = cpu_texture_normalized_coord(v, tex->use_normalized_coords, tex->height, tex->address_mode_v);
        float tx = cpu_apply_address_mode_1d(coord_u, w, tex->address_mode_u);
        float ty = cpu_apply_address_mode_1d(coord_v, h, tex->address_mode_v);
        int64 x0 = (int64)floor(tx);
        int64 y0 = (int64)floor(ty);
        int64 x1 = x0 + 1;
        int64 y1 = y0 + 1;

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

    if (tex->filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        const int64 x = cpu_nearest_texel_index(u, tex->use_normalized_coords, tex->width, w, tex->address_mode_u);
        const int64 y = cpu_nearest_texel_index(v, tex->use_normalized_coords, tex->height, h, tex->address_mode_v);
        const int64 z
            = cpu_nearest_texel_index(w_coord, tex->use_normalized_coords, tex->depth, d, tex->address_mode_w);
        return cpu_fetch_texel_3d(tex, level, x, y, z, channel);
    } else {
        float coord_u = cpu_texture_normalized_coord(u, tex->use_normalized_coords, tex->width, tex->address_mode_u);
        float coord_v = cpu_texture_normalized_coord(v, tex->use_normalized_coords, tex->height, tex->address_mode_v);
        float coord_w
            = cpu_texture_normalized_coord(w_coord, tex->use_normalized_coords, tex->depth, tex->address_mode_w);
        float tx = cpu_apply_address_mode_1d(coord_u, w, tex->address_mode_u);
        float ty = cpu_apply_address_mode_1d(coord_v, h, tex->address_mode_v);
        float tz = cpu_apply_address_mode_1d(coord_w, d, tex->address_mode_w);
        int64 x0 = (int64)floor(tx);
        int64 y0 = (int64)floor(ty);
        int64 z0 = (int64)floor(tz);
        int64 x1 = x0 + 1;
        int64 y1 = y0 + 1;
        int64 z1 = z0 + 1;

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
    if (level1 == level0 || fl == 0.0f)
        return v0;
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
    if (level1 == level0 || fl == 0.0f)
        return v0;
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
    if (level1 == level0 || fl == 0.0f)
        return v0;
    float v1 = cpu_sample_3d_channel_at_level(tex, level1, u, v, w, channel);
    return v0 * (1.0f - fl) + v1 * fl;
}

#endif  // !defined(__CUDA_ARCH__)

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
// Each ``sample_*`` helper accepts a single ``lod`` argument. When ``lod`` is negative
// (including :c:macro:`WP_TEXTURE_LOD_DISABLED`), the helper dispatches to the original
// non-LOD texture fetch (``tex1D`` / ``tex2D`` / ``tex3D`` on GPU, base mip level on CPU)
// so that callers that never opt into LOD preserve the pre-mipmap behavior exactly.
// Any nonnegative ``lod`` value routes through the mipmap-aware path (``tex1DLod`` / ...).
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

// NVCC and NVRTC versions before CUDA 13.1 can miscompile base-level vector
// texture sampling when functions mix texture return widths on sm_89 and older
// targets. Keep the compiler and architecture checks in this header so both
// NVRTC and external NVCC builds use the same call boundary. Clang CUDA does
// not define the CUDA compiler version macros and retains the inline path.
#if defined(__CUDA_ARCH__)
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__)
constexpr bool use_cuda_texture_mixed_width_workaround = (__CUDA_ARCH__ < 900)
    && ((__CUDACC_VER_MAJOR__ < 13) || ((__CUDACC_VER_MAJOR__ == 13) && (__CUDACC_VER_MINOR__ < 1)));
#else
constexpr bool use_cuda_texture_mixed_width_workaround = false;
#endif
#endif

template <> struct texture_sample_helper<vec2f> {
#if defined(__CUDA_ARCH__)
    static CUDA_CALLABLE_DEVICE __noinline__ float2 sample_base_1d(const texture1d_t& tex, float u)
    {
        return tex1D<float2>(tex.tex, u);
    }

    static CUDA_CALLABLE_DEVICE __noinline__ float2 sample_base_2d(const texture2d_t& tex, float u, float v)
    {
        return tex2D<float2>(tex.tex, u, v);
    }

    static CUDA_CALLABLE_DEVICE __noinline__ float2 sample_base_3d(const texture3d_t& tex, float u, float v, float w)
    {
        return tex3D<float2>(tex.tex, u, v, w);
    }
#endif

    static CUDA_CALLABLE vec2f sample_1d(const texture1d_t& tex, float u, float lod)
    {
#if defined(__CUDA_ARCH__)
        float2 val;
        if constexpr (use_cuda_texture_mixed_width_workaround)
            val = (lod < 0.0f) ? sample_base_1d(tex, u) : tex1DLod<float2>(tex.tex, u, lod);
        else
            val = (lod < 0.0f) ? tex1D<float2>(tex.tex, u) : tex1DLod<float2>(tex.tex, u, lod);
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
        float2 val;
        if constexpr (use_cuda_texture_mixed_width_workaround)
            val = (lod < 0.0f) ? sample_base_2d(tex, u, v) : tex2DLod<float2>(tex.tex, u, v, lod);
        else
            val = (lod < 0.0f) ? tex2D<float2>(tex.tex, u, v) : tex2DLod<float2>(tex.tex, u, v, lod);
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
        float2 val;
        if constexpr (use_cuda_texture_mixed_width_workaround)
            val = (lod < 0.0f) ? sample_base_3d(tex, u, v, w) : tex3DLod<float2>(tex.tex, u, v, w, lod);
        else
            val = (lod < 0.0f) ? tex3D<float2>(tex.tex, u, v, w) : tex3DLod<float2>(tex.tex, u, v, w, lod);
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
#if defined(__CUDA_ARCH__)
    static CUDA_CALLABLE_DEVICE __noinline__ float4 sample_base_1d(const texture1d_t& tex, float u)
    {
        return tex1D<float4>(tex.tex, u);
    }

    static CUDA_CALLABLE_DEVICE __noinline__ float4 sample_base_2d(const texture2d_t& tex, float u, float v)
    {
        return tex2D<float4>(tex.tex, u, v);
    }

    static CUDA_CALLABLE_DEVICE __noinline__ float4 sample_base_3d(const texture3d_t& tex, float u, float v, float w)
    {
        return tex3D<float4>(tex.tex, u, v, w);
    }
#endif

    static CUDA_CALLABLE vec4f sample_1d(const texture1d_t& tex, float u, float lod)
    {
#if defined(__CUDA_ARCH__)
        float4 val;
        if constexpr (use_cuda_texture_mixed_width_workaround)
            val = (lod < 0.0f) ? sample_base_1d(tex, u) : tex1DLod<float4>(tex.tex, u, lod);
        else
            val = (lod < 0.0f) ? tex1D<float4>(tex.tex, u) : tex1DLod<float4>(tex.tex, u, lod);
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
        float4 val;
        if constexpr (use_cuda_texture_mixed_width_workaround)
            val = (lod < 0.0f) ? sample_base_2d(tex, u, v) : tex2DLod<float4>(tex.tex, u, v, lod);
        else
            val = (lod < 0.0f) ? tex2D<float4>(tex.tex, u, v) : tex2DLod<float4>(tex.tex, u, v, lod);
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
        float4 val;
        if constexpr (use_cuda_texture_mixed_width_workaround)
            val = (lod < 0.0f) ? sample_base_3d(tex, u, v, w) : tex3DLod<float4>(tex.tex, u, v, w, lod);
        else
            val = (lod < 0.0f) ? tex3D<float4>(tex.tex, u, v, w) : tex3DLod<float4>(tex.tex, u, v, w, lod);
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

// 32-bit integer formats cannot be sampled. CUDA's texture units promote 8- and 16-bit integer
// formats to normalized floats but leave 32-bit ones alone, so sampling one returns the stored
// word reinterpreted as a float on CUDA and a normalized value on the CPU. Reject these formats
// at the sample site while preserving their use for storage, copies, and interop.
CUDA_CALLABLE inline bool texture_dtype_is_sampleable(int32 dtype)
{
    return dtype != WP_TEXTURE_DTYPE_UINT32 && dtype != WP_TEXTURE_DTYPE_INT32;
}

CUDA_CALLABLE inline void texture_assert_sampleable(int32 dtype)
{
    if (texture_dtype_is_sampleable(dtype)) {
        return;
    }

#if defined(__CUDA_ARCH__)
    printf("texture_sample() does not support 32-bit integer textures\n");
    __trap();
#else
    _wp_assert("texture_sample() does not support 32-bit integer textures", __FILE__, unsigned(__LINE__));
#endif
}

#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
// CUDA 12 can miscompile optimized CUBIN texture sampling on sm_101 and the
// sm_120 family. When active lanes use different handles, a lane may sample
// another lane's texture. The uniform case is unaffected, so keep it inline and
// route only divergent handles through a no-inline call boundary that prevents
// the miscompile.
CUDA_CALLABLE_DEVICE inline bool texture_handle_is_uniform(uint64 handle)
{
    int uniform;
    (void)__match_all_sync(__activemask(), handle, &uniform);
    return uniform != 0;
}

template <typename T>
CUDA_CALLABLE_DEVICE __noinline__ T texture_sample_divergent(const texture2d_t& tex, float u, float v, float lod)
{
    return texture_sample_helper<T>::sample_2d(tex, u, v, lod);
}

template <typename T>
CUDA_CALLABLE_DEVICE __noinline__ T
texture_sample_divergent(const texture3d_t& tex, float u, float v, float w, float lod)
{
    return texture_sample_helper<T>::sample_3d(tex, u, v, w, lod);
}
#endif

// 1D texture sampling with scalar coordinate
template <typename T> CUDA_CALLABLE T texture_sample(const texture1d_t& tex, float u, float lod)
{
    texture_assert_sampleable(tex.get_dtype());
    return texture_sample_helper<T>::sample_1d(tex, u, lod);
}

// 2D texture sampling with vec2 coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture2d_t& tex, const vec2f& uv, float lod)
{
    texture_assert_sampleable(tex.get_dtype());
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex.tex))
        return texture_sample_divergent<T>(tex, uv[0], uv[1], lod);
#endif
    return texture_sample_helper<T>::sample_2d(tex, uv[0], uv[1], lod);
}

// 2D texture sampling with separate u, v coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture2d_t& tex, float u, float v, float lod)
{
    texture_assert_sampleable(tex.get_dtype());
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex.tex))
        return texture_sample_divergent<T>(tex, u, v, lod);
#endif
    return texture_sample_helper<T>::sample_2d(tex, u, v, lod);
}

// 3D texture sampling with vec3 coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture3d_t& tex, const vec3f& uvw, float lod)
{
    texture_assert_sampleable(tex.get_dtype());
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex.tex))
        return texture_sample_divergent<T>(tex, uvw[0], uvw[1], uvw[2], lod);
#endif
    return texture_sample_helper<T>::sample_3d(tex, uvw[0], uvw[1], uvw[2], lod);
}

// 3D texture sampling with separate u, v, w coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture3d_t& tex, float u, float v, float w, float lod)
{
    texture_assert_sampleable(tex.get_dtype());
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex.tex))
        return texture_sample_divergent<T>(tex, u, v, w, lod);
#endif
    return texture_sample_helper<T>::sample_3d(tex, u, v, w, lod);
}

// ============================================================================
// Texture Sampling Adjoints
// ============================================================================
//
// Coordinate gradients interpolate differences of neighboring texels. Subtract
// before interpolation: differencing interpolated samples can lose small slopes
// when the texels share a large offset. CUDA reads exact corners with integer
// coordinates after applying the texture's address modes explicitly.
//
// LOD gradients use sample(level1) - sample(level0), the derivative of the
// linear blend between adjacent mip levels. They are zero when mip filtering is
// closest or the LOD is clamped.
//
// Gradients with respect to the texel data (``adj_tex``) are not implemented.

// Internal adjoint samples validate the dtype once in adj_texture_sample() on
// CUDA. The CPU path retains the public sampler's validation, where skipping it
// does not improve performance.
template <typename T> CUDA_CALLABLE inline T texture_adjoint_sample(const texture2d_t& tex, float u, float v, float lod)
{
#if defined(__CUDA_ARCH__)
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex.tex))
        return texture_sample_divergent<T>(tex, u, v, lod);
#endif
    return texture_sample_helper<T>::sample_2d(tex, u, v, lod);
#else
    return texture_sample<T>(tex, u, v, lod);
#endif
}

template <typename T>
CUDA_CALLABLE inline T texture_adjoint_sample(const texture3d_t& tex, float u, float v, float w, float lod)
{
#if defined(__CUDA_ARCH__)
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex.tex))
        return texture_sample_divergent<T>(tex, u, v, w, lod);
#endif
    return texture_sample_helper<T>::sample_3d(tex, u, v, w, lod);
#else
    return texture_sample<T>(tex, u, v, w, lod);
#endif
}

// Contract each active output channel independently so a zero seed cannot
// turn an overflowing difference in an inactive channel into NaN.
CUDA_CALLABLE inline float texture_grad_dot(float seed, float delta) { return seed == 0.0f ? 0.0f : seed * delta; }

CUDA_CALLABLE inline float texture_grad_dot(const vec2f& seed, const vec2f& delta)
{
    return texture_grad_dot(seed[0], delta[0]) + texture_grad_dot(seed[1], delta[1]);
}

CUDA_CALLABLE inline float texture_grad_dot(const vec4f& seed, const vec4f& delta)
{
    return texture_grad_dot(seed[0], delta[0]) + texture_grad_dot(seed[1], delta[1])
        + texture_grad_dot(seed[2], delta[2]) + texture_grad_dot(seed[3], delta[3]);
}

// Dimension of a mip level (mip chains always use floor-halving)
CUDA_CALLABLE inline int texture_mip_dim(int base_dim, int level)
{
    int d = base_dim >> level;
    return d < 1 ? 1 : d;
}

// Derivative of the raw texel-space coordinate with respect to the sampling
// coordinate: texel = coord * scale - 0.5. Unnormalized coordinates address
// mip levels in base-level texel space, hence the level/base ratio.
CUDA_CALLABLE inline float texture_coord_scale(int use_normalized_coords, int level_dim, int base_dim)
{
    return use_normalized_coords ? (float)level_dim : ((float)level_dim / (float)base_dim);
}

#if defined(__CUDA_ARCH__)
// Reduce periodic coordinates before scaling to texels so large offsets do not
// round away the half-texel shift or the reconstructed corner coordinates.
CUDA_CALLABLE inline float texture_grad_periodic_coord(float coord, int use_normalized_coords, int address_mode)
{
    if (use_normalized_coords) {
        if (address_mode == WP_TEXTURE_ADDRESS_WRAP)
            return coord - floor(coord);
        if (address_mode == WP_TEXTURE_ADDRESS_MIRROR)
            return coord - 2.0f * floor(coord * 0.5f);
    }
    return coord;
}
#endif

// CPU dimensions can approach INT_MAX; a reflected cell and its upper neighbor
// need wider indices even though every addressed texel lies inside the texture.
#if defined(__CUDA_ARCH__)
using texture_grad_index = int;
#else
using texture_grad_index = int64;
#endif

// Select an equivalent bounded interpolation cell before converting to int.
// Clamped axes must still select the correct edge for derivatives in other axes.
CUDA_CALLABLE inline texture_grad_index
texture_grad_cell(float coord, int normalized, int base_dim, int level_dim, int address_mode, float& fraction)
{
#if defined(__CUDA_ARCH__)
    // CUDA treats unnormalized WRAP/MIRROR as CLAMP. Normalized periodic
    // coordinates were already reduced by adj_texture_sample().
    if (!normalized && (address_mode == WP_TEXTURE_ADDRESS_WRAP || address_mode == WP_TEXTURE_ADDRESS_MIRROR))
        address_mode = WP_TEXTURE_ADDRESS_CLAMP;
    float t = coord * texture_coord_scale(normalized, level_dim, base_dim) - 0.5f;
#else
    // Use the same normalization as the CPU forward sampler, including the
    // order of division and multiplication for non-power-of-two dimensions.
    coord = cpu_texture_normalized_coord(coord, normalized, base_dim, address_mode);
    float t = coord * float(level_dim) - 0.5f;
#endif
    if (address_mode == WP_TEXTURE_ADDRESS_CLAMP)
        t = max(-1.0f, min(t, float(level_dim - 1)));
    else if (address_mode == WP_TEXTURE_ADDRESS_BORDER)
        t = max(-2.0f, min(t, float(level_dim)));
    const float cell = floor(t);
    fraction = t - cell;
#if defined(__CUDA_ARCH__)
    // Match the CUDA sampler's eight fractional bits on the other axes.
    fraction = rintf(fraction * 256.0f) / 256.0f;
#endif
    return (texture_grad_index)cell;
}

#if !defined(__CUDA_ARCH__)
// Read CPU corners directly so coordinate round trips cannot interpolate them.
template <typename T> inline T cpu_texture_grad_texel(const Texture* tex, int level, int64 x, int64 y = 0, int64 z = 0)
{
    x = cpu_apply_address_mode_index(x, tex->mip_widths_arr[level], tex->address_mode_u);
    y = cpu_apply_address_mode_index(y, tex->mip_heights_arr[level], tex->address_mode_v);
    z = cpu_apply_address_mode_index(z, tex->mip_depths_arr[level], tex->address_mode_w);
    if constexpr (sizeof(T) == sizeof(float)) {
        return cpu_fetch_texel_3d(tex, level, x, y, z, 0);
    } else {
        T value;
        for (int channel = 0; channel < sizeof(T) / sizeof(float); ++channel)
            value[channel] = cpu_fetch_texel_3d(tex, level, x, y, z, channel);
        return value;
    }
}

#endif  // !defined(__CUDA_ARCH__)

#if defined(__CUDA_ARCH__)
// Integer coordinates bypass texture filtering. CUDA arrays, including 1D
// arrays, use the 2D array-fetch instruction; the 1D integer instruction is for
// linear-memory textures. Always fetch four channels to avoid mixed-width
// texture instruction compiler bugs, then select the requested components.
CUDA_CALLABLE_DEVICE inline vec4f cuda_texture_grad_fetch(uint64 tex, int level, int x, int y)
{
    vec4f value;
    asm("tex.level.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
        : "=f"(value[0]), "=f"(value[1]), "=f"(value[2]), "=f"(value[3])
        : "l"(tex), "r"(x), "r"(y), "r"(level));
    return value;
}

CUDA_CALLABLE_DEVICE inline vec4f cuda_texture_grad_fetch(uint64 tex, int level, int x, int y, int z)
{
    vec4f value;
    asm("tex.level.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, 0}], %8;"
        : "=f"(value[0]), "=f"(value[1]), "=f"(value[2]), "=f"(value[3])
        : "l"(tex), "r"(x), "r"(y), "r"(z), "r"(level));
    return value;
}

#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
CUDA_CALLABLE_DEVICE __noinline__ vec4f cuda_texture_grad_fetch_divergent(uint64 tex, int level, int x, int y)
{
    return cuda_texture_grad_fetch(tex, level, x, y);
}

CUDA_CALLABLE_DEVICE __noinline__ vec4f cuda_texture_grad_fetch_divergent(uint64 tex, int level, int x, int y, int z)
{
    return cuda_texture_grad_fetch(tex, level, x, y, z);
}
#endif

CUDA_CALLABLE_DEVICE inline vec4f cuda_texture_grad_fetch_protected(uint64 tex, int level, int x, int y)
{
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex))
        return cuda_texture_grad_fetch_divergent(tex, level, x, y);
#endif
    return cuda_texture_grad_fetch(tex, level, x, y);
}

CUDA_CALLABLE_DEVICE inline vec4f cuda_texture_grad_fetch_protected(uint64 tex, int level, int x, int y, int z)
{
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex))
        return cuda_texture_grad_fetch_divergent(tex, level, x, y, z);
#endif
    return cuda_texture_grad_fetch(tex, level, x, y, z);
}

// Integer fetches do not apply periodic or border addressing. CUDA texture
// dimensions are hardware-bounded, so their indices and periods fit in int32.
CUDA_CALLABLE_DEVICE inline int cuda_texture_grad_index(int index, int size, int mode, int normalized)
{
    if (!normalized && (mode == WP_TEXTURE_ADDRESS_WRAP || mode == WP_TEXTURE_ADDRESS_MIRROR))
        mode = WP_TEXTURE_ADDRESS_CLAMP;
    if (mode == WP_TEXTURE_ADDRESS_WRAP) {
        const int m = index % size;
        return m < 0 ? m + size : m;
    }
    if (mode == WP_TEXTURE_ADDRESS_MIRROR) {
        const int period = 2 * size;
        int m = index % period;
        if (m < 0)
            m += period;
        return m < size ? m : period - m - 1;
    }
    if (mode == WP_TEXTURE_ADDRESS_CLAMP)
        return max(0, min(index, size - 1));
    return index;
}

template <typename T> CUDA_CALLABLE_DEVICE inline T texture_grad_components(const vec4f& value)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        return value[0];
    } else {
        T result;
        for (int channel = 0; channel < sizeof(T) / sizeof(float); ++channel)
            result[channel] = value[channel];
        return result;
    }
}
#endif

template <typename T> CUDA_CALLABLE inline T texture_grad_texel(const texture1d_t& tex, int level, texture_grad_index x)
{
#if defined(__CUDA_ARCH__)
    const int width = texture_mip_dim(tex.width, level);
    x = cuda_texture_grad_index(x, width, tex.get_address_mode_u(), tex.get_use_normalized_coords());
    if (x < 0 || x >= width)
        return texture_sample_helper<T>::zero();
    return texture_grad_components<T>(cuda_texture_grad_fetch_protected(tex.tex, level, x, 0));
#else
    return cpu_texture_grad_texel<T>((const Texture*)tex.tex, level, x);
#endif
}

template <typename T>
CUDA_CALLABLE inline T texture_grad_texel(const texture2d_t& tex, int level, texture_grad_index x, texture_grad_index y)
{
#if defined(__CUDA_ARCH__)
    const int width = texture_mip_dim(tex.width, level);
    const int height = texture_mip_dim(tex.height, level);
    x = cuda_texture_grad_index(x, width, tex.get_address_mode_u(), tex.get_use_normalized_coords());
    y = cuda_texture_grad_index(y, height, tex.get_address_mode_v(), tex.get_use_normalized_coords());
    if (x < 0 || x >= width || y < 0 || y >= height)
        return texture_sample_helper<T>::zero();
    return texture_grad_components<T>(cuda_texture_grad_fetch_protected(tex.tex, level, x, y));
#else
    return cpu_texture_grad_texel<T>((const Texture*)tex.tex, level, x, y);
#endif
}

template <typename T>
CUDA_CALLABLE inline T
texture_grad_texel(const texture3d_t& tex, int level, texture_grad_index x, texture_grad_index y, texture_grad_index z)
{
#if defined(__CUDA_ARCH__)
    const int width = texture_mip_dim(tex.width, level);
    const int height = texture_mip_dim(tex.height, level);
    const int depth = texture_mip_dim(tex.depth, level);
    x = cuda_texture_grad_index(x, width, tex.get_address_mode_u(), tex.get_use_normalized_coords());
    y = cuda_texture_grad_index(y, height, tex.get_address_mode_v(), tex.get_use_normalized_coords());
    z = cuda_texture_grad_index(z, depth, tex.get_address_mode_w(), tex.get_use_normalized_coords());
    if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth)
        return texture_sample_helper<T>::zero();
    return texture_grad_components<T>(cuda_texture_grad_fetch_protected(tex.tex, level, x, y, z));
#else
    return cpu_texture_grad_texel<T>((const Texture*)tex.tex, level, x, y, z);
#endif
}

// Keep the corner reference separate from the interpolated differences so a
// large shared texel offset cannot erase the difference between mip samples.
template <typename T> struct texture_grad_sample {
    T anchor;
    T residual;
};

// Opposite-sign finite texels can overflow centered differences even when the
// interpolated samples and their LOD difference are finite.
template <typename T> CUDA_CALLABLE inline bool texture_grad_lod_isfinite(const T& value)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        return isfinite(value);
    } else {
        for (int channel = 0; channel < sizeof(T) / sizeof(float); ++channel)
            if (!isfinite(value[channel]))
                return false;
        return true;
    }
}

// Preserve stable differences in the other channels, including small slopes
// atop large offsets. Forward sampling also preserves genuine NaN inputs.
template <typename T> CUDA_CALLABLE inline T texture_grad_lod_fallback(T value, const T& sampled_difference)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        return isfinite(value) ? value : sampled_difference;
    } else {
        for (int channel = 0; channel < sizeof(T) / sizeof(float); ++channel)
            if (!isfinite(value[channel]))
                value[channel] = sampled_difference[channel];
        return value;
    }
}

// The returned sample reuses the fetched corners when an LOD gradient is also
// needed. Ignored results let the compiler remove the residual interpolation.
template <typename T>
CUDA_CALLABLE inline texture_grad_sample<T>
texture_grad_coords_1d(const texture1d_t& tex, float u, int level, float weight, const T& adj_ret, float& adj_u)
{
    const int level_w = texture_mip_dim(tex.width, level);
    const float su = texture_coord_scale(tex.get_use_normalized_coords(), level_w, tex.width);

    float fu;
    const texture_grad_index x
        = texture_grad_cell(u, tex.get_use_normalized_coords(), tex.width, level_w, tex.get_address_mode_u(), fu);
    const T v0 = texture_grad_texel<T>(tex, level, x);
    const T v1 = texture_grad_texel<T>(tex, level, x + 1);
    // At integer LODs the next level is sampled only for the LOD derivative.
    if (weight != 0.0f)
        adj_u += weight * su * texture_grad_dot(adj_ret, v1 - v0);
    return { v0, (v1 - v0) * fu };
}

template <typename T>
CUDA_CALLABLE inline texture_grad_sample<T> texture_grad_coords_2d(
    const texture2d_t& tex, float u, float v, int level, float weight, const T& adj_ret, float& adj_u, float& adj_v
)
{
    const int level_w = texture_mip_dim(tex.width, level);
    const int level_h = texture_mip_dim(tex.height, level);
    const float su = texture_coord_scale(tex.get_use_normalized_coords(), level_w, tex.width);
    const float sv = texture_coord_scale(tex.get_use_normalized_coords(), level_h, tex.height);

    float fu, fv;
    const texture_grad_index x
        = texture_grad_cell(u, tex.get_use_normalized_coords(), tex.width, level_w, tex.get_address_mode_u(), fu);
    const texture_grad_index y
        = texture_grad_cell(v, tex.get_use_normalized_coords(), tex.height, level_h, tex.get_address_mode_v(), fv);
    const T v00 = texture_grad_texel<T>(tex, level, x, y);
    const T v10 = texture_grad_texel<T>(tex, level, x + 1, y);
    const T v01 = texture_grad_texel<T>(tex, level, x, y + 1);
    const T v11 = texture_grad_texel<T>(tex, level, x + 1, y + 1);

    if (weight != 0.0f) {
        adj_u += weight * su * lerp(texture_grad_dot(adj_ret, v10 - v00), texture_grad_dot(adj_ret, v11 - v01), fv);
        adj_v += weight * sv * lerp(texture_grad_dot(adj_ret, v01 - v00), texture_grad_dot(adj_ret, v11 - v10), fu);
    }
    return { v00, lerp((v10 - v00) * fu, lerp(v01 - v00, v11 - v00, fu), fv) };
}

template <typename T>
CUDA_CALLABLE inline texture_grad_sample<T> texture_grad_coords_3d(
    const texture3d_t& tex,
    float u,
    float v,
    float w,
    int level,
    float weight,
    const T& adj_ret,
    float& adj_u,
    float& adj_v,
    float& adj_w
)
{
    const int level_w = texture_mip_dim(tex.width, level);
    const int level_h = texture_mip_dim(tex.height, level);
    const int level_d = texture_mip_dim(tex.depth, level);
    const float su = texture_coord_scale(tex.get_use_normalized_coords(), level_w, tex.width);
    const float sv = texture_coord_scale(tex.get_use_normalized_coords(), level_h, tex.height);
    const float sw = texture_coord_scale(tex.get_use_normalized_coords(), level_d, tex.depth);

    float fu, fv, fw;
    const texture_grad_index x
        = texture_grad_cell(u, tex.get_use_normalized_coords(), tex.width, level_w, tex.get_address_mode_u(), fu);
    const texture_grad_index y
        = texture_grad_cell(v, tex.get_use_normalized_coords(), tex.height, level_h, tex.get_address_mode_v(), fv);
    const texture_grad_index z
        = texture_grad_cell(w, tex.get_use_normalized_coords(), tex.depth, level_d, tex.get_address_mode_w(), fw);
    const T v000 = texture_grad_texel<T>(tex, level, x, y, z);
    const T v100 = texture_grad_texel<T>(tex, level, x + 1, y, z);
    const T v010 = texture_grad_texel<T>(tex, level, x, y + 1, z);
    const T v110 = texture_grad_texel<T>(tex, level, x + 1, y + 1, z);
    const T v001 = texture_grad_texel<T>(tex, level, x, y, z + 1);
    const T v101 = texture_grad_texel<T>(tex, level, x + 1, y, z + 1);
    const T v011 = texture_grad_texel<T>(tex, level, x, y + 1, z + 1);
    const T v111 = texture_grad_texel<T>(tex, level, x + 1, y + 1, z + 1);

    if (weight != 0.0f) {
        adj_u += weight * su
            * lerp(lerp(texture_grad_dot(adj_ret, v100 - v000), texture_grad_dot(adj_ret, v110 - v010), fv),
                   lerp(texture_grad_dot(adj_ret, v101 - v001), texture_grad_dot(adj_ret, v111 - v011), fv), fw);
        adj_v += weight * sv
            * lerp(lerp(texture_grad_dot(adj_ret, v010 - v000), texture_grad_dot(adj_ret, v110 - v100), fu),
                   lerp(texture_grad_dot(adj_ret, v011 - v001), texture_grad_dot(adj_ret, v111 - v101), fu), fw);
        adj_w += weight * sw
            * lerp(lerp(texture_grad_dot(adj_ret, v001 - v000), texture_grad_dot(adj_ret, v101 - v100), fu),
                   lerp(texture_grad_dot(adj_ret, v011 - v010), texture_grad_dot(adj_ret, v111 - v110), fu), fv);
    }
    const T lower = lerp((v100 - v000) * fu, lerp(v010 - v000, v110 - v000, fu), fv);
    const T upper = lerp(lerp(v001 - v000, v101 - v000, fu), lerp(v011 - v000, v111 - v000, fu), fv);
    return { v000, lerp(lower, upper, fw) };
}

// Resolve the mip levels and blend weights the forward pass reads for a given
// LOD, mirroring cpu_clamp_lod and the mip filter selection. Returns the
// number of levels contributing spatial gradients (1 or 2).
CUDA_CALLABLE inline int
texture_grad_mip_levels(int num_mip_levels, int mip_filter_mode, float lod, int* levels, float* weights)
{
    const float max_lod = (float)(num_mip_levels - 1);
    float clamped = lod < 0.0f ? 0.0f : (lod > max_lod ? max_lod : lod);

    if (mip_filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        int nearest = (int)floor(clamped + 0.5f);
        if (nearest >= num_mip_levels)
            nearest = num_mip_levels - 1;
        levels[0] = nearest;
        weights[0] = 1.0f;
        return 1;
    }

    const int level0 = (int)floor(clamped);
    int level1 = level0 + 1;
    if (level1 >= num_mip_levels)
        level1 = num_mip_levels - 1;
    const float fl = clamped - (float)level0;

    levels[0] = level0;
    weights[0] = 1.0f - fl;
    if (level1 == level0 || fl == 0.0f)
        return 1;
    levels[1] = level1;
    weights[1] = fl;
    return 2;
}

// LOD gradient: the mip-linear blend is piecewise linear in the clamped LOD,
// with derivative sample(level1) - sample(level0) between levels and zero in
// the clamped ranges or with a CLOSEST mip filter.
CUDA_CALLABLE inline bool
texture_grad_lod_levels(int num_mip_levels, int mip_filter_mode, float lod, int& level0, int& level1)
{
    if (num_mip_levels <= 1 || mip_filter_mode != WP_TEXTURE_FILTER_LINEAR)
        return false;
    if (lod < 0.0f || lod >= (float)(num_mip_levels - 1))
        return false;
    level0 = (int)floor(lod);
    level1 = level0 + 1;
    return true;
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture1d_t& tex, float u, float lod, texture1d_t& adj_tex, float& adj_u, float& adj_lod, const T& adj_ret
)
{
    if (tex.tex == 0)
        return;

#if defined(__CUDA_ARCH__)
    texture_assert_sampleable(tex.get_dtype());
    u = texture_grad_periodic_coord(u, tex.get_use_normalized_coords(), tex.get_address_mode_u());
#endif

    const bool mipmapped = tex.get_num_mip_levels() > 1 && lod >= 0.0f;
    int level0, level1;
    const bool grad_lod = mipmapped
        && texture_grad_lod_levels(tex.get_num_mip_levels(), tex.get_mip_filter_mode(), lod, level0, level1);

    if (tex.get_filter_mode() == WP_TEXTURE_FILTER_LINEAR) {
        if (grad_lod) {
            const float fl = lod - (float)level0;
            const texture_grad_sample<T> sample0 = texture_grad_coords_1d(tex, u, level0, 1.0f - fl, adj_ret, adj_u);
            const texture_grad_sample<T> sample1 = texture_grad_coords_1d(tex, u, level1, fl, adj_ret, adj_u);
            T difference = (sample1.anchor - sample0.anchor) + (sample1.residual - sample0.residual);
            if (!texture_grad_lod_isfinite(difference)) {
                const T v0 = texture_sample_helper<T>::sample_1d(tex, u, (float)level0);
                const T v1 = texture_sample_helper<T>::sample_1d(tex, u, (float)level1);
                difference = texture_grad_lod_fallback(difference, v1 - v0);
            }
            adj_lod += texture_grad_dot(adj_ret, difference);
        } else if (!mipmapped) {
            texture_grad_coords_1d(tex, u, 0, 1.0f, adj_ret, adj_u);
        } else {
            int levels[2];
            float weights[2];
            const int n
                = texture_grad_mip_levels(tex.get_num_mip_levels(), tex.get_mip_filter_mode(), lod, levels, weights);
            for (int i = 0; i < n; ++i)
                texture_grad_coords_1d(tex, u, levels[i], weights[i], adj_ret, adj_u);
        }
    } else if (grad_lod) {
        // CLOSEST spatial filtering needs only the two selected texels.
        const T v0 = texture_sample_helper<T>::sample_1d(tex, u, (float)level0);
        const T v1 = texture_sample_helper<T>::sample_1d(tex, u, (float)level1);
        adj_lod += texture_grad_dot(adj_ret, v1 - v0);
    }
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
    if (tex.tex == 0)
        return;

#if defined(__CUDA_ARCH__)
    texture_assert_sampleable(tex.get_dtype());
    u = texture_grad_periodic_coord(u, tex.get_use_normalized_coords(), tex.get_address_mode_u());
    v = texture_grad_periodic_coord(v, tex.get_use_normalized_coords(), tex.get_address_mode_v());
#endif

    const bool mipmapped = tex.get_num_mip_levels() > 1 && lod >= 0.0f;
    int level0, level1;
    const bool grad_lod = mipmapped
        && texture_grad_lod_levels(tex.get_num_mip_levels(), tex.get_mip_filter_mode(), lod, level0, level1);

    if (tex.get_filter_mode() == WP_TEXTURE_FILTER_LINEAR) {
        if (grad_lod) {
            const float fl = lod - (float)level0;
            const texture_grad_sample<T> sample0
                = texture_grad_coords_2d(tex, u, v, level0, 1.0f - fl, adj_ret, adj_u, adj_v);
            const texture_grad_sample<T> sample1 = texture_grad_coords_2d(tex, u, v, level1, fl, adj_ret, adj_u, adj_v);
            T difference = (sample1.anchor - sample0.anchor) + (sample1.residual - sample0.residual);
            if (!texture_grad_lod_isfinite(difference)) {
                const T v0 = texture_adjoint_sample<T>(tex, u, v, (float)level0);
                const T v1 = texture_adjoint_sample<T>(tex, u, v, (float)level1);
                difference = texture_grad_lod_fallback(difference, v1 - v0);
            }
            adj_lod += texture_grad_dot(adj_ret, difference);
        } else if (!mipmapped) {
            texture_grad_coords_2d(tex, u, v, 0, 1.0f, adj_ret, adj_u, adj_v);
        } else {
            int levels[2];
            float weights[2];
            const int n
                = texture_grad_mip_levels(tex.get_num_mip_levels(), tex.get_mip_filter_mode(), lod, levels, weights);
            for (int i = 0; i < n; ++i)
                texture_grad_coords_2d(tex, u, v, levels[i], weights[i], adj_ret, adj_u, adj_v);
        }
    } else if (grad_lod) {
        // CLOSEST spatial filtering needs only the two selected texels.
        const T v0 = texture_adjoint_sample<T>(tex, u, v, (float)level0);
        const T v1 = texture_adjoint_sample<T>(tex, u, v, (float)level1);
        adj_lod += texture_grad_dot(adj_ret, v1 - v0);
    }
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
    adj_texture_sample(tex, uv[0], uv[1], lod, adj_tex, adj_uv[0], adj_uv[1], adj_lod, adj_ret);
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
    if (tex.tex == 0)
        return;

#if defined(__CUDA_ARCH__)
    texture_assert_sampleable(tex.get_dtype());
    u = texture_grad_periodic_coord(u, tex.get_use_normalized_coords(), tex.get_address_mode_u());
    v = texture_grad_periodic_coord(v, tex.get_use_normalized_coords(), tex.get_address_mode_v());
    w = texture_grad_periodic_coord(w, tex.get_use_normalized_coords(), tex.get_address_mode_w());
#endif

    const bool mipmapped = tex.get_num_mip_levels() > 1 && lod >= 0.0f;
    int level0, level1;
    const bool grad_lod = mipmapped
        && texture_grad_lod_levels(tex.get_num_mip_levels(), tex.get_mip_filter_mode(), lod, level0, level1);

    if (tex.get_filter_mode() == WP_TEXTURE_FILTER_LINEAR) {
        if (grad_lod) {
            const float fl = lod - (float)level0;
            const texture_grad_sample<T> sample0
                = texture_grad_coords_3d(tex, u, v, w, level0, 1.0f - fl, adj_ret, adj_u, adj_v, adj_w);
            const texture_grad_sample<T> sample1
                = texture_grad_coords_3d(tex, u, v, w, level1, fl, adj_ret, adj_u, adj_v, adj_w);
            T difference = (sample1.anchor - sample0.anchor) + (sample1.residual - sample0.residual);
            if (!texture_grad_lod_isfinite(difference)) {
                const T v0 = texture_adjoint_sample<T>(tex, u, v, w, (float)level0);
                const T v1 = texture_adjoint_sample<T>(tex, u, v, w, (float)level1);
                difference = texture_grad_lod_fallback(difference, v1 - v0);
            }
            adj_lod += texture_grad_dot(adj_ret, difference);
        } else if (!mipmapped) {
            texture_grad_coords_3d(tex, u, v, w, 0, 1.0f, adj_ret, adj_u, adj_v, adj_w);
        } else {
            int levels[2];
            float weights[2];
            const int n
                = texture_grad_mip_levels(tex.get_num_mip_levels(), tex.get_mip_filter_mode(), lod, levels, weights);
            for (int i = 0; i < n; ++i)
                texture_grad_coords_3d(tex, u, v, w, levels[i], weights[i], adj_ret, adj_u, adj_v, adj_w);
        }
    } else if (grad_lod) {
        // CLOSEST spatial filtering needs only the two selected texels.
        const T v0 = texture_adjoint_sample<T>(tex, u, v, w, (float)level0);
        const T v1 = texture_adjoint_sample<T>(tex, u, v, w, (float)level1);
        adj_lod += texture_grad_dot(adj_ret, v1 - v0);
    }
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
    adj_texture_sample(tex, uvw[0], uvw[1], uvw[2], lod, adj_tex, adj_uvw[0], adj_uvw[1], adj_uvw[2], adj_lod, adj_ret);
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
