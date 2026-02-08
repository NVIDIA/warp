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

#include "warp.h"

#include "cuda_util.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// CUDA texture API implementation for Warp
// This file implements texture creation and destruction for 2D and 3D CUDA textures

// Data type constants (must match Python side)
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

// ============================================================================
// CPU Host Texture Data Structures
// ============================================================================

// CPU texture descriptor - stores texture data and sampling parameters
// This is allocated and pointed to by the tex handle (as uint64_t pointer)
struct cpu_texture2d_data {
    void* data;  // Pointer to the texture data (owned, allocated with malloc)
    int width;
    int height;
    int num_channels;
    int dtype;  // WP_TEXTURE_DTYPE_*
    int filter_mode;  // WP_TEXTURE_FILTER_*
    int address_mode_u;  // WP_TEXTURE_ADDRESS_* for U axis
    int address_mode_v;  // WP_TEXTURE_ADDRESS_* for V axis
    bool use_normalized_coords;  // If true, coords in [0,1]; if false, in texel space
    int num_mip_levels;  // 1 = no mipmaps
    int mip_filter_mode;  // 0=closest, 1=linear
    void** mip_data;
    int* mip_widths;
    int* mip_heights;
};

struct cpu_texture3d_data {
    void* data;  // Pointer to the texture data (owned, allocated with malloc)
    int width;
    int height;
    int depth;
    int num_channels;
    int dtype;  // WP_TEXTURE_DTYPE_*
    int filter_mode;  // WP_TEXTURE_FILTER_*
    int address_mode_u;  // WP_TEXTURE_ADDRESS_* for U axis
    int address_mode_v;  // WP_TEXTURE_ADDRESS_* for V axis
    int address_mode_w;  // WP_TEXTURE_ADDRESS_* for W axis
    bool use_normalized_coords;  // If true, coords in [0,1]; if false, in texel space
    int num_mip_levels;  // 1 = no mipmaps
    int mip_filter_mode;  // 0=closest, 1=linear
    void** mip_data;
    int* mip_widths;
    int* mip_heights;
    int* mip_depths;
};

// Helper function to get bytes per channel from dtype
static int get_bytes_per_channel(int dtype)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return 1;
    case WP_TEXTURE_DTYPE_UINT16:
        return 2;
    case WP_TEXTURE_DTYPE_FLOAT32:
    default:
        return 4;
    }
}

// ============================================================================
// CPU Host Texture Implementation
// ============================================================================

bool wp_texture2d_create_host(
    int width,
    int height,
    int num_channels,
    int dtype,
    int filter_mode,
    int mip_filter_mode,
    int address_mode_u,
    int address_mode_v,
    bool use_normalized_coords,
    int num_mip_levels,
    const void* data,
    const int* mip_widths,
    const int* mip_heights,
    uint64_t* tex_handle_out
)
{
    if (width <= 0 || height <= 0 || data == nullptr) {
        return false;
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        fprintf(
            stderr,
            "Warp texture error: Only 1, 2, or 4 channel textures are supported (got %d channels). "
            "3-channel textures are not supported.\n",
            num_channels
        );
        return false;
    }

    // Allocate the CPU texture descriptor
    cpu_texture2d_data* tex_data = (cpu_texture2d_data*)malloc(sizeof(cpu_texture2d_data));
    if (tex_data == nullptr) {
        return false;
    }

    int bytes_per_channel = get_bytes_per_channel(dtype);

    if (num_mip_levels > 1 && mip_widths != nullptr && mip_heights != nullptr) {
        // Mipmapped path: compute total data size across all levels
        size_t total_size = 0;
        for (int i = 0; i < num_mip_levels; ++i) {
            total_size += (size_t)mip_widths[i] * mip_heights[i] * num_channels * bytes_per_channel;
        }

        void* all_data = malloc(total_size);
        if (all_data == nullptr) {
            free(tex_data);
            return false;
        }
        memcpy(all_data, data, total_size);

        // Allocate mipmap arrays
        tex_data->mip_data = (void**)malloc(sizeof(void*) * num_mip_levels);
        tex_data->mip_widths = (int*)malloc(sizeof(int) * num_mip_levels);
        tex_data->mip_heights = (int*)malloc(sizeof(int) * num_mip_levels);

        // Set up per-level pointers into the contiguous buffer
        char* ptr = (char*)all_data;
        for (int i = 0; i < num_mip_levels; ++i) {
            tex_data->mip_data[i] = ptr;
            tex_data->mip_widths[i] = mip_widths[i];
            tex_data->mip_heights[i] = mip_heights[i];
            ptr += (size_t)mip_widths[i] * mip_heights[i] * num_channels * bytes_per_channel;
        }

        tex_data->data = all_data;
        tex_data->num_mip_levels = num_mip_levels;
        tex_data->mip_filter_mode = mip_filter_mode;
    } else {
        // Non-mipmapped path
        size_t data_size = (size_t)width * height * num_channels * bytes_per_channel;

        tex_data->data = malloc(data_size);
        if (tex_data->data == nullptr) {
            free(tex_data);
            return false;
        }
        memcpy(tex_data->data, data, data_size);

        tex_data->num_mip_levels = 1;
        tex_data->mip_filter_mode = 0;
        tex_data->mip_data = nullptr;
        tex_data->mip_widths = nullptr;
        tex_data->mip_heights = nullptr;
    }

    // Store metadata
    tex_data->width = width;
    tex_data->height = height;
    tex_data->num_channels = num_channels;
    tex_data->dtype = dtype;
    tex_data->filter_mode = filter_mode;
    tex_data->address_mode_u = address_mode_u;
    tex_data->address_mode_v = address_mode_v;
    tex_data->use_normalized_coords = use_normalized_coords;

    // Return the pointer as uint64_t handle
    *tex_handle_out = (uint64_t)tex_data;

    return true;
}

void wp_texture2d_destroy_host(uint64_t tex_handle)
{
    if (tex_handle == 0)
        return;

    cpu_texture2d_data* tex_data = (cpu_texture2d_data*)tex_handle;
    if (tex_data->data != nullptr)
        free(tex_data->data);
    if (tex_data->mip_data != nullptr)
        free(tex_data->mip_data);
    if (tex_data->mip_widths != nullptr)
        free(tex_data->mip_widths);
    if (tex_data->mip_heights != nullptr)
        free(tex_data->mip_heights);
    free(tex_data);
}

bool wp_texture3d_create_host(
    int width,
    int height,
    int depth,
    int num_channels,
    int dtype,
    int filter_mode,
    int mip_filter_mode,
    int address_mode_u,
    int address_mode_v,
    int address_mode_w,
    bool use_normalized_coords,
    int num_mip_levels,
    const void* data,
    const int* mip_widths,
    const int* mip_heights,
    const int* mip_depths,
    uint64_t* tex_handle_out
)
{
    if (width <= 0 || height <= 0 || depth <= 0 || data == nullptr) {
        return false;
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        fprintf(
            stderr,
            "Warp texture error: Only 1, 2, or 4 channel textures are supported (got %d channels). "
            "3-channel textures are not supported.\n",
            num_channels
        );
        return false;
    }

    // Allocate the CPU texture descriptor
    cpu_texture3d_data* tex_data = (cpu_texture3d_data*)malloc(sizeof(cpu_texture3d_data));
    if (tex_data == nullptr) {
        return false;
    }

    int bytes_per_channel = get_bytes_per_channel(dtype);

    if (num_mip_levels > 1 && mip_widths != nullptr && mip_heights != nullptr && mip_depths != nullptr) {
        // Mipmapped path
        size_t total_size = 0;
        for (int i = 0; i < num_mip_levels; ++i) {
            total_size += (size_t)mip_widths[i] * mip_heights[i] * mip_depths[i] * num_channels * bytes_per_channel;
        }

        void* all_data = malloc(total_size);
        if (all_data == nullptr) {
            free(tex_data);
            return false;
        }
        memcpy(all_data, data, total_size);

        tex_data->mip_data = (void**)malloc(sizeof(void*) * num_mip_levels);
        tex_data->mip_widths = (int*)malloc(sizeof(int) * num_mip_levels);
        tex_data->mip_heights = (int*)malloc(sizeof(int) * num_mip_levels);
        tex_data->mip_depths = (int*)malloc(sizeof(int) * num_mip_levels);

        char* ptr = (char*)all_data;
        for (int i = 0; i < num_mip_levels; ++i) {
            tex_data->mip_data[i] = ptr;
            tex_data->mip_widths[i] = mip_widths[i];
            tex_data->mip_heights[i] = mip_heights[i];
            tex_data->mip_depths[i] = mip_depths[i];
            ptr += (size_t)mip_widths[i] * mip_heights[i] * mip_depths[i] * num_channels * bytes_per_channel;
        }

        tex_data->data = all_data;
        tex_data->num_mip_levels = num_mip_levels;
        tex_data->mip_filter_mode = mip_filter_mode;
    } else {
        // Non-mipmapped path
        size_t data_size = (size_t)width * height * depth * num_channels * bytes_per_channel;

        tex_data->data = malloc(data_size);
        if (tex_data->data == nullptr) {
            free(tex_data);
            return false;
        }
        memcpy(tex_data->data, data, data_size);

        tex_data->num_mip_levels = 1;
        tex_data->mip_filter_mode = 0;
        tex_data->mip_data = nullptr;
        tex_data->mip_widths = nullptr;
        tex_data->mip_heights = nullptr;
        tex_data->mip_depths = nullptr;
    }

    // Store metadata
    tex_data->width = width;
    tex_data->height = height;
    tex_data->depth = depth;
    tex_data->num_channels = num_channels;
    tex_data->dtype = dtype;
    tex_data->filter_mode = filter_mode;
    tex_data->address_mode_u = address_mode_u;
    tex_data->address_mode_v = address_mode_v;
    tex_data->address_mode_w = address_mode_w;
    tex_data->use_normalized_coords = use_normalized_coords;

    // Return the pointer as uint64_t handle
    *tex_handle_out = (uint64_t)tex_data;

    return true;
}

void wp_texture3d_destroy_host(uint64_t tex_handle)
{
    if (tex_handle == 0)
        return;

    cpu_texture3d_data* tex_data = (cpu_texture3d_data*)tex_handle;
    if (tex_data->data != nullptr)
        free(tex_data->data);
    if (tex_data->mip_data != nullptr)
        free(tex_data->mip_data);
    if (tex_data->mip_widths != nullptr)
        free(tex_data->mip_widths);
    if (tex_data->mip_heights != nullptr)
        free(tex_data->mip_heights);
    if (tex_data->mip_depths != nullptr)
        free(tex_data->mip_depths);
    free(tex_data);
}

// ============================================================================
// CUDA Device Texture Implementation
// ============================================================================

#if WP_ENABLE_CUDA

// Helper function to get CUDA array format from dtype
static CUarray_format get_cuda_format(int dtype)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return CU_AD_FORMAT_UNSIGNED_INT8;
    case WP_TEXTURE_DTYPE_UINT16:
        return CU_AD_FORMAT_UNSIGNED_INT16;
    case WP_TEXTURE_DTYPE_FLOAT32:
    default:
        return CU_AD_FORMAT_FLOAT;
    }
}

// Helper function to convert address mode int to CUDA address mode
static CUaddress_mode get_cuda_address_mode(int address_mode)
{
    switch (address_mode) {
    case 0:
        return CU_TR_ADDRESS_MODE_WRAP;
    case 1:
        return CU_TR_ADDRESS_MODE_CLAMP;
    case 2:
        return CU_TR_ADDRESS_MODE_MIRROR;
    case 3:
        return CU_TR_ADDRESS_MODE_BORDER;
    default:
        return CU_TR_ADDRESS_MODE_CLAMP;
    }
}

bool wp_texture2d_create_device(
    void* context,
    int width,
    int height,
    int num_channels,
    int dtype,
    int filter_mode,
    int mip_filter_mode,
    int address_mode_u,
    int address_mode_v,
    bool use_normalized_coords,
    int num_mip_levels,
    const void* data,
    const int* mip_widths,
    const int* mip_heights,
    uint64_t* tex_handle_out,
    uint64_t* array_handle_out,
    uint64_t* mipmap_handle_out
)
{
    if (width <= 0 || height <= 0 || data == nullptr) {
        return false;
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        fprintf(
            stderr,
            "Warp texture error: Only 1, 2, or 4 channel textures are supported (got %d channels). "
            "3-channel textures are not supported due to CUDA hardware limitations.\n",
            num_channels
        );
        return false;
    }

    ContextGuard guard(context);

    // Determine the CUDA array format
    CUarray_format format = get_cuda_format(dtype);
    int bytes_per_channel = get_bytes_per_channel(dtype);
    int bytes_per_texel = num_channels * bytes_per_channel;

    bool use_mip = (num_mip_levels > 1 && mip_widths != nullptr && mip_heights != nullptr);

    CUDA_RESOURCE_DESC res_desc = {};
    res_desc.flags = 0;

    CUarray cuda_array = nullptr;
    CUmipmappedArray mip_array = nullptr;

    if (use_mip) {
        // Mipmapped path: create mipmapped array
        CUDA_ARRAY3D_DESCRIPTOR arr_desc = {};
        arr_desc.Width = width;
        arr_desc.Height = height;
        arr_desc.Depth = 0;  // 0 for 2D mipmapped arrays
        arr_desc.Format = format;
        arr_desc.NumChannels = num_channels;
        arr_desc.Flags = 0;

        CUresult result = cuMipmappedArrayCreate_f(&mip_array, &arr_desc, num_mip_levels);
        if (result != CUDA_SUCCESS) {
            return false;
        }

        // Copy data to each mip level
        const char* src_ptr = (const char*)data;
        for (int level = 0; level < num_mip_levels; ++level) {
            CUarray level_array;
            result = cuMipmappedArrayGetLevel_f(&level_array, mip_array, level);
            if (result != CUDA_SUCCESS) {
                cuMipmappedArrayDestroy_f(mip_array);
                return false;
            }

            int lw = mip_widths[level];
            int lh = mip_heights[level];

            CUDA_MEMCPY2D copy_params = {};
            copy_params.srcMemoryType = CU_MEMORYTYPE_HOST;
            copy_params.srcHost = src_ptr;
            copy_params.srcPitch = lw * bytes_per_texel;
            copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            copy_params.dstArray = level_array;
            copy_params.WidthInBytes = lw * bytes_per_texel;
            copy_params.Height = lh;

            result = cuMemcpy2D_f(&copy_params);
            if (result != CUDA_SUCCESS) {
                cuMipmappedArrayDestroy_f(mip_array);
                return false;
            }

            src_ptr += (size_t)lw * lh * bytes_per_texel;
        }

        res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        res_desc.res.mipmap.hMipmappedArray = mip_array;
    } else {
        // Non-mipmapped path: create regular CUDA array
        CUDA_ARRAY_DESCRIPTOR arr_desc = {};
        arr_desc.Width = width;
        arr_desc.Height = height;
        arr_desc.Format = format;
        arr_desc.NumChannels = num_channels;

        CUresult result = cuArrayCreate_f(&cuda_array, &arr_desc);
        if (result != CUDA_SUCCESS) {
            return false;
        }

        CUDA_MEMCPY2D copy_params = {};
        copy_params.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy_params.srcHost = data;
        copy_params.srcPitch = width * bytes_per_texel;
        copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy_params.dstArray = cuda_array;
        copy_params.WidthInBytes = width * bytes_per_texel;
        copy_params.Height = height;

        result = cuMemcpy2D_f(&copy_params);
        if (result != CUDA_SUCCESS) {
            cuArrayDestroy_f(cuda_array);
            return false;
        }

        res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = cuda_array;
    }

    // Create texture descriptor
    CUDA_TEXTURE_DESC tex_desc = {};
    tex_desc.addressMode[0] = get_cuda_address_mode(address_mode_u);
    tex_desc.addressMode[1] = get_cuda_address_mode(address_mode_v);
    tex_desc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    tex_desc.filterMode = (filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;
    tex_desc.flags = use_normalized_coords ? CU_TRSF_NORMALIZED_COORDINATES : 0;
    tex_desc.maxAnisotropy = 0;

    if (use_mip) {
        tex_desc.mipmapFilterMode = (mip_filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;
        tex_desc.maxMipmapLevelClamp = (float)(num_mip_levels - 1);
    } else {
        tex_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    }
    tex_desc.mipmapLevelBias = 0;
    tex_desc.minMipmapLevelClamp = 0;

    // Create texture object
    CUtexObject tex_object;
    CUresult result = cuTexObjectCreate_f(&tex_object, &res_desc, &tex_desc, nullptr);
    if (result != CUDA_SUCCESS) {
        if (use_mip)
            cuMipmappedArrayDestroy_f(mip_array);
        else
            cuArrayDestroy_f(cuda_array);
        return false;
    }

    *tex_handle_out = (uint64_t)tex_object;
    *array_handle_out = use_mip ? 0 : (uint64_t)cuda_array;
    *mipmap_handle_out = use_mip ? (uint64_t)mip_array : 0;

    return true;
}

void wp_texture2d_destroy_device(void* context, uint64_t tex_handle, uint64_t array_handle, uint64_t mipmap_handle)
{
    if (tex_handle == 0 && array_handle == 0 && mipmap_handle == 0)
        return;

    ContextGuard guard(context);

    if (tex_handle != 0)
        cuTexObjectDestroy_f((CUtexObject)tex_handle);
    if (array_handle != 0)
        cuArrayDestroy_f((CUarray)array_handle);
    if (mipmap_handle != 0)
        cuMipmappedArrayDestroy_f((CUmipmappedArray)mipmap_handle);
}

bool wp_texture3d_create_device(
    void* context,
    int width,
    int height,
    int depth,
    int num_channels,
    int dtype,
    int filter_mode,
    int mip_filter_mode,
    int address_mode_u,
    int address_mode_v,
    int address_mode_w,
    bool use_normalized_coords,
    int num_mip_levels,
    const void* data,
    const int* mip_widths,
    const int* mip_heights,
    const int* mip_depths,
    uint64_t* tex_handle_out,
    uint64_t* array_handle_out,
    uint64_t* mipmap_handle_out
)
{
    if (width <= 0 || height <= 0 || depth <= 0 || data == nullptr) {
        return false;
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        fprintf(
            stderr,
            "Warp texture error: Only 1, 2, or 4 channel textures are supported (got %d channels). "
            "3-channel textures are not supported due to CUDA hardware limitations.\n",
            num_channels
        );
        return false;
    }

    ContextGuard guard(context);

    CUarray_format format = get_cuda_format(dtype);
    int bytes_per_channel = get_bytes_per_channel(dtype);
    int bytes_per_texel = num_channels * bytes_per_channel;

    bool use_mip = (num_mip_levels > 1 && mip_widths != nullptr && mip_heights != nullptr && mip_depths != nullptr);

    CUDA_RESOURCE_DESC res_desc = {};
    res_desc.flags = 0;

    CUarray cuda_array = nullptr;
    CUmipmappedArray mip_array = nullptr;

    CUDA_ARRAY3D_DESCRIPTOR arr_desc = {};
    arr_desc.Width = width;
    arr_desc.Height = height;
    arr_desc.Depth = depth;
    arr_desc.Format = format;
    arr_desc.NumChannels = num_channels;
    arr_desc.Flags = 0;

    if (use_mip) {
        // Mipmapped path
        CUresult result = cuMipmappedArrayCreate_f(&mip_array, &arr_desc, num_mip_levels);
        if (result != CUDA_SUCCESS) {
            return false;
        }

        const char* src_ptr = (const char*)data;
        for (int level = 0; level < num_mip_levels; ++level) {
            CUarray level_array;
            result = cuMipmappedArrayGetLevel_f(&level_array, mip_array, level);
            if (result != CUDA_SUCCESS) {
                cuMipmappedArrayDestroy_f(mip_array);
                return false;
            }

            int lw = mip_widths[level];
            int lh = mip_heights[level];
            int ld = mip_depths[level];

            CUDA_MEMCPY3D copy_params = {};
            copy_params.srcMemoryType = CU_MEMORYTYPE_HOST;
            copy_params.srcHost = src_ptr;
            copy_params.srcPitch = lw * bytes_per_texel;
            copy_params.srcHeight = lh;
            copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            copy_params.dstArray = level_array;
            copy_params.WidthInBytes = lw * bytes_per_texel;
            copy_params.Height = lh;
            copy_params.Depth = ld;

            result = cuMemcpy3D_f(&copy_params);
            if (result != CUDA_SUCCESS) {
                cuMipmappedArrayDestroy_f(mip_array);
                return false;
            }

            src_ptr += (size_t)lw * lh * ld * bytes_per_texel;
        }

        res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        res_desc.res.mipmap.hMipmappedArray = mip_array;
    } else {
        // Non-mipmapped path
        CUresult result = cuArray3DCreate_f(&cuda_array, &arr_desc);
        if (result != CUDA_SUCCESS) {
            return false;
        }

        CUDA_MEMCPY3D copy_params = {};
        copy_params.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy_params.srcHost = data;
        copy_params.srcPitch = width * bytes_per_texel;
        copy_params.srcHeight = height;
        copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy_params.dstArray = cuda_array;
        copy_params.WidthInBytes = width * bytes_per_texel;
        copy_params.Height = height;
        copy_params.Depth = depth;

        result = cuMemcpy3D_f(&copy_params);
        if (result != CUDA_SUCCESS) {
            cuArrayDestroy_f(cuda_array);
            return false;
        }

        res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = cuda_array;
    }

    // Create texture descriptor
    CUDA_TEXTURE_DESC tex_desc = {};
    tex_desc.addressMode[0] = get_cuda_address_mode(address_mode_u);
    tex_desc.addressMode[1] = get_cuda_address_mode(address_mode_v);
    tex_desc.addressMode[2] = get_cuda_address_mode(address_mode_w);
    tex_desc.filterMode = (filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;
    tex_desc.flags = use_normalized_coords ? CU_TRSF_NORMALIZED_COORDINATES : 0;
    tex_desc.maxAnisotropy = 0;

    if (use_mip) {
        tex_desc.mipmapFilterMode = (mip_filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;
        tex_desc.maxMipmapLevelClamp = (float)(num_mip_levels - 1);
    } else {
        tex_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    }
    tex_desc.mipmapLevelBias = 0;
    tex_desc.minMipmapLevelClamp = 0;

    // Create texture object
    CUtexObject tex_object;
    CUresult result = cuTexObjectCreate_f(&tex_object, &res_desc, &tex_desc, nullptr);
    if (result != CUDA_SUCCESS) {
        if (use_mip)
            cuMipmappedArrayDestroy_f(mip_array);
        else
            cuArrayDestroy_f(cuda_array);
        return false;
    }

    *tex_handle_out = (uint64_t)tex_object;
    *array_handle_out = use_mip ? 0 : (uint64_t)cuda_array;
    *mipmap_handle_out = use_mip ? (uint64_t)mip_array : 0;

    return true;
}

void wp_texture3d_destroy_device(void* context, uint64_t tex_handle, uint64_t array_handle, uint64_t mipmap_handle)
{
    if (tex_handle == 0 && array_handle == 0 && mipmap_handle == 0)
        return;

    ContextGuard guard(context);

    if (tex_handle != 0)
        cuTexObjectDestroy_f((CUtexObject)tex_handle);
    if (array_handle != 0)
        cuArrayDestroy_f((CUarray)array_handle);
    if (mipmap_handle != 0)
        cuMipmappedArrayDestroy_f((CUmipmappedArray)mipmap_handle);
}

#else  // WP_ENABLE_CUDA

// Stub implementations for non-CUDA builds
bool wp_texture2d_create_device(
    void* context,
    int width,
    int height,
    int num_channels,
    int dtype,
    int filter_mode,
    int mip_filter_mode,
    int address_mode_u,
    int address_mode_v,
    bool use_normalized_coords,
    int num_mip_levels,
    const void* data,
    const int* mip_widths,
    const int* mip_heights,
    uint64_t* tex_handle_out,
    uint64_t* array_handle_out,
    uint64_t* mipmap_handle_out
)
{
    return false;
}

void wp_texture2d_destroy_device(void* context, uint64_t tex_handle, uint64_t array_handle, uint64_t mipmap_handle) { }

bool wp_texture3d_create_device(
    void* context,
    int width,
    int height,
    int depth,
    int num_channels,
    int dtype,
    int filter_mode,
    int mip_filter_mode,
    int address_mode_u,
    int address_mode_v,
    int address_mode_w,
    bool use_normalized_coords,
    int num_mip_levels,
    const void* data,
    const int* mip_widths,
    const int* mip_heights,
    const int* mip_depths,
    uint64_t* tex_handle_out,
    uint64_t* array_handle_out,
    uint64_t* mipmap_handle_out
)
{
    return false;
}

void wp_texture3d_destroy_device(void* context, uint64_t tex_handle, uint64_t array_handle, uint64_t mipmap_handle) { }

#endif  // WP_ENABLE_CUDA
