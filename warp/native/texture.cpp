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
    int address_mode_u,
    int address_mode_v,
    bool use_normalized_coords,
    const void* data,
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

    // Calculate data size and allocate storage
    int bytes_per_channel = get_bytes_per_channel(dtype);
    size_t data_size = (size_t)width * height * num_channels * bytes_per_channel;

    tex_data->data = malloc(data_size);
    if (tex_data->data == nullptr) {
        free(tex_data);
        return false;
    }

    // Copy the texture data
    memcpy(tex_data->data, data, data_size);

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
    if (tex_data->data != nullptr) {
        free(tex_data->data);
    }
    free(tex_data);
}

bool wp_texture3d_create_host(
    int width,
    int height,
    int depth,
    int num_channels,
    int dtype,
    int filter_mode,
    int address_mode_u,
    int address_mode_v,
    int address_mode_w,
    bool use_normalized_coords,
    const void* data,
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

    // Calculate data size and allocate storage
    int bytes_per_channel = get_bytes_per_channel(dtype);
    size_t data_size = (size_t)width * height * depth * num_channels * bytes_per_channel;

    tex_data->data = malloc(data_size);
    if (tex_data->data == nullptr) {
        free(tex_data);
        return false;
    }

    // Copy the texture data
    memcpy(tex_data->data, data, data_size);

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
    if (tex_data->data != nullptr) {
        free(tex_data->data);
    }
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
    int address_mode_u,
    int address_mode_v,
    bool use_normalized_coords,
    const void* data,
    uint64_t* tex_handle_out,
    uint64_t* array_handle_out
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

    // Create CUDA array descriptor
    CUDA_ARRAY_DESCRIPTOR arr_desc = {};
    arr_desc.Width = width;
    arr_desc.Height = height;
    arr_desc.Format = format;
    arr_desc.NumChannels = num_channels;

    // Create the array
    CUarray cuda_array;
    CUresult result = cuArrayCreate_f(&cuda_array, &arr_desc);
    if (result != CUDA_SUCCESS) {
        return false;
    }

    // Copy data to the array
    int bytes_per_texel = num_channels * bytes_per_channel;
    CUDA_MEMCPY2D copy_params = {};
    copy_params.srcXInBytes = 0;
    copy_params.srcY = 0;
    copy_params.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy_params.srcHost = data;
    copy_params.srcPitch = width * bytes_per_texel;

    copy_params.dstXInBytes = 0;
    copy_params.dstY = 0;
    copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy_params.dstArray = cuda_array;

    copy_params.WidthInBytes = width * bytes_per_texel;
    copy_params.Height = height;

    result = cuMemcpy2D_f(&copy_params);
    if (result != CUDA_SUCCESS) {
        cuArrayDestroy_f(cuda_array);
        return false;
    }

    // Create resource descriptor
    CUDA_RESOURCE_DESC res_desc = {};
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = cuda_array;
    res_desc.flags = 0;

    // Create texture descriptor
    CUDA_TEXTURE_DESC tex_desc = {};

    // Per-axis address modes
    tex_desc.addressMode[0] = get_cuda_address_mode(address_mode_u);
    tex_desc.addressMode[1] = get_cuda_address_mode(address_mode_v);
    tex_desc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;  // Not used for 2D, but set a default

    // Filter mode: 0=nearest, 1=linear
    tex_desc.filterMode = (filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;

    // Coordinate mode: normalized [0,1] or texel space [0,width/height]
    // For uint8/uint16 textures, CUDA automatically normalizes values to [0,1] when sampled
    // (since CU_TRSF_READ_AS_INTEGER is NOT set). Float32 textures return values as-is.
    tex_desc.flags = use_normalized_coords ? CU_TRSF_NORMALIZED_COORDINATES : 0;

    tex_desc.maxAnisotropy = 0;
    tex_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    tex_desc.mipmapLevelBias = 0;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.maxMipmapLevelClamp = 0;

    // Create texture object
    CUtexObject tex_object;
    result = cuTexObjectCreate_f(&tex_object, &res_desc, &tex_desc, nullptr);
    if (result != CUDA_SUCCESS) {
        cuArrayDestroy_f(cuda_array);
        return false;
    }

    *tex_handle_out = (uint64_t)tex_object;
    *array_handle_out = (uint64_t)cuda_array;

    return true;
}

void wp_texture2d_destroy_device(void* context, uint64_t tex_handle, uint64_t array_handle)
{
    if (tex_handle == 0 && array_handle == 0)
        return;

    ContextGuard guard(context);

    if (tex_handle != 0) {
        cuTexObjectDestroy_f((CUtexObject)tex_handle);
    }

    if (array_handle != 0) {
        cuArrayDestroy_f((CUarray)array_handle);
    }
}

bool wp_texture3d_create_device(
    void* context,
    int width,
    int height,
    int depth,
    int num_channels,
    int dtype,
    int filter_mode,
    int address_mode_u,
    int address_mode_v,
    int address_mode_w,
    bool use_normalized_coords,
    const void* data,
    uint64_t* tex_handle_out,
    uint64_t* array_handle_out
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

    // Determine the CUDA array format
    CUarray_format format = get_cuda_format(dtype);
    int bytes_per_channel = get_bytes_per_channel(dtype);

    // Create CUDA 3D array descriptor
    CUDA_ARRAY3D_DESCRIPTOR arr_desc = {};
    arr_desc.Width = width;
    arr_desc.Height = height;
    arr_desc.Depth = depth;
    arr_desc.Format = format;
    arr_desc.NumChannels = num_channels;
    arr_desc.Flags = 0;

    // Create the 3D array
    CUarray cuda_array;
    CUresult result = cuArray3DCreate_f(&cuda_array, &arr_desc);
    if (result != CUDA_SUCCESS) {
        return false;
    }

    // Copy data to the 3D array
    int bytes_per_texel = num_channels * bytes_per_channel;
    CUDA_MEMCPY3D copy_params = {};
    copy_params.srcXInBytes = 0;
    copy_params.srcY = 0;
    copy_params.srcZ = 0;
    copy_params.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy_params.srcHost = data;
    copy_params.srcPitch = width * bytes_per_texel;
    copy_params.srcHeight = height;

    copy_params.dstXInBytes = 0;
    copy_params.dstY = 0;
    copy_params.dstZ = 0;
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

    // Create resource descriptor
    CUDA_RESOURCE_DESC res_desc = {};
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = cuda_array;
    res_desc.flags = 0;

    // Create texture descriptor
    CUDA_TEXTURE_DESC tex_desc = {};

    // Per-axis address modes
    tex_desc.addressMode[0] = get_cuda_address_mode(address_mode_u);
    tex_desc.addressMode[1] = get_cuda_address_mode(address_mode_v);
    tex_desc.addressMode[2] = get_cuda_address_mode(address_mode_w);

    // Filter mode: 0=nearest, 1=linear
    tex_desc.filterMode = (filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;

    // Coordinate mode: normalized [0,1] or texel space [0,width/height/depth]
    // For uint8/uint16 textures, CUDA automatically normalizes values to [0,1] when sampled
    // (since CU_TRSF_READ_AS_INTEGER is NOT set). Float32 textures return values as-is.
    tex_desc.flags = use_normalized_coords ? CU_TRSF_NORMALIZED_COORDINATES : 0;

    tex_desc.maxAnisotropy = 0;
    tex_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    tex_desc.mipmapLevelBias = 0;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.maxMipmapLevelClamp = 0;

    // Create texture object
    CUtexObject tex_object;
    result = cuTexObjectCreate_f(&tex_object, &res_desc, &tex_desc, nullptr);
    if (result != CUDA_SUCCESS) {
        cuArrayDestroy_f(cuda_array);
        return false;
    }

    *tex_handle_out = (uint64_t)tex_object;
    *array_handle_out = (uint64_t)cuda_array;

    return true;
}

void wp_texture3d_destroy_device(void* context, uint64_t tex_handle, uint64_t array_handle)
{
    if (tex_handle == 0 && array_handle == 0)
        return;

    ContextGuard guard(context);

    if (tex_handle != 0) {
        cuTexObjectDestroy_f((CUtexObject)tex_handle);
    }

    if (array_handle != 0) {
        cuArrayDestroy_f((CUarray)array_handle);
    }
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
    int address_mode_u,
    int address_mode_v,
    bool use_normalized_coords,
    const void* data,
    uint64_t* tex_handle_out,
    uint64_t* array_handle_out
)
{
    return false;
}

void wp_texture2d_destroy_device(void* context, uint64_t tex_handle, uint64_t array_handle) { }

bool wp_texture3d_create_device(
    void* context,
    int width,
    int height,
    int depth,
    int num_channels,
    int dtype,
    int filter_mode,
    int address_mode_u,
    int address_mode_v,
    int address_mode_w,
    bool use_normalized_coords,
    const void* data,
    uint64_t* tex_handle_out,
    uint64_t* array_handle_out
)
{
    return false;
}

void wp_texture3d_destroy_device(void* context, uint64_t tex_handle, uint64_t array_handle) { }

#endif  // WP_ENABLE_CUDA
