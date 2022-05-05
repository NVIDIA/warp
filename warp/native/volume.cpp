/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "volume.h"

#include "warp.h"

#ifndef WP_CUDA

#include <map>

using namespace wp;

namespace
{
struct VolumeDesc
{
    // NanoVDB buffer either in device or host memory
    void* buffer; 
    uint64_t size_in_bytes;

    // offset to the voxel values of the first leaf node realtive to buffer
    uint64_t first_voxel_data_offs;

    // copy of the grids's metadata to keep on the host for device volumes
    pnanovdb_grid_t grid_data;

    // copy of the tree's metadata to keep on the host for device volumes
    pnanovdb_tree_t tree_data;
};

// Host-side volume desciptors. Maps each CPU/GPU volume buffer address (id) to a CPU desc
std::map<uint64_t, VolumeDesc> g_volume_descriptors;

bool volume_get_descriptor(uint64_t id, VolumeDesc& volumeDesc)
{
    if (id == 0) return false;

    const auto& iter = g_volume_descriptors.find(id);
    if (iter == g_volume_descriptors.end())
        return false;
    else
        volumeDesc = iter->second;
    return true;
}

void volume_add_descriptor(uint64_t id, const VolumeDesc& volumeDesc)
{
    g_volume_descriptors[id] = volumeDesc;
}

void volume_rem_descriptor(uint64_t id)
{
    g_volume_descriptors.erase(id);
}
} // anonymous namespace

// Creates a volume on the specified device
// NB: buf must be a pointer on the same device
template<Device device>
uint64_t volume_create(void* buf, uint64_t size)
{
    if (size < sizeof(pnanovdb_grid_t) + sizeof(pnanovdb_tree_t)) { // This cannot be a valid NanoVDB grid with data
        return 0;
    }

    VolumeDesc volumeDesc;
    memcpy<device, Device::CPU>(&volumeDesc.grid_data, buf, sizeof(pnanovdb_grid_t));
    memcpy<device, Device::CPU>(&volumeDesc.tree_data, (pnanovdb_grid_t*)buf + 1, sizeof(pnanovdb_tree_t));

    if (volumeDesc.grid_data.magic != PNANOVDB_MAGIC_NUMBER) {
        return 0;
    }

    volumeDesc.size_in_bytes = size;
    volumeDesc.buffer = alloc<device>(size);
    memcpy<device, device>(volumeDesc.buffer, buf, size);

    volumeDesc.first_voxel_data_offs =
        sizeof(pnanovdb_grid_t) + volumeDesc.tree_data.node_offset_leaf + PNANOVDB_GRID_TYPE_GET(PNANOVDB_GRID_TYPE_FLOAT, leaf_off_table);

    const uint64_t id = (uint64_t)volumeDesc.buffer;

    volume_add_descriptor(id, volumeDesc);

    return id;
}

uint64_t volume_create_host(void* buf, uint64_t size)
{
    return volume_create<Device::CPU>(buf, size);
}

uint64_t volume_create_device(void* buf, uint64_t size)
{
    return volume_create<Device::CUDA>(buf, size);
}


template<Device device>
void volume_get_buffer_info(uint64_t id, void** buf, uint64_t* size)
{
    *buf = 0;
    *size = 0;

    VolumeDesc volumeDesc;
    if (volume_get_descriptor(id, volumeDesc)) {
        *buf = volumeDesc.buffer;
        *size = volumeDesc.size_in_bytes;
    }
}

void volume_get_buffer_info_host(uint64_t id, void** buf, uint64_t* size)
{
    volume_get_buffer_info<Device::CPU>(id, buf, size);
}

void volume_get_buffer_info_device(uint64_t id, void** buf, uint64_t* size)
{
    volume_get_buffer_info<Device::CUDA>(id, buf, size);
}

template<Device device>
void volume_destroy(uint64_t id)
{
    free<device>((void*)id);
    volume_rem_descriptor(id);
}

void volume_destroy_host(uint64_t id)
{
    volume_destroy<Device::CPU>(id);
}

void volume_destroy_device(uint64_t id)
{
    volume_destroy<Device::CUDA>(id);
}

// stubs for non-CUDA platforms
#if WP_DISABLE_CUDA

#endif // WP_DISABLE_CUDA

#endif // WP_CUDA
