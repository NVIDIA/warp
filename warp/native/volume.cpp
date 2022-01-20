#include "volume.h"

#ifndef WP_CUDA
#include <nanovdb/util/Primitives.h>

using namespace wp;

uint64_t volume_create_device(void* buf, uint64_t size, bool on_device, bool copy)
{
    void* device_buf;
    if (on_device) {
        if (copy) {
            device_buf = alloc_device(size);
            memcpy_d2d(device_buf, buf, size);
        } else {
            device_buf = buf;
        }
    } else {
        device_buf = alloc_device(size);
        memcpy_h2d(device_buf, buf, size);
    }

    Volume volume;
    volume.buf = pnanovdb_make_buf((pnanovdb_uint32_t*)device_buf, size / sizeof(uint32_t));
    volume.grid = { 0u };
    volume.tree = pnanovdb_grid_get_tree(volume.buf, volume.grid);
    volume.size_in_bytes = size;

    Volume* volume_device = (Volume*)alloc_device(sizeof(Volume));
    memcpy_h2d(volume_device, &volume, sizeof(Volume));
    return (uint64_t)volume_device;
}

void volume_get_buffer_info(uint64_t id, void** buf, uint64_t* size)
{
    if (!id) {
        *buf = 0;
        *size = 0;
        return;
    }

    // TODO: decide if the volume is on the device or host
    // assuming device for now

    Volume volume;
    Volume* volume_device = (Volume*)(id);
    memcpy_d2h(&volume, volume_device, sizeof(Volume));
    *buf = volume.buf.data;
    *size = volume.size_in_bytes;
}

void volume_destroy_device(uint64_t id)
{
    if (!id) return;

    Volume volume;
    Volume* volume_device = (Volume*)(id);
    memcpy_d2h(&volume, volume_device, sizeof(Volume));
    free_device(volume.buf.data);
    free_device(volume_device);
}

// Utils

uint64_t volume_create_sphere_device(float radius, float center_x, float center_y, float center_z, float voxel_size)
{
    nanovdb::GridHandle<nanovdb::HostBuffer> handle =
        nanovdb::createLevelSetSphere<float, float>(radius, nanovdb::Vec3f(center_x, center_y, center_z), voxel_size);
    nanovdb::NanoGrid<float>* grid = handle.grid<float>();

    return volume_create_device(handle.data(), grid->gridSize(), false, true);
}

// stubs for non-CUDA platforms
#if __APPLE__

#endif // __APPLE_

#endif // WP_CUDA
