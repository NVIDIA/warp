#include "volume.h"

#include "warp.h"

#ifndef WP_CUDA

using namespace wp;

// Creates a Volume on the specified device
// NB: buf must be a pointer on the same device
template<Device device>
uint64_t volume_create(void* buf, uint64_t size)
{
    { // Checking for a valid NanoVDB buffer
        uint64_t nanovdb_magic;
        memcpy<device, Device::CPU>(&nanovdb_magic, buf, sizeof(uint64_t));
        if (nanovdb_magic != PNANOVDB_MAGIC_NUMBER) {
            return 0; // NanoVDB signature missing!
        }
    }
    
    void* target_buf = alloc<device>(size);
    memcpy<device, device>(target_buf, buf, size);

    Volume *volume = new Volume;
    volume->buf = pnanovdb_make_buf((pnanovdb_uint32_t*)target_buf, size / sizeof(uint32_t));
    volume->grid = { 0u };
    volume->tree = pnanovdb_grid_get_tree(volume->buf, volume->grid);
    volume->size_in_bytes = size;

    Volume *volume_result = volume;

    if (device == Device::CUDA) {
        volume_result = (Volume*)alloc<device>(sizeof(Volume));
        memcpy<Device::CPU, device>(volume_result, volume, sizeof(Volume));
        delete volume;
    }

    return (uint64_t)volume_result;
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
    if (!id) {
        *buf = 0;
        *size = 0;
        return;
    }

    Volume* volume_src = (Volume*)(id);
    if (device == Device::CUDA) {
        Volume volume;
        memcpy_d2h(&volume, volume_src, sizeof(Volume));
        *buf = volume.buf.data;
        *size = volume.size_in_bytes;
    } else {
        *buf = volume_src->buf.data;
        *size = volume_src->size_in_bytes;
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
    if (!id) return;

    Volume* volume_src = (Volume*)(id);
    if (device == Device::CUDA) {
        Volume volume;
        memcpy_d2h(&volume, volume_src, sizeof(Volume));
        free_device(volume.buf.data);
    } else {
        free_host(volume_src->buf.data);
    }
    free<device>(volume_src);
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
#if __APPLE__

#endif // __APPLE_

#endif // WP_CUDA
