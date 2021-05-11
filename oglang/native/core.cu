#include "core.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

void* alloc_device(size_t s)
{
    void* ptr;
    cudaMalloc(&ptr, s);

    return ptr;
}

void free_device(void* ptr)
{
    cudaFree(ptr);
}


void memcpy_h2d(void* dest, void* src, size_t n)
{
    cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice);
}

void memcpy_d2h(void* dest, void* src, size_t n)
{
    cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost);
}

void memcpy_d2d(void* dest, void* src, size_t n)
{
    cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToDevice);
}

void memset_device(void* dest, int value, size_t n)
{
    cudaMemsetAsync(dest, value, n);
}

void synchronize()
{
    cudaStreamSynchronize(0);
}

// impl. files
#include "bvh.cu"
#include "mesh.cu"
//#include "spline.inl"
//#include "volume.inl"

