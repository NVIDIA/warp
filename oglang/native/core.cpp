#include "core.h"

#include "string.h"

void init()
{
}

void shutdown()
{
}

void* alloc_host(size_t s)
{
    return malloc(s);
}

void free_host(void* ptr)
{
    free(ptr);
}

void memcpy_h2h(void* dest, void* src, size_t n)
{
    memcpy(dest, src, n);
}

void memset_host(void* dest, int value, size_t n)
{
    memset(dest, value, n);
}

#if CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif // CUDA

void* alloc_device(size_t s)
{
#if CUDA
    void* ptr;
    cudaMalloc(&ptr, s);

    return ptr;
#else
    return NULL;
#endif

}

void free_device(void* ptr)
{
#if CUDA
    cudaFree(ptr);
#endif
}


void memcpy_h2d(void* dest, void* src, size_t n)
{
#if CUDA
    cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice);
#endif
}

void memcpy_d2h(void* dest, void* src, size_t n)
{
#if CUDA
    cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost);
#endif
}

void memcpy_d2d(void* dest, void* src, size_t n)
{
#if CUDA
    cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToDevice);
#endif
}

void memset_device(void* dest, int value, size_t n)
{
#if CUDA
    cudaMemsetAsync(dest, value, n);
#endif
}

void synchronize()
{
#if CUDA
    cudaStreamSynchronize(0);
#endif
}

// impl. files
#include "bvh.inl"
#include "mesh.inl"
//#include "spline.inl"
//#include "volume.inl"

