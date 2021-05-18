#include "core.h"

#include "stdlib.h"
#include "string.h"


bool cuda_init();

void init()
{
    cuda_init();
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


// impl. files
#include "bvh.cpp"
#include "mesh.cpp"
//#include "spline.inl"
//#include "volume.inl"


// stubs for MacOS where there is no CUDA
#if __APPLE__


void* alloc_device(size_t s)
{
}

void free_device(void* ptr)
{
}


void memcpy_h2d(void* dest, void* src, size_t n)
{
}

void memcpy_d2h(void* dest, void* src, size_t n)
{
}

void memcpy_d2d(void* dest, void* src, size_t n)
{
}

void memset_device(void* dest, int value, size_t n)
{
}

void synchronize()
{
}

#endif // __APPLE__