#include "core.h"

#include "stdlib.h"
#include "string.h"

int cuda_init();

int init()
{
    return cuda_init();
}

void shutdown()
{
}

// void* alloc_host(size_t s)
// {
//     return malloc(s);
// }

// void free_host(void* ptr)
// {
//     free(ptr);
// }

void memcpy_h2h(void* dest, void* src, size_t n)
{
    memcpy(dest, src, n);
}

void memset_host(void* dest, int value, size_t n)
{
    memset(dest, value, n);
}

void array_inner_host(uint64_t a, uint64_t b, uint64_t out, int len)
{
    const float* ptr_a = (const float*)(a);
    const float* ptr_b = (const float*)(b);
    float* ptr_out = (float*)(out);

    *ptr_out = 0.0f;
    for (int i=0; i < len; ++i)
        *ptr_out += ptr_a[i]*ptr_b[i];
}

void array_sum_host(uint64_t a, uint64_t out, int len)
{
    const float* ptr_a = (const float*)(a);
    float* ptr_out = (float*)(out);

    *ptr_out = 0.0f;
    for (int i=0; i < len; ++i)
        *ptr_out += ptr_a[i];
}



// impl. files
#include "bvh.cpp"
#include "mesh.cpp"
//#include "spline.inl"
//#include "volume.inl"


// stubs for MacOS where there is no CUDA
#if __APPLE__

bool cuda_init() { return false; }

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

uint64_t cuda_check_device() { return 0; }
void cuda_acquire_context() {}
void cuda_restore_context() {}
void* cuda_get_context() { return NULL; }
void cuda_set_context(void* ctx) {}

#endif // __APPLE__