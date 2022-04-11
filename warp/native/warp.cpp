/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "warp.h"

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
#include "hashgrid.cpp"
#include "sort.cpp"
#include "volume.cpp"
//#include "spline.inl"


// stubs for platforms where there is no CUDA
#if WP_DISABLE_CUDA

int cuda_init() { return false; }

void* alloc_device(size_t s)
{
    return NULL;
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

WP_API uint64_t cuda_check_device() { return 0;}
WP_API void cuda_report_error(int code, const char* file, int line) {}
WP_API void cuda_acquire_context() {}
WP_API void cuda_restore_context() {}
WP_API void* cuda_get_context() { return NULL;}
WP_API void cuda_set_context(void* ctx) {}
WP_API void* cuda_get_stream() { return NULL; }
WP_API const char* cuda_get_device_name() { return "Not supported"; }
WP_API void cuda_graph_begin_capture() {}
WP_API void* cuda_graph_end_capture() { return NULL; }
WP_API void cuda_graph_launch(void* graph) {}
WP_API void cuda_graph_destroy(void* graph) {}
WP_API size_t cuda_compile_program(const char* cuda_src, const char* include_dir, bool debug, bool verbose, const char* output_file) { return 0; }
WP_API void* cuda_load_module(const char* ptx) { return NULL; }
WP_API void cuda_unload_module(void* module) {}
WP_API void* cuda_get_kernel(void* module, const char* name) { return NULL; }
WP_API size_t cuda_launch_kernel(void* kernel, size_t dim, void** args) { return 0;}

#endif // WP_DISABLE_CUDA