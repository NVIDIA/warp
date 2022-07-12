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


namespace wp
{

extern "C"
{
    #include "exports.h"
}

} // namespace wp

int cuda_init();

int init()
{
#if WP_DISABLE_CUDA
    return 0;
#else
    return cuda_init();
#endif
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
    if ((n%4) > 0)
    {
        memset(dest, value, n);
    }
    else
    {
        const int num_words = n/4;
        for (int i=0; i < num_words; ++i)
            ((int*)dest)[i] = value;
    }
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
#include "cuda_util.cpp"
#include "bvh.cpp"
#include "mesh.cpp"
#include "hashgrid.cpp"
#include "sort.cpp"
#include "volume.cpp"
#include "marching.cpp"
//#include "spline.inl"


// stubs for platforms where there is no CUDA
#if WP_DISABLE_CUDA

int cuda_init() { return -1; }

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

void memcpy_peer(void* dest, void* dest_ctx, void* src, void* src_ctx, size_t n)
{
}

void memset_device(void* dest, int value, size_t n)
{
}

WP_API uint64_t cuda_check_device() { return 0;}

WP_API int cuda_device_get_count() { return 0; }
WP_API void* cuda_device_get_primary_context(int ordinal) { return NULL; }
WP_API const char* cuda_device_get_name(int ordinal) { return NULL; }
WP_API int cuda_device_get_arch(int ordinal) { return 0; }
WP_API int cuda_device_is_uva(int ordinal) { return 0; }

WP_API void* cuda_context_get_current() { return NULL; }
WP_API void cuda_context_set_current(void* ctx) {}
WP_API void cuda_context_push_current(void* context) {}
WP_API void cuda_context_pop_current() {}
WP_API void* cuda_context_create(int device_ordinal) { return NULL; }
WP_API void cuda_context_destroy(void* context) {}
WP_API void cuda_context_synchronize() {}
WP_API int cuda_context_get_device_ordinal(void* context) { return -1; }
WP_API int cuda_context_is_primary(void* context) { return 0; }
WP_API void* cuda_context_get_stream(void* context) { return NULL; }
WP_API int cuda_context_can_access_peer(void* context, void* peer_context) { return 0; }
WP_API int cuda_context_enable_peer_access(void* context, void* peer_context) { return 0; }

WP_API void* cuda_stream_get_current() { return NULL; }

WP_API void cuda_graph_begin_capture() {}
WP_API void* cuda_graph_end_capture() { return NULL; }
WP_API void cuda_graph_launch(void* graph) {}
WP_API void cuda_graph_destroy(void* graph) {}
WP_API size_t cuda_compile_program(const char* cuda_src, int arch, const char* include_dir, bool debug, bool verbose, bool verify_fp, const char* output_file) { return 0; }
WP_API void* cuda_load_module(const char* ptx) { return NULL; }
WP_API void cuda_unload_module(void* module) {}
WP_API void* cuda_get_kernel(void* module, const char* name) { return NULL; }
WP_API size_t cuda_launch_kernel(void* kernel, size_t dim, void** args) { return 0;}

#endif // WP_DISABLE_CUDA