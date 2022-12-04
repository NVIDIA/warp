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
#if !WP_DISABLE_CUDA
    // note: it's safe to proceed even if CUDA initialization failed
    cuda_init();
#endif

    return 0;
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

void* alloc_pinned(size_t s)
{
    // CUDA is not available, fall back on system allocator
    return alloc_host(s);
}

void free_pinned(void* ptr)
{
    // CUDA is not available, fall back on system allocator
    free_host(ptr);
}

void* alloc_device(void* context, size_t s)
{
    return NULL;
}

void free_device(void* context, void* ptr)
{
}


void memcpy_h2d(void* context, void* dest, void* src, size_t n)
{
}

void memcpy_d2h(void* context, void* dest, void* src, size_t n)
{
}

void memcpy_d2d(void* context, void* dest, void* src, size_t n)
{
}

void memcpy_peer(void* context, void* dest, void* src, size_t n)
{
}

void memset_device(void* context, void* dest, int value, size_t n)
{
}

WP_API int cuda_driver_version() { return 0; }
WP_API int cuda_toolkit_version() { return 0; }

WP_API int nvrtc_supported_arch_count() { return 0; }
WP_API void nvrtc_supported_archs(int* archs) {}

WP_API int cuda_device_get_count() { return 0; }
WP_API void* cuda_device_primary_context_retain(int ordinal) { return NULL; }
WP_API void cuda_device_primary_context_release(int ordinal) {}
WP_API const char* cuda_device_get_name(int ordinal) { return NULL; }
WP_API int cuda_device_get_arch(int ordinal) { return 0; }
WP_API int cuda_device_is_uva(int ordinal) { return 0; }

WP_API void* cuda_context_get_current() { return NULL; }
WP_API void cuda_context_set_current(void* ctx) {}
WP_API void cuda_context_push_current(void* context) {}
WP_API void cuda_context_pop_current() {}
WP_API void* cuda_context_create(int device_ordinal) { return NULL; }
WP_API void cuda_context_destroy(void* context) {}
WP_API void cuda_context_synchronize(void* context) {}
WP_API uint64_t cuda_context_check(void* context) { return 0; }
WP_API int cuda_context_get_device_ordinal(void* context) { return -1; }
WP_API int cuda_context_is_primary(void* context) { return 0; }
WP_API void* cuda_context_get_stream(void* context) { return NULL; }
WP_API void cuda_context_set_stream(void* context, void* stream) {}
WP_API int cuda_context_can_access_peer(void* context, void* peer_context) { return 0; }
WP_API int cuda_context_enable_peer_access(void* context, void* peer_context) { return 0; }

WP_API void* cuda_stream_create(void* context) { return NULL; }
WP_API void cuda_stream_destroy(void* context, void* stream) {}
WP_API void* cuda_stream_get_current() { return NULL; }
WP_API void cuda_stream_synchronize(void* context, void* stream) {}
WP_API void cuda_stream_wait_event(void* context, void* stream, void* event) {}
WP_API void cuda_stream_wait_stream(void* context, void* stream, void* other_stream, void* event) {}

WP_API void* cuda_event_create(void* context, unsigned flags) { return NULL; }
WP_API void cuda_event_destroy(void* context, void* event) {}
WP_API void cuda_event_record(void* context, void* event, void* stream) {}

WP_API void cuda_graph_begin_capture(void* context) {}
WP_API void* cuda_graph_end_capture(void* context) { return NULL; }
WP_API void cuda_graph_launch(void* context, void* graph) {}
WP_API void cuda_graph_destroy(void* context, void* graph) {}

WP_API size_t cuda_compile_program(const char* cuda_src, int arch, const char* include_dir, bool debug, bool verbose, bool verify_fp, bool fast_math, const char* output_file) { return 0; }

WP_API void* cuda_load_module(void* context, const char* ptx) { return NULL; }
WP_API void cuda_unload_module(void* context, void* module) {}
WP_API void* cuda_get_kernel(void* context, void* module, const char* name) { return NULL; }
WP_API size_t cuda_launch_kernel(void* context, void* kernel, size_t dim, void** args) { return 0;}

WP_API void cuda_set_context_restore_policy(bool always_restore) {}
WP_API int cuda_get_context_restore_policy() { return false; }

#endif // WP_DISABLE_CUDA