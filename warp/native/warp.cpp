/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "warp.h"
#include "scan.h"
#include "array.h"

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


uint16_t float_to_half_bits(float x)
{
    return wp::half(x).u;
}

float half_bits_to_float(uint16_t u)
{
    wp::half h;
    h.u = u;
    return half_to_float(h);
}

int init()
{
#if WP_ENABLE_CUDA
    // note: it's safe to proceed even if CUDA initialization failed
    cuda_init();
#endif

    return 0;
}

void shutdown()
{
}

int is_cuda_enabled()
{
    return int(WP_ENABLE_CUDA);
}

int is_cuda_compatibility_enabled()
{
    return int(WP_ENABLE_CUDA_COMPATIBILITY);
}

int is_cutlass_enabled()
{
    return int(WP_ENABLE_CUTLASS);
}

int is_debug_enabled()
{
    return int(WP_ENABLE_DEBUG);
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
        const size_t num_words = n/4;
        for (size_t i=0; i < num_words; ++i)
            ((int*)dest)[i] = value;
    }
}

// fill memory buffer with a value: this is a faster memtile variant
// for types bigger than one byte, but requires proper alignment of dst
template <typename T>
void memtile_value_host(T* dst, T value, size_t n)
{
    while (n--)
        *dst++ = value;
}

void memtile_host(void* dst, const void* src, size_t srcsize, size_t n)
{
    size_t dst_addr = reinterpret_cast<size_t>(dst);
    size_t src_addr = reinterpret_cast<size_t>(src);

    // try memtile_value first because it should be faster, but we need to ensure proper alignment
    if (srcsize == 8 && (dst_addr & 7) == 0 && (src_addr & 7) == 0)
        memtile_value_host(reinterpret_cast<int64_t*>(dst), *reinterpret_cast<const int64_t*>(src), n);
    else if (srcsize == 4 && (dst_addr & 3) == 0 && (src_addr & 3) == 0)
        memtile_value_host(reinterpret_cast<int32_t*>(dst), *reinterpret_cast<const int32_t*>(src), n);
    else if (srcsize == 2 && (dst_addr & 1) == 0 && (src_addr & 1) == 0)
        memtile_value_host(reinterpret_cast<int16_t*>(dst), *reinterpret_cast<const int16_t*>(src), n);
    else if (srcsize == 1)
        memset(dst, *reinterpret_cast<const int8_t*>(src), n);
    else
    {
        // generic version
        while (n--)
        {
            memcpy(dst, src, srcsize);
            dst = (int8_t*)dst + srcsize;
        }
    }
}

void array_scan_int_host(uint64_t in, uint64_t out, int len, bool inclusive)
{
    scan_host((const int*)in, (int*)out, len, inclusive);
}

void array_scan_float_host(uint64_t in, uint64_t out, int len, bool inclusive)
{
    scan_host((const float*)in, (float*)out, len, inclusive);
}


static void array_copy_nd(void* dst, const void* src,
                      const int* dst_strides, const int* src_strides,
                      const int*const* dst_indices, const int*const* src_indices,
                      const int* shape, int ndim, int elem_size)
{
    if (ndim == 1)
    {
        for (int i = 0; i < shape[0]; i++)
        {
            int src_idx = src_indices[0] ? src_indices[0][i] : i;
            int dst_idx = dst_indices[0] ? dst_indices[0][i] : i;
            const char* p = (const char*)src + src_idx * src_strides[0];
            char* q = (char*)dst + dst_idx * dst_strides[0];
            // copy element
            memcpy(q, p, elem_size);
        }
    }
    else
    {
        for (int i = 0; i < shape[0]; i++)
        {
            int src_idx = src_indices[0] ? src_indices[0][i] : i;
            int dst_idx = dst_indices[0] ? dst_indices[0][i] : i;
            const char* p = (const char*)src + src_idx * src_strides[0];
            char* q = (char*)dst + dst_idx * dst_strides[0];
            // recurse on next inner dimension
            array_copy_nd(q, p, dst_strides + 1, src_strides + 1, dst_indices + 1, src_indices + 1, shape + 1, ndim - 1, elem_size);
        }
    }
}


WP_API size_t array_copy_host(void* dst, void* src, int dst_type, int src_type, int elem_size)
{
    if (!src || !dst)
        return 0;

    const void* src_data = NULL;
    const void* src_grad = NULL;
    void* dst_data = NULL;
    void* dst_grad = NULL;
    int src_ndim = 0;
    int dst_ndim = 0;
    const int* src_shape = NULL;
    const int* dst_shape = NULL;
    const int* src_strides = NULL;
    const int* dst_strides = NULL;
    const int*const* src_indices = NULL;
    const int*const* dst_indices = NULL;

    const int* null_indices[wp::ARRAY_MAX_DIMS] = { NULL };

    if (src_type == wp::ARRAY_TYPE_REGULAR)
    {
        const wp::array_t<void>& src_arr = *static_cast<const wp::array_t<void>*>(src);
        src_data = src_arr.data;
        src_grad = src_arr.grad;
        src_ndim = src_arr.ndim;
        src_shape = src_arr.shape.dims;
        src_strides = src_arr.strides;
        src_indices = null_indices;
    }
    else if (src_type == wp::ARRAY_TYPE_INDEXED)
    {
        const wp::indexedarray_t<void>& src_arr = *static_cast<const wp::indexedarray_t<void>*>(src);
        src_data = src_arr.arr.data;
        src_ndim = src_arr.arr.ndim;
        src_shape = src_arr.shape.dims;
        src_strides = src_arr.arr.strides;
        src_indices = src_arr.indices;
    }
    else
    {
        fprintf(stderr, "Warp error: Invalid array type (%d)\n", src_type);
        return 0;
    }

    if (dst_type == wp::ARRAY_TYPE_REGULAR)
    {
        const wp::array_t<void>& dst_arr = *static_cast<const wp::array_t<void>*>(dst);
        dst_data = dst_arr.data;
        dst_grad = dst_arr.grad;
        dst_ndim = dst_arr.ndim;
        dst_shape = dst_arr.shape.dims;
        dst_strides = dst_arr.strides;
        dst_indices = null_indices;
    }
    else if (dst_type == wp::ARRAY_TYPE_INDEXED)
    {
        const wp::indexedarray_t<void>& dst_arr = *static_cast<const wp::indexedarray_t<void>*>(dst);
        dst_data = dst_arr.arr.data;
        dst_ndim = dst_arr.arr.ndim;
        dst_shape = dst_arr.shape.dims;
        dst_strides = dst_arr.arr.strides;
        dst_indices = dst_arr.indices;
    }
    else
    {
        fprintf(stderr, "Warp error: Invalid array type (%d)\n", dst_type);
        return 0;
    }

    if (src_ndim != dst_ndim)
    {
        fprintf(stderr, "Warp error: Incompatible array dimensionalities (%d and %d)\n", src_ndim, dst_ndim);
        return 0;
    }

    bool has_grad = (src_grad && dst_grad);
    size_t n = 1;

    for (int i = 0; i < src_ndim; i++)
    {
        if (src_shape[i] != dst_shape[i])
        {
            fprintf(stderr, "Warp error: Incompatible array shapes\n");
            return 0;
        }
        n *= src_shape[i];
    }

    array_copy_nd(dst_data, src_data,
              dst_strides, src_strides,
              dst_indices, src_indices,
              src_shape, src_ndim, elem_size);

    if (has_grad)
    {
        array_copy_nd(dst_grad, src_grad,
                dst_strides, src_strides,
                dst_indices, src_indices,
                src_shape, src_ndim, elem_size);
    }

    return n;
}


static void array_fill_strided(void* data, const int* shape, const int* strides, int ndim, const void* value, int value_size)
{
    if (ndim == 1)
    {
        char* p = (char*)data;
        for (int i = 0; i < shape[0]; i++)
        {
            memcpy(p, value, value_size);
            p += strides[0];
        }
    }
    else
    {
        for (int i = 0; i < shape[0]; i++)
        {
            char* p = (char*)data + i * strides[0];
            // recurse on next inner dimension
            array_fill_strided(p, shape + 1, strides + 1, ndim - 1, value, value_size);
        }
    }
}


static void array_fill_indexed(void* data, const int* shape, const int* strides, const int*const* indices, int ndim, const void* value, int value_size)
{
    if (ndim == 1)
    {
        for (int i = 0; i < shape[0]; i++)
        {
            int idx = indices[0] ? indices[0][i] : i;
            char* p = (char*)data + idx * strides[0];
            memcpy(p, value, value_size);
        }
    }
    else
    {
        for (int i = 0; i < shape[0]; i++)
        {
            int idx = indices[0] ? indices[0][i] : i;
            char* p = (char*)data + idx * strides[0];
            // recurse on next inner dimension
            array_fill_indexed(p, shape + 1, strides + 1, indices + 1, ndim - 1, value, value_size);
        }
    }
}


WP_API void array_fill_host(void* arr_ptr, int arr_type, const void* value_ptr, int value_size)
{
    if (!arr_ptr || !value_ptr)
        return;

    if (arr_type == wp::ARRAY_TYPE_REGULAR)
    {
        wp::array_t<void>& arr = *static_cast<wp::array_t<void>*>(arr_ptr);
        array_fill_strided(arr.data, arr.shape.dims, arr.strides, arr.ndim, value_ptr, value_size);
    }
    else if (arr_type == wp::ARRAY_TYPE_INDEXED)
    {
        wp::indexedarray_t<void>& ia = *static_cast<wp::indexedarray_t<void>*>(arr_ptr);
        array_fill_indexed(ia.arr.data, ia.shape.dims, ia.arr.strides, ia.indices, ia.arr.ndim, value_ptr, value_size);
    }
    else
    {
        fprintf(stderr, "Warp error: Invalid array type id %d\n", arr_type);
    }
}


// impl. files
// TODO: compile as separate translation units
#include "bvh.cpp"
#include "scan.cpp"


// stubs for platforms where there is no CUDA
#if !WP_ENABLE_CUDA

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

void memtile_device(void* context, void* dest, const void* src, size_t srcsize, size_t n)
{
}

size_t array_copy_device(void* context, void* dst, void* src, int dst_type, int src_type, int elem_size)
{
    return 0;
}

void array_fill_device(void* context, void* arr, int arr_type, const void* value, int value_size)
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

WP_API void array_scan_int_device(uint64_t in, uint64_t out, int len, bool inclusive) {}
WP_API void array_scan_float_device(uint64_t in, uint64_t out, int len, bool inclusive) {}

WP_API void cuda_graphics_map(void* context, void* resource) {}
WP_API void cuda_graphics_unmap(void* context, void* resource) {}
WP_API void cuda_graphics_device_ptr_and_size(void* context, void* resource, uint64_t* ptr, size_t* size) {}
WP_API void* cuda_graphics_register_gl_buffer(void* context, uint32_t gl_buffer, unsigned int flags) { return NULL; }
WP_API void cuda_graphics_unregister_resource(void* context, void* resource) {}

#endif // !WP_ENABLE_CUDA
