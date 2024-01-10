/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "warp.h"
#include "scan.h"
#include "cuda_util.h"

#include <nvrtc.h>
#include <nvPTXCompiler.h>

#include <map>
#include <vector>

#define check_nvrtc(code) (check_nvrtc_result(code, __FILE__, __LINE__))
#define check_nvptx(code) (check_nvptx_result(code, __FILE__, __LINE__))

bool check_nvrtc_result(nvrtcResult result, const char* file, int line)
{
    if (result == NVRTC_SUCCESS)
        return true;

    const char* error_string = nvrtcGetErrorString(result);
    fprintf(stderr, "Warp NVRTC compilation error %u: %s (%s:%d)\n", unsigned(result), error_string, file, line);
    return false;
}

bool check_nvptx_result(nvPTXCompileResult result, const char* file, int line)
{
    if (result == NVPTXCOMPILE_SUCCESS)
        return true;

    const char* error_string;
    switch (result)
    {
    case NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE:
        error_string = "Invalid compiler handle";
        break;
    case NVPTXCOMPILE_ERROR_INVALID_INPUT:
        error_string = "Invalid input";
        break;
    case NVPTXCOMPILE_ERROR_COMPILATION_FAILURE:
        error_string = "Compilation failure";
        break;
    case NVPTXCOMPILE_ERROR_INTERNAL:
        error_string = "Internal error";
        break;
    case NVPTXCOMPILE_ERROR_OUT_OF_MEMORY:
        error_string = "Out of memory";
        break;
    case NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE:
        error_string = "Incomplete compiler invocation";
        break;
    case NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION:
        error_string = "Unsupported PTX version";
        break;
    default:
        error_string = "Unknown error";
        break;
    }

    fprintf(stderr, "Warp PTX compilation error %u: %s (%s:%d)\n", unsigned(result), error_string, file, line);
    return false;
}


struct DeviceInfo
{
    static constexpr int kNameLen = 128;

    CUdevice device = -1;
    int ordinal = -1;
    char name[kNameLen] = "";
    int arch = 0;
    int is_uva = 0;
    int is_memory_pool_supported = 0;
};

struct ContextInfo
{
    DeviceInfo* device_info = NULL;

    CUstream stream = NULL; // created when needed
};

// cached info for all devices, indexed by ordinal
static std::vector<DeviceInfo> g_devices;

// maps CUdevice to DeviceInfo
static std::map<CUdevice, DeviceInfo*> g_device_map;

// cached info for all known contexts
static std::map<CUcontext, ContextInfo> g_contexts;


void cuda_set_context_restore_policy(bool always_restore)
{
    ContextGuard::always_restore = always_restore;
}

int cuda_get_context_restore_policy()
{
    return int(ContextGuard::always_restore);
}

int cuda_init()
{
    if (!init_cuda_driver())
        return -1;

    int deviceCount = 0;
    if (check_cu(cuDeviceGetCount_f(&deviceCount)))
    {
        g_devices.resize(deviceCount);

        for (int i = 0; i < deviceCount; i++)
        {
            CUdevice device;
            if (check_cu(cuDeviceGet_f(&device, i)))
            {
                // query device info
                g_devices[i].device = device;
                g_devices[i].ordinal = i;
                check_cu(cuDeviceGetName_f(g_devices[i].name, DeviceInfo::kNameLen, device));
                check_cu(cuDeviceGetAttribute_f(&g_devices[i].is_uva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
                check_cu(cuDeviceGetAttribute_f(&g_devices[i].is_memory_pool_supported, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device));
                int major = 0;
                int minor = 0;
                check_cu(cuDeviceGetAttribute_f(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
                check_cu(cuDeviceGetAttribute_f(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
                g_devices[i].arch = 10 * major + minor;

                g_device_map[device] = &g_devices[i];
            }
            else
            {
                return -1;
            }
        }
    }
    else
    {
        return -1;
    }

    return 0;
}


static inline CUcontext get_current_context()
{
    CUcontext ctx;
    if (check_cu(cuCtxGetCurrent_f(&ctx)))
        return ctx;
    else
        return NULL;
}

static inline CUstream get_current_stream()
{
    return static_cast<CUstream>(cuda_context_get_stream(NULL));
}

static ContextInfo* get_context_info(CUcontext ctx)
{
    if (!ctx)
    {
        ctx = get_current_context();
        if (!ctx)
            return NULL;
    }

    auto it = g_contexts.find(ctx);
    if (it != g_contexts.end())
    {
        return &it->second;
    }
    else
    {
        // previously unseen context, add the info
        ContextGuard guard(ctx, true);
        ContextInfo context_info;
        CUdevice device;
        if (check_cu(cuCtxGetDevice_f(&device)))
        {
            context_info.device_info = g_device_map[device];
            auto result = g_contexts.insert(std::make_pair(ctx, context_info));
            return &result.first->second;
        }
    }

    return NULL;
}


void* alloc_pinned(size_t s)
{
    void* ptr;
    check_cuda(cudaMallocHost(&ptr, s));
    return ptr;
}

void free_pinned(void* ptr)
{
    cudaFreeHost(ptr);
}

void* alloc_device(void* context, size_t s)
{
    ContextGuard guard(context);

    void* ptr;
    check_cuda(cudaMalloc(&ptr, s));
    return ptr;
}

void* alloc_temp_device(void* context, size_t s)
{
    // "cudaMallocAsync ignores the current device/context when determining where the allocation will reside. Instead,
    // cudaMallocAsync determines the resident device based on the specified memory pool or the supplied stream."
    ContextGuard guard(context);

    void* ptr;

    if (cuda_context_is_memory_pool_supported(context))
    {
        check_cuda(cudaMallocAsync(&ptr, s, get_current_stream()));
    }
    else
    {
        check_cuda(cudaMalloc(&ptr, s));
    }

    return ptr;
}

void free_device(void* context, void* ptr)
{
    ContextGuard guard(context);

    check_cuda(cudaFree(ptr));
}

void free_temp_device(void* context, void* ptr)
{
    ContextGuard guard(context);

    if (cuda_context_is_memory_pool_supported(context))
    {
        check_cuda(cudaFreeAsync(ptr, get_current_stream()));
    }
    else
    {
        check_cuda(cudaFree(ptr));
    }
}

void memcpy_h2d(void* context, void* dest, void* src, size_t n)
{
    ContextGuard guard(context);
    
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice, get_current_stream()));
}

void memcpy_d2h(void* context, void* dest, void* src, size_t n)
{
    ContextGuard guard(context);

    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost, get_current_stream()));
}

void memcpy_d2d(void* context, void* dest, void* src, size_t n)
{
    ContextGuard guard(context);

    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToDevice, get_current_stream()));
}

void memcpy_peer(void* context, void* dest, void* src, size_t n)
{
    ContextGuard guard(context);

    // NB: assumes devices involved support UVA
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDefault, get_current_stream()));
}

__global__ void memset_kernel(int* dest, int value, size_t n)
{
    const size_t tid = wp::grid_index();
    
    if (tid < n)
    {
        dest[tid] = value;
    }
}

void memset_device(void* context, void* dest, int value, size_t n)
{
    ContextGuard guard(context);

    if (true)// ((n%4) > 0)
    {
        // for unaligned lengths fallback to CUDA memset
        check_cuda(cudaMemsetAsync(dest, value, n, get_current_stream()));
    }
    else
    {
        // custom kernel to support 4-byte values (and slightly lower host overhead)
        const size_t num_words = n/4;
        wp_launch_device(WP_CURRENT_CONTEXT, memset_kernel, num_words, ((int*)dest, value, num_words));
    }
}

// fill memory buffer with a value: generic memtile kernel using memcpy for each element
__global__ void memtile_kernel(void* dst, const void* src, size_t srcsize, size_t n)
{
    size_t tid = wp::grid_index();
    if (tid < n)
    {
        memcpy((int8_t*)dst + srcsize * tid, src, srcsize);
    }
}

// this should be faster than memtile_kernel, but requires proper alignment of dst
template <typename T>
__global__ void memtile_value_kernel(T* dst, T value, size_t n)
{
    size_t tid = wp::grid_index();
    if (tid < n)
    {
        dst[tid] = value;
    }
}

void memtile_device(void* context, void* dst, const void* src, size_t srcsize, size_t n)
{
    ContextGuard guard(context);

    size_t dst_addr = reinterpret_cast<size_t>(dst);
    size_t src_addr = reinterpret_cast<size_t>(src);

    // try memtile_value first because it should be faster, but we need to ensure proper alignment
    if (srcsize == 8 && (dst_addr & 7) == 0 && (src_addr & 7) == 0)
    {
        int64_t* p = reinterpret_cast<int64_t*>(dst);
        int64_t value = *reinterpret_cast<const int64_t*>(src);
        wp_launch_device(WP_CURRENT_CONTEXT, memtile_value_kernel, n, (p, value, n));
    }
    else if (srcsize == 4 && (dst_addr & 3) == 0 && (src_addr & 3) == 0)
    {
        int32_t* p = reinterpret_cast<int32_t*>(dst);
        int32_t value = *reinterpret_cast<const int32_t*>(src);
        wp_launch_device(WP_CURRENT_CONTEXT, memtile_value_kernel, n, (p, value, n));
    }
    else if (srcsize == 2 && (dst_addr & 1) == 0 && (src_addr & 1) == 0)
    {
        int16_t* p = reinterpret_cast<int16_t*>(dst);
        int16_t value = *reinterpret_cast<const int16_t*>(src);
        wp_launch_device(WP_CURRENT_CONTEXT, memtile_value_kernel, n, (p, value, n));
    }
    else if (srcsize == 1)
    {
        check_cuda(cudaMemset(dst, *reinterpret_cast<const int8_t*>(src), n));
    }
    else
    {
        // generic version

        // TODO: use a persistent stream-local staging buffer to avoid allocs?
        void* src_device;
        check_cuda(cudaMalloc(&src_device, srcsize));
        check_cuda(cudaMemcpyAsync(src_device, src, srcsize, cudaMemcpyHostToDevice, get_current_stream()));

        wp_launch_device(WP_CURRENT_CONTEXT, memtile_kernel, n, (dst, src_device, srcsize, n));

        check_cuda(cudaFree(src_device));
    }
}


static __global__ void array_copy_1d_kernel(void* dst, const void* src,
                                        int dst_stride, int src_stride,
                                        const int* dst_indices, const int* src_indices,
                                        int n, int elem_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int src_idx = src_indices ? src_indices[i] : i;
        int dst_idx = dst_indices ? dst_indices[i] : i;
        const char* p = (const char*)src + src_idx * src_stride;
        char* q = (char*)dst + dst_idx * dst_stride;
        memcpy(q, p, elem_size);
    }
}

static __global__ void array_copy_2d_kernel(void* dst, const void* src,
                                        wp::vec_t<2, int> dst_strides, wp::vec_t<2, int> src_strides,
                                        wp::vec_t<2, const int*> dst_indices, wp::vec_t<2, const int*> src_indices,
                                        wp::vec_t<2, int> shape, int elem_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = shape[1];
    int i = tid / n;
    int j = tid % n;
    if (i < shape[0] /*&& j < shape[1]*/)
    {
        int src_idx0 = src_indices[0] ? src_indices[0][i] : i;
        int dst_idx0 = dst_indices[0] ? dst_indices[0][i] : i;
        int src_idx1 = src_indices[1] ? src_indices[1][j] : j;
        int dst_idx1 = dst_indices[1] ? dst_indices[1][j] : j;
        const char* p = (const char*)src + src_idx0 * src_strides[0] + src_idx1 * src_strides[1];
        char* q = (char*)dst + dst_idx0 * dst_strides[0] + dst_idx1 * dst_strides[1];
        memcpy(q, p, elem_size);
    }
}

static __global__ void array_copy_3d_kernel(void* dst, const void* src,
                                        wp::vec_t<3, int> dst_strides, wp::vec_t<3, int> src_strides,
                                        wp::vec_t<3, const int*> dst_indices, wp::vec_t<3, const int*> src_indices,
                                        wp::vec_t<3, int> shape, int elem_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = shape[1];
    int o = shape[2];
    int i = tid / (n * o);
    int j = tid % (n * o) / o;
    int k = tid % o;
    if (i < shape[0] && j < shape[1] /*&& k < shape[2]*/)
    {
        int src_idx0 = src_indices[0] ? src_indices[0][i] : i;
        int dst_idx0 = dst_indices[0] ? dst_indices[0][i] : i;
        int src_idx1 = src_indices[1] ? src_indices[1][j] : j;
        int dst_idx1 = dst_indices[1] ? dst_indices[1][j] : j;
        int src_idx2 = src_indices[2] ? src_indices[2][k] : k;
        int dst_idx2 = dst_indices[2] ? dst_indices[2][k] : k;
        const char* p = (const char*)src + src_idx0 * src_strides[0]
                                         + src_idx1 * src_strides[1]
                                         + src_idx2 * src_strides[2];
        char* q = (char*)dst + dst_idx0 * dst_strides[0]
                             + dst_idx1 * dst_strides[1]
                             + dst_idx2 * dst_strides[2];
        memcpy(q, p, elem_size);
    }
}

static __global__ void array_copy_4d_kernel(void* dst, const void* src,
                                        wp::vec_t<4, int> dst_strides, wp::vec_t<4, int> src_strides,
                                        wp::vec_t<4, const int*> dst_indices, wp::vec_t<4, const int*> src_indices,
                                        wp::vec_t<4, int> shape, int elem_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = shape[1];
    int o = shape[2];
    int p = shape[3];
    int i = tid / (n * o * p);
    int j = tid % (n * o * p) / (o * p);
    int k = tid % (o * p) / p;
    int l = tid % p;
    if (i < shape[0] && j < shape[1] && k < shape[2] /*&& l < shape[3]*/)
    {
        int src_idx0 = src_indices[0] ? src_indices[0][i] : i;
        int dst_idx0 = dst_indices[0] ? dst_indices[0][i] : i;
        int src_idx1 = src_indices[1] ? src_indices[1][j] : j;
        int dst_idx1 = dst_indices[1] ? dst_indices[1][j] : j;
        int src_idx2 = src_indices[2] ? src_indices[2][k] : k;
        int dst_idx2 = dst_indices[2] ? dst_indices[2][k] : k;
        int src_idx3 = src_indices[3] ? src_indices[3][l] : l;
        int dst_idx3 = dst_indices[3] ? dst_indices[3][l] : l;
        const char* p = (const char*)src + src_idx0 * src_strides[0]
                                         + src_idx1 * src_strides[1]
                                         + src_idx2 * src_strides[2]
                                         + src_idx3 * src_strides[3];
        char* q = (char*)dst + dst_idx0 * dst_strides[0]
                             + dst_idx1 * dst_strides[1]
                             + dst_idx2 * dst_strides[2]
                             + dst_idx3 * dst_strides[3];
        memcpy(q, p, elem_size);
    }
}


static __global__ void array_copy_from_fabric_kernel(wp::fabricarray_t<void> src,
                                                     void* dst_data, int dst_stride, const int* dst_indices,
                                                     int elem_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < src.size)
    {
        int dst_idx = dst_indices ? dst_indices[tid] : tid;
        void* dst_ptr = (char*)dst_data + dst_idx * dst_stride;
        const void* src_ptr = fabricarray_element_ptr(src, tid, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}

static __global__ void array_copy_from_fabric_indexed_kernel(wp::indexedfabricarray_t<void> src,
                                                             void* dst_data, int dst_stride, const int* dst_indices,
                                                             int elem_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < src.size)
    {
        int src_index = src.indices[tid];
        int dst_idx = dst_indices ? dst_indices[tid] : tid;
        void* dst_ptr = (char*)dst_data + dst_idx * dst_stride;
        const void* src_ptr = fabricarray_element_ptr(src.fa, src_index, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}

static __global__ void array_copy_to_fabric_kernel(wp::fabricarray_t<void> dst,
                                                   const void* src_data, int src_stride, const int* src_indices,
                                                   int elem_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dst.size)
    {
        int src_idx = src_indices ? src_indices[tid] : tid;
        const void* src_ptr = (const char*)src_data + src_idx * src_stride;
        void* dst_ptr = fabricarray_element_ptr(dst, tid, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}

static __global__ void array_copy_to_fabric_indexed_kernel(wp::indexedfabricarray_t<void> dst,
                                                           const void* src_data, int src_stride, const int* src_indices,
                                                           int elem_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dst.size)
    {
        int src_idx = src_indices ? src_indices[tid] : tid;
        const void* src_ptr = (const char*)src_data + src_idx * src_stride;
        int dst_idx = dst.indices[tid];
        void* dst_ptr = fabricarray_element_ptr(dst.fa, dst_idx, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}


static __global__ void array_copy_fabric_to_fabric_kernel(wp::fabricarray_t<void> dst, wp::fabricarray_t<void> src, int elem_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dst.size)
    {
        const void* src_ptr = fabricarray_element_ptr(src, tid, elem_size);
        void* dst_ptr = fabricarray_element_ptr(dst, tid, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}


static __global__ void array_copy_fabric_to_fabric_indexed_kernel(wp::indexedfabricarray_t<void> dst, wp::fabricarray_t<void> src, int elem_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dst.size)
    {
        const void* src_ptr = fabricarray_element_ptr(src, tid, elem_size);
        int dst_index = dst.indices[tid];
        void* dst_ptr = fabricarray_element_ptr(dst.fa, dst_index, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}


static __global__ void array_copy_fabric_indexed_to_fabric_kernel(wp::fabricarray_t<void> dst, wp::indexedfabricarray_t<void> src, int elem_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dst.size)
    {
        int src_index = src.indices[tid];
        const void* src_ptr = fabricarray_element_ptr(src.fa, src_index, elem_size);
        void* dst_ptr = fabricarray_element_ptr(dst, tid, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}


static __global__ void array_copy_fabric_indexed_to_fabric_indexed_kernel(wp::indexedfabricarray_t<void> dst, wp::indexedfabricarray_t<void> src, int elem_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dst.size)
    {
        int src_index = src.indices[tid];
        int dst_index = dst.indices[tid];
        const void* src_ptr = fabricarray_element_ptr(src.fa, src_index, elem_size);
        void* dst_ptr = fabricarray_element_ptr(dst.fa, dst_index, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}


WP_API size_t array_copy_device(void* context, void* dst, void* src, int dst_type, int src_type, int elem_size)
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

    const wp::fabricarray_t<void>* src_fabricarray = NULL;
    wp::fabricarray_t<void>* dst_fabricarray = NULL;

    const wp::indexedfabricarray_t<void>* src_indexedfabricarray = NULL;
    wp::indexedfabricarray_t<void>* dst_indexedfabricarray = NULL;

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
    else if (src_type == wp::ARRAY_TYPE_FABRIC)
    {
        src_fabricarray = static_cast<const wp::fabricarray_t<void>*>(src);
        src_ndim = 1;
    }
    else if (src_type == wp::ARRAY_TYPE_FABRIC_INDEXED)
    {
        src_indexedfabricarray = static_cast<const wp::indexedfabricarray_t<void>*>(src);
        src_ndim = 1;
    }
    else
    {
        fprintf(stderr, "Warp copy error: Invalid array type (%d)\n", src_type);
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
    else if (dst_type == wp::ARRAY_TYPE_FABRIC)
    {
        dst_fabricarray = static_cast<wp::fabricarray_t<void>*>(dst);
        dst_ndim = 1;
    }
    else if (dst_type == wp::ARRAY_TYPE_FABRIC_INDEXED)
    {
        dst_indexedfabricarray = static_cast<wp::indexedfabricarray_t<void>*>(dst);
        dst_ndim = 1;
    }
    else
    {
        fprintf(stderr, "Warp copy error: Invalid array type (%d)\n", dst_type);
        return 0;
    }

    if (src_ndim != dst_ndim)
    {
        fprintf(stderr, "Warp copy error: Incompatible array dimensionalities (%d and %d)\n", src_ndim, dst_ndim);
        return 0;
    }

    ContextGuard guard(context);

    // handle fabric arrays
    if (dst_fabricarray)
    {
        size_t n = dst_fabricarray->size;
        if (src_fabricarray)
        {
            // copy from fabric to fabric
            if (src_fabricarray->size != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return 0;
            }
            wp_launch_device(WP_CURRENT_CONTEXT, array_copy_fabric_to_fabric_kernel, n,
                            (*dst_fabricarray, *src_fabricarray, elem_size));
            return n;
        }
        else if (src_indexedfabricarray)
        {
            // copy from fabric indexed to fabric
            if (src_indexedfabricarray->size != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return 0;
            }
            wp_launch_device(WP_CURRENT_CONTEXT, array_copy_fabric_indexed_to_fabric_kernel, n,
                            (*dst_fabricarray, *src_indexedfabricarray, elem_size));
            return n;
        }
        else
        {
            // copy to fabric
            if (size_t(src_shape[0]) != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return 0;
            }
            wp_launch_device(WP_CURRENT_CONTEXT, array_copy_to_fabric_kernel, n,
                            (*dst_fabricarray, src_data, src_strides[0], src_indices[0], elem_size));
            return n;
        }
    }
    if (dst_indexedfabricarray)
    {
        size_t n = dst_indexedfabricarray->size;
        if (src_fabricarray)
        {
            // copy from fabric to fabric indexed
            if (src_fabricarray->size != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return 0;
            }
            wp_launch_device(WP_CURRENT_CONTEXT, array_copy_fabric_to_fabric_indexed_kernel, n,
                            (*dst_indexedfabricarray, *src_fabricarray, elem_size));
            return n;
        }
        else if (src_indexedfabricarray)
        {
            // copy from fabric indexed to fabric indexed
            if (src_indexedfabricarray->size != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return 0;
            }
            wp_launch_device(WP_CURRENT_CONTEXT, array_copy_fabric_indexed_to_fabric_indexed_kernel, n,
                            (*dst_indexedfabricarray, *src_indexedfabricarray, elem_size));
            return n;
        }
        else
        {
            // copy to fabric indexed
            if (size_t(src_shape[0]) != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return 0;
            }
            wp_launch_device(WP_CURRENT_CONTEXT, array_copy_to_fabric_indexed_kernel, n,
                             (*dst_indexedfabricarray, src_data, src_strides[0], src_indices[0], elem_size));
            return n;
        }
    }
    else if (src_fabricarray)
    {
        // copy from fabric
        size_t n = src_fabricarray->size;
        if (size_t(dst_shape[0]) != n)
        {
            fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
            return 0;
        }
        wp_launch_device(WP_CURRENT_CONTEXT, array_copy_from_fabric_kernel, n,
                         (*src_fabricarray, dst_data, dst_strides[0], dst_indices[0], elem_size));
        return n;
    }
    else if (src_indexedfabricarray)
    {
        // copy from fabric indexed
        size_t n = src_indexedfabricarray->size;
        if (size_t(dst_shape[0]) != n)
        {
            fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
            return 0;
        }
        wp_launch_device(WP_CURRENT_CONTEXT, array_copy_from_fabric_indexed_kernel, n,
                         (*src_indexedfabricarray, dst_data, dst_strides[0], dst_indices[0], elem_size));
        return n;
    }

    size_t n = 1;
    for (int i = 0; i < src_ndim; i++)
    {
        if (src_shape[i] != dst_shape[i])
        {
            fprintf(stderr, "Warp copy error: Incompatible array shapes\n");
            return 0;
        }
        n *= src_shape[i];
    }

    switch (src_ndim)
    {
    case 1:
    {
        wp_launch_device(WP_CURRENT_CONTEXT, array_copy_1d_kernel, n, (dst_data, src_data,
                                                                   dst_strides[0], src_strides[0],
                                                                   dst_indices[0], src_indices[0],
                                                                   src_shape[0], elem_size));
        break;
    }
    case 2:
    {
        wp::vec_t<2, int> shape_v(src_shape[0], src_shape[1]);
        wp::vec_t<2, int> src_strides_v(src_strides[0], src_strides[1]);
        wp::vec_t<2, int> dst_strides_v(dst_strides[0], dst_strides[1]);
        wp::vec_t<2, const int*> src_indices_v(src_indices[0], src_indices[1]);
        wp::vec_t<2, const int*> dst_indices_v(dst_indices[0], dst_indices[1]);

        wp_launch_device(WP_CURRENT_CONTEXT, array_copy_2d_kernel, n, (dst_data, src_data,
                                                                   dst_strides_v, src_strides_v,
                                                                   dst_indices_v, src_indices_v,
                                                                   shape_v, elem_size));
        break;
    }
    case 3:
    {
        wp::vec_t<3, int> shape_v(src_shape[0], src_shape[1], src_shape[2]);
        wp::vec_t<3, int> src_strides_v(src_strides[0], src_strides[1], src_strides[2]);
        wp::vec_t<3, int> dst_strides_v(dst_strides[0], dst_strides[1], dst_strides[2]);
        wp::vec_t<3, const int*> src_indices_v(src_indices[0], src_indices[1], src_indices[2]);
        wp::vec_t<3, const int*> dst_indices_v(dst_indices[0], dst_indices[1], dst_indices[2]);

        wp_launch_device(WP_CURRENT_CONTEXT, array_copy_3d_kernel, n, (dst_data, src_data,
                                                                   dst_strides_v, src_strides_v,
                                                                   dst_indices_v, src_indices_v,
                                                                   shape_v, elem_size));
        break;
    }
    case 4:
    {
        wp::vec_t<4, int> shape_v(src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
        wp::vec_t<4, int> src_strides_v(src_strides[0], src_strides[1], src_strides[2], src_strides[3]);
        wp::vec_t<4, int> dst_strides_v(dst_strides[0], dst_strides[1], dst_strides[2], dst_strides[3]);
        wp::vec_t<4, const int*> src_indices_v(src_indices[0], src_indices[1], src_indices[2], src_indices[3]);
        wp::vec_t<4, const int*> dst_indices_v(dst_indices[0], dst_indices[1], dst_indices[2], dst_indices[3]);

        wp_launch_device(WP_CURRENT_CONTEXT, array_copy_4d_kernel, n, (dst_data, src_data,
                                                                   dst_strides_v, src_strides_v,
                                                                   dst_indices_v, src_indices_v,
                                                                   shape_v, elem_size));
        break;
    }
    default:
        fprintf(stderr, "Warp copy error: invalid array dimensionality (%d)\n", src_ndim);
        return 0;
    }

    if (check_cuda(cudaGetLastError()))
        return n;
    else
        return 0;
}


static __global__ void array_fill_1d_kernel(void* data,
                                            int n,
                                            int stride,
                                            const int* indices,
                                            const void* value,
                                            int value_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int idx = indices ? indices[i] : i;
        char* p = (char*)data + idx * stride;
        memcpy(p, value, value_size);
    }
}

static __global__ void array_fill_2d_kernel(void* data,
                                            wp::vec_t<2, int> shape,
                                            wp::vec_t<2, int> strides,
                                            wp::vec_t<2, const int*> indices,
                                            const void* value,
                                            int value_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = shape[1];
    int i = tid / n;
    int j = tid % n;
    if (i < shape[0] /*&& j < shape[1]*/)
    {
        int idx0 = indices[0] ? indices[0][i] : i;
        int idx1 = indices[1] ? indices[1][j] : j;
        char* p = (char*)data + idx0 * strides[0] + idx1 * strides[1];
        memcpy(p, value, value_size);
    }
}

static __global__ void array_fill_3d_kernel(void* data,
                                            wp::vec_t<3, int> shape,
                                            wp::vec_t<3, int> strides,
                                            wp::vec_t<3, const int*> indices,
                                            const void* value,
                                            int value_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = shape[1];
    int o = shape[2];
    int i = tid / (n * o);
    int j = tid % (n * o) / o;
    int k = tid % o;
    if (i < shape[0] && j < shape[1] /*&& k < shape[2]*/)
    {
        int idx0 = indices[0] ? indices[0][i] : i;
        int idx1 = indices[1] ? indices[1][j] : j;
        int idx2 = indices[2] ? indices[2][k] : k;
        char* p = (char*)data + idx0 * strides[0] + idx1 * strides[1] + idx2 * strides[2];
        memcpy(p, value, value_size);
    }
}

static __global__ void array_fill_4d_kernel(void* data,
                                            wp::vec_t<4, int> shape,
                                            wp::vec_t<4, int> strides,
                                            wp::vec_t<4, const int*> indices,
                                            const void* value,
                                            int value_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = shape[1];
    int o = shape[2];
    int p = shape[3];
    int i = tid / (n * o * p);
    int j = tid % (n * o * p) / (o * p);
    int k = tid % (o * p) / p;
    int l = tid % p;
    if (i < shape[0] && j < shape[1] && k < shape[2] /*&& l < shape[3]*/)
    {
        int idx0 = indices[0] ? indices[0][i] : i;
        int idx1 = indices[1] ? indices[1][j] : j;
        int idx2 = indices[2] ? indices[2][k] : k;
        int idx3 = indices[3] ? indices[3][l] : l;
        char* p = (char*)data + idx0 * strides[0] + idx1 * strides[1] + idx2 * strides[2] + idx3 * strides[3];
        memcpy(p, value, value_size);
    }
}


static __global__ void array_fill_fabric_kernel(wp::fabricarray_t<void> fa, const void* value, int value_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < fa.size)
    {
        void* dst_ptr = fabricarray_element_ptr(fa, tid, value_size);
        memcpy(dst_ptr, value, value_size);
    }
}


static __global__ void array_fill_fabric_indexed_kernel(wp::indexedfabricarray_t<void> ifa, const void* value, int value_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < ifa.size)
    {
        size_t idx = size_t(ifa.indices[tid]);
        if (idx < ifa.fa.size)
        {
            void* dst_ptr = fabricarray_element_ptr(ifa.fa, idx, value_size);
            memcpy(dst_ptr, value, value_size);
        }
    }
}


WP_API void array_fill_device(void* context, void* arr_ptr, int arr_type, const void* value_ptr, int value_size)
{
    if (!arr_ptr || !value_ptr)
        return;

    void* data = NULL;
    int ndim = 0;
    const int* shape = NULL;
    const int* strides = NULL;
    const int*const* indices = NULL;

    wp::fabricarray_t<void>* fa = NULL;
    wp::indexedfabricarray_t<void>* ifa = NULL;

    const int* null_indices[wp::ARRAY_MAX_DIMS] = { NULL };

    if (arr_type == wp::ARRAY_TYPE_REGULAR)
    {
        wp::array_t<void>& arr = *static_cast<wp::array_t<void>*>(arr_ptr);
        data = arr.data;
        ndim = arr.ndim;
        shape = arr.shape.dims;
        strides = arr.strides;
        indices = null_indices;
    }
    else if (arr_type == wp::ARRAY_TYPE_INDEXED)
    {
        wp::indexedarray_t<void>& ia = *static_cast<wp::indexedarray_t<void>*>(arr_ptr);
        data = ia.arr.data;
        ndim = ia.arr.ndim;
        shape = ia.shape.dims;
        strides = ia.arr.strides;
        indices = ia.indices;
    }
    else if (arr_type == wp::ARRAY_TYPE_FABRIC)
    {
        fa = static_cast<wp::fabricarray_t<void>*>(arr_ptr);
    }
    else if (arr_type == wp::ARRAY_TYPE_FABRIC_INDEXED)
    {
        ifa = static_cast<wp::indexedfabricarray_t<void>*>(arr_ptr);
    }
    else
    {
        fprintf(stderr, "Warp fill error: Invalid array type id %d\n", arr_type);
        return;
    }

    size_t n = 1;
    for (int i = 0; i < ndim; i++)
        n *= shape[i];

    ContextGuard guard(context);

    // copy value to device memory
    void* value_devptr;
    check_cuda(cudaMalloc(&value_devptr, value_size));
    check_cuda(cudaMemcpyAsync(value_devptr, value_ptr, value_size, cudaMemcpyHostToDevice, get_current_stream()));

    // handle fabric arrays
    if (fa)
    {
        wp_launch_device(WP_CURRENT_CONTEXT, array_fill_fabric_kernel, n,
                         (*fa, value_devptr, value_size));
        return;
    }
    else if (ifa)
    {
        wp_launch_device(WP_CURRENT_CONTEXT, array_fill_fabric_indexed_kernel, n,
                         (*ifa, value_devptr, value_size));
        return;
    }

    // handle regular or indexed arrays
    switch (ndim)
    {
    case 1:
    {
        wp_launch_device(WP_CURRENT_CONTEXT, array_fill_1d_kernel, n,
                         (data, shape[0], strides[0], indices[0], value_devptr, value_size));
        break;
    }
    case 2:
    {
        wp::vec_t<2, int> shape_v(shape[0], shape[1]);
        wp::vec_t<2, int> strides_v(strides[0], strides[1]);
        wp::vec_t<2, const int*> indices_v(indices[0], indices[1]);
        wp_launch_device(WP_CURRENT_CONTEXT, array_fill_2d_kernel, n,
                         (data, shape_v, strides_v, indices_v, value_devptr, value_size));
        break;
    }
    case 3:
    {
        wp::vec_t<3, int> shape_v(shape[0], shape[1], shape[2]);
        wp::vec_t<3, int> strides_v(strides[0], strides[1], strides[2]);
        wp::vec_t<3, const int*> indices_v(indices[0], indices[1], indices[2]);
        wp_launch_device(WP_CURRENT_CONTEXT, array_fill_3d_kernel, n,
                         (data, shape_v, strides_v, indices_v, value_devptr, value_size));
        break;
    }
    case 4:
    {
        wp::vec_t<4, int> shape_v(shape[0], shape[1], shape[2], shape[3]);
        wp::vec_t<4, int> strides_v(strides[0], strides[1], strides[2], strides[3]);
        wp::vec_t<4, const int*> indices_v(indices[0], indices[1], indices[2], indices[3]);
        wp_launch_device(WP_CURRENT_CONTEXT, array_fill_4d_kernel, n,
                         (data, shape_v, strides_v, indices_v, value_devptr, value_size));
        break;
    }
    default:
        fprintf(stderr, "Warp fill error: invalid array dimensionality (%d)\n", ndim);
        return;
    }
}

void array_scan_int_device(uint64_t in, uint64_t out, int len, bool inclusive)
{
    scan_device((const int*)in, (int*)out, len, inclusive);
}

void array_scan_float_device(uint64_t in, uint64_t out, int len, bool inclusive)
{
    scan_device((const float*)in, (float*)out, len, inclusive);
}

int cuda_driver_version()
{
    int version;
    if (check_cu(cuDriverGetVersion_f(&version)))
        return version;
    else
        return 0;
}

int cuda_toolkit_version()
{
    return CUDA_VERSION;
}

bool cuda_driver_is_initialized()
{
    return is_cuda_driver_initialized();
}

int nvrtc_supported_arch_count()
{
    int count;
    if (check_nvrtc(nvrtcGetNumSupportedArchs(&count)))
        return count;
    else
        return 0;
}

void nvrtc_supported_archs(int* archs)
{
    if (archs)
    {
        check_nvrtc(nvrtcGetSupportedArchs(archs));
    }
}

int cuda_device_get_count()
{
    int count = 0;
    check_cu(cuDeviceGetCount_f(&count));
    return count;
}

void* cuda_device_primary_context_retain(int ordinal)
{
    CUcontext context = NULL;
    CUdevice device;
    if (check_cu(cuDeviceGet_f(&device, ordinal)))
        check_cu(cuDevicePrimaryCtxRetain_f(&context, device));
    return context;
}

void cuda_device_primary_context_release(int ordinal)
{
    CUdevice device;
    if (check_cu(cuDeviceGet_f(&device, ordinal)))
        check_cu(cuDevicePrimaryCtxRelease_f(device));
}

const char* cuda_device_get_name(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].name;
    return NULL;
}

int cuda_device_get_arch(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].arch;
    return 0;
}

int cuda_device_is_uva(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].is_uva;
    return 0;
}

int cuda_device_is_memory_pool_supported(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].is_memory_pool_supported;
    return false;
}

void* cuda_context_get_current()
{
    return get_current_context();
}

void cuda_context_set_current(void* context)
{
    CUcontext ctx = static_cast<CUcontext>(context);
    CUcontext prev_ctx = NULL;
    check_cu(cuCtxGetCurrent_f(&prev_ctx));
    if (ctx != prev_ctx)
    {
        check_cu(cuCtxSetCurrent_f(ctx));
    }
}

void cuda_context_push_current(void* context)
{
    check_cu(cuCtxPushCurrent_f(static_cast<CUcontext>(context)));
}

void cuda_context_pop_current()
{
    CUcontext context;
    check_cu(cuCtxPopCurrent_f(&context));
}

void* cuda_context_create(int device_ordinal)
{
    CUcontext ctx = NULL;
    CUdevice device;
    if (check_cu(cuDeviceGet_f(&device, device_ordinal)))
        check_cu(cuCtxCreate_f(&ctx, 0, device));
    return ctx;
}

void cuda_context_destroy(void* context)
{
    if (context)
    {
        CUcontext ctx = static_cast<CUcontext>(context);

        // ensure this is not the current context
        if (ctx == cuda_context_get_current())
            cuda_context_set_current(NULL);

        // release the cached info about this context
        ContextInfo* info = get_context_info(ctx);
        if (info)
        {
            if (info->stream)
                check_cu(cuStreamDestroy_f(info->stream));
            
            g_contexts.erase(ctx);
        }

        check_cu(cuCtxDestroy_f(ctx));
    }
}

void cuda_context_synchronize(void* context)
{
    ContextGuard guard(context);

    check_cu(cuCtxSynchronize_f());
}

uint64_t cuda_context_check(void* context)
{
    ContextGuard guard(context);

    cudaStreamCaptureStatus status;
    cudaStreamIsCapturing(get_current_stream(), &status);
    
    // do not check during cuda stream capture
    // since we cannot synchronize the device
    if (status == cudaStreamCaptureStatusNone)
    {
        cudaDeviceSynchronize();
        return cudaPeekAtLastError(); 
    }
    else
    {
        return 0;
    }
}


int cuda_context_get_device_ordinal(void* context)
{
    ContextInfo* info = get_context_info(static_cast<CUcontext>(context));
    return info && info->device_info ? info->device_info->ordinal : -1;
}

int cuda_context_is_primary(void* context)
{
    int ordinal = cuda_context_get_device_ordinal(context);
    if (ordinal != -1)
    {
        // there is no CUDA API to check if a context is primary, but we can temporarily
        // acquire the device's primary context to check the pointer
        void* device_primary_context = cuda_device_primary_context_retain(ordinal);
        cuda_device_primary_context_release(ordinal);
        return int(context == device_primary_context);
    }
    return 0;
}

int cuda_context_is_memory_pool_supported(void* context)
{
    int ordinal = cuda_context_get_device_ordinal(context);
    if (ordinal != -1)
    {
        return cuda_device_is_memory_pool_supported(ordinal);
    }
    return 0;
}

void* cuda_context_get_stream(void* context)
{
    ContextInfo* info = get_context_info(static_cast<CUcontext>(context));
    if (info)
    {
        return info->stream;
    }
    return NULL;
}

void cuda_context_set_stream(void* context, void* stream)
{
    ContextInfo* info = get_context_info(static_cast<CUcontext>(context));
    if (info)
    {
        info->stream = static_cast<CUstream>(stream);
    }
}

int cuda_context_enable_peer_access(void* context, void* peer_context)
{
    if (!context || !peer_context)
    {
        fprintf(stderr, "Warp error: Failed to enable peer access: invalid argument\n");
        return 0;
    }

    if (context == peer_context)
        return 1;  // ok

    CUcontext ctx = static_cast<CUcontext>(context);
    CUcontext peer_ctx = static_cast<CUcontext>(peer_context);

    ContextInfo* info = get_context_info(ctx);
    ContextInfo* peer_info = get_context_info(peer_ctx);
    if (!info || !peer_info)
    {
        fprintf(stderr, "Warp error: Failed to enable peer access: failed to get context info\n");
        return 0;
    }

    // check if same device
    if (info->device_info == peer_info->device_info)
    {
        if (info->device_info->is_uva)
        {
            return 1;  // ok
        }
        else
        {
            fprintf(stderr, "Warp error: Failed to enable peer access: device doesn't support UVA\n");
            return 0;
        }
    }
    else
    {
        // different devices, try to enable
        ContextGuard guard(ctx, true);
        CUresult result = cuCtxEnablePeerAccess_f(peer_ctx, 0);
        if (result == CUDA_SUCCESS || result == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
        {
            return 1;  // ok
        }
        else
        {
            check_cu(result);
            return 0;
        }
    }
}

int cuda_context_can_access_peer(void* context, void* peer_context)
{
    if (!context || !peer_context)
        return 0;

    if (context == peer_context)
        return 1;

    CUcontext ctx = static_cast<CUcontext>(context);
    CUcontext peer_ctx = static_cast<CUcontext>(peer_context);
    
    ContextInfo* info = get_context_info(ctx);
    ContextInfo* peer_info = get_context_info(peer_ctx);
    if (!info || !peer_info)
        return 0;

    // check if same device
    if (info->device_info == peer_info->device_info)
    {
        if (info->device_info->is_uva)
            return 1;
        else
            return 0;
    }
    else
    {
        // different devices, try to enable
        // TODO: is there a better way to check?
        ContextGuard guard(ctx, true);
        CUresult result = cuCtxEnablePeerAccess_f(peer_ctx, 0);
        if (result == CUDA_SUCCESS || result == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
            return 1;
        else
            return 0;
    }
}

void* cuda_stream_create(void* context)
{
    CUcontext ctx = context ? static_cast<CUcontext>(context) : get_current_context();
    if (!ctx)
        return NULL;

    ContextGuard guard(context, true);

    CUstream stream;
    if (check_cu(cuStreamCreate_f(&stream, CU_STREAM_DEFAULT)))
        return stream;
    else
        return NULL;
}

void cuda_stream_destroy(void* context, void* stream)
{
    if (!stream)
        return;

    CUcontext ctx = context ? static_cast<CUcontext>(context) : get_current_context();
    if (!ctx)
        return;

    ContextGuard guard(context, true);

    check_cu(cuStreamDestroy_f(static_cast<CUstream>(stream)));
}

void cuda_stream_synchronize(void* context, void* stream)
{
    ContextGuard guard(context);

    check_cu(cuStreamSynchronize_f(static_cast<CUstream>(stream)));
}

void* cuda_stream_get_current()
{
    return get_current_stream();
}

void cuda_stream_wait_event(void* context, void* stream, void* event)
{
    ContextGuard guard(context);

    check_cu(cuStreamWaitEvent_f(static_cast<CUstream>(stream), static_cast<CUevent>(event), 0));
}

void cuda_stream_wait_stream(void* context, void* stream, void* other_stream, void* event)
{
    ContextGuard guard(context);

    check_cu(cuEventRecord_f(static_cast<CUevent>(event), static_cast<CUstream>(other_stream)));
    check_cu(cuStreamWaitEvent_f(static_cast<CUstream>(stream), static_cast<CUevent>(event), 0));
}

void* cuda_event_create(void* context, unsigned flags)
{
    ContextGuard guard(context);

    CUevent event;
    if (check_cu(cuEventCreate_f(&event, flags)))
        return event;
    else
        return NULL;
}

void cuda_event_destroy(void* context, void* event)
{
    ContextGuard guard(context, true);

    check_cu(cuEventDestroy_f(static_cast<CUevent>(event)));
}

void cuda_event_record(void* context, void* event, void* stream)
{
    ContextGuard guard(context);

    check_cu(cuEventRecord_f(static_cast<CUevent>(event), static_cast<CUstream>(stream)));
}

void cuda_graph_begin_capture(void* context)
{
    ContextGuard guard(context);

    check_cuda(cudaStreamBeginCapture(get_current_stream(), cudaStreamCaptureModeGlobal));
}

void* cuda_graph_end_capture(void* context)
{
    ContextGuard guard(context);

    cudaGraph_t graph = NULL;
    check_cuda(cudaStreamEndCapture(get_current_stream(), &graph));

    if (graph)
    {
        // enable to create debug GraphVis visualization of graph
        //cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose);

        cudaGraphExec_t graph_exec = NULL;
        //check_cuda(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
        
        // can use after CUDA 11.4 to permit graphs to capture cudaMallocAsync() operations
        check_cuda(cudaGraphInstantiateWithFlags(&graph_exec, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch));

        // free source graph
        check_cuda(cudaGraphDestroy(graph));

        return graph_exec;
    }
    else
    {
        return NULL;
    }
}

void cuda_graph_launch(void* context, void* graph_exec)
{
    ContextGuard guard(context);

    check_cuda(cudaGraphLaunch((cudaGraphExec_t)graph_exec, get_current_stream()));
}

void cuda_graph_destroy(void* context, void* graph_exec)
{
    ContextGuard guard(context);

    check_cuda(cudaGraphExecDestroy((cudaGraphExec_t)graph_exec));
}

size_t cuda_compile_program(const char* cuda_src, int arch, const char* include_dir, bool debug, bool verbose, bool verify_fp, bool fast_math, const char* output_path)
{
    // use file extension to determine whether to output PTX or CUBIN
    const char* output_ext = strrchr(output_path, '.');
    bool use_ptx = output_ext && strcmp(output_ext + 1, "ptx") == 0;

    // check include dir path len (path + option)
    const int max_path = 4096 + 16;
    if (strlen(include_dir) > max_path)
    {
        fprintf(stderr, "Warp error: Include path too long\n");
        return size_t(-1);
    }

    char include_opt[max_path];
    strcpy(include_opt, "--include-path=");
    strcat(include_opt, include_dir);

    const int max_arch = 128;
    char arch_opt[max_arch];

    if (use_ptx)
        snprintf(arch_opt, max_arch, "--gpu-architecture=compute_%d", arch);
    else
        snprintf(arch_opt, max_arch, "--gpu-architecture=sm_%d", arch);

    std::vector<const char*> opts;
    opts.push_back(arch_opt);
    opts.push_back(include_opt);
    opts.push_back("--std=c++11");
    
    if (debug)
    {
        opts.push_back("--define-macro=_DEBUG");
        opts.push_back("--generate-line-info");
        // disabling since it causes issues with `Unresolved extern function 'cudaGetParameterBufferV2'
        //opts.push_back("--device-debug");
    }
    else
        opts.push_back("--define-macro=NDEBUG");

    if (verify_fp)
        opts.push_back("--define-macro=WP_VERIFY_FP");
    else
        opts.push_back("--undefine-macro=WP_VERIFY_FP");
    
    if (fast_math)
        opts.push_back("--use_fast_math");


    nvrtcProgram prog;
    nvrtcResult res;

    res = nvrtcCreateProgram(
        &prog,         // prog
        cuda_src,      // buffer
        NULL,          // name
        0,             // numHeaders
        NULL,          // headers
        NULL);         // includeNames

    if (!check_nvrtc(res))
        return size_t(res);

    res = nvrtcCompileProgram(prog, int(opts.size()), opts.data());

    if (!check_nvrtc(res) || verbose)
    {
        // get program log
        size_t log_size;
        if (check_nvrtc(nvrtcGetProgramLogSize(prog, &log_size)))
        {
            std::vector<char> log(log_size);
            if (check_nvrtc(nvrtcGetProgramLog(prog, log.data())))
            {
                // todo: figure out better way to return this to python
                if (res != NVRTC_SUCCESS)
                    fprintf(stderr, "%s", log.data());
                else
                    fprintf(stdout, "%s", log.data());
            }
        }

        if (res != NVRTC_SUCCESS)
        {
            nvrtcDestroyProgram(&prog);
            return size_t(res);
        }
    }

    nvrtcResult (*get_output_size)(nvrtcProgram, size_t*);
    nvrtcResult (*get_output_data)(nvrtcProgram, char*);
    const char* output_mode;
    if (use_ptx)
    {
        get_output_size = nvrtcGetPTXSize;
        get_output_data = nvrtcGetPTX;
        output_mode = "wt";
    }
    else
    {
        get_output_size = nvrtcGetCUBINSize;
        get_output_data = nvrtcGetCUBIN;
        output_mode = "wb";
    }

    // save output
    size_t output_size;
    res = get_output_size(prog, &output_size);
    if (check_nvrtc(res))
    {
        std::vector<char> output(output_size);
        res = get_output_data(prog, output.data());
        if (check_nvrtc(res))
        {
            FILE* file = fopen(output_path, output_mode);
            if (file)
            {
                if (fwrite(output.data(), 1, output_size, file) != output_size)
                {
                    fprintf(stderr, "Warp error: Failed to write output file '%s'\n", output_path);
                    res = nvrtcResult(-1);
                }
                fclose(file);
            }
            else
            {
                fprintf(stderr, "Warp error: Failed to open output file '%s'\n", output_path);
                res = nvrtcResult(-1);
            }
        }
    }

    check_nvrtc(nvrtcDestroyProgram(&prog));

    return res;
}

void* cuda_load_module(void* context, const char* path)
{
    ContextGuard guard(context);

    // use file extension to determine whether to load PTX or CUBIN
    const char* input_ext = strrchr(path, '.');
    bool load_ptx = input_ext && strcmp(input_ext + 1, "ptx") == 0;

    std::vector<char> input;

    FILE* file = fopen(path, "rb");
    if (file)
    {
        fseek(file, 0, SEEK_END);
        size_t length = ftell(file);
        fseek(file, 0, SEEK_SET);

        input.resize(length + 1);
        if (fread(input.data(), 1, length, file) != length)
        {
            fprintf(stderr, "Warp error: Failed to read input file '%s'\n", path);
            fclose(file);
            return NULL;
        }
        fclose(file);

        input[length] = '\0';
    }
    else
    {
        fprintf(stderr, "Warp error: Failed to open input file '%s'\n", path);
        return NULL;
    }

    int driver_cuda_version = 0;
    CUmodule module = NULL;

    if (load_ptx)
    {
        if (check_cu(cuDriverGetVersion_f(&driver_cuda_version)) && driver_cuda_version >= CUDA_VERSION)
        {
            // let the driver compile the PTX

            CUjit_option options[2];
            void *option_vals[2];
            char error_log[8192] = "";
            unsigned int log_size = 8192;
            // Set up loader options
            // Pass a buffer for error message
            options[0] = CU_JIT_ERROR_LOG_BUFFER;
            option_vals[0] = (void*)error_log;
            // Pass the size of the error buffer
            options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
            option_vals[1] = (void*)(size_t)log_size;

            if (!check_cu(cuModuleLoadDataEx_f(&module, input.data(), 2, options, option_vals)))
            {
                fprintf(stderr, "Warp error: Loading PTX module failed\n");
                // print error log if not empty
                if (*error_log)
                    fprintf(stderr, "PTX loader error:\n%s\n", error_log);
                return NULL;
            }
        }
        else
        {
            // manually compile the PTX and load as CUBIN

            ContextInfo* context_info = get_context_info(static_cast<CUcontext>(context));
            if (!context_info || !context_info->device_info)
            {
                fprintf(stderr, "Warp error: Failed to determine target architecture\n");
                return NULL;
            }

            int arch = context_info->device_info->arch;

            char arch_opt[128];
            sprintf(arch_opt, "--gpu-name=sm_%d", arch);

            const char* compiler_options[] = { arch_opt };

            nvPTXCompilerHandle compiler = NULL;
            if (!check_nvptx(nvPTXCompilerCreate(&compiler, input.size(), input.data())))
                return NULL;

            if (!check_nvptx(nvPTXCompilerCompile(compiler, sizeof(compiler_options) / sizeof(*compiler_options), compiler_options)))
                return NULL;

            size_t cubin_size = 0;
            if (!check_nvptx(nvPTXCompilerGetCompiledProgramSize(compiler, &cubin_size)))
                return NULL;

            std::vector<char> cubin(cubin_size);
            if (!check_nvptx(nvPTXCompilerGetCompiledProgram(compiler, cubin.data())))
                return NULL;

            check_nvptx(nvPTXCompilerDestroy(&compiler));

            if (!check_cu(cuModuleLoadDataEx_f(&module, cubin.data(), 0, NULL, NULL)))
            {
                fprintf(stderr, "Warp CUDA error: Loading module failed\n");
                return NULL;
            }
        }
    }
    else
    {
        // load CUBIN
        if (!check_cu(cuModuleLoadDataEx_f(&module, input.data(), 0, NULL, NULL)))
        {
            fprintf(stderr, "Warp CUDA error: Loading module failed\n");
            return NULL;
        }
    }

    return module;
}

void cuda_unload_module(void* context, void* module)
{
    ContextGuard guard(context);

    check_cu(cuModuleUnload_f((CUmodule)module));
}

void* cuda_get_kernel(void* context, void* module, const char* name)
{
    ContextGuard guard(context);

    CUfunction kernel = NULL;
    if (!check_cu(cuModuleGetFunction_f(&kernel, (CUmodule)module, name)))
        fprintf(stderr, "Warp CUDA error: Failed to lookup kernel function %s in module\n", name);

    return kernel;
}

size_t cuda_launch_kernel(void* context, void* kernel, size_t dim, int max_blocks, void** args)
{
    ContextGuard guard(context);

    const int block_dim = 256;
    // CUDA specs up to compute capability 9.0 says the max x-dim grid is 2**31-1, so
    // grid_dim is fine as an int for the near future
    int grid_dim = (dim + block_dim - 1)/block_dim;

    if (max_blocks <= 0) {
        max_blocks = 2147483647;
    }

    if (grid_dim < 0)
    {
#if defined(_DEBUG)
        fprintf(stderr, "Warp warning: Overflow in grid dimensions detected for %zu total elements and 256 threads "
                "per block.\n    Setting block count to %d.\n", dim, max_blocks);
#endif
        grid_dim =  max_blocks;
    }
    else 
    {
        if (grid_dim > max_blocks)
        {
            grid_dim = max_blocks;
        }
    }

    CUresult res = cuLaunchKernel_f(
        (CUfunction)kernel,
        grid_dim, 1, 1,
        block_dim, 1, 1,
        0, get_current_stream(),
        args,
        0);

    check_cu(res);

    return res;
}

void cuda_graphics_map(void* context, void* resource)
{
    ContextGuard guard(context);

    check_cu(cuGraphicsMapResources_f(1, (CUgraphicsResource*)resource, get_current_stream()));
}

void cuda_graphics_unmap(void* context, void* resource)
{
    ContextGuard guard(context);

    check_cu(cuGraphicsUnmapResources_f(1, (CUgraphicsResource*)resource, get_current_stream()));
}

void cuda_graphics_device_ptr_and_size(void* context, void* resource, uint64_t* ptr, size_t* size)
{
    ContextGuard guard(context);

    CUdeviceptr device_ptr;
    size_t bytes;
    check_cu(cuGraphicsResourceGetMappedPointer_f(&device_ptr, &bytes, *(CUgraphicsResource*)resource));

    *ptr = device_ptr;
    *size = bytes;
}

void* cuda_graphics_register_gl_buffer(void* context, uint32_t gl_buffer, unsigned int flags)
{
    ContextGuard guard(context);

    CUgraphicsResource *resource = new CUgraphicsResource;
    check_cu(cuGraphicsGLRegisterBuffer_f(resource, gl_buffer, flags));

    return resource;
}

void cuda_graphics_unregister_resource(void* context, void* resource)
{
    ContextGuard guard(context);

    CUgraphicsResource *res = (CUgraphicsResource*)resource;
    check_cu(cuGraphicsUnregisterResource_f(*res));
    delete res;
}


// impl. files
#include "bvh.cu"
#include "mesh.cu"
#include "sort.cu"
#include "hashgrid.cu"
#include "reduce.cu"
#include "runlength_encode.cu"
#include "scan.cu"
#include "marching.cu"
#include "sparse.cu"
#include "volume.cu"
#include "volume_builder.cu"
#if WP_ENABLE_CUTLASS
    #include "cutlass_gemm.cu"
#endif

//#include "spline.inl"
//#include "volume.inl"
