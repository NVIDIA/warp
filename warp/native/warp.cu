/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "warp.h"
#include "scan.h"
#include "cuda_util.h"
#include "error.h"

#include <cstdlib>
#include <fstream>
#include <nvrtc.h>
#include <nvPTXCompiler.h>
#if WP_ENABLE_MATHDX
    #include <nvJitLink.h>
    #include <libmathdx.h>
#endif

#include <array>
#include <algorithm>
#include <iterator>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define check_any(result) (check_generic(result, __FILE__, __LINE__))
#define check_nvrtc(code) (check_nvrtc_result(code, __FILE__, __LINE__))
#define check_nvptx(code) (check_nvptx_result(code, __FILE__, __LINE__))
#define check_nvjitlink(handle, code) (check_nvjitlink_result(handle, code, __FILE__, __LINE__))
#define check_cufftdx(code) (check_cufftdx_result(code, __FILE__, __LINE__))
#define check_cublasdx(code) (check_cublasdx_result(code, __FILE__, __LINE__))
#define check_cusolver(code) (check_cusolver_result(code, __FILE__, __LINE__))
#define CHECK_ANY(code) \
{ \
    do { \
        bool out = (check_any(code)); \
        if(!out) { \
            return out; \
        } \
    } while(0); \
}
#define CHECK_CUFFTDX(code) \
{ \
    do { \
        bool out = (check_cufftdx(code)); \
        if(!out) { \
            return out; \
        } \
    } while(0); \
}
#define CHECK_CUBLASDX(code) \
{ \
    do { \
        bool out = (check_cufftdx(code)); \
        if(!out) { \
            return out; \
        } \
    } while(0); \
}
#define CHECK_CUSOLVER(code) \
{ \
    do { \
        bool out = (check_cusolver(code)); \
        if(!out) { \
            return out; \
        } \
    } while(0); \
}

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

bool check_generic(int result, const char* file, int line)
{
    if (!result) {
        fprintf(stderr, "Error %d on %s:%d\n", (int)result, file, line);
        return false;
    } else {
        return true;
    }
}

struct DeviceInfo
{
    static constexpr int kNameLen = 128;

    CUdevice device = -1;
    CUuuid uuid = {0};
    int ordinal = -1;
    int pci_domain_id = -1;
    int pci_bus_id = -1;
    int pci_device_id = -1;
    char name[kNameLen] = "";
    int arch = 0;
    int is_uva = 0;
    int is_mempool_supported = 0;
    int sm_count = 0;
    int is_ipc_supported = -1;
    int max_smem_bytes = 0;
    CUcontext primary_context = NULL;
};

struct ContextInfo
{
    DeviceInfo* device_info = NULL;

    // the current stream, managed from Python (see cuda_context_set_stream() and cuda_context_get_stream())
    CUstream stream = NULL;
};

struct CaptureInfo
{
    CUstream stream = NULL;  // the main stream where capture begins and ends
    uint64_t id = 0;  // unique capture id from CUDA
    bool external = false;  // whether this is an external capture
};

struct StreamInfo
{
    CUevent cached_event = NULL;  // event used for stream synchronization (cached to avoid creating temporary events)
    CaptureInfo* capture = NULL;  // capture info (only if started on this stream)
};

struct GraphInfo
{
    std::vector<void*> unfreed_allocs;
};

// Information for graph allocations that are not freed by the graph.
// These allocations have a shared ownership:
// - The graph instance allocates/maps the memory on each launch, even if the user reference is released.
// - The user reference must remain valid even if the graph is destroyed.
// The memory will be freed once the user reference is released and the graph is destroyed.
struct GraphAllocInfo
{
    uint64_t capture_id = 0;
    void* context = NULL;
    bool ref_exists = false;  // whether user reference still exists
    bool graph_destroyed = false;  // whether graph instance was destroyed
};

// Information used when deferring deallocations.
struct FreeInfo
{
    void* context = NULL;
    void* ptr = NULL;
    bool is_async = false;
};

// Information used when deferring module unloading.
struct ModuleInfo
{
    void* context = NULL;
    void* module = NULL;
};

static std::unordered_map<CUfunction, std::string> g_kernel_names;

// cached info for all devices, indexed by ordinal
static std::vector<DeviceInfo> g_devices;

// maps CUdevice to DeviceInfo
static std::map<CUdevice, DeviceInfo*> g_device_map;

// cached info for all known contexts
static std::map<CUcontext, ContextInfo> g_contexts;

// cached info for all known streams (including registered external streams)
static std::unordered_map<CUstream, StreamInfo> g_streams;

// Ongoing graph captures registered using wp.capture_begin().
// This maps the capture id to the stream where capture was started.
// See cuda_graph_begin_capture(), cuda_graph_end_capture(), and free_device_async().
static std::unordered_map<uint64_t, CaptureInfo*> g_captures;

// Memory allocated during graph capture requires special handling.
// See alloc_device_async() and free_device_async().
static std::unordered_map<void*, GraphAllocInfo> g_graph_allocs;

// Memory that cannot be freed immediately gets queued here.
// Call free_deferred_allocs() to release.
static std::vector<FreeInfo> g_deferred_free_list;

// Modules that cannot be unloaded immediately get queued here.
// Call unload_deferred_modules() to release.
static std::vector<ModuleInfo> g_deferred_module_list;

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

    int device_count = 0;
    if (check_cu(cuDeviceGetCount_f(&device_count)))
    {
        g_devices.resize(device_count);

        for (int i = 0; i < device_count; i++)
        {
            CUdevice device;
            if (check_cu(cuDeviceGet_f(&device, i)))
            {
                // query device info
                g_devices[i].device = device;
                g_devices[i].ordinal = i;
                check_cu(cuDeviceGetName_f(g_devices[i].name, DeviceInfo::kNameLen, device));
                check_cu(cuDeviceGetUuid_f(&g_devices[i].uuid, device));
                check_cu(cuDeviceGetAttribute_f(&g_devices[i].pci_domain_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device));
                check_cu(cuDeviceGetAttribute_f(&g_devices[i].pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device));
                check_cu(cuDeviceGetAttribute_f(&g_devices[i].pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device));
                check_cu(cuDeviceGetAttribute_f(&g_devices[i].is_uva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
                check_cu(cuDeviceGetAttribute_f(&g_devices[i].is_mempool_supported, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device));
                check_cu(cuDeviceGetAttribute_f(&g_devices[i].sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
#ifdef CUDA_VERSION
#if CUDA_VERSION >= 12000
                int device_attribute_integrated = 0;
                check_cu(cuDeviceGetAttribute_f(&device_attribute_integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device));
                if (device_attribute_integrated == 0)
                {
                    check_cu(cuDeviceGetAttribute_f(&g_devices[i].is_ipc_supported, CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED, device));
                }
                else
                {
                    // integrated devices do not support CUDA IPC
                    g_devices[i].is_ipc_supported = 0;
                }
#endif
#endif
                check_cu(cuDeviceGetAttribute_f(&g_devices[i].max_smem_bytes, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
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

    // initialize default timing state
    static CudaTimingState default_timing_state(0, NULL);
    g_cuda_timing_state = &default_timing_state;

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

static inline CUstream get_current_stream(void* context=NULL)
{
    return static_cast<CUstream>(cuda_context_get_stream(context));
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

        CUdevice device;
        if (check_cu(cuCtxGetDevice_f(&device)))
        {
            DeviceInfo* device_info = g_device_map[device];

            // workaround for https://nvbugspro.nvidia.com/bug/4456003
            if (device_info->is_mempool_supported)
            {
                void* dummy = NULL;
                check_cuda(cudaMallocAsync(&dummy, 1, NULL));
                check_cuda(cudaFreeAsync(dummy, NULL));
            }

            ContextInfo context_info;
            context_info.device_info = device_info;
            auto result = g_contexts.insert(std::make_pair(ctx, context_info));
            return &result.first->second;
        }
    }

    return NULL;
}

static inline ContextInfo* get_context_info(void* context)
{
    return get_context_info(static_cast<CUcontext>(context));
}

static inline StreamInfo* get_stream_info(CUstream stream)
{
    auto it = g_streams.find(stream);
    if (it != g_streams.end())
        return &it->second;
    else
        return NULL;
}

static void deferred_free(void* ptr, void* context, bool is_async)
{
    FreeInfo free_info;
    free_info.ptr = ptr;
    free_info.context = context ? context : get_current_context();
    free_info.is_async = is_async;
    g_deferred_free_list.push_back(free_info);
}

static int free_deferred_allocs(void* context = NULL)
{
    if (g_deferred_free_list.empty() || !g_captures.empty())
        return 0;

    int num_freed_allocs = 0;
    for (auto it = g_deferred_free_list.begin(); it != g_deferred_free_list.end(); /*noop*/)
    {
        const FreeInfo& free_info = *it;

        // free the pointer if it matches the given context or if the context is unspecified
        if (free_info.context == context || !context)
        {
            ContextGuard guard(free_info.context);

            if (free_info.is_async)
            {
                // this could be a regular stream-ordered allocation or a graph allocation
                cudaError_t res = cudaFreeAsync(free_info.ptr, NULL);
                if (res != cudaSuccess)
                {
                    if (res == cudaErrorInvalidValue)
                    {
                        // This can happen if we try to release the pointer but the graph was
                        // never launched, so the memory isn't mapped.
                        // This is fine, so clear the error.
                        cudaGetLastError();
                    }
                    else
                    {
                        // something else went wrong, report error
                        check_cuda(res);
                    }
                }
            }
            else
            {
                check_cuda(cudaFree(free_info.ptr));
            }

            ++num_freed_allocs;

            it = g_deferred_free_list.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return num_freed_allocs;
}

static int unload_deferred_modules(void* context = NULL)
{
    if (g_deferred_module_list.empty() || !g_captures.empty())
        return 0;

    int num_unloaded_modules = 0;
    for (auto it = g_deferred_module_list.begin(); it != g_deferred_module_list.end(); /*noop*/)
    {
        // free the module if it matches the given context or if the context is unspecified
        const ModuleInfo& module_info = *it;
        if (module_info.context == context || !context)
        {
            cuda_unload_module(module_info.context, module_info.module);
            ++num_unloaded_modules;
            it = g_deferred_module_list.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return num_unloaded_modules;
}

static void CUDART_CB on_graph_destroy(void* user_data)
{
    if (!user_data)
        return;

    GraphInfo* graph_info = static_cast<GraphInfo*>(user_data);

    for (void* ptr : graph_info->unfreed_allocs)
    {
        auto alloc_iter = g_graph_allocs.find(ptr);
        if (alloc_iter != g_graph_allocs.end())
        {
            GraphAllocInfo& alloc_info = alloc_iter->second;
            if (alloc_info.ref_exists)
            {
                // unreference from graph so the pointer will be deallocated when the user reference goes away
                alloc_info.graph_destroyed = true;
            }
            else
            {
                // the pointer can be freed, but we can't call CUDA functions in this callback, so defer it
                deferred_free(ptr, alloc_info.context, true);
                g_graph_allocs.erase(alloc_iter);
            }
        }
    }

    delete graph_info;
}

static inline const char* get_cuda_kernel_name(void* kernel)
{
    CUfunction cuda_func = static_cast<CUfunction>(kernel);
    auto name_iter = g_kernel_names.find((CUfunction)cuda_func);
    if (name_iter != g_kernel_names.end())
        return name_iter->second.c_str();
    else
        return "unknown_kernel";
}


void* alloc_pinned(size_t s)
{
    void* ptr = NULL;
    check_cuda(cudaMallocHost(&ptr, s));
    return ptr;
}

void free_pinned(void* ptr)
{
    cudaFreeHost(ptr);
}

void* alloc_device(void* context, size_t s)
{
    int ordinal = cuda_context_get_device_ordinal(context);

    // use stream-ordered allocator if available
    if (cuda_device_is_mempool_supported(ordinal))
        return alloc_device_async(context, s);
    else
        return alloc_device_default(context, s);
}

void free_device(void* context, void* ptr)
{
    int ordinal = cuda_context_get_device_ordinal(context);

    // use stream-ordered allocator if available
    if (cuda_device_is_mempool_supported(ordinal))
        free_device_async(context, ptr);
    else
        free_device_default(context, ptr);
}

void* alloc_device_default(void* context, size_t s)
{
    ContextGuard guard(context);

    void* ptr = NULL;
    check_cuda(cudaMalloc(&ptr, s));

    return ptr;
}

void free_device_default(void* context, void* ptr)
{
    ContextGuard guard(context);

    // check if a capture is in progress
    if (g_captures.empty())
    {
        check_cuda(cudaFree(ptr));
    }
    else
    {
        // we must defer the operation until graph captures complete
        deferred_free(ptr, context, false);
    }
}

void* alloc_device_async(void* context, size_t s)
{
    // stream-ordered allocations don't rely on the current context,
    // but we set the context here for consistent behaviour
    ContextGuard guard(context);

    ContextInfo* context_info = get_context_info(context);
    if (!context_info)
        return NULL;

    CUstream stream = context_info->stream;

    void* ptr = NULL;
    check_cuda(cudaMallocAsync(&ptr, s, stream));

    if (ptr)
    {
        // if the stream is capturing, the allocation requires special handling
        if (cuda_stream_is_capturing(stream))
        {
            // check if this is a known capture
            uint64_t capture_id = get_capture_id(stream);
            auto capture_iter = g_captures.find(capture_id);
            if (capture_iter != g_captures.end())
            {
                // remember graph allocation details
                GraphAllocInfo alloc_info;
                alloc_info.capture_id = capture_id;
                alloc_info.context = context ? context : get_current_context();
                alloc_info.ref_exists = true;  // user reference created and returned here
                alloc_info.graph_destroyed = false;  // graph not destroyed yet
                g_graph_allocs[ptr] = alloc_info;
            }
        }
    }

    return ptr;
}

void free_device_async(void* context, void* ptr)
{
    // stream-ordered allocators generally don't rely on the current context,
    // but we set the context here for consistent behaviour
    ContextGuard guard(context);

    // NB: Stream-ordered deallocations are tricky, because the memory could still be used on another stream
    // or even multiple streams.  To avoid use-after-free errors, we need to ensure that all preceding work
    // completes before releasing the memory.  The strategy is different for regular stream-ordered allocations
    // and allocations made during graph capture.  See below for details.

    // check if this allocation was made during graph capture
    auto alloc_iter = g_graph_allocs.find(ptr);
    if (alloc_iter == g_graph_allocs.end())
    {
        // Not a graph allocation.
        // Check if graph capture is ongoing.
        if (g_captures.empty())
        {
            // cudaFreeAsync on the null stream does not block or trigger synchronization, but it postpones
            // the deallocation until a synchronization point is reached, so preceding work on this pointer
            // should safely complete.
            check_cuda(cudaFreeAsync(ptr, NULL));
        }
        else
        {
            // We must defer the free operation until graph capture completes.
            deferred_free(ptr, context, true);
        }
    }
    else
    {
        // get the graph allocation details
        GraphAllocInfo& alloc_info = alloc_iter->second;

        uint64_t capture_id = alloc_info.capture_id;

        // check if the capture is still active
        auto capture_iter = g_captures.find(capture_id);
        if (capture_iter != g_captures.end())
        {
            // Add a mem free node.  Use all current leaf nodes as dependencies to ensure that all prior
            // work completes before deallocating.  This works with both Warp-initiated and external captures
            // and avoids the need to explicitly track all streams used during the capture.
            CaptureInfo* capture = capture_iter->second;
            cudaGraph_t graph = get_capture_graph(capture->stream);
            std::vector<cudaGraphNode_t> leaf_nodes;
            if (graph && get_graph_leaf_nodes(graph, leaf_nodes))
            {
                cudaGraphNode_t free_node;
                check_cuda(cudaGraphAddMemFreeNode(&free_node, graph, leaf_nodes.data(), leaf_nodes.size(), ptr));
            }

            // we're done with this allocation, it's owned by the graph
            g_graph_allocs.erase(alloc_iter);
        }
        else
        {
            // the capture has ended
            // if the owning graph was already destroyed, we can free the pointer now
            if (alloc_info.graph_destroyed)
            {
                if (g_captures.empty())
                {
                    // try to free the pointer now
                    cudaError_t res = cudaFreeAsync(ptr, NULL);
                    if (res == cudaErrorInvalidValue)
                    {
                        // This can happen if we try to release the pointer but the graph was
                        // never launched, so the memory isn't mapped.
                        // This is fine, so clear the error.
                        cudaGetLastError();
                    }
                    else
                    {
                        // check for other errors
                        check_cuda(res);
                    }
                }
                else
                {
                    // We must defer the operation until graph capture completes.
                    deferred_free(ptr, context, true);
                }

                // we're done with this allocation
                g_graph_allocs.erase(alloc_iter);
            }
            else
            {
                // graph still exists
                // unreference the pointer so it will be deallocated once the graph instance is destroyed
                alloc_info.ref_exists = false;
            }
        }
    }
}

bool memcpy_h2d(void* context, void* dest, void* src, size_t n, void* stream)
{
    ContextGuard guard(context);

    CUstream cuda_stream;
    if (stream != WP_CURRENT_STREAM)
        cuda_stream = static_cast<CUstream>(stream);
    else
        cuda_stream = get_current_stream(context);

    begin_cuda_range(WP_TIMING_MEMCPY, cuda_stream, context, "memcpy HtoD");

    bool result = check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice, cuda_stream));

    end_cuda_range(WP_TIMING_MEMCPY, cuda_stream);

    return result;
}

bool memcpy_d2h(void* context, void* dest, void* src, size_t n, void* stream)
{
    ContextGuard guard(context);

    CUstream cuda_stream;
    if (stream != WP_CURRENT_STREAM)
        cuda_stream = static_cast<CUstream>(stream);
    else
        cuda_stream = get_current_stream(context);

    begin_cuda_range(WP_TIMING_MEMCPY, cuda_stream, context, "memcpy DtoH");

    bool result = check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost, cuda_stream));

    end_cuda_range(WP_TIMING_MEMCPY, cuda_stream);

    return result;
}

bool memcpy_d2d(void* context, void* dest, void* src, size_t n, void* stream)
{
    ContextGuard guard(context);

    CUstream cuda_stream;
    if (stream != WP_CURRENT_STREAM)
        cuda_stream = static_cast<CUstream>(stream);
    else
        cuda_stream = get_current_stream(context);

    begin_cuda_range(WP_TIMING_MEMCPY, cuda_stream, context, "memcpy DtoD");

    bool result = check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToDevice, cuda_stream));

    end_cuda_range(WP_TIMING_MEMCPY, cuda_stream);

    return result;
}

bool memcpy_p2p(void* dst_context, void* dst, void* src_context, void* src, size_t n, void* stream)
{
    // ContextGuard guard(context);

    CUstream cuda_stream;
    if (stream != WP_CURRENT_STREAM)
        cuda_stream = static_cast<CUstream>(stream);
    else
        cuda_stream = get_current_stream(dst_context);

    // Notes:
    // - cuMemcpyPeerAsync() works fine with both regular and pooled allocations (cudaMalloc() and cudaMallocAsync(), respectively)
    //   when not capturing a graph.
    // - cuMemcpyPeerAsync() is not supported during graph capture, so we must use cudaMemcpyAsync() with kind=cudaMemcpyDefault.
    // - cudaMemcpyAsync() works fine with regular allocations, but doesn't work with pooled allocations
    //   unless mempool access has been enabled.
    // - There is no reliable way to check if mempool access is enabled during graph capture,
    //   because cudaMemPoolGetAccess() cannot be called during graph capture.
    // - CUDA will report error 1 (invalid argument) if cudaMemcpyAsync() is called but mempool access is not enabled.

    if (!cuda_stream_is_capturing(stream))
    {
        begin_cuda_range(WP_TIMING_MEMCPY, cuda_stream, get_stream_context(stream), "memcpy PtoP");

        bool result = check_cu(cuMemcpyPeerAsync_f(
            (CUdeviceptr)dst, (CUcontext)dst_context,
            (CUdeviceptr)src, (CUcontext)src_context,
            n, cuda_stream));

        end_cuda_range(WP_TIMING_MEMCPY, cuda_stream);

        return result;
    }
    else
    {
        cudaError_t result = cudaSuccess;

        // cudaMemcpyAsync() is sensitive to the bound context to resolve pointer locations.
        // If fails with cudaErrorInvalidValue if it cannot resolve an argument.
        // We first try the copy in the destination context, then if it fails we retry in the source context.
        // The cudaErrorInvalidValue error doesn't cause graph capture to fail, so it's ok to retry.
        // Since this trial-and-error shenanigans only happens during capture, there
        // is no perf impact when the graph is launched.
        // For bonus points, this approach simplifies memory pool access requirements.
        // Access only needs to be enabled one way, either from the source device to the destination device
        // or vice versa.  Sometimes, when it's really quiet, you can actually hear my genius.
        {
            // try doing the copy in the destination context
            ContextGuard guard(dst_context);
            result = cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, cuda_stream);

            if (result != cudaSuccess)
            {
                // clear error in destination context
                cudaGetLastError();

                // try doing the copy in the source context
                ContextGuard guard(src_context);
                result = cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, cuda_stream);

                // clear error in source context
                cudaGetLastError();
            }
        }

        // If the copy failed, try to detect if mempool allocations are involved to generate a helpful error message.
        if (!check_cuda(result))
        {
            if (result == cudaErrorInvalidValue && src != NULL && dst != NULL)
            {
                // check if either of the pointers was allocated from a mempool
                void* src_mempool = NULL;
                void* dst_mempool = NULL;
                cuPointerGetAttribute_f(&src_mempool, CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, (CUdeviceptr)src);
                cuPointerGetAttribute_f(&dst_mempool, CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, (CUdeviceptr)dst);
                cudaGetLastError();  // clear any errors
                // check if either of the pointers was allocated during graph capture
                auto src_alloc = g_graph_allocs.find(src);
                auto dst_alloc = g_graph_allocs.find(dst);
                if (src_mempool != NULL || src_alloc != g_graph_allocs.end() ||
                    dst_mempool != NULL || dst_alloc != g_graph_allocs.end())
                {
                    wp::append_error_string("*** CUDA mempool allocations were used in a peer-to-peer copy during graph capture.");
                    wp::append_error_string("*** This operation fails if mempool access is not enabled between the peer devices.");
                    wp::append_error_string("*** Either enable mempool access between the devices or use the default CUDA allocator");
                    wp::append_error_string("*** to pre-allocate the arrays before graph capture begins.");
                }
            }

            return false;
        }

        return true;
    }
}


__global__ void memset_kernel(int* dest, int value, size_t n)
{
    const size_t tid = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
    
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
        cudaStream_t stream = get_current_stream();

        begin_cuda_range(WP_TIMING_MEMSET, stream, context, "memset");

        // for unaligned lengths fallback to CUDA memset
        check_cuda(cudaMemsetAsync(dest, value, n, stream));

        end_cuda_range(WP_TIMING_MEMSET, stream);
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
    size_t tid = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
    if (tid < n)
    {
        memcpy((int8_t*)dst + srcsize * tid, src, srcsize);
    }
}

// this should be faster than memtile_kernel, but requires proper alignment of dst
template <typename T>
__global__ void memtile_value_kernel(T* dst, T value, size_t n)
{
    size_t tid = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
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

        // copy value to device memory
        // TODO: use a persistent stream-local staging buffer to avoid allocs?
        void* src_devptr = alloc_device(WP_CURRENT_CONTEXT, srcsize);
        check_cuda(cudaMemcpyAsync(src_devptr, src, srcsize, cudaMemcpyHostToDevice, get_current_stream()));

        wp_launch_device(WP_CURRENT_CONTEXT, memtile_kernel, n, (dst, src_devptr, srcsize, n));

        free_device(WP_CURRENT_CONTEXT, src_devptr);

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


WP_API bool array_copy_device(void* context, void* dst, void* src, int dst_type, int src_type, int elem_size)
{
    if (!src || !dst)
        return false;

    const void* src_data = NULL;
    void* dst_data = NULL;
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
        return false;
    }

    if (dst_type == wp::ARRAY_TYPE_REGULAR)
    {
        const wp::array_t<void>& dst_arr = *static_cast<const wp::array_t<void>*>(dst);
        dst_data = dst_arr.data;
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
        return false;
    }

    if (src_ndim != dst_ndim)
    {
        fprintf(stderr, "Warp copy error: Incompatible array dimensionalities (%d and %d)\n", src_ndim, dst_ndim);
        return false;
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
                return false;
            }
            wp_launch_device(WP_CURRENT_CONTEXT, array_copy_fabric_to_fabric_kernel, n,
                            (*dst_fabricarray, *src_fabricarray, elem_size));
            return true;
        }
        else if (src_indexedfabricarray)
        {
            // copy from fabric indexed to fabric
            if (src_indexedfabricarray->size != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            wp_launch_device(WP_CURRENT_CONTEXT, array_copy_fabric_indexed_to_fabric_kernel, n,
                            (*dst_fabricarray, *src_indexedfabricarray, elem_size));
            return true;
        }
        else
        {
            // copy to fabric
            if (size_t(src_shape[0]) != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            wp_launch_device(WP_CURRENT_CONTEXT, array_copy_to_fabric_kernel, n,
                            (*dst_fabricarray, src_data, src_strides[0], src_indices[0], elem_size));
            return true;
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
                return false;
            }
            wp_launch_device(WP_CURRENT_CONTEXT, array_copy_fabric_to_fabric_indexed_kernel, n,
                            (*dst_indexedfabricarray, *src_fabricarray, elem_size));
            return true;
        }
        else if (src_indexedfabricarray)
        {
            // copy from fabric indexed to fabric indexed
            if (src_indexedfabricarray->size != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            wp_launch_device(WP_CURRENT_CONTEXT, array_copy_fabric_indexed_to_fabric_indexed_kernel, n,
                            (*dst_indexedfabricarray, *src_indexedfabricarray, elem_size));
            return true;
        }
        else
        {
            // copy to fabric indexed
            if (size_t(src_shape[0]) != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            wp_launch_device(WP_CURRENT_CONTEXT, array_copy_to_fabric_indexed_kernel, n,
                             (*dst_indexedfabricarray, src_data, src_strides[0], src_indices[0], elem_size));
            return true;
        }
    }
    else if (src_fabricarray)
    {
        // copy from fabric
        size_t n = src_fabricarray->size;
        if (size_t(dst_shape[0]) != n)
        {
            fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
            return false;
        }
        wp_launch_device(WP_CURRENT_CONTEXT, array_copy_from_fabric_kernel, n,
                         (*src_fabricarray, dst_data, dst_strides[0], dst_indices[0], elem_size));
        return true;
    }
    else if (src_indexedfabricarray)
    {
        // copy from fabric indexed
        size_t n = src_indexedfabricarray->size;
        if (size_t(dst_shape[0]) != n)
        {
            fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
            return false;
        }
        wp_launch_device(WP_CURRENT_CONTEXT, array_copy_from_fabric_indexed_kernel, n,
                         (*src_indexedfabricarray, dst_data, dst_strides[0], dst_indices[0], elem_size));
        return true;
    }

    size_t n = 1;
    for (int i = 0; i < src_ndim; i++)
    {
        if (src_shape[i] != dst_shape[i])
        {
            fprintf(stderr, "Warp copy error: Incompatible array shapes\n");
            return false;
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
        return false;
    }

    return check_cuda(cudaGetLastError());
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
    // TODO: use a persistent stream-local staging buffer to avoid allocs?
    void* value_devptr = alloc_device(WP_CURRENT_CONTEXT, value_size);
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

    free_device(WP_CURRENT_CONTEXT, value_devptr);
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

void* cuda_device_get_primary_context(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
    {
        DeviceInfo& device_info = g_devices[ordinal];

        // acquire the primary context if we haven't already
        if (!device_info.primary_context)
            check_cu(cuDevicePrimaryCtxRetain_f(&device_info.primary_context, device_info.device));

        return device_info.primary_context;
    }

    return NULL;
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

int cuda_device_get_sm_count(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].sm_count;
    return 0;
}

void cuda_device_get_uuid(int ordinal, char uuid[16])
{
    memcpy(uuid, g_devices[ordinal].uuid.bytes, sizeof(char)*16);
}

int cuda_device_get_pci_domain_id(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].pci_domain_id;
    return -1;
}

int cuda_device_get_pci_bus_id(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].pci_bus_id;
    return -1;
}

int cuda_device_get_pci_device_id(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].pci_device_id;
    return -1;
}

int cuda_device_is_uva(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].is_uva;
    return 0;
}

int cuda_device_is_mempool_supported(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].is_mempool_supported;
    return 0;
}

int cuda_device_is_ipc_supported(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].is_ipc_supported;
    return 0;
}

int cuda_device_set_mempool_release_threshold(int ordinal, uint64_t threshold)
{
    if (ordinal < 0 || ordinal > int(g_devices.size()))
    {
        fprintf(stderr, "Invalid device ordinal %d\n", ordinal);
        return 0;
    }

    if (!g_devices[ordinal].is_mempool_supported)
        return 0;

    cudaMemPool_t pool;
    if (!check_cuda(cudaDeviceGetDefaultMemPool(&pool, ordinal)))
    {
        fprintf(stderr, "Warp error: Failed to get memory pool on device %d\n", ordinal);
        return 0;
    }

    if (!check_cuda(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold)))
    {
        fprintf(stderr, "Warp error: Failed to set memory pool attribute on device %d\n", ordinal);
        return 0;
    }

    return 1;  // success
}

uint64_t cuda_device_get_mempool_release_threshold(int ordinal)
{
    if (ordinal < 0 || ordinal > int(g_devices.size()))
    {
        fprintf(stderr, "Invalid device ordinal %d\n", ordinal);
        return 0;
    }

    if (!g_devices[ordinal].is_mempool_supported)
        return 0;

    cudaMemPool_t pool;
    if (!check_cuda(cudaDeviceGetDefaultMemPool(&pool, ordinal)))
    {
        fprintf(stderr, "Warp error: Failed to get memory pool on device %d\n", ordinal);
        return 0;
    }

    uint64_t threshold = 0;
    if (!check_cuda(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold)))
    {
        fprintf(stderr, "Warp error: Failed to get memory pool release threshold on device %d\n", ordinal);
        return 0;
    }

    return threshold;
}

uint64_t cuda_device_get_mempool_used_mem_current(int ordinal)
{
    if (ordinal < 0 || ordinal > int(g_devices.size()))
    {
        fprintf(stderr, "Invalid device ordinal %d\n", ordinal);
        return 0;
    }

    if (!g_devices[ordinal].is_mempool_supported)
        return 0;

    cudaMemPool_t pool;
    if (!check_cuda(cudaDeviceGetDefaultMemPool(&pool, ordinal)))
    {
        fprintf(stderr, "Warp error: Failed to get memory pool on device %d\n", ordinal);
        return 0;
    }

    uint64_t mem_used = 0;
    if (!check_cuda(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &mem_used)))
    {
        fprintf(stderr, "Warp error: Failed to get amount of currently used memory from the memory pool on device %d\n", ordinal);
        return 0;
    }

    return mem_used;
}

uint64_t cuda_device_get_mempool_used_mem_high(int ordinal)
{
    if (ordinal < 0 || ordinal > int(g_devices.size()))
    {
        fprintf(stderr, "Invalid device ordinal %d\n", ordinal);
        return 0;
    }

    if (!g_devices[ordinal].is_mempool_supported)
        return 0;

    cudaMemPool_t pool;
    if (!check_cuda(cudaDeviceGetDefaultMemPool(&pool, ordinal)))
    {
        fprintf(stderr, "Warp error: Failed to get memory pool on device %d\n", ordinal);
        return 0;
    }

    uint64_t mem_high_water_mark = 0;
    if (!check_cuda(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &mem_high_water_mark)))
    {
        fprintf(stderr, "Warp error: Failed to get memory usage high water mark from the memory pool on device %d\n", ordinal);
        return 0;
    }

    return mem_high_water_mark;
}

void cuda_device_get_memory_info(int ordinal, size_t* free_mem, size_t* total_mem)
{
    // use temporary storage if user didn't specify pointers
    size_t tmp_free_mem, tmp_total_mem;

    if (free_mem)
        *free_mem = 0;
    else
        free_mem = &tmp_free_mem;

    if (total_mem)
        *total_mem = 0;
    else
        total_mem = &tmp_total_mem;

    if (ordinal >= 0 && ordinal < int(g_devices.size()))
    {
        if (g_devices[ordinal].primary_context)
        {
            ContextGuard guard(g_devices[ordinal].primary_context, true);
            check_cu(cuMemGetInfo_f(free_mem, total_mem));
        }
        else
        {
            // if we haven't acquired the primary context yet, acquire it temporarily
            CUcontext primary_context = NULL;
            check_cu(cuDevicePrimaryCtxRetain_f(&primary_context, g_devices[ordinal].device));
            {
                ContextGuard guard(primary_context, true);
                check_cu(cuMemGetInfo_f(free_mem, total_mem));
            }
            check_cu(cuDevicePrimaryCtxRelease_f(g_devices[ordinal].device));
        }
    }
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

    if (free_deferred_allocs(context ? context : get_current_context()) > 0)
    {
        // ensure deferred asynchronous deallocations complete
        check_cu(cuCtxSynchronize_f());
    }

    unload_deferred_modules(context);

    // check_cuda(cudaDeviceGraphMemTrim(cuda_context_get_device_ordinal(context)));
}

uint64_t cuda_context_check(void* context)
{
    ContextGuard guard(context);

    // check errors before syncing
    cudaError_t e = cudaGetLastError();
    check_cuda(e);

    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    check_cuda(cudaStreamIsCapturing(get_current_stream(), &status));
    
    // synchronize if the stream is not capturing
    if (status == cudaStreamCaptureStatusNone)
    {
        check_cuda(cudaDeviceSynchronize());
        e = cudaGetLastError();
    }

    return static_cast<uint64_t>(e);
}


int cuda_context_get_device_ordinal(void* context)
{
    ContextInfo* info = get_context_info(static_cast<CUcontext>(context));
    return info && info->device_info ? info->device_info->ordinal : -1;
}

int cuda_context_is_primary(void* context)
{
    CUcontext ctx = static_cast<CUcontext>(context);
    ContextInfo* context_info = get_context_info(ctx);
    if (!context_info)
    {
        fprintf(stderr, "Warp error: Failed to get context info\n");
        return 0;
    }

    // if the device primary context is known, check if it matches the given context
    DeviceInfo* device_info = context_info->device_info;
    if (device_info->primary_context)
        return int(ctx == device_info->primary_context);

    // there is no CUDA API to check if a context is primary, but we can temporarily
    // acquire the device's primary context to check the pointer
    CUcontext primary_ctx;
    if (check_cu(cuDevicePrimaryCtxRetain_f(&primary_ctx, device_info->device)))
    {
        check_cu(cuDevicePrimaryCtxRelease_f(device_info->device));
        return int(ctx == primary_ctx);
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

void cuda_context_set_stream(void* context, void* stream, int sync)
{
    ContextInfo* context_info = get_context_info(static_cast<CUcontext>(context));
    if (context_info)
    {
        CUstream new_stream = static_cast<CUstream>(stream);

        // check whether we should sync with the previous stream on this device
        if (sync)
        {
            CUstream old_stream = context_info->stream;
            StreamInfo* old_stream_info = get_stream_info(old_stream);
            if (old_stream_info)
            {
                CUevent cached_event = old_stream_info->cached_event;
                check_cu(cuEventRecord_f(cached_event, old_stream));
                check_cu(cuStreamWaitEvent_f(new_stream, cached_event, CU_EVENT_WAIT_DEFAULT));
            }
        }

        context_info->stream = new_stream;
    }
}

int cuda_is_peer_access_supported(int target_ordinal, int peer_ordinal)
{
    int num_devices = int(g_devices.size());

    if (target_ordinal < 0 || target_ordinal > num_devices)
    {
        fprintf(stderr, "Warp error: Invalid target device ordinal %d\n", target_ordinal);
        return 0;
    }

    if (peer_ordinal < 0 || peer_ordinal > num_devices)
    {
        fprintf(stderr, "Warp error: Invalid peer device ordinal %d\n", peer_ordinal);
        return 0;
    }

    if (target_ordinal == peer_ordinal)
        return 1;

    int can_access = 0;
    check_cuda(cudaDeviceCanAccessPeer(&can_access, peer_ordinal, target_ordinal));

    return can_access;
}

int cuda_is_peer_access_enabled(void* target_context, void* peer_context)
{
    if (!target_context || !peer_context)
    {
        fprintf(stderr, "Warp error: invalid CUDA context\n");
        return 0;
    }

    if (target_context == peer_context)
        return 1;

    int target_ordinal = cuda_context_get_device_ordinal(target_context);
    int peer_ordinal = cuda_context_get_device_ordinal(peer_context);

    // check if peer access is supported
    int can_access = 0;
    check_cuda(cudaDeviceCanAccessPeer(&can_access, peer_ordinal, target_ordinal));
    if (!can_access)
        return 0;

    // There is no CUDA API to query if peer access is enabled, but we can try to enable it and check the result.

    ContextGuard guard(peer_context, true);

    CUcontext target_ctx = static_cast<CUcontext>(target_context);

    CUresult result = cuCtxEnablePeerAccess_f(target_ctx, 0);
    if (result == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
    {
        return 1;
    }
    else if (result == CUDA_SUCCESS)
    {
        // undo enablement
        check_cu(cuCtxDisablePeerAccess_f(target_ctx));
        return 0;
    }
    else
    {
        // report error
        check_cu(result);
        return 0;
    }
}

int cuda_set_peer_access_enabled(void* target_context, void* peer_context, int enable)
{
    if (!target_context || !peer_context)
    {
        fprintf(stderr, "Warp error: invalid CUDA context\n");
        return 0;
    }

    if (target_context == peer_context)
        return 1;  // no-op
        
    int target_ordinal = cuda_context_get_device_ordinal(target_context);
    int peer_ordinal = cuda_context_get_device_ordinal(peer_context);

    // check if peer access is supported
    int can_access = 0;
    check_cuda(cudaDeviceCanAccessPeer(&can_access, peer_ordinal, target_ordinal));
    if (!can_access)
    {
        // failure if enabling, success if disabling
        if (enable)
        {
            fprintf(stderr, "Warp error: device %d cannot access device %d\n", peer_ordinal, target_ordinal);
            return 0;
        }
        else
            return 1;
    }

    ContextGuard guard(peer_context, true);

    CUcontext target_ctx = static_cast<CUcontext>(target_context);

    if (enable)
    {
        CUresult status = cuCtxEnablePeerAccess_f(target_ctx, 0);
        if (status != CUDA_SUCCESS && status != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
        {
            check_cu(status);
            fprintf(stderr, "Warp error: failed to enable peer access from device %d to device %d\n", peer_ordinal, target_ordinal);
            return 0;
        }
    }
    else
    {
        CUresult status = cuCtxDisablePeerAccess_f(target_ctx);
        if (status != CUDA_SUCCESS && status != CUDA_ERROR_PEER_ACCESS_NOT_ENABLED)
        {
            check_cu(status);
            fprintf(stderr, "Warp error: failed to disable peer access from device %d to device %d\n", peer_ordinal, target_ordinal);
            return 0;
        }
    }

    return 1;  // success
}

int cuda_is_mempool_access_enabled(int target_ordinal, int peer_ordinal)
{
    int num_devices = int(g_devices.size());

    if (target_ordinal < 0 || target_ordinal > num_devices)
    {
        fprintf(stderr, "Warp error: Invalid device ordinal %d\n", target_ordinal);
        return 0;
    }

    if (peer_ordinal < 0 || peer_ordinal > num_devices)
    {
        fprintf(stderr, "Warp error: Invalid peer device ordinal %d\n", peer_ordinal);
        return 0;
    }

    if (target_ordinal == peer_ordinal)
        return 1;

    cudaMemPool_t pool;
    if (!check_cuda(cudaDeviceGetDefaultMemPool(&pool, target_ordinal)))
    {
        fprintf(stderr, "Warp error: Failed to get memory pool of device %d\n", target_ordinal);
        return 0;
    }

    cudaMemAccessFlags flags = cudaMemAccessFlagsProtNone;
    cudaMemLocation location;
    location.id = peer_ordinal;
    location.type = cudaMemLocationTypeDevice;
    if (check_cuda(cudaMemPoolGetAccess(&flags, pool, &location)))
        return int(flags != cudaMemAccessFlagsProtNone);

    return 0;
}

int cuda_set_mempool_access_enabled(int target_ordinal, int peer_ordinal, int enable)
{
    int num_devices = int(g_devices.size());

    if (target_ordinal < 0 || target_ordinal > num_devices)
    {
        fprintf(stderr, "Warp error: Invalid device ordinal %d\n", target_ordinal);
        return 0;
    }

    if (peer_ordinal < 0 || peer_ordinal > num_devices)
    {
        fprintf(stderr, "Warp error: Invalid peer device ordinal %d\n", peer_ordinal);
        return 0;
    }

    if (target_ordinal == peer_ordinal)
        return 1;  // no-op

    // get the memory pool
    cudaMemPool_t pool;
    if (!check_cuda(cudaDeviceGetDefaultMemPool(&pool, target_ordinal)))
    {
        fprintf(stderr, "Warp error: Failed to get memory pool of device %d\n", target_ordinal);
        return 0;
    }

    cudaMemAccessDesc desc;
    desc.location.type = cudaMemLocationTypeDevice;
    desc.location.id = peer_ordinal;

    // only cudaMemAccessFlagsProtReadWrite and cudaMemAccessFlagsProtNone are supported
    if (enable)
        desc.flags = cudaMemAccessFlagsProtReadWrite;
    else
        desc.flags = cudaMemAccessFlagsProtNone;

    if (!check_cuda(cudaMemPoolSetAccess(pool, &desc, 1)))
    {
        fprintf(stderr, "Warp error: Failed to set mempool access from device %d to device %d\n", peer_ordinal, target_ordinal);
        return 0;
    }

    return 1;  // success
}

void cuda_ipc_get_mem_handle(void* ptr, char* out_buffer) {
    CUipcMemHandle memHandle;
    check_cu(cuIpcGetMemHandle_f(&memHandle, (CUdeviceptr)ptr));
    memcpy(out_buffer, memHandle.reserved, CU_IPC_HANDLE_SIZE);
}

void* cuda_ipc_open_mem_handle(void* context, char* handle) {
    ContextGuard guard(context);

    CUipcMemHandle memHandle;
    memcpy(memHandle.reserved, handle, CU_IPC_HANDLE_SIZE);

    CUdeviceptr device_ptr;

    // Strangely, the CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS flag is required
    if check_cu(cuIpcOpenMemHandle_f(&device_ptr, memHandle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS))
        return (void*) device_ptr;
    else
        return NULL;
}

void cuda_ipc_close_mem_handle(void* ptr) {
    check_cu(cuIpcCloseMemHandle_f((CUdeviceptr) ptr));
}

void cuda_ipc_get_event_handle(void* context, void* event, char* out_buffer) {
    ContextGuard guard(context);

    CUipcEventHandle eventHandle;
    check_cu(cuIpcGetEventHandle_f(&eventHandle, static_cast<CUevent>(event)));
    memcpy(out_buffer, eventHandle.reserved, CU_IPC_HANDLE_SIZE);
}

void* cuda_ipc_open_event_handle(void* context, char* handle) {
    ContextGuard guard(context);

    CUipcEventHandle eventHandle;
    memcpy(eventHandle.reserved, handle, CU_IPC_HANDLE_SIZE);

    CUevent event;

    if (check_cu(cuIpcOpenEventHandle_f(&event, eventHandle)))
        return event;
    else
        return NULL;
}

void* cuda_stream_create(void* context, int priority)
{
    ContextGuard guard(context, true);

    CUstream stream;
    if (check_cu(cuStreamCreateWithPriority_f(&stream, CU_STREAM_DEFAULT, priority)))
    {
        cuda_stream_register(WP_CURRENT_CONTEXT, stream);
        return stream;
    }
    else
        return NULL;
}

void cuda_stream_destroy(void* context, void* stream)
{
    if (!stream)
        return;

    cuda_stream_unregister(context, stream);

    check_cu(cuStreamDestroy_f(static_cast<CUstream>(stream)));
}

int cuda_stream_query(void* stream)
{
    CUresult res =  cuStreamQuery_f(static_cast<CUstream>(stream));

    if ((res != CUDA_SUCCESS) && (res != CUDA_ERROR_NOT_READY))
    {
        // Abnormal, print out error
        check_cu(res);
    }

    return res;
}

void cuda_stream_register(void* context, void* stream)
{
    if (!stream)
        return;

    ContextGuard guard(context);

    // populate stream info
    StreamInfo& stream_info = g_streams[static_cast<CUstream>(stream)];
    check_cu(cuEventCreate_f(&stream_info.cached_event, CU_EVENT_DISABLE_TIMING));
}

void cuda_stream_unregister(void* context, void* stream)
{
    if (!stream)
        return;

    CUstream cuda_stream = static_cast<CUstream>(stream);
    
    StreamInfo* stream_info = get_stream_info(cuda_stream);
    if (stream_info)
    {
        // release stream info
        check_cu(cuEventDestroy_f(stream_info->cached_event));
        g_streams.erase(cuda_stream);
    }

    // make sure we don't leave dangling references to this stream
    ContextInfo* context_info = get_context_info(context);
    if (context_info)
    {
        if (cuda_stream == context_info->stream)
            context_info->stream = NULL;
    }
}

void* cuda_stream_get_current()
{
    return get_current_stream();
}

void cuda_stream_synchronize(void* stream)
{
    check_cu(cuStreamSynchronize_f(static_cast<CUstream>(stream)));
}

void cuda_stream_wait_event(void* stream, void* event)
{
    check_cu(cuStreamWaitEvent_f(static_cast<CUstream>(stream), static_cast<CUevent>(event), 0));
}

void cuda_stream_wait_stream(void* stream, void* other_stream, void* event)
{
    check_cu(cuEventRecord_f(static_cast<CUevent>(event), static_cast<CUstream>(other_stream)));
    check_cu(cuStreamWaitEvent_f(static_cast<CUstream>(stream), static_cast<CUevent>(event), 0));
}

int cuda_stream_is_capturing(void* stream)
{
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    check_cuda(cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &status));
    
    return int(status != cudaStreamCaptureStatusNone);
}

uint64_t cuda_stream_get_capture_id(void* stream)
{
    return get_capture_id(static_cast<CUstream>(stream));
}

int cuda_stream_get_priority(void* stream)
{
    int priority = 0;
    check_cuda(cuStreamGetPriority_f(static_cast<CUstream>(stream), &priority));

    return priority;
}

void* cuda_event_create(void* context, unsigned flags)
{
    ContextGuard guard(context, true);

    CUevent event;
    if (check_cu(cuEventCreate_f(&event, flags)))
        return event;
    else
        return NULL;
}

void cuda_event_destroy(void* event)
{
    check_cu(cuEventDestroy_f(static_cast<CUevent>(event)));
}

int cuda_event_query(void* event)
{
    CUresult res = cuEventQuery_f(static_cast<CUevent>(event));

    if ((res != CUDA_SUCCESS) && (res != CUDA_ERROR_NOT_READY))
    {
        // Abnormal, print out error
        check_cu(res);
    }

    return res;
}

void cuda_event_record(void* event, void* stream, bool timing)
{
    if (timing && !g_captures.empty() && cuda_stream_is_capturing(stream))
    {
        // record timing event during graph capture
        check_cu(cuEventRecordWithFlags_f(static_cast<CUevent>(event), static_cast<CUstream>(stream), CU_EVENT_RECORD_EXTERNAL));
    }
    else
    {
        check_cu(cuEventRecord_f(static_cast<CUevent>(event), static_cast<CUstream>(stream)));
    }
}

void cuda_event_synchronize(void* event)
{
    check_cu(cuEventSynchronize_f(static_cast<CUevent>(event)));
}

float cuda_event_elapsed_time(void* start_event, void* end_event)
{
    float elapsed = 0.0f;
    cudaEvent_t start = static_cast<cudaEvent_t>(start_event);
    cudaEvent_t end = static_cast<cudaEvent_t>(end_event);
    check_cuda(cudaEventElapsedTime(&elapsed, start, end));
    return elapsed;
}

bool cuda_graph_begin_capture(void* context, void* stream, int external)
{
    ContextGuard guard(context);

    CUstream cuda_stream = static_cast<CUstream>(stream);
    StreamInfo* stream_info = get_stream_info(cuda_stream);
    if (!stream_info)
    {
        wp::set_error_string("Warp error: unknown stream");
        return false;
    }

    if (external)
    {
        // if it's an external capture, make sure it's already active so we can get the capture id
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (!check_cuda(cudaStreamIsCapturing(cuda_stream, &status)))
            return false;
        if (status != cudaStreamCaptureStatusActive)
        {
            wp::set_error_string("Warp error: stream is not capturing");
            return false;
        }
    }
    else
    {
        // start the capture
        if (!check_cuda(cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeGlobal)))
            return false;
    }

    uint64_t capture_id = get_capture_id(cuda_stream);

    CaptureInfo* capture = new CaptureInfo();
    capture->stream = cuda_stream;
    capture->id = capture_id;
    capture->external = bool(external);

    // update stream info
    stream_info->capture = capture;

    // add to known captures
    g_captures[capture_id] = capture;

    return true;
}

bool cuda_graph_end_capture(void* context, void* stream, void** graph_ret)
{
    ContextGuard guard(context);

    // check if this is a known stream
    CUstream cuda_stream = static_cast<CUstream>(stream);
    StreamInfo* stream_info = get_stream_info(cuda_stream);
    if (!stream_info)
    {
        wp::set_error_string("Warp error: unknown capture stream");
        return false;
    }

    // check if this stream was used to start a capture
    CaptureInfo* capture = stream_info->capture;
    if (!capture)
    {
        wp::set_error_string("Warp error: stream has no capture started");
        return false;
    }

    // get capture info
    bool external = capture->external;
    uint64_t capture_id = capture->id;

    // clear capture info
    stream_info->capture = NULL;
    g_captures.erase(capture_id);
    delete capture;

    // a lambda to clean up on exit in case of error
    auto clean_up = [cuda_stream, capture_id, external]()
    {
        // unreference outstanding graph allocs so that they will be released with the user reference
        for (auto it = g_graph_allocs.begin(); it != g_graph_allocs.end(); ++it)
        {
            GraphAllocInfo& alloc_info = it->second;
            if (alloc_info.capture_id == capture_id)
                alloc_info.graph_destroyed = true;
        }

        // make sure we terminate the capture
        if (!external)
        {
            cudaGraph_t graph = NULL;
            cudaStreamEndCapture(cuda_stream, &graph);
            cudaGetLastError();
        }
    };

    // get captured graph without ending the capture in case it is external
    cudaGraph_t graph = get_capture_graph(cuda_stream);
    if (!graph)
    {
        clean_up();
        return false;
    }
    
    // ensure that all forked streams are joined to the main capture stream by manually
    // adding outstanding capture dependencies gathered from the graph leaf nodes
    std::vector<cudaGraphNode_t> stream_dependencies;
    std::vector<cudaGraphNode_t> leaf_nodes;
    if (get_capture_dependencies(cuda_stream, stream_dependencies) && get_graph_leaf_nodes(graph, leaf_nodes))
    {
        // compute set difference to get unjoined dependencies
        std::vector<cudaGraphNode_t> unjoined_dependencies;
        std::sort(stream_dependencies.begin(), stream_dependencies.end());
        std::sort(leaf_nodes.begin(), leaf_nodes.end());
        std::set_difference(leaf_nodes.begin(), leaf_nodes.end(),
                            stream_dependencies.begin(), stream_dependencies.end(),
                            std::back_inserter(unjoined_dependencies));
        if (!unjoined_dependencies.empty())
        {
            check_cu(cuStreamUpdateCaptureDependencies_f(cuda_stream, unjoined_dependencies.data(), unjoined_dependencies.size(),
                                                         CU_STREAM_ADD_CAPTURE_DEPENDENCIES));
            // ensure graph is still valid
            if (get_capture_graph(cuda_stream) != graph)
            {
                clean_up();
                return false;
            }
        }
    }

    // check if this graph has unfreed allocations, which require special handling
    std::vector<void*> unfreed_allocs;
    for (auto it = g_graph_allocs.begin(); it != g_graph_allocs.end(); ++it)
    {
        GraphAllocInfo& alloc_info = it->second;
        if (alloc_info.capture_id == capture_id)
            unfreed_allocs.push_back(it->first);
    }

    if (!unfreed_allocs.empty())
    {
        // Create a user object that will notify us when the instantiated graph is destroyed.
        // This works for external captures also, since we wouldn't otherwise know when
        // the externally-created graph instance gets deleted.
        // This callback is guaranteed to arrive after the graph has finished executing on the device,
        // not necessarily when cudaGraphExecDestroy() is called.
        GraphInfo* graph_info = new GraphInfo;
        graph_info->unfreed_allocs = unfreed_allocs;
        cudaUserObject_t user_object;
        check_cuda(cudaUserObjectCreate(&user_object, graph_info, on_graph_destroy, 1, cudaUserObjectNoDestructorSync));
        check_cuda(cudaGraphRetainUserObject(graph, user_object, 1, cudaGraphUserObjectMove));

        // ensure graph is still valid
        if (get_capture_graph(cuda_stream) != graph)
        {
            clean_up();
            return false;
        }
    }

    // for external captures, we don't instantiate the graph ourselves, so we're done
    if (external)
        return true;

    cudaGraphExec_t graph_exec = NULL;

    // end the capture
    if (!check_cuda(cudaStreamEndCapture(cuda_stream, &graph)))
        return false;

    // enable to create debug GraphVis visualization of graph
    // cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose);
    
    // can use after CUDA 11.4 to permit graphs to capture cudaMallocAsync() operations
    if (!check_cuda(cudaGraphInstantiateWithFlags(&graph_exec, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch)))
        return false;

    // free source graph
    check_cuda(cudaGraphDestroy(graph));

    // process deferred free list if no more captures are ongoing
    if (g_captures.empty())
    {
        free_deferred_allocs();
        unload_deferred_modules();
    }

    if (graph_ret)
        *graph_ret = graph_exec;

    return true;
}

bool cuda_graph_launch(void* graph_exec, void* stream)
{
    // TODO: allow naming graphs?
    begin_cuda_range(WP_TIMING_GRAPH, stream, get_stream_context(stream), "graph");

    bool result = check_cuda(cudaGraphLaunch((cudaGraphExec_t)graph_exec, (cudaStream_t)stream));

    end_cuda_range(WP_TIMING_GRAPH, stream);

    return result;
}

bool cuda_graph_destroy(void* context, void* graph_exec)
{
    ContextGuard guard(context);

    return check_cuda(cudaGraphExecDestroy((cudaGraphExec_t)graph_exec));
}

bool write_file(const char* data, size_t size, std::string filename, const char* mode)
{
    const bool print_debug = (std::getenv("WARP_DEBUG") != nullptr);
    if (print_debug) 
    {
        printf("Writing %zu B to %s (%s)\n", size, filename.c_str(), mode);
    }
    FILE* file = fopen(filename.c_str(), mode);
    if (file)
    {
        if (fwrite(data, 1, size, file) != size) {
            fprintf(stderr, "Warp error: Failed to write to output file '%s'\n", filename.c_str());
            return false;
        }
        fclose(file);
        return true;
    }
    else
    {
        fprintf(stderr, "Warp error: Failed to open file '%s'\n", filename.c_str());
        return false;
    }
}

#if WP_ENABLE_MATHDX
    bool check_nvjitlink_result(nvJitLinkHandle handle, nvJitLinkResult result, const char* file, int line)
    {
        if (result != NVJITLINK_SUCCESS) {
            fprintf(stderr, "nvJitLink error: %d on %s:%d\n", (int)result, file, line);
            size_t lsize;
            result = nvJitLinkGetErrorLogSize(handle, &lsize);
            if (result == NVJITLINK_SUCCESS && lsize > 0) {
                std::vector<char> log(lsize);
                result = nvJitLinkGetErrorLog(handle, log.data());
                if (result == NVJITLINK_SUCCESS) {
                    fprintf(stderr, "%s\n", log.data());
                }
            }
            return false;
        } else {
            return true;
        }
    }
#endif

size_t cuda_compile_program(const char* cuda_src, const char* program_name, int arch, const char* include_dir, int num_cuda_include_dirs, const char** cuda_include_dirs, bool debug, bool verbose, bool verify_fp, bool fast_math, bool fuse_fp, bool lineinfo, const char* output_path, size_t num_ltoirs, char** ltoirs, size_t* ltoir_sizes, int* ltoir_input_types)
{
    // use file extension to determine whether to output PTX or CUBIN
    const char* output_ext = strrchr(output_path, '.');
    bool use_ptx = output_ext && strcmp(output_ext + 1, "ptx") == 0;
    const bool print_debug = (std::getenv("WARP_DEBUG") != nullptr);

    // check include dir path len (path + option)
    const int max_path = 4096 + 16;
    if (strlen(include_dir) > max_path)
    {
        fprintf(stderr, "Warp error: Include path too long\n");
        return size_t(-1);
    }

    if (print_debug)
    {
        // Not available in all nvJitLink versions
        // unsigned major = 0;
        // unsigned minor = 0;
        // nvJitLinkVersion(&major, &minor);
        // printf("nvJitLink version %d.%d\n", major, minor);
        int major = 0;
        int minor = 0;
        nvrtcVersion(&major, &minor);
        printf("NVRTC version %d.%d\n", major, minor);
    }

    char include_opt[max_path];
    strcpy(include_opt, "--include-path=");
    strcat(include_opt, include_dir);

    const int max_arch = 128;
    char arch_opt[max_arch];
    char arch_opt_lto[max_arch];

    if (use_ptx)
    {
        snprintf(arch_opt, max_arch, "--gpu-architecture=compute_%d", arch);
        snprintf(arch_opt_lto, max_arch, "-arch=compute_%d", arch);
    }
    else
    {
        snprintf(arch_opt, max_arch, "--gpu-architecture=sm_%d", arch);
        snprintf(arch_opt_lto, max_arch, "-arch=sm_%d", arch);
    }

    std::vector<const char*> opts;
    opts.push_back(arch_opt);
    opts.push_back(include_opt);
    opts.push_back("--std=c++17");
    
    if (debug)
    {
        opts.push_back("--define-macro=_DEBUG");
        opts.push_back("--generate-line-info");

        // disabling since it causes issues with `Unresolved extern function 'cudaGetParameterBufferV2'
        //opts.push_back("--device-debug");
    }
    else
    {
        opts.push_back("--define-macro=NDEBUG");

        if (lineinfo)
            opts.push_back("--generate-line-info");
    }

    if (verify_fp)
        opts.push_back("--define-macro=WP_VERIFY_FP");
    else
        opts.push_back("--undefine-macro=WP_VERIFY_FP");

#if WP_ENABLE_MATHDX
    opts.push_back("--define-macro=WP_ENABLE_MATHDX=1");
#else
    opts.push_back("--define-macro=WP_ENABLE_MATHDX=0");
#endif
    
    if (fast_math)
        opts.push_back("--use_fast_math");

    if (fuse_fp)
        opts.push_back("--fmad=true");
    else
        opts.push_back("--fmad=false");

    std::vector<std::string> cuda_include_opt;
    for(int i = 0; i < num_cuda_include_dirs; i++)
    {
        cuda_include_opt.push_back(std::string("--include-path=") + cuda_include_dirs[i]);
        opts.push_back(cuda_include_opt.back().c_str());
    }

    opts.push_back("--device-as-default-execution-space");
    opts.push_back("--extra-device-vectorization");
    opts.push_back("--restrict");

    if (num_ltoirs > 0)
    {
        opts.push_back("-dlto");
        opts.push_back("--relocatable-device-code=true");
    }

    nvrtcProgram prog;
    nvrtcResult res;

    res = nvrtcCreateProgram(
        &prog,         // prog
        cuda_src,      // buffer
        program_name,  // name
        0,             // numHeaders
        NULL,          // headers
        NULL);         // includeNames

    if (!check_nvrtc(res))
        return size_t(res);

    if (print_debug) 
    {
        printf("NVRTC options:\n");
        for(auto o: opts) {
            printf("%s\n", o);
        }
    }
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
    if(num_ltoirs > 0) {
#if WP_ENABLE_MATHDX
        get_output_size = nvrtcGetLTOIRSize;
        get_output_data = nvrtcGetLTOIR;
        output_mode = "wb";
#else
        fprintf(stderr, "Warp error: num_ltoirs > 0 but Warp was not built with MathDx support\n");
        return size_t(-1);
#endif
    }
    else if (use_ptx)
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

            // LTOIR case - need an extra step
            if (num_ltoirs > 0) 
            {
#if WP_ENABLE_MATHDX
                if(ltoir_input_types == nullptr || ltoirs == nullptr || ltoir_sizes == nullptr) {
                    fprintf(stderr, "Warp error: num_ltoirs > 0 but ltoir_input_types, ltoirs or ltoir_sizes are NULL\n");
                    return size_t(-1);
                }
                nvJitLinkHandle handle = nullptr;
                std::vector<const char *> lopts = {"-dlto", arch_opt_lto};
                if (use_ptx) {
                    lopts.push_back("-ptx");
                }
                if (print_debug) 
                {
                    printf("nvJitLink options:\n");
                    for(auto o: lopts) {
                        printf("%s\n", o);
                    }
                }
                if(!check_nvjitlink(handle, nvJitLinkCreate(&handle, lopts.size(), lopts.data())))
                {
                    res = nvrtcResult(-1);
                }
                // Links
                if(std::getenv("WARP_DUMP_LTOIR"))
                {
                    write_file(output.data(), output.size(), "nvrtc_output.ltoir", "wb");
                }
                if(!check_nvjitlink(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, output.data(), output.size(), "nvrtc_output"))) // NVRTC business
                {
                    res = nvrtcResult(-1);
                }
                for(size_t ltoidx = 0; ltoidx < num_ltoirs; ltoidx++) 
                {
                    nvJitLinkInputType input_type = static_cast<nvJitLinkInputType>(ltoir_input_types[ltoidx]);
                    const char* ext = ".unknown";
                    switch(input_type) {
                        case NVJITLINK_INPUT_CUBIN:
                            ext = ".cubin";
                            break;
                        case NVJITLINK_INPUT_LTOIR:
                            ext = ".ltoir";
                            break;
                        case NVJITLINK_INPUT_FATBIN:
                            ext = ".fatbin";
                            break;
                        default:
                            break;
                    }
                    if(std::getenv("WARP_DUMP_LTOIR"))
                    {
                        write_file(ltoirs[ltoidx], ltoir_sizes[ltoidx], std::string("lto_online_") + std::to_string(ltoidx) + ext, "wb");
                    }
                    if(!check_nvjitlink(handle, nvJitLinkAddData(handle, input_type, ltoirs[ltoidx], ltoir_sizes[ltoidx], "lto_online"))) // External LTOIR
                    {
                        res = nvrtcResult(-1);
                    }
                }
                if(!check_nvjitlink(handle, nvJitLinkComplete(handle)))
                {
                    res = nvrtcResult(-1);
                } 
                else 
                {
                    if(use_ptx) 
                    {
                        size_t ptx_size = 0;
                        check_nvjitlink(handle, nvJitLinkGetLinkedPtxSize(handle, &ptx_size));
                        std::vector<char> ptx(ptx_size);
                        check_nvjitlink(handle, nvJitLinkGetLinkedPtx(handle, ptx.data()));
                        output = ptx;
                    } 
                    else
                    {
                        size_t cubin_size = 0;
                        check_nvjitlink(handle, nvJitLinkGetLinkedCubinSize(handle, &cubin_size));
                        std::vector<char> cubin(cubin_size);
                        check_nvjitlink(handle, nvJitLinkGetLinkedCubin(handle, cubin.data()));
                        output = cubin;
                    }
                }
                check_nvjitlink(handle, nvJitLinkDestroy(&handle));
#else
                fprintf(stderr, "Warp error: num_ltoirs > 0 but Warp was not built with MathDx support\n");
                return size_t(-1);
#endif
            }

            if(!write_file(output.data(), output.size(), output_path, output_mode)) {
                res = nvrtcResult(-1);
            }
        }
    }

    check_nvrtc(nvrtcDestroyProgram(&prog));

    return res;
}

#if WP_ENABLE_MATHDX
    bool check_cufftdx_result(commondxStatusType result, const char* file, int line)
    {
        if (result != commondxStatusType::COMMONDX_SUCCESS) {
            fprintf(stderr, "libmathdx cuFFTDx error: %d on %s:%d\n", (int)result, file, line);
            return false;
        } else {
            return true;
        }
    }

    bool check_cublasdx_result(commondxStatusType result, const char* file, int line)
    {
        if (result != commondxStatusType::COMMONDX_SUCCESS) {
            fprintf(stderr, "libmathdx cuBLASDx error: %d on %s:%d\n", (int)result, file, line);
            return false;
        } else {
            return true;
        }
    }

    bool check_cusolver_result(commondxStatusType result, const char* file, int line) 
    {
        if (result != commondxStatusType::COMMONDX_SUCCESS) {
            fprintf(stderr, "libmathdx cuSOLVER error: %d on %s:%d\n", (int)result, file, line);
            return false;
        } else {
            return true;
        }
    }

    bool cuda_compile_fft(const char* ltoir_output_path, const char* symbol_name, int num_include_dirs, const char** include_dirs, const char* mathdx_include_dir, int arch, int size, int elements_per_thread, int direction, int precision, int* shared_memory_size)
    {

        CHECK_ANY(ltoir_output_path != nullptr);
        CHECK_ANY(symbol_name != nullptr);
        CHECK_ANY(shared_memory_size != nullptr);
        // Includes currently unused
        CHECK_ANY(include_dirs == nullptr);
        CHECK_ANY(mathdx_include_dir == nullptr);
        CHECK_ANY(num_include_dirs == 0);

        bool res = true;
        cufftdxHandle h;
        CHECK_CUFFTDX(cufftdxCreate(&h));

        // CUFFTDX_API_BLOCK_LMEM means each thread starts with a subset of the data
        CHECK_CUFFTDX(cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_API, cufftdxApi::CUFFTDX_API_BLOCK_LMEM));
        CHECK_CUFFTDX(cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK));
        CHECK_CUFFTDX(cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_SIZE, (long long)size));
        CHECK_CUFFTDX(cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_DIRECTION, (cufftdxDirection)direction));
        CHECK_CUFFTDX(cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_PRECISION, (commondxPrecision)precision));
        CHECK_CUFFTDX(cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_SM, (long long)(arch * 10)));
        CHECK_CUFFTDX(cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_ELEMENTS_PER_THREAD, (long long)(elements_per_thread)));
        CHECK_CUFFTDX(cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_FFTS_PER_BLOCK, 1));

        CHECK_CUFFTDX(cufftdxSetOptionStr(h, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, symbol_name));

        size_t lto_size = 0;
        CHECK_CUFFTDX(cufftdxGetLTOIRSize(h, &lto_size));

        std::vector<char> lto(lto_size);
        CHECK_CUFFTDX(cufftdxGetLTOIR(h, lto.size(), lto.data()));    

        long long int smem = 0;
        CHECK_CUFFTDX(cufftdxGetTraitInt64(h, cufftdxTraitType::CUFFTDX_TRAIT_SHARED_MEMORY_SIZE, &smem));
        *shared_memory_size = (int)smem;

        if(!write_file(lto.data(), lto.size(), ltoir_output_path, "wb")) {
            res = false;
        }

        CHECK_CUFFTDX(cufftdxDestroy(h));

        return res;
    }

    bool cuda_compile_dot(const char* ltoir_output_path, const char* symbol_name, int num_include_dirs, const char** include_dirs, const char* mathdx_include_dir, int arch, int M, int N, int K, int precision_A, int precision_B, int precision_C, int type, int arrangement_A, int arrangement_B, int arrangement_C, int num_threads)
    {

        CHECK_ANY(ltoir_output_path != nullptr);
        CHECK_ANY(symbol_name != nullptr);
        // Includes currently unused
        CHECK_ANY(include_dirs == nullptr);
        CHECK_ANY(mathdx_include_dir == nullptr);
        CHECK_ANY(num_include_dirs == 0);

        bool res = true;
        cublasdxHandle h;
        CHECK_CUBLASDX(cublasdxCreate(&h));

        CHECK_CUBLASDX(cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_FUNCTION, cublasdxFunction::CUBLASDX_FUNCTION_MM));
        CHECK_CUBLASDX(cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK));
        CHECK_CUBLASDX(cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_API, cublasdxApi::CUBLASDX_API_BLOCK_SMEM));
        std::array<long long int, 3> precisions = {precision_A, precision_B, precision_C};
        CHECK_CUBLASDX(cublasdxSetOperatorInt64Array(h, cublasdxOperatorType::CUBLASDX_OPERATOR_PRECISION, 3, precisions.data()));
        CHECK_CUBLASDX(cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_SM, (long long)(arch * 10)));
        CHECK_CUBLASDX(cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_TYPE, (cublasdxType)type));
        std::array<long long int, 3> block_dim = {num_threads, 1, 1};
        CHECK_CUBLASDX(cublasdxSetOperatorInt64Array(h, cublasdxOperatorType::CUBLASDX_OPERATOR_BLOCK_DIM, block_dim.size(), block_dim.data()));
        std::array<long long int, 3> size = {M, N, K};
        CHECK_CUBLASDX(cublasdxSetOperatorInt64Array(h, cublasdxOperatorType::CUBLASDX_OPERATOR_SIZE, size.size(), size.data()));
        std::array<long long int, 3> arrangement = {arrangement_A, arrangement_B, arrangement_C};
        CHECK_CUBLASDX(cublasdxSetOperatorInt64Array(h, cublasdxOperatorType::CUBLASDX_OPERATOR_ARRANGEMENT, arrangement.size(), arrangement.data()));
        
        CHECK_CUBLASDX(cublasdxSetOptionStr(h, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, symbol_name));

        size_t lto_size = 0;
        CHECK_CUBLASDX(cublasdxGetLTOIRSize(h, &lto_size));

        std::vector<char> lto(lto_size);
        CHECK_CUBLASDX(cublasdxGetLTOIR(h, lto.size(), lto.data()));    

        if(!write_file(lto.data(), lto.size(), ltoir_output_path, "wb")) {
            res = false;
        }

        CHECK_CUBLASDX(cublasdxDestroy(h));

        return res;
    }

    bool cuda_compile_solver(const char* fatbin_output_path, const char* ltoir_output_path, const char* symbol_name, int num_include_dirs, const char** include_dirs, const char* mathdx_include_dir, int arch, int M, int N, int function, int precision, int fill_mode, int num_threads)
    {

        CHECK_ANY(ltoir_output_path != nullptr);
        CHECK_ANY(symbol_name != nullptr);
        CHECK_ANY(mathdx_include_dir == nullptr);
        CHECK_ANY(num_include_dirs == 0);
        CHECK_ANY(include_dirs == nullptr);

        bool res = true;

        cusolverHandle h { 0 };
        CHECK_CUSOLVER(cusolverCreate(&h));
        long long int size[2] = {M, N};
        long long int block_dim[3] = {num_threads, 1, 1};
        CHECK_CUSOLVER(cusolverSetOperatorInt64Array(h, cusolverOperatorType::CUSOLVER_OPERATOR_SIZE, 2, size));
        CHECK_CUSOLVER(cusolverSetOperatorInt64Array(h, cusolverOperatorType::CUSOLVER_OPERATOR_BLOCK_DIM, 3, block_dim));
        CHECK_CUSOLVER(cusolverSetOperatorInt64(h, cusolverOperatorType::CUSOLVER_OPERATOR_TYPE, cusolverType::CUSOLVER_TYPE_REAL));
        CHECK_CUSOLVER(cusolverSetOperatorInt64(h, cusolverOperatorType::CUSOLVER_OPERATOR_API, cusolverApi::CUSOLVER_API_BLOCK_SMEM));
        CHECK_CUSOLVER(cusolverSetOperatorInt64(h, cusolverOperatorType::CUSOLVER_OPERATOR_FUNCTION, (cusolverFunction)function));
        CHECK_CUSOLVER(cusolverSetOperatorInt64(h, cusolverOperatorType::CUSOLVER_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK));
        CHECK_CUSOLVER(cusolverSetOperatorInt64(h, cusolverOperatorType::CUSOLVER_OPERATOR_PRECISION, (commondxPrecision)precision));
        CHECK_CUSOLVER(cusolverSetOperatorInt64(h, cusolverOperatorType::CUSOLVER_OPERATOR_FILL_MODE, (cusolverFillMode)fill_mode));
        CHECK_CUSOLVER(cusolverSetOperatorInt64(h, cusolverOperatorType::CUSOLVER_OPERATOR_SM, (long long)(arch * 10)));
        
        CHECK_CUSOLVER(cusolverSetOptionStr(h, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, symbol_name));

        size_t lto_size = 0;
        CHECK_CUSOLVER(cusolverGetLTOIRSize(h, &lto_size));

        std::vector<char> lto(lto_size);
        CHECK_CUSOLVER(cusolverGetLTOIR(h, lto.size(), lto.data()));   

        // This fatbin is universal, ie it is the same for any instantiations of a cusolver device function
        size_t fatbin_size = 0;
        CHECK_CUSOLVER(cusolverGetUniversalFATBINSize(h, &fatbin_size));

        std::vector<char> fatbin(fatbin_size);
        CHECK_CUSOLVER(cusolverGetUniversalFATBIN(h, fatbin.size(), fatbin.data()));     

        if(!write_file(lto.data(), lto.size(), ltoir_output_path, "wb")) {
            res = false;
        }

        if(!write_file(fatbin.data(), fatbin.size(), fatbin_output_path, "wb")) {
            res = false;
        }

        CHECK_CUSOLVER(cusolverDestroy(h));

        return res;
    }

#endif

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
    // ensure there are no graph captures in progress
    if (g_captures.empty())
    {
        ContextGuard guard(context);
        check_cu(cuModuleUnload_f((CUmodule)module));
    }
    else
    {
        // defer until graph capture completes
        ModuleInfo module_info;
        module_info.context = context ? context : get_current_context();
        module_info.module = module;
        g_deferred_module_list.push_back(module_info);
    }
}


int cuda_get_max_shared_memory(void* context)
{
    ContextInfo* info = get_context_info(context);
    if (!info)
        return -1;

    int max_smem_bytes = info->device_info->max_smem_bytes;
    return max_smem_bytes;
}

bool cuda_configure_kernel_shared_memory(void* kernel, int size)
{
    int requested_smem_bytes = size;

    // configure shared memory 
    CUresult res = cuFuncSetAttribute_f((CUfunction)kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, requested_smem_bytes);
    if (res != CUDA_SUCCESS)
        return false;

    return true;
}

void* cuda_get_kernel(void* context, void* module, const char* name)
{
    ContextGuard guard(context);

    CUfunction kernel = NULL;
    if (!check_cu(cuModuleGetFunction_f(&kernel, (CUmodule)module, name)))
    {
        fprintf(stderr, "Warp CUDA error: Failed to lookup kernel function %s in module\n", name);
        return NULL;
    }

    g_kernel_names[kernel] = name;
    return kernel;
}

size_t cuda_launch_kernel(void* context, void* kernel, size_t dim, int max_blocks, int block_dim, int shared_memory_bytes, void** args, void* stream)
{
    ContextGuard guard(context);

    if (block_dim <= 0)
    {
#if defined(_DEBUG)
        fprintf(stderr, "Warp warning: Launch got block_dim %d. Setting to 256.\n", block_dim);
#endif
        block_dim = 256;
    }

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

    begin_cuda_range(WP_TIMING_KERNEL, stream, context, get_cuda_kernel_name(kernel));

    CUresult res = cuLaunchKernel_f(
        (CUfunction)kernel,
        grid_dim, 1, 1,
        block_dim, 1, 1,
        shared_memory_bytes,
        static_cast<CUstream>(stream),
        args,
        0);

    check_cu(res);

    end_cuda_range(WP_TIMING_KERNEL, stream);

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
    bool success = check_cu(cuGraphicsGLRegisterBuffer_f(resource, gl_buffer, flags));
    if (!success)
    {
        delete resource;
        return NULL;
    }

    return resource;
}

void cuda_graphics_unregister_resource(void* context, void* resource)
{
    ContextGuard guard(context);

    CUgraphicsResource *res = (CUgraphicsResource*)resource;
    check_cu(cuGraphicsUnregisterResource_f(*res));
    delete res;
}

void cuda_timing_begin(int flags)
{
    g_cuda_timing_state = new CudaTimingState(flags, g_cuda_timing_state);
}

int cuda_timing_get_result_count()
{
    if (g_cuda_timing_state)
        return int(g_cuda_timing_state->ranges.size());
    return 0;
}

void cuda_timing_end(timing_result_t* results, int size)
{
    if (!g_cuda_timing_state)
        return;

    // number of results to write to the user buffer
    int count = std::min(cuda_timing_get_result_count(), size);

    // compute timings and write results
    for (int i = 0; i < count; i++)
    {
        const CudaTimingRange& range = g_cuda_timing_state->ranges[i];
        timing_result_t& result = results[i];
        result.context = range.context;
        result.name = range.name;
        result.flag = range.flag;
        check_cuda(cudaEventElapsedTime(&result.elapsed, range.start, range.end));
    }

    // release events
    for (CudaTimingRange& range : g_cuda_timing_state->ranges)
    {
        check_cu(cuEventDestroy_f(range.start));
        check_cu(cuEventDestroy_f(range.end));
    }

    // restore previous state
    CudaTimingState* parent_state = g_cuda_timing_state->parent;
    delete g_cuda_timing_state;
    g_cuda_timing_state = parent_state;
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

//#include "spline.inl"
//#include "volume.inl"
