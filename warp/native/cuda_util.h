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

#pragma once

#include "builtin.h"

#if WP_ENABLE_CUDA

#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>

#include <stdio.h>

#include <vector>

#define check_cuda(code) (check_cuda_result(code, __FUNCTION__, __FILE__, __LINE__))
#define check_cu(code) (check_cu_result(code, __FUNCTION__, __FILE__, __LINE__))


#if defined(__CUDACC__)
#if _DEBUG   
    // helper for launching kernels (synchronize + error checking after each kernel)
    #define wp_launch_device(context, kernel, dim, args) { \
        if (dim) { \
        ContextGuard guard(context); \
        cudaStream_t stream = (cudaStream_t)cuda_stream_get_current(); \
        const int num_threads = 256; \
        const int num_blocks = (dim+num_threads-1)/num_threads; \
        begin_cuda_range(WP_TIMING_KERNEL_BUILTIN, stream, context, #kernel); \
        kernel<<<num_blocks, 256, 0, stream>>>args; \
        check_cuda(cuda_context_check(WP_CURRENT_CONTEXT)); \
        end_cuda_range(WP_TIMING_KERNEL_BUILTIN, stream); }}
#else
    // helper for launching kernels (no error checking)
    #define wp_launch_device(context, kernel, dim, args) { \
        if (dim) { \
        ContextGuard guard(context); \
        cudaStream_t stream = (cudaStream_t)cuda_stream_get_current(); \
        const int num_threads = 256; \
        const int num_blocks = (dim+num_threads-1)/num_threads; \
        begin_cuda_range(WP_TIMING_KERNEL_BUILTIN, stream, context, #kernel); \
        kernel<<<num_blocks, 256, 0, stream>>>args; \
        end_cuda_range(WP_TIMING_KERNEL_BUILTIN, stream); }}
#endif // _DEBUG
#endif // defined(__CUDACC__)


CUresult cuDriverGetVersion_f(int* version);
CUresult cuGetErrorName_f(CUresult result, const char** pstr);
CUresult cuGetErrorString_f(CUresult result, const char** pstr);
CUresult cuInit_f(unsigned int flags);
CUresult cuDeviceGet_f(CUdevice *dev, int ordinal);
CUresult cuDeviceGetCount_f(int* count);
CUresult cuDeviceGetName_f(char* name, int len, CUdevice dev);
CUresult cuDeviceGetAttribute_f(int* value, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDeviceGetUuid_f(CUuuid* uuid, CUdevice dev);
CUresult cuDevicePrimaryCtxRetain_f(CUcontext* ctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease_f(CUdevice dev);
CUresult cuDeviceCanAccessPeer_f(int* can_access, CUdevice dev, CUdevice peer_dev);
CUresult cuMemGetInfo_f(size_t* free, size_t* total);
CUresult cuCtxGetCurrent_f(CUcontext* ctx);
CUresult cuCtxSetCurrent_f(CUcontext ctx);
CUresult cuCtxPushCurrent_f(CUcontext ctx);
CUresult cuCtxPopCurrent_f(CUcontext* ctx);
CUresult cuCtxSynchronize_f();
CUresult cuCtxGetDevice_f(CUdevice* dev);
CUresult cuCtxCreate_f(CUcontext* ctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy_f(CUcontext ctx);
CUresult cuCtxEnablePeerAccess_f(CUcontext peer_ctx, unsigned int flags);
CUresult cuCtxDisablePeerAccess_f(CUcontext peer_ctx);
CUresult cuStreamCreate_f(CUstream* stream, unsigned int flags);
CUresult cuStreamDestroy_f(CUstream stream);
CUresult cuStreamQuery_f(CUstream stream);
CUresult cuStreamSynchronize_f(CUstream stream);
CUresult cuStreamWaitEvent_f(CUstream stream, CUevent event, unsigned int flags);
CUresult cuStreamGetCtx_f(CUstream stream, CUcontext* pctx);
CUresult cuStreamGetCaptureInfo_f(CUstream stream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out);
CUresult cuStreamUpdateCaptureDependencies_f(CUstream stream, CUgraphNode *dependencies, size_t numDependencies, unsigned int flags);
CUresult cuStreamCreateWithPriority_f(CUstream* phStream, unsigned int flags, int priority);
CUresult cuStreamGetPriority_f(CUstream hStream, int* priority);
CUresult cuEventCreate_f(CUevent* event, unsigned int flags);
CUresult cuEventDestroy_f(CUevent event);
CUresult cuEventQuery_f(CUevent event);
CUresult cuEventRecord_f(CUevent event, CUstream stream);
CUresult cuEventRecordWithFlags_f(CUevent event, CUstream stream, unsigned int flags);
CUresult cuEventSynchronize_f(CUevent event);
CUresult cuModuleUnload_f(CUmodule hmod);
CUresult cuModuleLoadDataEx_f(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
CUresult cuModuleGetFunction_f(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult cuLaunchKernel_f(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
CUresult cuMemcpyPeerAsync_f(CUdeviceptr dst_ptr, CUcontext dst_ctx, CUdeviceptr src_ptr, CUcontext src_ctx, size_t n, CUstream stream);
CUresult cuPointerGetAttribute_f(void* data, CUpointer_attribute attribute, CUdeviceptr ptr);
CUresult cuGraphicsMapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream stream);
CUresult cuGraphicsUnmapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
CUresult cuGraphicsResourceGetMappedPointer_f(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource);
CUresult cuGraphicsGLRegisterBuffer_f(CUgraphicsResource *pCudaResource, unsigned int buffer, unsigned int flags);
CUresult cuGraphicsUnregisterResource_f(CUgraphicsResource resource);
CUresult cuModuleGetGlobal_f(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name );
CUresult cuFuncSetAttribute_f(CUfunction hfunc, CUfunction_attribute attrib, int value);
CUresult cuIpcGetEventHandle_f(CUipcEventHandle *pHandle, CUevent event);
CUresult cuIpcOpenEventHandle_f(CUevent *phEvent, CUipcEventHandle handle);
CUresult cuIpcGetMemHandle_f(CUipcMemHandle *pHandle, CUdeviceptr dptr);
CUresult cuIpcOpenMemHandle_f(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int flags);
CUresult cuIpcCloseMemHandle_f(CUdeviceptr dptr);

bool init_cuda_driver();
bool is_cuda_driver_initialized();

bool check_cuda_result(cudaError_t code, const char* func, const char* file, int line);

inline bool check_cuda_result(uint64_t code, const char* func, const char* file, int line)
{
    return check_cuda_result(static_cast<cudaError_t>(code), func, file, line);
}

bool check_cu_result(CUresult result, const char* func, const char* file, int line);

inline uint64_t get_capture_id(CUstream stream)
{
    CUstreamCaptureStatus status;
    uint64_t id = 0;
    check_cu(cuStreamGetCaptureInfo_f(stream, &status, &id, NULL, NULL, NULL));
    return id;
}

inline CUgraph get_capture_graph(CUstream stream)
{
    CUstreamCaptureStatus status;
    CUgraph graph = NULL;
    check_cu(cuStreamGetCaptureInfo_f(stream, &status, NULL, &graph, NULL, NULL));
    return graph;
}

bool get_capture_dependencies(CUstream stream, std::vector<CUgraphNode>& dependencies_ret);

bool get_graph_leaf_nodes(cudaGraph_t graph, std::vector<cudaGraphNode_t>& leaf_nodes_ret);

inline CUcontext get_stream_context(CUstream stream)
{
    CUcontext context;
    if (check_cu(cuStreamGetCtx_f(stream, &context)))
        return context;
    else
        return NULL;
}

inline CUcontext get_stream_context(void* stream)
{
    return get_stream_context(static_cast<CUstream>(stream));
}


//
// Scoped CUDA context guard
//
// Behaviour on entry
// - If the given `context` is NULL, do nothing.
// - If the given `context` is the same as the current context, do nothing.
// - If the given `context` is different from the current context, make the given context current.
//
// Behaviour on exit
// - If the current context did not change on entry, do nothing.
// - If the `restore` flag was true on entry, make the previous context current.
//
// Default exit behaviour policy
// - If the `restore` flag is omitted on entry, fall back on the global `always_restore` flag.
// - This allows us to easily change the default behaviour of the guards.
//
class ContextGuard
{
public:
    // default policy for restoring contexts
    static bool always_restore;

    explicit ContextGuard(CUcontext context, bool restore=always_restore)
        : need_restore(false)
    {
        if (context)
        {
            if (check_cu(cuCtxGetCurrent_f(&prev_context)) && context != prev_context)
                need_restore = check_cu(cuCtxSetCurrent_f(context)) && restore;
        }
    }

    explicit ContextGuard(void* context, bool restore=always_restore)
        : ContextGuard(static_cast<CUcontext>(context), restore)
    {
    }

    ~ContextGuard()
    {
        if (need_restore)
            check_cu(cuCtxSetCurrent_f(prev_context));
    }

private:
    CUcontext prev_context;
    bool need_restore;
};


// CUDA timing range used during event-based timing
struct CudaTimingRange
{
    void* context;
    const char* name;
    int flag;
    CUevent start;
    CUevent end;
};

// Timing result used to pass timings to Python
struct timing_result_t
{
    void* context;
    const char* name;
    int flag;
    float elapsed;
};

struct CudaTimingState
{
    int flags;
    std::vector<CudaTimingRange> ranges;
    CudaTimingState* parent;

    CudaTimingState(int flags, CudaTimingState* parent)
        : flags(flags), parent(parent)
    {
    }
};

// timing flags
constexpr int WP_TIMING_KERNEL = 1;  // Warp kernel
constexpr int WP_TIMING_KERNEL_BUILTIN = 2;  // internal kernel
constexpr int WP_TIMING_MEMCPY = 4;  // memcpy operation
constexpr int WP_TIMING_MEMSET = 8;  // memset operation
constexpr int WP_TIMING_GRAPH = 16;  // graph launch

#define begin_cuda_range(_flag, _stream, _context, _name) \
    CudaTimingRange _timing_range; \
    bool _timing_enabled; \
    if ((g_cuda_timing_state->flags & _flag) && !cuda_stream_is_capturing(_stream)) { \
        ContextGuard guard(_context, true); \
        _timing_enabled = true; \
        _timing_range.context = _context ? _context : get_current_context(); \
        _timing_range.name = _name; \
        _timing_range.flag = _flag; \
        check_cu(cuEventCreate_f(&_timing_range.start, CU_EVENT_DEFAULT)); \
        check_cu(cuEventCreate_f(&_timing_range.end, CU_EVENT_DEFAULT)); \
        check_cu(cuEventRecord_f(_timing_range.start, static_cast<CUstream>(_stream))); \
    } else { \
        _timing_enabled = false; \
    }

#define end_cuda_range(_flag, _stream) \
    if (_timing_enabled) { \
        check_cu(cuEventRecord_f(_timing_range.end, static_cast<CUstream>(_stream))); \
        g_cuda_timing_state->ranges.push_back(_timing_range); \
    }

extern CudaTimingState* g_cuda_timing_state;


#else

typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;

class ContextGuard
{
public:
    explicit ContextGuard(CUcontext context, bool restore=false)
    {
        (void)context;
        (void)restore;
    }

    explicit ContextGuard(void* context, bool restore=false)
    {
        (void)context;
        (void)restore;
    }
};

#endif // WP_ENABLE_CUDA

// Pass this value to device functions as the `context` parameter to bypass unnecessary context management.
// This works in conjunction with ContextGuards, which do nothing if the given context is NULL.
// Using this variable instead of passing NULL directly aids readability and makes the intent clear.
constexpr void* WP_CURRENT_CONTEXT = NULL;
