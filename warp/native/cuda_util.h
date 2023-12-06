/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "builtin.h"

#if WP_ENABLE_CUDA

#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>

#include <stdio.h>

#define check_cuda(code) (check_cuda_result(code, __FILE__, __LINE__))
#define check_cu(code) (check_cu_result(code, __FILE__, __LINE__))


#if defined(__CUDACC__)
#if _DEBUG   
    // helper for launching kernels (synchronize + error checking after each kernel)
    #define wp_launch_device(context, kernel, dim, args) { \
        if (dim) { \
        ContextGuard guard(context); \
        const int num_threads = 256; \
        const int num_blocks = (dim+num_threads-1)/num_threads; \
        kernel<<<num_blocks, 256, 0, (cudaStream_t)cuda_stream_get_current()>>>args; \
        check_cuda(cuda_context_check(WP_CURRENT_CONTEXT)); } }
#else
    // helper for launching kernels (no error checking)
    #define wp_launch_device(context, kernel, dim, args) { \
        if (dim) { \
        ContextGuard guard(context); \
        const int num_threads = 256; \
        const int num_blocks = (dim+num_threads-1)/num_threads; \
        kernel<<<num_blocks, 256, 0, (cudaStream_t)cuda_stream_get_current()>>>args; } }
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
CUresult cuDevicePrimaryCtxRetain_f(CUcontext* ctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease_f(CUdevice dev);
CUresult cuDeviceCanAccessPeer_f(int* can_access, CUdevice dev, CUdevice peer_dev);
CUresult cuCtxGetCurrent_f(CUcontext* ctx);
CUresult cuCtxSetCurrent_f(CUcontext ctx);
CUresult cuCtxPushCurrent_f(CUcontext ctx);
CUresult cuCtxPopCurrent_f(CUcontext* ctx);
CUresult cuCtxSynchronize_f();
CUresult cuCtxGetDevice_f(CUdevice* dev);
CUresult cuCtxCreate_f(CUcontext* ctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy_f(CUcontext ctx);
CUresult cuCtxEnablePeerAccess_f(CUcontext peer_ctx, unsigned int flags);
CUresult cuStreamCreate_f(CUstream* stream, unsigned int flags);
CUresult cuStreamDestroy_f(CUstream stream);
CUresult cuStreamSynchronize_f(CUstream stream);
CUresult cuStreamWaitEvent_f(CUstream stream, CUevent event, unsigned int flags);
CUresult cuEventCreate_f(CUevent* event, unsigned int flags);
CUresult cuEventDestroy_f(CUevent event);
CUresult cuEventRecord_f(CUevent event, CUstream stream);
CUresult cuModuleUnload_f(CUmodule hmod);
CUresult cuModuleLoadDataEx_f(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
CUresult cuModuleGetFunction_f(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult cuLaunchKernel_f(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
CUresult cuMemcpyPeerAsync_f(CUdeviceptr dst_ptr, CUcontext dst_ctx, CUdeviceptr src_ptr, CUcontext src_ctx, size_t n, CUstream stream);
CUresult cuGraphicsMapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream stream);
CUresult cuGraphicsUnmapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
CUresult cuGraphicsResourceGetMappedPointer_f(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource);
CUresult cuGraphicsGLRegisterBuffer_f(CUgraphicsResource *pCudaResource, unsigned int buffer, unsigned int flags);
CUresult cuGraphicsUnregisterResource_f(CUgraphicsResource resource);


bool init_cuda_driver();
bool is_cuda_driver_initialized();

bool check_cuda_result(cudaError_t code, const char* file, int line);
inline bool check_cuda_result(uint64_t code, const char* file, int line)
{
    return check_cuda_result(static_cast<cudaError_t>(code), file, line);
}

bool check_cu_result(CUresult result, const char* file, int line);


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
// This works in conjuntion with ContextGuards, which do nothing if the given context is NULL.
// Using this variable instead of passing NULL directly aids readability and makes the intent clear.
constexpr void* WP_CURRENT_CONTEXT = NULL;
