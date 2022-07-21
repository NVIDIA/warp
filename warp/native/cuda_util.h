/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#if !WP_DISABLE_CUDA

#include <cuda.h>
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


typedef CUresult CUDAAPI cuInit_t(unsigned int);
typedef CUresult CUDAAPI cuGetErrorName_t(CUresult result, const char** pstr);
typedef CUresult CUDAAPI cuGetErrorString_t(CUresult result, const char** pstr);

typedef CUresult CUDAAPI cuDeviceGet_t(CUdevice *dev, int ordinal);
typedef CUresult CUDAAPI cuDeviceGetCount_t(int* count);
typedef CUresult CUDAAPI cuDeviceGetName_t(char* name, int len, CUdevice dev);
typedef CUresult CUDAAPI cuDevicePrimaryCtxRetain_t(CUcontext* ctx, CUdevice dev);
typedef CUresult CUDAAPI cuDeviceGetAttribute_t(int* value, CUdevice_attribute attrib, CUdevice dev);
typedef CUresult CUDAAPI cuDeviceCanAccessPeer_t(int* can_access, CUdevice dev, CUdevice peer_dev);

typedef CUresult CUDAAPI cuCtxGetCurrent_t(CUcontext* ctx);
typedef CUresult CUDAAPI cuCtxSetCurrent_t(CUcontext ctx);
typedef CUresult CUDAAPI cuCtxPushCurrent_t(CUcontext ctx);
typedef CUresult CUDAAPI cuCtxPopCurrent_t(CUcontext* ctx);
typedef CUresult CUDAAPI cuCtxSynchronize_t();
typedef CUresult CUDAAPI cuCtxGetDevice_t(CUdevice* dev);
typedef CUresult CUDAAPI cuCtxCreate_t(CUcontext* ctx, unsigned int flags, CUdevice dev);
typedef CUresult CUDAAPI cuCtxDestroy_t(CUcontext ctx);
typedef CUresult CUDAAPI cuCtxEnablePeerAccess_t(CUcontext peer_ctx, unsigned int flags);

typedef CUresult CUDAAPI cuStreamCreate_t(CUstream* stream, unsigned int flags);
typedef CUresult CUDAAPI cuStreamDestroy_t(CUstream stream);
typedef CUresult CUDAAPI cuStreamSynchronize_t(CUstream stream);

typedef CUresult CUDAAPI cuModuleUnload_t(CUmodule hmod);
typedef CUresult CUDAAPI cuModuleLoadDataEx_t(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
typedef CUresult CUDAAPI cuModuleGetFunction_t(CUfunction *hfunc, CUmodule hmod, const char *name);

typedef CUresult CUDAAPI cuLaunchKernel_t(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);

typedef CUresult CUDAAPI cuMemcpyPeerAsync_t(CUdeviceptr dst_ptr, CUcontext dst_ctx, CUdeviceptr src_ptr, CUcontext src_ctx, size_t n, CUstream stream);

extern cuInit_t* cuInit_f;
extern cuGetErrorName_t* cuGetErrorName_f;
extern cuGetErrorString_t* cuGetErrorString_f;

extern cuDeviceGet_t* cuDeviceGet_f;
extern cuDeviceGetCount_t* cuDeviceGetCount_f;
extern cuDeviceGetName_t* cuDeviceGetName_f;
extern cuDevicePrimaryCtxRetain_t* cuDevicePrimaryCtxRetain_f;
extern cuDeviceGetAttribute_t* cuDeviceGetAttribute_f;
extern cuDeviceCanAccessPeer_t* cuDeviceCanAccessPeer_f;

extern cuCtxGetCurrent_t* cuCtxGetCurrent_f;
extern cuCtxSetCurrent_t* cuCtxSetCurrent_f;
extern cuCtxPushCurrent_t* cuCtxPushCurrent_f;
extern cuCtxPopCurrent_t* cuCtxPopCurrent_f;
extern cuCtxSynchronize_t* cuCtxSynchronize_f;
extern cuCtxGetDevice_t* cuCtxGetDevice_f;
extern cuCtxCreate_t* cuCtxCreate_f;
extern cuCtxDestroy_t* cuCtxDestroy_f;
extern cuCtxEnablePeerAccess_t* cuCtxEnablePeerAccess_f;

extern cuStreamCreate_t* cuStreamCreate_f;
extern cuStreamDestroy_t* cuStreamDestroy_f;
extern cuStreamSynchronize_t* cuStreamSynchronize_f;

extern cuModuleUnload_t* cuModuleUnload_f;
extern cuModuleLoadDataEx_t* cuModuleLoadDataEx_f;
extern cuModuleGetFunction_t* cuModuleGetFunction_f;

extern cuLaunchKernel_t* cuLaunchKernel_f;

extern cuMemcpyPeerAsync_t* cuMemcpyPeerAsync_f;

bool init_cuda_driver();

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

#endif // !WP_DISABLE_CUDA

// Pass this value to device functions as the `context` parameter to bypass unnecessary context management.
// This works in conjuntion with ContextGuards, which do nothing if the given context is NULL.
// Using this variable instead of passing NULL directly aids readability and makes the intent clear.
constexpr void* WP_CURRENT_CONTEXT = NULL;
