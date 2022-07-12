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
    #define wp_launch_device(kernel, dim, args) { \
        if (dim) { \
        const int num_threads = 256; \
        const int num_blocks = (dim+num_threads-1)/num_threads; \
        kernel<<<num_blocks, 256, 0, (cudaStream_t)cuda_stream_get_current()>>>args; \
        check_cuda(cuda_check_device()); } }
#else
    // helper for launching kernels (no error checking)
    #define wp_launch_device(kernel, dim, args) { \
        if (dim) { \
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
bool check_cu_result(CUresult result, const char* file, int line);

class ContextGuard
{
public:
    explicit ContextGuard(CUcontext ctx)
        : need_pop(false)
    {
        if (ctx)
        {
            CUcontext current_ctx;
            if (check_cu(cuCtxGetCurrent_f(&current_ctx)) && ctx != current_ctx)
                need_pop = check_cu(cuCtxPushCurrent_f(ctx));
        }
    }

    ~ContextGuard()
    {
        if (need_pop)
        {
            CUcontext ctx;
            check_cu(cuCtxPopCurrent_f(&ctx));
        }
    }

private:
    bool need_pop;
};

#else

typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;

#endif // !WP_DISABLE_CUDA
