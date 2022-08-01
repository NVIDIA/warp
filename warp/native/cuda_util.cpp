/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#if !WP_DISABLE_CUDA

#include "cuda_util.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

static PFN_cuGetErrorName pfn_cuGetErrorName;
static PFN_cuGetErrorString pfn_cuGetErrorString;
static PFN_cuInit pfn_cuInit;
static PFN_cuDeviceGet pfn_cuDeviceGet;
static PFN_cuDeviceGetCount pfn_cuDeviceGetCount;
static PFN_cuDeviceGetName pfn_cuDeviceGetName;
static PFN_cuDeviceGetAttribute pfn_cuDeviceGetAttribute;
static PFN_cuDevicePrimaryCtxRetain pfn_cuDevicePrimaryCtxRetain;
static PFN_cuDeviceCanAccessPeer pfn_cuDeviceCanAccessPeer;
static PFN_cuCtxGetCurrent pfn_cuCtxGetCurrent;
static PFN_cuCtxSetCurrent pfn_cuCtxSetCurrent;
static PFN_cuCtxPushCurrent pfn_cuCtxPushCurrent;
static PFN_cuCtxPopCurrent pfn_cuCtxPopCurrent;
static PFN_cuCtxSynchronize pfn_cuCtxSynchronize;
static PFN_cuCtxGetDevice pfn_cuCtxGetDevice;
static PFN_cuCtxCreate pfn_cuCtxCreate;
static PFN_cuCtxDestroy pfn_cuCtxDestroy;
static PFN_cuCtxEnablePeerAccess pfn_cuCtxEnablePeerAccess;
static PFN_cuStreamCreate pfn_cuStreamCreate;
static PFN_cuStreamDestroy pfn_cuStreamDestroy;
static PFN_cuStreamSynchronize pfn_cuStreamSynchronize;
static PFN_cuModuleLoadDataEx pfn_cuModuleLoadDataEx;
static PFN_cuModuleUnload pfn_cuModuleUnload;
static PFN_cuModuleGetFunction pfn_cuModuleGetFunction;
static PFN_cuLaunchKernel pfn_cuLaunchKernel;
static PFN_cuMemcpyPeerAsync pfn_cuMemcpyPeerAsync;

bool ContextGuard::always_restore = false;


bool init_cuda_driver()
{
#if defined(_WIN32)
    static HMODULE hCudaDriver = LoadLibraryA("nvcuda.dll");
    if (hCudaDriver == NULL) {
        fprintf(stderr, "Warp CUDA Error: Could not open nvcuda.dll.\n");
        return false;
    }
    PFN_cuGetProcAddress pfn_cuGetProcAddress = (PFN_cuGetProcAddress)GetProcAddress(hCudaDriver, "cuGetProcAddress");
#elif defined(__linux__)
    static void* hCudaDriver = dlopen("libcuda.so", RTLD_NOW);
    if (hCudaDriver == NULL) {
        fprintf(stderr, "Warp CUDA Error: Could not open libcuda.so.\n");
        return false;
    }
    PFN_cuGetProcAddress pfn_cuGetProcAddress = (PFN_cuGetProcAddress)dlsym(hCudaDriver, "cuGetProcAddress");
#endif

    if (!pfn_cuGetProcAddress)
    {
        fprintf(stderr, "Warp CUDA error: Failed to get function cuGetProcAddress\n");
        return false;
    }

    bool success = true;

    {
        int version = CUDA_VERSION;
        cuuint64_t flags = CU_GET_PROC_ADDRESS_DEFAULT;

        success = success && check_cu(pfn_cuGetProcAddress("cuGetErrorString", &(void*&)pfn_cuGetErrorString, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuGetErrorName", &(void*&)pfn_cuGetErrorName, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuInit", &(void*&)pfn_cuInit, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuDeviceGet", &(void*&)pfn_cuDeviceGet, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuDeviceGetCount", &(void*&)pfn_cuDeviceGetCount, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuDeviceGetName", &(void*&)pfn_cuDeviceGetName, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuDeviceGetAttribute", &(void*&)pfn_cuDeviceGetAttribute, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuDevicePrimaryCtxRetain", &(void*&)pfn_cuDevicePrimaryCtxRetain, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuDeviceCanAccessPeer", &(void*&)pfn_cuDeviceCanAccessPeer, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuCtxSetCurrent", &(void*&)pfn_cuCtxSetCurrent, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuCtxGetCurrent", &(void*&)pfn_cuCtxGetCurrent, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuCtxPushCurrent", &(void*&)pfn_cuCtxPushCurrent, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuCtxPopCurrent", &(void*&)pfn_cuCtxPopCurrent, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuCtxSynchronize", &(void*&)pfn_cuCtxSynchronize, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuCtxGetDevice", &(void*&)pfn_cuCtxGetDevice, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuCtxCreate", &(void*&)pfn_cuCtxCreate, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuCtxDestroy", &(void*&)pfn_cuCtxDestroy, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuCtxEnablePeerAccess", &(void*&)pfn_cuCtxEnablePeerAccess, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuStreamCreate", &(void*&)pfn_cuStreamCreate, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuStreamDestroy", &(void*&)pfn_cuStreamDestroy, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuStreamSynchronize", &(void*&)pfn_cuStreamSynchronize, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuModuleLoadDataEx", &(void*&)pfn_cuModuleLoadDataEx, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuModuleUnload", &(void*&)pfn_cuModuleUnload, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuModuleGetFunction", &(void*&)pfn_cuModuleGetFunction, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuLaunchKernel", &(void*&)pfn_cuLaunchKernel, version, flags));
        success = success && check_cu(pfn_cuGetProcAddress("cuMemcpyPeerAsync", &(void*&)pfn_cuMemcpyPeerAsync, version, flags));
    }

    return success;
}


bool check_cuda_result(cudaError_t code, const char* file, int line)
{
    if (code == cudaSuccess)
        return true;

    fprintf(stderr, "Warp CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file, line);
    return false;
}

bool check_cu_result(CUresult result, const char* file, int line)
{
    if (result == CUDA_SUCCESS)
        return true;

    const char* errString = NULL;
    if (pfn_cuGetErrorString)
        pfn_cuGetErrorString(result, &errString);

    if (errString)
        fprintf(stderr, "Warp CUDA error %u: %s (%s:%d)\n", unsigned(result), errString, file, line);
    else
        fprintf(stderr, "Warp CUDA error %u (%s:%d)\n", unsigned(result), file, line);

    return false;
}


#define DRIVER_ENTRY_POINT_ERROR driver_entry_point_error(__FUNCTION__)

static CUresult driver_entry_point_error(const char* function)
{
    fprintf(stderr, "Warp CUDA error: Function %s: a suitable driver entry point was not found\n", function);
    return (CUresult)cudaErrorCallRequiresNewerDriver; // this matches what cudart would do
}

CUresult cuGetErrorName_f(CUresult result, const char** pstr)
{
    return pfn_cuGetErrorName ? pfn_cuGetErrorName(result, pstr) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGetErrorString_f(CUresult result, const char** pstr)
{
    return pfn_cuGetErrorString ? pfn_cuGetErrorString(result, pstr) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuInit_f(unsigned int flags)
{
    return pfn_cuInit ? pfn_cuInit(flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDeviceGet_f(CUdevice *dev, int ordinal)
{
    return pfn_cuDeviceGet ? pfn_cuDeviceGet(dev, ordinal) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDeviceGetCount_f(int* count)
{
    return pfn_cuDeviceGetCount ? pfn_cuDeviceGetCount(count) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDeviceGetName_f(char* name, int len, CUdevice dev)
{
    return pfn_cuDeviceGetName ? pfn_cuDeviceGetName(name, len, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDeviceGetAttribute_f(int* value, CUdevice_attribute attrib, CUdevice dev)
{
    return pfn_cuDeviceGetAttribute ? pfn_cuDeviceGetAttribute(value, attrib, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDevicePrimaryCtxRetain_f(CUcontext* ctx, CUdevice dev)
{
    return pfn_cuDevicePrimaryCtxRetain ? pfn_cuDevicePrimaryCtxRetain (ctx, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDeviceCanAccessPeer_f(int* can_access, CUdevice dev, CUdevice peer_dev)
{
    return pfn_cuDeviceCanAccessPeer ? pfn_cuDeviceCanAccessPeer(can_access, dev, peer_dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxGetCurrent_f(CUcontext* ctx)
{
    return pfn_cuCtxGetCurrent ? pfn_cuCtxGetCurrent(ctx) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxSetCurrent_f(CUcontext ctx)
{
    return pfn_cuCtxSetCurrent ? pfn_cuCtxSetCurrent(ctx) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxPushCurrent_f(CUcontext ctx)
{
    return pfn_cuCtxPushCurrent ? pfn_cuCtxPushCurrent(ctx) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxPopCurrent_f(CUcontext* ctx)
{
    return pfn_cuCtxPopCurrent ? pfn_cuCtxPopCurrent(ctx) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxSynchronize_f()
{
    return pfn_cuCtxSynchronize ? pfn_cuCtxSynchronize() : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxGetDevice_f(CUdevice* dev)
{
    return pfn_cuCtxGetDevice ? pfn_cuCtxGetDevice(dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxCreate_f(CUcontext* ctx, unsigned int flags, CUdevice dev)
{
    return pfn_cuCtxCreate ? pfn_cuCtxCreate(ctx, flags, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxDestroy_f(CUcontext ctx)
{
    return pfn_cuCtxDestroy ? pfn_cuCtxDestroy(ctx) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxEnablePeerAccess_f(CUcontext peer_ctx, unsigned int flags)
{
    return pfn_cuCtxEnablePeerAccess ? pfn_cuCtxEnablePeerAccess(peer_ctx, flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamCreate_f(CUstream* stream, unsigned int flags)
{
    return pfn_cuStreamCreate ? pfn_cuStreamCreate(stream, flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamDestroy_f(CUstream stream)
{
    return pfn_cuStreamDestroy ? pfn_cuStreamDestroy(stream) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamSynchronize_f(CUstream stream)
{
    return pfn_cuStreamSynchronize ? pfn_cuStreamSynchronize(stream) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuModuleLoadDataEx_f(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    return pfn_cuModuleLoadDataEx ? pfn_cuModuleLoadDataEx(module, image, numOptions, options, optionValues) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuModuleUnload_f(CUmodule hmod)
{
    return pfn_cuModuleUnload ? pfn_cuModuleUnload(hmod) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuModuleGetFunction_f(CUfunction *hfunc, CUmodule hmod, const char *name)
{
    return pfn_cuModuleGetFunction ? pfn_cuModuleGetFunction(hfunc, hmod, name) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuLaunchKernel_f(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)
{
    return pfn_cuLaunchKernel ? pfn_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuMemcpyPeerAsync_f(CUdeviceptr dst_ptr, CUcontext dst_ctx, CUdeviceptr src_ptr, CUcontext src_ctx, size_t n, CUstream stream)
{
    return pfn_cuMemcpyPeerAsync ? pfn_cuMemcpyPeerAsync(dst_ptr, dst_ctx, src_ptr, src_ctx, n, stream) : DRIVER_ENTRY_POINT_ERROR;
}


#endif // !WP_DISABLE_CUDA
