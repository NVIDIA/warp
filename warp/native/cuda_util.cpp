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
static void* GetProcAddress(void* handle, const char* name) { return dlsym(handle, name); }
#endif


cuInit_t* cuInit_f;
cuGetErrorName_t* cuGetErrorName_f;
cuGetErrorString_t* cuGetErrorString_f;

cuDeviceGet_t* cuDeviceGet_f;
cuDeviceGetCount_t* cuDeviceGetCount_f;
cuDeviceGetName_t* cuDeviceGetName_f;
cuDevicePrimaryCtxRetain_t* cuDevicePrimaryCtxRetain_f;
cuDeviceGetAttribute_t* cuDeviceGetAttribute_f;
cuDeviceCanAccessPeer_t* cuDeviceCanAccessPeer_f;

cuCtxGetCurrent_t* cuCtxGetCurrent_f;
cuCtxSetCurrent_t* cuCtxSetCurrent_f;
cuCtxPushCurrent_t* cuCtxPushCurrent_f;
cuCtxPopCurrent_t* cuCtxPopCurrent_f;
cuCtxSynchronize_t* cuCtxSynchronize_f;
cuCtxGetDevice_t* cuCtxGetDevice_f;
cuCtxCreate_t* cuCtxCreate_f;
cuCtxDestroy_t* cuCtxDestroy_f;
cuCtxEnablePeerAccess_t* cuCtxEnablePeerAccess_f;

cuStreamCreate_t* cuStreamCreate_f;
cuStreamDestroy_t* cuStreamDestroy_f;
cuStreamSynchronize_t* cuStreamSynchronize_f;

cuModuleUnload_t* cuModuleUnload_f;
cuModuleLoadDataEx_t* cuModuleLoadDataEx_f;
cuModuleGetFunction_t* cuModuleGetFunction_f;

cuLaunchKernel_t* cuLaunchKernel_f;

cuMemcpyPeerAsync_t* cuMemcpyPeerAsync_f;


bool ContextGuard::always_restore = false;


bool init_cuda_driver()
{
#if defined(_WIN32)
    static HMODULE hCudaDriver = LoadLibrary("nvcuda.dll");
    if (hCudaDriver == NULL) {
        fprintf(stderr, "Error: Could not open nvcuda.dll.\n");
        return false;
    }
#elif defined(__linux__)
    static void* hCudaDriver = dlopen("libcuda.so", RTLD_NOW);
    if (hCudaDriver == NULL) {
        fprintf(stderr, "Error: Could not open libcuda.so.\n");
        return false;
    }
#endif

    //
    // CAUTION: Need to be careful about version suffixes like _v2!
    //
    // TODO: should use cuGetProcAddress
    //
    cuInit_f = (cuInit_t*)GetProcAddress(hCudaDriver, "cuInit");
    cuGetErrorName_f = (cuGetErrorName_t*)GetProcAddress(hCudaDriver, "cuGetErrorName");
    cuGetErrorString_f = (cuGetErrorString_t*)GetProcAddress(hCudaDriver, "cuGetErrorString");

    cuDeviceGet_f = (cuDeviceGet_t*)GetProcAddress(hCudaDriver, "cuDeviceGet");
    cuDeviceGetCount_f = (cuDeviceGetCount_t*)GetProcAddress(hCudaDriver, "cuDeviceGetCount");
    cuDeviceGetName_f = (cuDeviceGetName_t*)GetProcAddress(hCudaDriver, "cuDeviceGetName");
    cuDevicePrimaryCtxRetain_f = (cuDevicePrimaryCtxRetain_t*)GetProcAddress(hCudaDriver, "cuDevicePrimaryCtxRetain");
    cuDeviceGetAttribute_f = (cuDeviceGetAttribute_t*)GetProcAddress(hCudaDriver, "cuDeviceGetAttribute");
    cuDeviceCanAccessPeer_f = (cuDeviceCanAccessPeer_t*)GetProcAddress(hCudaDriver, "cuDeviceCanAccessPeer");

    cuCtxSetCurrent_f = (cuCtxSetCurrent_t*)GetProcAddress(hCudaDriver, "cuCtxSetCurrent");
    cuCtxGetCurrent_f = (cuCtxGetCurrent_t*)GetProcAddress(hCudaDriver, "cuCtxGetCurrent");
    cuCtxPushCurrent_f = (cuCtxPushCurrent_t*)GetProcAddress(hCudaDriver, "cuCtxPushCurrent_v2"); // !!! _v2
    cuCtxPopCurrent_f = (cuCtxPopCurrent_t*)GetProcAddress(hCudaDriver, "cuCtxPopCurrent_v2"); // !!! _v2
    cuCtxSynchronize_f = (cuCtxSynchronize_t*)GetProcAddress(hCudaDriver, "cuCtxSynchronize");
    cuCtxGetDevice_f = (cuCtxGetDevice_t*)GetProcAddress(hCudaDriver, "cuCtxGetDevice");
    cuCtxCreate_f = (cuCtxCreate_t*)GetProcAddress(hCudaDriver, "cuCtxCreate_v2");  // !!! _v2
    cuCtxDestroy_f = (cuCtxDestroy_t*)GetProcAddress(hCudaDriver, "cuCtxDestroy_v2");  // !!! _v2
    cuCtxEnablePeerAccess_f = (cuCtxEnablePeerAccess_t*)GetProcAddress(hCudaDriver, "cuCtxEnablePeerAccess");

    cuStreamCreate_f = (cuStreamCreate_t*)GetProcAddress(hCudaDriver, "cuStreamCreate");
    cuStreamDestroy_f = (cuStreamDestroy_t*)GetProcAddress(hCudaDriver, "cuStreamDestroy");
    cuStreamSynchronize_f = (cuStreamSynchronize_t*)GetProcAddress(hCudaDriver, "cuStreamSynchronize");

    cuModuleUnload_f = (cuModuleUnload_t*)GetProcAddress(hCudaDriver, "cuModuleUnload");
    cuModuleLoadDataEx_f = (cuModuleLoadDataEx_t*)GetProcAddress(hCudaDriver, "cuModuleLoadDataEx");
    cuModuleGetFunction_f = (cuModuleGetFunction_t*)GetProcAddress(hCudaDriver, "cuModuleGetFunction");

    cuLaunchKernel_f = (cuLaunchKernel_t*)GetProcAddress(hCudaDriver, "cuLaunchKernel");

    cuMemcpyPeerAsync_f = (cuMemcpyPeerAsync_t*)GetProcAddress(hCudaDriver, "cuMemcpyPeerAsync");

    if (!cuInit_f)
    {
        fprintf(stderr, "Warp CUDA error: failed to load CUDA symbols\n");
        return false;
    }

    return true;
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
    if (cuGetErrorString_f)
        cuGetErrorString_f(result, &errString);

    if (errString)
        fprintf(stderr, "Warp CUDA error %u: %s (%s:%d)\n", unsigned(result), errString, file, line);
    else
        fprintf(stderr, "Warp CUDA error %u (%s:%d)\n", unsigned(result), file, line);

    return false;
}

#endif // !WP_DISABLE_CUDA
