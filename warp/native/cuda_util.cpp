/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#if WP_ENABLE_CUDA

#include "cuda_util.h"
#include "error.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <wingdi.h>  // needed for OpenGL includes
#elif defined(__linux__)
#include <dlfcn.h>
#endif

#include <set>
#include <stack>

// the minimum CUDA version required from the driver
#define WP_CUDA_DRIVER_VERSION 11040

// the minimum CUDA Toolkit version required to build Warp
#define WP_CUDA_TOOLKIT_VERSION 11050

// check if the CUDA Toolkit is too old
#if CUDA_VERSION < WP_CUDA_TOOLKIT_VERSION
#error Building Warp requires CUDA Toolkit version 11.5 or higher
#endif

// Avoid including <cudaGLTypedefs.h>, which requires OpenGL headers to be installed.
// We define our own GL types, based on the spec here: https://www.khronos.org/opengl/wiki/OpenGL_Type
namespace wp
{
typedef uint32_t GLuint;
}

// function prototypes adapted from <cudaGLTypedefs.h>
typedef CUresult (CUDAAPI *PFN_cuGraphicsGLRegisterBuffer_v3000)(CUgraphicsResource *pCudaResource, wp::GLuint buffer, unsigned int Flags);


// function pointers to driver API entry points
// these are explicitly versioned according to cudaTypedefs.h from CUDA Toolkit WP_CUDA_TOOLKIT_VERSION
#if CUDA_VERSION < 12000
static PFN_cuGetProcAddress_v11030 pfn_cuGetProcAddress;
#else
static PFN_cuGetProcAddress_v12000 pfn_cuGetProcAddress;
#endif
static PFN_cuDriverGetVersion_v2020 pfn_cuDriverGetVersion;
static PFN_cuGetErrorName_v6000 pfn_cuGetErrorName;
static PFN_cuGetErrorString_v6000 pfn_cuGetErrorString;
static PFN_cuInit_v2000 pfn_cuInit;
static PFN_cuDeviceGet_v2000 pfn_cuDeviceGet;
static PFN_cuDeviceGetCount_v2000 pfn_cuDeviceGetCount;
static PFN_cuDeviceGetName_v2000 pfn_cuDeviceGetName;
static PFN_cuDeviceGetAttribute_v2000 pfn_cuDeviceGetAttribute;
static PFN_cuDeviceGetUuid_v11040 pfn_cuDeviceGetUuid;
static PFN_cuDevicePrimaryCtxRetain_v7000 pfn_cuDevicePrimaryCtxRetain;
static PFN_cuDevicePrimaryCtxRelease_v11000 pfn_cuDevicePrimaryCtxRelease;
static PFN_cuDeviceCanAccessPeer_v4000 pfn_cuDeviceCanAccessPeer;
static PFN_cuMemGetInfo_v3020 pfn_cuMemGetInfo;
static PFN_cuCtxGetCurrent_v4000 pfn_cuCtxGetCurrent;
static PFN_cuCtxSetCurrent_v4000 pfn_cuCtxSetCurrent;
static PFN_cuCtxPushCurrent_v4000 pfn_cuCtxPushCurrent;
static PFN_cuCtxPopCurrent_v4000 pfn_cuCtxPopCurrent;
static PFN_cuCtxSynchronize_v2000 pfn_cuCtxSynchronize;
static PFN_cuCtxGetDevice_v2000 pfn_cuCtxGetDevice;
static PFN_cuCtxCreate_v3020 pfn_cuCtxCreate;
static PFN_cuCtxDestroy_v4000 pfn_cuCtxDestroy;
static PFN_cuCtxEnablePeerAccess_v4000 pfn_cuCtxEnablePeerAccess;
static PFN_cuCtxDisablePeerAccess_v4000 pfn_cuCtxDisablePeerAccess;
static PFN_cuStreamCreate_v2000 pfn_cuStreamCreate;
static PFN_cuStreamDestroy_v4000 pfn_cuStreamDestroy;
static PFN_cuStreamSynchronize_v2000 pfn_cuStreamSynchronize;
static PFN_cuStreamWaitEvent_v3020 pfn_cuStreamWaitEvent;
static PFN_cuStreamGetCtx_v9020 pfn_cuStreamGetCtx;
static PFN_cuStreamGetCaptureInfo_v11030 pfn_cuStreamGetCaptureInfo;
static PFN_cuStreamUpdateCaptureDependencies_v11030 pfn_cuStreamUpdateCaptureDependencies;
static PFN_cuEventCreate_v2000 pfn_cuEventCreate;
static PFN_cuEventDestroy_v4000 pfn_cuEventDestroy;
static PFN_cuEventRecord_v2000 pfn_cuEventRecord;
static PFN_cuEventRecordWithFlags_v11010 pfn_cuEventRecordWithFlags;
static PFN_cuEventSynchronize_v2000 pfn_cuEventSynchronize;
static PFN_cuModuleLoadDataEx_v2010 pfn_cuModuleLoadDataEx;
static PFN_cuModuleUnload_v2000 pfn_cuModuleUnload;
static PFN_cuModuleGetFunction_v2000 pfn_cuModuleGetFunction;
static PFN_cuLaunchKernel_v4000 pfn_cuLaunchKernel;
static PFN_cuMemcpyPeerAsync_v4000 pfn_cuMemcpyPeerAsync;
static PFN_cuPointerGetAttribute_v4000 pfn_cuPointerGetAttribute;
static PFN_cuGraphicsMapResources_v3000 pfn_cuGraphicsMapResources;
static PFN_cuGraphicsUnmapResources_v3000 pfn_cuGraphicsUnmapResources;
static PFN_cuGraphicsResourceGetMappedPointer_v3020 pfn_cuGraphicsResourceGetMappedPointer;
static PFN_cuGraphicsGLRegisterBuffer_v3000 pfn_cuGraphicsGLRegisterBuffer;
static PFN_cuGraphicsUnregisterResource_v3000 pfn_cuGraphicsUnregisterResource;

static bool cuda_driver_initialized = false;

bool ContextGuard::always_restore = false;

CudaTimingState* g_cuda_timing_state = NULL;


static inline int get_major(int version)
{
    return version / 1000;
}

static inline int get_minor(int version)
{
    return (version % 1000) / 10;
}

static bool get_driver_entry_point(const char* name, void** pfn)
{
    if (!pfn_cuGetProcAddress || !name || !pfn)
        return false;

#if CUDA_VERSION < 12000
    CUresult r = pfn_cuGetProcAddress(name, pfn, WP_CUDA_DRIVER_VERSION, CU_GET_PROC_ADDRESS_DEFAULT);
#else
    CUresult r = pfn_cuGetProcAddress(name, pfn, WP_CUDA_DRIVER_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, NULL);
#endif

    if (r != CUDA_SUCCESS)
    {
        fprintf(stderr, "Warp CUDA error: Failed to get driver entry point '%s' (CUDA error %u)\n", name, unsigned(r));
        return false;
    }

    return true;
}

bool init_cuda_driver()
{
#if defined(_WIN32)
    static HMODULE hCudaDriver = LoadLibraryA("nvcuda.dll");
    if (hCudaDriver == NULL) {
        fprintf(stderr, "Warp CUDA error: Could not open nvcuda.dll.\n");
        return false;
    }
    pfn_cuGetProcAddress = (PFN_cuGetProcAddress)GetProcAddress(hCudaDriver, "cuGetProcAddress");
#elif defined(__linux__)
    static void* hCudaDriver = dlopen("libcuda.so", RTLD_NOW);
    if (hCudaDriver == NULL) {
        // WSL and possibly other systems might require the .1 suffix
        hCudaDriver = dlopen("libcuda.so.1", RTLD_NOW);
        if (hCudaDriver == NULL) {
            fprintf(stderr, "Warp CUDA error: Could not open libcuda.so.\n");
            return false;
        }
    }
    pfn_cuGetProcAddress = (PFN_cuGetProcAddress)dlsym(hCudaDriver, "cuGetProcAddress");
#endif

    if (!pfn_cuGetProcAddress)
    {
        fprintf(stderr, "Warp CUDA error: Failed to get function cuGetProcAddress\n");
        return false;
    }

    // check the CUDA driver version and report an error if it's too low
    int driver_version = 0;
    if (get_driver_entry_point("cuDriverGetVersion", &(void*&)pfn_cuDriverGetVersion) && check_cu(pfn_cuDriverGetVersion(&driver_version)))
    {
        if (driver_version < WP_CUDA_DRIVER_VERSION)
        {
            fprintf(stderr, "Warp CUDA error: Warp requires CUDA driver %d.%d or higher, but the current driver only supports CUDA %d.%d\n",
                get_major(WP_CUDA_DRIVER_VERSION), get_minor(WP_CUDA_DRIVER_VERSION),
                get_major(driver_version), get_minor(driver_version));
            return false;
        }
    }
    else
    {
        fprintf(stderr, "Warp CUDA warning: Unable to determine CUDA driver version\n");
    }

    // initialize driver entry points
    get_driver_entry_point("cuGetErrorString", &(void*&)pfn_cuGetErrorString);
    get_driver_entry_point("cuGetErrorName", &(void*&)pfn_cuGetErrorName);
    get_driver_entry_point("cuInit", &(void*&)pfn_cuInit);
    get_driver_entry_point("cuDeviceGet", &(void*&)pfn_cuDeviceGet);
    get_driver_entry_point("cuDeviceGetCount", &(void*&)pfn_cuDeviceGetCount);
    get_driver_entry_point("cuDeviceGetName", &(void*&)pfn_cuDeviceGetName);
    get_driver_entry_point("cuDeviceGetAttribute", &(void*&)pfn_cuDeviceGetAttribute);
    get_driver_entry_point("cuDeviceGetUuid", &(void*&)pfn_cuDeviceGetUuid);
    get_driver_entry_point("cuDevicePrimaryCtxRetain", &(void*&)pfn_cuDevicePrimaryCtxRetain);
    get_driver_entry_point("cuDevicePrimaryCtxRelease", &(void*&)pfn_cuDevicePrimaryCtxRelease);
    get_driver_entry_point("cuDeviceCanAccessPeer", &(void*&)pfn_cuDeviceCanAccessPeer);
    get_driver_entry_point("cuMemGetInfo", &(void*&)pfn_cuMemGetInfo);
    get_driver_entry_point("cuCtxSetCurrent", &(void*&)pfn_cuCtxSetCurrent);
    get_driver_entry_point("cuCtxGetCurrent", &(void*&)pfn_cuCtxGetCurrent);
    get_driver_entry_point("cuCtxPushCurrent", &(void*&)pfn_cuCtxPushCurrent);
    get_driver_entry_point("cuCtxPopCurrent", &(void*&)pfn_cuCtxPopCurrent);
    get_driver_entry_point("cuCtxSynchronize", &(void*&)pfn_cuCtxSynchronize);
    get_driver_entry_point("cuCtxGetDevice", &(void*&)pfn_cuCtxGetDevice);
    get_driver_entry_point("cuCtxCreate", &(void*&)pfn_cuCtxCreate);
    get_driver_entry_point("cuCtxDestroy", &(void*&)pfn_cuCtxDestroy);
    get_driver_entry_point("cuCtxEnablePeerAccess", &(void*&)pfn_cuCtxEnablePeerAccess);
    get_driver_entry_point("cuCtxDisablePeerAccess", &(void*&)pfn_cuCtxDisablePeerAccess);
    get_driver_entry_point("cuStreamCreate", &(void*&)pfn_cuStreamCreate);
    get_driver_entry_point("cuStreamDestroy", &(void*&)pfn_cuStreamDestroy);
    get_driver_entry_point("cuStreamSynchronize", &(void*&)pfn_cuStreamSynchronize);
    get_driver_entry_point("cuStreamWaitEvent", &(void*&)pfn_cuStreamWaitEvent);
    get_driver_entry_point("cuStreamGetCtx", &(void*&)pfn_cuStreamGetCtx);
    get_driver_entry_point("cuStreamGetCaptureInfo", &(void*&)pfn_cuStreamGetCaptureInfo);
    get_driver_entry_point("cuStreamUpdateCaptureDependencies", &(void*&)pfn_cuStreamUpdateCaptureDependencies);
    get_driver_entry_point("cuEventCreate", &(void*&)pfn_cuEventCreate);
    get_driver_entry_point("cuEventDestroy", &(void*&)pfn_cuEventDestroy);
    get_driver_entry_point("cuEventRecord", &(void*&)pfn_cuEventRecord);
    get_driver_entry_point("cuEventRecordWithFlags", &(void*&)pfn_cuEventRecordWithFlags);
    get_driver_entry_point("cuEventSynchronize", &(void*&)pfn_cuEventSynchronize);
    get_driver_entry_point("cuModuleLoadDataEx", &(void*&)pfn_cuModuleLoadDataEx);
    get_driver_entry_point("cuModuleUnload", &(void*&)pfn_cuModuleUnload);
    get_driver_entry_point("cuModuleGetFunction", &(void*&)pfn_cuModuleGetFunction);
    get_driver_entry_point("cuLaunchKernel", &(void*&)pfn_cuLaunchKernel);
    get_driver_entry_point("cuMemcpyPeerAsync", &(void*&)pfn_cuMemcpyPeerAsync);
    get_driver_entry_point("cuPointerGetAttribute", &(void*&)pfn_cuPointerGetAttribute);
    get_driver_entry_point("cuGraphicsMapResources", &(void*&)pfn_cuGraphicsMapResources);
    get_driver_entry_point("cuGraphicsUnmapResources", &(void*&)pfn_cuGraphicsUnmapResources);
    get_driver_entry_point("cuGraphicsResourceGetMappedPointer", &(void*&)pfn_cuGraphicsResourceGetMappedPointer);
    get_driver_entry_point("cuGraphicsGLRegisterBuffer", &(void*&)pfn_cuGraphicsGLRegisterBuffer);
    get_driver_entry_point("cuGraphicsUnregisterResource", &(void*&)pfn_cuGraphicsUnregisterResource);

    if (pfn_cuInit)
        cuda_driver_initialized = check_cu(pfn_cuInit(0));
    
    return cuda_driver_initialized;
}

bool is_cuda_driver_initialized()
{
    return cuda_driver_initialized;
}

bool check_cuda_result(cudaError_t code, const char* func, const char* file, int line)
{
    if (code == cudaSuccess)
        return true;

    wp::set_error_string("Warp CUDA error %u: %s (in function %s, %s:%d)", unsigned(code), cudaGetErrorString(code), func, file, line);
    return false;
}

bool check_cu_result(CUresult result, const char* func, const char* file, int line)
{
    if (result == CUDA_SUCCESS)
        return true;

    const char* errString = NULL;
    if (pfn_cuGetErrorString)
        pfn_cuGetErrorString(result, &errString);

    if (errString)
        wp::set_error_string("Warp CUDA error %u: %s (in function %s, %s:%d)", unsigned(result), errString, func, file, line);
    else
        wp::set_error_string("Warp CUDA error %u (in function %s, %s:%d)", unsigned(result), func, file, line);

    return false;
}

bool get_capture_dependencies(CUstream stream, std::vector<CUgraphNode>& dependencies_ret)
{
    CUstreamCaptureStatus status;
    size_t num_dependencies = 0;
    const CUgraphNode* dependencies = NULL;
    dependencies_ret.clear();
    if (check_cu(cuStreamGetCaptureInfo_f(stream, &status, NULL, NULL, &dependencies, &num_dependencies)))
    {
        if (dependencies && num_dependencies > 0)
            dependencies_ret.insert(dependencies_ret.begin(), dependencies, dependencies + num_dependencies);
        return true;
    }
    return false;
}

bool get_graph_leaf_nodes(cudaGraph_t graph, std::vector<cudaGraphNode_t>& leaf_nodes_ret)
{
    if (!graph)
        return false;

    size_t node_count = 0;
    if (!check_cuda(cudaGraphGetNodes(graph, NULL, &node_count)))
        return false;

    std::vector<cudaGraphNode_t> nodes(node_count);
    if (!check_cuda(cudaGraphGetNodes(graph, nodes.data(), &node_count)))
        return false;

    leaf_nodes_ret.clear();

    for (cudaGraphNode_t node : nodes)
    {
        size_t dependent_count;
        if (!check_cuda(cudaGraphNodeGetDependentNodes(node, NULL, &dependent_count)))
            return false;

        if (dependent_count == 0)
            leaf_nodes_ret.push_back(node);
    }

    return true;
}


#define DRIVER_ENTRY_POINT_ERROR driver_entry_point_error(__FUNCTION__)

static CUresult driver_entry_point_error(const char* function)
{
    fprintf(stderr, "Warp CUDA error: Function %s: a suitable driver entry point was not found\n", function);
    return (CUresult)cudaErrorCallRequiresNewerDriver; // this matches what cudart would do
}

CUresult cuDriverGetVersion_f(int* version)
{
    return pfn_cuDriverGetVersion ? pfn_cuDriverGetVersion(version) : DRIVER_ENTRY_POINT_ERROR;
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
    if (pfn_cuDeviceGetCount)
        return pfn_cuDeviceGetCount(count);

    // allow calling this function even if CUDA is not available
    if (count)
        *count = 0;

    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName_f(char* name, int len, CUdevice dev)
{
    return pfn_cuDeviceGetName ? pfn_cuDeviceGetName(name, len, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDeviceGetAttribute_f(int* value, CUdevice_attribute attrib, CUdevice dev)
{
    return pfn_cuDeviceGetAttribute ? pfn_cuDeviceGetAttribute(value, attrib, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDeviceGetUuid_f(CUuuid* uuid, CUdevice dev)
{
    return pfn_cuDeviceGetUuid ? pfn_cuDeviceGetUuid(uuid, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDevicePrimaryCtxRetain_f(CUcontext* ctx, CUdevice dev)
{
    return pfn_cuDevicePrimaryCtxRetain ? pfn_cuDevicePrimaryCtxRetain(ctx, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDevicePrimaryCtxRelease_f(CUdevice dev)
{
    return pfn_cuDevicePrimaryCtxRelease ? pfn_cuDevicePrimaryCtxRelease(dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDeviceCanAccessPeer_f(int* can_access, CUdevice dev, CUdevice peer_dev)
{
    return pfn_cuDeviceCanAccessPeer ? pfn_cuDeviceCanAccessPeer(can_access, dev, peer_dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuMemGetInfo_f(size_t* free, size_t* total)
{
    return pfn_cuMemGetInfo ? pfn_cuMemGetInfo(free, total) : DRIVER_ENTRY_POINT_ERROR;
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

CUresult cuCtxDisablePeerAccess_f(CUcontext peer_ctx)
{
    return pfn_cuCtxDisablePeerAccess ? pfn_cuCtxDisablePeerAccess(peer_ctx) : DRIVER_ENTRY_POINT_ERROR;
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

CUresult cuStreamWaitEvent_f(CUstream stream, CUevent event, unsigned int flags)
{
    return pfn_cuStreamWaitEvent ? pfn_cuStreamWaitEvent(stream, event, flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamGetCtx_f(CUstream stream, CUcontext* pctx)
{
    return pfn_cuStreamGetCtx ? pfn_cuStreamGetCtx(stream, pctx) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamGetCaptureInfo_f(CUstream stream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out)
{
    return pfn_cuStreamGetCaptureInfo ? pfn_cuStreamGetCaptureInfo(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamUpdateCaptureDependencies_f(CUstream stream, CUgraphNode *dependencies, size_t numDependencies, unsigned int flags)
{
    return pfn_cuStreamUpdateCaptureDependencies ? pfn_cuStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuEventCreate_f(CUevent* event, unsigned int flags)
{
    return pfn_cuEventCreate ? pfn_cuEventCreate(event, flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuEventDestroy_f(CUevent event)
{
    return pfn_cuEventDestroy ? pfn_cuEventDestroy(event) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuEventRecord_f(CUevent event, CUstream stream)
{
    return pfn_cuEventRecord ? pfn_cuEventRecord(event, stream) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuEventRecordWithFlags_f(CUevent event, CUstream stream, unsigned int flags)
{
    return pfn_cuEventRecordWithFlags ? pfn_cuEventRecordWithFlags(event, stream, flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuEventSynchronize_f(CUevent event)
{
    return pfn_cuEventSynchronize ? pfn_cuEventSynchronize(event) : DRIVER_ENTRY_POINT_ERROR;
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

CUresult cuPointerGetAttribute_f(void* data, CUpointer_attribute attribute, CUdeviceptr ptr)
{
    return pfn_cuPointerGetAttribute ? pfn_cuPointerGetAttribute(data, attribute, ptr) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGraphicsMapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream stream)
{
    return pfn_cuGraphicsMapResources ? pfn_cuGraphicsMapResources(count, resources, stream) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGraphicsUnmapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream hStream)
{
    return pfn_cuGraphicsUnmapResources ? pfn_cuGraphicsUnmapResources(count, resources, hStream) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGraphicsResourceGetMappedPointer_f(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource)
{
    return pfn_cuGraphicsResourceGetMappedPointer ? pfn_cuGraphicsResourceGetMappedPointer(pDevPtr, pSize, resource) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGraphicsGLRegisterBuffer_f(CUgraphicsResource *pCudaResource, unsigned int buffer, unsigned int flags)
{
    return pfn_cuGraphicsGLRegisterBuffer ? pfn_cuGraphicsGLRegisterBuffer(pCudaResource, (wp::GLuint) buffer, flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGraphicsUnregisterResource_f(CUgraphicsResource resource)
{
    return pfn_cuGraphicsUnregisterResource ? pfn_cuGraphicsUnregisterResource(resource) : DRIVER_ENTRY_POINT_ERROR;
}

#endif // WP_ENABLE_CUDA
