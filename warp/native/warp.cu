/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "warp.h"
#include "cuda_util.h"

#include <nvrtc.h>

#include <map>
#include <vector>

struct DeviceInfo
{
    static constexpr int kNameLen = 128;

    CUdevice device = -1;
    int ordinal = -1;
    char name[kNameLen] = "";
    int arch = 0;
    int is_uva = 0;
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


// void* alloc_host(size_t s)
// {
//     void* ptr;
//     check_cuda(cudaMallocHost(&ptr, s));
//     return ptr;
// }

// void free_host(void* ptr)
// {
//     cudaFreeHost(ptr);
// }

void* alloc_device(void* context, size_t s)
{
    ContextGuard guard(context);

    void* ptr;
    check_cuda(cudaMalloc(&ptr, s));
    return ptr;
}

void free_device(void* context, void* ptr)
{
    ContextGuard guard(context);

    check_cuda(cudaFree(ptr));
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

void memcpy_peer(void* dest_context, void* dest, void* src_context, void* src, size_t n)
{
    ContextGuard guard(dest_context);

    check_cu(cuMemcpyPeerAsync_f(
        reinterpret_cast<CUdeviceptr>(dest), static_cast<CUcontext>(dest_context),
        reinterpret_cast<CUdeviceptr>(src), static_cast<CUcontext>(src_context),
        n, get_current_stream()
    ));
}

__global__ void memset_kernel(int* dest, int value, int n)
{
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (tid < n)
    {
        dest[tid] = value;
    }
}

void memset_device(void* context, void* dest, int value, size_t n)
{
    ContextGuard guard(context);

    if ((n%4) > 0)
    {
        // for unaligned lengths fallback to CUDA memset
        check_cuda(cudaMemsetAsync(dest, value, n, get_current_stream()));
    }
    else
    {
        // custom kernel to support 4-byte values (and slightly lower host overhead)
        const int num_words = n/4;
        wp_launch_device(WP_CURRENT_CONTEXT, memset_kernel, num_words, ((int*)dest, value, num_words));
    }
}


void array_inner_device(uint64_t a, uint64_t b, uint64_t out, int len)
{

}

void array_sum_device(uint64_t a, uint64_t out, int len)
{
    
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

void* cuda_context_get_stream(void* context)
{
    ContextInfo* info = get_context_info(static_cast<CUcontext>(context));
    if (info)
    {
        // create stream on demand
        if (!info->stream)
        {
            ContextGuard guard(context, true);
            check_cu(cuStreamCreate_f(&info->stream, CU_STREAM_DEFAULT));
        }
        return info->stream;
    }
    return NULL;
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

void* cuda_stream_get_current()
{
    return get_current_stream();
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
        check_cuda(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
        
        // can use after CUDA 11.4 to permit graphs to capture cudaMallocAsync() operations
        //check_cuda(cudaGraphInstantiateWithFlags(&graph_exec, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch));

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

size_t cuda_compile_program(const char* cuda_src, int arch, const char* include_dir, bool debug, bool verbose, bool verify_fp, const char* output_file)
{
    nvrtcResult res;

    nvrtcProgram prog;
    res = nvrtcCreateProgram(
        &prog,          // prog
        cuda_src,      // buffer
        NULL,          // name
        0,             // numHeaders
        NULL,          // headers
        NULL);         // includeNames

    if (res != NVRTC_SUCCESS)
        return res;

    // check include dir path len (path + option)
    const int max_path = 4096 + 16;
    if (strlen(include_dir) > max_path)
    {
        printf("Include path too long\n");
        return size_t(-1);
    }

    char include_opt[max_path];
    strcpy(include_opt, "--include-path=");
    strcat(include_opt, include_dir);

    const int max_arch = 256;
    char arch_opt[max_arch];
    sprintf(arch_opt, "--gpu-architecture=compute_%d", arch);

    const char *opts[] = 
    {
        "--device-as-default-execution-space",
        arch_opt,
        "--use_fast_math",
        "--std=c++11",
        "--define-macro=WP_CUDA",
        (verify_fp ? "--define-macro=WP_VERIFY_FP" : "--undefine-macro=WP_VERIFY_FP"),
        "--define-macro=WP_NO_CRT",
        (debug ? "--define-macro=DEBUG" : "--define-macro=NDEBUG"),
        include_opt
    };

    res = nvrtcCompileProgram(prog, 9, opts);

    if (res == NVRTC_SUCCESS)
    {
        // save ptx
        size_t ptx_size;
        nvrtcGetPTXSize(prog, &ptx_size);

        char* ptx = (char*)malloc(ptx_size);
        nvrtcGetPTX(prog, ptx);

        // write to file
        FILE* file = fopen(output_file, "w");
        fwrite(ptx, 1, ptx_size, file);
        fclose(file);

        free(ptx);
    }

    if (res != NVRTC_SUCCESS || verbose)
    {
        // get program log
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);

        char* log = (char*)malloc(log_size);
        nvrtcGetProgramLog(prog, log);

        // todo: figure out better way to return this to python
        printf("%s", log);
        free(log);
    }

    nvrtcDestroyProgram(&prog);
    return res;
}

void* cuda_load_module(void* context, const char* path)
{
    ContextGuard guard(context);

    FILE* file = fopen(path, "rb");
    fseek(file, 0, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* buf = (char*)malloc(length);
    size_t result = fread(buf, 1, length, file);
    fclose(file);

    if (result != length)
    {
        printf("Warp: Failed to load PTX from disk, unexpected number of bytes\n");
        return NULL;
    }

    CUjit_option options[2];
    void *optionVals[2];
    char error_log[8192];
    unsigned int logSize = 8192;
    // Setup linker options
    // Pass a buffer for error message
    options[0] = CU_JIT_ERROR_LOG_BUFFER;
    optionVals[0] = (void *) error_log;
    // Pass the size of the error buffer
    options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optionVals[1] = (void *) (long) logSize;

    CUmodule module = NULL;
    if (!check_cu(cuModuleLoadDataEx_f(&module, buf, 2, options, optionVals)))
    {
        printf("Warp: Loading PTX module failed\n");
        // print error log
        fprintf(stderr, "PTX linker error:\n%s\n", error_log);
    }

    free(buf);

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
        printf("Warp: Failed to lookup kernel function %s in module\n", name);

    return kernel;
}

size_t cuda_launch_kernel(void* context, void* kernel, size_t dim, void** args)
{
    ContextGuard guard(context);

    const int block_dim = 256;
    const int grid_dim = (dim + block_dim - 1)/block_dim;

    CUresult res = cuLaunchKernel_f(
        (CUfunction)kernel,
        grid_dim, 1, 1,
        block_dim, 1, 1,
        0, get_current_stream(),
        args,
        0);

    return res;

}

// impl. files
#include "bvh.cu"
#include "mesh.cu"
#include "sort.cu"
#include "hashgrid.cu"
#include "marching.cu"
#include "volume_builder.cu"

//#include "spline.inl"
//#include "volume.inl"

