#include "warp.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#if defined(__linux__)
#include <dlfcn.h>
static void* GetProcAddress(void* handle, const char* name) { return dlsym(handle, name); }
#endif

#if defined(_WIN32)
#include <windows.h>
#endif

#include "nvrtc.h"

typedef CUresult CUDAAPI cuInit_t(unsigned int);
typedef CUresult CUDAAPI cuDeviceGet_t(CUdevice *dev, int ordinal);
typedef CUresult CUDAAPI cuCtxGetCurrent_t(CUcontext* ctx);
typedef CUresult CUDAAPI cuCtxSetCurrent_t(CUcontext ctx);
typedef CUresult CUDAAPI cuCtxCreate_t(CUcontext* pctx, unsigned int flags, CUdevice dev);
typedef CUresult CUDAAPI cuCtxDestroy_t(CUcontext pctx);

typedef CUresult CUDAAPI cuModuleUnload_t(CUmodule hmod);
typedef CUresult CUDAAPI cuModuleLoadDataEx_t(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
typedef CUresult CUDAAPI cuModuleGetFunction_t(CUfunction *hfunc, CUmodule hmod, const char *name);

typedef CUresult CUDAAPI cuLaunchKernel_t(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);

static cuInit_t* cuInit_f;
static cuCtxGetCurrent_t* cuCtxGetCurrent_f;
static cuCtxSetCurrent_t* cuCtxSetCurrent_f;

static cuModuleUnload_t* cuModuleUnload_f;
static cuModuleLoadDataEx_t* cuModuleLoadDataEx_f;
static cuModuleGetFunction_t* cuModuleGetFunction_f;
static cuLaunchKernel_t* cuLaunchKernel_f;

//static cuCtxCreate_t* cuCtxCreate_f;
//static cuCtxDestroy_t* cuCtxDestroy_f;
//static cuDeviceGet_t* cuDeviceGet_f;

static CUcontext g_cuda_context;
static CUcontext g_save_context;

static cudaStream_t g_cuda_stream;

int cuda_init()
{
    #if defined(_WIN32)
        static HMODULE hCudaDriver = LoadLibrary("nvcuda.dll");
    #elif defined(__linux__)
        static void* hCudaDriver = dlopen("libcuda.so", RTLD_NOW);
    #endif

    if (hCudaDriver == NULL)
        return false;

	cuInit_f = (cuInit_t*)GetProcAddress(hCudaDriver, "cuInit");
	cuCtxSetCurrent_f = (cuCtxSetCurrent_t*)GetProcAddress(hCudaDriver, "cuCtxSetCurrent");
	cuCtxGetCurrent_f = (cuCtxGetCurrent_t*)GetProcAddress(hCudaDriver, "cuCtxGetCurrent");
    cuModuleUnload_f = (cuModuleUnload_t*)GetProcAddress(hCudaDriver, "cuModuleUnload");
    cuModuleLoadDataEx_f = (cuModuleLoadDataEx_t*)GetProcAddress(hCudaDriver, "cuModuleLoadDataEx");
    cuModuleGetFunction_f = (cuModuleGetFunction_t*)GetProcAddress(hCudaDriver, "cuModuleGetFunction");
    cuLaunchKernel_f = (cuLaunchKernel_t*)GetProcAddress(hCudaDriver, "cuLaunchKernel");

    if (cuInit_f == NULL)
        return -1;

    CUresult err = cuInit_f(0);
    if (err != CUDA_SUCCESS)
		return err;

    CUcontext ctx;
    cuCtxGetCurrent_f(&ctx);

    if (ctx == NULL)
    {
        // create a new default runtime context
        cudaSetDevice(0);
        cuCtxGetCurrent_f(&ctx);
    }
    
    // save the context, all API calls must have this context set on the calling thread
    g_cuda_context = ctx;
    
    check_cuda(cudaStreamCreate(&g_cuda_stream));
    
    return 0;
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

void* alloc_device(size_t s)
{
    void* ptr;
    check_cuda(cudaMalloc(&ptr, s));
    return ptr;
}

void free_device(void* ptr)
{
    check_cuda(cudaFree(ptr));
}

void memcpy_h2d(void* dest, void* src, size_t n)
{
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice, g_cuda_stream));
}

void memcpy_d2h(void* dest, void* src, size_t n)
{
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost, g_cuda_stream));
}

void memcpy_d2d(void* dest, void* src, size_t n)
{
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToDevice, g_cuda_stream));
}

__global__ void memset_kernel(int* dest, int value, int n)
{
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (tid < n)
    {
        dest[tid] = value;
    }
}

void memset_device(void* dest, int value, size_t n)
{
    if ((n%4) > 0)
    {
        // for unaligned lengths fallback to CUDA memset
        check_cuda(cudaMemsetAsync(dest, value, n, g_cuda_stream));
    }
    else
    {
        // custom kernel mostly to reduce launch overhead
        const int num_words = n/4;
        wp_launch_device(memset_kernel, num_words, ((int*)dest, value, num_words));
    }
}

void synchronize()
{
    check_cuda(cudaStreamSynchronize(g_cuda_stream));
}

void array_inner_device(uint64_t a, uint64_t b, uint64_t out, int len)
{

}

void array_sum_device(uint64_t a, uint64_t out, int len)
{
    
}


uint64_t cuda_check_device()
{
    cudaDeviceSynchronize();
    return cudaPeekAtLastError(); 
}

void cuda_report_error(int code, const char* file, int line)
{
    if (code != cudaSuccess) 
    {
        printf("CUDA Error: %s %s %d\n", cudaGetErrorString((cudaError_t)code), file, line);
    }
}

void* cuda_get_stream()
{
    return g_cuda_stream;
}

void cuda_graph_begin_capture()
{
    check_cuda(cudaStreamBeginCapture(g_cuda_stream, cudaStreamCaptureModeGlobal));
}

void* cuda_graph_end_capture()
{
    cudaGraph_t graph;
    check_cuda(cudaStreamEndCapture(g_cuda_stream, &graph));

    cudaGraphExec_t graph_exec;
    check_cuda(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0))

    // free source graph
    check_cuda(cudaGraphDestroy(graph));

    return graph_exec;
}

void cuda_graph_launch(void* graph_exec)
{
    check_cuda(cudaGraphLaunch((cudaGraphExec_t)graph_exec, g_cuda_stream));
}

void cuda_graph_destroy(void* graph_exec)
{
    check_cuda(cudaGraphExecDestroy((cudaGraphExec_t)graph_exec));
}

void cuda_acquire_context()
{
    cuCtxGetCurrent_f(&g_save_context);
    cuCtxSetCurrent_f(g_cuda_context);
}

void cuda_restore_context()
{
    cuCtxSetCurrent_f(g_save_context);
}


void* cuda_get_context()
{
	return g_cuda_context;
}

void cuda_set_context(void* ctx)
{
    g_cuda_context = (CUcontext)ctx;
}

const char* cuda_get_device_name()
{
    static cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    return prop.name;
}

size_t cuda_compile_program(const char* cuda_src, const char* include_dir, bool debug, bool verbose, const char* output_file)
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

    const char *opts[] = 
    {   
        "--device-as-default-execution-space",
        "--gpu-architecture=compute_52",
//        "--use_fast_math",
        "--std=c++11",
        "--define-macro=WP_CUDA",
        "--define-macro=WP_NO_CRT",
        "--define-macro=NDEBUG",
        include_opt
    };

    res = nvrtcCompileProgram(prog, 7, opts);

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

void* cuda_load_module(const char* path)
{
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

    CUmodule module = NULL;
    CUresult res = cuModuleLoadDataEx_f(&module, buf, 0, 0, 0);
    if (res != CUDA_SUCCESS)
        printf("Warp: Loading PTX module failed with error: %d\n", res);

    free(buf);

    return module;
}

void cuda_unload_module(void* module)
{
    cuModuleUnload_f((CUmodule)module);
}

void* cuda_get_kernel(void* module, const char* name)
{
    CUfunction kernel = NULL;
    CUresult res = cuModuleGetFunction_f(&kernel, (CUmodule)module, name);
    if (res != CUDA_SUCCESS)
        printf("Warp: Failed to lookup kernel function %s in module\n", name);

    return kernel;
}

size_t cuda_launch_kernel(void* kernel, size_t dim, void** args)
{
    const int block_dim = 256;
    const int grid_dim = (dim + block_dim - 1)/block_dim;

    CUresult res = cuLaunchKernel_f(
        (CUfunction)kernel,
        grid_dim, 1, 1,
        block_dim, 1, 1,
        0, g_cuda_stream,
        args,
        0);

    return res;

}

// impl. files
#include "bvh.cu"
#include "mesh.cu"
#include "sort.cu"
#include "hashgrid.cu"

//#include "spline.inl"
//#include "volume.inl"

