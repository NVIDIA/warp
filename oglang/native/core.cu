#include "core.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

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
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice));
}

void memcpy_d2h(void* dest, void* src, size_t n)
{
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost));
}

void memcpy_d2d(void* dest, void* src, size_t n)
{
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToDevice));
}

void memset_device(void* dest, int value, size_t n)
{
    check_cuda(cudaMemsetAsync(dest, value, n));
}

void synchronize()
{
    check_cuda(cudaStreamSynchronize(0));
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


#if defined(__linux__)
#include <dlfcn.h>
static void* GetProcAddress(void* handle, const char* name) { return dlsym(handle, name); }
#endif

#if defined(_WIN32)
#include <windows.h>
#endif

typedef CUresult CUDAAPI cuInit_t(unsigned int);
typedef CUresult CUDAAPI cuDeviceGet_t(CUdevice *dev, int ordinal);
typedef CUresult CUDAAPI cuCtxGetCurrent_t(CUcontext* ctx);
typedef CUresult CUDAAPI cuCtxSetCurrent_t(CUcontext ctx);
typedef CUresult CUDAAPI cuCtxCreate_t(CUcontext* pctx, unsigned int flags, CUdevice dev);
typedef CUresult CUDAAPI cuCtxDestroy_t(CUcontext pctx);

static cuInit_t* cuInit_f;
static cuCtxGetCurrent_t* cuCtxGetCurrent_f;
static cuCtxSetCurrent_t* cuCtxSetCurrent_f;
//static cuCtxCreate_t* cuCtxCreate_f;
//static cuCtxDestroy_t* cuCtxDestroy_f;
//static cuDeviceGet_t* cuDeviceGet_f;

static CUcontext g_cuda_context;
static CUcontext g_save_context;

bool cuda_init()
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
//	cuCtxCreate_f = (cuCtxCreate_t*)GetProcAddress(hCudaDriver, "cuCtxCreate");
//	cuCtxDestroy_f = (cuCtxDestroy_t*)GetProcAddress(hCudaDriver, "cuCtxDestroy");
//	cuDeviceGet_f = (cuDeviceGet_t*)GetProcAddress(hCudaDriver, "cuDeviceGet");

    if (cuInit_f == NULL)
        return false;

    if (cuInit_f(0) != CUDA_SUCCESS)
		return false;

    CUcontext ctx;
    cuCtxGetCurrent_f(&ctx);

    if (ctx == NULL)
    {
        // create a new default runtime context
        cudaSetDevice(0);
        cuCtxGetCurrent_f(&ctx);
    }
    
    g_cuda_context = ctx;
    
    return true;
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
	CUcontext ctx;
	if (cuCtxGetCurrent_f(&ctx) == CUDA_SUCCESS)
	    return ctx;
    else
        return NULL;
}

void cuda_set_context(void* ctx)
{
    cuCtxSetCurrent_f((CUcontext)ctx);
}

// impl. files
#include "bvh.cu"
#include "mesh.cu"
//#include "spline.inl"
//#include "volume.inl"

