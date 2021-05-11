# CUDA driver API wrapper from https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA/tree/master/Chapter10

from ctypes import *
import sys

if 'linux' in sys.platform:
	cuda = CDLL('libcuda.so')
elif 'win' in sys.platform:
	cuda = CDLL('nvcuda.dll')

if (cuda != None):
        
    CUDA_ERRORS = {0 : 'CUDA_SUCCESS', 1 : 'CUDA_ERROR_INVALID_VALUE', 200 : 'CUDA_ERROR_INVALID_IMAGE', 201 : 'CUDA_ERROR_INVALID_CONTEXT ', 400 : 'CUDA_ERROR_INVALID_HANDLE' }

    cuInit = cuda.cuInit
    cuInit.argtypes = [c_uint]
    cuInit.restype = int

    cuDeviceGetCount = cuda.cuDeviceGetCount
    cuDeviceGetCount.argtypes = [POINTER(c_int)]
    cuDeviceGetCount.restype = int

    cuDeviceGet = cuda.cuDeviceGet
    cuDeviceGet.argtypes = [POINTER(c_int), c_int]
    cuDeviceGet.restype = int

    cuCtxCreate = cuda.cuCtxCreate_v2
    cuCtxCreate.argtypes = [c_void_p, c_uint, c_int]
    cuCtxCreate.restype = int

    cuDevicePrimaryCtxRetain = cuda.cuDevicePrimaryCtxRetain
    cuDevicePrimaryCtxRetain.argtypes = [c_void_p, c_int]
    cuDevicePrimaryCtxRetain.restype = int

    cuCtxSetCurrent = cuda.cuCtxSetCurrent 
    cuCtxSetCurrent.argtypes = [c_void_p]
    cuCtxSetCurrent.restype = int

    cuCtxGetCurrent = cuda.cuCtxGetCurrent
    cuCtxGetCurrent.argtypes = [c_void_p]
    cuCtxGetCurrent.restype = int

    cuCtxPushCurrent = cuda.cuCtxPushCurrent_v2 
    cuCtxPushCurrent.argtypes = [c_void_p]
    cuCtxPushCurrent.restype = int

    cuCtxPopCurrent = cuda.cuCtxPopCurrent_v2
    cuCtxPopCurrent.argtypes = [c_void_p]
    cuCtxPopCurrent.restype = int

    #CUresult cuDevicePrimaryCtxGetState ( CUdevice dev, unsigned int* flags, int* active ) 
    cuDevicePrimaryCtxGetState = cuda.cuDevicePrimaryCtxGetState
    cuDevicePrimaryCtxGetState.argtypes = [c_int, POINTER(c_uint), POINTER(c_int)]
    cuDevicePrimaryCtxGetState.restype = int

    #CUresult cuCtxGetApiVersion ( CUcontext ctx, unsigned int* version ) 
    cuCtxGetApiVersion = cuda.cuCtxGetApiVersion
    cuCtxGetApiVersion.argtypes = [c_void_p, POINTER(c_uint)]
    cuCtxGetApiVersion.restype = int


    cuModuleLoad = cuda.cuModuleLoad
    cuModuleLoad.argtypes = [c_void_p, c_char_p]
    cuModuleLoad.restype  = int

    cuCtxSynchronize = cuda.cuCtxSynchronize
    cuCtxSynchronize.argtypes = []
    cuCtxSynchronize.restype = int

    cuModuleGetFunction = cuda.cuModuleGetFunction
    cuModuleGetFunction.argtypes = [c_void_p, c_void_p, c_char_p ]
    cuModuleGetFunction.restype = int

    # alloc functions
    cuMemAlloc = cuda.cuMemAlloc_v2
    cuMemAlloc.argtypes = [c_void_p, c_size_t]
    cuMemAlloc.restype = int

    cuMemFree = cuda.cuMemFree_v2
    cuMemFree.argtypes = [c_void_p] 
    cuMemFree.restype = int

    # mempcpy
    cuMemcpyHtoD = cuda.cuMemcpyHtoD_v2
    cuMemcpyHtoD.argtypes = [c_void_p, c_void_p, c_size_t]
    cuMemcpyHtoD.restype = int

    cuMemcpyDtoH = cuda.cuMemcpyDtoH_v2 
    cuMemcpyDtoH.argtypes = [c_void_p, c_void_p, c_size_t]
    cuMemcpyDtoH.restype = int

    # memcpy async
    cuMemcpyHtoDAsync = cuda.cuMemcpyHtoDAsync_v2 
    cuMemcpyHtoDAsync.argtypes = [c_void_p, c_void_p, c_size_t, c_void_p]
    cuMemcpyHtoDAsync.restype = int

    cuMemcpyDtoHAsync = cuda.cuMemcpyDtoHAsync_v2
    cuMemcpyDtoHAsync.argtypes = [c_void_p, c_void_p, c_size_t, c_void_p]
    cuMemcpyDtoHAsync.restype = int

    cuMemsetD32Async = cuda.cuMemsetD32Async 
    cuMemsetD32Async.argtypes = [c_void_p, c_uint, c_size_t, c_void_p]
    cuMemsetD32Async.restype = int

    cuLaunchKernel = cuda.cuLaunchKernel
    cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_void_p, c_void_p, c_void_p]
    cuLaunchKernel.restype = int

    cuCtxDestroy = cuda.cuCtxDestroy
    cuCtxDestroy.argtypes = [c_void_p]
    cuCtxDestroy.restype = int

    cuCtxAttach = cuda.cuCtxAttach
    cuCtxAttach.argtypes = [c_void_p, c_uint]
    cuCtxAttach.restype = int

