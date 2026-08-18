// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
#error "hip_util.h should only be included for HIP builds."
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#ifndef HIP_VERSION
#if defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR) && defined(HIP_VERSION_PATCH)
#define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)
#else
#define HIP_VERSION 0
#endif  // defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR) && defined(HIP_VERSION_PATCH)
#endif  // HIP_VERSION
#ifndef CUDA_VERSION
#define CUDA_VERSION HIP_VERSION
#endif  // CUDA_VERSION
#ifndef NVRTC_SUCCESS
#define NVRTC_SUCCESS HIPRTC_SUCCESS
#endif  // NVRTC_SUCCESS
#ifndef nvrtcGetErrorString
#define nvrtcGetErrorString hiprtcGetErrorString
#endif  // nvrtcGetErrorString
#ifndef nvrtcCreateProgram
#define nvrtcCreateProgram hiprtcCreateProgram
#endif  // nvrtcCreateProgram
#ifndef nvrtcCompileProgram
#define nvrtcCompileProgram hiprtcCompileProgram
#endif  // nvrtcCompileProgram
#ifndef nvrtcDestroyProgram
#define nvrtcDestroyProgram hiprtcDestroyProgram
#endif  // nvrtcDestroyProgram
#ifndef nvrtcGetProgramLogSize
#define nvrtcGetProgramLogSize hiprtcGetProgramLogSize
#endif  // nvrtcGetProgramLogSize
#ifndef nvrtcGetProgramLog
#define nvrtcGetProgramLog hiprtcGetProgramLog
#endif  // nvrtcGetProgramLog
#ifndef nvrtcGetPTXSize
#define nvrtcGetPTXSize hiprtcGetCodeSize
#endif  // nvrtcGetPTXSize
#ifndef nvrtcGetPTX
#define nvrtcGetPTX hiprtcGetCode
#endif  // nvrtcGetPTX
#ifndef nvrtcGetCUBINSize
#define nvrtcGetCUBINSize hiprtcGetBitcodeSize
#endif  // nvrtcGetCUBINSize
#ifndef nvrtcGetCUBIN
#define nvrtcGetCUBIN hiprtcGetBitcode
#endif  // nvrtcGetCUBIN
#if defined(nvrtcGetNumSupportedArchs)
#undef nvrtcGetNumSupportedArchs
#endif  // defined(nvrtcGetNumSupportedArchs)
static inline hiprtcResult nvrtcGetNumSupportedArchs(int* count)
{
    if (count) {
        *count = 0;
    }
    return HIPRTC_SUCCESS;
}

#if defined(nvrtcGetSupportedArchs)
#undef nvrtcGetSupportedArchs
#endif  // defined(nvrtcGetSupportedArchs)
static inline hiprtcResult nvrtcGetSupportedArchs(int* archs)
{
    (void)archs;
    return HIPRTC_SUCCESS;
}
#ifndef nvrtcVersion
#define nvrtcVersion hiprtcVersion
#endif  // nvrtcVersion
#ifndef CUDAAPI
#define CUDAAPI
#endif  // CUDAAPI
#if defined(CUDART_CB)
#undef CUDART_CB
#endif  // defined(CUDART_CB)
#define CUDART_CB
#ifndef CU_GET_PROC_ADDRESS_DEFAULT
#define CU_GET_PROC_ADDRESS_DEFAULT 0
#endif  // CU_GET_PROC_ADDRESS_DEFAULT
#ifndef CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
#define CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
#endif  // CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
#ifndef CU_IPC_HANDLE_SIZE
#define CU_IPC_HANDLE_SIZE sizeof(CUipcMemHandle)
#endif  // CU_IPC_HANDLE_SIZE
#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS hipSuccess
#endif  // CUDA_SUCCESS
#ifndef cudaErrorInvalidValue
#define cudaErrorInvalidValue hipErrorInvalidValue
#endif  // cudaErrorInvalidValue
#ifndef cudaSuccess
#define cudaSuccess hipSuccess
#endif  // cudaSuccess
#ifndef cudaGetErrorString
#define cudaGetErrorString hipGetErrorString
#endif  // cudaGetErrorString
#ifndef cudaGetLastError
#define cudaGetLastError hipGetLastError
#endif  // cudaGetLastError
#ifndef cudaDeviceSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#endif  // cudaDeviceSynchronize
#ifndef cudaGetDevice
#define cudaGetDevice hipGetDevice
#endif  // cudaGetDevice
#ifndef cudaGetDeviceCount
#define cudaGetDeviceCount hipGetDeviceCount
#endif  // cudaGetDeviceCount
#ifndef cudaGetDeviceProperties
#define cudaGetDeviceProperties hipGetDeviceProperties
#endif  // cudaGetDeviceProperties
#ifndef cudaDeviceCanAccessPeer
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#endif  // cudaDeviceCanAccessPeer
#ifndef cudaPointerGetAttributes
#define cudaPointerGetAttributes hipPointerGetAttributes
#endif  // cudaPointerGetAttributes
#ifndef cudaMemcpy
#define cudaMemcpy hipMemcpy
#endif  // cudaMemcpy
#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync hipMemcpyAsync
#endif  // cudaMemcpyAsync
#ifndef cudaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#endif  // cudaMemcpyDeviceToDevice
#ifndef cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#endif  // cudaMemcpyDeviceToHost
#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#endif  // cudaMemcpyHostToDevice
#ifndef cudaMemcpyHostToHost
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#endif  // cudaMemcpyHostToHost
#ifndef cudaMemcpyDefault
#define cudaMemcpyDefault hipMemcpyDefault
#endif  // cudaMemcpyDefault
#ifndef cudaMemset
#define cudaMemset hipMemset
#endif  // cudaMemset
#ifndef cudaMemsetAsync
#define cudaMemsetAsync hipMemsetAsync
#endif  // cudaMemsetAsync
#ifndef cudaMalloc
#define cudaMalloc hipMalloc
#endif  // cudaMalloc
#ifndef cudaMallocHost
#define cudaMallocHost hipHostMalloc
#endif  // cudaMallocHost
#ifndef cudaFree
#define cudaFree hipFree
#endif  // cudaFree
#ifndef cudaFreeHost
#define cudaFreeHost hipHostFree
#endif  // cudaFreeHost
#ifndef cudaMallocManaged
#define cudaMallocManaged hipMallocManaged
#endif  // cudaMallocManaged
#ifndef cudaMallocAsync
#define cudaMallocAsync hipMallocAsync
#endif  // cudaMallocAsync
#ifndef cudaFreeAsync
#define cudaFreeAsync hipFreeAsync
#endif  // cudaFreeAsync
#ifndef cudaDeviceGetDefaultMemPool
#define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#endif  // cudaDeviceGetDefaultMemPool
#ifndef cudaMemPoolSetAttribute
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#endif  // cudaMemPoolSetAttribute
#ifndef cudaMemPoolGetAttribute
#define cudaMemPoolGetAttribute hipMemPoolGetAttribute
#endif  // cudaMemPoolGetAttribute
#ifndef cudaMemPoolGetAccess
#define cudaMemPoolGetAccess hipMemPoolGetAccess
#endif  // cudaMemPoolGetAccess
#ifndef cudaMemPoolSetAccess
#define cudaMemPoolSetAccess hipMemPoolSetAccess
#endif  // cudaMemPoolSetAccess
#ifndef cudaStreamSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#endif  // cudaStreamSynchronize
#ifndef cudaStreamIsCapturing
#define cudaStreamIsCapturing hipStreamIsCapturing
#endif  // cudaStreamIsCapturing
#ifndef cudaStreamBeginCapture
#define cudaStreamBeginCapture hipStreamBeginCapture
#endif  // cudaStreamBeginCapture
#ifndef cudaStreamBeginCaptureToGraph
#define cudaStreamBeginCaptureToGraph hipStreamBeginCaptureToGraph
#endif  // cudaStreamBeginCaptureToGraph
#ifndef cudaStreamEndCapture
#define cudaStreamEndCapture hipStreamEndCapture
#endif  // cudaStreamEndCapture
// NOTE: cudaStreamSetCaptureDependencies is mapped to hipStreamSetCaptureDependencies
// further below. Do not add a literal fallback (e.g. `0`) here: because this header
// is included first, a literal definition would win over the function mapping and
// turn call sites into `0(...)`, which fails to compile.
#ifndef cudaEventCreate
#define cudaEventCreate hipEventCreate
#endif  // cudaEventCreate
#ifndef cudaEventRecord
#define cudaEventRecord hipEventRecord
#endif  // cudaEventRecord
#ifndef cudaEventSynchronize
#define cudaEventSynchronize hipEventSynchronize
#endif  // cudaEventSynchronize
#ifndef cudaEventElapsedTime
#define cudaEventElapsedTime hipEventElapsedTime
#endif  // cudaEventElapsedTime
#ifndef cudaEventDestroy
#define cudaEventDestroy hipEventDestroy
#endif  // cudaEventDestroy
#ifndef cudaMemAdvise
#define cudaMemAdvise hipMemAdvise
#endif  // cudaMemAdvise
#ifndef cudaMemPrefetchAsync
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#endif  // cudaMemPrefetchAsync
#ifndef cudaFuncSetAttribute
#define cudaFuncSetAttribute hipFuncSetAttribute
#endif  // cudaFuncSetAttribute
#ifndef cudaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#endif  // cudaFuncAttributeMaxDynamicSharedMemorySize
#ifndef CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES hipFuncAttributeMaxDynamicSharedMemorySize
#endif  // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#ifndef cudaCpuDeviceId
#define cudaCpuDeviceId hipCpuDeviceId
#endif  // cudaCpuDeviceId
#ifndef cudaInvalidDeviceId
#define cudaInvalidDeviceId hipInvalidDeviceId
#endif  // cudaInvalidDeviceId
#ifndef cudaGraphDestroy
#define cudaGraphDestroy hipGraphDestroy
#endif  // cudaGraphDestroy
#ifndef cudaGraphExecDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#endif  // cudaGraphExecDestroy
#ifndef cudaGraphAddMemFreeNode
#define cudaGraphAddMemFreeNode hipGraphAddMemFreeNode
#endif  // cudaGraphAddMemFreeNode
#ifndef cudaGraphAddMemcpyNode1D
#define cudaGraphAddMemcpyNode1D hipGraphAddMemcpyNode1D
#endif  // cudaGraphAddMemcpyNode1D

#ifndef CU_STREAM_ADD_CAPTURE_DEPENDENCIES
#define CU_STREAM_ADD_CAPTURE_DEPENDENCIES 0
#endif  // CU_STREAM_ADD_CAPTURE_DEPENDENCIES
#ifndef CU_STREAM_CAPTURE_STATUS_NONE
#define CU_STREAM_CAPTURE_STATUS_NONE hipStreamCaptureStatusNone
#endif  // CU_STREAM_CAPTURE_STATUS_NONE
#ifndef CU_STREAM_CAPTURE_STATUS_ACTIVE
#define CU_STREAM_CAPTURE_STATUS_ACTIVE hipStreamCaptureStatusActive
#endif  // CU_STREAM_CAPTURE_STATUS_ACTIVE
#ifndef cudaGraphExecMemcpyNodeSetParams1D
#define cudaGraphExecMemcpyNodeSetParams1D hipGraphExecMemcpyNodeSetParams1D
#endif  // cudaGraphExecMemcpyNodeSetParams1D
#ifndef cudaGraphInstantiateWithFlags
#define cudaGraphInstantiateWithFlags hipGraphInstantiateWithFlags
#endif  // cudaGraphInstantiateWithFlags
#ifndef cudaGraphUpload
#define cudaGraphUpload hipGraphUpload
#endif  // cudaGraphUpload
#ifndef cudaGraphLaunch
#define cudaGraphLaunch hipGraphLaunch
#endif  // cudaGraphLaunch
#ifndef cudaGraphGetNodes
#define cudaGraphGetNodes hipGraphGetNodes
#endif  // cudaGraphGetNodes
#ifndef cudaGraphChildGraphNodeGetGraph
#define cudaGraphChildGraphNodeGetGraph hipGraphChildGraphNodeGetGraph
#endif  // cudaGraphChildGraphNodeGetGraph
#ifndef cudaGraphAddChildGraphNode
#define cudaGraphAddChildGraphNode hipGraphAddChildGraphNode
#endif  // cudaGraphAddChildGraphNode
#ifndef cudaGraphDebugDotPrint
#define cudaGraphDebugDotPrint hipGraphDebugDotPrint
#endif  // cudaGraphDebugDotPrint
#ifndef cudaUserObjectCreate
#define cudaUserObjectCreate hipUserObjectCreate
#endif  // cudaUserObjectCreate
#ifndef cudaGraphRetainUserObject
#define cudaGraphRetainUserObject hipGraphRetainUserObject
#endif  // cudaGraphRetainUserObject
#ifndef cudaStreamSetCaptureDependencies
#define cudaStreamSetCaptureDependencies hipStreamSetCaptureDependencies
#endif  // cudaStreamSetCaptureDependencies

#ifndef cudaGraphInstantiateFlagAutoFreeOnLaunch
#define cudaGraphInstantiateFlagAutoFreeOnLaunch hipGraphInstantiateFlagAutoFreeOnLaunch
#endif  // cudaGraphInstantiateFlagAutoFreeOnLaunch
#ifndef cudaGraphUserObjectMove
#define cudaGraphUserObjectMove hipGraphUserObjectMove
#endif  // cudaGraphUserObjectMove
#ifndef cudaUserObjectNoDestructorSync
#define cudaUserObjectNoDestructorSync hipUserObjectNoDestructorSync
#endif  // cudaUserObjectNoDestructorSync
#ifndef cudaMemPoolAttrReleaseThreshold
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#endif  // cudaMemPoolAttrReleaseThreshold
#ifndef cudaMemPoolAttrUsedMemCurrent
#define cudaMemPoolAttrUsedMemCurrent hipMemPoolAttrUsedMemCurrent
#endif  // cudaMemPoolAttrUsedMemCurrent
#ifndef cudaMemPoolAttrUsedMemHigh
#define cudaMemPoolAttrUsedMemHigh hipMemPoolAttrUsedMemHigh
#endif  // cudaMemPoolAttrUsedMemHigh
#ifndef cudaMemAccessFlagsProtNone
#define cudaMemAccessFlagsProtNone hipMemAccessFlagsProtNone
#endif  // cudaMemAccessFlagsProtNone
#ifndef cudaMemAccessFlagsProtReadWrite
#define cudaMemAccessFlagsProtReadWrite hipMemAccessFlagsProtReadWrite
#endif  // cudaMemAccessFlagsProtReadWrite
#ifndef cudaMemLocationTypeDevice
#define cudaMemLocationTypeDevice hipMemLocationTypeDevice
#endif  // cudaMemLocationTypeDevice
#ifndef cudaStreamCaptureStatusNone
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#endif  // cudaStreamCaptureStatusNone
#ifndef cudaStreamCaptureStatusActive
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#endif  // cudaStreamCaptureStatusActive
#ifndef cudaStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#endif  // cudaStreamCaptureModeThreadLocal

#ifndef CUDA_ERROR_NOT_INITIALIZED
#define CUDA_ERROR_NOT_INITIALIZED hipErrorNotInitialized
#endif  // CUDA_ERROR_NOT_INITIALIZED
#ifndef CUDA_ERROR_NOT_READY
#define CUDA_ERROR_NOT_READY hipErrorNotReady
#endif  // CUDA_ERROR_NOT_READY
#ifndef CUDA_ERROR_NOT_SUPPORTED
#define CUDA_ERROR_NOT_SUPPORTED hipErrorNotSupported
#endif  // CUDA_ERROR_NOT_SUPPORTED
#ifndef CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
#define CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED hipErrorPeerAccessAlreadyEnabled
#endif  // CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
#ifndef CUDA_ERROR_PEER_ACCESS_NOT_ENABLED
#define CUDA_ERROR_PEER_ACCESS_NOT_ENABLED hipErrorPeerAccessNotEnabled
#endif  // CUDA_ERROR_PEER_ACCESS_NOT_ENABLED
#ifndef cudaErrorCallRequiresNewerDriver
#define cudaErrorCallRequiresNewerDriver hipErrorCallRequiresNewerDriver
#endif  // cudaErrorCallRequiresNewerDriver
#ifndef CU_STREAM_DEFAULT
#define CU_STREAM_DEFAULT hipStreamDefault
#endif  // CU_STREAM_DEFAULT
#ifndef CU_EVENT_DEFAULT
#define CU_EVENT_DEFAULT hipEventDefault
#endif  // CU_EVENT_DEFAULT
#ifndef CU_EVENT_DISABLE_TIMING
#define CU_EVENT_DISABLE_TIMING hipEventDisableTiming
#endif  // CU_EVENT_DISABLE_TIMING
#ifndef CU_EVENT_RECORD_DEFAULT
#define CU_EVENT_RECORD_DEFAULT 0
#endif  // CU_EVENT_RECORD_DEFAULT
#ifndef CU_EVENT_WAIT_DEFAULT
#define CU_EVENT_WAIT_DEFAULT 0
#endif  // CU_EVENT_WAIT_DEFAULT
#ifndef CU_EVENT_RECORD_EXTERNAL
#define CU_EVENT_RECORD_EXTERNAL 0
#endif  // CU_EVENT_RECORD_EXTERNAL
#ifndef CU_EVENT_WAIT_EXTERNAL
#define CU_EVENT_WAIT_EXTERNAL 0
#endif  // CU_EVENT_WAIT_EXTERNAL
#ifndef CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
#define CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS 0
#endif  // CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
#ifndef WP_HAS_MEMCPY_BATCH
#define WP_HAS_MEMCPY_BATCH (HIP_VERSION >= 70100000)
#endif  // WP_HAS_MEMCPY_BATCH

#ifndef CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
#if WP_HAS_MEMCPY_BATCH
#define CU_MEMCPY_SRC_ACCESS_ORDER_STREAM hipMemcpySrcAccessOrderStream
#else
#define CU_MEMCPY_SRC_ACCESS_ORDER_STREAM 0
#endif  // WP_HAS_MEMCPY_BATCH
#endif  // CU_MEMCPY_SRC_ACCESS_ORDER_STREAM

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaPointerAttributes = hipPointerAttribute_t;
using cudaMemcpyKind = hipMemcpyKind;
using cudaMemoryAdvise = hipMemoryAdvise;
using cudaStreamCaptureStatus = hipStreamCaptureStatus;
using cudaStreamCaptureMode = hipStreamCaptureMode;
using cudaGraph_t = hipGraph_t;
using cudaGraphNode_t = hipGraphNode_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaMemPool_t = hipMemPool_t;
using cudaMemAccessFlags = hipMemAccessFlags;
using cudaMemLocation = hipMemLocation;
using cudaMemAccessDesc = hipMemAccessDesc;
using cudaUserObject_t = hipUserObject_t;
using cudaResourceDesc = hipResourceDesc;
using cudaArray_t = hipArray_t;
using cudaSurfaceObject_t = hipSurfaceObject_t;
#ifndef cudaResourceTypeArray
#define cudaResourceTypeArray hipResourceTypeArray
#endif  // cudaResourceTypeArray
#ifndef cudaCreateSurfaceObject
#define cudaCreateSurfaceObject hipCreateSurfaceObject
#endif  // cudaCreateSurfaceObject
#ifndef cudaDestroySurfaceObject
#define cudaDestroySurfaceObject hipDestroySurfaceObject
#endif  // cudaDestroySurfaceObject
using nvrtcProgram = hiprtcProgram;
using nvrtcResult = hiprtcResult;
using CUresult = hipError_t;
using CUdevice = hipDevice_t;
struct HipContext {
    int device;
};
using CUcontext = HipContext*;
using CUstream = hipStream_t;
using CUevent = hipEvent_t;
using CUmodule = hipModule_t;
using CUfunction = hipFunction_t;
using CUdeviceptr = hipDeviceptr_t;
using CUuuid = hipUUID;
using CUdevice_attribute = hipDeviceAttribute_t;
using CUipcEventHandle = hipIpcEventHandle_t;
using CUipcMemHandle = hipIpcMemHandle_t;
using cuuint64_t = uint64_t;
using CUgraphicsResource = hipGraphicsResource_t;
using CUarray = hipArray_t;
using CUtexObject = hipTextureObject_t;
using CUgraph = hipGraph_t;
using CUgraphNode = hipGraphNode_t;
using CUgraphNodeType = hipGraphNodeType;
using CUgraphNodeParams = void;
using CUgraphEdgeData = void;
using CUstreamCaptureStatus = hipStreamCaptureStatus;
using CUjit_option = int;
using CUpointer_attribute = hipPointer_attribute;
using CUfunction_attribute = hipFuncAttribute;
// HIP has no equivalent of the CUDA `CUoccupancyB2DSize` callback (the only HIP
// equivalents take a fixed `size_t` shared-memory size). Provide a stub typedef
// so call sites that reference the type compile under HIP; HIP code paths must
// ignore the callback (see cuOccupancyMaxPotentialBlockSize_f).
typedef size_t(CUDAAPI* CUoccupancyB2DSize)(int blockSize);
#if HIP_VERSION >= 70100000
using CUmemcpyAttributes = hipMemcpyAttributes;
#else
struct CUmemcpyAttributes {
    int dummy;
};
#endif  // HIP_VERSION >= 70100000
using CUDA_ARRAY_DESCRIPTOR = HIP_ARRAY_DESCRIPTOR;
using CUDA_ARRAY3D_DESCRIPTOR = HIP_ARRAY3D_DESCRIPTOR;
#if HIP_VERSION >= 70000000
using CUDA_MEMCPY2D = hip_Memcpy2D;
using CUDA_MEMCPY3D = HIP_MEMCPY3D;
using CUDA_RESOURCE_DESC = HIP_RESOURCE_DESC;
using CUDA_TEXTURE_DESC = HIP_TEXTURE_DESC;
using CUDA_RESOURCE_VIEW_DESC = HIP_RESOURCE_VIEW_DESC;

using CUarray_format = hipArray_Format;
using CUaddress_mode = HIPaddress_mode;

#ifndef CU_AD_FORMAT_UNSIGNED_INT8
#define CU_AD_FORMAT_UNSIGNED_INT8 HIP_AD_FORMAT_UNSIGNED_INT8
#endif  // CU_AD_FORMAT_UNSIGNED_INT8
#ifndef CU_AD_FORMAT_UNSIGNED_INT16
#define CU_AD_FORMAT_UNSIGNED_INT16 HIP_AD_FORMAT_UNSIGNED_INT16
#endif  // CU_AD_FORMAT_UNSIGNED_INT16
#ifndef CU_AD_FORMAT_UNSIGNED_INT32
#define CU_AD_FORMAT_UNSIGNED_INT32 HIP_AD_FORMAT_UNSIGNED_INT32
#endif  // CU_AD_FORMAT_UNSIGNED_INT32
#ifndef CU_AD_FORMAT_SIGNED_INT8
#define CU_AD_FORMAT_SIGNED_INT8 HIP_AD_FORMAT_SIGNED_INT8
#endif  // CU_AD_FORMAT_SIGNED_INT8
#ifndef CU_AD_FORMAT_SIGNED_INT16
#define CU_AD_FORMAT_SIGNED_INT16 HIP_AD_FORMAT_SIGNED_INT16
#endif  // CU_AD_FORMAT_SIGNED_INT16
#ifndef CU_AD_FORMAT_SIGNED_INT32
#define CU_AD_FORMAT_SIGNED_INT32 HIP_AD_FORMAT_SIGNED_INT32
#endif  // CU_AD_FORMAT_SIGNED_INT32
#ifndef CU_AD_FORMAT_HALF
#define CU_AD_FORMAT_HALF HIP_AD_FORMAT_HALF
#endif  // CU_AD_FORMAT_HALF
#ifndef CU_AD_FORMAT_FLOAT
#define CU_AD_FORMAT_FLOAT HIP_AD_FORMAT_FLOAT
#endif  // CU_AD_FORMAT_FLOAT

#ifndef CU_TR_ADDRESS_MODE_WRAP
#define CU_TR_ADDRESS_MODE_WRAP HIP_TR_ADDRESS_MODE_WRAP
#endif  // CU_TR_ADDRESS_MODE_WRAP
#ifndef CU_TR_ADDRESS_MODE_CLAMP
#define CU_TR_ADDRESS_MODE_CLAMP HIP_TR_ADDRESS_MODE_CLAMP
#endif  // CU_TR_ADDRESS_MODE_CLAMP
#ifndef CU_TR_ADDRESS_MODE_MIRROR
#define CU_TR_ADDRESS_MODE_MIRROR HIP_TR_ADDRESS_MODE_MIRROR
#endif  // CU_TR_ADDRESS_MODE_MIRROR
#ifndef CU_TR_ADDRESS_MODE_BORDER
#define CU_TR_ADDRESS_MODE_BORDER HIP_TR_ADDRESS_MODE_BORDER
#endif  // CU_TR_ADDRESS_MODE_BORDER

#ifndef CU_TR_FILTER_MODE_POINT
#define CU_TR_FILTER_MODE_POINT HIP_TR_FILTER_MODE_POINT
#endif  // CU_TR_FILTER_MODE_POINT
#ifndef CU_TR_FILTER_MODE_LINEAR
#define CU_TR_FILTER_MODE_LINEAR HIP_TR_FILTER_MODE_LINEAR
#endif  // CU_TR_FILTER_MODE_LINEAR

#ifndef CU_TRSF_NORMALIZED_COORDINATES
#define CU_TRSF_NORMALIZED_COORDINATES HIP_TRSF_NORMALIZED_COORDINATES
#endif  // CU_TRSF_NORMALIZED_COORDINATES

#ifndef CU_RESOURCE_TYPE_ARRAY
#define CU_RESOURCE_TYPE_ARRAY HIP_RESOURCE_TYPE_ARRAY
#endif  // CU_RESOURCE_TYPE_ARRAY

#ifndef CU_MEMORYTYPE_HOST
#define CU_MEMORYTYPE_HOST hipMemoryTypeHost
#endif  // CU_MEMORYTYPE_HOST
#ifndef CU_MEMORYTYPE_DEVICE
#define CU_MEMORYTYPE_DEVICE hipMemoryTypeDevice
#endif  // CU_MEMORYTYPE_DEVICE
#ifndef CU_MEMORYTYPE_ARRAY
#define CU_MEMORYTYPE_ARRAY hipMemoryTypeArray
#endif  // CU_MEMORYTYPE_ARRAY
using CUmemorytype = hipMemoryType;
#ifndef CUDA_ARRAY3D_SURFACE_LDST
#define CUDA_ARRAY3D_SURFACE_LDST hipArraySurfaceLoadStore
#endif  // CUDA_ARRAY3D_SURFACE_LDST
#else
// Warp's texture path (texture.cpp) uses the driver-style HIP_RESOURCE_DESC layout
// (res.array.hArray / res.mipmap.hMipmappedArray / flags). The runtime hipResourceDesc
// available before ROCm 7.0 exposes different member names (res.array.array /
// res.mipmap.mipmap and has no flags member), so those descriptors do not compile
// there. Require ROCm 7.0+ for the HIP resource/texture descriptor path.
#error "Warp's HIP build requires ROCm 7.0 or newer (driver-style resource descriptors)."
#endif  // HIP_VERSION >= 70000000

#ifndef CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID hipDeviceAttributePciDomainId
#endif  // CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID
#ifndef CU_DEVICE_ATTRIBUTE_PCI_BUS_ID
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID hipDeviceAttributePciBusId
#endif  // CU_DEVICE_ATTRIBUTE_PCI_BUS_ID
#ifndef CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID hipDeviceAttributePciDeviceId
#endif  // CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID
#ifndef CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
#define CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING hipDeviceAttributeUnifiedAddressing
#endif  // CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
#ifndef CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED
#define CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED hipDeviceAttributeMemoryPoolsSupported
#endif  // CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED
#ifndef CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT hipDeviceAttributeMultiprocessorCount
#endif  // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
#ifndef CU_DEVICE_ATTRIBUTE_INTEGRATED
#define CU_DEVICE_ATTRIBUTE_INTEGRATED hipDeviceAttributeIntegrated
#endif  // CU_DEVICE_ATTRIBUTE_INTEGRATED
#ifndef CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN hipDeviceAttributeSharedMemPerBlockOptin
#endif  // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
#ifndef CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR hipDeviceAttributeComputeCapabilityMajor
#endif  // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#ifndef CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR hipDeviceAttributeComputeCapabilityMinor
#endif  // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
// HIP has no equivalent of CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED. Map it to an
// invalid sentinel (-1): a cuDeviceGetAttribute query with this value fails at
// runtime rather than reporting a value, so callers must treat the query error as
// "capability not supported" (mirrors the CUoccupancyB2DSize contract above).
#ifndef CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED
#define CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED ((CUdevice_attribute)-1)
#endif  // CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED
