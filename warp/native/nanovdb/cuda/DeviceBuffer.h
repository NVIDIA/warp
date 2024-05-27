// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file DeviceBuffer.h

    \author Ken Museth

    \date January 8, 2020

    \brief Implements a simple dual (host/device) CUDA buffer.

    \note This file has no device-only kernel functions,
          which explains why it's a .h and not .cuh file.
*/

#ifndef NANOVDB_CUDA_DEVICEBUFFER_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_DEVICEBUFFER_H_HAS_BEEN_INCLUDED

#include <nanovdb/HostBuffer.h>// for BufferTraits
#include <nanovdb/util/cuda/Util.h>// for cudaMalloc/cudaMallocManaged/cudaFree

namespace nanovdb {// ================================================================

namespace cuda {// ===================================================================

// ----------------------------> DeviceBuffer <--------------------------------------

/// @brief Simple memory buffer using un-managed pinned host memory when compiled with NVCC.
///        Obviously this class is making explicit used of CUDA so replace it with your own memory
///        allocator if you are not using CUDA.
/// @note  While CUDA's pinned host memory allows for asynchronous memory copy between host and device
///        it is significantly slower then cached (un-pinned) memory on the host.
class DeviceBuffer
{
    uint64_t mSize; // total number of bytes managed by this buffer (assumed to be identical for host and device)
    void *mCpuData, *mGpuData; // raw pointers to the host and device buffers
    bool mManaged;

public:
    /// @brief Static factory method that return an instance of this buffer
    /// @param size byte size of buffer to be initialized
    /// @param dummy this argument is currently ignored but required to match the API of the HostBuffer
    /// @param host If true buffer is initialized only on the host/CPU, else on the device/GPU
    /// @param stream optional stream argument (defaults to stream NULL)
    /// @return An instance of this class using move semantics
    static DeviceBuffer create(uint64_t size, const DeviceBuffer* dummy = nullptr, bool host = true, void* stream = nullptr);

    /// @brief Static factory method that return an instance of this buffer that wraps externally managed memory
    /// @param size byte size of buffer specified by external memory
    /// @param cpuData pointer to externally managed host memory
    /// @param gpuData pointer to externally managed device memory
    /// @return An instance of this class using move semantics
    static DeviceBuffer create(uint64_t size, void* cpuData, void* gpuData);

    /// @brief Constructor
    /// @param size byte size of buffer to be initialized
    /// @param host If true buffer is initialized only on the host/CPU, else on the device/GPU
    /// @param stream optional stream argument (defaults to stream NULL)
    DeviceBuffer(uint64_t size = 0, bool host = true, void* stream = nullptr)
        : mSize(0)
        , mCpuData(nullptr)
        , mGpuData(nullptr)
        , mManaged(false)
    {
        if (size > 0) this->init(size, host, stream);
    }

    DeviceBuffer(uint64_t size, void* cpuData, void* gpuData)
        : mSize(size)
        , mCpuData(cpuData)
        , mGpuData(gpuData)
        , mManaged(false)
    {
    }

    /// @brief Disallow copy-construction
    DeviceBuffer(const DeviceBuffer&) = delete;

    /// @brief Move copy-constructor
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : mSize(other.mSize)
        , mCpuData(other.mCpuData)
        , mGpuData(other.mGpuData)
        , mManaged(other.mManaged)
    {
        other.mSize = 0;
        other.mCpuData = nullptr;
        other.mGpuData = nullptr;
        other.mManaged = false;
    }

    /// @brief Disallow copy assignment operation
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /// @brief Move copy assignment operation
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        this->clear();
        mSize = other.mSize;
        mCpuData = other.mCpuData;
        mGpuData = other.mGpuData;
        mManaged = other.mManaged;
        other.mSize = 0;
        other.mCpuData = nullptr;
        other.mGpuData = nullptr;
        other.mManaged = false;
        return *this;
    }

    /// @brief Destructor frees memory on both the host and device
    ~DeviceBuffer() { this->clear(); };

    /// @brief Initialize buffer
    /// @param size byte size of buffer to be initialized
    /// @param host If true buffer is initialized only on the host/CPU, else on the device/GPU
    /// @note All existing buffers are first cleared
    /// @warning size is expected to be non-zero. Use clear() clear buffer!
    void init(uint64_t size, bool host = true, void* stream = nullptr);

    /// @brief Retuns a raw pointer to the host/CPU buffer managed by this allocator.
    /// @warning Note that the pointer can be NULL!
    void* data() const { return mCpuData; }

    /// @brief Retuns a raw pointer to the device/GPU buffer managed by this allocator.
    /// @warning Note that the pointer can be NULL!
    void* deviceData() const { return mGpuData; }

    /// @brief  Upload this buffer from the host to the device, i.e. CPU -> GPU.
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    /// @param sync if false the memory copy is asynchronous
    /// @note If the device/GPU buffer does not exist it is first allocated
    /// @warning Assumes that the host/CPU buffer already exists
    void deviceUpload(void* stream = nullptr, bool sync = true) const;

    /// @brief Upload this buffer from the device to the host, i.e. GPU -> CPU.
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    /// @param sync if false the memory copy is asynchronous
    /// @note If the host/CPU buffer does not exist it is first allocated
    /// @warning Assumes that the device/GPU buffer already exists
    void deviceDownload(void* stream = nullptr, bool sync = true) const;

    /// @brief Returns the size in bytes of the raw memory buffer managed by this allocator.
    uint64_t size() const { return mSize; }

    //@{
    /// @brief Returns true if this allocator is empty, i.e. has no allocated memory
    bool empty() const { return mSize == 0; }
    bool isEmpty() const { return mSize == 0; }
    //@}

    /// @brief De-allocate all memory managed by this allocator and set all pointers to NULL
    void clear(void* stream = nullptr);

}; // DeviceBuffer class

// --------------------------> Implementations below <------------------------------------

inline DeviceBuffer DeviceBuffer::create(uint64_t size, const DeviceBuffer*, bool host, void* stream)
{
    return DeviceBuffer(size, host, stream);
}

inline DeviceBuffer DeviceBuffer::create(uint64_t size, void* cpuData, void* gpuData)
{
    return DeviceBuffer(size, cpuData, gpuData);
}

inline void DeviceBuffer::init(uint64_t size, bool host, void* stream)
{
    if (mSize>0) this->clear(stream);
    NANOVDB_ASSERT(size > 0);
    if (host) {
        cudaCheck(cudaMallocHost((void**)&mCpuData, size)); // un-managed pinned memory on the host (can be slow to access!). Always 32B aligned
        checkPtr(mCpuData, "cuda::DeviceBuffer::init: failed to allocate host buffer");
    } else {
        cudaCheck(util::cuda::mallocAsync((void**)&mGpuData, size, reinterpret_cast<cudaStream_t>(stream))); // un-managed memory on the device, always 32B aligned!
        checkPtr(mGpuData, "cuda::DeviceBuffer::init: failed to allocate device buffer");
    }
    mSize = size;
    mManaged = true;
} // DeviceBuffer::init

inline void DeviceBuffer::deviceUpload(void* stream, bool sync) const
{
    if (!mManaged) throw std::runtime_error("DeviceBuffer::deviceUpload called on externally managed memory. Replace deviceUpload call with the appropriate external copy operation.");

    checkPtr(mCpuData, "uninitialized cpu data");
    if (mGpuData == nullptr) {
        cudaCheck(util::cuda::mallocAsync((void**)&mGpuData, mSize, reinterpret_cast<cudaStream_t>(stream))); // un-managed memory on the device, always 32B aligned!
    }
    checkPtr(mGpuData, "uninitialized gpu data");
    cudaCheck(cudaMemcpyAsync(mGpuData, mCpuData, mSize, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
    if (sync) cudaCheck(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
} // DeviceBuffer::gpuUpload

inline void DeviceBuffer::deviceDownload(void* stream, bool sync) const
{
    if (!mManaged) throw std::runtime_error("DeviceBuffer::deviceDownload called on externally managed memory. Replace deviceDownload call with the appropriate external copy operation.");

    checkPtr(mGpuData, "uninitialized gpu data");
    if (mCpuData == nullptr) {
        cudaCheck(cudaMallocHost((void**)&mCpuData, mSize)); // un-managed pinned memory on the host (can be slow to access!). Always 32B aligned
    }
    checkPtr(mCpuData, "uninitialized cpu data");
    cudaCheck(cudaMemcpyAsync(mCpuData, mGpuData, mSize, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
    if (sync) cudaCheck(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
} // DeviceBuffer::gpuDownload

inline void DeviceBuffer::clear(void *stream)
{
    if (mManaged && mGpuData) cudaCheck(util::cuda::freeAsync(mGpuData, reinterpret_cast<cudaStream_t>(stream)));
    if (mManaged && mCpuData) cudaCheck(cudaFreeHost(mCpuData));
    mCpuData = mGpuData = nullptr;
    mSize = 0;
    mManaged = false;
} // DeviceBuffer::clear

}// namespace cuda

using CudaDeviceBuffer [[deprecated("Use nanovdb::cuda::DeviceBudder instead")]] = cuda::DeviceBuffer;

template<>
struct BufferTraits<cuda::DeviceBuffer>
{
    static constexpr bool hasDeviceDual = true;
};

}// namespace nanovdb

#endif // end of NANOVDB_CUDA_DEVICEBUFFER_H_HAS_BEEN_INCLUDED
