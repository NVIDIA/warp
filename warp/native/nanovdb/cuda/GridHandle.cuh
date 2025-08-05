// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/cuda/GridHandle.cuh

    \author Ken Museth, Doyub Kim

    \date August 3, 2023

    \brief Contains cuda kernels for GridHandle

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_CUDA_GRIDHANDLE_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_GRIDHANDLE_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/cuda/DeviceBuffer.h>// required for instantiation of move c-tor of GridHandle
#include <nanovdb/tools/cuda/GridChecksum.cuh>// for cuda::updateChecksum
#include <nanovdb/GridHandle.h>

namespace nanovdb {

namespace cuda {

namespace {// anonymous namespace
__global__ void cpyGridHandleMeta(const GridData *d_data, GridHandleMetaData *d_meta)
{
    nanovdb::cpyGridHandleMeta(d_data, d_meta);
}

__global__ void updateGridCount(GridData *d_data, uint32_t gridIndex, uint32_t gridCount, bool *d_dirty)
{
    NANOVDB_ASSERT(gridIndex < gridCount);
    *d_dirty = (d_data->mGridIndex != gridIndex) || (d_data->mGridCount != gridCount);
    if (*d_dirty) {
        d_data->mGridIndex = gridIndex;
        d_data->mGridCount = gridCount;
        if (d_data->mChecksum.isEmpty()) *d_dirty = false;// no need to update checksum if it didn't already exist
    }
}
}// anonymous namespace

template<typename BufferT, template <class, class...> class VectorT = std::vector>
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, VectorT<GridHandle<BufferT>>>::type
splitGridHandles(const GridHandle<BufferT> &handle, const BufferT* other = nullptr, cudaStream_t stream = 0)
{
    const void *ptr = handle.deviceData();
    if (ptr == nullptr) return VectorT<GridHandle<BufferT>>();
    VectorT<GridHandle<BufferT>> handles(handle.gridCount());
    bool dirty, *d_dirty;// use this to check if the checksum needs to be recomputed
    cudaCheck(util::cuda::mallocAsync((void**)&d_dirty, sizeof(bool), stream));
    int device = 0;
    cudaCheck(cudaGetDevice(&device));
    for (uint32_t n=0; n<handle.gridCount(); ++n) {
        auto buffer = BufferT::create(handle.gridSize(n), other, device, stream);
        GridData *dst = reinterpret_cast<GridData*>(buffer.deviceData());
        const GridData *src = reinterpret_cast<const GridData*>(ptr);
        cudaCheck(cudaMemcpyAsync(dst, src, handle.gridSize(n), cudaMemcpyDeviceToDevice, stream));
        updateGridCount<<<1, 1, 0, stream>>>(dst, 0u, 1u, d_dirty);
        cudaCheckError();
        cudaCheck(cudaMemcpyAsync(&dirty, d_dirty, sizeof(bool), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        if (dirty) tools::cuda::updateChecksum(dst, CheckMode::Partial, stream);
        handles[n] = nanovdb::GridHandle<BufferT>(std::move(buffer));
        ptr = util::PtrAdd(ptr, handle.gridSize(n));
    }
    cudaCheck(util::cuda::freeAsync(d_dirty, stream));
    return std::move(handles);
}// cuda::splitGridHandles

template<typename BufferT, template <class, class...> class VectorT>
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, GridHandle<BufferT>>::type
mergeGridHandles(const VectorT<GridHandle<BufferT>> &handles, const BufferT* other = nullptr, cudaStream_t stream = 0)
{
    uint64_t size = 0u;
    uint32_t counter = 0u, gridCount = 0u;
    for (auto &h : handles) {
        gridCount += h.gridCount();
        for (uint32_t n=0; n<h.gridCount(); ++n) size += h.gridSize(n);
    }
    int device = 0;
    cudaCheck(cudaGetDevice(&device));
    auto buffer = BufferT::create(size, other, device, stream);
    void *dst = buffer.deviceData();
    bool dirty, *d_dirty;// use this to check if the checksum needs to be recomputed
    cudaCheck(util::cuda::mallocAsync((void**)&d_dirty, sizeof(bool), stream));
    for (auto &h : handles) {
        const void *src = h.deviceData();
        for (uint32_t n=0; n<h.gridCount(); ++n) {
            cudaCheck(cudaMemcpyAsync(dst, src, h.gridSize(n), cudaMemcpyDeviceToDevice, stream));
            GridData *data = reinterpret_cast<GridData*>(dst);
            updateGridCount<<<1, 1, 0, stream>>>(data, counter++, gridCount, d_dirty);
            cudaCheckError();
            cudaCheck(cudaMemcpyAsync(&dirty, d_dirty, sizeof(bool), cudaMemcpyDeviceToHost, stream));
            cudaCheck(cudaStreamSynchronize(stream));
            if (dirty) tools::cuda::updateChecksum(data, CheckMode::Partial, stream);
            dst = util::PtrAdd(dst, h.gridSize(n));
            src = util::PtrAdd(src, h.gridSize(n));
        }
    }
    cudaCheck(util::cuda::freeAsync(d_dirty, stream));
    return GridHandle<BufferT>(std::move(buffer));
}// cuda::mergeGridHandles

}// namespace cuda

template<typename BufferT, template <class, class...> class VectorT = std::vector>
[[deprecated("Use nanovdb::cuda::splitGridHandles instead")]]
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, VectorT<GridHandle<BufferT>>>::type
splitDeviceGrids(const GridHandle<BufferT> &handle, const BufferT* other = nullptr, cudaStream_t stream = 0)
{ return cuda::splitGridHandles(handle, other, stream); }

template<typename BufferT, template <class, class...> class VectorT>
[[deprecated("Use nanovdb::cuda::mergeGridHandles instead")]]
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, GridHandle<BufferT>>::type
mergeDeviceGrids(const VectorT<GridHandle<BufferT>> &handles, const BufferT* other = nullptr, cudaStream_t stream = 0)
{ return cuda::mergeGridHandles<BufferT, VectorT>(handles, other, stream); }

template<typename BufferT>
template<typename T, typename util::enable_if<BufferTraits<T>::hasDeviceDual, int>::type>
GridHandle<BufferT>::GridHandle(T&& buffer)
{
    static_assert(util::is_same<T,BufferT>::value, "Expected U==BufferT");
    mBuffer = std::move(buffer);
    if (auto *data = reinterpret_cast<const GridData*>(mBuffer.data())) {
        if (!data->isValid()) throw std::runtime_error("GridHandle was constructed with an invalid host buffer");
        mMetaData.resize(data->mGridCount);
        cpyGridHandleMeta(data, mMetaData.data());
    } else {
        if (auto *d_data = reinterpret_cast<const GridData*>(mBuffer.deviceData())) {
            GridData tmp;
            cudaCheck(cudaMemcpy(&tmp, d_data, sizeof(GridData), cudaMemcpyDeviceToHost));
            if (!tmp.isValid()) throw std::runtime_error("GridHandle was constructed with an invalid device buffer");
            GridHandleMetaData *d_metaData;
            cudaMalloc((void**)&d_metaData, tmp.mGridCount*sizeof(GridHandleMetaData));
            cuda::cpyGridHandleMeta<<<1,1>>>(d_data, d_metaData);
            mMetaData.resize(tmp.mGridCount);
            cudaCheck(cudaMemcpy(mMetaData.data(), d_metaData,tmp.mGridCount*sizeof(GridHandleMetaData), cudaMemcpyDeviceToHost));
            cudaCheck(cudaFree(d_metaData));
        }
    }
}// GridHandle(T&& buffer)

// Dummy function that ensures instantiation of the move-constructor above when BufferT=cuda::DeviceBuffer
namespace {auto __dummy(){return GridHandle<cuda::DeviceBuffer>(std::move(cuda::DeviceBuffer()));}}

} // namespace nanovdb

#endif // NANOVDB_CUDA_GRIDHANDLE_CUH_HAS_BEEN_INCLUDED
