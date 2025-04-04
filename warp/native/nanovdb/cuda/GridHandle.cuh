// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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
    if ((*d_dirty = (d_data->mGridIndex != gridIndex || d_data->mGridCount != gridCount))) {
        d_data->mGridIndex = gridIndex;
        d_data->mGridCount = gridCount;
        if (d_data->mChecksum.isEmpty()) *d_dirty = false;// no need to update checksum if it didn't already exist
    }
}
}// anonymous namespace

}// namespace cuda

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
