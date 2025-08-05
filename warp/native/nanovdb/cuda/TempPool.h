// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NANOVDB_CUDA_TEMPPOOL_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_TEMPPOOL_H_HAS_BEEN_INCLUDED

#include <nanovdb/cuda/DeviceResource.h>

namespace nanovdb {

namespace cuda {

template <class Resource>
class TempPool {
public:
    TempPool() : mData(nullptr), mSize(0), mRequestedSize(0) {}
    ~TempPool() {
        mRequestedSize = 0;
        Resource::deallocateAsync(mData, mSize, Resource::DEFAULT_ALIGNMENT, nullptr);
        mData = nullptr;
        mSize = 0;
    }

    void* data() {
        return mData;
    }

    size_t& size() {
        return mSize;
    }

    size_t& requestedSize() {
        return mRequestedSize;
    }

    void reallocate(cudaStream_t stream) {
        if (!mData || mRequestedSize > mSize) {
            Resource::deallocateAsync(mData, mSize, Resource::DEFAULT_ALIGNMENT, stream);
            mData = Resource::allocateAsync(mRequestedSize, Resource::DEFAULT_ALIGNMENT, stream);
            mSize = mRequestedSize;
        }
    }
private:
    void* mData;
    size_t mSize;
    size_t mRequestedSize;
};

using TempDevicePool = TempPool<DeviceResource>;

}

} // namespace nanovdb::cuda

#endif // end of NANOVDB_CUDA_TEMPPOOL_H_HAS_BEEN_INCLUDED
