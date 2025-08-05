// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NANOVDB_CUDA_DEVICERESOURCE_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_DEVICERESOURCE_H_HAS_BEEN_INCLUDED

#include <cuda_runtime_api.h>
#include <nanovdb/util/cuda/Util.h>

namespace nanovdb {

namespace cuda {

class DeviceResource
{
public:
    // cudaMalloc aligns memory to 256 bytes by default
    static constexpr size_t DEFAULT_ALIGNMENT = 256;

    static void* allocateAsync(size_t bytes, size_t, cudaStream_t stream) {
        void* p = nullptr;
        cudaCheck(util::cuda::mallocAsync(&p, bytes, stream));
        return p;
    }

    static void deallocateAsync(void *p, size_t, size_t, cudaStream_t stream) {
        cudaCheck(util::cuda::freeAsync(p, stream));
    }
};

}

} // namespace nanovdb::cuda

#endif // end of NANOVDB_CUDA_DEVICERESOURCE_H_HAS_BEEN_INCLUDED
