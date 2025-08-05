// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/GridChecksum.cuh

    \author Ken Museth

    \date September 28, 2023

    \brief Compute CRC32 checksum of NanoVDB grids

    \note before v32.6.0: checksum[0] = Grid+Tree+Root, checksum[1] = nodes
          after  v32.6.0: checksum[0] = Grid+Tree,      checksum[1] = nodes + blind data in 4K blocks

    When serialized:
                                [Grid,Tree][Root][ROOT TILES...][Node<5>...][Node<4>...][Leaf<3>...][BlindMeta...][BlindData...]
    checksum[2] before v32.6.0: <------------- [0] ------------><-------------- [1] --------------->
    checksum[]2 after  v32.6.0: <---[0]---><----------------------------------------[1]---------------------------------------->
*/

#ifndef NANOVDB_TOOLS_CUDA_GRIDCHECKSUM_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_GRIDCHECKSUM_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/util/cuda/Util.h>
#include <nanovdb/cuda/DeviceBuffer.h>// required for instantiation of move c-tor of GridHandle
#include <nanovdb/cuda/NodeManager.cuh>
#include <nanovdb/tools/GridChecksum.h>
#include <nanovdb/GridHandle.h>

namespace nanovdb {// =======================================================================

namespace tools::cuda {// ===================================================================

/// @brief Compute the (2 x CRC32) checksum of the specified @c d_gridData on the device
/// @param d_gridData Device base pointer to the grid from which the checksum is computed.
/// @param mode Defines the mode of computation for the checksum.
/// @param stream optional cuda stream (defaults to zero)
/// @return Return the (2 x CRC32) checksum of the specified @c d_gridData
Checksum evalChecksum(const GridData *d_gridData, CheckMode mode = CheckMode::Default, cudaStream_t stream = 0);

/// @brief Extract the checksum of a device grid
/// @param d_gridData Device basepointer to grid with a checksum
/// @param stream optional cuda stream (defaults to zero)
/// @return Checksum encoded in the specified grid
Checksum getChecksum(const GridData *d_gridData, cudaStream_t stream = 0);

/// @brief Return true if the checksum of @c d_gridData matches the expected
///        value already encoded into the grid's meta data.
/// @tparam BuildT Template parameter used to build NanoVDB grid.
/// @param d_gridData Grid whose checksum is validated.
/// @param mode Defines the mode of computation for the checksum.
/// @param stream optional cuda stream (defaults to zero)
bool validateChecksum(const GridData *d_gridData, CheckMode mode = CheckMode::Default, cudaStream_t stream = 0);

/// @brief Update the checksum of a device grid
/// @param d_gridData device pointer to GridData
/// @param mode Mode of computation for the checksum.
/// @param stream optional cuda stream (defaults to zero)
void updateChecksum(GridData *d_gridData, CheckMode mode, cudaStream_t stream = 0);

/// @brief  Updates the checksum of a device grid by preserving its mode
/// @param d_gridData Device base pointer to grid
/// @param stream optional cuda stream (defaults to zero)
inline void updateChecksum(GridData *d_gridData, cudaStream_t stream = 0)
{
    updateChecksum(d_gridData, getChecksum(d_gridData, stream).mode(), stream);
}

}// namespace tools::cuda

namespace util::cuda {

/// @brief Cuda kernel that computes CRC32 checksums of blocks of data using a look-up-table
/// @param d_data device pointer to raw data from which to compute the CRC32 checksums
/// @param d_blockCRC device pointer to array of @c blockCount checksums for each block
/// @param blockCount number of blocks and checksums
/// @param blockSize size of each block in bytes
/// @param d_lut device pointer to CRC32 Lookup Table
template <typename T>
__global__ void crc32Kernel(const T *d_data, uint32_t* d_blockCRC, uint32_t blockCount, uint32_t blockSize, const uint32_t *d_lut)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < blockCount) d_blockCRC[tid] = crc32((const uint8_t*)d_data + tid * blockSize, blockSize, d_lut);
}

/// @brief Cuda kernel that computes CRC32 checksums of blocks of data (without using a look-up-table)
/// @param d_data device pointer to raw data from which to compute the CRC32 checksums
/// @param d_blockCRC device pointer to array of @c blockCount checksums for each block
/// @param blockCount number of blocks and checksums
/// @param blockSize size of each block in bytes
template <typename T>
__global__ void crc32Kernel(const T *d_data, uint32_t* d_blockCRC, uint32_t blockCount, uint32_t blockSize)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < blockCount) d_blockCRC[tid] = crc32((const uint8_t*)d_data + tid * blockSize, blockSize);
}

/// @brief Host function to allocate and initiate a Look-Up-Table of size 256 for subsequent CRC32 computation on the device
/// @param extra number of extra elements in the LUT
/// @param stream optional cuda stream (defaults to zero)
/// @return returns a nanovdb::util::cuda::unique_ptr point to a lookup-table for CRC32 computation
inline unique_ptr<uint32_t> createCrc32Lut(size_t extra = 0, cudaStream_t stream = 0)
{
    unique_ptr<uint32_t> lut(256 + extra, stream);
    uint32_t *d_lut = lut.get();
    lambdaKernel<<<1, 256, 0, stream>>>(256, [=] __device__(size_t tid) {initCrc32Lut(d_lut, tid);});
    cudaCheckError();
    return lut;
}

/// @brief Compute CRC32 checksum of 4K block
/// @param d_data device pointer to start of data
/// @param size number of bytes
/// @param d_lut Look-Up-Table for CRC32 computation
/// @param stream optional cuda stream (defaults to zero)
inline void blockedCRC32(const void *d_data, size_t size, const uint32_t *d_lut, uint32_t *d_crc, cudaStream_t stream)
{
    NANOVDB_ASSERT(d_data && d_lut && d_crc);
    static constexpr unsigned int threadsPerBlock = 128;// seems faster than the old value of 256!
    const uint64_t checksumCount = size >> NANOVDB_CRC32_LOG2_BLOCK_SIZE;// 4 KB (4096 byte)
    unique_ptr<uint32_t> buffer(checksumCount, stream);// for checksums of 4 KB blocks
    uint32_t *d_checksums = buffer.get();
    lambdaKernel<<<blocksPerGrid(checksumCount, threadsPerBlock), threadsPerBlock, 0, stream>>>(checksumCount, [=] __device__(size_t tid) {
        uint32_t blockSize = 1 << NANOVDB_CRC32_LOG2_BLOCK_SIZE;
        if (tid+1 == checksumCount) blockSize += size - (checksumCount<<NANOVDB_CRC32_LOG2_BLOCK_SIZE);
        d_checksums[tid] = crc32((const uint8_t*)d_data + (tid<<NANOVDB_CRC32_LOG2_BLOCK_SIZE), blockSize, d_lut);
    }); cudaCheckError();
    lambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {// Compute CRC32 of all the 4K blocks
        *d_crc = crc32((const uint8_t*)d_checksums, checksumCount*sizeof(uint32_t), d_lut);
    }); cudaCheckError();
}// void cudaBlockedCRC32(const void *d_data, size_t size, const uint32_t *d_lut, uint32_t *d_crc, cudaStream_t stream)

/// @brief Compute CRC32 checksum of 4K block
/// @param d_begin device pointer to start of data (inclusive)
/// @param d_end device pointer to end of data (exclusive)
/// @param d_lut pointer to Look-Up-Table for accelerated CRC32 computation
/// @param stream optional cuda stream (defaults to zero)
inline void blockedCRC32(const void *d_begin, const void *d_end, const uint32_t *d_lut, uint32_t *d_crc, cudaStream_t stream)
{
    blockedCRC32(d_begin, PtrDiff(d_end, d_begin), d_lut, d_crc, stream);
}

}// namespace util::cuda

namespace tools::cuda {

/// @brief
/// @param d_gridData
/// @param d_lut pointer to Look-Up-Table for accelerated CRC32 computation
/// @param d_crc
/// @param stream optional cuda stream (defaults to zero)
inline void crc32Head(const GridData *d_gridData, const uint32_t *d_lut, uint32_t *d_crc, cudaStream_t stream)
{
    NANOVDB_ASSERT(d_gridData && d_lut && d_crc);
    util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t){*d_crc = tools::crc32Head(d_gridData, d_lut);});
}// void cudaCrc32Head(const GridData *d_gridData, const uint32_t *d_lut, uint32_t *d_crc, cudaStream_t stream)

/// @brief
/// @param d_gridData
/// @param gridData
/// @param d_lut pointer to Look-Up-Table for accelerated CRC32 computation
/// @param stream optional cuda stream (defaults to zero)
inline void crc32Tail(const GridData *d_gridData, const GridData *gridData, const uint32_t *d_lut, uint32_t *d_crc, cudaStream_t stream)
{
    NANOVDB_ASSERT(d_gridData && gridData && d_lut && d_crc);
    NANOVDB_ASSERT(gridData->mVersion > Version(32,6,0));
    const uint8_t *d_begin = (const uint8_t*)d_gridData;
    util::cuda::blockedCRC32(d_begin + sizeof(GridData) + sizeof(TreeData), d_begin + gridData->mGridSize, d_lut, d_crc, stream);
}

/// @brief
/// @tparam ValueT
/// @param d_grid
/// @param gridData
/// @param d_lut pointer to Look-Up-Table for accelerated CRC32 computation
/// @param d_crc
/// @param stream
template <typename ValueT>
void crc32TailOld(const NanoGrid<ValueT> *d_grid, const GridData *gridData, const uint32_t *d_lut, uint32_t *d_crc, cudaStream_t stream)
{
    static constexpr unsigned int threadsPerBlock = 128;// seems faster than the old value of 256!
    auto nodeMgrHandle = nanovdb::cuda::createNodeManager<ValueT, nanovdb::cuda::DeviceBuffer>(d_grid, nanovdb::cuda::DeviceBuffer(), stream);
    auto *d_nodeMgr = nodeMgrHandle.template deviceMgr<ValueT>();
    NANOVDB_ASSERT(isAligned(d_nodeMgr));
    const uint32_t nodeCount[3]={gridData->template nodeCount<0>(), gridData->template nodeCount<1>(), gridData->template nodeCount<2>()};
    util::cuda::unique_ptr<uint32_t> d_checksumsUP(nodeCount[0]+nodeCount[1]+nodeCount[2]);
    uint32_t *d_checksums = d_checksumsUP.get(), *d_ptr = d_checksums;

    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(nodeCount[2], threadsPerBlock), threadsPerBlock, 0, stream>>>(nodeCount[2], [=] __device__(size_t tid) {
        auto &node = d_nodeMgr->upper(uint32_t(tid));
        d_ptr[tid] = util::crc32((const uint8_t*)&node, node.memUsage(), d_lut);
    }); cudaCheckError();

    d_ptr += nodeCount[2];
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(nodeCount[1], threadsPerBlock), threadsPerBlock, 0, stream>>>(nodeCount[1], [=] __device__(size_t tid) {
        auto &node = d_nodeMgr->lower(uint32_t(tid));
        d_ptr[tid] = util::crc32((const uint8_t*)&node, node.memUsage(), d_lut);
    }); cudaCheckError();

    d_ptr += nodeCount[1];
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(nodeCount[0], threadsPerBlock), threadsPerBlock, 0, stream>>>(nodeCount[0], [=] __device__(size_t tid) {
        auto &node = d_nodeMgr->leaf(uint32_t(tid));
        d_ptr[tid] = util::crc32((const uint8_t*)&node, node.memUsage(), d_lut);
    }); cudaCheckError();

    util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
        *d_crc = util::crc32(d_checksums, d_nodeMgr->tree().totalNodeCount()*sizeof(uint32_t), d_lut);
    }); cudaCheckError();
}// void cudaCrc32TailOld(const NanoGrid<ValueT> *d_grid, const GridData *gridData, uint32_t *d_lut, cudaStream_t stream)

struct Crc32TailOld {
    template <typename BuildT>
    static void known(const GridData *d_gridData, const GridData *gridData, const uint32_t *d_lut, uint32_t *d_crc, cudaStream_t stream)
    {
        crc32TailOld((const NanoGrid<BuildT>*)d_gridData, gridData, d_lut, d_crc, stream);
    }
    static void unknown(const GridData*, const GridData*, const uint32_t*, uint32_t*, cudaStream_t)
    {
        throw std::runtime_error("Cannot call cudaCrc32TailOld with grid of unknown type");
    }
};// Crc32TailOld

/// @brief
/// @param d_gridData
/// @param mode
/// @param stream
/// @return
inline Checksum evalChecksum(const GridData *d_gridData, CheckMode mode, cudaStream_t stream)
{
    static const int headSize = sizeof(GridData) + sizeof(TreeData);
    NANOVDB_ASSERT(d_gridData);
    Checksum cs;
    if (mode != CheckMode::Empty) {
        auto d_lut = util::cuda::createCrc32Lut(1, stream);
        crc32Head(d_gridData, d_lut.get(), d_lut.get() + 256, stream);
        cudaCheck(cudaMemcpyAsync(&(cs.head()), d_lut.get() + 256, headSize, cudaMemcpyDeviceToHost, stream));
        if (mode == CheckMode::Full) {
            std::unique_ptr<char[]> buffer(new char[headSize]);
            auto *gridData = (GridData*)(buffer.get());
            cudaCheck(cudaMemcpyAsync(gridData, d_gridData, headSize, cudaMemcpyDeviceToHost, stream));
            if (gridData->mVersion > Version(32,6,0)) {
                crc32Tail(d_gridData, gridData, d_lut.get(), d_lut.get() + 256, stream);
            } else {
                callNanoGrid<Crc32TailOld>(d_gridData, gridData, d_lut.get(), d_lut.get() + 256, stream);
            }
            cudaCheck(cudaMemcpyAsync(&(cs.tail()), d_lut.get() + 256, headSize, cudaMemcpyDeviceToHost, stream));
        }
    }
    return cs;
}

/// @brief
/// @tparam BuildT
/// @param d_grid
/// @param mode
/// @param stream
/// @return
template <typename BuildT>
Checksum evalChecksum(const NanoGrid<BuildT> *d_grid, CheckMode mode, cudaStream_t stream = 0)
{
    static const int headSize = sizeof(GridData) + sizeof(TreeData);
    NANOVDB_ASSERT(d_grid);
    Checksum cs;
    if (mode != CheckMode::Empty) {
        auto d_lut = util::cuda::createCrc32Lut(1, stream);
        crc32Head(d_grid, d_lut.get(), d_lut.get() + 256, stream);
        cudaCheck(cudaMemcpyAsync(&(cs.head()), d_lut.get() + 256, headSize, cudaMemcpyDeviceToHost, stream));
        if (mode == CheckMode::Full) {
            std::unique_ptr<char[]> buffer(new char[headSize]);
            auto *gridData = (GridData*)(buffer.get());
            cudaCheck(cudaMemcpyAsync(gridData, d_grid, headSize, cudaMemcpyDeviceToHost, stream));
            if (gridData->mVersion > Version(32,6,0)) {
                crc32Tail(d_grid, gridData, d_lut.get(), d_lut.get() + 256, stream);
            } else {
                crc32TailOld(d_grid, gridData, d_lut.get(), d_lut.get() + 256, stream);
            }
            cudaCheck(cudaMemcpyAsync(&(cs.tail()), d_lut.get() + 256, headSize, cudaMemcpyDeviceToHost, stream));
        }
    }
    return cs;
}

/// @brief
/// @param d_gridData
/// @param mode
/// @param stream
/// @return
inline bool validateChecksum(const GridData *d_gridData, CheckMode mode, cudaStream_t stream)
{
    static const int headSize = sizeof(GridData) + sizeof(TreeData);
    NANOVDB_ASSERT(d_gridData);
    if (mode == CheckMode::Empty) return true;

    // Copy just the GridData from the device to the host
    std::unique_ptr<char[]> buffer(new char[headSize]);
    auto *gridData = (GridData*)(buffer.get());
    cudaCheck(cudaMemcpyAsync(gridData, d_gridData, headSize, cudaMemcpyDeviceToHost, stream));
    if (gridData->mChecksum.isEmpty()) return true;// checksum is empty so nothing to check

    // Allocate device LUT for CRC32 computation
    auto d_lut = util::cuda::createCrc32Lut(1, stream);// unique pointer
    uint32_t crc = 0, *d_crc = d_lut.get() + 256;

    // Check head checksum
    crc32Head(d_gridData, d_lut.get(), d_crc, stream);
    cudaCheck(cudaMemcpyAsync(&crc, d_crc, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    const bool checkHead = (crc == gridData->mChecksum.head());
    if (gridData->mChecksum.isHalf() || mode == CheckMode::Half || !checkHead) return checkHead;

    // Check tail checksum
    if (gridData->mVersion > Version(32,6,0)) {
        crc32Tail(d_gridData, gridData, d_lut.get(), d_crc, stream);
    } else {
        callNanoGrid<Crc32TailOld>(d_gridData, gridData, d_lut.get(), d_crc, stream);
    }
    cudaCheck(cudaMemcpyAsync(&crc, d_crc, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    return crc == gridData->mChecksum.tail();
}// bool cudaValidateChecksum(const GridData *d_gridData, CheckMode mode, cudaStream_t stream = 0)

/// @brief
/// @tparam BuildT
/// @param d_grid
/// @param mode
/// @param stream
/// @return
template <typename BuildT>
bool validateChecksum(const NanoGrid<BuildT> *d_grid, CheckMode mode, cudaStream_t stream = 0)
{
    static const int headSize = sizeof(GridData) + sizeof(TreeData);
    NANOVDB_ASSERT(d_grid);
    if (mode == CheckMode::Empty) return true;

    // Copy just the GridData from the device to the host
    std::unique_ptr<char[]> buffer(new char[headSize]);
    auto *gridData = (GridData*)(buffer.get());
    cudaCheck(cudaMemcpyAsync(gridData, d_grid, headSize, cudaMemcpyDeviceToHost, stream));
    if (gridData->mChecksum.isEmpty()) return true;// checksum is empty so nothing to check

    // Allocate device LUT for CRC32 computation
    auto d_lut = util::cuda::createCrc32Lut(1, stream);// unique pointer
    uint32_t crc = 0, *d_crc = d_lut.get() + 256;

    // Check head checksum
    crc32Head(d_grid, d_lut.get(), d_crc, stream);
    cudaCheck(cudaMemcpyAsync(&crc, d_crc, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    const bool checkHead = (crc == gridData->mChecksum.head());
    if (gridData->mChecksum.isHalf() || mode == CheckMode::Half || !checkHead) return checkHead;

    // Check tail checksum
    if (gridData->mVersion > Version(32,6,0)) {
        crc32Tail(d_grid, gridData, d_lut.get(), d_crc, stream);
    } else {
        crc32TailOld(d_grid, gridData, d_lut.get(), d_crc, stream);
    }
    cudaCheck(cudaMemcpyAsync(&crc, d_crc, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    return crc == gridData->mChecksum.tail();
}// bool cudaValidateChecksum(const GridData *d_gridData, CheckMode mode, cudaStream_t stream = 0)

/// @brief Extract the checksum of a device grid
/// @param d_gridData Device pointer to grid with a checksum
/// @param stream optional cuda stream (defaults to zero)
inline Checksum getChecksum(const GridData *d_gridData, cudaStream_t stream)
{
    NANOVDB_ASSERT(d_gridData);
    Checksum cs;
    cudaCheck(cudaMemcpyAsync(&cs, (const uint8_t*)d_gridData + 8, sizeof(cs), cudaMemcpyDeviceToHost, stream));
    return cs;
}

/// @brief Update the checksum of a device grid
/// @param d_gridData device pointer to GridData
/// @param mode Mode of computation for the checksum.
/// @param stream optional cuda stream (defaults to zero)
/// @return The actual mode used for checksum computation. Eg. if @c d_gridData is NULL (or @c mode = CheckMode::Empty)
///         then CheckMode::Empty is always returned. Else if the grid has no nodes or blind data CheckMode::Partial
///         is always returned (even if @c mode = CheckMode::Full).
inline void updateChecksum(GridData *d_gridData, CheckMode mode, cudaStream_t stream)
{
    NANOVDB_ASSERT(d_gridData);
    if (mode == CheckMode::Empty) return;

    // Allocate device LUT for CRC32 computation
    auto d_lut = util::cuda::createCrc32Lut(0, stream);// unique pointers

    // Update head checksum
    crc32Head(d_gridData, d_lut.get(), (uint32_t*)d_gridData + 2, stream);

    if (mode == CheckMode::Half) return;

    // Copy just the GridData from the device to the host
    std::unique_ptr<char[]> buffer(new char[sizeof(GridData) + sizeof(TreeData)]);
    auto *gridData = (GridData*)(buffer.get());
    cudaCheck(cudaMemcpyAsync(gridData, d_gridData, sizeof(GridData) + sizeof(TreeData), cudaMemcpyDeviceToHost, stream));

    // Update tail checksum
    uint32_t *d_tail = (uint32_t*)d_gridData + 3;
    if (gridData->mVersion > Version(32,6,0)) {
        crc32Tail(d_gridData, gridData, d_lut.get(), d_tail, stream);
    } else {
        callNanoGrid<Crc32TailOld>(d_gridData, gridData, d_lut.get(), d_tail, stream);
    }
}// cudaUpdateChecksum

/// @brief
/// @tparam ValueT
/// @param d_grid
/// @param mode
/// @param stream
template <typename ValueT>
void updateChecksum(NanoGrid<ValueT> *d_grid, CheckMode mode, cudaStream_t stream = 0)
{
    NANOVDB_ASSERT(d_grid);
    if (mode == CheckMode::Empty) return;

    // Allocate device LUT for CRC32 computation
    auto d_lut = util::cuda::createCrc32Lut(0, stream);// unique pointers

    // Update head checksum
    cuda::crc32Head(d_grid, d_lut.get(), (uint32_t*)d_grid + 2, stream);
    if (mode == CheckMode::Half) return;

    // Copy just the GridData from the device to the host
    std::unique_ptr<char[]> buffer(new char[sizeof(GridData) + sizeof(TreeData)]);
    auto *gridData = (GridData*)(buffer.get());
    cudaCheck(cudaMemcpyAsync(gridData, d_grid, sizeof(GridData) + sizeof(TreeData), cudaMemcpyDeviceToHost, stream));

    // Update tail checksum
    uint32_t *d_tail = (uint32_t*)d_grid + 3;
    if (gridData->mVersion > Version(32,6,0)) {
        crc32Tail(d_grid->data(), gridData, d_lut.get(), d_tail, stream);
    } else {
        crc32TailOld(d_grid, gridData, d_lut.get(), d_tail, stream);
    }
}

}// namespace tools::cuda // ================================================

}// namespace nanovdb // ====================================================

#endif // NANOVDB_TOOLS_CUDA_GRIDCHECKSUM_CUH_HAS_BEEN_INCLUDED
