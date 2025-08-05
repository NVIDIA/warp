// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/GridChecksum.h

    \author Ken Museth

    \brief Computes a pair of uint32_t checksums, of a Grid, by means of 32 bit Cyclic Redundancy Check (CRC32)

    \details A CRC32 is the 32 bit remainder, or residue, of binary division of a message, by a polynomial.


    \note before v32.6.0: checksum[0] = Grid+Tree+Root, checksum[1] = nodes
          after  v32.6.0: checksum[0] = Grid+Tree,      checksum[1] = nodes + blind data in 4K blocks

    When serialized:
                                [Grid,Tree][Root][ROOT TILES...][Node<5>...][Node<4>...][Leaf<3>...][BlindMeta...][BlindData...]
    checksum[2] before v32.6.0: <------------- [0] ------------><-------------- [1] --------------->
    checksum[2] after  v32.6.0: <---[0]---><----------------------------------------[1]---------------------------------------->
*/

#ifndef NANOVDB_TOOLS_GRIDCHECKSUM_H_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_GRIDCHECKSUM_H_HAS_BEEN_INCLUDED

#include <algorithm>// for std::generate
#include <array>
#include <vector>
#include <cstdint>
#include <cstddef>// offsetof macro
#include <numeric>
#include <type_traits>
#include <memory>// for std::unique_ptr

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/ForEach.h>
#include <nanovdb/NodeManager.h>

// Define log of block size for FULL CRC32 computation.
// A value of 12 corresponds to a block size of 4KB (2^12 = 4096).
#define NANOVDB_CRC32_LOG2_BLOCK_SIZE 12

namespace nanovdb {// ==================================================================

namespace tools {// ====================================================================

/// @brief Compute the (2 x CRC32) checksum of the specified @c gridData
/// @param gridData  Base pointer to the grid from which the checksum is computed.
/// @param mode Defines the mode of computation for the checksum.
/// @return Return the (2 x CRC32) checksum of the specified @c gridData
Checksum evalChecksum(const GridData *gridData, CheckMode mode = CheckMode::Default);

/// @brief Extract the checksum of a grid
/// @param gridData Base pointer to grid with a checksum
/// @return Checksum encoded in the specified grid
inline Checksum getChecksum(const GridData *gridData)
{
    NANOVDB_ASSERT(gridData);
    return gridData->mChecksum;
}

/// @brief Return true if the checksum of @c gridData matches the expected
///        value already encoded into the grid's meta data.
/// @tparam BuildT Template parameter used to build NanoVDB grid.
/// @param grid Grid whose checksum is validated.
/// @param mode Defines the mode of computation for the checksum.
bool validateChecksum(const GridData *gridData, CheckMode mode = CheckMode::Default);

/// @brief Updates the checksum of a grid
/// @param grid Grid whose checksum will be updated.
/// @param mode Defines the mode of computation for the checksum.
inline void updateChecksum(GridData *gridData, CheckMode mode)
{
    NANOVDB_ASSERT(gridData);
    gridData->mChecksum = evalChecksum(gridData, mode);
}

/// @brief Updates the checksum of a grid by preserving its mode
/// @param gridData Base pointer to grid
inline void updateChecksum(GridData *gridData)
{
    updateChecksum(gridData, gridData->mChecksum.mode());
}

}// namespace tools

namespace util {

/// @brief Initiate single entry in look-up-table for CRC32 computations
/// @param lut pointer of size 256 for look-up-table
/// @param n entry in table (assumed n < 256)
inline __hostdev__ void initCrc32Lut(uint32_t lut[256], uint32_t n)
{
    lut[n] = n;
    uint32_t &cs = lut[n];
    for (int i = 0; i < 8; ++i) cs = (cs >> 1) ^ ((cs & 1) ? 0xEDB88320 : 0);
}

/// @brief Initiate entire look-up-table for CRC32 computations
/// @param lut pointer of size 256 for look-up-table
inline __hostdev__ void initCrc32Lut(uint32_t lut[256]){for (uint32_t n = 0u; n < 256u; ++n) initCrc32Lut(lut, n);}

/// @brief Create and initiate entire look-up-table for CRC32 computations
/// @return returns a unique pointer to the lookup table of size 256.
inline std::unique_ptr<uint32_t[]> createCrc32Lut()
{
    std::unique_ptr<uint32_t[]> lut(new uint32_t[256]);
    initCrc32Lut(lut.get());
    return lut;
}

/// @brief Compute crc32 checksum of @c data of @c size bytes (without a lookup table))
/// @param data pointer to beginning of data
/// @param size byte size of data
/// @param crc initial value of crc32 checksum
/// @return return crc32 checksum of @c data
inline __hostdev__ uint32_t crc32(const void* data, size_t size, uint32_t crc = 0)
{
    NANOVDB_ASSERT(data);
    crc = ~crc;
    for (auto *p = (const uint8_t*)data, *q = p + size; p != q; ++p) {
        crc ^= *p;
        for (int j = 0; j < 8; ++j) crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
    }
    return ~crc;
}

/// @brief Compute crc32 checksum of data between @c begin and @c end
/// @param begin points to beginning of data
/// @param end points to end of @data, (exclusive)
/// @param crc initial value of crc32 checksum
/// @return return crc32 checksum
inline __hostdev__ uint32_t crc32(const void *begin, const void *end, uint32_t crc = 0)
{
    NANOVDB_ASSERT(begin && end);
    NANOVDB_ASSERT(end >= begin);
    return crc32(begin, (const char*)end - (const char*)begin, crc);
}

/// @brief Compute crc32 checksum of @c data with @c size bytes using a lookup table
/// @param data pointer to begenning of data
/// @param size byte size
/// @param lut pointer to loopup table for accelerated crc32 computation
/// @param crc initial value of the checksum
/// @return crc32 checksum of @c data with @c size bytes
inline __hostdev__ uint32_t crc32(const void *data, size_t size, const uint32_t lut[256], uint32_t crc = 0)
{
    NANOVDB_ASSERT(data);
    crc = ~crc;
    for (auto *p = (const uint8_t*)data, *q = p + size; p != q; ++p) crc = lut[(crc ^ *p) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

/// @brief Compute crc32 checksum of data between @c begin and @c end using a lookup table
/// @param begin points to beginning of data
/// @param end points to end of @data, (exclusive)
/// @param lut pointer to loopup table for accelerated crc32 computation
/// @param crc initial value of crc32 checksum
/// @return return crc32 checksum
inline __hostdev__ uint32_t crc32(const void *begin, const void *end, const uint32_t lut[256], uint32_t crc = 0)
{
    NANOVDB_ASSERT(begin && end);
    NANOVDB_ASSERT(end >= begin);
    return crc32(begin, (const char*)end - (const char*)begin, lut, crc);
}// uint32_t util::crc32(const void *begin, const void *end, const uint32_t lut[256], uint32_t crc = 0)

/// @brief
/// @param data
/// @param size
/// @param lut
/// @return
inline uint32_t blockedCrc32(const void *data, size_t size, const uint32_t *lut)
{
    if (size == 0 ) return ~uint32_t(0);
    const uint64_t blockCount = size >> NANOVDB_CRC32_LOG2_BLOCK_SIZE;// number of 4 KB (4096 byte) blocks
    std::unique_ptr<uint32_t[]> checksums(new uint32_t[blockCount]);
    forEach(0, blockCount, 64, [&](const Range1D &r) {
        uint32_t blockSize = 1 << NANOVDB_CRC32_LOG2_BLOCK_SIZE, *p = checksums.get() + r.begin();
        for (auto i = r.begin(); i != r.end(); ++i) {
            if (i+1 == blockCount) blockSize += static_cast<uint32_t>(size - (blockCount<<NANOVDB_CRC32_LOG2_BLOCK_SIZE));
            *p++ = crc32((const uint8_t*)data + (i<<NANOVDB_CRC32_LOG2_BLOCK_SIZE), blockSize, lut);
        }
    });
    return crc32(checksums.get(), sizeof(uint32_t)*blockCount, lut);
}// uint32_t util::blockedCrc32(const void *data, size_t size, const uint32_t *lut)

/// @brief
/// @param begin
/// @param end
/// @param lut
/// @return
inline uint32_t blockedCrc32(const void *begin, const void *end, const uint32_t *lut)
{
    return blockedCrc32(begin, PtrDiff(end, begin), lut);
}

}// namespace util =======================================================================================

namespace tools {// ======================================================================================

//    When serialized:
//                                [Grid,Tree][Root][ROOT TILES...][Node<5>...][Node<4>...][Leaf<3>...][BlindMeta...][BlindData...]
//    checksum[2] before v32.6.0: <------------- [0] ------------><-------------- [1] --------------->
//    checksum[]2 after  v32.6.0: <---[0]---><----------------------------------------[1]---------------------------------------->

// ----------------------------> crc32Head <--------------------------------------

/// @brief
/// @tparam ValueT
/// @param grid
/// @param mode
/// @return
inline __hostdev__ uint32_t crc32Head(const GridData *gridData, const uint32_t *lut)
{
    NANOVDB_ASSERT(gridData);
    const uint8_t *begin = (const uint8_t*)(gridData), *mid = begin + sizeof(GridData) + sizeof(TreeData);
    if (gridData->mVersion <= Version(32,6,0)) mid = (const uint8_t*)(gridData->template nodePtr<2>());
    return util::crc32(begin + 16u, mid, lut);// exclude GridData::mMagic and GridData::mChecksum
}// uint32_t crc32Head(const GridData *gridData, const uint32_t *lut)

/// @brief
/// @param gridData
/// @return
inline __hostdev__ uint32_t crc32Head(const GridData *gridData)
{
    NANOVDB_ASSERT(gridData);
    const uint8_t *begin = (const uint8_t*)(gridData), *mid = begin + sizeof(GridData) + sizeof(TreeData);
    if (gridData->mVersion <= Version(32,6,0)) mid = (const uint8_t*)(gridData->template nodePtr<2>());
    return util::crc32(begin + 16, mid);// exclude GridData::mMagic and GridData::mChecksum
}// uint32_t crc32Head(const GridData *gridData)

// ----------------------------> crc32TailOld <--------------------------------------

// Old checksum
template <typename ValueT>
uint32_t crc32TailOld(const NanoGrid<ValueT> *grid, const uint32_t *lut)
{
    NANOVDB_ASSERT(grid->mVersion <= Version(32,6,0));
    const auto &tree = grid->tree();
    auto nodeMgrHandle = createNodeManager(*grid);
    auto *nodeMgr = nodeMgrHandle.template mgr<ValueT>();
    assert(nodeMgr && isAligned(nodeMgr));
    const auto nodeCount = tree.nodeCount(0) + tree.nodeCount(1) + tree.nodeCount(2);
    std::vector<uint32_t> checksums(nodeCount, 0);
    util::forEach(0, tree.nodeCount(2), 1,[&](const util::Range1D &r) {// process upper internal nodes
        uint32_t *p = checksums.data() + r.begin();
        for (auto i = r.begin(); i != r.end(); ++i) {
            const auto &node = nodeMgr->upper(static_cast<uint32_t>(i));
            *p++ = util::crc32(&node, node.memUsage(), lut);
        }
    });
    util::forEach(0, tree.nodeCount(1), 1, [&](const util::Range1D &r) { // process lower internal nodes
        uint32_t *p = checksums.data() + r.begin() + tree.nodeCount(2);
        for (auto i = r.begin(); i != r.end(); ++i) {
            const auto &node = nodeMgr->lower(static_cast<uint32_t>(i));
            *p++ = util::crc32(&node, node.memUsage(), lut);
        }
    });
    util::forEach(0, tree.nodeCount(0), 8, [&](const util::Range1D &r) { // process leaf nodes
        uint32_t *p = checksums.data() + r.begin() + tree.nodeCount(1) + tree.nodeCount(2);
        for (auto i = r.begin(); i != r.end(); ++i) {
            const auto &leaf = nodeMgr->leaf(static_cast<uint32_t>(i));
            *p++ = util::crc32(&leaf, leaf.memUsage(), lut);
        }
    });
    return util::crc32(checksums.data(), sizeof(uint32_t)*checksums.size(), lut);
}// uint32_t crc32TailOld(const NanoGrid<ValueT> *grid, const uint32_t *lut)

struct Crc32TailOld {
    template <typename BuildT>
    static uint32_t   known(const GridData *gridData, const uint32_t *lut)
    {
        return crc32TailOld((const NanoGrid<BuildT>*)gridData, lut);
    }
    static uint32_t unknown(const GridData*, const uint32_t*)
    {
        throw std::runtime_error("Cannot call Crc32TailOld with grid of unknown type");
        return 0u;//dummy
    }
};// struct Crc32TailOld

inline uint32_t crc32Tail(const GridData *gridData, const uint32_t *lut)
{
    NANOVDB_ASSERT(gridData);
    if (gridData->mVersion > Version(32,6,0)) {
        const uint8_t *begin = (const uint8_t*)(gridData);
        return util::blockedCrc32(begin + sizeof(GridData) + sizeof(TreeData), begin + gridData->mGridSize, lut);
    } else {
        return callNanoGrid<Crc32TailOld>(gridData, lut);
    }
}// uint32_t crc32Tail(const GridData *gridData, const uint32_t *lut)

template <typename ValueT>
uint32_t crc32Tail(const NanoGrid<ValueT> *grid, const uint32_t *lut)
{
    NANOVDB_ASSERT(grid);
    if (grid->mVersion > Version(32,6,0)) {
        const uint8_t *begin = (const uint8_t*)(grid);
        return util::blockedCrc32(begin + sizeof(GridData) + sizeof(TreeData), begin + grid->mGridSize, lut);
    } else {
        return crc32TailOld(grid, lut);
    }
}// uint32_t crc32Tail(const NanoGrid<ValueT> *gridData, const uint32_t *lut)

// ----------------------------> evalChecksum <--------------------------------------

/// @brief
/// @tparam ValueT
/// @param grid
/// @param mode
/// @return
template <typename ValueT>
Checksum evalChecksum(const NanoGrid<ValueT> *grid, CheckMode mode)
{
    NANOVDB_ASSERT(grid);
    Checksum cs;
    if (mode != CheckMode::Empty) {
        auto lut  = util::createCrc32Lut();
        cs.head() = crc32Head(grid, lut.get());
        if (mode == CheckMode::Full) cs.tail() = crc32Tail(grid, lut.get());
    }
    return cs;
}// checksum(const NanoGrid*, CheckMode)

template <typename ValueT>
[[deprecated("Use evalChecksum(const NanoGrid<ValueT> *grid, CheckMode mode) instead")]]
Checksum checksum(const NanoGrid<ValueT> *grid, CheckMode mode){return evalChecksum(grid, mode);}

inline Checksum evalChecksum(const GridData *gridData, CheckMode mode)
{
    NANOVDB_ASSERT(gridData);
    Checksum cs;
    if (mode != CheckMode::Disable) {
        auto lut  = util::createCrc32Lut();
        cs.head() = crc32Head(gridData, lut.get());
        if (mode == CheckMode::Full) cs.tail() = crc32Tail(gridData, lut.get());
    }
    return cs;
}// evalChecksum(GridData *data, CheckMode mode)

[[deprecated("Use evalChecksum(const NanoGrid*, CheckMode) instead")]]
inline Checksum checksum(const GridData *gridData, CheckMode mode){return evalChecksum(gridData, mode);}

template <typename ValueT>
[[deprecated("Use checksum(const NanoGrid*, CheckMode) instead")]]
Checksum checksum(const NanoGrid<ValueT> &grid, CheckMode mode){return checksum(&grid, mode);}

// ----------------------------> validateChecksum <--------------------------------------

/// @brief
/// @tparam ValueT
/// @param grid
/// @param mode
/// @return
template <typename ValueT>
bool validateChecksum(const NanoGrid<ValueT> *grid, CheckMode mode)
{
    if (grid->mChecksum.isEmpty() || mode == CheckMode::Empty) return true;
    auto lut = util::createCrc32Lut();
    bool checkHead = grid->mChecksum.head() == crc32Head(grid->data(), lut.get());
    if (grid->mChecksum.isHalf() || mode == CheckMode::Half || !checkHead) {
        return checkHead;
    } else {
        return grid->mChecksum.tail() == crc32Tail(grid, lut.get());
    }
}

/// @brief
/// @tparam ValueT
/// @param grid
/// @param mode
/// @return
inline bool validateChecksum(const GridData *gridData, CheckMode mode)
{
    if (gridData->mChecksum.isEmpty()|| mode == CheckMode::Empty) return true;
    auto lut = util::createCrc32Lut();
    bool checkHead = gridData->mChecksum.head() == crc32Head(gridData, lut.get());
    if (gridData->mChecksum.isHalf() || mode == CheckMode::Half || !checkHead) {
        return checkHead;
    } else {
        return gridData->mChecksum.tail() == crc32Tail(gridData, lut.get());
    }
}//  bool validateChecksum(const GridData *gridData, CheckMode mode)

template <typename ValueT>
[[deprecated("Use validateChecksum(const NanoGrid*, CheckMode) instead")]]
bool validateChecksum(const NanoGrid<ValueT> &grid, CheckMode mode){return validateChecksum(&grid, mode);}

// ----------------------------> updateChecksum <--------------------------------------

/// @brief
/// @tparam ValueT
/// @param grid
/// @param mode
template <typename ValueT>
void updateChecksum(NanoGrid<ValueT> *grid, CheckMode mode){grid->mChecksum = evalChecksum(grid, mode);}

template <typename ValueT>
void updateChecksum(NanoGrid<ValueT> *grid){grid->mChecksum = evalChecksum(grid, grid->mChecksum.mode());}

// deprecated method that takes a reference vs a pointer
template <typename ValueT>
[[deprecated("Use updateChecksum(const NanoGrid*, CheckMode) instead")]]
void updateChecksum(NanoGrid<ValueT> &grid, CheckMode mode){updateChecksum(&grid, mode);}

// ----------------------------> updateGridCount <--------------------------------------

/// @brief Updates the ground index and count, as well as the head checksum if needed
/// @param data Pointer to grid data
/// @param gridIndex New value of the index
/// @param gridCount New value of the grid count
inline void updateGridCount(GridData *data, uint32_t gridIndex, uint32_t gridCount)
{
    NANOVDB_ASSERT(data && gridIndex < gridCount);
    if (data->mGridIndex != gridIndex || data->mGridCount != gridCount) {
        data->mGridIndex  = gridIndex;
        data->mGridCount  = gridCount;
        if (!data->mChecksum.isEmpty()) data->mChecksum.head() = crc32Head(data);
    }
}

} // namespace tools ======================================================================


} // namespace nanovdb ====================================================================

#endif // NANOVDB_TOOLS_GRIDCHECKSUM_H_HAS_BEEN_INCLUDED
