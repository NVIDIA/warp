// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/NodeManager.h

    \author Ken Museth

    \date February 12, 2021

    \brief This class allows for sequential access to nodes
           in a NanoVDB tree on both the host and device.

    \details The ordering of the sequential access to nodes is always breadth-first!
*/

#include <nanovdb/NanoVDB.h>// for NanoGrid etc
#include <nanovdb/HostBuffer.h>// for HostBuffer

#ifndef NANOVDB_NODEMANAGER_H_HAS_BEEN_INCLUDED
#define NANOVDB_NODEMANAGER_H_HAS_BEEN_INCLUDED

namespace nanovdb {

/// @brief NodeManager allows for sequential access to nodes
template <typename BuildT>
class NodeManager;

/// @brief NodeManagerHandle manages the memory of a NodeManager
template<typename BufferT = HostBuffer>
class NodeManagerHandle;

/// @brief brief Construct a NodeManager and return its handle
///
/// @param grid grid whose nodes will be accessed sequentially
/// @param buffer buffer from which to allocate the output handle
///
/// @note This is the only way to create a NodeManager since it's using
///       managed memory pointed to by a NodeManagerHandle.
template <typename BuildT, typename BufferT = HostBuffer>
NodeManagerHandle<BufferT> createNodeManager(const NanoGrid<BuildT> &grid,
                                             const BufferT& buffer = BufferT());

struct NodeManagerData
{// 40B = 5*8B
    __hostdev__ NodeManagerData(void *grid) : mPadding{0}, mGrid(grid), mPtr{0,0,0}{}
    union {int64_t  mPadding; uint8_t mLinear;};// 8B of which 1B is used for a binary flag
    void           *mGrid;//  8B pointer to either host or device grid
    union {int64_t *mPtr[3], mOff[3];};// 24B, use mOff if mLinear!=0
};

/// @brief This class serves to manage a raw memory buffer of a NanoVDB NodeManager or LeafManager.
template<typename BufferT>
class NodeManagerHandle
{
    GridType mGridType{GridType::Unknown};
    BufferT  mBuffer;

    template<typename BuildT>
    const NodeManager<BuildT>* getMgr() const {
        return mGridType == toGridType<BuildT>() ? (const NodeManager<BuildT>*)mBuffer.data() : nullptr;
    }

    template<typename BuildT, typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, const NodeManager<BuildT>*>::type
    getDeviceMgr() const {
        return mGridType == toGridType<BuildT>() ? (const NodeManager<BuildT>*)mBuffer.deviceData() : nullptr;
    }

    template <typename T>
    static T* no_const(const T* ptr) { return const_cast<T*>(ptr); }

public:
    /// @brief Move constructor from a buffer
    NodeManagerHandle(GridType gridType, BufferT&& buffer) : mGridType(gridType) { mBuffer = std::move(buffer); }
    /// @brief Empty ctor
    NodeManagerHandle() = default;
    /// @brief Disallow copy-construction
    NodeManagerHandle(const NodeManagerHandle&) = delete;
    /// @brief Disallow copy assignment operation
    NodeManagerHandle& operator=(const NodeManagerHandle&) = delete;
    /// @brief Move copy assignment operation
    NodeManagerHandle& operator=(NodeManagerHandle&& other) noexcept {
        mGridType = other.mGridType;
        mBuffer = std::move(other.mBuffer);
        other.mGridType = GridType::Unknown;
        return *this;
    }
    /// @brief Move copy-constructor
    NodeManagerHandle(NodeManagerHandle&& other) noexcept {
        mGridType = other.mGridType;
        mBuffer = std::move(other.mBuffer);
        other.mGridType = GridType::Unknown;
    }
    /// @brief Default destructor
    ~NodeManagerHandle() { this->reset(); }
    /// @brief clear the buffer
    void reset() { mBuffer.clear(); }

    /// @brief Return a reference to the buffer
    BufferT& buffer() { return mBuffer; }

    /// @brief Return a const reference to the buffer
    const BufferT& buffer() const { return mBuffer; }

    /// @brief Returns a non-const pointer to the data.
    ///
    /// @warning Note that the return pointer can be NULL if the NodeManagerHandle was not initialized
    void* data() { return mBuffer.data(); }

    /// @brief Returns a const pointer to the data.
    ///
    /// @warning Note that the return pointer can be NULL if the NodeManagerHandle was not initialized
    const void* data() const { return mBuffer.data(); }

    /// @brief Returns the size in bytes of the raw memory buffer managed by this NodeManagerHandle's allocator.
    uint64_t size() const { return mBuffer.size(); }

    /// @brief Returns a const pointer to the NodeManager encoded in this NodeManagerHandle.
    ///
    /// @warning Note that the return pointer can be NULL if the template parameter does not match the specified grid!
    template<typename BuildT>
    const NodeManager<BuildT>* mgr() const { return this->template getMgr<BuildT>(); }

    /// @brief Returns a pointer to the NodeManager encoded in this NodeManagerHandle.
    ///
    /// @warning Note that the return pointer can be NULL if the template parameter does not match the specified grid!
    template<typename BuildT>
    NodeManager<BuildT>* mgr() { return no_const(this->template getMgr<BuildT>()); }

    /// @brief Return a const pointer to the NodeManager encoded in this NodeManagerHandle on the device, e.g. GPU
    ///
    /// @warning Note that the return pointer can be NULL if the template parameter does not match the specified grid!
    template<typename BuildT, typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, const NodeManager<BuildT>*>::type
    deviceMgr() const { return this->template getDeviceMgr<BuildT>(); }

    /// @brief Return a const pointer to the NodeManager encoded in this NodeManagerHandle on the device, e.g. GPU
    ///
    /// @warning Note that the return pointer can be NULL if the template parameter does not match the specified grid!
    template<typename BuildT, typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, NodeManager<BuildT>*>::type
    deviceMgr() { return no_const(this->template getDeviceMgr<BuildT>()); }

    /// @brief Upload the NodeManager to the device, e.g. from CPU to GPU
    ///
    /// @note This method is only available if the buffer supports devices
    template<typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, void>::type
    deviceUpload(void* deviceGrid, void* stream = nullptr, bool sync = true)
    {
        assert(deviceGrid);
        auto *data = reinterpret_cast<NodeManagerData*>(mBuffer.data());
        void *tmp = data->mGrid;
        data->mGrid = deviceGrid;
        mBuffer.deviceUpload(stream, sync);
        data->mGrid = tmp;
    }

    /// @brief Download the NodeManager to from the device, e.g. from GPU to CPU
    ///
    /// @note This method is only available if the buffer supports devices
    template<typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, void>::type
    deviceDownload(void* stream = nullptr, bool sync = true)
    {
        auto *data = reinterpret_cast<NodeManagerData*>(mBuffer.data());
        void *tmp = data->mGrid;
        mBuffer.deviceDownload(stream, sync);
        data->mGrid = tmp;
    }
};// NodeManagerHandle

/// @brief This class allows for sequential access to nodes in a NanoVDB tree
///
/// @details Nodes are always arranged breadth first during sequential access of nodes
///          at a particular level.
template<typename BuildT>
class NodeManager : private NodeManagerData
{
    using DataT = NodeManagerData;
    using GridT = NanoGrid<BuildT>;
    using TreeT = typename GridTree<GridT>::type;
    template<int LEVEL>
    using NodeT = typename NodeTrait<TreeT, LEVEL>::type;
    using RootT = NodeT<3>;// root node
    using Node2 = NodeT<2>;// upper internal node
    using Node1 = NodeT<1>;// lower internal node
    using Node0 = NodeT<0>;// leaf node

public:
    static constexpr bool FIXED_SIZE = Node0::FIXED_SIZE && Node1::FIXED_SIZE && Node2::FIXED_SIZE;

    NodeManager(const NodeManager&) = delete;
    NodeManager(NodeManager&&) = delete;
    NodeManager& operator=(const NodeManager&) = delete;
    NodeManager& operator=(NodeManager&&) = delete;
    ~NodeManager() = delete;

    /// @brief return true if the nodes have both fixed size and are arranged breadth-first in memory.
    ///        This allows for direct and memory-efficient linear access to nodes.
    __hostdev__ static bool isLinear(const GridT &grid) {return FIXED_SIZE && grid.isBreadthFirst();}

    /// @brief return true if the nodes have both fixed size and are arranged breadth-first in memory.
    ///        This allows for direct and memory-efficient linear access to nodes.
    __hostdev__ bool isLinear() const {return DataT::mLinear!=0u;}

    /// @brief Return the memory footprint in bytes of the NodeManager derived from the specified grid
    __hostdev__ static uint64_t memUsage(const GridT &grid) {
        uint64_t size = sizeof(NodeManagerData);
        if (!NodeManager::isLinear(grid)) {
            const uint32_t *p = grid.tree().mNodeCount;
            size += sizeof(int64_t)*(p[0]+p[1]+p[2]);
        }
        return size;
    }

    /// @brief Return the memory footprint in bytes of this instance
    __hostdev__ uint64_t memUsage() const {return NodeManager::memUsage(this->grid());}

    /// @brief Return a reference to the grid
    __hostdev__       GridT& grid()       { return *reinterpret_cast<GridT*>(DataT::mGrid); }
    __hostdev__ const GridT& grid() const { return *reinterpret_cast<const GridT*>(DataT::mGrid); }

    /// @brief Return a reference to the tree
    __hostdev__       TreeT& tree()       { return this->grid().tree(); }
    __hostdev__ const TreeT& tree() const { return this->grid().tree(); }

    /// @brief Return a reference to the root
    __hostdev__       RootT& root()       { return this->tree().root(); }
    __hostdev__ const RootT& root() const { return this->tree().root(); }

    /// @brief Return the number of tree nodes at the specified level
    /// @details 0 is leaf, 1 is lower internal, and 2 is upper internal level
    __hostdev__ uint64_t nodeCount(int level) const { return this->tree().nodeCount(level); }

    __hostdev__ uint64_t leafCount()  const { return this->tree().nodeCount(0); }
    __hostdev__ uint64_t lowerCount() const { return this->tree().nodeCount(1); }
    __hostdev__ uint64_t upperCount() const { return this->tree().nodeCount(2); }

    /// @brief Return the i'th leaf node with respect to breadth-first ordering
    template <int LEVEL>
    __hostdev__ const NodeT<LEVEL>& node(uint32_t i) const {
        NANOVDB_ASSERT(i < this->nodeCount(LEVEL));
        const NodeT<LEVEL>* ptr = nullptr;
        if (DataT::mLinear) {
            ptr = util::PtrAdd<const NodeT<LEVEL>>(DataT::mGrid, DataT::mOff[LEVEL]) + i;
        } else {
            ptr = util::PtrAdd<const NodeT<LEVEL>>(DataT::mGrid, DataT::mPtr[LEVEL][i]);
        }
        NANOVDB_ASSERT(ptr && isAligned(ptr));
        return *ptr;
    }

    /// @brief Return the i'th node with respect to breadth-first ordering
    template <int LEVEL>
    __hostdev__ NodeT<LEVEL>& node(uint32_t i) {
        NANOVDB_ASSERT(i < this->nodeCount(LEVEL));
        NodeT<LEVEL>* ptr = nullptr;
        if (DataT::mLinear) {
            ptr = util::PtrAdd<NodeT<LEVEL>>(DataT::mGrid, DataT::mOff[LEVEL]) + i;
        } else {
            ptr = util::PtrAdd<NodeT<LEVEL>>(DataT::mGrid, DataT::mPtr[LEVEL][i]);
        }
        NANOVDB_ASSERT(ptr && isAligned(ptr));
        return *ptr;
    }

    /// @brief Return the i'th leaf node with respect to breadth-first ordering
    __hostdev__ const Node0& leaf(uint32_t i) const { return this->node<0>(i); }
    __hostdev__       Node0& leaf(uint32_t i)       { return this->node<0>(i); }

    /// @brief Return the i'th lower internal node with respect to breadth-first ordering
    __hostdev__ const Node1& lower(uint32_t i) const { return this->node<1>(i); }
    __hostdev__       Node1& lower(uint32_t i)       { return this->node<1>(i); }

    /// @brief Return the i'th upper internal node with respect to breadth-first ordering
    __hostdev__ const Node2& upper(uint32_t i) const { return this->node<2>(i); }
    __hostdev__       Node2& upper(uint32_t i)       { return this->node<2>(i); }

}; // NodeManager<BuildT> class

template <typename BuildT, typename BufferT>
NodeManagerHandle<BufferT> createNodeManager(const NanoGrid<BuildT> &grid,
                                             const BufferT& buffer)
{
    NodeManagerHandle<BufferT> handle(toGridType<BuildT>(), BufferT::create(NodeManager<BuildT>::memUsage(grid), &buffer));
    auto *data = reinterpret_cast<NodeManagerData*>(handle.data());
    NANOVDB_ASSERT(data && isAligned(data));
    NANOVDB_ASSERT(toGridType<BuildT>() == grid.gridType());
    *data = NodeManagerData((void*)&grid);

    if (NodeManager<BuildT>::isLinear(grid)) {
        data->mLinear = uint8_t(1u);
        data->mOff[0] = util::PtrDiff(grid.tree().template getFirstNode<0>(), &grid);
        data->mOff[1] = util::PtrDiff(grid.tree().template getFirstNode<1>(), &grid);
        data->mOff[2] = util::PtrDiff(grid.tree().template getFirstNode<2>(), &grid);
    } else {
        int64_t *ptr0 = data->mPtr[0] = reinterpret_cast<int64_t*>(data + 1);
        int64_t *ptr1 = data->mPtr[1] = data->mPtr[0] + grid.tree().nodeCount(0);
        int64_t *ptr2 = data->mPtr[2] = data->mPtr[1] + grid.tree().nodeCount(1);
        // Performs depth first traversal but breadth first insertion
        for (auto it2 = grid.tree().root().cbeginChild(); it2; ++it2) {
            *ptr2++ = util::PtrDiff(&*it2, &grid);
            for (auto it1 = it2->cbeginChild(); it1; ++it1) {
                *ptr1++ = util::PtrDiff(&*it1, &grid);
                for (auto it0 = it1->cbeginChild(); it0; ++it0) {
                    *ptr0++ = util::PtrDiff(&*it0, &grid);
                }// loop over child nodes of the lower internal node
            }// loop over child nodes of the upper internal node
        }// loop over child nodes of the root node
    }

    return handle;// // is converted to r-value so return value is move constructed!
}// createNodeManager

} // namespace nanovdb

#if defined(__CUDACC__)
#include <nanovdb/cuda/NodeManager.cuh>
#endif// defined(__CUDACC__)

#endif // NANOVDB_NODEMANAGER_H_HAS_BEEN_INCLUDED
