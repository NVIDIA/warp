/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "volume_builder.h"

#include <nanovdb/tools/cuda/PointsToGrid.cuh>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cub/cub.cuh>

#if defined(__NVCC_DIAG_PRAGMA_SUPPORT__)
// dynamic initialization is not supported for a function-scope static __shared__ variable within a
// __device__/__global__ function
#pragma nv_diag_suppress 20054
#elif defined(__NVCC__)
#pragma diag_suppress 20054
#endif
namespace
{
/// Allocator class following interface of cub::cachingDeviceAllocator, as expected by naovdb::PointsToGrid
struct Allocator
{

    cudaError_t DeviceAllocate(void **d_ptr,               ///< [out] Reference to pointer to the allocation
                               size_t bytes,               ///< [in] Minimum number of bytes for the allocation
                               cudaStream_t active_stream) ///< [in] The stream to be associated with this allocation
    {
        // in PointsToGrid stream argument always coincide with current stream, ignore
        *d_ptr = wp_alloc_device(WP_CURRENT_CONTEXT, bytes);
        cudaCheckError();
        return cudaSuccess;
    }

    cudaError_t DeviceFree(void *d_ptr)
    {
        wp_free_device(WP_CURRENT_CONTEXT, d_ptr);
        return cudaSuccess;
    }

    cudaError_t FreeAllCached()
    {
        return cudaSuccess;
    }
};

/// @brief  Implementation of NanoVDB's DeviceBuffer that uses warp allocators
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
    static DeviceBuffer create(uint64_t size, const DeviceBuffer *dummy = nullptr, bool host = true,
                               void *stream = nullptr)
    {
        return DeviceBuffer(size, host, stream);
    }

    /// @brief Static factory method that return an instance of this buffer that wraps externally managed memory
    /// @param size byte size of buffer specified by external memory
    /// @param cpuData pointer to externally managed host memory
    /// @param gpuData pointer to externally managed device memory
    /// @return An instance of this class using move semantics
    static DeviceBuffer create(uint64_t size, void *cpuData, void *gpuData)
    {
        return DeviceBuffer(size, cpuData, gpuData);
    }

    /// @brief Constructor
    /// @param size byte size of buffer to be initialized
    /// @param host If true buffer is initialized only on the host/CPU, else on the device/GPU
    /// @param stream optional stream argument (defaults to stream NULL)
    DeviceBuffer(uint64_t size = 0, bool host = true, void *stream = nullptr)
        : mSize(0), mCpuData(nullptr), mGpuData(nullptr), mManaged(false)
    {
        if (size > 0)
            this->init(size, host, stream);
    }

    DeviceBuffer(uint64_t size, void *cpuData, void *gpuData)
        : mSize(size), mCpuData(cpuData), mGpuData(gpuData), mManaged(false)
    {
    }

    /// @brief Disallow copy-construction
    DeviceBuffer(const DeviceBuffer &) = delete;

    /// @brief Move copy-constructor
    DeviceBuffer(DeviceBuffer &&other) noexcept
        : mSize(other.mSize), mCpuData(other.mCpuData), mGpuData(other.mGpuData), mManaged(other.mManaged)
    {
        other.mSize = 0;
        other.mCpuData = nullptr;
        other.mGpuData = nullptr;
        other.mManaged = false;
    }

    /// @brief Disallow copy assignment operation
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    /// @brief Move copy assignment operation
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
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
    ~DeviceBuffer()
    {
        this->clear();
    };

    /// @brief Initialize buffer
    /// @param size byte size of buffer to be initialized
    /// @param host If true buffer is initialized only on the host/CPU, else on the device/GPU
    /// @note All existing buffers are first cleared
    /// @warning size is expected to be non-zero. Use clear() clear buffer!
    void init(uint64_t size, bool host = true, void *stream = nullptr)
    {
        if (mSize > 0)
            this->clear(stream);
        NANOVDB_ASSERT(size > 0);
        if (host)
        {
            mCpuData =
                wp_alloc_pinned(size); // un-managed pinned memory on the host (can be slow to access!). Always 32B aligned
        }
        else
        {
            mGpuData = wp_alloc_device(WP_CURRENT_CONTEXT, size);
        }
        cudaCheckError();
        mSize = size;
        mManaged = true;
    }

    /// @brief Returns a raw pointer to the host/CPU buffer managed by this allocator.
    /// @warning Note that the pointer can be NULL!
    void *data() const
    {
        return mCpuData;
    }

    /// @brief Returns a raw pointer to the device/GPU buffer managed by this allocator.
    /// @warning Note that the pointer can be NULL!
    void *deviceData() const
    {
        return mGpuData;
    }

    /// @brief Returns the size in bytes of the raw memory buffer managed by this allocator.
    uint64_t size() const
    {
        return mSize;
    }

    //@{
    /// @brief Returns true if this allocator is empty, i.e. has no allocated memory
    bool empty() const
    {
        return mSize == 0;
    }
    bool isEmpty() const
    {
        return mSize == 0;
    }
    //@}

    /// @brief Detach device data so it is not dealloced when this buffer is destroyed
    void detachDeviceData()
    {
        mGpuData = nullptr;
        if (!mCpuData)
        {
            mSize = 0;
        }
    }

    /// @brief De-allocate all memory managed by this allocator and set all pointers to NULL
    void clear(void *stream = nullptr)
    {
        if (mManaged && mGpuData)
            wp_free_device(WP_CURRENT_CONTEXT, mGpuData);
        if (mManaged && mCpuData)
            wp_free_pinned(mCpuData);
        mCpuData = mGpuData = nullptr;
        mSize = 0;
        mManaged = false;
    }

}; // DeviceBuffer class

template <typename Tree> __global__ void activateAllLeafVoxels(Tree *tree)
{
    const unsigned leaf_count = tree->mNodeCount[0];

    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < leaf_count)
    {
        // activate all leaf voxels
        typename Tree::LeafNodeType *const leaf_nodes = tree->getFirstLeaf();
        typename Tree::LeafNodeType &leaf = leaf_nodes[tid];
        leaf.mValueMask.setOn();
        leaf.updateBBox();
    }

    if (tid == 0)
    {
        tree->mVoxelCount = Tree::LeafNodeType::SIZE * leaf_count; // full leaves
    }
}

template <typename Node>
__device__ std::enable_if_t<!nanovdb::BuildTraits<typename Node::BuildType>::is_index> setBackgroundValue(
    Node &node, unsigned tile_id, const typename Node::BuildType background_value)
{
    node.setValue(tile_id, background_value);
}

template <typename Node>
__device__ std::enable_if_t<nanovdb::BuildTraits<typename Node::BuildType>::is_index> setBackgroundValue(
    Node &node, unsigned tile_id, const typename Node::BuildType background_value)
{
}

template <typename Node>
__device__ std::enable_if_t<!nanovdb::BuildTraits<typename Node::BuildType>::is_index> setBackgroundValue(
    Node &node, const typename Node::BuildType background_value)
{
    node.mBackground = background_value;
}

template <typename Node>
__device__ std::enable_if_t<nanovdb::BuildTraits<typename Node::BuildType>::is_index> setBackgroundValue(
    Node &node, const typename Node::BuildType background_value)
{
}

template <typename T>
struct alignas(alignof(T)) AlignedProxy
{
    char data[sizeof(T)];
};

template <typename Tree, typename NodeT>
__global__ void setInternalBBoxAndBackgroundValue(Tree *tree, const typename Tree::BuildType background_value)
{
    using BBox = nanovdb::math::BBox<typename NodeT::CoordT>;
    using BBoxProxy = AlignedProxy<BBox>;

    __shared__ BBoxProxy bbox_mem;

    BBox& bbox = reinterpret_cast<BBox&>(bbox_mem);

    const unsigned node_count = tree->mNodeCount[NodeT::LEVEL];
    const unsigned node_id = blockIdx.x;

    if (node_id < node_count)
    {

        if (threadIdx.x == 0)
        {
            new(&bbox) BBox();
        }

        __syncthreads();

        NodeT &node = tree->template getFirstNode<NodeT>()[node_id];
        for (unsigned child_id = threadIdx.x; child_id < NodeT::SIZE; child_id += blockDim.x)
        {
            if (node.isChild(child_id))
            {
                bbox.expandAtomic(node.getChild(child_id)->bbox());
            }
            else
            {
                setBackgroundValue(node, child_id, background_value);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0)
        {
            node.mBBox = bbox;
        }
    }
}

template <typename Tree>
__global__ void setRootBBoxAndBackgroundValue(nanovdb::Grid<Tree> *grid,
                                              const typename Tree::BuildType background_value)
{
    using BBox = typename Tree::RootNodeType::BBoxType;
    using BBoxProxy = AlignedProxy<BBox>;
    __shared__ BBoxProxy bbox_mem;

    BBox& bbox = reinterpret_cast<BBox&>(bbox_mem);

    Tree &tree = grid->tree();
    const unsigned upper_count = tree.mNodeCount[2];

    if (threadIdx.x == 0)
    {
        new(&bbox) BBox();
    }

    __syncthreads();

    for (unsigned upper_id = threadIdx.x; upper_id < upper_count; upper_id += blockDim.x)
    {
        typename Tree::UpperNodeType &upper = tree.getFirstUpper()[upper_id];
        bbox.expandAtomic(upper.bbox());
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        typename Tree::RootNodeType &root = tree.root();
        setBackgroundValue(root, background_value);
        root.mBBox = bbox;

        grid->mWorldBBox = root.mBBox.transform(grid->map());
    }
}

template <typename BuildT>
void finalize_grid(nanovdb::Grid<nanovdb::NanoTree<BuildT>> &out_grid, const BuildGridParams<BuildT> &params)
{
    // set background value, activate all voxels for allocated tiles and update bbox

    using Tree = nanovdb::NanoTree<BuildT>;
    Tree *tree = &out_grid.tree();

    int node_counts[3];
    wp_memcpy_d2h(WP_CURRENT_CONTEXT, node_counts, tree->mNodeCount, sizeof(node_counts));
    // synchronization below is unnecessary as node_counts is in pageable memory.
    // keep it for clarity
    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());
    wp_cuda_stream_synchronize(stream);

    const unsigned int leaf_count = node_counts[0];
    const unsigned int lower_count = node_counts[1];
    const unsigned int upper_count = node_counts[2];

    constexpr unsigned NUM_THREADS = 256;
    const unsigned leaf_blocks = (leaf_count + NUM_THREADS - 1) / NUM_THREADS;
    activateAllLeafVoxels<Tree><<<leaf_blocks, NUM_THREADS, 0, stream>>>(tree);

    setInternalBBoxAndBackgroundValue<Tree, typename Tree::LowerNodeType>
        <<<lower_count, NUM_THREADS, 0, stream>>>(tree, params.background_value);
    setInternalBBoxAndBackgroundValue<Tree, typename Tree::UpperNodeType>
        <<<upper_count, NUM_THREADS, 0, stream>>>(tree, params.background_value);
    setRootBBoxAndBackgroundValue<Tree><<<1, NUM_THREADS, 0, stream>>>(&out_grid, params.background_value);

    check_cuda(wp_cuda_context_check(WP_CURRENT_CONTEXT));
}

template <>
void finalize_grid(nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>> &out_grid,
                   const BuildGridParams<nanovdb::ValueOnIndex> &params)
{
    // nothing to do for OnIndex grids
}

/// "fancy-pointer" that transforms from world to index coordinates
struct WorldSpacePointsPtr
{
    const nanovdb::Vec3f *points;
    const nanovdb::Map map;

    __device__ nanovdb::Vec3f operator[](int idx) const
    {
        return map.applyInverseMapF(points[idx]);
    }

    __device__ nanovdb::Vec3f operator*() const
    {
        return (*this)[0];
    }
};

} // namespace

namespace nanovdb
{
template <> struct BufferTraits<DeviceBuffer>
{
    static constexpr bool hasDeviceDual = true;
};

} // namespace nanovdb

template <typename BuildT>
void build_grid_from_points(nanovdb::Grid<nanovdb::NanoTree<BuildT>> *&out_grid, size_t &out_grid_size,
                            const void *points, size_t num_points, bool points_in_world_space,
                            const BuildGridParams<BuildT> &params)
{

    out_grid = nullptr;
    out_grid_size = 0;

    try
    {

        cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());
        nanovdb::tools::cuda::PointsToGrid<BuildT, Allocator> p2g(params.map, stream);

        // p2g.setVerbose(2);
        p2g.setGridName(params.name);
        p2g.setChecksum(nanovdb::CheckMode::Disable);

        // Only compute bbox for OnIndex grids. Otherwise bbox will be computed after activating all leaf voxels
        p2g.includeBBox(nanovdb::BuildTraits<BuildT>::is_onindex);

        nanovdb::GridHandle<DeviceBuffer> grid_handle;

        if (points_in_world_space)
        {
            grid_handle = p2g.getHandle(WorldSpacePointsPtr{static_cast<const nanovdb::Vec3f*>(points), params.map},
                                        num_points, DeviceBuffer());
        }
        else
        {
            grid_handle = p2g.getHandle(static_cast<const nanovdb::Coord*>(points), num_points, DeviceBuffer());
        }

        out_grid = grid_handle.deviceGrid<BuildT>();
        out_grid_size = grid_handle.gridSize();

        finalize_grid(*out_grid, params);

        // So that buffer is not destroyed when handles goes out of scope
        grid_handle.buffer().detachDeviceData();
    }
    catch (const std::runtime_error& exc)
    {
        out_grid = nullptr;
        out_grid_size = 0;
    }
}


#define EXPAND_BUILDER_TYPE(type) \
template void build_grid_from_points(nanovdb::Grid<nanovdb::NanoTree<type>> *&, size_t &, const void *, size_t, bool, \
                                     const BuildGridParams<type> &);

WP_VOLUME_BUILDER_INSTANTIATE_TYPES
#undef EXPAND_BUILDER_TYPE

template void build_grid_from_points(nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueIndex>> *&, size_t &, const void *,
                                     size_t, bool, const BuildGridParams<nanovdb::ValueIndex> &);
template void build_grid_from_points(nanovdb::Grid<nanovdb::NanoTree<nanovdb::ValueOnIndex>> *&, size_t &, const void *,
                                     size_t, bool, const BuildGridParams<nanovdb::ValueOnIndex> &);
