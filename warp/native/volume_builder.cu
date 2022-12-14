#include "volume_builder.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>

// Explanation of key types
// ------------------------
//
// leaf_key:
// .__.__. .... .__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.__.
//  63 62  ....  27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 09 08 07 06 05 04 03 02 01 00
//  XX|< tile key >|<               upper offset               >|<           lower offset          >|
//
// tile key (36 bit):
//   (uint32(ijk[2]) >> ChildT::TOTAL) |
//   (uint64_t(uint32(ijk[1]) >> ChildT::TOTAL)) << 12 |
//   (uint64_t(uint32(ijk[0]) >> ChildT::TOTAL)) << 24 
//
// lower_key (51 bits) == leaf_key >> 12
//
// upper_key (36 bits) == lower_key >> 15 == leaf_key >> 27 == tile key

CUDA_CALLABLE inline uint64_t coord_to_full_key(const nanovdb::Coord& ijk) 
{
    using Tree = nanovdb::FloatTree; // any type is fine at this point
    assert((abs(ijk[0]) >> 24) == 0);
    assert((abs(ijk[1]) >> 24) == 0);
    assert((abs(ijk[2]) >> 24) == 0);
    constexpr uint32_t MASK_12BITS = (1u << 12) - 1u;
    const uint64_t     tile_key36 =
        ((uint32_t(ijk[2]) >> 12) & MASK_12BITS) | // z is the lower 12 bits
        (uint64_t((uint32_t(ijk[1]) >> 12) & MASK_12BITS) << 12) | // y is the middle 12 bits
        (uint64_t((uint32_t(ijk[0]) >> 12) & MASK_12BITS) << 24); // x is the upper 12 bits
    const uint32_t upper_offset = Tree::Node2::CoordToOffset(ijk);
    const uint32_t lower_offset = Tree::Node1::CoordToOffset(ijk);
    return (tile_key36 << 27) | (upper_offset << 12) | lower_offset; 
}

__global__
void generate_keys(size_t num_points, const nanovdb::Coord* points, uint64_t* all_leaf_keys)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;

    all_leaf_keys[tid] = coord_to_full_key(points[tid]); 
}

__global__
void generate_keys(size_t num_points, const nanovdb::Vec3f* points, uint64_t* all_leaf_keys, float one_over_voxel_size, nanovdb::Vec3f translation)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;

    const nanovdb::Coord ijk = ((points[tid] - translation) * one_over_voxel_size).round();
    all_leaf_keys[tid] = coord_to_full_key(ijk); 
}

// Convert a 36 bit tile key to the ijk origin of the addressed tile
CUDA_CALLABLE inline nanovdb::Coord tile_key36_to_coord(uint64_t tile_key36) {
    auto extend_sign = [](uint32_t i) -> int32_t { return i | ((i>>11 & 1) * 0xFFFFF800);};
    constexpr uint32_t MASK_12BITS = (1u << 12) - 1u;
    const int32_t i = extend_sign(uint32_t(tile_key36 >> 24) & MASK_12BITS);
    const int32_t j = extend_sign(uint32_t(tile_key36 >> 12) & MASK_12BITS);
    const int32_t k = extend_sign(uint32_t(tile_key36) & MASK_12BITS);
    return nanovdb::Coord(i, j, k) << 12;
}


// --- CUB helpers ---
template<uint8_t bits, typename InType, typename OutType>
struct ShiftRight {
    CUDA_CALLABLE inline OutType operator()(const InType& v) const {
        return static_cast<OutType>(v >> bits);
    }
};

template<uint8_t bits, typename InType = uint64_t, typename OutType = uint64_t>
struct ShiftRightIterator : public cub::TransformInputIterator<OutType, ShiftRight<bits, InType, OutType>, InType*> {
    using BASE = cub::TransformInputIterator<OutType, ShiftRight<bits, InType, OutType>, InType*>;
    CUDA_CALLABLE inline ShiftRightIterator(uint64_t* input_itr)
        : BASE(input_itr, ShiftRight<bits, InType, OutType>()) {}
};


// --- Atomic instructions for NanoVDB construction ---
template<typename MaskT>
CUDA_CALLABLE_DEVICE void set_mask_atomic(MaskT& mask, uint32_t n) {
    unsigned long long int* words = reinterpret_cast<unsigned long long int*>(&mask);
    atomicOr(words + (n / 64), 1ull << (n & 63));
}

template<typename Vec3T>
CUDA_CALLABLE_DEVICE void expand_cwise_atomic(nanovdb::BBox<Vec3T>& bbox, const Vec3T& v) {
    atomicMin(&bbox.mCoord[0][0], v[0]);
    atomicMin(&bbox.mCoord[0][1], v[1]);
    atomicMin(&bbox.mCoord[0][2], v[2]);
    atomicMax(&bbox.mCoord[1][0], v[0]);
    atomicMax(&bbox.mCoord[1][1], v[1]);
    atomicMax(&bbox.mCoord[1][2], v[2]);
}

template<typename RootDataType>
__hostdev__ const typename RootDataType::Tile* find_tile(const RootDataType* root_data, const nanovdb::Coord& ijk)
{
    using Tile = typename RootDataType::Tile;
    const Tile *tiles = reinterpret_cast<const Tile *>(root_data + 1);
    const auto key = RootDataType::CoordToKey(ijk);

    for (uint32_t i = 0; i < root_data->mTableSize; ++i)
    {
        if (tiles[i].key == key)
            return &tiles[i];
    }
    return nullptr;
}

// --- Wrapper for launching lambda kernels
template<typename Func, typename... Args>
__global__ void kernel(const size_t num_items, Func f, Args... args)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) return;
    f(tid, args...);
}

template <typename BuildT>
void build_grid_from_tiles(nanovdb::Grid<nanovdb::NanoTree<BuildT>> *&out_grid,
                           size_t &out_grid_size,
                           const void *points,
                           size_t num_points,
                           bool points_in_world_space,
                           const BuildGridParams<BuildT> &params)
{
    using FloatT = typename nanovdb::FloatTraits<BuildT>::FloatType;
    const BuildT ZERO_VAL{0};
    const FloatT ZERO_SCALAR{0};

    // Don't want to access "params" in kernels
    const double dx = params.voxel_size;
    const double Tx = params.translation[0], Ty = params.translation[1], Tz = params.translation[2];
    const BuildT background_value = params.background_value;

    const unsigned int num_threads = 256;
    unsigned int num_blocks;

    out_grid = nullptr;
    out_grid_size = 0;

    cub::CachingDeviceAllocator allocator;
    
    uint64_t* leaf_keys;
    uint64_t* lower_keys;
    uint64_t* upper_keys;
    uint32_t* node_counts;
    uint32_t leaf_count, lower_node_count, upper_node_count;

    allocator.DeviceAllocate((void**)&leaf_keys, sizeof(uint64_t) * num_points);
    allocator.DeviceAllocate((void**)&node_counts, sizeof(uint32_t) * 3);

    // Phase 1: counting the nodes
    {
        // Generating keys from coords
        uint64_t* all_leaf_keys;
        uint64_t* all_leaf_keys_sorted;
        allocator.DeviceAllocate((void**)&all_leaf_keys, sizeof(uint64_t) * num_points);
        allocator.DeviceAllocate((void**)&all_leaf_keys_sorted, sizeof(uint64_t) * num_points);

        num_blocks = (static_cast<unsigned int>(num_points) + num_threads - 1) / num_threads;
        if (points_in_world_space) {
            generate_keys<<<num_blocks, num_threads>>>(num_points, static_cast<const nanovdb::Vec3f*>(points), all_leaf_keys, static_cast<float>(1.0 / dx), nanovdb::Vec3f(params.translation));
        } else {
            generate_keys<<<num_blocks, num_threads>>>(num_points, static_cast<const nanovdb::Coord*>(points), all_leaf_keys);
        }

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        // Sort the keys, then get an array of unique keys
        cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, all_leaf_keys, all_leaf_keys_sorted, static_cast<int>(num_points), /* begin_bit = */ 0, /* end_bit = */ 63);
        allocator.DeviceAllocate((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, all_leaf_keys, all_leaf_keys_sorted, static_cast<int>(num_points), /* begin_bit = */ 0, /* end_bit = */ 63);
        allocator.DeviceFree(d_temp_storage);

        cub::DeviceSelect::Unique(nullptr, temp_storage_bytes, all_leaf_keys_sorted, leaf_keys, node_counts, static_cast<int>(num_points));
        allocator.DeviceAllocate((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, all_leaf_keys_sorted, leaf_keys, node_counts, static_cast<int>(num_points));
        allocator.DeviceFree(d_temp_storage);
        check_cuda(cudaMemcpy(&leaf_count, node_counts, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        allocator.DeviceFree(all_leaf_keys);
        all_leaf_keys = nullptr;
        allocator.DeviceFree(all_leaf_keys_sorted);
        all_leaf_keys_sorted = nullptr;


        // Get the keys unique to lower nodes and the number of them
        allocator.DeviceAllocate((void**)&lower_keys, sizeof(uint64_t) * leaf_count);
        cub::DeviceSelect::Unique(nullptr, temp_storage_bytes, ShiftRightIterator<12>(leaf_keys), lower_keys, node_counts + 1, leaf_count);
        allocator.DeviceAllocate((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, ShiftRightIterator<12>(leaf_keys), lower_keys, node_counts + 1, leaf_count);
        allocator.DeviceFree(d_temp_storage);
        check_cuda(cudaMemcpy(&lower_node_count, node_counts + 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // Get the keys unique to upper nodes and the number of them
        allocator.DeviceAllocate((void**)&upper_keys, sizeof(uint64_t) * lower_node_count);
        cub::DeviceSelect::Unique(nullptr, temp_storage_bytes, ShiftRightIterator<15>(lower_keys), upper_keys, node_counts + 2, lower_node_count);
        allocator.DeviceAllocate((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, ShiftRightIterator<15>(lower_keys), upper_keys, node_counts + 2, lower_node_count);
        allocator.DeviceFree(d_temp_storage);
        check_cuda(cudaMemcpy(&upper_node_count, node_counts + 2, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }

    using Tree = nanovdb::NanoTree<BuildT>;
    using Grid = nanovdb::Grid<Tree>;

    const size_t total_bytes =
        sizeof(Grid) +
        sizeof(Tree) +
        sizeof(typename Tree::RootType) +
        sizeof(typename Tree::RootType::Tile) * upper_node_count +
        sizeof(typename Tree::Node2) * upper_node_count +
        sizeof(typename Tree::Node1) * lower_node_count +
        sizeof(typename Tree::Node0) * leaf_count;

    const int64_t upper_mem_offset =
        sizeof(nanovdb::GridData) + sizeof(Tree) + sizeof(typename Tree::RootType) + 
        sizeof(typename Tree::RootType::Tile) * upper_node_count;
    const int64_t lower_mem_offset = upper_mem_offset + sizeof(typename Tree::Node2) * upper_node_count;
    const int64_t leaf_mem_offset = lower_mem_offset + sizeof(typename Tree::Node1) * lower_node_count;

    typename Grid::DataType* grid;
    check_cuda(cudaMalloc(&grid, total_bytes));

    typename Tree::DataType* const tree = reinterpret_cast<typename Tree::DataType*>(grid + 1); // The tree is immediately after the grid
    typename Tree::RootType::DataType* const root = reinterpret_cast<typename Tree::RootType::DataType*>(tree + 1); // The root is immediately after the tree
    typename Tree::RootType::Tile* const tiles = reinterpret_cast<typename Tree::RootType::Tile*>(root + 1);
    typename Tree::Node2::DataType* const upper_nodes = nanovdb::PtrAdd<typename Tree::Node2::DataType>(grid, upper_mem_offset);
    typename Tree::Node1::DataType* const lower_nodes = nanovdb::PtrAdd<typename Tree::Node1::DataType>(grid, lower_mem_offset);
    typename Tree::Node0::DataType* const leaf_nodes  = nanovdb::PtrAdd<typename Tree::Node0::DataType>(grid, leaf_mem_offset);

    // Phase 2: building the tree
    {
        // Setting up the tree and root node
        kernel<<<1, 1>>>(1, [=] __device__(size_t i) {
            tree->mNodeOffset[3] = sizeof(Tree);
            tree->mNodeOffset[2] = tree->mNodeOffset[3] + sizeof(typename Tree::RootType) + sizeof(typename Tree::RootType::Tile) * upper_node_count;
            tree->mNodeOffset[1] = tree->mNodeOffset[2] + sizeof(typename Tree::Node2) * upper_node_count;
            tree->mNodeOffset[0] = tree->mNodeOffset[1] + sizeof(typename Tree::Node1) * lower_node_count;
            tree->mNodeCount[2] = tree->mTileCount[2] = upper_node_count;
            tree->mNodeCount[1] = tree->mTileCount[1] = lower_node_count;
            tree->mNodeCount[0] = tree->mTileCount[0] = leaf_count;
            tree->mVoxelCount = Tree::Node0::SIZE * leaf_count; // assuming full leaves

            root->mBBox = nanovdb::CoordBBox(); // init to empty
            root->mTableSize = upper_node_count;
            root->mBackground = background_value;
            root->mMinimum = ZERO_VAL;
            root->mMaximum = ZERO_VAL;
            root->mAverage = ZERO_SCALAR;
            root->mStdDevi = ZERO_SCALAR;
        });
    }

    // Add tiles and upper nodes
    // i : 0 .. upper_node_count-1 
    num_blocks = (upper_node_count + num_threads - 1) / num_threads;
    {
        kernel<<<num_blocks, num_threads>>>(upper_node_count, [=] __device__(size_t i) {
            tiles[i].key = root->CoordToKey(tile_key36_to_coord(upper_keys[i]));
            tiles[i].child = sizeof(typename Tree::RootType) + sizeof(typename Tree::RootType::Tile) * upper_node_count + sizeof(typename Tree::Node2) * i;
            tiles[i].state = 0;
            tiles[i].value = background_value;

            assert(reinterpret_cast<const char*>(root->getChild(tiles + i)) == reinterpret_cast<const char*>(upper_nodes + i));
            auto& node = upper_nodes[i];
            node.mBBox = nanovdb::CoordBBox();
            node.mFlags = 0;
            node.mValueMask.setOff();
            node.mChildMask.setOff();
            node.mMinimum = ZERO_VAL;
            node.mMaximum = ZERO_VAL;
            node.mAverage = ZERO_SCALAR;
            node.mStdDevi = ZERO_SCALAR;
            for (size_t n = 0; n < Tree::Node2::SIZE; ++n) {
                node.mTable[n].value = background_value;
            }
        });
    }

    constexpr uint32_t MASK_15BITS = (1u << 15) - 1u;
    constexpr uint32_t MASK_12BITS = (1u << 12) - 1u;

    // Init lower nodes and register to parent
    // i : 0 .. lower_node_count-1 
    num_blocks = (lower_node_count + num_threads - 1) / num_threads;
    {
        kernel<<<num_blocks, num_threads>>>(lower_node_count, [=] __device__(size_t i) {
            uint32_t upper_offset = lower_keys[i] & MASK_15BITS;
            auto*    upper_node = root->getChild(find_tile(root, tile_key36_to_coord(lower_keys[i] >> 15)))->data();
            set_mask_atomic(upper_node->mChildMask, upper_offset);
            upper_node->setChild(upper_offset, lower_nodes + i);

            auto& node = lower_nodes[i];
            node.mBBox = nanovdb::CoordBBox();
            node.mFlags = 0;
            node.mValueMask.setOff();
            node.mChildMask.setOff();
            node.mMinimum = ZERO_VAL;
            node.mMaximum = ZERO_VAL;
            node.mAverage = ZERO_SCALAR;
            node.mStdDevi = ZERO_SCALAR;
            for (size_t n = 0; n < Tree::Node1::SIZE; ++n) {
                node.mTable[n].value = background_value;
            }
        });
    }

    // Init leaf nodes and register to parent
    // i : 0 .. leaf_count-1 
    num_blocks = (leaf_count + num_threads - 1) / num_threads;
    {
        kernel<<<num_blocks, num_threads>>>(leaf_count, [=] __device__(size_t i) {
            uint32_t lower_offset = leaf_keys[i] & MASK_12BITS;
            uint32_t upper_offset = (leaf_keys[i] >> 12) & MASK_15BITS;
            const nanovdb::Coord ijk = tile_key36_to_coord(leaf_keys[i] >> 27);

            auto* upper_node = root->getChild(find_tile(root, ijk))->data();
            auto* lower_node = upper_node->getChild(upper_offset)->data();
            set_mask_atomic(lower_node->mChildMask, lower_offset);
            lower_node->setChild(lower_offset, leaf_nodes + i);

            const nanovdb::Coord localUpperIjk = Tree::Node2::OffsetToLocalCoord(upper_offset) << Tree::Node1::TOTAL;
            const nanovdb::Coord localLowerIjk = Tree::Node1::OffsetToLocalCoord(lower_offset) << Tree::Node0::TOTAL;
            const nanovdb::Coord leafOrigin = ijk + localUpperIjk + localLowerIjk;

            auto& node = leaf_nodes[i];
            node.mBBoxMin = leafOrigin;
            node.mBBoxDif[0] = leaf_nodes[i].mBBoxDif[1] = leaf_nodes[i].mBBoxDif[2] = Tree::Node0::DIM;
            node.mFlags = 0;
            node.mValueMask.setOn();
            node.mMinimum = ZERO_VAL;
            node.mMaximum = ZERO_VAL;
            node.mAverage = ZERO_SCALAR;
            node.mStdDevi = ZERO_SCALAR;
            // mValues is undefined

            // propagating bbox up:
            expand_cwise_atomic(lower_node->mBBox, leafOrigin);
            expand_cwise_atomic(lower_node->mBBox, leafOrigin + nanovdb::Coord(Tree::Node0::DIM));
        });
    }

    // Propagating bounding boxes from lower nodes to upper nodes
    // i : 0 .. lower_node_count-1 
    num_blocks = (lower_node_count + num_threads - 1) / num_threads;
    {
        kernel<<<num_blocks, num_threads>>>(lower_node_count, [=] __device__(size_t i) {
            auto* upper_node = root->getChild(find_tile(root, tile_key36_to_coord(lower_keys[i] >> 15)))->data();
            expand_cwise_atomic(upper_node->mBBox, lower_nodes[i].mBBox.min());
            expand_cwise_atomic(upper_node->mBBox, lower_nodes[i].mBBox.max());
        });
    }

    // Setting up root bounding box and grid
    {
        kernel<<<1, 1>>>(1, [=] __device__(size_t i) {
            for (int i = 0; i < upper_node_count; ++i) {
                root->mBBox.expand(upper_nodes[i].mBBox.min());
                root->mBBox.expand(upper_nodes[i].mBBox.max());
            }

            nanovdb::Map map;
            {
                const double mat[4][4] = {
                    {dx, 0.0, 0.0, 0.0}, // row 0
                    {0.0, dx, 0.0, 0.0}, // row 1
                    {0.0, 0.0, dx, 0.0}, // row 2
                    {Tx, Ty, Tz, 1.0}, // row 3
                };
                const double invMat[4][4] = {
                    {1 / dx, 0.0, 0.0, 0.0}, // row 0
                    {0.0, 1 / dx, 0.0, 0.0}, // row 1
                    {0.0, 0.0, 1 / dx, 0.0}, // row 2
                    {0.0, 0.0, 0.0, 0.0}, // row 3, ignored by Map::set
                };
                map.set(mat, invMat, 1.0);
            }

            grid->mMagic = NANOVDB_MAGIC_NUMBER;
            grid->mChecksum = 0xFFFFFFFFFFFFFFFFull;
            grid->mVersion = nanovdb::Version();
            grid->mFlags = static_cast<uint32_t>(nanovdb::GridFlags::HasBBox) | 
                           static_cast<uint32_t>(nanovdb::GridFlags::IsBreadthFirst);
            grid->mGridIndex = 0;
            grid->mGridCount = 1;
            grid->mGridSize = total_bytes;
            // mGridName is set below
            grid->mWorldBBox.mCoord[0] = map.applyMap(nanovdb::Vec3R(root->mBBox.mCoord[0]));
            grid->mWorldBBox.mCoord[1] = map.applyMap(nanovdb::Vec3R(root->mBBox.mCoord[1]));
            grid->mVoxelSize = nanovdb::Vec3d(dx);
            grid->mMap = map;
            grid->mGridClass = nanovdb::GridClass::Unknown;
            grid->mGridType = nanovdb::mapToGridType<BuildT>();
            grid->mBlindMetadataOffset = total_bytes;
            grid->mBlindMetadataCount = 0;
        });
    }

    check_cuda(cudaMemcpy(grid->mGridName, params.name, 256, cudaMemcpyHostToDevice));

    allocator.DeviceFree(lower_keys);
    allocator.DeviceFree(upper_keys);
    allocator.DeviceFree(leaf_keys);
    allocator.DeviceFree(node_counts);

    out_grid = reinterpret_cast<Grid*>(grid);
    out_grid_size = total_bytes;
}

template void build_grid_from_tiles(nanovdb::Grid<nanovdb::NanoTree<float>>*&, size_t&, const void*, size_t, bool, const BuildGridParams<float>&);
template void build_grid_from_tiles(nanovdb::Grid<nanovdb::NanoTree<nanovdb::Vec3f>>*&, size_t&, const void*, size_t, bool, const BuildGridParams<nanovdb::Vec3f>&);
template void build_grid_from_tiles(nanovdb::Grid<nanovdb::NanoTree<int32_t>>*&, size_t&, const void*, size_t, bool, const BuildGridParams<int32_t>&);
