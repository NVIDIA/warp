// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/PointsToGrid.cuh

    \authors Greg Klar (initial version) and Ken Museth (final version)

    \brief Generates NanoVDB grids from a list of voxels or points on the device

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_TOOLS_CUDA_POINTSTOGRID_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_POINTSTOGRID_CUH_HAS_BEEN_INCLUDED

#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>
#include <vector>
#include <tuple>
#include <cinttypes>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/tools/cuda/GridChecksum.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>

/*
   Note: 4.29 billion (=2^32) coordinates of type Vec3f have a memory footprint of 48 GB!
*/

namespace nanovdb {// ================================================================================

namespace tools::cuda {// ============================================================================

/// @brief Generates a NanoGrid<Point> from a list of point coordinates on the device. This method is
///        mainly used as a means to build a BVH acceleration structure for points, e.g. for efficient rendering.
/// @tparam PtrT Template type to a raw or fancy-pointer of point coordinates in world space. Dereferencing should return Vec3f or Vec3d.
/// @tparam BufferT Template type of buffer used for memory allocation on the device
/// @tparam ResourceT Template type of optional resource used for internal temporary memory
/// @param dWorldPoints Raw or fancy pointer to list of point coordinates in world space on the device
/// @param pointCount number of point in the list @c d_world
/// @param voxelSize Size of a voxel in world units used for the output grid
/// @param type Defined the way point information is represented in the output grid (see PointType enum NanoVDB.h)
///             Should not be PointType::Disable!
/// @param buffer Instance of the device buffer used for memory allocation
/// @param stream optional CUDA stream (defaults to CUDA stream 0)
/// @return Returns a handle with a grid of type NanoGrid<Point> where point information, e.g. coordinates,
///         are represented as blind data defined by @c type.
template<typename PtrT, typename BufferT = nanovdb::cuda::DeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
GridHandle<BufferT>
pointsToGrid(const PtrT dWorldPoints,
             int pointCount,
             double voxelSize,
             PointType type = PointType::Default,
             const BufferT &buffer = BufferT(),
             cudaStream_t stream = 0);

//-----------------------------------------------------------------------------------------------------

/// @brief Generates a NanoGrid<Point> from a list of point coordinates on the device. This method is
///        mainly used as a means to build a BVH acceleration structure for points, e.g. for efficient rendering.
/// @tparam PtrT Template type to a raw or fancy-pointer of point coordinates in world space. Dereferencing should return Vec3f or Vec3d.
/// @tparam BufferT Template type of buffer used for memory allocation on the device
/// @tparam ResourceT Template type of optional resource used for internal temporary memory
/// @param dWorldPoints Raw or fancy pointer to list of point coordinates in world space on the device
/// @param pointCount total number of point in the list @c d_world
/// @param maxPointsPerVoxel Max density of points per voxel, i.e. maximum number of points in any voxel
/// @param tolerance allow for point density to vary by the specified tolerance (defaults to 1). That is, the voxel size
///                  is selected such that the max density is +/- the tolerance.
/// @param maxIterations Maximum number of iterations used to search for a voxel size that produces a point density
///                      with specified tolerance takes.
/// @param type Defined the way point information is represented in the output grid (see PointType enum in NanoVDB.h)
///             Should not be PointType::Disable!
/// @param buffer Instance of the device buffer used for memory allocation
/// @param stream optional CUDA stream (defaults to CUDA stream 0)
/// @return Returns a handle with a grid of type NanoGrid<Point> where point information, e.g. coordinates,
///         are represented as blind data defined by @c type.
template<typename PtrT, typename BufferT = nanovdb::cuda::DeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
GridHandle<BufferT>
pointsToGrid(const PtrT dWorldPoints,
             int pointCount,
             int maxPointPerVoxel,
             int tolerance = 1,
             int maxIterations = 10,
             PointType type = PointType::Default,
             const BufferT &buffer = BufferT(),
             cudaStream_t stream = 0);

//-----------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT = nanovdb::cuda::DeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
GridHandle<BufferT>
pointsToGrid(std::vector<std::tuple<const PtrT,size_t,double,PointType>> pointSet,
            const BufferT &buffer = BufferT(),
            cudaStream_t stream = 0);

//-----------------------------------------------------------------------------------------------------

/// @brief Generates a NanoGrid of any type from a list of voxel coordinates on the device. Unlike @c cudaPointsToGrid
///        this method only builds the grid but does not encode the coordinates as blind data. It is mainly useful as a
///        means to generate a grid that is know to contain the voxels given in the list.
/// @tparam BuildT Template type of the return grid
/// @tparam PtrT Template type to a raw or fancy-pointer of point coordinates in world space. Dereferencing should return Vec3f or Vec3d.
/// @tparam BufferT Template type of buffer used for memory allocation on the device
/// @tparam ResourceT Template type of optional resource used for internal temporary memory
/// @param dGridVoxels Raw or fancy pointer to list of voxel coordinates in grid (or index) space on the device
/// @param pointCount number of voxel in the list @c dGridVoxels
/// @param voxelSize Size of a voxel in world units used for the output grid
/// @param buffer Instance of the device buffer used for memory allocation
/// @return Returns a handle with the grid of type NanoGrid<BuildT>
template<typename BuildT, typename PtrT, typename BufferT = nanovdb::cuda::DeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
GridHandle<BufferT>
voxelsToGrid(const PtrT dGridVoxels,
             size_t voxelCount,
             double voxelSize = 1.0,
             const BufferT &buffer = BufferT(),
             cudaStream_t stream = 0);

//-------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT = nanovdb::cuda::DeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
GridHandle<BufferT>
voxelsToGrid(std::vector<std::tuple<const PtrT, size_t, double>> pointSet,
             const BufferT &buffer = BufferT(),
             cudaStream_t stream = 0);

}// namespace tools::cuda ========================================================================

/// @brief Example class of a fancy pointer that can optionally be used as a template for writing
///        a custom fancy pointer that allows for particle coordinates to be arrange non-linearly
///        in memory. For instance, when coordinates are interlaced with other data, e.g. an array
///        of structs, a custom implementation of fancy_ptr::operator[](size_t i) can account for
///        strides that skip other interlaces data.
/// @tparam T Template type that specifies the type use for the coordinates of the points
template<typename T>
class fancy_ptr
{
    const T* mPtr;
public:
    /// @brief Default constructor.
    /// @note  This method is actually not required by cuda::PointsToGrid
    /// @param ptr Pointer to array of elements
    __hostdev__ explicit fancy_ptr(const T* ptr = nullptr) : mPtr(ptr) {}
    /// @brief Index acces into the array pointed to by the stored pointer.
    /// @note  This method is required by cuda::PointsToGrid!
    /// @param i Unsigned index of the element to be returned
    /// @return Const reference to the element at the i'th position
    __hostdev__ inline const T& operator[](size_t i) const {return mPtr[i];}
    /// @brief Dummy implementation required by pointer_traits.
    /// @note  Note that only the return type matters!
    /// @details Unlike operator[] it is safe to assume that all pointer types have operator*,
    ///          which is why pointer_traits makes use of it to determine the element_type that
    ///          a pointer class is pointing to. E.g. operator[] is not always defined for std::shared_ptr!
    __hostdev__ inline const T& operator*() const {return *mPtr;}
};// fancy_ptr<T>

/// @brief Simple stand-alone function that can be used to conveniently construct a fancy_ptr
/// @tparam T Template type that specifies the type use for the coordinates of the points
/// @param ptr Raw pointer to data
/// @return a new instance of a fancy_ptr
template<typename T>
fancy_ptr<T> make_fancy(const T* ptr = nullptr) {return fancy_ptr<T>(ptr);}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// @brief Trait of points, like type of pointer and size of the pointer type
template<typename>
struct pointer_traits;

template<typename T>
struct pointer_traits<T*> {
    using element_type = T;
    static constexpr size_t element_size = sizeof(T);
};

template<typename T>
struct pointer_traits {
    using element_type = typename util::remove_reference<decltype(*util::declval<T>())>::type;// assumes T::operator*() exists!
    static constexpr size_t element_size = sizeof(element_type);
};

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// @brief computes the relative 8-bit voxel offsets from a world coordinate
/// @tparam Vec3T Type of the world coordinate
/// @param voxel 8-bit output coordinates that are relative to a voxel
/// @param world input world coordinates
/// @param indexToWorld Transform from index to world space
template<typename Vec3T>
__hostdev__ inline static void worldToVoxel(Vec3u8 &voxel, const Vec3T &world, const Map &indexToWorld)
{
    const Vec3d ijk = indexToWorld.applyInverseMap(world);// world -> index
    static constexpr double encode = double((1<<8) - 1);
    voxel[0] = uint8_t( encode*(ijk[0] - math::Floor(ijk[0] + 0.5) + 0.5) );
    voxel[1] = uint8_t( encode*(ijk[1] - math::Floor(ijk[1] + 0.5) + 0.5) );
    voxel[2] = uint8_t( encode*(ijk[2] - math::Floor(ijk[2] + 0.5) + 0.5) );
}

/// @brief computes the relative 16-bit voxel offsets from a world coordinate
/// @tparam Vec3T Type of the world coordinate
/// @param voxel 16-bit output coordinates that are relative to a voxel
/// @param world input world coordinates
/// @param indexToWorld Transform from index to world space
template<typename Vec3T>
__hostdev__ inline static void worldToVoxel(Vec3u16 &voxel, const Vec3T &world, const Map &indexToWorld)
{
    const Vec3d ijk = indexToWorld.applyInverseMap(world);// world -> index
    static constexpr double encode = double((1<<16) - 1);
    voxel[0] = uint16_t( encode*(ijk[0] - math::Floor(ijk[0] + 0.5) + 0.5) );
    voxel[1] = uint16_t( encode*(ijk[1] - math::Floor(ijk[1] + 0.5) + 0.5) );
    voxel[2] = uint16_t( encode*(ijk[2] - math::Floor(ijk[2] + 0.5) + 0.5) );
}

/// @brief computes the relative float voxel offsets from a world coordinate
/// @tparam Vec3T Type of the world coordinate
/// @param voxel float output coordinates that are relative to a voxel
/// @param world input world coordinates
/// @param indexToWorld Transform from index to world space
template<typename Vec3T>
__hostdev__ inline static void worldToVoxel(Vec3f &voxel, const Vec3T &world, const Map &indexToWorld)
{
    const Vec3d ijk = indexToWorld.applyInverseMap(world);// world -> index
    voxel[0] = float( ijk[0] - math::Floor(ijk[0] + 0.5) );
    voxel[1] = float( ijk[1] - math::Floor(ijk[1] + 0.5) );
    voxel[2] = float( ijk[2] - math::Floor(ijk[2] + 0.5) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename Vec3T = Vec3d>
__hostdev__ inline static Vec3T voxelToWorld(const Vec3u8 &voxel, const Coord &ijk, const Map &map)
{
    static constexpr double decode = 1.0/double((1<<8) - 1);
    if constexpr(util::is_same<Vec3T,Vec3d>::value) {
        return map.applyMap( Vec3d(ijk[0] + decode*voxel[0] - 0.5, ijk[1] + decode*voxel[1] - 0.5, ijk[2] + decode*voxel[2] - 0.5));
    } else {
        return map.applyMapF(Vec3f(ijk[0] + decode*voxel[0] - 0.5f, ijk[1] + decode*voxel[1] - 0.5f, ijk[2] + decode*voxel[2] - 0.5f));
    }
}

template<typename Vec3T = Vec3d>
__hostdev__ inline static Vec3T voxelToWorld(const Vec3u16 &voxel, const Coord &ijk, const Map &map)
{
    static constexpr double decode = 1.0/double((1<<16) - 1);
    if constexpr(util::is_same<Vec3T,Vec3d>::value) {
        return map.applyMap( Vec3d(ijk[0] + decode*voxel[0] - 0.5, ijk[1] + decode*voxel[1] - 0.5, ijk[2] + decode*voxel[2] - 0.5));
    } else {
        return map.applyMapF(Vec3f(ijk[0] + decode*voxel[0] - 0.5f, ijk[1] + decode*voxel[1] - 0.5f, ijk[2] + decode*voxel[2] - 0.5f));
    }
}

template<typename Vec3T = Vec3d>
__hostdev__ inline static Vec3T voxelToWorld(const Vec3f &voxel, const Coord &ijk, const Map &map)
{
    if constexpr(util::is_same<Vec3T,Vec3d>::value) {
        return map.applyMap( Vec3d(ijk[0] + voxel[0], ijk[1] + voxel[1], ijk[2] + voxel[2]));
    } else {
        return map.applyMapF(Vec3f(ijk[0] + voxel[0], ijk[1] + voxel[1], ijk[2] + voxel[2]));
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace tools::cuda {

template<typename BuildT>
struct PointsToGridData {
    Map map;
    void     *d_bufferPtr;
    uint64_t *d_keys, *d_tile_keys, *d_lower_keys, *d_leaf_keys;// device pointer to 64 bit keys
    uint64_t  grid, tree, root, upper, lower, leaf, meta, blind, size;// byte offsets to nodes in buffer
    uint32_t *d_indx;// device pointer to point indices (or IDs)
    uint32_t  nodeCount[3], *pointsPerLeafPrefix, *pointsPerLeaf;// 0=leaf,1=lower, 2=upper
    uint32_t  voxelCount,  *pointsPerVoxelPrefix, *pointsPerVoxel;
    BitFlags<16> flags;
    __hostdev__ NanoGrid<BuildT>&  getGrid() const {return *util::PtrAdd<NanoGrid<BuildT>>(d_bufferPtr, grid);}
    __hostdev__ NanoTree<BuildT>&  getTree() const {return *util::PtrAdd<NanoTree<BuildT>>(d_bufferPtr, tree);}
    __hostdev__ NanoRoot<BuildT>&  getRoot() const {return *util::PtrAdd<NanoRoot<BuildT>>(d_bufferPtr, root);}
    __hostdev__ NanoUpper<BuildT>& getUpper(int i) const {return *(util::PtrAdd<NanoUpper<BuildT>>(d_bufferPtr, upper)+i);}
    __hostdev__ NanoLower<BuildT>& getLower(int i) const {return *(util::PtrAdd<NanoLower<BuildT>>(d_bufferPtr, lower)+i);}
    __hostdev__ NanoLeaf<BuildT>&  getLeaf(int i) const {return *(util::PtrAdd<NanoLeaf<BuildT>>(d_bufferPtr, leaf)+i);}
    __hostdev__ GridBlindMetaData& getMeta() const { return *util::PtrAdd<GridBlindMetaData>(d_bufferPtr, meta);};
     template<typename Vec3T>
    __hostdev__ Vec3T& getPoint(int i) const {return *(util::PtrAdd<Vec3T>(d_bufferPtr, blind)+i);}
};// PointsToGridData


template<typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class PointsToGrid
{
public:
    /// @brief Map constructor, which other constructors might call
    /// @param map Map to be used for the output device grid
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    PointsToGrid(const Map &map, cudaStream_t stream = 0)
        : mStream(stream)
        , mTimer(stream)
        , mPointType(util::is_same<BuildT,Point>::value ? PointType::Default : PointType::Disable)
    {
        mData.map = map;
        mData.flags.initMask({GridFlags::HasBBox, GridFlags::IsBreadthFirst});
        mDeviceData = static_cast<PointsToGridData<BuildT>*>(ResourceT::allocateAsync(sizeof(PointsToGridData<BuildT>), ResourceT::DEFAULT_ALIGNMENT, mStream));
    }

    /// @brief Default constructor that calls the Map constructor defined above
    /// @param scale Voxel size in world units
    /// @param trans Translation of origin in world units
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    PointsToGrid(const double scale = 1.0, const Vec3d &trans = Vec3d(0.0), cudaStream_t stream = 0)
        : PointsToGrid(Map(scale, trans), stream){}

    /// @brief Constructor from a target maximum number of particles per voxel. Calls the Map constructor defined above
    /// @param maxPointsPerVoxel Maximum number of points oer voxel
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    PointsToGrid(int maxPointsPerVoxel, int tolerance = 1, int maxIterations = 10, cudaStream_t stream = 0)
        : PointsToGrid(Map(1.0), stream)
    {
        mMaxPointsPerVoxel = maxPointsPerVoxel;
        mTolerance = tolerance;
        mMaxIterations = maxIterations;
    }

    ~PointsToGrid(){ ResourceT::deallocateAsync(mDeviceData, sizeof(PointsToGridData<BuildT>), ResourceT::DEFAULT_ALIGNMENT, mStream); }

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) {mVerbose = level; mData.flags.setBit(7u, level); }

    /// @brief Set the mode for checksum computation, which is disabled by default
    /// @param mode Mode of checksum computation
    void setChecksum(CheckMode mode = CheckMode::Disable){mChecksum = mode;}

    /// @brief Toggle on and off the computation of a bounding-box
    /// @param on If true bbox will be computed
    void includeBBox(bool on = true) { mData.flags.setMask(GridFlags::HasBBox, on); }

    /// @brief Set the name of the output grid
    /// @param name name of the output grid
    void setGridName(const std::string &name) {mGridName = name;}

    // only available when BuildT == Point
    template<typename T = BuildT> typename util::enable_if<util::is_same<T, Point>::value>::type
    setPointType(PointType type) { mPointType = type; }

    /// @brief Creates a handle to a grid with the specified build type from a list of points in index or world space
    /// @tparam BuildT Build type of the output grid, i.e NanoGrid<BuildT>
    /// @tparam PtrT Template type to a raw or fancy-pointer of point coordinates in world or index space.
    /// @tparam BufferT Buffer type used for allocation of the grid handle
    /// @param points device point to an array of points in world space
    /// @param pointCount number of input points or voxels
    /// @param buffer optional buffer (currently ignored)
    /// @return returns a handle with a grid of type NanoGrid<BuildT>
    template<typename PtrT, typename BufferT = nanovdb::cuda::DeviceBuffer>
    GridHandle<BufferT> getHandle(const PtrT points,
                                  size_t pointCount,
                                  const BufferT &buffer = BufferT());

    template<typename PtrT>
    void countNodes(const PtrT points, size_t pointCount);

    template<typename PtrT>
    void processGridTreeRoot(const PtrT points, size_t pointCount);

    void processUpperNodes();

    void processLowerNodes();

    void processLeafNodes(size_t pointCount);

    template<typename PtrT>
    void processPoints(const PtrT points, size_t pointCount);

    void processBBox();

    // the following methods are only defined when BuildT == Point
    template<typename T = BuildT> typename util::enable_if<util::is_same<T, Point>::value, uint32_t>::type
    maxPointsPerVoxel() const {return mMaxPointsPerVoxel;}
    template<typename T = BuildT> typename util::enable_if<util::is_same<T, Point>::value, uint32_t>::type
    maxPointsPerLeaf()  const {return mMaxPointsPerLeaf;}

private:
    static constexpr unsigned int mNumThreads = 128;// seems faster than the old value of 256!
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    cudaStream_t             mStream{0};
    util::cuda::Timer        mTimer;
    PointType                mPointType;
    std::string              mGridName;
    int                      mVerbose{0};
    PointsToGridData<BuildT> mData, *mDeviceData;
    uint32_t                 mMaxPointsPerVoxel{0u}, mMaxPointsPerLeaf{0u};
    int                      mTolerance{1}, mMaxIterations{1};
    CheckMode                mChecksum{CheckMode::Disable};

    nanovdb::cuda::TempPool<ResourceT> mTempDevicePool;

    template<typename PtrT, typename BufferT>
    BufferT getBuffer(const PtrT points, size_t pointCount, const BufferT &buffer);
};// tools::cuda::PointsToGrid<BuildT, ResourceT>

namespace kernels {
/// @details Used by cuda::PointsToGrid<BuildT, ResourceT>::processLeafNodes before the computation
/// of prefix-sum for index grid.
/// Moving this away from an implementation using the lambdaKernel wrapper
/// to fix the following on Windows platform:
/// error : For this host platform/dialect, an extended lambda cannot be defined inside the 'if'
/// or 'else' block of a constexpr if statement.
/// function in a lambda through lambdaKernel wrapper defined in CudaUtils.h.
template<typename BuildT>
__global__ void fillValueIndexKernel(const size_t numItems, unsigned int offset, uint64_t* devValueIndex, PointsToGridData<BuildT>* d_data) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numItems) return;
    devValueIndex[tid + offset] = static_cast<uint64_t>(d_data->getLeaf(tid + offset).mValueMask.countOn());
}

/// @details Used by PointsToGrid<BuildT, ResourceT>::processLeafNodes for the computation
/// of prefix-sum for index grid.
/// Moving this away from an implementation using the lambdaKernel wrapper
/// to fix the following on Windows platform:
/// error : For this host platform/dialect, an extended lambda cannot be defined inside the 'if'
/// or 'else' block of a constexpr if statement.
template<typename BuildT>
__global__ void leafPrefixSumKernel(const size_t numItems, unsigned int offset, uint64_t* devValueIndexPrefix, PointsToGridData<BuildT>* d_data) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numItems) return;

    auto &leaf = d_data->getLeaf(tid + offset);
    leaf.mOffset = 1u;// will be re-set below
    const uint64_t *w = leaf.mValueMask.words();
    uint64_t &prefixSum = leaf.mPrefixSum, sum = util::countOn(*w++);
    prefixSum = sum;
    for (int n = 9; n < 55; n += 9) {// n=i*9 where i=1,2,..6
        sum += util::countOn(*w++);
        prefixSum |= sum << n;// each pre-fixed sum is encoded in 9 bits
    }
    if ((tid + offset) == 0) {
        d_data->getGrid().mData1 = 1u + devValueIndexPrefix[d_data->nodeCount[0]-1];// set total count
        d_data->getTree().mVoxelCount = devValueIndexPrefix[d_data->nodeCount[0]-1];
    } else {
        leaf.mOffset = 1u + devValueIndexPrefix[tid + offset -1];// background is index 0
    }
}

/// @details Used by PointsToGrid<BuildT, ResourceT>::processLeafNodes to make sure leaf.mMask - leaf.mValueMask.
/// Moving this away from an implementation using the lambdaKernel wrapper
/// to fix the following on Windows platform:
/// error : For this host platform/dialect, an extended lambda cannot be defined inside the 'if'
/// or 'else' block of a constexpr if statement.
template<typename BuildT>
__global__ void setMaskEqValMaskKernel(const size_t numItems, unsigned int offset, PointsToGridData<BuildT>* d_data) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numItems) return;
    auto &leaf = d_data->getLeaf(tid + offset);
    leaf.mMask = leaf.mValueMask;
}
} // namespace kernels

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Define utility macro used to call cub functions that use dynamic temporary storage
#ifndef CALL_CUBS
#ifdef _WIN32
#define CALL_CUBS(func, ...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), __VA_ARGS__, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), __VA_ARGS__, mStream));
#else// fdef _WIN32
#define CALL_CUBS(func, args...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), args, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), args, mStream));
#endif// ifdef _WIN32
#endif// ifndef CALL_CUBS

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
template<typename PtrT, typename BufferT>
inline GridHandle<BufferT>
PointsToGrid<BuildT, ResourceT>::getHandle(const PtrT points,
                                size_t pointCount,
                                const BufferT &pool)
{
    if (mVerbose==1) mTimer.start("\nCounting nodes");
    this->countNodes(points, pointCount);

    if (mVerbose==1) mTimer.restart("Initiate buffer");
    auto buffer = this->getBuffer(points, pointCount, pool);

    if (mVerbose==1) mTimer.restart("Process grid,tree,root");
    this->processGridTreeRoot(points, pointCount);

    if (mVerbose==1) mTimer.restart("Process upper nodes");
    this->processUpperNodes();

    if (mVerbose==1) mTimer.restart("Process lower nodes");
    this->processLowerNodes();

    if (mVerbose==1) mTimer.restart("Process leaf nodes");
    this->processLeafNodes(pointCount);

    if (mVerbose==1) mTimer.restart("Process points");
    this->processPoints(points, pointCount);

    if (mVerbose==1) mTimer.restart("Process bbox");
    this->processBBox();
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.restart("Computation of checksum");
    tools::cuda::updateChecksum((GridData*)buffer.deviceData(), mChecksum, mStream);
    if (mVerbose==1) mTimer.stop();

    cudaStreamSynchronize(mStream);

    return GridHandle<BufferT>(std::move(buffer));
}// PointsToGrid<BuildT, ResourceT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// --- CUB helpers ---
template<uint8_t BitCount, typename InT = uint64_t, typename OutT = uint64_t>
struct ShiftRight
{
    __hostdev__ inline OutT operator()(const InT& v) const {return static_cast<OutT>(v >> BitCount);}
};

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT>
struct TileKeyFunctor {
    using Vec3T = typename util::remove_const<typename pointer_traits<PtrT>::element_type>::type;

    __device__
    void operator()(size_t tid, const PointsToGridData<BuildT> *d_data, const PtrT points, uint64_t* d_keys, uint32_t* d_indx) {
        auto coordToKey = [](const Coord &ijk)->uint64_t{
            // Note: int32_t has a range of -2^31 to 2^31 - 1 whereas uint32_t has a range of 0 to 2^32 - 1
            static constexpr int64_t kOffset = 1 << 31;
            return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)      ) | // z is the lower 21 bits
                   (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) | // y is the middle 21 bits
                   (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42); //  x is the upper 21 bits
        };// coordToKey lambda functor
        d_indx[tid] = uint32_t(tid);
        uint64_t &key = d_keys[tid];
        if constexpr(util::is_same<BuildT, Point>::value) {// points are in world space
            if constexpr(util::is_same<Vec3T, Vec3f>::value) {
                key = coordToKey(d_data->map.applyInverseMapF(points[tid]).round());
            } else {// points are Vec3d
                key = coordToKey(d_data->map.applyInverseMap(points[tid]).round());
            }
        } else if constexpr(util::is_same<Vec3T, Coord>::value) {// points Coord are in index space
            key = coordToKey(points[tid]);
        } else {// points are Vec3f or Vec3d in index space
            key = coordToKey(points[tid].round());
        }
    }
};

template<typename BuildT, typename PtrT>
struct VoxelKeyFunctor {
    using Vec3T = typename util::remove_const<typename pointer_traits<PtrT>::element_type>::type;

    __device__
    void operator()(size_t tid, const PointsToGridData<BuildT> *d_data, const PtrT points, uint64_t id, uint64_t *d_keys, const uint32_t *d_indx) {
        auto voxelKey = [] __device__ (uint64_t tileID, const Coord &ijk){
            return tileID << 36 |                                       // upper offset: 64-15-12-9=28, i.e. last 28 bits
                uint64_t(NanoUpper<BuildT>::CoordToOffset(ijk)) << 21 | // lower offset: 32^3 = 2^15,   i.e. next 15 bits
                uint64_t(NanoLower<BuildT>::CoordToOffset(ijk)) <<  9 | // leaf  offset: 16^3 = 2^12,   i.e. next 12 bits
                uint64_t(NanoLeaf< BuildT>::CoordToOffset(ijk));        // voxel offset:  8^3 =  2^9,   i.e. first 9 bits
        };// voxelKey lambda functor
        Vec3T p = points[d_indx[tid]];
        if constexpr(util::is_same<BuildT, Point>::value) p = util::is_same<Vec3T, Vec3f>::value ? d_data->map.applyInverseMapF(p) : d_data->map.applyInverseMap(p);
        d_keys[tid] = voxelKey(id, p.round());
    }
};

template<typename BuildT, typename ResourceT>
template<typename PtrT>
void PointsToGrid<BuildT, ResourceT>::countNodes(const PtrT points, size_t pointCount)
{
    using Vec3T = typename util::remove_const<typename pointer_traits<PtrT>::element_type>::type;
    if constexpr(util::is_same<BuildT, Point>::value) {
        static_assert(util::is_same<Vec3T, Vec3f, Vec3d>::value, "Point (vs voxels) coordinates should be represented as Vec3f or Vec3d");
    } else {
        static_assert(util::is_same<Vec3T, Coord, Vec3f, Vec3d>::value, "Voxel coordinates should be represented as Coord, Vec3f or Vec3d");
    }

    mMaxPointsPerVoxel = math::Min(mMaxPointsPerVoxel, pointCount);
    int iterCounter = 0;
    struct Foo {// pairs current voxel size, dx, with the corresponding particle density, i.e. maximum number of points per voxel
        double   dx;
        uint32_t density;
        bool operator<(const Foo &rhs) const {return density < rhs.density || (density == rhs.density && dx < rhs.dx);}
    } min{0.0, 1}, max{0.0, 0};// min: as dx -> 0 density -> 1 point per voxel, max: density is 0 i.e. undefined

jump:// this marks the beginning of the actual algorithm

    mData.d_keys = static_cast<uint64_t*>(ResourceT::allocateAsync(pointCount*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
    mData.d_indx = static_cast<uint32_t*>(ResourceT::allocateAsync(pointCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream));// uint32_t can index 4.29 billion Coords, corresponding to 48 GB
    cudaCheck(cudaMemcpyAsync(mDeviceData, &mData, sizeof(PointsToGridData<BuildT>), cudaMemcpyHostToDevice, mStream));// copy mData from CPU -> GPU

    if (mVerbose==2) mTimer.start("\nAllocating arrays for keys and indices");
    auto *d_keys = static_cast<uint64_t*>(ResourceT::allocateAsync(pointCount*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
    auto *d_indx = static_cast<uint32_t*>(ResourceT::allocateAsync(pointCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream));

    if (mVerbose==2) mTimer.restart("Generate tile keys");
    util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, TileKeyFunctor<BuildT, PtrT>(), mDeviceData, points, d_keys, d_indx);
    cudaCheckError();
    if (mVerbose==2) mTimer.restart("DeviceRadixSort of "+std::to_string(pointCount)+" tile keys");
    CALL_CUBS(DeviceRadixSort::SortPairs, d_keys, mData.d_keys, d_indx, mData.d_indx, pointCount, 0, 63);// 21 bits per coord
    std::swap(d_indx, mData.d_indx);// sorted indices are now in d_indx

    if (mVerbose==2) mTimer.restart("Allocate runs");
    auto *d_points_per_tile = static_cast<uint32_t*>(ResourceT::allocateAsync(pointCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
    uint32_t *d_node_count  = static_cast<uint32_t*>(ResourceT::allocateAsync(3*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream));

    if (mVerbose==2) mTimer.restart("DeviceRunLengthEncode tile keys");
    CALL_CUBS(DeviceRunLengthEncode::Encode, mData.d_keys, d_keys, d_points_per_tile, d_node_count+2, pointCount);
    cudaCheck(cudaMemcpyAsync(mData.nodeCount+2, d_node_count+2, sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    mData.d_tile_keys = static_cast<uint64_t*>(ResourceT::allocateAsync(mData.nodeCount[2]*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
    cudaCheck(cudaMemcpyAsync(mData.d_tile_keys, d_keys, mData.nodeCount[2]*sizeof(uint64_t), cudaMemcpyDeviceToDevice, mStream));

    if (mVerbose==2) mTimer.restart("DeviceRadixSort of " + std::to_string(pointCount) + " voxel keys in " + std::to_string(mData.nodeCount[2]) + " tiles");
    uint32_t *points_per_tile = new uint32_t[mData.nodeCount[2]];
    cudaCheck(cudaMemcpyAsync(points_per_tile, d_points_per_tile, mData.nodeCount[2]*sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
    ResourceT::deallocateAsync(d_points_per_tile, pointCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);

    for (uint32_t id = 0, offset = 0; id < mData.nodeCount[2]; ++id) {
        const uint32_t count = points_per_tile[id];
        util::cuda::offsetLambdaKernel<<<numBlocks(count), mNumThreads, 0, mStream>>>(count, offset, VoxelKeyFunctor<BuildT, PtrT>(), mDeviceData, points, id, d_keys, d_indx);
        cudaCheckError();
        CALL_CUBS(DeviceRadixSort::SortPairs, d_keys + offset, mData.d_keys + offset, d_indx + offset, mData.d_indx + offset, count, 0, 36);// 9+12+15=36
        offset += count;
    }
    ResourceT::deallocateAsync(d_indx, pointCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
    delete [] points_per_tile;

    if (mVerbose==2) mTimer.restart("Count points per voxel");

    cudaEvent_t copyEvent;
    cudaCheck(cudaEventCreate(&copyEvent));
    mData.pointsPerVoxel    = static_cast<uint32_t*>(ResourceT::allocateAsync(pointCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
    uint32_t *d_voxel_count = static_cast<uint32_t*>(ResourceT::allocateAsync(sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
    CALL_CUBS(DeviceRunLengthEncode::Encode, mData.d_keys, d_keys, mData.pointsPerVoxel, d_voxel_count, pointCount);
    cudaCheck(cudaMemcpyAsync(&mData.voxelCount, d_voxel_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaEventRecord(copyEvent, mStream));
    ResourceT::deallocateAsync(d_voxel_count, sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);

    if (util::is_same<BuildT, Point>::value) {
        if (mVerbose==2) mTimer.restart("Count max points per voxel");
        uint32_t *d_maxPointsPerVoxel = static_cast<uint32_t*>(ResourceT::allocateAsync(sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream)), maxPointsPerVoxel;
        cudaCheck(cudaEventSynchronize(copyEvent));
        CALL_CUBS(DeviceReduce::Max, mData.pointsPerVoxel, d_maxPointsPerVoxel, mData.voxelCount);
        cudaCheck(cudaMemcpyAsync(&maxPointsPerVoxel, d_maxPointsPerVoxel, sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
        cudaCheck(cudaEventRecord(copyEvent, mStream));
        ResourceT::deallocateAsync(d_maxPointsPerVoxel, sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
        double dx = mData.map.getVoxelSize()[0];
        cudaCheck(cudaEventSynchronize(copyEvent));
        if (++iterCounter >= mMaxIterations || pointCount == 1u || math::Abs((int)maxPointsPerVoxel - (int)mMaxPointsPerVoxel) <= mTolerance) {
            mMaxPointsPerVoxel = maxPointsPerVoxel;
        } else {
            const Foo tmp{dx, maxPointsPerVoxel};
            if (maxPointsPerVoxel < mMaxPointsPerVoxel) {
                if (min < tmp) min = tmp;
            } else if (max.density == 0 || tmp < max) {
                max = tmp;
            }
            if (max.density) {
                dx = (min.dx*(max.density - mMaxPointsPerVoxel) + max.dx*(mMaxPointsPerVoxel-min.density))/double(max.density-min.density);
            } else if (maxPointsPerVoxel > 1u) {
                dx *= (mMaxPointsPerVoxel-1.0)/(maxPointsPerVoxel-1.0);
            } else {// maxPointsPerVoxel = 1 so increase dx significantly
                dx *= 10.0;
            }
            if (mVerbose==2) printf("\ntarget density = %" PRIu32 ", current density = %" PRIu32 ", current dx = %f, next dx = %f\n", mMaxPointsPerVoxel, maxPointsPerVoxel, tmp.dx, dx);
            mData.map = Map(dx);
            ResourceT::deallocateAsync(mData.d_keys, pointCount*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
            ResourceT::deallocateAsync(mData.d_indx, pointCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
            ResourceT::deallocateAsync(d_keys, pointCount*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
            ResourceT::deallocateAsync(mData.d_tile_keys, mData.nodeCount[2]*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
            ResourceT::deallocateAsync(d_node_count, 3*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
            ResourceT::deallocateAsync(mData.pointsPerVoxel, pointCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
            goto jump;
        }
    }
    if (iterCounter>1 && mVerbose) std::cerr << "Used " << iterCounter << " attempts to determine dx that produces a target dpoint denisty\n\n";

    if (mVerbose==2) mTimer.restart("Compute prefix sum of points per voxel");
    cudaCheck(cudaEventSynchronize(copyEvent));
    mData.pointsPerVoxelPrefix = static_cast<uint32_t*>(ResourceT::allocateAsync(mData.voxelCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
    CALL_CUBS(DeviceScan::ExclusiveSum, mData.pointsPerVoxel, mData.pointsPerVoxelPrefix, mData.voxelCount);

    mData.pointsPerLeaf = static_cast<uint32_t*>(ResourceT::allocateAsync(pointCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
    CALL_CUBS(DeviceRunLengthEncode::Encode, thrust::make_transform_iterator(mData.d_keys, ShiftRight<9>()), d_keys, mData.pointsPerLeaf, d_node_count, pointCount);
    cudaCheck(cudaMemcpyAsync(mData.nodeCount, d_node_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaEventRecord(copyEvent, mStream));

    if constexpr(util::is_same<BuildT, Point>::value) {
        uint32_t *d_maxPointsPerLeaf = static_cast<uint32_t*>(ResourceT::allocateAsync(sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
        cudaCheck(cudaEventSynchronize(copyEvent));
        CALL_CUBS(DeviceReduce::Max, mData.pointsPerLeaf, d_maxPointsPerLeaf, mData.nodeCount[0]);
        cudaCheck(cudaMemcpyAsync(&mMaxPointsPerLeaf, d_maxPointsPerLeaf, sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
        //printf("\n Leaf count = %u, max points per leaf = %u\n", mData.nodeCount[0], mMaxPointsPerLeaf);
        if (mMaxPointsPerLeaf > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("Too many points per leaf: "+std::to_string(mMaxPointsPerLeaf));
        }
        ResourceT::deallocateAsync(d_maxPointsPerLeaf, sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
    }

    cudaCheck(cudaEventSynchronize(copyEvent));
    mData.pointsPerLeafPrefix = static_cast<uint32_t*>(ResourceT::allocateAsync(mData.nodeCount[0]*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
    CALL_CUBS(DeviceScan::ExclusiveSum, mData.pointsPerLeaf, mData.pointsPerLeafPrefix, mData.nodeCount[0]);

    cudaCheck(cudaStreamSynchronize(mStream));
    mData.d_leaf_keys = static_cast<uint64_t*>(ResourceT::allocateAsync(mData.nodeCount[0]*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
    cudaCheck(cudaMemcpyAsync(mData.d_leaf_keys, d_keys, mData.nodeCount[0]*sizeof(uint64_t), cudaMemcpyDeviceToDevice, mStream));

    CALL_CUBS(DeviceSelect::Unique, thrust::make_transform_iterator(mData.d_leaf_keys, ShiftRight<12>()), d_keys, d_node_count+1, mData.nodeCount[0]);// count lower nodes
    cudaCheck(cudaMemcpyAsync(mData.nodeCount+1, d_node_count+1, sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    mData.d_lower_keys = static_cast<uint64_t*>(ResourceT::allocateAsync(mData.nodeCount[1]*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
    cudaCheck(cudaMemcpyAsync(mData.d_lower_keys, d_keys, mData.nodeCount[1]*sizeof(uint64_t), cudaMemcpyDeviceToDevice, mStream));

    ResourceT::deallocateAsync(d_keys, pointCount*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
    ResourceT::deallocateAsync(d_node_count, 3*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
    if (mVerbose==2) mTimer.stop();
    cudaCheck(cudaEventDestroy(copyEvent));

    //printf("Leaf count = %u, lower count = %u, upper count = %u\n", mData.nodeCount[0], mData.nodeCount[1], mData.nodeCount[2]);
}// PointsToGrid<BuildT, ResourceT>::countNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
template<typename PtrT, typename BufferT>
inline BufferT PointsToGrid<BuildT, ResourceT>::getBuffer(const PtrT, size_t pointCount, const BufferT &pool)
{
    auto sizeofPoint = [&]()->size_t{
        switch (mPointType){
        case PointType::PointID: return sizeof(uint32_t);
        case PointType::World64: return sizeof(Vec3d);
        case PointType::World32: return sizeof(Vec3f);
        case PointType::Grid64:  return sizeof(Vec3d);
        case PointType::Grid32:  return sizeof(Vec3f);
        case PointType::Voxel32: return sizeof(Vec3f);
        case PointType::Voxel16: return sizeof(Vec3u16);
        case PointType::Voxel8:  return sizeof(Vec3u8);
        case PointType::Default: return pointer_traits<PtrT>::element_size;
        default: return size_t(0);// PointType::Disable
        }
    };

    mData.grid  = 0;// grid is always stored at the start of the buffer!
    mData.tree  = NanoGrid<BuildT>::memUsage(); // grid ends and tree begins
    mData.root  = mData.tree  + NanoTree<BuildT>::memUsage(); // tree ends and root node begins
    mData.upper = mData.root  + NanoRoot<BuildT>::memUsage(mData.nodeCount[2]); // root node ends and upper internal nodes begin
    mData.lower = mData.upper + NanoUpper<BuildT>::memUsage()*mData.nodeCount[2]; // upper internal nodes ends and lower internal nodes begin
    mData.leaf  = mData.lower + NanoLower<BuildT>::memUsage()*mData.nodeCount[1]; // lower internal nodes ends and leaf nodes begin
    mData.meta  = mData.leaf  + NanoLeaf<BuildT>::DataType::memUsage()*mData.nodeCount[0];// leaf nodes end and blind meta data begins
    mData.blind = mData.meta  + sizeof(GridBlindMetaData)*int( mPointType!=PointType::Disable ); // meta data ends and blind data begins
    mData.size  = mData.blind + pointCount*sizeofPoint();// end of buffer

    int device = 0;
    cudaGetDevice(&device);
    auto buffer = BufferT::create(mData.size, &pool, device, mStream);// only allocate buffer on the device

    mData.d_bufferPtr = buffer.deviceData();
    if (mData.d_bufferPtr == nullptr) throw std::runtime_error("Failed to allocate grid buffer on the device");
    cudaCheck(cudaMemcpyAsync(mDeviceData, &mData, sizeof(PointsToGridData<BuildT>), cudaMemcpyHostToDevice, mStream));// copy Data CPU -> GPU
    return buffer;
}// PointsToGrid<BuildT, ResourceT>::getBuffer

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT>
struct BuildGridTreeRootFunctor
{
    using Vec3T = typename util::remove_const<typename pointer_traits<PtrT>::element_type>::type;

    __device__
    void operator()(size_t, PointsToGridData<BuildT> *d_data, PointType pointType, size_t pointCount) {
       // process Root
        auto &root = d_data->getRoot();
        root.mBBox = CoordBBox(); // init to empty
        root.mTableSize = d_data->nodeCount[2];
        root.mBackground = typename NanoRoot<BuildT>::ValueType(0);// background_value
        root.mMinimum = root.mMaximum = typename NanoRoot<BuildT>::ValueType(0);
        root.mAverage = root.mStdDevi = typename NanoRoot<BuildT>::FloatType(0);

        // process Tree
        auto &tree = d_data->getTree();
        tree.setRoot(&root);
        tree.setFirstNode(&d_data->getUpper(0));
        tree.setFirstNode(&d_data->getLower(0));
        tree.setFirstNode(&d_data->getLeaf(0));
        tree.mNodeCount[2] = tree.mTileCount[2] = d_data->nodeCount[2];
        tree.mNodeCount[1] = tree.mTileCount[1] = d_data->nodeCount[1];
        tree.mNodeCount[0] = tree.mTileCount[0] = d_data->nodeCount[0];
        tree.mVoxelCount = d_data->voxelCount;

        // process Grid
        auto &grid = d_data->getGrid();
        grid.init({GridFlags::HasBBox, GridFlags::IsBreadthFirst}, d_data->size, d_data->map, toGridType<BuildT>());
        grid.mChecksum = ~uint64_t(0);// set all bits on which means it's disabled
        grid.mBlindMetadataCount  = util::is_same<BuildT, Point>::value;// ? 1u : 0u;
        grid.mBlindMetadataOffset = d_data->meta;
        if (pointType != PointType::Disable) {
            const auto lastLeaf = tree.mNodeCount[0] - 1;
            grid.mData1 = d_data->pointsPerLeafPrefix[lastLeaf] + d_data->pointsPerLeaf[lastLeaf];
            auto &meta = d_data->getMeta();
            meta.mDataOffset = sizeof(GridBlindMetaData);// blind data is placed right after this meta data
            meta.mValueCount = pointCount;
            // Blind meta data
            switch (pointType){
            case PointType::PointID:
                grid.mGridClass = GridClass::PointIndex;
                meta.mSemantic  = GridBlindDataSemantic::PointId;
                meta.mDataClass = GridBlindDataClass::IndexArray;
                meta.mDataType  = toGridType<uint32_t>();
                meta.mValueSize = sizeof(uint32_t);
                util::strcpy(meta.mName, "PointID: uint32_t indices to points");
                break;
            case PointType::World64:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::WorldCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3d>();
                meta.mValueSize = sizeof(Vec3d);
                util::strcpy(meta.mName, "World64: Vec3<double> point coordinates in world space");
                break;
            case PointType::World32:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::WorldCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3f>();
                meta.mValueSize = sizeof(Vec3f);
                util::strcpy(meta.mName, "World32: Vec3<float> point coordinates in world space");
                break;
            case PointType::Grid64:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::GridCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3d>();
                meta.mValueSize = sizeof(Vec3d);
                util::strcpy(meta.mName, "Grid64: Vec3<double> point coordinates in grid space");
                break;
            case PointType::Grid32:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::GridCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3f>();
                meta.mValueSize = sizeof(Vec3f);
                util::strcpy(meta.mName, "Grid32: Vec3<float> point coordinates in grid space");
                break;
            case PointType::Voxel32:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::VoxelCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3f>();
                meta.mValueSize = sizeof(Vec3f);
                util::strcpy(meta.mName, "Voxel32: Vec3<float> point coordinates in voxel space");
                break;
            case PointType::Voxel16:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::VoxelCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3u16>();
                meta.mValueSize = sizeof(Vec3u16);
                util::strcpy(meta.mName, "Voxel16: Vec3<uint16_t> point coordinates in voxel space");
                break;
            case PointType::Voxel8:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::VoxelCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3u8>();
                meta.mValueSize = sizeof(Vec3u8);
                util::strcpy(meta.mName, "Voxel8: Vec3<uint8_t> point coordinates in voxel space");
                break;
            case PointType::Default:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::WorldCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3T>();
                meta.mValueSize = sizeof(Vec3T);
                if constexpr(util::is_same<Vec3T, Vec3f>::value) {
                    util::strcpy(meta.mName, "World32: Vec3<float> point coordinates in world space");
                } else if constexpr(util::is_same<Vec3T, Vec3d>::value){
                    util::strcpy(meta.mName, "World64: Vec3<double> point coordinates in world space");
                } else {
                    printf("Error in PointsToGrid<BuildT, ResourceT>::processGridTreeRoot: expected Vec3T = Vec3f or Vec3d\n");
                }
                break;
            default:
                printf("Error in PointsToGrid<BuildT, ResourceT>::processGridTreeRoot: invalid pointType\n");
            }
        } else if constexpr(BuildTraits<BuildT>::is_offindex) {
            grid.mData1 = 1u + 512u*d_data->nodeCount[0];
            grid.mGridClass = GridClass::IndexGrid;
        }
    }
};

template<typename BuildT, typename ResourceT>
template<typename PtrT>
inline void PointsToGrid<BuildT, ResourceT>::processGridTreeRoot(const PtrT points, size_t pointCount)
{
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, BuildGridTreeRootFunctor<BuildT, PtrT>(), mDeviceData, mPointType, pointCount);// lambdaKernel
    cudaCheckError();

    char *dst = mData.getGrid().mGridName;
    if (const char *src = mGridName.data()) {
        cudaCheck(cudaMemcpyAsync(dst, src, GridData::MaxNameSize, cudaMemcpyHostToDevice, mStream));
    } else {
        cudaCheck(cudaMemsetAsync(dst, 0, GridData::MaxNameSize, mStream));
    }
}// PointsToGrid<BuildT, ResourceT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
struct BuildUpperNodesFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        auto &root  = d_data->getRoot();
        auto &upper = d_data->getUpper(tid);
#if 1
        auto keyToCoord = [](uint64_t key)->nanovdb::Coord{
            static constexpr int64_t offset = 1 << 31;// max values of uint32_t is 2^31 - 1
            static constexpr uint64_t MASK = (1u << 21) - 1; // used to mask out 21 lower bits
            return nanovdb::Coord(int(int64_t(((key >> 42) & MASK) << 12) - offset),  // x are the upper 21 bits
                                  int(int64_t(((key >> 21) & MASK) << 12) - offset),  // y are the middle 21 bits
                                  int(int64_t(( key        & MASK) << 12) - offset)); // z are the lower 21 bits
        };
        const Coord ijk = keyToCoord(d_data->d_tile_keys[tid]);
#else
        const Coord ijk = typename NanoRoot<uint32_t>::KeyToCoord(d_data->d_tile_keys[tid]);
#endif
        root.tile(tid)->setChild(ijk, &upper, &root);
        upper.mBBox[0] = ijk;
        upper.mFlags = 0;
        upper.mValueMask.setOff();
        upper.mChildMask.setOff();
        upper.mMinimum = upper.mMaximum = typename NanoLower<BuildT>::ValueType(0);
        upper.mAverage = upper.mStdDevi = typename NanoLower<BuildT>::FloatType(0);
    }
};

template<typename BuildT>
struct SetUpperBackgroundValuesFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        auto &upper = d_data->getUpper(tid >> 15);
        upper.mTable[tid & 32767u].value = typename NanoUpper<BuildT>::ValueType(0);// background
    }
};

template<typename BuildT, typename ResourceT>
inline void PointsToGrid<BuildT, ResourceT>::processUpperNodes()
{
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[2]), mNumThreads, 0, mStream>>>(mData.nodeCount[2], BuildUpperNodesFunctor<BuildT>(), mDeviceData);
    cudaCheckError();

    ResourceT::deallocateAsync(mData.d_tile_keys, mData.nodeCount[2]*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream);

    const uint64_t valueCount = mData.nodeCount[2] << 15;
    util::cuda::lambdaKernel<<<numBlocks(valueCount), mNumThreads, 0, mStream>>>(valueCount, SetUpperBackgroundValuesFunctor<BuildT>(), mDeviceData);
    cudaCheckError();
}// PointsToGrid<BuildT, ResourceT>::processUpperNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
struct BuildLowerNodesFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        auto &root  = d_data->getRoot();
        const uint64_t lowerKey = d_data->d_lower_keys[tid];
        auto &upper = d_data->getUpper(lowerKey >> 15);
        const uint32_t upperOffset = lowerKey & 32767u;// (1 << 15) - 1 = 32767
        upper.mChildMask.setOnAtomic(upperOffset);
        auto &lower = d_data->getLower(tid);
        upper.setChild(upperOffset, &lower);
        lower.mBBox[0] = upper.offsetToGlobalCoord(upperOffset);
        lower.mFlags = 0;
        lower.mValueMask.setOff();
        lower.mChildMask.setOff();
        lower.mMinimum = lower.mMaximum = typename NanoLower<BuildT>::ValueType(0);// background;
        lower.mAverage = lower.mStdDevi = typename NanoLower<BuildT>::FloatType(0);
    }
};

template<typename BuildT>
struct SetLowerBackgroundValuesFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        auto &lower = d_data->getLower(tid >> 12);
        lower.mTable[tid & 4095u].value = typename NanoLower<BuildT>::ValueType(0);// background
    }
};

template<typename BuildT, typename ResourceT>
inline void PointsToGrid<BuildT, ResourceT>::processLowerNodes()
{
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[1]), mNumThreads, 0, mStream>>>(mData.nodeCount[1], BuildLowerNodesFunctor<BuildT>(), mDeviceData);
    cudaCheckError();

    const uint64_t valueCount = mData.nodeCount[1] << 12;
    util::cuda::lambdaKernel<<<numBlocks(valueCount), mNumThreads, 0, mStream>>>(valueCount, SetLowerBackgroundValuesFunctor<BuildT>(), mDeviceData);
    cudaCheckError();
}// PointsToGrid<BuildT, ResourceT>::processLowerNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
struct ProcessLeafMetaDataFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data, uint8_t flags) {
        const uint64_t leafKey = d_data->d_leaf_keys[tid], tile_id = leafKey >> 27;
        auto &upper = d_data->getUpper(tile_id);
        const uint32_t lowerOffset = leafKey & 4095u, upperOffset = (leafKey >> 12) & 32767u;
        auto &lower = *upper.getChild(upperOffset);
        lower.mChildMask.setOnAtomic(lowerOffset);
        auto &leaf = d_data->getLeaf(tid);
        lower.setChild(lowerOffset, &leaf);
        leaf.mBBoxMin = lower.offsetToGlobalCoord(lowerOffset);
        leaf.mFlags = flags;
        auto &valueMask = leaf.mValueMask;
        valueMask.setOff();// initiate all bits to off

        if constexpr(util::is_same<Point, BuildT>::value) {
            leaf.mOffset = d_data->pointsPerLeafPrefix[tid];
            leaf.mPointCount = d_data->pointsPerLeaf[tid];
        } else if constexpr(BuildTraits<BuildT>::is_offindex) {
            leaf.mOffset = tid*512u + 1u;// background is index 0
            leaf.mPrefixSum = 0u;
        } else if constexpr(!BuildTraits<BuildT>::is_special) {
            leaf.mAverage = leaf.mStdDevi = typename NanoLeaf<BuildT>::FloatType(0);
            leaf.mMinimum = leaf.mMaximum = typename NanoLeaf<BuildT>::ValueType(0);
        }
    }
};

template<typename BuildT>
struct SetLeafActiveVoxelStateAndValuesFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        const uint32_t pointID  = d_data->pointsPerVoxelPrefix[tid];
        const uint64_t voxelKey = d_data->d_keys[pointID];
        auto &upper = d_data->getUpper(voxelKey >> 36);
        auto &lower = *upper.getChild((voxelKey >> 21) & 32767u);
        auto &leaf  = *lower.getChild((voxelKey >>  9) &  4095u);
        const uint32_t n = voxelKey & 511u;
        leaf.mValueMask.setOnAtomic(n);// <--- slow!
        if constexpr(util::is_same<Point, BuildT>::value) {
            leaf.mValues[n] = uint16_t(pointID + d_data->pointsPerVoxel[tid] - leaf.offset());
        } else if constexpr(!BuildTraits<BuildT>::is_special) {
            leaf.mValues[n] = typename NanoLeaf<BuildT>::ValueType(1);// set value of active voxels that are not points (or index)
        }
    }
};

template<typename BuildT>
struct SetLeafInactiveVoxelValuesFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        auto &leaf = d_data->getLeaf(tid >> 9u);
        const uint32_t n = tid & 511u;
        if (leaf.mValueMask.isOn(n)) return;
        if constexpr(util::is_same<BuildT, Point>::value) {
            const uint32_t m = leaf.mValueMask.findPrev<true>(n - 1);
            leaf.mValues[n] = m < 512u ? leaf.mValues[m] : 0u;
        } else if constexpr(!BuildTraits<BuildT>::is_special) {
            leaf.mValues[n] = typename NanoLeaf<BuildT>::ValueType(0);// value of inactive voxels
        }
    }
};

template<typename BuildT, typename ResourceT>
inline void PointsToGrid<BuildT, ResourceT>::processLeafNodes(size_t pointCount)
{
    const uint8_t flags = static_cast<uint8_t>(mData.flags.data());// mIncludeStats ? 16u : 0u;// 4th bit indicates stats

    if (mVerbose==2) mTimer.start("process leaf meta data");
    // loop over leaf nodes and add it to its parent node
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[0]), mNumThreads, 0, mStream>>>(mData.nodeCount[0], ProcessLeafMetaDataFunctor<BuildT>(), mDeviceData, flags);
    cudaCheckError();

    if (mVerbose==2) mTimer.restart("set active voxel state and values");
    // loop over all active voxels and set LeafNode::mValueMask and LeafNode::mValues
    util::cuda::lambdaKernel<<<numBlocks(mData.voxelCount), mNumThreads, 0, mStream>>>(mData.voxelCount, SetLeafActiveVoxelStateAndValuesFunctor<BuildT>(), mDeviceData);
    cudaCheckError();

    ResourceT::deallocateAsync(mData.d_keys, pointCount*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
    ResourceT::deallocateAsync(mData.pointsPerVoxel, pointCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
    ResourceT::deallocateAsync(mData.pointsPerVoxelPrefix, mData.voxelCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
    ResourceT::deallocateAsync(mData.pointsPerLeafPrefix, pointCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
    ResourceT::deallocateAsync(mData.pointsPerLeaf,mData.nodeCount[0]*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);

    if (mVerbose==2) mTimer.restart("set inactive voxel values");
    const uint64_t denseVoxelCount = mData.nodeCount[0] << 9;
    util::cuda::lambdaKernel<<<numBlocks(denseVoxelCount), mNumThreads, 0, mStream>>>(denseVoxelCount, SetLeafInactiveVoxelValuesFunctor<BuildT>(), mDeviceData);
    cudaCheckError();

    if constexpr(BuildTraits<BuildT>::is_onindex) {
        if (mVerbose==2) mTimer.restart("prefix-sum for index grid");
        auto devValueIndex = static_cast<uint64_t*>(ResourceT::allocateAsync(mData.nodeCount[0]*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
        auto devValueIndexPrefix = static_cast<uint64_t*>(ResourceT::allocateAsync(mData.nodeCount[0]*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream));
        kernels::fillValueIndexKernel<BuildT><<<numBlocks(mData.nodeCount[0]), mNumThreads, 0, mStream>>>(mData.nodeCount[0], 0, devValueIndex, mDeviceData);
        cudaCheckError();
        CALL_CUBS(DeviceScan::InclusiveSum, devValueIndex, devValueIndexPrefix, mData.nodeCount[0]);
        ResourceT::deallocateAsync(devValueIndex, mData.nodeCount[0]*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
        kernels::leafPrefixSumKernel<BuildT><<<numBlocks(mData.nodeCount[0]), mNumThreads, 0, mStream>>>(mData.nodeCount[0], 0, devValueIndexPrefix, mDeviceData);
        cudaCheckError();
        ResourceT::deallocateAsync(devValueIndexPrefix, mData.nodeCount[0]*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
    }

    if constexpr(BuildTraits<BuildT>::is_indexmask) {
        if (mVerbose==2) mTimer.restart("leaf.mMask = leaf.mValueMask");
        kernels::setMaskEqValMaskKernel<BuildT><<<numBlocks(mData.nodeCount[0]), mNumThreads, 0, mStream>>>(mData.nodeCount[0], 0, mDeviceData);
        cudaCheckError();
    }
    if (mVerbose==2) mTimer.stop();
}// PointsToGrid<BuildT, ResourceT>::processLeafNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
template<typename PtrT>
inline void PointsToGrid<BuildT, ResourceT>::processPoints(const PtrT, size_t pointCount)
{
    ResourceT::deallocateAsync(mData.d_indx, pointCount*sizeof(uint32_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Template specialization with BuildT = Point
template<>
template<typename PtrT>
inline void PointsToGrid<Point>::processPoints(const PtrT points, size_t pointCount)
{
    switch (mPointType){
    case PointType::Disable:
        throw std::runtime_error("PointsToGrid<Point>::processPoints: mPointType == PointType::Disable\n");
    case PointType::PointID:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            d_data->template getPoint<uint32_t>(tid) = d_data->d_indx[tid];
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::World64:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            d_data->template getPoint<Vec3d>(tid) = points[d_data->d_indx[tid]];
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::World32:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            d_data->template getPoint<Vec3f>(tid) = points[d_data->d_indx[tid]];
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Grid64:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            d_data->template getPoint<Vec3d>(tid) = d_data->map.applyInverseMap(points[d_data->d_indx[tid]]);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Grid32:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            d_data->template getPoint<Vec3f>(tid) = d_data->map.applyInverseMapF(points[d_data->d_indx[tid]]);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Voxel32:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            worldToVoxel(d_data->template getPoint<Vec3f>(tid), points[d_data->d_indx[tid]], d_data->map);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Voxel16:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            worldToVoxel(d_data->template getPoint<Vec3u16>(tid), points[d_data->d_indx[tid]], d_data->map);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Voxel8:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            worldToVoxel(d_data->template getPoint<Vec3u8>(tid), points[d_data->d_indx[tid]], d_data->map);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Default:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            d_data->template getPoint<typename pointer_traits<PtrT>::element_type>(tid) = points[d_data->d_indx[tid]];
        }, mDeviceData); cudaCheckError();
        break;
    default:
        printf("Internal error in PointsToGrid<Point>::processPoints\n");
    }
    nanovdb::cuda::DeviceResource::deallocateAsync(mData.d_indx, pointCount*sizeof(uint32_t), nanovdb::cuda::DeviceResource::DEFAULT_ALIGNMENT, mStream);
}// PointsToGrid<Point>::processPoints

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
struct ResetLowerNodeBBoxFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        d_data->getLower(tid).mBBox = CoordBBox();
    }
};

template<typename BuildT>
struct UpdateAndPropagateLeafBBoxFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        const uint64_t leafKey = d_data->d_leaf_keys[tid];
        auto &upper = d_data->getUpper(leafKey >> 27);
        auto &lower = *upper.getChild((leafKey >> 12) & 32767u);
        auto &leaf = d_data->getLeaf(tid);
        leaf.updateBBox();
        lower.mBBox.expandAtomic(leaf.bbox());
    }
};

template<typename BuildT>
struct ResetUpperNodeBBoxFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        d_data->getUpper(tid).mBBox = CoordBBox();
    }
};

template<typename BuildT>
struct PropagateLowerBBoxFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        const uint64_t lowerKey = d_data->d_lower_keys[tid];
        auto &upper = d_data->getUpper(lowerKey >> 15);
        auto &lower = d_data->getLower(tid);
        upper.mBBox.expandAtomic(lower.bbox());
    }
};

template<typename BuildT>
struct PropagateUpperBBoxFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        d_data->getRoot().mBBox.expandAtomic(d_data->getUpper(tid).bbox());
    }
};

template<typename BuildT>
struct UpdateRootWorldBBoxFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        d_data->getGrid().mWorldBBox = d_data->getRoot().mBBox.transform(d_data->map);
    }
};

template<typename BuildT, typename ResourceT>
inline void PointsToGrid<BuildT, ResourceT>::processBBox()
{
    if (mData.flags.isMaskOff(GridFlags::HasBBox)) {
        ResourceT::deallocateAsync(mData.d_leaf_keys, mData.nodeCount[0]*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
        ResourceT::deallocateAsync(mData.d_lower_keys, mData.nodeCount[1]*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
        return;
    }

    // reset bbox in lower nodes
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[1]), mNumThreads, 0, mStream>>>(mData.nodeCount[1], ResetLowerNodeBBoxFunctor<BuildT>(), mDeviceData);
    cudaCheckError();

    // update and propagate bbox from leaf -> lower/parent nodes
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[0]), mNumThreads, 0, mStream>>>(mData.nodeCount[0], UpdateAndPropagateLeafBBoxFunctor<BuildT>(), mDeviceData);
    ResourceT::deallocateAsync(mData.d_leaf_keys, mData.nodeCount[0]*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
    cudaCheckError();

    // reset bbox in upper nodes
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[2]), mNumThreads, 0, mStream>>>(mData.nodeCount[2], ResetUpperNodeBBoxFunctor<BuildT>(), mDeviceData);
    cudaCheckError();

    // propagate bbox from lower -> upper/parent node
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[1]), mNumThreads, 0, mStream>>>(mData.nodeCount[1], PropagateLowerBBoxFunctor<BuildT>(), mDeviceData);
    ResourceT::deallocateAsync(mData.d_lower_keys, mData.nodeCount[1]*sizeof(uint64_t), ResourceT::DEFAULT_ALIGNMENT, mStream);
    cudaCheckError()

    // propagate bbox from upper -> root/parent node
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[2]), mNumThreads, 0, mStream>>>(mData.nodeCount[2], PropagateUpperBBoxFunctor<BuildT>(), mDeviceData);
    cudaCheckError();

    // update the world-bbox in the root node
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, UpdateRootWorldBBoxFunctor<BuildT>(), mDeviceData);
    cudaCheckError();
}// PointsToGrid<BuildT, ResourceT>::processBBox

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT, typename ResourceT>
GridHandle<BufferT>// Grid<BuildT>
voxelsToGrid(const PtrT d_ijk, size_t voxelCount, double voxelSize, const BufferT &buffer, cudaStream_t stream)
{
    PointsToGrid<BuildT, ResourceT> converter(voxelSize, Vec3d(0.0), stream);
    return converter.getHandle(d_ijk, voxelCount, buffer);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename PtrT, typename BufferT, typename ResourceT>
GridHandle<BufferT>// Grid<Point> with PointType coordinates as blind data
pointsToGrid(const PtrT d_xyz, int pointCount, int maxPointsPerVoxel, int tolerance, int maxIterations, PointType type, const BufferT &buffer, cudaStream_t stream)
{
    PointsToGrid<Point, ResourceT> converter(maxPointsPerVoxel, tolerance, maxIterations, stream);
    converter.setPointType(type);
    return converter.getHandle(d_xyz, pointCount, buffer);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT, typename ResourceT>
GridHandle<BufferT>
pointsToGrid(std::vector<std::tuple<const PtrT,size_t,double,PointType>> vec, const BufferT &buffer, cudaStream_t stream)
{
    std::vector<GridHandle<BufferT>> handles;
    for (auto &p : vec) handles.push_back(pointsToGrid<BuildT, PtrT, BufferT, ResourceT>(std::get<0>(p), std::get<1>(p), std::get<2>(p), std::get<3>(p), buffer, stream));
    return mergeDeviceGrids(handles, stream);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT, typename ResourceT>
GridHandle<BufferT>
voxelsToGrid(std::vector<std::tuple<const PtrT,size_t,double>> vec, const BufferT &buffer, cudaStream_t stream)
{
    std::vector<GridHandle<BufferT>> handles;
    for (auto &p : vec) handles.push_back(voxelsToGrid<BuildT, PtrT, BufferT, ResourceT>(std::get<0>(p), std::get<1>(p), std::get<2>(p), buffer, stream));
    return mergeDeviceGrids(handles, stream);
}

}// namespace tools::cuda ======================================================================================================================================

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename PtrT, typename BufferT = cuda::DeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
[[deprecated("Use cuda::pointsToGrid instead")]]
GridHandle<BufferT>
cudaPointsToGrid(const PtrT dWorldPoints,
                 int pointCount,
                 double voxelSize = 1.0,
                 PointType type = PointType::Default,
                 const BufferT &buffer = BufferT(),
                 cudaStream_t stream = 0)
{
    return tools::cuda::pointsToGrid<PtrT, BufferT, ResourceT>(dWorldPoints, pointCount, voxelSize, type, buffer, stream);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT = cuda::DeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
[[deprecated("Use cuda::pointsToGrid instead")]]
GridHandle<BufferT>
cudaPointsToGrid(std::vector<std::tuple<const PtrT,size_t,double,PointType>> pointSet,
                 const BufferT &buffer = BufferT(),
                 cudaStream_t stream = 0)
{
    return tools::cuda::pointsToGrid<BuildT, PtrT, BufferT, ResourceT>(pointSet, buffer, stream);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT = cuda::DeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
[[deprecated("Use cuda::voxelsToGrid instead")]]
GridHandle<BufferT>
cudaVoxelsToGrid(const PtrT dGridVoxels,
                 size_t voxelCount,
                 double voxelSize = 1.0,
                 const BufferT &buffer = BufferT(),
                 cudaStream_t stream = 0)
{
    return tools::cuda::voxelsToGrid<BuildT, PtrT, BufferT, ResourceT>(dGridVoxels, voxelCount, voxelSize, buffer, stream);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT = cuda::DeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
[[deprecated("Use cuda::voxelsToGrid instead")]]
GridHandle<BufferT>
cudaVoxelsToGrid(std::vector<std::tuple<const PtrT, size_t, double>> pointSet,
                 const BufferT &buffer = BufferT(),
                 cudaStream_t stream = 0)
{
    return tools::cuda::voxelsToGrid<BuildT, PtrT, BufferT, ResourceT>(pointSet, buffer, stream);
}

}// namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_POINTSTOGRID_CUH_HAS_BEEN_INCLUDED
