// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/NanoVDB.h

    \author Ken Museth

    \date  January 8, 2020

    \brief Implements a light-weight self-contained VDB data-structure in a
           single file! In other words, this is a significantly watered-down
           version of the OpenVDB implementation, with few dependencies - so
           a one-stop-shop for a minimalistic VDB data structure that run on
           most platforms!

    \note It is important to note that NanoVDB (by design) is a read-only
          sparse GPU (and CPU) friendly data structure intended for applications
          like rendering and collision detection. As such it obviously lacks
          a lot of the functionality and features of OpenVDB grids. NanoVDB
          is essentially a compact linearized (or serialized) representation of
          an OpenVDB tree with getValue methods only. For best performance use
          the ReadAccessor::getValue method as opposed to the Tree::getValue
          method. Note that since a ReadAccessor caches previous access patterns
          it is by design not thread-safe, so use one instantiation per thread
          (it is very light-weight). Also, it is not safe to copy accessors between
          the GPU and CPU! In fact, client code should only interface
          with the API of the Grid class (all other nodes of the NanoVDB data
          structure can safely be ignored by most client codes)!


    \warning NanoVDB grids can only be constructed via tools like createNanoGrid
             or the GridBuilder. This explains why none of the grid nodes defined below
             have public constructors or destructors.

    \details Please see the following paper for more details on the data structure:
          K. Museth, “VDB: High-Resolution Sparse Volumes with Dynamic Topology”,
          ACM Transactions on Graphics 32(3), 2013, which can be found here:
          http://www.museth.org/Ken/Publications_files/Museth_TOG13.pdf

          NanoVDB was first published there: https://dl.acm.org/doi/fullHtml/10.1145/3450623.3464653


    Overview: This file implements the following fundamental class that when combined
          forms the backbone of the VDB tree data structure:

          Coord- a signed integer coordinate
          Vec3 - a 3D vector
          Vec4 - a 4D vector
          BBox - a bounding box
          Mask - a bitmask essential to the non-root tree nodes
          Map  - an affine coordinate transformation
          Grid - contains a Tree and a map for world<->index transformations. Use
                 this class as the main API with client code!
          Tree - contains a RootNode and getValue methods that should only be used for debugging
          RootNode - the top-level node of the VDB data structure
          InternalNode - the internal nodes of the VDB data structure
          LeafNode - the lowest level tree nodes that encode voxel values and state
          ReadAccessor - implements accelerated random access operations

    Semantics: A VDB data structure encodes values and (binary) states associated with
          signed integer coordinates. Values encoded at the leaf node level are
          denoted voxel values, and values associated with other tree nodes are referred
          to as tile values, which by design cover a larger coordinate index domain.


    Memory layout:

    It's important to emphasize that all the grid data (defined below) are explicitly 32 byte
    aligned, which implies that any memory buffer that contains a NanoVDB grid must also be at
    32 byte aligned. That is, the memory address of the beginning of a buffer (see ascii diagram below)
    must be divisible by 32, i.e. uintptr_t(&buffer)%32 == 0! If this is not the case, the C++ standard
    says the behaviour is undefined! Normally this is not a concerns on GPUs, because they use 256 byte
    aligned allocations, but the same cannot be said about the CPU.

    GridData is always at the very beginning of the buffer immediately followed by TreeData!
    The remaining nodes and blind-data are allowed to be scattered throughout the buffer,
    though in practice they are arranged as:

    GridData: 672 bytes (e.g. magic, checksum, major, flags, index, count, size, name, map, world bbox, voxel size, class, type, offset, count)

    TreeData: 64 bytes (node counts and byte offsets)

    ... optional padding ...

    RootData: size depends on ValueType (index bbox, voxel count, tile count, min/max/avg/standard deviation)

    Array of: RootData::Tile

    ... optional padding ...

    Array of: Upper InternalNodes of size 32^3:  bbox, two bit masks, 32768 tile values, and min/max/avg/standard deviation values

    ... optional padding ...

    Array of: Lower InternalNodes of size 16^3:  bbox, two bit masks, 4096 tile values, and min/max/avg/standard deviation values

    ... optional padding ...

    Array of: LeafNodes of size 8^3: bbox, bit masks, 512 voxel values, and min/max/avg/standard deviation values

    ... optional padding ...

    Array of: GridBlindMetaData (288 bytes). The offset and count are defined in GridData::mBlindMetadataOffset and GridData::mBlindMetadataCount

    ... optional padding ...

    Array of: blind data

    Notation: "]---[" implies it has optional padding, and "][" implies zero padding

    [GridData(672B)][TreeData(64B)]---[RootData][N x Root::Tile]---[InternalData<5>]---[InternalData<4>]---[LeafData<3>]---[BLINDMETA...]---[BLIND0]---[BLIND1]---etc.
    ^                                 ^         ^                  ^                   ^                   ^               ^
    |                                 |         |                  |                   |                   |               GridBlindMetaData*
    +-- Start of 32B aligned buffer   |         |                  |                   |                   +-- Node0::DataType* leafData
        GridType::DataType* gridData  |         |                  |                   |
                                      |         |                  |                   +-- Node1::DataType* lowerData
       RootType::DataType* rootData --+         |                  |
                                                |                  +-- Node2::DataType* upperData
                                                |
                                                +-- RootType::DataType::Tile* tile

*/

#ifndef NANOVDB_NANOVDB_H_HAS_BEEN_INCLUDED
#define NANOVDB_NANOVDB_H_HAS_BEEN_INCLUDED

// The following two header files are the only mandatory dependencies
#include <nanovdb/util/Util.h>// for __hostdev__ and lots of other utility functions
#include <nanovdb/math/Math.h>// for Coord, BBox, Vec3, Vec4 etc

// Do not change this value! 32 byte alignment is fixed in NanoVDB
#define NANOVDB_DATA_ALIGNMENT 32

// NANOVDB_MAGIC_NUMB previously used for both grids and files (starting with v32.6.0)
// NANOVDB_MAGIC_GRID currently used exclusively for grids (serialized to a single buffer)
// NANOVDB_MAGIC_FILE currently used exclusively for files
//                             | : 0 in 30 corresponds to 0 in NanoVDB0
#define NANOVDB_MAGIC_NUMB  0x304244566f6e614eUL // "NanoVDB0" in hex - little endian (uint64_t)
#define NANOVDB_MAGIC_GRID  0x314244566f6e614eUL // "NanoVDB1" in hex - little endian (uint64_t)
#define NANOVDB_MAGIC_FILE  0x324244566f6e614eUL // "NanoVDB2" in hex - little endian (uint64_t)
#define NANOVDB_MAGIC_MASK  0x00FFFFFFFFFFFFFFUL // use this mask to remove the number

#define NANOVDB_USE_NEW_MAGIC_NUMBERS// enables use of the new magic numbers described above

#define NANOVDB_MAJOR_VERSION_NUMBER 32 // reflects changes to the ABI and hence also the file format
#define NANOVDB_MINOR_VERSION_NUMBER 8 //  reflects changes to the API but not ABI
#define NANOVDB_PATCH_VERSION_NUMBER 0 //  reflects changes that does not affect the ABI or API

#define TBB_SUPPRESS_DEPRECATED_MESSAGES 1

// This replaces a Coord key at the root level with a single uint64_t
#define NANOVDB_USE_SINGLE_ROOT_KEY

// This replaces three levels of Coord keys in the ReadAccessor with one Coord
//#define NANOVDB_USE_SINGLE_ACCESSOR_KEY

// Use this to switch between std::ofstream or FILE implementations
//#define NANOVDB_USE_IOSTREAMS

#define NANOVDB_FPN_BRANCHLESS

#if !defined(NANOVDB_ALIGN)
#define NANOVDB_ALIGN(n) alignas(n)
#endif // !defined(NANOVDB_ALIGN)

namespace nanovdb {// =================================================================

// --------------------------> Build types <------------------------------------

/// @brief Dummy type for a voxel whose value equals an offset into an external value array
class ValueIndex{};

/// @brief Dummy type for a voxel whose value equals an offset into an external value array of active values
class ValueOnIndex{};

/// @brief Like @c ValueIndex but with a mutable mask
class ValueIndexMask{};

/// @brief Like @c ValueOnIndex but with a mutable mask
class ValueOnIndexMask{};

/// @brief Dummy type for a voxel whose value equals its binary active state
class ValueMask{};

/// @brief Dummy type for a 16 bit floating point values (placeholder for IEEE 754 Half)
class Half{};

/// @brief Dummy type for a 4bit quantization of float point values
class Fp4{};

/// @brief Dummy type for a 8bit quantization of float point values
class Fp8{};

/// @brief Dummy type for a 16bit quantization of float point values
class Fp16{};

/// @brief Dummy type for a variable bit quantization of floating point values
class FpN{};

/// @brief Dummy type for indexing points into voxels
class Point{};

// --------------------------> GridType <------------------------------------

/// @brief return the number of characters (including null termination) required to convert enum type to a string
///
/// @note This curious implementation, which subtracts End from StrLen, avoids duplicate values in the enum!
template <class EnumT>
__hostdev__ inline constexpr uint32_t strlen(){return (uint32_t)EnumT::StrLen - (uint32_t)EnumT::End;}

/// @brief List of types that are currently supported by NanoVDB
///
/// @note To expand on this list do:
///       1) Add the new type between Unknown and End in the enum below
///       2) Add the new type to OpenToNanoVDB::processGrid that maps OpenVDB types to GridType
///       3) Verify that the ConvertTrait in NanoToOpenVDB.h works correctly with the new type
///       4) Add the new type to toGridType (defined below) that maps NanoVDB types to GridType
///       5) Add the new type to toStr (defined below)
enum class GridType : uint32_t { Unknown = 0, //  unknown value type - should rarely be used
                                 Float = 1, //  single precision floating point value
                                 Double = 2, //  double precision floating point value
                                 Int16 = 3, //  half precision signed integer value
                                 Int32 = 4, //  single precision signed integer value
                                 Int64 = 5, //  double precision signed integer value
                                 Vec3f = 6, //  single precision floating 3D vector
                                 Vec3d = 7, //  double precision floating 3D vector
                                 Mask = 8, //  no value, just the active state
                                 Half = 9, //  half precision floating point value (placeholder for IEEE 754 Half)
                                 UInt32 = 10, // single precision unsigned integer value
                                 Boolean = 11, // boolean value, encoded in bit array
                                 RGBA8 = 12, // RGBA packed into 32bit word in reverse-order, i.e. R is lowest byte.
                                 Fp4 = 13, // 4bit quantization of floating point value
                                 Fp8 = 14, // 8bit quantization of floating point value
                                 Fp16 = 15, // 16bit quantization of floating point value
                                 FpN = 16, // variable bit quantization of floating point value
                                 Vec4f = 17, // single precision floating 4D vector
                                 Vec4d = 18, // double precision floating 4D vector
                                 Index = 19, // index into an external array of active and inactive values
                                 OnIndex = 20, // index into an external array of active values
                                 IndexMask = 21, // like Index but with a mutable mask
                                 OnIndexMask = 22, // like OnIndex but with a mutable mask
                                 PointIndex = 23, // voxels encode indices to co-located points
                                 Vec3u8 = 24, // 8bit quantization of floating point 3D vector (only as blind data)
                                 Vec3u16 = 25, // 16bit quantization of floating point 3D vector (only as blind data)
                                 UInt8 = 26, // 8 bit unsigned integer values (eg 0 -> 255 gray scale)
                                 End = 27,// total number of types in this enum (excluding StrLen since it's not a type)
                                 StrLen = End + 12};// this entry is used to determine the minimum size of c-string

/// @brief Maps a GridType to a c-string
/// @param dst destination string of size 12 or larger
/// @param gridType GridType enum to be mapped to a string
/// @return Retuns a c-string used to describe a GridType
__hostdev__ inline char* toStr(char *dst, GridType gridType)
{
    switch (gridType){
        case GridType::Unknown:     return util::strcpy(dst, "?");
        case GridType::Float:       return util::strcpy(dst, "float");
        case GridType::Double:      return util::strcpy(dst, "double");
        case GridType::Int16:       return util::strcpy(dst, "int16");
        case GridType::Int32:       return util::strcpy(dst, "int32");
        case GridType::Int64:       return util::strcpy(dst, "int64");
        case GridType::Vec3f:       return util::strcpy(dst, "Vec3f");
        case GridType::Vec3d:       return util::strcpy(dst, "Vec3d");
        case GridType::Mask:        return util::strcpy(dst, "Mask");
        case GridType::Half:        return util::strcpy(dst, "Half");
        case GridType::UInt32:      return util::strcpy(dst, "uint32");
        case GridType::Boolean:     return util::strcpy(dst, "bool");
        case GridType::RGBA8:       return util::strcpy(dst, "RGBA8");
        case GridType::Fp4:         return util::strcpy(dst, "Float4");
        case GridType::Fp8:         return util::strcpy(dst, "Float8");
        case GridType::Fp16:        return util::strcpy(dst, "Float16");
        case GridType::FpN:         return util::strcpy(dst, "FloatN");
        case GridType::Vec4f:       return util::strcpy(dst, "Vec4f");
        case GridType::Vec4d:       return util::strcpy(dst, "Vec4d");
        case GridType::Index:       return util::strcpy(dst, "Index");
        case GridType::OnIndex:     return util::strcpy(dst, "OnIndex");
        case GridType::IndexMask:   return util::strcpy(dst, "IndexMask");
        case GridType::OnIndexMask: return util::strcpy(dst, "OnIndexMask");// StrLen = 11 + 1 + End
        case GridType::PointIndex:  return util::strcpy(dst, "PointIndex");
        case GridType::Vec3u8:      return util::strcpy(dst, "Vec3u8");
        case GridType::Vec3u16:     return util::strcpy(dst, "Vec3u16");
        case GridType::UInt8:       return util::strcpy(dst, "uint8");
        default:                    return util::strcpy(dst, "End");
    }
}

// --------------------------> GridClass <------------------------------------

/// @brief Classes (superset of OpenVDB) that are currently supported by NanoVDB
enum class GridClass : uint32_t { Unknown = 0,
                                  LevelSet = 1, // narrow band level set, e.g. SDF
                                  FogVolume = 2, // fog volume, e.g. density
                                  Staggered = 3, // staggered MAC grid, e.g. velocity
                                  PointIndex = 4, // point index grid
                                  PointData = 5, // point data grid
                                  Topology = 6, // grid with active states only (no values)
                                  VoxelVolume = 7, // volume of geometric cubes, e.g. colors cubes in Minecraft
                                  IndexGrid = 8, // grid whose values are offsets, e.g. into an external array
                                  TensorGrid = 9, // Index grid for indexing learnable tensor features
                                  End = 10,// total number of types in this enum (excluding StrLen since it's not a type)
                                  StrLen = End + 7};// this entry is used to determine the minimum size of c-string


/// @brief Retuns a c-string used to describe a GridClass
/// @param dst destination string of size 7 or larger
/// @param gridClass GridClass enum to be converted to a string
__hostdev__ inline char* toStr(char *dst, GridClass gridClass)
{
    switch (gridClass){
        case GridClass::Unknown:     return util::strcpy(dst, "?");
        case GridClass::LevelSet:    return util::strcpy(dst, "SDF");
        case GridClass::FogVolume:   return util::strcpy(dst, "FOG");
        case GridClass::Staggered:   return util::strcpy(dst, "MAC");
        case GridClass::PointIndex:  return util::strcpy(dst, "PNTIDX");// StrLen = 6 + 1 + End
        case GridClass::PointData:   return util::strcpy(dst, "PNTDAT");
        case GridClass::Topology:    return util::strcpy(dst, "TOPO");
        case GridClass::VoxelVolume: return util::strcpy(dst, "VOX");
        case GridClass::IndexGrid:   return util::strcpy(dst, "INDEX");
        case GridClass::TensorGrid:  return util::strcpy(dst, "TENSOR");
        default:                     return util::strcpy(dst, "END");
    }
}

// --------------------------> GridFlags <------------------------------------

/// @brief Grid flags which indicate what extra information is present in the grid buffer.
enum class GridFlags : uint32_t {
    HasLongGridName = 1 << 0, // grid name is longer than 256 characters
    HasBBox = 1 << 1, // nodes contain bounding-boxes of active values
    HasMinMax = 1 << 2, // nodes contain min/max of active values
    HasAverage = 1 << 3, // nodes contain averages of active values
    HasStdDeviation = 1 << 4, // nodes contain standard deviations of active values
    IsBreadthFirst = 1 << 5, // nodes are typically arranged breadth-first in memory
    End = 1 << 6, // use End - 1 as a mask for the 5 lower bit flags
    StrLen = End + 23,// this entry is used to determine the minimum size of c-string
};

/// @brief Retuns a c-string used to describe a GridFlags
/// @param dst destination string of size 23 or larger
/// @param gridFlags GridFlags enum to be converted to a string
__hostdev__ inline const char* toStr(char *dst, GridFlags gridFlags)
{
    switch (gridFlags){
        case GridFlags::HasLongGridName: return util::strcpy(dst, "has long grid name");
        case GridFlags::HasBBox:         return util::strcpy(dst, "has bbox");
        case GridFlags::HasMinMax:       return util::strcpy(dst, "has min/max");
        case GridFlags::HasAverage:      return util::strcpy(dst, "has average");
        case GridFlags::HasStdDeviation: return util::strcpy(dst, "has standard deviation");// StrLen = 22 + 1 + End
        case GridFlags::IsBreadthFirst:  return util::strcpy(dst, "is breadth-first");
        default:                         return util::strcpy(dst, "end");
    }
}

// --------------------------> MagicType <------------------------------------

/// @brief Enums used to identify magic numbers recognized by NanoVDB
enum class MagicType : uint32_t { Unknown  = 0,// first 64 bits are neither of the cases below
                                  OpenVDB  = 1,// first 32 bits = 0x56444220UL
                                  NanoVDB  = 2,// first 64 bits = NANOVDB_MAGIC_NUMB
                                  NanoGrid = 3,// first 64 bits = NANOVDB_MAGIC_GRID
                                  NanoFile = 4,// first 64 bits = NANOVDB_MAGIC_FILE
                                  End      = 5,
                                  StrLen   = End + 14};// this entry is used to determine the minimum size of c-string

/// @brief maps 64 bits of magic number to enum
__hostdev__ inline MagicType toMagic(uint64_t magic)
{
    switch (magic){
        case NANOVDB_MAGIC_NUMB:   return MagicType::NanoVDB;
        case NANOVDB_MAGIC_GRID:   return MagicType::NanoGrid;
        case NANOVDB_MAGIC_FILE:   return MagicType::NanoFile;
        default: return (magic & ~uint32_t(0)) == 0x56444220UL ? MagicType::OpenVDB : MagicType::Unknown;
    }
}

/// @brief print 64-bit magic number to string
/// @param dst destination string of size 25 or larger
/// @param magic 64 bit magic number to be printed
/// @return return destination string @c dst
__hostdev__ inline char* toStr(char *dst, MagicType magic)
{
    switch (magic){
        case MagicType::Unknown:  return util::strcpy(dst, "unknown");
        case MagicType::NanoVDB:  return util::strcpy(dst, "nanovdb");
        case MagicType::NanoGrid: return util::strcpy(dst, "nanovdb::Grid");// StrLen = 13 + 1 + End
        case MagicType::NanoFile: return util::strcpy(dst, "nanovdb::File");
        case MagicType::OpenVDB:  return util::strcpy(dst, "openvdb");
        default:                  return util::strcpy(dst, "end");
    }
}

// --------------------------> PointType enums <------------------------------------

// Define the type used when the points are encoded as blind data in the output grid
enum class PointType : uint32_t { Disable = 0,// no point information e.g. when BuildT != Point
                                  PointID = 1,// linear index of type uint32_t to points
                                  World64 = 2,// Vec3d in world space
                                  World32 = 3,// Vec3f in world space
                                  Grid64  = 4,// Vec3d in grid space
                                  Grid32  = 5,// Vec3f in grid space
                                  Voxel32 = 6,// Vec3f in voxel space
                                  Voxel16 = 7,// Vec3u16 in voxel space
                                  Voxel8  = 8,// Vec3u8 in voxel space
                                  Default = 9,// output matches input, i.e. Vec3d or Vec3f in world space
                                  End     =10 };

// --------------------------> GridBlindData enums <------------------------------------

/// @brief Blind-data Classes that are currently supported by NanoVDB
enum class GridBlindDataClass : uint32_t { Unknown = 0,
                                           IndexArray = 1,
                                           AttributeArray = 2,
                                           GridName = 3,
                                           ChannelArray = 4,
                                           End = 5 };

/// @brief Blind-data Semantics that are currently understood by NanoVDB
enum class GridBlindDataSemantic : uint32_t { Unknown = 0,
                                              PointPosition = 1, // 3D coordinates in an unknown space
                                              PointColor = 2,
                                              PointNormal = 3,
                                              PointRadius = 4,
                                              PointVelocity = 5,
                                              PointId = 6,
                                              WorldCoords = 7, // 3D coordinates in world space, e.g. (0.056, 0.8, 1,8)
                                              GridCoords = 8, // 3D coordinates in grid space, e.g. (1.2, 4.0, 5.7), aka index-space
                                              VoxelCoords = 9, // 3D coordinates in voxel space, e.g. (0.2, 0.0, 0.7)
                                              End = 10 };

// --------------------------> BuildTraits <------------------------------------

/// @brief Define static boolean tests for template build types
template<typename T>
struct BuildTraits
{
    // check if T is an index type
    static constexpr bool is_index     = util::is_same<T, ValueIndex, ValueIndexMask, ValueOnIndex, ValueOnIndexMask>::value;
    static constexpr bool is_onindex   = util::is_same<T, ValueOnIndex, ValueOnIndexMask>::value;
    static constexpr bool is_offindex  = util::is_same<T, ValueIndex, ValueIndexMask>::value;
    static constexpr bool is_indexmask = util::is_same<T, ValueIndexMask, ValueOnIndexMask>::value;
    // check if T is a compressed float type with fixed bit precision
    static constexpr bool is_FpX = util::is_same<T, Fp4, Fp8, Fp16>::value;
    // check if T is a compressed float type with fixed or variable bit precision
    static constexpr bool is_Fp = util::is_same<T, Fp4, Fp8, Fp16, FpN>::value;
    // check if T is a POD float type, i.e float or double
    static constexpr bool is_float = util::is_floating_point<T>::value;
    // check if T is a template specialization of LeafData<T>, i.e. has T mValues[512]
    static constexpr bool is_special = is_index || is_Fp || util::is_same<T, Point, bool, ValueMask>::value;
}; // BuildTraits

// --------------------------> BuildToValueMap <------------------------------------

/// @brief Maps one type (e.g. the build types above) to other (actual) types
template<typename T>
struct BuildToValueMap
{
    using Type = T;
    using type = T;
};

template<>
struct BuildToValueMap<ValueIndex>
{
    using Type = uint64_t;
    using type = uint64_t;
};

template<>
struct BuildToValueMap<ValueOnIndex>
{
    using Type = uint64_t;
    using type = uint64_t;
};

template<>
struct BuildToValueMap<ValueIndexMask>
{
    using Type = uint64_t;
    using type = uint64_t;
};

template<>
struct BuildToValueMap<ValueOnIndexMask>
{
    using Type = uint64_t;
    using type = uint64_t;
};

template<>
struct BuildToValueMap<ValueMask>
{
    using Type = bool;
    using type = bool;
};

template<>
struct BuildToValueMap<Half>
{
    using Type = float;
    using type = float;
};

template<>
struct BuildToValueMap<Fp4>
{
    using Type = float;
    using type = float;
};

template<>
struct BuildToValueMap<Fp8>
{
    using Type = float;
    using type = float;
};

template<>
struct BuildToValueMap<Fp16>
{
    using Type = float;
    using type = float;
};

template<>
struct BuildToValueMap<FpN>
{
    using Type = float;
    using type = float;
};

template<>
struct BuildToValueMap<Point>
{
    using Type = uint64_t;
    using type = uint64_t;
};

// --------------------------> utility functions related to alignment <------------------------------------

/// @brief return true if the specified pointer is 32 byte aligned
__hostdev__ inline static bool isAligned(const void* p){return uint64_t(p) % NANOVDB_DATA_ALIGNMENT == 0;}

/// @brief return the smallest number of bytes that when added to the specified pointer results in a 32 byte aligned pointer.
__hostdev__ inline static uint64_t alignmentPadding(const void* p)
{
    NANOVDB_ASSERT(p);
    return (NANOVDB_DATA_ALIGNMENT - (uint64_t(p) % NANOVDB_DATA_ALIGNMENT)) % NANOVDB_DATA_ALIGNMENT;
}

/// @brief offset the specified pointer so it is 32 byte aligned. Works with both const and non-const pointers.
template <typename T>
__hostdev__ inline static T* alignPtr(T* p){return util::PtrAdd<T>(p, alignmentPadding(p));}

// --------------------------> isFloatingPoint(GridType) <------------------------------------

/// @brief return true if the GridType maps to a floating point type
__hostdev__ inline bool isFloatingPoint(GridType gridType)
{
    return gridType == GridType::Float ||
           gridType == GridType::Double ||
           gridType == GridType::Half ||
           gridType == GridType::Fp4 ||
           gridType == GridType::Fp8 ||
           gridType == GridType::Fp16 ||
           gridType == GridType::FpN;
}

// --------------------------> isFloatingPointVector(GridType) <------------------------------------

/// @brief return true if the GridType maps to a floating point vec3.
__hostdev__ inline bool isFloatingPointVector(GridType gridType)
{
    return gridType == GridType::Vec3f ||
           gridType == GridType::Vec3d ||
           gridType == GridType::Vec4f ||
           gridType == GridType::Vec4d;
}

// --------------------------> isInteger(GridType) <------------------------------------

/// @brief Return true if the GridType maps to a POD integer type.
/// @details These types are used to associate a voxel with a POD integer type
__hostdev__ inline bool isInteger(GridType gridType)
{
    return gridType == GridType::Int16 ||
           gridType == GridType::Int32 ||
           gridType == GridType::Int64 ||
           gridType == GridType::UInt32||
           gridType == GridType::UInt8;
}

// --------------------------> isIndex(GridType) <------------------------------------

/// @brief Return true if the GridType maps to a special index type (not a POD integer type).
/// @details These types are used to index from a voxel into an external array of values, e.g. sidecar or blind data.
__hostdev__ inline bool isIndex(GridType gridType)
{
    return gridType == GridType::Index ||// index both active and inactive values
           gridType == GridType::OnIndex ||// index active values only
           gridType == GridType::IndexMask ||// as Index, but with an additional mask
           gridType == GridType::OnIndexMask;// as OnIndex, but with an additional mask
}

// --------------------------> isValue(GridType, GridClass) <------------------------------------

/// @brief return true if the combination of GridType and GridClass is valid.
__hostdev__ inline bool isValid(GridType gridType, GridClass gridClass)
{
    if (gridClass == GridClass::LevelSet || gridClass == GridClass::FogVolume) {
        return isFloatingPoint(gridType);
    } else if (gridClass == GridClass::Staggered) {
        return isFloatingPointVector(gridType);
    } else if (gridClass == GridClass::PointIndex || gridClass == GridClass::PointData) {
        return gridType == GridType::PointIndex || gridType == GridType::UInt32;
    } else if (gridClass == GridClass::Topology) {
        return gridType == GridType::Mask;
    } else if (gridClass == GridClass::IndexGrid) {
        return isIndex(gridType);
    } else if (gridClass == GridClass::VoxelVolume) {
        return gridType == GridType::RGBA8 || gridType == GridType::Float ||
               gridType == GridType::Double || gridType == GridType::Vec3f ||
               gridType == GridType::Vec3d || gridType == GridType::UInt32 ||
               gridType == GridType::UInt8;
    }
    return gridClass < GridClass::End && gridType < GridType::End; // any valid combination
}

// --------------------------> validation of blind data meta data <------------------------------------

/// @brief return true if the combination of GridBlindDataClass, GridBlindDataSemantic and GridType is valid.
__hostdev__ inline bool isValid(const GridBlindDataClass&    blindClass,
                                const GridBlindDataSemantic& blindSemantics,
                                const GridType&              blindType)
{
    bool test = false;
    switch (blindClass) {
    case GridBlindDataClass::IndexArray:
        test = (blindSemantics == GridBlindDataSemantic::Unknown ||
                blindSemantics == GridBlindDataSemantic::PointId) &&
               isInteger(blindType);
        break;
    case GridBlindDataClass::AttributeArray:
        if (blindSemantics == GridBlindDataSemantic::PointPosition ||
            blindSemantics == GridBlindDataSemantic::WorldCoords) {
            test = blindType == GridType::Vec3f || blindType == GridType::Vec3d;
        } else if (blindSemantics == GridBlindDataSemantic::GridCoords) {
            test = blindType == GridType::Vec3f;
        } else if (blindSemantics == GridBlindDataSemantic::VoxelCoords) {
            test = blindType == GridType::Vec3f || blindType == GridType::Vec3u8 || blindType == GridType::Vec3u16;
        } else {
            test = blindSemantics != GridBlindDataSemantic::PointId;
        }
        break;
    case GridBlindDataClass::GridName:
        test = blindSemantics == GridBlindDataSemantic::Unknown && blindType == GridType::Unknown;
        break;
    default: // captures blindClass == Unknown and ChannelArray
        test = blindClass < GridBlindDataClass::End &&
               blindSemantics < GridBlindDataSemantic::End &&
               blindType < GridType::End; // any valid combination
        break;
    }
    //if (!test) printf("Invalid combination: GridBlindDataClass=%u, GridBlindDataSemantic=%u, GridType=%u\n",(uint32_t)blindClass, (uint32_t)blindSemantics, (uint32_t)blindType);
    return test;
}

// ----------------------------> Version class <-------------------------------------

/// @brief Bit-compacted representation of all three version numbers
///
/// @details major is the top 11 bits, minor is the 11 middle bits and patch is the lower 10 bits
class Version
{
    uint32_t mData; // 11 + 11 + 10 bit packing of major + minor + patch
public:
    static constexpr uint32_t End = 0, StrLen = 8;// for strlen<Version>()
    /// @brief Default constructor
    __hostdev__ Version()
        : mData(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER) << 21 |
                uint32_t(NANOVDB_MINOR_VERSION_NUMBER) << 10 |
                uint32_t(NANOVDB_PATCH_VERSION_NUMBER))
    {
    }
    /// @brief Constructor from a raw uint32_t data representation
    __hostdev__ Version(uint32_t data) : mData(data) {}
    /// @brief Constructor from major.minor.patch version numbers
    __hostdev__ Version(uint32_t major, uint32_t minor, uint32_t patch)
        : mData(major << 21 | minor << 10 | patch)
    {
        NANOVDB_ASSERT(major < (1u << 11)); // max value of major is 2047
        NANOVDB_ASSERT(minor < (1u << 11)); // max value of minor is 2047
        NANOVDB_ASSERT(patch < (1u << 10)); // max value of patch is 1023
    }
    __hostdev__ bool     operator==(const Version& rhs) const { return mData == rhs.mData; }
    __hostdev__ bool     operator<( const Version& rhs) const { return mData < rhs.mData; }
    __hostdev__ bool     operator<=(const Version& rhs) const { return mData <= rhs.mData; }
    __hostdev__ bool     operator>( const Version& rhs) const { return mData > rhs.mData; }
    __hostdev__ bool     operator>=(const Version& rhs) const { return mData >= rhs.mData; }
    __hostdev__ uint32_t id() const { return mData; }
    __hostdev__ uint32_t getMajor() const { return (mData >> 21) & ((1u << 11) - 1); }
    __hostdev__ uint32_t getMinor() const { return (mData >> 10) & ((1u << 11) - 1); }
    __hostdev__ uint32_t getPatch() const { return  mData        & ((1u << 10) - 1); }
    __hostdev__ bool isCompatible() const { return this->getMajor() == uint32_t(NANOVDB_MAJOR_VERSION_NUMBER); }
    /// @brief Returns the difference between major version of this instance and NANOVDB_MAJOR_VERSION_NUMBER
    /// @return return 0 if the major version equals NANOVDB_MAJOR_VERSION_NUMBER, else a negative age if this
    ///         instance has a smaller major verion (is older), and a positive age if it is newer, i.e. larger.
    __hostdev__ int age() const {return int(this->getMajor()) - int(NANOVDB_MAJOR_VERSION_NUMBER);}
}; // Version

/// @brief print the verion number to a c-string
/// @param dst destination string of size 8 or more
/// @param v version to be printed
/// @return returns destination string @c dst
__hostdev__ inline char* toStr(char *dst, const Version &v)
{
    return util::sprint(dst, v.getMajor(), ".",v.getMinor(), ".",v.getPatch());
}

// ----------------------------> TensorTraits <--------------------------------------

template<typename T, int Rank = (util::is_specialization<T, math::Vec3>::value || util::is_specialization<T, math::Vec4>::value || util::is_same<T, math::Rgba8>::value) ? 1 : 0>
struct TensorTraits;

template<typename T>
struct TensorTraits<T, 0>
{
    static const int  Rank = 0; // i.e. scalar
    static const bool IsScalar = true;
    static const bool IsVector = false;
    static const int  Size = 1;
    using ElementType = T;
    static T scalar(const T& s) { return s; }
};

template<typename T>
struct TensorTraits<T, 1>
{
    static const int  Rank = 1; // i.e. vector
    static const bool IsScalar = false;
    static const bool IsVector = true;
    static const int  Size = T::SIZE;
    using ElementType = typename T::ValueType;
    static ElementType scalar(const T& v) { return v.length(); }
};

// ----------------------------> FloatTraits <--------------------------------------

template<typename T, int = sizeof(typename TensorTraits<T>::ElementType)>
struct FloatTraits
{
    using FloatType = float;
};

template<typename T>
struct FloatTraits<T, 8>
{
    using FloatType = double;
};

template<>
struct FloatTraits<bool, 1>
{
    using FloatType = bool;
};

template<>
struct FloatTraits<ValueIndex, 1> // size of empty class in C++ is 1 byte and not 0 byte
{
    using FloatType = uint64_t;
};

template<>
struct FloatTraits<ValueIndexMask, 1> // size of empty class in C++ is 1 byte and not 0 byte
{
    using FloatType = uint64_t;
};

template<>
struct FloatTraits<ValueOnIndex, 1> // size of empty class in C++ is 1 byte and not 0 byte
{
    using FloatType = uint64_t;
};

template<>
struct FloatTraits<ValueOnIndexMask, 1> // size of empty class in C++ is 1 byte and not 0 byte
{
    using FloatType = uint64_t;
};

template<>
struct FloatTraits<ValueMask, 1> // size of empty class in C++ is 1 byte and not 0 byte
{
    using FloatType = bool;
};

template<>
struct FloatTraits<Point, 1> // size of empty class in C++ is 1 byte and not 0 byte
{
    using FloatType = double;
};

// ----------------------------> mapping BuildType -> GridType <--------------------------------------

/// @brief Maps from a templated build type to a GridType enum
template<typename BuildT>
__hostdev__ inline GridType toGridType()
{
    if constexpr(util::is_same<BuildT, float>::value) { // resolved at compile-time
        return GridType::Float;
    } else if constexpr(util::is_same<BuildT, double>::value) {
        return GridType::Double;
    } else if constexpr(util::is_same<BuildT, int16_t>::value) {
        return GridType::Int16;
    } else if constexpr(util::is_same<BuildT, int32_t>::value) {
        return GridType::Int32;
    } else if constexpr(util::is_same<BuildT, int64_t>::value) {
        return GridType::Int64;
    } else if constexpr(util::is_same<BuildT, Vec3f>::value) {
        return GridType::Vec3f;
    } else if constexpr(util::is_same<BuildT, Vec3d>::value) {
        return GridType::Vec3d;
    } else if constexpr(util::is_same<BuildT, uint32_t>::value) {
        return GridType::UInt32;
    } else if constexpr(util::is_same<BuildT, ValueMask>::value) {
        return GridType::Mask;
    } else if constexpr(util::is_same<BuildT, Half>::value) {
        return GridType::Half;
    } else if constexpr(util::is_same<BuildT, ValueIndex>::value) {
        return GridType::Index;
    } else if constexpr(util::is_same<BuildT, ValueOnIndex>::value) {
        return GridType::OnIndex;
    } else if constexpr(util::is_same<BuildT, ValueIndexMask>::value) {
        return GridType::IndexMask;
    } else if constexpr(util::is_same<BuildT, ValueOnIndexMask>::value) {
        return GridType::OnIndexMask;
    } else if constexpr(util::is_same<BuildT, bool>::value) {
        return GridType::Boolean;
    } else if constexpr(util::is_same<BuildT, math::Rgba8>::value) {
        return GridType::RGBA8;
    } else if constexpr(util::is_same<BuildT, Fp4>::value) {
        return GridType::Fp4;
    } else if constexpr(util::is_same<BuildT, Fp8>::value) {
        return GridType::Fp8;
    } else if constexpr(util::is_same<BuildT, Fp16>::value) {
        return GridType::Fp16;
    } else if constexpr(util::is_same<BuildT, FpN>::value) {
        return GridType::FpN;
    } else if constexpr(util::is_same<BuildT, Vec4f>::value) {
        return GridType::Vec4f;
    } else if constexpr(util::is_same<BuildT, Vec4d>::value) {
        return GridType::Vec4d;
    } else if constexpr(util::is_same<BuildT, Point>::value) {
        return GridType::PointIndex;
    } else if constexpr(util::is_same<BuildT, Vec3u8>::value) {
        return GridType::Vec3u8;
    } else if constexpr(util::is_same<BuildT, Vec3u16>::value) {
        return GridType::Vec3u16;
    } else if constexpr(util::is_same<BuildT, uint8_t>::value) {
        return GridType::UInt8;
    }
    return GridType::Unknown;
}// toGridType

template<typename BuildT>
[[deprecated("Use toGridType<T>() instead.")]]
__hostdev__ inline GridType mapToGridType(){return toGridType<BuildT>();}

// ----------------------------> mapping BuildType -> GridClass <--------------------------------------

/// @brief Maps from a templated build type to a GridClass enum
template<typename BuildT>
__hostdev__ inline GridClass toGridClass(GridClass defaultClass = GridClass::Unknown)
{
    if constexpr(util::is_same<BuildT, ValueMask>::value) {
        return GridClass::Topology;
    } else if constexpr(BuildTraits<BuildT>::is_index) {
        return GridClass::IndexGrid;
    } else if constexpr(util::is_same<BuildT, math::Rgba8>::value) {
        return GridClass::VoxelVolume;
    } else if constexpr(util::is_same<BuildT, Point>::value) {
        return GridClass::PointIndex;
    }
    return defaultClass;
}

template<typename BuildT>
[[deprecated("Use toGridClass<T>() instead.")]]
__hostdev__ inline GridClass mapToGridClass(GridClass defaultClass = GridClass::Unknown)
{
    return toGridClass<BuildT>();
}

//  ----------------------------> BitFlags <--------------------------------------

template<int N>
struct BitArray;
template<>
struct BitArray<8>
{
    uint8_t mFlags{0};
};
template<>
struct BitArray<16>
{
    uint16_t mFlags{0};
};
template<>
struct BitArray<32>
{
    uint32_t mFlags{0};
};
template<>
struct BitArray<64>
{
    uint64_t mFlags{0};
};

template<int N>
class BitFlags : public BitArray<N>
{
protected:
    using BitArray<N>::mFlags;

public:
    using Type = decltype(mFlags);
    BitFlags() {}
    BitFlags(Type mask) : BitArray<N>{mask} {}
    BitFlags(std::initializer_list<uint8_t> list)
    {
        for (auto bit : list) mFlags |= static_cast<Type>(1 << bit);
    }
    template<typename MaskT>
    BitFlags(std::initializer_list<MaskT> list)
    {
        for (auto mask : list) mFlags |= static_cast<Type>(mask);
    }
    __hostdev__ Type  data() const { return mFlags; }
    __hostdev__ Type& data() { return mFlags; }
    __hostdev__ void  initBit(std::initializer_list<uint8_t> list)
    {
        mFlags = 0u;
        for (auto bit : list) mFlags |= static_cast<Type>(1 << bit);
    }
    template<typename MaskT>
    __hostdev__ void initMask(std::initializer_list<MaskT> list)
    {
        mFlags = 0u;
        for (auto mask : list) mFlags |= static_cast<Type>(mask);
    }
    __hostdev__ Type getFlags() const { return mFlags & (static_cast<Type>(GridFlags::End) - 1u); } // mask out everything except relevant bits

    __hostdev__ void setOn() { mFlags = ~Type(0u); }
    __hostdev__ void setOff() { mFlags = Type(0u); }

    __hostdev__ void setBitOn(uint8_t bit) { mFlags |= static_cast<Type>(1 << bit); }
    __hostdev__ void setBitOff(uint8_t bit) { mFlags &= ~static_cast<Type>(1 << bit); }

    __hostdev__ void setBitOn(std::initializer_list<uint8_t> list)
    {
        for (auto bit : list) mFlags |= static_cast<Type>(1 << bit);
    }
    __hostdev__ void setBitOff(std::initializer_list<uint8_t> list)
    {
        for (auto bit : list) mFlags &= ~static_cast<Type>(1 << bit);
    }

    template<typename MaskT>
    __hostdev__ void setMaskOn(MaskT mask) { mFlags |= static_cast<Type>(mask); }
    template<typename MaskT>
    __hostdev__ void setMaskOff(MaskT mask) { mFlags &= ~static_cast<Type>(mask); }

    template<typename MaskT>
    __hostdev__ void setMaskOn(std::initializer_list<MaskT> list)
    {
        for (auto mask : list) mFlags |= static_cast<Type>(mask);
    }
    template<typename MaskT>
    __hostdev__ void setMaskOff(std::initializer_list<MaskT> list)
    {
        for (auto mask : list) mFlags &= ~static_cast<Type>(mask);
    }

    __hostdev__ void setBit(uint8_t bit, bool on) { on ? this->setBitOn(bit) : this->setBitOff(bit); }
    template<typename MaskT>
    __hostdev__ void setMask(MaskT mask, bool on) { on ? this->setMaskOn(mask) : this->setMaskOff(mask); }

    __hostdev__ bool isOn() const { return mFlags == ~Type(0u); }
    __hostdev__ bool isOff() const { return mFlags == Type(0u); }
    __hostdev__ bool isBitOn(uint8_t bit) const { return 0 != (mFlags & static_cast<Type>(1 << bit)); }
    __hostdev__ bool isBitOff(uint8_t bit) const { return 0 == (mFlags & static_cast<Type>(1 << bit)); }
    template<typename MaskT>
    __hostdev__ bool isMaskOn(MaskT mask) const { return 0 != (mFlags & static_cast<Type>(mask)); }
    template<typename MaskT>
    __hostdev__ bool isMaskOff(MaskT mask) const { return 0 == (mFlags & static_cast<Type>(mask)); }
    /// @brief return true if any of the masks in the list are on
    template<typename MaskT>
    __hostdev__ bool isMaskOn(std::initializer_list<MaskT> list) const
    {
        for (auto mask : list) {
            if (0 != (mFlags & static_cast<Type>(mask))) return true;
        }
        return false;
    }
    /// @brief return true if any of the masks in the list are off
    template<typename MaskT>
    __hostdev__ bool isMaskOff(std::initializer_list<MaskT> list) const
    {
        for (auto mask : list) {
            if (0 == (mFlags & static_cast<Type>(mask))) return true;
        }
        return false;
    }
    /// @brief required for backwards compatibility
    __hostdev__ BitFlags& operator=(Type n)
    {
        mFlags = n;
        return *this;
    }
}; // BitFlags<N>

// ----------------------------> Mask <--------------------------------------

/// @brief Bit-mask to encode active states and facilitate sequential iterators
/// and a fast codec for I/O compression.
template<uint32_t LOG2DIM>
class Mask
{
public:
    static constexpr uint32_t SIZE = 1U << (3 * LOG2DIM); // Number of bits in mask
    static constexpr uint32_t WORD_COUNT = SIZE >> 6; // Number of 64 bit words

    /// @brief Return the memory footprint in bytes of this Mask
    __hostdev__ static size_t memUsage() { return sizeof(Mask); }

    /// @brief Return the number of bits available in this Mask
    __hostdev__ static uint32_t bitCount() { return SIZE; }

    /// @brief Return the number of machine words used by this Mask
    __hostdev__ static uint32_t wordCount() { return WORD_COUNT; }

    /// @brief Return the total number of set bits in this Mask
    __hostdev__ uint32_t countOn() const
    {
        uint32_t sum = 0;
        for (const uint64_t *w = mWords, *q = w + WORD_COUNT; w != q; ++w)
            sum += util::countOn(*w);
        return sum;
    }

    /// @brief Return the number of lower set bits in mask up to but excluding the i'th bit
    inline __hostdev__ uint32_t countOn(uint32_t i) const
    {
        uint32_t n = i >> 6, sum = util::countOn(mWords[n] & ((uint64_t(1) << (i & 63u)) - 1u));
        for (const uint64_t* w = mWords; n--; ++w)
            sum += util::countOn(*w);
        return sum;
    }

    template<bool On>
    class Iterator
    {
    public:
        __hostdev__ Iterator()
            : mPos(Mask::SIZE)
            , mParent(nullptr)
        {
        }
        __hostdev__ Iterator(uint32_t pos, const Mask* parent)
            : mPos(pos)
            , mParent(parent)
        {
        }
        Iterator&            operator=(const Iterator&) = default;
        __hostdev__ uint32_t operator*() const { return mPos; }
        __hostdev__ uint32_t pos() const { return mPos; }
        __hostdev__ operator bool() const { return mPos != Mask::SIZE; }
        __hostdev__ Iterator& operator++()
        {
            mPos = mParent->findNext<On>(mPos + 1);
            return *this;
        }
        __hostdev__ Iterator operator++(int)
        {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

    private:
        uint32_t    mPos;
        const Mask* mParent;
    }; // Member class Iterator

    class DenseIterator
    {
    public:
        __hostdev__ DenseIterator(uint32_t pos = Mask::SIZE)
            : mPos(pos)
        {
        }
        DenseIterator&       operator=(const DenseIterator&) = default;
        __hostdev__ uint32_t operator*() const { return mPos; }
        __hostdev__ uint32_t pos() const { return mPos; }
        __hostdev__ operator bool() const { return mPos != Mask::SIZE; }
        __hostdev__ DenseIterator& operator++()
        {
            ++mPos;
            return *this;
        }
        __hostdev__ DenseIterator operator++(int)
        {
            auto tmp = *this;
            ++mPos;
            return tmp;
        }

    private:
        uint32_t mPos;
    }; // Member class DenseIterator

    using OnIterator = Iterator<true>;
    using OffIterator = Iterator<false>;

    __hostdev__ OnIterator beginOn() const { return OnIterator(this->findFirst<true>(), this); }

    __hostdev__ OffIterator beginOff() const { return OffIterator(this->findFirst<false>(), this); }

    __hostdev__ DenseIterator beginAll() const { return DenseIterator(0); }

    /// @brief Initialize all bits to zero.
    __hostdev__ Mask()
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = 0;
    }
    __hostdev__ Mask(bool on)
    {
        const uint64_t v = on ? ~uint64_t(0) : uint64_t(0);
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = v;
    }

    /// @brief Copy constructor
    __hostdev__ Mask(const Mask& other)
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = other.mWords[i];
    }

    /// @brief Return a pointer to the list of words of the bit mask
    __hostdev__ uint64_t*       words() { return mWords; }
    __hostdev__ const uint64_t* words() const { return mWords; }

    template<typename WordT>
    __hostdev__ WordT getWord(uint32_t n) const
    {
        static_assert(util::is_same<WordT, uint8_t, uint16_t, uint32_t, uint64_t>::value);
        NANOVDB_ASSERT(n*8*sizeof(WordT) < WORD_COUNT);
        return reinterpret_cast<WordT*>(mWords)[n];
    }
    template<typename WordT>
    __hostdev__ void setWord(WordT w, uint32_t n)
    {
        static_assert(util::is_same<WordT, uint8_t, uint16_t, uint32_t, uint64_t>::value);
        NANOVDB_ASSERT(n*8*sizeof(WordT) < WORD_COUNT);
        reinterpret_cast<WordT*>(mWords)[n] = w;
    }

    /// @brief Assignment operator that works with openvdb::util::NodeMask
    template<typename MaskT = Mask>
    __hostdev__ typename util::enable_if<!util::is_same<MaskT, Mask>::value, Mask&>::type operator=(const MaskT& other)
    {
        static_assert(sizeof(Mask) == sizeof(MaskT), "Mismatching sizeof");
        static_assert(WORD_COUNT == MaskT::WORD_COUNT, "Mismatching word count");
        static_assert(LOG2DIM == MaskT::LOG2DIM, "Mismatching LOG2DIM");
        auto* src = reinterpret_cast<const uint64_t*>(&other);
        for (uint64_t *dst = mWords, *end = dst + WORD_COUNT; dst != end; ++dst)
            *dst = *src++;
        return *this;
    }

    //__hostdev__ Mask& operator=(const Mask& other){return *util::memcpy(this, &other);}
    Mask& operator=(const Mask&) = default;

    __hostdev__ bool operator==(const Mask& other) const
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            if (mWords[i] != other.mWords[i])
                return false;
        }
        return true;
    }

    __hostdev__ bool operator!=(const Mask& other) const { return !((*this) == other); }

    /// @brief Return true if the given bit is set.
    __hostdev__ bool isOn(uint32_t n) const { return 0 != (mWords[n >> 6] & (uint64_t(1) << (n & 63))); }

    /// @brief Return true if the given bit is NOT set.
    __hostdev__ bool isOff(uint32_t n) const { return 0 == (mWords[n >> 6] & (uint64_t(1) << (n & 63))); }

    /// @brief Return true if all the bits are set in this Mask.
    __hostdev__ bool isOn() const
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            if (mWords[i] != ~uint64_t(0))
                return false;
        return true;
    }

    /// @brief Return true if none of the bits are set in this Mask.
    __hostdev__ bool isOff() const
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            if (mWords[i] != uint64_t(0))
                return false;
        return true;
    }

    /// @brief Set the specified bit on.
    __hostdev__ void setOn(uint32_t n) { mWords[n >> 6] |= uint64_t(1) << (n & 63); }
    /// @brief Set the specified bit off.
    __hostdev__ void setOff(uint32_t n) { mWords[n >> 6] &= ~(uint64_t(1) << (n & 63)); }

#if defined(__CUDACC__) // the following functions only run on the GPU!
    __device__ inline void setOnAtomic(uint32_t n)
    {
        atomicOr(reinterpret_cast<unsigned long long int*>(this) + (n >> 6), 1ull << (n & 63));
    }
    __device__ inline void setOffAtomic(uint32_t n)
    {
        atomicAnd(reinterpret_cast<unsigned long long int*>(this) + (n >> 6), ~(1ull << (n & 63)));
    }
    __device__ inline void setAtomic(uint32_t n, bool on)
    {
        on ? this->setOnAtomic(n) : this->setOffAtomic(n);
    }
/*
    template<typename WordT>
    __device__ inline void setWordAtomic(WordT w, uint32_t n)
    {
        static_assert(util::is_same<WordT, uint8_t, uint16_t, uint32_t, uint64_t>::value);
        NANOVDB_ASSERT(n*8*sizeof(WordT) < WORD_COUNT);
        if constexpr(util::is_same<WordT,uint8_t>::value) {
            mask <<= x;
        } else if constexpr(util::is_same<WordT,uint16_t>::value) {
            unsigned int mask = w;
            if (n >> 1) mask <<= 16;
            atomicOr(reinterpret_cast<unsigned int*>(this) + n, mask);
        } else if constexpr(util::is_same<WordT,uint32_t>::value) {
            atomicOr(reinterpret_cast<unsigned int*>(this) + n, w);
        } else {
            atomicOr(reinterpret_cast<unsigned long long int*>(this) + n, w);
        }
    }
*/
#endif
    /// @brief Set the specified bit on or off.
    __hostdev__ void set(uint32_t n, bool on)
    {
#if 1 // switch between branchless
        auto& word = mWords[n >> 6];
        n &= 63;
        word &= ~(uint64_t(1) << n);
        word |= uint64_t(on) << n;
#else
        on ? this->setOn(n) : this->setOff(n);
#endif
    }

    /// @brief Set all bits on
    __hostdev__ void setOn()
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)mWords[i] = ~uint64_t(0);
    }

    /// @brief Set all bits off
    __hostdev__ void setOff()
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) mWords[i] = uint64_t(0);
    }

    /// @brief Set all bits off
    __hostdev__ void set(bool on)
    {
        const uint64_t v = on ? ~uint64_t(0) : uint64_t(0);
        for (uint32_t i = 0; i < WORD_COUNT; ++i) mWords[i] = v;
    }
    /// brief Toggle the state of all bits in the mask
    __hostdev__ void toggle()
    {
        uint32_t n = WORD_COUNT;
        for (auto* w = mWords; n--; ++w) *w = ~*w;
    }
    __hostdev__ void toggle(uint32_t n) { mWords[n >> 6] ^= uint64_t(1) << (n & 63); }

    /// @brief Bitwise intersection
    __hostdev__ Mask& operator&=(const Mask& other)
    {
        uint64_t*       w1 = mWords;
        const uint64_t* w2 = other.mWords;
        for (uint32_t n = WORD_COUNT; n--; ++w1, ++w2) *w1 &= *w2;
        return *this;
    }
    /// @brief Bitwise union
    __hostdev__ Mask& operator|=(const Mask& other)
    {
        uint64_t*       w1 = mWords;
        const uint64_t* w2 = other.mWords;
        for (uint32_t n = WORD_COUNT; n--; ++w1, ++w2) *w1 |= *w2;
        return *this;
    }
    /// @brief Bitwise difference
    __hostdev__ Mask& operator-=(const Mask& other)
    {
        uint64_t*       w1 = mWords;
        const uint64_t* w2 = other.mWords;
        for (uint32_t n = WORD_COUNT; n--; ++w1, ++w2) *w1 &= ~*w2;
        return *this;
    }
    /// @brief Bitwise XOR
    __hostdev__ Mask& operator^=(const Mask& other)
    {
        uint64_t*       w1 = mWords;
        const uint64_t* w2 = other.mWords;
        for (uint32_t n = WORD_COUNT; n--; ++w1, ++w2) *w1 ^= *w2;
        return *this;
    }

    NANOVDB_HOSTDEV_DISABLE_WARNING
    template<bool ON>
    __hostdev__ uint32_t findFirst() const
    {
        uint32_t        n = 0u;
        const uint64_t* w = mWords;
        for (; n < WORD_COUNT && !(ON ? *w : ~*w); ++w, ++n);
        return n < WORD_COUNT ? (n << 6) + util::findLowestOn(ON ? *w : ~*w) : SIZE;
    }

    NANOVDB_HOSTDEV_DISABLE_WARNING
    template<bool ON>
    __hostdev__ uint32_t findNext(uint32_t start) const
    {
        uint32_t n = start >> 6; // initiate
        if (n >= WORD_COUNT) return SIZE; // check for out of bounds
        uint32_t m = start & 63u;
        uint64_t b = ON ? mWords[n] : ~mWords[n];
        if (b & (uint64_t(1u) << m)) return start; // simple case: start is on/off
        b &= ~uint64_t(0u) << m; // mask out lower bits
        while (!b && ++n < WORD_COUNT) b = ON ? mWords[n] : ~mWords[n]; // find next non-zero word
        return b ? (n << 6) + util::findLowestOn(b) : SIZE; // catch last word=0
    }

    NANOVDB_HOSTDEV_DISABLE_WARNING
    template<bool ON>
    __hostdev__ uint32_t findPrev(uint32_t start) const
    {
        uint32_t n = start >> 6; // initiate
        if (n >= WORD_COUNT) return SIZE; // check for out of bounds
        uint32_t m = start & 63u;
        uint64_t b = ON ? mWords[n] : ~mWords[n];
        if (b & (uint64_t(1u) << m)) return start; // simple case: start is on/off
        b &= (uint64_t(1u) << m) - 1u; // mask out higher bits
        while (!b && n) b = ON ? mWords[--n] : ~mWords[--n]; // find previous non-zero word
        return b ? (n << 6) + util::findHighestOn(b) : SIZE; // catch first word=0
    }

private:
    uint64_t mWords[WORD_COUNT];
}; // Mask class

// ----------------------------> Map <--------------------------------------

/// @brief Defines an affine transform and its inverse represented as a 3x3 matrix and a vec3 translation
struct Map
{ // 264B (not 32B aligned!)
    float  mMatF[9]; // 9*4B <- 3x3 matrix
    float  mInvMatF[9]; // 9*4B <- 3x3 matrix
    float  mVecF[3]; // 3*4B <- translation
    float  mTaperF; // 4B, placeholder for taper value
    double mMatD[9]; // 9*8B <- 3x3 matrix
    double mInvMatD[9]; // 9*8B <- 3x3 matrix
    double mVecD[3]; // 3*8B <- translation
    double mTaperD; // 8B, placeholder for taper value

    /// @brief Default constructor for the identity map
    __hostdev__ Map()
        : mMatF{   1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}
        , mInvMatF{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}
        , mVecF{0.0f, 0.0f, 0.0f}
        , mTaperF{1.0f}
        , mMatD{   1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}
        , mInvMatD{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}
        , mVecD{0.0, 0.0, 0.0}
        , mTaperD{1.0}
    {
    }
    __hostdev__ Map(double s, const Vec3d& t = Vec3d(0.0, 0.0, 0.0))
        : mMatF{float(s), 0.0f, 0.0f, 0.0f, float(s), 0.0f, 0.0f, 0.0f, float(s)}
        , mInvMatF{1.0f / float(s), 0.0f, 0.0f, 0.0f, 1.0f / float(s), 0.0f, 0.0f, 0.0f, 1.0f / float(s)}
        , mVecF{float(t[0]), float(t[1]), float(t[2])}
        , mTaperF{1.0f}
        , mMatD{s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s}
        , mInvMatD{1.0 / s, 0.0, 0.0, 0.0, 1.0 / s, 0.0, 0.0, 0.0, 1.0 / s}
        , mVecD{t[0], t[1], t[2]}
        , mTaperD{1.0}
    {
    }

    /// @brief Initialize the member data from 3x3 or 4x4 matrices
    /// @note This is not _hostdev__ since then MatT=openvdb::Mat4d will produce warnings
    template<typename MatT, typename Vec3T>
    void set(const MatT& mat, const MatT& invMat, const Vec3T& translate, double taper = 1.0);

    /// @brief Initialize the member data from 4x4 matrices
    /// @note  The last (4th) row of invMat is actually ignored.
    ///        This is not _hostdev__ since then Mat4T=openvdb::Mat4d will produce warnings
    template<typename Mat4T>
    void set(const Mat4T& mat, const Mat4T& invMat, double taper = 1.0) { this->set(mat, invMat, mat[3], taper); }

    template<typename Vec3T>
    void set(double scale, const Vec3T& translation, double taper = 1.0);

    /// @brief Apply the forward affine transformation to a vector using 64bit floating point arithmetics.
    /// @note Typically this operation is used for the scale, rotation and translation of index -> world mapping
    /// @tparam Vec3T Template type of the 3D vector to be mapped
    /// @param ijk 3D vector to be mapped - typically floating point index coordinates
    /// @return Forward mapping for affine transformation, i.e. (mat x ijk) + translation
    template<typename Vec3T>
    __hostdev__ Vec3T applyMap(const Vec3T& ijk) const { return math::matMult(mMatD, mVecD, ijk); }

    /// @brief Apply the forward affine transformation to a vector using 32bit floating point arithmetics.
    /// @note Typically this operation is used for the scale, rotation and translation of index -> world mapping
    /// @tparam Vec3T Template type of the 3D vector to be mapped
    /// @param ijk 3D vector to be mapped - typically floating point index coordinates
    /// @return Forward mapping for affine transformation, i.e. (mat x ijk) + translation
    template<typename Vec3T>
    __hostdev__ Vec3T applyMapF(const Vec3T& ijk) const { return math::matMult(mMatF, mVecF, ijk); }

    /// @brief Apply the linear forward 3x3 transformation to an input 3d vector using 64bit floating point arithmetics,
    ///        e.g. scale and rotation WITHOUT translation.
    /// @note Typically this operation is used for scale and rotation from index -> world mapping
    /// @tparam Vec3T Template type of the 3D vector to be mapped
    /// @param ijk 3D vector to be mapped - typically floating point index coordinates
    /// @return linear forward 3x3 mapping of the input vector
    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobian(const Vec3T& ijk) const { return math::matMult(mMatD, ijk); }

    /// @brief Apply the linear forward 3x3 transformation to an input 3d vector using 32bit floating point arithmetics,
    ///        e.g. scale and rotation WITHOUT translation.
    /// @note Typically this operation is used for scale and rotation from index -> world mapping
    /// @tparam Vec3T Template type of the 3D vector to be mapped
    /// @param ijk 3D vector to be mapped - typically floating point index coordinates
    /// @return linear forward 3x3 mapping of the input vector
    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobianF(const Vec3T& ijk) const { return math::matMult(mMatF, ijk); }

    /// @brief Apply the inverse affine mapping to a vector using 64bit floating point arithmetics.
    /// @note Typically this operation is used for the world -> index mapping
    /// @tparam Vec3T Template type of the 3D vector to be mapped
    /// @param xyz 3D vector to be mapped - typically floating point world coordinates
    /// @return Inverse affine mapping of the input @c xyz i.e. (xyz - translation) x mat^-1
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMap(const Vec3T& xyz) const
    {
        return math::matMult(mInvMatD, Vec3T(xyz[0] - mVecD[0], xyz[1] - mVecD[1], xyz[2] - mVecD[2]));
    }

    /// @brief Apply the inverse affine mapping to a vector using 32bit floating point arithmetics.
    /// @note Typically this operation is used for the world -> index mapping
    /// @tparam Vec3T Template type of the 3D vector to be mapped
    /// @param xyz 3D vector to be mapped - typically floating point world coordinates
    /// @return Inverse affine mapping of the input @c xyz i.e. (xyz - translation) x mat^-1
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMapF(const Vec3T& xyz) const
    {
        return math::matMult(mInvMatF, Vec3T(xyz[0] - mVecF[0], xyz[1] - mVecF[1], xyz[2] - mVecF[2]));
    }

    /// @brief Apply the linear inverse 3x3 transformation to an input 3d vector using 64bit floating point arithmetics,
    ///        e.g. inverse scale and inverse rotation WITHOUT translation.
    /// @note Typically this operation is used for scale and rotation from world -> index mapping
    /// @tparam Vec3T Template type of the 3D vector to be mapped
    /// @param ijk 3D vector to be mapped - typically floating point index coordinates
    /// @return linear inverse 3x3 mapping of the input vector i.e. xyz x mat^-1
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobian(const Vec3T& xyz) const { return math::matMult(mInvMatD, xyz); }

    /// @brief Apply the linear inverse 3x3 transformation to an input 3d vector using 32bit floating point arithmetics,
    ///        e.g. inverse scale and inverse rotation WITHOUT translation.
    /// @note Typically this operation is used for scale and rotation from world -> index mapping
    /// @tparam Vec3T Template type of the 3D vector to be mapped
    /// @param ijk 3D vector to be mapped - typically floating point index coordinates
    /// @return linear inverse 3x3 mapping of the input vector i.e. xyz x mat^-1
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobianF(const Vec3T& xyz) const { return math::matMult(mInvMatF, xyz); }

    /// @brief Apply the transposed inverse 3x3 transformation to an input 3d vector using 64bit floating point arithmetics,
    ///        e.g. inverse scale and inverse rotation WITHOUT translation.
    /// @note Typically this operation is used for scale and rotation from world -> index mapping
    /// @tparam Vec3T Template type of the 3D vector to be mapped
    /// @param ijk 3D vector to be mapped - typically floating point index coordinates
    /// @return linear inverse 3x3 mapping of the input vector i.e. xyz x mat^-1
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJT(const Vec3T& xyz) const { return math::matMultT(mInvMatD, xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJTF(const Vec3T& xyz) const { return math::matMultT(mInvMatF, xyz); }

    /// @brief Return a voxels size in each coordinate direction, measured at the origin
    __hostdev__ Vec3d getVoxelSize() const { return this->applyMap(Vec3d(1)) - this->applyMap(Vec3d(0)); }
}; // Map

template<typename MatT, typename Vec3T>
inline void Map::set(const MatT& mat, const MatT& invMat, const Vec3T& translate, double taper)
{
    float * mf = mMatF, *vf = mVecF, *mif = mInvMatF;
    double *md = mMatD, *vd = mVecD, *mid = mInvMatD;
    mTaperF = static_cast<float>(taper);
    mTaperD = taper;
    for (int i = 0; i < 3; ++i) {
        *vd++ = translate[i]; //translation
        *vf++ = static_cast<float>(translate[i]); //translation
        for (int j = 0; j < 3; ++j) {
            *md++ = mat[j][i]; //transposed
            *mid++ = invMat[j][i];
            *mf++ = static_cast<float>(mat[j][i]); //transposed
            *mif++ = static_cast<float>(invMat[j][i]);
        }
    }
}

template<typename Vec3T>
inline void Map::set(double dx, const Vec3T& trans, double taper)
{
    NANOVDB_ASSERT(dx > 0.0);
    const double mat[3][3] = { {dx, 0.0, 0.0},   // row 0
                               {0.0, dx, 0.0},   // row 1
                               {0.0, 0.0, dx} }; // row 2
    const double idx = 1.0 / dx;
    const double invMat[3][3] = { {idx, 0.0, 0.0},   // row 0
                                  {0.0, idx, 0.0},   // row 1
                                  {0.0, 0.0, idx} }; // row 2
    this->set(mat, invMat, trans, taper);
}

// ----------------------------> GridBlindMetaData <--------------------------------------

struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) GridBlindMetaData
{ // 288 bytes
    static const int      MaxNameSize = 256; // due to NULL termination the maximum length is one less!
    int64_t               mDataOffset; // byte offset to the blind data, relative to GridBlindMetaData::this.
    uint64_t              mValueCount; // number of blind values, e.g. point count
    uint32_t              mValueSize;// byte size of each value, e.g. 4 if mDataType=Float and 1 if mDataType=Unknown since that amounts to char
    GridBlindDataSemantic mSemantic; // semantic meaning of the data.
    GridBlindDataClass    mDataClass; // 4 bytes
    GridType              mDataType; // 4 bytes
    char                  mName[MaxNameSize]; // note this includes the NULL termination
    // no padding required for 32 byte alignment

    /// @brief Empty constructor
    GridBlindMetaData()
        : mDataOffset(0)
        , mValueCount(0)
        , mValueSize(0)
        , mSemantic(GridBlindDataSemantic::Unknown)
        , mDataClass(GridBlindDataClass::Unknown)
        , mDataType(GridType::Unknown)
    {
        util::memzero(mName, MaxNameSize);
    }

    GridBlindMetaData(int64_t dataOffset, uint64_t valueCount, uint32_t valueSize, GridBlindDataSemantic semantic, GridBlindDataClass dataClass, GridType dataType)
        : mDataOffset(dataOffset)
        , mValueCount(valueCount)
        , mValueSize(valueSize)
        , mSemantic(semantic)
        , mDataClass(dataClass)
        , mDataType(dataType)
    {
        util::memzero(mName, MaxNameSize);
    }

    /// @brief Copy constructor that resets mDataOffset and zeros out mName
    GridBlindMetaData(const GridBlindMetaData &other)
        : mDataOffset(util::PtrDiff(util::PtrAdd(&other, other.mDataOffset), this))
        , mValueCount(other.mValueCount)
        , mValueSize(other.mValueSize)
        , mSemantic(other.mSemantic)
        , mDataClass(other.mDataClass)
        , mDataType(other.mDataType)
    {
        util::strncpy(mName, other.mName, MaxNameSize);
    }

    /// @brief Copy assignment operator that resets mDataOffset and copies mName
    /// @param rhs right-hand instance to copy
    /// @return reference to itself
    const GridBlindMetaData& operator=(const GridBlindMetaData& rhs)
    {
        mDataOffset = util::PtrDiff(util::PtrAdd(&rhs, rhs.mDataOffset), this);
        mValueCount = rhs.mValueCount;
        mValueSize  = rhs. mValueSize;
        mSemantic   = rhs.mSemantic;
        mDataClass  = rhs.mDataClass;
        mDataType   = rhs.mDataType;
        util::strncpy(mName, rhs.mName, MaxNameSize);
        return *this;
    }

    __hostdev__ void setBlindData(const void* blindData)
    {
        mDataOffset = util::PtrDiff(blindData, this);
    }

    /// @brief Sets the name string
    /// @param name c-string source name
    /// @return returns false if @c name has too many characters
    __hostdev__  bool setName(const char* name){return util::strncpy(mName, name, MaxNameSize)[MaxNameSize-1] == '\0';}

    /// @brief returns a const void point to the blind data
    /// @note assumes that setBlinddData was called
    __hostdev__ const void* blindData() const
    {
        NANOVDB_ASSERT(mDataOffset != 0);
        return util::PtrAdd(this, mDataOffset);
    }

    /// @brief Get a const pointer to the blind data represented by this meta data
    /// @tparam BlindDataT Expected value type of the blind data.
    /// @return Returns NULL if mGridType!=toGridType<BlindDataT>(), else a const point of type BlindDataT.
    /// @note Use mDataType=Unknown if BlindDataT is a custom data type unknown to NanoVDB.
    template<typename BlindDataT>
    __hostdev__ const BlindDataT* getBlindData() const
    {
        return mDataOffset && (mDataType == toGridType<BlindDataT>()) ? util::PtrAdd<BlindDataT>(this, mDataOffset) : nullptr;
    }

    /// @brief return true if this meta data has a valid combination of semantic, class and value tags
    /// @note this does not check if the mDataOffset has been set!
    __hostdev__ bool isValid() const
    {
        auto check = [&]()->bool{
            switch (mDataType){
            case GridType::Unknown: return mValueSize==1u;// i.e. we encode data as mValueCount chars
            case GridType::Float:   return mValueSize==4u;
            case GridType::Double:  return mValueSize==8u;
            case GridType::Int16:   return mValueSize==2u;
            case GridType::Int32:   return mValueSize==4u;
            case GridType::Int64:   return mValueSize==8u;
            case GridType::Vec3f:   return mValueSize==12u;
            case GridType::Vec3d:   return mValueSize==24u;
            case GridType::Half:    return mValueSize==2u;
            case GridType::RGBA8:   return mValueSize==4u;
            case GridType::Fp8:     return mValueSize==1u;
            case GridType::Fp16:    return mValueSize==2u;
            case GridType::Vec4f:   return mValueSize==16u;
            case GridType::Vec4d:   return mValueSize==32u;
            case GridType::Vec3u8:  return mValueSize==3u;
            case GridType::Vec3u16: return mValueSize==6u;
            default: return true;}// all other combinations are valid
        };
        return nanovdb::isValid(mDataClass, mSemantic, mDataType) && check();
    }

    /// @brief return size in bytes of the blind data represented by this blind meta data
    /// @note This size includes possible padding for 32 byte alignment. The actual amount
    ///       of bind data is mValueCount * mValueSize
    __hostdev__ uint64_t blindDataSize() const
    {
        return math::AlignUp<NANOVDB_DATA_ALIGNMENT>(mValueCount * mValueSize);
    }
}; // GridBlindMetaData

// ----------------------------> NodeTrait <--------------------------------------

/// @brief Struct to derive node type from its level in a given
///        grid, tree or root while preserving constness
template<typename GridOrTreeOrRootT, int LEVEL>
struct NodeTrait;

// Partial template specialization of above Node struct
template<typename GridOrTreeOrRootT>
struct NodeTrait<GridOrTreeOrRootT, 0>
{
    static_assert(GridOrTreeOrRootT::RootNodeType::LEVEL == 3, "Tree depth is not supported");
    using Type = typename GridOrTreeOrRootT::LeafNodeType;
    using type = typename GridOrTreeOrRootT::LeafNodeType;
};
template<typename GridOrTreeOrRootT>
struct NodeTrait<const GridOrTreeOrRootT, 0>
{
    static_assert(GridOrTreeOrRootT::RootNodeType::LEVEL == 3, "Tree depth is not supported");
    using Type = const typename GridOrTreeOrRootT::LeafNodeType;
    using type = const typename GridOrTreeOrRootT::LeafNodeType;
};

template<typename GridOrTreeOrRootT>
struct NodeTrait<GridOrTreeOrRootT, 1>
{
    static_assert(GridOrTreeOrRootT::RootNodeType::LEVEL == 3, "Tree depth is not supported");
    using Type = typename GridOrTreeOrRootT::RootNodeType::ChildNodeType::ChildNodeType;
    using type = typename GridOrTreeOrRootT::RootNodeType::ChildNodeType::ChildNodeType;
};
template<typename GridOrTreeOrRootT>
struct NodeTrait<const GridOrTreeOrRootT, 1>
{
    static_assert(GridOrTreeOrRootT::RootNodeType::LEVEL == 3, "Tree depth is not supported");
    using Type = const typename GridOrTreeOrRootT::RootNodeType::ChildNodeType::ChildNodeType;
    using type = const typename GridOrTreeOrRootT::RootNodeType::ChildNodeType::ChildNodeType;
};
template<typename GridOrTreeOrRootT>
struct NodeTrait<GridOrTreeOrRootT, 2>
{
    static_assert(GridOrTreeOrRootT::RootNodeType::LEVEL == 3, "Tree depth is not supported");
    using Type = typename GridOrTreeOrRootT::RootNodeType::ChildNodeType;
    using type = typename GridOrTreeOrRootT::RootNodeType::ChildNodeType;
};
template<typename GridOrTreeOrRootT>
struct NodeTrait<const GridOrTreeOrRootT, 2>
{
    static_assert(GridOrTreeOrRootT::RootNodeType::LEVEL == 3, "Tree depth is not supported");
    using Type = const typename GridOrTreeOrRootT::RootNodeType::ChildNodeType;
    using type = const typename GridOrTreeOrRootT::RootNodeType::ChildNodeType;
};
template<typename GridOrTreeOrRootT>
struct NodeTrait<GridOrTreeOrRootT, 3>
{
    static_assert(GridOrTreeOrRootT::RootNodeType::LEVEL == 3, "Tree depth is not supported");
    using Type = typename GridOrTreeOrRootT::RootNodeType;
    using type = typename GridOrTreeOrRootT::RootNodeType;
};

template<typename GridOrTreeOrRootT>
struct NodeTrait<const GridOrTreeOrRootT, 3>
{
    static_assert(GridOrTreeOrRootT::RootNodeType::LEVEL == 3, "Tree depth is not supported");
    using Type = const typename GridOrTreeOrRootT::RootNodeType;
    using type = const typename GridOrTreeOrRootT::RootNodeType;
};

// ------------> Froward decelerations of accelerated random access methods <---------------

template<typename BuildT>
struct GetValue;
template<typename BuildT>
struct SetValue;
template<typename BuildT>
struct SetVoxel;
template<typename BuildT>
struct GetState;
template<typename BuildT>
struct GetDim;
template<typename BuildT>
struct GetLeaf;
template<typename BuildT>
struct ProbeValue;
template<typename BuildT>
struct GetNodeInfo;

// ----------------------------> CheckMode <----------------------------------

/// @brief List of different modes for computing for a checksum
enum class CheckMode : uint32_t { Disable = 0,  // no computation
                                  Empty   = 0,
                                  Half    = 1,
                                  Partial = 1,  // fast but approximate
                                  Default = 1,  // defaults to Partial
                                  Full    = 2,  // slow but accurate
                                  End     = 3, // marks the end of the enum list
                                  StrLen  = 9 + End};

/// @brief Prints CheckMode enum to a c-string
/// @param dst Destination c-string
/// @param mode CheckMode enum to be converted to string
/// @return destinations string @c dst
__hostdev__ inline char* toStr(char *dst, CheckMode mode)
{
    switch (mode){
        case CheckMode::Half: return util::strcpy(dst, "half");
        case CheckMode::Full: return util::strcpy(dst, "full");
        default: return util::strcpy(dst, "disabled");// StrLen = 8 + 1 + End
    }
}

// ----------------------------> Checksum <----------------------------------

/// @brief Class that encapsulates two CRC32 checksums, one for the Grid, Tree and Root node meta data
///        and one for the remaining grid nodes.
class Checksum
{
    /// Three types of checksums:
    ///   1) Empty: all 64 bits are on (used to signify a disabled or undefined checksum)
    ///   2) Half: Upper 32 bits are on and not all of lower 32 bits are on (lower 32 bits checksum head of grid)
    ///   3) Full: Not all of the 64 bits are one (lower 32 bits checksum head of grid and upper 32 bits checksum tail of grid)
    union { uint32_t mCRC32[2]; uint64_t mCRC64; };// mCRC32[0] is checksum of Grid, Tree and Root, and mCRC32[1] is checksum of nodes

public:

    static constexpr uint32_t EMPTY32 = ~uint32_t{0};
    static constexpr uint64_t EMPTY64 = ~uint64_t(0);

    /// @brief default constructor initiates checksum to EMPTY
    __hostdev__ Checksum() : mCRC64{EMPTY64} {}

    /// @brief Constructor that allows the two 32bit checksums to be initiated explicitly
    /// @param head Initial 32bit CRC checksum of grid, tree and root data
    /// @param tail Initial 32bit CRC checksum of all the nodes and blind data
    __hostdev__ Checksum(uint32_t head, uint32_t tail) :  mCRC32{head, tail} {}

    /// @brief
    /// @param checksum
    /// @param mode
    __hostdev__ Checksum(uint64_t checksum, CheckMode mode = CheckMode::Full) : mCRC64{mode == CheckMode::Disable ? EMPTY64 : checksum}
    {
        if (mode == CheckMode::Partial) mCRC32[1] = EMPTY32;
    }

    /// @brief return the 64 bit checksum of this instance
    [[deprecated("Use Checksum::data instead.")]]
    __hostdev__ uint64_t checksum() const { return mCRC64; }
    [[deprecated("Use Checksum::head and Ckecksum::tail instead.")]]
    __hostdev__ uint32_t& checksum(int i) {NANOVDB_ASSERT(i==0 || i==1); return mCRC32[i]; }
    [[deprecated("Use Checksum::head and Ckecksum::tail instead.")]]
    __hostdev__ uint32_t checksum(int i) const {NANOVDB_ASSERT(i==0 || i==1); return mCRC32[i]; }

    __hostdev__ uint64_t  full() const { return mCRC64; }
    __hostdev__ uint64_t& full()       { return mCRC64; }
    __hostdev__ uint32_t  head() const { return mCRC32[0]; }
    __hostdev__ uint32_t& head()       { return mCRC32[0]; }
    __hostdev__ uint32_t  tail() const { return mCRC32[1]; }
    __hostdev__ uint32_t& tail()       { return mCRC32[1]; }

    /// @brief return true if the 64 bit checksum is partial, i.e. of head only
    [[deprecated("Use Checksum::isHalf instead.")]]
    __hostdev__ bool isPartial() const { return mCRC32[0] != EMPTY32 && mCRC32[1] == EMPTY32; }
    __hostdev__ bool isHalf() const { return mCRC32[0] != EMPTY32 && mCRC32[1] == EMPTY32; }

    /// @brief return true if the 64 bit checksum is fill, i.e. of both had and nodes
    __hostdev__ bool isFull() const { return mCRC64 != EMPTY64 && mCRC32[1] != EMPTY32; }

    /// @brief return true if the 64 bit checksum is disables (unset)
    __hostdev__ bool isEmpty() const { return mCRC64 == EMPTY64; }

    __hostdev__ void disable() { mCRC64 = EMPTY64; }

    /// @brief return the mode of the 64 bit checksum
    __hostdev__ CheckMode mode() const
    {
        return mCRC64    == EMPTY64 ? CheckMode::Disable :
               mCRC32[1] == EMPTY32 ? CheckMode::Partial : CheckMode::Full;
    }

    /// @brief return true if the checksums are identical
    /// @param rhs other Checksum
    __hostdev__ bool operator==(const Checksum &rhs) const {return mCRC64 == rhs.mCRC64;}

    /// @brief return true if the checksums are not identical
    /// @param rhs other Checksum
    __hostdev__ bool operator!=(const Checksum &rhs) const {return mCRC64 != rhs.mCRC64;}
};// Checksum

/// @brief Maps 64 bit checksum to CheckMode enum
/// @param checksum 64 bit checksum with two CRC32 codes
/// @return CheckMode enum
__hostdev__ inline CheckMode toCheckMode(const Checksum &checksum){return checksum.mode();}

// ----------------------------> Grid <--------------------------------------

/*
    The following class and comment is for internal use only

    Memory layout:

    Grid ->       39 x double                          (world bbox and affine transformation)
    Tree -> Root  3 x ValueType + int32_t + N x Tiles  (background,min,max,tileCount + tileCount x Tiles)

    N2 upper InternalNodes each with 2 bit masks, N2 tiles, and min/max values

    N1 lower InternalNodes each with 2 bit masks, N1 tiles, and min/max values

    N0 LeafNodes each with a bit mask, N0 ValueTypes and min/max

    Example layout: ("---" implies it has a custom offset, "..." implies zero or more)
    [GridData][TreeData]---[RootData][ROOT TILES...]---[InternalData<5>]---[InternalData<4>]---[LeafData<3>]---[BLINDMETA...]---[BLIND0]---[BLIND1]---etc.
*/

/// @brief Struct with all the member data of the Grid (useful during serialization of an openvdb grid)
///
/// @note The transform is assumed to be affine (so linear) and have uniform scale! So frustum transforms
///       and non-uniform scaling are not supported (primarily because they complicate ray-tracing in index space)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) GridData
{ // sizeof(GridData) = 672B
    static const int MaxNameSize = 256; // due to NULL termination the maximum length is one less
    uint64_t         mMagic; // 8B (0) magic to validate it is valid grid data.
    Checksum         mChecksum; // 8B (8). Checksum of grid buffer.
    Version          mVersion; // 4B (16) major, minor, and patch version numbers
    BitFlags<32>     mFlags; // 4B (20). flags for grid.
    uint32_t         mGridIndex; // 4B (24). Index of this grid in the buffer
    uint32_t         mGridCount; // 4B (28). Total number of grids in the buffer
    uint64_t         mGridSize; // 8B (32). byte count of this entire grid occupied in the buffer.
    char             mGridName[MaxNameSize]; // 256B (40)
    Map              mMap; // 264B (296). affine transformation between index and world space in both single and double precision
    Vec3dBBox        mWorldBBox; // 48B (560). floating-point AABB of active values in WORLD SPACE (2 x 3 doubles)
    Vec3d            mVoxelSize; // 24B (608). size of a voxel in world units
    GridClass        mGridClass; // 4B (632).
    GridType         mGridType; //  4B (636).
    int64_t          mBlindMetadataOffset; // 8B (640). offset to beginning of GridBlindMetaData structures that follow this grid.
    uint32_t         mBlindMetadataCount; // 4B (648). count of GridBlindMetaData structures that follow this grid.
    uint32_t         mData0; // 4B (652) unused
    uint64_t         mData1; // 8B (656) is use for the total number of values indexed by an IndexGrid
    uint64_t         mData2; // 8B (664) padding to 32 B alignment
    /// @brief Use this method to initiate most member data
    GridData& operator=(const GridData&) = default;
    //__hostdev__ GridData& operator=(const GridData& other){return *util::memcpy(this, &other);}
    __hostdev__ void init(std::initializer_list<GridFlags> list = {GridFlags::IsBreadthFirst},
                          uint64_t                         gridSize = 0u,
                          const Map&                       map = Map(),
                          GridType                         gridType = GridType::Unknown,
                          GridClass                        gridClass = GridClass::Unknown)
    {
#ifdef NANOVDB_USE_NEW_MAGIC_NUMBERS
        mMagic = NANOVDB_MAGIC_GRID;
#else
        mMagic = NANOVDB_MAGIC_NUMB;
#endif
        mChecksum.disable();// all 64 bits ON means checksum is disabled
        mVersion = Version();
        mFlags.initMask(list);
        mGridIndex = 0u;
        mGridCount = 1u;
        mGridSize = gridSize;
        mGridName[0] = '\0';
        mMap = map;
        mWorldBBox = Vec3dBBox();// invalid bbox
        mVoxelSize = map.getVoxelSize();
        mGridClass = gridClass;
        mGridType = gridType;
        mBlindMetadataOffset = mGridSize; // i.e. no blind data
        mBlindMetadataCount = 0u; // i.e. no blind data
        mData0 = 0u; // zero padding
        mData1 = 0u; // only used for index and point grids
#ifdef NANOVDB_USE_NEW_MAGIC_NUMBERS
        mData2 = 0u;// unused
#else
        mData2 = NANOVDB_MAGIC_GRID; // since version 32.6.0 (will change in the future)
#endif
    }
    /// @brief return true if the magic number and the version are both valid
    __hostdev__ bool isValid() const {
        // Before v32.6.0: toMagic(mMagic) = MagicType::NanoVDB  and mData2 was undefined
        // For    v32.6.0: toMagic(mMagic) = MagicType::NanoVDB  and toMagic(mData2) = MagicType::NanoGrid
        // After  v32.7.X: toMagic(mMagic) = MagicType::NanoGrid and mData2 will again be undefined
        const MagicType magic = toMagic(mMagic);
        if (magic == MagicType::NanoGrid || toMagic(mData2) == MagicType::NanoGrid) return true;
        bool test = magic == MagicType::NanoVDB;// could be GridData or io::FileHeader
        if (test) test = mVersion.isCompatible();
        if (test) test = mGridCount > 0u && mGridIndex < mGridCount;
        if (test) test = mGridClass < GridClass::End && mGridType < GridType::End;
        return test;
    }
    // Set and unset various bit flags
    __hostdev__ void setMinMaxOn(bool on = true) { mFlags.setMask(GridFlags::HasMinMax, on); }
    __hostdev__ void setBBoxOn(bool on = true) { mFlags.setMask(GridFlags::HasBBox, on); }
    __hostdev__ void setLongGridNameOn(bool on = true) { mFlags.setMask(GridFlags::HasLongGridName, on); }
    __hostdev__ void setAverageOn(bool on = true) { mFlags.setMask(GridFlags::HasAverage, on); }
    __hostdev__ void setStdDeviationOn(bool on = true) { mFlags.setMask(GridFlags::HasStdDeviation, on); }
    __hostdev__ bool setGridName(const char* src)
    {
        const bool success = (util::strncpy(mGridName, src, MaxNameSize)[MaxNameSize-1] == '\0');
        if (!success) mGridName[MaxNameSize-1] = '\0';
        return success; // returns true if input grid name is NOT longer than MaxNameSize characters
    }
    // Affine transformations based on double precision
    template<typename Vec3T>
    __hostdev__ Vec3T applyMap(const Vec3T& xyz) const { return mMap.applyMap(xyz); } // Pos: index -> world
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMap(const Vec3T& xyz) const { return mMap.applyInverseMap(xyz); } // Pos: world -> index
    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobian(const Vec3T& xyz) const { return mMap.applyJacobian(xyz); } // Dir: index -> world
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobian(const Vec3T& xyz) const { return mMap.applyInverseJacobian(xyz); } // Dir: world -> index
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJT(const Vec3T& xyz) const { return mMap.applyIJT(xyz); }
    // Affine transformations based on single precision
    template<typename Vec3T>
    __hostdev__ Vec3T applyMapF(const Vec3T& xyz) const { return mMap.applyMapF(xyz); } // Pos: index -> world
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMapF(const Vec3T& xyz) const { return mMap.applyInverseMapF(xyz); } // Pos: world -> index
    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobianF(const Vec3T& xyz) const { return mMap.applyJacobianF(xyz); } // Dir: index -> world
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobianF(const Vec3T& xyz) const { return mMap.applyInverseJacobianF(xyz); } // Dir: world -> index
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJTF(const Vec3T& xyz) const { return mMap.applyIJTF(xyz); }

    // @brief Return a non-const void pointer to the tree
    __hostdev__ void* treePtr() { return this + 1; }// TreeData is always right after GridData

    // @brief Return a const void pointer to the tree
    __hostdev__ const void* treePtr() const { return this + 1; }// TreeData is always right after GridData

    /// @brief Return a non-const void pointer to the first node at @c LEVEL
    /// @tparam LEVEL Level of the node. LEVEL 0 means leaf node and LEVEL 3 means root node
    template <uint32_t LEVEL>
    __hostdev__ const void* nodePtr() const
    {
        static_assert(LEVEL >= 0 && LEVEL <= 3, "invalid LEVEL template parameter");
        const void *treeData = this + 1;// TreeData is always right after GridData
        const uint64_t nodeOffset = *util::PtrAdd<uint64_t>(treeData, 8*LEVEL);// skip LEVEL uint64_t
        return nodeOffset ? util::PtrAdd(treeData, nodeOffset) : nullptr;
    }

    /// @brief Return a non-const void pointer to the first node at @c LEVEL
    /// @tparam LEVEL of the node. LEVEL 0 means leaf node and LEVEL 3 means root node
    /// @warning If not nodes exist at @c LEVEL NULL is returned
    template <uint32_t LEVEL>
    __hostdev__ void* nodePtr()
    {
        static_assert(LEVEL >= 0 && LEVEL <= 3, "invalid LEVEL template parameter");
        void *treeData  = this + 1;// TreeData is always right after GridData
        const uint64_t nodeOffset = *util::PtrAdd<uint64_t>(treeData, 8*LEVEL);// skip LEVEL uint64_t
        return nodeOffset ? util::PtrAdd(treeData, nodeOffset) : nullptr;
    }

    /// @brief Return number of nodes at @c LEVEL
    /// @tparam Level of the node. LEVEL 0 means leaf node and LEVEL 2 means upper node
    template <uint32_t LEVEL>
    __hostdev__ uint32_t nodeCount() const
    {
        static_assert(LEVEL >= 0 && LEVEL < 3, "invalid LEVEL template parameter");
        return *util::PtrAdd<uint32_t>(this + 1, 4*(8 + LEVEL));// TreeData is always right after GridData
    }

    /// @brief Returns a const reference to the blindMetaData at the specified linear offset.
    ///
    /// @warning The linear offset is assumed to be in the valid range
    __hostdev__ const GridBlindMetaData* blindMetaData(uint32_t n) const
    {
        NANOVDB_ASSERT(n < mBlindMetadataCount);
        return util::PtrAdd<GridBlindMetaData>(this, mBlindMetadataOffset) + n;
    }

    __hostdev__ const char* gridName() const
    {
        if (mFlags.isMaskOn(GridFlags::HasLongGridName)) {// search for first blind meta data that contains a name
            NANOVDB_ASSERT(mBlindMetadataCount > 0);
            for (uint32_t i = 0; i < mBlindMetadataCount; ++i) {
                const auto* metaData = this->blindMetaData(i);// EXTREMELY important to be a pointer
                if (metaData->mDataClass == GridBlindDataClass::GridName) {
                    NANOVDB_ASSERT(metaData->mDataType == GridType::Unknown);
                    return metaData->template getBlindData<const char>();
                }
            }
            NANOVDB_ASSERT(false); // should never hit this!
        }
        return mGridName;
    }

    /// @brief Return memory usage in bytes for this class only.
    __hostdev__ static uint64_t memUsage() { return sizeof(GridData); }

    /// @brief return AABB of active values in world space
    __hostdev__ const Vec3dBBox& worldBBox() const { return mWorldBBox; }

    /// @brief return AABB of active values in index space
    __hostdev__ const CoordBBox& indexBBox() const {return *(const CoordBBox*)(this->nodePtr<3>());}

    /// @brief return the root table has size
    __hostdev__ uint32_t rootTableSize() const
    {
        const void *root = this->nodePtr<3>();
        return root ? *util::PtrAdd<uint32_t>(root, sizeof(CoordBBox)) : 0u;
    }

    /// @brief test if the grid is empty, e.i the root table has size 0
    /// @return  true if this grid contains not data whatsoever
    __hostdev__ bool isEmpty() const {return this->rootTableSize() == 0u;}

    /// @brief  return true if RootData follows TreeData in memory without any extra padding
    /// @details TreeData is always following right after GridData, but the same might not be true for RootData
    __hostdev__ bool isRootConnected() const { return *(const uint64_t*)((const char*)(this + 1) + 24) == 64u;}
}; // GridData

// Forward declaration of accelerated random access class
template<typename BuildT, int LEVEL0 = -1, int LEVEL1 = -1, int LEVEL2 = -1>
class ReadAccessor;

template<typename BuildT>
using DefaultReadAccessor = ReadAccessor<BuildT, 0, 1, 2>;

/// @brief Highest level of the data structure. Contains a tree and a world->index
///        transform (that currently only supports uniform scaling and translation).
///
/// @note This the API of this class to interface with client code
template<typename TreeT>
class Grid : public GridData
{
public:
    using TreeType = TreeT;
    using RootType = typename TreeT::RootType;
    using RootNodeType = RootType;
    using UpperNodeType = typename RootNodeType::ChildNodeType;
    using LowerNodeType = typename UpperNodeType::ChildNodeType;
    using LeafNodeType = typename RootType::LeafNodeType;
    using DataType = GridData;
    using ValueType = typename TreeT::ValueType;
    using BuildType = typename TreeT::BuildType; // in rare cases BuildType != ValueType, e.g. then BuildType = ValueMask and ValueType = bool
    using CoordType = typename TreeT::CoordType;
    using AccessorType = DefaultReadAccessor<BuildType>;

    /// @brief Disallow constructions, copy and assignment
    ///
    /// @note Only a Serializer, defined elsewhere, can instantiate this class
    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;
    ~Grid() = delete;

    __hostdev__ Version version() const { return DataType::mVersion; }

    __hostdev__ DataType* data() { return reinterpret_cast<DataType*>(this); }

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief Return memory usage in bytes for this class only.
    //__hostdev__ static uint64_t memUsage() { return sizeof(GridData); }

    /// @brief Return the memory footprint of the entire grid, i.e. including all nodes and blind data
    __hostdev__ uint64_t gridSize() const { return DataType::mGridSize; }

    /// @brief Return index of this grid in the buffer
    __hostdev__ uint32_t gridIndex() const { return DataType::mGridIndex; }

    /// @brief Return total number of grids in the buffer
    __hostdev__ uint32_t gridCount() const { return DataType::mGridCount; }

    /// @brief  @brief Return the total number of values indexed by this IndexGrid
    ///
    /// @note This method is only defined for IndexGrid = NanoGrid<ValueIndex || ValueOnIndex || ValueIndexMask || ValueOnIndexMask>
    template<typename T = BuildType>
    __hostdev__ typename util::enable_if<BuildTraits<T>::is_index, const uint64_t&>::type
    valueCount() const { return DataType::mData1; }

    /// @brief  @brief Return the total number of points indexed by this PointGrid
    ///
    /// @note This method is only defined for PointGrid = NanoGrid<Point>
    template<typename T = BuildType>
    __hostdev__ typename util::enable_if<util::is_same<T, Point>::value, const uint64_t&>::type
    pointCount() const { return DataType::mData1; }

    /// @brief Return a const reference to the tree
    __hostdev__ const TreeT& tree() const { return *reinterpret_cast<const TreeT*>(this->treePtr()); }

    /// @brief Return a non-const reference to the tree
    __hostdev__ TreeT& tree() { return *reinterpret_cast<TreeT*>(this->treePtr()); }

    /// @brief Return a new instance of a ReadAccessor used to access values in this grid
    __hostdev__ AccessorType getAccessor() const { return AccessorType(this->tree().root()); }

    /// @brief Return a const reference to the size of a voxel in world units
    __hostdev__ const Vec3d& voxelSize() const { return DataType::mVoxelSize; }

    /// @brief Return a const reference to the Map for this grid
    __hostdev__ const Map& map() const { return DataType::mMap; }

    /// @brief world to index space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndex(const Vec3T& xyz) const { return this->applyInverseMap(xyz); }

    /// @brief index to world space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorld(const Vec3T& xyz) const { return this->applyMap(xyz); }

    /// @brief transformation from index space direction to world space direction
    /// @warning assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldDir(const Vec3T& dir) const { return this->applyJacobian(dir); }

    /// @brief transformation from world space direction to index space direction
    /// @warning assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndexDir(const Vec3T& dir) const { return this->applyInverseJacobian(dir); }

    /// @brief transform the gradient from index space to world space.
    /// @details Applies the inverse jacobian transform map.
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldGrad(const Vec3T& grad) const { return this->applyIJT(grad); }

    /// @brief world to index space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndexF(const Vec3T& xyz) const { return this->applyInverseMapF(xyz); }

    /// @brief index to world space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldF(const Vec3T& xyz) const { return this->applyMapF(xyz); }

    /// @brief transformation from index space direction to world space direction
    /// @warning assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldDirF(const Vec3T& dir) const { return this->applyJacobianF(dir); }

    /// @brief transformation from world space direction to index space direction
    /// @warning assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndexDirF(const Vec3T& dir) const { return this->applyInverseJacobianF(dir); }

    /// @brief Transforms the gradient from index space to world space.
    /// @details Applies the inverse jacobian transform map.
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldGradF(const Vec3T& grad) const { return DataType::applyIJTF(grad); }

    /// @brief Computes a AABB of active values in world space
    //__hostdev__ const Vec3dBBox& worldBBox() const { return DataType::mWorldBBox; }

    /// @brief Computes a AABB of active values in index space
    ///
    /// @note This method is returning a floating point bounding box and not a CoordBBox. This makes
    ///       it more useful for clipping rays.
    //__hostdev__ const BBox<CoordType>& indexBBox() const { return this->tree().bbox(); }

    /// @brief Return the total number of active voxels in this tree.
    __hostdev__ uint64_t activeVoxelCount() const { return this->tree().activeVoxelCount(); }

    /// @brief Methods related to the classification of this grid
    __hostdev__ bool             isValid() const { return DataType::isValid(); }
    __hostdev__ const GridType&  gridType() const { return DataType::mGridType; }
    __hostdev__ const GridClass& gridClass() const { return DataType::mGridClass; }
    __hostdev__ bool             isLevelSet() const { return DataType::mGridClass == GridClass::LevelSet; }
    __hostdev__ bool             isFogVolume() const { return DataType::mGridClass == GridClass::FogVolume; }
    __hostdev__ bool             isStaggered() const { return DataType::mGridClass == GridClass::Staggered; }
    __hostdev__ bool             isPointIndex() const { return DataType::mGridClass == GridClass::PointIndex; }
    __hostdev__ bool             isGridIndex() const { return DataType::mGridClass == GridClass::IndexGrid; }
    __hostdev__ bool             isPointData() const { return DataType::mGridClass == GridClass::PointData; }
    __hostdev__ bool             isMask() const { return DataType::mGridClass == GridClass::Topology; }
    __hostdev__ bool             isUnknown() const { return DataType::mGridClass == GridClass::Unknown; }
    __hostdev__ bool             hasMinMax() const { return DataType::mFlags.isMaskOn(GridFlags::HasMinMax); }
    __hostdev__ bool             hasBBox() const { return DataType::mFlags.isMaskOn(GridFlags::HasBBox); }
    __hostdev__ bool             hasLongGridName() const { return DataType::mFlags.isMaskOn(GridFlags::HasLongGridName); }
    __hostdev__ bool             hasAverage() const { return DataType::mFlags.isMaskOn(GridFlags::HasAverage); }
    __hostdev__ bool             hasStdDeviation() const { return DataType::mFlags.isMaskOn(GridFlags::HasStdDeviation); }
    __hostdev__ bool             isBreadthFirst() const { return DataType::mFlags.isMaskOn(GridFlags::IsBreadthFirst); }

    /// @brief return true if the specified node type is laid out breadth-first in memory and has a fixed size.
    ///        This allows for sequential access to the nodes.
    template<typename NodeT>
    __hostdev__ bool isSequential() const { return NodeT::FIXED_SIZE && this->isBreadthFirst(); }

    /// @brief return true if the specified node level is laid out breadth-first in memory and has a fixed size.
    ///        This allows for sequential access to the nodes.
    template<int LEVEL>
    __hostdev__ bool isSequential() const { return NodeTrait<TreeT, LEVEL>::type::FIXED_SIZE && this->isBreadthFirst(); }

   /// @brief return true if nodes at all levels can safely be accessed with simple linear offsets
    __hostdev__ bool isSequential() const { return UpperNodeType::FIXED_SIZE && LowerNodeType::FIXED_SIZE && LeafNodeType::FIXED_SIZE && this->isBreadthFirst(); }

    /// @brief Return a c-string with the name of this grid
    __hostdev__ const char* gridName() const { return DataType::gridName(); }

    /// @brief Return a c-string with the name of this grid, truncated to 255 characters
    __hostdev__ const char* shortGridName() const { return DataType::mGridName; }

    /// @brief Return checksum of the grid buffer.
    __hostdev__ const Checksum& checksum() const { return DataType::mChecksum; }

    /// @brief Return true if this grid is empty, i.e. contains no values or nodes.
    //__hostdev__ bool isEmpty() const { return this->tree().isEmpty(); }

    /// @brief Return the count of blind-data encoded in this grid
    __hostdev__ uint32_t blindDataCount() const { return DataType::mBlindMetadataCount; }

    /// @brief Return the index of the first blind data with specified name if found, otherwise -1.
    __hostdev__ int findBlindData(const char* name) const;

    /// @brief Return the index of the first blind data with specified semantic if found, otherwise -1.
    __hostdev__ int findBlindDataForSemantic(GridBlindDataSemantic semantic) const;

    /// @brief Returns a const pointer to the blindData at the specified linear offset.
    ///
    /// @warning Pointer might be NULL and the linear offset is assumed to be in the valid range
    // this method is deprecated !!!!
    [[deprecated("Use Grid::getBlindData<T>() instead.")]]
    __hostdev__ const void* blindData(uint32_t n) const
    {
        printf("\nnanovdb::Grid::blindData is unsafe and hence deprecated! Please use nanovdb::Grid::getBlindData instead.\n\n");
        NANOVDB_ASSERT(n < DataType::mBlindMetadataCount);
        return this->blindMetaData(n).blindData();
    }

    template <typename BlindDataT>
     __hostdev__ const BlindDataT* getBlindData(uint32_t n) const
    {
        if (n >= DataType::mBlindMetadataCount) return nullptr;// index is out of bounds
        return this->blindMetaData(n).template getBlindData<BlindDataT>();// NULL if mismatching BlindDataT
    }

    template <typename BlindDataT>
     __hostdev__ BlindDataT* getBlindData(uint32_t n)
    {
        if (n >= DataType::mBlindMetadataCount) return nullptr;// index is out of bounds
        return const_cast<BlindDataT*>(this->blindMetaData(n).template getBlindData<BlindDataT>());// NULL if mismatching BlindDataT
    }

    __hostdev__ const GridBlindMetaData& blindMetaData(uint32_t n) const { return *DataType::blindMetaData(n); }

private:
    static_assert(sizeof(GridData) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(GridData) is misaligned");
}; // Class Grid

template<typename TreeT>
__hostdev__ int Grid<TreeT>::findBlindDataForSemantic(GridBlindDataSemantic semantic) const
{
    for (uint32_t i = 0, n = this->blindDataCount(); i < n; ++i) {
        if (this->blindMetaData(i).mSemantic == semantic)
            return int(i);
    }
    return -1;
}

template<typename TreeT>
__hostdev__ int Grid<TreeT>::findBlindData(const char* name) const
{
    auto test = [&](int n) {
        const char* str = this->blindMetaData(n).mName;
        for (int i = 0; i < GridBlindMetaData::MaxNameSize; ++i) {
            if (name[i] != str[i])
                return false;
            if (name[i] == '\0' && str[i] == '\0')
                return true;
        }
        return true; // all len characters matched
    };
    for (int i = 0, n = this->blindDataCount(); i < n; ++i)
        if (test(i))
            return i;
    return -1;
}

// ----------------------------> Tree <--------------------------------------

struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) TreeData
{ // sizeof(TreeData) == 64B
    int64_t  mNodeOffset[4];// 32B, byte offset from this tree to first leaf, lower, upper and root node. If mNodeCount[N]=0 => mNodeOffset[N]==mNodeOffset[N+1]
    uint32_t mNodeCount[3]; // 12B, total number of nodes of type: leaf, lower internal, upper internal
    uint32_t mTileCount[3]; // 12B, total number of active tile values at the lower internal, upper internal and root node levels
    uint64_t mVoxelCount; //    8B, total number of active voxels in the root and all its child nodes.
    // No padding since it's always 32B aligned
    TreeData& operator=(const TreeData&) = default;
    __hostdev__ void setRoot(const void* root) {
        NANOVDB_ASSERT(root);
        mNodeOffset[3] = util::PtrDiff(root, this);
    }

    /// @brief Get a non-const void pointer to the root node (never NULL)
    __hostdev__ void* getRoot() { return util::PtrAdd(this, mNodeOffset[3]); }

    /// @brief Get a const void pointer to the root node (never NULL)
    __hostdev__ const void* getRoot() const { return util::PtrAdd(this, mNodeOffset[3]); }

    template<typename NodeT>
    __hostdev__ void setFirstNode(const NodeT* node) {mNodeOffset[NodeT::LEVEL] = (node ? util::PtrDiff(node, this) : 0);}

    /// @brief Return true if the root is empty, i.e. has not child nodes or constant tiles
    __hostdev__ bool isEmpty() const  {return  mNodeOffset[3] ? *util::PtrAdd<uint32_t>(this, mNodeOffset[3] + sizeof(CoordBBox)) == 0 : true;}

    /// @brief Return the index bounding box of all the active values in this tree, i.e. in all nodes of the tree
    __hostdev__ CoordBBox bbox() const {return  mNodeOffset[3] ? *util::PtrAdd<CoordBBox>(this, mNodeOffset[3]) : CoordBBox();}

    /// @brief  return true if RootData is layout out immediately after TreeData in memory
    __hostdev__ bool isRootNext() const {return mNodeOffset[3] ? mNodeOffset[3] == sizeof(TreeData) : false; }
};// TreeData

// ----------------------------> GridTree <--------------------------------------

/// @brief defines a tree type from a grid type while preserving constness
template<typename GridT>
struct GridTree
{
    using Type = typename GridT::TreeType;
    using type = typename GridT::TreeType;
};
template<typename GridT>
struct GridTree<const GridT>
{
    using Type = const typename GridT::TreeType;
    using type = const typename GridT::TreeType;
};

// ----------------------------> Tree <--------------------------------------

/// @brief VDB Tree, which is a thin wrapper around a RootNode.
template<typename RootT>
class Tree : public TreeData
{
    static_assert(RootT::LEVEL == 3, "Tree depth is not supported");
    static_assert(RootT::ChildNodeType::LOG2DIM == 5, "Tree configuration is not supported");
    static_assert(RootT::ChildNodeType::ChildNodeType::LOG2DIM == 4, "Tree configuration is not supported");
    static_assert(RootT::LeafNodeType::LOG2DIM == 3, "Tree configuration is not supported");

public:
    using DataType = TreeData;
    using RootType = RootT;
    using RootNodeType = RootT;
    using UpperNodeType = typename RootNodeType::ChildNodeType;
    using LowerNodeType = typename UpperNodeType::ChildNodeType;
    using LeafNodeType = typename RootType::LeafNodeType;
    using ValueType = typename RootT::ValueType;
    using BuildType = typename RootT::BuildType; // in rare cases BuildType != ValueType, e.g. then BuildType = ValueMask and ValueType = bool
    using CoordType = typename RootT::CoordType;
    using AccessorType = DefaultReadAccessor<BuildType>;

    using Node3 = RootT;
    using Node2 = typename RootT::ChildNodeType;
    using Node1 = typename Node2::ChildNodeType;
    using Node0 = LeafNodeType;

    /// @brief This class cannot be constructed or deleted
    Tree() = delete;
    Tree(const Tree&) = delete;
    Tree& operator=(const Tree&) = delete;
    ~Tree() = delete;

    __hostdev__ DataType* data() { return reinterpret_cast<DataType*>(this); }

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief return memory usage in bytes for the class
    __hostdev__ static uint64_t memUsage() { return sizeof(DataType); }

    __hostdev__ RootT& root() {return *reinterpret_cast<RootT*>(DataType::getRoot());}

    __hostdev__ const RootT& root() const {return *reinterpret_cast<const RootT*>(DataType::getRoot());}

    __hostdev__ AccessorType getAccessor() const { return AccessorType(this->root()); }

    /// @brief Return the value of the given voxel (regardless of state or location in the tree.)
    __hostdev__ ValueType getValue(const CoordType& ijk) const { return this->root().getValue(ijk); }
    __hostdev__ ValueType getValue(int i, int j, int k) const { return this->root().getValue(CoordType(i, j, k)); }

    /// @brief Return the active state of the given voxel (regardless of state or location in the tree.)
    __hostdev__ bool isActive(const CoordType& ijk) const { return this->root().isActive(ijk); }

    /// @brief Return true if this tree is empty, i.e. contains no values or nodes
    //__hostdev__ bool isEmpty() const { return this->root().isEmpty(); }

    /// @brief Combines the previous two methods in a single call
    __hostdev__ bool probeValue(const CoordType& ijk, ValueType& v) const { return this->root().probeValue(ijk, v); }

    /// @brief Return a const reference to the background value.
    __hostdev__ const ValueType& background() const { return this->root().background(); }

    /// @brief Sets the extrema values of all the active values in this tree, i.e. in all nodes of the tree
    __hostdev__ void extrema(ValueType& min, ValueType& max) const;

    /// @brief Return a const reference to the index bounding box of all the active values in this tree, i.e. in all nodes of the tree
    //__hostdev__ const BBox<CoordType>& bbox() const { return this->root().bbox(); }

    /// @brief Return the total number of active voxels in this tree.
    __hostdev__ uint64_t activeVoxelCount() const { return DataType::mVoxelCount; }

    /// @brief   Return the total number of active tiles at the specified level of the tree.
    ///
    /// @details level = 1,2,3 corresponds to active tile count in lower internal nodes, upper
    ///          internal nodes, and the root level. Note active values at the leaf level are
    ///          referred to as active voxels (see activeVoxelCount defined above).
    __hostdev__ const uint32_t& activeTileCount(uint32_t level) const
    {
        NANOVDB_ASSERT(level > 0 && level <= 3); // 1, 2, or 3
        return DataType::mTileCount[level - 1];
    }

    template<typename NodeT>
    __hostdev__ uint32_t nodeCount() const
    {
        static_assert(NodeT::LEVEL < 3, "Invalid NodeT");
        return DataType::mNodeCount[NodeT::LEVEL];
    }

    __hostdev__ uint32_t nodeCount(int level) const
    {
        NANOVDB_ASSERT(level < 3);
        return DataType::mNodeCount[level];
    }

    __hostdev__ uint32_t totalNodeCount() const
    {
        return DataType::mNodeCount[0] + DataType::mNodeCount[1] + DataType::mNodeCount[2];
    }

    /// @brief return a pointer to the first node of the specified type
    ///
    /// @warning Note it may return NULL if no nodes exist
    template<typename NodeT>
    __hostdev__ NodeT* getFirstNode()
    {
        const int64_t nodeOffset = DataType::mNodeOffset[NodeT::LEVEL];
        return nodeOffset ? util::PtrAdd<NodeT>(this, nodeOffset) : nullptr;
    }

    /// @brief return a const pointer to the first node of the specified type
    ///
    /// @warning Note it may return NULL if no nodes exist
    template<typename NodeT>
    __hostdev__ const NodeT* getFirstNode() const
    {
        const int64_t nodeOffset = DataType::mNodeOffset[NodeT::LEVEL];
        return nodeOffset ? util::PtrAdd<NodeT>(this, nodeOffset) : nullptr;
    }

    /// @brief return a pointer to the first node at the specified level
    ///
    /// @warning Note it may return NULL if no nodes exist
    template<int LEVEL>
    __hostdev__ typename NodeTrait<RootT, LEVEL>::type* getFirstNode()
    {
        return this->template getFirstNode<typename NodeTrait<RootT, LEVEL>::type>();
    }

    /// @brief return a const pointer to the first node of the specified level
    ///
    /// @warning Note it may return NULL if no nodes exist
    template<int LEVEL>
    __hostdev__ const typename NodeTrait<RootT, LEVEL>::type* getFirstNode() const
    {
        return this->template getFirstNode<typename NodeTrait<RootT, LEVEL>::type>();
    }

    /// @brief Template specializations of getFirstNode
    __hostdev__ LeafNodeType*                             getFirstLeaf() { return this->getFirstNode<LeafNodeType>(); }
    __hostdev__ const LeafNodeType*                       getFirstLeaf() const { return this->getFirstNode<LeafNodeType>(); }
    __hostdev__ typename NodeTrait<RootT, 1>::type*       getFirstLower() { return this->getFirstNode<1>(); }
    __hostdev__ const typename NodeTrait<RootT, 1>::type* getFirstLower() const { return this->getFirstNode<1>(); }
    __hostdev__ typename NodeTrait<RootT, 2>::type*       getFirstUpper() { return this->getFirstNode<2>(); }
    __hostdev__ const typename NodeTrait<RootT, 2>::type* getFirstUpper() const { return this->getFirstNode<2>(); }

    template<typename OpT, typename... ArgsT>
    __hostdev__ auto get(const CoordType& ijk, ArgsT&&... args) const
    {
        return this->root().template get<OpT>(ijk, args...);
    }

    template<typename OpT, typename... ArgsT>
    __hostdev__ auto set(const CoordType& ijk, ArgsT&&... args)
    {
        return this->root().template set<OpT>(ijk, args...);
    }

private:
    static_assert(sizeof(DataType) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(TreeData) is misaligned");

}; // Tree class

template<typename RootT>
__hostdev__ void Tree<RootT>::extrema(ValueType& min, ValueType& max) const
{
    min = this->root().minimum();
    max = this->root().maximum();
}

// --------------------------> RootData <------------------------------------

/// @brief Struct with all the member data of the RootNode (useful during serialization of an openvdb RootNode)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!
template<typename ChildT>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) RootData
{
    using ValueT = typename ChildT::ValueType;
    using BuildT = typename ChildT::BuildType; // in rare cases BuildType != ValueType, e.g. then BuildType = ValueMask and ValueType = bool
    using CoordT = typename ChildT::CoordType;
    using StatsT = typename ChildT::FloatType;
    static constexpr bool FIXED_SIZE = false;

    /// @brief Return a key based on the coordinates of a voxel
#ifdef NANOVDB_USE_SINGLE_ROOT_KEY
    using KeyT = uint64_t;
    template<typename CoordType>
    __hostdev__ static KeyT CoordToKey(const CoordType& ijk)
    {
        static_assert(sizeof(CoordT) == sizeof(CoordType), "Mismatching sizeof");
        static_assert(32 - ChildT::TOTAL <= 21, "Cannot use 64 bit root keys");
        return (KeyT(uint32_t(ijk[2]) >> ChildT::TOTAL)) | //       z is the lower 21 bits
               (KeyT(uint32_t(ijk[1]) >> ChildT::TOTAL) << 21) | // y is the middle 21 bits
               (KeyT(uint32_t(ijk[0]) >> ChildT::TOTAL) << 42); //  x is the upper 21 bits
    }
    __hostdev__ static CoordT KeyToCoord(const KeyT& key)
    {
        static constexpr uint64_t MASK = (1u << 21) - 1; // used to mask out 21 lower bits
        return CoordT(((key >> 42) & MASK) << ChildT::TOTAL, // x are the upper 21 bits
                      ((key >> 21) & MASK) << ChildT::TOTAL, // y are the middle 21 bits
                      (key & MASK) << ChildT::TOTAL); // z are the lower 21 bits
    }
#else
    using KeyT = CoordT;
    __hostdev__ static KeyT   CoordToKey(const CoordT& ijk) { return ijk & ~ChildT::MASK; }
    __hostdev__ static CoordT KeyToCoord(const KeyT& key) { return key; }
#endif
    math::BBox<CoordT> mBBox; // 24B. AABB of active values in index space.
    uint32_t           mTableSize; // 4B. number of tiles and child pointers in the root node

    ValueT mBackground; // background value, i.e. value of any unset voxel
    ValueT mMinimum; // typically 4B, minimum of all the active values
    ValueT mMaximum; // typically 4B, maximum of all the active values
    StatsT mAverage; // typically 4B, average of all the active values in this node and its child nodes
    StatsT mStdDevi; // typically 4B, standard deviation of all the active values in this node and its child nodes

    /// @brief Return padding of this class in bytes, due to aliasing and 32B alignment
    ///
    /// @note The extra bytes are not necessarily at the end, but can come from aliasing of individual data members.
    __hostdev__ static constexpr uint32_t padding()
    {
        return sizeof(RootData) - (24 + 4 + 3 * sizeof(ValueT) + 2 * sizeof(StatsT));
    }

    struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) Tile
    {
        template<typename CoordType>
        __hostdev__ void setChild(const CoordType& k, const void* ptr, const RootData* data)
        {
            key = CoordToKey(k);
            state = false;
            child = util::PtrDiff(ptr, data);
        }
        template<typename CoordType, typename ValueType>
        __hostdev__ void setValue(const CoordType& k, bool s, const ValueType& v)
        {
            key = CoordToKey(k);
            state = s;
            value = v;
            child = 0;
        }
        __hostdev__ bool   isChild() const { return child != 0; }
        __hostdev__ bool   isValue() const { return child == 0; }
        __hostdev__ bool   isActive() const { return child == 0 && state; }
        __hostdev__ CoordT origin() const { return KeyToCoord(key); }
        KeyT               key; // NANOVDB_USE_SINGLE_ROOT_KEY ? 8B : 12B
        int64_t            child; // 8B. signed byte offset from this node to the child node.  0 means it is a constant tile, so use value.
        uint32_t           state; // 4B. state of tile value
        ValueT             value; // value of tile (i.e. no child node)
    }; // Tile

    /// @brief Returns a pointer to the tile at the specified linear offset.
    ///
    /// @warning The linear offset is assumed to be in the valid range
    __hostdev__ const Tile* tile(uint32_t n) const
    {
        NANOVDB_ASSERT(n < mTableSize);
        return reinterpret_cast<const Tile*>(this + 1) + n;
    }
    __hostdev__ Tile* tile(uint32_t n)
    {
        NANOVDB_ASSERT(n < mTableSize);
        return reinterpret_cast<Tile*>(this + 1) + n;
    }

    template<typename DataT>
    class TileIter
    {
    protected:
        using TileT = typename util::match_const<Tile,   DataT>::type;
        using NodeT = typename util::match_const<ChildT, DataT>::type;
        TileT *mBegin, *mPos, *mEnd;

    public:
        __hostdev__ TileIter() : mBegin(nullptr), mPos(nullptr), mEnd(nullptr) {}
        __hostdev__ TileIter(DataT* data, uint32_t pos = 0)
            : mBegin(reinterpret_cast<TileT*>(data + 1))// tiles reside right after the RootData
            , mPos(mBegin + pos)
            , mEnd(mBegin + data->mTableSize)
        {
            NANOVDB_ASSERT(data);
            NANOVDB_ASSERT(mBegin <= mPos);// pos > mTableSize is allowed
            NANOVDB_ASSERT(mBegin <= mEnd);// mTableSize = 0 is possible
        }
        __hostdev__ inline operator bool() const { return mPos < mEnd; }
        __hostdev__ inline auto pos() const {return mPos - mBegin; }
        __hostdev__ inline TileIter& operator++()
        {
            ++mPos;
            return *this;
        }
        __hostdev__ inline TileT& operator*() const
        {
            NANOVDB_ASSERT(mPos < mEnd);
            return *mPos;
        }
        __hostdev__ inline TileT* operator->() const
        {
            NANOVDB_ASSERT(mPos < mEnd);
            return mPos;
        }
        __hostdev__ inline DataT* data() const
        {
            NANOVDB_ASSERT(mBegin);
            return reinterpret_cast<DataT*>(mBegin) - 1;
        }
        __hostdev__ inline bool isChild() const
        {
            NANOVDB_ASSERT(mPos < mEnd);
            return mPos->child != 0;
        }
        __hostdev__ inline bool isValue() const
        {
            NANOVDB_ASSERT(mPos < mEnd);
            return mPos->child == 0;
        }
        __hostdev__ inline bool isValueOn() const
        {
            NANOVDB_ASSERT(mPos < mEnd);
            return mPos->child == 0 && mPos->state != 0;
        }
        __hostdev__ inline NodeT* child() const
        {
            NANOVDB_ASSERT(mPos < mEnd && mPos->child != 0);
            return util::PtrAdd<NodeT>(this->data(), mPos->child);// byte offset relative to RootData::this
        }
        __hostdev__ inline ValueT value() const
        {
            NANOVDB_ASSERT(mPos < mEnd && mPos->child == 0);
            return mPos->value;
        }
    };// TileIter

    using TileIterator      = TileIter<RootData>;
    using ConstTileIterator = TileIter<const RootData>;

    __hostdev__ TileIterator       beginTile()       { return      TileIterator(this); }
    __hostdev__ ConstTileIterator cbeginTile() const { return ConstTileIterator(this); }

    __hostdev__ inline TileIterator probe(const CoordT& ijk)
    {
        const auto key = CoordToKey(ijk);
        TileIterator iter(this);
        for(; iter; ++iter) if (iter->key == key) break;
        return iter;
    }

    __hostdev__ inline ConstTileIterator probe(const CoordT& ijk) const
    {
        const auto key = CoordToKey(ijk);
        ConstTileIterator iter(this);
        for(; iter; ++iter) if (iter->key == key) break;
        return iter;
    }

    __hostdev__ inline Tile* probeTile(const CoordT& ijk)
    {
        auto iter = this->probe(ijk);
        return iter ? iter.operator->() : nullptr;
    }

    __hostdev__ inline const Tile* probeTile(const CoordT& ijk) const
    {
        return const_cast<RootData*>(this)->probeTile(ijk);
    }

    __hostdev__ inline ChildT* probeChild(const CoordT& ijk)
    {
        auto iter = this->probe(ijk);
        return iter && iter.isChild() ? iter.child() : nullptr;
    }

     __hostdev__ inline const ChildT* probeChild(const CoordT& ijk) const
    {
        return const_cast<RootData*>(this)->probeChild(ijk);
    }

    /// @brief Returns a const reference to the child node in the specified tile.
    ///
    /// @warning A child node is assumed to exist in the specified tile
    __hostdev__ ChildT* getChild(const Tile* tile)
    {
        NANOVDB_ASSERT(tile->child);
        return util::PtrAdd<ChildT>(this, tile->child);
    }
    __hostdev__ const ChildT* getChild(const Tile* tile) const
    {
        NANOVDB_ASSERT(tile->child);
        return util::PtrAdd<ChildT>(this, tile->child);
    }

    __hostdev__ const ValueT& getMin() const { return mMinimum; }
    __hostdev__ const ValueT& getMax() const { return mMaximum; }
    __hostdev__ const StatsT& average() const { return mAverage; }
    __hostdev__ const StatsT& stdDeviation() const { return mStdDevi; }

    __hostdev__ void setMin(const ValueT& v) { mMinimum = v; }
    __hostdev__ void setMax(const ValueT& v) { mMaximum = v; }
    __hostdev__ void setAvg(const StatsT& v) { mAverage = v; }
    __hostdev__ void setDev(const StatsT& v) { mStdDevi = v; }

    /// @brief This class cannot be constructed or deleted
    RootData() = delete;
    RootData(const RootData&) = delete;
    RootData& operator=(const RootData&) = delete;
    ~RootData() = delete;
}; // RootData

// --------------------------> RootNode <------------------------------------

/// @brief Top-most node of the VDB tree structure.
template<typename ChildT>
class RootNode : public RootData<ChildT>
{
public:
    using DataType = RootData<ChildT>;
    using ChildNodeType = ChildT;
    using RootType = RootNode<ChildT>; // this allows RootNode to behave like a Tree
    using RootNodeType = RootType;
    using UpperNodeType = ChildT;
    using LowerNodeType = typename UpperNodeType::ChildNodeType;
    using LeafNodeType = typename ChildT::LeafNodeType;
    using ValueType = typename DataType::ValueT;
    using FloatType = typename DataType::StatsT;
    using BuildType = typename DataType::BuildT; // in rare cases BuildType != ValueType, e.g. then BuildType = ValueMask and ValueType = bool

    using CoordType = typename ChildT::CoordType;
    using BBoxType = math::BBox<CoordType>;
    using AccessorType = DefaultReadAccessor<BuildType>;
    using Tile = typename DataType::Tile;
    static constexpr bool FIXED_SIZE = DataType::FIXED_SIZE;

    static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL; // level 0 = leaf

    template<typename RootT>
    class BaseIter
    {
    protected:
        using DataT = typename util::match_const<DataType, RootT>::type;
        using TileT = typename util::match_const<Tile, RootT>::type;
        typename DataType::template TileIter<DataT> mTileIter;
        __hostdev__ BaseIter() : mTileIter() {}
        __hostdev__ BaseIter(DataT* data) : mTileIter(data){}

    public:
        __hostdev__ operator bool() const { return bool(mTileIter); }
        __hostdev__ uint32_t  pos() const { return uint32_t(mTileIter.pos()); }
        __hostdev__ TileT*    tile() const { return mTileIter.operator->(); }
        __hostdev__ CoordType getOrigin() const {return mTileIter->origin();}
        __hostdev__ CoordType getCoord()  const {return this->getOrigin();}
    }; // Member class BaseIter

    template<typename RootT>
    class ChildIter : public BaseIter<RootT>
    {
        static_assert(util::is_same<typename util::remove_const<RootT>::type, RootNode>::value, "Invalid RootT");
        using BaseT = BaseIter<RootT>;
        using NodeT = typename util::match_const<ChildT, RootT>::type;
        using BaseT::mTileIter;

    public:
        __hostdev__ ChildIter() : BaseT() {}
        __hostdev__ ChildIter(RootT* parent) : BaseT(parent->data())
        {
            while (mTileIter && mTileIter.isValue()) ++mTileIter;
        }
        __hostdev__ NodeT& operator*()  const {return *mTileIter.child();}
        __hostdev__ NodeT* operator->() const {return  mTileIter.child();}
        __hostdev__ ChildIter& operator++()
        {
            ++mTileIter;
            while (mTileIter && mTileIter.isValue()) ++mTileIter;
            return *this;
        }
        __hostdev__ ChildIter operator++(int)
        {
            auto tmp = *this;
            this->operator++();
            return tmp;
        }
    }; // Member class ChildIter

    using ChildIterator      = ChildIter<RootNode>;
    using ConstChildIterator = ChildIter<const RootNode>;

    __hostdev__ ChildIterator       beginChild()       { return      ChildIterator(this); }
    __hostdev__ ConstChildIterator cbeginChild() const { return ConstChildIterator(this); }

    template<typename RootT>
    class ValueIter : public BaseIter<RootT>
    {
        using BaseT = BaseIter<RootT>;
        using BaseT::mTileIter;

    public:
        __hostdev__ ValueIter() : BaseT(){}
        __hostdev__ ValueIter(RootT* parent) : BaseT(parent->data())
        {
            while (mTileIter && mTileIter.isChild()) ++mTileIter;
        }
        __hostdev__ ValueType operator*() const {return mTileIter.value();}
        __hostdev__ bool isActive() const {return mTileIter.isValueOn();}
        __hostdev__ ValueIter& operator++()
        {
            ++mTileIter;
            while (mTileIter && mTileIter.isChild()) ++mTileIter;
            return *this;
        }
        __hostdev__ ValueIter operator++(int)
        {
            auto tmp = *this;
            this->operator++();
            return tmp;
        }
    }; // Member class ValueIter

    using ValueIterator = ValueIter<RootNode>;
    using ConstValueIterator = ValueIter<const RootNode>;

    __hostdev__ ValueIterator       beginValue()          { return      ValueIterator(this); }
    __hostdev__ ConstValueIterator cbeginValueAll() const { return ConstValueIterator(this); }

    template<typename RootT>
    class ValueOnIter : public BaseIter<RootT>
    {
        using BaseT = BaseIter<RootT>;
        using BaseT::mTileIter;

    public:
        __hostdev__ ValueOnIter() : BaseT(){}
        __hostdev__ ValueOnIter(RootT* parent) : BaseT(parent->data())
        {
            while (mTileIter && !mTileIter.isValueOn()) ++mTileIter;
        }
        __hostdev__ ValueType operator*() const {return mTileIter.value();}
        __hostdev__ ValueOnIter& operator++()
        {
            ++mTileIter;
            while (mTileIter && !mTileIter.isValueOn()) ++mTileIter;
            return *this;
        }
        __hostdev__ ValueOnIter operator++(int)
        {
            auto tmp = *this;
            this->operator++();
            return tmp;
        }
    }; // Member class ValueOnIter

    using ValueOnIterator = ValueOnIter<RootNode>;
    using ConstValueOnIterator = ValueOnIter<const RootNode>;

    __hostdev__ ValueOnIterator       beginValueOn()       { return      ValueOnIterator(this); }
    __hostdev__ ConstValueOnIterator cbeginValueOn() const { return ConstValueOnIterator(this); }

    template<typename RootT>
    class DenseIter : public BaseIter<RootT>
    {
        using BaseT = BaseIter<RootT>;
        using NodeT = typename util::match_const<ChildT, RootT>::type;
        using BaseT::mTileIter;

    public:
        __hostdev__ DenseIter() : BaseT(){}
        __hostdev__ DenseIter(RootT* parent) : BaseT(parent->data()){}
        __hostdev__ NodeT* probeChild(ValueType& value) const
        {
            if (mTileIter.isChild()) return mTileIter.child();
            value = mTileIter.value();
            return nullptr;
        }
        __hostdev__ bool isValueOn() const{return mTileIter.isValueOn();}
        __hostdev__ DenseIter& operator++()
        {
            ++mTileIter;
            return *this;
        }
        __hostdev__ DenseIter operator++(int)
        {
            auto tmp = *this;
            ++mTileIter;
            return tmp;
        }
    }; // Member class DenseIter

    using DenseIterator      = DenseIter<RootNode>;
    using ConstDenseIterator = DenseIter<const RootNode>;

    __hostdev__ DenseIterator       beginDense()          { return      DenseIterator(this); }
    __hostdev__ ConstDenseIterator cbeginDense() const    { return ConstDenseIterator(this); }
    __hostdev__ ConstDenseIterator cbeginChildAll() const { return ConstDenseIterator(this); }

    /// @brief This class cannot be constructed or deleted
    RootNode() = delete;
    RootNode(const RootNode&) = delete;
    RootNode& operator=(const RootNode&) = delete;
    ~RootNode() = delete;

    __hostdev__ AccessorType getAccessor() const { return AccessorType(*this); }

    __hostdev__ DataType* data() { return reinterpret_cast<DataType*>(this); }

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief Return a const reference to the index bounding box of all the active values in this tree, i.e. in all nodes of the tree
    __hostdev__ const BBoxType& bbox() const { return DataType::mBBox; }

    /// @brief Return the total number of active voxels in the root and all its child nodes.

    /// @brief Return a const reference to the background value, i.e. the value associated with
    ///        any coordinate location that has not been set explicitly.
    __hostdev__ const ValueType& background() const { return DataType::mBackground; }

    /// @brief Return the number of tiles encoded in this root node
    __hostdev__ const uint32_t& tileCount() const { return DataType::mTableSize; }
    __hostdev__ const uint32_t& getTableSize() const { return DataType::mTableSize; }

    /// @brief Return a const reference to the minimum active value encoded in this root node and any of its child nodes
    __hostdev__ const ValueType& minimum() const { return DataType::mMinimum; }

    /// @brief Return a const reference to the maximum active value encoded in this root node and any of its child nodes
    __hostdev__ const ValueType& maximum() const { return DataType::mMaximum; }

    /// @brief Return a const reference to the average of all the active values encoded in this root node and any of its child nodes
    __hostdev__ const FloatType& average() const { return DataType::mAverage; }

    /// @brief Return the variance of all the active values encoded in this root node and any of its child nodes
    __hostdev__ FloatType variance() const { return math::Pow2(DataType::mStdDevi); }

    /// @brief Return a const reference to the standard deviation of all the active values encoded in this root node and any of its child nodes
    __hostdev__ const FloatType& stdDeviation() const { return DataType::mStdDevi; }

    /// @brief Return the expected memory footprint in bytes with the specified number of tiles
    __hostdev__ static uint64_t memUsage(uint32_t tableSize) { return sizeof(RootNode) + tableSize * sizeof(Tile); }

    /// @brief Return the actual memory footprint of this root node
    __hostdev__ uint64_t memUsage() const { return sizeof(RootNode) + DataType::mTableSize * sizeof(Tile); }

    /// @brief Return true if this RootNode is empty, i.e. contains no values or nodes
    __hostdev__ bool isEmpty() const { return DataType::mTableSize == uint32_t(0); }

    /// @brief Return the value of the given voxel
    __hostdev__ ValueType getValue(const CoordType& ijk) const { return this->template get<GetValue<BuildType>>(ijk); }
    __hostdev__ ValueType getValue(int i, int j, int k) const { return this->template get<GetValue<BuildType>>(CoordType(i, j, k)); }
    __hostdev__ bool      isActive(const CoordType& ijk) const { return this->template get<GetState<BuildType>>(ijk); }
    /// @brief return the state and updates the value of the specified voxel
    __hostdev__ bool                probeValue(const CoordType& ijk, ValueType& v) const { return this->template get<ProbeValue<BuildType>>(ijk, v); }
    __hostdev__ const LeafNodeType* probeLeaf(const CoordType& ijk) const { return this->template get<GetLeaf<BuildType>>(ijk); }

    template<typename OpT, typename... ArgsT>
    __hostdev__ typename OpT::Type get(const CoordType& ijk, ArgsT&&... args) const
    {
        if (const Tile* tile = this->probeTile(ijk)) {
            if constexpr(OpT::LEVEL < LEVEL) if (tile->isChild()) return this->getChild(tile)->template get<OpT>(ijk, args...);
            return OpT::get(*tile, args...);
        }
        return OpT::get(*this, args...);
    }

    template<typename OpT, typename... ArgsT>
    __hostdev__ void set(const CoordType& ijk, ArgsT&&... args)
    {
        if (Tile* tile = DataType::probeTile(ijk)) {
            if constexpr(OpT::LEVEL < LEVEL) if (tile->isChild()) return this->getChild(tile)->template set<OpT>(ijk, args...);
            return OpT::set(*tile, args...);
        }
        return OpT::set(*this, args...);
    }

private:
    static_assert(sizeof(DataType) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(RootData) is misaligned");
    static_assert(sizeof(typename DataType::Tile) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(RootData::Tile) is misaligned");

    template<typename, int, int, int>
    friend class ReadAccessor;

    template<typename>
    friend class Tree;

    template<typename RayT, typename AccT>
    __hostdev__ uint32_t getDimAndCache(const CoordType& ijk, const RayT& ray, const AccT& acc) const
    {
        if (const Tile* tile = this->probeTile(ijk)) {
            if (tile->isChild()) {
                const auto* child = this->getChild(tile);
                acc.insert(ijk, child);
                return child->getDimAndCache(ijk, ray, acc);
            }
            return 1 << ChildT::TOTAL; //tile value
        }
        return ChildNodeType::dim(); // background
    }

    template<typename OpT, typename AccT, typename... ArgsT>
    __hostdev__ typename OpT::Type getAndCache(const CoordType& ijk, const AccT& acc, ArgsT&&... args) const
    {
        if (const Tile* tile = this->probeTile(ijk)) {
            if constexpr(OpT::LEVEL < LEVEL) {
                if (tile->isChild()) {
                    const ChildT* child = this->getChild(tile);
                    acc.insert(ijk, child);
                    return child->template getAndCache<OpT>(ijk, acc, args...);
                }
            }
            return OpT::get(*tile, args...);
        }
        return OpT::get(*this, args...);
    }

    template<typename OpT, typename AccT, typename... ArgsT>
    __hostdev__ void setAndCache(const CoordType& ijk, const AccT& acc, ArgsT&&... args)
    {
        if (Tile* tile = DataType::probeTile(ijk)) {
            if constexpr(OpT::LEVEL < LEVEL) {
                if (tile->isChild()) {
                    ChildT* child = this->getChild(tile);
                    acc.insert(ijk, child);
                    return child->template setAndCache<OpT>(ijk, acc, args...);
                }
            }
            return OpT::set(*tile, args...);
        }
        return OpT::set(*this, args...);
    }

}; // RootNode class

// After the RootNode the memory layout is assumed to be the sorted Tiles

// --------------------------> InternalNode <------------------------------------

/// @brief Struct with all the member data of the InternalNode (useful during serialization of an openvdb InternalNode)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!
template<typename ChildT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) InternalData
{
    using ValueT = typename ChildT::ValueType;
    using BuildT = typename ChildT::BuildType; // in rare cases BuildType != ValueType, e.g. then BuildType = ValueMask and ValueType = bool
    using StatsT = typename ChildT::FloatType;
    using CoordT = typename ChildT::CoordType;
    using MaskT = typename ChildT::template MaskType<LOG2DIM>;
    static constexpr bool FIXED_SIZE = true;

    union Tile
    {
        ValueT  value;
        int64_t child; //signed 64 bit byte offset relative to this InternalData, i.e. child-pointer = Tile::child + this
        /// @brief This class cannot be constructed or deleted
        Tile() = delete;
        Tile(const Tile&) = delete;
        Tile& operator=(const Tile&) = delete;
        ~Tile() = delete;
    };

    math::BBox<CoordT> mBBox; // 24B. node bounding box.                   |
    uint64_t     mFlags; // 8B. node flags.                          | 32B aligned
    MaskT        mValueMask; // LOG2DIM(5): 4096B, LOG2DIM(4): 512B  | 32B aligned
    MaskT        mChildMask; // LOG2DIM(5): 4096B, LOG2DIM(4): 512B  | 32B aligned

    ValueT mMinimum; // typically 4B
    ValueT mMaximum; // typically 4B
    StatsT mAverage; // typically 4B, average of all the active values in this node and its child nodes
    StatsT mStdDevi; // typically 4B, standard deviation of all the active values in this node and its child nodes
    // possible padding, e.g. 28 byte padding when ValueType = bool

    /// @brief Return padding of this class in bytes, due to aliasing and 32B alignment
    ///
    /// @note The extra bytes are not necessarily at the end, but can come from aliasing of individual data members.
    __hostdev__ static constexpr uint32_t padding()
    {
        return sizeof(InternalData) - (24u + 8u + 2 * (sizeof(MaskT) + sizeof(ValueT) + sizeof(StatsT)) + (1u << (3 * LOG2DIM)) * (sizeof(ValueT) > 8u ? sizeof(ValueT) : 8u));
    }
    alignas(32) Tile mTable[1u << (3 * LOG2DIM)]; // sizeof(ValueT) x (16*16*16 or 32*32*32)

    __hostdev__ static uint64_t memUsage() { return sizeof(InternalData); }

    __hostdev__ void setChild(uint32_t n, const void* ptr)
    {
        NANOVDB_ASSERT(mChildMask.isOn(n));
        mTable[n].child = util::PtrDiff(ptr, this);
    }

    template<typename ValueT>
    __hostdev__ void setValue(uint32_t n, const ValueT& v)
    {
        NANOVDB_ASSERT(!mChildMask.isOn(n));
        mTable[n].value = v;
    }

    /// @brief Returns a pointer to the child node at the specifed linear offset.
    __hostdev__ ChildT* getChild(uint32_t n)
    {
        NANOVDB_ASSERT(mChildMask.isOn(n));
        return util::PtrAdd<ChildT>(this, mTable[n].child);
    }
    __hostdev__ const ChildT* getChild(uint32_t n) const
    {
        NANOVDB_ASSERT(mChildMask.isOn(n));
        return util::PtrAdd<ChildT>(this, mTable[n].child);
    }

    __hostdev__ ValueT getValue(uint32_t n) const
    {
        NANOVDB_ASSERT(mChildMask.isOff(n));
        return mTable[n].value;
    }

    __hostdev__ bool isActive(uint32_t n) const
    {
        NANOVDB_ASSERT(mChildMask.isOff(n));
        return mValueMask.isOn(n);
    }

    __hostdev__ bool isChild(uint32_t n) const { return mChildMask.isOn(n); }

    template<typename T>
    __hostdev__ void setOrigin(const T& ijk) { mBBox[0] = ijk; }

    __hostdev__ const ValueT& getMin() const { return mMinimum; }
    __hostdev__ const ValueT& getMax() const { return mMaximum; }
    __hostdev__ const StatsT& average() const { return mAverage; }
    __hostdev__ const StatsT& stdDeviation() const { return mStdDevi; }

// GCC 11 (and possibly prior versions) has a regression that results in invalid
// warnings when -Wstringop-overflow is turned on. For details, refer to
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=101854
#if defined(__GNUC__) && (__GNUC__ < 12) && !defined(__APPLE__) && !defined(__llvm__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
    __hostdev__ void setMin(const ValueT& v) { mMinimum = v; }
    __hostdev__ void setMax(const ValueT& v) { mMaximum = v; }
    __hostdev__ void setAvg(const StatsT& v) { mAverage = v; }
    __hostdev__ void setDev(const StatsT& v) { mStdDevi = v; }
#if defined(__GNUC__) && (__GNUC__ < 12) && !defined(__APPLE__) && !defined(__llvm__)
#pragma GCC diagnostic pop
#endif

    /// @brief This class cannot be constructed or deleted
    InternalData() = delete;
    InternalData(const InternalData&) = delete;
    InternalData& operator=(const InternalData&) = delete;
    ~InternalData() = delete;
}; // InternalData

/// @brief Internal nodes of a VDB tree
template<typename ChildT, uint32_t Log2Dim = ChildT::LOG2DIM + 1>
class InternalNode : public InternalData<ChildT, Log2Dim>
{
public:
    using DataType = InternalData<ChildT, Log2Dim>;
    using ValueType = typename DataType::ValueT;
    using FloatType = typename DataType::StatsT;
    using BuildType = typename DataType::BuildT; // in rare cases BuildType != ValueType, e.g. then BuildType = ValueMask and ValueType = bool
    using LeafNodeType = typename ChildT::LeafNodeType;
    using ChildNodeType = ChildT;
    using CoordType = typename ChildT::CoordType;
    static constexpr bool FIXED_SIZE = DataType::FIXED_SIZE;
    template<uint32_t LOG2>
    using MaskType = typename ChildT::template MaskType<LOG2>;
    template<bool On>
    using MaskIterT = typename Mask<Log2Dim>::template Iterator<On>;

    static constexpr uint32_t LOG2DIM = Log2Dim;
    static constexpr uint32_t TOTAL = LOG2DIM + ChildT::TOTAL; // dimension in index space
    static constexpr uint32_t DIM = 1u << TOTAL; // number of voxels along each axis of this node
    static constexpr uint32_t SIZE = 1u << (3 * LOG2DIM); // number of tile values (or child pointers)
    static constexpr uint32_t MASK = (1u << TOTAL) - 1u;
    static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL); // total voxel count represented by this node

    /// @brief Visits child nodes of this node only
    template <typename ParentT>
    class ChildIter : public MaskIterT<true>
    {
        static_assert(util::is_same<typename util::remove_const<ParentT>::type, InternalNode>::value, "Invalid ParentT");
        using BaseT = MaskIterT<true>;
        using NodeT = typename util::match_const<ChildT, ParentT>::type;
        ParentT* mParent;

    public:
        __hostdev__ ChildIter()
            : BaseT()
            , mParent(nullptr)
        {
        }
        __hostdev__ ChildIter(ParentT* parent)
            : BaseT(parent->mChildMask.beginOn())
            , mParent(parent)
        {
        }
        ChildIter& operator=(const ChildIter&) = default;
        __hostdev__ NodeT& operator*() const
        {
            NANOVDB_ASSERT(*this);
            return *mParent->getChild(BaseT::pos());
        }
        __hostdev__ NodeT* operator->() const
        {
            NANOVDB_ASSERT(*this);
            return mParent->getChild(BaseT::pos());
        }
        __hostdev__ CoordType getOrigin() const
        {
            NANOVDB_ASSERT(*this);
            return (*this)->origin();
        }
        __hostdev__ CoordType getCoord() const {return this->getOrigin();}
    }; // Member class ChildIter

    using ChildIterator      = ChildIter<InternalNode>;
    using ConstChildIterator = ChildIter<const InternalNode>;

    __hostdev__ ChildIterator       beginChild()       { return ChildIterator(this); }
    __hostdev__ ConstChildIterator cbeginChild() const { return ConstChildIterator(this); }

    /// @brief Visits all tile values in this node, i.e. both inactive and active tiles
    class ValueIterator : public MaskIterT<false>
    {
        using BaseT = MaskIterT<false>;
        const InternalNode* mParent;

    public:
        __hostdev__ ValueIterator()
            : BaseT()
            , mParent(nullptr)
        {
        }
        __hostdev__ ValueIterator(const InternalNode* parent)
            : BaseT(parent->data()->mChildMask.beginOff())
            , mParent(parent)
        {
        }
        ValueIterator&        operator=(const ValueIterator&) = default;
        __hostdev__ ValueType operator*() const
        {
            NANOVDB_ASSERT(*this);
            return mParent->data()->getValue(BaseT::pos());
        }
        __hostdev__ CoordType getOrigin() const
        {
            NANOVDB_ASSERT(*this);
            return mParent->offsetToGlobalCoord(BaseT::pos());
        }
        __hostdev__ CoordType getCoord() const {return this->getOrigin();}
        __hostdev__ bool isActive() const
        {
            NANOVDB_ASSERT(*this);
            return mParent->data()->isActive(BaseT::mPos);
        }
    }; // Member class ValueIterator

    __hostdev__ ValueIterator beginValue() const { return ValueIterator(this); }
    __hostdev__ ValueIterator cbeginValueAll() const { return ValueIterator(this); }

    /// @brief Visits active tile values of this node only
    class ValueOnIterator : public MaskIterT<true>
    {
        using BaseT = MaskIterT<true>;
        const InternalNode* mParent;

    public:
        __hostdev__ ValueOnIterator()
            : BaseT()
            , mParent(nullptr)
        {
        }
        __hostdev__ ValueOnIterator(const InternalNode* parent)
            : BaseT(parent->data()->mValueMask.beginOn())
            , mParent(parent)
        {
        }
        ValueOnIterator&      operator=(const ValueOnIterator&) = default;
        __hostdev__ ValueType operator*() const
        {
            NANOVDB_ASSERT(*this);
            return mParent->data()->getValue(BaseT::pos());
        }
        __hostdev__ CoordType getOrigin() const
        {
            NANOVDB_ASSERT(*this);
            return mParent->offsetToGlobalCoord(BaseT::pos());
        }
        __hostdev__ CoordType getCoord() const {return this->getOrigin();}
    }; // Member class ValueOnIterator

    __hostdev__ ValueOnIterator beginValueOn() const { return ValueOnIterator(this); }
    __hostdev__ ValueOnIterator cbeginValueOn() const { return ValueOnIterator(this); }

    /// @brief Visits all tile values and child nodes of this node
    class DenseIterator : public Mask<Log2Dim>::DenseIterator
    {
        using BaseT = typename Mask<Log2Dim>::DenseIterator;
        const DataType* mParent;

    public:
        __hostdev__ DenseIterator()
            : BaseT()
            , mParent(nullptr)
        {
        }
        __hostdev__ DenseIterator(const InternalNode* parent)
            : BaseT(0)
            , mParent(parent->data())
        {
        }
        DenseIterator&            operator=(const DenseIterator&) = default;
        __hostdev__ const ChildT* probeChild(ValueType& value) const
        {
            NANOVDB_ASSERT(mParent && bool(*this));
            const ChildT* child = nullptr;
            if (mParent->mChildMask.isOn(BaseT::pos())) {
                child = mParent->getChild(BaseT::pos());
            } else {
                value = mParent->getValue(BaseT::pos());
            }
            return child;
        }
        __hostdev__ bool isValueOn() const
        {
            NANOVDB_ASSERT(mParent && bool(*this));
            return mParent->isActive(BaseT::pos());
        }
        __hostdev__ CoordType getOrigin() const
        {
            NANOVDB_ASSERT(mParent && bool(*this));
            return mParent->offsetToGlobalCoord(BaseT::pos());
        }
        __hostdev__ CoordType getCoord() const {return this->getOrigin();}
    }; // Member class DenseIterator

    __hostdev__ DenseIterator beginDense() const { return DenseIterator(this); }
    __hostdev__ DenseIterator cbeginChildAll() const { return DenseIterator(this); } // matches openvdb

    /// @brief This class cannot be constructed or deleted
    InternalNode() = delete;
    InternalNode(const InternalNode&) = delete;
    InternalNode& operator=(const InternalNode&) = delete;
    ~InternalNode() = delete;

    __hostdev__ DataType* data() { return reinterpret_cast<DataType*>(this); }

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief Return the dimension, in voxel units, of this internal node (typically 8*16 or 8*16*32)
    __hostdev__ static uint32_t dim() { return 1u << TOTAL; }

    /// @brief Return memory usage in bytes for the class
    __hostdev__ static size_t memUsage() { return DataType::memUsage(); }

    /// @brief Return a const reference to the bit mask of active voxels in this internal node
    __hostdev__ const MaskType<LOG2DIM>& valueMask() const { return DataType::mValueMask; }
    __hostdev__ const MaskType<LOG2DIM>& getValueMask() const { return DataType::mValueMask; }

    /// @brief Return a const reference to the bit mask of child nodes in this internal node
    __hostdev__ const MaskType<LOG2DIM>& childMask() const { return DataType::mChildMask; }
    __hostdev__ const MaskType<LOG2DIM>& getChildMask() const { return DataType::mChildMask; }

    /// @brief Return the origin in index space of this leaf node
    __hostdev__ CoordType origin() const { return DataType::mBBox.min() & ~MASK; }

    /// @brief Return a const reference to the minimum active value encoded in this internal node and any of its child nodes
    __hostdev__ const ValueType& minimum() const { return this->getMin(); }

    /// @brief Return a const reference to the maximum active value encoded in this internal node and any of its child nodes
    __hostdev__ const ValueType& maximum() const { return this->getMax(); }

    /// @brief Return a const reference to the average of all the active values encoded in this internal node and any of its child nodes
    __hostdev__ const FloatType& average() const { return DataType::mAverage; }

    /// @brief Return the variance of all the active values encoded in this internal node and any of its child nodes
    __hostdev__ FloatType variance() const { return DataType::mStdDevi * DataType::mStdDevi; }

    /// @brief Return a const reference to the standard deviation of all the active values encoded in this internal node and any of its child nodes
    __hostdev__ const FloatType& stdDeviation() const { return DataType::mStdDevi; }

    /// @brief Return a const reference to the bounding box in index space of active values in this internal node and any of its child nodes
    __hostdev__ const math::BBox<CoordType>& bbox() const { return DataType::mBBox; }

    /// @brief If the first entry in this node's table is a tile, return the tile's value.
    ///        Otherwise, return the result of calling getFirstValue() on the child.
    __hostdev__ ValueType getFirstValue() const
    {
        return DataType::mChildMask.isOn(0) ? this->getChild(0)->getFirstValue() : DataType::getValue(0);
    }

    /// @brief If the last entry in this node's table is a tile, return the tile's value.
    ///        Otherwise, return the result of calling getLastValue() on the child.
    __hostdev__ ValueType getLastValue() const
    {
        return DataType::mChildMask.isOn(SIZE - 1) ? this->getChild(SIZE - 1)->getLastValue() : DataType::getValue(SIZE - 1);
    }

    /// @brief Return the value of the given voxel
    __hostdev__ ValueType getValue(const CoordType& ijk) const { return this->template get<GetValue<BuildType>>(ijk); }
    __hostdev__ bool      isActive(const CoordType& ijk) const { return this->template get<GetState<BuildType>>(ijk); }
    /// @brief return the state and updates the value of the specified voxel
    __hostdev__ bool                probeValue(const CoordType& ijk, ValueType& v) const { return this->template get<ProbeValue<BuildType>>(ijk, v); }
    __hostdev__ const LeafNodeType* probeLeaf(const CoordType& ijk) const { return this->template get<GetLeaf<BuildType>>(ijk); }

    __hostdev__ ChildNodeType* probeChild(const CoordType& ijk)
    {
        const uint32_t n = CoordToOffset(ijk);
        return DataType::mChildMask.isOn(n) ? this->getChild(n) : nullptr;
    }
    __hostdev__ const ChildNodeType* probeChild(const CoordType& ijk) const
    {
        const uint32_t n = CoordToOffset(ijk);
        return DataType::mChildMask.isOn(n) ? this->getChild(n) : nullptr;
    }

    /// @brief Return the linear offset corresponding to the given coordinate
    __hostdev__ static uint32_t CoordToOffset(const CoordType& ijk)
    {
        return (((ijk[0] & MASK) >> ChildT::TOTAL) << (2 * LOG2DIM)) | // note, we're using bitwise OR instead of +
               (((ijk[1] & MASK) >> ChildT::TOTAL) << (LOG2DIM)) |
               ((ijk[2] & MASK) >> ChildT::TOTAL);
    }

    /// @return the local coordinate of the n'th tile or child node
    __hostdev__ static Coord OffsetToLocalCoord(uint32_t n)
    {
        NANOVDB_ASSERT(n < SIZE);
        const uint32_t m = n & ((1 << 2 * LOG2DIM) - 1);
        return Coord(n >> 2 * LOG2DIM, m >> LOG2DIM, m & ((1 << LOG2DIM) - 1));
    }

    /// @brief modifies local coordinates to global coordinates of a tile or child node
    __hostdev__ void localToGlobalCoord(Coord& ijk) const
    {
        ijk <<= ChildT::TOTAL;
        ijk += this->origin();
    }

    __hostdev__ Coord offsetToGlobalCoord(uint32_t n) const
    {
        Coord ijk = InternalNode::OffsetToLocalCoord(n);
        this->localToGlobalCoord(ijk);
        return ijk;
    }

    /// @brief Return true if this node or any of its child nodes contain active values
    __hostdev__ bool isActive() const { return DataType::mFlags & uint32_t(2); }

    template<typename OpT, typename... ArgsT>
    __hostdev__ typename OpT::Type get(const CoordType& ijk, ArgsT&&... args) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if constexpr(OpT::LEVEL < LEVEL) if (this->isChild(n)) return this->getChild(n)->template get<OpT>(ijk, args...);
        return OpT::get(*this, n, args...);
    }

    template<typename OpT, typename... ArgsT>
    __hostdev__ void set(const CoordType& ijk, ArgsT&&... args)
    {
        const uint32_t n = CoordToOffset(ijk);
        if constexpr(OpT::LEVEL < LEVEL) if (this->isChild(n)) return this->getChild(n)->template set<OpT>(ijk, args...);
        return OpT::set(*this, n, args...);
    }

private:
    static_assert(sizeof(DataType) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(InternalData) is misaligned");

    template<typename, int, int, int>
    friend class ReadAccessor;

    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;

    template<typename RayT, typename AccT>
    __hostdev__ uint32_t getDimAndCache(const CoordType& ijk, const RayT& ray, const AccT& acc) const
    {
        if (DataType::mFlags & uint32_t(1u))
            return this->dim(); // skip this node if the 1st bit is set
        //if (!ray.intersects( this->bbox() )) return 1<<TOTAL;

        const uint32_t n = CoordToOffset(ijk);
        if (DataType::mChildMask.isOn(n)) {
            const ChildT* child = this->getChild(n);
            acc.insert(ijk, child);
            return child->getDimAndCache(ijk, ray, acc);
        }
        return ChildNodeType::dim(); // tile value
    }

    template<typename OpT, typename AccT, typename... ArgsT>
    __hostdev__ typename OpT::Type getAndCache(const CoordType& ijk, const AccT& acc, ArgsT&&... args) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if constexpr(OpT::LEVEL < LEVEL) {
            if (this->isChild(n)) {
                const ChildT* child = this->getChild(n);
                acc.insert(ijk, child);
                return child->template getAndCache<OpT>(ijk, acc, args...);
            }
        }
        return OpT::get(*this, n, args...);
    }

    template<typename OpT, typename AccT, typename... ArgsT>
    __hostdev__ void setAndCache(const CoordType& ijk, const AccT& acc, ArgsT&&... args)
    {
        const uint32_t n = CoordToOffset(ijk);
        if constexpr(OpT::LEVEL < LEVEL) {
            if (this->isChild(n)) {
                ChildT* child = this->getChild(n);
                acc.insert(ijk, child);
                return child->template setAndCache<OpT>(ijk, acc, args...);
            }
        }
        return OpT::set(*this, n, args...);
    }

}; // InternalNode class

// --------------------------> LeafData<T> <------------------------------------

/// @brief Stuct with all the member data of the LeafNode (useful during serialization of an openvdb LeafNode)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!
template<typename ValueT, typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData
{
    static_assert(sizeof(CoordT) == sizeof(Coord), "Mismatching sizeof");
    static_assert(sizeof(MaskT<LOG2DIM>) == sizeof(Mask<LOG2DIM>), "Mismatching sizeof");
    using ValueType = ValueT;
    using BuildType = ValueT;
    using FloatType = typename FloatTraits<ValueT>::FloatType;
    using ArrayType = ValueT; // type used for the internal mValue array
    static constexpr bool FIXED_SIZE = true;

    CoordT         mBBoxMin; // 12B.
    uint8_t        mBBoxDif[3]; // 3B.
    uint8_t        mFlags; // 1B. bit0: skip render?, bit1: has bbox?, bit3: unused, bit4: has stats, bits5,6,7: bit-width for FpN
    MaskT<LOG2DIM> mValueMask; // LOG2DIM(3): 64B.

    ValueType mMinimum; // typically 4B
    ValueType mMaximum; // typically 4B
    FloatType mAverage; // typically 4B, average of all the active values in this node and its child nodes
    FloatType mStdDevi; // typically 4B, standard deviation of all the active values in this node and its child nodes
    alignas(32) ValueType mValues[1u << 3 * LOG2DIM];

    /// @brief Return padding of this class in bytes, due to aliasing and 32B alignment
    ///
    /// @note The extra bytes are not necessarily at the end, but can come from aliasing of individual data members.
    __hostdev__ static constexpr uint32_t padding()
    {
        return sizeof(LeafData) - (12 + 3 + 1 + sizeof(MaskT<LOG2DIM>) + 2 * (sizeof(ValueT) + sizeof(FloatType)) + (1u << (3 * LOG2DIM)) * sizeof(ValueT));
    }
    __hostdev__ static uint64_t memUsage() { return sizeof(LeafData); }

    __hostdev__ static bool hasStats() { return true; }

    __hostdev__ ValueType getValue(uint32_t i) const { return mValues[i]; }
    __hostdev__ void      setValueOnly(uint32_t offset, const ValueType& value) { mValues[offset] = value; }
    __hostdev__ void      setValue(uint32_t offset, const ValueType& value)
    {
        mValueMask.setOn(offset);
        mValues[offset] = value;
    }
    __hostdev__ void setOn(uint32_t offset) { mValueMask.setOn(offset); }

    __hostdev__ ValueType getMin() const { return mMinimum; }
    __hostdev__ ValueType getMax() const { return mMaximum; }
    __hostdev__ FloatType getAvg() const { return mAverage; }
    __hostdev__ FloatType getDev() const { return mStdDevi; }

// GCC 11 (and possibly prior versions) has a regression that results in invalid
// warnings when -Wstringop-overflow is turned on. For details, refer to
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=101854
#if defined(__GNUC__) && (__GNUC__ < 12) && !defined(__APPLE__) && !defined(__llvm__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
    __hostdev__ void setMin(const ValueType& v) { mMinimum = v; }
    __hostdev__ void setMax(const ValueType& v) { mMaximum = v; }
    __hostdev__ void setAvg(const FloatType& v) { mAverage = v; }
    __hostdev__ void setDev(const FloatType& v) { mStdDevi = v; }
#if defined(__GNUC__) && (__GNUC__ < 12) && !defined(__APPLE__) && !defined(__llvm__)
#pragma GCC diagnostic pop
#endif

    template<typename T>
    __hostdev__ void setOrigin(const T& ijk) { mBBoxMin = ijk; }

    __hostdev__ void fill(const ValueType& v)
    {
        for (auto *p = mValues, *q = p + 512; p != q; ++p)
            *p = v;
    }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<ValueT>

// --------------------------> LeafFnBase <------------------------------------

/// @brief Base-class for quantized float leaf nodes
template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafFnBase
{
    static_assert(sizeof(CoordT) == sizeof(Coord), "Mismatching sizeof");
    static_assert(sizeof(MaskT<LOG2DIM>) == sizeof(Mask<LOG2DIM>), "Mismatching sizeof");
    using ValueType = float;
    using FloatType = float;

    CoordT         mBBoxMin; // 12B.
    uint8_t        mBBoxDif[3]; // 3B.
    uint8_t        mFlags; // 1B. bit0: skip render?, bit1: has bbox?, bit3: unused, bit4: has stats, bits5,6,7: bit-width for FpN
    MaskT<LOG2DIM> mValueMask; // LOG2DIM(3): 64B.

    float    mMinimum; //  4B - minimum of ALL values in this node
    float    mQuantum; //  = (max - min)/15 4B
    uint16_t mMin, mMax, mAvg, mDev; // quantized representations of statistics of active values
    // no padding since it's always 32B aligned
    __hostdev__ static uint64_t memUsage() { return sizeof(LeafFnBase); }

    __hostdev__ static bool hasStats() { return true; }

    /// @brief Return padding of this class in bytes, due to aliasing and 32B alignment
    ///
    /// @note The extra bytes are not necessarily at the end, but can come from aliasing of individual data members.
    __hostdev__ static constexpr uint32_t padding()
    {
        return sizeof(LeafFnBase) - (12 + 3 + 1 + sizeof(MaskT<LOG2DIM>) + 2 * 4 + 4 * 2);
    }
    __hostdev__ void init(float min, float max, uint8_t bitWidth)
    {
        mMinimum = min;
        mQuantum = (max - min) / float((1 << bitWidth) - 1);
    }

    __hostdev__ void setOn(uint32_t offset) { mValueMask.setOn(offset); }

    /// @brief return the quantized minimum of the active values in this node
    __hostdev__ float getMin() const { return mMin * mQuantum + mMinimum; }

    /// @brief return the quantized maximum of the active values in this node
    __hostdev__ float getMax() const { return mMax * mQuantum + mMinimum; }

    /// @brief return the quantized average of the active values in this node
    __hostdev__ float getAvg() const { return mAvg * mQuantum + mMinimum; }
    /// @brief return the quantized standard deviation of the active values in this node

    /// @note 0 <= StdDev <= max-min or 0 <= StdDev/(max-min) <= 1
    __hostdev__ float getDev() const { return mDev * mQuantum; }

    /// @note min <= X <= max or 0 <= (X-min)/(min-max) <= 1
    __hostdev__ void setMin(float min) { mMin = uint16_t((min - mMinimum) / mQuantum + 0.5f); }

    /// @note min <= X <= max or 0 <= (X-min)/(min-max) <= 1
    __hostdev__ void setMax(float max) { mMax = uint16_t((max - mMinimum) / mQuantum + 0.5f); }

    /// @note min <= avg <= max or 0 <= (avg-min)/(min-max) <= 1
    __hostdev__ void setAvg(float avg) { mAvg = uint16_t((avg - mMinimum) / mQuantum + 0.5f); }

    /// @note 0 <= StdDev <= max-min or 0 <= StdDev/(max-min) <= 1
    __hostdev__ void setDev(float dev) { mDev = uint16_t(dev / mQuantum + 0.5f); }

    template<typename T>
    __hostdev__ void setOrigin(const T& ijk) { mBBoxMin = ijk; }
}; // LeafFnBase

// --------------------------> LeafData<Fp4> <------------------------------------

/// @brief Stuct with all the member data of the LeafNode (useful during serialization of an openvdb LeafNode)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!
template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<Fp4, CoordT, MaskT, LOG2DIM>
    : public LeafFnBase<CoordT, MaskT, LOG2DIM>
{
    using BaseT = LeafFnBase<CoordT, MaskT, LOG2DIM>;
    using BuildType = Fp4;
    using ArrayType = uint8_t; // type used for the internal mValue array
    static constexpr bool FIXED_SIZE = true;
    alignas(32) uint8_t mCode[1u << (3 * LOG2DIM - 1)]; // LeafFnBase is 32B aligned and so is mCode

    __hostdev__ static constexpr uint64_t memUsage() { return sizeof(LeafData); }
    __hostdev__ static constexpr uint32_t padding()
    {
        static_assert(BaseT::padding() == 0, "expected no padding in LeafFnBase");
        return sizeof(LeafData) - sizeof(BaseT) - (1u << (3 * LOG2DIM - 1));
    }

    __hostdev__ static constexpr uint8_t bitWidth() { return 4u; }
    __hostdev__ float                    getValue(uint32_t i) const
    {
#if 0
        const uint8_t c = mCode[i>>1];
        return ( (i&1) ? c >> 4 : c & uint8_t(15) )*BaseT::mQuantum + BaseT::mMinimum;
#else
        return ((mCode[i >> 1] >> ((i & 1) << 2)) & uint8_t(15)) * BaseT::mQuantum + BaseT::mMinimum;
#endif
    }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<Fp4>

// --------------------------> LeafBase<Fp8> <------------------------------------

template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<Fp8, CoordT, MaskT, LOG2DIM>
    : public LeafFnBase<CoordT, MaskT, LOG2DIM>
{
    using BaseT = LeafFnBase<CoordT, MaskT, LOG2DIM>;
    using BuildType = Fp8;
    using ArrayType = uint8_t; // type used for the internal mValue array
    static constexpr bool FIXED_SIZE = true;
    alignas(32) uint8_t mCode[1u << 3 * LOG2DIM];
    __hostdev__ static constexpr int64_t  memUsage() { return sizeof(LeafData); }
    __hostdev__ static constexpr uint32_t padding()
    {
        static_assert(BaseT::padding() == 0, "expected no padding in LeafFnBase");
        return sizeof(LeafData) - sizeof(BaseT) - (1u << 3 * LOG2DIM);
    }

    __hostdev__ static constexpr uint8_t bitWidth() { return 8u; }
    __hostdev__ float                    getValue(uint32_t i) const
    {
        return mCode[i] * BaseT::mQuantum + BaseT::mMinimum; // code * (max-min)/255 + min
    }
    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<Fp8>

// --------------------------> LeafData<Fp16> <------------------------------------

template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<Fp16, CoordT, MaskT, LOG2DIM>
    : public LeafFnBase<CoordT, MaskT, LOG2DIM>
{
    using BaseT = LeafFnBase<CoordT, MaskT, LOG2DIM>;
    using BuildType = Fp16;
    using ArrayType = uint16_t; // type used for the internal mValue array
    static constexpr bool FIXED_SIZE = true;
    alignas(32) uint16_t mCode[1u << 3 * LOG2DIM];

    __hostdev__ static constexpr uint64_t memUsage() { return sizeof(LeafData); }
    __hostdev__ static constexpr uint32_t padding()
    {
        static_assert(BaseT::padding() == 0, "expected no padding in LeafFnBase");
        return sizeof(LeafData) - sizeof(BaseT) - 2 * (1u << 3 * LOG2DIM);
    }

    __hostdev__ static constexpr uint8_t bitWidth() { return 16u; }
    __hostdev__ float                    getValue(uint32_t i) const
    {
        return mCode[i] * BaseT::mQuantum + BaseT::mMinimum; // code * (max-min)/65535 + min
    }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<Fp16>

// --------------------------> LeafData<FpN> <------------------------------------

template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<FpN, CoordT, MaskT, LOG2DIM>
    : public LeafFnBase<CoordT, MaskT, LOG2DIM>
{ // this class has no additional data members, however every instance is immediately followed by
    //  bitWidth*64 bytes. Since its base class is 32B aligned so are the bitWidth*64 bytes
    using BaseT = LeafFnBase<CoordT, MaskT, LOG2DIM>;
    using BuildType = FpN;
    static constexpr bool FIXED_SIZE = false;
    __hostdev__ static constexpr uint32_t padding()
    {
        static_assert(BaseT::padding() == 0, "expected no padding in LeafFnBase");
        return 0;
    }

    __hostdev__ uint8_t       bitWidth() const { return 1 << (BaseT::mFlags >> 5); } // 4,8,16,32 = 2^(2,3,4,5)
    __hostdev__ size_t        memUsage() const { return sizeof(*this) + this->bitWidth() * 64; }
    __hostdev__ static size_t memUsage(uint32_t bitWidth) { return 96u + bitWidth * 64; }
    __hostdev__ float         getValue(uint32_t i) const
    {
#ifdef NANOVDB_FPN_BRANCHLESS // faster
        const int b = BaseT::mFlags >> 5; // b = 0, 1, 2, 3, 4 corresponding to 1, 2, 4, 8, 16 bits
#if 0 // use LUT
        uint16_t code = reinterpret_cast<const uint16_t*>(this + 1)[i >> (4 - b)];
        const static uint8_t shift[5] = {15, 7, 3, 1, 0};
        const static uint16_t mask[5] = {1, 3, 15, 255, 65535};
        code >>= (i & shift[b]) << b;
        code  &= mask[b];
#else // no LUT
        uint32_t code = reinterpret_cast<const uint32_t*>(this + 1)[i >> (5 - b)];
        code >>= (i & ((32 >> b) - 1)) << b;
        code &= (1 << (1 << b)) - 1;
#endif
#else // use branched version (slow)
        float code;
        auto* values = reinterpret_cast<const uint8_t*>(this + 1);
        switch (BaseT::mFlags >> 5) {
        case 0u: // 1 bit float
            code = float((values[i >> 3] >> (i & 7)) & uint8_t(1));
            break;
        case 1u: // 2 bits float
            code = float((values[i >> 2] >> ((i & 3) << 1)) & uint8_t(3));
            break;
        case 2u: // 4 bits float
            code = float((values[i >> 1] >> ((i & 1) << 2)) & uint8_t(15));
            break;
        case 3u: // 8 bits float
            code = float(values[i]);
            break;
        default: // 16 bits float
            code = float(reinterpret_cast<const uint16_t*>(values)[i]);
        }
#endif
        return float(code) * BaseT::mQuantum + BaseT::mMinimum; // code * (max-min)/UNITS + min
    }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<FpN>

// --------------------------> LeafData<bool> <------------------------------------

// Partial template specialization of LeafData with bool
template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<bool, CoordT, MaskT, LOG2DIM>
{
    static_assert(sizeof(CoordT) == sizeof(Coord), "Mismatching sizeof");
    static_assert(sizeof(MaskT<LOG2DIM>) == sizeof(Mask<LOG2DIM>), "Mismatching sizeof");
    using ValueType = bool;
    using BuildType = bool;
    using FloatType = bool; // dummy value type
    using ArrayType = MaskT<LOG2DIM>; // type used for the internal mValue array
    static constexpr bool FIXED_SIZE = true;

    CoordT         mBBoxMin; // 12B.
    uint8_t        mBBoxDif[3]; // 3B.
    uint8_t        mFlags; // 1B. bit0: skip render?, bit1: has bbox?, bit3: unused, bit4: has stats, bits5,6,7: bit-width for FpN
    MaskT<LOG2DIM> mValueMask; // LOG2DIM(3): 64B.
    MaskT<LOG2DIM> mValues; // LOG2DIM(3): 64B.
    uint64_t       mPadding[2]; // 16B padding to 32B alignment

    __hostdev__ static constexpr uint32_t padding() { return sizeof(LeafData) - 12u - 3u - 1u - 2 * sizeof(MaskT<LOG2DIM>) - 16u; }
    __hostdev__ static uint64_t           memUsage() { return sizeof(LeafData); }
    __hostdev__ static bool hasStats() { return false; }
    __hostdev__ bool getValue(uint32_t i) const { return mValues.isOn(i); }
    __hostdev__ bool getMin() const { return false; } // dummy
    __hostdev__ bool getMax() const { return false; } // dummy
    __hostdev__ bool getAvg() const { return false; } // dummy
    __hostdev__ bool getDev() const { return false; } // dummy
    __hostdev__ void setValue(uint32_t offset, bool v)
    {
        mValueMask.setOn(offset);
        mValues.set(offset, v);
    }
    __hostdev__ void setOn(uint32_t offset) { mValueMask.setOn(offset); }
    __hostdev__ void setMin(const bool&) {} // no-op
    __hostdev__ void setMax(const bool&) {} // no-op
    __hostdev__ void setAvg(const bool&) {} // no-op
    __hostdev__ void setDev(const bool&) {} // no-op

    template<typename T>
    __hostdev__ void setOrigin(const T& ijk) { mBBoxMin = ijk; }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<bool>

// --------------------------> LeafData<ValueMask> <------------------------------------

// Partial template specialization of LeafData with ValueMask
template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<ValueMask, CoordT, MaskT, LOG2DIM>
{
    static_assert(sizeof(CoordT) == sizeof(Coord), "Mismatching sizeof");
    static_assert(sizeof(MaskT<LOG2DIM>) == sizeof(Mask<LOG2DIM>), "Mismatching sizeof");
    using ValueType = bool;
    using BuildType = ValueMask;
    using FloatType = bool; // dummy value type
    using ArrayType = void; // type used for the internal mValue array - void means missing
    static constexpr bool FIXED_SIZE = true;

    CoordT         mBBoxMin; // 12B.
    uint8_t        mBBoxDif[3]; // 3B.
    uint8_t        mFlags; // 1B. bit0: skip render?, bit1: has bbox?, bit3: unused, bit4: has stats, bits5,6,7: bit-width for FpN
    MaskT<LOG2DIM> mValueMask; // LOG2DIM(3): 64B.
    uint64_t       mPadding[2]; // 16B padding to 32B alignment

    __hostdev__ static uint64_t memUsage() { return sizeof(LeafData); }
    __hostdev__ static bool hasStats() { return false; }
    __hostdev__ static constexpr uint32_t padding()
    {
        return sizeof(LeafData) - (12u + 3u + 1u + sizeof(MaskT<LOG2DIM>) + 2 * 8u);
    }

    __hostdev__ bool getValue(uint32_t i) const { return mValueMask.isOn(i); }
    __hostdev__ bool getMin() const { return false; } // dummy
    __hostdev__ bool getMax() const { return false; } // dummy
    __hostdev__ bool getAvg() const { return false; } // dummy
    __hostdev__ bool getDev() const { return false; } // dummy
    __hostdev__ void setValue(uint32_t offset, bool) { mValueMask.setOn(offset); }
    __hostdev__ void setOn(uint32_t offset) { mValueMask.setOn(offset); }
    __hostdev__ void setMin(const ValueType&) {} // no-op
    __hostdev__ void setMax(const ValueType&) {} // no-op
    __hostdev__ void setAvg(const FloatType&) {} // no-op
    __hostdev__ void setDev(const FloatType&) {} // no-op

    template<typename T>
    __hostdev__ void setOrigin(const T& ijk) { mBBoxMin = ijk; }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<ValueMask>

// --------------------------> LeafIndexBase <------------------------------------

// Partial template specialization of LeafData with ValueIndex
template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafIndexBase
{
    static_assert(sizeof(CoordT) == sizeof(Coord), "Mismatching sizeof");
    static_assert(sizeof(MaskT<LOG2DIM>) == sizeof(Mask<LOG2DIM>), "Mismatching sizeof");
    using ValueType = uint64_t;
    using FloatType = uint64_t;
    using ArrayType = void; // type used for the internal mValue array - void means missing
    static constexpr bool FIXED_SIZE = true;

    CoordT         mBBoxMin; // 12B.
    uint8_t        mBBoxDif[3]; // 3B.
    uint8_t        mFlags; // 1B. bit0: skip render?, bit1: has bbox?, bit3: unused, bit4: has stats, bits5,6,7: bit-width for FpN
    MaskT<LOG2DIM> mValueMask; // LOG2DIM(3): 64B.
    uint64_t mOffset, mPrefixSum; // 8B offset to first value in this leaf node and 9-bit prefix sum
    __hostdev__ static constexpr uint32_t padding()
    {
        return sizeof(LeafIndexBase) - (12u + 3u + 1u + sizeof(MaskT<LOG2DIM>) + 2 * 8u);
    }
    __hostdev__ static uint64_t memUsage() { return sizeof(LeafIndexBase); }
    __hostdev__ bool            hasStats() const { return mFlags & (uint8_t(1) << 4); }
    // return the offset to the first value indexed by this leaf node
    __hostdev__ const uint64_t& firstOffset() const { return mOffset; }
    __hostdev__ void            setMin(const ValueType&) {} // no-op
    __hostdev__ void            setMax(const ValueType&) {} // no-op
    __hostdev__ void            setAvg(const FloatType&) {} // no-op
    __hostdev__ void            setDev(const FloatType&) {} // no-op
    __hostdev__ void            setOn(uint32_t offset) { mValueMask.setOn(offset); }
    template<typename T>
    __hostdev__ void setOrigin(const T& ijk) { mBBoxMin = ijk; }

protected:
    /// @brief This class should be used as an abstract class and only constructed or deleted via child classes
    LeafIndexBase() = default;
    LeafIndexBase(const LeafIndexBase&) = default;
    LeafIndexBase& operator=(const LeafIndexBase&) = default;
    ~LeafIndexBase() = default;
}; // LeafIndexBase

// --------------------------> LeafData<ValueIndex> <------------------------------------

// Partial template specialization of LeafData with ValueIndex
template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<ValueIndex, CoordT, MaskT, LOG2DIM>
    : public LeafIndexBase<CoordT, MaskT, LOG2DIM>
{
    using BaseT = LeafIndexBase<CoordT, MaskT, LOG2DIM>;
    using BuildType = ValueIndex;
    // return the total number of values indexed by this leaf node, excluding the optional 4 stats
    __hostdev__ static uint32_t valueCount() { return uint32_t(512); } // 8^3 = 2^9
    // return the offset to the last value indexed by this leaf node (disregarding optional stats)
    __hostdev__ uint64_t lastOffset() const { return BaseT::mOffset + 511u; } // 2^9 - 1
    // if stats are available, they are always placed after the last voxel value in this leaf node
    __hostdev__ uint64_t getMin() const { return this->hasStats() ? BaseT::mOffset + 512u : 0u; }
    __hostdev__ uint64_t getMax() const { return this->hasStats() ? BaseT::mOffset + 513u : 0u; }
    __hostdev__ uint64_t getAvg() const { return this->hasStats() ? BaseT::mOffset + 514u : 0u; }
    __hostdev__ uint64_t getDev() const { return this->hasStats() ? BaseT::mOffset + 515u : 0u; }
    __hostdev__ uint64_t getValue(uint32_t i) const { return BaseT::mOffset + i; } // dense leaf node with active and inactive voxels
}; // LeafData<ValueIndex>

// --------------------------> LeafData<ValueOnIndex> <------------------------------------

template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<ValueOnIndex, CoordT, MaskT, LOG2DIM>
    : public LeafIndexBase<CoordT, MaskT, LOG2DIM>
{
    using BaseT = LeafIndexBase<CoordT, MaskT, LOG2DIM>;
    using BuildType = ValueOnIndex;
    __hostdev__ uint32_t valueCount() const
    {
        return util::countOn(BaseT::mValueMask.words()[7]) + (BaseT::mPrefixSum >> 54u & 511u); // last 9 bits of mPrefixSum do not account for the last word in mValueMask
    }
    __hostdev__ uint64_t lastOffset() const { return BaseT::mOffset + this->valueCount() - 1u; }
    __hostdev__ uint64_t getMin() const { return this->hasStats() ? this->lastOffset() + 1u : 0u; }
    __hostdev__ uint64_t getMax() const { return this->hasStats() ? this->lastOffset() + 2u : 0u; }
    __hostdev__ uint64_t getAvg() const { return this->hasStats() ? this->lastOffset() + 3u : 0u; }
    __hostdev__ uint64_t getDev() const { return this->hasStats() ? this->lastOffset() + 4u : 0u; }
    __hostdev__ uint64_t getValue(uint32_t i) const
    {
        //return mValueMask.isOn(i) ? mOffset + mValueMask.countOn(i) : 0u;// for debugging
        uint32_t       n = i >> 6;
        const uint64_t w = BaseT::mValueMask.words()[n], mask = uint64_t(1) << (i & 63u);
        if (!(w & mask)) return uint64_t(0); // if i'th value is inactive return offset to background value
        uint64_t sum  = BaseT::mOffset + util::countOn(w & (mask - 1u));
        if (n--) sum += BaseT::mPrefixSum >> (9u * n) & 511u;
        return sum;
    }
}; // LeafData<ValueOnIndex>

// --------------------------> LeafData<ValueIndexMask> <------------------------------------

template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<ValueIndexMask, CoordT, MaskT, LOG2DIM>
    : public LeafData<ValueIndex, CoordT, MaskT, LOG2DIM>
{
    using BuildType = ValueIndexMask;
    MaskT<LOG2DIM>              mMask;
    __hostdev__ static uint64_t memUsage() { return sizeof(LeafData); }
    __hostdev__ bool            isMaskOn(uint32_t offset) const { return mMask.isOn(offset); }
    __hostdev__ void            setMask(uint32_t offset, bool v) { mMask.set(offset, v); }
}; // LeafData<ValueIndexMask>

template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<ValueOnIndexMask, CoordT, MaskT, LOG2DIM>
    : public LeafData<ValueOnIndex, CoordT, MaskT, LOG2DIM>
{
    using BuildType = ValueOnIndexMask;
    MaskT<LOG2DIM>              mMask;
    __hostdev__ static uint64_t memUsage() { return sizeof(LeafData); }
    __hostdev__ bool            isMaskOn(uint32_t offset) const { return mMask.isOn(offset); }
    __hostdev__ void            setMask(uint32_t offset, bool v) { mMask.set(offset, v); }
}; // LeafData<ValueOnIndexMask>

// --------------------------> LeafData<Point> <------------------------------------

template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<Point, CoordT, MaskT, LOG2DIM>
{
    static_assert(sizeof(CoordT) == sizeof(Coord), "Mismatching sizeof");
    static_assert(sizeof(MaskT<LOG2DIM>) == sizeof(Mask<LOG2DIM>), "Mismatching sizeof");
    using ValueType = uint64_t;
    using BuildType = Point;
    using FloatType = typename FloatTraits<ValueType>::FloatType;
    using ArrayType = uint16_t; // type used for the internal mValue array
    static constexpr bool FIXED_SIZE = true;

    CoordT         mBBoxMin; // 12B.
    uint8_t        mBBoxDif[3]; // 3B.
    uint8_t        mFlags; // 1B. bit0: skip render?, bit1: has bbox?, bit3: unused, bit4: has stats, bits5,6,7: bit-width for FpN
    MaskT<LOG2DIM> mValueMask; // LOG2DIM(3): 64B.

    uint64_t mOffset; //  8B
    uint64_t mPointCount; //  8B
    alignas(32) uint16_t mValues[1u << 3 * LOG2DIM]; // 1KB
    // no padding

    /// @brief Return padding of this class in bytes, due to aliasing and 32B alignment
    ///
    /// @note The extra bytes are not necessarily at the end, but can come from aliasing of individual data members.
    __hostdev__ static constexpr uint32_t padding()
    {
        return sizeof(LeafData) - (12u + 3u + 1u + sizeof(MaskT<LOG2DIM>) + 2 * 8u + (1u << 3 * LOG2DIM) * 2u);
    }
    __hostdev__ static uint64_t memUsage() { return sizeof(LeafData); }

    __hostdev__ uint64_t offset() const { return mOffset; }
    __hostdev__ uint64_t pointCount() const { return mPointCount; }
    __hostdev__ uint64_t first(uint32_t i) const { return i ? uint64_t(mValues[i - 1u]) + mOffset : mOffset; }
    __hostdev__ uint64_t last(uint32_t i) const { return uint64_t(mValues[i]) + mOffset; }
    __hostdev__ uint64_t getValue(uint32_t i) const { return uint64_t(mValues[i]); }
    __hostdev__ void     setValueOnly(uint32_t offset, uint16_t value) { mValues[offset] = value; }
    __hostdev__ void     setValue(uint32_t offset, uint16_t value)
    {
        mValueMask.setOn(offset);
        mValues[offset] = value;
    }
    __hostdev__ void setOn(uint32_t offset) { mValueMask.setOn(offset); }

    __hostdev__ ValueType getMin() const { return mOffset; }
    __hostdev__ ValueType getMax() const { return mPointCount; }
    __hostdev__ FloatType getAvg() const { return 0.0f; }
    __hostdev__ FloatType getDev() const { return 0.0f; }

    __hostdev__ void setMin(const ValueType&) {}
    __hostdev__ void setMax(const ValueType&) {}
    __hostdev__ void setAvg(const FloatType&) {}
    __hostdev__ void setDev(const FloatType&) {}

    template<typename T>
    __hostdev__ void setOrigin(const T& ijk) { mBBoxMin = ijk; }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<Point>

// --------------------------> LeafNode<T> <------------------------------------

/// @brief Leaf nodes of the VDB tree. (defaults to 8x8x8 = 512 voxels)
template<typename BuildT,
         typename CoordT = Coord,
         template<uint32_t> class MaskT = Mask,
         uint32_t Log2Dim = 3>
class LeafNode : public LeafData<BuildT, CoordT, MaskT, Log2Dim>
{
public:
    struct ChildNodeType
    {
        static constexpr uint32_t   TOTAL = 0;
        static constexpr uint32_t   DIM = 1;
        __hostdev__ static uint32_t dim() { return 1u; }
    }; // Voxel
    using LeafNodeType = LeafNode<BuildT, CoordT, MaskT, Log2Dim>;
    using DataType = LeafData<BuildT, CoordT, MaskT, Log2Dim>;
    using ValueType = typename DataType::ValueType;
    using FloatType = typename DataType::FloatType;
    using BuildType = typename DataType::BuildType;
    using CoordType = CoordT;
    static constexpr bool FIXED_SIZE = DataType::FIXED_SIZE;
    template<uint32_t LOG2>
    using MaskType = MaskT<LOG2>;
    template<bool ON>
    using MaskIterT = typename Mask<Log2Dim>::template Iterator<ON>;

    /// @brief Visits all active values in a leaf node
    class ValueOnIterator : public MaskIterT<true>
    {
        using BaseT = MaskIterT<true>;
        const LeafNode* mParent;

    public:
        __hostdev__ ValueOnIterator()
            : BaseT()
            , mParent(nullptr)
        {
        }
        __hostdev__ ValueOnIterator(const LeafNode* parent)
            : BaseT(parent->data()->mValueMask.beginOn())
            , mParent(parent)
        {
        }
        ValueOnIterator&      operator=(const ValueOnIterator&) = default;
        __hostdev__ ValueType operator*() const
        {
            NANOVDB_ASSERT(*this);
            return mParent->getValue(BaseT::pos());
        }
        __hostdev__ CoordT getCoord() const
        {
            NANOVDB_ASSERT(*this);
            return mParent->offsetToGlobalCoord(BaseT::pos());
        }
    }; // Member class ValueOnIterator

    __hostdev__ ValueOnIterator beginValueOn() const { return ValueOnIterator(this); }
    __hostdev__ ValueOnIterator cbeginValueOn() const { return ValueOnIterator(this); }

    /// @brief Visits all inactive values in a leaf node
    class ValueOffIterator : public MaskIterT<false>
    {
        using BaseT = MaskIterT<false>;
        const LeafNode* mParent;

    public:
        __hostdev__ ValueOffIterator()
            : BaseT()
            , mParent(nullptr)
        {
        }
        __hostdev__ ValueOffIterator(const LeafNode* parent)
            : BaseT(parent->data()->mValueMask.beginOff())
            , mParent(parent)
        {
        }
        ValueOffIterator&     operator=(const ValueOffIterator&) = default;
        __hostdev__ ValueType operator*() const
        {
            NANOVDB_ASSERT(*this);
            return mParent->getValue(BaseT::pos());
        }
        __hostdev__ CoordT getCoord() const
        {
            NANOVDB_ASSERT(*this);
            return mParent->offsetToGlobalCoord(BaseT::pos());
        }
    }; // Member class ValueOffIterator

    __hostdev__ ValueOffIterator  beginValueOff() const { return ValueOffIterator(this); }
    __hostdev__ ValueOffIterator cbeginValueOff() const { return ValueOffIterator(this); }

    /// @brief Visits all values in a leaf node, i.e. both active and inactive values
    class ValueIterator
    {
        const LeafNode* mParent;
        uint32_t        mPos;

    public:
        __hostdev__ ValueIterator()
            : mParent(nullptr)
            , mPos(1u << 3 * Log2Dim)
        {
        }
        __hostdev__ ValueIterator(const LeafNode* parent)
            : mParent(parent)
            , mPos(0)
        {
            NANOVDB_ASSERT(parent);
        }
        ValueIterator&        operator=(const ValueIterator&) = default;
        __hostdev__ ValueType operator*() const
        {
            NANOVDB_ASSERT(*this);
            return mParent->getValue(mPos);
        }
        __hostdev__ CoordT getCoord() const
        {
            NANOVDB_ASSERT(*this);
            return mParent->offsetToGlobalCoord(mPos);
        }
        __hostdev__ bool isActive() const
        {
            NANOVDB_ASSERT(*this);
            return mParent->isActive(mPos);
        }
        __hostdev__ operator bool() const { return mPos < (1u << 3 * Log2Dim); }
        __hostdev__ ValueIterator& operator++()
        {
            ++mPos;
            return *this;
        }
        __hostdev__ ValueIterator operator++(int)
        {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
    }; // Member class ValueIterator

    __hostdev__ ValueIterator  beginValue()    const { return ValueIterator(this); }
    __hostdev__ ValueIterator cbeginValueAll() const { return ValueIterator(this); }

    static_assert(util::is_same<ValueType, typename BuildToValueMap<BuildType>::Type>::value, "Mismatching BuildType");
    static constexpr uint32_t LOG2DIM = Log2Dim;
    static constexpr uint32_t TOTAL = LOG2DIM; // needed by parent nodes
    static constexpr uint32_t DIM = 1u << TOTAL; // number of voxels along each axis of this node
    static constexpr uint32_t SIZE = 1u << 3 * LOG2DIM; // total number of voxels represented by this node
    static constexpr uint32_t MASK = (1u << LOG2DIM) - 1u; // mask for bit operations
    static constexpr uint32_t LEVEL = 0; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL); // total voxel count represented by this node

    __hostdev__ DataType* data() { return reinterpret_cast<DataType*>(this); }

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief Return a const reference to the bit mask of active voxels in this leaf node
    __hostdev__ const MaskType<LOG2DIM>& valueMask() const { return DataType::mValueMask; }
    __hostdev__ const MaskType<LOG2DIM>& getValueMask() const { return DataType::mValueMask; }

    /// @brief Return a const reference to the minimum active value encoded in this leaf node
    __hostdev__ ValueType minimum() const { return DataType::getMin(); }

    /// @brief Return a const reference to the maximum active value encoded in this leaf node
    __hostdev__ ValueType maximum() const { return DataType::getMax(); }

    /// @brief Return a const reference to the average of all the active values encoded in this leaf node
    __hostdev__ FloatType average() const { return DataType::getAvg(); }

    /// @brief Return the variance of all the active values encoded in this leaf node
    __hostdev__ FloatType variance() const { return Pow2(DataType::getDev()); }

    /// @brief Return a const reference to the standard deviation of all the active values encoded in this leaf node
    __hostdev__ FloatType stdDeviation() const { return DataType::getDev(); }

    __hostdev__ uint8_t flags() const { return DataType::mFlags; }

    /// @brief Return the origin in index space of this leaf node
    __hostdev__ CoordT origin() const { return DataType::mBBoxMin & ~MASK; }

    /// @brief  Compute the local coordinates from a linear offset
    /// @param n Linear offset into this nodes dense table
    /// @return Local (vs global) 3D coordinates
    __hostdev__ static CoordT OffsetToLocalCoord(uint32_t n)
    {
        NANOVDB_ASSERT(n < SIZE);
        const uint32_t m = n & ((1 << 2 * LOG2DIM) - 1);
        return CoordT(n >> 2 * LOG2DIM, m >> LOG2DIM, m & MASK);
    }

    /// @brief Converts (in place) a local index coordinate to a global index coordinate
    __hostdev__ void localToGlobalCoord(Coord& ijk) const { ijk += this->origin(); }

    __hostdev__ CoordT offsetToGlobalCoord(uint32_t n) const
    {
        return OffsetToLocalCoord(n) + this->origin();
    }

    /// @brief Return the dimension, in index space, of this leaf node (typically 8 as for openvdb leaf nodes!)
    __hostdev__ static uint32_t dim() { return 1u << LOG2DIM; }

    /// @brief Return the bounding box in index space of active values in this leaf node
    __hostdev__ math::BBox<CoordT> bbox() const
    {
        math::BBox<CoordT> bbox(DataType::mBBoxMin, DataType::mBBoxMin);
        if (this->hasBBox()) {
            bbox.max()[0] += DataType::mBBoxDif[0];
            bbox.max()[1] += DataType::mBBoxDif[1];
            bbox.max()[2] += DataType::mBBoxDif[2];
        } else { // very rare case
            bbox = math::BBox<CoordT>(); // invalid
        }
        return bbox;
    }

    /// @brief Return the total number of voxels (e.g. values) encoded in this leaf node
    __hostdev__ static uint32_t voxelCount() { return 1u << (3 * LOG2DIM); }

    __hostdev__ static uint32_t padding() { return DataType::padding(); }

    /// @brief return memory usage in bytes for the leaf node
    __hostdev__ uint64_t memUsage() const { return DataType::memUsage(); }

    /// @brief This class cannot be constructed or deleted
    LeafNode() = delete;
    LeafNode(const LeafNode&) = delete;
    LeafNode& operator=(const LeafNode&) = delete;
    ~LeafNode() = delete;

    /// @brief Return the voxel value at the given offset.
    __hostdev__ ValueType getValue(uint32_t offset) const { return DataType::getValue(offset); }

    /// @brief Return the voxel value at the given coordinate.
    __hostdev__ ValueType getValue(const CoordT& ijk) const { return DataType::getValue(CoordToOffset(ijk)); }

    /// @brief Return the first value in this leaf node.
    __hostdev__ ValueType getFirstValue() const { return this->getValue(0); }
    /// @brief Return the last value in this leaf node.
    __hostdev__ ValueType getLastValue() const { return this->getValue(SIZE - 1); }

    /// @brief Sets the value at the specified location and activate its state.
    ///
    /// @note This is safe since it does not change the topology of the tree (unlike setValue methods on the other nodes)
    __hostdev__ void setValue(const CoordT& ijk, const ValueType& v) { DataType::setValue(CoordToOffset(ijk), v); }

    /// @brief Sets the value at the specified location but leaves its state unchanged.
    ///
    /// @note This is safe since it does not change the topology of the tree (unlike setValue methods on the other nodes)
    __hostdev__ void setValueOnly(uint32_t offset, const ValueType& v) { DataType::setValueOnly(offset, v); }
    __hostdev__ void setValueOnly(const CoordT& ijk, const ValueType& v) { DataType::setValueOnly(CoordToOffset(ijk), v); }

    /// @brief Return @c true if the voxel value at the given coordinate is active.
    __hostdev__ bool isActive(const CoordT& ijk) const { return DataType::mValueMask.isOn(CoordToOffset(ijk)); }
    __hostdev__ bool isActive(uint32_t n) const { return DataType::mValueMask.isOn(n); }

    /// @brief Return @c true if any of the voxel value are active in this leaf node.
    __hostdev__ bool isActive() const
    {
        //NANOVDB_ASSERT( bool(DataType::mFlags & uint8_t(2)) != DataType::mValueMask.isOff() );
        //return DataType::mFlags & uint8_t(2);
        return !DataType::mValueMask.isOff();
    }

    __hostdev__ bool hasBBox() const { return DataType::mFlags & uint8_t(2); }

    /// @brief Return @c true if the voxel value at the given coordinate is active and updates @c v with the value.
    __hostdev__ bool probeValue(const CoordT& ijk, ValueType& v) const
    {
        const uint32_t n = CoordToOffset(ijk);
        v = DataType::getValue(n);
        return DataType::mValueMask.isOn(n);
    }

    __hostdev__ const LeafNode* probeLeaf(const CoordT&) const { return this; }

    /// @brief Return the linear offset corresponding to the given coordinate
    __hostdev__ static uint32_t CoordToOffset(const CoordT& ijk)
    {
        return ((ijk[0] & MASK) << (2 * LOG2DIM)) | ((ijk[1] & MASK) << LOG2DIM) | (ijk[2] & MASK);
    }

    /// @brief Updates the local bounding box of active voxels in this node. Return true if bbox was updated.
    ///
    /// @warning It assumes that the origin and value mask have already been set.
    ///
    /// @details This method is based on few (intrinsic) bit operations and hence is relatively fast.
    ///          However, it should only only be called if either the value mask has changed or if the
    ///          active bounding box is still undefined. e.g. during construction of this node.
    __hostdev__ bool updateBBox();

    template<typename OpT, typename... ArgsT>
    __hostdev__ auto get(const CoordType& ijk, ArgsT&&... args) const
    {
        return OpT::get(*this, CoordToOffset(ijk), args...);
    }

    template<typename OpT, typename... ArgsT>
    __hostdev__ auto get(const uint32_t n, ArgsT&&... args) const
    {
        return OpT::get(*this, n, args...);
    }

    template<typename OpT, typename... ArgsT>
    __hostdev__ auto set(const CoordType& ijk, ArgsT&&... args)
    {
        return OpT::set(*this, CoordToOffset(ijk), args...);
    }

    template<typename OpT, typename... ArgsT>
    __hostdev__ auto set(const uint32_t n, ArgsT&&... args)
    {
        return OpT::set(*this, n, args...);
    }

private:
    static_assert(sizeof(DataType) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(LeafData) is misaligned");

    template<typename, int, int, int>
    friend class ReadAccessor;

    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;

    template<typename RayT, typename AccT>
    __hostdev__ uint32_t getDimAndCache(const CoordT&, const RayT& /*ray*/, const AccT&) const
    {
        if (DataType::mFlags & uint8_t(1u))
            return this->dim(); // skip this node if the 1st bit is set

        //if (!ray.intersects( this->bbox() )) return 1 << LOG2DIM;
        return ChildNodeType::dim();
    }

    template<typename OpT, typename AccT, typename... ArgsT>
    __hostdev__ auto
    //__hostdev__  decltype(OpT::get(util::declval<const LeafNode&>(), util::declval<uint32_t>(), util::declval<ArgsT>()...))
    getAndCache(const CoordType& ijk, const AccT&, ArgsT&&... args) const
    {
        return OpT::get(*this, CoordToOffset(ijk), args...);
    }

    template<typename OpT, typename AccT, typename... ArgsT>
    //__hostdev__ auto // occasionally fails with NVCC
    __hostdev__ decltype(OpT::set(util::declval<LeafNode&>(), util::declval<uint32_t>(), util::declval<ArgsT>()...))
    setAndCache(const CoordType& ijk, const AccT&, ArgsT&&... args)
    {
        return OpT::set(*this, CoordToOffset(ijk), args...);
    }

}; // LeafNode class

// --------------------------> LeafNode<T>::updateBBox <------------------------------------

template<typename ValueT, typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
__hostdev__ inline bool LeafNode<ValueT, CoordT, MaskT, LOG2DIM>::updateBBox()
{
    static_assert(LOG2DIM == 3, "LeafNode::updateBBox: only supports LOGDIM = 3!");
    if (DataType::mValueMask.isOff()) {
        DataType::mFlags &= ~uint8_t(2); // set 2nd bit off, which indicates that this nodes has no bbox
        return false;
    }
    auto update = [&](uint32_t min, uint32_t max, int axis) {
        NANOVDB_ASSERT(min <= max && max < 8);
        DataType::mBBoxMin[axis] = (DataType::mBBoxMin[axis] & ~MASK) + int(min);
        DataType::mBBoxDif[axis] = uint8_t(max - min);
    };
    uint64_t *w = DataType::mValueMask.words(), word64 = *w;
    uint32_t  Xmin = word64 ? 0u : 8u, Xmax = Xmin;
    for (int i = 1; i < 8; ++i) { // last loop over 7 remaining 64 bit words
        if (w[i]) { // skip if word has no set bits
            word64 |= w[i]; // union 8 x 64 bits words into one 64 bit word
            if (Xmin == 8)
                Xmin = i; // only set once
            Xmax = i;
        }
    }
    NANOVDB_ASSERT(word64);
    update(Xmin, Xmax, 0);
    update(util::findLowestOn(word64) >> 3, util::findHighestOn(word64) >> 3, 1);
    const uint32_t *p = reinterpret_cast<const uint32_t*>(&word64), word32 = p[0] | p[1];
    const uint16_t *q = reinterpret_cast<const uint16_t*>(&word32), word16 = q[0] | q[1];
    const uint8_t  *b = reinterpret_cast<const uint8_t*>(&word16), byte = b[0] | b[1];
    NANOVDB_ASSERT(byte);
    update(util::findLowestOn(static_cast<uint32_t>(byte)), util::findHighestOn(static_cast<uint32_t>(byte)), 2);
    DataType::mFlags |= uint8_t(2); // set 2nd bit on, which indicates that this nodes has a bbox
    return true;
} // LeafNode::updateBBox

// --------------------------> Template specializations and traits <------------------------------------

/// @brief Template specializations to the default configuration used in OpenVDB:
///        Root -> 32^3 -> 16^3 -> 8^3
template<typename BuildT>
using NanoLeaf = LeafNode<BuildT, Coord, Mask, 3>;
template<typename BuildT>
using NanoLower = InternalNode<NanoLeaf<BuildT>, 4>;
template<typename BuildT>
using NanoUpper = InternalNode<NanoLower<BuildT>, 5>;
template<typename BuildT>
using NanoRoot = RootNode<NanoUpper<BuildT>>;
template<typename BuildT>
using NanoTree = Tree<NanoRoot<BuildT>>;
template<typename BuildT>
using NanoGrid = Grid<NanoTree<BuildT>>;

/// @brief Trait to map from LEVEL to node type
template<typename BuildT, int LEVEL>
struct NanoNode;

// Partial template specialization of above Node struct
template<typename BuildT>
struct NanoNode<BuildT, 0>
{
    using Type = NanoLeaf<BuildT>;
    using type = NanoLeaf<BuildT>;
};
template<typename BuildT>
struct NanoNode<BuildT, 1>
{
    using Type = NanoLower<BuildT>;
    using type = NanoLower<BuildT>;
};
template<typename BuildT>
struct NanoNode<BuildT, 2>
{
    using Type = NanoUpper<BuildT>;
    using type = NanoUpper<BuildT>;
};
template<typename BuildT>
struct NanoNode<BuildT, 3>
{
    using Type = NanoRoot<BuildT>;
    using type = NanoRoot<BuildT>;
};

using FloatTree = NanoTree<float>;
using Fp4Tree = NanoTree<Fp4>;
using Fp8Tree = NanoTree<Fp8>;
using Fp16Tree = NanoTree<Fp16>;
using FpNTree = NanoTree<FpN>;
using DoubleTree = NanoTree<double>;
using Int32Tree = NanoTree<int32_t>;
using UInt32Tree = NanoTree<uint32_t>;
using Int64Tree = NanoTree<int64_t>;
using Vec3fTree = NanoTree<Vec3f>;
using Vec3dTree = NanoTree<Vec3d>;
using Vec4fTree = NanoTree<Vec4f>;
using Vec4dTree = NanoTree<Vec4d>;
using Vec3ITree = NanoTree<Vec3i>;
using MaskTree = NanoTree<ValueMask>;
using BoolTree = NanoTree<bool>;
using IndexTree = NanoTree<ValueIndex>;
using OnIndexTree = NanoTree<ValueOnIndex>;
using IndexMaskTree = NanoTree<ValueIndexMask>;
using OnIndexMaskTree = NanoTree<ValueOnIndexMask>;

using FloatGrid = Grid<FloatTree>;
using Fp4Grid = Grid<Fp4Tree>;
using Fp8Grid = Grid<Fp8Tree>;
using Fp16Grid = Grid<Fp16Tree>;
using FpNGrid = Grid<FpNTree>;
using DoubleGrid = Grid<DoubleTree>;
using Int32Grid = Grid<Int32Tree>;
using UInt32Grid = Grid<UInt32Tree>;
using Int64Grid = Grid<Int64Tree>;
using Vec3fGrid = Grid<Vec3fTree>;
using Vec3dGrid = Grid<Vec3dTree>;
using Vec4fGrid = Grid<Vec4fTree>;
using Vec4dGrid = Grid<Vec4dTree>;
using Vec3IGrid = Grid<Vec3ITree>;
using MaskGrid = Grid<MaskTree>;
using BoolGrid = Grid<BoolTree>;
using PointGrid = Grid<Point>;
using IndexGrid = Grid<IndexTree>;
using OnIndexGrid = Grid<OnIndexTree>;
using IndexMaskGrid = Grid<IndexMaskTree>;
using OnIndexMaskGrid = Grid<OnIndexMaskTree>;

// --------------------------> callNanoGrid <------------------------------------

/**
* @brief Below is an example of the struct used for generic programming with callNanoGrid
* @details For an example see "struct Crc32TailOld" in nanovdb/tools/GridChecksum.h or
*          "struct IsNanoGridValid" in nanovdb/tools/GridValidator.h
* @code
*   struct OpT {
        // define these two static functions with non-const GridData
*       template <typename BuildT>
*       static auto   known(      GridData *gridData, args...);
*       static auto unknown(      GridData *gridData, args...);
*       // or alternatively these two static functions with const GridData
*       template <typename BuildT>
*       static auto   known(const GridData *gridData, args...);
*       static auto unknown(const GridData *gridData, args...);
*   };
*  @endcode
*
* @brief Here is an example of how to use callNanoGrid in client code
* @code
*    return callNanoGrid<OpT>(gridData, args...);
* @endcode
*/

/// @brief Use this function, which depends a pointer to GridData, to call
///        other functions that depend on a NanoGrid of a known ValueType.
/// @details This function allows for generic programming by converting GridData
///          to a NanoGrid of the type encoded in GridData::mGridType.
template<typename OpT, typename GridDataT, typename... ArgsT>
auto callNanoGrid(GridDataT *gridData, ArgsT&&... args)
{
    static_assert(util::is_same<GridDataT, GridData, const GridData>::value, "Expected gridData to be of type GridData* or const GridData*");
    switch (gridData->mGridType){
        case GridType::Float:
            return OpT::template known<float>(gridData, args...);
        case GridType::Double:
            return OpT::template known<double>(gridData, args...);
        case GridType::Int16:
            return OpT::template known<int16_t>(gridData, args...);
        case GridType::Int32:
            return OpT::template known<int32_t>(gridData, args...);
        case GridType::Int64:
            return OpT::template known<int64_t>(gridData, args...);
        case GridType::Vec3f:
            return OpT::template known<Vec3f>(gridData, args...);
        case GridType::Vec3d:
            return OpT::template known<Vec3d>(gridData, args...);
        case GridType::UInt32:
            return OpT::template known<uint32_t>(gridData, args...);
        case GridType::Mask:
            return OpT::template known<ValueMask>(gridData, args...);
        case GridType::Index:
            return OpT::template known<ValueIndex>(gridData, args...);
        case GridType::OnIndex:
            return OpT::template known<ValueOnIndex>(gridData, args...);
        case GridType::IndexMask:
            return OpT::template known<ValueIndexMask>(gridData, args...);
        case GridType::OnIndexMask:
            return OpT::template known<ValueOnIndexMask>(gridData, args...);
        case GridType::Boolean:
            return OpT::template known<bool>(gridData, args...);
        case GridType::RGBA8:
            return OpT::template known<math::Rgba8>(gridData, args...);
        case GridType::Fp4:
            return OpT::template known<Fp4>(gridData, args...);
        case GridType::Fp8:
            return OpT::template known<Fp8>(gridData, args...);
        case GridType::Fp16:
            return OpT::template known<Fp16>(gridData, args...);
        case GridType::FpN:
            return OpT::template known<FpN>(gridData, args...);
        case GridType::Vec4f:
            return OpT::template known<Vec4f>(gridData, args...);
        case GridType::Vec4d:
            return OpT::template known<Vec4d>(gridData, args...);
        case GridType::UInt8:
            return OpT::template known<uint8_t>(gridData, args...);
        default:
            return OpT::unknown(gridData, args...);
    }
}// callNanoGrid

// --------------------------> ReadAccessor <------------------------------------

/// @brief A read-only value accessor with three levels of node caching. This allows for
///        inverse tree traversal during lookup, which is on average significantly faster
///        than calling the equivalent method on the tree (i.e. top-down traversal).
///
/// @note  By virtue of the fact that a value accessor accelerates random access operations
///        by re-using cached access patterns, this access should be reused for multiple access
///        operations. In other words, never create an instance of this accessor for a single
///        access only. In general avoid single access operations with this accessor, and
///        if that is not possible call the corresponding method on the tree instead.
///
/// @warning Since this ReadAccessor internally caches raw pointers to the nodes of the tree
///          structure, it is not safe to copy between host and device, or even to share among
///          multiple threads on the same host or device. However, it is light-weight so simple
///          instantiate one per thread (on the host and/or device).
///
/// @details Used to accelerated random access into a VDB tree. Provides on average
///          O(1) random access operations by means of inverse tree traversal,
///          which amortizes the non-const time complexity of the root node.

template<typename BuildT>
class ReadAccessor<BuildT, -1, -1, -1>
{
    using GridT = NanoGrid<BuildT>; // grid
    using TreeT = NanoTree<BuildT>; // tree
    using RootT = NanoRoot<BuildT>; // root node
    using LeafT = NanoLeaf<BuildT>; // Leaf node
    using FloatType = typename RootT::FloatType;
    using CoordValueType = typename RootT::CoordType::ValueType;

    mutable const RootT* mRoot; // 8 bytes (mutable to allow for access methods to be const)
public:
    using BuildType = BuildT;
    using ValueType = typename RootT::ValueType;
    using CoordType = typename RootT::CoordType;

    static const int CacheLevels = 0;

    /// @brief Constructor from a root node
    __hostdev__ ReadAccessor(const RootT& root)
        : mRoot{&root}
    {
    }

    /// @brief Constructor from a grid
    __hostdev__ ReadAccessor(const GridT& grid)
        : ReadAccessor(grid.tree().root())
    {
    }

    /// @brief Constructor from a tree
    __hostdev__ ReadAccessor(const TreeT& tree)
        : ReadAccessor(tree.root())
    {
    }

    /// @brief Reset this access to its initial state, i.e. with an empty cache
    /// @node Noop since this template specialization has no cache
    __hostdev__ void clear() {}

    __hostdev__ const RootT& root() const { return *mRoot; }

    /// @brief Defaults constructors
    ReadAccessor(const ReadAccessor&) = default;
    ~ReadAccessor() = default;
    ReadAccessor& operator=(const ReadAccessor&) = default;
    __hostdev__ ValueType getValue(const CoordType& ijk) const
    {
        return this->template get<GetValue<BuildT>>(ijk);
    }
    __hostdev__ ValueType    getValue(int i, int j, int k) const { return this->template get<GetValue<BuildT>>(CoordType(i, j, k)); }
    __hostdev__ ValueType    operator()(const CoordType& ijk) const { return this->template get<GetValue<BuildT>>(ijk); }
    __hostdev__ ValueType    operator()(int i, int j, int k) const { return this->template get<GetValue<BuildT>>(CoordType(i, j, k)); }
    __hostdev__ auto         getNodeInfo(const CoordType& ijk) const { return this->template get<GetNodeInfo<BuildT>>(ijk); }
    __hostdev__ bool         isActive(const CoordType& ijk) const { return this->template get<GetState<BuildT>>(ijk); }
    __hostdev__ bool         probeValue(const CoordType& ijk, ValueType& v) const { return this->template get<ProbeValue<BuildT>>(ijk, v); }
    __hostdev__ const LeafT* probeLeaf(const CoordType& ijk) const { return this->template get<GetLeaf<BuildT>>(ijk); }
    template<typename RayT>
    __hostdev__ uint32_t getDim(const CoordType& ijk, const RayT& ray) const
    {
        return mRoot->getDimAndCache(ijk, ray, *this);
    }
    template<typename OpT, typename... ArgsT>
    __hostdev__ auto get(const CoordType& ijk, ArgsT&&... args) const
    {
        return mRoot->template get<OpT>(ijk, args...);
    }

    template<typename OpT, typename... ArgsT>
    __hostdev__ auto set(const CoordType& ijk, ArgsT&&... args) const
    {
        return const_cast<RootT*>(mRoot)->template set<OpT>(ijk, args...);
    }

private:
    /// @brief Allow nodes to insert themselves into the cache.
    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;
    template<typename, typename, template<uint32_t> class, uint32_t>
    friend class LeafNode;

    /// @brief No-op
    template<typename NodeT>
    __hostdev__ void insert(const CoordType&, const NodeT*) const {}
}; // ReadAccessor<ValueT, -1, -1, -1> class

/// @brief Node caching at a single tree level
template<typename BuildT, int LEVEL0>
class ReadAccessor<BuildT, LEVEL0, -1, -1> //e.g. 0, 1, 2
{
    static_assert(LEVEL0 >= 0 && LEVEL0 <= 2, "LEVEL0 should be 0, 1, or 2");

    using GridT = NanoGrid<BuildT>; // grid
    using TreeT = NanoTree<BuildT>;
    using RootT = NanoRoot<BuildT>; //  root node
    using LeafT = NanoLeaf<BuildT>; // Leaf node
    using NodeT = typename NodeTrait<TreeT, LEVEL0>::type;
    using CoordT = typename RootT::CoordType;
    using ValueT = typename RootT::ValueType;

    using FloatType = typename RootT::FloatType;
    using CoordValueType = typename RootT::CoordT::ValueType;

    // All member data are mutable to allow for access methods to be const
    mutable CoordT       mKey; // 3*4 = 12 bytes
    mutable const RootT* mRoot; // 8 bytes
    mutable const NodeT* mNode; // 8 bytes

public:
    using BuildType = BuildT;
    using ValueType = ValueT;
    using CoordType = CoordT;

    static const int CacheLevels = 1;

    /// @brief Constructor from a root node
    __hostdev__ ReadAccessor(const RootT& root)
        : mKey(CoordType::max())
        , mRoot(&root)
        , mNode(nullptr)
    {
    }

    /// @brief Constructor from a grid
    __hostdev__ ReadAccessor(const GridT& grid)
        : ReadAccessor(grid.tree().root())
    {
    }

    /// @brief Constructor from a tree
    __hostdev__ ReadAccessor(const TreeT& tree)
        : ReadAccessor(tree.root())
    {
    }

    /// @brief Reset this access to its initial state, i.e. with an empty cache
    __hostdev__ void clear()
    {
        mKey = CoordType::max();
        mNode = nullptr;
    }

    __hostdev__ const RootT& root() const { return *mRoot; }

    /// @brief Defaults constructors
    ReadAccessor(const ReadAccessor&) = default;
    ~ReadAccessor() = default;
    ReadAccessor& operator=(const ReadAccessor&) = default;

    __hostdev__ bool isCached(const CoordType& ijk) const
    {
        return (ijk[0] & int32_t(~NodeT::MASK)) == mKey[0] &&
               (ijk[1] & int32_t(~NodeT::MASK)) == mKey[1] &&
               (ijk[2] & int32_t(~NodeT::MASK)) == mKey[2];
    }

    __hostdev__ ValueType getValue(const CoordType& ijk) const
    {
        return this->template get<GetValue<BuildT>>(ijk);
    }
    __hostdev__ ValueType    getValue(int i, int j, int k) const { return this->template get<GetValue<BuildT>>(CoordType(i, j, k)); }
    __hostdev__ ValueType    operator()(const CoordType& ijk) const { return this->template get<GetValue<BuildT>>(ijk); }
    __hostdev__ ValueType    operator()(int i, int j, int k) const { return this->template get<GetValue<BuildT>>(CoordType(i, j, k)); }
    __hostdev__ auto         getNodeInfo(const CoordType& ijk) const { return this->template get<GetNodeInfo<BuildT>>(ijk); }
    __hostdev__ bool         isActive(const CoordType& ijk) const { return this->template get<GetState<BuildT>>(ijk); }
    __hostdev__ bool         probeValue(const CoordType& ijk, ValueType& v) const { return this->template get<ProbeValue<BuildT>>(ijk, v); }
    __hostdev__ const LeafT* probeLeaf(const CoordType& ijk) const { return this->template get<GetLeaf<BuildT>>(ijk); }

    template<typename RayT>
    __hostdev__ uint32_t getDim(const CoordType& ijk, const RayT& ray) const
    {
        if (this->isCached(ijk)) return mNode->getDimAndCache(ijk, ray, *this);
        return mRoot->getDimAndCache(ijk, ray, *this);
    }

    template<typename OpT, typename... ArgsT>
    __hostdev__ typename OpT::Type get(const CoordType& ijk, ArgsT&&... args) const
    {
        if constexpr(OpT::LEVEL <= LEVEL0) if (this->isCached(ijk)) return mNode->template getAndCache<OpT>(ijk, *this, args...);
        return mRoot->template getAndCache<OpT>(ijk, *this, args...);
    }

    template<typename OpT, typename... ArgsT>
    __hostdev__ void set(const CoordType& ijk, ArgsT&&... args) const
    {
        if constexpr(OpT::LEVEL <= LEVEL0) if (this->isCached(ijk)) return const_cast<NodeT*>(mNode)->template setAndCache<OpT>(ijk, *this, args...);
        return const_cast<RootT*>(mRoot)->template setAndCache<OpT>(ijk, *this, args...);
    }

private:
    /// @brief Allow nodes to insert themselves into the cache.
    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;
    template<typename, typename, template<uint32_t> class, uint32_t>
    friend class LeafNode;

    /// @brief Inserts a leaf node and key pair into this ReadAccessor
    __hostdev__ void insert(const CoordType& ijk, const NodeT* node) const
    {
        mKey = ijk & ~NodeT::MASK;
        mNode = node;
    }

    // no-op
    template<typename OtherNodeT>
    __hostdev__ void insert(const CoordType&, const OtherNodeT*) const {}

}; // ReadAccessor<ValueT, LEVEL0>

template<typename BuildT, int LEVEL0, int LEVEL1>
class ReadAccessor<BuildT, LEVEL0, LEVEL1, -1> //e.g. (0,1), (1,2), (0,2)
{
    static_assert(LEVEL0 >= 0 && LEVEL0 <= 2, "LEVEL0 must be 0, 1, 2");
    static_assert(LEVEL1 >= 0 && LEVEL1 <= 2, "LEVEL1 must be 0, 1, 2");
    static_assert(LEVEL0 < LEVEL1, "Level 0 must be lower than level 1");
    using GridT = NanoGrid<BuildT>; // grid
    using TreeT = NanoTree<BuildT>;
    using RootT = NanoRoot<BuildT>;
    using LeafT = NanoLeaf<BuildT>;
    using Node1T = typename NodeTrait<TreeT, LEVEL0>::type;
    using Node2T = typename NodeTrait<TreeT, LEVEL1>::type;
    using CoordT = typename RootT::CoordType;
    using ValueT = typename RootT::ValueType;
    using FloatType = typename RootT::FloatType;
    using CoordValueType = typename RootT::CoordT::ValueType;

    // All member data are mutable to allow for access methods to be const
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY // 44 bytes total
    mutable CoordT mKey; // 3*4 = 12 bytes
#else // 68 bytes total
    mutable CoordT mKeys[2]; // 2*3*4 = 24 bytes
#endif
    mutable const RootT*  mRoot;
    mutable const Node1T* mNode1;
    mutable const Node2T* mNode2;

public:
    using BuildType = BuildT;
    using ValueType = ValueT;
    using CoordType = CoordT;

    static const int CacheLevels = 2;

    /// @brief Constructor from a root node
    __hostdev__ ReadAccessor(const RootT& root)
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
        : mKey(CoordType::max())
#else
        : mKeys{CoordType::max(), CoordType::max()}
#endif
        , mRoot(&root)
        , mNode1(nullptr)
        , mNode2(nullptr)
    {
    }

    /// @brief Constructor from a grid
    __hostdev__ ReadAccessor(const GridT& grid)
        : ReadAccessor(grid.tree().root())
    {
    }

    /// @brief Constructor from a tree
    __hostdev__ ReadAccessor(const TreeT& tree)
        : ReadAccessor(tree.root())
    {
    }

    /// @brief Reset this access to its initial state, i.e. with an empty cache
    __hostdev__ void clear()
    {
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
        mKey = CoordType::max();
#else
        mKeys[0] = mKeys[1] = CoordType::max();
#endif
        mNode1 = nullptr;
        mNode2 = nullptr;
    }

    __hostdev__ const RootT& root() const { return *mRoot; }

    /// @brief Defaults constructors
    ReadAccessor(const ReadAccessor&) = default;
    ~ReadAccessor() = default;
    ReadAccessor& operator=(const ReadAccessor&) = default;

#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
    __hostdev__ bool isCached1(CoordValueType dirty) const
    {
        if (!mNode1)
            return false;
        if (dirty & int32_t(~Node1T::MASK)) {
            mNode1 = nullptr;
            return false;
        }
        return true;
    }
    __hostdev__ bool isCached2(CoordValueType dirty) const
    {
        if (!mNode2)
            return false;
        if (dirty & int32_t(~Node2T::MASK)) {
            mNode2 = nullptr;
            return false;
        }
        return true;
    }
    __hostdev__ CoordValueType computeDirty(const CoordType& ijk) const
    {
        return (ijk[0] ^ mKey[0]) | (ijk[1] ^ mKey[1]) | (ijk[2] ^ mKey[2]);
    }
#else
    __hostdev__ bool isCached1(const CoordType& ijk) const
    {
        return (ijk[0] & int32_t(~Node1T::MASK)) == mKeys[0][0] &&
               (ijk[1] & int32_t(~Node1T::MASK)) == mKeys[0][1] &&
               (ijk[2] & int32_t(~Node1T::MASK)) == mKeys[0][2];
    }
    __hostdev__ bool isCached2(const CoordType& ijk) const
    {
        return (ijk[0] & int32_t(~Node2T::MASK)) == mKeys[1][0] &&
               (ijk[1] & int32_t(~Node2T::MASK)) == mKeys[1][1] &&
               (ijk[2] & int32_t(~Node2T::MASK)) == mKeys[1][2];
    }
#endif

    __hostdev__ ValueType getValue(const CoordType& ijk) const
    {
        return this->template get<GetValue<BuildT>>(ijk);
    }
    __hostdev__ ValueType    getValue(int i, int j, int k) const { return this->template get<GetValue<BuildT>>(CoordType(i, j, k)); }
    __hostdev__ ValueType    operator()(const CoordType& ijk) const { return this->template get<GetValue<BuildT>>(ijk); }
    __hostdev__ ValueType    operator()(int i, int j, int k) const { return this->template get<GetValue<BuildT>>(CoordType(i, j, k)); }
    __hostdev__ auto         getNodeInfo(const CoordType& ijk) const { return this->template get<GetNodeInfo<BuildT>>(ijk); }
    __hostdev__ bool         isActive(const CoordType& ijk) const { return this->template get<GetState<BuildT>>(ijk); }
    __hostdev__ bool         probeValue(const CoordType& ijk, ValueType& v) const { return this->template get<ProbeValue<BuildT>>(ijk, v); }
    __hostdev__ const LeafT* probeLeaf(const CoordType& ijk) const { return this->template get<GetLeaf<BuildT>>(ijk); }

    template<typename RayT>
    __hostdev__ uint32_t getDim(const CoordType& ijk, const RayT& ray) const
    {
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached1(dirty)) {
            return mNode1->getDimAndCache(ijk, ray, *this);
        } else if (this->isCached2(dirty)) {
            return mNode2->getDimAndCache(ijk, ray, *this);
        }
        return mRoot->getDimAndCache(ijk, ray, *this);
    }

    template<typename OpT, typename... ArgsT>
    __hostdev__ typename OpT::Type get(const CoordType& ijk, ArgsT&&... args) const
    {
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if constexpr(OpT::LEVEL <= LEVEL0) {
            if (this->isCached1(dirty)) return mNode1->template getAndCache<OpT>(ijk, *this, args...);
        } else if constexpr(OpT::LEVEL <= LEVEL1) {
            if (this->isCached2(dirty)) return mNode2->template getAndCache<OpT>(ijk, *this, args...);
        }
        return mRoot->template getAndCache<OpT>(ijk, *this, args...);
    }

    template<typename OpT, typename... ArgsT>
    __hostdev__ void set(const CoordType& ijk, ArgsT&&... args) const
    {
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if constexpr(OpT::LEVEL <= LEVEL0) {
            if (this->isCached1(dirty)) return const_cast<Node1T*>(mNode1)->template setAndCache<OpT>(ijk, *this, args...);
        } else if constexpr(OpT::LEVEL <= LEVEL1) {
            if (this->isCached2(dirty)) return const_cast<Node2T*>(mNode2)->template setAndCache<OpT>(ijk, *this, args...);
        }
        return const_cast<RootT*>(mRoot)->template setAndCache<OpT>(ijk, *this, args...);
    }

private:
    /// @brief Allow nodes to insert themselves into the cache.
    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;
    template<typename, typename, template<uint32_t> class, uint32_t>
    friend class LeafNode;

    /// @brief Inserts a leaf node and key pair into this ReadAccessor
    __hostdev__ void insert(const CoordType& ijk, const Node1T* node) const
    {
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
        mKey = ijk;
#else
        mKeys[0] = ijk & ~Node1T::MASK;
#endif
        mNode1 = node;
    }
    __hostdev__ void insert(const CoordType& ijk, const Node2T* node) const
    {
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
        mKey = ijk;
#else
        mKeys[1] = ijk & ~Node2T::MASK;
#endif
        mNode2 = node;
    }
    template<typename OtherNodeT>
    __hostdev__ void insert(const CoordType&, const OtherNodeT*) const {}
}; // ReadAccessor<BuildT, LEVEL0, LEVEL1>

/// @brief Node caching at all (three) tree levels
template<typename BuildT>
class ReadAccessor<BuildT, 0, 1, 2>
{
    using GridT = NanoGrid<BuildT>; // grid
    using TreeT = NanoTree<BuildT>;
    using RootT = NanoRoot<BuildT>; //  root node
    using NodeT2 = NanoUpper<BuildT>; // upper internal node
    using NodeT1 = NanoLower<BuildT>; // lower internal node
    using LeafT = NanoLeaf<BuildT>; // Leaf node
    using CoordT = typename RootT::CoordType;
    using ValueT = typename RootT::ValueType;

    using FloatType = typename RootT::FloatType;
    using CoordValueType = typename RootT::CoordT::ValueType;

    // All member data are mutable to allow for access methods to be const
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY // 44 bytes total
    mutable CoordT mKey; // 3*4 = 12 bytes
#else // 68 bytes total
    mutable CoordT mKeys[3]; // 3*3*4 = 36 bytes
#endif
    mutable const RootT* mRoot;
    mutable const void*  mNode[3]; // 4*8 = 32 bytes

public:
    using BuildType = BuildT;
    using ValueType = ValueT;
    using CoordType = CoordT;

    static const int CacheLevels = 3;

    /// @brief Constructor from a root node
    __hostdev__ ReadAccessor(const RootT& root)
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
        : mKey(CoordType::max())
#else
        : mKeys{CoordType::max(), CoordType::max(), CoordType::max()}
#endif
        , mRoot(&root)
        , mNode{nullptr, nullptr, nullptr}
    {
    }

    /// @brief Constructor from a grid
    __hostdev__ ReadAccessor(const GridT& grid)
        : ReadAccessor(grid.tree().root())
    {
    }

    /// @brief Constructor from a tree
    __hostdev__ ReadAccessor(const TreeT& tree)
        : ReadAccessor(tree.root())
    {
    }

    __hostdev__ const RootT& root() const { return *mRoot; }

    /// @brief Defaults constructors
    ReadAccessor(const ReadAccessor&) = default;
    ~ReadAccessor() = default;
    ReadAccessor& operator=(const ReadAccessor&) = default;

    /// @brief Return a const point to the cached node of the specified type
    ///
    /// @warning The return value could be NULL.
    template<typename NodeT>
    __hostdev__ const NodeT* getNode() const
    {
        using T = typename NodeTrait<TreeT, NodeT::LEVEL>::type;
        static_assert(util::is_same<T, NodeT>::value, "ReadAccessor::getNode: Invalid node type");
        return reinterpret_cast<const T*>(mNode[NodeT::LEVEL]);
    }

    template<int LEVEL>
    __hostdev__ const typename NodeTrait<TreeT, LEVEL>::type* getNode() const
    {
        using T = typename NodeTrait<TreeT, LEVEL>::type;
        static_assert(LEVEL >= 0 && LEVEL <= 2, "ReadAccessor::getNode: Invalid node type");
        return reinterpret_cast<const T*>(mNode[LEVEL]);
    }

    /// @brief Reset this access to its initial state, i.e. with an empty cache
    __hostdev__ void clear()
    {
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
        mKey = CoordType::max();
#else
        mKeys[0] = mKeys[1] = mKeys[2] = CoordType::max();
#endif
        mNode[0] = mNode[1] = mNode[2] = nullptr;
    }

#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
    template<typename NodeT>
    __hostdev__ bool isCached(CoordValueType dirty) const
    {
        if (!mNode[NodeT::LEVEL])
            return false;
        if (dirty & int32_t(~NodeT::MASK)) {
            mNode[NodeT::LEVEL] = nullptr;
            return false;
        }
        return true;
    }

    __hostdev__ CoordValueType computeDirty(const CoordType& ijk) const
    {
        return (ijk[0] ^ mKey[0]) | (ijk[1] ^ mKey[1]) | (ijk[2] ^ mKey[2]);
    }
#else
    template<typename NodeT>
    __hostdev__ bool isCached(const CoordType& ijk) const
    {
        return (ijk[0] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][0] &&
               (ijk[1] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][1] &&
               (ijk[2] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][2];
    }
#endif

    __hostdev__ ValueType getValue(const CoordType& ijk) const {return this->template get<GetValue<BuildT>>(ijk);}
    __hostdev__ ValueType    getValue(int i, int j, int k) const { return this->template get<GetValue<BuildT>>(CoordType(i, j, k)); }
    __hostdev__ ValueType    operator()(const CoordType& ijk) const { return this->template get<GetValue<BuildT>>(ijk); }
    __hostdev__ ValueType    operator()(int i, int j, int k) const { return this->template get<GetValue<BuildT>>(CoordType(i, j, k)); }
    __hostdev__ auto         getNodeInfo(const CoordType& ijk) const { return this->template get<GetNodeInfo<BuildT>>(ijk); }
    __hostdev__ bool         isActive(const CoordType& ijk) const { return this->template get<GetState<BuildT>>(ijk); }
    __hostdev__ bool         probeValue(const CoordType& ijk, ValueType& v) const { return this->template get<ProbeValue<BuildT>>(ijk, v); }
    __hostdev__ const LeafT* probeLeaf(const CoordType& ijk) const { return this->template get<GetLeaf<BuildT>>(ijk); }

    template<typename OpT, typename... ArgsT>
    __hostdev__ typename OpT::Type get(const CoordType& ijk, ArgsT&&... args) const
    {
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if constexpr(OpT::LEVEL <=0) {
            if (this->isCached<LeafT>(dirty)) return ((const LeafT*)mNode[0])->template getAndCache<OpT>(ijk, *this, args...);
        } else if constexpr(OpT::LEVEL <= 1) {
            if (this->isCached<NodeT1>(dirty)) return ((const NodeT1*)mNode[1])->template getAndCache<OpT>(ijk, *this, args...);
        } else if constexpr(OpT::LEVEL <= 2) {
            if (this->isCached<NodeT2>(dirty)) return ((const NodeT2*)mNode[2])->template getAndCache<OpT>(ijk, *this, args...);
        }
        return mRoot->template getAndCache<OpT>(ijk, *this, args...);
    }

    template<typename OpT, typename... ArgsT>
    __hostdev__ void set(const CoordType& ijk, ArgsT&&... args) const
    {
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if constexpr(OpT::LEVEL <= 0) {
            if (this->isCached<LeafT>(dirty)) return ((LeafT*)mNode[0])->template setAndCache<OpT>(ijk, *this, args...);
        } else if constexpr(OpT::LEVEL <= 1) {
            if (this->isCached<NodeT1>(dirty)) return ((NodeT1*)mNode[1])->template setAndCache<OpT>(ijk, *this, args...);
        } else if constexpr(OpT::LEVEL <= 2) {
            if (this->isCached<NodeT2>(dirty)) return ((NodeT2*)mNode[2])->template setAndCache<OpT>(ijk, *this, args...);
        }
        return ((RootT*)mRoot)->template setAndCache<OpT>(ijk, *this, args...);
    }

    template<typename RayT>
    __hostdev__ uint32_t getDim(const CoordType& ijk, const RayT& ray) const
    {
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached<LeafT>(dirty)) {
            return ((LeafT*)mNode[0])->getDimAndCache(ijk, ray, *this);
        } else if (this->isCached<NodeT1>(dirty)) {
            return ((NodeT1*)mNode[1])->getDimAndCache(ijk, ray, *this);
        } else if (this->isCached<NodeT2>(dirty)) {
            return ((NodeT2*)mNode[2])->getDimAndCache(ijk, ray, *this);
        }
        return mRoot->getDimAndCache(ijk, ray, *this);
    }

private:
    /// @brief Allow nodes to insert themselves into the cache.
    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;
    template<typename, typename, template<uint32_t> class, uint32_t>
    friend class LeafNode;

    /// @brief Inserts a leaf node and key pair into this ReadAccessor
    template<typename NodeT>
    __hostdev__ void insert(const CoordType& ijk, const NodeT* node) const
    {
#ifdef NANOVDB_USE_SINGLE_ACCESSOR_KEY
        mKey = ijk;
#else
        mKeys[NodeT::LEVEL] = ijk & ~NodeT::MASK;
#endif
        mNode[NodeT::LEVEL] = node;
    }
}; // ReadAccessor<BuildT, 0, 1, 2>

//////////////////////////////////////////////////

/// @brief Free-standing function for convenient creation of a ReadAccessor with
///        optional and customizable node caching.
///
/// @details createAccessor<>(grid):  No caching of nodes and hence it's thread-safe but slow
///          createAccessor<0>(grid): Caching of leaf nodes only
///          createAccessor<1>(grid): Caching of lower internal nodes only
///          createAccessor<2>(grid): Caching of upper internal nodes only
///          createAccessor<0,1>(grid): Caching of leaf and lower internal nodes
///          createAccessor<0,2>(grid): Caching of leaf and upper internal nodes
///          createAccessor<1,2>(grid): Caching of lower and upper internal nodes
///          createAccessor<0,1,2>(grid): Caching of all nodes at all tree levels

template<int LEVEL0 = -1, int LEVEL1 = -1, int LEVEL2 = -1, typename ValueT = float>
ReadAccessor<ValueT, LEVEL0, LEVEL1, LEVEL2> createAccessor(const NanoGrid<ValueT>& grid)
{
    return ReadAccessor<ValueT, LEVEL0, LEVEL1, LEVEL2>(grid);
}

template<int LEVEL0 = -1, int LEVEL1 = -1, int LEVEL2 = -1, typename ValueT = float>
ReadAccessor<ValueT, LEVEL0, LEVEL1, LEVEL2> createAccessor(const NanoTree<ValueT>& tree)
{
    return ReadAccessor<ValueT, LEVEL0, LEVEL1, LEVEL2>(tree);
}

template<int LEVEL0 = -1, int LEVEL1 = -1, int LEVEL2 = -1, typename ValueT = float>
ReadAccessor<ValueT, LEVEL0, LEVEL1, LEVEL2> createAccessor(const NanoRoot<ValueT>& root)
{
    return ReadAccessor<ValueT, LEVEL0, LEVEL1, LEVEL2>(root);
}

//////////////////////////////////////////////////

/// @brief This is a convenient class that allows for access to grid meta-data
///        that are independent of the value type of a grid. That is, this class
///        can be used to get information about a grid without actually knowing
///        its ValueType.
class GridMetaData
{ // 768 bytes (32 byte aligned)
    GridData  mGridData; // 672B
    TreeData  mTreeData; // 64B
    CoordBBox mIndexBBox; // 24B. AABB of active values in index space.
    uint32_t  mRootTableSize, mPadding{0}; // 8B

public:
    template<typename T>
    GridMetaData(const NanoGrid<T>& grid)
    {
        mGridData = *grid.data();
        mTreeData = *grid.tree().data();
        mIndexBBox = grid.indexBBox();
        mRootTableSize = grid.tree().root().getTableSize();
    }
    GridMetaData(const GridData* gridData)
    {
        if (GridMetaData::safeCast(gridData)) {
            *this = *reinterpret_cast<const GridMetaData*>(gridData);
            //util::memcpy(this, (const GridMetaData*)gridData);
        } else {// otherwise copy each member individually
            mGridData  = *gridData;
            mTreeData  = *reinterpret_cast<const TreeData*>(gridData->treePtr());
            mIndexBBox = gridData->indexBBox();
            mRootTableSize = gridData->rootTableSize();
        }
    }
    GridMetaData& operator=(const GridMetaData&) = default;
    /// @brief return true if the RootData follows right after the TreeData.
    ///        If so, this implies that it's safe to cast the grid from which
    ///        this instance was constructed to a GridMetaData
    __hostdev__ bool safeCast() const { return mTreeData.isRootNext(); }

    /// @brief return true if it is safe to cast the grid to a pointer
    ///        of type GridMetaData, i.e. construction can be avoided.
    __hostdev__ static bool      safeCast(const GridData *gridData){
        NANOVDB_ASSERT(gridData && gridData->isValid());
        return gridData->isRootConnected();
    }
    /// @brief return true if it is safe to cast the grid to a pointer
    ///        of type GridMetaData, i.e. construction can be avoided.
    template<typename T>
    __hostdev__ static bool      safeCast(const NanoGrid<T>& grid){return grid.tree().isRootNext();}
    __hostdev__ bool             isValid() const { return mGridData.isValid(); }
    __hostdev__ const GridType&  gridType() const { return mGridData.mGridType; }
    __hostdev__ const GridClass& gridClass() const { return mGridData.mGridClass; }
    __hostdev__ bool             isLevelSet() const { return mGridData.mGridClass == GridClass::LevelSet; }
    __hostdev__ bool             isFogVolume() const { return mGridData.mGridClass == GridClass::FogVolume; }
    __hostdev__ bool             isStaggered() const { return mGridData.mGridClass == GridClass::Staggered; }
    __hostdev__ bool             isPointIndex() const { return mGridData.mGridClass == GridClass::PointIndex; }
    __hostdev__ bool             isGridIndex() const { return mGridData.mGridClass == GridClass::IndexGrid; }
    __hostdev__ bool             isPointData() const { return mGridData.mGridClass == GridClass::PointData; }
    __hostdev__ bool             isMask() const { return mGridData.mGridClass == GridClass::Topology; }
    __hostdev__ bool             isUnknown() const { return mGridData.mGridClass == GridClass::Unknown; }
    __hostdev__ bool             hasMinMax() const { return mGridData.mFlags.isMaskOn(GridFlags::HasMinMax); }
    __hostdev__ bool             hasBBox() const { return mGridData.mFlags.isMaskOn(GridFlags::HasBBox); }
    __hostdev__ bool             hasLongGridName() const { return mGridData.mFlags.isMaskOn(GridFlags::HasLongGridName); }
    __hostdev__ bool             hasAverage() const { return mGridData.mFlags.isMaskOn(GridFlags::HasAverage); }
    __hostdev__ bool             hasStdDeviation() const { return mGridData.mFlags.isMaskOn(GridFlags::HasStdDeviation); }
    __hostdev__ bool             isBreadthFirst() const { return mGridData.mFlags.isMaskOn(GridFlags::IsBreadthFirst); }
    __hostdev__ uint64_t         gridSize() const { return mGridData.mGridSize; }
    __hostdev__ uint32_t         gridIndex() const { return mGridData.mGridIndex; }
    __hostdev__ uint32_t         gridCount() const { return mGridData.mGridCount; }
    __hostdev__ const char*      shortGridName() const { return mGridData.mGridName; }
    __hostdev__ const Map&       map() const { return mGridData.mMap; }
    __hostdev__ const Vec3dBBox& worldBBox() const { return mGridData.mWorldBBox; }
    __hostdev__ const CoordBBox& indexBBox() const { return mIndexBBox; }
    __hostdev__ Vec3d              voxelSize() const { return mGridData.mVoxelSize; }
    __hostdev__ int                blindDataCount() const { return mGridData.mBlindMetadataCount; }
    __hostdev__ uint64_t        activeVoxelCount() const { return mTreeData.mVoxelCount; }
    __hostdev__ const uint32_t& activeTileCount(uint32_t level) const { return mTreeData.mTileCount[level - 1]; }
    __hostdev__ uint32_t        nodeCount(uint32_t level) const { return mTreeData.mNodeCount[level]; }
    __hostdev__ const Checksum& checksum() const { return mGridData.mChecksum; }
    __hostdev__ uint32_t        rootTableSize() const { return mRootTableSize; }
    __hostdev__ bool            isEmpty() const { return mRootTableSize == 0; }
    __hostdev__ Version         version() const { return mGridData.mVersion; }
}; // GridMetaData

/// @brief Class to access points at a specific voxel location
///
/// @note If GridClass::PointIndex AttT should be uint32_t and if GridClass::PointData Vec3f
template<typename AttT, typename BuildT = uint32_t>
class PointAccessor : public DefaultReadAccessor<BuildT>
{
    using AccT = DefaultReadAccessor<BuildT>;
    const NanoGrid<BuildT>& mGrid;
    const AttT*             mData;

public:
    PointAccessor(const NanoGrid<BuildT>& grid)
        : AccT(grid.tree().root())
        , mGrid(grid)
        , mData(grid.template getBlindData<AttT>(0))
    {
        NANOVDB_ASSERT(grid.gridType() == toGridType<BuildT>());
        NANOVDB_ASSERT((grid.gridClass() == GridClass::PointIndex && util::is_same<uint32_t, AttT>::value) ||
                       (grid.gridClass() == GridClass::PointData && util::is_same<Vec3f, AttT>::value));
    }

    /// @brief  return true if this access was initialized correctly
    __hostdev__ operator bool() const { return mData != nullptr; }

    __hostdev__ const NanoGrid<BuildT>& grid() const { return mGrid; }

    /// @brief Return the total number of point in the grid and set the
    ///        iterators to the complete range of points.
    __hostdev__ uint64_t gridPoints(const AttT*& begin, const AttT*& end) const
    {
        const uint64_t count = mGrid.blindMetaData(0u).mValueCount;
        begin = mData;
        end = begin + count;
        return count;
    }
    /// @brief Return the number of points in the leaf node containing the coordinate @a ijk.
    ///        If this return value is larger than zero then the iterators @a begin and @a end
    ///        will point to all the attributes contained within that leaf node.
    __hostdev__ uint64_t leafPoints(const Coord& ijk, const AttT*& begin, const AttT*& end) const
    {
        auto* leaf = this->probeLeaf(ijk);
        if (leaf == nullptr) {
            return 0;
        }
        begin = mData + leaf->minimum();
        end = begin + leaf->maximum();
        return leaf->maximum();
    }

    /// @brief get iterators over attributes to points at a specific voxel location
    __hostdev__ uint64_t voxelPoints(const Coord& ijk, const AttT*& begin, const AttT*& end) const
    {
        begin = end = nullptr;
        if (auto* leaf = this->probeLeaf(ijk)) {
            const uint32_t offset = NanoLeaf<BuildT>::CoordToOffset(ijk);
            if (leaf->isActive(offset)) {
                begin = mData + leaf->minimum();
                end = begin + leaf->getValue(offset);
                if (offset > 0u)
                    begin += leaf->getValue(offset - 1);
            }
        }
        return end - begin;
    }
}; // PointAccessor

template<typename AttT>
class PointAccessor<AttT, Point> : public DefaultReadAccessor<Point>
{
    using AccT = DefaultReadAccessor<Point>;
    const NanoGrid<Point>& mGrid;
    const AttT*             mData;

public:
    PointAccessor(const NanoGrid<Point>& grid)
        : AccT(grid.tree().root())
        , mGrid(grid)
        , mData(grid.template getBlindData<AttT>(0))
    {
        NANOVDB_ASSERT(mData);
        NANOVDB_ASSERT(grid.gridType() == GridType::PointIndex);
        NANOVDB_ASSERT((grid.gridClass() == GridClass::PointIndex && util::is_same<uint32_t, AttT>::value) ||
                       (grid.gridClass() == GridClass::PointData && util::is_same<Vec3f, AttT>::value) ||
                       (grid.gridClass() == GridClass::PointData && util::is_same<Vec3d, AttT>::value) ||
                       (grid.gridClass() == GridClass::PointData && util::is_same<Vec3u16, AttT>::value) ||
                       (grid.gridClass() == GridClass::PointData && util::is_same<Vec3u8, AttT>::value));
    }

    /// @brief  return true if this access was initialized correctly
    __hostdev__ operator bool() const { return mData != nullptr; }

    __hostdev__ const NanoGrid<Point>& grid() const { return mGrid; }

    /// @brief Return the total number of point in the grid and set the
    ///        iterators to the complete range of points.
    __hostdev__ uint64_t gridPoints(const AttT*& begin, const AttT*& end) const
    {
        const uint64_t count = mGrid.blindMetaData(0u).mValueCount;
        begin = mData;
        end = begin + count;
        return count;
    }
    /// @brief Return the number of points in the leaf node containing the coordinate @a ijk.
    ///        If this return value is larger than zero then the iterators @a begin and @a end
    ///        will point to all the attributes contained within that leaf node.
    __hostdev__ uint64_t leafPoints(const Coord& ijk, const AttT*& begin, const AttT*& end) const
    {
        auto* leaf = this->probeLeaf(ijk);
        if (leaf == nullptr)
            return 0;
        begin = mData + leaf->offset();
        end = begin + leaf->pointCount();
        return leaf->pointCount();
    }

    /// @brief get iterators over attributes to points at a specific voxel location
    __hostdev__ uint64_t voxelPoints(const Coord& ijk, const AttT*& begin, const AttT*& end) const
    {
        if (auto* leaf = this->probeLeaf(ijk)) {
            const uint32_t n = NanoLeaf<Point>::CoordToOffset(ijk);
            if (leaf->isActive(n)) {
                begin = mData + leaf->first(n);
                end = mData + leaf->last(n);
                return end - begin;
            }
        }
        begin = end = nullptr;
        return 0u; // no leaf or inactive voxel
    }
}; // PointAccessor<AttT, Point>

/// @brief Class to access values in channels at a specific voxel location.
///
/// @note The ChannelT template parameter can be either const and non-const.
template<typename ChannelT, typename IndexT = ValueIndex>
class ChannelAccessor : public DefaultReadAccessor<IndexT>
{
    static_assert(BuildTraits<IndexT>::is_index, "Expected an index build type");
    using BaseT = DefaultReadAccessor<IndexT>;

    const NanoGrid<IndexT>& mGrid;
    ChannelT*               mChannel;

public:
    using ValueType = ChannelT;
    using TreeType = NanoTree<IndexT>;
    using AccessorType = ChannelAccessor<ChannelT, IndexT>;

    /// @brief Ctor from an IndexGrid and an integer ID of an internal channel
    ///        that is assumed to exist as blind data in the IndexGrid.
    __hostdev__ ChannelAccessor(const NanoGrid<IndexT>& grid, uint32_t channelID = 0u)
        : BaseT(grid.tree().root())
        , mGrid(grid)
        , mChannel(nullptr)
    {
        NANOVDB_ASSERT(isIndex(grid.gridType()));
        NANOVDB_ASSERT(grid.gridClass() == GridClass::IndexGrid);
        this->setChannel(channelID);
    }

    /// @brief Ctor from an IndexGrid and an external channel
    __hostdev__ ChannelAccessor(const NanoGrid<IndexT>& grid, ChannelT* channelPtr)
        : BaseT(grid.tree().root())
        , mGrid(grid)
        , mChannel(channelPtr)
    {
        NANOVDB_ASSERT(isIndex(grid.gridType()));
        NANOVDB_ASSERT(grid.gridClass() == GridClass::IndexGrid);
    }

    /// @brief  return true if this access was initialized correctly
    __hostdev__ operator bool() const { return mChannel != nullptr; }

    /// @brief Return a const reference to the IndexGrid
    __hostdev__ const NanoGrid<IndexT>& grid() const { return mGrid; }

    /// @brief Return a const reference to the tree of the IndexGrid
    __hostdev__ const TreeType& tree() const { return mGrid.tree(); }

    /// @brief Return a vector of the axial voxel sizes
    __hostdev__ const Vec3d& voxelSize() const { return mGrid.voxelSize(); }

    /// @brief Return total number of values indexed by the IndexGrid
    __hostdev__ const uint64_t& valueCount() const { return mGrid.valueCount(); }

    /// @brief Change to an external channel
    /// @return Pointer to channel data
    __hostdev__ ChannelT* setChannel(ChannelT* channelPtr) {return mChannel = channelPtr;}

    /// @brief Change to an internal channel, assuming it exists as as blind data
    ///        in the IndexGrid.
    /// @return Pointer to channel data, which could be NULL if channelID is out of range or
    ///         if ChannelT does not match the value type of the blind data
    __hostdev__ ChannelT* setChannel(uint32_t channelID)
    {
        return mChannel = const_cast<ChannelT*>(mGrid.template getBlindData<ChannelT>(channelID));
    }

    /// @brief Return the linear offset into a channel that maps to the specified coordinate
    __hostdev__ uint64_t getIndex(const math::Coord& ijk) const { return BaseT::getValue(ijk); }
    __hostdev__ uint64_t idx(int i, int j, int k) const { return BaseT::getValue(math::Coord(i, j, k)); }

    /// @brief Return the value from a cached channel that maps to the specified coordinate
    __hostdev__ ChannelT& getValue(const math::Coord& ijk) const { return mChannel[BaseT::getValue(ijk)]; }
    __hostdev__ ChannelT& operator()(const math::Coord& ijk) const { return this->getValue(ijk); }
    __hostdev__ ChannelT& operator()(int i, int j, int k) const { return this->getValue(math::Coord(i, j, k)); }

    /// @brief return the state and updates the value of the specified voxel
    __hostdev__ bool probeValue(const math::Coord& ijk, typename util::remove_const<ChannelT>::type& v) const
    {
        uint64_t   idx;
        const bool isActive = BaseT::probeValue(ijk, idx);
        v = mChannel[idx];
        return isActive;
    }
    /// @brief Return the value from a specified channel that maps to the specified coordinate
    ///
    /// @note The template parameter can be either const or non-const
    template<typename T>
    __hostdev__ T& getValue(const math::Coord& ijk, T* channelPtr) const { return channelPtr[BaseT::getValue(ijk)]; }

}; // ChannelAccessor

#if 0
// This MiniGridHandle class is only included as a stand-alone example. Note that aligned_alloc is a C++17 feature!
// Normally we recommend using GridHandle defined in util/GridHandle.h but this minimal implementation could be an
// alternative when using the IO methods defined below.
struct MiniGridHandle {
    struct BufferType {
        uint8_t *data;
        uint64_t size;
        BufferType(uint64_t n=0) : data(std::aligned_alloc(NANOVDB_DATA_ALIGNMENT, n)), size(n) {assert(isValid(data));}
        BufferType(BufferType &&other) : data(other.data), size(other.size) {other.data=nullptr; other.size=0;}
        ~BufferType() {std::free(data);}
        BufferType& operator=(const BufferType &other) = delete;
        BufferType& operator=(BufferType &&other){data=other.data; size=other.size; other.data=nullptr; other.size=0; return *this;}
        static BufferType create(size_t n, BufferType* dummy = nullptr) {return BufferType(n);}
    } buffer;
    MiniGridHandle(BufferType &&buf) : buffer(std::move(buf)) {}
    const uint8_t* data() const {return buffer.data;}
};// MiniGridHandle
#endif

namespace io {

/// @brief Define compression codecs
///
/// @note NONE is the default, ZIP is slow but compact and BLOSC offers a great balance.
///
/// @throw NanoVDB optionally supports ZIP and BLOSC compression and will throw an exception
///        if its support is required but missing.
enum class Codec : uint16_t { NONE = 0,
                              ZIP = 1,
                              BLOSC = 2,
                              End = 3,
                              StrLen = 6 + End };

__hostdev__ inline const char* toStr(char *dst, Codec codec)
{
    switch (codec){
        case Codec::NONE:   return util::strcpy(dst, "NONE");
        case Codec::ZIP:    return util::strcpy(dst, "ZIP");
        case Codec::BLOSC : return util::strcpy(dst, "BLOSC");// StrLen = 5 + 1 + End
        default:            return util::strcpy(dst, "END");
    }
}

__hostdev__ inline Codec toCodec(const char *str)
{
    if (util::streq(str, "none"))  return Codec::NONE;
    if (util::streq(str, "zip"))   return Codec::ZIP;
    if (util::streq(str, "blosc")) return Codec::BLOSC;
    return Codec::End;
}

/// @brief Data encoded at the head of each segment of a file or stream.
///
/// @note A file or stream is composed of one or more segments that each contain
//        one or more grids.
struct FileHeader {// 16 bytes
    uint64_t magic;//     8 bytes
    Version  version;//   4 bytes version numbers
    uint16_t gridCount;// 2 bytes
    Codec    codec;//     2 bytes
    bool isValid() const {return magic == NANOVDB_MAGIC_NUMB || magic == NANOVDB_MAGIC_FILE;}
}; // FileHeader ( 16 bytes = 2 words )

// @brief Data encoded for each of the grids associated with a segment.
// Grid size in memory             (uint64_t)   |
// Grid size on disk               (uint64_t)   |
// Grid name hash key              (uint64_t)   |
// Numer of active voxels          (uint64_t)   |
// Grid type                       (uint32_t)   |
// Grid class                      (uint32_t)   |
// Characters in grid name         (uint32_t)   |
// AABB in world space             (2*3*double) | one per grid in file
// AABB in index space             (2*3*int)    |
// Size of a voxel in world units  (3*double)   |
// Byte size of the grid name      (uint32_t)   |
// Number of nodes per level       (4*uint32_t) |
// Numer of active tiles per level (3*uint32_t) |
// Codec for file compression      (uint16_t)   |
// Padding due to 8B alignment     (uint16_t)   |
// Version number                  (uint32_t)   |
struct FileMetaData
{// 176 bytes
    uint64_t    gridSize, fileSize, nameKey, voxelCount; // 4 * 8 = 32B.
    GridType    gridType;  // 4B.
    GridClass   gridClass; // 4B.
    Vec3dBBox   worldBBox; // 2 * 3 * 8 = 48B.
    CoordBBox   indexBBox; // 2 * 3 * 4 = 24B.
    Vec3d       voxelSize; // 24B.
    uint32_t    nameSize;  // 4B.
    uint32_t    nodeCount[4]; //4 x 4 = 16B
    uint32_t    tileCount[3];// 3 x 4 = 12B
    Codec       codec;  // 2B
    uint16_t    padding;// 2B, due to 8B alignment from uint64_t
    Version     version;// 4B
}; // FileMetaData

// the following code block uses std and therefore needs to be ignored by CUDA and HIP
#if !defined(__CUDA_ARCH__) && !defined(__HIP__)

// Note that starting with version 32.6.0 it is possible to write and read raw grid buffers to
// files, e.g. os.write((const char*)&buffer.data(), buffer.size()) or more conveniently as
// handle.write(fileName). In addition to this simple approach we offer the methods below to
// write traditional uncompressed nanovdb files that unlike raw files include metadata that
// is used for tools like nanovdb_print.

///
/// @brief This is a standalone alternative to io::writeGrid(...,Codec::NONE) defined in util/IO.h
///        Unlike the latter this function has no dependencies at all, not even NanoVDB.h, so it also
///        works if client code only includes PNanoVDB.h!
///
/// @details Writes a raw NanoVDB buffer, possibly with multiple grids, to a stream WITHOUT compression.
///          It follows all the conventions in util/IO.h so the stream can be read by all existing client
///          code of NanoVDB.
///
/// @note This method will always write uncompressed grids to the stream, i.e. Blosc or ZIP compression
///       is never applied! This is a fundamental limitation and feature of this standalone function.
///
/// @throw std::invalid_argument if buffer does not point to a valid NanoVDB grid.
///
/// @warning This is pretty ugly code that involves lots of pointer and bit manipulations - not for the faint of heart :)
template<typename StreamT> // StreamT class must support: "void write(const char*, size_t)"
void writeUncompressedGrid(StreamT& os, const GridData* gridData, bool raw = false)
{
    NANOVDB_ASSERT(gridData->mMagic == NANOVDB_MAGIC_NUMB || gridData->mMagic == NANOVDB_MAGIC_GRID);
    NANOVDB_ASSERT(gridData->mVersion.isCompatible());
    if (!raw) {// segment with a single grid:  FileHeader, FileMetaData, gridName, Grid
#ifdef NANOVDB_USE_NEW_MAGIC_NUMBERS
        FileHeader head{NANOVDB_MAGIC_FILE, gridData->mVersion, 1u, Codec::NONE};
#else
        FileHeader head{NANOVDB_MAGIC_NUMB, gridData->mVersion, 1u, Codec::NONE};
#endif
        const char* gridName = gridData->gridName();
        const uint32_t nameSize = util::strlen(gridName) + 1;// include '\0'
        const TreeData* treeData = (const TreeData*)(gridData->treePtr());
        FileMetaData meta{gridData->mGridSize, gridData->mGridSize, 0u, treeData->mVoxelCount,
                          gridData->mGridType, gridData->mGridClass, gridData->mWorldBBox,
                          treeData->bbox(), gridData->mVoxelSize, nameSize,
                          {treeData->mNodeCount[0], treeData->mNodeCount[1], treeData->mNodeCount[2], 1u},
                          {treeData->mTileCount[0], treeData->mTileCount[1], treeData->mTileCount[2]},
                          Codec::NONE, 0u, gridData->mVersion }; // FileMetaData
        os.write((const char*)&head, sizeof(FileHeader)); // write header
        os.write((const char*)&meta, sizeof(FileMetaData)); // write meta data
        os.write(gridName, nameSize); // write grid name
    }
    os.write((const char*)gridData, gridData->mGridSize);// write the grid
}// writeUncompressedGrid

/// @brief  write multiple NanoVDB grids to a single file, without compression.
/// @note To write all grids in a single GridHandle simply use handle.write("fieNane")
template<typename GridHandleT, template<typename...> class VecT>
void writeUncompressedGrids(const char* fileName, const VecT<GridHandleT>& handles, bool raw = false)
{
#ifdef NANOVDB_USE_IOSTREAMS // use this to switch between std::ofstream or FILE implementations
    std::ofstream os(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
#else
    struct StreamT {
        FILE* fptr;
        StreamT(const char* name) { fptr = fopen(name, "wb"); }
        ~StreamT() { fclose(fptr); }
        void write(const char* data, size_t n) { fwrite(data, 1, n, fptr); }
        bool is_open() const { return fptr != NULL; }
    } os(fileName);
#endif
    if (!os.is_open()) {
        fprintf(stderr, "nanovdb::writeUncompressedGrids: Unable to open file \"%s\"for output\n", fileName);
        exit(EXIT_FAILURE);
    }
    for (auto& h : handles) {
        for (uint32_t n=0; n<h.gridCount(); ++n) writeUncompressedGrid(os, h.gridData(n), raw);
    }
} // writeUncompressedGrids

/// @brief read all uncompressed grids from a stream and return their handles.
///
/// @throw std::invalid_argument if stream does not contain a single uncompressed valid NanoVDB grid
///
/// @details StreamT class must support: "bool read(char*, size_t)" and "void skip(uint32_t)"
template<typename GridHandleT, typename StreamT, template<typename...> class VecT>
VecT<GridHandleT> readUncompressedGrids(StreamT& is, const typename GridHandleT::BufferType& pool = typename GridHandleT::BufferType())
{
    VecT<GridHandleT> handles;
    GridData data;
    is.read((char*)&data, sizeof(GridData));
    if (data.isValid()) {// stream contains a raw grid buffer
        uint64_t size = data.mGridSize, sum = 0u;
        while(data.mGridIndex + 1u < data.mGridCount) {
            is.skip(data.mGridSize - sizeof(GridData));// skip grid
            is.read((char*)&data, sizeof(GridData));// read sizeof(GridData) bytes
            sum += data.mGridSize;
        }
        is.skip(-int64_t(sum + sizeof(GridData)));// rewind to start
        auto buffer = GridHandleT::BufferType::create(size + sum, &pool);
        is.read((char*)(buffer.data()), buffer.size());
        handles.emplace_back(std::move(buffer));
    } else {// Header0, MetaData0, gridName0, Grid0...HeaderN, MetaDataN, gridNameN, GridN
        is.skip(-sizeof(GridData));// rewind
        FileHeader head;
        while(is.read((char*)&head, sizeof(FileHeader))) {
            if (!head.isValid()) {
                fprintf(stderr, "nanovdb::readUncompressedGrids: invalid magic number = \"%s\"\n", (const char*)&(head.magic));
                exit(EXIT_FAILURE);
            } else if (!head.version.isCompatible()) {
                char str[20];
                fprintf(stderr, "nanovdb::readUncompressedGrids: invalid major version = \"%s\"\n", toStr(str, head.version));
                exit(EXIT_FAILURE);
            } else if (head.codec != Codec::NONE) {
                char str[8];
                fprintf(stderr, "nanovdb::readUncompressedGrids: invalid codec = \"%s\"\n", toStr(str, head.codec));
                exit(EXIT_FAILURE);
            }
            FileMetaData meta;
            for (uint16_t i = 0; i < head.gridCount; ++i) { // read all grids in segment
                is.read((char*)&meta, sizeof(FileMetaData));// read meta data
                is.skip(meta.nameSize); // skip grid name
                auto buffer = GridHandleT::BufferType::create(meta.gridSize, &pool);
                is.read((char*)buffer.data(), meta.gridSize);// read grid
                handles.emplace_back(std::move(buffer));
            }// loop over grids in segment
        }// loop over segments
    }
    return handles;
} // readUncompressedGrids

/// @brief Read a multiple un-compressed NanoVDB grids from a file and return them as a vector.
template<typename GridHandleT, template<typename...> class VecT>
VecT<GridHandleT> readUncompressedGrids(const char* fileName, const typename GridHandleT::BufferType& buffer = typename GridHandleT::BufferType())
{
#ifdef NANOVDB_USE_IOSTREAMS // use this to switch between std::ifstream or FILE implementations
    struct StreamT : public std::ifstream {
        StreamT(const char* name) : std::ifstream(name, std::ios::in | std::ios::binary){}
        void skip(int64_t off) { this->seekg(off, std::ios_base::cur); }
    };
#else
    struct StreamT {
        FILE* fptr;
        StreamT(const char* name) { fptr = fopen(name, "rb"); }
        ~StreamT() { fclose(fptr); }
        bool read(char* data, size_t n) {
            size_t m = fread(data, 1, n, fptr);
            return n == m;
        }
        void skip(int64_t off) { fseek(fptr, (long int)off, SEEK_CUR); }
        bool is_open() const { return fptr != NULL; }
    };
#endif
    StreamT is(fileName);
    if (!is.is_open()) {
        fprintf(stderr, "nanovdb::readUncompressedGrids: Unable to open file \"%s\"for input\n", fileName);
        exit(EXIT_FAILURE);
    }
    return readUncompressedGrids<GridHandleT, StreamT, VecT>(is, buffer);
} // readUncompressedGrids

#endif // if !defined(__CUDA_ARCH__) && !defined(__HIP__)

} // namespace io

// ----------------------------> Implementations of random access methods <--------------------------------------

/**
* @brief Below is an example of a struct used for random get methods.
* @note All member methods, data, and types are mandatory.
* @code
    template<typename BuildT>
    struct GetOpT {
        using Type = typename BuildToValueMap<BuildT>::Type;// return type
        static constexpr int LEVEL = 0;// minimum level for the descent during top-down traversal
        __hostdev__ static Type get(const NanoRoot<BuildT>& root, args...) { }
        __hostdev__ static Type get(const typename NanoRoot<BuildT>::Tile& tile, args...) { }
        __hostdev__ static Type get(const NanoUpper<BuildT>& node, uint32_t n, args...) { }
        __hostdev__ static Type get(const NanoLower<BuildT>& node, uint32_t n, args...) { }
        __hostdev__ static Type get(const NanoLeaf<BuildT>& leaf,  uint32_t n, args...) { }
   };
  @endcode

  * @brief Below is an example of the struct used for random set methods
  * @note All member methods and data are mandatory.
  * @code
    template<typename BuildT>
    struct SetOpT {
        static constexpr int LEVEL = 0;// minimum level for the descent during top-down traversal
        __hostdev__ static void set(NanoRoot<BuildT>& root, args...) { }
        __hostdev__ static void set(typename NanoRoot<BuildT>::Tile& tile, args...) { }
        __hostdev__ static void set(NanoUpper<BuildT>& node, uint32_t n, args...) { }
        __hostdev__ static void set(NanoLower<BuildT>& node, uint32_t n, args...) { }
        __hostdev__ static void set(NanoLeaf<BuildT>& leaf,  uint32_t n, args...) { }
   };
  @endcode
**/

/// @brief Implements Tree::getValue(math::Coord), i.e. return the value associated with a specific coordinate @c ijk.
/// @tparam BuildT Build type of the grid being called
/// @details The value at a coordinate either maps to the background, a tile value or a leaf value.
template<typename BuildT>
struct GetValue
{
    using Type = typename NanoLeaf<BuildT>::ValueType;
    static constexpr int LEVEL = 0;// minimum level for the descent during top-down traversal
    __hostdev__ static Type get(const NanoRoot<BuildT>& root) { return root.mBackground; }
    __hostdev__ static Type get(const typename NanoRoot<BuildT>::Tile& tile) { return tile.value; }
    __hostdev__ static Type get(const NanoUpper<BuildT>& node, uint32_t n) { return node.mTable[n].value; }
    __hostdev__ static Type get(const NanoLower<BuildT>& node, uint32_t n) { return node.mTable[n].value; }
    __hostdev__ static Type get(const NanoLeaf<BuildT>& leaf,  uint32_t n) { return leaf.getValue(n); } // works with all build types
}; // GetValue<BuildT>

template<typename BuildT>
struct SetValue
{
    static_assert(!BuildTraits<BuildT>::is_special, "SetValue does not support special value types, e.g. Fp4, Fp8, Fp16, FpN");
    using ValueT = typename NanoLeaf<BuildT>::ValueType;
    static constexpr int LEVEL = 0;// minimum level for the descent during top-down traversal
    __hostdev__ static void set(NanoRoot<BuildT>&, const ValueT&) {} // no-op
    __hostdev__ static void set(typename NanoRoot<BuildT>::Tile& tile, const ValueT& v) { tile.value = v; }
    __hostdev__ static void set(NanoUpper<BuildT>& node, uint32_t n, const ValueT& v) { node.mTable[n].value = v; }
    __hostdev__ static void set(NanoLower<BuildT>& node, uint32_t n, const ValueT& v) { node.mTable[n].value = v; }
    __hostdev__ static void set(NanoLeaf<BuildT>& leaf,  uint32_t n, const ValueT& v) { leaf.mValues[n] = v; }
}; // SetValue<BuildT>

template<typename BuildT>
struct SetVoxel
{
    static_assert(!BuildTraits<BuildT>::is_special, "SetVoxel does not support special value types. e.g. Fp4, Fp8, Fp16, FpN");
    using ValueT = typename NanoLeaf<BuildT>::ValueType;
    static constexpr int LEVEL = 0;// minimum level for the descent during top-down traversal
    __hostdev__ static void set(NanoRoot<BuildT>&, const ValueT&) {} // no-op
    __hostdev__ static void set(typename NanoRoot<BuildT>::Tile&, const ValueT&) {} // no-op
    __hostdev__ static void set(NanoUpper<BuildT>&, uint32_t, const ValueT&) {} // no-op
    __hostdev__ static void set(NanoLower<BuildT>&, uint32_t, const ValueT&) {} // no-op
    __hostdev__ static void set(NanoLeaf<BuildT>& leaf, uint32_t n, const ValueT& v) { leaf.mValues[n] = v; }
}; // SetVoxel<BuildT>

/// @brief Implements Tree::isActive(math::Coord)
/// @tparam BuildT Build type of the grid being called
template<typename BuildT>
struct GetState
{
    using Type = bool;
    static constexpr int LEVEL = 0;// minimum level for the descent during top-down traversal
    __hostdev__ static Type get(const NanoRoot<BuildT>&) { return false; }
    __hostdev__ static Type get(const typename NanoRoot<BuildT>::Tile& tile) { return tile.state > 0; }
    __hostdev__ static Type get(const NanoUpper<BuildT>& node, uint32_t n) { return node.mValueMask.isOn(n); }
    __hostdev__ static Type get(const NanoLower<BuildT>& node, uint32_t n) { return node.mValueMask.isOn(n); }
    __hostdev__ static Type get(const NanoLeaf<BuildT>& leaf,  uint32_t n) { return leaf.mValueMask.isOn(n); }
}; // GetState<BuildT>

/// @brief Implements Tree::getDim(math::Coord)
/// @tparam BuildT Build type of the grid being called
template<typename BuildT>
struct GetDim
{
    using Type = uint32_t;
    static constexpr int LEVEL = 0;// minimum level for the descent during top-down traversal
    __hostdev__ static Type get(const NanoRoot<BuildT>&) { return 0u; } // background
    __hostdev__ static Type get(const typename NanoRoot<BuildT>::Tile&) { return 4096u; }
    __hostdev__ static Type get(const NanoUpper<BuildT>&, uint32_t) { return 128u; }
    __hostdev__ static Type get(const NanoLower<BuildT>&, uint32_t) { return 8u; }
    __hostdev__ static Type get(const NanoLeaf<BuildT>&, uint32_t) { return 1u; }
}; // GetDim<BuildT>

/// @brief Return the pointer to the leaf node that contains math::Coord. Implements Tree::probeLeaf(math::Coord)
/// @tparam BuildT Build type of the grid being called
template<typename BuildT>
struct GetLeaf
{
    using Type = const NanoLeaf<BuildT>*;
    static constexpr int LEVEL = 0;// minimum level for the descent during top-down traversal
    __hostdev__ static Type get(const NanoRoot<BuildT>&) { return nullptr; }
    __hostdev__ static Type get(const typename NanoRoot<BuildT>::Tile&) { return nullptr; }
    __hostdev__ static Type get(const NanoUpper<BuildT>&, uint32_t) { return nullptr; }
    __hostdev__ static Type get(const NanoLower<BuildT>&, uint32_t) { return nullptr; }
    __hostdev__ static Type get(const NanoLeaf<BuildT>& leaf, uint32_t) { return &leaf; }
}; // GetLeaf<BuildT>

/// @brief Return point to the lower internal node where math::Coord maps to one of its values, i.e. terminates
/// @tparam BuildT Build type of the grid being called
template<typename BuildT>
struct GetLower
{
    using Type = const NanoLower<BuildT>*;
    static constexpr int LEVEL = 1;// minimum level for the descent during top-down traversal
    __hostdev__ static Type get(const NanoRoot<BuildT>&) { return nullptr; }
    __hostdev__ static Type get(const typename NanoRoot<BuildT>::Tile&) { return nullptr; }
    __hostdev__ static Type get(const NanoUpper<BuildT>&, uint32_t) { return nullptr; }
    __hostdev__ static Type get(const NanoLower<BuildT>& node, uint32_t) { return &node; }
}; // GetLower<BuildT>

/// @brief Return point to the upper internal node where math::Coord maps to one of its values, i.e. terminates
/// @tparam BuildT Build type of the grid being called
template<typename BuildT>
struct GetUpper
{
    using Type = const NanoUpper<BuildT>*;
    static constexpr int LEVEL = 2;// minimum level for the descent during top-down traversal
    __hostdev__ static Type get(const NanoRoot<BuildT>&) { return nullptr; }
    __hostdev__ static Type get(const typename NanoRoot<BuildT>::Tile&) { return nullptr; }
    __hostdev__ static Type get(const NanoUpper<BuildT>& node, uint32_t) { return &node; }
}; // GetUpper<BuildT>

/// @brief Return point to the root Tile where math::Coord maps to one of its values, i.e. terminates
/// @tparam BuildT Build type of the grid being called
template<typename BuildT>
struct GetTile
{
    using Type = const typename NanoRoot<BuildT>::Tile*;
    static constexpr int LEVEL = 3;// minimum level for the descent during top-down traversal
    __hostdev__ static Type get(const NanoRoot<BuildT>&) { return nullptr; }
    __hostdev__ static Type get(const typename NanoRoot<BuildT>::Tile &tile) { return &tile; }
}; // GetTile<BuildT>

/// @brief Implements Tree::probeLeaf(math::Coord)
/// @tparam BuildT Build type of the grid being called
template<typename BuildT>
struct ProbeValue
{
    using Type = bool;
    static constexpr int LEVEL = 0;// minimum level for the descent during top-down traversal
    using ValueT = typename BuildToValueMap<BuildT>::Type;
    __hostdev__ static Type get(const NanoRoot<BuildT>& root, ValueT& v)
    {
        v = root.mBackground;
        return false;
    }
    __hostdev__ static Type get(const typename NanoRoot<BuildT>::Tile& tile, ValueT& v)
    {
        v = tile.value;
        return tile.state > 0u;
    }
    __hostdev__ static Type get(const NanoUpper<BuildT>& node, uint32_t n, ValueT& v)
    {
        v = node.mTable[n].value;
        return node.mValueMask.isOn(n);
    }
    __hostdev__ static Type get(const NanoLower<BuildT>& node, uint32_t n, ValueT& v)
    {
        v = node.mTable[n].value;
        return node.mValueMask.isOn(n);
    }
    __hostdev__ static Type get(const NanoLeaf<BuildT>& leaf, uint32_t n, ValueT& v)
    {
        v = leaf.getValue(n);
        return leaf.mValueMask.isOn(n);
    }
}; // ProbeValue<BuildT>

/// @brief Implements Tree::getNodeInfo(math::Coord)
/// @tparam BuildT Build type of the grid being called
template<typename BuildT>
struct GetNodeInfo
{
    using ValueType = typename NanoLeaf<BuildT>::ValueType;
    using FloatType = typename NanoLeaf<BuildT>::FloatType;
    struct NodeInfo
    {
        uint32_t level, dim;
        ValueType minimum, maximum;
        FloatType average, stdDevi;
        CoordBBox bbox;
    };
    static constexpr int LEVEL = 0;
    using Type = NodeInfo;
    __hostdev__ static Type get(const NanoRoot<BuildT>& root)
    {
        return NodeInfo{3u, NanoUpper<BuildT>::DIM, root.minimum(), root.maximum(), root.average(), root.stdDeviation(), root.bbox()};
    }
    __hostdev__ static Type get(const typename NanoRoot<BuildT>::Tile& tile)
    {
        return NodeInfo{3u, NanoUpper<BuildT>::DIM, tile.value, tile.value, static_cast<FloatType>(tile.value), 0, CoordBBox::createCube(tile.origin(), NanoUpper<BuildT>::DIM)};
    }
    __hostdev__ static Type get(const NanoUpper<BuildT>& node, uint32_t n)
    {
        return NodeInfo{2u, node.dim(), node.minimum(), node.maximum(), node.average(), node.stdDeviation(), node.bbox()};
    }
    __hostdev__ static Type get(const NanoLower<BuildT>& node, uint32_t n)
    {
        return NodeInfo{1u, node.dim(), node.minimum(), node.maximum(), node.average(), node.stdDeviation(), node.bbox()};
    }
    __hostdev__ static Type get(const NanoLeaf<BuildT>& leaf, uint32_t n)
    {
        return NodeInfo{0u, leaf.dim(), leaf.minimum(), leaf.maximum(), leaf.average(), leaf.stdDeviation(), leaf.bbox()};
    }
}; // GetNodeInfo<BuildT>

} // namespace nanovdb ===================================================================

#endif // end of NANOVDB_NANOVDB_H_HAS_BEEN_INCLUDED
