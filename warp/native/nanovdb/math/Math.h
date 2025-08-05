// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   Math.h

    \author Ken Museth

    \date  January 8, 2020

    \brief Math functions and classes

*/

#ifndef NANOVDB_MATH_MATH_H_HAS_BEEN_INCLUDED
#define NANOVDB_MATH_MATH_H_HAS_BEEN_INCLUDED

#include <nanovdb/util/Util.h>// for __hostdev__ and lots of other utility functions

namespace nanovdb {// =================================================================

namespace math {// =============================================================

// ----------------------------> Various math functions <-------------------------------------

//@{
/// @brief Pi constant taken from Boost to match old behaviour
template<typename T>
inline __hostdev__ constexpr T pi()
{
    return 3.141592653589793238462643383279502884e+00;
}
template<>
inline __hostdev__ constexpr float pi()
{
    return 3.141592653589793238462643383279502884e+00F;
}
template<>
inline __hostdev__ constexpr double pi()
{
    return 3.141592653589793238462643383279502884e+00;
}
template<>
inline __hostdev__ constexpr long double pi()
{
    return 3.141592653589793238462643383279502884e+00L;
}
//@}

//@{
/// Tolerance for floating-point comparison
template<typename T>
struct Tolerance;
template<>
struct Tolerance<float>
{
    __hostdev__ static float value() { return 1e-8f; }
};
template<>
struct Tolerance<double>
{
    __hostdev__ static double value() { return 1e-15; }
};
//@}

//@{
/// Delta for small floating-point offsets
template<typename T>
struct Delta;
template<>
struct Delta<float>
{
    __hostdev__ static float value() { return 1e-5f; }
};
template<>
struct Delta<double>
{
    __hostdev__ static double value() { return 1e-9; }
};
//@}

//@{
/// Maximum floating-point values
template<typename T>
struct Maximum;
#if defined(__CUDA_ARCH__) || defined(__HIP__)
template<>
struct Maximum<int>
{
    __hostdev__ static int value() { return 2147483647; }
};
template<>
struct Maximum<uint32_t>
{
    __hostdev__ static uint32_t value() { return 4294967295u; }
};
template<>
struct Maximum<float>
{
    __hostdev__ static float value() { return 1e+38f; }
};
template<>
struct Maximum<double>
{
    __hostdev__ static double value() { return 1e+308; }
};
#else
template<typename T>
struct Maximum
{
    static T value() { return std::numeric_limits<T>::max(); }
};
#endif
//@}

template<typename Type>
__hostdev__ inline bool isApproxZero(const Type& x)
{
    return !(x > Tolerance<Type>::value()) && !(x < -Tolerance<Type>::value());
}

template<typename Type>
__hostdev__ inline Type Min(Type a, Type b)
{
    return (a < b) ? a : b;
}
__hostdev__ inline int32_t Min(int32_t a, int32_t b)
{
    return int32_t(fminf(float(a), float(b)));
}
__hostdev__ inline uint32_t Min(uint32_t a, uint32_t b)
{
    return uint32_t(fminf(float(a), float(b)));
}
__hostdev__ inline float Min(float a, float b)
{
    return fminf(a, b);
}
__hostdev__ inline double Min(double a, double b)
{
    return fmin(a, b);
}
template<typename Type>
__hostdev__ inline Type Max(Type a, Type b)
{
    return (a > b) ? a : b;
}

__hostdev__ inline int32_t Max(int32_t a, int32_t b)
{
    return int32_t(fmaxf(float(a), float(b)));
}
__hostdev__ inline uint32_t Max(uint32_t a, uint32_t b)
{
    return uint32_t(fmaxf(float(a), float(b)));
}
__hostdev__ inline float Max(float a, float b)
{
    return fmaxf(a, b);
}
__hostdev__ inline double Max(double a, double b)
{
    return fmax(a, b);
}
__hostdev__ inline float Clamp(float x, float a, float b)
{
    return Max(Min(x, b), a);
}
__hostdev__ inline double Clamp(double x, double a, double b)
{
    return Max(Min(x, b), a);
}

__hostdev__ inline float Fract(float x)
{
    return x - floorf(x);
}
__hostdev__ inline double Fract(double x)
{
    return x - floor(x);
}

__hostdev__ inline int32_t Floor(float x)
{
    return int32_t(floorf(x));
}
__hostdev__ inline int32_t Floor(double x)
{
    return int32_t(floor(x));
}

__hostdev__ inline int32_t Ceil(float x)
{
    return int32_t(ceilf(x));
}
__hostdev__ inline int32_t Ceil(double x)
{
    return int32_t(ceil(x));
}

template<typename T>
__hostdev__ inline T Pow2(T x)
{
    return x * x;
}

template<typename T>
__hostdev__ inline T Pow3(T x)
{
    return x * x * x;
}

template<typename T>
__hostdev__ inline T Pow4(T x)
{
    return Pow2(x * x);
}
template<typename T>
__hostdev__ inline T Abs(T x)
{
    return x < 0 ? -x : x;
}

template<>
__hostdev__ inline float Abs(float x)
{
    return fabsf(x);
}

template<>
__hostdev__ inline double Abs(double x)
{
    return fabs(x);
}

template<>
__hostdev__ inline int Abs(int x)
{
    return abs(x);
}

template<typename CoordT, typename RealT, template<typename> class Vec3T>
__hostdev__ inline CoordT Round(const Vec3T<RealT>& xyz);

template<typename CoordT, template<typename> class Vec3T>
__hostdev__ inline CoordT Round(const Vec3T<float>& xyz)
{
    return CoordT(int32_t(rintf(xyz[0])), int32_t(rintf(xyz[1])), int32_t(rintf(xyz[2])));
    //return CoordT(int32_t(roundf(xyz[0])), int32_t(roundf(xyz[1])), int32_t(roundf(xyz[2])) );
    //return CoordT(int32_t(floorf(xyz[0] + 0.5f)), int32_t(floorf(xyz[1] + 0.5f)), int32_t(floorf(xyz[2] + 0.5f)));
}

template<typename CoordT, template<typename> class Vec3T>
__hostdev__ inline CoordT Round(const Vec3T<double>& xyz)
{
    return CoordT(int32_t(floor(xyz[0] + 0.5)), int32_t(floor(xyz[1] + 0.5)), int32_t(floor(xyz[2] + 0.5)));
}

template<typename CoordT, typename RealT, template<typename> class Vec3T>
__hostdev__ inline CoordT RoundDown(const Vec3T<RealT>& xyz)
{
    return CoordT(Floor(xyz[0]), Floor(xyz[1]), Floor(xyz[2]));
}

//@{
/// Return the square root of a floating-point value.
__hostdev__ inline float Sqrt(float x)
{
    return sqrtf(x);
}
__hostdev__ inline double Sqrt(double x)
{
    return sqrt(x);
}
//@}

/// Return the sign of the given value as an integer (either -1, 0 or 1).
template<typename T>
__hostdev__ inline T Sign(const T& x)
{
    return ((T(0) < x) ? T(1) : T(0)) - ((x < T(0)) ? T(1) : T(0));
}

template<typename Vec3T>
__hostdev__ inline int MinIndex(const Vec3T& v)
{
#if 0
    static const int hashTable[8] = {2, 1, 9, 1, 2, 9, 0, 0}; //9 are dummy values
    const int        hashKey = ((v[0] < v[1]) << 2) + ((v[0] < v[2]) << 1) + (v[1] < v[2]); // ?*4+?*2+?*1
    return hashTable[hashKey];
#else
    if (v[0] < v[1] && v[0] < v[2])
        return 0;
    if (v[1] < v[2])
        return 1;
    else
        return 2;
#endif
}

template<typename Vec3T>
__hostdev__ inline int MaxIndex(const Vec3T& v)
{
#if 0
    static const int hashTable[8] = {2, 1, 9, 1, 2, 9, 0, 0}; //9 are dummy values
    const int        hashKey = ((v[0] > v[1]) << 2) + ((v[0] > v[2]) << 1) + (v[1] > v[2]); // ?*4+?*2+?*1
    return hashTable[hashKey];
#else
    if (v[0] > v[1] && v[0] > v[2])
        return 0;
    if (v[1] > v[2])
        return 1;
    else
        return 2;
#endif
}

/// @brief round up byteSize to the nearest wordSize, e.g. to align to machine word: AlignUp<sizeof(size_t)(n)
///
/// @details both wordSize and byteSize are in byte units
template<uint64_t wordSize>
__hostdev__ inline uint64_t AlignUp(uint64_t byteCount)
{
    const uint64_t r = byteCount % wordSize;
    return r ? byteCount - r + wordSize : byteCount;
}

// ------------------------------> Coord <--------------------------------------

// forward declaration so we can define Coord::asVec3s and Coord::asVec3d
template<typename>
class Vec3;

/// @brief Signed (i, j, k) 32-bit integer coordinate class, similar to openvdb::math::Coord
class Coord
{
    int32_t mVec[3]; // private member data - three signed index coordinates
public:
    using ValueType = int32_t;
    using IndexType = uint32_t;

    /// @brief Initialize all coordinates to zero.
    __hostdev__ Coord()
        : mVec{0, 0, 0}
    {
    }

    /// @brief Initializes all coordinates to the given signed integer.
    __hostdev__ explicit Coord(ValueType n)
        : mVec{n, n, n}
    {
    }

    /// @brief Initializes coordinate to the given signed integers.
    __hostdev__ Coord(ValueType i, ValueType j, ValueType k)
        : mVec{i, j, k}
    {
    }

    __hostdev__ Coord(ValueType* ptr)
        : mVec{ptr[0], ptr[1], ptr[2]}
    {
    }

    __hostdev__ int32_t x() const { return mVec[0]; }
    __hostdev__ int32_t y() const { return mVec[1]; }
    __hostdev__ int32_t z() const { return mVec[2]; }

    __hostdev__ int32_t& x() { return mVec[0]; }
    __hostdev__ int32_t& y() { return mVec[1]; }
    __hostdev__ int32_t& z() { return mVec[2]; }

    __hostdev__ static Coord max() { return Coord(int32_t((1u << 31) - 1)); }

    __hostdev__ static Coord min() { return Coord(-int32_t((1u << 31) - 1) - 1); }

    __hostdev__ static size_t memUsage() { return sizeof(Coord); }

    /// @brief Return a const reference to the given Coord component.
    /// @warning The argument is assumed to be 0, 1, or 2.
    __hostdev__ const ValueType& operator[](IndexType i) const { return mVec[i]; }

    /// @brief Return a non-const reference to the given Coord component.
    /// @warning The argument is assumed to be 0, 1, or 2.
    __hostdev__ ValueType& operator[](IndexType i) { return mVec[i]; }

    /// @brief Assignment operator that works with openvdb::Coord
    template<typename CoordT>
    __hostdev__ Coord& operator=(const CoordT& other)
    {
        static_assert(sizeof(Coord) == sizeof(CoordT), "Mis-matched sizeof");
        mVec[0] = other[0];
        mVec[1] = other[1];
        mVec[2] = other[2];
        return *this;
    }

    /// @brief Return a new instance with coordinates masked by the given unsigned integer.
    __hostdev__ Coord operator&(IndexType n) const { return Coord(mVec[0] & n, mVec[1] & n, mVec[2] & n); }

    // @brief Return a new instance with coordinates left-shifted by the given unsigned integer.
    __hostdev__ Coord operator<<(IndexType n) const { return Coord(mVec[0] << n, mVec[1] << n, mVec[2] << n); }

    // @brief Return a new instance with coordinates right-shifted by the given unsigned integer.
    __hostdev__ Coord operator>>(IndexType n) const { return Coord(mVec[0] >> n, mVec[1] >> n, mVec[2] >> n); }

    /// @brief Return true if this Coord is lexicographically less than the given Coord.
    __hostdev__ bool operator<(const Coord& rhs) const
    {
        return mVec[0] < rhs[0] ? true
             : mVec[0] > rhs[0] ? false
             : mVec[1] < rhs[1] ? true
             : mVec[1] > rhs[1] ? false
             : mVec[2] < rhs[2] ? true : false;
    }

    /// @brief Return true if this Coord is lexicographically less or equal to the given Coord.
    __hostdev__ bool operator<=(const Coord& rhs) const
    {
        return mVec[0] < rhs[0] ? true
             : mVec[0] > rhs[0] ? false
             : mVec[1] < rhs[1] ? true
             : mVec[1] > rhs[1] ? false
             : mVec[2] <=rhs[2] ? true : false;
    }

    // @brief Return true if this Coord is lexicographically greater than the given Coord.
    __hostdev__ bool operator>(const Coord& rhs) const
    {
        return mVec[0] > rhs[0] ? true
             : mVec[0] < rhs[0] ? false
             : mVec[1] > rhs[1] ? true
             : mVec[1] < rhs[1] ? false
             : mVec[2] > rhs[2] ? true : false;
    }

    // @brief Return true if this Coord is lexicographically greater or equal to the given Coord.
    __hostdev__ bool operator>=(const Coord& rhs) const
    {
        return mVec[0] > rhs[0] ? true
             : mVec[0] < rhs[0] ? false
             : mVec[1] > rhs[1] ? true
             : mVec[1] < rhs[1] ? false
             : mVec[2] >=rhs[2] ? true : false;
    }

    // @brief Return true if the Coord components are identical.
    __hostdev__ bool   operator==(const Coord& rhs) const { return mVec[0] == rhs[0] && mVec[1] == rhs[1] && mVec[2] == rhs[2]; }
    __hostdev__ bool   operator!=(const Coord& rhs) const { return mVec[0] != rhs[0] || mVec[1] != rhs[1] || mVec[2] != rhs[2]; }
    __hostdev__ Coord& operator&=(int n)
    {
        mVec[0] &= n;
        mVec[1] &= n;
        mVec[2] &= n;
        return *this;
    }
    __hostdev__ Coord& operator<<=(uint32_t n)
    {
        mVec[0] <<= n;
        mVec[1] <<= n;
        mVec[2] <<= n;
        return *this;
    }
    __hostdev__ Coord& operator>>=(uint32_t n)
    {
        mVec[0] >>= n;
        mVec[1] >>= n;
        mVec[2] >>= n;
        return *this;
    }
    __hostdev__ Coord& operator+=(int n)
    {
        mVec[0] += n;
        mVec[1] += n;
        mVec[2] += n;
        return *this;
    }
    __hostdev__ Coord  operator+(const Coord& rhs) const { return Coord(mVec[0] + rhs[0], mVec[1] + rhs[1], mVec[2] + rhs[2]); }
    __hostdev__ Coord  operator-(const Coord& rhs) const { return Coord(mVec[0] - rhs[0], mVec[1] - rhs[1], mVec[2] - rhs[2]); }
    __hostdev__ Coord  operator-() const { return Coord(-mVec[0], -mVec[1], -mVec[2]); }
    __hostdev__ Coord& operator+=(const Coord& rhs)
    {
        mVec[0] += rhs[0];
        mVec[1] += rhs[1];
        mVec[2] += rhs[2];
        return *this;
    }
    __hostdev__ Coord& operator-=(const Coord& rhs)
    {
        mVec[0] -= rhs[0];
        mVec[1] -= rhs[1];
        mVec[2] -= rhs[2];
        return *this;
    }

    /// @brief Perform a component-wise minimum with the other Coord.
    __hostdev__ Coord& minComponent(const Coord& other)
    {
        if (other[0] < mVec[0])
            mVec[0] = other[0];
        if (other[1] < mVec[1])
            mVec[1] = other[1];
        if (other[2] < mVec[2])
            mVec[2] = other[2];
        return *this;
    }

    /// @brief Perform a component-wise maximum with the other Coord.
    __hostdev__ Coord& maxComponent(const Coord& other)
    {
        if (other[0] > mVec[0])
            mVec[0] = other[0];
        if (other[1] > mVec[1])
            mVec[1] = other[1];
        if (other[2] > mVec[2])
            mVec[2] = other[2];
        return *this;
    }
#if defined(__CUDACC__) // the following functions only run on the GPU!
    __device__ inline Coord& minComponentAtomic(const Coord& other)
    {
        atomicMin(&mVec[0], other[0]);
        atomicMin(&mVec[1], other[1]);
        atomicMin(&mVec[2], other[2]);
        return *this;
    }
    __device__ inline Coord& maxComponentAtomic(const Coord& other)
    {
        atomicMax(&mVec[0], other[0]);
        atomicMax(&mVec[1], other[1]);
        atomicMax(&mVec[2], other[2]);
        return *this;
    }
#endif

    __hostdev__ Coord offsetBy(ValueType dx, ValueType dy, ValueType dz) const
    {
        return Coord(mVec[0] + dx, mVec[1] + dy, mVec[2] + dz);
    }

    __hostdev__ Coord offsetBy(ValueType n) const { return this->offsetBy(n, n, n); }

    /// Return true if any of the components of @a a are smaller than the
    /// corresponding components of @a b.
    __hostdev__ static inline bool lessThan(const Coord& a, const Coord& b)
    {
        return (a[0] < b[0] || a[1] < b[1] || a[2] < b[2]);
    }

    /// @brief Return the largest integer coordinates that are not greater
    /// than @a xyz (node centered conversion).
    template<typename Vec3T>
    __hostdev__ static Coord Floor(const Vec3T& xyz) { return Coord(math::Floor(xyz[0]), math::Floor(xyz[1]), math::Floor(xyz[2])); }

    /// @brief Return a hash key derived from the existing coordinates.
    /// @details The hash function is originally taken from the SIGGRAPH paper:
    ///          "VDB: High-resolution sparse volumes with dynamic topology"
    ///          and the prime numbers are modified based on the ACM Transactions on Graphics paper:
    ///          "Real-time 3D reconstruction at scale using voxel hashing" (the second number had a typo!)
    template<int Log2N = 3 + 4 + 5>
    __hostdev__ uint32_t hash() const { return ((1 << Log2N) - 1) & (mVec[0] * 73856093 ^ mVec[1] * 19349669 ^ mVec[2] * 83492791); }

    /// @brief Return the octant of this Coord
    //__hostdev__ size_t octant() const { return (uint32_t(mVec[0])>>31) | ((uint32_t(mVec[1])>>31)<<1) | ((uint32_t(mVec[2])>>31)<<2); }
    __hostdev__ uint8_t octant() const { return (uint8_t(bool(mVec[0] & (1u << 31)))) |
                                                (uint8_t(bool(mVec[1] & (1u << 31))) << 1) |
                                                (uint8_t(bool(mVec[2] & (1u << 31))) << 2); }

    /// @brief Return a single precision floating-point vector of this coordinate
    __hostdev__ inline Vec3<float> asVec3s() const;

    /// @brief Return a double precision floating-point vector of this coordinate
    __hostdev__ inline Vec3<double> asVec3d() const;

    // returns a copy of itself, so it mimics the behaviour of Vec3<T>::round()
    __hostdev__ inline Coord round() const { return *this; }
}; // Coord class

// ----------------------------> Vec3 <--------------------------------------

/// @brief A simple vector class with three components, similar to openvdb::math::Vec3
template<typename T>
class Vec3
{
    T mVec[3];

public:
    static const int SIZE = 3;
    static const int size = 3; // in openvdb::math::Tuple
    using ValueType = T;
    Vec3() = default;
    __hostdev__ explicit Vec3(T x)
        : mVec{x, x, x}
    {
    }
    __hostdev__ Vec3(T x, T y, T z)
        : mVec{x, y, z}
    {
    }
    template<template<class> class Vec3T, class T2>
    __hostdev__ Vec3(const Vec3T<T2>& v)
        : mVec{T(v[0]), T(v[1]), T(v[2])}
    {
        static_assert(Vec3T<T2>::size == size, "expected Vec3T::size==3!");
    }
    template<typename T2>
    __hostdev__ explicit Vec3(const Vec3<T2>& v)
        : mVec{T(v[0]), T(v[1]), T(v[2])}
    {
    }
    __hostdev__ explicit Vec3(const Coord& ijk)
        : mVec{T(ijk[0]), T(ijk[1]), T(ijk[2])}
    {
    }
    __hostdev__ bool operator==(const Vec3& rhs) const { return mVec[0] == rhs[0] && mVec[1] == rhs[1] && mVec[2] == rhs[2]; }
    __hostdev__ bool operator!=(const Vec3& rhs) const { return mVec[0] != rhs[0] || mVec[1] != rhs[1] || mVec[2] != rhs[2]; }
    template<template<class> class Vec3T, class T2>
    __hostdev__ Vec3& operator=(const Vec3T<T2>& rhs)
    {
        static_assert(Vec3T<T2>::size == size, "expected Vec3T::size==3!");
        mVec[0] = rhs[0];
        mVec[1] = rhs[1];
        mVec[2] = rhs[2];
        return *this;
    }
    __hostdev__ const T& operator[](int i) const { return mVec[i]; }
    __hostdev__ T&       operator[](int i) { return mVec[i]; }
    template<typename Vec3T>
    __hostdev__ T dot(const Vec3T& v) const { return mVec[0] * v[0] + mVec[1] * v[1] + mVec[2] * v[2]; }
    template<typename Vec3T>
    __hostdev__ Vec3 cross(const Vec3T& v) const
    {
        return Vec3(mVec[1] * v[2] - mVec[2] * v[1],
                    mVec[2] * v[0] - mVec[0] * v[2],
                    mVec[0] * v[1] - mVec[1] * v[0]);
    }
    __hostdev__ T lengthSqr() const
    {
        return mVec[0] * mVec[0] + mVec[1] * mVec[1] + mVec[2] * mVec[2]; // 5 flops
    }
    __hostdev__ T     length() const { return Sqrt(this->lengthSqr()); }
    __hostdev__ Vec3  operator-() const { return Vec3(-mVec[0], -mVec[1], -mVec[2]); }
    __hostdev__ Vec3  operator*(const Vec3& v) const { return Vec3(mVec[0] * v[0], mVec[1] * v[1], mVec[2] * v[2]); }
    __hostdev__ Vec3  operator/(const Vec3& v) const { return Vec3(mVec[0] / v[0], mVec[1] / v[1], mVec[2] / v[2]); }
    __hostdev__ Vec3  operator+(const Vec3& v) const { return Vec3(mVec[0] + v[0], mVec[1] + v[1], mVec[2] + v[2]); }
    __hostdev__ Vec3  operator-(const Vec3& v) const { return Vec3(mVec[0] - v[0], mVec[1] - v[1], mVec[2] - v[2]); }
    __hostdev__ Vec3  operator+(const Coord& ijk) const { return Vec3(mVec[0] + ijk[0], mVec[1] + ijk[1], mVec[2] + ijk[2]); }
    __hostdev__ Vec3  operator-(const Coord& ijk) const { return Vec3(mVec[0] - ijk[0], mVec[1] - ijk[1], mVec[2] - ijk[2]); }
    __hostdev__ Vec3  operator*(const T& s) const { return Vec3(s * mVec[0], s * mVec[1], s * mVec[2]); }
    __hostdev__ Vec3  operator/(const T& s) const { return (T(1) / s) * (*this); }
    __hostdev__ Vec3& operator+=(const Vec3& v)
    {
        mVec[0] += v[0];
        mVec[1] += v[1];
        mVec[2] += v[2];
        return *this;
    }
    __hostdev__ Vec3& operator+=(const Coord& ijk)
    {
        mVec[0] += T(ijk[0]);
        mVec[1] += T(ijk[1]);
        mVec[2] += T(ijk[2]);
        return *this;
    }
    __hostdev__ Vec3& operator-=(const Vec3& v)
    {
        mVec[0] -= v[0];
        mVec[1] -= v[1];
        mVec[2] -= v[2];
        return *this;
    }
    __hostdev__ Vec3& operator-=(const Coord& ijk)
    {
        mVec[0] -= T(ijk[0]);
        mVec[1] -= T(ijk[1]);
        mVec[2] -= T(ijk[2]);
        return *this;
    }
    __hostdev__ Vec3& operator*=(const T& s)
    {
        mVec[0] *= s;
        mVec[1] *= s;
        mVec[2] *= s;
        return *this;
    }
    __hostdev__ Vec3& operator/=(const T& s) { return (*this) *= T(1) / s; }
    __hostdev__ Vec3& normalize() { return (*this) /= this->length(); }
    /// @brief Perform a component-wise minimum with the other Coord.
    __hostdev__ Vec3& minComponent(const Vec3& other)
    {
        if (other[0] < mVec[0])
            mVec[0] = other[0];
        if (other[1] < mVec[1])
            mVec[1] = other[1];
        if (other[2] < mVec[2])
            mVec[2] = other[2];
        return *this;
    }

    /// @brief Perform a component-wise maximum with the other Coord.
    __hostdev__ Vec3& maxComponent(const Vec3& other)
    {
        if (other[0] > mVec[0])
            mVec[0] = other[0];
        if (other[1] > mVec[1])
            mVec[1] = other[1];
        if (other[2] > mVec[2])
            mVec[2] = other[2];
        return *this;
    }
    /// @brief Return the smallest vector component
    __hostdev__ ValueType min() const
    {
        return mVec[0] < mVec[1] ? (mVec[0] < mVec[2] ? mVec[0] : mVec[2]) : (mVec[1] < mVec[2] ? mVec[1] : mVec[2]);
    }
    /// @brief Return the largest vector component
    __hostdev__ ValueType max() const
    {
        return mVec[0] > mVec[1] ? (mVec[0] > mVec[2] ? mVec[0] : mVec[2]) : (mVec[1] > mVec[2] ? mVec[1] : mVec[2]);
    }
    /// @brief Round each component if this Vec<T> up to its integer value
    /// @return Return an integer Coord
    __hostdev__ Coord floor() const { return Coord(Floor(mVec[0]), Floor(mVec[1]), Floor(mVec[2])); }
    /// @brief Round each component if this Vec<T> down to its integer value
    /// @return Return an integer Coord
    __hostdev__ Coord ceil() const { return Coord(Ceil(mVec[0]), Ceil(mVec[1]), Ceil(mVec[2])); }
    /// @brief Round each component if this Vec<T> to its closest integer value
    /// @return Return an integer Coord
    __hostdev__ Coord round() const
    {
        if constexpr(util::is_same<T, float>::value) {
            return Coord(Floor(mVec[0] + 0.5f), Floor(mVec[1] + 0.5f), Floor(mVec[2] + 0.5f));
        } else if constexpr(util::is_same<T, int>::value) {
            return Coord(mVec[0], mVec[1], mVec[2]);
        } else {
            return Coord(Floor(mVec[0] + 0.5), Floor(mVec[1] + 0.5), Floor(mVec[2] + 0.5));
        }
    }

    /// @brief return a non-const raw constant pointer to array of three vector components
    __hostdev__ T* asPointer() { return mVec; }
    /// @brief return a const raw constant pointer to array of three vector components
    __hostdev__ const T* asPointer() const { return mVec; }
}; // Vec3<T>

template<typename T1, typename T2>
__hostdev__ inline Vec3<T2> operator*(T1 scalar, const Vec3<T2>& vec)
{
    return Vec3<T2>(scalar * vec[0], scalar * vec[1], scalar * vec[2]);
}
template<typename T1, typename T2>
__hostdev__ inline Vec3<T2> operator/(T1 scalar, const Vec3<T2>& vec)
{
    return Vec3<T2>(scalar / vec[0], scalar / vec[1], scalar / vec[2]);
}

/// @brief Return a single precision floating-point vector of this coordinate
__hostdev__ inline Vec3<float> Coord::asVec3s() const
{
    return Vec3<float>(float(mVec[0]), float(mVec[1]), float(mVec[2]));
}

/// @brief Return a double precision floating-point vector of this coordinate
__hostdev__ inline Vec3<double> Coord::asVec3d() const
{
    return Vec3<double>(double(mVec[0]), double(mVec[1]), double(mVec[2]));
}

// ----------------------------> Vec4 <--------------------------------------

/// @brief A simple vector class with four components, similar to openvdb::math::Vec4
template<typename T>
class Vec4
{
    T mVec[4];

public:
    static const int SIZE = 4;
    static const int size = 4;
    using ValueType = T;
    Vec4() = default;
    __hostdev__ explicit Vec4(T x)
        : mVec{x, x, x, x}
    {
    }
    __hostdev__ Vec4(T x, T y, T z, T w)
        : mVec{x, y, z, w}
    {
    }
    template<typename T2>
    __hostdev__ explicit Vec4(const Vec4<T2>& v)
        : mVec{T(v[0]), T(v[1]), T(v[2]), T(v[3])}
    {
    }
    template<template<class> class Vec4T, class T2>
    __hostdev__ Vec4(const Vec4T<T2>& v)
        : mVec{T(v[0]), T(v[1]), T(v[2]), T(v[3])}
    {
        static_assert(Vec4T<T2>::size == size, "expected Vec4T::size==4!");
    }
    __hostdev__ bool operator==(const Vec4& rhs) const { return mVec[0] == rhs[0] && mVec[1] == rhs[1] && mVec[2] == rhs[2] && mVec[3] == rhs[3]; }
    __hostdev__ bool operator!=(const Vec4& rhs) const { return mVec[0] != rhs[0] || mVec[1] != rhs[1] || mVec[2] != rhs[2] || mVec[3] != rhs[3]; }
    template<template<class> class Vec4T, class T2>
    __hostdev__ Vec4& operator=(const Vec4T<T2>& rhs)
    {
        static_assert(Vec4T<T2>::size == size, "expected Vec4T::size==4!");
        mVec[0] = rhs[0];
        mVec[1] = rhs[1];
        mVec[2] = rhs[2];
        mVec[3] = rhs[3];
        return *this;
    }

    __hostdev__ const T& operator[](int i) const { return mVec[i]; }
    __hostdev__ T&       operator[](int i) { return mVec[i]; }
    template<typename Vec4T>
    __hostdev__ T dot(const Vec4T& v) const { return mVec[0] * v[0] + mVec[1] * v[1] + mVec[2] * v[2] + mVec[3] * v[3]; }
    __hostdev__ T lengthSqr() const
    {
        return mVec[0] * mVec[0] + mVec[1] * mVec[1] + mVec[2] * mVec[2] + mVec[3] * mVec[3]; // 7 flops
    }
    __hostdev__ T     length() const { return Sqrt(this->lengthSqr()); }
    __hostdev__ Vec4  operator-() const { return Vec4(-mVec[0], -mVec[1], -mVec[2], -mVec[3]); }
    __hostdev__ Vec4  operator*(const Vec4& v) const { return Vec4(mVec[0] * v[0], mVec[1] * v[1], mVec[2] * v[2], mVec[3] * v[3]); }
    __hostdev__ Vec4  operator/(const Vec4& v) const { return Vec4(mVec[0] / v[0], mVec[1] / v[1], mVec[2] / v[2], mVec[3] / v[3]); }
    __hostdev__ Vec4  operator+(const Vec4& v) const { return Vec4(mVec[0] + v[0], mVec[1] + v[1], mVec[2] + v[2], mVec[3] + v[3]); }
    __hostdev__ Vec4  operator-(const Vec4& v) const { return Vec4(mVec[0] - v[0], mVec[1] - v[1], mVec[2] - v[2], mVec[3] - v[3]); }
    __hostdev__ Vec4  operator*(const T& s) const { return Vec4(s * mVec[0], s * mVec[1], s * mVec[2], s * mVec[3]); }
    __hostdev__ Vec4  operator/(const T& s) const { return (T(1) / s) * (*this); }
    __hostdev__ Vec4& operator+=(const Vec4& v)
    {
        mVec[0] += v[0];
        mVec[1] += v[1];
        mVec[2] += v[2];
        mVec[3] += v[3];
        return *this;
    }
    __hostdev__ Vec4& operator-=(const Vec4& v)
    {
        mVec[0] -= v[0];
        mVec[1] -= v[1];
        mVec[2] -= v[2];
        mVec[3] -= v[3];
        return *this;
    }
    __hostdev__ Vec4& operator*=(const T& s)
    {
        mVec[0] *= s;
        mVec[1] *= s;
        mVec[2] *= s;
        mVec[3] *= s;
        return *this;
    }
    __hostdev__ Vec4& operator/=(const T& s) { return (*this) *= T(1) / s; }
    __hostdev__ Vec4& normalize() { return (*this) /= this->length(); }
    /// @brief Perform a component-wise minimum with the other Coord.
    __hostdev__ Vec4& minComponent(const Vec4& other)
    {
        if (other[0] < mVec[0])
            mVec[0] = other[0];
        if (other[1] < mVec[1])
            mVec[1] = other[1];
        if (other[2] < mVec[2])
            mVec[2] = other[2];
        if (other[3] < mVec[3])
            mVec[3] = other[3];
        return *this;
    }

    /// @brief Perform a component-wise maximum with the other Coord.
    __hostdev__ Vec4& maxComponent(const Vec4& other)
    {
        if (other[0] > mVec[0])
            mVec[0] = other[0];
        if (other[1] > mVec[1])
            mVec[1] = other[1];
        if (other[2] > mVec[2])
            mVec[2] = other[2];
        if (other[3] > mVec[3])
            mVec[3] = other[3];
        return *this;
    }
}; // Vec4<T>

template<typename T1, typename T2>
__hostdev__ inline Vec4<T2> operator*(T1 scalar, const Vec4<T2>& vec)
{
    return Vec4<T2>(scalar * vec[0], scalar * vec[1], scalar * vec[2], scalar * vec[3]);
}
template<typename T1, typename T2>
__hostdev__ inline Vec4<T2> operator/(T1 scalar, const Vec4<T2>& vec)
{
    return Vec4<T2>(scalar / vec[0], scalar / vec[1], scalar / vec[2], scalar / vec[3]);
}

// ----------------------------> matMult <--------------------------------------

/// @brief Multiply a 3x3 matrix and a 3d vector using 32bit floating point arithmetics
/// @note This corresponds to a linear mapping, e.g. scaling, rotation etc.
/// @tparam Vec3T Template type of the input and output 3d vectors
/// @param mat pointer to an array of floats with the 3x3 matrix
/// @param xyz input vector to be multiplied by the matrix
/// @return result of matrix-vector multiplication, i.e. mat x xyz
template<typename Vec3T>
__hostdev__ inline Vec3T matMult(const float* mat, const Vec3T& xyz)
{
    return Vec3T(fmaf(static_cast<float>(xyz[0]), mat[0], fmaf(static_cast<float>(xyz[1]), mat[1], static_cast<float>(xyz[2]) * mat[2])),
                 fmaf(static_cast<float>(xyz[0]), mat[3], fmaf(static_cast<float>(xyz[1]), mat[4], static_cast<float>(xyz[2]) * mat[5])),
                 fmaf(static_cast<float>(xyz[0]), mat[6], fmaf(static_cast<float>(xyz[1]), mat[7], static_cast<float>(xyz[2]) * mat[8]))); // 6 fmaf + 3 mult = 9 flops
}

/// @brief Multiply a 3x3 matrix and a 3d vector using 64bit floating point arithmetics
/// @note This corresponds to a linear mapping, e.g. scaling, rotation etc.
/// @tparam Vec3T Template type of the input and output 3d vectors
/// @param mat pointer to an array of floats with the 3x3 matrix
/// @param xyz input vector to be multiplied by the matrix
/// @return result of matrix-vector multiplication, i.e. mat x xyz
template<typename Vec3T>
__hostdev__ inline Vec3T matMult(const double* mat, const Vec3T& xyz)
{
    return Vec3T(fma(static_cast<double>(xyz[0]), mat[0], fma(static_cast<double>(xyz[1]), mat[1], static_cast<double>(xyz[2]) * mat[2])),
                 fma(static_cast<double>(xyz[0]), mat[3], fma(static_cast<double>(xyz[1]), mat[4], static_cast<double>(xyz[2]) * mat[5])),
                 fma(static_cast<double>(xyz[0]), mat[6], fma(static_cast<double>(xyz[1]), mat[7], static_cast<double>(xyz[2]) * mat[8]))); // 6 fmaf + 3 mult = 9 flops
}

/// @brief Multiply a 3x3 matrix to a 3d vector and add another 3d vector using 32bit floating point arithmetics
/// @note This corresponds to an affine transformation, i.e a linear mapping followed by a translation. e.g. scale/rotation and translation
/// @tparam Vec3T Template type of the input and output 3d vectors
/// @param mat pointer to an array of floats with the 3x3 matrix
/// @param vec 3d vector to be added AFTER the matrix multiplication
/// @param xyz input vector to be multiplied by the matrix and a translated by @c vec
/// @return result of affine transformation, i.e. (mat x xyz) + vec
template<typename Vec3T>
__hostdev__ inline Vec3T matMult(const float* mat, const float* vec, const Vec3T& xyz)
{
    return Vec3T(fmaf(static_cast<float>(xyz[0]), mat[0], fmaf(static_cast<float>(xyz[1]), mat[1], fmaf(static_cast<float>(xyz[2]), mat[2], vec[0]))),
                 fmaf(static_cast<float>(xyz[0]), mat[3], fmaf(static_cast<float>(xyz[1]), mat[4], fmaf(static_cast<float>(xyz[2]), mat[5], vec[1]))),
                 fmaf(static_cast<float>(xyz[0]), mat[6], fmaf(static_cast<float>(xyz[1]), mat[7], fmaf(static_cast<float>(xyz[2]), mat[8], vec[2])))); // 9 fmaf = 9 flops
}

/// @brief Multiply a 3x3 matrix to a 3d vector and add another 3d vector using 64bit floating point arithmetics
/// @note This corresponds to an affine transformation, i.e a linear mapping followed by a translation. e.g. scale/rotation and translation
/// @tparam Vec3T Template type of the input and output 3d vectors
/// @param mat pointer to an array of floats with the 3x3 matrix
/// @param vec 3d vector to be added AFTER the matrix multiplication
/// @param xyz input vector to be multiplied by the matrix and a translated by @c vec
/// @return result of affine transformation, i.e. (mat x xyz) + vec
template<typename Vec3T>
__hostdev__ inline Vec3T matMult(const double* mat, const double* vec, const Vec3T& xyz)
{
    return Vec3T(fma(static_cast<double>(xyz[0]), mat[0], fma(static_cast<double>(xyz[1]), mat[1], fma(static_cast<double>(xyz[2]), mat[2], vec[0]))),
                 fma(static_cast<double>(xyz[0]), mat[3], fma(static_cast<double>(xyz[1]), mat[4], fma(static_cast<double>(xyz[2]), mat[5], vec[1]))),
                 fma(static_cast<double>(xyz[0]), mat[6], fma(static_cast<double>(xyz[1]), mat[7], fma(static_cast<double>(xyz[2]), mat[8], vec[2])))); // 9 fma = 9 flops
}

/// @brief Multiply the transposed of a 3x3 matrix and a 3d vector using 32bit floating point arithmetics
/// @note This corresponds to an inverse linear mapping, e.g. inverse scaling, inverse rotation etc.
/// @tparam Vec3T Template type of the input and output 3d vectors
/// @param mat pointer to an array of floats with the 3x3 matrix
/// @param xyz input vector to be multiplied by the transposed matrix
/// @return result of matrix-vector multiplication, i.e. mat^T x xyz
template<typename Vec3T>
__hostdev__ inline Vec3T matMultT(const float* mat, const Vec3T& xyz)
{
    return Vec3T(fmaf(static_cast<float>(xyz[0]), mat[0], fmaf(static_cast<float>(xyz[1]), mat[3], static_cast<float>(xyz[2]) * mat[6])),
                 fmaf(static_cast<float>(xyz[0]), mat[1], fmaf(static_cast<float>(xyz[1]), mat[4], static_cast<float>(xyz[2]) * mat[7])),
                 fmaf(static_cast<float>(xyz[0]), mat[2], fmaf(static_cast<float>(xyz[1]), mat[5], static_cast<float>(xyz[2]) * mat[8]))); // 6 fmaf + 3 mult = 9 flops
}

/// @brief Multiply the transposed of a 3x3 matrix and a 3d vector using 64bit floating point arithmetics
/// @note This corresponds to an inverse linear mapping, e.g. inverse scaling, inverse rotation etc.
/// @tparam Vec3T Template type of the input and output 3d vectors
/// @param mat pointer to an array of floats with the 3x3 matrix
/// @param xyz input vector to be multiplied by the transposed matrix
/// @return result of matrix-vector multiplication, i.e. mat^T x xyz
template<typename Vec3T>
__hostdev__ inline Vec3T matMultT(const double* mat, const Vec3T& xyz)
{
    return Vec3T(fma(static_cast<double>(xyz[0]), mat[0], fma(static_cast<double>(xyz[1]), mat[3], static_cast<double>(xyz[2]) * mat[6])),
                 fma(static_cast<double>(xyz[0]), mat[1], fma(static_cast<double>(xyz[1]), mat[4], static_cast<double>(xyz[2]) * mat[7])),
                 fma(static_cast<double>(xyz[0]), mat[2], fma(static_cast<double>(xyz[1]), mat[5], static_cast<double>(xyz[2]) * mat[8]))); // 6 fmaf + 3 mult = 9 flops
}

template<typename Vec3T>
__hostdev__ inline Vec3T matMultT(const float* mat, const float* vec, const Vec3T& xyz)
{
    return Vec3T(fmaf(static_cast<float>(xyz[0]), mat[0], fmaf(static_cast<float>(xyz[1]), mat[3], fmaf(static_cast<float>(xyz[2]), mat[6], vec[0]))),
                 fmaf(static_cast<float>(xyz[0]), mat[1], fmaf(static_cast<float>(xyz[1]), mat[4], fmaf(static_cast<float>(xyz[2]), mat[7], vec[1]))),
                 fmaf(static_cast<float>(xyz[0]), mat[2], fmaf(static_cast<float>(xyz[1]), mat[5], fmaf(static_cast<float>(xyz[2]), mat[8], vec[2])))); // 9 fmaf = 9 flops
}

template<typename Vec3T>
__hostdev__ inline Vec3T matMultT(const double* mat, const double* vec, const Vec3T& xyz)
{
    return Vec3T(fma(static_cast<double>(xyz[0]), mat[0], fma(static_cast<double>(xyz[1]), mat[3], fma(static_cast<double>(xyz[2]), mat[6], vec[0]))),
                 fma(static_cast<double>(xyz[0]), mat[1], fma(static_cast<double>(xyz[1]), mat[4], fma(static_cast<double>(xyz[2]), mat[7], vec[1]))),
                 fma(static_cast<double>(xyz[0]), mat[2], fma(static_cast<double>(xyz[1]), mat[5], fma(static_cast<double>(xyz[2]), mat[8], vec[2])))); // 9 fma = 9 flops
}

// ----------------------------> BBox <-------------------------------------

// Base-class for static polymorphism (cannot be constructed directly)
template<typename Vec3T>
struct BaseBBox
{
    Vec3T                    mCoord[2];
    __hostdev__ bool         operator==(const BaseBBox& rhs) const { return mCoord[0] == rhs.mCoord[0] && mCoord[1] == rhs.mCoord[1]; };
    __hostdev__ bool         operator!=(const BaseBBox& rhs) const { return mCoord[0] != rhs.mCoord[0] || mCoord[1] != rhs.mCoord[1]; };
    __hostdev__ const Vec3T& operator[](int i) const { return mCoord[i]; }
    __hostdev__ Vec3T&       operator[](int i) { return mCoord[i]; }
    __hostdev__ Vec3T&       min() { return mCoord[0]; }
    __hostdev__ Vec3T&       max() { return mCoord[1]; }
    __hostdev__ const Vec3T& min() const { return mCoord[0]; }
    __hostdev__ const Vec3T& max() const { return mCoord[1]; }
    __hostdev__ BaseBBox&    translate(const Vec3T& xyz)
    {
        mCoord[0] += xyz;
        mCoord[1] += xyz;
        return *this;
    }
    /// @brief Expand this bounding box to enclose point @c xyz.
    __hostdev__ BaseBBox& expand(const Vec3T& xyz)
    {
        mCoord[0].minComponent(xyz);
        mCoord[1].maxComponent(xyz);
        return *this;
    }

    /// @brief Expand this bounding box to enclose the given bounding box.
    __hostdev__ BaseBBox& expand(const BaseBBox& bbox)
    {
        mCoord[0].minComponent(bbox[0]);
        mCoord[1].maxComponent(bbox[1]);
        return *this;
    }

    /// @brief Intersect this bounding box with the given bounding box.
    __hostdev__ BaseBBox& intersect(const BaseBBox& bbox)
    {
        mCoord[0].maxComponent(bbox[0]);
        mCoord[1].minComponent(bbox[1]);
        return *this;
    }

//    __hostdev__ BaseBBox expandBy(typename Vec3T::ValueType padding) const
//    {
//        return BaseBBox(mCoord[0].offsetBy(-padding),mCoord[1].offsetBy(padding));
//    }
    __hostdev__ bool isInside(const Vec3T& xyz)
    {
        if (xyz[0] < mCoord[0][0] || xyz[1] < mCoord[0][1] || xyz[2] < mCoord[0][2])
            return false;
        if (xyz[0] > mCoord[1][0] || xyz[1] > mCoord[1][1] || xyz[2] > mCoord[1][2])
            return false;
        return true;
    }

protected:
    __hostdev__ BaseBBox() {}
    __hostdev__ BaseBBox(const Vec3T& min, const Vec3T& max)
        : mCoord{min, max}
    {
    }
}; // BaseBBox

template<typename Vec3T, bool = util::is_floating_point<typename Vec3T::ValueType>::value>
struct BBox;

/// @brief Partial template specialization for floating point coordinate types.
///
/// @note Min is inclusive and max is exclusive. If min = max the dimension of
///       the bounding box is zero and therefore it is also empty.
template<typename Vec3T>
struct BBox<Vec3T, true> : public BaseBBox<Vec3T>
{
    using Vec3Type = Vec3T;
    using ValueType = typename Vec3T::ValueType;
    static_assert(util::is_floating_point<ValueType>::value, "Expected a floating point coordinate type");
    using BaseT = BaseBBox<Vec3T>;
    using BaseT::mCoord;
    /// @brief Default construction sets BBox to an empty bbox
    __hostdev__ BBox()
        : BaseT(Vec3T( Maximum<typename Vec3T::ValueType>::value()),
                Vec3T(-Maximum<typename Vec3T::ValueType>::value()))
    {
    }
    __hostdev__ BBox(const Vec3T& min, const Vec3T& max)
        : BaseT(min, max)
    {
    }
    __hostdev__ BBox(const Coord& min, const Coord& max)
        : BaseT(Vec3T(ValueType(min[0]), ValueType(min[1]), ValueType(min[2])),
                Vec3T(ValueType(max[0] + 1), ValueType(max[1] + 1), ValueType(max[2] + 1)))
    {
    }
    __hostdev__ static BBox createCube(const Coord& min, typename Coord::ValueType dim)
    {
        return BBox(min, min.offsetBy(dim));
    }

    __hostdev__ BBox(const BaseBBox<Coord>& bbox)
        : BBox(bbox[0], bbox[1])
    {
    }
    __hostdev__ bool  empty() const { return mCoord[0][0] >= mCoord[1][0] ||
                                             mCoord[0][1] >= mCoord[1][1] ||
                                             mCoord[0][2] >= mCoord[1][2]; }
    __hostdev__ operator bool() const { return mCoord[0][0] < mCoord[1][0] &&
                                               mCoord[0][1] < mCoord[1][1] &&
                                               mCoord[0][2] < mCoord[1][2]; }
    __hostdev__ Vec3T dim() const { return *this ? this->max() - this->min() : Vec3T(0); }
    __hostdev__ bool  isInside(const Vec3T& p) const
    {
        return p[0] > mCoord[0][0] && p[1] > mCoord[0][1] && p[2] > mCoord[0][2] &&
               p[0] < mCoord[1][0] && p[1] < mCoord[1][1] && p[2] < mCoord[1][2];
    }

}; // BBox<Vec3T, true>

/// @brief Partial template specialization for integer coordinate types
///
/// @note Both min and max are INCLUDED in the bbox so dim = max - min + 1. So,
///       if min = max the bounding box contains exactly one point and dim = 1!
template<typename CoordT>
struct BBox<CoordT, false> : public BaseBBox<CoordT>
{
    static_assert(util::is_same<int, typename CoordT::ValueType>::value, "Expected \"int\" coordinate type");
    using BaseT = BaseBBox<CoordT>;
    using BaseT::mCoord;
    /// @brief Iterator over the domain covered by a BBox
    /// @details z is the fastest-moving coordinate.
    class Iterator
    {
        const BBox& mBBox;
        CoordT      mPos;

    public:
        __hostdev__ Iterator(const BBox& b)
            : mBBox(b)
            , mPos(b.min())
        {
        }
        __hostdev__ Iterator(const BBox& b, const Coord& p)
            : mBBox(b)
            , mPos(p)
        {
        }
        __hostdev__ Iterator& operator++()
        {
            if (mPos[2] < mBBox[1][2]) { // this is the most common case
                ++mPos[2];// increment z
            } else if (mPos[1] < mBBox[1][1]) {
                mPos[2] = mBBox[0][2];// reset z
                ++mPos[1];// increment y
            } else if (mPos[0] <= mBBox[1][0]) {
                mPos[2] = mBBox[0][2];// reset z
                mPos[1] = mBBox[0][1];// reset y
                ++mPos[0];// increment x
            }
            return *this;
        }
        __hostdev__ Iterator operator++(int)
        {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        __hostdev__ bool operator==(const Iterator& rhs) const
        {
            NANOVDB_ASSERT(mBBox == rhs.mBBox);
            return mPos == rhs.mPos;
        }
        __hostdev__ bool operator!=(const Iterator& rhs) const
        {
            NANOVDB_ASSERT(mBBox == rhs.mBBox);
            return mPos != rhs.mPos;
        }
        __hostdev__ bool operator<(const Iterator& rhs) const
        {
            NANOVDB_ASSERT(mBBox == rhs.mBBox);
            return mPos < rhs.mPos;
        }
        __hostdev__ bool operator<=(const Iterator& rhs) const
        {
            NANOVDB_ASSERT(mBBox == rhs.mBBox);
            return mPos <= rhs.mPos;
        }
        /// @brief Return @c true if the iterator still points to a valid coordinate.
        __hostdev__ operator bool() const { return mPos <= mBBox[1]; }
        __hostdev__ const CoordT& operator*() const { return mPos; }
    }; // Iterator
    __hostdev__ Iterator begin() const { return Iterator{*this}; }
    __hostdev__ Iterator end()   const { return Iterator{*this, CoordT(mCoord[1][0]+1, mCoord[0][1], mCoord[0][2])}; }
    __hostdev__          BBox()
        : BaseT(CoordT::max(), CoordT::min())
    {
    }
    __hostdev__ BBox(const CoordT& min, const CoordT& max)
        : BaseT(min, max)
    {
    }

    template<typename SplitT>
    __hostdev__ BBox(BBox& other, const SplitT&)
        : BaseT(other.mCoord[0], other.mCoord[1])
    {
        NANOVDB_ASSERT(this->is_divisible());
        const int n = MaxIndex(this->dim());
        mCoord[1][n] = (mCoord[0][n] + mCoord[1][n]) >> 1;
        other.mCoord[0][n] = mCoord[1][n] + 1;
    }

    __hostdev__ static BBox createCube(const CoordT& min, typename CoordT::ValueType dim)
    {
        return BBox(min, min.offsetBy(dim - 1));
    }

    __hostdev__ static BBox createCube(typename CoordT::ValueType min, typename CoordT::ValueType max)
    {
        return BBox(CoordT(min), CoordT(max));
    }

    __hostdev__ bool is_divisible() const { return mCoord[0][0] < mCoord[1][0] &&
                                                   mCoord[0][1] < mCoord[1][1] &&
                                                   mCoord[0][2] < mCoord[1][2]; }
    /// @brief Return true if this bounding box is empty, e.g. uninitialized
    __hostdev__ bool     empty() const { return mCoord[0][0] > mCoord[1][0] ||
                                                mCoord[0][1] > mCoord[1][1] ||
                                                mCoord[0][2] > mCoord[1][2]; }
    /// @brief Convert this BBox to boolean true if it is not empty
    __hostdev__ operator bool() const { return mCoord[0][0] <= mCoord[1][0] &&
                                               mCoord[0][1] <= mCoord[1][1] &&
                                               mCoord[0][2] <= mCoord[1][2]; }
    __hostdev__ CoordT   dim() const { return *this ? this->max() - this->min() + Coord(1) : Coord(0); }
    __hostdev__ uint64_t volume() const
    {
        auto d = this->dim();
        return uint64_t(d[0]) * uint64_t(d[1]) * uint64_t(d[2]);
    }
    __hostdev__ bool isInside(const CoordT& p) const { return !(CoordT::lessThan(p, this->min()) || CoordT::lessThan(this->max(), p)); }
    /// @brief Return @c true if the given bounding box is inside this bounding box.
    __hostdev__ bool isInside(const BBox& b) const
    {
        return !(CoordT::lessThan(b.min(), this->min()) || CoordT::lessThan(this->max(), b.max()));
    }

    /// @brief Return @c true if the given bounding box overlaps with this bounding box.
    __hostdev__ bool hasOverlap(const BBox& b) const
    {
        return !(CoordT::lessThan(this->max(), b.min()) || CoordT::lessThan(b.max(), this->min()));
    }

    /// @warning This converts a CoordBBox into a floating-point bounding box which implies that max += 1 !
    template<typename RealT = double>
    __hostdev__ BBox<Vec3<RealT>> asReal() const
    {
        static_assert(util::is_floating_point<RealT>::value, "CoordBBox::asReal: Expected a floating point coordinate");
        return BBox<Vec3<RealT>>(Vec3<RealT>(RealT(mCoord[0][0]), RealT(mCoord[0][1]), RealT(mCoord[0][2])),
                                 Vec3<RealT>(RealT(mCoord[1][0] + 1), RealT(mCoord[1][1] + 1), RealT(mCoord[1][2] + 1)));
    }
    /// @brief Return a new instance that is expanded by the specified padding.
    __hostdev__ BBox expandBy(typename CoordT::ValueType padding) const
    {
        return BBox(mCoord[0].offsetBy(-padding), mCoord[1].offsetBy(padding));
    }

    /// @brief  @brief transform this coordinate bounding box by the specified map
    /// @param map mapping of index to world coordinates
    /// @return world bounding box
    template<typename Map>
    __hostdev__ auto transform(const Map& map) const
    {
        using Vec3T = Vec3<double>;
        const Vec3T tmp = map.applyMap(Vec3T(mCoord[0][0], mCoord[0][1], mCoord[0][2]));
        BBox<Vec3T> bbox(tmp, tmp);// return value
        bbox.expand(map.applyMap(Vec3T(mCoord[0][0], mCoord[0][1], mCoord[1][2])));
        bbox.expand(map.applyMap(Vec3T(mCoord[0][0], mCoord[1][1], mCoord[0][2])));
        bbox.expand(map.applyMap(Vec3T(mCoord[1][0], mCoord[0][1], mCoord[0][2])));
        bbox.expand(map.applyMap(Vec3T(mCoord[1][0], mCoord[1][1], mCoord[0][2])));
        bbox.expand(map.applyMap(Vec3T(mCoord[1][0], mCoord[0][1], mCoord[1][2])));
        bbox.expand(map.applyMap(Vec3T(mCoord[0][0], mCoord[1][1], mCoord[1][2])));
        bbox.expand(map.applyMap(Vec3T(mCoord[1][0], mCoord[1][1], mCoord[1][2])));
        return bbox;
    }

#if defined(__CUDACC__) // the following functions only run on the GPU!
    __device__ inline BBox& expandAtomic(const CoordT& ijk)
    {
        mCoord[0].minComponentAtomic(ijk);
        mCoord[1].maxComponentAtomic(ijk);
        return *this;
    }
    __device__ inline BBox& expandAtomic(const BBox& bbox)
    {
        mCoord[0].minComponentAtomic(bbox[0]);
        mCoord[1].maxComponentAtomic(bbox[1]);
        return *this;
    }
    __device__ inline BBox& intersectAtomic(const BBox& bbox)
    {
        mCoord[0].maxComponentAtomic(bbox[0]);
        mCoord[1].minComponentAtomic(bbox[1]);
        return *this;
    }
#endif
}; // BBox<CoordT, false>

// --------------------------> Rgba8 <------------------------------------

/// @brief 8-bit red, green, blue, alpha packed into 32 bit unsigned int
class Rgba8
{
    union
    {
        uint8_t  c[4];   // 4 integer color channels of red, green, blue and alpha components.
        uint32_t packed; // 32 bit packed representation
    } mData;

public:
    static const int SIZE = 4;
    using ValueType = uint8_t;

    /// @brief Default copy constructor
    Rgba8(const Rgba8&) = default;

    /// @brief Default move constructor
    Rgba8(Rgba8&&) = default;

    /// @brief Default move assignment operator
    /// @return non-const reference to this instance
    Rgba8&      operator=(Rgba8&&) = default;

    /// @brief Default copy assignment operator
    /// @return non-const reference to this instance
    Rgba8&      operator=(const Rgba8&) = default;

    /// @brief Default ctor initializes all channels to zero
    __hostdev__ Rgba8()
        : mData{{0, 0, 0, 0}}
    {
        static_assert(sizeof(uint32_t) == sizeof(Rgba8), "Unexpected sizeof");
    }

    /// @brief integer r,g,b,a ctor where alpha channel defaults to opaque
    /// @note all values should be in the range 0u to 255u
    __hostdev__ Rgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255u)
        : mData{{r, g, b, a}}
    {
    }

    /// @brief  @brief ctor where all channels are initialized to the same value
    /// @note value should be in the range 0u to 255u
    explicit __hostdev__ Rgba8(uint8_t v)
        : mData{{v, v, v, v}}
    {
    }

    /// @brief floating-point r,g,b,a ctor where alpha channel defaults to opaque
    /// @note all values should be in the range 0.0f to 1.0f
    __hostdev__ Rgba8(float r, float g, float b, float a = 1.0f)
        : mData{{static_cast<uint8_t>(0.5f + r * 255.0f), // round floats to nearest integers
                 static_cast<uint8_t>(0.5f + g * 255.0f), // double {{}} is needed due to union
                 static_cast<uint8_t>(0.5f + b * 255.0f),
                 static_cast<uint8_t>(0.5f + a * 255.0f)}}
    {
    }

    /// @brief Vec3f r,g,b ctor (alpha channel it set to 1)
    /// @note all values should be in the range 0.0f to 1.0f
    __hostdev__ Rgba8(const Vec3<float>& rgb)
        : Rgba8(rgb[0], rgb[1], rgb[2])
    {
    }

    /// @brief Vec4f r,g,b,a ctor
    /// @note all values should be in the range 0.0f to 1.0f
    __hostdev__ Rgba8(const Vec4<float>& rgba)
        : Rgba8(rgba[0], rgba[1], rgba[2], rgba[3])
    {
    }

    __hostdev__ bool  operator< (const Rgba8& rhs) const { return mData.packed < rhs.mData.packed; }
    __hostdev__ bool  operator==(const Rgba8& rhs) const { return mData.packed == rhs.mData.packed; }
    __hostdev__ float lengthSqr() const
    {
        return 0.0000153787005f * (float(mData.c[0]) * mData.c[0] +
                                   float(mData.c[1]) * mData.c[1] +
                                   float(mData.c[2]) * mData.c[2]); //1/255^2
    }
    __hostdev__ float           length() const { return sqrtf(this->lengthSqr()); }
    /// @brief return n'th color channel as a float in the range 0 to 1
    __hostdev__ float           asFloat(int n) const { return 0.003921569f*float(mData.c[n]); }// divide by 255
    __hostdev__ const uint8_t&  operator[](int n) const { return mData.c[n]; }
    __hostdev__ uint8_t&        operator[](int n) { return mData.c[n]; }
    __hostdev__ const uint32_t& packed() const { return mData.packed; }
    __hostdev__ uint32_t&       packed() { return mData.packed; }
    __hostdev__ const uint8_t&  r() const { return mData.c[0]; }
    __hostdev__ const uint8_t&  g() const { return mData.c[1]; }
    __hostdev__ const uint8_t&  b() const { return mData.c[2]; }
    __hostdev__ const uint8_t&  a() const { return mData.c[3]; }
    __hostdev__ uint8_t&        r() { return mData.c[0]; }
    __hostdev__ uint8_t&        g() { return mData.c[1]; }
    __hostdev__ uint8_t&        b() { return mData.c[2]; }
    __hostdev__ uint8_t&        a() { return mData.c[3]; }
    __hostdev__                 operator Vec3<float>() const {
        return Vec3<float>(this->asFloat(0), this->asFloat(1), this->asFloat(2));
    }
    __hostdev__                 operator Vec4<float>() const {
        return Vec4<float>(this->asFloat(0), this->asFloat(1), this->asFloat(2), this->asFloat(3));
    }
}; // Rgba8

using Vec3d  = Vec3<double>;
using Vec3f  = Vec3<float>;
using Vec3i  = Vec3<int32_t>;
using Vec3u  = Vec3<uint32_t>;
using Vec3u8 = Vec3<uint8_t>;
using Vec3u16 = Vec3<uint16_t>;

using Vec4R  = Vec4<double>;
using Vec4d  = Vec4<double>;
using Vec4f  = Vec4<float>;
using Vec4i  = Vec4<int>;

}// namespace math ===============================================================

using Rgba8 [[deprecated("Use math::Rgba8 instead.")]] = math::Rgba8;
using math::Coord;

using Vec3d = math::Vec3<double>;
using Vec3f = math::Vec3<float>;
using Vec3i = math::Vec3<int32_t>;
using Vec3u = math::Vec3<uint32_t>;
using Vec3u8 = math::Vec3<uint8_t>;
using Vec3u16 = math::Vec3<uint16_t>;

using Vec4R = math::Vec4<double>;
using Vec4d = math::Vec4<double>;
using Vec4f = math::Vec4<float>;
using Vec4i = math::Vec4<int>;

using CoordBBox = math::BBox<Coord>;
using Vec3dBBox = math::BBox<Vec3d>;
using BBoxR [[deprecated("Use Vec3dBBox instead.")]] = math::BBox<Vec3d>;

} // namespace nanovdb ===================================================================

#endif // end of NANOVDB_MATH_MATH_H_HAS_BEEN_INCLUDED
