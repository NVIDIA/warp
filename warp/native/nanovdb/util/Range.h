// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/util/Range.h

    \author Ken Museth

    \date August 28, 2020

    \brief Custom Range class that is compatible with the tbb::blocked_range classes
*/

#ifndef NANOVDB_UTIL_RANGE_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_RANGE_H_HAS_BEEN_INCLUDED

#include <cassert>
#include <cstddef>// for size_t

#ifdef NANOVDB_USE_TBB
#include <tbb/blocked_range.h>// for tbb::split
#endif

namespace nanovdb {

namespace util {

class Split {};// Dummy class used by split constructors

template <int, typename>
class Range;

using Range1D = Range<1, size_t>;
using Range2D = Range<2, size_t>;
using Range3D = Range<3, size_t>;

// template specialization for Rank = 1
template <typename T>
class Range<1, T>
{
    T mBegin, mEnd;
    size_t mGrainsize;
    template<int, typename>
    friend class Range;
public:
    using const_iterator = T;
    using size_type = size_t;
    Range(const Range&) = default;
    Range(T begin, T end, size_type grainsize = size_type(1))
        : mBegin(begin), mEnd(end), mGrainsize(grainsize)
    {
        assert(grainsize > size_type(0));
    }
    /// @brief Split constructor: r[a,b[ -> r[a,b/2[ & this[b/2,b[
    Range(Range &r, Split) : mBegin(r.mBegin), mEnd(r.mEnd), mGrainsize(r.mGrainsize) {
        assert(r.is_divisible());
        r.mEnd = mBegin = this->middle();
    }
#ifdef NANOVDB_USE_TBB
    Range(Range &r, tbb::split) : Range(r, Split()) {}
#endif
    bool operator==(const Range& rhs) const { return mBegin == rhs.mBegin && mEnd == rhs.mEnd && mGrainsize == rhs.mGrainsize; }
    T middle() const {return mBegin + (mEnd - mBegin) / T(2);}
    size_type size()  const { assert(!this->empty()); return size_type(mEnd - mBegin); }
    bool empty()   const { return !(mBegin < mEnd); }
    size_type grainsize() const {return mGrainsize;}
    bool is_divisible() const {return mGrainsize < this->size();}
    const_iterator begin() const { return mBegin; }
    const_iterator end()   const { return mEnd; }
};// Range<1, T>

// template specialization for Rank = 2
template <typename T>
class Range<2, T>
{
    Range<1, T> mRange[2];
public:
    using size_type = typename Range<1, T>::size_type;
    Range(const Range<1, T> &rangeRow, const Range<1, T> &rangeCol) : mRange{ rangeRow, rangeCol } {}
    Range(T beginRow, T endRow, size_type grainsizeRow, T beginCol, T endCol, size_type grainsizeCol)
        : Range( Range<1,T>(beginRow, endRow, grainsizeRow), Range<1,T>(beginCol, endCol, grainsizeCol) )
    {
    }
    Range(T beginRow, T endRow, T beginCol, T endCol) : Range(Range<1,T>(beginRow, endRow), Range<1,T>(beginCol, endCol) )
    {
    }
    Range(Range &r, Split) : Range(r.mRange[0], r.mRange[1]) {
        assert( r.is_divisible() );// at least one of the two dimensions must be divisible!
        if( mRange[0].size()*double(mRange[1].grainsize()) < mRange[1].size()*double(mRange[0].grainsize()) ) {
            r.mRange[1].mEnd = mRange[1].mBegin = mRange[1].middle();
        } else {
            r.mRange[0].mEnd = mRange[0].mBegin = mRange[0].middle();
        }
    }
#ifdef NANOVDB_USE_TBB
    Range(Range &r, tbb::split) : Range(r, Split()) {}
#endif
    bool operator==(const Range& rhs) const {return mRange[0] == rhs[0] && mRange[1] == rhs[1]; }
    bool empty() const { return mRange[0].empty() || mRange[1].empty(); }
    bool is_divisible() const {return mRange[0].is_divisible() || mRange[1].is_divisible();}
    const Range<1, T>& operator[](int i) const { assert(i==0 || i==1); return mRange[i]; }
};// Range<2, T>

// template specialization for Rank = 3
template <typename T>
class Range<3, T>
{
    Range<1, T> mRange[3];
public:
    using size_type = typename Range<1, T>::size_type;
    Range(const Range<1, T> &rangeX, const Range<1, T> &rangeY, const Range<1, T> &rangeZ) : mRange{ rangeX, rangeY, rangeZ } {}
    Range(T beginX, T endX, size_type grainsizeX,
          T beginY, T endY, size_type grainsizeY,
          T beginZ, T endZ, size_type grainsizeZ)
        : Range( Range<1,T>(beginX, endX, grainsizeX),
                 Range<1,T>(beginY, endY, grainsizeY),
                 Range<1,T>(beginZ, endZ, grainsizeZ) )
    {
    }
    Range(T beginX, T endX, T beginY, T endY, T beginZ, T endZ)
        : Range( Range<1,T>(beginX, endX), Range<1,T>(beginY, endY), Range<1,T>(beginZ, endZ) )
    {
    }
    Range(Range &r, Split) : Range(r.mRange[0], r.mRange[1], r.mRange[2])
    {
        assert( r.is_divisible() );// at least one of the three dimensions must be divisible!
        if ( mRange[2].size()*double(mRange[0].grainsize()) < mRange[0].size()*double(mRange[2].grainsize()) ) {
            if ( mRange[0].size()*double(mRange[1].grainsize()) < mRange[1].size()*double(mRange[0].grainsize()) ) {
                r.mRange[1].mEnd = mRange[1].mBegin = mRange[1].middle();
            } else {
                r.mRange[0].mEnd = mRange[0].mBegin = mRange[0].middle();
            }
        } else {
            if ( mRange[2].size()*double(mRange[1].grainsize()) < mRange[1].size()*double(mRange[2].grainsize()) ) {
                r.mRange[1].mEnd = mRange[1].mBegin = mRange[1].middle();
            } else {
                r.mRange[2].mEnd = mRange[2].mBegin = mRange[2].middle();
            }
        }
    }
#ifdef NANOVDB_USE_TBB
    Range(Range &r, tbb::split) : Range(r, Split()) {}
#endif
    bool operator==(const Range& rhs) const {return mRange[0] == rhs[0] && mRange[1] == rhs[1] && mRange[2] == rhs[2]; }
    bool empty() const { return mRange[0].empty() || mRange[1].empty() || mRange[2].empty(); }
    bool is_divisible() const {return mRange[0].is_divisible() || mRange[1].is_divisible() || mRange[2].is_divisible();}
    const Range<1, T>& operator[](int i) const { assert(i==0 || i==1 || i==2); return mRange[i]; }
};// Range<3, T>

}// namespace util

using Range1D [[deprecated("Use nanovdb::util::Range1D instead")]] = util::Range<1, size_t>;
using Range2D [[deprecated("Use nanovdb::util::Range2D instead")]] = util::Range<2, size_t>;
using Range3D [[deprecated("Use nanovdb::util::Range3D instead")]] = util::Range<3, size_t>;

}// namespace nanovdb

#endif // NANOVDB_UTIL_RANGE_H_HAS_BEEN_INCLUDED
