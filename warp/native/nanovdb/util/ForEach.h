// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/util/ForEach.h

    \author Ken Museth

    \date August 24, 2020

    \brief A unified wrapper for tbb::parallel_for and a naive std::thread fallback
*/

#ifndef NANOVDB_UTIL_FOREACH_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_FOREACH_H_HAS_BEEN_INCLUDED

#include <nanovdb/util/Range.h>// for Range1D

#ifdef NANOVDB_USE_TBB
#include <tbb/parallel_for.h>
#else
#include <thread>
#include <mutex>
#include <vector>
#endif

namespace nanovdb {

namespace util {

/// @brief simple wrapper for tbb::parallel_for with a naive std fallback
///
/// @param range Range, CoordBBox, tbb::blocked_range, blocked_range2D, or blocked_range3D.
/// @param func functor with the signature [](const RangeT&){...},
///
/// @code
///     std::vector<int> array(100);
///     auto func = [&array](auto &r){for (auto i=r.begin(); i!=r.end(); ++i) array[i]=i;};
///     forEach(array, func);
/// @endcode
template <typename RangeT, typename FuncT>
inline void forEach(RangeT range, const FuncT &func)
{
    if (range.empty()) return;
#ifdef NANOVDB_USE_TBB
    tbb::parallel_for(range, func);
#else// naive and likely slow alternative based on std::thread
    if (const size_t threadCount = std::thread::hardware_concurrency()>>1) {
        std::vector<RangeT> rangePool{ range };
        while(rangePool.size() < threadCount) {
            const size_t oldSize = rangePool.size();
            for (size_t i = 0; i < oldSize && rangePool.size() < threadCount; ++i) {
                auto &r = rangePool[i];
                if (r.is_divisible()) rangePool.push_back(RangeT(r, Split()));
            }
            if (rangePool.size() == oldSize) break;// none of the ranges were divided so stop
        }
        std::vector<std::thread> threadPool;
        for (auto &r : rangePool) threadPool.emplace_back(func, r);// launch threads
        for (auto &t : threadPool) t.join();// synchronize threads
    } else {//serial
        func(range);
    }
#endif
}

/// @brief Simple wrapper for the function defined above
template <typename FuncT>
inline void forEach(size_t begin, size_t end, size_t grainSize, const FuncT& func)
{
    forEach(Range1D(begin, end, grainSize), func);
}

/// @brief Simple wrapper for the function defined above, which works with std::containers
template <template<typename...> class ContainerT, typename... T, typename FuncT>
inline void forEach(const ContainerT<T...> &c, const FuncT& func)
{
    forEach(Range1D(0, c.size(), 1), func);
}

/// @brief Simple wrapper for the function defined above, which works with std::containers
template <template<typename...> class ContainerT, typename... T, typename FuncT>
inline void forEach(const ContainerT<T...> &c, size_t grainSize, const FuncT& func)
{
    forEach(Range1D(0, c.size(), grainSize), func);
}

}// namespace util

/// @brief Simple wrapper for the function defined above
template <typename FuncT>
[[deprecated("Use nanovdb::util::forEach instead")]]
inline void forEach(size_t begin, size_t end, size_t grainSize, const FuncT& func)
{
    util::forEach(util::Range1D(begin, end, grainSize), func);
}

/// @brief Simple wrapper for the function defined above, which works with std::containers
template <template<typename...> class ContainerT, typename... T, typename FuncT>
[[deprecated("Use nanovdb::util::forEach instead")]]
inline void forEach(const ContainerT<T...> &c, const FuncT& func)
{
    util::forEach(util::Range1D(0, c.size(), 1), func);
}

/// @brief Simple wrapper for the function defined above, which works with std::containers
template <template<typename...> class ContainerT, typename... T, typename FuncT>
[[deprecated("Use nanovdb::util::forEach instead")]]
inline void forEach(const ContainerT<T...> &c, size_t grainSize, const FuncT& func)
{
    util::forEach(util::Range1D(0, c.size(), grainSize), func);
}

}// namespace nanovdb

#endif // NANOVDB_UTIL_FOREACH_H_HAS_BEEN_INCLUDED
