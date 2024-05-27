// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file   nanovdb/util/Util.h

    \author Ken Museth

    \date  January 8, 2020

    \brief Utility functions
*/

#ifndef NANOVDB_UTIL_UTIL_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_UTIL_H_HAS_BEEN_INCLUDED

#ifdef __CUDACC_RTC__

typedef signed char        int8_t;
typedef short              int16_t;
typedef int                int32_t;
typedef long long          int64_t;
typedef unsigned char      uint8_t;
typedef unsigned int       uint32_t;
typedef unsigned short     uint16_t;
typedef unsigned long long uint64_t;

#define NANOVDB_ASSERT(x)

#ifndef UINT64_C
#define UINT64_C(x) (x ## ULL)
#endif

#else // !__CUDACC_RTC__

#include <stdlib.h> //    for abs in clang7
#include <stdint.h> //    for types like int32_t etc
#include <stddef.h> //    for size_t type
#include <cassert> //     for assert
#include <cstdio> //      for stderr and snprintf
#include <cmath> //       for sqrt and fma
#include <limits> //      for numeric_limits
#include <utility>//      for std::move
#ifdef NANOVDB_USE_IOSTREAMS
#include <fstream>//      for read/writeUncompressedGrids
#endif// ifdef NANOVDB_USE_IOSTREAMS

// All asserts can be disabled here, even for debug builds
#if 1
#define NANOVDB_ASSERT(x) assert(x)
#else
#define NANOVDB_ASSERT(x)
#endif

#if defined(NANOVDB_USE_INTRINSICS) && defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanForward)
#pragma intrinsic(_BitScanReverse64)
#pragma intrinsic(_BitScanForward64)
#endif

#endif // __CUDACC_RTC__

#if defined(__CUDACC__) || defined(__HIP__)
// Only define __hostdev__ qualifier when using NVIDIA CUDA or HIP compilers
#ifndef __hostdev__
#define __hostdev__ __host__ __device__ // Runs on the CPU and GPU, called from the CPU or the GPU
#endif
#else
// Dummy definitions of macros only defined by CUDA and HIP compilers
#ifndef __hostdev__
#define __hostdev__ // Runs on the CPU and GPU, called from the CPU or the GPU
#endif
#ifndef __global__
#define __global__ // Runs on the GPU, called from the CPU or the GPU
#endif
#ifndef __device__
#define __device__ // Runs on the GPU, called from the GPU
#endif
#ifndef __host__
#define __host__ // Runs on the CPU, called from the CPU
#endif

#endif // if defined(__CUDACC__) || defined(__HIP__)

// The following macro will suppress annoying warnings when nvcc
// compiles functions that call (host) intrinsics (which is perfectly valid)
#if defined(_MSC_VER) && defined(__CUDACC__)
#define NANOVDB_HOSTDEV_DISABLE_WARNING __pragma("hd_warning_disable")
#elif defined(__GNUC__) && defined(__CUDACC__)
#define NANOVDB_HOSTDEV_DISABLE_WARNING _Pragma("hd_warning_disable")
#else
#define NANOVDB_HOSTDEV_DISABLE_WARNING
#endif

// Define compiler warnings that work with all compilers
//#if defined(_MSC_VER)
//#define NANO_WARNING(msg) _pragma("message" #msg)
//#else
//#define NANO_WARNING(msg) _Pragma("message" #msg)
//#endif

//==============================================
/// @brief Defines macros that issues warnings for deprecated header files
/// @details Example:
/// @code
/// #include <nanovdb/util/Util.h> // for NANOVDB_DEPRECATED_HEADER
/// #include <nanovdb/path/Alternative.h>
/// NANOVDB_DEPRECATED_HEADER("This header file is deprecated, please use <nanovdb/path/Alternative.h> instead")
/// @endcode
#ifdef __GNUC__
#define NANOVDB_PRAGMA(X) _Pragma(#X)
#define NANOVDB_DEPRECATED_HEADER(MSG) NANOVDB_PRAGMA(GCC warning MSG)
#elif defined(_MSC_VER)
#define NANOVDB_STRINGIZE_(MSG) #MSG
#define NANOVDB_STRINGIZE(MSG) NANOVDB_STRINGIZE_(MSG)
#define NANOVDB_DEPRECATED_HEADER(MSG) \
    __pragma(message(__FILE__ "(" NANOVDB_STRINGIZE(__LINE__) ") : Warning: " MSG))
#endif

// A portable implementation of offsetof - unfortunately it doesn't work with static_assert
#define NANOVDB_OFFSETOF(CLASS, MEMBER) ((int)(size_t)((char*)&((CLASS*)0)->MEMBER - (char*)0))

namespace nanovdb {// =================================================================

namespace util {// ====================================================================

/// @brief Minimal implementation of std::declval, which converts any type @c T to
////       a reference type, making it possible to use member functions in the operand
///        of the decltype specifier without the need to go through constructors.
/// @tparam T Template type to be converted to T&&
/// @return T&&
/// @warning Unlike std::declval, this version does not work when T = void! However,
///          NVRTC does not like std::declval, so we provide our own implementation.
template<typename T>
T&& declval() noexcept;

// --------------------------> string utility functions <------------------------------------

/// @brief tests if a c-string @c str is empty, that is its first value is '\0'
/// @param str c-string to be tested for null termination
/// @return true if str[0] = '\0'
__hostdev__ inline bool empty(const char* str)
{
    NANOVDB_ASSERT(str != nullptr);
    return *str == '\0';
}// util::empty

/// @brief length of a c-sting, excluding '\0'.
/// @param str c-string
/// @return the number of characters that precede the terminating null character.
__hostdev__ inline size_t strlen(const char *str)
{
    NANOVDB_ASSERT(str != nullptr);
    const char *s = str;
    while(*s) ++s;               ;
    return (s - str);
}// util::strlen

/// @brief Copy characters from @c src to @c dst.
/// @param dst pointer to the destination string.
/// @param src pointer to the null-terminated source string.
/// @return destination string @c dst.
/// @note Emulates the behaviour of std::strcpy, except this version also runs on the GPU.
__hostdev__ inline char* strcpy(char *dst, const char *src)
{
    NANOVDB_ASSERT(dst != nullptr && src != nullptr);
    for (char *p = dst; (*p++ = *src) != '\0'; ++src);
    return dst;
}// util::strcpy(char*, const char*)

/// @brief Copies the first num characters of @c src to @c dst.
///        If the end of the source C string (which is signaled by a
///        null-character) is found before @c max characters have been
///        copied, @c dst is padded with zeros until a total of @c max
///        characters have been written to it.
/// @param dst destination string
/// @param src source string
/// @param max maximum number of character in destination string
/// @return destination string @c dst
/// @warning if strncpy(dst, src, max)[max-1]!='\0' then @c src has more
///          characters than @c max and the return string needs to be
///          manually null-terminated, i.e. strncpy(dst, src, max)[max-1]='\0'
__hostdev__ inline char* strncpy(char *dst, const char *src, size_t max)
{
    NANOVDB_ASSERT(dst != nullptr && src != nullptr);
    size_t i = 0;
    for (; i < max && src[i] != '\0'; ++i) dst[i] = src[i];
    for (; i < max; ++i) dst[i] = '\0';
    return dst;
}// util::strncpy(char *dst, const char *src, size_t max)

/// @brief converts a number to a string using a specific base
/// @param dst destination string
/// @param num signed number to be concatenated after @c dst
/// @param bas base used when converting @c num to a string
/// @return destination string @c dst
/// @note Emulates the behaviour of itoa, except this verion also works on the GPU.
__hostdev__ inline char* strcpy(char* dst, int num, int bas = 10)
{
    NANOVDB_ASSERT(dst != nullptr && bas > 0);
    int len = 0;// length of number once converted to a string
    if (num == 0) dst[len++] = '0';
    for (int abs = num < 0 && bas == 10 ? -num : num; abs; abs /= bas) {
        const int rem = abs % bas;
        dst[len++] = rem > 9 ? rem - 10 + 'a' : rem + '0';
    }
    if (num < 0) dst[len++] = '-';// append '-' if negative
    for (char *a = dst, *b = a + len - 1; a < b; ++a, --b) {// reverse dst
        dst[len] = *a;// use end of string as temp
        *a = *b;
        *b = dst[len];
    }
    dst[len] = '\0';// explicitly terminate end of string
    return dst;
}// util::strcpy(char*, int, int)

/// @brief Appends a copy of the character string pointed to by @c src to
///        the end of the character string pointed to by @c dst on the device.
/// @param dst pointer to the null-terminated byte string to append to.
/// @param src pointer to the null-terminated byte string to copy from.
/// @return pointer to the character array being appended to.
/// @note Emulates the behaviour of std::strcat, except this version also runs on the GPU.
__hostdev__ inline char* strcat(char *dst, const char *src)
{
    NANOVDB_ASSERT(dst != nullptr && src != nullptr);
    char *p = dst;
    while (*p != '\0') ++p;// advance till end of dst
    strcpy(p, src);// append src
    return dst;
}// util::strcat(char*, const char*)

/// @brief concatenates a number after a string using a specific base
/// @param dst null terminated destination string
/// @param num signed number to be concatenated after @c dst
/// @param bas base used when converting @c num to a string
/// @return destination string @c dst
__hostdev__ inline char* strcat(char* dst, int num, int bas = 10)
{
    NANOVDB_ASSERT(dst != nullptr);
    char *p = dst;
    while (*p != '\0') ++p;
    strcpy(p, num, bas);
    return dst;
}// util::strcat(char*, int, int)

/// @brief Compares two null-terminated byte strings lexicographically.
/// @param lhs pointer to the null-terminated byte strings to compare
/// @param rhs pointer to the null-terminated byte strings to compare
/// @return Negative value if @c lhs appears before @c rhs in lexicographical order.
///         Zero if @c lhs and @c rhs compare equal. Positive value if @c lhs appears
///         after @c rhs in lexicographical order.
/// @note Emulates the behaviour of std::strcmp, except this version also runs on the GPU.
__hostdev__ inline int strcmp(const char *lhs, const char *rhs)
{
    while(*lhs != '\0' && (*lhs == *rhs)){
        lhs++;
        rhs++;
    }
    return *(const unsigned char*)lhs - *(const unsigned char*)rhs;// zero if lhs == rhs
}// util::strcmp(const char*, const char*)

/// @brief Test if two null-terminated byte strings are the same
/// @param lhs pointer to the null-terminated byte strings to compare
/// @param rhs pointer to the null-terminated byte strings to compare
/// @return true if the two c-strings are identical
__hostdev__ inline bool streq(const char *lhs, const char *rhs)
{
    return strcmp(lhs, rhs) == 0;
}// util::streq

namespace impl {// =======================================================
// Base-case implementation of Variadic Template function impl::sprint
__hostdev__ inline char* sprint(char *dst){return dst;}
// Variadic Template function impl::sprint
template <typename T, typename... Types>
__hostdev__ inline char* sprint(char *dst, T var1, Types... var2)
{
    return impl::sprint(strcat(dst, var1), var2...);
}
}// namespace impl =========================================================

/// @brief prints a variable number of string and/or numbers to a destination string
template <typename T, typename... Types>
__hostdev__ inline char* sprint(char *dst, T var1, Types... var2)
{
    return impl::sprint(strcpy(dst, var1), var2...);
}// util::sprint

// --------------------------> memzero <------------------------------------

/// @brief Zero initialization of memory
/// @param dst pointer to destination
/// @param byteCount number of bytes to be initialized to zero
/// @return destination pointer @c dst
__hostdev__ inline static void* memzero(void *dst, size_t byteCount)
{
    NANOVDB_ASSERT(dst);
    const size_t wordCount = byteCount >> 3;
    if (wordCount << 3 == byteCount) {
        for (auto *d = (uint64_t*)dst, *e = d + wordCount; d != e; ++d) *d = 0ULL;
    } else {
        for (auto *d = (char*)dst, *e = d + byteCount; d != e; ++d) *d = '\0';
    }
    return dst;
}// util::memzero

// --------------------------> util::is_same <------------------------------------

/// @brief C++11 implementation of std::is_same
/// @note When more than two arguments are provided value = T0==T1 || T0==T2 || ...
template<typename T0, typename T1, typename ...T>
struct is_same
{
    static constexpr bool value = is_same<T0, T1>::value || is_same<T0, T...>::value;
};

template<typename T0, typename T1>
struct is_same<T0, T1> {static constexpr bool value = false;};

template<typename T>
struct is_same<T, T> {static constexpr bool value = true;};

// --------------------------> util::is_floating_point <------------------------------------

/// @brief C++11 implementation of std::is_floating_point
template<typename T>
struct is_floating_point {static constexpr bool value = is_same<T, float, double>::value;};

// --------------------------> util::enable_if <------------------------------------

/// @brief C++11 implementation of std::enable_if
template <bool, typename T = void>
struct enable_if {};

template <typename T>
struct enable_if<true, T> {using type = T;};

// --------------------------> util::disable_if <------------------------------------

template<bool, typename T = void>
struct disable_if {using type = T;};

template<typename T>
struct disable_if<true, T> {};

// --------------------------> util::is_const <------------------------------------

template<typename T>
struct is_const {static constexpr bool value = false;};

template<typename T>
struct is_const<const T> {static constexpr bool value = true;};

// --------------------------> util::is_pointer <------------------------------------

/// @brief Trait used to identify template parameter that are pointers
/// @tparam T Template parameter to be tested
template<class T>
struct is_pointer {static constexpr bool value = false;};

/// @brief Template specialization of pointers
/// @tparam T Template parameter to be tested
/// @note T can be both a non-const and const type
template<class T>
struct is_pointer<T*> {static constexpr bool value = true;};

// --------------------------> util::conditional <------------------------------------

/// @brief C++11 implementation of std::conditional
template<bool, class TrueT, class FalseT>
struct conditional { using type = TrueT; };

/// @brief Template specialization of conditional
/// @tparam FalseT Type used when boolean is false
/// @tparam TrueT Type used when boolean is true
template<class TrueT, class FalseT>
struct conditional<false, TrueT, FalseT> { using type = FalseT; };

// --------------------------> util::remove_const <------------------------------------

/// @brief Trait use to const from type. Default implementation is just a pass-through
/// @tparam T Type
/// @details remove_pointer<float>::type = float
template<typename T>
struct remove_const {using type = T;};

/// @brief Template specialization of trait class use to remove const qualifier type from a type
/// @tparam T Type of the const type
/// @details remove_pointer<const float>::type = float
template<typename T>
struct remove_const<const T> {using type = T;};

// --------------------------> util::remove_reference <------------------------------------

/// @brief Trait use to remove reference, i.e. "&", qualifier from a type. Default implementation is just a pass-through
/// @tparam T Type
/// @details remove_pointer<float>::type = float
template <typename T>
struct remove_reference {using type = T;};

/// @brief Template specialization of trait class use to remove reference, i.e. "&", qualifier from a type
/// @tparam T Type of the reference
/// @details remove_pointer<float&>::type = float
template <typename T>
struct remove_reference<T&> {using type = T;};

// --------------------------> util::remove_pointer <------------------------------------

/// @brief Trait use to remove pointer, i.e. "*", qualifier from a type. Default implementation is just a pass-through
/// @tparam T Type
/// @details remove_pointer<float>::type = float
template <typename T>
struct remove_pointer {using type = T;};

/// @brief Template specialization of trait class use to to remove pointer, i.e. "*", qualifier from a type
/// @tparam T Type of the pointer
/// @details remove_pointer<float*>::type = float
template <typename T>
struct remove_pointer<T*> {using type = T;};

// --------------------------> util::match_const <------------------------------------

/// @brief Trait used to transfer the const-ness of a reference type to another type
/// @tparam T Type whose const-ness needs to match the reference type
/// @tparam ReferenceT Reference type that is not const
/// @details match_const<const int, float>::type = int
///          match_const<int, float>::type = int
template<typename T, typename ReferenceT>
struct match_const {using type = typename remove_const<T>::type;};

/// @brief Template specialization used to transfer the const-ness of a reference type to another type
/// @tparam T Type that will adopt the const-ness of the reference type
/// @tparam ReferenceT Reference type that is const
/// @details match_const<const int, const float>::type = const int
///          match_const<int, const float>::type = const int
template<typename T, typename ReferenceT>
struct match_const<T, const ReferenceT> {using type = const typename remove_const<T>::type;};

// --------------------------> util::is_specialization <------------------------------------

/// @brief Metafunction used to determine if the first template
///        parameter is a specialization of the class template
///        given in the second template parameter.
///
/// @details is_specialization<Vec3<float>, Vec3>::value == true;
///          is_specialization<Vec3f, Vec3>::value == true;
///          is_specialization<std::vector<float>, std::vector>::value == true;
template<typename AnyType, template<typename...> class TemplateType>
struct is_specialization {static const bool value = false;};
template<typename... Args, template<typename...> class TemplateType>
struct is_specialization<TemplateType<Args...>, TemplateType>
{
    static const bool value = true;
};// util::is_specialization

// --------------------------> util::PtrDiff <------------------------------------

/// @brief Compute the distance, in bytes, between two pointers, dist = p - q
/// @param p fist pointer, assumed to NOT be NULL
/// @param q second pointer, assumed to NOT be NULL
/// @return signed distance between pointer, p - q, addresses in units of bytes
__hostdev__ inline static int64_t PtrDiff(const void* p, const void* q)
{
    NANOVDB_ASSERT(p && q);
    return reinterpret_cast<const char*>(p) - reinterpret_cast<const char*>(q);
}// util::PtrDiff

// --------------------------> util::PtrAdd <------------------------------------

/// @brief Adds a byte offset to a non-const pointer to produce another non-const pointer
/// @tparam DstT Type of the return pointer (defaults to void)
/// @param p non-const input pointer, assumed to NOT be NULL
/// @param offset signed byte offset
/// @return a non-const pointer defined as the offset of an input pointer
template<typename DstT = void>
__hostdev__ inline static DstT* PtrAdd(void* p, int64_t offset)
{
    NANOVDB_ASSERT(p);
    return reinterpret_cast<DstT*>(reinterpret_cast<char*>(p) + offset);
}// util::PtrAdd

/// @brief Adds a byte offset to a const pointer to produce another const pointer
/// @tparam DstT Type of the return pointer (defaults to void)
/// @param p const input pointer, assumed to NOT be NULL
/// @param offset signed byte offset
/// @return a const pointer defined as the offset of a const input pointer
template<typename DstT = void>
__hostdev__ inline static const DstT* PtrAdd(const void* p, int64_t offset)
{
    NANOVDB_ASSERT(p);
    return reinterpret_cast<const DstT*>(reinterpret_cast<const char*>(p) + offset);
}// util::PtrAdd

// -------------------> findLowestOn <----------------------------

/// @brief Returns the index of the lowest, i.e. least significant, on bit in the specified 32 bit word
///
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint32_t(0)!
NANOVDB_HOSTDEV_DISABLE_WARNING
__hostdev__ inline uint32_t findLowestOn(uint32_t v)
{
    NANOVDB_ASSERT(v);
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(NANOVDB_USE_INTRINSICS)
    return __ffs(v) - 1; // one based indexing
#elif defined(_MSC_VER) && defined(NANOVDB_USE_INTRINSICS)
    unsigned long index;
    _BitScanForward(&index, v);
    return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(NANOVDB_USE_INTRINSICS)
    return static_cast<uint32_t>(__builtin_ctzl(v));
#else
    //NANO_WARNING("Using software implementation for findLowestOn(uint32_t v)")
    static const unsigned char DeBruijn[32] = {
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9};
// disable unary minus on unsigned warning
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
    return DeBruijn[uint32_t((v & -v) * 0x077CB531U) >> 27];
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(pop)
#endif

#endif
}// util::findLowestOn(uint32_t)

/// @brief Returns the index of the lowest, i.e. least significant, on bit in the specified 64 bit word
///
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint32_t(0)!
NANOVDB_HOSTDEV_DISABLE_WARNING
__hostdev__ inline uint32_t findLowestOn(uint64_t v)
{
    NANOVDB_ASSERT(v);
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(NANOVDB_USE_INTRINSICS)
    return __ffsll(static_cast<unsigned long long int>(v)) - 1; // one based indexing
#elif defined(_MSC_VER) && defined(NANOVDB_USE_INTRINSICS)
    unsigned long index;
    _BitScanForward64(&index, v);
    return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(NANOVDB_USE_INTRINSICS)
    return static_cast<uint32_t>(__builtin_ctzll(v));
#else
    //NANO_WARNING("Using software implementation for util::findLowestOn(uint64_t)")
    static const unsigned char DeBruijn[64] = {
        0,   1,  2, 53,  3,  7, 54, 27, 4,  38, 41,  8, 34, 55, 48, 28,
        62,  5, 39, 46, 44, 42, 22,  9, 24, 35, 59, 56, 49, 18, 29, 11,
        63, 52,  6, 26, 37, 40, 33, 47, 61, 45, 43, 21, 23, 58, 17, 10,
        51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12,
    };
// disable unary minus on unsigned warning
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
    return DeBruijn[uint64_t((v & -v) * UINT64_C(0x022FDD63CC95386D)) >> 58];
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(pop)
#endif

#endif
}// util::findLowestOn(uint64_t)

// -------------------> findHighestOn <----------------------------

/// @brief Returns the index of the highest, i.e. most significant, on bit in the specified 32 bit word
///
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint32_t(0)!
NANOVDB_HOSTDEV_DISABLE_WARNING
__hostdev__ inline uint32_t findHighestOn(uint32_t v)
{
    NANOVDB_ASSERT(v);
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(NANOVDB_USE_INTRINSICS)
    return sizeof(uint32_t) * 8 - 1 - __clz(v); // Return the number of consecutive high-order zero bits in a 32-bit integer.
#elif defined(_MSC_VER) && defined(NANOVDB_USE_INTRINSICS)
    unsigned long index;
    _BitScanReverse(&index, v);
    return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(NANOVDB_USE_INTRINSICS)
    return sizeof(unsigned long) * 8 - 1 - __builtin_clzl(v);
#else
    //NANO_WARNING("Using software implementation for util::findHighestOn(uint32_t)")
    static const unsigned char DeBruijn[32] = {
        0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
        8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31};
    v |= v >> 1; // first round down to one less than a power of 2
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return DeBruijn[uint32_t(v * 0x07C4ACDDU) >> 27];
#endif
}// util::findHighestOn

/// @brief Returns the index of the highest, i.e. most significant, on bit in the specified 64 bit word
///
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint32_t(0)!
NANOVDB_HOSTDEV_DISABLE_WARNING
__hostdev__ inline uint32_t findHighestOn(uint64_t v)
{
    NANOVDB_ASSERT(v);
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(NANOVDB_USE_INTRINSICS)
    return sizeof(unsigned long) * 8 - 1 - __clzll(static_cast<unsigned long long int>(v));
#elif defined(_MSC_VER) && defined(NANOVDB_USE_INTRINSICS)
    unsigned long index;
    _BitScanReverse64(&index, v);
    return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(NANOVDB_USE_INTRINSICS)
    return sizeof(unsigned long) * 8 - 1 - __builtin_clzll(v);
#else
    const uint32_t* p = reinterpret_cast<const uint32_t*>(&v);
    return p[1] ? 32u + findHighestOn(p[1]) : findHighestOn(p[0]);
#endif
}// util::findHighestOn

// ----------------------------> util::countOn <--------------------------------------

/// @return Number of bits that are on in the specified 64-bit word
NANOVDB_HOSTDEV_DISABLE_WARNING
__hostdev__ inline uint32_t countOn(uint64_t v)
{
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(NANOVDB_USE_INTRINSICS)
    //#warning Using popcll for util::countOn
    return __popcll(v);
// __popcnt64 intrinsic support was added in VS 2019 16.8
#elif defined(_MSC_VER) && defined(_M_X64) && (_MSC_VER >= 1928) && defined(NANOVDB_USE_INTRINSICS)
    //#warning Using popcnt64 for util::countOn
    return uint32_t(__popcnt64(v));
#elif (defined(__GNUC__) || defined(__clang__)) && defined(NANOVDB_USE_INTRINSICS)
    //#warning Using builtin_popcountll for util::countOn
    return __builtin_popcountll(v);
#else // use software implementation
    //NANO_WARNING("Using software implementation for util::countOn")
    v = v - ((v >> 1) & uint64_t(0x5555555555555555));
    v = (v & uint64_t(0x3333333333333333)) + ((v >> 2) & uint64_t(0x3333333333333333));
    return (((v + (v >> 4)) & uint64_t(0xF0F0F0F0F0F0F0F)) * uint64_t(0x101010101010101)) >> 56;
#endif
}// util::countOn(uint64_t)

}// namespace util ==================================================================

[[deprecated("Use nanovdb::util::findLowestOn instead")]]
__hostdev__ inline uint32_t FindLowestOn(uint32_t v){return util::findLowestOn(v);}
[[deprecated("Use nanovdb::util::findLowestOn instead")]]
__hostdev__ inline uint32_t FindLowestOn(uint64_t v){return util::findLowestOn(v);}
[[deprecated("Use nanovdb::util::findHighestOn instead")]]
__hostdev__ inline uint32_t FindHighestOn(uint32_t v){return util::findHighestOn(v);}
[[deprecated("Use nanovdb::util::findHighestOn instead")]]
__hostdev__ inline uint32_t FindHighestOn(uint64_t v){return util::findHighestOn(v);}
[[deprecated("Use nanovdb::util::countOn instead")]]
__hostdev__ inline uint32_t CountOn(uint64_t v){return util::countOn(v);}

} // namespace nanovdb ===================================================================

#endif // end of NANOVDB_UTIL_UTIL_H_HAS_BEEN_INCLUDED
