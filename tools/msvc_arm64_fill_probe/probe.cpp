// Throwaway probe: which shapes of the strided fill loop from warp.cpp
// (array_fill_1d) does MSVC for ARM64 compile correctly?
//
// Each variant is a full copy of array_fill_1d's dispatch with a different
// inner store loop, so the inlining context matches warp.cpp. Every variant is
// checked for all widths (1, 2, 4, 8 bytes), indexed and non-indexed, over many
// lengths and strides, and timed on a large 2-byte strided fill.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

// --- inner store loops ------------------------------------------------------

// Current warp.cpp shape: pointer bump.
struct Base {
    template <typename T> static void run(char* data, int n, size_t stride, const int* indices, T value)
    {
        if (indices) {
            for (int i = 0; i < n; i++)
                *reinterpret_cast<T*>(data + size_t(indices[i]) * stride) = value;
        } else {
            for (int i = 0; i < n; i++, data += stride)
                *reinterpret_cast<T*>(data) = value;
        }
    }
};

// Byte offset computed from the index each iteration.
struct IndexOffset {
    template <typename T> static void run(char* data, int n, size_t stride, const int* indices, T value)
    {
        for (int i = 0; i < n; i++) {
            size_t idx = indices ? size_t(indices[i]) : size_t(i);
            *reinterpret_cast<T*>(data + idx * stride) = value;
        }
    }
};

// Two loops; the non-indexed one computes the byte offset from the index.
struct TwoLoopIndexOffset {
    template <typename T> static void run(char* data, int n, size_t stride, const int* indices, T value)
    {
        if (indices) {
            for (int i = 0; i < n; i++)
                *reinterpret_cast<T*>(data + size_t(indices[i]) * stride) = value;
        } else {
            for (int i = 0; i < n; i++)
                *reinterpret_cast<T*>(data + size_t(i) * stride) = value;
        }
    }
};

// Stride in elements of T (valid because the caller checked alignment).
struct ElementStride {
    template <typename T> static void run(char* data, int n, size_t stride, const int* indices, T value)
    {
        T* p = reinterpret_cast<T*>(data);
        size_t s = stride / sizeof(T);
        if (indices) {
            for (int i = 0; i < n; i++)
                p[size_t(indices[i]) * s] = value;
        } else {
            for (int i = 0; i < n; i++)
                p[size_t(i) * s] = value;
        }
    }
};

// Constant-size memcpy per element (compilers turn it into one store).
struct MemcpyConst {
    template <typename T> static void run(char* data, int n, size_t stride, const int* indices, T value)
    {
        if (indices) {
            for (int i = 0; i < n; i++)
                memcpy(data + size_t(indices[i]) * stride, &value, sizeof(T));
        } else {
            for (int i = 0; i < n; i++, data += stride)
                memcpy(data, &value, sizeof(T));
        }
    }
};

// Pointer bump with a size_t counter.
struct SizeTCounter {
    template <typename T> static void run(char* data, int n, size_t stride, const int* indices, T value)
    {
        if (indices) {
            for (size_t i = 0; i < size_t(n); i++)
                *reinterpret_cast<T*>(data + size_t(indices[i]) * stride) = value;
        } else {
            for (size_t i = 0; i < size_t(n); i++, data += stride)
                *reinterpret_cast<T*>(data) = value;
        }
    }
};

// Pointer bump through an end pointer instead of a counter.
struct EndPointer {
    template <typename T> static void run(char* data, int n, size_t stride, const int* indices, T value)
    {
        if (indices) {
            for (int i = 0; i < n; i++)
                *reinterpret_cast<T*>(data + size_t(indices[i]) * stride) = value;
        } else {
            char* end = data + size_t(n) * stride;
            for (; data != end; data += stride)
                *reinterpret_cast<T*>(data) = value;
        }
    }
};

// --- dispatch, copied from warp.cpp array_fill_1d ---------------------------

template <typename Loop>
NOINLINE void array_fill_1d(char* data, int n, size_t stride, const int* indices, const void* value, size_t value_size)
{
    size_t alignment = reinterpret_cast<size_t>(data) | stride;
    if (value_size == 8 && alignment % 8 == 0) {
        int64_t v;
        memcpy(&v, value, sizeof(v));
        Loop::run(data, n, stride, indices, v);
    } else if (value_size == 4 && alignment % 4 == 0) {
        int32_t v;
        memcpy(&v, value, sizeof(v));
        Loop::run(data, n, stride, indices, v);
    } else if (value_size == 2 && alignment % 2 == 0) {
        int16_t v;
        memcpy(&v, value, sizeof(v));
        Loop::run(data, n, stride, indices, v);
    } else if (value_size == 1) {
        int8_t v;
        memcpy(&v, value, sizeof(v));
        Loop::run(data, n, stride, indices, v);
    } else {
        for (int i = 0; i < n; i++) {
            size_t idx = indices ? size_t(indices[i]) : size_t(i);
            memcpy(data + idx * stride, value, value_size);
        }
    }
}

// Pre-GH-2196 per-element memcpy loop: the current workaround's path.
NOINLINE void array_fill_1d_memcpy(char* data, int n, size_t stride, const int* indices, const void* value, size_t value_size)
{
    for (int i = 0; i < n; i++) {
        size_t idx = indices ? size_t(indices[i]) : size_t(i);
        memcpy(data + idx * stride, value, value_size);
    }
}

using FillFn = void (*)(char*, int, size_t, const int*, const void*, size_t);

// --- checks -----------------------------------------------------------------

// Opaque so the compiler can't specialize the calls on constant arguments.
volatile int g_opaque = 0;

static int check(FillFn fn, size_t width, bool indexed)
{
    const unsigned char pattern[8] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
    const size_t strides_in_width[] = {1, 2, 3, 4, 5, 8, 17};
    int failures = 0;
    for (size_t sw : strides_in_width) {
        size_t stride = sw * width + size_t(g_opaque);
        for (int n = 0; n <= 37; n++) {
            int count = indexed ? (n + 1) / 2 : n;  // indexed: every other slot, reversed
            std::vector<int> idx(size_t(count) + 1);
            for (int i = 0; i < count; i++)
                idx[size_t(i)] = 2 * (count - 1 - i);
            size_t slots = indexed ? size_t(2 * count) : size_t(n);
            std::vector<uint64_t> storage((slots * stride + 64) / 8 + 1, 0);  // 8-byte aligned
            char* buf = reinterpret_cast<char*>(storage.data());
            size_t bytes = slots * stride + 16;
            memset(buf, 0, bytes);
            fn(buf, count, stride, indexed ? idx.data() : nullptr, pattern, width);
            std::vector<unsigned char> want(bytes, 0);
            for (int i = 0; i < count; i++) {
                size_t slot = indexed ? size_t(idx[size_t(i)]) : size_t(i);
                memcpy(want.data() + slot * stride, pattern, width);
            }
            if (memcmp(buf, want.data(), bytes) != 0) {
                if (failures < 2) {
                    printf("    FAIL width=%zu stride=%zu n=%d indexed=%d missing slots:", width, stride, count, int(indexed));
                    for (size_t s = 0; s < slots; s++)
                        if (memcmp(buf + s * stride, want.data() + s * stride, width) != 0)
                            printf(" %zu", s);
                    printf("\n");
                }
                failures++;
            }
        }
    }
    return failures;
}

static double time_fill(FillFn fn)
{
    const int n = 1 << 20;
    const size_t stride = 4;  // a[::2] of an int16 array
    std::vector<uint64_t> storage(size_t(n) * stride / 8 + 1);
    char* buf = reinterpret_cast<char*>(storage.data());
    const int16_t value = 17;
    double best = 1e30;
    for (int rep = 0; rep < 20; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        fn(buf, n, stride + size_t(g_opaque), nullptr, &value, sizeof(value));
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best)
            best = ms;
    }
    return best;
}

int main()
{
#if defined(_MSC_FULL_VER)
    printf("compiler: MSVC _MSC_FULL_VER=%d\n", _MSC_FULL_VER);
#elif defined(__clang__)
    printf("compiler: clang %d.%d\n", __clang_major__, __clang_minor__);
#elif defined(__GNUC__)
    printf("compiler: gcc %d.%d\n", __GNUC__, __GNUC_MINOR__);
#endif

    struct Variant {
        const char* name;
        FillFn fn;
    };
    const Variant variants[] = {
        {"Base (current warp.cpp)", &array_fill_1d<Base>},
        {"IndexOffset", &array_fill_1d<IndexOffset>},
        {"TwoLoopIndexOffset", &array_fill_1d<TwoLoopIndexOffset>},
        {"ElementStride", &array_fill_1d<ElementStride>},
        {"MemcpyConst", &array_fill_1d<MemcpyConst>},
        {"SizeTCounter", &array_fill_1d<SizeTCounter>},
        {"EndPointer", &array_fill_1d<EndPointer>},
        {"memcpy loop (workaround)", &array_fill_1d_memcpy},
    };

    int base_failures = 0;
    printf("%-26s %8s %8s %8s %8s %8s %10s\n", "variant", "w1", "w2", "w4", "w8", "w2-idx", "2B ms");
    for (const Variant& v : variants) {
        int f[5];
        const size_t widths[4] = {1, 2, 4, 8};
        // w1/w4/w8 cover both branches; w2 is split so the indexed result stands alone.
        for (int w = 0; w < 4; w++)
            f[w] = check(v.fn, widths[w], false) + (w == 1 ? 0 : check(v.fn, widths[w], true));
        f[4] = check(v.fn, 2, true);
        printf("%-26s", v.name);
        for (int k = 0; k < 5; k++)
            printf(" %8s", f[k] ? "FAIL" : "ok");
        printf(" %10.3f\n", time_fill(v.fn));
        if (v.fn == variants[0].fn)
            base_failures = f[0] + f[1] + f[2] + f[3] + f[4];
    }
    printf("baseline reproduces miscompile: %s\n", base_failures ? "yes" : "no");
    return 0;
}
