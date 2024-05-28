// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    @file nanovdb/HostBuffer.h

    @date April 20, 2021

    @brief HostBuffer - a buffer that contains a shared or private bump
           pool to either externally or internally managed host memory.

    @details This HostBuffer can be used in multiple ways, most of which are
             demonstrated in the examples below. Memory in the pool can
             be managed or unmanged (e.g. internal or external) and can
             be shared between multiple buffers or belong to a single buffer.

   Example that uses HostBuffer::create inside io::readGrids to create a
   full self-managed buffer, i.e. not shared and without padding, per grid in the file.
   @code
        auto handles = nanovdb::io::readGrids("file.nvdb");
   @endcode

   Example that uses HostBuffer::createFull. Assuming you have a raw pointer
   to a NanoVDB grid of unknown type, this examples shows how to create its
   GridHandle which can be used to enquire about the grid type and meta data.
   @code
        void    *data;// pointer to a NanoVDB grid of unknown type
        uint64_t size;// byte size of NanoVDB grid of unknown type
        auto buffer = nanovdb::HostBuffer::createFull(size, data);
        nanovdb::GridHandle<> gridHandle(std::move(buffer));
   @endcode

   Example that uses HostBuffer::createPool for internally managed host memory.
   Suppose you want to read multiple grids in multiple files, but reuse the same
   fixed sized memory buffer to both avoid memory fragmentation as well as
   exceeding the fixed memory ceiling!
   @code
        auto pool = nanovdb::HostBuffer::createPool(1 << 30);// 1 GB memory pool
        std::vector<std::string>> frames;// vector of grid names
        for (int i=0; i<frames.size(); ++i) {
            auto handles = nanovdb::io::readGrids(frames[i], 0, pool);// throws if grids in file exceed 1 GB
            ...
            pool.reset();// clears all handles and resets the memory pool for reuse
        }
   @endcode

   Example that uses HostBuffer::createPool for externally managed host memory.
   Note that in this example @c handles are allowed to outlive @c pool since
   they internally store a shared pointer to the memory pool. However @c data
   MUST outlive @c handles since the pool does not own its memory in this example.
   @code
        const size_t poolSize = 1 << 30;// 1 GB
        void *data = std::malloc(size + NANOVDB_DATA_ALIGNMENT);// 1 GB pool with padding
        void *buffer = nanovdb::alignPtr(data);// 32B aligned buffer
        //void *buffer = std::aligned_alloc(NANOVDB_DATA_ALIGNMENT, poolSize);// in C++17
        auto pool = nanovdb::HostBuffer::createPool(poolSize, buffer);
        auto handles1 = nanovdb::io::readGrids("file1.nvdb", 0, pool);
        auto handles2 = nanovdb::io::readGrids("file2.nvdb", 0, pool);
        ....
        std::free(data);
        //std::free(buffer);
   @endcode

   Example that uses HostBuffer::createPool for externally managed host memory.
   Note that in this example @c handles are allowed to outlive @c pool since
   they internally store a shared pointer to the memory pool. However @c array
   MUST outlive @c handles since the pool does not own its memory in this example.
   @code
        const size_t poolSize = 1 << 30;// 1 GB
        std::unique_ptr<char[]> array(new char[size + NANOVDB_DATA_ALIGNMENT]);// scoped pool of 1 GB with padding
        void *buffer = nanovdb::alignPtr(array.get());// 32B aligned buffer
        auto pool = nanovdb::HostBuffer::createPool(poolSize, buffer);
        auto handles = nanovdb::io::readGrids("file.nvdb", 0, pool);
   @endcode
*/

#ifndef NANOVDB_HOSTBUFFER_H_HAS_BEEN_INCLUDED
#define NANOVDB_HOSTBUFFER_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>// for NANOVDB_DATA_ALIGNMENT;
#include <stdint.h> //         for types like int32_t etc
#include <cstdio> //           for fprintf
#include <cstdlib> //          for std::malloc/std::realloc/std::free
#include <memory>//            for std::make_shared
#include <mutex>//             for std::mutex
#include <unordered_set>//     for std::unordered_set
#include <cassert>//           for assert
#include <sstream>//           for std::stringstream
#include <cstring>//           for memcpy

#define checkPtr(ptr, msg) \
    { \
        ptrAssert((ptr), (msg), __FILE__, __LINE__); \
    }

namespace nanovdb {

template<typename BufferT>
struct BufferTraits
{
    static constexpr bool hasDeviceDual = false;
};

// ----------------------------> HostBuffer <--------------------------------------

/// @brief This is a buffer that contains a shared or private pool
///        to either externally or internally managed host memory.
///
/// @note  Terminology:
///        Pool:   0 = buffer.size() < buffer.poolSize()
///        Buffer: 0 < buffer.size() < buffer.poolSize()
///        Full:   0 < buffer.size() = buffer.poolSize()
///        Empty:  0 = buffer.size() = buffer.poolSize()
class HostBuffer
{
    struct Pool;// forward declaration of private pool struct
    std::shared_ptr<Pool> mPool;
    uint64_t              mSize; // total number of bytes for the NanoVDB grid.
    void*                 mData; // raw buffer for the NanoVDB grid.

#if defined(DEBUG) || defined(_DEBUG)
    static inline void ptrAssert(void* ptr, const char* msg, const char* file, int line, bool abort = true)
    {
        if (ptr == nullptr) {
            fprintf(stderr, "NULL pointer error: %s %s %d\n", msg, file, line);
            if (abort)
                exit(1);
        }
        if (uint64_t(ptr) % NANOVDB_DATA_ALIGNMENT) {
            fprintf(stderr, "Alignment pointer error: %s %s %d\n", msg, file, line);
            if (abort)
                exit(1);
        }
    }
#else
    static inline void ptrAssert(void*, const char*, const char*, int, bool = true)
    {
    }
#endif

public:
    /// @brief Return a full buffer or an empty buffer
    HostBuffer(uint64_t bufferSize = 0);

     /// @brief Move copy-constructor
    HostBuffer(HostBuffer&& other);

    /// @brief Custom descructor
    ~HostBuffer() { this->clear(); }

    /// @brief Move copy assignment operation
    HostBuffer& operator=(HostBuffer&& other);

    /// @brief Disallow copy-construction
    HostBuffer(const HostBuffer&) = delete;

    /// @brief Disallow copy assignment operation
    HostBuffer& operator=(const HostBuffer&) = delete;

    /// @brief Return a pool buffer which satisfies: buffer.size == 0,
    ///        buffer.poolSize() == poolSize, and buffer.data() == nullptr.
    ///        If data==nullptr, memory for the pool will be allocated.
    ///
    /// @throw If poolSize is zero.
    static HostBuffer createPool(uint64_t poolSize, void *data = nullptr);

    /// @brief Return a full buffer which satisfies: buffer.size == bufferSize,
    ///        buffer.poolSize() == bufferSize, and buffer.data() == data.
    ///        If data==nullptr, memory for the pool will be allocated.
    ///
    /// @throw If bufferSize is zero.
    static HostBuffer createFull(uint64_t bufferSize, void *data = nullptr);

    /// @brief Return a buffer with @c bufferSize bytes managed by
    ///        the specified memory @c pool. If none is provided, i.e.
    ///        @c pool == nullptr or @c pool->poolSize() == 0, one is
    ///        created with size @c bufferSize, i.e. a full buffer is returned.
    ///
    /// @throw If the specified @c pool has insufficient memory for
    ///        the requested buffer size.
    static HostBuffer create(uint64_t bufferSize, const HostBuffer* pool = nullptr);

    /// @brief Initialize as a full buffer with the specified size. If data is NULL
    ///        the memory is internally allocated.
    void init(uint64_t bufferSize, void *data = nullptr);

    //@{
    /// @brief Retuns a pointer to the raw memory buffer managed by this allocator.
    ///
    /// @warning Note that the pointer can be NULL if the allocator was not initialized!
    const void* data() const { return mData; }
    void* data() { return mData; }
    //@}

    //@{
    /// @brief Returns the size in bytes associated with this buffer.
    uint64_t bufferSize() const { return mSize; }
    uint64_t size() const { return this->bufferSize(); }
    //@}

    /// @brief Returns the size in bytes of the memory pool shared with this instance.
    uint64_t poolSize() const;

    /// @brief Return true if memory is managed (using std::malloc and std:free) by the
    ///        shared pool in this buffer. Else memory is assumed to be managed externally.
    bool isManaged() const;

    //@{
    /// @brief Returns true if this buffer has no memory associated with it
    bool isEmpty() const { return !mPool || mSize == 0 || mData == nullptr; }
    bool empty() const { return this->isEmpty(); }
    //@}

    /// @brief Return true if this is a pool, i.e. an empty buffer with a nonempty
    ///        internal pool, i.e. this->size() == 0 and this->poolSize() != 0
    bool isPool() const { return mSize == 0 && this->poolSize() > 0; }

    /// @brief Return true if the pool exists, is nonempty but has no more available memory
    bool isFull() const;

    /// @brief Clear this buffer so it is empty.
    void clear();

    /// @brief Clears all existing buffers that are registered against the memory pool
    ///        and resets the pool so it can be reused to create new buffers.
    ///
    /// @throw If this instance is not empty or contains no pool.
    ///
    /// @warning This method is not thread-safe!
    void reset();

    /// @brief Total number of bytes from the pool currently in use by buffers
    uint64_t poolUsage() const;

    /// @brief resize the pool size. It will attempt to resize the existing
    ///        memory block, but if that fails a deep copy is performed.
    ///        If @c data is not NULL it will be used as new externally
    ///        managed memory for the pool. All registered buffers are
    ///        updated so GridHandle::grid might return a new address (if
    ///        deep copy was performed).
    ///
    /// @note  This method can be use to resize the memory pool and even
    ///        change it from internally to externally managed memory or vice versa.
    ///
    /// @throw if @c poolSize is less than this->poolUsage() the used memory
    ///        or allocations fail.
    void resizePool(uint64_t poolSize, void *data = nullptr);

}; // HostBuffer class

// --------------------------> Implementation of HostBuffer::Pool <------------------------------------

// This is private struct of HostBuffer so you can safely ignore the API
struct HostBuffer::Pool
{
    using HashTableT = std::unordered_set<HostBuffer*>;
    std::mutex mMutex; // mutex for updating mRegister and mFree
    HashTableT mRegister;
    void      *mData, *mFree;
    uint64_t   mSize, mPadding;
    bool       mManaged;

    /// @brief External memory ctor
    Pool(uint64_t size = 0, void* data = nullptr)
        : mData(data)
        , mFree(mData)
        , mSize(size)
        , mPadding(0)
        , mManaged(data == nullptr)
    {
        if (mManaged) {
            mData = Pool::alloc(mSize);
            if (mData == nullptr) throw std::runtime_error("Pool::Pool malloc failed");
        }
        mPadding = alignmentPadding(mData);
        if (!mManaged && mPadding != 0) {
            throw std::runtime_error("Pool::Pool: external memory buffer is not aligned to " +
                                     std::to_string(NANOVDB_DATA_ALIGNMENT) +
                                     " bytes.\nHint: use nanovdb::alignPtr or std::aligned_alloc (C++17 only)");
        }
        mFree = util::PtrAdd(mData, mPadding);
    }

    /// @brief Custom destructor
    ~Pool()
    {
        assert(mRegister.empty());
        if (mManaged) std::free(mData);
    }

    /// @brief Disallow copy-construction
    Pool(const Pool&) = delete;

    /// @brief Disallow move-construction
    Pool(const Pool&&) = delete;

    /// @brief Disallow copy assignment operation
    Pool& operator=(const Pool&) = delete;

    /// @brief Disallow move assignment operation
    Pool& operator=(const Pool&&) = delete;

    /// @brief Return the total number of bytes used from this Pool by buffers
    uint64_t usage() const { return util::PtrDiff(mFree, mData) - mPadding; }

    /// @brief Allocate a buffer of the specified size and add it to the register
    void add(HostBuffer* buffer, uint64_t size)
    {
        void *alignedFree = util::PtrAdd(mFree, alignmentPadding(mFree));

        if (util::PtrAdd(alignedFree, size) > util::PtrAdd(mData, mPadding + mSize)) {
            std::stringstream ss;
            ss << "HostBuffer::Pool: insufficient memory\n"
               << "\tA buffer requested " << size << " bytes with " << NANOVDB_DATA_ALIGNMENT
               << "-bytes alignment from a pool with "
               << mSize << " bytes of which\n\t" << (util::PtrDiff(alignedFree, mData) - mPadding)
               << " bytes are used by " << mRegister.size() << " other buffer(s). "
               << "Pool is " << (mManaged ? "internally" : "externally") << " managed.\n";
            //std::cerr << ss.str();
            throw std::runtime_error(ss.str());
        }
        buffer->mSize = size;
        const std::lock_guard<std::mutex> lock(mMutex);
        mRegister.insert(buffer);
        buffer->mData = alignedFree;
        mFree = util::PtrAdd(alignedFree, size);
    }

    /// @brief Remove the specified buffer from the register
    void remove(HostBuffer *buffer)
    {
        const std::lock_guard<std::mutex> lock(mMutex);
        mRegister.erase(buffer);
    }

    /// @brief Replaces buffer1 with buffer2 in the register
    void replace(HostBuffer *buffer1, HostBuffer *buffer2)
    {
        const std::lock_guard<std::mutex> lock(mMutex);
        mRegister.erase( buffer1);
        mRegister.insert(buffer2);
    }

    /// @brief Reset the register and all its buffers
    void reset()
    {
        for (HostBuffer *buffer : mRegister) {
            buffer->mPool.reset();
            buffer->mSize = 0;
            buffer->mData = nullptr;
        }
        mRegister.clear();
        mFree = util::PtrAdd(mData, mPadding);
    }

    /// @brief Resize this Pool and update registered buffers as needed. If data is no NULL
    ///        it is used as externally managed memory.
    void resize(uint64_t size, void *data = nullptr)
    {
        const uint64_t memUsage = this->usage();

        const bool managed = (data == nullptr);

        if (!managed && alignmentPadding(data) != 0) {
            throw std::runtime_error("Pool::resize: external memory buffer is not aligned to " +
                                     std::to_string(NANOVDB_DATA_ALIGNMENT) + " bytes");
        }

        if (memUsage > size) {
            throw std::runtime_error("Pool::resize: insufficient memory");
        }

        uint64_t padding = 0;
        if (mManaged && managed && size != mSize) { // managed -> managed
            padding = mPadding;
            data = Pool::realloc(mData, memUsage, size, padding); // performs both copy and free of mData
        } else if (!mManaged && managed) { // un-managed -> managed
            data = Pool::alloc(size);
            padding = alignmentPadding(data);
        }

        if (data == nullptr) {
            throw std::runtime_error("Pool::resize: allocation failed");
        } else if (data != mData) {
            void* paddedData = util::PtrAdd(data, padding);

            if (!(mManaged && managed)) { // no need to copy if managed -> managed
                memcpy(paddedData, util::PtrAdd(mData, mPadding), memUsage);
            }

            for (HostBuffer* buffer : mRegister) { // update registered buffers
                //buffer->mData = paddedData + ptrdiff_t(buffer->mData - (mData + mPadding));
                buffer->mData = util::PtrAdd(paddedData, util::PtrDiff(buffer->mData, util::PtrAdd(mData, mPadding)));
            }
            mFree = util::PtrAdd(paddedData, memUsage); // update the free pointer
            if (mManaged && !managed) {// only free if managed -> un-managed
                std::free(mData);
            }

            mData = data;
            mPadding = padding;
        }
        mSize    = size;
        mManaged = managed;
    }
    /// @brief Return true is all the memory in this pool is in use.
    bool isFull() const
    {
        assert(mFree <= util::PtrAdd(mData, mPadding + mSize));
        return mSize > 0 ? mFree == util::PtrAdd(mData, mPadding + mSize) : false;
    }

private:

    static void* alloc(uint64_t size)
    {
//#if (__cplusplus >= 201703L)
//    return std::aligned_alloc(NANOVDB_DATA_ALIGNMENT, size);//C++17 or newer
//#else
    // make sure we alloc enough space to align the result
    return std::malloc(size + NANOVDB_DATA_ALIGNMENT);
//#endif
    }

    static void* realloc(void* const origData,
                         uint64_t    origSize,
                         uint64_t    desiredSize,
                         uint64_t&   padding)
    {
        // make sure we alloc enough space to align the result
        void* data = std::realloc(origData, desiredSize + NANOVDB_DATA_ALIGNMENT);

        if (data != nullptr && data != origData) {
            uint64_t newPadding = alignmentPadding(data);
            // Number of padding bytes may have changed -- move data if that's the case
            if (newPadding != padding) {
                // Realloc should not happen when shrinking down buffer, but let's be safe
                std::memmove(util::PtrAdd(data, newPadding),
                             util::PtrAdd(data, padding),
                             math::Min(origSize, desiredSize));
                padding = newPadding;
            }
        }

        return data;
    }

};// struct HostBuffer::Pool

// --------------------------> Implementation of HostBuffer <------------------------------------

inline HostBuffer::HostBuffer(uint64_t size) : mPool(nullptr), mSize(size), mData(nullptr)
{
    if (size>0) {
        mPool = std::make_shared<Pool>(size);
        mData = mPool->mFree;
        mPool->mRegister.insert(this);
        mPool->mFree = util::PtrAdd(mPool->mFree, size);
    }
}

inline HostBuffer::HostBuffer(HostBuffer&& other) : mPool(other.mPool), mSize(other.mSize), mData(other.mData)
{
    if (mPool && mSize != 0) {
        mPool->replace(&other, this);
    }
    other.mPool.reset();
    other.mSize = 0;
    other.mData = nullptr;
}

inline void HostBuffer::init(uint64_t bufferSize, void *data)
{
    if (bufferSize == 0) {
        throw std::runtime_error("HostBuffer: invalid buffer size");
    }
    if (mPool) {
        mPool.reset();
    }
    if (!mPool || mPool->mSize != bufferSize) {
        mPool = std::make_shared<Pool>(bufferSize, data);
    }
    mPool->add(this, bufferSize);
}

inline HostBuffer& HostBuffer::operator=(HostBuffer&& other)
{
    if (mPool) {
        mPool->remove(this);
    }
    mPool = other.mPool;
    mSize = other.mSize;
    mData = other.mData;
    if (mPool && mSize != 0) {
        mPool->replace(&other, this);
    }
    other.mPool.reset();
    other.mSize = 0;
    other.mData = nullptr;
    return *this;
}

inline uint64_t HostBuffer::poolSize() const
{
    return mPool ? mPool->mSize : 0u;
}

inline uint64_t HostBuffer::poolUsage() const
{
    return mPool ? mPool->usage(): 0u;
}

inline bool HostBuffer::isManaged() const
{
    return mPool ? mPool->mManaged : false;
}

inline bool HostBuffer::isFull() const
{
    return mPool ? mPool->isFull() : false;
}

inline HostBuffer HostBuffer::createPool(uint64_t poolSize, void *data)
{
    if (poolSize == 0) {
        throw std::runtime_error("HostBuffer: invalid pool size");
    }
    HostBuffer buffer;
    buffer.mPool = std::make_shared<Pool>(poolSize, data);
    // note the buffer is NOT registered by its pool since it is not using its memory
    buffer.mSize = 0;
    buffer.mData = nullptr;
    return buffer;
}

inline HostBuffer HostBuffer::createFull(uint64_t bufferSize, void *data)
{
    if (bufferSize == 0) {
        throw std::runtime_error("HostBuffer: invalid buffer size");
    }
    HostBuffer buffer;
    buffer.mPool = std::make_shared<Pool>(bufferSize, data);
    buffer.mPool->add(&buffer, bufferSize);
    return buffer;
}

inline HostBuffer HostBuffer::create(uint64_t bufferSize, const HostBuffer* pool)
{
    HostBuffer buffer;
    if (pool == nullptr || !pool->mPool) {
        buffer.mPool = std::make_shared<Pool>(bufferSize);
    } else {
       buffer.mPool = pool->mPool;
    }
    buffer.mPool->add(&buffer, bufferSize);
    return buffer;
}

inline void HostBuffer::clear()
{
    if (mPool) {// remove self from the buffer register in the pool
        mPool->remove(this);
    }
    mPool.reset();
    mSize = 0;
    mData = nullptr;
}

inline void HostBuffer::reset()
{
    if (this->size()>0) {
        throw std::runtime_error("HostBuffer: only empty buffers can call reset");
    }
    if (!mPool) {
        throw std::runtime_error("HostBuffer: this buffer contains no pool to reset");
    }
    mPool->reset();
}

inline void HostBuffer::resizePool(uint64_t size, void *data)
{
    if (!mPool) {
        throw std::runtime_error("HostBuffer: this buffer contains no pool to resize");
    }
    mPool->resize(size, data);
}

} // namespace nanovdb

#endif // end of NANOVDB_HOSTBUFFER_H_HAS_BEEN_INCLUDED
