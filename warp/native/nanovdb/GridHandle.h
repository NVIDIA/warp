// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file nanovdb/GridHandle.h

    \author Ken Museth

    \date January 8, 2020

    \brief Defines GridHandle, which manages a host, and possibly a device,
           memory buffer containing one or more NanoVDB grids.
*/

#ifndef NANOVDB_GRID_HANDLE_H_HAS_BEEN_INCLUDED
#define NANOVDB_GRID_HANDLE_H_HAS_BEEN_INCLUDED

#include <fstream> // for std::ifstream
#include <iostream> // for std::cerr/cout
#include <vector>
#include <initializer_list>

#include <nanovdb/NanoVDB.h>// for toGridType
#include <nanovdb/HostBuffer.h>

namespace nanovdb {

// --------------------------> GridHandle <------------------------------------

struct GridHandleMetaData {uint64_t offset, size; GridType gridType;};

/// @brief This class serves to manage a buffer containing one or more NanoVDB Grids.
///
/// @note  It is important to note that this class does NOT depend on OpenVDB.
template<typename BufferT = HostBuffer>
class GridHandle
{
    std::vector<GridHandleMetaData> mMetaData;
    BufferT mBuffer;

    template <typename T>
    static T* no_const(const T* ptr) { return const_cast<T*>(ptr); }

public:
    using BufferType = BufferT;

    /// @brief  Move constructor from a host buffer
    /// @param buffer buffer containing one or more NanoGrids that will be moved into this GridHandle
    /// @throw Will throw and error with the buffer does not contain a valid NanoGrid!
    template<typename T = BufferT, typename util::enable_if<BufferTraits<T>::hasDeviceDual, int>::type = 0>
    GridHandle(T&& buffer);

    /// @brief  Move constructor from a dual host-device buffer
    /// @param buffer buffer containing one or more NanoGrids that will be moved into this GridHandle
    /// @throw Will throw and error with the buffer does not contain a valid NanoGrid!
    template<typename T = BufferT, typename util::disable_if<BufferTraits<T>::hasDeviceDual, int>::type = 0>
    GridHandle(T&& buffer);

    /// @brief Constructs an empty GridHandle
    GridHandle() = default;

    /// @brief Disallow copy-construction
    GridHandle(const GridHandle&) = delete;

    /// @brief Move copy-constructor
    GridHandle(GridHandle&& other) noexcept {
        mBuffer   = std::move(other.mBuffer);
        mMetaData = std::move(other.mMetaData);
    }

    /// @brief clear this GridHandle to an empty handle
    void reset() {
        mBuffer.clear();
        mMetaData.clear();
    }

    /// @brief Disallow copy assignment operation
    GridHandle& operator=(const GridHandle&) = delete;

    /// @brief Move copy assignment operation
    GridHandle& operator=(GridHandle&& other) noexcept {
        mBuffer   = std::move(other.mBuffer);
        mMetaData = std::move(other.mMetaData);
        return *this;
    }

    /// @brief Performs a deep copy of the GridHandle, possibly templated on a different buffer type
    /// @tparam OtherBufferT Buffer type of the deep copy
    /// @param buffer optional buffer used for allocation
    /// @return A new handle of the specified buffer type that contains a deep copy of the current handle
    template <typename OtherBufferT = HostBuffer>
    GridHandle<OtherBufferT> copy(const OtherBufferT& buffer = OtherBufferT()) const;

    /// @brief Return a reference to the buffer
    BufferT&       buffer() { return mBuffer; }

    /// @brief Return a const reference to the buffer
    const BufferT& buffer() const { return mBuffer; }

    /// @brief Returns a non-const pointer to the data.
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized
    void* data() { return mBuffer.data(); }

    /// @brief Returns a const pointer to the data.
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized
    const void* data() const { return mBuffer.data(); }

    template<typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, const void*>::type
    deviceData() const { return mBuffer.deviceData(); }
    template<typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, void*>::type
    deviceData() { return mBuffer.deviceData(); }

    /// @brief Returns the size in bytes of the raw memory buffer managed by this GridHandle.
    uint64_t size() const { return mBuffer.size(); }

    //@{
    /// @brief Return true if this handle is empty, i.e. has no allocated memory
    bool empty()   const { return this->size() == 0; }
    bool isEmpty() const { return this->size() == 0; }
    //@}

    /// @brief Return true if this handle contains any grids
    operator bool() const { return !this->empty(); }

    /// @brief Returns a const host pointer to the @a n'th NanoVDB grid encoded in this GridHandle.
    /// @tparam ValueT Value type of the grid point to be returned
    /// @param n Index of the (host) grid pointer to be returned
    /// @warning Note that the return pointer can be NULL if the GridHandle no host grid, @a n is invalid
    ///          or if the template parameter does not match the specified grid!
    template<typename ValueT>
    const NanoGrid<ValueT>* grid(uint32_t n = 0) const;

    /// @brief Returns a host pointer to the @a n'th  NanoVDB grid encoded in this GridHandle.
    /// @tparam ValueT Value type of the grid point to be returned
    /// @param n Index of the (host) grid pointer to be returned
    /// @warning Note that the return pointer can be NULL if the GridHandle no host grid, @a n is invalid
    ///          or if the template parameter does not match the specified grid!
    template<typename ValueT>
    NanoGrid<ValueT>* grid(uint32_t n = 0) {return const_cast<NanoGrid<ValueT>*>(static_cast<const GridHandle*>(this)->template grid<ValueT>(n));}

    /// @brief Return a const pointer to the @a n'th grid encoded in this GridHandle on the device, e.g. GPU
    /// @tparam ValueT Value type of the grid point to be returned
    /// @param n Index of the (device) grid pointer to be returned
    /// @warning Note that the return pointer can be NULL if the GridHandle has no device grid, @a n is invalid,
    ///          or if the template parameter does not match the specified grid.
    template<typename ValueT, typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, const NanoGrid<ValueT>*>::type
    deviceGrid(uint32_t n=0) const;

    /// @brief Return a const pointer to the @a n'th grid encoded in this GridHandle on the device, e.g. GPU
    /// @tparam ValueT Value type of the grid point to be returned
    /// @param n Index if of the grid pointer to be returned
    /// @param verbose if non-zero error messages will be printed in case something failed
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized, @a n is invalid,
    ///          or if the template parameter does not match the specified grid.
    template<typename ValueT, typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, NanoGrid<ValueT>*>::type
    deviceGrid(uint32_t n=0){return const_cast<NanoGrid<ValueT>*>(static_cast<const GridHandle*>(this)->template deviceGrid<ValueT>(n));}

    /// @brief Upload the grid to the device, e.g. from CPU to GPU
    /// @note This method is only available if the buffer supports devices
    template<typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, void>::type
    deviceUpload(void* stream = nullptr, bool sync = true) { mBuffer.deviceUpload(stream, sync); }

    /// @brief Download the grid to from the device, e.g. from GPU to CPU
    /// @note This method is only available if the buffer supports devices
    template<typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, void>::type
    deviceDownload(void* stream = nullptr, bool sync = true) { mBuffer.deviceDownload(stream, sync); }

    /// @brief Check if the buffer is this handle has any padding, i.e. if the buffer is larger than the combined size of all its grids
    /// @return true is the combined size of all grid is smaller than the buffer size
    bool isPadded() const {return mMetaData.empty() ? false : mMetaData.back().offset + mMetaData.back().size != mBuffer.size();}

    /// @brief Return the total number of grids contained in this buffer
    uint32_t gridCount() const {return static_cast<uint32_t>(mMetaData.size());}

    /// @brief Return the grid size of the @a n'th grid in this GridHandle
    /// @param n index of the grid (assumed to be less than gridCount())
    /// @return Return the byte size of the specified grid
    uint64_t gridSize(uint32_t n = 0) const {return mMetaData[n].size; }

    /// @brief Return the GridType of the @a n'th grid in this GridHandle
    /// @param n index of the grid (assumed to be less than gridCount())
    /// @return Return the GridType of the specified grid
    GridType gridType(uint32_t n = 0) const {return mMetaData[n].gridType; }

    /// @brief Access to the GridData of the n'th grid in the current handle
    /// @param n zero-based ID of the grid
    /// @return Const pointer to the n'th GridData in the current handle
    const GridData* gridData(uint32_t n = 0) const;

    /// @brief Returns a const point to the @a n'th grid meta data
    /// @param n zero-based ID of the grid
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized
    const GridMetaData* gridMetaData(uint32_t n = 0) const;

    /// @brief Write a specific grid in this buffer to an output stream
    /// @param os  output stream that the buffer will be written to
    /// @param n zero-based index of the grid to be written to stream
    void write(std::ostream& os, uint32_t n) const {
        if (const GridData* data = this->gridData(n)) {
            os.write((const char*)data, data->mGridSize);
        } else {
            throw std::runtime_error("GridHandle does not contain a #" + std::to_string(n) + " grid");
        }
    }

    /// @brief Write the entire grid buffer to an output stream
    /// @param os output stream that the buffer will be written to
    void write(std::ostream& os) const {
        for (uint32_t n=0; n<this->gridCount(); ++n) this->write(os, n);
    }

    /// @brief Write this entire grid buffer to a file
    /// @param fileName string name of the output file
    void write(const std::string &fileName) const {
        std::ofstream os(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!os.is_open()) throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for output");
        this->write(os);
    }

    /// @brief Write a specific grid to file
    /// @param fileName string name of the output file
    /// @param n zero-based index of the grid to be written to file
    void write(const std::string &fileName, uint32_t n) const {
        std::ofstream os(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!os.is_open()) throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for output");
        this->write(os, n);
    }

    /// @brief Read an entire raw grid buffer from an input stream
    /// @param is input stream containing a raw grid buffer
    /// @param pool optional pool from which to allocate the new grid buffer
    /// @throw Will throw a std::logic_error if the stream does not contain a valid raw grid
    void read(std::istream& is, const BufferT& pool = BufferT());

    /// @brief Read a specific grid from an input stream containing a raw grid buffer
    /// @param is input stream containing a raw grid buffer
    /// @param n zero-based index of the grid to be read
    /// @param pool optional pool from which to allocate the new grid buffer
    /// @throw Will throw a std::logic_error if the stream does not contain a valid raw grid
    void read(std::istream& is, uint32_t n, const BufferT& pool = BufferT());

    /// @brief Read a specific grid from an input stream containing a raw grid buffer
    /// @param is input stream containing a raw grid buffer
    /// @param gridName string name of the grid to be read
    /// @param pool optional pool from which to allocate the new grid buffer
    /// @throw Will throw a std::logic_error if the stream does not contain a valid raw grid with the speficied name
    void read(std::istream& is, const std::string &gridName, const BufferT& pool = BufferT());

    /// @brief Read a raw grid buffer from a file
    /// @param filename string name of the input file containing a raw grid buffer
    /// @param pool optional pool from which to allocate the new grid buffe
    void read(const std::string &fileName, const BufferT& pool = BufferT()) {
        std::ifstream is(fileName, std::ios::in | std::ios::binary);
        if (!is.is_open()) throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for input");
        this->read(is, pool);
    }

    /// @brief Read a specific grid from a file containing a raw grid buffer
    /// @param filename string name of the input file containing a raw grid buffer
    /// @param n zero-based index of the grid to be read
    /// @param pool optional pool from which to allocate the new grid buffer
    /// @throw Will throw a std::ios_base::failure if the file does not exist and a
    ///        std::logic_error if the files does not contain a valid raw grid
    void read(const std::string &fileName, uint32_t n, const BufferT& pool = BufferT()) {
        std::ifstream is(fileName, std::ios::in | std::ios::binary);
        if (!is.is_open()) throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for input");
        this->read(is, n, pool);
    }

    /// @brief Read a specific grid from a file containing a raw grid buffer
    /// @param filename string name of the input file containing a raw grid buffer
    /// @param gridName string name of the grid to be read
    /// @param pool optional pool from which to allocate the new grid buffer
    /// @throw Will throw a std::ios_base::failure if the file does not exist and a
    ///        std::logic_error if the files does not contain a valid raw grid withe the specified name
    void read(const std::string &fileName, const std::string &gridName, const BufferT& pool = BufferT()) {
        std::ifstream is(fileName, std::ios::in | std::ios::binary);
        if (!is.is_open()) throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for input");
        this->read(is, gridName, pool);
    }
}; // GridHandle

// --------------------------> Implementation of private methods in GridHandle <------------------------------------

template<typename BufferT>
inline const GridData* GridHandle<BufferT>::gridData(uint32_t n) const
{
    const void *data = this->data();
    if (data == nullptr || n >= mMetaData.size()) return nullptr;
    return util::PtrAdd<GridData>(data, mMetaData[n].offset);
}// const GridData* GridHandle<BufferT>::gridData(uint32_t n) const

template<typename BufferT>
inline const GridMetaData* GridHandle<BufferT>::gridMetaData(uint32_t n) const
{
    const auto *data = this->data();
    if (data == nullptr || n >= mMetaData.size()) return nullptr;
    return util::PtrAdd<GridMetaData>(data, mMetaData[n].offset);
}// const GridMetaData* GridHandle<BufferT>::gridMetaData(uint32_t n) const

inline __hostdev__ void cpyGridHandleMeta(const GridData *data, GridHandleMetaData *meta)
{
    uint64_t offset = 0;
    for (auto *p=meta, *q=p+data->mGridCount; p!=q; ++p) {
        *p = {offset,  data->mGridSize, data->mGridType};
        offset += p->size;
        data = util::PtrAdd<GridData>(data, p->size);
    }
}// void cpyGridHandleMeta(const GridData *data, GridHandleMetaData *meta)

template<typename BufferT>
template<typename T, typename util::disable_if<BufferTraits<T>::hasDeviceDual, int>::type>
GridHandle<BufferT>::GridHandle(T&& buffer)
{
    static_assert(util::is_same<T,BufferT>::value, "Expected U==BufferT");
    mBuffer = std::move(buffer);
    if (auto *data = reinterpret_cast<const GridData*>(mBuffer.data())) {
        if (!data->isValid()) throw std::runtime_error("GridHandle was constructed with an invalid host buffer");
        mMetaData.resize(data->mGridCount);
        cpyGridHandleMeta(data, mMetaData.data());
    }
}// GridHandle<BufferT>::GridHandle(T&& buffer)

template<typename BufferT>
template <typename OtherBufferT>
inline GridHandle<OtherBufferT> GridHandle<BufferT>::copy(const OtherBufferT& other) const
{
    if (mBuffer.isEmpty()) return GridHandle<OtherBufferT>();// return an empty handle
    auto buffer = OtherBufferT::create(mBuffer.size(), &other);
    std::memcpy(buffer.data(), mBuffer.data(), mBuffer.size());// deep copy of buffer
    return GridHandle<OtherBufferT>(std::move(buffer));
}// GridHandle<OtherBufferT> GridHandle<BufferT>::copy(const OtherBufferT& other) const

template<typename BufferT>
template<typename ValueT>
inline const NanoGrid<ValueT>* GridHandle<BufferT>::grid(uint32_t n) const
{
    const void *data = mBuffer.data();
    if (data == nullptr || n >= mMetaData.size() || mMetaData[n].gridType != toGridType<ValueT>()) return nullptr;
    return util::PtrAdd<NanoGrid<ValueT>>(data, mMetaData[n].offset);
}// const NanoGrid<ValueT>* GridHandle<BufferT>::grid(uint32_t n) const

template<typename BufferT>
template<typename ValueT, typename U>
inline typename util::enable_if<BufferTraits<U>::hasDeviceDual, const NanoGrid<ValueT>*>::type
GridHandle<BufferT>::deviceGrid(uint32_t n) const
{
    const void *data = mBuffer.deviceData();
    if (data == nullptr || n >= mMetaData.size() || mMetaData[n].gridType != toGridType<ValueT>()) return nullptr;
    return util::PtrAdd<NanoGrid<ValueT>>(data, mMetaData[n].offset);
}// GridHandle<BufferT>::deviceGrid(uint32_t n) cons


} // namespace nanovdb

#if defined(__CUDACC__)
#include <nanovdb/cuda/GridHandle.cuh>
#endif// defined(__CUDACC__)

#endif // NANOVDB_GRID_HANDLE_H_HAS_BEEN_INCLUDED
