// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file nanovdb/util/cuda/Timer.h
///
/// @author Ken Museth
///
/// @brief A simple GPU timing class

#ifndef NANOVDB_UTIL_CUDA_TIMER_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_CUDA_TIMER_H_HAS_BEEN_INCLUDED

#include <iostream>// for std::cerr
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace nanovdb {

namespace util::cuda {

class Timer
{
    cudaStream_t mStream{0};
    cudaEvent_t  mStart, mStop;

public:
    /// @brief Default constructor
    /// @param stream CUDA stream to be timed (defaults to stream 0)
    /// @note Starts the timer
    /// @warning @c cudaEventCreate creates the event for the current device
    ///          and @c cudaEventRecord requires that the event and stream are
    ///          associated with the same device. So it's important to call
    ///          @c cudaSetDevice(device) so @c device matches the one used
    ///          when @c stream was created.
    Timer(cudaStream_t stream = 0) : mStream(stream)
    {
        cudaEventCreate(&mStart);
        cudaEventCreate(&mStop);
        cudaEventRecord(mStart, mStream);
    }

    /// @brief Construct and start the timer
    /// @param msg string message to be printed when timer is started
    /// @param stream CUDA stream to be timed (defaults to stream 0)
    /// @param os output stream for the message above
    /// @warning @c cudaEventCreate creates the event for the current device
    ///          and @c cudaEventRecord requires that the event and stream are
    ///          associated with the same device. So it's important to call
    ///          @c cudaSetDevice(device) so @c device matches the one used
    ///          when @c stream was created.
    Timer(const std::string &msg, cudaStream_t stream = 0, std::ostream& os = std::cerr)
        : mStream(stream)
    {
        os << msg << " ... " << std::flush;
        cudaEventCreate(&mStart);
        cudaEventCreate(&mStop);
        cudaEventRecord(mStart, mStream);
    }

    /// @brief Destructor
    ~Timer()
    {
        cudaEventDestroy(mStart);
        cudaEventDestroy(mStop);
    }

    /// @brief Start the timer
    /// @param stream CUDA stream to be timed (defaults to stream 0)
    /// @param os output stream for the message above
    /// @warning @c cudaEventRecord requires that the event and stream are
    ///          associated with the same device. So it's important to call
    ///          @c cudaSetDevice(device) so @c device matches the one used
    ///          when @c mStream was created.
    void start() {cudaEventRecord(mStart, mStream);}

    /// @brief Start the timer
    /// @param msg string message to be printed when timer is started

    /// @param os output stream for the message above
    void start(const std::string &msg, std::ostream& os = std::cerr)
    {
        os << msg << " ... " << std::flush;
        this->start();
    }

    /// @brief Start the timer
    /// @param msg string message to be printed when timer is started
    /// @param os output stream for the message above
    void start(const char* msg, std::ostream& os = std::cerr)
    {
        os << msg << " ... " << std::flush;
        this->start();
    }

    /// @warning @c cudaEventRecord requires that the event and stream are
    ///          associated with the same device. So it's important to call
    ///          @c cudaSetDevice(device) so it matches the device used when
    ///          @c mStream was created.
    inline void record()
    {
        cudaEventRecord(mStop, mStream);
        cudaEventSynchronize(mStop);
    }

    /// @brief Return the time in milliseconds since record was called
    inline float milliseconds() const
    {
        float msec = 0.0f;
        cudaEventElapsedTime(&msec, mStart, mStop);
        return msec;
    }

    /// @brief elapsed time (since start) in miliseconds
    /// @return elapsed time (since start) in miliseconds
    inline float elapsed()
    {
        this->record();
        return this->milliseconds();
    }

    /// @brief Prints the elapsed time in milliseconds to a stream
    /// @param os output stream to print to
    inline void print(std::ostream& os = std::cerr)
    {
        const float msec = this->milliseconds();
        os << "completed in " << msec << " milliseconds" << std::endl;
    }

    /// @brief Prints a message followed by the elapsed time in milliseconds to a stream
    /// @param msg message to print before the time
    /// @param os stream to print to
    inline void print(const char* msg, std::ostream& os = std::cerr)
    {
        os << msg;
        this->print(os);
    }

    /// @brief Like the above method but with a std::string arguments
    inline void print(const std::string &msg, std::ostream& os = std::cerr){this->print(msg.c_str(), os);}

    /// @brief stop the timer
    /// @param os output stream for the message above
    inline void stop(std::ostream& os = std::cerr)
    {
        this->record();
        this->print(os);
    }

    /// @brief stop and start the timer
    /// @param msg string message to be printed when timer is started
    /// @warning Remember to call start before restart
    inline void restart(const std::string &msg, std::ostream& os = std::cerr)
    {
        this->stop(os);
        this->start(msg, os);
    }
};// Timer

}// namespace util::cuda

using GpuTimer [[deprecated("Use nanovdb::util::cuda::Timer instead")]]= util::cuda::Timer;

} // namespace nanovdb

#endif // NANOVDB_UTIL_CUDA_TIMER_H_HAS_BEEN_INCLUDED
