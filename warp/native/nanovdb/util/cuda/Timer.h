// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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

namespace util{ namespace cuda {

class Timer
{
    cudaStream_t mStream{0};
    cudaEvent_t mStart, mStop;

public:
    /// @brief Default constructor
    /// @param stream CUDA stream to be timed (defaults to stream 0)
    /// @note Starts the timer
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

    /// @brief elapsed time (since start) in miliseconds
    /// @return elapsed time (since start) in miliseconds
    float elapsed()
    {
        cudaEventRecord(mStop, mStream);
        cudaEventSynchronize(mStop);
        float diff = 0.0f;
        cudaEventElapsedTime(&diff, mStart, mStop);
        return diff;
    }

    /// @brief stop the timer
    /// @param os output stream for the message above
    void stop(std::ostream& os = std::cerr)
    {
        float diff = this->elapsed();
        os << "completed in " << diff << " milliseconds" << std::endl;
    }

    /// @brief stop and start the timer
    /// @param msg string message to be printed when timer is started
    /// @warning Remember to call start before restart
    void restart(const std::string &msg, std::ostream& os = std::cerr)
    {
        this->stop();
        this->start(msg, os);
    }
};// Timer

}}// namespace util::cuda

using GpuTimer [[deprecated("Use nanovdb::util::cuda::Timer instead")]]= util::cuda::Timer;

} // namespace nanovdb

#endif // NANOVDB_UTIL_CUDA_TIMER_H_HAS_BEEN_INCLUDED
