/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "builtin.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/device_memory.h"

#define F16_STR "<f2"
#define F32_STR "<f4"
#define F64_STR "<f8"

namespace wp {

template <typename Gemm>
struct Allocation {
  // Allocations holding input and output tensors
  cutlass::DeviceAllocation<typename Gemm::ElementA> ptr_A;
  cutlass::DeviceAllocation<typename Gemm::ElementB> ptr_B;
  cutlass::DeviceAllocation<typename Gemm::ElementC> ptr_C;
  cutlass::DeviceAllocation<typename Gemm::ElementC> ptr_D;

  // Leading dimensions
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  int64_t ldd;

  Allocation(int m, int n, int k, int batch_count, const void* a, const void* b, const void* c, void* d) {
    ptr_A.reset(m * k * batch_count);
    ptr_B.reset(k * n * batch_count);
    ptr_C.reset(m * n * batch_count);
    ptr_D.reset(m * n * batch_count);

    ptr_A.copy_from_host((typename Gemm::ElementA*)a);
    ptr_B.copy_from_host((typename Gemm::ElementB*)b);
    ptr_C.copy_from_host((typename Gemm::ElementC*)c);
    ptr_D.copy_from_host((typename Gemm::ElementC*)d);

    lda = k;
    ldb = n;
    ldc = n;
    ldd = n;
  }
};


template <typename Gemm>
bool run_gemm(int m, int n, int k, int batch_count, const void* a, const void* b, const void* c, void* d, float alpha, float beta) {
    //
    // Allocate and initialize arguments
    //

    Allocation<Gemm> alloc(m, n, k, batch_count, a, b, c, d);
    typename Gemm::EpilogueOutputOp::Params epilogue_params(
        (typename Gemm::EpilogueOutputOp::ElementCompute)alpha,
        (typename Gemm::EpilogueOutputOp::ElementCompute)beta);

    typename Gemm::Arguments arguments{
        batch_count == 1 ? cutlass::gemm::GemmUniversalMode::kGemm : cutlass::gemm::GemmUniversalMode::kBatched ,
        cutlass::gemm::GemmCoord{m, n, k}, // Problem size
        batch_count,
        epilogue_params,
        (void const*)alloc.ptr_A.get(),
        (void const*)alloc.ptr_B.get(),
        (void const*)alloc.ptr_C.get(),
        (void      *)alloc.ptr_D.get(),
        int64_t(m * k), int64_t(k * n), int64_t(m * n), int64_t(m * n), // Batch strides
        alloc.lda,
        alloc.ldb,
        alloc.ldc,
        alloc.ldd
    };

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm.initialize(arguments, workspace.get());

    if (status != cutlass::Status::kSuccess) {
        cudaError_t error = cudaGetLastError();
        std::cerr << "Error initializing GEMM: " << cudaGetErrorString(error) << "\n";
        return false;
    }

    //
    // Run the GEMM
    //

    status = gemm();
    if (status != cutlass::Status::kSuccess) {
        cudaError_t error = cudaGetLastError();
        std::cerr << "Runtime error: " << cudaGetErrorString(error) << "\n";
        return false;
    }

    alloc.ptr_D.copy_to_host((typename Gemm::ElementC*)d);
    return true;
}


template <
    int ComputeCapability,
    typename Element_
>
struct DefaultGemmConfig;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Partial specialization for SM80 F64 Tensor Cores
template <>
struct DefaultGemmConfig<80, double> {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        double, cutlass::layout::RowMajor,                              // ElementA and LayoutA
        double, cutlass::layout::RowMajor,                              // ElementB and LayoutB
        double, cutlass::layout::RowMajor,                              // ElementC and LayoutC
        double,                                                         // ElementAccumulator
        cutlass::arch::OpClassTensorOp,                                 // Operation type
        cutlass::arch::Sm80,                                            // Architecture
        cutlass::gemm::GemmShape<128, 128, 16>,                         // ThreadblockShape
        cutlass::gemm::GemmShape<32, 64, 16>,                           // WarpShape
        cutlass::gemm::GemmShape<8, 8, 4>,                              // Instruction Shape
        cutlass::epilogue::thread::LinearCombination<                   // Epilogue
            double,
            1,
            double,
            double>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,   // Swizzling
        3                                                               // Stages
    >;
};

// Partial specialization for SM80 F32 Tensor Cores
template <>
struct DefaultGemmConfig<80, float> {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        float, cutlass::layout::RowMajor,                               // ElementA and LayoutA
        float, cutlass::layout::RowMajor,                               // ElementB and LayoutB
        float, cutlass::layout::RowMajor,                               // ElementC and LayoutC
        float,                                                          // ElementAccumulator
        cutlass::arch::OpClassTensorOp,                                 // Operation type
        cutlass::arch::Sm80,                                            // Architecture
        cutlass::gemm::GemmShape<256, 128, 16>,                         // ThreadblockShape
        cutlass::gemm::GemmShape<64, 64, 16>,                           // WarpShape
        cutlass::gemm::GemmShape<16, 8, 8>,                             // Instruction Shape
        cutlass::epilogue::thread::LinearCombination<                   // Epilogue
            float,
            128 / cutlass::sizeof_bits<float>::value,
            float,
            float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,   // Swizzling
        3,                                                              // Stages
        4, 4,                                                           // AlignmentA and AlignmentB
        cutlass::arch::OpMultiplyAddFastF32                             // Math mode -- use 3xTF32
    >;
};

// Partial specialization for SM80 F16 Tensor Cores
template <>
struct DefaultGemmConfig<80, cutlass::half_t> {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        cutlass::half_t, cutlass::layout::RowMajor,                     // ElementA and LayoutA
        cutlass::half_t, cutlass::layout::RowMajor,                     // ElementB and LayoutB
        cutlass::half_t, cutlass::layout::RowMajor,                     // ElementC and LayoutC
        cutlass::half_t,                                                // ElementAccumulator
        cutlass::arch::OpClassTensorOp,                                 // Operation type
        cutlass::arch::Sm80,                                            // Architecture
        cutlass::gemm::GemmShape<256, 128, 32>,                         // ThreadblockShape
        cutlass::gemm::GemmShape<64, 64, 32>,                           // WarpShape
        cutlass::gemm::GemmShape<16, 8, 16>,                            // Instruction Shape
        cutlass::epilogue::thread::LinearCombination<                   // Epilogue
            cutlass::half_t,
            128 / cutlass::sizeof_bits<cutlass::half_t>::value,
            cutlass::half_t,
            cutlass::half_t>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,   // Swizzling
        3                                                               // Stages
    >;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Partial specialization for SM75 F16 Tensor Cores
template <>
struct DefaultGemmConfig<75, cutlass::half_t> {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        cutlass::half_t, cutlass::layout::RowMajor,                     // ElementA and LayoutA
        cutlass::half_t, cutlass::layout::RowMajor,                     // ElementB and LayoutB
        cutlass::half_t, cutlass::layout::RowMajor,                     // ElementC and LayoutC
        cutlass::half_t,                                                // ElementAccumulator
        cutlass::arch::OpClassTensorOp,                                 // Operation type
        cutlass::arch::Sm75,                                            // Architecture
        cutlass::gemm::GemmShape<256, 128, 32>,                         // ThreadblockShape
        cutlass::gemm::GemmShape<64, 64, 32>,                           // WarpShape
        cutlass::gemm::GemmShape<16, 8, 8>,                             // Instruction Shape
        cutlass::epilogue::thread::LinearCombination<                   // Epilogue
            cutlass::half_t,
            128 / cutlass::sizeof_bits<cutlass::half_t>::value,
            cutlass::half_t,
            cutlass::half_t>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,   // Swizzling
        2                                                               // Stages
    >;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Partial specialization for SM70 F16 Tensor Cores
template <>
struct DefaultGemmConfig<70, cutlass::half_t> {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        cutlass::half_t, cutlass::layout::RowMajor,                     // ElementA and LayoutA
        cutlass::half_t, cutlass::layout::RowMajor,                     // ElementB and LayoutB
        cutlass::half_t, cutlass::layout::RowMajor,                     // ElementC and LayoutC
        cutlass::half_t,                                                // ElementAccumulator
        cutlass::arch::OpClassTensorOp,                                 // Operation type
        cutlass::arch::Sm70,                                            // Architecture
        cutlass::gemm::GemmShape<256, 128, 32>,                         // ThreadblockShape
        cutlass::gemm::GemmShape<64, 64, 32>,                           // WarpShape
        cutlass::gemm::GemmShape<8, 8, 4>,                              // Instruction Shape
        cutlass::epilogue::thread::LinearCombination<                   // Epilogue
            cutlass::half_t,
            128 / cutlass::sizeof_bits<cutlass::half_t>::value,
            cutlass::half_t,
            cutlass::half_t>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,   // Swizzling
        2                                                               // Stages
    >;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Partial specialization for SM50 SIMT
template <typename Element>
struct DefaultGemmConfig<50, Element> {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        Element, cutlass::layout::RowMajor,                             // ElementA and LayoutA
        Element, cutlass::layout::RowMajor,                             // ElementB and LayoutB
        Element, cutlass::layout::RowMajor,                             // ElementC and LayoutC
        Element,                                                        // ElementAccumulator
        cutlass::arch::OpClassSimt,                                     // Operation type
        cutlass::arch::Sm50,                                            // Architecture
        cutlass::gemm::GemmShape<128, 128, 8>,                          // ThreadblockShape
        cutlass::gemm::GemmShape<32, 64, 8>,                            // WarpShape
        cutlass::gemm::GemmShape<1, 1, 1>,                              // Instruction Shape
        cutlass::epilogue::thread::LinearCombination<                   // Epilogue
            Element,
            1,
            Element,
            Element>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,   // Swizzling
        2                                                               // Stages
    >;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" {

WP_API
bool cutlass_gemm(
                  int compute_capability,
                  int m, int n, int k,
                  const char* datatype_str,
                  const void* a, const void* b, const void* c, void* d,
                  float alpha, float beta,
                  bool allow_tf32x3_arith,
                  int batch_count) {

    std::string datatype(datatype_str);

    // Specializations for using Tensor Cores
    if (compute_capability == 80) {
        if (datatype == F64_STR) {
            using Gemm = DefaultGemmConfig<80, double>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        } else if (datatype == F32_STR && allow_tf32x3_arith) {
            using Gemm = DefaultGemmConfig<80, float>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        } else if (datatype == F16_STR) {
            using Gemm = DefaultGemmConfig<80, cutlass::half_t>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        }
    } else if (compute_capability == 75) {
        if (datatype == F16_STR) {
            using Gemm = DefaultGemmConfig<75, cutlass::half_t>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        }
    } else if (compute_capability == 70) {
        if (datatype == F16_STR) {
            using Gemm = DefaultGemmConfig<70, cutlass::half_t>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        }
    }

    // No Tensor Core capability available. Run a SIMT kernel
    if (datatype == F64_STR) {
        using Gemm = DefaultGemmConfig<50, double>::Gemm;
        return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
    } else if (datatype == F32_STR) {
        using Gemm = DefaultGemmConfig<50, float>::Gemm;
        return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
    } else if (datatype == F16_STR) {
        using Gemm = DefaultGemmConfig<50, cutlass::half_t>::Gemm;
        return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
    }

    std::cerr << "Data type " << datatype << " is not currently supported." << std::endl;
    return false;
}

}

} // namespace wp