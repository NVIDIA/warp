/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "builtin.h"
#include "temp_buffer.h"
#include "cuda_util.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/device_memory.h"

#define F16_STR "<f2"
#define F32_STR "<f4"
#define F64_STR "<f8"

namespace wp {

template <typename Gemm>
bool run_gemm(int m, int n, int k, int batch_count, const void* a, const void* b, const void* c, void* d, float alpha, float beta) {
    //
    // Initialize arguments
    //
    typename Gemm::EpilogueOutputOp::Params epilogue_params(
        (typename Gemm::EpilogueOutputOp::ElementCompute)alpha,
        (typename Gemm::EpilogueOutputOp::ElementCompute)beta);

    typename Gemm::Arguments arguments{
        batch_count == 1 ? cutlass::gemm::GemmUniversalMode::kGemm : cutlass::gemm::GemmUniversalMode::kBatched ,
        cutlass::gemm::GemmCoord{m, n, k}, // Problem size
        batch_count,
        epilogue_params,
        a, b, c, d,
        int64_t(m * k), int64_t(k * n), int64_t(m * n), int64_t(m * n), // Batch strides
        Gemm::LayoutA::packed({m, k}).stride(0), Gemm::LayoutB::packed({k, n}).stride(0), n, n
    };

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    ScopedTemporary<> workspace(WP_CURRENT_CONTEXT, workspace_size);
    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_get_current());
    cutlass::Status status = gemm.initialize(arguments, workspace.buffer(), stream);

    if (status != cutlass::Status::kSuccess) {
        cudaError_t error = cudaGetLastError();
        std::cerr << "Error initializing GEMM: " << cudaGetErrorString(error) << "\n";
        return false;
    }

    //
    // Run the GEMM
    //

    status = gemm(stream);
    if (status != cutlass::Status::kSuccess) {
        cudaError_t error = cudaGetLastError();
        std::cerr << "Runtime error: " << cudaGetErrorString(error) << "\n";
        return false;
    }

    return true;
}

template <
    int ComputeCapability,
    typename Element_,
    typename LayoutA,
    typename LayoutB
>
struct DefaultGemmConfig;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Partial specialization for SM80 F64 Tensor Cores
template <typename LayoutA, typename LayoutB>
struct DefaultGemmConfig<80, double, LayoutA, LayoutB> {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        double, LayoutA,                                                // ElementA and LayoutA
        double, LayoutB,                                                // ElementB and LayoutB
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
template <typename LayoutA, typename LayoutB>
struct DefaultGemmConfig<80, float, LayoutA, LayoutB> {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        float, LayoutA,                                                 // ElementA and LayoutA
        float, LayoutB,                                                 // ElementB and LayoutB
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
template <typename LayoutA, typename LayoutB>
struct DefaultGemmConfig<80, cutlass::half_t, LayoutA, LayoutB> {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        cutlass::half_t, LayoutA,                                       // ElementA and LayoutA
        cutlass::half_t, LayoutB,                                       // ElementB and LayoutB
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
template <typename LayoutA, typename LayoutB>
struct DefaultGemmConfig<75, cutlass::half_t, LayoutA, LayoutB> {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        cutlass::half_t, LayoutA,                                       // ElementA and LayoutA
        cutlass::half_t, LayoutB,                                       // ElementB and LayoutB
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
template <typename LayoutA, typename LayoutB>
struct DefaultGemmConfig<70, cutlass::half_t, LayoutA, LayoutB> {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        cutlass::half_t, LayoutA,                                       // ElementA and LayoutA
        cutlass::half_t, LayoutB,                                       // ElementB and LayoutB
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
template <typename Element, typename LayoutA, typename LayoutB>
struct DefaultGemmConfig<50, Element, LayoutA, LayoutB> {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        Element, LayoutA,                                               // ElementA and LayoutA
        Element, LayoutB,                                               // ElementB and LayoutB
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
                  void* context, int compute_capability,
                  int m, int n, int k,
                  const char* datatype_str,
                  const void* a, const void* b, const void* c, void* d,
                  float alpha, float beta,
                  bool row_major_a, bool row_major_b,
                  bool allow_tf32x3_arith,
                  int batch_count) {

    std::string datatype(datatype_str);

    ContextGuard guard(context);

    // Specializations for using Tensor Cores and A/B RowMajor/ColumnMajor designations
    if (compute_capability == 80) {
        if (datatype == F64_STR) {
            if (row_major_a && row_major_b) {
                using Gemm = DefaultGemmConfig<80, double, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            } else if (!row_major_a && row_major_b) {
                using Gemm = DefaultGemmConfig<80, double, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            } else if (row_major_a && !row_major_b) {
                using Gemm = DefaultGemmConfig<80, double, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            } else if (!row_major_a && !row_major_b) {
                using Gemm = DefaultGemmConfig<80, double, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            }
        } else if (datatype == F32_STR && allow_tf32x3_arith) {
            if (row_major_a && row_major_b) {
                using Gemm = DefaultGemmConfig<80, float, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            } else if (!row_major_a && row_major_b) {
                using Gemm = DefaultGemmConfig<80, float, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            } else if (row_major_a && !row_major_b) {
                using Gemm = DefaultGemmConfig<80, float, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            } else if (!row_major_a && !row_major_b) {
                using Gemm = DefaultGemmConfig<80, float, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            }
        } else if (datatype == F16_STR) {
            if (row_major_a && row_major_b) {
                using Gemm = DefaultGemmConfig<80, cutlass::half_t, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            } else if (!row_major_a && row_major_b) {
                using Gemm = DefaultGemmConfig<80, cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            } else if (row_major_a && !row_major_b) {
                using Gemm = DefaultGemmConfig<80, cutlass::half_t, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            } else if (!row_major_a && !row_major_b) {
                using Gemm = DefaultGemmConfig<80, cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            }
        }
    } else if (compute_capability == 75) {
        if (datatype == F16_STR) {
            if (row_major_a && row_major_b) {
                using Gemm = DefaultGemmConfig<75, cutlass::half_t, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);    
            } else if (!row_major_a && row_major_b) {
                using Gemm = DefaultGemmConfig<75, cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);    
            } else if (row_major_a && !row_major_b) {
                using Gemm = DefaultGemmConfig<75, cutlass::half_t, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);    
            } else if (!row_major_a && !row_major_b) {
                using Gemm = DefaultGemmConfig<75, cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);    
            }
        }
    } else if (compute_capability == 70) {
        if (datatype == F16_STR) {
            if (row_major_a && row_major_b) {
                using Gemm = DefaultGemmConfig<70, cutlass::half_t, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            } else if (!row_major_a && row_major_b) {
                using Gemm = DefaultGemmConfig<70, cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            } else if (row_major_a && !row_major_b) {
                using Gemm = DefaultGemmConfig<70, cutlass::half_t, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            } else if (!row_major_a && !row_major_b) {
                using Gemm = DefaultGemmConfig<70, cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>::Gemm;
                return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
            }
        }
    }

    // No Tensor Core capability available. Run a SIMT kernel
    if (datatype == F64_STR) {
        if (row_major_a && row_major_b) {
            using Gemm = DefaultGemmConfig<50, double, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        } else if (!row_major_a && row_major_b) {
            using Gemm = DefaultGemmConfig<50, double, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        } else if (row_major_a && !row_major_b) {
            using Gemm = DefaultGemmConfig<50, double, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        } else if (!row_major_a && !row_major_b) {
            using Gemm = DefaultGemmConfig<50, double, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        }
    } else if (datatype == F32_STR) {
        if (row_major_a && row_major_b) {
            using Gemm = DefaultGemmConfig<50, float, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        } else if (!row_major_a && row_major_b) {
            using Gemm = DefaultGemmConfig<50, float, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        } else if (row_major_a && !row_major_b) {
            using Gemm = DefaultGemmConfig<50, float, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        } else if (!row_major_a && !row_major_b) {
            using Gemm = DefaultGemmConfig<50, float, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        }
    } else if (datatype == F16_STR) {
        if (row_major_a && row_major_b) {
            using Gemm = DefaultGemmConfig<50, cutlass::half_t, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        } else if (!row_major_a && row_major_b) {
            using Gemm = DefaultGemmConfig<50, cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        } else if (row_major_a && !row_major_b) {
            using Gemm = DefaultGemmConfig<50, cutlass::half_t, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        } else if (!row_major_a && !row_major_b) {
            using Gemm = DefaultGemmConfig<50, cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>::Gemm;
            return run_gemm<Gemm>(m, n, k, batch_count, a, b, c, d, alpha, beta);
        }
    }

    std::cerr << "Data type " << datatype << " is not currently supported." << std::endl;
    return false;
}

}

} // namespace wp