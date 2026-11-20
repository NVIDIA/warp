# CPU Loop Vectorization

**Status**: Implemented

**Issue**: [GH-1954](https://github.com/NVIDIA/warp/issues/1954)

## Motivation

Warp generates C++ for CPU kernels and invokes Clang's frontend directly. The Clang driver normally enables LLVM's loop vectorizer at `-O2` and `-O3`, but that driver step is absent here. As a result, loops over adjacent array elements were compiled as scalar operations even on CPUs with suitable vector instructions.

Enabling the pass alone is insufficient for common kernels. A generic scalar `wp.tid()` call converts task indices through launch coordinates, and one dimensional array access uses runtime strides. Those operations make a simple unit-stride loop harder for LLVM to recognize. The optimized path should expose the simpler form without changing the behavior of strided views, multidimensional launches, gradients, or overlapping arrays.

## Design

### Enable the loop pass

`warp/native/clang/clang.cpp` enables `CodeGenOptions::VectorizeLoop` for CPU compilation at effective optimization level 2 or 3. Debug builds and lower optimization levels retain their existing behavior. A module can opt out with `-fno-vectorize` in `cpu_compiler_flags`; Warp handles this driver-style flag because it calls the frontend directly. This disables only LLVM's CPU loop vectorizer. Constant-range lowering and eligible CPU source specialization still apply. The pass switch does not affect CUDA compilation, but the shared constant-range lowering changes generated CUDA source.

LLVM still decides which loops to vectorize, the profitable vector width, and whether to use scalar remainders. This change does not enable floating-point reassociation or promise a particular instruction set.

### Make eligible loops easier to analyze

When a constant `range()` exceeds `max_unroll` and its body has no `break`, code generation emits an ordinary induction loop with the known start, stop, and step. This replaces an iterator helper that hides the trip count from LLVM. Reverse-mode code generation traverses the same range in reverse; empty ranges, dynamic ranges, and loops with `break` retain their existing semantics.

For `block_dim=1` kernels with one dimensional launch coordinates, scalar `wp.tid()`, and at least one direct one dimensional array argument, code generation may emit a fast kernel variant and a generic fallback. The fast variant is available when either every direct array argument is one dimensional, or the kernel contains a rolled constant loop or tile work. A runtime guard selects it only when launch coordinates are linear and the relevant array strides match their element sizes. Backward dispatch checks both primal and adjoint strides. The fast variant uses the task index directly and copies array descriptors with constant one dimensional strides so LLVM can see unit-stride access. The generic variant preserves arbitrary strides and coordinate conversion.

The guard makes no non-aliasing promise. In-place and overlapping arrays retain their sequential behavior. Cooperative `block_dim>1` kernels continue to use their existing fiber execution path. Python code generation emits only the serial or fiber entry path needed for the selected block size, and omits the contiguous dispatch path for ineligible kernels.

### Control known costs

The CPU row loop in `mat_t::outer()` has a targeted Clang pragma disabling loop vectorization. LLVM's chosen transformation for this small loop regressed measured workloads; the rest of the module can still use the loop pass.

Task and dispatch wrappers are force-inlined so LLVM sees the task loop early without repeatedly optimizing separate wrappers. The default CPU helper inlining thresholds are 9000 for ordinary calls and 3000 for cold call sites. The higher ordinary threshold restores inlining of helpers with multiple call sites in specialized kernels, including the XPBD contact helper.

The separate [CPU SLP vectorization design](cpu-slp-vectorization.md) documents the SLP pass and its tuning.

## Tests and benchmarks

`TestCpuVectorization` checks contiguous and strided one dimensional arrays, in-place and overlapping access, fast-path eligibility, and forward and backward constant-range behavior. `TestCpuBlockCodegen` checks serial and cooperative CPU source generation. `TestCodeGen` covers range edge cases across devices, and `TestFuncInline` checks generated helper attributes. These tests assert correctness and source structure without timing assertions.

The CPU ASV `time_linear_array_loop_arithmetic` case measures a fixed-size unit-stride loop. Broader ASV, Newton, and application workloads are measured separately to check runtime, code generation time, and fallback paths.

An isolated comparison of the shared constant-range lowering on an RTX 5080 found matching status for 57 runnable Newton CUDA examples. Rechecks of the apparent runtime outliers found no material performance change in those workloads, even though their generated CUDA source can differ.

## Tradeoffs

Specialization adds a second compiled kernel body for eligible code and can increase compilation time and object size. The eligibility and runtime guards limit that cost and preserve generic behavior. More aggressive helper inlining can also grow objects. LLVM's cost model and the selected CPU target determine the final vector instructions, so performance needs measurement on each target.

Both kernel bodies can be inlined into the same entry function. Adding the specialized body therefore also changes the generic fallback's machine code and instruction layout, including for launches that never take the specialized path. Some workloads can become slower even when they execute only the fallback.

On affected Skylake-derived Intel CPUs (Skylake through Comet Lake client processors, Skylake-SP and Cascade Lake Xeons), the microcode fix for the [JCC erratum](https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/best-practices/mitigation-strategies-jcc-microcode.html) prevents branches that cross or end on a 32-byte boundary from filling the decoded instruction cache. Tight loops can become sensitive to branch placement, so specialization's layout changes can cause speedups or slowdowns unrelated to vectorization.

An Intel Core i9-10900X investigation compared the loop-vectorization build and its main-branch baseline on these runtime-length row reductions:

```python
@wp.kernel(module="unique", enable_backward=False)
def row_dot(a: wp.array[float], b: wp.array[float], count: int, out: wp.array[float]):
    row = wp.tid()
    total = float(0.0)
    for j in range(count):
        i = row * count + j
        total += a[i] * b[i]
    out[row] = total
```

The `ragged_row_dot` variant shortens each row's reduction by the row index modulo four, while retaining `count` as the distance between row starts. The `reverse_stepped_row_dot` variant starts at the last element of each row and moves backward in steps of three while the element index within the row remains nonnegative. Both retain the same decorator, signature, accumulation, and output assignment; `count` is a runtime argument.

The benchmarks used 262,144 logical float32 elements per input and `262144 // count` output elements, with the counts and layouts in the table. The strided cases used separate backing arrays of twice those lengths and passed their `base[::2]` views for both inputs and the output. Recorded CPU launches used one task per row and `block_dim=1`; compilation and warmup preceded timing. The initial comparison collected 21 samples of 32 calls per process, with four fresh processes in base/branch/branch/base order pinned to one logical CPU.

The initial builds showed 17–18% slowdowns in all three reductions. A follow-up experiment added `-mllvm=-x86-branches-within-32B-boundaries` to both builds to mitigate the branch-layout penalty. The table reports timings in milliseconds; relative changes compare the loop-vectorization build with the main-branch baseline under the same alignment setting.

| Kernel | `count` | Array layout / runtime path | Base, no alignment | Branch, no alignment | Change | Base, aligned | Branch, aligned | Change |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `row_dot` | 64 | Contiguous / specialized | 0.484 | 0.567 | +17.0% | 0.484 | 0.435 | -10.1% |
| `ragged_row_dot` | 17 | Stride 2 / fallback | 0.465 | 0.546 | +17.2% | 0.455 | 0.465 | +2.2% |
| `reverse_stepped_row_dot` | 4 | Stride 2 / fallback | 0.413 | 0.486 | +17.7% | 0.377 | 0.374 | -0.9% |

The strided launches failed the contiguous-stride guard and executed only the generic fallback, confirming that the regression also affected launches that could not benefit from specialization. In controlled compiled-entry-point comparisons, disabling specialization restored baseline performance and byte-identical baseline machine code for all three kernels. Disabling loop vectorization or restoring the earlier inlining threshold retained the slowdown. Increased legacy-decoder activity and the alignment results supported a branch-layout explanation consistent with the JCC microcode penalty.

Alignment reduced the regressions to a 2.2% slowdown in the ragged case and speedups in the other two cases. The long-loop ASV benchmark retained its gain: about 0.694 ms with alignment and 0.693 ms without it on the loop-vectorization build, versus 5.44 ms on the main-branch baseline. However, the same flag slowed two AMD row-dot probes by 6.6% and 7.2%, so applying it globally would exchange one target's layout penalty for costs elsewhere.

We retained specialization and accepted the measured regressions in these short reductions on the i9-10900X as a tradeoff for exposing contiguous loops to vectorization. Disabling specialization would remove that optimized path, while the alignment workaround had CPU- and workload-dependent costs that did not justify enabling it by default. The decision preserved the vectorization gains without adding an architecture-specific workaround. These historical measurements document that tradeoff; they do not establish performance on every CPU or the final revision.
