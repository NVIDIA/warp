# CPU SLP Vectorization

**Status**: Implemented

**Issue**: [GH-1954](https://github.com/NVIDIA/warp/issues/1954)

## Motivation

Warp calls Clang's frontend directly to compile CPU kernels. This bypasses the Clang driver step that normally enables LLVM's SLP vectorizer at `-O2` and `-O3`. SLP can pack independent scalar operations, such as arithmetic on `vec4` components, into SIMD instructions.

## Design

`warp/native/clang/clang.cpp` enables `CodeGenOptions::VectorizeSLP` for CPU compilation at effective optimization level 2 or 3. By default, SLP remains disabled at lower CPU optimization levels; CUDA compilation is unchanged. A module can disable this pass with `-fno-slp-vectorize` in `cpu_compiler_flags`. Warp handles that driver-style flag because it invokes the frontend directly.

The default CPU flags set `-mllvm=-slp-recursion-max-depth=4` instead of LLVM's default 12. SLP at depth 4 forms four-lane `vec4` arithmetic on Jetson Thor, nearly doubling throughput relative to depth 3, which forms only two-lane groups. Depth 4 also leaves room for future code generation to expose more profitable four-lane trees. The SLP profitability threshold remains at LLVM's default.

The choice has measured costs. Six small-radius Mesh/BVH cases on Thor lose roughly 7–12% throughput at depth 4 relative to depth 2; all six remain within 1% at depth 3. Across 53 Benchy cases, depth 4's equal-case geometric mean is +0.60% versus depth 2, with a worst loss of 4.55% on tet streamlines. On the AMD Ryzen 7 9800X3D, depth 4 was closer than depth 3 to depth-2 XPBD and Teleop timings in the direct screen, though those depth-3 application results were not independently repeated.

LLVM's default depth 12 offers no consistent selected Thor application advantage over depth 4. In selected Thor H1, softbody, and Teleop builds, cold compilation cost rose roughly 5% at depth 4 versus 12–21% at depth 12, each relative to depth 2. This is a selected compile-cost observation, not a guarantee across workloads. The hidden recursion-depth option also affects load/store grouping and reduction discovery.

## Validation and tradeoffs

CPU correctness tests cover `vec4` arithmetic, its adjoint, strided access, and the opt-out; an optimization-record test checks that the SLP pass runs at CPU optimization level 2. Both depth sweeps held `-march=native` and the 9000/3000 inlining thresholds fixed. On Thor, depth 4 delivered about +181% `vec4` throughput, +15.8% softbody throughput, +27.3% H1 IK throughput, and +10.9% Teleop VBD throughput relative to depth 2, while selected small-radius queries slowed. On the AMD host, a custom two-computed-operand `vec4` kernel took about 0.405x depth-2 time at depth 4; the original ASV `vec4` case was unchanged. Tested numerical outputs agreed within the documented scopes. These are within-device depth comparisons, not main-versus-branch speedups.
