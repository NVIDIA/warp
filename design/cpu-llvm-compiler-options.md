# LLVM Options for CPU Kernel Compilation

**Status**: Implemented

**Issue**: [GH-2004](https://github.com/NVIDIA/warp/issues/2004)

## Motivation

Warp passes `warp.config.cpu_compiler_flags` and the per-module
`cpu_compiler_flags` option to its embedded Clang frontend. It invokes a
frontend action directly, so Clang does not parse `-mllvm` arguments for Warp.
This feature forwards those arguments to LLVM before CPU code generation.

LLVM options are process-wide. Parsing and resetting them needs coordination
with other compilations that use the same LLVM process
([LLVM command-line guide](https://llvm.org/docs/CommandLine.html),
[Clang backend source](https://github.com/llvm/llvm-project/blob/llvmorg-22.1.8/clang/lib/CodeGen/BackendUtil.cpp)).

## Scope

| ID | Requirement |
| --- | --- |
| R1 | Accept `-mllvm <argument>` and `-mllvm=<argument>` in global and per-module CPU flags. Parse the extracted arguments with LLVM in their original order. |
| R2 | Keep `cpu_compiler_flags=None` equivalent to `-march=native` with LLVM defaults. Put any future Warp tuning defaults in the configuration string, where a per-module string can replace them. |
| R3 | Coordinate Warp's temporary LLVM option parsing with CPU and LLVM-CUDA compilation. Compilations requesting the same ordered `-mllvm` arguments can run in parallel; different requests wait for one another. |
| R4 | Keep CPU objects for distinct resolved flag strings in distinct cache entries. Return ordinary syntax and LLVM parser errors as build failures. |
| R5 | Document that this is a power-user facility for straightforward optimization and code-generation options. Users must check that options have the intended effect and accept known and unknown LLVM state and pipeline caveats. |

Warp does not emulate the Clang driver or validate whether individual LLVM
options work with Warp's frontend and backend pipeline. Pipeline controls can
prevent object emission. Other options that select reports, register callbacks,
or read external files can behave differently from ordinary tuning options.
Users are responsible for checking the effect of their flags and for
option-specific failures; Warp does not manage LLVM help/version callbacks
that terminate Python or failures from Clang flags such as `-ftime-report`.
The flag string does not support arguments containing whitespace or LLVM
`@response` files.

## Design

The CPU compiler extracts both `-mllvm` forms from the whitespace-delimited
flag string and passes the remaining flags to the existing Clang frontend.
LLVM parses the extracted arguments before PCH generation and object emission.
For example, `-march=native -mllvm -inline-threshold=1000` and
`-march=native -mllvm=-inline-threshold=1000` express the same request. A
per-module flag string replaces the global string, so it must include
`-march=native` if host CPU detection is wanted.

The current `None` default supplies `-march=native` and leaves LLVM's defaults
untouched. Future Warp tuning belongs in a configuration default such as
`"-march=native -mllvm -example-option=value"`, rather than a native compiler
override. A per-module string omitting that argument selects LLVM's original
default. Option availability and behavior depend on the bundled LLVM version.

The native lease coordinator keys requests by the ordered LLVM argument vector;
ordinary Clang flags do not affect the key. The empty vector requests the
default LLVM option state. The first nonempty lease parses its arguments while
holding the coordinator mutex. Matching requests then share that state and
compile concurrently. A different request waits until the last active lease
releases; that release resets the parser options it can restore, including
occurrence counts and positions. Ordinary parser failures reset partial state.
Warp can reject a nonempty request if LLVM options were already set outside
this path.
CPU compilation holds its lease through PCH generation, retry, and object
emission.
The LLVM-CUDA path uses an empty-key lease, so it waits while a nonempty CPU
request is active. JIT object loading, lookup, and unloading use their own
mutex but do not wait for CPU option leases; they may observe temporary LLVM
option values. Different spellings of equivalent options may serialize; a
continuing stream of matching requests may delay a different request.

Resetting LLVM command-line options does not reverse every effect of using
them. LLVM or Clang can copy a value into another global, run a callback, or
latch it on first use. For example, `-regalloc` is latched on the first CPU
code-generation use: a later setting may be ignored, while an earlier setting
may affect later modules even after the option is reset. Those modules keep
their usual cache keys, and Warp writes new binaries to the kernel cache even
when `cache_kernels` is `False`, so affected binaries can outlive the process.
Users should run experiments with options that have lasting effects in a fresh
process with its own kernel cache directory. Warp rejects the Clang frontend
flags `-mdebug-pass` and `-mlimit-float-precision` because they parse LLVM
options outside the coordinator.

Resolved CPU flags contribute to the module hash, and the native PCH key uses
the ordered flag tokens. Different `-mllvm` strings therefore produce distinct
CPU objects and can produce distinct PCH files. A change to the configuration
default enters those same hashes. Options that read a file are keyed by its
path in the flag string, not by the file contents; callers can list such files
in `wp.ModuleBuildOptions(extra_build_dependencies=[...])` when needed.
Compiler setting changes are expected to be rare, so serializing PCH rebuilds
for different LLVM option requests is acceptable.

## Verification

- Check both argument forms with one observable option effect, followed by a
  default build, and confirm distinct flag strings produce distinct module hashes.
- Check that an invalid LLVM argument fails without poisoning a later build.
- Smoke-test matching and differing LLVM arguments in concurrent CPU builds,
  without timing assertions.
