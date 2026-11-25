# cuTile Foreign-Function Bridge

**Status**: Implemented (experimental)

**Issue**: [GH-2346](https://github.com/NVIDIA/warp/issues/2346)

## Motivation

CUDA Tile programs, written in cuTile Python, operate on whole tiles: there is
no thread index, and operations such as `ct.load` or `ct.mma` act on a tile as
one value. Many per-element operations that physics and graphics workloads need
have no tile-level equivalent: querying the signed distance to a mesh, casting
a ray against a BVH, sampling a sparse volume, or evaluating procedural noise.
Warp already implements all of these as SIMT device code behind its builtin
library, and users already write them as `@wp.func` functions.

The bridge lets a kernel written in cuTile Python call any user-defined
`@wp.func` with a supported signature as a foreign function, so that a tile program can mix tensor-core
work expressed in cuTile with per-element Warp code in one kernel. The
reference example trains a neural signed-distance field whose MLP runs as
cuTile matrix multiplies while the ground-truth labels come from a Warp mesh
query inside the same kernel.

The public entry point is `warp.experimental.cutile.export_func()`.

## Terminology

- **CUDA Tile**: the tile-based programming model.
- **Tile IR**: the intermediate representation CUDA Tile programs are compiled
  through. Its bytecode is exact-version-locked to the toolkit release.
- **cuTile Python** (`cuda-tile`): the Python frontend that traces
  `@ct.kernel` functions to Tile IR bytecode. "cuTile" is used as shorthand for
  it below. The `cuda-tile-preview` package carries the preview surfaces the
  bridge depends on, `Tilelib` and `foreign_call`.
- **`tileiras`**: the Tile IR assembler and linker from the CUDA Toolkit. It
  compiles a kernel's bytecode to a cubin, links foreign-function libraries
  into it, and also packs those libraries.
- **`.tilelib`**: the foreign-function library format that `tileiras` packs
  and links. It bundles an *interface* bytecode module (the extern
  declarations a kernel calls), an *implementation* bytecode module (the
  `extern_impl` entries binding each declaration to a linkable symbol, with
  per-operand fragment maps), and the LTOIR holding the device code.
- **Fragment map**: the implementation's description of how each operand tile
  is distributed over the kernel's worker threads, so the compiled function
  receives its slice by pointer.
- **Scope**: `"thread"` runs one independent invocation of the Warp function
  per tile element; `"block"` runs one cooperative invocation per block in
  which every worker thread participates.

## Requirements

| ID  | Requirement                                                                                          | Priority | Notes                                                    |
| --- | ---------------------------------------------------------------------------------------------------- | -------- | -------------------------------------------------------- |
| R1  | Export any concrete `@wp.func` with scalar, vector, matrix, quaternion, transform, struct, or array parameters (`float16`, `bfloat16`, and `bool` excluded as scalar operands and as vector or matrix element types) | Must     | Contract derived from the signature alone                |
| R2  | No per-builtin work: whatever the function body may call under `wp.launch` is available            | Must     | Geometry queries, volumes, noise, tile builtins at block scope |
| R3  | Compute what the same function computes under `wp.launch`                                           | Must     | Same module build options, same headers                  |
| R4  | Install with `pip install "warp-lang[cutile]"` next to PyTorch's CUDA 13 wheels                      | Must     | Compiler wheels, not the `cuda-toolkit` metapackage      |
| R5  | Thread scope and block-cooperative scope                                                             | Should   | block scope enables Warp tile builtins                     |
| R6  | Iterator and tiled geometry queries inside foreign functions                                        | Should   | Needs per-thread traversal stacks and shared memory      |
| R7  | Content-keyed caching that never replays stale device code                                           | Must     | cuTile caches kernels on the tilelib's `(path, version)` |
| R8  | Misconfiguration reported at declaration or trace time where possible                                | Should   | block-scope mismatches rejected at kernel compile time; thread-scope slab overruns trap, see below |

**Non-goals**: automatic differentiation across the call boundary (cuTile has
no autodiff to drive adjoints); tile-typed parameters (Warp tiles and cuTile
tiles never cross as objects); MathDx-accelerated tile builtins (they take
their portable code paths); API capture (`ct.launch()` is not a Warp API
call); CPU execution.

## Design

### Approach

A declaration runs four stages, all at the time `export_func()` is called:

1. **Standalone codegen.** Warp's code generator emits self-contained CUDA C++
   for the function and its transitive dependencies without any kernel, via a
   synthetic `ModuleBuilder`, the same pattern `@wp.func_grad` uses for return
   type inference. The module header is baked for a canonical block size, so
   the generated text, and with it the cache identity, does not vary with the
   `num_worker_warps` hint at thread scope. An `extern "C"` destination-passing
   wrapper is appended: one pointer parameter per operand, the Warp call in
   the middle, and the return value written through the output pointer.
2. **LTOIR compile.** Warp's embedded NVRTC compiles the source to raw LTOIR
   (`-rdc -dlto`, no nvJitLink step) for the target device's architecture.
3. **Bytecode emission.** Warp writes the Tile IR interface and implementation
   modules itself (`warp/_src/cutile_bytecode.py`). The toolkit ships no tool
   that assembles these modules from text, so the writer follows the format of
   cuTile Python's own bytecode writer, restricted to the zero-function
   modules the bridge needs: extern declarations, `extern_impl` entries with
   fragment maps, and the scratch and synchronization attributes.
4. **Packing.** `tileiras --preview-create-tile-lib` packs the two modules
   and the LTOIR into a `.tilelib` in the `cutile/` directory of the Warp
   kernel cache. The returned `TilelibForeignFunction` holds a
   `cuda.tile_preview.Tilelib` handle (path plus content-digest version) and a
   generated, traced call body that kernels invoke directly.

### Calling convention

The contract is derived from the function's signature. Each Warp parameter
becomes one call-site argument, and the call returns one SSA result.

- Scalars are rank-1 tiles of shape `(L,)`, one value per invocation, where
  `L` is the call site's tile length.
- Vectors, matrices, quaternions, and transforms are flat rank-1 tiles of
  shape `(L * width,)`, with `width` the element count padded to a power of
  two (`wp.vec3` pads to 4). Tile IR tile dimensions are powers of two, and a
  flat tile keeps the per-thread slice contiguous.
- Structs travel as flat rank-1 `int32` tiles carrying the struct bytes
  (`encode_struct()`), usually one shared tile.
- Arrays are always shared and take a tuple of `1 + ndim` plain values: the
  data pointer followed by one extent per dimension (`array_pointer_args()`).
  Only contiguous device arrays are accepted; the pointer travels as `int64`
  because Tile IR tiles have no unsigned dtypes.

**Shared arguments.** Whether an argument is shared by every invocation or
per-invocation is decided at the call site, from the argument alone: a plain
Python number, a rank-0 tile (a runtime kernel scalar parameter), or one
`(width,)` tile is shared; a `(L * width,)` tile is per-invocation; arrays are
always shared. The two forms cannot alias, since a shared operand has at most
`width` elements and a per-invocation one at least `32 * width`. The
declaration itself has one layout, every operand per-invocation, and the
traced call body (`ct.static_eval` classification, then `ct.full` or
`ct.broadcast_to` under tracer-folded branches) expands each shared value to
an `(L * width,)` tile before the foreign call. `tileiras` folds that
broadcast into the same machine code a stride-0 fragment map would produce
for operands of up to 64 `int32` words (byte-identical SASS, registers, and
shared memory in every case measured); see Known limitations for wider
operands. One build therefore serves every call pattern, and `L` for a call
whose arguments are all shared is `32 * num_worker_warps`.

Multi-dimensional tiles are rejected on purpose. Their linearization in
fragment-map element space is implementation-defined, so the contract is
rank-1 tiles only, and the traced call body checks ranks and shapes at trace
time with messages naming the Warp parameter.

**Declared lengths.** Thread-scope code is per-invocation and does not depend
on the tile length, so one LTOIR backs a declaration for every power-of-two
length from 32 to 1024, and the traced call body dispatches on the length of
the first per-invocation operand (a width-1 operand when there is one, since a
scalar shape cannot alias another declared length times a padding width).
Block scope pins the declaration to the single `num_worker_warps` value, as
the block size is baked into the cooperative code. The hint is always among
the declared lengths and is the length of a call whose arguments are all
shared.

**Memory effect.** Foreign calls carry a memory effect on this preview
surface, so `tileiras` never eliminates a call whose result is unused. A
function called only for its side effects still needs a return value, but the
result need not be consumed.

### Shared memory and scratch

A Tile IR kernel never allocates Warp's static or dynamic shared memory, so
everything a foreign function needs in shared memory comes from *scratch*: a
block of shared memory the kernel provides to the call, sized by the
implementation's declared `scratch` attribute. The wrapper receives it as one
trailing pointer and binds it before calling the function. The layout is

```
[ BVH query stack slab | mesh query stack slab | per-thread offset slots | tile storage ]
```

with each region present only when the function uses it. The scratch pointer
arrives in generic address space; the implementation's `frontend_abi` map
tells `tileiras` to pass the shared-memory pointer as a generic one.

**Tile storage.** When the function's transitive forward pass needs Warp tile
shared memory (`get_total_required_shared() > 0`), the source is compiled with
`WP_TILE_EXTERNAL_SHARED_STORAGE`. In that mode `tile_shared_storage_t` keeps
only a single static pointer slot; the per-thread allocation offsets and the
dynamic area live in the scratch. `bind_external_smem()` publishes the base
pointer and resets the offsets. It needs two barriers: the first so that no
thread still draining a previous call's tile operations observes the
republished base pointer mid-flight (callers also emit an exit barrier after
each call, so this is defense in depth), and the second so that every thread
sees thread 0's store before the offsets are reset. The launching kernel's
block must be exactly `WP_TILE_BLOCK_DIM x 1 x 1` threads: extra threads
would index past the offset slots, and multi-dimensional blocks alias them.
`WP_TILE_BLOCK_DIM` must also be a multiple of 4 so the dynamic area keeps the
16-byte alignment tile allocations assume. Tile storage is therefore accepted
only at block scope, where the block size is pinned.

**Traversal stacks.** Iterator queries (`wp.mesh_query_aabb`,
`wp.mesh_query_sphere`, `wp.bvh_query_*`) keep a per-thread traversal stack. In
a Warp kernel it lives in static shared memory sized for the kernel's block.
A foreign function does not know the calling kernel's block size, so with
`BVH_SCRATCH_STACK` the stacks come from scratch slabs, strided by block size
at runtime: one slab for BVH queries and one for mesh queries, each
`BVH_QUERY_STACK_SIZE * sizeof(int) * slab_threads` bytes. Two slabs mirror
the two static slabs of the shared-stack path, so a mesh query nested inside a
BVH query (or vice versa) never overwrites the other's live state. A
thread-scope declaration sizes the slabs for `max(8, num_worker_warps)` warps,
so the common 8-warp block needs no hint and a larger block is declared
through `num_worker_warps`; block scope sizes them to the pinned block
exactly. The cost is 128 bytes per worker thread per query family, so a
declaration whose scratch exceeds the device's per-block shared memory is
rejected at declaration with the numbers.

The slab bind in `bvh_bind_scratch_stacks()` deliberately uses **no barrier**,
unlike `bind_external_smem()`. Every thread performs the store itself and only
reads the pointer afterwards, so each read is ordered after that thread's own
store by program order, and every store carries the identical value, so no
thread can observe anything else. A barrier would also be wrong there:
thread-scope foreign functions run one invocation per tile element with no
convergence guarantee, so they must never synchronize, and the block-scope
wrapper already brackets its calls with block barriers.

A kernel may call several foreign functions back to back, and a wrapper may
start rebinding while a slower warp is still inside the previous call.
Rebinding a family to another valid slab is harmless as long as the kernel
hands consecutive calls scratch regions that are identical or disjoint, never
partially overlapping: every thread uses only its own column of whichever slab
is bound, and all slabs are sized for the block. Binding a null pointer is
not harmless, so a wrapper leaves a family it does not use bound as it was.

**Tiled queries.** The block-cooperative tiled queries (`wp.tile_mesh_query_*`,
`wp.tile_bvh_query_*`) traverse with the whole block and keep block-sized
result buffers in static shared memory. They are accepted only at block scope,
and that memory is in addition to what cuTile plans for the kernel.

**Known gap.** Thread scope rejects a function at declaration when its module
requests tile shared memory or when it calls a tiled query builtin. A builtin
that synchronizes through a private `__shared__` slot without touching the
tile shared-storage allocator passes both checks; `wp.tile_from_thread()` is
one. Calling it at thread scope is undefined behavior, as the guide states for
every block barrier, but it is not detected. A compile-time check was
prototyped (`WP_TILE_SYNC` expands to a dependent `static_assert` in
thread-scope builds, so any barrier fails NVRTC compilation; all tests pass
with it) and deferred: it touches `tile.h` and the error reaches the user as a
raw NVRTC build failure until the message is surfaced properly.

### Block-size validation

Two configuration errors depend on the calling kernel's block size, which a
declaration cannot know:

- a block-scope function called from a kernel whose worker-warp count differs
  from the declared `num_worker_warps`, which the cooperative code assumes;
- a thread-scope function with iterator queries called from a block larger
  than its slabs were sized for, which would stride past the slabs.

**A block-scope mismatch is rejected at kernel compile time.** The
implementation's fragment map declares `32 * num_worker_warps` participants, and
`tileiras` 13.5 requires a `cta`-synchronized callee (Tile IR's name for block
scope) to have a participant count equal to the kernel's
`32 * num_worker_warps`. A mismatch fails the kernel's compilation with a
`CompilerExecutionError` before anything launches, and the CUDA context
is unaffected. The release `tileiras` reports only "failed to compile Tile IR
program", so the user guide's troubleshooting section explains the cause. No
runtime check is emitted for this case.

**Thread-scope slab overruns trap.** The wrapper compares `blockDim` against
`slab_threads` and executes `__trap()` on a larger block, so the overrun
fails loudly instead of corrupting shared memory. The check is on the
compiled block, which can exceed `32 * num_worker_warps`: cuTile documents
that the compiler may add warps (for asynchronous memory transfers) that
`num_worker_warps` does not count, and exposes no way to read the final
block size before launch. The user guide therefore states the worker-warp
count as a lower bound and the trap as a known limitation. A trap is a
sticky device fault: the launch fails with a CUDA error, the context is
unusable afterwards, and the process must restart. Reporting this as a recoverable
error was requested in review; the preview API prevents each of the direct
routes:

- *Trace time.* The generated call body runs while cuTile traces the kernel,
  but compiler options, including `num_worker_warps`, are not passed to the
  tracer, and the default `None` leaves the count to `tileiras`.
- *Link time.* The `extern_impl` entry carries access, scratch size and
  alignment, architecture, ABI map, and synchronization. It has no field for
  a required or maximum block size that `tileiras` could check.
- *Scratch sized from the block.* The scratch size is a constant integer
  expression in the 1.8.1 bytecode encoding; no symbolic form exists.

The authoritative block size is the `EIATTR_REQNTID` attribute `tileiras`
writes into every kernel cubin. The driver enforces it, rejecting an explicit
launch with any other block as `CUDA_ERROR_INVALID_VALUE` without harming
the context, so the only mismatch that can reach the wrapper is between a
Warp declaration and that attribute.

Alternatives considered for the thread-scope case, none adopted in the
initial version:

- *Thread-local traversal stacks.* `bvh.h` already supports per-thread local
  stacks (the CPU path), which need no slab and no block-size assumption.
  Declined because it would put the foreign path on a different traversal
  stack model from Warp's GPU kernels, a divergence maintainers would have
  to carry.
- *Slabs indexed by invocation.* The call body could pass a hidden
  `ct.arange(L)` operand so each invocation learns its index; the fragment
  map guarantees invocation *i* receives element *i*, and a probe confirmed
  this across block and length combinations. The slab would then have `L`
  columns, sized per declared length, and the block size would drop out.
  Declined for now because it adds a hidden operand to every call and rests
  on a fragment-map guarantee most readers would not know to check.
- *Pre-launch validation from the cubin.* `kernel._compile` is a Python
  method cuTile's dispatcher calls by name; a Warp hook could request the
  final IR, collect the `PreviewForeignCall` operations, read
  `EIATTR_REQNTID` from the cubin, and raise a `ValueError` naming the
  function and both sizes. A proof of concept works. Declined for now
  because it wraps two private cuTile surfaces and must either be an opt-in
  decorator or patch cuTile's kernel class on import.
- *Slabs sized for the largest block.* 1024 threads need 128 KiB per query
  family, above the per-block shared-memory limit on current GPUs.
- *A device-side error flag instead of the trap.* The wrapper has no
  Warp-launched context to receive a flag address, cuTile's library
  lifecycle hook that would expose the loaded cubin is hard-coded to NVSHMEM
  symbol prefixes, and the error would arrive after the kernel produced
  wrong results.

The clean fix is upstream: a required-block-size attribute on the
`extern_impl` entry that `tileiras` validates at link time, or a tracer- or
load-time API through which a linked library can learn the resolved block
size. Until then the trap remains the memory-safety backstop for thread-scope
iterator queries.

### Build options and determinism

The function is compiled with its own Warp module's resolved options so that
it computes what the same `@wp.func` computes under `wp.launch`: `mode`,
`optimization_level`, `fast_math`, `fuse_fp`, `lineinfo`, the
`extra_build_options` preamble, include directories, and build dependencies,
plus the global `warp.config.verify_fp`. Two options do not carry over fully:

- `fuse_fp=False` is forwarded to NVRTC but cannot be guaranteed, because
  floating-point contraction is decided when `tileiras` link-time-optimizes
  the calling kernel, outside Warp's control. A warning is emitted once.
- Deterministic mode is disabled for the foreign build. Warp's deterministic
  codegen injects hidden context parameters between the regular arguments and
  the multi-return references; the foreign wrapper has no way to supply them,
  and the host kernel is not Warp's in any case.

MathDx-accelerated tile builtins are compiled with no target architecture set,
so they take their portable implementations and need no separately linked
device libraries.

Query and tile-builtin detection scans the generated source for the builtin
names. A `wp.func_native` helper whose body lives in an included header can
therefore reach a stack-based query or a block-cooperative builtin without the
bridge planning scratch for it; such calls must stay in the snippet text or in
the Warp function itself.

### Caching

cuTile caches compiled kernels on the tilelib's `(path, version)` alone, so
every input that shapes the packed tilelib must perturb the version string.
The bridge uses a content digest as that version, folding in the generated
source and wrapper (emitted against placeholder names so the digest can be
computed before the digest-derived symbol exists), the bytecode modules, the
LTOIR compile options and the embedded NVRTC identity, a hash of the native
headers, the `tileiras` path and version banner, the target architecture, the
declared lengths, the scope, and the scratch size. A schema version constant
is bumped whenever the emitted contract or packing layout changes.

Artifacts are named by that digest and published atomically, so concurrent
declarations in several processes converge on one file and a crash never
leaves a partial tilelib at the trusted path. The generated CUDA C++ is kept
beside its compiled code for inspection. Two caveats follow from caching the
header and toolchain fingerprints per process: editing native headers or
swapping a toolchain in place mid-session requires a restart, and the digested
source embeds the defining file's location, so moving a file rebuilds its
artifacts.

The generated call body's source file is also named by content, because cuTile
embeds the filename in the traced kernel's bytecode as a location; a
per-process name would defeat cuTile's own cross-process kernel cache.

### Toolchain discovery and packaging

`tileiras` is resolved from `WARP_TILEIR_TOOLCHAIN` if set (a configured
toolchain that resolves nothing raises rather than falling back, so a typo
never silently swaps toolchains), else from the `nvidia.cu13` pip namespace
package, else from `PATH`. The release must be exactly 13.5: the bridge
writes Tile IR 13.5 SIMT interoperability bytecode, and the
[Tile IR compatibility policy](https://docs.nvidia.com/cuda/tile-ir/13.5/sections/prog_model.html#compatibility)
states that preview bytecode is not portable to 13.6 or any later release,
that breaking changes may land in any release, and that the preview may end
as early as 13.7. `_check_tileiras_release` therefore rejects both older and
newer releases up front, with a message naming the required one, instead of
letting the mismatch surface as a packing or link error. Moving to a later
toolkit means updating `cutile_bytecode.py` to that release's format and
bumping `BYTECODE_VERSION`, which the check reads.

Two compilers are involved and must agree. Warp runs `tileiras` once to pack
the `.tilelib`; cuTile Python runs its own `tileiras` to compile the calling
kernel and link the `.tilelib` into it, found through its own lookup (pip
package, then `PATH`, then `CUDA_HOME`, then default toolkit paths).
`WARP_TILEIR_TOOLCHAIN` changes only Warp's packer, so it cannot repair a
mismatch on its own; it exists to point the packer at a specific toolkit
install (a local build, a toolkit outside `PATH`) when cuTile already
resolves the same release. The `cutile` extra pins `nvidia-cuda-tileiras` to
`>=13.5,<13.6`, and both lookups check the pip package first, so the
standard install gives both compilers one binary. A manual setup puts the
13.5 toolkit's `bin` first on `PATH`, which both lookups consult next.

The `cutile` extra lists the compiler wheels directly: `nvidia-cuda-tileiras`,
`nvidia-cuda-nvcc`, `nvidia-nvvm`, `nvidia-nvjitlink`, and `nvidia-cuda-crt`,
rather than `cuda-tile[tileiras]`. That extra pulls the `cuda-toolkit`
metapackage, which pins the CUDA runtime and NVRTC wheels at the toolkit
release and cannot coexist with PyTorch's CUDA 13 wheels, which pin them at
13.0. The compiler wheels leave the runtime unpinned, so one environment can
hold PyTorch's 13.0 runtime and the 13.5 compilers.

The `nvidia.cu13` namespace can span several site directories (a
`uv run --with` overlay in front of the project environment, a user site next
to a system site). Warp's lookup searches every `__path__` entry. cuTile
Python 1.8.1 checks only the first and falls back silently to a system
`tileiras`
([nvidia/cutile-python#131](https://github.com/NVIDIA/cutile-python/issues/131)),
which is why CI locks JAX and RMM into the `tests-cu13` dependency group
instead of adding them with `--with`: a single `nvidia/cu13` tree keeps
cuTile's own lookup working.

### Alternatives considered

- **Per-builtin wrappers.** Hand-writing a foreign function for each Warp
  builtin a tile program might need. Rejected: the set is open-ended, users'
  own `@wp.func` code would be excluded, and deriving the contract from a
  signature covers both with one mechanism.
- **Assembling the bytecode modules from text.** A `cuda-tile-translate`
  style assembler would have avoided writing bytecode directly. None ships in
  the toolkit, so the bridge carries a minimal writer validated against
  cuTile Python's own.
- **Multi-dimensional operand tiles.** Would let callers pass `(L, 3)` tiles
  for `wp.vec3`. Rejected because the fragment-map linearization of rank-2
  tiles is implementation-defined; flat rank-1 tiles with power-of-two padding
  are unambiguous.
- **Declaring shared parameters at export time.** An earlier revision took a
  `uniform=` set naming the parameters broadcast to every invocation and gave
  them stride-0 fragment maps. Rejected in review: the call site already shows
  which arguments are single values, arrays are always shared, so the set made
  callers say it twice.
- **Per-pattern stride-0 builds packed at trace time.** Keeps the zero-copy
  broadcast for every width by inferring the shared set when the kernel is
  traced and packing (or fetching) the `.tilelib` for that pattern then.
  Prototyped and measured at parity with the stride-0 path, including wide
  operands, but compilation and its errors move from `export_func()` into the
  kernel trace, a `tileiras` subprocess runs inside cuTile's tracer, and the
  handle carries a lock and per-pattern state. Rejected in favor of call-site
  expansion, whose only cost is the wide-operand case below.
- **A barrier in the BVH slab bind.** Rejected as unnecessary (each thread
  stores and reads its own copy of the same value) and incorrect at thread
  scope, where invocations have no convergence guarantee.
- **Static traversal stacks sized at declaration.** Would tie the generated
  code to one block size and so to one tile length, losing the universal
  thread-scope declaration. Scratch slabs strided at runtime keep one LTOIR
  for all lengths.
- **Running the cuTile tests in dedicated CI jobs.** The first CI layout,
  forced by the `cuda-toolkit` conflict with PyTorch. Superseded by the
  compiler-wheel extra, which lets the existing CUDA 13 jobs test the same
  extra users install.

### Known limitations

- **One architecture per declaration.** A `export_func()` result is compiled
  for the architecture of the device it was declared on (the `device`
  argument, defaulting to the current device). Warp kernels compile lazily
  per device; a foreign declaration does not, so a machine with two GPU
  generations needs one declaration per architecture. The implementation
  module constrains the LTOIR to its architecture, so a mismatch fails at
  link resolution rather than deep inside the device link. Lazy
  per-architecture compilation behind one handle is possible future work.
- **Wide shared operands.** Call-site expansion relies on `tileiras` folding
  the broadcast. Measured on sm_120 with CUDA 13.5 cuTile 1.8.1: operands of
  up to 64 `int32` words fold in every case (scalars, `vec3`, `mat44`, `mat8x8`,
  structs of 16 `vec4`); a 128-word struct at tile length 128 is staged through
  a static shared-memory allocation and runs 6.6x slower (5.05 vs 33.4 us),
  while the same operand at length 256 folds again; a 256-word operand
  (`mat12x12`) is staged at lengths 128 and 256 (9 to 12x). The threshold is a
  `tileiras` layout decision that is not monotonic in tile length. Operands of
  256 words and more are already slow in every variant because the bridge
  marshals structs by value (next item), so the regime where expansion alone
  costs is roughly 128-word shared operands at small tile lengths. The user
  guide states the 64-word rule and the remedy (smaller shared structs, or an
  array).
- **Wide struct operands are marshalled by value.** The wrapper copies a
  struct operand's bytes into a by-value local before the call. LLVM
  scalarizes that copy for structs of up to 64 words; at 256 words it falls
  back to a local-memory copy (3 to 5 KB of stack per thread) and such calls
  run roughly three orders of magnitude slower than 64-word ones in every
  variant measured. Follow-up: pass large structs by reference.
- **Forward only.** Explicit `wp.grad()` calls inside the function work, since
  they are part of its forward computation, but there is no adjoint of the
  foreign call.
- **Thread-scope slab overruns trap.** A block larger than a thread-scope
  declaration's iterator-query slabs is a sticky device fault; see
  [Block-size validation](#block-size-validation).
- **Preview surfaces.** `Tilelib`, `foreign_call`, the bytecode format, and
  `tileiras --preview-create-tile-lib` may change between CUDA Toolkit and
  cuTile Python releases.

## Testing Strategy

`warp/tests/interop/test_cutile.py` has three layers:

- **Helpers** (`TestCutileHelpers`): struct and array operand encoding, the
  public module surface, the traversal-stack depth constant matching
  `bvh.h`, tilelib packing failure reporting, and the `tileiras` lookup and
  release check with mocked filesystems. These need no cuTile installation or
  GPU, except the CUDA array encoding test, which runs on every CUDA test
  device.
- **Declaration** (`TestCutileDeclaration`): every validation error
  `export_func()` raises before any toolchain or GPU work, needing only
  `cuda.tile` importable.
- **Kernels** (`TestCutileKernels`): end-to-end calls from cuTile kernels, one
  test per documented call form or contract, registered with
  `add_function_test()` over `get_cuda_test_devices()` so they run on every
  CUDA test device. Each test declares on the device it is given and launches
  on that device's stream. They skip, visibly, when `cuda.tile_preview` or a
  supported `tileiras` is missing.

The neural SDF example is registered in `warp/tests/test_examples.py` under
`TestInteropExamples` with a `cutile_required` skip key built on the same
checks, so it appears in the suite's junit report and coverage.

The toolchain is present only in the CUDA 13 CI jobs, which install the
`cutile` extra alongside `torch-cu13` and the `tests-cu13` group; elsewhere the
kernel tests and the example skip. Multi-GPU and mixed-architecture coverage
depends on the CI runners' hardware.
