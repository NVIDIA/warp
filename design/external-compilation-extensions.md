# External Compilation Extension API

**Status**: In Progress

**Issue**: [GH-1575](https://github.com/NVIDIA/warp/issues/1575)

## Motivation

Packages built on Warp sometimes need to make external C++ declarations visible to generated
kernels, call external functions, exchange library-defined value types, and load the generated
code in another runtime. Today that requires patching Warp or importing from `warp._src`.

OptiX is the first consumer. An OptiX addon needs its headers available during NVRTC
compilation, callable wrappers around OptiX device functions, OptiX-compatible program names
and launch parameters, and the path to the generated PTX. None of those needs are
OptiX-specific, so the API should serve other C++/CUDA libraries without adding
library-specific concepts to Warp.

Warp keeps ownership of source generation, hashing, compilation, and artifact naming. An addon
only supplies the extra declarations and ABI contracts its own module needs. This is not a
general C++ build system.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1  | Add CPU and CUDA include directories and source preambles per module | Must | `wp.ModuleBuildOptions` |
| R2  | Recompile when an explicitly listed external dependency changes | Must | `extra_build_dependencies`, content-hashed |
| R3  | Compose build inputs from multiple addons without mutation | Must | `ModuleBuildOptions.merged()` |
| R4  | Register an external C++ function through a small public API | Must | `wp.build_experimental.add_builtin()` |
| R5  | Register opaque and field-described external C++ value types | Must | `wp.build_experimental.add_native_type()` |
| R6  | Validate host/native ABI assumptions during compilation | Must | Generated `static_assert`s |
| R7  | Support selectable external entry-point ABIs | Must | `@wp.kernel(entry_point_abi=...)` |
| R8  | Return AOT artifact paths instead of requiring cache-name reconstruction | Must | `wp.compile_aot_module()` |
| R9  | Include all Warp-visible extension contracts and explicitly declared build inputs in module hashing | Must | Module options, external builtin identities, `nt<schema>` type codes |
| R10 | Preserve normal JIT compilation and `wp.launch()` behavior by default | Must | Defaults unchanged |
| R11 | Reject conflicts early and make equivalent registration idempotent | Must | Registration is process-global. Reloaded classes are not yet interchangeable |
| R12 | Keep experimental external-compilation contracts narrowly scoped | Should | |

**Non-goals:**

- Compiling or linking arbitrary external translation units or libraries. External
  implementation code must be visible through an included header.
- Sandboxing external code. Addon headers and preambles are trusted native build inputs.
- Exposing the full internal builtin-registration API.
- Inferring arithmetic, methods, gradients, or resource ownership for native value types.
- Launching non-Warp entry-point ABIs through `wp.launch()`.

## Design

### Approach

The complete public surface added by this design:

```python
# --- Module-scoped compiler inputs -------------------------------------------------
class wp.ModuleBuildOptions:
    def __init__(
        self,
        *,
        extra_cuda_include_dirs: Sequence[str | os.PathLike[str]] | None = None,
        extra_cpu_include_dirs: Sequence[str | os.PathLike[str]] | None = None,
        extra_cuda_preamble: str = "",
        extra_cpu_preamble: str = "",
        extra_build_dependencies: Sequence[str | os.PathLike[str]] | None = None,
    ) -> None: ...

    # Non-mutating compose: paths keep their first occurrence, preambles concatenate.
    def merged(self, *others: ModuleBuildOptions) -> ModuleBuildOptions: ...

# Applied through the existing module-option API under the new key "extra_build_options".
# `module` now also accepts a Module object or a module name, not just a Python module.
wp.set_module_options(options: dict[str, Any], module: Any = None) -> None
wp.get_module_options(module: Any = None) -> dict[str, Any]

# --- External C++ functions and value types (process-global registries) ------------
wp.build_experimental.add_builtin(
    name: str,
    input_types: Mapping[str, type] | None = None,
    value_type: type | None = None,
    *,
    native_name: str | None = None,             # defaults to f"wp::{name}"
    doc: str = "",
) -> wp.Function

wp.build_experimental.add_native_type(
    ctype: type[ctypes.Structure],
    *,
    native_name: str,
    fields: Mapping[str, type] | None = None,   # None -> opaque
    initializer: str | None = None,             # None | "aggregate"
) -> type[ctypes.Structure]                     # returns ctype, used as annotation/dtype

# --- Exported kernels, with only the new parameter shown ----------------------------
@wp.kernel(
    entry_point_abi: Literal["warp", "external_constant_params"] | None = None,
)

# --- AOT: returns the artifacts it wrote, in target order (previously None) ---------
wp.compile_aot_module(
    module: Module | types.ModuleType | str,
    device: Device | str | list[Device] | list[str] | None = None,
    arch: int | Iterable[int] | None = None,
    module_dir: str | os.PathLike | None = None,
    use_ptx: bool | None = None,
    strip_hash: bool | None = None,
) -> list[pathlib.Path]
```

Builtin and native type registration is process-global, because registered names participate
in Warp's language and overload sets. Build inputs are module-scoped, so unrelated kernels do
not inherit another addon's headers, dependencies, or compiler state.

An addon normally registers its language elements at import time, then applies its build
options to each module that uses them:

```python
class Color(ctypes.Structure):
    _fields_ = [
        ("r", ctypes.c_float),
        ("g", ctypes.c_float),
        ("b", ctypes.c_float),
    ]


wp.build_experimental.add_native_type(
    Color,
    native_name="my_addon::Color",
    fields={"r": wp.float32, "g": wp.float32, "b": wp.float32},
    initializer="aggregate",
)
wp.build_experimental.add_builtin(
    "my_addon_scale_color",
    {"value": Color, "factor": wp.float32},
    Color,
    native_name="my_addon::scale_color",
)

addon_options = wp.ModuleBuildOptions(
    extra_cuda_include_dirs=[include_dir],
    extra_cuda_preamble='#include "my_addon.h"',
    extra_build_dependencies=[header_path],
)
```

The addon then merges `addon_options` into each consuming module's `extra_build_options`
via `wp.set_module_options()`, before that module's first JIT launch or AOT compilation.

### Key Implementation Details

#### Module build inputs

`ModuleBuildOptions` carries separate CPU and CUDA include-directory lists, separate CPU and
CUDA preambles, and a list of files whose contents affect the build.

A preamble is inserted after Warp's native headers, but before codegen-only cast macros and the
generated code. External headers can therefore use public Warp macros such as `CUDA_CALLABLE`,
ordinary C++ function-style casts are not rewritten by codegen macros, and the generated kernels
see everything the preamble declares. The cost is that a preamble cannot define macros consumed
by Warp's own headers.

Emitting the preamble *first* was the obvious choice and turned out to be wrong: on CPU, Clang
injects the precompiled `builtin.h` ahead of the translation unit, so a leading preamble lands
*after* Warp's headers there but *before* them under NVRTC — and the CPU behavior flipped with
`warp.config.use_precompiled_headers`. Placing it after the native headers makes both backends
agree regardless of PCH state.

Include directories and dependency files must be absolute and must exist, but the constructor
and `merged()` do not check that — validation happens when the module's options are resolved
for a build. This keeps composition cheap and puts the error at the point where a missing path
actually matters.

`merged()` returns a new object and mutates neither input. Include directories and dependencies
keep their first occurrence; preambles are concatenated in argument order with a newline
between non-empty values. Independently developed addons can therefore compose without copying
every option or overwriting each other.

Warp hashes the resolved paths, preamble text, dependency file *contents*, code-generation
options, and referenced Warp-visible extension contracts, including external builtin identities
and native type schemas. It does not scan transitive C++ includes — addons
must list any header whose contents should invalidate the cache in `extra_build_dependencies`.
Dependency contents are re-read when the module hash is recomputed (after
`wp.set_module_options()` or in a new process), not on every launch, so editing a header does
not rebuild an already-loaded module in a live process.

Mutating a `ModuleBuildOptions` instance in place does not invalidate a compiled module. The
caller reapplies it with `wp.set_module_options()`, which is the single explicit invalidation
point.

#### External builtins

`wp.build_experimental.add_builtin()` is a narrow wrapper over Warp's internal builtin registration,
exposing only the parameters that have a stable external meaning.

`name` is how the function is called in kernels (`wp.<name>`). `native_name` is the exact
qualified C++ function emitted into generated source; it defaults to `wp::<name>`.
`input_types` is an ordered mapping and defines exactly one overload.

External builtins are registered non-differentiable, non-exported, and hidden: they are
callable only from kernel code, and they do not appear in static stubs or the generated builtin
reference. Because all registrations share Warp's global builtin namespace, addons should
prefix their names. Adding an overload to an existing Warp operation is allowed and sometimes
intended.

Registering the same signature twice with the same result type and native target returns the
existing function. Reusing a signature with a different result type or native target raises.

The internal `add_builtin()` is unchanged. Its snippets, gradient hooks, replay functions,
dispatch callbacks, and export controls stay private.

#### Native value types

`wp.build_experimental.add_native_type()` registers a `ctypes.Structure` subclass. That one class is both
the host ABI declaration and the Warp annotation / array `dtype`, which avoids a parallel
wrapper hierarchy and keeps ctypes and NumPy interoperability direct.

The C++ type must be standard-layout and trivially copyable. Warp copies its bytes; it does not
own resources referenced by the value and never invokes an external destructor.

With `fields=None` the type is opaque: values can cross kernel and builtin boundaries, sit in
Warp structs, and be stored in arrays, but kernel code cannot read their members.

A `fields` mapping exposes named public C++ data members as Warp types. Default construction is
always available. `initializer="aggregate"` additionally allows construction from field values;
because C++17 aggregate initialization is positional, that mode requires `fields` to list every
ctypes field in declaration order. Without that constraint, values could silently initialize
the wrong C++ members.

Native types gain no operators or differentiation. Addons register operations as builtins, and
native arrays reject `requires_grad=True`.

#### ABI validation

The ctypes definition is a promise about the C++ layout, so Warp checks it on both sides.

At registration (Python): each exposed field's ctypes storage size must match the size of the
Warp type it maps to, and every exposed field must exist in `_fields_`. Bit-field layouts are
rejected.

At compilation: every module that references the type emits `static_assert`s against the real
C++ definition.

| Contract                               | CPU                    | CUDA           |
| -------------------------------------- | ---------------------- | -------------- |
| Trivially copyable and standard-layout | `static_assert`        | `static_assert` |
| Type size and alignment                | `static_assert`        | `static_assert` |
| Exposed field type                     | `decltype` / equality trait | `decltype` / equality trait |
| Exposed field offset                   | `__builtin_offsetof`   | not checked     |
| Exposed field size                     | not checked            | `sizeof(((T*)0)->field)` |

The two backends check complementary properties because NVRTC does not consistently provide a
usable `offsetof` expression in the supported configurations, while the pointer-based `sizeof`
trick works everywhere. A CUDA-only addon therefore does not get field-offset verification and
remains responsible for matching native member order and padding; compiling the same type for
CPU once is the cheapest way to get that check.

The registered schema is the qualified C++ name, size, alignment, initializer policy, and the
exposed field names, types, and offsets. It is hashed into a short type code (`nt<digest>`) that
feeds module hashing. Re-registering an equivalent schema is accepted and produces the same
stable type code. This makes registration idempotent, but it does not make module reloads safe. A
newly created ctypes class is not interchangeable with the original class during overload
resolution or argument matching. Reload-safe type matching is future work. Registering a
*different* schema under the same C++ name raises.

#### Entry-point ABIs

The default entry-point ABI is `"warp"`: unchanged CPU/CUDA, forward/backward code generation
and `wp.launch()` behavior.

The one external ABI is `"external_constant_params"`. It emits a no-argument `extern "C"`
CUDA entry point whose single Warp struct argument is read from a module-scope constant-memory
symbol named `params`:

```cuda
extern "C" {
__constant__ __align__(alignof(MyParams)) unsigned char params[sizeof(MyParams)];
}

extern "C" __global__ void my_kernel()
{
    MyParams var_p = *reinterpret_cast<const MyParams*>(params);
    // ... kernel body ...
}
```

The entry point uses the kernel's mangled name without the `_cuda_kernel_forward` suffix. With
`strip_hash=True`, Warp uses the kernel key without its hash suffix, giving the external runtime
a stable device-side symbol to look up.

The ABI is deliberately constrained. A kernel using it must be CUDA-only, take exactly one Warp
struct argument, set `enable_backward=False`, avoid `wp.tid()`, shared-memory tiles, and
deterministic atomics, and cannot be passed to `wp.launch()`. Because `params` is a single
module-scope symbol, all `external_constant_params` kernels in one module must use the same
struct type; mixing types raises at code-generation time.

Addons can wrap this generic option in a domain-specific decorator and record their own metadata.
Warp does not need to know about concepts such as OptiX program kinds.

#### AOT artifact paths

`wp.compile_aot_module()` returns `pathlib.Path` objects for the artifacts it created, in
target order (devices first, then explicit `arch` values). Consumers no longer need to
reconstruct Warp's private cache and architecture naming rules. Callers that ignored the
previous `None` return are unaffected.

#### Where the code lives

- `warp/build_experimental.py` — public re-exports for `add_builtin` and `add_native_type`.
- `warp/_src/external_build.py` — external registration implementation and registry state.
- `warp/_src/context.py` — module options, hashing, ABI assertion emission, AOT naming and
  compilation.
- `warp/_src/codegen.py` — lowering of native values, external calls, and entry-point
  signatures.
- `warp/_src/types.py` — native values in arrays, NumPy interop, type identity, type codes.
- `warp/_src/build.py` and `warp/native/clang/clang.cpp` — passing resolved include directories
  to the CPU (Clang) and CUDA (NVRTC) compilers.

`warp.build_experimental` makes the provisional status visible at import sites. If these APIs
later meet Warp's stability and support bar, they could be promoted to a stable `warp.build`
namespace, as `warp.jax_experimental` was promoted to `warp.jax`. Any promotion would define its
own compatibility and deprecation plan.

Native metadata is attached to the ctypes class as `_wp_native_type_` / `_wp_native_vars_` so
existing struct-oriented code generation stays generic. These attributes are private, not an
extension protocol; addons should use `wp.build_experimental` only.

### Alternatives Considered

**Expose internal registration directly.** Publishing the existing `add_builtin()` would be a
smaller code change but a much larger API commitment. Most of its parameters exist for Warp's
own builtin library and have no stable external contract.

**Process-global compiler settings.** An earlier peer draft proposed global `add_header()`,
`add_include_directory()`, macro, and C++ standard setters. Convenient for prototypes, but
every later module inherits the mutation, addons cannot isolate conflicting settings, cache
invalidation is ambiguous, and concurrent compilation can observe unrelated state.
Module-scoped options keep the useful capabilities and make ownership and hashing explicit;
macros go in a preamble, and broader compiler controls can wait for a demonstrated use case.

**Register source files or libraries.** Compiling separate translation units would require a
public link model spanning NVRTC, embedded Clang, NVJitLink, target architectures, relocatable
device code, and cache artifacts. Header-visible implementation covers the initial consumers,
and separate source/library support can be added later without changing this contract.

**Model native types as Warp structs or new wrapper objects.** Warp structs duplicate an
external definition rather than referring to it, and cannot prove its ABI. A new public wrapper
type would raise identity and conversion questions across annotations, arrays, constants, and
NumPy interop. The ctypes class already supplies the host layout with no extra user-visible
type.

## Limitations and Future Work

The experimental API surface is limited to module build inputs, external registrations,
entry-point ABIs, and the AOT artifact-path return contract. The registries are process-global
with no unregister, and dynamically registered builtin names are invisible to static type
checkers.

Plausible follow-ups: linked translation units, stronger CUDA ABI checks, gradient contracts
for external builtins, additional construction policies, more named entry-point ABIs, and
explicit registry lifetimes. These should follow real consumers rather than pre-emptively
exposing compiler internals.

## Testing Strategy

- `warp/tests/test_external_build.py`: registration validation, conflicts and idempotent
  re-registration, `merged()` ordering and non-mutation, dependency and external builtin
  contract hashing, preamble position on both backends, and opaque and field-described native
  values as kernel arguments, builtin results, struct fields, and array dtypes with NumPy
  round-trip.
- `warp/tests/test_codegen.py`: preamble and backend-specific include-directory hashing,
  `external_constant_params` code generation, and its validation errors.
- `warp/tests/test_compilation.py` and `warp/tests/aot/test_module_aot.py`: include directories
  reaching Clang, and AOT artifact paths for CPU, CUDA devices, and architecture lists.
- `warp/tests/cuda/test_occupancy.py` and `warp/tests/cuda/test_kernel_attributes.py`: occupancy
  and kernel-property queries for external entry points.
- `warp/tests/test_modules_lite.py`: confirmation that the internal `add_builtin` is not part of
  the public API.

Preambles, backend-specific include paths, external builtin contracts, and dependency contents
have dedicated hash-sensitivity assertions. Native schemas and entry-point ABI choices are
covered indirectly because native type codes and kernel options feed the module hash. Validation
against a real out-of-tree consumer (`otk-pyoptix`, importing nothing from `warp._src`) is manual.
