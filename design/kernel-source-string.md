# Execute Warp Definitions from Source Strings

**Status**: Proposed

**Issue**: [GH-1640](https://github.com/NVIDIA/warp/issues/1640)

## Motivation

Warp normally retrieves the source of `@wp.kernel` and `@wp.func` definitions
from a `.py` file. Functions created by `exec()` have no file-backed source for
`inspect.getsourcelines()`, so Warp decorators reject them and ask the user to
save the code to a file.

This prevents short `python -c` experiments and trusted code generators from
using normal Warp syntax. The existing workaround executes an undecorated
function in a manually prepared namespace, retrieves a Warp module, and then
constructs `wp.Kernel` with explicit `source` and `module` arguments. It is hard
to discover, does not generalize naturally to several definitions, and loses
useful filename information in code-generation errors.

The proposed API executes trusted in-memory Python source while preserving
Warp's existing decorators, modules, hashing, code generation, compiled-code
cache, and CPU execution path.

## Requirements

| ID | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | Execute source containing normal Warp decorators | Must | Supports `@wp.kernel`, `@wp.func`, and `@wp.struct` |
| R2 | Work when invoked through `python -c` | Must | Primary command-line use case |
| R3 | Preserve a meaningful virtual filename and source-block line numbers | Must | Applies to Python and Warp errors |
| R4 | Use the existing Warp module and CPU compilation paths | Must | No separate compiler or compiled-code cache |
| R5 | Support multiple top-level definitions in one source block | Must | Kernels, functions, structs, constants, and imports |
| R6 | Define deterministic module naming and collision behavior | Must | Prevents accidental replacement and stale state |
| R7 | Roll back partial Warp registration when execution fails | Must | Failed calls leave no generated module behind |
| R8 | Reuse identical named source without executing it again | Should | Avoids duplicate objects and registry growth |
| R9 | Leave file-defined kernel behavior unchanged | Must | Existing decorators retain their current behavior |
| R10 | Clearly identify the trusted-source security boundary | Must | The API does not sandbox Python |
| R11 | Specify and test CPU behavior only | Must | GPU/CUDA behavior is outside the design |

**Non-goals:**

- Sandboxing or safely executing untrusted source.
- GPU/CUDA behavior or compatibility guarantees.
- Hot reloading changed source under an existing module name.
- A new kernel language or parser.
- Package installation or dependency resolution.
- Persistent generated-module serialization.
- Cross-process namespace reuse beyond Warp's existing compiled-code cache.

## Design

### Approach

Add `wp.exec_source()`, which compiles trusted Python source with a synthetic
filename and executes it in a named namespace. The source uses normal Warp
decorators, so existing registration and compilation paths remain unchanged.

A synthetic filename is a descriptive filename that identifies source held in
memory rather than on disk. A namespace is the dictionary of names visible to
the executed source.

The source is temporarily registered in Python's `linecache`, the standard
cache used by inspection and traceback tools, while decorators inspect it.
Warp's `Adjoint` objects retain the parsed source needed for later code
generation, allowing the temporary entry to be removed when `exec()` finishes.

Each successful call owns a generated Warp module and a read-only mapping of
the names created by the source. Here, a mapping is a dictionary-like object
whose entries cannot be changed by the caller. An internal registry provides
deterministic same-source reuse and rejects unsafe name collisions.

### Alternatives Considered

**Automatically wrap undecorated functions:** This matches the existing
workaround, but the helper would have to guess which functions are kernels and
manually handle helpers, structs, overloads, and source slices. Explicit
`Kernel(source=...)` also reports an unknown filename and function-relative
line numbers.

**Return one kernel:** This is convenient for the smallest example but cannot
represent a source block containing several related definitions.

**Return the mutable execution namespace:** This exposes infrastructure names
and implies that mutating the dictionary is a supported way to alter a
generated module.

**Create a `types.ModuleType` and add it to `sys.modules`:** Attribute access is
convenient, but Python loaders, package semantics, imports, and removal become
part of the module lifecycle. A named namespace plus `linecache` is sufficient
for source inspection.

**Extend decorators with an explicit source provider:** This avoids
`linecache`, but broadens the `@wp.kernel`, `@wp.func`, and `Adjoint` interfaces
for a use case that can be handled at one public entry point.

**Replace changed source in place:** Warp can retain older same-key kernel
objects and lazily rebuild their shared module executable. Exposing that
behavior as hot reload would make old references and module dependencies hard
to reason about. Changed source requires a new module name instead.

**Use random default module names:** Random names avoid immediate collisions
but prevent deterministic reuse and grow process-global registry state across
identical calls.

### Key Implementation Details

#### Public API

```python
def exec_source(
    source: str,
    *,
    module_name: str | None = None,
) -> Mapping[str, Any]:
    ...
```

Example:

```python
import warp as wp

generated = wp.exec_source(
    """
@wp.func
def scale(x: float):
    return x * 2.0

@wp.kernel
def scale_array(values: wp.array[wp.float32]):
    i = wp.tid()
    values[i] = scale(values[i])
""",
    module_name="generated_scale",
)

values = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
wp.launch(generated["scale_array"], dim=3, inputs=[values], device="cpu")
```

`exec_source` reflects that the operation executes ordinary Python and can
produce several kernels, functions, structs, constants, or imported names. A
single-kernel API would be too narrow for a module-sized source block.

The name also makes the security boundary visible: unlike an API named
``compile_source``, it does not imply that the operation only translates code.
It executes top-level Python statements before any generated kernel is
launched.

Maintainers may prefer one of these alternatives before the API is released:

- ``load_source`` follows the vocabulary of ``load_module``, but understates
  that arbitrary top-level Python executes and suggests that a module object is
  returned.
- ``module_from_source`` emphasizes module identity, but the return value is a
  mapping rather than ``wp.Module``.
- ``kernel_from_source`` is clear for the smallest example, but cannot describe
  source containing multiple kernels, functions, and structs.
- ``compile_source`` sounds non-executing and would obscure the trusted-source
  warning.

No aliases are proposed. One final name keeps the public surface small and
avoids supporting redundant spellings after release.

The function is defined in `warp/_src/context.py` near the existing kernel and
module APIs and re-exported from `warp/__init__.py` under Kernel Programming.

#### Accepted Source

The source is ordinary trusted Python. Warp constructs use their normal
decorators:

- `@wp.kernel` defines a kernel.
- `@wp.func` defines a function callable from Warp code.
- `@wp.struct` defines a Warp structure.

Plain functions are executed as ordinary Python functions and are not
automatically converted into kernels. Definitions must be top-level within one
source block. A generated module cannot be extended across several calls.

Absolute imports behave normally. Relative imports are unsupported because a
generated namespace has no package and is not added to `sys.modules`.

#### Execution Namespace

The private namespace starts with:

```python
{
    "__name__": resolved_module_name,
    "__file__": synthetic_filename,
    "__package__": None,
    "wp": warp,
}
```

Python adds `__builtins__` during execution. The alias `warp` is not injected;
source that needs that spelling imports it normally.

The returned read-only mapping contains public names created by the submitted
source. The reserved names `wp`, `__name__`, `__file__`, `__package__`, and
`__builtins__` are excluded. Imports and ordinary Python values are retained
because they can be useful results or dependencies.

The private namespace remains alive with the generated-source record because
Python functions refer to their globals dictionary. Reusing identical source
returns the same mapping object.

#### Module Naming

`module_name` is optional. An explicit name must be a non-empty dotted Python
identifier: every component separated by `.` must satisfy
`str.isidentifier()` and must not be a Python keyword.

When omitted, the name is derived from the exact UTF-8 source bytes:

```text
__warp_source_<first 16 hexadecimal characters of SHA-256(source)>
```

The source digest, a SHA-256 fingerprint of the submitted text, is distinct
from Warp's module hash. Warp's module hash continues to identify compiled
content, including referenced constructs, types, options, and build settings.

#### Synthetic Filename and Source Inspection

The synthetic filename includes the resolved module name and source identity:

```text
<warp-source:generated_scale:4a1b2c3d4e5f>
```

The complete source is compiled with this filename. Its exact lines are placed
in `linecache` while `exec()` runs, allowing `inspect.getsourcelines()` to
extract decorated functions with correct complete-source line offsets.

Compilation does not inherit `__future__` flags from Warp's implementation
module. The submitted source controls its own future statements, so internal
compiler settings cannot silently change annotation behavior.

The helper removes its `linecache` entry in a `finally` block. `Adjoint`
instances retain their parsed source, and generic kernel/function overloads
construct new `Adjoint` instances from that stored source.

#### Generated-Source Registry

An internal registry in `warp/_src/context.py`, alongside `user_modules`, stores
one record per generated module. A record owns:

- the resolved module name;
- the exact source digest;
- the synthetic filename;
- the private execution namespace; and
- the read-only result mapping.

The registry is separate from `user_modules`: the former tracks source identity
and namespace lifetime, while the latter remains Warp's authority for kernels,
functions, structs, dependencies, hashes, and executables.

#### Reuse and Collisions

| Existing state | Request | Behavior |
| --- | --- | --- |
| No matching name | Any source | Execute and register the generated module |
| Generated name with identical source | Same source | Return the existing mapping without re-executing source |
| Generated name with different source | Changed source | Raise `ValueError` |
| Existing file-backed Python or Warp module | Any source | Raise `ValueError` without modifying existing state |
| Same source with a different explicit name | Same source | Create a distinct generated module |
| Same symbol name in different generated modules | Any source | Allow; Warp modules isolate the symbols |

Generated source is immutable after successful registration. Callers use a new
module name for changed source.

The helper does not create a `ModuleType` or add an entry to `sys.modules`.

#### Execution and Rollback

The operation proceeds in this order:

1. Validate `source` and `module_name`.
2. Calculate the source digest and synthetic filename.
3. Return an identical generated record or reject a collision.
4. Compile the complete source before modifying Warp registries.
5. Create the private namespace and temporary `linecache` entry.
6. Execute the source and let existing decorators register Warp constructs.
7. Build the read-only result mapping.
8. Publish the generated-source record.
9. Remove the temporary `linecache` entry.

Compilation before registry mutation makes syntax errors atomic: a failed
operation does not leave a partially registered generated module.
Other exceptions can occur after an earlier decorator has registered a
construct. A failed call removes all state owned by the generated name:

1. Remove the temporary `linecache` entry.
2. Detach dependency edges from the partial Warp module.
3. Unload CPU executables created by top-level source actions.
4. Remove the partial module from `user_modules`.
5. Discard the namespace and generated-source record.
6. Re-raise the original exception with its traceback.

Generated names cannot refer to pre-existing modules, so rollback does not
modify state owned by file-defined or previously generated modules.

#### CPU Scope

All specified behavior and tests target CPU kernels. The helper contains no
device-specific compilation logic; it prepares Warp definitions and delegates
later compilation to the existing module path.

GPU/CUDA behavior is unspecified and outside this design.

#### Security

`exec_source` executes ordinary Python. Source can import modules, access files,
start processes, mutate global state, or perform any action allowed to the
current process.

The API documentation includes this warning:

> Only execute source you trust. This function does not sandbox Python code.

## Testing Strategy

All launch tests explicitly use `device="cpu"`.

### Core Behavior

- Define and launch one kernel with verified CPU output.
- Run the complete workflow in a `python -c` subprocess.
- Define multiple kernels in one source block.
- Use a generated `@wp.func` from a kernel.
- Use a generated `@wp.struct` from a kernel.
- Use imports and constants from generated definitions.
- Verify that the result contains source-defined names and excludes reserved
  namespace entries.

### Diagnostics

- Report a Python syntax error with the synthetic filename and exact line.
- Report a Warp code-generation error with the synthetic filename and exact
  line.
- Preserve a useful traceback for a top-level execution exception.

### Identity and Collisions

- Derive a deterministic default name.
- Return the same result for an identical name and source.
- Reject changed source under an existing generated name without mutation.
- Allow identical symbols under distinct explicit names.
- Reject collision with a file-backed Warp module.
- Keep identical source under different explicit names distinct.

### Failure Atomicity and Lifecycle

- Clean up a failure before the first decorator.
- Clean up a failure after one decorated definition succeeds.
- Remove generated registry, Warp module, dependency, executable, and
  `linecache` state after failure.
- Compile on CPU after successful `linecache` cleanup.
- Create generic overloads after successful `linecache` cleanup.
- Leave ordinary file-defined CPU kernels unaffected.

Tests use `unittest`. A focused test module is added to `default_suite()` in
`warp/tests/unittest_suites.py`; subprocess coverage follows the pattern in
`test_modules_lite.py`.
