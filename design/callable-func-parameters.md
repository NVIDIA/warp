# Callable Function Parameters

**Status**: Implemented

**Issue**: [GH-1424](https://github.com/NVIDIA/warp/issues/1424)

## Motivation

Warp users can write higher-order Python helpers that accept a callable and
apply it to values, but the same pattern did not work inside user-defined
`@wp.func` code. A function parameter annotated as `Callable` was not matched
consistently during Warp overload resolution, especially across the two standard
library import paths:

```python
from typing import Callable as TypingCallable
from collections.abc import Callable as AbcCallable
```

The aliases are illustrative; Warp treats both origins as the same type-erased
`Callable` marker. This design implements the user-defined function portion of
GH-1424 as a first pass: support direct inline calls such as
`apply(double_it, 3.0)` from a kernel or another `@wp.func`, without depending
on local function-variable assignment behavior.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | ----------- | -------- | ----- |
| R1  | Recognize `typing.Callable` and `collections.abc.Callable` as callable annotations. | Must | Includes bare and parameterized forms. |
| R2  | Allow user-defined `@wp.func` objects to match `Callable` parameters. | Must | Callable values are type-erased. |
| R3  | Preserve the existing `"c"` type code for callable annotations. | Must | Used by module hashing and overload keys. |
| R4  | Keep annotation recognition compatible with Python 3.10 through 3.14. | Must | Avoid private `typing` internals. |
| R5  | Include callable argument and default targets in module hashes and dependencies. | Must | Prevent stale compiled modules when callable targets change. |
| R6  | Reject unsupported callable specializations explicitly. | Must | Built-ins are deferred; custom grad/replay functions are first-pass non-goals. |

**Non-goals**:

- Validate parameterized callable signatures such as
  `Callable[[float], float]`.
- Implement runtime dispatch for arbitrary Python callables.
- Support kernel-local function variable assignment.
- Support built-in Warp functions such as `wp.sin` or `wp.add` as callable
  arguments in this first pass. They are rejected until built-in callable
  identity can participate safely in specialization hashing and dependency
  tracking.
- Support callable-specialized functions that have custom gradient or replay
  functions.

Parameterized callable annotations are recognized as callable markers, but
their argument and return types remain unchecked.

## Design

### Approach

The implementation treats `Callable` parameters as compile-time function
references. A callable parameter never becomes a runtime C++ function pointer or
kernel parameter. Instead, code generation specializes the user-defined function
for the concrete user-defined Warp function passed at each call site.

This means:

- `apply(double_it, x)` and `apply(triple_it, x)` produce separate specialized
  native functions.
- Callable parameters are bound into the specialized function's codegen symbol
  table.
- Callable parameters are omitted from the emitted forward and reverse C++
  function signatures.
- Runtime calls pass only the non-callable arguments.

The callable annotation predicate lives in `warp/_src/types.py` as
`is_callable_annotation(annotation)`. It returns true for:

- bare `typing.Callable`,
- bare `collections.abc.Callable`,
- `typing.Callable[...]`,
- `collections.abc.Callable[...]`.

The helper uses `typing.get_origin()` plus the canonical
`collections.abc.Callable` object, which covers the supported Python versions
without relying on private implementation details.

### Alternatives Considered

One option was to normalize annotations globally during `@wp.func` registration.
That would make all `Callable` forms identical up front, but it would also
change the raw annotations stored on every function and could affect unrelated
signature handling. A small predicate keeps the behavior localized.

Another option was to support built-in Warp functions as callable values.
GH-1424 includes built-ins, but this first pass narrows support to user-defined
targets. Built-ins require hashing and dependency behavior for callable
identity, so they are rejected explicitly to avoid partial support that could
reuse stale cached modules.

### Key Implementation Details

`get_type_code()` returns `"c"` for callable annotations before the generic
type branches. This keeps module hash type-code behavior stable for bare and
parameterized `Callable` forms.

`func_match_args()` treats a `warp._src.context.Function` value as compatible
with any callable annotation. Non-function values continue to fail normal
overload resolution.

During `Adjoint.add_call()`, default arguments are applied first. Callable
arguments and callable defaults are then collected. Built-in function values are
rejected, and user-defined function values trigger creation of a specialized
`Function` clone. The clone receives:

- a hash-suffixed native function name,
- a fresh `Adjoint` with the original annotations, including the return
  annotation,
- `callable_arg_values` used to bind callable parameter names during codegen.

Module hashing and dependency discovery both inspect callable arguments and
callable defaults. User-defined callable targets are included in the referenced
function set for hashes and in module references/dependents for invalidation.

Callable-specialized functions with custom gradients or custom replay functions
are rejected. Their custom functions are tied to the unspecialized native
function signature, while callable specialization removes callable runtime
parameters from the emitted signature.

## Testing Strategy

`warp/tests/test_func.py` covers:

- bare `typing.Callable` and `collections.abc.Callable` runtime calls,
- parameterized `Callable[[float], float]` runtime calls,
- generic user functions that combine `Callable` with `Any`,
- default callable arguments,
- keyword callable arguments,
- nested user-defined function calls,
- callable targets affecting module hashes,
- callable targets updating cross-module dependents,
- return annotation preservation on specialized functions,
- explicit rejection for built-in callable targets,
- explicit rejection for callable-specialized functions with custom grad or
  replay functions.

Local verification should include the focused `TestFunc` suite and pre-commit
over the changed files. CI provides the full Python 3.10 through 3.14 matrix.
