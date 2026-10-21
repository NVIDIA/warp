# Array Subclass Input Compatibility

**Status**: Implemented

**Issue**: [GH-1723](https://github.com/NVIDIA/warp/issues/1723)

## Motivation and Scope

Applications subclass arrays to attach provenance, graph-node identity, or helper
methods. They should be able to pass those initialized objects directly through
Warp operations without repeatedly extracting a separate array and maintaining
its identity and ownership relationships.

Previously, generic inference rejected subclasses, and mapping could generate
references to nonexistent classes such as `wp.CustomArray`. A concrete regular
array launch could succeed through the external array-interface fallback, but
lose gradient pointers and `retain_grad` flags and bypass native dtype, rank,
and launch-access validation. Correct forward values did not establish parity:
two in-place doublings could produce gradient 9 instead of the parent's default 4.

Accept initialized subclasses of `wp.array`, `wp.indexedarray`, `wp.fabricarray`,
and `wp.indexedfabricarray` wherever their particular parent is accepted. Preserve
parent restrictions, original objects, storage ownership, gradients, and explicit
output identities. Inference must not invoke application constructors or retain
allocations in specialization types. No native ABI change is required.

Subclass propagation, application annotations, custom operation dispatch, arbitrary
proxy protocols, and serialization of Python metadata are outside this feature.
`wp.fixedarray` remains a compiler-only stack-array type. The `array1d` through
`array4d` facades are annotation/factory conveniences, not additional storage families.

## Design

### Separate Runtime Objects from Compiler Types

The compiler sees the built-in native family, dtype, and ndim. Runtime descriptors
carry shape, strides, storage, and gradient pointers. Application class names and
metadata do not enter specialization signatures. For example, an initialized
`TaggedArray(wp.array)` containing float32 values specializes as `wp.array[wp.float32]`.
Device code sees the parent's array operations, without Python metadata.

`types.concrete_array_type()` canonicalizes compiler array types, including
lightweight annotations. `_resolve_array_family()` classifies runtime values from
`type(value)` and its ancestry; annotations resolve to no runtime family. Exact
built-in classes and ordinary regular-array descendants have stateless fast paths.
Other families use ordinary `cls.__mro__` lookup. Keep the most specific native
ancestor, so `fixedarray` is not mistaken for `array`, and reject unrelated-family
multiple inheritance because its descriptor layout is ambiguous. Mixins and
same-family diamonds remain valid. Class identity avoids invoking application
metaclass hashing/equality; there is no cache retaining dynamic application classes.
Deliberately falsified metaclass MRO attributes are outside this contract.

### Normalize at Host Entry Points

`types.infer_argument_types()` creates storage-free built-in annotations instead
of constructing a subclass with application-specific constructor requirements.
Python calls to overloaded functions normalize live arrays before argument
inspection, including arrays with `Sequence` mixins that would otherwise be iterated.
`Kernel.add_overload()` normalizes live descriptors supplied by `wp.overload()` or
optimizers before signature lookup, preventing allocation retention and cache pollution.

Generic `launch()` already obtains canonical types from inference. It calls the
internal `Kernel._add_normalized_overload()` directly, avoiding a second normalization pass
on every launch. This internal method also keeps argument-name preparation after
the overload-cache hit. Callers with live descriptors continue using `add_overload()`.

Reject host `fixedarray` values before inference or overload lookup: their signature
can match a regular array even though their native representation is a stack value.
Preserve concrete extents when inferring legitimate compiler templates from `None`.
Reject lightweight annotations used as runtime arguments or struct values. Source
parameters and fields use built-in annotations, not application subclass instances;
exact built-in fixedarray compiler annotations remain supported.

### Pack Native Descriptors and Preserve Ownership

`context.pack_arg()` recognizes Warp families before trying external array
interfaces. Matching subclasses receive parent dtype, rank, and device-access
validation and pass their own `__ctype__()` descriptor, preserving gradient
pointers and flags. Exact built-in arrays retain their direct path. Indexed-array
adjoints remain regular arrays. Wrong-family or invalid-descriptor errors must
not fall back to interface conversion. Non-Warp interface providers keep their
existing regular-array conversion behavior.

For subclass descriptors, require the family's ctypes base and exact native size.
Accept compatible ctypes subclasses, copying their bytes to the canonical native
class and retaining the original descriptor. This prevents dynamic descriptor
classes from growing CPU invoke caches and bypasses shadowed Python rank properties.
Regular descriptors, including indexed backing descriptors, must match the expected
ndim. Fabric descriptors have no native rank field. Struct setters apply the same
checks, retain original objects, and leave bytes and ownership unchanged on failure.
Errors include kernel arguments or struct fields and preserve their original cause.

These checks establish representation, not arbitrary pointer validity or allocation
extent. Subclasses must preserve parent storage and ownership behavior. APIC additionally
requires packed regular-array data and gradient pointers to match the object's own
storage, including indexed backing arrays, before registering capture regions.
Otherwise eager execution and replay could access different allocations.

Tape, recorded commands, and views retain original objects according to their
parent contracts. `wp.launch(..., record_cmd=True)` retains arguments; manually
constructing `wp.Launch` does not acquire a new ownership guarantee. New allocations
and views have no subclass-propagation guarantee, while existing no-op returns and
caller-supplied outputs preserve identity. Capture follows each parent's capabilities
and does not serialize application metadata or reconstruct application classes.

### Apply the Same Identity to Generated Operations

`utils.map()` uses native families in input/output cache descriptors and generated
annotations, preserving existing dtype, ndim, broadcasting, and output-family
distinctions. Generated ranks use integers rather than injected `typing.Literal`,
which can collide with application names. Exact `wp.array` inputs and outputs
bypass family resolution on cache hits. Python binary, reflected, unary, and
in-place operators inherit this behavior through `map()`.

FEM interpolation also keys array destinations by native family. Copies use native
family identifiers; existing allocation helpers and other APIs already using array
attributes or family-aware checks need no mechanical rewrite.

### Public Contract and Migration

The [compatibility guide](../docs/user_guide/compatibility.rst) owns the detailed
subclass contract and family capability table; Basics links to it. Initialize
parent state, preserve storage and descriptor behavior, retain borrowed owners,
allow parent bookkeeping writes, and retain identity-based equality/hash for Tape.
Warp cannot prove arbitrary overrides correct.

The [interoperability guide](../docs/user_guide/interoperability.rst) explains the
intentional native-packing behavior change and explicit view, reshape, conversion,
or external-provider alternatives. Tape overwrite gradients and `retain_grad` now
match the parent; implicit dtype/shape reinterpretation is rejected. Launch access
checks still follow `wp.config.launch_array_access_mode`, with RELAXED unchecked.
The changed changelog fragment links to this guidance rather than duplicating it.
Accepted-input removals are recorded in [deprecations.md](deprecations.md), without
a warning period that would preserve the incorrect gradient and validation path.

## Decisions and Alternatives

| Alternative | Decision and justification |
| --- | --- |
| Replace exact checks with `isinstance` only | Insufficient: constructors, generated annotations, signatures, and native layouts would still disagree. |
| Unwrap or use the array interface for Warp subclasses | Reject: loses identity/gradient relationships and bypasses parent validation. Native descriptors already exist. |
| Construct application subclasses for compiler types | Reject: constructors require application arguments and may have side effects. Use lightweight parent annotations. |
| Normalize every overload lookup | Normalize live descriptors at host entry points; skip redundant work for inferred compiler types on generic launches. |
| Inherited family marker or resolver cache | Defer: markers can hide ambiguous families, strong caches retain classes, and weak caches add complexity. Consider a weak cache only if representative profiles justify it. |
| Bypass metaclass attribute lookup to read the MRO slot | Omit: ordinary inheritance is the supported contract; defending deliberately falsified MRO attributes adds machinery and tests without a supported use case. |
| Pick the first native family in multiple inheritance | Reject: base order does not make unrelated descriptor layouts compatible. |
| Require exact ctypes descriptor classes | Accept compatible subclasses but canonicalize their bytes, preserving extensions without unbounded invoke-cache class identities. |
| Validate arbitrary descriptor pointers and ownership | Defer: layout/rank checks cannot establish this generally. Preserve parent invariants; APIC checks the pointers needed for faithful replay. |
| Preserve subclasses in allocated results or add dispatch hooks | Separate features: constructor protocols, metadata propagation, precedence, and capture semantics need their own contracts. No hook name is reserved here. |
| Keep old interface fallback behind a switch | Reject: preserves incorrect gradients and inconsistent validation. Explicit conversion remains available. |

## Validation and Performance

Maintained coverage is in [test_array_subclass.py](../warp/tests/test_array_subclass.py)
and [aux_array_subclass_cache.py](../warp/tests/aux_array_subclass_cache.py), registered
in the existing default suite. It covers four-family typed/generic inputs and outputs,
constructor bypass, specialization sharing, wrong-family/dtype/rank rejection,
descriptor layout/cache stability, struct assignment atomicity, owner lifetimes,
Tape overwrite/in-place gradients and `retain_grad`, gradcheck/Jacobian, map/operators,
FEM reuse, access modes, capture/replay, and cold/warm parent/subclass cache reuse.

Host fixedarray misuse has a small in-process regression: generic launches and
list/dict `wp.overload()` calls must raise before and after a valid specialization,
leaving overloads and module hashes unchanged. Mock the module loader during invalid
calls so a regression cannot compile incompatible native code or poison disk caches.
The existing fixedarray suite exercises legitimate compiled stack arrays. Separate
compiler-template and source-annotation checks protect extents and annotation boundaries.

Use structural call-count checks for fast paths, not timing assertions in CI.
Measure warmed generic launches with vector arguments, concrete array launches,
and cached `map()` against the same merge base when changing these paths. Timings
depend on platform and driver; current measurements belong in the MR validation
summary. Earlier prototype timings are intentionally omitted because they predated
the current implementation. No extra CI job or standalone compatibility tool is needed.
