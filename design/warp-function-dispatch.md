# Warp Host Function Override Protocol

**Status**: Implemented with explicit application startup opt-in

**Issue**: [GH-1834](https://github.com/NVIDIA/warp/issues/1834)

**Dependency**: [GH-1723](https://github.com/NVIDIA/warp/issues/1723), array subclass input compatibility merged into `main`

**Activation**: `wp.config.enable_array_function_overrides = True`; enable at startup before saving Warp API aliases or importing integrations that cache or wrap them. Registered Warp bindings call native functions directly while disabled.

## Motivation

Add a `__warp_function__` protocol that lets array subclasses and wrappers intercept supported public Warp functions. Users keep calling APIs such as `wp.launch`, `wp.clone`, and `wp.copy`; downstream libraries can observe the operation, execute an alternative implementation, or delegate to Warp and process the result.

The `__warp_function__` spelling follows PyTorch's `__torch_function__` and NumPy's `__array_function__` terminology. Here, “function” means a public host Python API such as `wp.launch` or `wp.copy`. The hook and `enable_array_function_overrides` setting do not change compiled `@wp.func` definitions, kernel-side function calls, or `wp.Function` overload resolution.

LEAPP motivates this extension. Its proposed Warp integration associates arrays with tracing context and FX graph proxies through a `TracedWpArray` subclass. The background notes in `LEAPP--FC.md` describe a global patcher that wraps Warp functions to propagate tracing metadata and discover regions for API capture. An operand-based override protocol gives those responsibilities a supported entry point, including through API aliases obtained after startup activation.

The hook addresses interception of host Python calls involving tracked operands. LEAPP still owns region discovery, framework transitions, mutation tracking, and export. CUPTI monitoring, `.wrp` serialization, PTX portability, ONNX custom operators, and loading captures from bytes are separate concerns.

Subclass compatibility and API overrides solve different problems. GH-1723 allows a properly initialized subclass to pass the same validation as its base array family. This proposal adds control over an operation and its results; it does not require Warp to preserve subclass types or metadata automatically.

## Requirements

| ID | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | Intercept `wp.launch` before kernel specialization, argument packing, or submission | Must | Include positional and keyword calls and all four argument sequences |
| R2 | Provide the `func`, `types`, `args`, and `kwargs` protocol arguments | Must | Preserve the original call and omitted defaults |
| R3 | Support array subclasses and wrappers that implement the protocol | Must | Wrappers must adapt their operands before default execution when necessary |
| R4 | Specify deterministic precedence and `NotImplemented` fallback | Must | One invocation per participating concrete type |
| R5 | Provide safe delegation to the default implementation | Must | Call the supplied function directly; all calls inside hooks use context-local automatic suppression, distinct from application-wide opt-in |
| R6 | Preserve default validation, return values, ownership, streams, and side effects | Must | Include mutation, gradients, tape recording, and capture |
| R7 | Document an explicit set of participating APIs and intentional exclusions | Must | Coverage is part of the public contract |
| R8 | Keep disabled registered bindings native | Should | Registration initially returns native callables; startup activation installs cached wrappers. Retained wrapper aliases still check the live configuration flag. |

**Non-goals:**

- Dispatch from compiled kernel code, `@wp.func` calls inside kernels, or native runtime operations.
- Automatic subclass preservation in allocations, views, gradients, or conversions.
- Implicit wrapper unwrapping, a new array storage ABI, or subclass-specific kernel annotations such as `CustomArray[float]`.
- Observing every allocation, CUDA operation, command execution, or graph replay.
- Implementing LEAPP tracing or changing capture formats and deployment runtimes.

## Design

### Explicit application opt-in

Applications enable the protocol with `wp.config.enable_array_function_overrides = True`. The flag defaults to `False`, and registration initially returns native callables. While disabled, registered Warp API bindings and the array initializer call their native implementations directly. Implementing `__warp_function__` on a type alone does not enable its hooks. Retained wrapper aliases are a separate case described below.

Enable the flag at startup before saving Warp API aliases or importing integrations that cache or wrap them. This includes compatibility shims that retain a reference to an original Warp function before replacing its public binding. Changing the flag switches known native and wrapped bindings in loaded Warp modules, including public exports and internal aliases. Explicit class-attribute registration also switches `array.__init__`. Activation replaces bindings only when their current values match registered callables by identity; unrelated external patches remain untouched. Newly registered APIs honor the current flag.

Aliases saved in application variables, containers, or class attributes before enabling the feature retain their native outer entrypoints. They do not acquire an override for the outer operation, but their bodies may call helpers whose bindings are now wrapped. Obtain aliases after enabling the feature when consistent outer-operation interception is required.

One cached wrapper is reused for each native callable. Aliases saved while enabled retain that wrapper and check the live flag after later toggles. Disabling restores native registered Warp bindings, but cannot replace wrapper references retained by applications. Pickling follows normal Python function identity rules: the currently exported callable is supported, while an alias to the inactive native or wrapped version may fail because its module now exports the other version.

The flag is process-wide. Change it only at startup or between tracing sessions without concurrent Warp calls; switching bindings does not provide session isolation or an atomic transition across all APIs. Overrides are automatically suppressed in the current execution context while a hook or native implementation runs, without mutating the flag. Activation does not add hooks to graph replay or saved launch-command execution.

Startup activation is the selected design after the issue reporter accepted the early-enablement requirement. Applications that never opt in retain native call bindings without a forwarding wrapper or per-call flag check. Libraries implementing the protocol must document activation and alias ordering; importing an integration should not silently enable overrides for the entire process.

### Protocol contract

A participating type defines a synchronous class method:

```python
@classmethod
def __warp_function__(cls, func, types, args=(), kwargs=None): ...
```

The arguments have the following meaning:

| Argument | Meaning |
| --- | --- |
| `cls` | The concrete participating class, including when it inherits a user override |
| `func` | The wrapper for the active registered API, normally identical to the corresponding `wp` export |
| `types` | A tuple of unique participating concrete classes, in dispatch order |
| `args` | The original positional argument tuple |
| `kwargs` | A dictionary containing only the original keyword arguments |

Warp does not normalize the hook's arguments, insert defaults, replace array objects, or rewrite positional arguments as keywords. A hook can inspect all operands through `args` and `kwargs`; `types` is not a complete list of operand types because it excludes objects without an active override.

Any result other than the singleton `NotImplemented` completes dispatch. This includes `None`, `False`, and empty containers. Warp returns that result unchanged and does not then run its own implementation. Exceptions from a hook propagate; they do not select the next candidate.

A hook must decide whether it can handle the function and participating types before making execution or tracing changes. Returning `NotImplemented` must leave operands, argument containers, and tracing state unchanged. Warp cannot roll back an allocation, mutation, capsule consumption, kernel submission, or graph update performed by a declining hook.

### Candidate discovery

Resolve `__warp_function__` on `type(value)`, rather than on the instance. This avoids evaluating arbitrary instance properties and makes interception a type-level contract. An absent hook or a hook set to `None` does not participate. A non-callable hook produces `TypeError`. Define ordinary hooks using the synchronous class method convention above; a plain instance-method definition, including one inherited from a base class, raises `TypeError` with `@classmethod` guidance before invocation. Class-level descriptors and callable hook objects remain supported when their resolved callable follows the four-argument contract. Assigning a hook to an individual instance is unsupported.

Each registered API supplies a fixed signature prepared at registration time. The extractor uses `Signature.bind()` to validate supplied arguments, then visits them in declared parameter order regardless of keyword insertion order. Omitted defaults are not inserted. `launch_tiled` forwards `*args` and `**kwargs` in Python, but uses the explicit fixed `launch` signature for discovery. Arbitrary variadic registration is unsupported. Normal call-shape errors, such as duplicate arguments, remain errors before hook lookup or invocation; discovery does not call `inspect.signature()` on each call.

All direct argument values are checked. For `launch`, additionally inspect the elements of `inputs`, `outputs`, `adj_inputs`, and `adj_outputs`, whether those parameters were supplied positionally or by keyword. `launch_tiled` uses the same sequence extraction rules. Traverse only objects whose exact type is `list` or `tuple`, recursively, with identity-based cycle protection. An object that has an active hook is treated as a participant rather than traversed.

Do not iterate arbitrary generators or custom sequence implementations, traverse dictionary contents, inspect object attributes, or descend into Warp struct fields. A custom argument container can implement its own hook. Additional structured discovery can be added through explicit extractors later. Discovering a hook in a nested list does not make that list a valid native kernel argument; default execution still applies the existing argument rules.

Collect each concrete participating type once, even when several operands have that type or one object occurs in several parameters. Two concrete subclasses that inherit the same user hook still participate separately, with their own `cls` binding. Do not deduplicate by hook function identity.

### Dispatch order

Use actual Python inheritance, as represented by the class MRO, for precedence. For each newly encountered type, insert it immediately before the first existing type that is its strict superclass. If there is no such superclass, append it. The resulting list becomes `types` and determines hook invocation order.

For example, encountering `[Base, Other, Child]`, where `Child` inherits from `Base` and `Other` is unrelated, produces `[Child, Base, Other]`. This defines the interaction between subclass priority and encounter order without relying on an ambiguous comparison sort. Virtual ABC registration does not change precedence.

For `copy`, the destination is encountered before the source. For `launch`, the forward inputs precede forward outputs, then adjoint inputs and adjoint outputs, according to its declared parameter order. Reordering keyword arguments does not change that order.

### Default behavior and deferral

Add a neutral `__warp_function__` class method to the common `Array` base in [`types.py`](../warp/_src/types.py). It returns `NotImplemented`. Warp excludes that exact inherited implementation from candidate discovery, so ordinary arrays and subclasses without a user override do not become dispatch candidates.

The neutral base supports `super().__warp_function__(func, types, args, kwargs)` as a deferral mechanism. It does not execute the operation. An application superclass may implement its own behavior, in which case normal Python `super()` resolution applies. Calling `super()` does not itself restart Warp's candidate selection.

Try each candidate until one returns a result. If every candidate declines, run Warp's default implementation once with the original operands. This intentionally differs from NumPy's all-decline `TypeError`: a Warp subclass can decline a call and retain native behavior. NumPy's protocol and argument extraction model provide the comparison point in [NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html).

Fallback does not strip metadata or extract storage from wrappers. A wrapper that the native API cannot already accept must handle the call and supply valid operands itself. Otherwise the usual native validation error is raised. A wrapper already accepted through an existing array interface retains that behavior.

### Delegation and reentrancy

Automatically suppress overrides in the current execution context throughout every hook invocation. A hook calls the supplied public `func` directly with the original subclass operands, or with explicitly adapted wrapper operands, then processes the native result. No public disable scope is required.

This example observes successful operations without preserving the result's subclass type:

```python
import warp as wp

wp.config.enable_array_function_overrides = True


class ObservedArray(wp.array):
    calls = []

    @classmethod
    def __warp_function__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(t, ObservedArray) for t in types):
            return NotImplemented

        kwargs = {} if kwargs is None else kwargs
        result = func(*args, **kwargs)

        cls.calls.append(func)
        return result


x = ObservedArray([1.0, 2.0], dtype=wp.float32, device="cpu")

y = wp.clone(x)
wp.copy(dest=x, src=y)
assert ObservedArray.calls == [wp.clone, wp.copy]
```

This example uses the subclass compatibility delivered with the protocol. `super()` is appropriate for deferral; calling `func` executes Warp and lets the hook examine or wrap its result. A handler that takes ownership of a mixed-type operation must coordinate with those types explicitly, or return `NotImplemented` before doing work so the outer call can try the next candidate. After native execution, return the result rather than `NotImplemented` to avoid duplicate execution during fallback.

Suppression covers all supported calls inside a hook, including other functions and calls with other operand types. Those calls execute their native bodies without invoking any operand hooks. This deliberately gives up automatic composition of nested hooks in favor of straightforward delegation and one observable outer operation. It also removes active-function tracking and the same-function recursion error. A hook may still explicitly invoke another type's implementation when coordination is needed.

For calls through active wrappers, execute every default function body under override suppression, including when no custom candidate was found. Consequently, an intercepted `clone` call is one observed operation; its internal `empty_like` and `copy` calls do not generate additional hook events. Similarly, `launch_tiled` is observed under its own identity and its internal `launch` is suppressed. This boundary prevents duplicate tracing and makes observation independent of helper composition.

While overrides are enabled, the installed native array-initialization wrapper also suppresses hooks. For example, constructing an array from host data currently calls `warp.copy` internally; that copy must not invoke a subclass hook before its initialization is complete. Subclass authors should attach their own tracing metadata before making supported Warp calls after the native initializer returns. Native initialization is suppressed automatically; arbitrary subclass initialization code is not. Disabling restores the registered initializer binding to its native callable.

Use one `ContextVar` containing a suppression flag and the originating caller location. A single token covers the candidate hooks and any native fallback, and is restored in a `finally` block after success or failure. Suppression is isolated between execution contexts; the application opt-in flag is process-wide. Copied contexts retain suppression and caller attribution, so work intentionally scheduled into a copied hook context also bypasses overrides. This follows Python's normal context inheritance rules and does not change Warp's runtime, tape, or capture concurrency guarantees. Hooks are synchronous.

### Initial API coverage

The following APIs participate as one documented set.

| APIs | Discovery and behavior |
| --- | --- |
| `wp.launch`, `wp.launch_tiled` | Direct arguments and the four launch argument sequences |
| `wp.copy` | Include both destination and source; preserve the `None` return value |
| `wp.clone` | Observe the outer clone operation and allow explicit result wrapping |
| `wp.empty_like`, `wp.zeros_like`, `wp.ones_like`, `wp.full_like` | Include the template array and other direct operands |
| `wp.from_numpy` | Allow a NumPy subclass or wrapper with this hook to handle import |
| `wp.from_torch`, `wp.to_torch` | Include direct source and gradient operands; retain existing return variants |
| `wp.from_dlpack`, `wp.to_dlpack` | Discover before requesting or consuming a capsule |
| `wp.utils.array_scan` | Include input, output, and an explicitly supplied `temp_storage` array; preserve output mutation and the `None` return value |
| `wp.utils.array_sum`, `wp.utils.array_inner` | Include input arrays and an explicit `out`; retain host-value and output-array return forms |
| `wp.utils.radix_sort_pairs`, `wp.utils.segmented_sort_pairs` | Include keys, values, and provided segment-index arrays; retain in-place sorting and scratch-storage requirements |
| `wp.utils.runlength_encode` | Include input, run-value, run-length, and explicit `run_count` arrays; retain integer and count-array return forms |

Utility hooks run before their native bodies inspect array metadata or record API Capture (APIC) operations. A delegated call retains its existing storage, mutation, and return-value behavior, and internal allocation, copy, and readback calls remain suppressed as part of the outer utility operation. Existing graph-capture restrictions still apply: non-empty APIC reductions require an explicit `out`, and non-empty APIC run-length encoding requires an explicit `run_count`.

`wp.utils.array_scan_temp_storage_size()` has no array operand from which to discover a hook and is outside this set. Supplying scan scratch storage does not change the protocol's treatment of the other operands.

These are host function hooks. Plain NumPy arrays and PyTorch tensors do not implement this new protocol automatically. LEAPP can add it to its traced foreign types to observe calls such as `wp.from_torch(traced_tensor)`.

Array constructors, array methods such as `.numpy()`, `.to()`, `.zero_()`, and `.reshape()`, Python operators, `wp.map`, utilities outside the explicit coverage table, Paddle and JAX integrations, mesh and volume APIs, and other functions are outside the initial set. Their internal calls to a covered function can still be intercepted when they occur outside a suppressed default implementation, but this is not a stable hook for the outer operation.

Factories such as `wp.zeros(shape=...)` have no array operand from which to select a hook and are not registered in this phase. A library that needs to observe array-free operations or uncovered methods must use explicit tracing regions or additional instrumentation. The first release can replace LEAPP patches for the covered operand calls; removing its entire patcher requires broader coverage.

For `wp.launch(record_cmd=True)`, the hook observes command construction, which returns a `Launch`, rather than kernel execution. Subsequent `Launch.launch()` calls do not re-enter this protocol. CUDA graph replay and API capture replay also do not invoke these host hooks. Command execution and replay observation would require separately designed hooks.

### Mutation and framework interoperability

The protocol passes the actual operands, retaining repeated-object identity and aliasing. Warp does not clone an array to adapt its type. An override that delegates must preserve destination storage and ownership, especially for `copy` and `launch`. Replacing a mutated destination with a newly allocated wrapper changes the caller's observable behavior and is not transparent delegation.

The `inputs` and `outputs` launch groups are not complete read and write declarations. A kernel can mutate an array supplied through `inputs`. LEAPP must account for kernel mutation and shared storage when updating proxies, rather than infer all writes from `outputs` or a returned value.

Result propagation belongs to the override. A fresh allocation can receive new metadata, while a shared-storage conversion must preserve the relationship to the source and retain its storage owner. A generic constructor call or reassignment of `__class__` is not a supported wrapping recipe. Warp does not copy arbitrary subclass attributes or attach FX proxies.

Conversion hooks run before importing an optional framework, requesting DLPack data, inspecting an array interface, or synchronizing a stream. Discovery never reads array contents or calls `.numpy()`, `__array__`, `__dlpack__`, or `__cuda_array_interface__`. The delegated implementation retains existing device, stream, gradient, lifetime, and `return_ctype` behavior. A DLPack capsule must be consumed once, after the handling decision.

### Subclass compatibility prerequisite

This protocol builds on the separate subclass compatibility implementation for GH-1723 in the base branch. Forward argument packing in [`context.py`](../warp/_src/context.py) accepts subclasses through their registered array family. Generic type inference and type-code generation in [`types.py`](../warp/_src/types.py) use the same family normalization, and `map` uses it in generated annotations and cache keys.

Normalize subclass instances to their registered array family for kernel type matching, while retaining the original Python object for ownership, gradients, and tracing. Apply the ordinary dtype, dimensionality, layout, and configured device-access checks. Do not route a Warp subclass through foreign array-interface conversion as a substitute: that path can bypass native checks and does not carry the same gradient descriptor semantics.

This prerequisite does not change annotation support or require output types to match input subclasses. The dispatch layer likewise leaves annotations and the native array descriptor ABI unchanged.

### Implementation boundaries

The internal [`function_dispatch.py`](../warp/_src/function_dispatch.py) module contains the `warp_function` decorator, fixed-signature extractor, neutral hook, operand visitor, startup registration helpers, and automatic suppression state. Keep its imports lightweight so that early-loaded array types can use it without importing `context` or `utils` and creating a cycle. The implementation is entirely Python and requires no additional extension or build configuration.

Register supported host APIs with `@warp_function` where they are defined in `context.py`, `torch.py`, `dlpack.py`, and `utils.py`, before public re-export. Registration caches native and wrapped callables, uses `functools.wraps` to preserve metadata, and returns the version selected by the current flag. A configuration assignment callback switches registered bindings across known loaded Warp modules. Native array initialization has a suppression wrapper and explicit class-attribute registration.

For an active registered API, the hook receives the public wrapper, so comparisons such as `func is wp.launch` and registries keyed by that callable work. Re-enabling the feature reuses the same wrapper identity. External aliases follow the startup contract above.

An installed API wrapper invokes hooks before the original function body initializes devices, validates array metadata, specializes kernels, records work, or performs conversion. Once default execution is selected, existing validation and native execution remain responsible for those operations. No C++ or CUDA changes are needed for the hook itself.

Registration metadata associates each wrapper with its native callable and fixed-signature extractor. Startup activation uses those callable identities and explicit class-attribute registrations to switch known bindings without modifying function code. Avoid negative hook-lookup caches on user types without an invalidation strategy: replacing a class hook must affect the next intercepted call.

Capture the originating application call site in the dispatch context, without retaining Python frame objects. Use it for tape records and allocation/autograd diagnostics during default delegation. Merely skipping wrapper frames is insufficient: a user override would otherwise become the first external caller. Calls made inside the hook retain the outer application's location under automatic suppression. Preserve the existing default execution exactly once, including tape records, retained command operands, and capture recording.

When enabled, calls through installed wrappers perform argument binding, type-level eligibility checks, declared container traversal, and context bookkeeping. Discovery performs no data movement or optional framework import. Disabled registered Warp bindings enter native functions directly, while retained wrapper aliases check the flag and forward. Performance measurements remain separate from functional tests, with no timing assertions in unit tests.

### Performance implementation

Startup activation changes which callable is installed in registered Warp bindings: native functions while disabled and cached wrappers while enabled. This includes the explicitly registered array initializer. Public and suppression wrappers also capture the live `warp.config.__dict__` at registration time. Retained wrapper aliases read its current `enable_array_function_overrides` entry on each call and forward natively after disabling; capturing the dictionary does not cache the flag's value. Direct native bindings and retained wrapper aliases must be evaluated separately.

The decorator captures one fixed `inspect.Signature` when each API is registered. Enabled calls use `Signature.bind()` to validate arguments before inspecting hooks, without applying omitted defaults. Iterating the bound argument mapping preserves declaration order. One module-level recursive operand visitor handles direct operands and the declared list/tuple parameters. It receives candidate and identity sets explicitly, so discovery does not allocate a recursive closure or leave a per-call self-reference cycle for cyclic garbage collection.

Binding is used only for validation and discovery. Hooks and the default implementation receive the original `args` and `kwargs`, preserving positional and keyword choices, object identity, and omitted arguments. Native and wrapped callables remain ordinary Python functions; metadata supports introspection, method binding, and weak references. Pickling is supported for the currently exported version, subject to the inactive-alias limitation above. Activation replaces binding values without generating or rewriting function code.

On exceptional exits, forwarding frames explicitly release their own `args` and `kwargs` references before re-raising, without mutating the containers or replacing the exception. This preserves upstream failed-build cleanup: an exception retained by a notebook must not keep an obsolete kernel alive through an otherwise transparent forwarding frame. Successful calls use ordinary frame cleanup rather than an explicit argument-deletion step. Hooks that intentionally retain operands continue to own those references.

Discovery uses ordinary lists and sets for candidates, participating type identities, and visited container identities. Exact list and tuple traversal is limited to declared sequence parameters and shares cycle detection across argument groups. Dynamic type-level lookup resolves each visited operand's hook once. When the resolved hook is a Python function, static descriptor inspection detects a plain unbound instance method and raises the class-method guidance error; this does not re-run class-method descriptors or reject callable hook objects. Type deduplication uses identity, and subclass precedence uses actual MRO membership.

Suppression and the caller location share one `ContextVar` value with structured token cleanup. Caller capture eagerly stores an immutable `(filename, lineno, function)` tuple. Hook delegation and other nested calls retain the outer application's location rather than attributing work to hook internals. Copied contexts preserve that location without retaining application frames or locals. There is no active-function tuple, separate scope-token stack, code-offset representation, line-table cache, eviction policy, or diagnostic lock.

Applications requiring overrides explicitly accept enabled-path work, and graph replay bypasses these host wrappers. The implementation uses no generated scanners, handwritten signature binder, or deferred caller-location cache. Validate performance separately for upstream native calls, disabled registered bindings, retained wrapper aliases, enabled ordinary operands, participating hooks, and graph replay. Installation and packaging use Warp's existing Python build process and wheel tags, with no native accelerator or additional wheel variant.

### Local performance evaluation, November 13, 2026

Local host-call measurements compared upstream `b19c73487`, the permanent-wrapper review baseline `fc654c92b`, and the startup-activation implementation, using identical rebuilt native libraries. The environment was Python 3.12.12 on Windows 10 with an NVIDIA GeForce RTX 5080 and driver 591.44. Each variant ran in four fresh processes in balanced order, with nine batches of 1,000 calls per case. The table reports pooled medians in microseconds. Garbage collection and CUDA synchronization occurred outside timed batches; copy timings measure host submission rather than transfer completion. These are local workload-family measurements, not a CI ASV gate or Newton evaluation, and describe the November 13 snapshots before subsequent rebases. The review-baseline column predates the shared review fixes; the optimized permanent-wrapper implementation was measured separately.

| Case | Upstream native | Review baseline | Startup activation |
| --- | ---: | ---: | ---: |
| Disabled, CPU empty launch | 3.078 | 3.293 | 3.078 |
| Disabled, CUDA empty launch | 5.158 | 5.311 | 5.185 |
| Disabled, CUDA launch with three arrays | 7.540 | 7.741 | 7.394 |
| Disabled, pinned host-to-device copy | 2.679 | 2.868 | 2.704 |
| Disabled, pinned device-to-host copy | 2.671 | 2.878 | 2.730 |
| Disabled, CPU zeros | 3.400 | 3.606 | 3.353 |
| Enabled, CPU empty launch, ordinary operands | 3.079 | 8.185 | 7.241 |
| Enabled, CUDA empty launch, ordinary operands | 5.140 | 10.323 | 9.481 |
| Enabled, observed launch with delegation | 6.427 | 13.248 | 11.914 |
| Enabled, observed reduction with delegation | 7.405 | 10.963 | 10.704 |

Disabled registered bindings were between about 1.9% faster and 2.2% slower than upstream across these launch, copy, and allocation cases. CPU empty launches matched upstream at 3.078 microseconds. The native entrypoints remove the fixed forwarding call and flag lookup, while enabled calls retain the protocol's substantial binding, discovery, and context costs. The device-to-host copy result includes a candidate process with broad slowdowns also seen in other cases; all samples were retained. Graph replay follows an unchanged path and varied by about 3% in the enabled control case, so small differences should not be interpreted as reliable gains or regressions.

These measurements cover calls through registered Warp exports. Retained wrapper aliases still pay their flag-check and forwarding cost while disabled; early native aliases retain the interception limitations described above. Local raw samples, source hashes, and native-library hashes are preserved in `.cache/rebase-review-20261113/performance-startup/`.

### Historical permanent-wrapper comparison

A separate November 13 run used the same environment and methodology to measure the optimized permanent-wrapper implementation in `9f1c0320b` against upstream `b19c73487` and the earlier review baseline `fc654c92b`. Disabled CPU empty launches took 3.260 microseconds versus 3.086 upstream, about 5.6% slower; pinned host-to-device and device-to-host copies were about 7.0% and 8.2% slower. Enabled launch and copy cases improved about 7–12% over the review baseline, while keeping the same late-activation behavior for saved aliases. The remaining fixed forwarding cost motivated the adopted startup contract. These were separate measurements from the startup table, so small cross-run differences should not be interpreted as gains from activation alone. The historical raw samples and hashes are preserved in `.cache/rebase-review-20261113/performance-permanent/`.

### Alternatives considered

**Keep wrappers installed while disabled.** Earlier implementations kept permanent wrappers so aliases imported before activation could observe later flag changes. That behavior retained a forwarding call and flag lookup on ordinary calls. The issue reporter accepted obtaining aliases after activation, allowing the selected design to restore native registered bindings while disabled.

**Require explicit suppression for delegation.** An explicit disable scope lets a hook choose which nested operations invoke further hooks. Automatic suppression instead makes ordinary `func(*args, **kwargs)` delegation safe, avoids a public context manager and recursion policy, and matches the primary goal of observing and propagating metadata for an outer operation. Hooks that need cooperation from other custom operand types must arrange it explicitly.

**Add a CPython accelerator.** Native call-entry machinery would introduce Python-version support, binary distribution, lifecycle, and deployment obligations. The selected implementation stays Python-only.

**Continue downstream monkey-patching.** An integration can patch arbitrary operations, but it must manage installation order, alias coverage, interaction with other integrations, and restoration itself. Warp's startup registration centralizes the supported API set and native/wrapped bindings, with the external alias limitation stated explicitly.

**Add a global observer callback.** This can observe operations without tracked operands, but does not provide type-specific override selection or result propagation. It would be a separate instrumentation API.

**Wrap native launches only.** This sees submissions but misses allocation helpers, conversions, and the original Python operands needed for metadata propagation.

**Automatically preserve every subclass.** Warp cannot infer constructor requirements, metadata policies, or wrapper ownership rules. Explicit overrides let each library choose its result behavior.

**Raise an error whenever all hooks decline.** This follows the usual NumPy override rule but prevents an observing subclass from selectively deferring to native Warp. Native fallback better matches the requested behavior; invalid operands still fail native validation.

## Testing Strategy

Register protocol, recording, and interoperability classes in `default_suite` in `warp/tests/unittest_suites.py`. Use `unittest.TestCase` for device-independent protocol behavior and `add_function_test()` with `get_test_devices()` when native execution must cover CPU and CUDA. Focus signature coverage on documented fixed APIs and their supported call forms, including ordinary binding errors before hook lookup. Keep lifetime, context, and delegation tests focused on behavior exercised by supported operations.

| Area | Required checks |
| --- | --- |
| Activation | Direct native registered bindings while disabled; loaded Warp module aliases and array initializer switch on activation; early external aliases remain native; unrelated external patches stay intact |
| Discovery | Positional and keyword calls; declaration order; fixed-signature errors before lookup; output and adjoint operands; scan `temp_storage`; nested exact lists/tuples and cycles; no generator consumption or arbitrary attribute probing |
| Hook validation and lifetime | Direct and inherited instance-method rejection before invocation; class-method descriptors and callable objects remain valid; immediate release of discarded candidate hooks with GC disabled on success and lookup errors |
| Precedence | Child before parent; unrelated encounter order; multiple inheritance; repeated types; distinct types inheriting one hook; neutral base arrays do not preempt wrappers |
| Results | Deferral to another hook and then native fallback; `None`, `False`, and empty results; no extra default execution after a handled call; exceptions propagate |
| Delegation | Direct same-function delegation; adapted operands; suppression of other functions and operand hooks; restoration after exceptions; candidate fallthrough; isolation across execution contexts and inheritance into copied contexts |
| API identity | Active registered APIs pass public wrappers; cached wrappers retain identity across toggles; retained wrapper aliases honor later flags; only current exports promise pickling; signatures and documentation remain discoverable |
| Composition | One `clone` event with no inner allocation or copy events; one `launch_tiled` event; no hooks during native construction; `super()` defers rather than executes |
| Native validation | Ordinary arrays and subclasses retain dtype, dimensionality, layout, generic specialization, and configured device-access behavior |
| Mutation | Destination identity and storage aliases remain intact; kernels can mutate input operands; copies execute once |
| Autodiff and recording | Correct gradients; one tape and capture record; original user call site; recorded commands retain storage owners; hook timing matches command construction |
| Interoperability | Foreign traced types can intercept import; shared-storage conversion retains owners; existing gradient and descriptor return variants; DLPack consumption occurs once |
| Error recovery and diagnostics | Retained failed-build errors do not pin obsolete kernels through forwarding arguments; warning locations skip forwarding frames under Python's default filters and custom loggers |

Keep kernel fixtures in real `.py` files and use public `warp` imports. Verify numerical results with `np.testing.assert_allclose()` and synchronize the tested device when an asynchronous test has no implicit synchronization. Do not clear kernel or LTO caches from the tests and do not add timing assertions.

Run the new dispatch tests together with the affected array, launch, tape, interop, and API capture tests. Import and documentation checks should also exercise installations without optional PyTorch support. A small LEAPP-style test should delegate a launch, convert a tracked operand to PyTorch when available, and preserve tracing metadata through explicit result wrapping and the supported activation path.

## Delivery

The implementation builds on GH-1723 subclass compatibility and includes the full initial coverage table, protocol and device regression tests, and recording and interoperability tests. The public Python guide documents deferral versus execution, method and replay exclusions, and the supported API list. The protocol has its own Towncrier fragment; GH-1723 records the prerequisite separately.

Array methods and additional helpers are candidates for a follow-up. Extending coverage changes what a downstream tracer observes, so additions should be documented and tested as public behavior.
