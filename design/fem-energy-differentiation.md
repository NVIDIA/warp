# FEM Energy Differentiation With Bound Fields

**Status**: Finite-difference, jet, and reverse-over-forward (draft) backends implemented in `gdaviet/fem-energy-fd-bind`

**Related**: [GH-1813](https://github.com/NVIDIA/warp/issues/1813), `wp.JetSpace` / `wp.JetSpace2` (already in `main`)

## Motivation

A nonlinear FEM energy currently needs separately written integrands for its first and second variations. Bind a valued field to the test and trial directions so a single energy integrand can produce a scalar energy, residual vector, or tangent matrix. Forward-mode jets and centered finite differences consume the same public binding.

## Proposed API

```python
@fem.integrand
def energy(s: fem.Sample, u: fem.Field):
    return density(u(s), fem.grad(u, s))


fd_options = {"test_step": 1.0e-3, "trial_step": 1.0e-3}

E = fem.integrate(energy, domain=domain, fields={"u": u_current})
R = fem.integrate(
    energy,
    fields={"u": fem.bind(u_current, test=u_test)},
    differentiation="finite_difference",
    differentiation_options={"test_step": fd_options["test_step"]},
)
K = fem.integrate(
    energy,
    fields={"u": fem.bind(u_current, test=u_test, trial=u_trial)},
    differentiation="finite_difference",
    differentiation_options=fd_options,
)
```

`fem.bind()` is backend independent. It is a field binding accepted in `fields=`, not a request to differentiate at construction time. A binding with `test` selects a linear form; adding `trial` selects a bilinear form. A single binding may carry only a test or only a trial, but a binding needs at least one of them. The rule that a trial requires a test applies to the whole `fem.integrate()` call: across all bindings, a tangent needs one test direction and one trial direction. Plain fields retain current `fem.integrate()` behavior. A differentiated binding requires an explicit `differentiation` argument in this first version. `differentiation_options` owns the step sizes; the binding does not. `test_step` is required for residuals; `trial_step` is required when a trial direction is bound. A residual call may pass `trial_step`, but does not use it.

For a mixed block, different integrand fields can carry the two directions:

```python
fields = {
    "u": fem.bind(u_current, test=u_test),
    "p": fem.bind(p_current, trial=p_trial),
}
K_up = fem.integrate(
    coupled_energy,
    fields=fields,
    differentiation="finite_difference",
    differentiation_options=fd_options,
)
```

Only one test and one trial direction may appear across all bindings in one call, matching the existing assembly arity. Test gives rows and trial gives columns. Plain-valued fields and `values=` arguments are fixed parameters.

## Semantics

At an integration sample, a bound field evaluates as `current + a * test + b * trial`. The same coefficients apply consistently to `inner`, `outer`, `grad`, `grad_outer`, `div`, and `div_outer` for which the underlying fields provide implementations. All evaluations in one perturbed energy call see the same coefficients. Geometry, quadrature locations, and quadrature weights are fixed when differentiating.

For one test basis direction `v` and one trial basis direction `w`, the backend returns the following scalar density to the existing integration kernel:

```text
R(v)   = [E(u + ht*v) - E(u - ht*v)] / (2*ht)
K(v,w) = [E(u + ht*v + hr*w) - E(u + ht*v - hr*w)
          - E(u - ht*v + hr*w) + E(u - ht*v - hr*w)] / (4*ht*hr)
```

For coupled fields, `u` denotes the tuple of all valued fields and each direction perturbs only the field to which it is bound. Quadrature weighting happens after this density calculation, as with an ordinary form. The energy evaluation with plain fields does not use finite differences.

The first backend is intended to validate the API and assembly semantics. Its truncation and cancellation error, especially for second derivatives in float32, must be documented. The backend must not silently switch methods.

## Implementation Plan

1. Add `BoundField` and `bind()` in the internal FEM field package and export `bind` from `warp.fem`. Validate compatible geometry/domain, evaluated value type, and usable test/trial operators. Keep the valued field's runtime data live so updating its DOF values updates later integrations.
2. Add `differentiation` and `differentiation_options` keywords to `integrate()`. Validate the method and the required positive finite steps. Leave ordinary integrations unchanged.
3. Extend `IntegrandArguments` and `_parse_integrand_arguments()` to discover directions inside bindings. Keep the test/trial objects separate from user integrand parameter names: both directions may belong to the same parameter. Pass these objects to the existing linear and bilinear assembly kernels.
4. Generate an internal `FieldLike` evaluator for a binding. Its `EvalArg` stores the underlying valued/test/trial field arguments and perturbation coefficients. Each supported field operator evaluates the three underlying fields at the same `Sample` and forms the native scalar/vector/matrix linear combination. This avoids jet types and componentwise rewriting in the first backend.
5. Generate a scalar derivative wrapper around the specialized energy function. For each call from the existing integration kernel, evaluate the energy two or four times with coefficient pairs `(±ht, 0)` or `(±ht, ±hr)`, then return the centered difference. Include derivative order, field binding types, and backend kind in generated function/kernel cache keys; keep step sizes as runtime values so changing them does not require recompilation.
6. Support generic and dispatch quadrature assembly, with the usual default strategy selection, and report an explicit unsupported-assembly error for nodal assembly. For dispatch, rebind the bindings' test/trial directions to `LocalTestField`/`LocalTrialField` before making the bound evaluator.
7. Add API documentation, one user-facing changelog fragment, and a focused FEM example that shows energy, residual, and matrix from one integrand.

The bound field's quadrature degree is the sum of its valued field's and test direction's degrees, as for the two fields of a hand-written residual. The trial direction does not add to it, so a tangent uses the same default quadrature as its residual. Inside the energy, `fem.degree()` of a bound field returns the valued field's degree, which does not depend on the bound directions. As for existing nonlinear forms, users can supply an explicit quadrature when the automatic degree estimate is insufficient.

## Validation

Add a `unittest` FEM module and register it in `default_suite`. Use independently derived forms for a nonlinear scalar energy, a vector field energy using `fem.grad`, and a mixed-field cross block. Check matrix row/column orientation, updating valued DOFs after binding, unequal test/trial steps, invalid bindings, and the unsupported assembly modes. Test CPU and CUDA where available, using float64 for tight finite-difference comparisons and dtype-appropriate tolerances elsewhere. Run only this individual test module during development; do not run the project test runner or clear the kernel cache.

## Jet Backend

`differentiation="jets"` takes no options (`None` or an empty dictionary) and consumes the same `fem.bind()` objects. The jet variables are the binding coefficients `a` and `b` of `current + a * test + b * trial`, not the element degrees of freedom, so the jet width is 1 or 2 regardless of the element's node count and compile cost stays small.

- Residual: every bound field evaluates as a width-1 `wp.JetSpace` jet built with `directional(current, test)`. The energy is evaluated once and its derivative coefficient `coeff[0]` is `R(v)`.
- Tangent: every bound field evaluates as a width-2 `wp.JetSpace2` jet whose gradient is `(test, trial)` (zero for an unbound direction) and whose Hessian is zero. The energy is evaluated once and the mixed second derivative `hess[0, 1]` is `K(v, w)`.

`integrate()` replaces each `BoundField` with an internal `_JetBoundField` copy for the requested order. Its name carries the order, so evaluators, integrands, and kernels are cached separately from the finite-difference variant. Its runtime argument holds the three underlying field arguments without perturbation coefficients.

Limitations:

- The energy and any helper function it calls must accept jet types, for example with `Any` annotations. Jets overload arithmetic, transcendental functions, `dot`, `length`, `transpose`, `trace`, `determinant`, `inverse`, and componentwise operations, but not `ddot` or `outer`.
- `JetSpace2` has scalar, `vec2`, and `vec3` types only. Tangents of energies that evaluate a matrix, such as the gradient of a vector field, are rejected before code generation with a `NotImplementedError` naming the operator and field. Second-order matrix jets could be built from rows of second-order vector jets, since products, transposes, determinants, and traces reduce to existing `dot`, `cross`, and `extract` overloads. An in-kernel `wp.grad` over first-order jets is an alternative that avoids new jet types.

With dispatch assembly, jets are batched: instead of one call per pair of local
components, the jet directions are all local value and gradient components of the
test (and trial) functions at a quadrature point, and the energy is evaluated once
per point. A residual uses a first-order jet of width `K_test`; a tangent uses a
second-order jet whose Hessian block holds every local test/trial pair. When test
and trial are the same local functions of the same binding, they share one set of
`K` directions; otherwise the width is `K_test + K_trial`. `K` is the value
component count times one plus the dimension when both values and gradients are
used, so it does not depend on the element order. Dedicated local kernels launch
one thread per quadrature point and scatter the jet coefficients into the same
local buffers as ordinary dispatch. The quadrature sample carries placeholder
test and trial degrees of freedom; samples without them, such as those returned
by `lookup` or by `at_node` on a discrete field, get no direction, as with
generic assembly.

Local Gauss–Newton and Hessian projection operators remain separate API decisions.

## Reverse-Over-Forward Backend (Draft)

`differentiation="reverse_over_forward"` takes no options and consumes the same `fem.bind()` objects. Residuals are a single forward pass and use the jet backend unchanged. Tangents support generic and dispatch assembly. Dispatch goes through the batched local bilinear kernel that the jet backend also uses; generic assembly differentiates one test/trial pair at a time (see below).

It removes the limitation of `JetSpace2`, which has no matrix types: only first-order jets are needed, and they exist for scalar, vector, and matrix values. Tangents of energies that use `fem.grad` of a vector field work. It also avoids the `K x K` Hessian state of a second-order jet, with `K` local components. The state per quadrature point is `O(K)` plus the `K x K` output block, and compile time does not grow with `K^2`.

At a quadrature point, `_RofBoundField` evaluates `current + sum_i a[i] * test_i` as the value of a width-1 jet whose derivative is the local trial component `j` selected by `trial_index`; `a` is a coefficient vector with one entry per local test component. For each `j`, `directional_derivative` returns the jet's derivative coefficient `D_j(a)`, and `wp.grad` of that function returns the adjoint of `a` at `a = 0`, which is column `j` of the local tangent block `H[i, j]` for all test components `i` at once. The wrapper `reverse_over_forward_integrand` loops over `j`, fills a `K_test x K_trial` block, and returns it to the existing kernel with zero offsets, which scatters it as for jets. Argument parsing allows one binding with a test direction and one with a trial direction, which may be the same binding.

Under generic assembly each thread has a single test and a single trial direction, so `a` is one scalar, the jet follows the sample's trial degree of freedom, and one forward and reverse pass returns that pair's entry. This is what lets generic assembly handle matrix-valued operators, which second-order jets cannot.

### Measured cost

Tangent assembly into an existing BSR matrix, timed under a CUDA graph with CUDA events (RTX PRO 6000 Blackwell, float32, 3D hexahedral grids, five repetitions per round, minimum over five rounds). The scalar energy is `0.5 (1 + u^2) |grad u|^2 + 0.25 u^4`, and "hand" is its separately derived bilinear form. The vector energy is a compressible neo-Hookean density of `F = I + grad u`, for which `jets` tangents are unavailable. All methods agree with the reference to about `1e-7` (`jets`, `reverse_over_forward`) and `1e-3` (`finite_difference`).

| Case | Cells | hand | `jets` | `reverse_over_forward` | `finite_difference` |
|---|---|---|---|---|---|
| scalar Q1, dispatch | 8000 | 0.133 ms | 0.123 ms | 0.127 ms | 0.136 ms |
| scalar Q1, generic | 8000 | n/a | 0.140 ms | 0.184 ms | 0.159 ms |
| scalar Q2, dispatch | 1728 | 1.51 ms | 1.18 ms | 1.29 ms | 1.47 ms |
| scalar Q2, generic | 1728 | n/a | 16.0 ms | 25.6 ms | 17.3 ms |
| vector Q1, dispatch | 4096 | n/a | not supported | 0.226 ms | 0.248 ms |
| vector Q1, generic | 4096 | n/a | not supported | 0.534 ms | 0.570 ms |
| vector Q2, dispatch | 512 | n/a | not supported | 1.01 ms | 1.17 ms |
| vector Q2, generic | 512 | n/a | not supported | 60.6 ms | 20.7 ms |

Cold-cache module compile time was about 1 to 1.6 s for every method at these sizes, with no growth for `reverse_over_forward`. Timings of one kind of GPU only; other architectures and tetrahedral meshes are not measured.

Findings:

- Under dispatch, `reverse_over_forward` is within about 10% of `jets` and the hand-written form on scalar energies, and about 10 to 15% faster than finite differences on the vector energy, while being exact.
- Under generic assembly it is the slowest method on higher-order elements, and 60 ms against 21 ms for finite differences on vector Q2, which is 3 times slower. For `reverse_over_forward`, dispatch is 20 to 60 times faster than generic on the Q2 cases.

### Direction cache (design)

The slow generic result had a clear cause. Every generic thread evaluates the current field, which sums over all of the element's nodes, and the test and trial directions inside the function that `wp.grad` differentiates, so those evaluations happen once in the forward sweep and again when the reverse sweep replays it. The finite-difference backend pays that cost once per sample, by storing the values in the binding's runtime argument (`fill_cache`). The table above predates the generic cache, which now exists (below).

Proposed design, using the same mechanism as the finite-difference backend:

1. **Generic assembly (implemented).** In generic mode `_RofBoundField` shares the finite-difference binding's runtime argument: its perturbation coefficient `test_coeff` is the variable that reverse mode differentiates, and the per-operator cache entries hold `value`, `test`, and `trial`. The pair wrapper fills the cache once per sample, before `wp.grad`, and `evaluate` reads `value + test_coeff * entry.test` for the perturbed value and `entry.trial` for the jet direction. The key check falls back to uncached evaluation for samples at other places, such as `lookup`. Interleaved A/B runs on the same machine (minimum assembly time over five rounds, two sessions; `finite_difference` generic as a noise reference, which stayed within 1%):

   | Case, generic | before cache | with cache | `finite_difference` |
   |---|---|---|---|
   | scalar Q2 (1728 cells) | 27.3 ms | 18.3 ms | 17.3 ms |
   | vector Q2 (512 cells) | 64.6 ms | 34.9 ms | 22.4 ms |
   | vector Q1 (4096 cells) | 0.58 ms | 0.62 ms | 0.57 ms |

   The cache cuts generic time by a third to a half on quadratic elements, which is where gathering the current field over all nodes dominates, but generic reverse-over-forward remains slower than finite differences on vector Q2 (1.6 times). On linear elements the cache is slightly slower (about 7%), since gathering is cheap there and the filled entry and its adjoint cost more than they save. Possible next steps: skip the cache for elements with few nodes, or avoid the adjoint of the cached members, which `wp.grad` computes although they are not needed. Not yet covered by a test: evaluation at a sample other than the cached one.
2. **Dispatch assembly.** Fill, once per quadrature point and before the loop over trial components, the current value and two direction matrices per operator: test directions with `K_test` columns and trial directions with `K_trial` columns (one shared matrix when test and trial are the same local functions). This removes `K_trial * (K_test + 1)` basis evaluations per point, in both sweeps. `evaluate` computes `value + unflatten(test_dirs * a)` and takes `column(trial_dirs, trial_index)`, using the same flattened-row layout as the first-order jet coefficients. Expected gain is smaller than for generic, since dispatch already matches the hand-written form on Cartesian grids; the benefit should grow with more costly basis evaluation, such as higher-order simplices or non-affine geometry. Measure before keeping it.
3. **Costs and risks (dispatch).** The cache adds `flat_rows * (K_test + K_trial)` registers per operator, for example 108 floats for each direction matrix of a vector field's gradient with `K = 12`. `wp.grad` also computes adjoints for these members, which are not needed, so register pressure roughly doubles and may spill. If it does, store only the test directions in the differentiated struct, or process trial components in chunks.
4. **Validation.** The existing test module covers correctness for both assembly modes. Add a test that evaluates at a different sample to exercise the key-check fallback.

Not covered: reusing the primal (value) part of the jet across the trial passes. It is identical for every `j`, but `wp.grad` treats the function as a black box.

Open points:

- Benchmarks on other GPU architectures and on tetrahedral meshes.

## Prototype implementation notes

`BoundField` wraps a valued `GeometryField` and optional `TestField` and
`TrialField`. It generates field evaluators for inner/outer value, gradient,
divergence, and reference gradient. Operators that require direct access to a
field's nodes or that pass an explicit element-local node index are outside this
prototype. The wrapper's runtime field argument stores each underlying field
argument plus two perturbation coefficients. The default quadrature degree
counts each binding's valued field and test direction; an explicit quadrature
may still be necessary for nonlinear densities.

`integrate()` recognizes directions in the bindings and passes their original
test and trial objects to generic assembly. For dispatch assembly it rebinds each
binding to the `LocalTestField`/`LocalTrialField` wrappers of those directions and
passes the wrappers to the local kernels. The energy is then evaluated per pair of
local value/gradient components instead of per pair of element nodes, and the
second pass scatters the results to nodes; this relies on the derivative being
linear in each direction, which holds exactly for jets and to truncation error for
finite differences. The local wrappers choose their component layout from the
operators the energy applies, so a binding forwards operator usage to its
directions and then regenerates its own evaluators. A generated Warp function changes
the runtime perturbation coefficients and invokes the specialized energy two
or four times. Steps live in the generated value argument struct and are
therefore runtime values; changing them reuses the compiled kernel. Before
the perturbed evaluations, the wrapper stores the current field's and the
directions' values for every operator the energy applies into the binding's
runtime argument, keyed by the sample's element, coordinates, and test/trial
degrees of freedom; each perturbed evaluation then only forms
`current + a * test + b * trial` from those values. Evaluations at other samples,
for instance through `lookup`, bypass the cache. The
binding's specialized name and the differentiation method appear in cache
keys. The current field's DOF arrays are read again for every integration, so
updates made after binding are visible.

The focused test module compares scalar, vector-gradient, and mixed-field
forms with analytic integrands. It runs on CPU and CUDA when a CUDA device is
available. Native code is unchanged; in the independent worktree, local
validation used native binaries copied from the original checkout because a
fresh build could not fetch libmathdx in the offline environment.
