# Forward-Mode Jets

**Status**: Implemented

**Issue**: [GH-1813](https://github.com/NVIDIA/warp/issues/1813)

## Motivation

Warp differentiates kernels in reverse mode. That gives gradients cheaply, but
second derivatives require differentiating a gradient, and Warp has no
second-order tape.

For simulation work the quantity actually needed is usually not a global
Hessian but a *local* one: an energy written as a sum of small terms,

```
E(x) = sum_i g_i(S_i x)
```

where each `g_i` reads a handful of scalars (one spring, one triangle, one
tetrahedron). The per-term Hessian is small and dense, and the assembled
Hessian is sparse. Users writing such energies today either derive the local
Hessian by hand -- error-prone and hard to keep in sync with the energy -- or
give up on Newton-type solvers.

Forward-mode jets close this gap. A jet carries a value together with its
derivatives along `width` directions at once:

```
x = value + sum_q grad[q] * eps_q
```

Evaluating `g_i` over jets rather than floats yields `grad g_i` as a side
effect of evaluating it, with no hand-differentiation. Crucially, the jet
arithmetic is itself ordinary Warp code, so Warp's existing reverse-mode
autodiff can differentiate *through* it. Reverse-over-forward then gives the
local Hessian one Hessian-vector product at a time, using only machinery Warp
already has.

Jets are complementary to any scheme in which users supply local derivatives by
hand and a separate mechanism scatters them into a global sparse Hessian: jets
remove the need to write those local derivatives, and say nothing about the
assembly step.

## Requirements

| ID  | Requirement                                                                       | Priority | Notes                                            |
| --- | --------------------------------------------------------------------------------- | -------- | ------------------------------------------------ |
| R1  | Evaluating a `@wp.func` over jets yields its gradient with no hand-differentiation | Must     |                                                  |
| R2  | Warp's reverse mode differentiates through jet code, giving reverse-over-forward Hessians | Must | Via `wp.Tape`, or in-kernel with `wp.grad` for a local, tape-free sweep |
| R3  | Ordinary Warp syntax: `a * b`, `wp.sin(a)`, `wp.length(d)`, `v[0]`                 | Must     | No names bound into the caller's module          |
| R4  | Derivative width independent of geometric dimension                                | Must     | 6 directions over two `vec3` endpoints           |
| R5  | Several widths and dtypes coexist in one process                                   | Should   |                                                  |
| R6  | Geometry helpers: `dot`, `length`, `normalize`, `cross`                            | Should   |                                                  |
| R7  | Matrix jets: `determinant`, `trace`, `transpose`, `inverse`, matmul                | Should   | Square `mat22`/`mat33` and rectangular `mat32`/`mat23` for deformation gradients |
| R8  | Second-order jets giving a dense local Hessian in one forward pass                  | Should   | `wp.JetSpace2`; no tape required                 |

**Non-goals**: second-order jets for matrix payloads
([GH-2185](https://github.com/NVIDIA/warp/issues/2185)); quaternion jet types; a native symmetric matrix type for the
Hessian; comparison operators dispatching through the overload table so
`a < b` works on a jet ([GH-2184](https://github.com/NVIDIA/warp/issues/2184)); assembling the sparse
global Hessian from the local blocks; choosing a sparsity pattern or a linear
solver.

## Design

### What `width` is

`width` is the number of derivative directions a jet carries, and it is fixed at
type-specialization time: `wp.JetSpace((wp.vec3, wp.vec3))` (or `wp.JetSpace(6)`)
produces types whose `grad` is a 6-vector, unrolled into the generated code. It is a **compile-time constant, not
a runtime length**. This is the single most important thing to understand about
the API, because it decides what jets are and are not for.

The consequences follow directly:

- **Cost scales with `width`, whether or not you use it.** A first-order jet
  carries `width` derivative components through every intermediate; a
  second-order jet carries `width²`. Nothing is sparse and nothing is skipped,
  so the width you ask for is the width you pay for, in registers and in
  compile time.
- **Each distinct width is a distinct specialization.** A width-6 and a
  width-9 space generate separate struct types and separate builtin
  overloads. They coexist fine (R5), but a width chosen per launch, or read
  from data, is not expressible.
- **This is the right shape for a *local* derivative.** The energies this
  targets have a small, statically known arity: a spring reads 2 `vec3` nodes
  (`width = 6`), a triangle 3 (`width = 9`), a tetrahedron 4 (`width = 12`).
  The width is a property of the element type, known when the kernel is
  written, and the resulting dense local gradient or Hessian block is exactly
  what a Newton solver wants to scatter into a sparse global matrix.
- **It is the wrong shape for a global or unbounded derivative.** Differentiating
  a loss with respect to all `n` degrees of freedom is not a `width = n` jet.
  `n` is a runtime quantity, so it cannot specialize a type; and even where it
  is known, the cost is `n` forward directions to reverse mode's single
  backward pass. Reverse mode (`wp.Tape`, `wp.grad`) remains the right tool
  there, and jets do not replace it. Jets are for the inner, fixed-arity term;
  reverse mode is for the outer sum over an unbounded number of them.

A useful test: if you can write the width as a literal next to the `@wp.func`
that consumes it, jets fit. If the width depends on the size of the problem,
they do not.

### Approach

`wp.JetSpace(slots, dtype)` specializes a family of types at
code-generation time and returns them in a namespace. `slots` describes the
inputs being differentiated with respect to -- see [Slots and seeding](#slots-and-seeding):

| Name    | `value`         | `grad`              |
| ------- | --------------- | -------------------- |
| `scalar`| `dtype`         | `vector(width)`      |
| `vec2`  | `vector(2)`     | `matrix(2, width)`   |
| `vec3`  | `vector(3)`     | `matrix(3, width)`   |
| `mat22` | `matrix(2, 2)`  | `matrix(4, width)`   |
| `mat33` | `matrix(3, 3)`  | `matrix(9, width)`   |
| `mat32` | `matrix(3, 2)`  | `matrix(6, width)`   |
| `mat23` | `matrix(2, 3)`  | `matrix(6, width)`   |

Matrix `grad` stores the entry derivatives row-major (`grad[r * cols + c, q]`
is `d value[r, c] / d eps_q`). The namespace names each jet type after the
Warp type of its value (`J.vec2`, `J.mat22`) and each `grad` type
`J.grad_type_<name>` (`J.grad_type_vec2`, `J.grad_type_mat22`); a
`_type_` infix keeps type attributes distinct from the jet fields
`a.grad`, `a.grad` and `a.hess`. `J.jet_type(t)` and `J.grad_type(t)` look
the same types up by Warp type, for annotations and host code: Warp's kernel
codegen cannot call a Python function that returns a type, so kernel bodies
use the attributes. Each jet type is a `@wp.struct`. The arithmetic is
generated as ordinary `@wp.func`
overloads over those structs, with width-dependent loops statically unrolled
via `wp.static(width)`.

The key move is what happens to those overloads: they are **registered as
overloads of Warp's own builtins**, so user code needs no jet-specific syntax:

```python
J = wp.JetSpace((wp.vec3, wp.vec3))


@wp.func
def spring_energy(x0: J.vec3, x1: J.vec3) -> J.scalar:
    d = x1 - x0
    r = wp.length(d) - 1.0
    return 0.5 * r * r
```

`spring_energy` is a plain `@wp.func`. Nothing is imported or installed.

### Slots and seeding

A jet's derivative directions are numbered, and a user has to say which input
component each direction belongs to. Doing that with raw direction indices --
`seed_vec3(x0, 0, 1, 2)`, `seed_vec3(x1, 3, 4, 5)` -- makes the user do
bookkeeping the types already know, and needs one helper per input type. The
space is therefore described by the *types* of its inputs, called slots:

```python
J = wp.JetSpace((wp.vec3, wp.vec3))  # width 6
J = wp.JetSpace((wp.vec2, wp.float32, wp.mat22))  # width 7
J = wp.JetSpace(4)  # four scalar slots
```

Supported slot types are scalars, `vec2`, `vec3`, and the matrix shapes the
matrix jets cover. An integer `n` means `n` scalar slots. The scalar type is
inferred from the slots unless `dtype` is given; mixing precisions is an error.

`J.seed(*values)` takes one value per slot and returns one jet per slot, with
unit directions assigned in slot order (row-major within a matrix):

```python
a, b = J.seed(x0, x1)  # a, b are J.vec3; directions 0-2 and 3-5
```

A single call is therefore type-checked against the layout by ordinary overload
resolution: passing a `vec2` where a `vec3` slot was declared fails to compile
rather than seeding the wrong directions. Seeding is not a family of
type-specific helpers; one name covers every slot type.

`J.directional(*values, *directions)` takes the same values followed by the
directions explicitly, flat and direction-major: direction `q` of slot `s` is
argument `q * len(slots) + s` after the values. The number of directions is
`width`, which need not equal the number of components. The default `width` is
the component count, which makes `seed` the coordinate special case; passing
`width=1` gives one combined direction, the building block for a
Hessian-vector product:

```python
J1 = wp.JetSpace((wp.vec3, wp.vec3), width=1)
a, b = J1.directional(x0, x1, v0, v1)  # derivative along (v0, v1)
```

`seed` exists only when `width` equals the component count; otherwise reading
it raises an error pointing at `directional`.

`J.variable(x, direction)` remains as the scalar primitive, for seeding inside a
loop where the number of scalars is not a fixed argument list.

**Why positional arguments, not tuples.** Passing the values as one tuple
(`J.seed((x0, x1))`) and the directions as nested tuples reads more naturally, but
Warp does not propagate adjoints into a tuple *argument*: the call compiles, and
reverse mode then returns zero. That is fatal here, because differentiating
through `seed` is how reverse-over-forward Hessians are computed. Multiple
*return* values do propagate correctly, so `a, b = J.seed(x0, x1)` is safe.
Nested tuples are not supported at all. `test_jet_seed_is_differentiable` guards
the property.

**Why the helpers are generated from source.** The number of arguments depends
on the layout, so `seed` and `directional` cannot be written once. They are
generated per layout as source text, registered with `linecache` so Warp's
codegen can read it back, and cached by `(width, dtype, slots)`. All layouts of
one width and dtype share the same jet types and arithmetic, so a space declared
as `(wp.vec3, wp.vec3)` and one declared as `6` interoperate.

**Second-order spaces** take scalar, `vec2` and `vec3` slots
(`wp.JetSpace2((wp.vec3, wp.vec3))`), but not matrix slots. `seed(*values)` is
available and builds a vector slot from one `variable` per component;
`directional` is not, because a second-order jet is seeded along coordinates.

### Why registration on builtins is necessary

Warp resolves a binary operator in `Adjoint.emit_BinOp` roughly as:

```python
name = builtin_operators[type(node.op)]  # ast.Mult -> "mul"

user_func = adj.resolve_external_reference(name)
if isinstance(user_func, Function):
    return adj.add_call(user_func, (left, right), {}, {})

return adj.add_builtin_call(name, [left, right])
```

`resolve_external_reference` is `resolve_closure_or_global(adj.func, name)`: it
looks the name up in the closure or globals **of the function being parsed**.

That is the whole problem. A jet library that generates its overloads inside a
factory function has them in the factory's local scope, with no lexical
connection to the user's module. Operators then only work if the user injects
the names first:

```python
J.install(globals())  # what this design exists to avoid
```

Beyond being ugly, that approach leaks plumbing into the API: it has to merge
overload sets when two widths share a module, guard against clobbering a
user's own `add()`, and it shadows Python's `pow` builtin in the target module.

The final line of the resolution above is the way out. Ordinary expressions
like `vec3 * float` need no `mul` in user globals -- they fall through to
Warp's builtin overload registry and resolve by argument type. Registering the
jet arithmetic there puts jets on exactly the same footing:

```
a * b  ->  builtin "mul"  ->  type-directed overload resolution  ->  mul(scalar, scalar)
```

The same applies to `wp.sin`, `wp.dot`, `wp.cross`, `wp.normalize`, and to
`v[0]`, which routes through the `extract` builtin.

### Alternatives Considered

**`J.install(globals())`.** Inject the operator names into the caller's module
before it defines any `@wp.func`. Works, but every consumer module pays for it,
and the helper needs real machinery: merging overloads across widths, a
collision guard, and an unavoidable shadowing of the `pow` builtin. Rejected as
plumbing in the user API.

**Module-level operator names in the jet module,** with every specialization
added to them, so users write `from warp.jet import add, sub, mul, div`. Avoids
the `install` call but is the same idea: Warp still requires those bare names in
the consumer's globals. Rejected.

**A jet-aware decorator, `@J.func`,** that clones the user's function with a
private `__globals__` containing the operator names, then hands it to
`wp.func`. This keeps the caller's module clean and needs no changes to Warp,
which makes it the best option for a library living *outside* Warp. It still
forces a non-standard decorator on every jet function, and a matching
`@J.kernel` for kernels containing jet expressions. Rejected once the work moved
into Warp, where builtin registration is available and needs neither.

**Second-order jets** give the Hessian in one forward pass with no tape at all.
This was originally deferred -- it does not reuse Warp's reverse-mode autodiff
the way first-order jets do, and the state grows quadratically in the width --
but the tradeoff is worth it for small local energies, so it was added as
`wp.JetSpace2`, with scalar and vector payloads. See
[Second-order jets](#second-order-jets) below. Matrix payloads are left as a
non-goal ([GH-2185](https://github.com/NVIDIA/warp/issues/2185)).

### Key Implementation Details

Everything lives in `warp/_src/jet.py`, exported as `wp.JetSpace`.

**Registration.** `_register(name, fn)` appends each of `fn`'s signatures to
`warp._src.context.builtin_functions[name].overloads`. A Warp function defined
several times under one name accumulates its signatures in `user_overloads`;
builtin overload resolution matches against a function's own `input_types` and
does not recurse into that dict, so each signature is appended individually. A
`@wp.func` appended to a builtin's overload list is returned by `resolve_func`
like any other match and generates a normal user-function call with its
adjoint, which is what makes R2 work.

Registered names: `add`, `sub`, `mul`, `div`, `pow`, `neg`, `pos`, `sin`,
`cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`,
`log`, `log2`, `log10`, `sqrt`, `cbrt`, `erf`, `floor`, `lerp`, `smoothstep`,
`abs`, `sign`, `min`, `max`, `clamp`, `where`, `extract`, `assign_copy`, `dot`,
`ddot`, `outer`, `cw_mul`, `cw_div`, `length`, `length_sq`, `normalize`,
`cross`, `skew`, `diag`, `get_diag`, `transpose`, `determinant`, `trace`,
`inverse`, `svd3`, `svd2`, `eig3`. `wp.JetSpace2` registers the same scalar arithmetic, transcendental,
and branching families (up through `where`), plus the vector arithmetic,
`extract`, `assign_copy`, `dot`, `cw_mul`, `cw_div`, `length`, `length_sq`,
`normalize` and `cross`.

Subscript assignment on a local struct value (`v[i] = s`, `F[i, j] = s`) is
lowered to the struct's `assign_copy` overload, and the result is rebound to
the assigned name, so `w = v; w[i] = s` leaves `v` unchanged.

`svd3`, `svd2` and `eig3` do not differentiate the iterative float code. They
decompose the value with it, then apply the transpose of the analytic rules in
Warp's reverse-mode adjoints (`adj_svd3`, `adj_svd2`, `adj_eig3`), with the same
clamping of nearly repeated singular values or eigenvalues, so a jet and
`wp.grad` agree. One deliberate difference: `adj_eig3` leaves an exactly zero
eigenvalue gap unclamped and skips the eigenvector term when its adjoint is
zero, but forward mode cannot know whether the eigenvectors' derivatives will
be used, so it clamps a zero gap too, keeping them finite at repeated
eigenvalues such as the identity. Only the returning forms
(`U, sigma, V = wp.svd3(A)`) accept jets.

The value-branching builtins (`min`, `max`, `clamp`, `where`, `abs`, `sign`)
select or pass through the derivative of the branch they take, alongside the
value they select. A piecewise energy therefore carries the derivative of the
active piece, which is the one-sided derivative at a switch point rather than a
two-sided one: these functions are non-smooth where the branch changes, and jets
do not smooth them.

**Native conventions at singular points.** Where a derivative is undefined or
the result is locally constant, the jets pick what native reverse mode picks, so
generic code gets the same derivatives either way: `abs(0)` has derivative +1;
`min` and `max` treat a NaN operand as missing and route a tie to the second
operand; `clamp(x, lo, hi)` is `min(max(lo, x), hi)`, so its upper bound has
derivative 0; `atan2` has zero derivatives at the origin; `asin` and `acos` have
zero derivatives outside the open interval `(-1, 1)`; `length` and `normalize`
of a zero vector have zero derivatives; and `svd2` keeps the derivative of
`U V^T` at repeated singular values. `test_jet_ops.py` checks these against
native reverse mode.

**`wp.value()`.** Comparisons themselves (`a < b`) do not overload -- Warp lowers
them to raw C++ operators rather than through the builtin table -- so jet code
compares `wp.value(a) < wp.value(b)`. `wp.value()` is a builtin: on a plain
`float16`, `float32` or `float64` it returns its argument, and every
`JetSpace`/`JetSpace2` adds an overload taking that space's scalar jet
(`J.scalar`) and returning its `value` field. One function can therefore branch
on plain floats and jets alike. Extracting the value drops the jet's
*forward-mode* derivatives, which is what makes it a value, but `wp.value()` is
not a stop-gradient: in reverse mode it is the identity, so a tape or `wp.grad`
still differentiates through it. Being a builtin, its overloads are not part of
a module's hash, so creating a new jet space does not change the hash of
modules that call `wp.value()` on plain floats. A clearer error for comparing
jets directly is tracked in [GH-2184](https://github.com/NVIDIA/warp/issues/2184).

**What stays on the namespace.** Seeding and construction (`seed`,
`directional`, `variable`, `constant`, `make_vec3`, ...) have no builtin
counterpart, so they remain namespace members. So do `perp` and `cross2`, which
Warp does not define for `vec2`; adding them as builtin overloads would give
jets a surface Warp itself lacks.

**Global mutation.** Registration mutates a process-global table. It is
additive, happens once per `(width, dtype)` because `JetSpace` caches, and
appends after the existing overloads so builtin resolution order is unchanged.
Each `_make_jet_space` call produces fresh `Function` objects, so widths do not
interfere. The cost is that overload lists grow: `mul` already carries ~180
overloads and each jet space adds roughly a dozen, which lengthens a linear
scan during code generation only.

**No deferred annotations.** `warp/_src/jet.py` deliberately omits
`from __future__ import annotations`. Warp resolves struct annotations with
`inspect.get_annotations(eval_str=True)`, which evaluates them against module
globals; deferred annotations would turn the locally-generated types into
strings that no longer resolve.

### Usage

Seeding the identity makes one forward pass yield the whole local gradient:

```python
J = wp.JetSpace(2)


@wp.func
def local_energy(a: J.scalar, b: J.scalar) -> J.scalar:
    return wp.sin(a * b) + 0.1 * (a * a * a) + wp.exp(b)


@wp.kernel
def local_gradient(z: wp.array2d[float], grad_g: wp.array[J.grad_type_scalar]):
    i = wp.tid()
    a, b = J.seed(z[i, 0], z[i, 1])
    grad_g[i] = local_energy(a, b).grad
```

Taping that launch and running one backward pass per gradient component gives
the local Hessian a row at a time:

```python
tape.backward(grads={grad_g: seed_row})
# z.grad[i,b] = d grad_g[i,row] / d z[i,b] = H_i[row,b]
```

### Matrix jets

Square (`mat22`, `mat33`) and rectangular (`mat32`, `mat23`) matrix jets let a
deformation gradient be a first-class jet value, so an elastic energy written in
terms of `F`, `S = FᵀF`, `det F`, and `tr S` differentiates through the whole
chain without hand-derivatives. Overloads cover `transpose`, `trace`,
`determinant`, matmul, and `inverse`. Inverse uses the closed-form differential
`d(A⁻¹) = -A⁻¹ (dA) A⁻¹`, evaluated per direction from `A⁻¹` computed once. A
`mat33` deformation gradient is built column-by-column from three seeded `vec3`
endpoints (`make_mat33`), or seeded directly by declaring a `wp.mat33` slot; the rectangular pair supports triangles, whose `3×2`
`F` and `S = FᵀF` come from `make_mat32` and the `(2×3)(3×2)` product.

These reuse the native `wp.inverse`/`wp.determinant` on the `value` and only
add the derivative bookkeeping, so they are cheap relative to the transcendental
overloads.

### Second-order jets

`wp.JetSpace2(slots, dtype)` carries a value, a length-`width` gradient, and a
dense `width × width` Hessian, propagated by the second-order chain rule through
the same arithmetic, transcendental, and branching overloads as first-order
jets. One forward evaluation of a scalar energy yields its local Hessian with
**no tape** and no reverse sweep at all.

That is one of several ways to reach a local Hessian with this machinery, and
which one wins is energy-dependent:

| Route | Mechanism | Cost |
| ----- | --------- | ---- |
| Reverse-over-jet, tape | width-1 dual for each directional derivative, then one `tape.backward()` per direction for a Hessian column | `k` forward and backward launch pairs; needs a global `requires_grad` array |
| Reverse-over-jet, in-kernel | width-1 dual for each directional derivative, reverse sweep taken *in the kernel* with `wp.grad` | one launch, register-resident, no tape and no global `.grad` |
| Pure forward | second-order jet | one launch, no reverse at all; `O(k²)` state in registers |

The in-kernel `wp.grad` route matters because it does not force a choice between
tape overhead and second-order jets. It stays local -- each element's Hessian is
computed in registers and scattered out -- so it composes with sparse assembly
the same way the pure-forward route does, without `JetSpace2`'s compile cost.

That compile cost is the main reason to prefer it. A second-order jet holds an
`O(k²)` Hessian in registers through every intermediate, and NVCC's compile time
is super-linear in `k`. On an RTX PRO 6000 Blackwell (a MIG 1g.24gb slice) with CUDA 12.9, the
tetrahedron (`k=12`) module of `benchmark_jet_hessian_mesh.py`, which holds
the second-order jet kernel alongside the first-order routes, took about
8.5 minutes to compile on a cold cache, after which the second-order route ran
about as fast as in-kernel `wp.grad`. When compile time dominates -- during
iteration, or for a wide energy -- in-kernel `wp.grad` over first-order jets gets
the same Hessian with none of that cost. The tape route remains useful on CPU and
for long chains.

**Vector payloads.** `Jet2Vec2` and `Jet2Vec3` are structs of one `Jet2Scalar`
per component, and every vector operation is composed from the scalar
overloads (`dot` is products and sums, `length` is `sqrt(dot)`, and so on). A
vector expression therefore performs exactly the arithmetic of the same
expression written out per component, and needs no second-order rules of its
own. That also settles the cost: a `vec3` jet holds the three scalar jets the
hand-scalarized function would hold, so state, registers and compile time are
those of the scalar form. The width -- the number of differentiated components
-- is what drives the `O(width²)` cost, not whether the components arrive as
scalars or vectors. Its point is that one function generic over its argument
types serves every Hessian route, so the vector-form energies used with
first-order jets no longer need scalar-form twins for `JetSpace2`.

Matrix payloads are left out
([GH-2185](https://github.com/NVIDIA/warp/issues/2185)). The same componentwise
construction would apply, but the matrix overload surface (matmul, transpose,
determinant, inverse, …) is much larger, and a matrix input can be passed as
its columns or components in the meantime.

## Coverage

First-order (`wp.JetSpace`) and second-order (`wp.JetSpace2`) support, by
operation family. "FD" marks families verified against finite differences in
`test_jet_ops.py`; the rest are covered by closed-form checks in `test_jet.py`.

| Family                                                    | 1st-order scalar | 1st-order vec2/vec3 | 1st-order matrix        | 2nd-order scalar | 2nd-order vec2/vec3 |
| --------------------------------------------------------- | :--------------: | :-----------------: | :---------------------: | :--------------: | :-----------------: |
| Arithmetic (`+ - * / ** neg pos`)                         |        ✓         |          ✓          |            ✓            |        ✓         |       ✓ — FD        |
| Transcendental (`sin cos tan exp log sqrt`)               |        ✓         |          –          |            –            |        ✓         |          –          |
| Inverse-trig (`asin acos atan atan2`) — FD                |        ✓         |          –          |            –            |        ✓         |          –          |
| Branching (`min max clamp where abs sign`) — FD           |        ✓         |          –          |            –            |        ✓         |          –          |
| Indexing (`v[i]`, `extract`)                              |        ✓         |          ✓          |            ✓            |        –         |       ✓ — FD        |
| Geometry (`dot length length_sq normalize cross`)         |        ✓         |          ✓          |            –            |        –         |       ✓ — FD        |
| Component-wise (`cw_mul cw_div`)                          |        –         |          ✓          |            –            |        –         |          ✓          |
| Matrix (`transpose trace determinant inverse`, matmul)    |        –         |          –          |    ✓ (mat22/33/32/23)    |        –         |          –          |
| Matrix (`F[i, j]` `F[i]` `ddot outer diag get_diag skew`, scalar-jet scaling) — FD | – | ✓ | ✓ (mat22/33/32/23) | – | – |
| More scalar functions (`sinh cosh tanh log2 log10 cbrt erf floor lerp smoothstep`) — FD | ✓ | – | – | ✓ | – |
| Subscript assignment (`v[i] = s`, `F[i, j] = s`), incl. reverse mode — FD | – | ✓ | ✓ | – | ✓ |
| Decompositions (`svd3 svd2 eig3`) — against reverse mode and FD | – | – | ✓ (mat22/33) | – | – |
| Reverse-over-jet Hessian via `wp.Tape`                    |        ✓         |          ✓          |            ✓            |     n/a          |         n/a         |
| Reverse-over-jet Hessian via in-kernel `wp.grad` (no tape) |        ✓         |          ✓          |            ✓            |     n/a          |         n/a         |
| Pure-forward Hessian (no tape, no reverse)                |       n/a        |         n/a         |           n/a           |        ✓         |          ✓          |

The second-order vector column is verified in `test_jet.py` against finite
differences of a NumPy mirror (`test_jet2_vector_ops`).

Not covered: second-order matrix jets ([GH-2185](https://github.com/NVIDIA/warp/issues/2185)); a native symmetric
matrix type for the Hessian; overloaded comparisons ([GH-2184](https://github.com/NVIDIA/warp/issues/2184)); quaternion
jets.

## Testing Strategy

Two modules, both registered in `default_suite`:

**`warp/tests/test_jet.py`** (`TestJet`, `TestJetSpace`). Device-parametrized
tests via `add_function_test`, plus a fixed-device `TestJetSpace` for namespace
behavior. The Hessian is checked two ways against
`g(a,b) = sin(a*b) + 0.1*a^3 + exp(b)`: a closed-form NumPy Hessian, and float64
second differences of `g`. The second derives nothing by hand, so it catches an
error in the closed form; the first is exact, so it does not depend on a step
size. Symmetry is asserted separately, since the two off-diagonals come from
separate backward passes and agreement is not automatic. Other cases: the
forward value and gradient; `vec3` geometry through `wp.length` on a spring
energy with a known closed-form gradient; `v[i]`, `wp.dot`, `wp.cross`,
`wp.normalize`, `wp.length_sq`; two widths coexisting, asserting the first still
resolves after the second registers; a `float64` space, asserted to `1e-12`; and
`JetSpace` caching and width validation. Seeding by slot is checked directly --
the gradient blocks of a `vec2`/scalar/`mat22` layout must come out as
consecutive identity rows, a single slot takes and returns a bare value,
`directional` over a coordinate basis must reproduce `seed`, and one combined
direction must give the gradient contracted with it. A separate test takes the
Hessian by reverse mode through `seed` and through a width-1 `directional`
(against a closed form), because Warp silently zeroes adjoints that flow through
a tuple argument and this is the property that would break first.

**`warp/tests/test_jet_ops.py`** (`TestJetOps`, `TestJetMatrix`). Finite-
difference checks for the wider op surface: inverse-trig, `pow` variants, and
value-branching builtins for both first- and second-order jets (gradient and, for
second order, the Hessian); and matrix jets -- `mat22` determinant/trace, a `mat33`
tetrahedron elastic energy, `mat33` inverse, and a rectangular `mat32` triangle
energy. These run on CPU only (`DEVICES = ["cpu"]`, standard `unittest.TestCase`
rather than `add_function_test`): the file's kernels form one large module whose
CUDA build takes far longer than a test should (it had not finished after
18 minutes on an RTX PRO 6000 Blackwell). CUDA coverage of the jets comes from
`test_jet.py`, which runs on CPU and CUDA, and from the implicit-projection
example, which runs on both in the example tests. The benchmarks also run on CUDA
but only report their errors, so they do not count as coverage.

## Benchmarks

Three benchmarks under `warp/examples/benchmarks/` accompany the feature. Each
prints how far its strategies' results are from one another (and, for the
Hessian benchmark, from finite differences) next to the timings; these errors
are reported, not enforced, so they are no substitute for the tests:

- `benchmark_jet_gradient_mesh.py` -- first-order gradient throughput over
  spring, triangle and tetrahedron meshes, contrasting a width-`k` jet pass, a
  `wp.Tape`, and in-kernel `wp.grad`.
- `benchmark_jet_hessian_mesh.py` -- local-Hessian strategies on the same
  energies: width-1 tape, in-kernel width-1 `wp.grad`, and the pure-forward
  second-order jet.
- `benchmark_jet_jacobian.py` -- the Jacobian of a many-output function, where
  one jet pass is shared by every output.
