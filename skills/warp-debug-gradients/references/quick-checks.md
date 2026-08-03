# Known-pattern checklist

Scan the user's code for each pattern below. For every hit, note it with file
and line, but rank it against the failure signature before declaring it the
root cause — several of these can be present and harmless in a given script.
Doc section names refer to the Differentiability guide
(https://nvidia.github.io/warp/stable/user_guide/differentiability.html; in a Warp
source checkout, `docs/user_guide/differentiability.rst`).

## Tier 1 — cheap structural checks

### 1. Write-after-read array overwrites (the most common killer)

Docs: "Array Overwrites" and "Array Overwrite Tracking".

Warp propagates gradients only through the *final* write to an array. Any
array that is **read on the tape and then written later on the same tape**
computes its earlier adjoints from clobbered values. The rule is one
sentence; the disguises are many:

- `wp.copy(dst, src)` where `dst` was already an input to an earlier taped
  launch. `wp.copy`/`wp.clone`/`array.assign` are differentiable and
  tape-recorded — that is exactly why copying *onto* an already-read array
  is dangerous. On Warp 1.17+ the corruption occurs when the copy **changes
  the values** the earlier launches read (their adjoints replay against the
  new contents — the tape does not snapshot); a write-back that preserves
  values (e.g. an identity/clipping stage that did not engage) computes
  correct gradients, because the copy adjoint routes and consumes gradients
  properly. On Warp < 1.17 the pattern is corrupt *regardless* of values
  (the copy adjoint also misrouted gradients around the intervening
  stage). The tracker warns in all cases, and the fix is the same
  everywhere: rebind the Python reference (`state.q = new_q`) instead of
  copying in place.
- Python rebinding that makes two "different" states the same object:
  `states[0] = states[-1]` between frames makes substep 0's read and the last
  substep's write hit the same array on the next frame's tape. Copy data or
  rotate distinct buffers instead.
- Ping-pong / double-buffer schemes where a buffer written in step N+1 was
  read in step N of the same tape.
- `.zero_()` or `fill_()` on an array already read on the tape (that is a
  write too). The `requires_grad` branch in solvers typically allocates fresh
  arrays (`wp.zeros_like`, `wp.clone`) per iteration instead — that pattern
  is *correct*, not wasteful.

Benign lookalikes: `x[tid] += ...` and `wp.atomic_add` are fine (adjoint
accumulation is handled); write-then-read within a single kernel is fine;
rebinding a Python variable to a *fresh* array per iteration
(`out = wp.clone(inp)` ... `inp = out`) is fine because every write targets
new memory.

Version caveat (Warp < 1.17): the adjoint of a tape-recorded copy
(`wp.copy`/`wp.clone`/`array.assign`) *overwrites* the source array's
gradient instead of accumulating into it. The general rule: **a recorded
copy is safe only if it is the LAST consumer of its source in forward
order.** If any later taped operation reads the source, the backward pass
runs those adjoints first, and the copy's adjoint then replaces everything
they accumulated with the destination's adjoint alone. The observed
signature depends solely on what the destination's adjoint holds at that
moment:

- destination subsequently **fully overwritten** (its adjoint consumed and
  zeroed by the overwriter's backward): source gradient becomes **exactly
  zero** — e.g. `out = wp.clone(q); kernel(q -> out); q = out`, iterated;
- destination **still carries adjoint** (read by the loss, or only
  partially rewritten — snapshot copies, ping-pong buffers, partial-update
  solvers): source gradient is **silently wrong by a large, data-dependent
  factor** — no NaN, no warning, plausible-looking numbers.

No overwrite warning fires in either case (there is no write-after-read in
the forward pass) — invisible to the tracker; suspect it whenever a
clone/copy's source has any downstream reader. Fixed in Warp 1.17
(copy adjoints accumulate). Workarounds on affected versions, in order:
reorder the copy last; replace it with a trivial `@wp.kernel`
(`out[i] = src[i]`) — kernel adjoints accumulate correctly; or
`wp.empty_like` when the cloned contents are never consumed.

Always confirm with the tracker (`references/verification.md`) rather than by
eye — and remember it cannot see arrays stored inside Warp structs, so a
clean tracker run does not clear struct-held arrays.

### 2. Missing `requires_grad` in the chain

`wp.zeros`/`wp.empty`/`wp.array(...)` default to `requires_grad=False`;
`zeros_like`/`ones_like`/`full_like`/`clone`/`empty_like` **inherit** from the
source array. One non-differentiable array anywhere in the chain silently
zeroes everything upstream of it. Check every intermediate the tape flows
through, not just the optimized inputs and the loss. For long chains, dump
`tape.visualize()` — arrays with `requires_grad=True` render green, others
grey, which makes the break visible at a glance (pass `inputs=`/`outputs=` to
`wp.launch` or the graph loses structure).

Related: `enable_backward=False` at global, module
(`wp.set_module_options`), or kernel (`@wp.kernel(enable_backward=False)`)
level means no adjoint is even compiled. `tape.backward()` warns about this;
`gradcheck_tape` silently *skips* such kernels.

### 3. Missing `tape.zero()` / `tape.reset()` in the optimization loop

`tape.backward()` accumulates into `.grad` arrays. Without zeroing between
iterations, gradients grow monotonically and training "explodes" with
perfectly correct per-iteration gradients. Conversely, reading grads *after*
zeroing yields zeros.

### 3b. Tape-scope hygiene

Only the forward pass belongs inside `with tape:`, and the loss computation
should be the last thing recorded. A common indentation slip keeps
post-loss work (gradient reads, optimizer steps, logging, state resets)
inside the scope — those launches get recorded too, can introduce spurious
write-after-reads against arrays the forward pass consumed, and bloat the
backward pass. Real users have lost time to exactly this; it is a ten-second
check of the `with` block's extent.

The subtler half of the rule: moving such launches *outside the `with`
block* is not sufficient if they still run **before `tape.backward()`**. The
backward pass replays adjoints against the arrays' *current* contents, so a
state reset or in-place overwrite between recording and backward clobbers
values the replay needs — while silencing the tracker (the write is no
longer recorded, so nothing warns). Housekeeping that writes taped arrays
belongs strictly *after* `tape.backward()`.

### 4. In-place multiplication / division

Docs: "In-Place Math Operations". `+=` and `-=` are differentiable;
**`*=` and `/=` are not** — they produce wrong backward results, and the
only warning is emitted at codegen time when `wp.config.log_level =
wp.LOG_DEBUG`, so users never see it. Rewrite as `a[i] = a[i] * b[i]`... which
is itself an overwrite — the correct fix is writing the product to a distinct
output array.

### 5. Vector / matrix / quaternion component reassignment

Docs: "Vector, Matrix, and Quaternion Component Assignment". Each component
of a locally constructed vec/mat/quat may be *assigned* (`v[0] = x`) at most
once; after that, only `+=`/`-=` updates are safe. A second direct assignment
invalidates gradients for the whole object. Escape hatch:
`wp.config.enable_vector_component_overwrites = True` (significant compile
time cost).

### 6. Dynamic loops

Docs: "Dynamic Loops". Dynamic (non-unrollable) loops are **not replayed or
unrolled in the backward pass**, so local variables hold their *final* values
when the adjoint runs. Three documented failure modes:

- Multiplicative accumulation (`prod *= x[i]`) — wrong adjoints.
- Any adjoint that depends on a loop-carried intermediate.
- A local computed in the loop and used after it (e.g., a norm) is *zero* at
  adjoint time → `inf`/NaN gradients.

Loops with only `+=`/`-=` accumulation are safe. Workarounds, in order of
preference: make the trip count static so it unrolls (respect `max_unroll`,
default 16, per-module option); store intermediates in an array indexed by
iteration; move the loop body into a `@wp.func` (forces replay — but
documented as valid only for simple add/subtract accumulation).

### 7. `retain_grad=True` double-counting

`retain_grad` disables the gradient-zeroing that enforces final-write-wins.
On an array whose elements are written more than once, it double-counts.
Only safe on arrays written at most once.

### 8. Accidental gradient truncation (tape-per-step training loops)

The highest-value structural check, because it is invisible to every numeric
tool run at the wrong scope. The pattern:

```python
for frame in range(num_frames):          # inside one training iteration
    tape = wp.Tape()
    with tape:
        ...substeps...; loss_kernel(...)
    tape.backward(loss)                   # backward INSIDE the step loop
    accumulate(param_grad, params.grad)   # grads summed across frames
    carry_state_forward()                 # outside any tape
apply_optimizer(param_grad)
```

Every per-frame gradient here is correct — and their sum is still the wrong
gradient for the objective being optimized. Each backward treats the window's
incoming state as a constant, so all paths where the parameters influence
*later* windows through the carried state are dropped. For objectives where
accumulated state matters (locomotion, control, anything episodic), the
dropped terms dominate. The frame loop is just the most common shape: the
same error occurs with any partitioning of one objective into per-window
tapes — per-stage tapes in a multi-phase solve, per-chunk tapes over a long
sequence — whenever state or coupling crosses the window boundary. Symptoms:
per-window FD matches beautifully; full-horizon FD does not; the optimizer
converges to a stationary point of the truncated objective while the
reported loss stalls or worsens.

The fix is structural:

- **One tape over the entire horizon**: `with tape:` wraps *all* windows; a
  single `tape.backward(loss)` after the loop; the optimizer consumes
  `param.grad` directly — no cross-window accumulation buffer.
- **One more state buffer than total steps** (`total_steps + 1`), all
  distinct. Reusing per-window state buffers inside the full-horizon tape is
  a write-after-read overwrite.
- **Per-step allocations for every taped intermediate** (control inputs,
  activations, per-window loss contributions) — the same rule; a single
  reused scratch array overwritten each window corrupts the backward pass.
- **Fresh initial state each training iteration** — episodes should not
  start where the previous optimization step's rollout ended.
- Accumulate the scalar loss across windows *on the tape* (e.g.,
  `wp.atomic_add(loss, 0, window_term)`).

All of these are one principle: **every write inside one tape needs its own
memory, and the tape must span the horizon the objective spans.** The memory
cost is real (state count scales with the full horizon); when it does not
fit, the correct escape is gradient checkpointing
(`warp/examples/optim/example_fluid_checkpoint.py`), never silent per-frame
truncation. If the user knowingly wants truncated BPTT as a cheaper
approximation, that is a legitimate choice — but it must be named,
and "gradients verified" claims must then be scoped to the window.

The same truncation bug occurs in **solver space**: differentiating through
a partially converged iterative solver (fixed-point / Jacobi / relaxation
iterations inside the tape) optimizes the truncated map, not the
equilibrium. The trap signature is unique: FD and autodiff AGREE — both
differentiate the same truncated program — while a converged re-simulation
of the "optimal" result misses the target. Fixes, best first: (1) converge
the solve **outside the tape** (untaped, reusable buffers) and warm-start a
short taped iteration block from the fixed point — the taped steps map the
solution to itself, gradients become exact at the solution, and tape memory
is unchanged (a truncated-Neumann form of implicit differentiation; the
right answer when memory motivated the truncation in the first place);
(2) if memory allows, converge the taped solve itself; (3) full implicit
differentiation via an adjoint solve at the fixed point.

### 9. Under-determined forward choices (ties, ordering, nondeterminism)

A principle, not a pattern: **gradients are defined relative to the forward
pass that actually ran.** Autodiff differentiates the branches and discrete
selections the execution took (closest-element queries, argmin/argmax,
conditionals), not the mathematical ideal. Wherever those choices are
under-determined — exact ties, traversal order, atomic scheduling — different
implementations, or different runs, can legitimately make different choices
and produce **different-but-equally-valid gradients** at those points. The
objective is only piecewise smooth there; any subgradient is correct, and FD
cannot adjudicate between them (see `references/verification.md`).

Consequences for diagnosis:

- A gradient mismatch (vs a reference implementation, or run-to-run) that is
  **sparse, data-dependent, and localized to choice boundaries** — while
  forward outputs agree to float precision — is ambiguity, not corruption.
  Probe: does the choice differ at exactly the mismatching elements (e.g., a
  near-zero margin between best and second-best candidates)? Does nudging
  inputs off the boundary dissolve the mismatch? If so, report it as expected
  non-smoothness and surface the real question to the user: is bitwise
  gradient parity actually a requirement, or is any valid subgradient
  acceptable for their optimization?
- Mismatches that are dense, or sit far from any choice boundary, are
  corruption — go back to the tier-1 checks.
- Benign float jitter is the trivial sibling: atomics make reduction order
  nondeterministic, so run-to-run gradient noise at float precision is
  normal; O(1) differences are not.

The same principle in bug form: the backward pass must see the **same
execution** the forward took. Saving and reusing forward selections in a
custom backward, `@wp.func_replay` for side effects, and keeping taped
arrays unmutated until `tape.backward()` are all instances — violating any
of them means differentiating a function that never ran.

### 9b. NaN from the unselected branch of `wp.where`

`wp.where`/`wp.select` are selects, not branches: BOTH operand expressions
are evaluated in the forward pass, and BOTH adjoint chains run in the
backward pass. A guard like ``wp.where(z > 20.0, z, wp.log(1.0 +
wp.exp(z)))`` keeps the forward value finite everywhere, but the unselected
``exp(z)`` still overflows once z exceeds ~88.7 (float32); its adjoint then
contributes ``inf * 0 = NaN``, poisoning the gradient — while every forward
value stays correct and any test data below the overflow threshold passes.
Real ``if``/``else`` control flow in kernels does not have this property
(only the taken branch's adjoint executes). Fixes, best first: (1)
reformulate stably so no operand can overflow — e.g. ``softplus(z) =
wp.max(z, 0.0) + wp.log(1.0 + wp.exp(-wp.abs(z)))`` removes the guard
entirely; (2) use a real ``if``/``else``; (3) clamp the unselected
operand's input (the double-where trick) or supply a custom
``@wp.func_grad``. Prefer removing the hazard over guarding it.

### 9c. Piecewise-constant ops: zero gradient is *correct* — use a surrogate

Ops whose output is locally flat — `wp.round`, `wp.floor`, `wp.ceil`,
`wp.trunc`, `wp.sign`, integer casts, comparisons, quantize/threshold
stages built from them — have derivative zero almost everywhere. Autodiff
returning exactly zero gradients through such a stage is **mathematically
correct, not a bug**: there is no corruption to find, and no taping-pattern
fix will help. Two things make this case easy to misread:

- Finite differences *disagree* with autodiff here, loudly: an FD probe
  steps across quantization boundaries and reports staircase secants
  (large, step-size-dependent values) while autodiff reports zero. That
  FD-vs-AD mismatch is expected non-smoothness, not evidence of a broken
  backward pass — do not spend time hunting corruption. (Contrast §9:
  there FD cannot adjudicate; here FD is simply measuring the secant of a
  staircase.)
- The training symptom is identical to a severed `requires_grad` chain
  (§2): loss constant, upstream gradients exactly zero. Distinguish them
  by locating where gradients die — at a flat op, it is this case.

The fix is a **surrogate gradient**, not a repair: keep the exact forward
(the quantization is usually the point) and supply the backward you want
to optimize with. The standard choice is the straight-through estimator
(STE) — pass the incoming adjoint through as if the op were the identity —
via `@wp.func_grad` on a wrapped `@wp.func` (template in
`references/custom-gradients.md`); clipped or scaled variants slot in the
same way. Report the situation honestly: the zero gradients were correct,
the surrogate is a modeling decision the user should own.

## Tier 2 — custom gradient machinery

Check both directions (details in `references/custom-gradients.md`):

- **Present and misused**: `@wp.func_grad` adjoints that assign instead of
  accumulate (`wp.adjoint[x] = ...` vs `+=`), signature mismatches,
  replay functions that don't mirror the forward signature, generic-typed
  forwards (unsupported — raises at registration).
- **Absent but required**: forward passes that evaluate non-differentiable
  points (`sqrt`, `length`, `normalize` at zero; division) need a
  `@wp.func_grad`; kernels with side effects the replay would repeat
  (atomic counters) need `@wp.func_replay`; `@wp.func_native` snippets need
  an `adj_snippet` to participate in backward at all.

Note a correctly written `@wp.func_grad` can still be defeated by a Tier-1
bug around it — the gradient-clipping case study in
`references/case-studies.md` is exactly this.
