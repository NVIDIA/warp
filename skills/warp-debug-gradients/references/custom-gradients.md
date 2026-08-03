# Custom gradient machinery

Doc anchor: "Custom Gradient Functions", "Example 2: Custom Replay Function", and
"Custom Native Functions" in the Differentiability guide
(https://nvidia.github.io/warp/stable/user_guide/differentiability.html).

Three decorators, three distinct jobs:

| Decorator | Job |
|---|---|
| `@wp.func_grad(fwd)` | Replace the auto-generated adjoint of a `@wp.func` |
| `@wp.func_replay(fwd)` | Replace the *forward replay* executed during the backward pass |
| `@wp.func_native(snippet, adj_snippet=None, replay_snippet=None)` | C++/CUDA snippet; `adj_snippet` is its adjoint, `replay_snippet` its replay |

## When one is REQUIRED (absence is the bug)

- **Non-differentiable points**: `wp.sqrt` at zero, division by values that
  reach zero — the auto-adjoint produces `inf`/NaN. The canonical fix is the
  docs' `safe_sqrt` example: a `@wp.func_grad` that guards the singular
  point (optionally implemented via `wp.grad(wp.sqrt)(x) * adj_ret` away
  from it). Nuance worth checking before writing a guard: some builtins are
  already internally guarded — `wp.length`'s native adjoint has an epsilon
  check at zero, while a hand-rolled `wp.sqrt(wp.dot(v, v))` does not. So
  "replace the hand-rolled norm with `wp.length`" can be the whole fix, and
  a NaN that appears with one spelling but not the other localizes to the
  unguarded builtin.
- **Side effects the replay repeats**: the backward pass *re-runs* the
  forward (replay phase) to rebuild intermediates. A `wp.atomic_add` used as
  a counter/allocator increments **again** during replay, so adjoints get
  scattered to wrong indices. Fix with `@wp.func_replay` that reads the
  already-computed result instead of re-incrementing (docs: "Example 2:
  Custom Replay Function").
- **`@wp.func_native` without `adj_snippet`**: the snippet contributes no
  adjoint at all — downstream gradients silently vanish through it.
- **Piecewise-constant forwards** (`wp.round`, `wp.floor`, `wp.sign`,
  casts, thresholds — quick-checks §9c): the auto-adjoint's zero is
  *correct*, so this is the one case where the custom gradient deliberately
  lies about the math. Straight-through estimator template — keep the exact
  forward, pass the adjoint through as identity:

  ```python
  @wp.func
  def ste_round(x: float) -> float:
      return wp.round(x)

  @wp.func_grad(ste_round)
  def adj_ste_round(x: float, adj_ret: float):
      wp.adjoint[x] += adj_ret  # identity backward; clip/scale here for variants
  ```

  Call `ste_round` in the kernel instead of `wp.round`. The surrogate is a
  modeling decision — say so in the report rather than presenting it as a
  bug fix.

## When one is PRESENT, check for misuse

- **Assign vs accumulate**: the adjoint must *accumulate* into
  `wp.adjoint[arg] += ...`. A plain `=` clobbers contributions from other
  uses of the same variable.
- **Signature discipline**: `func_grad` takes the forward inputs plus the
  adjoints of the outputs and returns nothing; `func_replay` must mirror a
  forward overload exactly; generic (`Any`-typed) forwards are unsupported.
  Mismatches raise at registration, so if the code imports cleanly this is
  not the bug.
- **`wp.grad()` limits**: only participates in the backward pass when used
  *inside* a `@wp.func_grad`; in regular kernel code it is forward-only
  (treated as a constant in backward, with a warning).
- **A correct custom gradient defeated by its surroundings**: a custom
  adjoint that should visibly change gradients (clipping, scaling, zeroing,
  reweighting) has no observable effect. The adjoint is usually running
  fine — a Tier-1 bug in the dataflow around its call site (an overwrite of
  an already-read array is the classic) corrupts or bypasses the chain.
  Verify the taping pattern before doubting the custom adjoint; see the
  second case study in `references/case-studies.md`.

Two cheap probes for "is my custom adjoint actually in the chain?":

- **Zero probe**: hard-set the custom adjoint's output to zero
  (`wp.adjoint[x] += type(0)` only) and rerun the backward pass. Upstream
  gradients must change (generally toward zero). If they are byte-identical,
  the adjoint's output is being destroyed or bypassed downstream of it —
  hunt for an overwrite around the call site, not a mistake inside it.
- **Print probe**: a `wp.printf` inside the `@wp.func_grad` confirms it
  *executes* during backward — but execution is not influence; combine with
  the zero probe before concluding anything.
