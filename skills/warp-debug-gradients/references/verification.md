# Verification tooling

Everything here assumes a shrunk reproduction that runs in seconds. Doc
anchor: "Debugging Gradients" in the Differentiability guide
(https://nvidia.github.io/warp/stable/user_guide/differentiability.html).

## Array overwrite tracker

```python
import warp as wp
wp.config.verify_autograd_array_access = True  # BEFORE kernels load/launch
```

Then run the forward pass **under an active `wp.Tape()`** and capture
warnings ("written to but has already been read from..."). Two kinds: inter-
launch (at runtime) and intra-kernel (at codegen). They arrive via Python
logging, not exceptions. Caveats that matter in practice:

- **Requires an active tape** — the runtime check lives in `wp.launch`'s
  tape-recording path. No tape, no warnings.
- **Blind to arrays stored inside Warp structs** (documented limitation). A
  clean run does not clear struct-held state.
- **Disables kernel-cache reuse and forces JIT recompilation of the user's
  kernel modules** (not a rebuild of Warp's native library) — expect a slow
  first run; turn it off when done.
- Operations recorded via `Tape.record_func` are only tracked if the code
  calls `array.mark_read()`/`mark_write()` manually. Plain `wp.copy` *is*
  instrumented (including via `wp.clone` / `array.assign`). On Warp 1.17+
  its warning ("written to by an array copy at file:line") names the
  offending call site; on earlier versions the copy warning is
  **location-free** (it prints only the array contents) — attribute it
  yourself by searching for `wp.copy`/`.assign(` calls whose destination was
  previously a kernel input; users stare straight past the anonymous form.
  Kernel-launch warnings on all versions cite the *kernel definition* site,
  not the launch site — users often misread that as "the bug is at that
  line" when the overwriting launch is elsewhere. `.zero_()` / `.fill_()`
  writes are not tracked at all; audit those by hand.
- Read-flag lifetime: on Warp 1.17+, `tape.backward()` clears the read
  flags of the arrays it consumed, so a fresh-tape-per-iteration training
  loop tracks cleanly across iterations; a warning that fires is therefore
  meaningful — either an intra-tape write-after-read or a write while
  another tape's backward is still *pending*. (`tape.reset()` also clears
  flags; `tape.zero()` does not.) **Version caveat (Warp < 1.17):** flags
  were sticky until an explicit `tape.reset()`, so multi-iteration runs
  accumulate false-positive warnings — including on arrays that were just
  freshly zero-initialized, which reads as nonsense. On those versions,
  scope the tracker to a single iteration or call `tape.reset()` between
  windows, and discount cross-iteration warnings accordingly.
- Expect a flood on real programs: triage by deduplicating (array, argument,
  kernel) triples first, and treat warnings on arrays your fix hypothesis
  does not explain as either false positives (older-Warp stale flags) or
  additional findings — do not fixate on the first warning printed.

## End-to-end finite-difference check

This is the ground-truth measurement. The primary tool is
`wp.autograd.gradcheck` applied to a **Python callable wrapping the whole
forward pass**, not just a kernel:

```python
import warp.autograd

def forward(theta: wp.array, loss: wp.array):
    # run the full pipeline: inference, sim steps, loss kernel
    ...

ok = wp.autograd.gradcheck(
    forward, inputs=[theta], outputs=[loss],
    eps=1e-3, atol=1e-3, rtol=1e-2,          # scale eps to the parameters
    max_inputs_per_var=32,                    # sample large inputs
    raise_exception=False, show_summary=True,
)
```

Requirements: differentiated arrays must be *arguments* of the callable with
`requires_grad=True` (module-level/closure state the forward reads won't be
perturbed); inputs precede outputs; structs are not supported.
On Warp 1.17+, `restore_inputs=True` (the default) snapshots the callable's
Warp array inputs and restores them before every evaluation, so forwards
that integrate state in place are still checked from pristine values.
**Version caveat (Warp < 1.17):** the parameter does not exist — gradcheck
evaluates the callable repeatedly on the *same* input arrays, so for a
forward that mutates its inputs in place the comparison starts from drifted
state and can silently false-pass (or false-fail). On those versions, either
make the callable clone its inputs at entry, or use the manual harness
below.

## Manual FD harness (fallback)

When the pipeline can't be expressed as a callable taking its differentiated
state as arguments (state buried in objects, host-side control flow between
launches, RNG), fall back to the manual pattern: a `run_loss(theta_np) ->
float` function that **rebuilds all state from scratch** per call (no array
objects shared between evaluations), central differences over sampled
elements, compared against `theta.grad` copied *before* any
`tape.zero()`/`reset()`. Scale `eps` to the parameter magnitude — too small
drowns in sim noise.

Judge agreement with a relative criterion like
`|ad - fd| <= atol + rtol * |fd|` (Warp's gradcheck uses `atol=1e-3`,
`rtol=1e-2` by default) and report the actual numbers, not just pass/fail.
For chaotic or contact-heavy sims, shrink the time horizon until FD is stable
before trusting a disagreement; FD noise on a stiff sim is not a gradient bug.
For a **stochastic forward** (sampled noise, dropout-style masks, randomized
augmentation), pin the RNG so every FD evaluation replays the *same* random
draws — re-seed inside `run_loss` (or hoist the sampling out and pass the
draws in as data). Unpinned, each perturbation samples fresh noise and FD
measures the noise, not the derivative: it will "fail" against perfectly
correct autodiff gradients.
Likewise, FD disagreement at a **non-smooth point** (an argmin tie, a branch
boundary) is not a bug indicator: central differences straddle the kink and
average the two branch derivatives, matching neither valid subgradient —
check the tie margin at mismatching elements (quick-checks §9).

**The FD reference is defined by the user's objective, never by the tape
structure.** Perturb the parameters the optimizer updates, rerun the FULL
pipeline (every step, every frame), difference the loss the user actually
reports, and compare against the gradient the optimizer actually consumes
(including any accumulation across inner backward calls). If agreement only
appears after you narrow the comparison — freezing carried-in state, checking
one frame's tape at a time — that narrowing is not a validation technique, it
is a *finding*: the pipeline computes a truncated gradient of a different
objective (quick-checks §8). Report the full-horizon disagreement as the
defect; do not redefine the ground truth until it matches the implementation.
(Shrinking the horizon for FD *stability* is different: shrink the problem
for both the FD and AD sides equally, never just the reference.)

## wp.autograd utilities (per-kernel checks)

`import warp.autograd` (explicit import required). Public API: `gradcheck`,
`gradcheck_tape`, `jacobian`, `jacobian_fd`, `jacobian_plot` — signatures and
narrowing knobs are in `warp/_src/autograd.py`. What they can and cannot
tell you:

- `gradcheck` on a single kernel validates *that kernel's* adjoint math
  (input args must precede outputs; only `requires_grad=True` arrays
  checked; structs unsupported).
- `gradcheck_tape` runs `gradcheck` on **each recorded launch in isolation**.
  It is structurally blind to inter-kernel overwrites and taping-pattern bugs
  — the exact class that dominates real-world failures. All kernels passing
  `gradcheck_tape` while the end-to-end FD check fails is a *strong positive
  signal* that the bug is in the taping pattern (overwrites, aliasing,
  requires_grad breaks), not in any kernel.
- `gradcheck_tape` silently skips kernels with `enable_backward=False` and
  anything recorded via `Tape.record_func`.

## Localization by bisection

When the signature and checklist leave several candidates:

1. Truncate the sim to K steps (or K solver iterations) and rerun the
   end-to-end FD comparison. Binary-search the smallest K where FD and
   autodiff diverge; the step that tips it names the kernel/pattern.
2. Substitute the loss with a trivial one (e.g., sum of an intermediate
   array) to test progressively shorter prefixes of the pipeline.
3. Cross-check the suspect kernel alone with `wp.autograd.gradcheck` on
   synthetic inputs.

## Tape visualization

```python
tape.visualize("tape.dot")   # then: dot -Tsvg tape.dot -o tape.svg
```

Arrays render green when `requires_grad=True`, grey otherwise — a fast way to
spot a broken differentiability chain in a long pipeline. The graph is only
as good as the launch metadata: kernels launched without `inputs=`/`outputs=`
arguments lose structure. `array_labels={arr: "name"}` makes big graphs
readable.
