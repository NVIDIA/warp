---
name: warp-debug-gradients
description: >-
  Use to diagnose and fix incorrect gradients in differentiable Warp programs.
  Anything trained, optimized, calibrated, or fit through Warp kernels depends
  on wp.Tape gradients, so treat any misbehavior of such a workflow as a
  gradient problem until proven otherwise — use this when training diverges or
  NaNs, won't train at all, stalls or plateaus above the expected loss,
  converges to a wrong or biased answer, is worse than a reference
  implementation, works at small scale but fails at production scale, or fails
  a QA/validation recheck. Also for explicit symptoms — exploding, NaN/inf,
  zero, or subtly wrong gradients, suspected wp.Tape/backward issues, gradcheck
  failures — but users usually describe only the surface symptom ("the sim
  explodes", "the fit gets dragged toward outliers") without mentioning
  gradients: make that leap. Not for forward-only Warp work, build/install
  problems, or autograd issues in other frameworks without Warp.
license: Apache-2.0
compatibility: Requires a working NVIDIA Warp installation (>= 1.13 minimum; >= 1.17 recommended for reliable verification — copy-adjoint accumulation, overwrite-warning call sites, read-flag lifetime, and gradcheck's restore_inputs changed in 1.17 and are version-caveated in the references; on older versions a fixed bug class still exists and some tools need workarounds). Diagnosis runs the user's reproduction, so a functioning device (CPU or CUDA) is needed.
metadata:
  author: "Warp Team <warp-python@nvidia.com>"
  version: "0.1.0"
  tags:
  - warp
  - autodiff
  - gradients
  - differentiable-simulation
  - debugging
  upstream: https://github.com/NVIDIA/warp
---

# Debugging Gradients in Warp

Gradient bugs in Warp are almost never math bugs. The forward simulation looks
perfectly healthy while the backward pass silently reads clobbered values,
skips arrays, or double-counts adjoints. Users routinely burn days tuning
physics knobs, loss functions, and assets when the real cause is a two-line
taping-pattern fix. Your job is to find that fix with evidence, not intuition.

The single most important discipline: **measure before hypothesizing**. It is
cheap for you to run a shrunk reproduction and compare autodiff against finite
differences. The *way* the gradient is wrong (its signature) prunes the
hypothesis space far faster than reading code ever will. Do not start
proposing fixes from code reading alone — plausible-looking diagnoses of
differentiability bugs are very often wrong, and an unverified "fix" that
happens to perturb the numbers wastes everyone's time.

## When to Use This Skill

Anything trained, optimized, calibrated, or fit through Warp kernels flows
through `wp.Tape` gradients — so when such a workflow misbehaves, gradients
are the prime suspect even if the user never says the word. Activate on the
symptoms users actually report: training that diverges, NaNs, or does
nothing; loss that stalls or plateaus above where it should; fits that
converge to a wrong or biased answer or are worse than a reference
implementation; pipelines that work at small scale but fail at production
scale or fail a QA recheck. Also activate on explicit gradient symptoms —
exploding, NaN/inf, zero, or subtly wrong gradients,
`wp.autograd.gradcheck` failures, suspected `wp.Tape`/backward issues — and
when the user asks whether their gradients can be trusted.

Do not activate for forward-only Warp work (kernel authoring, rendering,
performance tuning), Warp build or installation problems, autograd questions
in other frameworks with no Warp involvement, or pure performance work on a
backward pass whose gradients the user has already validated.

The canonical background is Warp's own documentation — consult the relevant
section before diagnosing in its territory (online at
https://nvidia.github.io/warp/stable/; in a Warp source checkout the same content
is under `docs/user_guide/`; pip installs do not include it):

- The "Differentiability" guide — especially "Array Overwrites", "Debugging
  Gradients", "Array Overwrite Tracking", and "Limitations and Workarounds"
  (in-place math, component assignment, dynamic loops).
- The FAQ, section "Differentiation and Interoperability" — what state a
  tape does and does not preserve, and checkpointing.

## Prerequisites

Executing this skill assumes all of the following; if one is missing,
surface that to the user instead of improvising around it:

- The user's script (or a faithful reproduction) is available in the
  workspace, runnable, and modifiable — diagnosis executes it repeatedly and
  edits it to apply fixes.
- This skill's `references/` files (quick-checks.md, verification.md,
  custom-gradients.md, case-studies.md) accompany it and are consulted at
  the steps that cite them.

## Instructions

1. **Note the user's Warp version first** (`wp.__version__` or the banner
   Warp prints at init). Several verification behaviors changed in Warp
   1.17 — copy-adjoint accumulation, overwrite-warning call sites, read-flag
   lifetime, `gradcheck`'s `restore_inputs` — and the references mark each
   with a version caveat. On Warp < 1.17, a whole bug class exists that
   later versions fixed (quick-checks §1's version caveat), and some tools
   need workarounds.

2. **Reproduce and shrink.** Get the user's script running, then cut it
   down:
   fewer particles/elements, fewer time steps, fewer optimizer iterations,
   CPU device if the sim allows. You need a repro that runs in seconds,
   because you will run it many times. Keep the structure (number of kernels,
   the taping pattern, buffer reuse) intact — that is where the bug lives.
   Shrinking the *physics* is fine; restructuring the *dataflow* is not.

   If the script cannot be made to run (missing dependencies, broken code),
   report the blocking issue as the deliverable and stop — do not proceed to
   verify a program that never ran.

3. **Instrument and establish ground truth** (details and templates in
   `references/verification.md`):
   - Set `wp.config.verify_autograd_array_access = True` before module load
     and rerun under an active tape. Capture every warning. This catches the
     single most common bug class (write-after-read overwrites) nearly for
     free. Know its blind spots: it needs a tape, it cannot see arrays stored
     inside Warp structs, and it disables kernel caching (expect a kernel
     rebuild — JIT module recompilation only, not a rebuild of the native
     library). If the tracker runs clean but gradients are still wrong,
     specifically check for in-place mutations of arrays held inside Warp
     structs (quick-checks §1 and Limitations) before trusting the clean
     result.
   - Run one **end-to-end finite-difference check**: wrap the full forward
     pass (sim steps + loss) in a Python callable and hand it to
     `wp.autograd.gradcheck` with the true optimization inputs — it compares
     the autodiff gradient against central differences, restoring array
     inputs between evaluations (Warp 1.17+) so in-place-mutating forwards
     are checked from pristine state; on older Warp use the manual harness
     in `references/verification.md`. The reference is the user's *actual
     objective over
     the full horizon*, compared against the gradient the optimizer *actually
     consumes* — never a narrower window (see `references/verification.md`).
     This confirms gradients are actually wrong (users are sometimes wrong
     about this — report "gradients are correct" findings honestly) and
     yields the error signature. The template in
     `references/verification.md` fixes the eps/tolerance choices and the
     seed-pinning a stochastic forward needs — do not eyeball pass/fail
     against floating-point or sampling noise. If the backward pass runs out
     of memory while establishing ground truth, apply the checkpointing
     pattern from "Edge case: out of memory" below before proceeding.

4. **Match the signature** against the table below to rank hypotheses.

5. **Scan the code against the known-pattern checklist**
   (`references/quick-checks.md`). This is fast for you — do it in the same
   pass, but let the signature decide which findings are plausible causes
   versus incidental smells.

6. **Localize if still ambiguous.** Binary-search the pipeline: truncate to K
   steps and find where FD and autodiff first diverge; run
   `wp.autograd.gradcheck_tape` to test each recorded launch in isolation.
   Remember gradcheck_tape validates kernels *individually* — it is
   structurally blind to inter-kernel overwrites, so a clean per-kernel pass
   plus a wrong end-to-end gradient points *at* the taping pattern, not the
   kernels. It also silently *skips* kernels compiled with
   `enable_backward=False` (see Limitations) — if any kernel in the pipeline
   sets that, a clean pass says nothing about it; verify it separately.

7. **Fix minimally, then re-verify** with the exact same FD harness that
   established the failure. A gradient fix without a before/after FD
   comparison is not a fix. Verify the exact program you are shipping — the
   fixed file as it stands, every line included — never a re-implementation
   of it in a diagnostic script: a rebuilt pipeline silently drops whatever
   you believed was irrelevant, and if that belief is wrong the verification
   passes while the shipped code stays broken. Mechanically: the harness
   must *import the fixed module (or execute the fixed file) and call into
   it* — the only code that may live outside the shipped program is the FD
   driver itself. Also rerun the overwrite
   tracker to confirm the warnings are gone. "Minimally" applies to the code
   diff, not the diagnosis:
   when the root cause is structural (e.g., accidental gradient truncation,
   quick-checks §8), the minimal *correct* fix is the restructure — do not
   substitute a smaller change that only silences the surface symptom.

8. **Close the loop on the user's original complaint.** Rerun their actual
   workflow (their script, their printed metrics). The job is done when the
   symptom they reported is resolved — an optimization that was "exploding"
   should now demonstrably *improve its objective*, not merely avoid NaN. If
   gradients verify correct at the full horizon but training still fails,
   that is a new signature-table entry, not a victory; keep diagnosing (or
   report the verified gradients and the remaining non-gradient cause, e.g.
   learning rate).

## Failure signatures

| Signature | Leading hypotheses |
|---|---|
| Gradients exactly zero | Missing `requires_grad=True` somewhere in the chain (note `wp.zeros` defaults to `False`; `zeros_like`/`clone` inherit from source); `enable_backward=False` at module/kernel level; loss array not connected to the tape; grads read after `tape.zero()`; a piecewise-constant op (`round`/`floor`/`sign`/cast/threshold) in the chain — there zero is *correct* and the fix is a surrogate gradient such as a straight-through estimator, not a bug hunt (quick-checks §9c); on Warp < 1.17, a tape-recorded copy/clone whose source has other downstream readers (see the version caveat in `references/quick-checks.md`) |
| Gradients grow without bound across optimizer iterations | Missing `tape.zero()`/`tape.reset()` between iterations; state-object aliasing that carries an in-tape overwrite across frames (case study 1) |
| Off by an exact small factor (2x, Nx) | Double accumulation: a duplicate launch recorded on the tape — note that since Warp 1.13 the store adjoint consumes the output gradient on first use, so a bare duplicate is inert unless the rewritten array has `retain_grad=True` (quick-checks §7) or the Warp version is older; overlapping tape scopes taping the same work twice. Also: a backward seed that does not match the stated objective — seeding a per-element loss adjoint with ones backpropagates the *sum*, exactly N× the *mean* objective's gradient |
| NaN or inf | Non-differentiable point evaluated in the backward pass (`wp.sqrt(0)`, `wp.length(0)`, `wp.normalize(0)`, division) — needs a custom gradient (`references/custom-gradients.md`) or, better, a stable reformulation; an overflow evaluated in the *unselected* branch of `wp.where` (a select, not a branch — quick-checks §9b); dynamic-loop local not recomputed during replay (documented to produce `inf`) |
| Subtly wrong, often worse with more steps/iterations | Write-after-read overwrite: `wp.copy` onto an already-read array, ping-pong buffers within one tape, Python rebinding that aliases two "different" states (case studies); in-place `*=`/`/=`; vector/matrix component reassignment; dynamic-loop intermediates; on Warp < 1.17, a recorded copy/clone that is not the last consumer of its source (version caveat in `references/quick-checks.md`) |
| Per-window FD agrees but full-horizon FD disagrees; or gradients "verified" yet the optimizer stalls or worsens the loss | **Accidental gradient truncation**: a tape-per-step loop with backward inside it and state carried between tapes optimizes a different objective than the one being reported (see quick-checks §8). The structural fix is one tape over the whole horizon with `total_steps + 1` distinct state buffers. The solver-space analog: a partially converged iterative solve inside the tape makes FD and autodiff agree on the wrong program — converge it outside the tape and warm-start the taped iterations (quick-checks §8) |
| Gradients disagree (vs a reference implementation or run-to-run) only on a sparse, data-dependent subset; forward outputs match to float precision | Under-determined forward choice at a non-smooth point (quick-checks §9): both answers can be valid subgradients, and FD cannot adjudicate at a kink. Check whether the discrete choice differs at exactly the mismatching elements before hunting corruption |
| FD and autodiff agree *at the full horizon* but optimization still fails | Not a gradient bug. Say so. Look at learning rate, loss landscape, physics stability — and report the verified-correct gradients as the finding |

## Examples

A representative session, end to end. A user reports "my cloth sim trains for
a while, then the loss creeps back up — tuning the learning rate doesn't
help." No mention of gradients; the leap is made because the workflow
optimizes through Warp kernels.

1. Their script runs 512 particles for 200 steps per iteration. Shrink to 16
   particles, 10 steps, CPU — repro now runs in ~2 s and shows the same
   creep.
2. `wp.config.verify_autograd_array_access = True` under the tape prints:
   `array ... was read from kernel integrate and is now being written to by
   kernel integrate` — a write-after-read overwrite.
3. End-to-end `wp.autograd.gradcheck` on the shrunk repro: max relative error
   0.4 against finite differences. Gradients are confirmed wrong, with the
   "subtly wrong, worse with more steps" signature.
4. The signature row plus quick-checks §1 point at buffer reuse inside one
   tape: the sim steps `state_a → state_b → state_a`, ping-ponging two
   buffers, so the backward pass reads clobbered states.
5. Minimal fix: allocate `num_steps + 1` distinct state buffers recorded on
   the tape (physics untouched; only the dataflow changes).
6. Re-verify: same gradcheck harness now passes (max relative error 3e-4);
   the overwrite warning is gone; the user's full-size training run now
   decreases monotonically.

Report: root cause (in-tape buffer reuse), the evidence chain (warning +
before/after FD numbers), the two-line diff, and a pointer to the
"Array Overwrites" section of the Differentiability guide.

## Reporting

Lead with the root cause and the evidence chain: the FD-vs-autodiff numbers
that established the failure, the warning or localization step that found the
cause, the minimal diff, and the FD numbers after the fix. Name the
documentation section that covers the pattern so the user can read the
canonical explanation. If you checked patterns that came up clean (e.g., the
overwrite tracker found nothing), say so — it tells the user what has been
ruled out.

If the user is only asking *whether* their gradients are trustworthy, stop
after verification and report; apply fixes when they ask for fixes.

Preserve the evidence: leave the diagnostic scripts (FD harness, shrunk
repro) in the workspace and list them in the report instead of deleting
them — they are the reproducible half of the evidence chain, and the user
or a reviewer should be able to rerun the exact verification that
justified the fix. Never delete files you did not create.

## Limitations

The verification tooling has blind spots — a clean pass through any one
tool is not a clean bill of health (details in
`references/verification.md`):

- The overwrite tracker requires an active tape, cannot see arrays stored
  inside Warp structs, and disables kernel caching while enabled.
- `wp.autograd.gradcheck` does not accept struct inputs; wrap the forward
  in a callable over the underlying arrays. On Warp < 1.17 it does not
  restore mutated array inputs between evaluations (use the manual
  harness).
- `wp.autograd.gradcheck_tape` validates each recorded launch in
  isolation — it is structurally blind to inter-kernel overwrite bugs and
  silently skips kernels compiled with `enable_backward=False`.
- The `*=`/`/=` non-differentiability warning is emitted only at codegen
  time under `wp.LOG_DEBUG`, so its absence from a normal run means
  nothing.
- Warp has no built-in gradient checkpointing; long-horizon memory
  pressure needs the application-level pattern below.
- At non-smooth points (ties, kinks, argmin selections), finite
  differences cannot adjudicate between valid subgradients — FD-vs-AD
  disagreement there is not automatically a bug (quick-checks §9).

## Edge case: out of memory

If the backward pass fails to allocate (long simulations keep every
intermediate state alive on the tape), the fix is gradient checkpointing:
save periodic states, replay the segments between them during backward. Warp
has no built-in utility — applications implement it themselves. Use
`warp/examples/optim/example_fluid_checkpoint.py` as the reference pattern,
and see the FAQ's "Differentiation and Interoperability" section.

## Reference files

- `references/quick-checks.md` — the known-bug-pattern checklist with doc
  pointers and the caveats that make each pattern easy to miss.
- `references/verification.md` — tooling details: overwrite tracker setup and
  blind spots, end-to-end FD harness template, `wp.autograd`
  gradcheck/jacobian usage and caveats, tape visualization, bisection.
- `references/custom-gradients.md` — `@wp.func_grad`, `@wp.func_replay`,
  `@wp.func_native`: when they are required and how they are misused.
- `references/case-studies.md` — two real debugging sagas (state aliasing;
  differentiable-copy overwrite) showing how subtle the surface symptoms are.
  Read these when the checklist comes up clean — they calibrate what "subtle"
  means here.
