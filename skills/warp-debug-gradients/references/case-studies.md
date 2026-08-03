# Case studies

Two real debugging sagas. Both took experienced developers days; both were
one- or two-line fixes; both were instances of the write-after-read rule
wearing a disguise. Read them to calibrate how *indirect* the symptoms are —
and notice that in both cases, measurement (the overwrite tracker plus an FD
comparison) would have found in minutes what code-reading intuition missed
for days.

## Case 1: the walker that couldn't learn — two bugs, one shallow, one deep

A soft-body walker (a tet-mesh "bear" driven by a small phase network)
ported from a working PyTorch implementation. Symptom: gradients grew
unbounded across training iterations. The developer spent days tuning
stiffness/damping knobs, re-exporting the asset with a hardened coordinate
system, and reworking the loss function — the classic pattern of debugging
the *physics* when the bug is in the *taping*.

**The surface bug — state aliasing.** The training loop kept a list of
per-substep states, taped each frame separately, and carried state between
frames with:

```python
self.states[0] = self.states[-1]   # Python REBIND, not a copy
```

After the first frame, `states[0]` and `states[-1]` are the same object: on
each frame's tape, substep 0 *reads* the very arrays the final substep
*writes*. Write-after-read; the backward pass replays the first substep's
adjoint against clobbered values. The forward sim is bit-for-bit identical
either way, which is why no amount of physics tuning could expose it. The
overwrite tracker flags this immediately.

**The deep bug — accidental gradient truncation (quick-checks §8).** Fixing
the aliasing makes each frame's gradient *correct for that frame with the
carried-in state frozen* — and training still fails, because the loop
structure itself (tape per frame, backward per frame, gradients summed
across frames, optimizer applied at the end) optimizes the wrong objective.
Every path where the network weights influence *later* frames through the
carried state is dropped, and for locomotion those paths are the whole
point. A full-horizon FD check fails against the accumulated gradient even
after the aliasing fix; that disagreement IS the diagnosis, not noise to be
explained away by narrowing the check.

**The real fix was a restructure** (this is the shipped, working version):

- one tape wraps the *entire* run; a single `tape.backward(loss)` after all
  frames; the optimizer consumes `weights.grad` directly;
- `sim_steps * sim_substeps + 1` distinct state buffers — every substep
  write inside the tape lands in fresh memory;
- every taped intermediate becomes a per-frame list (`phases[i]`,
  `activations[i]`, `coms[i]`) instead of one reused scratch array;
- each training iteration starts from a **fresh** initial state
  (`model.state()`), not wherever the previous rollout ended — and note this
  rebind is *correct* (post-backward, outside the tape, fresh allocation),
  while the aliasing rebind was the bug: rebinding is not the sin, aliasing
  two live tape participants is;
- the example now asserts its own success: loss must decrease across
  training (the close-the-loop check).

Only after this restructure did the gradient train the weights to serve the
mesh over the full duration of the experiment.

Lessons: treat Python `=` between array/state objects as an aliasing red
flag; "unbounded growth across iterations" does not always mean a missing
`tape.zero()`; per-frame FD agreement is NOT gradient correctness — the FD
reference is the user's objective over the full horizon; and when the
correct fix is structural, prescribe and implement the restructure rather
than a small change that silences the symptom.

## Case 2: the differentiable copy that wasn't (in context)

A team making an iterative cloth/soft-body solver differentiable. Per-kernel
gradient checks all passed. A gradient-clipping stage was added via a
custom-adjoint identity function — and had *no observable effect* on the
exploding gradients, even when hard-coded to zero the adjoint. That
impossibility was the key clue: the custom adjoint was running correctly,
but its output was being destroyed elsewhere.

The stage looked like:

```python
clamped_q = wp.zeros_like(state.particle_q, requires_grad=True)
wp.launch(clip_kernel, dim=n, inputs=[state.particle_q], outputs=[clamped_q])
wp.copy(state.particle_q, clamped_q)     # <- the bug
```

`wp.copy` is differentiable and tape-recorded — but its *destination* is
`state.particle_q`, which the launch on the previous line just **read**.
Write-after-read again: the copy clobbers the values the clip kernel's
adjoint needs. Fix: rebind instead of copying in place:

```python
state.particle_q = clamped_q
```

The same team had earlier chased warnings from a ping-pong state driver
(reusing two state objects alternately across steps of one tape) and had
been suspicious of the per-iteration rebinding pattern
(`out = wp.clone(inp); solve(inp, out); inp = out`) — which is actually
*correct*, since every write lands in fresh memory. Distinguishing the
benign rebind from the malignant in-place copy is precisely the judgment
this skill's checklist encodes.

**Version note (Warp 1.17+):** the copy adjoint now routes gradients through
the intervening stage and consumes the destination's gradient, so this
*exact* failure — an identity-forward stage whose write-back preserves
values — computes correct gradients on 1.17+ (the tracker still warns, and
rebinding is still the clean pattern). The write-back remains genuinely
corrupt on all versions when the copied values differ from what earlier
launches read, e.g. a clamp or projection that engages.

**Post-script — the pattern everyone blessed was also broken.** The team's
per-iteration idiom (`out = wp.clone(q_in)`; solve kernel updates only the
current color's entries of `out`; `q_in = out`) was reviewed in discussion
and judged valid — every write lands in fresh memory, and the clone
legitimately carries the untouched entries. Conceptually correct; but on
Warp versions where copy adjoints overwrite instead of accumulate
(quick-checks §1 version caveat), this exact shape silently produces
wrong gradients — measured at 45-90% of their true values in this case
study's solver, the exact factor depending on how much adjoint the
partially rewritten destination still carried — after every other bug
had been fixed, with no warning of any kind. A one-thread FD probe of the
full pattern (not per-kernel gradcheck, which passes) is what exposes it.

Lessons: "differentiable op" does not mean "safe anywhere" — differentiable
copies still overwrite; a custom gradient with no effect means the corruption
is *around* it, not in it; per-kernel gradcheck passing while end-to-end
gradients are wrong points at the taping pattern between kernels; and
expert sign-off on a pattern's *concept* does not verify the framework's
*implementation* of it — only a numeric check does.
