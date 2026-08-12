# Evaluation Guidance

Developer notes for the evaluation datasets. Both files use the agentskills.io
schema: a top-level object with `skill_name` and `evals[]`, where each case
carries `prompt`, `expected_output` (the reference answer), and `assertions`
(the ordered, checkable workflow steps). SkillEvaluator normalizes those onto
its internal `question` / `ground_truth` / `expected_behavior` fields, so the
grading semantics are unchanged from the earlier flat-array form.

- `evals.json` — the **CI-gated P0 smoke set** (6 tasks: 4 positive, 2
  negative). NV-CARPS CI runners cap evaluation at one hour and the dataset
  schema has no priority field, so the CI-gated file itself is kept small
  per the publishing guidance ("prefer P0 smoke coverage first").

  Note on negative cases: `expected_skill` is deliberately set to
  `"warp-debug-gradients"` (with `should_trigger: false`) on every negative
  task, in both files. The grader searches the trajectory for the named
  skill and fails the case if it activated; with `expected_skill: null`
  (what `create-eval-dataset` generates and the tier-3 docs describe) the
  check short-circuits to a pass without inspecting the trajectory. Do not
  let a future `create-eval-dataset --refine` revert these to `null`.
- `evals-full.json` — the **full suite** (33 tasks: 29 positive, 4
  negative) for release gates and manual runs. SkillEvaluator auto-discovers
  only `evals/evals.*` and has no dataset-selection flag; to run the full
  suite, copy the skill directory to a scratch location and replace its
  `evals/evals.json` with the contents of `evals-full.json` (swapping in
  place would change the shipped P0 gate and the dataset digest). The positives comprise the original validated fixture suite
  (multi-file scenarios in per-fixture subdirectories such as
  `files/walker/`) plus single-file adversarial additions.

## Environment requirements

- NVIDIA Warp >= 1.13 with a working compute device. Every number quoted
  below was validated against `warp-lang` 1.16.0.
- The tasks do not assume a device or an installed Warp, and the prompts say
  nothing about either. A runner that lacks Warp is a legitimate condition
  for this skill: SKILL.md tells the agent to bootstrap non-destructively
  and, failing that, to report the blocker as the deliverable. Do not add
  environment claims to the prompts. An earlier revision asserted "I'm on a
  CPU-only box with Warp 1.16 already installed", which is a statement the
  dataset cannot guarantee on an arbitrary runner.
- Every fixture nonetheless reproduces its symptom in seconds without a GPU,
  so a CPU-only runner is sufficient. Measured in a `python:3.12-slim`
  container with no CUDA driver: `warp-lang` installs in ~10 s and each
  fixture runs in 3-13 s. The post-fix numbers in `expected_output`
  reproduce there to the digits quoted: `huber_fit` a=+0.5226 b=+2.4299,
  `gain_train` 8.879e-15, `deblur` AD/FD 2.0010 before removing the
  duplicate launch and 1.0005 after, `springs` E=1.023235e+07 with
  AD/analytic 1.9988 before the `dldu` fix and 0.9994 after, `dac` loss
  6.203e-3 to 2.384e-7.
- `interop-012` additionally requires PyTorch (`warp_layer.py` /
  `torch_train.py`), which full-suite runs must provide. It reproduces
  without a GPU: with `torch 2.13.0+cpu` the shipped fixture settles into
  its documented limit cycle (loss 1.845e-2, `|x.grad|` 2.12e-2) and zeroing
  the Warp-side gradient takes it to 5.8e-14.
- No fixture selects a device. Every Warp allocation passes `device=None`,
  which resolves to Warp's default (CUDA when present, CPU otherwise), so
  the corpus follows whatever device the runner has. Keep it that way:
  every bug class these fixtures plant (overwrites, missing `requires_grad`,
  double accumulation, custom-adjoint math, gradient truncation) is a taping
  or codegen property and is device-independent. The one place CPU and CUDA
  genuinely differ is float atomic accumulation *order*, which changes the
  noise signature of a dead FD instrument in `cancel-008` without changing
  its verdict.
- Multi-file tasks declare every file they need in `files`, so the staging
  layout under `/workspace/input/` preserves the subdirectory structure and
  imports resolve.
- Toolkit caveat: NVRTC 13.1-13.3 miscompiles some generated backward
  kernels ("readonly and writeonly attributes incompatible"), which breaks
  the `structwar` fixture at module load. Build Warp against CUDA Toolkit
  13.0 (or a toolkit without that NVRTC defect) when running these evals on
  a CUDA device. CPU runs are unaffected.

## Why there is no `evals/config.yml`

This dataset deliberately ships no Harbor execution policy. The tempting
one is a `harbor.pre_agent_setup` that pip-installs Warp so both arms start
from a working device. It was tried and reverted; do not re-add it without
reading this.

Harbor renders `pre_agent_setup` into `[environment.healthcheck]` and runs
it as `bash -lc` under `set -euo pipefail`. Two consequences follow, and
both bite.

A non-zero exit aborts environment preflight before a single trial starts,
which fails the gate outright rather than degrading. On the NV-CARPS k8s
sandbox the install exited 1 and the gate reported "canonical Tier 3
coverage is invalid" — strictly worse than the NEUTRAL verdict the file was
meant to fix. Whether a runner can reach PyPI is not observable from this
repository, so betting the gate on it is not a bet this dataset should
make.

The login shell is the subtler problem. `bash -l` re-reads `/etc/profile`
and rebuilds `PATH`, so the `python3` a healthcheck resolves is not
necessarily the `python3` the agent uses. Measured in a `python:3.12-slim`
container: a non-login shell resolved `/opt/venv/bin/python3` while
`bash -lc` in the same container resolved `/usr/local/bin/python3`. A probe
that reports "Warp is missing" may only be describing the wrong
interpreter, and an install driven from that probe can land in an
environment the agent never sees.

The one skill that has passed Tier 3 in this sandbox,
`warp-compile-time-optimizer`, ships no `config.yml` either, and its own
assertions require executing Warp. That is the only direct evidence
available about how this environment behaves, and it points the same way.

If a runner genuinely lacks Warp, that is a real condition this skill is
written to handle, and the fix belongs in the runner image rather than
here.

## Case-file staging

Every case declares `files`. Cases that declare nothing get the legacy
fallback, which stages the entire `evals/files/` corpus (31 entries) into
`/workspace/input/`, so an agent debugging one fixture also sees the
thirty others. Negative cases declare `files: []` and stage nothing.

Prompts name the container path (`/workspace/input/<fixture>`) rather than
the repository path. With the repository path, every observed trial opened
with a failed `Read` followed by a `find` to locate the file, in both arms.

## P0 smoke subset (CI-gated runs)

NV-CARPS CI runners cap evaluation at one hour. For CI-gated runs, prefer
this six-task subset; it spans the skill's main behavior classes while
staying well inside the budget:

| Task                                   | Why it is in P0                                                              |
| -------------------------------------- | ---------------------------------------------------------------------------- |
| `huber-001-custom-grad-outlier-branch` | Real bug in custom-gradient math, fast fixture                               |
| `masked-006-zero-grad-hides-2x`        | Fixing the reported symptom hides a second bug only a fresh FD check finds    |
| `cancel-008-fp32-ulp-hides-2x`         | Dead FD gate hiding an exact-2x adjoint bug; code reading gives the wrong verdict |
| `quant-011-round-ste`                  | "No bug" honesty plus reformulation: the deliverable is a straight-through estimator |
| `neg-015-forward-kernel-perf`          | Negative activation: Warp-adjacent performance task                          |
| `neg-017-pure-torch-grads`             | Negative activation: gradient vocabulary without Warp                        |

`seed-002-sum-vs-mean-adjoint` was in P0 and is not any more. It is a good
case, but its root cause (`terms.grad.fill_(1.0)` seeding the sum rather
than the mean) is visible on a first read, and a measured no-skill baseline
applied the exact one-line fix without running anything. A case a strong
baseline solves by inspection cannot show uplift. `cancel-008` replaced it
because reading the code there produces a confident wrong answer: the FD
numbers look like proof that autodiff is broken, and the real bug is a
factor of two the dead instrument cannot see.

Excluded from P0 on runtime/dependency grounds, not value:
`walker-019` (the flagship discriminating task, conceptual gradient
truncation under a state-aliasing bug, and the one task no-skill baselines
reproducibly fail; the fixture itself runs in 3 s on CPU, but passing
requires a full restructure and a multi-minute agent session, so it is the
first task to promote into `evals.json` if the CI budget allows),
`chaos-007` (hundreds of taped launches per training step), `solver-014`
(converged-solver re-optimization is the slowest fix path), `interop-012`
(extra PyTorch dependency), and `clamp-009` (32768-emitter production
config). Run the full 33-task `evals-full.json` at release gates or after
skill content changes.

## Grading notes

- Positive tasks: `assertions` entries are ordered, checkable steps;
  `expected_output` carries the validated numbers (symptom values, post-fix
  values) that a correct solution reproduces to within float noise.
- The fixtures deliberately contain the defects agents must find. Do not
  lint-fix, reformat, or otherwise "clean up" `evals/files/` — several
  unused-variable warnings there ARE the planted bugs (this is also why the
  repository suppresses `F841`/`RUF059` for these directories).
- When grading claims of verification, prefer re-measuring the shipped
  fixture over trusting the transcript: the strongest observed failure mode
  is a confident report whose verification script tested a re-implementation
  rather than the file as fixed.
- The whole skill directory is staged into `/workspace/skills/` for the
  with-skill arm, filtered only for `results`, `__pycache__`, and `.git`.
  Both dataset files, `expected_output` included, are therefore readable by
  that arm and not by the baseline. Audit each run for whether it mattered
  by grepping the trajectories for `expected_output` text (not the file
  name, which shows up in harmless directory listings).
