# Evaluation Guidance

Developer notes for the evaluation datasets. Two files:

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

- NVIDIA Warp >= 1.13 with a working compute device. All fixtures were
  validated on Warp 1.16.0-dev with CUDA; they also run on CPU devices,
  though the documented symptom numbers were measured on CUDA.
- `interop-012` additionally requires PyTorch with CUDA
  (`warp_layer.py` / `torch_train.py`).
- Each fixture reproduces its symptom in seconds; agent sessions on the
  hardest tasks take several minutes each. Multi-file tasks (per-fixture
  subdirectories, plus `optlib.py` + `sensor_train.py` and `warp_layer.py` +
  `torch_train.py`) must be staged into the workspace together, preserving
  the subdirectory layout so imports resolve.
- Toolkit caveat: NVRTC 13.1-13.3 miscompiles some generated backward
  kernels ("readonly and writeonly attributes incompatible"), which breaks
  the `structwar` fixture at module load. Build Warp against CUDA Toolkit
  13.0 (or a toolkit without that NVRTC defect) when running these evals.

## P0 smoke subset (CI-gated runs)

NV-CARPS CI runners cap evaluation at one hour. For CI-gated runs, prefer
this six-task subset; it spans the skill's main behavior classes while
staying well inside the budget:

| Task | Why it is in P0 |
|---|---|
| `huber-001-custom-grad-outlier-branch` | Core positive case: real bug, custom-gradient math, fast fixture |
| `seed-002-sum-vs-mean-adjoint` | Misdirection resistance (exact-factor bait), fast fixture |
| `masked-006-zero-grad-hides-2x` | The most discriminating task: post-fix re-verification of the shipped program (the one behavior baselines measurably fail) |
| `quant-011-round-ste` | "No bug" honesty + instrument inversion (FD wrong, AD right) + reformulation |
| `neg-015-forward-kernel-perf` | Negative activation: Warp-adjacent performance task |
| `neg-017-pure-torch-grads` | Negative activation: gradient vocabulary without Warp |

Excluded from P0 on runtime/dependency grounds, not value:
`walker-019` (the flagship discriminating task — conceptual gradient
truncation under a state-aliasing bug, and the one task no-skill baselines
reproducibly fail — but also among the slowest, with multi-minute solves
and heavy training loops; it is the first task to promote into `evals.json`
if the CI budget allows), `chaos-007` (hundreds of taped launches per
training step), `solver-014` (converged-solver re-optimization is the
slowest fix path), `interop-012` (extra PyTorch dependency), and
`clamp-009` (32768-emitter production config). Run the full 33-task
`evals-full.json` at release gates or after skill content changes.

## Grading notes

- Positive tasks: `expected_behavior` entries are ordered, checkable steps;
  `ground_truth` carries the validated numbers (symptom values, post-fix
  values) that a correct solution reproduces to within float noise.
- The fixtures deliberately contain the defects agents must find. Do not
  lint-fix, reformat, or otherwise "clean up" `evals/files/` — several
  unused-variable warnings there ARE the planted bugs (this is also why the
  repository suppresses `F841`/`RUF059` for these directories).
- When grading claims of verification, prefer re-measuring the shipped
  fixture over trusting the transcript: the strongest observed failure mode
  is a confident report whose verification script tested a re-implementation
  rather than the file as fixed.
