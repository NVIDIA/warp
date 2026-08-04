# Rejection gates

A gate asks: **does a stated fact make this Warp evaluation inapplicable to the
scoped seam or regime?** Each is a property of the environment, workload shape
or incumbent — never a statement about how much evidence is available. Missing
measurements are not a gate; report them as missing downstream.

- Run Gates A–E before profiling. Run Gate F before profiling only when
  representative evidence already exists; otherwise settle it from the
  authorized stage-2 profile.
- Any one gate resolves to `ABORT`, with no Warp prototype at all.
- There are no "adjacent", partial or analogous gates. If every stated condition
  does not hold, that gate does not fire.
- Gates B, C and D are visible in code and incumbent artifacts. Gate F requires
  representative measured or structural evidence.
- Gates A and E are product decisions — which hardware the product may require,
  which obligations its packaging can take. Fire them only on something stated;
  where nothing states it, ask and stop (`AWAITING INTENT`). Code that runs on
  the CPU today, or on another accelerator, or without a Warp dependency,
  records neither decision.

## Gate A — deployment excludes NVIDIA

Production is CPU-only, or requires accelerated non-NVIDIA portability, with no
acceptable optional CUDA path. Warp's CPU kernels are serial, so "run it on CPU
instead" is not a performance fallback.

**Fires on:**

- a deployment, support or packaging document naming target hardware with no
  NVIDIA part; a fleet or appliance SKU list without one; a stated hardware
  freeze;
- cross-vendor parity promised as a product property — CI covering several
  vendors from one source, a "runs anywhere" guarantee;
- a non-NVIDIA accelerator this path must keep (TPU pods, Metal, ROCm-only) with
  no optional CUDA path acceptable alongside;
- the user saying so.

**Does not fire on absence:** no GPU code, no CUDA in the dependency list, no
Warp, CPU-only CI, a NumPy implementation, or a framework that happens to be
running on the CPU here. None of that says the product *may not* use an NVIDIA
GPU.

**Does not fire on a second accelerator that merely exists.** A JAX-on-TPU
codebase, or a project with a Metal path, still faces the question of whether an
NVIDIA path is acceptable *here*, alongside what exists. Required parity across
targets fires the gate; a second target is a portability and ownership fact to
record.

## Gate B — the boundary dominates

Data must cross host/device per small or infrequent call and the boundary cannot
be widened. → `ABORT` unless the caller can be restructured to keep data
resident.

## Gate C — it is ordinary dense tensor algebra

Dense linear algebra, convolution, FFT, reductions, standard neural-network
layers, or anything already lowered efficiently by the framework or a vendor
library. The tuned framework or vendor path is the relevant baseline. → `ABORT`.

## Gate D — a mature CUDA implementation already satisfies the contract

All three required: a maintained implementation is confirmed to execute through
CUDA on an NVIDIA GPU; it already meets the semantics; it approaches the
relevant GPU's hardware limits. → `ABORT` unless the user explicitly requested
an allowed non-performance objective *and* defined an acceptable
performance-regression budget.

**Does not fire when the incumbent's measured work executes on the CPU, however
fast.** Native extensions, vectorized code, multithreaded code and CPU JITs are
strongest-baseline candidates, not CUDA paths. Do not infer CUDA execution from
implementation language, package type or performance. If the incumbent does not
execute on an NVIDIA GPU, Gate D is unavailable.

## Gate E — the project cannot own the obligations

Warp adds an optional dependency, runtime compilation, a kernel cache, a
version-qualification matrix, platform-specific tests and a fallback path.

**Fires on a stated policy:** no runtime code generation; no compiler or
toolchain in the image; a read-only or air-gapped deployment with no writable
cache; vendored prebuilt binaries only; a pure-Python wheel guarantee; a
dependency policy that excludes it.

**Does not fire on** a project that merely happens to be pure Python today,
ships a small dependency list, or has no compiled extension yet — that is a cost
to raise, not a decision the repository has made. Nor does it fire on a
contribution policy that requires prior discussion for dependency additions, or
states they are unlikely to be merged: that governs how a change is proposed and
by whom, not whether the project can own the obligations.

Ask Gate E together with Gate A's question — for the user they are one decision.
State the concrete optional-path shape when asking: a named optional dependency
group, a soft import, and a fallback to the existing implementation leave the
default install unchanged, so only opt-in users carry the runtime obligations.
Do not assume the project accepts it.

## Gate F — the region provably cannot matter

The candidate stage is too small a share of the requested metric for any backend
to move it. Fires only from a profile the user supplied, an obvious structural
bound, or arithmetic on figures they quoted. If you cannot tell without
measuring, leave the gate open and settle it in stage 2. A stage-2 Gate F result
aborts the already-authorized scope without another authorization question.

Unavailable representative coverage is not Gate F. Record the single missing
artifact and stop with missing evidence; do not convert absence into `ABORT`.

- **The profile must represent the requested objective.** A measured share fires
  the gate only when the profile's workload, execution regime and metric match
  the user's target. A share from a maintenance, synthetic, demonstration or
  otherwise unrelated workload proves nothing, and missing target frequency or
  distribution is missing evidence rather than Gate F.
- **Apply it against the metric that candidate's pattern exhausts**, not the
  study's headline objective. Classify first and use the resource named by the
  matched target pattern. A latency study still contains candidates whose whole
  value is memory or capacity.
- **Show the ceiling arithmetic:** what the requested metric becomes with the
  stage free, and the maximum ratio that implies. Gate F requires a ceiling you
  can state, not an impression.
- **State which metric the gate was applied against**, and the device memory
  budget the calculation assumes.

## Asking about intent

Gates A and E unclear, at least one candidate pattern standing, no other gate
fired: that is `AWAITING INTENT`. Ask one question, then stop — before
profiling, before the report directory, before any prototype.

A question worth asking:

- **states what you read, not what you want** — "nothing in the repo says
  whether CPU-only is a requirement or just where this landed", not "do you have
  a GPU?", which invites a yes from someone with a laptop GPU and no intention
  of shipping against it;
- **makes both branches concrete and finished**, naming whether Warp will be
  tested or the evaluation aborts;
- is **asked once**, batched with the authorization checkpoint;
- **does not argue.**

Ask: **"May this evaluation prototype and benchmark a Warp implementation?"**
Use choices whose labels and descriptions state the action explicitly:

| Choice | What follows |
|---|---|
| **Yes — test Warp as an optional backend** | Prototype and benchmark Warp behind the named extra, soft import and existing fallback; record those conditions in scope |
| **Yes — test Warp without additional packaging conditions** | Prototype and benchmark Warp across the widest otherwise-authorized scope |
| **No or undecided** | `ABORT` before profiling; cite the user-supplied constraint or unresolved intent |
| **No answer** | Stop. No report directory. This is **not** `ABORT`: Warp was plausible, but the evaluation never ran |

Do not offer "CPU-side findings only" or incumbent-only profiling as a branch of
`warp-eval`; that is a separate ordinary optimization task.

**Do not ask** when the repository answers it; when another gate already fired
(B, C, D and F end the study on their own); when no pattern matched; or when you
would proceed regardless — if "CPU-only, please" would leave you still assessing
Warp, the question was decoration.
