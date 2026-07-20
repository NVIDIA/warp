# Parallel Test Runner

**Status**: Implemented

## Motivation

Warp's test suite runs across multiple processes because kernel compilation
and GPU execution make a serial run take over an hour. The runner began as a
vendored copy of the third-party `unittest-parallel` script, which handled
healthy runs fine but fell apart when a worker process died (segfault in
native code, CUDA error, OOM kill): the entire run collapsed into a single
`BrokenProcessPool` error. CI logs could not say which worker died, which
test it was running, or which of the other suites had actually passed.
Engineers re-ran jobs and bisected by hand.

`warp/_src/test_runner/` replaces that script with a Warp-owned runner built
so that a parallel test failure can be diagnosed from a single CI run. It
identifies the crashed worker and its candidate test, keeps every result
that was confirmed before the failure, and leaves evidence on disk that
survives the job itself failing.

## Requirements

| ID  | Requirement                                                                | Priority | Notes |
| --- | -------------------------------------------------------------------------- | -------- | ----- |
| R1  | Attribute a worker crash to the worker, suite, and candidate test          | Must     | Fault logs + event journals |
| R2  | Preserve and report all results confirmed before a pool failure            | Must     | No "everything crashed" reports |
| R3  | Persist diagnostic evidence that survives job failure                      | Must     | Uploaded as CI artifacts |
| R4  | Surface a pool failure in JUnit XML                                        | Must     | GitLab MR test reports |
| R5  | Never let diagnostics failures mask test results                           | Must     | See invariants below |
| R6  | Report per-suite scheduling/timing for shard balancing                     | Should   | `suite-timings.json` + console top-20 |
| R7  | Detect redundant kernel compilation across workers                        | Should   | `module-load-summary.json` |
| R8  | Observe GIL state changes on free-threaded CPython                         | Should   | Caught a real bug: importing Blosc re-enables the GIL on 3.14t |
| R9  | Keep clean runs cheap (small artifacts, no leftover evidence)              | Should   | Evidence deleted after clean runs |

**Non-goals**:

- Cross-version artifact compatibility. JSON artifacts carry no schema
  version. They are per-commit CI outputs read by humans (or tools built
  against the current format), and journals are written and replayed within
  a single run, so writer and reader are always the same code.
- Crash recovery. There is no retry or rerun mode. A pool failure reports
  salvaged results, classifies the rest, and exits nonzero. Two historical
  modes were removed once result salvage made them redundant:
  `--fallback-on-crash` isolated reruns and `--serial-fallback`.
- Exact exit attribution on Windows. The provenance heuristic is
  POSIX-oriented. On Windows, liveness at the failure snapshot carries the
  attribution, and a termination that happened before the snapshot may be
  labeled abnormal.
- Bit-exact metric aggregation. Timing summaries use plain float
  accumulation; they are diagnostics, not numerics.
- A public API. Everything lives under `warp/_src/` and is invoked via
  `python -m warp.tests`.

## Design

### Approach

Two data paths carry worker diagnostics, ranked by trust:

1. Durable journals are the ground truth. Every worker appends lifecycle
   events (worker/suite/test start, outcome, cleanup, GIL changes) as JSON
   lines to `worker-*.events.jsonl` in the run directory, through an
   `O_APPEND` fd. Workers also redirect their stdout/stderr file descriptors
   to a per-worker `*.output.log`, which captures native-code output that
   Python-level capture cannot see, and point `faulthandler` at a
   `*.fault.log`. The fault handler fires on fatal signals only, so a
   non-empty fault log is strong evidence of a crash.
2. The live event queue is only a view of that record. Workers push the same
   events onto a multiprocessing queue; a monitor thread in the parent feeds
   a state tracker that powers live progress lines and instant snapshots. On
   the happy path the monitor is stopped with a `None` sentinel after the
   executor has shut down; no events can arrive after worker exit, so the
   sentinel drains everything. On a pool failure the queue is presumed
   unreliable and the journals are replayed into the tracker instead.
   Per-worker sequence numbers make the overlap between live and replayed
   events idempotent.

Pool supervision keeps results keyed by future index, so every future that
completed is salvaged and reported, even out of order, even during a
failure. After a failure, each discovered suite is classified from evidence:
`confirmed` (result received), `started` (evidence it began but no result;
these are the crash candidates), `skipped` (cancelled or failfast), or
`never_started`. Worker exit provenance comes from observation rather than
inference. A process still alive when the failure surfaced can only have
been killed by our own cleanup, so it is `parent_terminated`. An
already-dead worker with a fatal signal or nonzero exit is
`independently_abnormal`.

Liveness alone cannot carry that first rule. CPython's executor manager
thread publishes `BrokenProcessPool` to the pending futures before it
terminates the surviving workers, so the parent may not reach its liveness
snapshot until those workers are already reaped, and the label would then
track thread scheduling rather than the worker's fate. The worker's last
journaled phase is immune to that race, so an already-dead worker that had
not reached `shutdown` and exited on SIGTERM is also `parent_terminated`.
SIGKILL stays `unresolved`: our cleanup sends it, but so does the OOM
killer, and conflating the two would hide the more serious diagnosis.

The formatted pool-failure report orders workers by evidence strength
(abnormal first, unresolved next, parent-terminated compacted last), prints
fault and output log tails for workers the parent did not terminate, and
injects a synthetic `WorkerPoolCrash` ERROR record into the JUnit XML so
the failure is visible in merge-request test reports.

### Invariants

- Diagnostics never mask test results. Every diagnostic step in the parent
  runs through a best-effort wrapper: ordinary exceptions warn and continue,
  while `KeyboardInterrupt` and `SystemExit` mark evidence for retention and
  re-raise only after the remaining cleanup steps have run.
- Evidence outlives failure. The diagnostics run directory is written
  incrementally and atomically (`os.replace`), and CI uploads it with
  `if: always()`. After a fully clean run, bulky per-worker evidence is
  deleted and only compact metadata is kept (`run.json`,
  `suite-timings.json`, `module-load-summary.json`).
- Ambiguity is reported as ambiguity. The runner never infers crash
  causality; `unresolved` appears as its own label with an explanatory note
  in the report.
- Worker sink setup is best-effort. If output redirection or the
  faulthandler sink fails to configure, the worker warns and runs with
  whatever succeeded; handles are still registered for shutdown cleanup.

### Alternatives Considered

- Wrapping `terminate()`/`kill()` on executor-owned processes to prove
  parent-initiated shutdown. This gives airtight provenance but costs about
  140 lines of monkey-patching against `concurrent.futures` private
  internals, fragile across CPython versions. Corroborating the liveness
  snapshot with the worker's last journaled phase gets the same answer for
  the pre-snapshot SIGTERM race without touching executor internals.
- `Process.is_alive()` for the failure-time liveness snapshot. Rejected
  after deterministic misclassification in testing: at the instant
  `BrokenProcessPool` surfaces, a dead but unreaped child (or one whose reap
  raced the executor's manager thread) still reports alive. The snapshot
  checks whether `process.sentinel` is readable instead. The sentinel
  becomes readable at the kernel level the moment the child exits, does not
  depend on reaping, and is public API; `is_alive()` remains as a fallback.
- A barrier protocol to flush the live queue mid-run. Unnecessary: the only
  flush point is after executor shutdown, when no worker can publish again,
  so the ordinary stop sentinel drains exactly the same events without a
  second synchronization mechanism.
- Schema-versioning the artifacts. Write-only future-proofing: nothing reads
  the version, and the one in-code reader (journal replay) is same-run by
  construction. A marker key can be added if cross-version tooling ever
  materializes.
- Isolated rerun after a crash (`--fallback-on-crash`). This predated result
  salvage. Re-running everything to find out what passed is worse than
  reporting what provably passed and failing loudly.

### Key Implementation Details

Module responsibilities (`warp/_src/test_runner/`):

- `common.py`: shared dataclasses/enums (events, snapshots, classifications,
  process exits) and helpers (warnings, GIL probe, suite naming, coverage
  context, atomic writes). This is the stdlib-only bottom layer; everything
  imports it, and it imports nothing from the package.
- `events.py`: worker-side event reporter and journal/output/fault sinks;
  parent-side monitor thread and worker state tracker; journal replay.
- `worker.py`: per-suite execution (`ParallelTestManager`) and worker
  process initialization (per-worker or shared kernel cache, `wp.init()`).
- `pool.py`: parallel orchestration. Submits suites, salvages results by
  future index, snapshots liveness and journaled worker phase, kills the
  pool, derives exit provenance.
- `postmortem.py`: suite classification, crash-snapshot JSON, the formatted
  pool-failure report, evidence tails, the JUnit pool-failure record.
- `module_loads.py`: parses Warp module-load timer lines from worker output
  logs and reports hash churn, repeated compilations, and slowest builds.
- `artifacts.py`: run-directory lifecycle, `run.json` metadata (environment
  values are allowlisted, never dumped), suite-timing serialization, and
  clean-run evidence cleanup.
- `runner.py`: CLI, suite discovery, the best-effort diagnostics lifecycle
  (`_RunDiagnostics`), result aggregation and reporting, coverage reports.

The result class (`ParallelJunitTestResult` in `warp/tests/unittest_utils.py`)
is shared by all modes: it emits the diagnostic event lifecycle and records
JUnit tuples unconditionally, and XML is only written when
`--junit-report-xml` is passed. Fixture-level failures (`setUpClass` and
friends) get synthesized identities so they appear in both journals and XML.

## Testing Strategy

The runner is validated by a dedicated harness at
`tools/ci/test_unittest_parallel.py`, kept outside `warp/tests/**` and
the default suite on purpose: the repo rule bars test-harness meta-tests
there. Run it directly:

```bash
uv run python tools/ci/test_unittest_parallel.py
```

The harness spawns real process pools and exercises the failure paths end to
end: worker segfault/abort, hangs, failfast cancellation, salvage ordering,
provenance labels, journal replay, GIL-change events, sink-failure
isolation, JUnit fixture records, and metadata allowlisting. It also pins
the diagnostics-never-mask-results invariant, including `KeyboardInterrupt`
and `SystemExit` propagation with evidence retention.

Known gaps and follow-ups: the harness is not yet wired into a CI job, so
nothing runs it automatically, and `tools/` is excluded from Ruff in
`pyproject.toml`, so the harness gets no lint coverage.
