#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement primitives implementing references/benchmark-protocol.md.

Import these from one driver script per bottleneck rather than hand-rolling
timing and memory code, so that two evaluations — or two agents — produce
comparable numbers. The protocol is the specification; this is its
implementation, and every default here encodes one of its rules:

  time_it        One discarded warm-up, then ONE timed run. Not a sampling
                 loop: more samples only sharpen a figure contaminated by a
                 machine you cannot quiet. Synchronizes before and after.
  comparison     Ratio plus protocol resolution. >=1.5x is directional; below
                 1.5x is no measured difference at this precision.
  sig2           Two significant figures at most.
  host_memory    Reads /proc VmHWM, NOT resource.getrusage().ru_maxrss.
                 ru_maxrss is inherited across a process spawn: a 787 MB parent
                 yields a child reporting 787 MB before it allocates anything,
                 which silently defeats the protocol's "measure in fresh
                 processes" rule. VmHWM is per-process and correct after exec.
  device_memory  NVML per-process GPU memory — the device-side counterpart of
                 process RSS, and the only domain that sees every framework and
                 the CUDA context alike. A bare context is 445 MB that CuPy's
                 own pool counter reports as 0. Framework allocator counters
                 supplement it; they never replace it.
  run_isolated   Fresh process per case; peak counters are monotonic within a
                 process, so the second config measured in one process inherits
                 the first's high-water mark.
  null_test      Proves the harness runs the work. A harness that silently
                 fails to run reports a very fast time, which reads exactly
                 like an apparent speedup.

Timings are synchronized wall clock unless you call device_time.

Usage:
    Import this module from one copied driver-template.py per bottleneck.
Arguments:
    Each function documents its callable, device, case and environment inputs.
Output:
    JSON-serializable measurement records; emit() writes JSON Lines.
Exit codes:
    This is an import library and does not define a command-line exit code.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from math import floor, isinf, log10
from pathlib import Path

__all__ = [
    "comparison",
    "device_baseline",
    "device_memory",
    "device_time",
    "emit",
    "env_fingerprint",
    "host_memory",
    "isolated_case",
    "null_test",
    "require_warp_cuda",
    "run_isolated",
    "sig2",
    "time_it",
]

NOT_MEASURED = "not measured"
_BLOCKED_ENV_KEYS = frozenset({"PATH", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"})
_BLOCKED_ENV_PREFIXES = ("DYLD_", "LD_")


# ------------------------------------------------------------------ reporting


def sig2(x: float | None) -> float | None:
    """Round to two significant figures. A single sample does not support
    '1.87x'; it supports 'about 1.9x', often only 'about 2x'."""
    if x is None or x == 0 or x != x or isinf(x):
        return x

    return round(x, -int(floor(log10(abs(x)))) + 1)


def comparison(baseline_s: float, candidate_s: float) -> dict:
    """Report a ratio at the protocol's measurement resolution."""
    r = baseline_s / candidate_s if candidate_s else float("inf")
    if r >= 1.5 or r <= 1 / 1.5:
        resolution = "directional difference"
    else:
        resolution = "no measured difference at this precision"
    return {
        "ratio": sig2(r),
        "comparison": resolution,
    }


# ------------------------------------------------------------------------ env

# Distribution names differ from import names, and probing a version must never
# import the module: importing an uninitialized runtime makes it look live to _sync.
_DISTS = {"numpy": "numpy", "scipy": "scipy", "warp": "warp-lang", "cupy": "cupy-cuda12x", "torch": "torch"}


def _version(mod: str):
    try:
        return version(_DISTS.get(mod, mod))
    except PackageNotFoundError:
        return None


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def env_fingerprint() -> dict:
    """Everything that decides whether two records are comparable. The thread
    count is not optional: a CPU baseline measured below the deployment's thread
    count has not been compared."""
    gpu = driver = NOT_MEASURED
    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if smi:
        gpu, driver = (s.strip() for s in smi.splitlines()[0].split(","))
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cores": os.cpu_count(),
        "threads": {
            v: os.environ.get(v)
            for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "gpu": gpu,
        "driver": driver,
        "versions": {m: _version(m) for m in _DISTS},
        "commit": _run(["git", "rev-parse", "HEAD"]),
    }


# --------------------------------------------------------------------- timing


def require_warp_cuda(device="cuda:0"):
    """Resolve and return an explicitly selected CUDA device through Warp.

    Warp's CPU backend is outside this evaluation's scope. Call this before
    correctness work; ``time_it(..., warp_candidate=True)`` calls it
    automatically before warm-up or timed execution.
    """
    if device is None or str(device) == "cpu":
        raise ValueError("Warp evaluation requires an explicit CUDA device, not CPU")
    wp = sys.modules.get("warp")
    if wp is None:
        raise RuntimeError("import and initialize Warp before resolving its CUDA device")
    try:
        resolved = wp.get_device(device)
    except Exception as exc:
        raise RuntimeError(f"Warp could not resolve device={device!r}: {exc}") from exc
    if not getattr(resolved, "is_cuda", False):
        raise ValueError(f"Warp evaluation requires a CUDA device; resolved {resolved!s}")
    return resolved


def _sync(device) -> None:
    """Synchronize `device` before and after a timed region.

    Silently doing nothing here is how asynchronous dispatch time gets compared
    against synchronized end-to-end time, so this raises rather than returns if
    nothing could be synchronized.
    """
    if device is None or str(device) == "cpu":
        return
    d = str(device)
    # Syncing "whichever runtime is live" syncs its *current* device, which on a
    # multi-GPU host need not be the one named. Resolve the index explicitly.
    if d == "cuda":
        idx = None
    elif d.startswith("cuda:") and d[5:].isdigit():
        idx = int(d[5:])
    else:
        raise ValueError(f"device={device!r} is not 'cpu', 'cuda', or 'cuda:<n>'")

    errors = {}
    for mod, fn in (
        ("cupy", lambda m: (m.cuda.Device(idx) if idx is not None else m.cuda.Device()).synchronize()),
        ("warp", lambda m: m.synchronize_device(d)),
        ("torch", lambda m: m.cuda.synchronize(idx)),
    ):
        m = sys.modules.get(mod)
        if m is None:
            continue
        try:
            fn(m)
            return
        except Exception as exc:  # imported but uninitialized, or no device
            errors[mod] = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(
        f"device={device!r} but no GPU runtime could be synchronized: {errors or 'none imported'}. "
        "Initialize warp/cupy/torch first, or pass device=None for pure-CPU work."
    )


def time_it(fn, *, device=None, warp_candidate: bool = False, warmup: int = 1, samples: int = 1) -> dict:
    """One discarded warm-up, then one timed run, synchronized both sides.

    Warp candidates must set ``warp_candidate=True``. This rejects Warp's CPU
    backend and records the resolved CUDA device and architecture.

    `samples` defaults to 1 deliberately. Do not raise it to tighten a reported
    figure — spend that budget on another size or regime instead. Raise it only
    for a harness self-check (see null_test), and say so in the report.

    Skipping the warm-up has moved a baseline by ~34%, enough to invert a
    comparison, which is why it is not optional.
    """
    warp_device = require_warp_cuda(device) if warp_candidate else None
    if warp_device is not None:
        device = str(warp_device)

    for _ in range(warmup):
        fn()
    _sync(device)

    times = []
    for _ in range(samples):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)

    record = {
        "seconds": min(times),
        "samples": samples,
        "warmup_discarded": warmup,
        "statistic": "synchronized_wall_clock",
        "device": str(device) if device is not None else "cpu",
        "warp_candidate": warp_candidate,
    }
    if warp_device is not None:
        record["warp_device"] = {
            "alias": str(warp_device),
            "architecture": str(getattr(warp_device, "arch", NOT_MEASURED)),
        }
    return record


def device_time(fn, *, device="cuda:0", warmup: int = 1) -> dict:
    """Device-event time. Report beside time_it where wrapper or launch behavior
    matters — never instead of it, since the evaluation covers the whole stage."""
    cp = sys.modules.get("cupy")
    if cp is None:
        return {"seconds": None, "note": f"{NOT_MEASURED}: cupy not imported"}
    for _ in range(warmup):
        fn()
    _sync(device)
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    fn()
    end.record()
    end.synchronize()
    return {
        "seconds": cp.cuda.get_elapsed_time(start, end) / 1000.0,
        "statistic": "cuda_event_device_time",
        "device": str(device),
    }


# --------------------------------------------------------------------- memory


def peak_host_mb() -> float:
    """Peak process RSS in MB, from /proc/self/status VmHWM on Linux.

    NOT resource.getrusage().ru_maxrss: that is inherited across a subprocess
    spawn, so a heavy parent censors every isolated measurement (measured: a
    787 MB parent yields a child reporting 787 MB before allocating; the same
    child's VmHWM correctly reads 24.1 MB, then 405.9 MB after a 400 MB array).
    """
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) / 1024
        except OSError:
            pass
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 if sys.platform.startswith("linux") else rss / (1024 * 1024)


_FLOOR_MB = peak_host_mb()


def host_memory() -> dict:
    """Host peak RSS — the domain that sees both sides of a comparison.

    Take headline memory ratios from here. A per-language counter such as
    tracemalloc cannot see a CUDA context at all and structurally understates a
    candidate that reserves hundreds of MiB; it may supplement this, never
    replace it.
    """
    peak = peak_host_mb()
    return {
        "peak_mb": peak,
        "floor_mb": _FLOOR_MB,
        "workload_mb": peak - _FLOOR_MB,
        "domain": "process RSS high-water (VmHWM)",
    }


_NVML = {"handle": None, "ready": False, "error": None}


def _nvml(index: int = 0):
    """Lazily initialize NVML once. Returns the device handle, or None."""
    if _NVML["ready"]:
        return _NVML["handle"]
    _NVML["ready"] = True
    try:
        import pynvml  # noqa: PLC0415
    except ImportError:
        _NVML["error"] = "nvidia-ml-py not installed (pip install nvidia-ml-py)"
        return None
    try:
        pynvml.nvmlInit()
        _NVML["handle"] = pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception as exc:
        _NVML["error"] = f"{type(exc).__name__}: {exc}"
    return _NVML["handle"]


def _nvml_process_mb(index: int = 0):
    """GPU memory this process holds, as the driver sees it. Framework-agnostic.

    Returns None where per-process accounting is unavailable — inside MIG, in
    some container configurations, and under restricted driver permissions —
    rather than substituting a device-wide figure that includes other tenants.
    """
    h = _nvml(index)
    if h is None:
        return None
    import pynvml  # noqa: PLC0415  # guaranteed importable: _nvml() returned a handle

    try:
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(h)
        except AttributeError:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
    except Exception as exc:
        _NVML["error"] = f"per-process accounting unavailable: {exc}"
        return None
    for p in procs:
        if p.pid == os.getpid():
            return (p.usedGpuMemory or 0) / 1e6
    return 0.0


_DEVICE_BASELINE: dict = {}


def device_baseline(index: int = 0) -> dict:
    """Record device memory already resident — CUDA context, driver, library
    handles — before the workload allocates.

    That context is a floor the incumbent never pays, so without this a small
    workload charges the candidate for it and can look worse at every small
    size. Measured here: 445 MB for a bare CUDA context.
    """
    global _DEVICE_BASELINE
    _DEVICE_BASELINE = {"context_fixed_mb": _nvml_process_mb(index)}
    return dict(_DEVICE_BASELINE)


def device_memory(index: int = 0, backend: str | None = None) -> dict:
    """Device memory for this process, from NVML — the domain that sees both sides.

    NVML is framework-agnostic and is the device-side counterpart of process RSS:
    it sees the CUDA context, every framework's allocations and any native
    library's alike, through one code path. A framework's own allocator counter
    cannot do this. Measured on an L40: a bare CUDA context is 445 MB of device
    memory that CuPy's pool counter reports as 0.0 MB, and the same NVML call
    tracked a 480 MB CuPy array and a 480 MB Warp array with no branch between
    them. Take headline device-memory ratios from `process_mb`.

    `backend` optionally attaches that framework's allocator detail — live versus
    pool-reserved, which NVML cannot distinguish. It supplements the NVML figure
    and never replaces it.

    NVML has 1 MiB granularity, so a zero delta means "below snapshot
    resolution", not "zero storage".
    """
    proc_mb = _nvml_process_mb(index)
    rec = {
        "process_mb": proc_mb,
        "context_fixed_mb": _DEVICE_BASELINE.get("context_fixed_mb"),
        "workload_mb": (proc_mb - _DEVICE_BASELINE["context_fixed_mb"])
        if proc_mb is not None and _DEVICE_BASELINE.get("context_fixed_mb") is not None
        else None,
        "domain": "NVML per-process GPU memory",
        "granularity_mb": 1,
    }
    if proc_mb is None:
        rec["note"] = f"{NOT_MEASURED}: {_NVML['error'] or 'NVML unavailable'}"

    # Optional supplement: allocator detail NVML cannot provide.
    if backend == "cupy" and "cupy" in sys.modules:
        pool = sys.modules["cupy"].get_default_memory_pool()
        rec["allocator"] = {
            "backend": "cupy",
            "live_mb": pool.used_bytes() / 1e6,
            "reserved_mb": pool.total_bytes() / 1e6,
        }
    elif backend == "torch" and "torch" in sys.modules:
        t = sys.modules["torch"]
        rec["allocator"] = {
            "backend": "torch",
            "live_mb": t.cuda.memory_allocated() / 1e6,
            "reserved_mb": t.cuda.memory_reserved() / 1e6,
        }
    elif backend:
        rec["allocator"] = {"backend": backend, "note": f"{NOT_MEASURED}: no counter wired"}
    return rec


# ------------------------------------------------------------------ null test


def null_test(make_fn, *, device=None, warp_candidate: bool = False, scale: int = 4, tol: float = 0.4) -> dict:
    """Prove the harness measures the work.

    `make_fn(k)` returns a callable doing k times the base work. If scaling the
    work does not scale the measurement, the harness is not running it — a
    missing sync, a cached result, a kernel never launched, a test binary that
    exits 0 without executing. All of those report a very fast time, which reads
    exactly like an apparent speedup.

    This is a harness self-check, not a reported figure, so it may take a few
    samples without violating the one-timed-run rule.
    """
    t1 = time_it(make_fn(1), device=device, warp_candidate=warp_candidate, samples=3)["seconds"]
    tk = time_it(make_fn(scale), device=device, warp_candidate=warp_candidate, samples=3)["seconds"]
    ratio = tk / t1 if t1 > 0 else float("inf")
    return {
        "scale": scale,
        "observed_ratio": sig2(ratio),
        "tolerance": tol,
        "pass": abs(ratio - scale) / scale <= tol,
    }


# -------------------------------------------------------- process isolation


def isolated_case() -> str | None:
    """Inside a run_isolated child, the case name; None in the parent."""
    return os.environ.get("WARP_EVAL_CASE")


def _driver_path(script: str | None) -> Path:
    """Resolve a Python driver confined to the current workspace."""
    candidate = Path(script or sys.argv[0])
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"measurement driver does not exist: {candidate}") from exc

    workspace = Path.cwd().resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"measurement driver must be inside {workspace}: {resolved}")
    if resolved.suffix != ".py" or not resolved.is_file():
        raise ValueError(f"measurement driver must be a Python file: {resolved}")
    return resolved


def _extra_environment(extra_env: dict | None) -> dict[str, str]:
    """Validate case parameters without allowing runtime-loader overrides."""
    safe: dict[str, str] = {}
    for raw_key, raw_value in (extra_env or {}).items():
        key = str(raw_key)
        if not key or "=" in key or "\0" in key:
            raise ValueError(f"invalid environment key: {key!r}")
        if key in _BLOCKED_ENV_KEYS or key.startswith(_BLOCKED_ENV_PREFIXES):
            raise ValueError(f"measurement environment may not override {key}")
        value = str(raw_value)
        if "\0" in value:
            raise ValueError(f"environment value for {key} contains a null byte")
        safe[key] = value
    return safe


def run_isolated(case: str, *, script: str | None = None, extra_env: dict | None = None, timeout: int = 3600) -> dict:
    """Re-execute this script in a fresh process for `case` and return the JSON
    record it printed. Required wherever allocators retain storage, and for any
    peak-memory comparison: peak counters are monotonic within a process.

    Driver shape:

        if (case := measure.isolated_case()):
            measure.emit(None, run_case(case))      # child: one JSON line
        else:
            for c in CASES:                         # parent: fresh process each
                measure.emit(RESULTS, measure.run_isolated(c))
    """
    if not isinstance(case, str) or not case or "\0" in case:
        raise ValueError("case must be a non-empty string without null bytes")
    driver = _driver_path(script)
    proc = subprocess.run(
        [sys.executable, str(driver)],
        env={**os.environ, "WARP_EVAL_CASE": case, **_extra_environment(extra_env)},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    for raw_line in reversed(proc.stdout.splitlines()):
        line = raw_line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {
        "case": case,
        "error": "no JSON record emitted",
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-2000:],
    }


# --------------------------------------------------------------------- output


def emit(path, record: dict) -> dict:
    """Append one record. Report tables are transcribed from this file; a cell
    with no record here reads "not measured". Use run-unique filenames — a
    second lock acquisition that re-runs a phase will otherwise silently
    overwrite results the report already cites."""
    record = {"env": env_fingerprint(), **record}
    line = json.dumps(record, default=str)
    if path is None:
        print(line, flush=True)
    else:
        with open(path, "a") as fh:
            fh.write(line + "\n")
    return record
