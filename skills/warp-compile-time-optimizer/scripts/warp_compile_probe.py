#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure and compare Warp cold-start module compilation for any command.

This probe runs an unmodified application command in a fresh subprocess with an
isolated Warp kernel cache, so every module it needs is compiled from scratch.
It records two things at once:

* what Warp compiled -- module identities, variants, native compiler time; and
* what the application launched -- the ordered kernel launch topology.

Both matter. Cold compilation time is only meaningful if the application still
does the same work, and the launch topology is the cheapest evidence that it
does. ``compare`` checks them together and refuses to call a change an
improvement when the workload moved.

Usage:
    # Baseline, before any edits.
    python warp_compile_probe.py measure --json before.json -- python -m your_app

    # ... make changes, then measure again with the same command.
    python warp_compile_probe.py measure --json after.json -- python -m your_app

    python warp_compile_probe.py compare before.json after.json

Nothing here is specific to any application, and no Warp source checkout is
required. The probe never writes to the project it measures.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# In-process instrumentation
# ---------------------------------------------------------------------------
# Injected into the target process through PYTHONPATH as ``sitecustomize``. It
# runs before the application's first import, so it can turn on Warp's module
# timers and wrap the launch entry points no matter how the app is started
# (``python -m pkg``, ``python script.py``, ``uv run ...``, a test runner, ...).
_SITECUSTOMIZE = r'''
import builtins
import json
import os
import sys

_RECORD_PATH = os.environ.get("_WARP_PROBE_RECORD")

# Chain to a sitecustomize the environment already had, so this injection stays
# additive rather than shadowing existing startup configuration.
_self = os.path.dirname(os.path.abspath(__file__))
_saved = list(sys.path)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _self]
try:
    del sys.modules["sitecustomize"]
    import sitecustomize  # noqa: F401
except Exception:
    pass
finally:
    sys.path[:] = _saved

_launches = []
_state = {"installed": False}


def _dtype_names(values):
    names = []
    for item in values or ():
        dtype = getattr(item, "dtype", None)
        if dtype is not None:
            names.append(getattr(dtype, "__name__", str(dtype)))
    return names


def _describe(kernel, dim, inputs, outputs, device, block_dim, adjoint):
    if isinstance(dim, int):
        shape = [dim]
    elif dim is None:
        shape = []
    else:
        try:
            shape = [int(v) for v in dim]
        except TypeError:
            shape = [int(dim)]
    module = getattr(kernel, "module", None)
    return {
        "kernel": getattr(kernel, "key", str(kernel)),
        "module": getattr(module, "name", None),
        "dim": shape,
        "device": str(device) if device is not None else None,
        "dtypes": _dtype_names(inputs) + _dtype_names(outputs),
        "block_dim": block_dim,
        "adjoint": bool(adjoint),
    }


def _install():
    """Enable Warp module timers and record every kernel launch."""
    if _state["installed"]:
        return
    warp = sys.modules.get("warp")
    config = sys.modules.get("warp.config")
    if warp is None or config is None or not hasattr(warp, "launch"):
        return
    _state["installed"] = True

    config.verbose = True

    original = warp.launch

    # Positional signature of warp.launch, used only to normalize calls that
    # pass arguments positionally.
    names = (
        "kernel dim inputs outputs adj_inputs adj_outputs device stream "
        "adjoint record_tape record_cmd max_blocks block_dim"
    ).split()

    def _wrapper(*args, **kwargs):
        try:
            bound = dict(zip(names, args))
            bound.update(kwargs)
            device = bound.get("device")
            if device is None:
                for item in list(bound.get("inputs") or ()) + list(bound.get("outputs") or ()):
                    if hasattr(item, "device"):
                        device = item.device
                        break
            _launches.append(
                _describe(
                    bound.get("kernel"),
                    bound.get("dim"),
                    bound.get("inputs"),
                    bound.get("outputs"),
                    device,
                    bound.get("block_dim"),
                    bound.get("adjoint"),
                )
            )
        except Exception:
            # Instrumentation must never change whether the application runs.
            pass
        return original(*args, **kwargs)

    _wrapper.__wrapped__ = original
    warp.launch = _wrapper
    # warp.launch_tiled() resolves `launch` as a module global of the module it
    # is defined in, so the internal binding has to be replaced as well for
    # tiled launches to be recorded.
    for internal in ("warp._src.context", "warp.context"):
        target = sys.modules.get(internal)
        if target is not None and getattr(target, "launch", None) is original:
            target.launch = _wrapper


_real_import = builtins.__import__


def _hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _real_import(name, globals, locals, fromlist, level)
    if not _state["installed"] and name.split(".")[0] == "warp":
        _install()
    return module


builtins.__import__ = _hooked_import


def _flush():
    if not _RECORD_PATH:
        return
    try:
        with open(_RECORD_PATH, "w", encoding="utf-8") as handle:
            json.dump(_launches, handle)
    except Exception:
        pass


if _RECORD_PATH:
    import atexit

    atexit.register(_flush)
'''


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
import re  # noqa: E402  (kept next to the patterns it serves)

_ANSI = re.compile(r"\x1B(?:[@-_][0-?]*[ -/]*[@-~]|\[[0-?]*[ -/]*[@-~])")
_MODULE = re.compile(
    r"^\s*Module (?P<label>.+?) load on device '(?P<device>[^']+)'"
    r"(?: \(block_dim=(?P<block_dim>\d+)\))?"
    r" took (?P<ms>\d+(?:\.\d+)?) ms\s+\((?P<status>compiled|cached|error)\)\s*$"
)
_COMPILER = re.compile(r"^\s*Compile (?P<compiler>x86|CUDA)(?: \((?P<details>.*)\))? took (?P<ms>\d+(?:\.\d+)?) ms\s*$")


_HASH = re.compile(r"^[0-9a-f]{6,}$")


def split_label(label: str) -> tuple[str, str]:
    """Split ``"pkg.module abc1234"`` into its module name and content hash.

    Warp appends the module hash to the label it logs. The hash is the single
    most useful field in the log: two events with the same name but different
    hashes mean the module's identity changed mid-run and it was rebuilt from
    scratch, which is a different problem from the same hash appearing at
    several block dimensions.
    """
    name, _, tail = label.rpartition(" ")
    if name and _HASH.match(tail):
        return name, tail
    return label, ""


@dataclass
class ModuleEvent:
    label: str
    device: str
    block_dim: int | None
    ms: float
    status: str
    name: str = ""
    hash: str = ""


@dataclass
class Sample:
    """One cold run of the command in a private, empty cache."""

    cold_module_ms: float = 0.0
    native_compile_ms: float = 0.0
    warm_module_ms: float | None = None
    # Elapsed process time. `cold_module_ms` is a *sum* of per-module timers, so
    # it counts overlapping builds twice; wall clock is what the user waits on.
    cold_wall_ms: float = 0.0
    warm_wall_ms: float | None = None
    compiled_events: int = 0
    cached_events: int = 0
    error_events: int = 0
    identities: int = 0
    modules: int = 0
    generated_source_bytes: int = 0
    lto_artifacts: int = 0
    lto_bytes: int = 0
    exit_code: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    launches: list[dict[str, Any]] = field(default_factory=list)


def parse_events(text: str) -> tuple[list[ModuleEvent], float]:
    """Return module load events and total native compiler time from Warp output."""
    events: list[ModuleEvent] = []
    native_ms = 0.0
    for raw in text.splitlines():
        line = _ANSI.sub("", raw)
        if match := _MODULE.match(line):
            block = match.group("block_dim")
            label = match.group("label")
            name, digest = split_label(label)
            events.append(
                ModuleEvent(
                    label=label,
                    device=match.group("device"),
                    block_dim=int(block) if block is not None else None,
                    ms=float(match.group("ms")),
                    status=match.group("status"),
                    name=name,
                    hash=digest,
                )
            )
        elif match := _COMPILER.match(line):
            native_ms += float(match.group("ms"))
    return events, native_ms


def inventory(cache_dir: Path) -> tuple[int, int, int]:
    """Return generated source bytes, LTO artifact count, and LTO bytes."""
    source_bytes = lto_count = lto_bytes = 0
    if not cache_dir.is_dir():
        return 0, 0, 0
    for item in cache_dir.rglob("*"):
        if not item.is_file():
            continue
        size = item.stat().st_size
        if item.suffix in {".cpp", ".cu"}:
            source_bytes += size
        elif item.suffix == ".lto" or item.suffix == ".ltoir":
            lto_count += 1
            lto_bytes += size
    return source_bytes, lto_count, lto_bytes


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def run_once(command: list[str], workdir: Path, *, timeout: float) -> Sample:
    """Run ``command`` cold in a private cache, optionally re-running it warm."""
    sample = Sample()
    with tempfile.TemporaryDirectory(prefix="warp-probe-") as scratch:
        scratch_path = Path(scratch)
        cache = scratch_path / "warp-cache"
        cuda_cache = scratch_path / "cuda-cache"
        inject = scratch_path / "inject"
        record = scratch_path / "launches.json"
        cache.mkdir()
        cuda_cache.mkdir()
        inject.mkdir()
        (inject / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")

        # The measured command inherits this process's environment, then has its
        # cache paths redirected below so the run is cold and self-contained.
        env = dict(os.environ)
        env["WARP_CACHE_PATH"] = str(cache)
        # WARP_CACHE_ROOT is honored by some project setups; keep the two aligned
        # so nothing falls back to a shared user-level cache.
        env["WARP_CACHE_ROOT"] = str(cache)
        env["CUDA_CACHE_PATH"] = str(cuda_cache)
        env["PYTHONPATH"] = os.pathsep.join([str(inject), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        env["_WARP_PROBE_RECORD"] = str(record)

        # A command that carries its own PYTHONPATH (`env PYTHONPATH=... python -m pkg`)
        # would otherwise overwrite the inject directory and silently disable launch
        # recording, so merge into that assignment instead of fighting it.
        command = list(command)
        for index, token in enumerate(command):
            if token.startswith("PYTHONPATH="):
                existing = token[len("PYTHONPATH=") :]
                merged = os.pathsep.join([str(inject), existing]).rstrip(os.pathsep)
                command[index] = f"PYTHONPATH={merged}"

        try:
            started = time.perf_counter()
            cold = subprocess.run(
                command,
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            sample.cold_wall_ms = (time.perf_counter() - started) * 1000.0
        except subprocess.TimeoutExpired:
            sample.exit_code = -1
            return sample

        output = cold.stdout + cold.stderr
        sample.exit_code = cold.returncode
        events, native_ms = parse_events(output)
        sample.native_compile_ms = native_ms
        sample.cold_module_ms = sum(e.ms for e in events if e.status == "compiled")
        sample.compiled_events = sum(1 for e in events if e.status == "compiled")
        sample.cached_events = sum(1 for e in events if e.status == "cached")
        sample.error_events = sum(1 for e in events if e.status == "error")
        sample.identities = len({(e.label, e.device, e.block_dim) for e in events if e.status == "compiled"})
        sample.modules = len({e.name for e in events if e.status == "compiled"})
        sample.events = [asdict(e) for e in events]
        if record.is_file():
            try:
                sample.launches = json.loads(record.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                sample.launches = []
        sample.generated_source_bytes, sample.lto_artifacts, sample.lto_bytes = inventory(cache)

        if sample.exit_code == 0:
            # Same command, same populated cache: what a repeat invocation pays.
            # Always run this. It is what separates a cold-start cost from a
            # per-run one (CS-2), and `cold_wall - warm_wall` is the only measure
            # of compile time that does not double-count overlapping builds. A
            # cold number without it cannot be judged.
            env.pop("_WARP_PROBE_RECORD", None)
            try:
                started = time.perf_counter()
                repeat = subprocess.run(
                    command,
                    cwd=str(workdir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                if repeat.returncode == 0:
                    sample.warm_wall_ms = (time.perf_counter() - started) * 1000.0
                    warm_events, _ = parse_events(repeat.stdout + repeat.stderr)
                    sample.warm_module_ms = sum(e.ms for e in warm_events)
                else:
                    # A warm run that fails exits early, so its wall clock is not
                    # the cost of a repeat invocation. Crediting it would inflate
                    # `cold_wall - warm_wall` and overstate the compile time a
                    # candidate saved, which is the one error this tool must not
                    # make. Drop the sample the way a timeout does.
                    sample.warm_module_ms = None
                    sample.warm_wall_ms = None
            except subprocess.TimeoutExpired:
                sample.warm_module_ms = None
                sample.warm_wall_ms = None
    return sample


def median_mad(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    med = statistics.median(values)
    return med, statistics.median(abs(v - med) for v in values)


def measure(args: argparse.Namespace) -> int:
    if not args.command:
        print("error: provide the application command after `--`", file=sys.stderr)
        return 2

    workdir = Path(args.cwd).resolve()
    samples: list[Sample] = []
    for index in range(args.samples):
        sample = run_once(args.command, workdir, timeout=args.timeout)
        samples.append(sample)
        status = "ok" if sample.exit_code == 0 else f"exit {sample.exit_code}"
        print(
            f"  sample {index + 1}/{args.samples}: "
            f"cold module work {sample.cold_module_ms:9.2f} ms  "
            f"native {sample.native_compile_ms:8.2f} ms  "
            f"identities {sample.identities:2d}  ({status})",
            file=sys.stderr,
        )

    failed = [s for s in samples if s.exit_code != 0]
    ok = [s for s in samples if s.exit_code == 0]
    if not ok:
        # Reporting a median over failing runs would show near-zero module work,
        # which reads as a large improvement rather than a broken command.
        print(
            f"\nERROR: all {len(samples)} runs exited non-zero, so there is nothing to "
            "measure. Fix the command first, then measure again.",
            file=sys.stderr,
        )
        return 1
    if failed:
        print(
            f"\nWARNING: {len(failed)}/{len(samples)} runs exited non-zero. "
            "Timings from a failing run are not evidence -- fix the run first.",
            file=sys.stderr,
        )

    cold_median, cold_mad = median_mad([s.cold_module_ms for s in ok])
    native_median, _ = median_mad([s.native_compile_ms for s in ok])
    warm_values = [s.warm_module_ms for s in ok if s.warm_module_ms is not None]
    warm_median = statistics.median(warm_values) if warm_values else None

    cold_wall_median, cold_wall_mad = median_mad([s.cold_wall_ms for s in ok])
    warm_wall_values = [s.warm_wall_ms for s in ok if s.warm_wall_ms is not None]
    warm_wall_median = statistics.median(warm_wall_values) if warm_wall_values else None

    # Elapsed time attributable to compilation. The warm pass runs the same
    # command against a populated cache, so process startup, imports, and the
    # workload itself appear in both and cancel. Unlike the sum of per-module
    # timers, this does not double-count builds that overlapped.
    elapsed_values = [s.cold_wall_ms - s.warm_wall_ms for s in ok if s.warm_wall_ms is not None]
    compile_elapsed_median, compile_elapsed_mad = median_mad(elapsed_values) if elapsed_values else (None, None)

    # >1 means the summed module timers exceed the elapsed compile time, which
    # only happens when builds ran concurrently.
    overlap = cold_median / compile_elapsed_median if compile_elapsed_median and compile_elapsed_median > 0 else None

    reference = ok[0]
    report = {
        "command": args.command,
        "cwd": str(workdir),
        "label": args.label,
        "samples": len(samples),
        "failed_samples": len(failed),
        "cold_module_ms_median": cold_median,
        "cold_module_ms_mad": cold_mad,
        "cold_module_ms_values": [s.cold_module_ms for s in ok],
        "native_compile_ms_median": native_median,
        "warm_module_ms_median": warm_median,
        "cold_wall_ms_median": cold_wall_median,
        "cold_wall_ms_mad": cold_wall_mad,
        "cold_wall_ms_values": [s.cold_wall_ms for s in ok],
        "warm_wall_ms_median": warm_wall_median,
        "compile_elapsed_ms_median": compile_elapsed_median,
        "compile_elapsed_ms_mad": compile_elapsed_mad,
        "overlap_factor": overlap,
        "identities": reference.identities,
        "modules": reference.modules,
        "compiled_events": reference.compiled_events,
        "cached_events": reference.cached_events,
        "error_events": reference.error_events,
        "generated_source_bytes": reference.generated_source_bytes,
        "lto_artifacts": reference.lto_artifacts,
        "lto_bytes": reference.lto_bytes,
        "module_events": reference.events,
        "launches": reference.launches,
    }

    print_measure_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}", file=sys.stderr)
    return 0


def print_measure_report(report: dict[str, Any]) -> None:
    print()
    print(f"Cold module work   {report['cold_module_ms_median']:10.2f} ms  (median of {report['samples']} samples)")
    print(f"  native compile   {report['native_compile_ms_median']:10.2f} ms")
    if report.get("cold_wall_ms_median"):
        print(f"  cold wall clock  {report['cold_wall_ms_median']:10.2f} ms")
    elapsed = report.get("compile_elapsed_ms_median")
    if elapsed is not None:
        print(f"  compile elapsed  {elapsed:10.2f} ms  (cold wall - warm wall)")
    overlap = report.get("overlap_factor")
    if overlap is not None and overlap > 1.15:
        print(
            f"\n  BUILDS OVERLAPPED: summed module timers are {overlap:.2f}x the elapsed\n"
            "  compile time, so modules were built concurrently. Each module's timer\n"
            "  absorbs the others' contention, which inflates 'cold module work' while\n"
            "  elapsed time falls. Use compile elapsed as the headline for this run.\n"
        )
    warm = report["warm_module_ms_median"]
    cold = report["cold_module_ms_median"]
    if warm is not None:
        print(f"  warm repeat      {warm:10.2f} ms")
        # A second run against a populated cache should cost almost nothing. If
        # it still pays most of the cold cost, the application is destroying its
        # own cache and no amount of module restructuring will help.
        if cold > 0 and warm > 0.25 * cold:
            print(
                f"\n  CACHE NOT REUSED: a repeat run still pays {warm / cold:.0%} of the cold cost.\n"
                "  Something is discarding compiled kernels between runs. Look for\n"
                "  wp.clear_kernel_cache() or wp.clear_lto_cache() left in application code,\n"
                "  wp.config.cache_kernels = False, a kernel_cache_dir that is a fresh temp\n"
                "  directory each run or is deleted on exit, or CUDA_CACHE_DISABLE in the\n"
                "  environment. Fix that before restructuring modules -- see CS-2."
            )
    print(f"  generated source {report['generated_source_bytes']:10d} bytes")
    if report["lto_artifacts"]:
        print(f"  LTO artifacts    {report['lto_artifacts']:10d}  ({report['lto_bytes']} bytes)")
    print(f"  launches         {len(report['launches']):10d}")

    # Warp's module timer covers more than the native compiler. When tile/MathDx
    # code is present and the remainder dominates, the build is bounded by
    # device-code linking, and compiler-oriented fixes will not move it.
    remainder = cold - report["native_compile_ms_median"]
    if report["lto_artifacts"] and cold > 0 and remainder > 0.5 * cold:
        print(
            f"\n  LINK-DOMINATED BUILD: {remainder / cold:.0%} of module time is outside the native\n"
            "  compiler, alongside LTO artifacts in the cache. This is device-code linking for tile\n"
            "  operations, not compilation. Reducing generated source or module count will barely\n"
            "  help; what matters is whether each link is needed and whether linked code is being\n"
            "  reused across modules and runs -- see CS-7 and CS-2."
        )
    if not report["launches"]:
        print(
            "\n  NOTE: no kernel launches were recorded. The workload-equivalence check in\n"
            "  `compare` needs these, so it will report UNVERIFIED. This usually means the\n"
            "  instrumentation did not attach -- a wrapper script that re-execs, or a command\n"
            "  that rewrites PYTHONPATH. Timing below is still valid; equivalence is not checked."
        )

    compiled = [e for e in report["module_events"] if e["status"] == "compiled"]
    if not compiled:
        print("\nNo modules were compiled. Either the cache was not isolated or no kernels ran.")
        return

    print(f"\n{len(compiled)} compiled module identities across {report['modules']} distinct module(s):")
    print(f"  {'ms':>9}  {'device':<10} {'block_dim':>9}  {'hash':<9} module")
    by_name: dict[str, list[dict[str, Any]]] = {}
    for event in sorted(compiled, key=lambda e: -e["ms"]):
        block = "-" if event["block_dim"] is None else str(event["block_dim"])
        digest = event.get("hash") or "-"
        print(
            f"  {event['ms']:9.2f}  {event['device']:<10} {block:>9}  {digest:<9} {event.get('name') or event['label']}"
        )
        by_name.setdefault(event.get("name") or event["label"], []).append(event)

    repeats = {name: events for name, events in by_name.items() if len(events) > 1}
    if not repeats:
        return

    # Every extra identity is a full rebuild of that module's whole live kernel
    # set. Naming which of the two causes applies is the point of this report.
    print("\nModules compiled more than once -- each extra build is the whole module again:")
    for name, events in sorted(repeats.items(), key=lambda item: -sum(e["ms"] for e in item[1])):
        hashes = {e.get("hash") for e in events}
        blocks = sorted({e["block_dim"] for e in events}, key=lambda b: (b is None, b))
        total = sum(e["ms"] for e in events)
        print(f"  {name}: {len(events)} builds, {total:.2f} ms total")
        if len(hashes) > 1:
            print(
                f"      identity churn: {len(hashes)} distinct hashes. Something changed the module's "
                "kernel set, options, or generic instances after it first loaded."
            )
        if len(blocks) > 1:
            print(
                f"      block_dim spread: built at {blocks}. On CUDA the module's entire live kernel "
                "set is compiled once per requested block_dim."
            )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def launch_signature(launch: dict[str, Any]) -> tuple:
    """The structural identity of a launch -- deliberately excluding the kernel name.

    Moving a kernel from a closure to module scope is one of the most common
    correct compile-time fixes, and it necessarily changes ``kernel.key``
    (``make_scale_kernel__locals__scale_kernel`` becomes ``scale_kernel``).
    Keying equivalence on the name would flag that fix as a workload change,
    which is exactly backwards. What must not move is the work: the order,
    shape, device, dtypes, block dimension, and adjoint flag of each launch.
    Renames are surfaced separately as an advisory.
    """
    return (
        tuple(launch.get("dim") or ()),
        launch.get("device"),
        tuple(launch.get("dtypes") or ()),
        launch.get("block_dim"),
        launch.get("adjoint"),
    )


def launch_name(launch: dict[str, Any]) -> str:
    """Kernel name with closure qualifiers stripped, for rename reporting."""
    name = str(launch.get("kernel") or "?")
    for marker in ("__locals__", "<locals>."):
        if marker in name:
            name = name.rsplit(marker, 1)[-1].lstrip(".")
    return name


def compare(args: argparse.Namespace) -> int:
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))

    print(f"Baseline : {args.before}")
    print(f"Candidate: {args.after}")

    if before["command"] != after["command"]:
        print("\nWARNING: the two runs used different commands, so they are not comparable:")
        print(f"  baseline : {' '.join(before['command'])}")
        print(f"  candidate: {' '.join(after['command'])}")

    # --- Workload equivalence -------------------------------------------------
    old_launches = before.get("launches") or []
    new_launches = after.get("launches") or []
    old_sig = [launch_signature(item) for item in old_launches]
    new_sig = [launch_signature(item) for item in new_launches]
    workload_ok = old_sig == new_sig
    workload_known = bool(old_launches or new_launches)

    print("\nWorkload")
    if not workload_known:
        # Instrumentation did not attach. Reporting this as a workload change
        # would be a false alarm on an unchanged program, so say what is true:
        # nothing is known either way.
        print("  UNVERIFIED: no launches were recorded in either run, so this check did not run.")
        print("  The probe injects instrumentation via PYTHONPATH; a command that sets its own")
        print("  PYTHONPATH, or a wrapper that re-execs, can drop it. Verify the workload another")
        print("  way (diff numeric output) before trusting any compile-time number below.")
    elif workload_ok:
        renames = sorted(
            {
                (launch_name(o), launch_name(n))
                for o, n in zip(old_launches, new_launches, strict=True)
                if launch_name(o) != launch_name(n)
            }
        )
        print(f"  Preserved: {len(new_sig)} launches, identical in order, dim, dtype, device, and block_dim.")
        if renames:
            print(f"  {len(renames)} kernel(s) renamed (expected when kernels move to module scope):")
            for old_name, new_name in renames:
                print(f"    {old_name} -> {new_name}")
            print("  Renaming is not a workload change, but confirm these are the same computation.")
    else:
        print(f"  CHANGED: baseline ran {len(old_sig)} launches, candidate ran {len(new_sig)}.")
        old_counts: dict[Any, int] = {}
        new_counts: dict[Any, int] = {}
        for launch, sig in zip(old_launches, old_sig, strict=True):
            key = (launch_name(launch), sig)
            old_counts[key] = old_counts.get(key, 0) + 1
        for launch, sig in zip(new_launches, new_sig, strict=True):
            key = (launch_name(launch), sig)
            new_counts[key] = new_counts.get(key, 0) + 1
        for key in sorted(set(old_counts) | set(new_counts), key=str):
            old_n, new_n = old_counts.get(key, 0), new_counts.get(key, 0)
            if old_n == new_n:
                continue
            kernel, (dim, device, dtypes, block_dim, _) = key
            detail = f"dim={list(dim)} device={device} block_dim={block_dim} dtypes={list(dtypes)}"
            print(f"    {kernel:<32} {old_n} -> {new_n}   {detail}")
        for index, (old, new) in enumerate(zip(old_sig, new_sig, strict=False)):
            if old != new:
                old_label = launch_name(old_launches[index])
                new_label = launch_name(new_launches[index])
                print(f"  First divergence at launch #{index + 1}: {old_label} -> {new_label}")
                break

    # --- Compile time ---------------------------------------------------------
    baseline_median = before["cold_module_ms_median"]
    candidate_median = after["cold_module_ms_median"]
    band = max(0.01 * baseline_median, 2.0 * before["cold_module_ms_mad"])
    delta = baseline_median - candidate_median
    percent = (delta / baseline_median * 100.0) if baseline_median else 0.0

    print("\nCold module work")
    print(f"  baseline  {baseline_median:10.2f} ms  (MAD {before['cold_module_ms_mad']:.2f}, n={before['samples']})")
    print(f"  candidate {candidate_median:10.2f} ms  (MAD {after['cold_module_ms_mad']:.2f}, n={after['samples']})")
    print(f"  delta     {delta:10.2f} ms  ({percent:+.1f}%)")
    print(f"  noise band ±{band:.2f} ms")

    if abs(delta) <= band:
        timing = "inconclusive"
        print("  -> Inside the noise band. This is not a measured improvement.")
    elif delta > 0:
        timing = "improved"
        print("  -> Improvement is larger than the noise band.")
    else:
        timing = "regressed"
        print("  -> REGRESSION beyond the noise band.")

    # --- Elapsed compile time -------------------------------------------------
    # "Cold module work" sums per-module timers, so builds that overlap inflate it
    # even when nothing extra was compiled. Elapsed time cannot double-count.
    old_elapsed = before.get("compile_elapsed_ms_median")
    new_elapsed = after.get("compile_elapsed_ms_median")
    elapsed_timing = "unavailable"
    elapsed_percent = 0.0
    if old_elapsed is not None and new_elapsed is not None and old_elapsed > 0:
        elapsed_band = max(0.01 * old_elapsed, 2.0 * (before.get("compile_elapsed_ms_mad") or 0.0))
        elapsed_delta = old_elapsed - new_elapsed
        elapsed_percent = elapsed_delta / old_elapsed * 100.0
        print("\nCompile elapsed (cold wall clock - warm wall clock)")
        print(f"  baseline  {old_elapsed:10.2f} ms")
        print(f"  candidate {new_elapsed:10.2f} ms")
        print(f"  delta     {elapsed_delta:10.2f} ms  ({elapsed_percent:+.1f}%)")
        print(f"  noise band ±{elapsed_band:.2f} ms")
        if abs(elapsed_delta) <= elapsed_band:
            elapsed_timing = "inconclusive"
        elif elapsed_delta > 0:
            elapsed_timing = "improved"
        else:
            elapsed_timing = "regressed"

    # The skill tells the reader to treat the module hash as the receipt for an
    # option change, so the verification step has to show it. An unchanged hash
    # on a module you meant to reconfigure means the option never arrived.
    def _hashes(report: dict[str, Any]) -> dict[str, str]:
        return {
            event.get("name") or event.get("label", ""): event.get("hash", "")
            for event in report.get("module_events", [])
            if event.get("status") == "compiled"
        }

    old_hashes, new_hashes = _hashes(before), _hashes(after)
    if old_hashes or new_hashes:
        print("\nModule identities")
        for name in sorted(set(old_hashes) | set(new_hashes)):
            was, now = old_hashes.get(name, "-"), new_hashes.get(name, "-")
            mark = "" if was == now else "   <- changed"
            print(f"  {name[:44]:<46} {was:>10} -> {now:<10}{mark}")

    def _identities(report: dict[str, Any]) -> set[tuple[Any, ...]]:
        return {
            (
                event.get("name") or event.get("label", ""),
                event.get("hash", ""),
                event.get("device"),
                event.get("block_dim"),
            )
            for event in report.get("module_events", [])
            if event.get("status") == "compiled"
        }

    print("\nStructure")
    # Equal counts do not mean the same code was built: a module keeps its count
    # when its hash changes, which is what an option change looks like. Seed this
    # from the identities themselves so a rebuilt module is never written off as
    # contention below.
    work_identical = _identities(before) == _identities(after)
    for key, name in (
        ("identities", "compiled module identities"),
        ("modules", "distinct modules"),
        ("generated_source_bytes", "generated source bytes"),
        ("lto_artifacts", "LTO artifacts"),
    ):
        old_value, new_value = before.get(key, 0), after.get(key, 0)
        arrow = "" if old_value == new_value else f"   ({new_value - old_value:+d})"
        if old_value != new_value:
            work_identical = False
        print(f"  {name:<28} {old_value:>10} -> {new_value:<10}{arrow}")

    if before.get("warm_module_ms_median") is not None and after.get("warm_module_ms_median") is not None:
        print(
            f"  {'warm repeat module ms':<28} "
            f"{before['warm_module_ms_median']:>10.2f} -> {after['warm_module_ms_median']:<10.2f}"
        )

    print("\nVerdict")
    if after.get("error_events") or after.get("failed_samples"):
        print("  INVALID: the candidate run reported errors or failing samples.")
        return 1
    if not workload_known:
        print("  UNVERIFIED: workload equivalence could not be checked (no launches recorded).")
        print("  Confirm the numeric output is unchanged before reporting the timing below.")
        return 0
    if not workload_ok:
        print("  INVALID: compile time cannot be credited because the workload changed.")
        print("  Restore the missing or altered launches, then measure again.")
        return 1
    # Same launches, same module identities, and byte-identical generated source
    # mean the candidate compiled exactly the same code. Summed module time cannot
    # represent added work in that state, so it is measuring contention -- either
    # builds that now overlap, or a busier machine. Crediting it as a regression
    # would reject a change that made the user wait less.
    if timing == "regressed" and work_identical and workload_ok:
        print("  SAME WORK COMPILED: identical launches, module identities, and generated")
        print("  source bytes, so nothing extra was built. The rise in summed module time")
        print("  is contention between overlapping builds, not new work.")
        if elapsed_timing == "improved":
            print(f"  IMPROVED: elapsed compile time down {elapsed_percent:.1f}%. Report that, not module work.")
            return 0
        if elapsed_timing == "unavailable":
            print("  UNVERIFIED: one of these reports has no elapsed compile time, so this")
            print("  change cannot be judged. Re-measure both arms with the current probe,")
            print("  which always records it, and compare on elapsed rather than module work.")
            return 0
        print("  INCONCLUSIVE: elapsed compile time did not improve beyond noise either.")
        return 0
    if timing == "regressed":
        print("  REGRESSION: workload preserved, but cold compilation got slower.")
        return 1
    if timing == "inconclusive":
        if elapsed_timing == "improved":
            print("  IMPROVED: summed module work is within noise, but elapsed compile time")
            print(f"  is down {elapsed_percent:.1f}%. A change to when modules build shows up here.")
            return 0
        print("  INCONCLUSIVE: workload preserved, compile-time change within noise.")
        return 0
    print(f"  IMPROVED: workload preserved, cold module work down {percent:.1f}%.")
    return 0


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="warp_compile_probe.py",
        description="Measure and compare Warp cold-start module compilation.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    run = sub.add_parser("measure", help="run a command cold in an isolated Warp cache")
    run.add_argument("--samples", type=int, default=3, help="cold samples to collect (default: 3)")
    # The warm pass is no longer optional: two of the probe's diagnoses depend on
    # it. Accepted and ignored so older commands and scripts keep working.
    run.add_argument("--warm", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--json", help="write the full report here for later comparison")
    run.add_argument("--label", default="", help="free-form label recorded in the report")
    run.add_argument("--cwd", default=".", help="working directory for the command (default: .)")
    run.add_argument("--timeout", type=float, default=900.0, help="per-run timeout in seconds (default: 900)")
    run.add_argument("command", nargs=argparse.REMAINDER, help="-- followed by the command to run")
    run.set_defaults(func=measure)

    diff = sub.add_parser("compare", help="compare two measure reports")
    diff.add_argument("before")
    diff.add_argument("after")
    diff.set_defaults(func=compare)

    args = parser.parse_args(argv)
    if args.mode == "measure":
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
