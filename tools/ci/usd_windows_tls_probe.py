# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe Windows USD native-library loading from fresh Python processes."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import ctypes
import dataclasses
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import traceback
from ctypes import wintypes
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class CommandResult:
    return_code: int
    timed_out: bool
    duration_seconds: float
    stdout_path: str
    stderr_path: str

    @property
    def return_code_hex(self) -> str:
        return format_exit_code(self.return_code)


def format_exit_code(return_code: int) -> str:
    """Format a process exit code as an unsigned 32-bit Windows status."""
    return f"0x{return_code & 0xFFFFFFFF:08X}"


def _module_priority(path: str) -> tuple[int, str]:
    name = Path(path).name.lower()
    normalized_path = path.lower()
    if name.startswith(("msvcp", "vcruntime", "concrt", "ucrt")):
        priority = 0
    elif name.startswith(("usd", "_tf", "_sdf")) or "site-packages\\pxr\\" in normalized_path:
        priority = 1
    elif name.startswith("tbb"):
        priority = 2
    elif name.startswith("torch") or "blosc" in normalized_path:
        priority = 3
    elif name.startswith(("warp", "wp_")):
        priority = 4
    elif name.startswith(("cuda", "nvcuda", "nvrtc", "nvjitlink", "cudart")):
        priority = 5
    elif name.startswith("python"):
        priority = 6
    else:
        priority = 99
    return priority, normalized_path


def filter_relevant_module_paths(paths: list[str]) -> list[str]:
    """Return modules relevant to the USD, MSVC, Torch, Warp, and CUDA interaction."""
    relevant = [path for path in paths if _module_priority(path)[0] < 99]
    return sorted(set(relevant), key=_module_priority)


def loaded_module_paths() -> list[str]:
    """Enumerate modules loaded in the current Windows process."""
    if os.name != "nt":
        return []

    list_modules_all = 0x03
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = ()
    get_current_process.restype = wintypes.HANDLE
    enum_process_modules = psapi.EnumProcessModulesEx
    enum_process_modules.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_void_p),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
    )
    enum_process_modules.restype = wintypes.BOOL
    get_module_filename = psapi.GetModuleFileNameExW
    get_module_filename.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.LPWSTR, wintypes.DWORD)
    get_module_filename.restype = wintypes.DWORD
    process = get_current_process()

    capacity = 2048
    modules = (ctypes.c_void_p * capacity)()
    bytes_needed = wintypes.DWORD()
    if not enum_process_modules(
        process,
        modules,
        ctypes.sizeof(modules),
        ctypes.byref(bytes_needed),
        list_modules_all,
    ):
        raise ctypes.WinError()

    count = min(capacity, bytes_needed.value // ctypes.sizeof(ctypes.c_void_p))
    paths = []
    for module in modules[:count]:
        buffer = ctypes.create_unicode_buffer(32768)
        if get_module_filename(process, module, buffer, len(buffer)):
            paths.append(buffer.value)
    return paths


def native_thread_count() -> int | None:
    """Count all native threads owned by the current Windows process."""
    if os.name != "nt":
        return None

    class ThreadEntry32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    create_snapshot.restype = wintypes.HANDLE
    thread_first = kernel32.Thread32First
    thread_first.argtypes = (wintypes.HANDLE, ctypes.POINTER(ThreadEntry32))
    thread_first.restype = wintypes.BOOL
    thread_next = kernel32.Thread32Next
    thread_next.argtypes = (wintypes.HANDLE, ctypes.POINTER(ThreadEntry32))
    thread_next.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    snapshot = create_snapshot(0x00000004, 0)
    invalid_handle_value = ctypes.c_void_p(-1).value
    if not snapshot or snapshot == invalid_handle_value:
        raise ctypes.WinError()

    entry = ThreadEntry32()
    entry.dwSize = ctypes.sizeof(entry)
    count = 0
    try:
        has_entry = thread_first(snapshot, ctypes.byref(entry))
        while has_entry:
            if entry.th32OwnerProcessID == os.getpid():
                count += 1
            has_entry = thread_next(snapshot, ctypes.byref(entry))
    finally:
        close_handle(snapshot)
    return count


def package_versions() -> dict[str, str]:
    """Return the package versions that shape the process loader state."""
    versions = {}
    for distribution in ("usd-core", "warp-lang", "torch", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def emit_event(kind: str, **fields: object) -> None:
    """Write a structured event immediately so it survives a later native crash."""
    event = {
        "kind": kind,
        "pid": os.getpid(),
        "time": time.time(),
        **fields,
    }
    print(json.dumps(event, sort_keys=True), flush=True)


def emit_checkpoint(stage: str) -> None:
    """Record process state at an import boundary."""
    emit_event(
        "checkpoint",
        stage=stage,
        native_thread_count=native_thread_count(),
        modules=filter_relevant_module_paths(loaded_module_paths()),
    )


@contextlib.contextmanager
def hold_threads(count: int):
    """Keep ``count`` Python threads alive across a measured DLL import."""
    release = threading.Event()
    ready_events = [threading.Event() for _ in range(count)]

    def wait_for_release(ready: threading.Event) -> None:
        ready.set()
        release.wait()

    threads = [
        threading.Thread(target=wait_for_release, args=(ready,), name=f"usd-tls-holder-{index}", daemon=True)
        for index, ready in enumerate(ready_events)
    ]
    for thread in threads:
        thread.start()
    try:
        for ready in ready_events:
            if not ready.wait(timeout=30):
                raise TimeoutError("Timed out starting TLS pressure threads")
        yield
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=30)


def warm_cuda_runtimes() -> tuple[object, object]:
    """Initialize Warp, Torch, and their CUDA runtime state."""
    emit_checkpoint("before-warp")

    import warp as wp  # noqa: PLC0415

    wp.init()
    devices = wp.get_cuda_devices()
    if not devices:
        raise RuntimeError("No CUDA device is visible to Warp")
    warp_array = wp.zeros(1, dtype=wp.float32, device=devices[0])
    wp.synchronize_device(devices[0])
    emit_checkpoint("after-warp-cuda")

    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device is visible to PyTorch")
    torch_tensor = torch.ones(1, device="cuda")
    torch_tensor.add_(1)
    torch.cuda.synchronize()
    emit_checkpoint("after-torch-cuda")
    return warp_array, torch_tensor


def import_tf() -> None:
    """Import the USD Tf extension at a measured boundary."""
    emit_checkpoint("before-pxr-tf")
    from pxr import Tf  # noqa: F401, PLC0415

    emit_checkpoint("after-pxr-tf")


def import_sdf() -> None:
    """Import the modular USD Sdf extension at a measured boundary."""
    emit_checkpoint("before-pxr-sdf")
    from pxr import Sdf  # noqa: F401, PLC0415

    emit_checkpoint("after-pxr-sdf")


def run_child(case: str, held_thread_count: int, attempt: int) -> int:
    """Run one import experiment in the process that may crash."""
    emit_event(
        "environment",
        attempt=attempt,
        case=case,
        held_thread_count=held_thread_count,
        hostname=socket.gethostname(),
        packages=package_versions(),
        platform=platform.platform(),
        python=sys.version,
    )

    try:
        if case == "early-tf":
            with hold_threads(held_thread_count):
                import_tf()
            runtime_state = warm_cuda_runtimes()
        else:
            runtime_state = warm_cuda_runtimes()
            with hold_threads(held_thread_count):
                import_tf()
                if case == "late-sdf":
                    import_sdf()
                elif case != "late-tf":
                    raise ValueError(f"Unknown case: {case}")

        # Retain CUDA allocations through the final USD import.
        assert runtime_state
        emit_event("complete", attempt=attempt, case=case)
        return 0
    except BaseException as error:
        emit_event(
            "exception",
            attempt=attempt,
            case=case,
            error_type=type(error).__name__,
            error=repr(error),
            errno=getattr(error, "errno", None),
            winerror=getattr(error, "winerror", None),
        )
        traceback.print_exc()
        return 1


def run_command(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> CommandResult:
    """Run a child command without raising when it fails or crashes."""
    started = time.monotonic()
    timed_out = False
    environment = os.environ.copy()
    environment["PYTHONFAULTHANDLER"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"

    with stdout_path.open("w", encoding="utf-8", errors="backslashreplace") as stdout_file:
        with stderr_path.open("w", encoding="utf-8", errors="backslashreplace") as stderr_file:
            process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, env=environment)
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                return_code = process.wait()
                print(f"Child timed out after {timeout_seconds} seconds", file=stderr_file, flush=True)

    return CommandResult(
        return_code=return_code,
        timed_out=timed_out,
        duration_seconds=time.monotonic() - started,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def run_attempt(
    *,
    case: str,
    held_thread_count: int,
    attempt: int,
    output_directory: Path,
    timeout_seconds: int,
) -> dict[str, object]:
    """Launch and summarize one fresh child process."""
    stem = f"{case}-threads-{held_thread_count:03d}-attempt-{attempt:03d}"
    result = run_command(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--case",
            case,
            "--held-threads",
            str(held_thread_count),
            "--attempt",
            str(attempt),
        ],
        stdout_path=output_directory / f"{stem}.stdout.log",
        stderr_path=output_directory / f"{stem}.stderr.log",
        timeout_seconds=timeout_seconds,
    )
    return {
        "attempt": attempt,
        "case": case,
        "held_thread_count": held_thread_count,
        **dataclasses.asdict(result),
        "return_code_hex": result.return_code_hex,
    }


def run_parent(args: argparse.Namespace) -> int:
    """Run repeated child experiments and retain results from every attempt."""
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    environment = {
        "hostname": socket.gethostname(),
        "packages": package_versions(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    (output_directory / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_attempt,
                case=args.case,
                held_thread_count=args.held_threads,
                attempt=attempt,
                output_directory=output_directory,
                timeout_seconds=args.child_timeout_seconds,
            )
            for attempt in range(1, args.repeat + 1)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    results.sort(key=lambda result: int(result["attempt"]))
    failures = sum(result["return_code"] != 0 or result["timed_out"] for result in results)
    summary = {
        "case": args.case,
        "failures": failures,
        "held_thread_count": args.held_threads,
        "repeat": args.repeat,
        "results": results,
        "workers": args.workers,
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, sort_keys=True))
    return int(failures != 0)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("early-tf", "late-tf", "late-sdf"), required=True)
    parser.add_argument("--held-threads", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--child-timeout-seconds", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("usd-tls-artifacts"))
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.held_threads < 0:
        parser.error("--held-threads must not be negative")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main() -> int:
    args = parse_arguments()
    if args.child:
        return run_child(args.case, args.held_threads, args.attempt)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
