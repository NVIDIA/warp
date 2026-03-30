# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Informal overhead benchmark: wp.log() vs wp.printf() vs baseline.

Run directly:
    uv run warp/tests/bench_kernel_log.py

Prints a results table and exits 0.  No timing assertions are made.
"""

import contextlib
import ctypes
import os
import statistics
import time

import warp as wp

wp.init()

# ---------------------------------------------------------------------------
# Kernel definitions at module scope (required for inspect.getsourcelines)
# ---------------------------------------------------------------------------

N = 1_000_000  # threads per launch
N_PRINTF = 10_000  # smaller N for printf (avoids GPU printf buffer overflow / hang)
REPEATS = 5


@wp.kernel
def _bench_baseline(out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    out[i] = i * 2 + 1


@wp.kernel
def _bench_log_dead(out: wp.array(dtype=wp.int32)):
    """wp.log() on a dead path — measures codegen/arg-passing overhead only."""
    i = wp.tid()
    if i < 0:  # always false
        wp.log(wp.LOG_DEBUG, "dead path", i)
    out[i] = i * 2 + 1


@wp.kernel
def _bench_log_live(out: wp.array(dtype=wp.int32)):
    """Every thread logs — ring-buffer atomic + 16-byte store overhead."""
    i = wp.tid()
    wp.log(wp.LOG_DEBUG, "bench", i)
    out[i] = i * 2 + 1


@wp.kernel
def _bench_printf(out: wp.array(dtype=wp.int32)):
    """Every thread calls wp.printf — serialised I/O reference."""
    i = wp.tid()
    wp.printf("%d\n", i)
    out[i] = i * 2 + 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _silence_stdout():
    """Redirect C-level stdout to /dev/null (suppresses wp.printf output)."""
    libc = ctypes.CDLL(None)
    libc.fflush(None)  # flush any pending C output before redirect
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)
    try:
        yield
    finally:
        libc.fflush(None)  # flush buffered writes to /dev/null before restore
        os.dup2(saved, 1)
        os.close(saved)


def _time(kernel, out, device, n):
    """Wall-clock timing for a kernel (ms), median over REPEATS."""
    wp.launch(kernel, dim=n, inputs=[out], device=device)
    wp.synchronize_device(device)  # warm-up

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        wp.launch(kernel, dim=n, inputs=[out], device=device)
        wp.synchronize_device(device)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run(device):
    is_cuda = str(device).startswith("cuda")

    print(f"\n{'=' * 64}")
    print(f"Device: {device}   N = {N:,} threads   repeats = {REPEATS}")
    if not is_cuda:
        print(f"  (wp.printf measured separately at N = {N_PRINTF:,})")
    print(f"{'=' * 64}")

    out = wp.zeros(N, dtype=wp.int32, device=device)
    out_small = wp.zeros(N_PRINTF, dtype=wp.int32, device=device)

    kernels = [
        ("baseline (no log)", _bench_baseline, N, False),
        ("wp.log  dead path  (if i<0)", _bench_log_dead, N, False),
        ("wp.log  live       (every thread)", _bench_log_live, N, False),
    ]
    if not is_cuda:
        # CUDA printf blocks/hangs with large N; skip on GPU
        kernels.append((f"wp.printf          (N={N_PRINTF:,})", _bench_printf, N_PRINTF, True))

    results = {}
    for label, kernel, n, use_silence in kernels:
        buf = out if n == N else out_small
        ctx = _silence_stdout() if use_silence else contextlib.nullcontext()
        with ctx:
            ms = _time(kernel, buf, device, n)
        results[label] = (ms, n)

    baseline_ms = results["baseline (no log)"][0]
    print(f"\n  {'Variant':<40} {'Median ms':>10}  {'vs baseline':>12}")
    print("  " + "-" * 66)
    for label, (ms, n) in results.items():
        if n == N:
            pct = (ms / baseline_ms - 1) * 100
            pct_str = f"{pct:>+11.1f}%"
        else:
            pct_str = f"{'(diff N)':>12}"
        print(f"  {label:<40} {ms:>10.4f}  {pct_str}")

    dead_pct = (results["wp.log  dead path  (if i<0)"][0] / baseline_ms - 1) * 100
    live_ms = results["wp.log  live       (every thread)"][0]

    print(f"\n  Dead-path overhead : {dead_pct:+.1f}%")
    if not is_cuda:
        printf_ms = results[f"wp.printf          (N={N_PRINTF:,})"][0]
        # Scale printf to N for a fair per-thread comparison
        printf_scaled = printf_ms * N / N_PRINTF
        speedup = printf_scaled / live_ms
        print(f"  wp.log is ~{speedup:.0f}x faster than wp.printf per thread (scaled)")
    print(
        f"  Note: live overhead with N={N:,} >> capacity={wp.config.kernel_log_capacity}; "
        "most atomic_adds are overflow increments"
    )


if __name__ == "__main__":
    devices = ["cpu"]
    if wp.is_cuda_available():
        devices.append("cuda:0")

    for dev in devices:
        _run(dev)

    print()
