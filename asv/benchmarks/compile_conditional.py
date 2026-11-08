# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cold CUDA compilation of nested searches with short-circuit convergence guards.

Reduced from the FSAI compilation regression caused by copying unnamed value
conditions. Retain the local vectors, nested searches, single-iteration loops
with early exits, and convergence guard: simpler reductions lose the compilation
time difference. No factorization or triangular solve is performed.
"""

from pathlib import Path

import warp as wp
from warp import sparse as sp

WIDTH = 24
Indices = wp.types.vector(length=WIDTH, dtype=wp.int32)
Vector = wp.types.vector(length=WIDTH, dtype=wp.float64)
BestIndex = wp.types.vector(length=1, dtype=wp.int32)
BestScore = wp.types.vector(length=1, dtype=wp.float64)


@wp.func
def lookup_value(offsets: wp.array[int], columns: wp.array[int], values: wp.array[wp.float64], r: int, c: int):
    slot = sp.bsr_block_index(r, c, offsets, columns)
    result = wp.float64(0)
    if slot >= 0:
        result = values[slot]
    return result


@wp.kernel(enable_backward=False)
def nested_search(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.float64],
    tolerance: wp.float64,
    out: wp.array[int],
):
    i = wp.tid()
    pattern = Indices(-1)
    diagonal = Vector(wp.float64(0))
    weights = Vector(wp.float64(0))
    pattern[0] = i
    diagonal[0] = wp.float64(1)
    weights[0] = wp.float64(1)
    size = int(1)
    # The one-iteration loops retain early exits from unrolled candidate batches.
    while size < wp.static(WIDTH):
        candidates = BestIndex(-1)
        scores = BestScore(wp.float64(0))
        for p in range(size):
            row = pattern[p]
            for e in range(offsets[row], offsets[row + 1]):
                c = columns[e]
                selected = bool(c >= i)
                for q in range(size):
                    if pattern[q] == c:
                        selected = True
                if candidates[0] == c:
                    selected = True
                if not selected:
                    residual = wp.float64(0)
                    for q in range(size):
                        residual += lookup_value(offsets, columns, values, c, pattern[q]) * weights[q]
                    score = wp.abs(residual)
                    for rank in range(1):
                        if score > scores[rank] or (score == scores[rank] and c < candidates[rank]):
                            candidates[rank] = c
                            scores[rank] = score
                            break
        if candidates[0] < 0:
            break
        old_weight = weights[0]
        stopped = bool(False)
        for rank in range(1):
            best = candidates[rank]
            if best < 0 or size == wp.static(WIDTH):
                break
            work = Vector(wp.float64(0))
            pivot = wp.float64(1)
            for p in range(size):
                work[p] = lookup_value(offsets, columns, values, best, pattern[p]) / diagonal[p]
                pivot -= work[p] * work[p]
            if pivot <= wp.float64(0.001) or not wp.isfinite(pivot):
                stopped = True
                break
            pattern[size] = best
            diagonal[size] = wp.sqrt(pivot)
            size += 1
        if stopped or (tolerance > wp.float64(0) and wp.float64(1) - old_weight / weights[0] <= tolerance):
            break
    out[i] = size


class CompileNestedSearch:
    """Measure compilation without Warp or NVRTC cache hits between samples."""

    number = 1
    # Four samples per round met a 5% relative 99% CI-width target on sm_120.
    repeat = 4
    warmup_time = 0
    timeout = 120

    def timeraw_cuda(self):
        # ASV runs each sample in a fresh process. Keep initialization outside
        # timing, disable NVRTC's own cache before CUDA initializes, and avoid
        # clearing the kernel cache shared by other benchmarks.
        return (
            "wp.load_module(module=nested_search.module, device='cuda:0')",
            f"""
            import os
            import sys
            from tempfile import TemporaryDirectory

            os.environ["CUDA_CACHE_DISABLE"] = "1"
            sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})

            import warp as wp

            wp.config.log_level = wp.LOG_WARNING
            cache = TemporaryDirectory(prefix="warp-asv-compile-")
            wp.config.kernel_cache_dir = cache.name
            wp.config.use_precompiled_headers = False
            wp.config.cuda_output = "cubin"
            wp.config.mode = "release"
            wp.init()

            from compile_conditional import nested_search
            """,
        )
