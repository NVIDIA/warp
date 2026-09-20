# Coordinate reconstruction for tiled CUDA launches: measurement record

Companion record for `warp/native/builtin.h` (`launch_coord_tile`) and the
`builtin_tid*` macros in `warp/_src/codegen.py`. Everything below was produced on the
two machines listed under *Tested configurations*; no number here is estimated.

## What was measured

Two harnesses, both driven by the same contract:

- **E2E**: a three-stage dependent tiled pipeline (`tile_load` → map → `tile_sum` →
  `tile_store`) over one buffer, captured once into a CUDA graph and replayed. This is
  the shape a tiled Warp workload actually has.
- **Micro**: a minimal tiled kernel whose two variants differ only in launch form
  (`coord_mult == blockDim.x` versus `coord_mult == 1`), so the coordinate path is
  isolated from tile arithmetic.

Method, applied identically to both variants:

- one CUDA graph per arm; `wp.Event(enable_timing=True)` around `wp.capture_launch`,
  with the synchronize outside the timed window
- fresh process per arm, native library built separately per tree
- alternated `B P P B` order so label order cannot be confounded with drift
- an arm whose within-arm sample spread exceeds 5% is discarded **together with every
  arm of the same workload**; selection is by spread, never by which arm looks better
- per-workload speedup is `median(baseline) / median(patch)` over arm medians; the
  pooled figure is the geometric mean of those per-workload ratios

## Tested configurations

| | host A | host B |
|---|---|---|
| GPU | NVIDIA L20, sm_89 | NVIDIA GeForce RTX 5090, sm_120 |
| CUDA | 12.8.61 | 13.0.88 |
| base | upstream `main` `015e5a1` | same |
| interpreter | Python 3.12.13 | Python 3.12.13 |

Baseline and patched trees are separate checkouts of the same base commit with the
patch applied to one of them only.

## Results

### E2E, three-stage tiled pipeline

| host | workload | baseline (ms/pipeline) | patch (ms/pipeline) | speedup | max spread |
|---|---|---:|---:|---:|---:|
| L20 | n=262144, block_dim=256, max_blocks=256 | 0.06751 | 0.06748 | 1.0004x | 0.43% |
| L20 | n=262144, block_dim=256, max_blocks=32 | 0.48396 | 0.48392 | 1.0001x | 0.06% |
| L20 | n=1048576, block_dim=128, max_blocks=512 | 0.13315 | 0.13313 | 1.0002x | 0.21% |
| L20 | n=1048576, block_dim=256, max_blocks=256 | 0.26021 | 0.26021 | 1.0000x | 0.12% |
| 5090 | n=65536, block_dim=256, max_blocks=256 | 0.01970 | 0.01970 | 0.9999x | 0.13% |
| 5090 | n=262144, block_dim=256, max_blocks=256 | 0.07192 | 0.07190 | 1.0003x | 0.07% |
| 5090 | n=262144, block_dim=256, max_blocks=32 | 0.54875 | 0.54864 | 1.0002x | 0.02% |
| 5090 | n=1048576, block_dim=256, max_blocks=256 | 0.28079 | 0.28040 | 1.0014x | 0.64% |

Geometric means: **1.0002x** (sm_89), **1.0005x** (sm_120).

### Micro, coordinate isolation

| host | workload | baseline (ms/launch) | patch (ms/launch) | speedup | max spread |
|---|---|---:|---:|---:|---:|
| L20 | n_tiles=2048, max_blocks=128 | 0.01471 | 0.01471 | 1.0002x | 0.44% |
| L20 | n_tiles=16384, max_blocks=128 | 0.10026 | 0.10026 | 1.0000x | 0.31% |
| L20 | n_tiles=16384, max_blocks=32 | 0.38109 | 0.38108 | 1.0000x | 0.09% |
| 5090 | n_tiles=2048, max_blocks=128 | 0.00910 | 0.00910 | 1.0001x | 0.35% |
| 5090 | n_tiles=16384, max_blocks=128 | 0.06259 | 0.06261 | 0.9996x | 0.08% |
| 5090 | n_tiles=16384, max_blocks=32 | 0.24609 | 0.24609 | 1.0000x | 0.02% |
| 5090 | n_tiles=65536, max_blocks=8 | 3.83509 | 3.83506 | 1.0000x | 0.05% |

Geometric means: **1.0000x** (sm_89), **0.9999x** (sm_120).

The `n_tiles=65536, max_blocks=8` row is the heaviest grid-stride case (8192 tiles per
thread). On shared devices it was unmeasurable at 37–76% spread; the row above was
taken on an exclusively held device and lands at 0.05%.

### Clean re-validation

Repeated from a fresh detached worktree at the patch SHA with a freshly compiled native
library and a fresh process per arm:

| workload | baseline | patch | speedup | max spread |
|---|---:|---:|---:|---:|
| E2E, n=262144, max_blocks=256 | 0.07206 | 0.07206 | 1.0001x | 0.48% |
| E2E, n=1048576, max_blocks=256 | 0.28078 | 0.28034 | 1.0016x | 0.04% |
| Micro, n_tiles=16384, max_blocks=128 | 0.06258 | 0.06259 | 1.0000x | 0.13% |

## Reading of these numbers

Every paired ratio lands between 0.9996x and 1.0016x, i.e. inside the spread of the
measurement, on both architectures and in every build state. This is consistent with
`launch_coord_tile` being the algebraic identity for `coord_mult > 1`
(`(linear / coord_mult) * coord_mult + linear % coord_mult == linear`): it moves the
coordinate computation around rather than removing any of it, so there is nothing for a
speedup to come from.

**No speedup is claimed for this change.** If the tile/lane split is not considered
clearer at the call site, closing this without merging is a reasonable outcome.

## Correctness evidence

| suite | baseline | patched |
|---|---|---|
| `warp/tests/tile/test_tile.py` (sm_89) | 157 pass / 0 fail | 157 pass / 0 fail |
| `warp/tests/tile/*` (sm_89, 34 suites) | 1080 pass / 0 fail | 1080 pass / 0 fail |
| `warp/tests/tile/*` (sm_120, 35 suites) | 502 pass / 6 fail | 502 pass / 6 fail |
| `warp/tests/tile/test_tile_launch_coord.py` | — | pass |

The six sm_120 failures are identical on both trees and unrelated to this change
(`test_register_tile_cpu_blocks`, the `*_oob_reports_*` group, and
`test_thread_tile_uses_logical_block_dimension`).

`launch_coord_tile` was also compared against `wp::launch_coord` over 9354
(`coord_mult`, thread) pairs across 1D/2D/3D/4D launch shapes, folded and unfolded:
0 mismatches.
