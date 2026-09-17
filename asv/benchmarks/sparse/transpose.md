# Captured BSR transpose benchmark

`transpose.py` provides both the ASV benchmark and a standalone JSON exporter.
Both time a graph containing 100 transposes, after an untimed warmup replay.
ASV reports seconds **per batch** and calibrates its own repetitions. The
standalone runner reports milliseconds **per transpose**, with five samples by
default. Initialization, compilation, and capture are excluded.

## Compare two builds

Build both Warp checkouts using the same compiler, CUDA toolkit, build flags,
and Python environment. The original reference for this experiment is
[`0e76b6a0`](https://github.com/NVIDIA/warp/commit/0e76b6a0e21ce0b06e1dab3b98efd8b5afa9f4f9).
Use the benchmark script from the candidate checkout for **both** builds:

```bash
# Set absolute paths to the two independently built checkouts.
WARP_REFERENCE=/path/to/warp-reference
WARP_CANDIDATE=/path/to/warp-candidate

cd "$WARP_CANDIDATE"
PYTHONPATH="$WARP_REFERENCE" uv run --no-sync python \
  asv/benchmarks/sparse/transpose.py --output reference-1.json
PYTHONPATH="$WARP_CANDIDATE" uv run --no-sync python \
  asv/benchmarks/sparse/transpose.py --output candidate-1.json
PYTHONPATH="$WARP_CANDIDATE" uv run --no-sync python \
  asv/benchmarks/sparse/transpose.py --output candidate-2.json
PYTHONPATH="$WARP_REFERENCE" uv run --no-sync python \
  asv/benchmarks/sparse/transpose.py --output reference-2.json
```

Check `warp_path` in each JSON to verify the selected package. Each checkout
must contain its corresponding native library; changing `PYTHONPATH` does not
rebuild it. Record both commit IDs and build flags with the results. Run on an
otherwise idle GPU and discard measurements affected by other compute work.
Compare geometric means of the process medians; do not treat samples from a
single process as independent runs. Throughput gain is `reference / candidate - 1`.

## Why these shapes?

| Pattern | Purpose |
|---|---|
| `stencil8`, `stencil32` | About 512K active blocks at two stencil widths. |
| `dense` | Exercise the global-sort fallback. |
| `overallocated` | Reserve S2/Q1 spaces for 64K cells, with 24K active blocks approximating a measured MPM contact map. |
| `wide_hot_column` | Concentrate all entries in one output row. |
| `shared_columns8` | Same dimensions and active count as `stencil8`, concentrated in eight output rows. |

These are synthetic matrices, not full MPM simulations. Include the skewed
patterns when reporting results: the experimental row-sort implementation has
a known shared-column regression, tracked in [#1971](https://github.com/NVIDIA/warp/issues/1971).
Correctness is checked separately by the sparse transpose tests, including
captured topology changes, padding, rectangular blocks, and exact value bytes.
