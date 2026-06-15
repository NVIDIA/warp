# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare exact NanoVDB point-to-grid allocation with rebuildable volumes.

Example:

    CUDA_PATH=/usr/local/cuda-13.2 uv run warp/examples/benchmarks/benchmark_volume_points_to_grid.py \
        --points 200000 --unique-ratio 0.5 --iterations 20 --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import warp as wp


@dataclass(frozen=True)
class TopologyCapacities:
    leaf_nodes: int
    lower_nodes: int
    upper_nodes: int
    active_voxels: int


@dataclass
class BenchmarkData:
    points: wp.array
    capacities: TopologyCapacities
    point_count: int
    unique_count: int


@dataclass(frozen=True)
class TimingResult:
    name: str
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class AllocationSizes:
    exact_bytes: int
    rebuildable_bytes: int


def unique_rows(values: np.ndarray) -> np.ndarray:
    return np.unique(np.ascontiguousarray(values), axis=0)


def make_lattice_points(point_count: int, unique_count: int, *, scale: int, seed: int) -> np.ndarray:
    side = math.ceil(unique_count ** (1.0 / 3.0))
    ids = np.arange(unique_count, dtype=np.int64)
    coords = np.empty((unique_count, 3), dtype=np.int32)
    coords[:, 0] = ids % side
    coords[:, 1] = (ids // side) % side
    coords[:, 2] = ids // (side * side)
    coords -= side // 2

    rng = np.random.default_rng(seed)
    if point_count > unique_count:
        selected = np.concatenate(
            (np.arange(unique_count, dtype=np.int64), rng.integers(0, unique_count, point_count - unique_count))
        )
        rng.shuffle(selected)
        coords = coords[selected]

    return coords * scale


def hierarchy_capacities(points: np.ndarray, *, active_voxels: bool, capacity_factor: float) -> TopologyCapacities:
    if active_voxels:
        active_voxel_count = len(unique_rows(points))
    else:
        active_voxel_count = 0

    leaves = unique_rows(np.floor_divide(points, 8))
    lower_nodes = unique_rows(np.floor_divide(points, 128))
    upper_nodes = unique_rows(np.floor_divide(points, 4096))

    def apply_factor(count: int) -> int:
        return max(1, math.ceil(count * capacity_factor))

    return TopologyCapacities(
        leaf_nodes=apply_factor(len(leaves)),
        lower_nodes=apply_factor(len(lower_nodes)),
        upper_nodes=apply_factor(len(upper_nodes)),
        active_voxels=apply_factor(active_voxel_count),
    )


def make_benchmark_data(
    kind: str, point_count: int, unique_ratio: float, capacity_factor: float, seed: int, device: wp.Device
) -> BenchmarkData:
    unique_count = max(1, min(point_count, math.ceil(point_count * unique_ratio)))
    scale = 8 if kind == "tiles" else 1
    points_np = make_lattice_points(point_count, unique_count, scale=scale, seed=seed)
    capacities = hierarchy_capacities(points_np, active_voxels=kind == "voxels", capacity_factor=capacity_factor)
    points = wp.array(points_np, dtype=wp.int32, device=device)
    return BenchmarkData(points=points, capacities=capacities, point_count=point_count, unique_count=unique_count)


def synchronize(device: wp.Device) -> None:
    wp.synchronize_device(device)


def timed_samples(
    label: str,
    func: Callable[[], object],
    *,
    iterations: int,
    warmup: int,
    device: wp.Device,
    cleanup: Callable[[object], None] | None = None,
) -> TimingResult:
    for _ in range(warmup):
        value = func()
        synchronize(device)
        if cleanup:
            cleanup(value)
            value = None
            gc.collect()
            synchronize(device)

    samples = []
    for _ in range(iterations):
        gc.collect()
        synchronize(device)
        start = time.perf_counter_ns()
        value = func()
        synchronize(device)
        end = time.perf_counter_ns()
        samples.append((end - start) / 1.0e6)
        if cleanup:
            cleanup(value)
            value = None
            gc.collect()
            synchronize(device)

    return TimingResult(
        name=label,
        median_ms=statistics.median(samples),
        mean_ms=statistics.mean(samples),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def timed_enqueued_loop(
    label: str,
    func: Callable[[], None],
    *,
    iterations: int,
    warmup: int,
    device: wp.Device,
) -> TimingResult:
    for _ in range(warmup):
        func()
    synchronize(device)

    samples = []
    for _ in range(3):
        synchronize(device)
        start = time.perf_counter_ns()
        for _ in range(iterations):
            func()
        synchronize(device)
        end = time.perf_counter_ns()
        samples.append((end - start) / (1.0e6 * iterations))

    return TimingResult(
        name=label,
        median_ms=statistics.median(samples),
        mean_ms=statistics.mean(samples),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def allocate_exact(kind: str, data: BenchmarkData, device: wp.Device) -> wp.Volume:
    if kind == "tiles":
        return wp.Volume.allocate_by_tiles(data.points, voxel_size=1.0, bg_value=0.0, device=device)
    return wp.Volume.allocate_by_voxels(data.points, voxel_size=1.0, device=device)


def allocate_rebuildable(kind: str, data: BenchmarkData, status: wp.array, device: wp.Device) -> wp.Volume:
    capacities = data.capacities
    if kind == "tiles":
        return wp.Volume.allocate_by_tiles(
            data.points,
            voxel_size=1.0,
            bg_value=0.0,
            device=device,
            graph_rebuildable=True,
            max_tiles=capacities.leaf_nodes,
            max_lower_nodes=capacities.lower_nodes,
            max_upper_nodes=capacities.upper_nodes,
            status=status,
        )

    return wp.Volume.allocate_by_voxels(
        data.points,
        voxel_size=1.0,
        device=device,
        graph_rebuildable=True,
        max_active_voxels=capacities.active_voxels,
        max_leaf_nodes=capacities.leaf_nodes,
        max_lower_nodes=capacities.lower_nodes,
        max_upper_nodes=capacities.upper_nodes,
        status=status,
    )


def rebuild_volume(kind: str, volume: wp.Volume, data: BenchmarkData, status: wp.array) -> None:
    if kind == "tiles":
        volume.rebuild_by_tiles(data.points, status=status)
    else:
        volume.rebuild_by_voxels(data.points, status=status)


def check_status(label: str, status: wp.array) -> None:
    value = int(status.numpy()[0])
    if value != wp.Volume.REBUILD_SUCCESS:
        raise RuntimeError(f"{label} failed with rebuild status 0x{value:08x}")


def cleanup_volume(volume: wp.Volume) -> None:
    del volume


def get_volume_size(volume: wp.Volume) -> int:
    return volume.get_grid_info().size_in_bytes


def measure_allocation_sizes(kind: str, data: BenchmarkData, device: wp.Device) -> AllocationSizes:
    status = wp.zeros(1, dtype=wp.uint32, device=device)
    exact_volume = allocate_exact(kind, data, device)
    rebuildable_volume = allocate_rebuildable(kind, data, status, device)
    synchronize(device)
    check_status(f"{kind} rebuildable size probe", status)

    sizes = AllocationSizes(
        exact_bytes=get_volume_size(exact_volume),
        rebuildable_bytes=get_volume_size(rebuildable_volume),
    )
    del exact_volume
    del rebuildable_volume
    gc.collect()
    synchronize(device)
    return sizes


def benchmark_kind(kind: str, data: BenchmarkData, args: argparse.Namespace, device: wp.Device) -> list[TimingResult]:
    setup_status = wp.zeros(1, dtype=wp.uint32, device=device)
    rebuild_status = wp.zeros(1, dtype=wp.uint32, device=device)

    results = [
        timed_samples(
            "exact allocate",
            lambda: allocate_exact(kind, data, device),
            iterations=args.iterations,
            warmup=args.warmup,
            device=device,
            cleanup=cleanup_volume,
        ),
        timed_samples(
            "rebuildable setup",
            lambda: allocate_rebuildable(kind, data, setup_status, device),
            iterations=args.iterations,
            warmup=args.warmup,
            device=device,
            cleanup=cleanup_volume,
        ),
    ]
    check_status(f"{kind} rebuildable setup", setup_status)

    volume = allocate_rebuildable(kind, data, rebuild_status, device)
    synchronize(device)
    check_status(f"{kind} initial rebuildable allocation", rebuild_status)

    results.append(
        timed_enqueued_loop(
            "rebuild enqueue",
            lambda: rebuild_volume(kind, volume, data, rebuild_status),
            iterations=args.iterations,
            warmup=args.warmup,
            device=device,
        )
    )
    check_status(f"{kind} direct rebuild", rebuild_status)

    if not args.skip_graph:
        if not device.is_mempool_supported:
            print(f"Skipping graph replay for {kind}: device {device} does not support CUDA memory pools.")
        else:
            if not wp.is_mempool_enabled(device):
                wp.set_mempool_enabled(device, True)

            graph_volume = allocate_rebuildable(kind, data, rebuild_status, device)
            synchronize(device)
            check_status(f"{kind} graph setup", rebuild_status)

            with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                rebuild_volume(kind, graph_volume, data, rebuild_status)

            results.append(
                timed_enqueued_loop(
                    "graph replay",
                    lambda: wp.capture_launch(capture.graph),
                    iterations=args.iterations,
                    warmup=args.warmup,
                    device=device,
                )
            )
            check_status(f"{kind} graph replay", rebuild_status)

    return results


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    scaled = float(value)
    for unit in units:
        if scaled < 1024.0 or unit == units[-1]:
            return f"{scaled:.2f} {unit}" if unit != "B" else f"{value} B"
        scaled /= 1024.0

    raise AssertionError("unreachable")


def print_case_header(kind: str, data: BenchmarkData, sizes: AllocationSizes) -> None:
    capacities = data.capacities
    print()
    print(f"{kind.upper()}: {data.point_count} points, {data.unique_count} unique generated points")
    print(
        "capacities: "
        f"leaf={capacities.leaf_nodes}, lower={capacities.lower_nodes}, "
        f"upper={capacities.upper_nodes}, active_voxels={capacities.active_voxels}"
    )
    print(f"grid bytes: exact={format_bytes(sizes.exact_bytes)}, rebuildable={format_bytes(sizes.rebuildable_bytes)}")


def print_results(results: list[TimingResult]) -> None:
    baseline = results[0].median_ms
    print(f"{'path':<20s} {'median ms':>12s} {'mean ms':>12s} {'min ms':>12s} {'max ms':>12s} {'speedup':>10s}")
    print("-" * 84)
    for result in results:
        speedup = baseline / result.median_ms if result.median_ms > 0.0 else float("inf")
        print(
            f"{result.name:<20s} {result.median_ms:12.4f} {result.mean_ms:12.4f} "
            f"{result.min_ms:12.4f} {result.max_ms:12.4f} {speedup:10.2f}x"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="CUDA device to benchmark.")
    parser.add_argument("--mode", choices=("tiles", "voxels", "both"), default="both")
    parser.add_argument("--points", type=int, default=100_000, help="Total input points, including duplicates.")
    parser.add_argument(
        "--unique-ratio",
        type=float,
        default=0.5,
        help="Fraction of total points that should be unique before duplicates are added.",
    )
    parser.add_argument(
        "--capacity-factor",
        type=float,
        default=1.1,
        help="Multiplier applied to exact generated topology counts for rebuildable capacities.",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-graph", action="store_true", help="Skip CUDA graph replay measurements.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points <= 0:
        raise ValueError("--points must be positive")
    if not (0.0 < args.unique_ratio <= 1.0):
        raise ValueError("--unique-ratio must be in the interval (0, 1]")
    if args.capacity_factor < 1.0:
        raise ValueError("--capacity-factor must be at least 1.0")
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("--iterations must be positive and --warmup must be non-negative")

    wp.config.log_level = wp.LOG_WARNING
    wp.init()

    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("This benchmark requires a CUDA device.")

    if device.is_mempool_supported and not wp.is_mempool_enabled(device):
        wp.set_mempool_enabled(device, True)

    kinds = ("tiles", "voxels") if args.mode == "both" else (args.mode,)
    print(f"device: {device}")
    print(f"iterations: {args.iterations}, warmup: {args.warmup}")

    for index, kind in enumerate(kinds):
        data = make_benchmark_data(
            kind,
            point_count=args.points,
            unique_ratio=args.unique_ratio,
            capacity_factor=args.capacity_factor,
            seed=args.seed + index,
            device=device,
        )
        sizes = measure_allocation_sizes(kind, data, device)
        print_case_header(kind, data, sizes)
        print_results(benchmark_kind(kind, data, args, device))


if __name__ == "__main__":
    main()
