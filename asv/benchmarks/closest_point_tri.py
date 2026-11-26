# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Point-triangle benchmarks against the native mesh helper and Newton.

Newton's function is copied verbatim from geometry/kernels.py at
f39d8199f18d5117f9dd8073c1eb71c9a196f020 (Apache-2.0).
Native comparisons call the actual compiled header, without BVH traversal.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import warp as wp
import warp.geometry

TRI_CONTACT_FEATURE_VERTEX_A = wp.constant(0)
TRI_CONTACT_FEATURE_VERTEX_B = wp.constant(1)
TRI_CONTACT_FEATURE_VERTEX_C = wp.constant(2)
TRI_CONTACT_FEATURE_EDGE_AB = wp.constant(3)
TRI_CONTACT_FEATURE_EDGE_AC = wp.constant(4)
TRI_CONTACT_FEATURE_EDGE_BC = wp.constant(5)
TRI_CONTACT_FEATURE_FACE_INTERIOR = wp.constant(6)


@wp.func
def triangle_closest_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    """
    feature_type type:
        TRI_CONTACT_FEATURE_VERTEX_A
        TRI_CONTACT_FEATURE_VERTEX_B
        TRI_CONTACT_FEATURE_VERTEX_C
        TRI_CONTACT_FEATURE_EDGE_AB      : at edge A-B
        TRI_CONTACT_FEATURE_EDGE_AC      : at edge A-C
        TRI_CONTACT_FEATURE_EDGE_BC      : at edge B-C
        TRI_CONTACT_FEATURE_FACE_INTERIOR
    """
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_A
        bary = wp.vec3(1.0, 0.0, 0.0)
        return a, bary, feature_type

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_B
        bary = wp.vec3(0.0, 1.0, 0.0)
        return b, bary, feature_type

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_C
        bary = wp.vec3(0.0, 0.0, 1.0)
        return c, bary, feature_type

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AB
        bary = wp.vec3(1.0 - v, v, 0.0)
        return a + v * ab, bary, feature_type

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        v = d2 / (d2 - d6)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AC
        bary = wp.vec3(1.0 - v, 0.0, v)
        return a + v * ac, bary, feature_type

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        feature_type = TRI_CONTACT_FEATURE_EDGE_BC
        bary = wp.vec3(0.0, 1.0 - v, v)
        return b + v * (c - b), bary, feature_type

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    feature_type = TRI_CONTACT_FEATURE_FACE_INTERIOR
    bary = wp.vec3(1.0 - v - w, v, w)
    return a + v * ab + w * ac, bary, feature_type


@wp.func_native("return wp::closest_point_to_triangle(a, b, c, p);")
def native_coordinates(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec2: ...


@wp.func(inline=True)
def native_feature(bary: wp.vec3):
    if bary[2] == 0.0:
        if bary[1] == 0.0:
            return 0
        if bary[0] == 0.0:
            return 1
        return 3
    if bary[1] == 0.0:
        if bary[0] == 0.0:
            return 2
        return 4
    if bary[0] == 0.0:
        return 5
    return 6


def make_kernel(method, features=True, newton_function=triangle_closest_point):
    @wp.kernel(enable_backward=False)
    def query(
        points: wp.array2d[wp.vec3], barycentrics: wp.array[wp.vec3], distances: wp.array[float], ids: wp.array[int]
    ):
        i = wp.tid()
        p, a, b, c = points[0, i], points[1, i], points[2, i], points[3, i]
        if wp.static(method == "warp"):
            bary, distance, feature = wp.geometry.closest_point_point_tri(p, a, b, c)
        elif wp.static(method == "newton"):
            closest, bary, feature = newton_function(a, b, c, p)
            distance = wp.length_sq(closest - p)
        else:
            uv = native_coordinates(p, a, b, c)
            bary = wp.vec3(uv[0], uv[1], 1.0 - uv[0] - uv[1])
            closest = bary[0] * a + bary[1] * b + bary[2] * c
            distance = wp.length_sq(closest - p)
            feature = native_feature(bary)
        barycentrics[i] = bary
        distances[i] = distance
        if wp.static(features):
            ids[i] = feature

    return query


def workload(count, name, seed=418):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(4, count, 3))
    if name == "near":
        bary = rng.dirichlet([1, 1, 1], count)
        x[0] = np.einsum("ij,jik->ik", bary, x[1:]) + rng.normal(0, 1e-4, (count, 3))
    elif name in ("face", "edge", "vertex", "boundary", "thin", "degenerate"):
        # Mix rotations and translations; use a right triangle with known regions.
        q, _ = np.linalg.qr(rng.normal(size=(count, 3, 3)))
        a = x[1].copy()
        e1, e2, normal = q[:, :, 0], q[:, :, 1], q[:, :, 2]
        x[2], x[3] = a + e1, a + e2
        if name == "face":
            x[0] = a + 0.25 * e1 + 0.25 * e2 + rng.normal(size=(count, 1)) * normal
        elif name == "edge":
            x[0] = a + 0.5 * e1 - 0.25 * e2 + rng.normal(size=(count, 1)) * normal
        elif name == "vertex":
            x[0] = a - 0.25 * e1 - 0.25 * e2 + rng.normal(size=(count, 1)) * normal
        elif name == "boundary":
            x[0] = a + 0.5 * e1
        elif name == "thin":
            x[3] = a + 0.5 * e1 + 1e-5 * e2
            x[0] = a + 0.5 * e1 + 3e-6 * e2 + 1e-4 * normal
        else:
            x[3] = x[1]
    elif name == "small":
        x *= 1e-12
    elif name == "large":
        x *= 1e12
    return x.astype(np.float32)


def capture(kernel, inputs, count, launches, timed=False):
    wp.launch(kernel, count, inputs=inputs, device="cuda:0", block_dim=128)
    wp.synchronize_device("cuda:0")
    start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    with wp.ScopedCapture(device="cuda:0") as cap:
        if timed:
            wp.record_event(start, external=True)
        for _ in range(launches):
            wp.launch(kernel, count, inputs=inputs, device="cuda:0", block_dim=128)
        if timed:
            wp.record_event(end, external=True)
    return (cap.graph, start, end) if timed else cap.graph


def time_graphs(graphs, launches, samples):
    # Both timestamps are in the graph so host throttling cannot insert an
    # idle gap between the kernels and the end event.
    times = {key: [] for key in graphs}
    for graph, _, _ in graphs.values():
        wp.capture_launch(graph)
    for iteration in range(samples):
        for key in list(graphs)[:: 1 if iteration % 2 == 0 else -1]:
            graph, start, end = graphs[key]
            wp.capture_launch(graph)
            wp.synchronize_event(end)
            times[key].append(wp.get_event_elapsed_time(start, end) * 1000.0 / launches)
    return {key: float(np.median(value)) for key, value in times.items()}


class PointTriangle:
    params = [
        ["random", "near", "face", "edge", "vertex", "thin", "degenerate", "boundary"],
        ["native", "newton", "warp"],
    ]
    param_names = ["workload", "implementation"]
    number = 1
    repeat = 5

    def setup(self, name, method):
        n = 262144
        self.inputs = [
            wp.array(workload(n, name), dtype=wp.vec3, device="cuda:0"),
            wp.empty(n, dtype=wp.vec3, device="cuda:0"),
            wp.empty(n, dtype=float, device="cuda:0"),
            wp.empty(n, dtype=int, device="cuda:0"),
        ]
        self.graph = capture(make_kernel(method), self.inputs, n, 30)

    def time_query(self, name, method):
        wp.capture_launch(self.graph)
        wp.synchronize_device("cuda:0")


def accuracy(raw, kernels, sample_count, device="cuda:0"):
    # The oracle enumerates exact rational active sets, independently of the
    # region-test implementations being measured. Quantize before computing it.
    from warp.tests.geometry.test_closest_point_tri import exact_reference  # noqa: PLC0415

    count = min(sample_count, raw.shape[1])
    indices = np.linspace(0, raw.shape[1] - 1, count, dtype=int)
    cases = raw[:, indices].transpose(1, 0, 2)
    reference = [exact_reference(case) for case in cases]
    expected_bary = np.array([row[0] for row in reference])
    expected_d2 = np.array([row[1] for row in reference])
    expected_ids = np.array([row[2] for row in reference])
    full_rank = np.array([row[3] for row in reference])
    xyz = cases.astype(np.float64)
    scale2 = np.max(np.sum((xyz[:, 1:] - xyz[:, :1]) ** 2, axis=2), axis=1)
    scale2 = np.maximum(scale2, np.finfo(np.float64).tiny)
    expected_point = np.einsum("ij,ijk->ik", expected_bary, xyz[:, 1:])
    data = wp.array(raw[:, indices].copy(), dtype=wp.vec3, device=device)
    bary = wp.empty(count, dtype=wp.vec3, device=device)
    distances = wp.empty(count, dtype=float, device=device)
    ids = wp.empty(count, dtype=int, device=device)
    result = {}
    for key, kernel in kernels.items():
        wp.launch(kernel, count, inputs=[data, bary, distances, ids], device=device)
        weights, d2, features = bary.numpy(), distances.numpy(), ids.numpy()
        finite = np.isfinite(weights).all(axis=1) & np.isfinite(d2)
        point = np.einsum("ij,ijk->ik", weights.astype(np.float64), xyz[:, 1:])
        result[key] = {
            "samples": count,
            "nonfinite": int(np.count_nonzero(~finite)),
            "invalid_barycentrics": int(np.count_nonzero(~(weights >= 0).all(axis=1))),
            "max_normalized_distance_squared_error": float(
                np.max(np.abs(d2[finite] - expected_d2[finite]) / scale2[finite])
            )
            if np.any(finite)
            else None,
            "max_normalized_witness_error": float(
                np.max(np.linalg.norm(point[finite] - expected_point[finite], axis=1) / np.sqrt(scale2[finite]))
            )
            if np.any(finite)
            else None,
            "unique_feature_samples": int(np.count_nonzero(full_rank)),
            "feature_mismatches": int(np.count_nonzero(full_rank & ((features != expected_ids) | ~finite))),
        }
    return result


def cpu_resources():
    """Record cgroup quota/throttling; GPU exclusivity does not reserve CPUs."""
    result = {}
    for name in ("cpu.max", "cpu.stat"):
        path = Path("/sys/fs/cgroup") / name
        result[name] = path.read_text().strip() if path.exists() else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--newton-source", type=Path)
    parser.add_argument("--accuracy-samples", type=int, default=256)
    parser.add_argument("--count", type=int, default=262144)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--launches", type=int, default=30)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["random", "near", "face", "edge", "vertex", "thin", "degenerate", "small", "large", "boundary"],
    )
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("point-triangle-benchmark.json"))
    args = parser.parse_args()
    cpu_before = cpu_resources()
    wp.init()
    newton_function = triangle_closest_point
    reference_commit = "f39d8199f18d5117f9dd8073c1eb71c9a196f020"
    if args.newton_source:
        source_file = args.newton_source / "newton/_src/geometry/kernels.py"
        source = ast.parse(source_file.read_text())
        copied = ast.parse(Path(__file__).read_text())

        def function(tree):
            return ast.dump(
                next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "triangle_closest_point"
                )
            )

        if function(source) != function(copied):
            raise ValueError("The Newton reference changed; review the benchmark copy before comparing.")
        sys.path.insert(0, str(args.newton_source.resolve()))
        from newton._src.geometry.kernels import triangle_closest_point as original  # noqa: PLC0415

        newton_function = original
        reference_commit = subprocess.check_output(
            ["git", "-C", str(args.newton_source), "rev-parse", "HEAD"], text=True
        ).strip()
    kernels = {key: make_kernel(key, not args.no_features, newton_function) for key in ("native", "newton", "warp")}
    accuracy_kernels = {key: make_kernel(key, True, newton_function) for key in kernels}
    rows = []
    for name in args.workloads:
        raw = workload(args.count, name)
        data = wp.array(raw, dtype=wp.vec3, device="cuda:0")
        bary = wp.empty(args.count, dtype=wp.vec3, device="cuda:0")
        distances = wp.empty(args.count, dtype=float, device="cuda:0")
        ids = wp.empty(args.count, dtype=int, device="cuda:0")
        graphs = {
            key: capture(kernel, [data, bary, distances, ids], args.count, args.launches, timed=True)
            for key, kernel in kernels.items()
        }
        times = time_graphs(graphs, args.launches, args.samples)
        print(name, times, flush=True)
        errors = accuracy(raw, accuracy_kernels, args.accuracy_samples) if args.accuracy_samples else {}
        print("accuracy", errors, flush=True)
        rows.append({"workload": name, "microseconds": times, "accuracy": errors})
    args.output.write_text(
        json.dumps(
            {
                "device": wp.get_device("cuda:0").name,
                "kernel_properties": {
                    key: wp.get_cuda_kernel_properties(kernel, device="cuda:0", block_dim=128)
                    for key, kernel in kernels.items()
                },
                "cpu_before": cpu_before,
                "cpu_after": cpu_resources(),
                "timing": "median CUDA graph event intervals; events captured inside graph",
                "newton_commit": reference_commit,
                "newton_original_imported": bool(args.newton_source),
                "warp_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "implementation_sha256": hashlib.sha256(
                    (Path(__file__).resolve().parents[2] / "warp/_src/geometry/closest.py").read_bytes()
                ).hexdigest(),
                "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "count": args.count,
                "samples": args.samples,
                "launches": args.launches,
                "features": not args.no_features,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
