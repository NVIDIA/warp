# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Closest-pair benchmarks and a capsule-specialized Newton reference.

Run this file directly for alternating CUDA-graph timings and accuracy statistics.
ASV discovers ``ClosestPairs`` for maintained regression tracking.

The Newton reference below is adapted from ``soft_contacts_sdf.py`` and
``kernels.py`` at 8a6560683ca99d17a5b5369a27f6fb886e2dcd23 (Apache-2.0).
Only the capsule branch is retained, as in Newton's geometry-specialized
compacted kernels. Loop counts, initialization, line search, and gap exit are
unchanged. ``--newton-source`` compares against those original functions, including
timings, and reports differences caused by specialization rather than assuming parity.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import warp as wp
import warp.geometry

# ruff: noqa: PLC0415

NEWTON_COMMIT = "8a6560683ca99d17a5b5369a27f6fb886e2dcd23"


@wp.func
def capsule_sdf(x: wp.vec3, radius: float, half_height: float):
    if x[2] > half_height:
        return wp.length(wp.vec3(x[0], x[1], x[2] - half_height)) - radius
    if x[2] < -half_height:
        return wp.length(wp.vec3(x[0], x[1], x[2] + half_height)) - radius
    return wp.length(wp.vec3(x[0], x[1], 0.0)) - radius


@wp.func
def capsule_grad(x: wp.vec3, half_height: float):
    v = wp.vec3(x[0], x[1], 0.0)
    grad = wp.vec3(0.0, 0.0, 1.0)
    if x[2] > half_height:
        v = wp.vec3(x[0], x[1], x[2] - half_height)
    elif x[2] < -half_height:
        v = wp.vec3(x[0], x[1], x[2] + half_height)
        grad = wp.vec3(0.0, 0.0, -1.0)
    length = wp.length(v)
    if length > 1.0e-8:
        grad = v / length
    return grad


@wp.func
def newton_edge(p: wp.vec3, q: wp.vec3, radius: float, half_height: float):
    inv_phi = float(0.6180339887498949)
    lo = float(0.0)
    hi = float(1.0)
    c = hi - (hi - lo) * inv_phi
    d = lo + (hi - lo) * inv_phi
    fc = capsule_sdf((1.0 - c) * p + c * q, radius, half_height)
    fd = capsule_sdf((1.0 - d) * p + d * q, radius, half_height)
    for _i in range(12):
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - (hi - lo) * inv_phi
            fc = capsule_sdf((1.0 - c) * p + c * q, radius, half_height)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + (hi - lo) * inv_phi
            fd = capsule_sdf((1.0 - d) * p + d * q, radius, half_height)
    u = 0.5 * (lo + hi)
    x = (1.0 - u) * p + u * q
    return u, x, capsule_sdf(x, radius, half_height), capsule_grad(x, half_height)


@wp.func
def newton_face(a: wp.vec3, b: wp.vec3, c: wp.vec3, radius: float, half_height: float):
    bary = wp.vec3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    x = bary[0] * a + bary[1] * b + bary[2] * c
    phi = capsule_sdf(x, radius, half_height)
    grad = capsule_grad(x, half_height)
    for _i in range(12):
        da = wp.dot(grad, a)
        db = wp.dot(grad, b)
        dc = wp.dot(grad, c)
        s = wp.vec3(1.0, 0.0, 0.0)
        if db <= da and db <= dc:
            s = wp.vec3(0.0, 1.0, 0.0)
        elif dc <= da and dc <= db:
            s = wp.vec3(0.0, 0.0, 1.0)
        target = s[0] * a + s[1] * b + s[2] * c
        if wp.dot(grad, x - target) <= 1.0e-6:
            break
        gamma, line_x, line_phi, line_grad = newton_edge(x, target, radius, half_height)
        bary = (1.0 - gamma) * bary + gamma * s
        x = line_x
        phi = line_phi
        grad = line_grad
    return bary, x, phi, grad


def make_kernel(method, triangle, contact=False, precision="float32", capsule=True):
    vector = wp.vec3 if precision == "float32" else wp.vec3d
    scalar = wp.float32 if precision == "float32" else wp.float64
    result_type = wp.types.vector(8, scalar)

    @wp.kernel(enable_backward=False)
    def kernel(points: wp.array2d[vector], out: wp.array[result_type]):
        i = wp.tid()
        a = points[i, 0]
        b = points[i, 1]
        c = points[i, 2]
        p = vector(scalar(0.0), scalar(0.0), scalar(-0.5))
        q = vector(scalar(0.0), scalar(0.0), scalar(0.5))
        if wp.static(not capsule):
            p = points[i, 3]
            q = points[i, 4]
        if wp.static(contact):
            center = (a + b) * 0.5
            reach = wp.length(a - b) * 0.5
            if wp.static(triangle):
                center = (a + b + c) / 3.0
                reach = wp.max(wp.length(a - center), wp.max(wp.length(b - center), wp.length(c - center)))
            if capsule_sdf(center, 0.5, 0.5) > 0.05 + reach:
                rejected = result_type(scalar(0.0))
                rejected[7] = scalar(1.0)
                out[i] = rejected
                return

        coords = vector(scalar(0.0))
        t = scalar(0.0)
        d2 = scalar(0.0)
        x = a
        normal = vector(scalar(0.0))
        phi = scalar(0.0)
        if wp.static(method == "newton"):
            if wp.static(triangle):
                coords, x, phi, normal = newton_face(a, b, c, 0.5, 0.5)
            else:
                t, x, phi, normal = newton_edge(a, b, 0.5, 0.5)
                coords = wp.vec3(1.0 - t, t, 0.0)
        else:
            if wp.static(triangle):
                coords, t, d2 = wp.geometry.closest_points_tri_seg(a, b, c, p, q)
                x = coords[0] * a + coords[1] * b + coords[2] * c
            else:
                s = scalar(0.0)
                if wp.static(method == "native"):
                    result = wp.closest_point_edge_edge(a, b, p, q, 0.0)
                    s = result[0]
                    t = result[1]
                    d2 = result[2] * result[2]
                else:
                    s, t, d2 = wp.geometry.closest_points_seg_seg(a, b, p, q)
                coords = vector(scalar(1.0) - s, s, scalar(0.0))
                x = a + s * (b - a)
            phi = wp.sqrt(d2) - scalar(0.5)
            delta = x - (p + t * (q - p))
            normal = wp.normalize(delta)
            if wp.static(capsule and precision == "float32"):
                # Match Newton's downstream SDF evaluation and normal fallback.
                # The only change is how the minimizing point is found.
                phi = capsule_sdf(x, 0.5, 0.5)
                normal = capsule_grad(x, 0.5)
        out[i] = result_type(coords[0], coords[1], coords[2], phi, normal[0], normal[1], normal[2], scalar(0.0))

    return kernel


def make_original_kernel(triangle, contact=False):
    # Optional direct source validation; not an ASV dependency.
    from newton._src.geometry.sdf_texture import TextureSDFData
    from newton._src.geometry.soft_contacts_sdf import optimize_edge_sdf, optimize_face_sdf
    from newton._src.geometry.types import GeoType

    capsule_type = int(GeoType.CAPSULE)
    result_type = wp.types.vector(8, wp.float32)

    @wp.kernel(enable_backward=False)
    def kernel(points: wp.array2d[wp.vec3], out: wp.array[result_type], table: wp.array[TextureSDFData]):
        i = wp.tid()
        a = points[i, 0]
        b = points[i, 1]
        c = points[i, 2]
        if wp.static(contact):
            center = (a + b) * 0.5
            reach = wp.length(a - b) * 0.5
            if wp.static(triangle):
                center = (a + b + c) / 3.0
                reach = wp.max(wp.length(a - center), wp.max(wp.length(b - center), wp.length(c - center)))
            if capsule_sdf(center, 0.5, 0.5) > 0.05 + reach:
                rejected = result_type(0.0)
                rejected[7] = 1.0
                out[i] = rejected
                return
        bary = wp.vec3()
        _x = wp.vec3()
        phi = float(0.0)
        normal = wp.vec3()
        if wp.static(triangle):
            bary, _x, phi, normal = optimize_face_sdf(capsule_type, wp.vec3(0.5, 0.5, 0.0), a, b, c, -1, table, 12, 12)
        else:
            t, _x, phi, normal = optimize_edge_sdf(capsule_type, wp.vec3(0.5, 0.5, 0.0), a, b, -1, table, 12)
            bary = wp.vec3(1.0 - t, t, 0.0)
        out[i] = result_type(bary[0], bary[1], bary[2], phi, normal[0], normal[1], normal[2], 0.0)

    return kernel, wp.empty(0, dtype=TextureSDFData, device="cuda:0")


def make_segment_kernel(native=False, precision="float32"):
    vector = wp.vec3 if precision == "float32" else wp.vec3d

    @wp.kernel(enable_backward=False)
    def kernel(points: wp.array2d[vector], out: wp.array[vector]):
        i = wp.tid()
        if wp.static(native):
            out[i] = wp.closest_point_edge_edge(points[i, 0], points[i, 1], points[i, 2], points[i, 3], 0.0)
        else:
            s, t, d2 = wp.geometry.closest_points_seg_seg(points[i, 0], points[i, 1], points[i, 2], points[i, 3])
            out[i] = vector(s, t, wp.sqrt(d2))

    return kernel


def segment_workload(count, name, precision="float32"):
    rng = np.random.default_rng(25)
    points = rng.uniform(-1, 1, (count, 4, 3))
    if name in ("parallel", "near_parallel"):
        u = rng.normal(size=(count, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        v = rng.normal(size=(count, 3))
        v -= np.sum(u * v, axis=1, keepdims=True) * u
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        angle = 0.0 if name == "parallel" else 1e-4
        offset = np.cross(u, v) * 1e-3
        points[:, 0], points[:, 1] = -u, u
        points[:, 2], points[:, 3] = -u - angle * v + offset, u + angle * v + offset
    elif name == "degenerate":
        points[:, 1] = points[:, 0]
        points[::2, 3] = points[::2, 2]
    return points.astype(precision)


class SegmentPairs:
    params = [["random", "parallel", "near_parallel", "degenerate"], ["warp", "native"], ["float32", "float64"]]
    param_names = ["workload", "implementation", "precision"]
    number = 1
    repeat = 5

    def setup(self, name, implementation, precision):
        if implementation == "native" and precision == "float64":
            raise NotImplementedError("The native edge-edge builtin only supports float32.")
        vector = wp.vec3 if precision == "float32" else wp.vec3d
        self.points = wp.array(segment_workload(262144, name, precision), dtype=vector, device="cuda:0")
        self.out = wp.empty(262144, dtype=vector, device="cuda:0")
        self.graph = capture(
            make_segment_kernel(implementation == "native", precision), [self.points, self.out], 262144
        )

    def time_queries(self, name, implementation, precision):
        wp.capture_launch(self.graph)
        wp.synchronize_device("cuda:0")


def workload(count, name, precision="float32", capsule=True):
    rng = np.random.default_rng(25)
    points = rng.uniform(-1.5, 1.5, (count, 5, 3))
    if name == "near":
        centers = rng.normal(size=(count, 1, 3))
        centers[:, :, 2] = 0.0
        centers /= np.linalg.norm(centers, axis=2, keepdims=True)
        points[:, :3] = centers * rng.uniform(0.45, 0.60, (count, 1, 1)) + rng.normal(0, 0.04, (count, 3, 3))
    elif name == "separated":
        points[:, :3] = points[:, :3] * 0.1 + np.array([3, 0, 0])
    elif name == "parallel":
        points[:, 1] = points[:, 0] + np.array([0, 0, 1])
        points[:, 2] = points[:, 0] + np.array([0.1, 0, 0.5])
    elif name == "degenerate":
        points[:, 1] = points[:, 0]
        points[::2, 2] = points[::2, 0]
    if capsule:
        points[:, 3] = [0, 0, -0.5]
        points[:, 4] = [0, 0, 0.5]
    return points.astype(precision)


def capture(kernel, inputs, count, launches=30):
    wp.launch(kernel, count, inputs=inputs, device="cuda:0", block_dim=128)
    wp.synchronize_device("cuda:0")
    with wp.ScopedCapture(device="cuda:0") as cap:
        for _ in range(launches):
            wp.launch(kernel, count, inputs=inputs, device="cuda:0", block_dim=128)
    return cap.graph


def graph_times(graphs, launches=30, samples=21):
    times = {key: [] for key in graphs}
    start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    for graph in graphs.values():
        wp.capture_launch(graph)
    for iteration in range(samples):
        for key in list(graphs)[:: 1 if iteration % 2 == 0 else -1]:
            wp.record_event(start)
            wp.capture_launch(graphs[key])
            wp.record_event(end)
            wp.synchronize_event(end)
            times[key].append(wp.get_event_elapsed_time(start, end) * 1000.0 / launches)
    return {key: float(np.median(values)) for key, values in times.items()}


class ClosestPairs:
    params = [["seg", "tri"], ["random", "near", "parallel", "separated"], ["warp", "newton"], [False, True]]
    param_names = ["primitive", "workload", "implementation", "initial_cull"]
    number = 1
    repeat = 5

    def setup(self, primitive, name, implementation, initial_cull):
        self.points = wp.array(workload(262144, name), dtype=wp.vec3, device="cuda:0")
        self.out = wp.empty(262144, dtype=wp.types.vector(8, wp.float32), device="cuda:0")
        kernel = make_kernel(implementation, primitive == "tri", initial_cull)
        self.graph = capture(kernel, [self.points, self.out], 262144)

    def time_queries(self, primitive, name, implementation, initial_cull):
        wp.capture_launch(self.graph)
        wp.synchronize_device("cuda:0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=262144)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--launches", type=int, default=30)
    parser.add_argument("--newton-source", type=Path)
    parser.add_argument("--initial-cull", action="store_true")
    parser.add_argument(
        "--segments", action="store_true", help="Compare general segment pairs with the native builtin."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("closest-benchmark.json"),
        help="JSON results file; defaults to the current directory rather than a shared temporary directory",
    )
    args = parser.parse_args()
    if min(args.count, args.samples, args.launches) <= 0:
        parser.error("Count, samples, and launches must be positive.")
    newton_commit = None
    if args.newton_source:
        newton_commit = subprocess.check_output(
            ["git", "-C", str(args.newton_source), "rev-parse", "HEAD"], text=True
        ).strip()
        if newton_commit != NEWTON_COMMIT:
            parser.error(f"The reference requires Newton {NEWTON_COMMIT}, got {newton_commit}.")
        sys.path.insert(0, str(args.newton_source.resolve()))
    wp.init()
    metadata = {
        "device": wp.get_device("cuda:0").name,
        "warp_version": wp.__version__,
        "newton_commit": newton_commit,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[2] / "warp/_src/geometry/closest.py").read_bytes()
        ).hexdigest(),
        "samples": args.samples,
        "launches": args.launches,
        "block_dim": 128,
        "seed": 25,
    }
    results = []
    if args.segments:
        for precision in ("float32", "float64"):
            for name in ("random", "parallel", "near_parallel", "degenerate"):
                raw = segment_workload(args.count, name, precision)
                vector = wp.vec3 if precision == "float32" else wp.vec3d
                data = wp.array(raw, dtype=vector, device="cuda:0")
                graphs, outputs = {}, {}
                for method in ["warp", "native"] if precision == "float32" else ["warp"]:
                    out = wp.empty(args.count, dtype=vector, device="cuda:0")
                    outputs[method] = out
                    graphs[method] = capture(
                        make_segment_kernel(method == "native", precision), [data, out], args.count, args.launches
                    )
                timing = graph_times(graphs, args.launches, args.samples)
                from warp.tests.geometry.test_closest import hull_distance_squared

                subset = np.linspace(0, args.count - 1, min(256, args.count), dtype=int)
                reference = np.array(
                    [np.sqrt(hull_distance_squared(raw[i, :2], raw[i, 2:], exact=True)) for i in subset]
                )
                errors = {
                    key: float(np.max(np.abs(value.numpy()[subset, 2] - reference))) for key, value in outputs.items()
                }
                row = {
                    "workload": name,
                    "precision": precision,
                    "count": args.count,
                    "us": timing,
                    "max_abs_distance_error_subset": errors,
                }
                results.append(row)
                print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps({**metadata, "results": results}, indent=2) + "\n")
        return
    for triangle in (False, True):
        for name in ("random", "near", "parallel", "separated", "degenerate"):
            raw = workload(args.count, name)
            data = wp.array(raw, dtype=wp.vec3, device="cuda:0")
            outputs, graphs = {}, {}
            methods = ["warp", "newton"] + ([] if triangle else ["native"])
            for method in methods:
                out = wp.empty(args.count, dtype=wp.types.vector(8, wp.float32), device="cuda:0")
                outputs[method] = out
                graphs[method] = capture(
                    make_kernel(method, triangle, args.initial_cull), [data, out], args.count, args.launches
                )
            if args.newton_source:
                kernel, table = make_original_kernel(triangle, args.initial_cull)
                out = wp.empty(args.count, dtype=wp.types.vector(8, wp.float32), device="cuda:0")
                outputs["original"] = out
                graphs["original"] = capture(kernel, [data, out, table], args.count, args.launches)
                # Keep the original as the authoritative baseline. Specializing
                # dispatch can change rounding at line-search ties and hence the
                # subsequent iterates; report the difference rather than hide it.
            timing = graph_times(graphs, args.launches, args.samples)
            values = {key: value.numpy() for key, value in outputs.items()}
            # Independent float64 active-set QP oracle from the unit tests, on a
            # reproducible subset. Evaluate quantized float32 inputs as float64.
            from warp.tests.geometry.test_closest import hull_distance_squared

            subset = np.linspace(0, args.count - 1, min(256, args.count), dtype=int)
            reference = np.array(
                [
                    np.sqrt(
                        hull_distance_squared(raw[i, : 3 if triangle else 2].astype(float), raw[i, 3:].astype(float))
                    )
                    - 0.5
                    for i in subset
                ]
            )
            active = values["warp"][subset, 7] == 0
            errors = {
                key: float(np.max(np.abs(value[subset, 3][active] - reference[active]), initial=0.0))
                for key, value in values.items()
            }
            row = {
                "primitive": "tri" if triangle else "seg",
                "workload": name,
                "count": args.count,
                "initial_cull": args.initial_cull,
                "active_queries": int(np.sum(values["warp"][:, 7] == 0)),
                "us": timing,
                "max_abs_sdf_error_subset": errors,
                "newton_misses_vs_warp_at_margin_0.05": int(
                    np.sum(
                        (values["newton"][:, 3] >= 0.05) & (values["warp"][:, 3] < 0.05) & (values["warp"][:, 7] == 0)
                    )
                ),
            }
            if "original" in values:
                row["specialization_max_sdf_delta"] = float(
                    np.max(np.abs(values["original"][:, 3] - values["newton"][:, 3]))
                )
                disagreements = np.flatnonzero(
                    (values["original"][:, 3] >= 0.05) & (values["warp"][:, 3] < 0.05) & (values["warp"][:, 7] == 0)
                )
                confirmed = 0
                ambiguous = 0
                for i in disagreements:
                    truth = (
                        np.sqrt(
                            hull_distance_squared(
                                raw[i, : 3 if triangle else 2].astype(float), raw[i, 3:].astype(float)
                            )
                        )
                        - 0.5
                    )
                    confirmed += int(truth < 0.05 - 3e-6)
                    ambiguous += int(abs(truth - 0.05) <= 3e-6)
                row["original_misses_confirmed_by_oracle"] = confirmed
                row["threshold_disagreements_within_3e-6"] = ambiguous
                row["original_misses_vs_warp_at_margin_0.05"] = len(disagreements)
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            {**metadata, "results": results},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
