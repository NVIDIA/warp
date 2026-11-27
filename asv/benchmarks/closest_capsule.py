# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare closest pairs and capsule contacts with Newton's analytic implementation.

The reference formulas are adapted from Newton's ``collision_primitive.py`` at
03b36585233ef0ae5dcdb7ebdd9a40a5e127c330 (Apache-2.0). Use ``--newton-source``
to validate and time the unmodified source as the authoritative baseline.
The hybrid replaces only the nonparallel closest-pair calculation; it retains
Newton's parallel threshold, two candidate contacts, and shared normal policy.
It is a benchmark adapter, not a new public capsule collision API.
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

if __package__:
    from .closest_points import capture, graph_times
else:
    from closest_points import capture, graph_times

NEWTON_COMMIT = "03b36585233ef0ae5dcdb7ebdd9a40a5e127c330"
MAXVAL = wp.constant(1e10)
MINVAL = wp.constant(1e-15)
mat23 = wp.types.matrix((2, 3), wp.float32)


@wp.func
def sphere_contact(a: wp.vec3, ra: float, b: wp.vec3, rb: float):
    direction = b - a
    distance = wp.length(direction)
    normal = wp.vec3(1.0, 0.0, 0.0)
    if distance != 0.0:
        normal = direction / distance
    distance = distance - (ra + rb)
    position = a + normal * (ra + 0.5 * distance)
    return distance, position, normal


@wp.func
def axis_determinant(a: wp.vec3, b: wp.vec3):
    ma = wp.dot(a, a)
    mb = -wp.dot(a, b)
    mc = wp.dot(b, b)
    return ma * mc - mb * mb


@wp.func
def newton_axis_parameters(a: wp.vec3, axis_a: wp.vec3, b: wp.vec3, axis_b: wp.vec3):
    """Newton's nonparallel branch, with centered parameters in [-1, 1]."""
    dif = a - b
    ma = wp.dot(axis_a, axis_a)
    mb = -wp.dot(axis_a, axis_b)
    mc = wp.dot(axis_b, axis_b)
    u = -wp.dot(axis_a, dif)
    v = wp.dot(axis_b, dif)
    det = ma * mc - mb * mb
    inv_det = 1.0 / det
    x1 = (mc * u - mb * v) * inv_det
    x2 = (ma * v - mb * u) * inv_det
    if x1 > 1.0:
        x1 = 1.0
        x2 = (v - mb) / mc
    elif x1 < -1.0:
        x1 = -1.0
        x2 = (v + mb) / mc
    if x2 > 1.0:
        x2 = 1.0
        x1 = wp.clamp((u - mb) / ma, -1.0, 1.0)
    elif x2 < -1.0:
        x2 = -1.0
        x1 = wp.clamp((u + mb) / ma, -1.0, 1.0)
    return x1, x2


def make_contact_function(hybrid):
    @wp.func
    def contact(a: wp.vec3, da: wp.vec3, ra: float, ha: float, b: wp.vec3, db: wp.vec3, rb: float, hb: float):
        distances = wp.vec2(MAXVAL, MAXVAL)
        positions = mat23()
        normal = wp.vec3()
        axis_a = da * ha
        axis_b = db * hb
        dif = a - b
        ma = wp.dot(axis_a, axis_a)
        mb = -wp.dot(axis_a, axis_b)
        mc = wp.dot(axis_b, axis_b)
        v = wp.dot(axis_b, dif)
        det = ma * mc - mb * mb
        if wp.abs(det) >= MINVAL:
            p = wp.vec3()
            q = wp.vec3()
            if wp.static(hybrid):
                s, t, _d2 = wp.geometry.closest_point_seg_seg(a - axis_a, a + axis_a, b - axis_b, b + axis_b)
                p = a + axis_a * (2.0 * s - 1.0)
                q = b + axis_b * (2.0 * t - 1.0)
            else:
                x1, x2 = newton_axis_parameters(a, axis_a, b, axis_b)
                p = a + axis_a * x1
                q = b + axis_b * x2
            distance, position, n = sphere_contact(p, ra, q, rb)
            distances[0] = distance
            positions[0] = position
            normal = n
        else:
            # Preserve Newton's parallel contact candidates, including its
            # behavior for unequal lengths and zero half-height. Do not silently
            # improve this policy while calling the comparison equivalent.
            p = a + axis_a
            x2 = wp.clamp((v - mb) / mc, -1.0, 1.0)
            q = b + axis_b * x2
            distance, position, n = sphere_contact(p, ra, q, rb)
            distances[0] = distance
            positions[0] = position
            normal = n
            p = a - axis_a
            x2 = wp.clamp((v + mb) / mc, -1.0, 1.0)
            q = b + axis_b * x2
            distance, position, _n = sphere_contact(p, ra, q, rb)
            distances[1] = distance
            positions[1] = position
        return distances, positions, normal

    return contact


newton_contacts = make_contact_function(False)
hybrid_contacts = make_contact_function(True)


def make_contact_kernel(function):
    @wp.kernel(enable_backward=False)
    def kernel(points: wp.array2d[wp.vec3], sizes: wp.array[wp.vec4], out: wp.array2d[float]):
        i = wp.tid()
        size = sizes[i]
        d, p, n = function(points[i, 0], points[i, 1], size[0], size[1], points[i, 2], points[i, 3], size[2], size[3])
        # Components are contiguous across queries, so output stores coalesce.
        # Every variant writes the same eleven scalar components.
        out[0, i] = d[0]
        out[1, i] = d[1]
        out[2, i] = p[0, 0]
        out[3, i] = p[0, 1]
        out[4, i] = p[0, 2]
        out[5, i] = p[1, 0]
        out[6, i] = p[1, 1]
        out[7, i] = p[1, 2]
        out[8, i] = n[0]
        out[9, i] = n[1]
        out[10, i] = n[2]

    return kernel


def make_axis_kernel(use_warp):
    @wp.kernel(enable_backward=False)
    def kernel(points: wp.array2d[wp.vec3], sizes: wp.array[wp.vec4], out: wp.array[wp.vec3]):
        i = wp.tid()
        a = points[i, 0]
        b = points[i, 2]
        axis_a = points[i, 1] * sizes[i][1]
        axis_b = points[i, 3] * sizes[i][3]
        if wp.static(use_warp):
            s, t, d2 = wp.geometry.closest_point_seg_seg(a - axis_a, a + axis_a, b - axis_b, b + axis_b)
            out[i] = wp.vec3(s, t, wp.sqrt(d2))
        else:
            x1, x2 = newton_axis_parameters(a, axis_a, b, axis_b)
            distance = wp.length((a + axis_a * x1) - (b + axis_b * x2))
            out[i] = wp.vec3(0.5 * (x1 + 1.0), 0.5 * (x2 + 1.0), distance)

    return kernel


def make_witness_kernel(use_warp, newton_function):
    @wp.kernel(enable_backward=False)
    def kernel(points: wp.array2d[wp.vec3], sizes: wp.array[wp.vec4], out: wp.array2d[float]):
        i = wp.tid()
        a = points[i, 0]
        b = points[i, 2]
        axis_a = points[i, 1] * sizes[i][1]
        axis_b = points[i, 3] * sizes[i][3]
        p0 = a - axis_a
        p1 = a + axis_a
        q0 = b - axis_b
        q1 = b + axis_b
        p = wp.vec3()
        q = wp.vec3()
        if wp.static(use_warp):
            s, t, _d2 = wp.geometry.closest_point_seg_seg(p0, p1, q0, q1)
            p = p0 + s * (p1 - p0)
            q = q0 + t * (q1 - q0)
        else:
            p, q = newton_function(p0, p1, q0, q1)
        out[0, i] = p[0]
        out[1, i] = p[1]
        out[2, i] = p[2]
        out[3, i] = q[0]
        out[4, i] = q[1]
        out[5, i] = q[2]
        out[6, i] = wp.length(p - q)

    return kernel


@wp.kernel(enable_backward=False)
def classify_axes(points: wp.array2d[wp.vec3], sizes: wp.array[wp.vec4], parallel: wp.array[bool]):
    i = wp.tid()
    parallel[i] = wp.abs(axis_determinant(points[i, 1] * sizes[i][1], points[i, 3] * sizes[i][3])) < MINVAL


def workload(count, name):
    rng = np.random.default_rng(2360)
    points = rng.uniform(-1, 1, (count, 4, 3))
    directions = rng.normal(size=(count, 2, 3))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    points[:, 1] = directions[:, 0]
    points[:, 3] = directions[:, 1]
    sizes = rng.uniform(0.1, 0.7, (count, 4))
    sizes[:, 0::2] = rng.uniform(0.02, 0.15, (count, 2))
    if name == "near":
        # Small relative displacement, independent axes: mixed side/cap contact.
        points[:, 2] = points[:, 0] + rng.normal(0, 0.15, (count, 3))
    elif name == "endcap":
        points[:, 2] = points[:, 0] + points[:, 1] * (sizes[:, 1:2] + sizes[:, 3:4] + 0.2)
    elif name in ("parallel", "parallel_varied_lengths", "contained_parallel"):
        points[:, 1] = [0, 0, 1]
        points[:, 3] = [0, 0, 1]
        points[:, 2] = points[:, 0] + [0.15, 0, 0]
        sizes[:, 0::2] = 0.1
        if name == "parallel":
            # Exactly representable products keep the determinant zero even
            # under fused multiply-add. Arbitrary lengths are covered below.
            sizes[:, 1] = 0.5
            sizes[:, 3] = 0.5
        elif name == "contained_parallel":
            sizes[:, 1] = 2.0
            sizes[:, 3] = 0.5
    elif name == "near_parallel":
        u = points[:, 1]
        v = points[:, 3] - np.sum(points[:, 3] * u, axis=1, keepdims=True) * u
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        angle = 10.0 ** rng.uniform(-6, -3, (count, 1))
        direction = u + angle * v
        points[:, 3] = direction / np.linalg.norm(direction, axis=1, keepdims=True)
        points[:, 2] = points[:, 0] + np.cross(u, v) * 0.05
        sizes[:, 1] = 0.5
        sizes[:, 3] = 0.5
    elif name == "small":
        points[:, 0::2] *= 1e-4
        sizes *= 1e-4
    elif name == "collapsed":
        sizes[::2, 1] = 0.0
        sizes[1::2, 3] = 0.0
    return points.astype(np.float32), sizes.astype(np.float32)


class CapsuleContacts:
    params = [["random", "near", "endcap", "parallel", "near_parallel"], ["newton", "hybrid"]]
    param_names = ["workload", "implementation"]
    number = 1
    repeat = 5

    def setup(self, name, implementation):
        points, sizes = workload(262144, name)
        self.points = wp.array(points, dtype=wp.vec3, device="cuda:0")
        self.sizes = wp.array(sizes, dtype=wp.vec4, device="cuda:0")
        self.out = wp.empty((11, 262144), dtype=float, device="cuda:0")
        function = newton_contacts if implementation == "newton" else hybrid_contacts
        self.graph = capture(make_contact_kernel(function), [self.points, self.sizes, self.out], 262144)

    def time_contacts(self, name, implementation):
        wp.capture_launch(self.graph)
        wp.synchronize_device("cuda:0")


def reference_distances(points, sizes, indices):
    from warp.tests.geometry.test_closest import hull_distance_squared  # noqa: PLC0415

    distances = []
    for i in indices:
        p = points[i].astype(float)
        s = sizes[i].astype(float)
        a = p[1] * s[1]
        b = p[3] * s[3]
        distances.append(np.sqrt(hull_distance_squared([p[0] - a, p[0] + a], [p[2] - b, p[2] + b], exact=True)))
    return np.array(distances)


def validate(original):
    # Exact parallelism isolates manifold preservation from roundoff in the
    # determinant threshold. Test zero height explicitly instead of hiding NaNs.
    for device in ("cpu", "cuda:0"):
        for name in ("random", "parallel", "contained_parallel", "collapsed"):
            raw, size = workload(64, name)
            data = wp.array(raw, dtype=wp.vec3, device=device)
            sizes = wp.array(size, dtype=wp.vec4, device=device)
            values = {}
            for key, function in (("copy", newton_contacts), ("original", original), ("hybrid", hybrid_contacts)):
                out = wp.empty((11, 64), dtype=float, device=device)
                wp.launch(make_contact_kernel(function), 64, inputs=[data, sizes, out], device=device)
                values[key] = out.numpy().T
            np.testing.assert_allclose(values["copy"], values["original"], atol=2e-6, rtol=2e-5, equal_nan=True)
            if name != "random":
                np.testing.assert_allclose(values["hybrid"], values["original"], atol=2e-6, rtol=2e-5, equal_nan=True)
            else:
                extracted = wp.empty(64, dtype=wp.vec3, device=device)
                wp.launch(make_axis_kernel(False), 64, inputs=[data, sizes, extracted], device=device)
                np.testing.assert_allclose(
                    extracted.numpy()[:, 2] - size[:, 0] - size[:, 2],
                    values["original"][:, 0],
                    atol=2e-6,
                    rtol=2e-5,
                )
            out = wp.empty(64, dtype=wp.vec3, device=device)
            wp.launch(make_axis_kernel(True), 64, inputs=[data, sizes, out], device=device)
            value = out.numpy()
            np.testing.assert_allclose(value[:, 2], reference_distances(raw, size, range(64)), atol=2e-6, rtol=2e-5)
            if not np.isfinite(value).all():
                raise AssertionError("The closest-pair query returned a nonfinite result.")
            if not ((value[:, :2] >= 0.0) & (value[:, :2] <= 1.0)).all():
                raise AssertionError("Closest-pair parameters must lie in [0, 1].")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--newton-source", type=Path, required=True)
    parser.add_argument("--count", type=int, default=262144)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--launches", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("capsule-analytic-benchmark.json"))
    args = parser.parse_args()
    if min(args.count, args.samples, args.launches) <= 0:
        parser.error("Count, samples, and launches must be positive.")
    revision = subprocess.check_output(["git", "-C", str(args.newton_source), "rev-parse", "HEAD"], text=True).strip()
    if revision != NEWTON_COMMIT:
        parser.error(f"The reference requires Newton {NEWTON_COMMIT}, got {revision}.")
    sys.path.insert(0, str(args.newton_source.resolve()))
    from newton._src.geometry.collision_primitive import (  # noqa: PLC0415
        closest_segment_to_segment_points,
        collide_capsule_capsule,
    )

    wp.init()
    validate(collide_capsule_capsule)
    print("CPU/CUDA reference and manifold checks passed.", flush=True)
    rows = []
    for name in (
        "random",
        "near",
        "endcap",
        "parallel",
        "parallel_varied_lengths",
        "contained_parallel",
        "near_parallel",
        "small",
        "collapsed",
    ):
        raw, size = workload(args.count, name)
        data = wp.array(raw, dtype=wp.vec3, device="cuda:0")
        sizes = wp.array(size, dtype=wp.vec4, device="cuda:0")
        parallel = wp.empty(args.count, dtype=bool, device="cuda:0")
        wp.launch(classify_axes, args.count, inputs=[data, sizes, parallel], device="cuda:0")
        mask = parallel.numpy()
        graphs, outputs = {}, {}
        for key, function in (
            ("original", collide_capsule_capsule),
            ("copy", newton_contacts),
            ("hybrid", hybrid_contacts),
        ):
            out = wp.empty((11, args.count), dtype=float, device="cuda:0")
            outputs[key] = out
            graphs[key] = capture(make_contact_kernel(function), [data, sizes, out], args.count, args.launches)
        timing = graph_times(graphs, args.launches, args.samples)
        values = {key: out.numpy().T for key, out in outputs.items()}
        original_parallel = values["original"][:, 1] != 1e10
        shared_parallel = original_parallel & (values["hybrid"][:, 1] != 1e10)
        axis_out = wp.empty(args.count, dtype=wp.vec3, device="cuda:0")
        warp_graph = capture(make_axis_kernel(True), [data, sizes, axis_out], args.count, args.launches)
        axis_values = axis_out.numpy()
        witness_graphs, witness_outputs = {}, {}
        for key, use_warp in (("newton_helper", False), ("warp", True)):
            out = wp.empty((7, args.count), dtype=float, device="cuda:0")
            witness_outputs[key] = out
            witness_graphs[key] = capture(
                make_witness_kernel(use_warp, closest_segment_to_segment_points),
                [data, sizes, out],
                args.count,
                args.launches,
            )
        witness_timing = graph_times(witness_graphs, args.launches, args.samples)
        witness_values = {key: out.numpy().T for key, out in witness_outputs.items()}
        axis_timing = None
        axis_baseline = None
        # Timing a skipped/undefined nonparallel formula against a complete
        # query would not be a like-for-like performance comparison.
        if not mask.any() and not original_parallel.any():
            native_out = wp.empty(args.count, dtype=wp.vec3, device="cuda:0")
            native_graph = capture(make_axis_kernel(False), [data, sizes, native_out], args.count, args.launches)
            axis_timing = graph_times(
                {"newton_extracted": native_graph, "warp": warp_graph}, args.launches, args.samples
            )
            axis_baseline = native_out.numpy()
        subset = np.linspace(0, args.count - 1, min(256, args.count), dtype=int)
        reference = reference_distances(raw, size, subset)
        radii = size[subset, 0].astype(float) + size[subset, 2].astype(float)
        errors = {"warp_closest": float(np.max(np.abs(axis_values[subset, 2] - reference)))}
        if axis_baseline is not None:
            errors["newton_extracted"] = float(np.max(np.abs(axis_baseline[subset, 2] - reference)))
        for key, value in witness_values.items():
            errors[key + "_witness"] = float(np.max(np.abs(value[subset, 6] - reference)))
        minima = {key: np.min(value[:, :2], axis=1) for key, value in values.items()}
        for key, value in minima.items():
            active = np.isfinite(value[subset])
            errors[key + "_contact_minimum"] = float(
                np.max(np.abs(value[subset][active] - (reference - radii)[active]), initial=0.0)
            )
        row = {
            "workload": name,
            "count": args.count,
            "parallel_branch_count": int(mask.sum()),
            "original_two_candidate_count": int(original_parallel.sum()),
            "hybrid_two_candidate_count": int(np.sum(values["hybrid"][:, 1] != 1e10)),
            "branch_disagreements": int(np.sum(original_parallel != (values["hybrid"][:, 1] != 1e10))),
            "shared_parallel_max_record_delta": float(
                np.max(np.abs(values["hybrid"][shared_parallel] - values["original"][shared_parallel]), initial=0.0)
            ),
            "contact_us": timing,
            "axis_us": axis_timing,
            "witness_us": witness_timing,
            "max_abs_distance_error_subset": errors,
            "nonfinite_contact_records": {
                key: int(np.sum(~np.isfinite(value).all(axis=1))) for key, value in values.items()
            },
            "contacts_below_zero": {key: int(np.sum(value[:, :2] < 0)) for key, value in values.items()},
            "axis_closest_below_zero": int(np.sum(axis_values[:, 2] < size[:, 0] + size[:, 2])),
            "original_missed_overlaps_confirmed_subset": int(
                np.sum((reference - radii < -3e-6) & (minima["original"][subset] >= 0))
            ),
            "copy_max_record_delta": float(np.max(np.abs(values["copy"] - values["original"]))),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    metadata = {
        "newton_commit": revision,
        "warp_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": wp.get_device("cuda:0").name,
        "samples": args.samples,
        "launches": args.launches,
        "block_dim": 128,
        "seed": 2360,
        "output_layout": "SoA (11 scalars/contact, 7 scalars/witness); vec3 for axis parameters/distance",
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[2] / "warp/_src/geometry/closest.py").read_bytes()
        ).hexdigest(),
        "results": rows,
    }
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
