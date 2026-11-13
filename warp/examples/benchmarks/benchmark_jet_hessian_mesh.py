# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Local element Hessian: reverse-over-forward via tape vs in-kernel wp.grad
###########################################################################

"""Assemble each element's local k x k Hessian, two reverse-over-forward ways and one forward way.

Same generic vec3 energies as the gradient benchmark (spring k=6, triangle k=9,
tet k=12), on per-element data. The two reverse-over-forward routes differ only
in HOW the reverse sweep is taken:

    width-1 (tape)   for each basis direction e_j, a width-1 dual forward gives
                     the directional derivative Dg.e_j, then one tape.backward
                     -> H e_j (a column). k (forward + backward) launches + tape.

    width-1 (ink)    the same k directional derivatives, but each reverse sweep
                     is taken IN the kernel with wp.grad -- register-resident, one
                     launch, no tape, no global .grad. Local, so it scatters into
                     an assembled sparse Hessian without the shared-array aliasing.

    forward-2        a second-order jet: one forward pass writes the whole k x k
                     Hessian, no reverse at all, on the same vector-form energies.
                     Its single fused pass is the fastest runtime at k=12, but its
                     O(k^2) state makes it the slowest to compile: on an RTX PRO
                     6000 Blackwell (MIG 1g.24gb) with CUDA 12.9, the tet module
                     took about 8.5 minutes to compile on a cold cache.

    uv run warp/examples/benchmarks/benchmark_jet_hessian_mesh.py --device cuda:0
    uv run warp/examples/benchmarks/benchmark_jet_hessian_mesh.py --device cuda:0 --graph
"""

import argparse
import time
from collections import namedtuple
from typing import Any

import numpy as np

import warp as wp

_REST = wp.constant(1.0)
_MU = wp.constant(1.0)
_LAM = wp.constant(1.0)

# One jet per node (a spring has two, a triangle three, a tetrahedron four).
# JD is the width-1 dual for the directional derivatives; J2K the second-order jet.
JD = {k: wp.JetSpace((wp.vec3,) * (k // 3), width=1) for k in (6, 9, 12)}
J2K = {k: wp.JetSpace2((wp.vec3,) * (k // 3)) for k in (6, 9, 12)}  # forward-2


# ---------------------------------------------------------------------------
# Energies -- generic vec3 wp.funcs (plain, jet, and width-1 dual).
# ---------------------------------------------------------------------------


@wp.func
def spring_energy(a: Any, b: Any):
    s = wp.length(a - b) - _REST
    return 0.5 * s * s


@wp.func
def triangle_energy(p0: Any, p1: Any, p2: Any):
    j0 = p1 - p0
    j1 = p2 - p0
    s00 = wp.dot(j0, j0)
    s01 = wp.dot(j0, j1)
    s11 = wp.dot(j1, j1)
    tr = s00 + s11
    det = s00 * s11 - s01 * s01
    return tr + tr / det


@wp.func
def tet_energy(p0: Any, p1: Any, p2: Any, p3: Any):
    f0 = p1 - p0
    f1 = p2 - p0
    f2 = p3 - p0
    i1 = wp.dot(f0, f0) + wp.dot(f1, f1) + wp.dot(f2, f2)
    logj = wp.log(wp.dot(f0, wp.cross(f1, f2)))
    return 0.5 * _MU * (i1 - 3.0) - _MU * logj + 0.5 * _LAM * logj * logj


# ---------------------------------------------------------------------------
# NumPy references for the FD Hessian gate.
# ---------------------------------------------------------------------------


def spring_np(z):
    d = z[:, 0:3] - z[:, 3:6]
    return 0.5 * (np.linalg.norm(d, axis=1) - 1.0) ** 2


def triangle_np(z):
    p0, p1, p2 = z[:, 0:3], z[:, 3:6], z[:, 6:9]
    j0, j1 = p1 - p0, p2 - p0
    s00 = (j0 * j0).sum(1)
    s01 = (j0 * j1).sum(1)
    s11 = (j1 * j1).sum(1)
    tr = s00 + s11
    return tr + tr / (s00 * s11 - s01 * s01)


def tet_np(z):
    p0, p1, p2, p3 = z[:, 0:3], z[:, 3:6], z[:, 6:9], z[:, 9:12]
    f = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=2)
    i1 = (f * f).sum((1, 2))
    logj = np.log(np.linalg.det(f))
    return 0.5 * (i1 - 3.0) - logj + 0.5 * logj**2


def hess_fd(fn, z, h=2e-4):
    n, k = z.shape
    H = np.empty((n, k, k))
    for p in range(k):
        for q in range(k):
            ep = np.zeros((n, k))
            ep[:, p] = h
            eq = np.zeros((n, k))
            eq[:, q] = h
            H[:, p, q] = (fn(z + ep + eq) - fn(z + ep - eq) - fn(z - ep + eq) + fn(z - ep - eq)) / (4 * h * h)
    return H


def sample(k, m, seed=0):
    rng = np.random.default_rng(seed)
    rest = {6: [0, 0, 0, 1, 0, 0], 9: [0, 0, 0, 1, 0, 0, 0.5, 0.8660254, 0], 12: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]}[
        k
    ]
    x = (np.array(rest, np.float32) + 0.12 * rng.standard_normal((m, k)).astype(np.float32)).reshape(m, k // 3, 3)
    if k == 12:
        # Keep det(F) > 0 so the neo-Hookean log is defined (flip inverted tets).
        f = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2).astype(np.float64)
        neg = np.linalg.det(f) < 0.0  # flip only inverted tets (swap negates det)
        x[neg, 2], x[neg, 3] = x[neg, 3].copy(), x[neg, 2].copy()
    return x


# ---------------------------------------------------------------------------
# Per-element kernels. build_* returns (k, directional, hess_w1, hess_fwd2).
# ---------------------------------------------------------------------------


def build_spring():
    J1 = JD[6]

    @wp.kernel
    def directional(x: wp.array2d[wp.vec3], d: wp.array2d[wp.vec3], dv: wp.array[wp.float32]):
        i = wp.tid()
        a, b = J1.directional(x[i, 0], x[i, 1], d[i, 0], d[i, 1])
        dv[i] = spring_energy(a, b).grad[0]

    @wp.func
    def dirderiv(a: wp.vec3, b: wp.vec3, da: wp.vec3, db: wp.vec3):
        ja, jb = J1.directional(a, b, da, db)
        return spring_energy(ja, jb).grad[0]

    @wp.kernel(enable_backward=False)
    def hess_w1(x: wp.array2d[wp.vec3], out: wp.array3d[wp.float32]):
        i = wp.tid()
        a, b = x[i, 0], x[i, 1]
        for j in range(wp.static(6)):
            da = wp.vec3(0.0, 0.0, 0.0)
            db = wp.vec3(0.0, 0.0, 0.0)
            if wp.static(j < 3):
                da[wp.static(j % 3)] = 1.0
            else:
                db[wp.static(j % 3)] = 1.0
            ga, gb, _a, _b = wp.grad(dirderiv)(a, b, da, db)
            for c in range(3):
                out[i, 0 * 3 + c, j] = ga[c]
                out[i, 1 * 3 + c, j] = gb[c]

    J2 = J2K[6]

    @wp.kernel(enable_backward=False)
    def hess_fwd2(x: wp.array2d[wp.vec3], out: wp.array3d[wp.float32]):
        i = wp.tid()
        ja, jb = J2.seed(x[i, 0], x[i, 1])
        h = spring_energy(ja, jb).hess
        for p in range(6):
            for q in range(6):
                out[i, p, q] = h[p, q]

    return 6, directional, hess_w1, hess_fwd2


def build_triangle():
    J1 = JD[9]

    @wp.kernel
    def directional(x: wp.array2d[wp.vec3], d: wp.array2d[wp.vec3], dv: wp.array[wp.float32]):
        i = wp.tid()
        a, b, c = J1.directional(x[i, 0], x[i, 1], x[i, 2], d[i, 0], d[i, 1], d[i, 2])
        dv[i] = triangle_energy(a, b, c).grad[0]

    @wp.func
    def dirderiv(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, d0: wp.vec3, d1: wp.vec3, d2: wp.vec3):
        a, b, c = J1.directional(p0, p1, p2, d0, d1, d2)
        return triangle_energy(a, b, c).grad[0]

    @wp.kernel(enable_backward=False)
    def hess_w1(x: wp.array2d[wp.vec3], out: wp.array3d[wp.float32]):
        i = wp.tid()
        p0, p1, p2 = x[i, 0], x[i, 1], x[i, 2]
        for j in range(wp.static(9)):
            d0 = wp.vec3(0.0, 0.0, 0.0)
            d1 = wp.vec3(0.0, 0.0, 0.0)
            d2 = wp.vec3(0.0, 0.0, 0.0)
            if wp.static(j < 3):
                d0[wp.static(j % 3)] = 1.0
            elif wp.static(j < 6):
                d1[wp.static(j % 3)] = 1.0
            else:
                d2[wp.static(j % 3)] = 1.0
            g0, g1, g2, _0, _1, _2 = wp.grad(dirderiv)(p0, p1, p2, d0, d1, d2)
            for c in range(3):
                out[i, 0 * 3 + c, j] = g0[c]
                out[i, 1 * 3 + c, j] = g1[c]
                out[i, 2 * 3 + c, j] = g2[c]

    J2 = J2K[9]

    @wp.kernel(enable_backward=False)
    def hess_fwd2(x: wp.array2d[wp.vec3], out: wp.array3d[wp.float32]):
        i = wp.tid()
        j0, j1, j2 = J2.seed(x[i, 0], x[i, 1], x[i, 2])
        h = triangle_energy(j0, j1, j2).hess
        for p in range(9):
            for q in range(9):
                out[i, p, q] = h[p, q]

    return 9, directional, hess_w1, hess_fwd2


def build_tet():
    J1 = JD[12]

    @wp.kernel
    def directional(x: wp.array2d[wp.vec3], d: wp.array2d[wp.vec3], dv: wp.array[wp.float32]):
        i = wp.tid()
        a, b, c, e = J1.directional(x[i, 0], x[i, 1], x[i, 2], x[i, 3], d[i, 0], d[i, 1], d[i, 2], d[i, 3])
        dv[i] = tet_energy(a, b, c, e).grad[0]

    @wp.func
    def dirderiv(
        p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, d0: wp.vec3, d1: wp.vec3, d2: wp.vec3, d3: wp.vec3
    ):
        a, b, c, e = J1.directional(p0, p1, p2, p3, d0, d1, d2, d3)
        return tet_energy(a, b, c, e).grad[0]

    @wp.kernel(enable_backward=False)
    def hess_w1(x: wp.array2d[wp.vec3], out: wp.array3d[wp.float32]):
        i = wp.tid()
        p0, p1, p2, p3 = x[i, 0], x[i, 1], x[i, 2], x[i, 3]
        for j in range(wp.static(12)):
            d0 = wp.vec3(0.0, 0.0, 0.0)
            d1 = wp.vec3(0.0, 0.0, 0.0)
            d2 = wp.vec3(0.0, 0.0, 0.0)
            d3 = wp.vec3(0.0, 0.0, 0.0)
            if wp.static(j < 3):
                d0[wp.static(j % 3)] = 1.0
            elif wp.static(j < 6):
                d1[wp.static(j % 3)] = 1.0
            elif wp.static(j < 9):
                d2[wp.static(j % 3)] = 1.0
            else:
                d3[wp.static(j % 3)] = 1.0
            g0, g1, g2, g3, _0, _1, _2, _3 = wp.grad(dirderiv)(p0, p1, p2, p3, d0, d1, d2, d3)
            for c in range(3):
                out[i, 0 * 3 + c, j] = g0[c]
                out[i, 1 * 3 + c, j] = g1[c]
                out[i, 2 * 3 + c, j] = g2[c]
                out[i, 3 * 3 + c, j] = g3[c]

    J2 = J2K[12]

    @wp.kernel(enable_backward=False)
    def hess_fwd2(x: wp.array2d[wp.vec3], out: wp.array3d[wp.float32]):
        i = wp.tid()
        j0, j1, j2, j3 = J2.seed(x[i, 0], x[i, 1], x[i, 2], x[i, 3])
        h = tet_energy(j0, j1, j2, j3).hess
        for p in range(12):
            for q in range(12):
                out[i, p, q] = h[p, q]

    return 12, directional, hess_w1, hess_fwd2


@wp.kernel
def _scatter_grad_row(xgrad: wp.array2d[wp.vec3], row: int, nodes: int, hess: wp.array3d[wp.float32]):
    i = wp.tid()
    for n in range(nodes):
        g = xgrad[i, n]
        hess[i, row, n * 3 + 0] = g[0]
        hess[i, row, n * 3 + 1] = g[1]
        hess[i, row, n * 3 + 2] = g[2]


Spec = namedtuple("Spec", "k build energy_np")
ELEMENTS = {
    "spring": Spec(6, build_spring, spring_np),
    "triangle": Spec(9, build_triangle, triangle_np),
    "tet": Spec(12, build_tet, tet_np),
}


class Width1Tape:
    label = "width-1(tape)"

    def __init__(self, spec, kernels, x_np, device):
        self.k, self.device = spec.k, device
        self.nodes = spec.k // 3
        self.n = x_np.shape[0]
        self.directional = kernels[0]
        self.x = wp.array(x_np, dtype=wp.vec3, device=device, requires_grad=True)
        self.dv = wp.zeros(self.n, dtype=wp.float32, device=device, requires_grad=True)
        self.ones = wp.ones(self.n, dtype=wp.float32, device=device)
        self.hessian = wp.zeros((self.n, self.k, self.k), dtype=wp.float32, device=device)
        self.dirs = []
        for j in range(self.k):
            e = np.zeros((self.n, self.nodes, 3), np.float32)
            e[:, j // 3, j % 3] = 1.0
            self.dirs.append(wp.array(e, dtype=wp.vec3, device=device))

    def run(self):
        for j in range(self.k):
            self.dv.zero_()
            tape = wp.Tape()
            with tape:
                wp.launch(
                    self.directional, dim=self.n, inputs=[self.x, self.dirs[j]], outputs=[self.dv], device=self.device
                )
            tape.backward(grads={self.dv: self.ones})
            wp.launch(
                _scatter_grad_row, dim=self.n, inputs=[self.x.grad, j, self.nodes, self.hessian], device=self.device
            )
            tape.zero()

    def hess(self):
        # tape gives columns H e_j; the Hessian is symmetric so rows == columns.
        return self.hessian.numpy()


class Width1InKernel:
    label = "width-1(ink)"

    def __init__(self, spec, kernels, x_np, device):
        self.k, self.device, self.n = spec.k, device, x_np.shape[0]
        self.hess_w1 = kernels[1]
        self.x = wp.array(x_np, dtype=wp.vec3, device=device)
        self.hessian = wp.zeros((self.n, self.k, self.k), dtype=wp.float32, device=device)

    def run(self):
        wp.launch(self.hess_w1, dim=self.n, inputs=[self.x], outputs=[self.hessian], device=self.device)

    def hess(self):
        return self.hessian.numpy()


class Forward2:
    """One second-order jet forward pass writes the whole k x k Hessian; no reverse."""

    label = "forward-2"

    def __init__(self, spec, kernels, x_np, device):
        self.k, self.device, self.n = spec.k, device, x_np.shape[0]
        self.kernel = kernels[2]  # None for elements above the compile cap
        self.x = wp.array(x_np, dtype=wp.vec3, device=device)
        self.hessian = wp.zeros((self.n, self.k, self.k), dtype=wp.float32, device=device)

    def run(self):
        wp.launch(self.kernel, dim=self.n, inputs=[self.x], outputs=[self.hessian], device=self.device)

    def hess(self):
        return self.hessian.numpy()


def timeit(fn, device, reps):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)
    return best


def time_strategy(s, device, reps, use_graph):
    s.run()
    wp.synchronize_device(device)
    if not use_graph:
        return timeit(s.run, device, reps)
    with wp.ScopedCapture(device) as cap:
        s.run()
    return timeit(lambda: wp.capture_launch(cap.graph), device, reps)


STRATEGIES = [Width1Tape, Width1InKernel]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--element", type=str, default=None, choices=list(ELEMENTS))
    parser.add_argument("--m", type=int, nargs="+", default=[200_000])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--graph", action="store_true", help="Time warm graph-captured replay.")
    args = parser.parse_args()

    device = args.device
    mode = "graph replay (warm)" if args.graph else "per-launch"
    print(f"device: {wp.get_device(device)}   reps: {args.reps}   timing: {mode}\n")

    header = (
        f"{'element':>9} {'k':>3} {'m':>9} {'w1-tape':>9} {'w1-ink':>9} {'fwd-2':>9}"
        f" {'ink/w1':>7} {'ink/f2':>7} {'|err|':>9}"
    )
    print(header)
    print("-" * len(header))
    print("(ink = in-kernel wp.grad reverse; fwd-2 = second-order jet, no reverse; ratios <1 mean ink faster;")
    print(" |err| = max of inter-method disagreement and FD error on well-conditioned elements)")

    for name in [args.element] if args.element else list(ELEMENTS):
        spec = ELEMENTS[name]
        kernels = spec.build()[1:]
        strategies = list(STRATEGIES) + ([Forward2] if kernels[2] is not None else [])
        for m in args.m:
            x_np = sample(spec.k, m)
            runs = {S.label: S(spec, kernels, x_np, device) for S in strategies}
            for s in runs.values():
                s.run()

            hs = [s.hess() for s in runs.values()]
            ref = hess_fd(spec.energy_np, x_np.reshape(m, spec.k).astype(np.float64))
            # Two checks. (1) The independent routes must agree with each other to
            # float precision (the strong test). (2) They must match FD on WELL-
            # CONDITIONED elements -- a rare near-degenerate element (tiny det) has an
            # enormous, stiff Hessian that FD can't resolve, so exclude it.
            mag = np.abs(ref).reshape(m, -1).max(1)
            well = np.isfinite(ref).reshape(m, -1).all(1) & (mag < 1e2)
            fin = np.isfinite(hs[0]).reshape(m, -1).all(1)
            inter = max(np.abs(h[fin] - hs[0][fin]).max() for h in hs) / max(np.abs(hs[0][fin]).max(), 1.0)
            fd = max(np.abs(h[well] - ref[well]).max() for h in hs) / max(np.abs(ref[well]).max(), 1.0)
            err = max(inter, fd)

            t = {lbl: time_strategy(s, device, args.reps, args.graph) for lbl, s in runs.items()}
            tw1, tik = t["width-1(tape)"], t["width-1(ink)"]
            tf2 = t.get("forward-2")
            f2s = f"{tf2 * 1e3:>9.3f}" if tf2 else f"{'-':>9}"
            f2r = f"{tik / tf2:>6.2f}x" if tf2 else f"{'-':>7}"
            print(
                f"{name:>9} {spec.k:>3} {m:>9} {tw1 * 1e3:>9.3f} {tik * 1e3:>9.3f} {f2s}"
                f" {tik / tw1:>6.2f}x {f2r} {err:>9.1e}"
            )


if __name__ == "__main__":
    main()
