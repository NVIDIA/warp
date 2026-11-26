# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Implicit Projection (forward-mode jets)
#
# Two flows push a cloud of 2D points onto the zero level set of a metaball
# (Gaussian blob) field f, one point per thread, integrated with RK4:
#
#   projection    dx/dt = -f grad(f) / ||grad(f)||^2
#                 moves each point onto the surface ALONG the gradient. Needs
#                 only grad(f) -- a FIRST-order jet.
#
#   closest-point Newton flow on the KKT system of  min ||x - p0||^2  s.t.
#                 f(x) = 0, with state z = (x, lambda):
#
#                     R = [ x - p0 + lambda grad(f) ;  f ]
#                     J = [ I + lambda H,  grad(f) ;  grad(f)^T,  0 ]
#                     dz/dt = -J^{-1} R
#
#                 moves each point to a surface point where the KKT conditions
#                 hold: a local stationary point of the distance, usually the
#                 nearest one, but not guaranteed to be on this nonconvex
#                 surface. Its Jacobian contains the Hessian H = grad^2(f) -- a
#                 SECOND-order jet. There is no first-order shortcut here.
#
# The field is written ONCE as a plain @wp.func; evaluating it over a jet
# yields f, grad(f) (and H) as a side effect -- no hand-differentiation. The two
# flows land at different points, which is exactly why the Hessian earns its
# keep for the closest-point problem but not for the plain projection.
#
###########################################################################

from typing import Any

import numpy as np

import warp as wp

# ---------------------------------------------------------------------------
# Metaball field  f(x) = sum_j exp(-||x - c_j||^2 / sigma_j^2) / iso_j - 1
# ---------------------------------------------------------------------------

# One blob per row: (cx, cy, sigma^2, iso).
_BLOBS = (
    (-0.60, 0.00, 0.45**2, 0.65),
    (0.45, 0.00, 0.55**2, 0.65),
    (0.30, 0.90, 0.40**2, 0.99),
)

J1 = wp.JetSpace(wp.vec2)  # first-order: value + gradient, over one 2D point
J2 = wp.JetSpace2(2)  # second-order: value + gradient + Hessian, over two scalars

# Tiny guard so the projection step is finite at exact grad(f) = 0 (measure zero),
# where it evaluates to 0 -- matching the Moore-Penrose inverse. Not damping: both
# flows converge fine with plain inverses on this field.
_EPS = wp.constant(1.0e-9)


@wp.func
def _blob(x0: Any, x1: Any, cx: float, cy: float, s2: float, iso: float):
    dx = x0 - cx
    dy = x1 - cy
    return wp.exp(-(dx * dx + dy * dy) / s2) / iso


@wp.func
def field(x0: Any, x1: Any):
    # Generic in the argument type: specializes on plain float, first-order jet,
    # and second-order jet from this single definition.
    return (
        _blob(x0, x1, -0.60, 0.00, 0.45 * 0.45, 0.65)
        + _blob(x0, x1, 0.45, 0.00, 0.55 * 0.55, 0.65)
        + _blob(x0, x1, 0.30, 0.90, 0.40 * 0.40, 0.99)
        - 1.0
    )


@wp.func
def projection_rhs(x: wp.vec2):
    # FIRST-order jet: evaluating the field over J1 gives f (.value) and grad(f)
    # (.grad). The step is the Moore-Penrose pseudo-inverse of the 1x2 constraint
    # Jacobian grad(f)^T applied to -f -- the min-norm solution of grad(f) . dx = -f,
    # which for a row vector is exactly -f grad(f) / ||grad(f)||^2.
    p = J1.seed(x)
    e = field(p[0], p[1])
    g = wp.vec2(e.grad[0], e.grad[1])
    return -(e.value / (wp.dot(g, g) + _EPS)) * g


@wp.func
def closest_point_rhs(z: wp.vec3, p0: wp.vec2):
    # SECOND-order jet: evaluating the field over J2 gives f, grad(f), AND the 2x2
    # Hessian H in a single forward pass -- and H is exactly what the KKT Jacobian
    # needs (its I + lambda*H block). No hand-written second derivatives.
    a, b = J2.seed(z[0], z[1])
    e = field(a, b)
    g0, g1 = e.grad[0], e.grad[1]
    lam = z[2]
    r = wp.vec3(z[0] - p0[0] + lam * g0, z[1] - p0[1] + lam * g1, e.value)
    jac = wp.mat33(
        1.0 + lam * e.hess[0, 0],
        lam * e.hess[0, 1],
        g0,
        lam * e.hess[1, 0],
        1.0 + lam * e.hess[1, 1],
        g1,
        g0,
        g1,
        0.0,
    )
    # Plain Newton -J^-1 r. No damping needed: the flow moves x toward the surface
    # (where grad(f) != 0) and J's identity block keeps it well conditioned, so it
    # converges to r = 0 exactly.
    return -(wp.inverse(jac) * r)


@wp.kernel
def integrate_projection(x0: wp.array[wp.vec2], dt: float, steps: int, traj: wp.array2d[wp.vec2]):
    i = wp.tid()
    x = x0[i]
    traj[i, 0] = x
    for s in range(steps):
        k1 = projection_rhs(x)
        k2 = projection_rhs(x + (0.5 * dt) * k1)
        k3 = projection_rhs(x + (0.5 * dt) * k2)
        k4 = projection_rhs(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[i, s + 1] = x


@wp.kernel
def integrate_closest_point(x0: wp.array[wp.vec2], dt: float, steps: int, traj: wp.array2d[wp.vec2]):
    i = wp.tid()
    p0 = x0[i]
    z = wp.vec3(p0[0], p0[1], 0.0)  # lambda starts at 0
    traj[i, 0] = wp.vec2(z[0], z[1])
    for s in range(steps):
        k1 = closest_point_rhs(z, p0)
        k2 = closest_point_rhs(z + (0.5 * dt) * k1, p0)
        k3 = closest_point_rhs(z + (0.5 * dt) * k2, p0)
        k4 = closest_point_rhs(z + dt * k3, p0)
        z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[i, s + 1] = wp.vec2(z[0], z[1])


def field_np(xy):
    f = np.full(xy.shape[0], -1.0)
    for cx, cy, s2, iso in _BLOBS:
        d = xy - np.array([cx, cy])
        f += np.exp(-(d * d).sum(1) / s2) / iso
    return f


class Example:
    def __init__(self, num_points=100, steps=200, dt=0.05, device=None):
        self.device = device
        self.steps = steps
        self.dt = dt
        rng = np.random.default_rng(0)
        p = (2.0 * rng.random((num_points, 2)) - 1.0).astype(np.float32)
        self.x0 = wp.array(p, dtype=wp.vec2, device=device)
        self.proj = wp.zeros((num_points, steps + 1), dtype=wp.vec2, device=device)
        self.cp = wp.zeros((num_points, steps + 1), dtype=wp.vec2, device=device)

    def run(self):
        n = self.x0.shape[0]
        wp.launch(
            integrate_projection, dim=n, inputs=[self.x0, self.dt, self.steps], outputs=[self.proj], device=self.device
        )
        wp.launch(
            integrate_closest_point, dim=n, inputs=[self.x0, self.dt, self.steps], outputs=[self.cp], device=self.device
        )
        wp.synchronize_device(self.device)

    def report(self):
        for name, traj in (("projection", self.proj), ("closest-point", self.cp)):
            final = traj.numpy()[:, -1]
            resid = np.abs(field_np(final.astype(np.float64)))
            print(f"  {name:14} max |f| at final points = {resid.max():.2e}   mean = {resid.mean():.2e}")

    def plot(self, path):
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            print("  (matplotlib not available; skipping plot)")
            return
        gx, gy = np.meshgrid(np.linspace(-1.5, 1.5, 400), np.linspace(-1.2, 1.2, 400))
        F = field_np(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(gx.shape)
        p0 = self.x0.numpy()
        proj = self.proj.numpy()
        cp = self.cp.numpy()

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_aspect("equal")
        ax.contourf(gx, gy, F, levels=np.linspace(-1.0, 1.6, 40), cmap="Greys", alpha=0.30)
        ax.contour(gx, gy, F, levels=[0.0], colors="k", linewidths=2.0)
        for i in range(proj.shape[0]):
            ax.plot(proj[i, :, 0], proj[i, :, 1], "-", color="#1f77b4", lw=0.6, alpha=0.7)
            ax.plot(cp[i, :, 0], cp[i, :, 1], "-", color="#d62728", lw=0.6, alpha=0.7)
        ax.scatter(p0[:, 0], p0[:, 1], s=14, c="k", alpha=0.35, label="initial")
        ax.scatter(
            proj[:, -1, 0],
            proj[:, -1, 1],
            s=26,
            facecolors="none",
            edgecolors="#1f77b4",
            label="projection (grad only)",
        )
        ax.scatter(cp[:, -1, 0], cp[:, -1, 1], s=34, c="#d62728", marker="x", label="closest point (Hessian)")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.2, 1.2)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("Metaball projection flows via forward-mode jets")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        print(f"  saved {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num-points", type=int, default=100, help="Number of points to project.")
    parser.add_argument("--steps", type=int, default=200, help="Number of RK4 integration steps.")
    parser.add_argument("--dt", type=float, default=0.05, help="Integration step size.")
    parser.add_argument("--headless", action="store_true", help="Do not save a plot.")
    parser.add_argument("--plot", type=str, default="example_implicit_projection.png", help="Output image path.")
    parser.add_argument("--stage-path", type=str, default=None, help="Unused; accepted for harness compatibility.")
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(num_points=args.num_points, steps=args.steps, dt=args.dt)
        example.run()
        example.report()
        if not args.headless and args.plot:
            example.plot(args.plot)
