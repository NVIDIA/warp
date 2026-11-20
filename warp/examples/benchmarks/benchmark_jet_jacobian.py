# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Jacobian of a many-output function: forward jet vs reverse mode
###########################################################################

"""Three ways to compute the m x k Jacobian of r: R^k -> R^m at every point of a batch, compared.

The function is a small forward model with k = 3 parameters. It integrates a damped
pendulum for a fixed number of steps (the *shared stage*, parameters = initial angle,
initial velocity and forcing) and then reads out m values of the final state (the
*readouts*). Many outputs from few inputs is the case where forward mode does less work
than reverse mode: one forward pass carries all k directions, while reverse mode needs one
backward sweep per output.

    jet      : one launch. The shared stage runs once on a width-3 jet and every readout
               is evaluated on it; each readout's k coeffs are one row of the Jacobian.

    wp.grad  : one launch. Each row is wp.grad(residual)(x, j), a register-resident
               reverse sweep. The scalar residual re-runs the shared stage for every row.

    tape     : the user-guide pattern of m tape.backward launches with a one-hot seed per
               output. Timed on the backward sweeps alone, so it favours the tape.

All three write the Jacobian output-major (jac[j * n + i]), so neighbouring threads
store consecutive elements. Storing it point-major (jac[i * m + j]) leaves each thread
writing its own contiguous block, and the much faster jet then becomes bound by those
strided stores.

    uv run warp/examples/benchmarks/benchmark_jet_jacobian.py --device cuda:0
"""

import argparse
import time
from typing import Any

import numpy as np

import warp as wp

J3 = wp.JetSpace(wp.vec3)  # k = 3: pendulum angle, velocity and forcing

STEPS = 16  # shared-stage length, within Warp's default loop-unroll limit


@wp.func
def shared_state(x0: Any, x1: Any, x2: Any):
    """Integrate a damped, forced pendulum and return its final angle and velocity."""
    u = x0
    v = x1
    for _step in range(STEPS):
        v = v + 0.1 * (x2 - wp.sin(u) - 0.05 * v)
        u = u + 0.1 * v
    return u, v


@wp.func
def readout(x2: Any, u: Any, v: Any, j: int):
    """Return the j-th cheap readout of the shared state."""
    a = float((j * 7) % 11 - 5) * 0.2
    b = float((j * 5) % 13 - 6) * 0.15
    c = float((j * 3) % 7 - 3) * 0.1
    d = float((j * 11) % 5 - 2) * 0.3
    return a * u + b * v + c * u * v + d * x2


@wp.func
def residual(x: wp.vec3, j: int):
    u, v = shared_state(x[0], x[1], x[2])
    return readout(x[2], u, v, j)


@wp.kernel(enable_backward=False)
def jacobian_jet_kernel(x: wp.array[wp.vec3], m: int, jac: wp.array[J3.grad_type_scalar]):
    i = wp.tid()
    n = x.shape[0]
    p = J3.seed(x[i])
    u, v = shared_state(p[0], p[1], p[2])
    for j in range(m):
        jac[j * n + i] = readout(p[2], u, v, j).grad


@wp.kernel(enable_backward=False)
def jacobian_grad_kernel(x: wp.array[wp.vec3], m: int, jac: wp.array[wp.vec3]):
    i = wp.tid()
    n = x.shape[0]
    for j in range(m):
        grad_x = wp.grad(residual)(x[i], j)
        jac[j * n + i] = grad_x[0]


@wp.kernel
def residual_kernel(x: wp.array[wp.vec3], m: int, r: wp.array[float]):
    i = wp.tid()
    n = x.shape[0]
    for j in range(m):
        r[j * n + i] = residual(x[i], j)


@wp.kernel
def one_hot_kernel(row: int, n: int, seed: wp.array[float]):
    k = wp.tid()
    seed[k] = float(k // n == row)


def best_time_ms(fn, reps, device):
    times = []
    for _ in range(reps):
        wp.synchronize_device(device)
        start = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        times.append(time.perf_counter() - start)
    return min(times) * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default=None, help="Warp device to run on")
    parser.add_argument("--n", type=int, default=200_000, help="number of points in the batch")
    parser.add_argument("--reps", type=int, default=10, help="launches timed per method (best is reported)")
    parser.add_argument("--ms", type=int, nargs="+", default=[1, 4, 16, 32, 64], help="output counts m to sweep")
    parser.add_argument(
        "--tape-max-m",
        type=int,
        default=32,
        help="skip the tape beyond this m: it stores one n*m seed per output, so memory grows as n*m^2",
    )
    args = parser.parse_args()

    device = wp.get_device(args.device)
    n = args.n
    rng = np.random.default_rng(0)
    x = wp.array(rng.uniform(0.3, 1.2, (n, 3)).astype(np.float32), dtype=wp.vec3, device=device, requires_grad=True)

    print(f"Jacobian of r: R^3 -> R^m at {n} points on {device} (times in ms, best of {args.reps})")
    print(f"{'m':>4} {'jet':>8} {'wp.grad':>9} {'tape':>9} {'grad/jet':>9} {'tape/jet':>9} {'max rel err':>12}")

    for m in args.ms:
        jac_jet = wp.zeros(m * n, dtype=J3.grad_type_scalar, device=device)
        jac_grad = wp.zeros(m * n, dtype=wp.vec3, device=device)

        def run_jet(m=m, jac_jet=jac_jet):
            wp.launch(jacobian_jet_kernel, dim=n, inputs=[x, m], outputs=[jac_jet], device=device)

        def run_grad(m=m, jac_grad=jac_grad):
            wp.launch(jacobian_grad_kernel, dim=n, inputs=[x, m], outputs=[jac_grad], device=device)

        run_jet()
        run_grad()
        t_jet = best_time_ms(run_jet, args.reps, device)
        t_grad = best_time_ms(run_grad, args.reps, device)

        reference = jac_grad.numpy()
        scale = max(1.0, np.abs(reference).max())
        error = np.abs(jac_jet.numpy() - reference).max() / scale

        t_tape = None
        if m <= args.tape_max_m:
            r = wp.zeros(m * n, dtype=float, device=device, requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(residual_kernel, dim=n, inputs=[x, m], outputs=[r], device=device)
            seeds = []
            for row in range(m):
                seed = wp.zeros(m * n, dtype=float, device=device)
                wp.launch(one_hot_kernel, dim=m * n, inputs=[row, n], outputs=[seed], device=device)
                seeds.append(seed)
            jac_tape = wp.zeros(m * n, dtype=wp.vec3, device=device)

            def run_tape(m=m, r=r, tape=tape, seeds=seeds, jac_tape=jac_tape):
                for row in range(m):
                    tape.backward(grads={r: seeds[row]})
                    wp.copy(jac_tape, x.grad, dest_offset=row * n, count=n)
                    tape.zero()

            run_tape()
            t_tape = best_time_ms(run_tape, max(3, args.reps // 2), device)
            error = max(error, np.abs(jac_tape.numpy() - reference).max() / scale)

        tape_text = f"{t_tape:>9.3f}" if t_tape is not None else f"{'-':>9}"
        tape_ratio = f"{t_tape / t_jet:>8.1f}x" if t_tape is not None else f"{'-':>9}"
        print(f"{m:>4} {t_jet:>8.3f} {t_grad:>9.3f} {tape_text} {t_grad / t_jet:>8.1f}x {tape_ratio} {error:>12.1e}")


if __name__ == "__main__":
    main()
