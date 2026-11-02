# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import warp.fem as fem
import warp.sparse as sp


@fem.integrand
def _screened_poisson(s: fem.Sample, u: fem.Field, v: fem.Field):
    return 0.01 * wp.dot(fem.grad(u, s), fem.grad(v, s)) + u(s) * v(s)


@wp.kernel
def _rhs(x: wp.array2d[wp.float64]):
    i, j = wp.tid()
    state = wp.rand_init(37, i * x.shape[1] + j)
    x[i, j] = wp.float64(wp.randn(state))


def problem(nrhs, resolution=32):
    geo = fem.Grid2D(res=(resolution, resolution))
    space = fem.make_polynomial_space(geo)
    a = fem.integrate(
        _screened_poisson, fields={"u": fem.make_trial(space), "v": fem.make_test(space)}, output_dtype=wp.float64
    )
    sp.bsr_compress(a)
    b = wp.empty((nrhs, a.shape[0]), dtype=wp.float64)
    wp.launch(_rhs, b.shape, [b])
    return a, b
