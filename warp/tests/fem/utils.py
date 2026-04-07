# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared integrands and type aliases for warp.fem tests."""

import warp as wp
import warp.fem as fem
from warp.fem import Field, Sample, integrand

vec6f = wp.types.vector(length=6, dtype=float)
mat66f = wp.types.matrix(shape=(6, 6), dtype=float)


@integrand
def linear_form(s: Sample, u: Field):
    return u(s)


@integrand
def bilinear_form(s: Sample, u: Field, v: Field):
    return u(s) * v(s)


@integrand
def scaled_linear_form(s: Sample, u: Field, scale: wp.array(dtype=float)):
    return u(s) * scale[0]


@integrand
def bilinear_field(s: fem.Sample, domain: fem.Domain):
    return fem.position(domain, s)[0] * fem.position(domain, s)[1]


@integrand
def grad_field(s: fem.Sample, p: fem.Field):
    return fem.grad(p, s)


@integrand
def piecewise_constant(s: Sample):
    return float(s.element_index)
