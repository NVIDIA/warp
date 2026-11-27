# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracked device transforms, cumulative coefficients and weighted isovalues."""

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from .field import LevelField, basis_key, find_key, query_vjp_function, reference_evaluation_kernel
from .keyed import coordinates

wp.set_module_options({"fuse_fp": False})


@lru_cache(None)
def precision_kernels(dtype):
    vec3 = wp.vec3d if dtype == wp.float64 else wp.vec3
    mat33 = wp.mat33d if dtype == wp.float64 else wp.mat33
    mat44 = wp.mat44d if dtype == wp.float64 else wp.mat44

    @wp.kernel
    def cumulative_backward_kernel(
        keys: wp.array[wp.int64], grad: wp.array[dtype], grad_coefficients: wp.array[dtype], depth: int
    ):
        i = wp.tid()
        if int(keys[i] >> wp.int64(48)) == depth:
            g = grad[i]
            wp.atomic_add(grad_coefficients, i, g)
            if depth > 0:
                c = coordinates(keys[i])
                for corner in range(8):
                    p = wp.vec3i(c[0] // 2 + corner // 4, c[1] // 2 + (corner // 2) % 2, c[2] // 2 + corner % 2)
                    h = (
                        (dtype(1) - wp.abs(dtype(c[0]) * dtype(0.5) - dtype(p[0])))
                        * (dtype(1) - wp.abs(dtype(c[1]) * dtype(0.5) - dtype(p[1])))
                        * (dtype(1) - wp.abs(dtype(c[2]) * dtype(0.5) - dtype(p[2])))
                    )
                    if h > dtype(0):
                        j = find_key(keys, basis_key(depth - 1, p[0], p[1], p[2]))
                        if j >= 0:
                            wp.atomic_add(grad, j, g * h)

    query_vjp = query_vjp_function(dtype)

    @wp.kernel
    def evaluate_backward_kernel(
        keys: wp.array[wp.int64],
        coefficients: wp.array[dtype],
        cumulative: wp.array[dtype],
        queries: wp.array[vec3],
        depth: int,
        grad_value: wp.array[dtype],
        grad_spatial: wp.array[vec3],
        grad_coefficients: wp.array[dtype],
        grad_cumulative: wp.array[dtype],
        grad_queries: wp.array[vec3],
    ):
        i = wp.tid()
        if grad_value[i] == dtype(0) and grad_spatial[i] == vec3(dtype(0)):
            return
        gradient = query_vjp(
            keys,
            coefficients,
            cumulative,
            queries[i],
            depth,
            grad_value[i],
            grad_spatial[i],
            grad_coefficients,
            grad_cumulative,
        )
        wp.atomic_add(grad_queries, i, gradient)

    @wp.kernel
    def to_cube_kernel(queries: wp.array[vec3], transforms: wp.array[mat44], out: wp.array[vec3]):
        i = wp.tid()
        t = transforms[1]
        linear = mat33(t[0, 0], t[0, 1], t[0, 2], t[1, 0], t[1, 1], t[1, 2], t[2, 0], t[2, 1], t[2, 2])
        out[i] = linear * queries[i] + vec3(t[0, 3], t[1, 3], t[2, 3])

    @wp.kernel
    def to_model_gradient_kernel(gradient: wp.array[vec3], transforms: wp.array[mat44], out: wp.array[vec3]):
        i = wp.tid()
        t = transforms[1]
        linear = mat33(t[0, 0], t[0, 1], t[0, 2], t[1, 0], t[1, 1], t[1, 2], t[2, 0], t[2, 1], t[2, 2])
        out[i] = wp.transpose(linear) * gradient[i]

    @wp.kernel
    def to_cube_backward_kernel(
        queries: wp.array[vec3],
        transforms: wp.array[mat44],
        seed: wp.array[vec3],
        grad_queries: wp.array[vec3],
        grad_transforms: wp.array[mat44],
    ):
        i = wp.tid()
        g = seed[i]
        if g[0] == dtype(0) and g[1] == dtype(0) and g[2] == dtype(0):
            return
        p, t = queries[i], transforms[1]
        linear = mat33(t[0, 0], t[0, 1], t[0, 2], t[1, 0], t[1, 1], t[1, 2], t[2, 0], t[2, 1], t[2, 2])
        if grad_queries.shape[0]:
            wp.atomic_add(grad_queries, i, wp.transpose(linear) * g)
        if grad_transforms.shape[0]:
            gt = mat44(dtype(0))
            for a in range(3):
                for b in range(3):
                    gt[a, b] = g[a] * p[b]
                gt[a, 3] = g[a]
            wp.atomic_add(grad_transforms, 1, gt)

    @wp.kernel
    def to_model_gradient_backward_kernel(
        gradient: wp.array[vec3],
        transforms: wp.array[mat44],
        seed: wp.array[vec3],
        grad_gradient: wp.array[vec3],
        grad_transforms: wp.array[mat44],
    ):
        i = wp.tid()
        g = seed[i]
        if g[0] == dtype(0) and g[1] == dtype(0) and g[2] == dtype(0):
            return
        v, t = gradient[i], transforms[1]
        linear = mat33(t[0, 0], t[0, 1], t[0, 2], t[1, 0], t[1, 1], t[1, 2], t[2, 0], t[2, 1], t[2, 2])
        wp.atomic_add(grad_gradient, i, linear * g)
        if grad_transforms.shape[0]:
            gt = mat44(dtype(0))
            for a in range(3):
                for b in range(3):
                    gt[a, b] = v[a] * g[b]
            wp.atomic_add(grad_transforms, 1, gt)

    @wp.kernel
    def center_kernel(values: wp.array[dtype], iso: wp.array[dtype], out: wp.array[dtype]):
        i = wp.tid()
        out[i] = values[i] - iso[0]

    @wp.kernel
    def center_backward_kernel(seed: wp.array[dtype], grad_values: wp.array[dtype], grad_iso: wp.array[dtype]):
        i = wp.tid()
        g = seed[i]
        if g != dtype(0):
            wp.atomic_add(grad_values, i, g)
            if grad_iso.shape[0]:
                wp.atomic_add(grad_iso, 0, -g)

    @wp.kernel
    def mass_partials_kernel(weights: wp.array[dtype], partials: wp.array[wp.float64]):
        block = wp.tid()
        total = wp.float64(0)
        for j in range(block * 256, wp.min((block + 1) * 256, weights.shape[0])):
            total += wp.float64(weights[j])
        partials[block] = total

    @wp.kernel
    def mass_finish_kernel(partials: wp.array[wp.float64], total: wp.array[wp.float64]):
        value = wp.float64(0)
        for i in range(partials.shape[0]):
            value += partials[i]
        total[0] = value

    @wp.kernel
    def iso_backward_kernel(
        values: wp.array[dtype],
        weights: wp.array[dtype],
        iso: wp.array[dtype],
        mass: wp.array[wp.float64],
        grad_iso: wp.array[dtype],
        grad_values: wp.array[dtype],
        grad_weights: wp.array[dtype],
    ):
        i = wp.tid()
        scale = dtype(wp.float64(grad_iso[0]) / mass[0])
        wp.atomic_add(grad_values, i, scale * weights[i])
        if grad_weights.shape[0]:
            wp.atomic_add(grad_weights, i, scale * (values[i] - iso[0]))

    return SimpleNamespace(
        cumulative_backward=cumulative_backward_kernel,
        evaluate_backward=evaluate_backward_kernel,
        to_cube=to_cube_kernel,
        to_model_gradient=to_model_gradient_kernel,
        center=center_kernel,
        to_cube_backward=to_cube_backward_kernel,
        to_model_gradient_backward=to_model_gradient_backward_kernel,
        center_backward=center_backward_kernel,
        mass_partials=mass_partials_kernel,
        mass_finish=mass_finish_kernel,
        iso_backward=iso_backward_kernel,
    )


def record_cumulative_vjp(keys, coefficients, cumulative, depth, *, tape):
    def backward():
        work = wp.clone(cumulative.grad, requires_grad=False)
        for d in range(depth, -1, -1):
            wp.launch(
                precision_kernels(coefficients.dtype).cumulative_backward,
                len(keys),
                inputs=[keys, work, coefficients.grad, d],
                device=keys.device,
                record_tape=False,
            )

    tape.record_func(backward, [coefficients, cumulative])


def record_isovalue_vjp(values, weights, iso, *, tape):
    kernels, device = precision_kernels(values.dtype), values.device
    partials = wp.empty((len(weights) + 255) // 256, dtype=wp.float64, device=device)
    mass = wp.empty(1, dtype=wp.float64, device=device)
    wp.launch(kernels.mass_partials, len(partials), inputs=[weights, partials], device=device, record_tape=False)
    wp.launch(kernels.mass_finish, 1, inputs=[partials, mass], device=device, record_tape=False)
    empty = wp.empty(0, dtype=weights.dtype, device=device)

    def backward():
        wp.launch(
            kernels.iso_backward,
            len(values),
            inputs=[
                values,
                weights,
                iso,
                mass,
                iso.grad,
                values.grad,
                weights.grad if weights.requires_grad else empty,
            ],
            device=device,
            record_tape=False,
        )

    tape.record_func(backward, [values, iso] + ([weights] if weights.requires_grad else []))


def cube_field(keys, coefficients, cumulative, depth):
    return LevelField(
        keys, coefficients, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), depth, 0.0, cumulative
    )


def center_values(values, iso, *, tape):
    """Center a tracked field without zero-seed atomics to the shared isovalue."""
    dtype, device = values.dtype, values.device
    kernels = precision_kernels(dtype)
    out = wp.empty_like(values, requires_grad=True)
    wp.launch(kernels.center, len(values), inputs=[values, iso, out], device=device, record_tape=False)
    empty = wp.empty(0, dtype=dtype, device=device)

    def backward():
        wp.launch(
            kernels.center_backward,
            len(values),
            inputs=[out.grad, values.grad, iso.grad if iso.requires_grad else empty],
            device=device,
            record_tape=False,
        )

    tape.record_func(backward, [values, out] + ([iso] if iso.requires_grad else []))
    return out


def evaluate_cube(field, queries, *, tape):
    dtype, device = field.coefficients.dtype, queries.device
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    matrix = wp.mat33d if dtype == wp.float64 else wp.mat33
    values = wp.empty(len(queries), dtype=dtype, device=device, requires_grad=tape is not None)
    gradients = wp.empty_like(queries, requires_grad=tape is not None)
    valid = wp.empty(len(queries), dtype=int, device=device)
    wp.launch(
        reference_evaluation_kernel(dtype),
        len(queries),
        inputs=[
            field.keys,
            field.coefficients,
            field.cumulative,
            queries,
            matrix(1, 0, 0, 0, 1, 0, 0, 0, 1),
            vector(0),
            field.depth,
            values,
            gradients,
            valid,
        ],
        device=device,
        record_tape=False,
    )

    def backward():
        wp.launch(
            precision_kernels(dtype).evaluate_backward,
            len(queries),
            inputs=[
                field.keys,
                field.coefficients,
                field.cumulative,
                queries,
                field.depth,
                values.grad,
                gradients.grad,
                field.coefficients.grad,
                field.cumulative.grad,
                queries.grad,
            ],
            device=device,
            record_tape=False,
        )

    if tape is not None:
        tape.record_func(backward, [field.coefficients, field.cumulative, queries, values, gradients])
    return values, gradients, valid


@dataclass
class DifferentiableField:
    cube: LevelField
    transforms: wp.array
    iso_value: wp.array
    tape: wp.Tape
    ready: object = None

    @property
    def coefficients(self):
        return self.cube.coefficients

    @property
    def cumulative(self):
        return self.cube.cumulative

    def device_field(self):
        """Return a zero-copy handle for queries recorded on this field's tape.

        Call on the query stream so it waits for reconstruction readiness.
        Record application kernels on the supplied reconstruction tape. The
        handle shares tracked coefficients and normalization transforms.
        """
        from .device_field import _device_field  # noqa: PLC0415

        if self.ready is not None:
            wp.get_stream(self.coefficients.device).wait_event(self.ready)
        return _device_field(self.cube, transforms=self.transforms)

    def evaluate(self, queries, *, centered=False, requires_grad=True):
        """Evaluate with device transforms; opt out of recording for fixed selection.

        ``requires_grad=False`` returns untracked values/spatial gradients and
        records no operations, even inside the reconstruction tape. This is for
        discrete classification, not for differentiable losses.
        """
        dtype, device = self.cube.coefficients.dtype, self.cube.coefficients.device
        vector = wp.vec3d if dtype == wp.float64 else wp.vec3
        if queries.dtype != vector or queries.device != device:
            raise ValueError("queries must match field precision and device")
        if self.ready is not None:
            wp.get_stream(device).wait_event(self.ready)
        kernels = precision_kernels(dtype)
        transforms = self.transforms
        local = wp.empty_like(queries, requires_grad=requires_grad)
        wp.launch(
            kernels.to_cube, len(queries), inputs=[queries, self.transforms, local], device=device, record_tape=False
        )
        if requires_grad:
            empty_vector = wp.empty(0, dtype=vector, device=device)
            empty_matrix = wp.empty(0, dtype=self.transforms.dtype, device=device)
            grad_transforms = self.transforms.grad if self.transforms.requires_grad else empty_matrix

            def query_backward():
                wp.launch(
                    kernels.to_cube_backward,
                    len(queries),
                    inputs=[
                        queries,
                        transforms,
                        local.grad,
                        queries.grad if queries.requires_grad else empty_vector,
                        grad_transforms,
                    ],
                    device=device,
                    record_tape=False,
                )

            self.tape.record_func(query_backward, [local] + [a for a in (queries, self.transforms) if a.requires_grad])
        values, gradients, valid = evaluate_cube(self.cube, local, tape=self.tape if requires_grad else None)
        model_gradients = wp.empty_like(gradients, requires_grad=requires_grad)
        wp.launch(
            kernels.to_model_gradient,
            len(queries),
            inputs=[gradients, self.transforms, model_gradients],
            device=device,
            record_tape=False,
        )
        if requires_grad:

            def gradient_backward():
                wp.launch(
                    kernels.to_model_gradient_backward,
                    len(queries),
                    inputs=[gradients, transforms, model_gradients.grad, gradients.grad, grad_transforms],
                    device=device,
                    record_tape=False,
                )

            self.tape.record_func(
                gradient_backward,
                [gradients, model_gradients] + ([self.transforms] if self.transforms.requires_grad else []),
            )
        if centered:
            if requires_grad:
                values = center_values(values, self.iso_value, tape=self.tape)
            else:
                centered_values = wp.empty_like(values, requires_grad=False)
                wp.launch(
                    kernels.center,
                    len(values),
                    inputs=[values, self.iso_value, centered_values],
                    device=device,
                    record_tape=False,
                )
                values = centered_values
        return values, model_gradients, valid
