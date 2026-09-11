# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp


@wp.func
def add(x: wp.float32, y: wp.float32) -> wp.float32:
    return wp.sub(x, y)


@wp.func
def sub(x: wp.float32, y: wp.float32) -> wp.float32:
    return wp.mul(x, y)


@wp.kernel(module="unique", enable_backward=False)
def overloaded_slot_augassign(values: wp.array[wp.vec3], rhs: wp.float32):
    values[0].x += rhs
    values[0][1] += rhs
    values[0].z -= rhs


@wp.kernel(module="unique")
def update_double_components_with_float_overloads(values: wp.array[wp.vec3d], rhs: wp.array[wp.float64]):
    values[0].x += rhs[0]
    values[0][1] += rhs[0]
    values[0].z -= rhs[0]


@wp.kernel(module="unique")
def overloaded_slot_add_backward(values: wp.array[wp.vec3], rhs: wp.array[wp.float32]):
    values[0].x += rhs[0]


@wp.kernel(module="unique")
def overloaded_slot_sub_backward(values: wp.array[wp.vec3], rhs: wp.array[wp.float32]):
    values[0][1] -= rhs[0]


@wp.func
def update_overloaded_component(values: wp.array[wp.vec3], rhs: wp.array[float]):
    values[0].x += rhs[0]


@wp.kernel(module="test_shared_slot_operators", enable_backward=False)
def update_shared_component_forward(values: wp.array[wp.vec3], rhs: wp.array[float]):
    update_overloaded_component(values, rhs)


@wp.kernel(module="test_shared_slot_operators")
def update_shared_component_backward(values: wp.array[wp.vec3], rhs: wp.array[float]):
    update_overloaded_component(values, rhs)
