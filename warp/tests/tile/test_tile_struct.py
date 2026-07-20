# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_M = wp.constant(8)
TILE_DIM = 64
TILE_ROWS = wp.constant(2)
TILE_COLS = wp.constant(4)


@wp.struct
class TileMapStruct:
    x: wp.float32
    y: wp.vec3


@wp.struct
class TileMapNestedStruct:
    payload: TileMapStruct
    weight: wp.float32


@wp.struct
class TileMapArrayPayload:
    data: wp.array[float]
    tag: wp.int32


@wp.struct
class TileMapIndexedArrayPayload:
    data: wp.indexedarray[float]
    tag: wp.int32


@wp.struct
class HalfFieldStruct:
    s: wp.float16
    v: wp.vec3h
    m: wp.mat22h


@wp.struct
class NonAtomicFieldStruct:
    flag: wp.bool
    small: wp.int8
    value: wp.float32


@wp.struct
class QuatTransformFieldStruct:
    q: wp.quat
    xform: wp.transform
    v: wp.vec3
    s: wp.float32


@wp.struct
class Float64FieldStruct:
    s: wp.float64
    v: wp.vec3d
    m: wp.mat22d


@wp.func
def tile_map_struct_scale(s: TileMapStruct) -> TileMapStruct:
    out = TileMapStruct()
    out.x = s.x + wp.float32(1.0)
    out.y = s.y * wp.float32(2.0)
    return out


@wp.func
def tile_map_struct_add(a: TileMapStruct, b: TileMapStruct) -> TileMapStruct:
    out = TileMapStruct()
    out.x = a.x + b.x
    out.y = a.y + b.y
    return out


@wp.func
def tile_map_struct_sum(s: TileMapStruct) -> float:
    return s.x + s.y[0] + s.y[1] + s.y[2]


@wp.func
def tile_map_nested_struct_scale(s: TileMapNestedStruct) -> TileMapNestedStruct:
    out = TileMapNestedStruct()
    out.payload = tile_map_struct_scale(s.payload)
    out.weight = s.weight + wp.float32(3.0)
    return out


@wp.func
def tile_map_nested_struct_sum(s: TileMapNestedStruct) -> float:
    return tile_map_struct_sum(s.payload) + s.weight


@wp.kernel
def tile_map_custom_struct_kernel(input: wp.array[TileMapStruct], output: wp.array[TileMapStruct]):
    i = wp.tid()
    a = wp.tile_load(input, shape=TILE_M, offset=i * TILE_M)
    b = wp.tile_map(tile_map_struct_scale, a)

    bias = TileMapStruct()
    bias.x = wp.float32(10.0)
    bias.y = wp.vec3(1.0, 2.0, 3.0)
    c = wp.tile_map(tile_map_struct_add, b, bias)

    wp.tile_store(output, c, offset=i * TILE_M)


@wp.kernel
def tile_map_custom_struct_grad_kernel(input: wp.array[TileMapStruct], loss: wp.array[float]):
    i = wp.tid()
    a = wp.tile_load(input, shape=TILE_M, offset=i * TILE_M)
    b = wp.tile_map(tile_map_struct_scale, a)

    bias = TileMapStruct()
    bias.x = wp.float32(10.0)
    bias.y = wp.vec3(1.0, 2.0, 3.0)
    c = wp.tile_map(tile_map_struct_add, b, bias)

    values = wp.tile_map(tile_map_struct_sum, c)
    total = wp.tile_sum(values)
    wp.tile_store(loss, total, offset=i)


@wp.kernel
def tile_nested_struct_ops_kernel(
    input: wp.array[TileMapNestedStruct],
    sort_keys: wp.array[int],
    mapped_out: wp.array[TileMapNestedStruct],
    sum_out: wp.array[TileMapNestedStruct],
    atomic_out: wp.array[TileMapNestedStruct],
    sort_out: wp.array[TileMapNestedStruct],
):
    t = wp.tile_load(input, shape=TILE_M, storage="shared")

    mapped = wp.tile_map(tile_map_nested_struct_scale, t)
    wp.tile_store(mapped_out, mapped)
    wp.tile_store(sum_out, wp.tile_sum(t))
    wp.tile_atomic_add(atomic_out, t)

    keys = wp.tile_load(sort_keys, shape=TILE_M, storage="shared")
    values = wp.tile_load(input, shape=TILE_M, storage="shared")
    wp.tile_sort(keys, values)
    wp.tile_store(sort_out, values)


@wp.kernel
def tile_nested_struct_grad_kernel(input: wp.array[TileMapNestedStruct], loss: wp.array[float]):
    t = wp.tile_load(input, shape=TILE_M, storage="shared")
    mapped = wp.tile_map(tile_map_nested_struct_scale, t)
    values = wp.tile_map(tile_map_nested_struct_sum, mapped)
    wp.tile_store(loss, wp.tile_sum(values))


@wp.func
def tile_map_nested_field_store(s: TileMapNestedStruct) -> TileMapNestedStruct:
    # Direct nested struct field stores (out.payload.x = ...) rather than
    # whole-struct assignment (out.payload = ...). Exercises the reverse pass
    # for stores into a field of a nested struct field.
    out = TileMapNestedStruct()
    out.payload.x = s.payload.x * wp.float32(2.0)
    out.payload.y = s.payload.y * wp.float32(3.0)
    out.weight = s.weight + wp.float32(1.0)
    return out


@wp.kernel
def tile_nested_field_store_grad_kernel(input: wp.array[TileMapNestedStruct], loss: wp.array[float]):
    t = wp.tile_load(input, shape=TILE_M, storage="shared")
    mapped = wp.tile_map(tile_map_nested_field_store, t)
    values = wp.tile_map(tile_map_nested_struct_sum, mapped)
    wp.tile_store(loss, wp.tile_sum(values))


@wp.kernel
def tile_struct_array_payload_sort_kernel(
    input: wp.array[TileMapArrayPayload],
    sort_keys: wp.array[int],
    sorted_payloads: wp.array[TileMapArrayPayload],
):
    keys = wp.tile_load(sort_keys, shape=TILE_M, storage="shared")
    values = wp.tile_load(input, shape=TILE_M, storage="shared")
    wp.tile_sort(keys, values)
    wp.tile_store(sorted_payloads, values)


@wp.kernel
def tile_struct_array_payload_read_kernel(
    payloads: wp.array[TileMapArrayPayload],
    values_out: wp.array[float],
    tags_out: wp.array[int],
):
    i = wp.tid()
    value = payloads[i]
    values_out[i] = value.data[0]
    tags_out[i] = value.tag


@wp.kernel
def tile_struct_indexed_array_payload_sort_kernel(
    input: wp.array[TileMapIndexedArrayPayload],
    sort_keys: wp.array[int],
    sorted_payloads: wp.array[TileMapIndexedArrayPayload],
):
    keys = wp.tile_load(sort_keys, shape=TILE_M, storage="shared")
    values = wp.tile_load(input, shape=TILE_M, storage="shared")
    wp.tile_sort(keys, values)
    wp.tile_store(sorted_payloads, values)


@wp.kernel
def tile_struct_indexed_array_payload_read_kernel(
    payloads: wp.array[TileMapIndexedArrayPayload],
    values_out: wp.array[float],
    tags_out: wp.array[int],
):
    i = wp.tid()
    value = payloads[i]
    values_out[i] = value.data[0]
    tags_out[i] = value.tag


@wp.kernel
def tile_struct_value_ops_kernel(
    input: wp.array[TileMapStruct],
    load_indices: wp.array[int],
    store_indices: wp.array[int],
    sort_keys: wp.array[int],
    indexed_out: wp.array[TileMapStruct],
    indexed_store_out: wp.array[TileMapStruct],
    add_out: wp.array[TileMapStruct],
    sub_out: wp.array[TileMapStruct],
    sum_out: wp.array[TileMapStruct],
    transpose_out: wp.array2d[TileMapStruct],
    extract_out: wp.array[TileMapStruct],
    atomic_out: wp.array[TileMapStruct],
    sort_out: wp.array[TileMapStruct],
    tile_inplace_out: wp.array[TileMapStruct],
    tile_sub_inplace_out: wp.array[TileMapStruct],
    element_inplace_out: wp.array[TileMapStruct],
    element_sub_inplace_out: wp.array[TileMapStruct],
):
    _tile, lane = wp.tid()

    t = wp.tile_load(input, shape=TILE_M, storage="shared")

    load_idx = wp.tile_load(load_indices, shape=TILE_M, storage="shared")
    indexed = wp.tile_load_indexed(input, indices=load_idx, shape=(TILE_M,), axis=0)
    wp.tile_store(indexed_out, indexed)

    store_idx = wp.tile_load(store_indices, shape=TILE_M, storage="shared")
    wp.tile_store_indexed(indexed_store_out, indices=store_idx, t=t, axis=0)

    wp.tile_store(add_out, t + t)
    # subtract distinct operands so the check fails if subtraction returns zero or the lhs
    wp.tile_store(sub_out, (t + t) - t)
    wp.tile_store(sum_out, wp.tile_sum(t))

    reshaped = wp.tile_reshape(t, shape=(2, 4))
    wp.tile_store(transpose_out, wp.tile_transpose(reshaped))

    extracted = wp.tile_extract(t, 3)
    if lane == 0:
        extract_out[0] = extracted

    wp.tile_atomic_add(atomic_out, t)

    keys = wp.tile_load(sort_keys, shape=TILE_M, storage="shared")
    values = wp.tile_load(input, shape=TILE_M, storage="shared")
    wp.tile_sort(keys, values)
    wp.tile_store(sort_out, values)

    tile_inplace = wp.tile_load(input, shape=TILE_M, storage="register")
    tile_inplace += t
    wp.tile_store(tile_inplace_out, tile_inplace)

    tile_sub_inplace = t + t
    tile_sub_inplace -= t
    wp.tile_store(tile_sub_inplace_out, tile_sub_inplace)

    value = TileMapStruct()
    if lane == 0:
        value = input[0]

    element_inplace = wp.tile_zeros(shape=1, dtype=TileMapStruct, storage="shared")
    element_inplace[0] += value
    element_total = wp.tile_extract(element_inplace, 0)
    if lane == 0:
        element_inplace_out[0] = element_total

    element_sub_inplace = wp.tile_zeros(shape=1, dtype=TileMapStruct, storage="shared")
    element_sub_inplace[0] -= value
    element_sub_total = wp.tile_extract(element_sub_inplace, 0)
    if lane == 0:
        element_sub_inplace_out[0] = element_sub_total


@wp.kernel
def tile_struct_half_fields_kernel(
    input: wp.array[HalfFieldStruct],
    sort_keys: wp.array[int],
    store_out: wp.array[HalfFieldStruct],
    sum_out: wp.array[HalfFieldStruct],
    sort_out: wp.array[HalfFieldStruct],
):
    t = wp.tile_load(input, shape=TILE_M)
    # Plain load/store already forces NVRTC to instantiate the struct shuffle
    # helpers, which failed to compile for half-precision fields before the fix.
    wp.tile_store(store_out, t)

    # tile_sum reduces across lanes via warp_shuffle_down on the half fields.
    wp.tile_store(sum_out, wp.tile_sum(t))

    # tile_sort permutes the struct value payload via warp_shuffle_xor.
    keys = wp.tile_load(sort_keys, shape=TILE_M)
    values = wp.tile_load(input, shape=TILE_M)
    wp.tile_sort(keys, values)
    wp.tile_store(sort_out, values)


@wp.kernel
def tile_struct_quat_transform_fields_kernel(
    input: wp.array[QuatTransformFieldStruct],
    sort_keys: wp.array[int],
    store_out: wp.array[QuatTransformFieldStruct],
    sum_out: wp.array[QuatTransformFieldStruct],
    sort_out: wp.array[QuatTransformFieldStruct],
):
    t = wp.tile_load(input, shape=TILE_M)
    # Plain load/store forces NVRTC to instantiate the struct shuffle helpers. The
    # quaternion and transform fields are value types, so they fall through to the
    # generic warp_shuffle_down / warp_shuffle_xor templates, which failed to compile
    # before the fix (deleted union default constructor / no __shfl_xor_sync overload).
    wp.tile_store(store_out, t)

    # tile_sum reduces across lanes via warp_shuffle_down on each value field.
    wp.tile_store(sum_out, wp.tile_sum(t))

    # tile_sort permutes the struct value payload via warp_shuffle_xor.
    keys = wp.tile_load(sort_keys, shape=TILE_M)
    values = wp.tile_load(input, shape=TILE_M)
    wp.tile_sort(keys, values)
    wp.tile_store(sort_out, values)


@wp.kernel
def tile_struct_float64_fields_kernel(
    input: wp.array[Float64FieldStruct],
    sort_keys: wp.array[int],
    store_out: wp.array[Float64FieldStruct],
    sum_out: wp.array[Float64FieldStruct],
    sort_out: wp.array[Float64FieldStruct],
):
    t = wp.tile_load(input, shape=TILE_M)
    # Plain load/store forces NVRTC to instantiate the struct shuffle helpers. The
    # float64 fields fall through to the generic warp_shuffle_down / warp_shuffle_xor
    # templates, whose word buffers are only 4-byte aligned; reinterpreting them as a
    # T* is undefined for the 8-byte-aligned float64 unless the buffers are aligned for T.
    wp.tile_store(store_out, t)

    # tile_sum reduces across lanes via warp_shuffle_down on each float64 field.
    wp.tile_store(sum_out, wp.tile_sum(t))

    # tile_sort permutes the struct value payload via warp_shuffle_xor.
    keys = wp.tile_load(sort_keys, shape=TILE_M)
    values = wp.tile_load(input, shape=TILE_M)
    wp.tile_sort(keys, values)
    wp.tile_store(sort_out, values)


@wp.kernel
def tile_half_scalar_kernel(input: wp.array[wp.float16], output: wp.array[wp.float16]):
    # Control: non-struct half tile load/store, which already compiled and ran.
    t = wp.tile_load(input, shape=TILE_M)
    wp.tile_store(output, t)


@wp.kernel
def tile_struct_non_atomic_fields_kernel(
    input: wp.array[NonAtomicFieldStruct],
    store_out: wp.array[NonAtomicFieldStruct],
    atomic_out: wp.array[NonAtomicFieldStruct],
):
    t = wp.tile_load(input, shape=TILE_M)
    # Load/store forces NVRTC to instantiate the struct atomic_add helper, which
    # unconditionally emitted a field atomic for the bool / int8 fields before the
    # fix, and CUDA has no atomicAdd overload for those scalar types.
    wp.tile_store(store_out, t)

    # tile_atomic_add accumulates the float field; the bool / int8 fields ride along
    # with the struct value but are not accumulated.
    wp.tile_atomic_add(atomic_out, t)


@wp.kernel
def tile_struct_non_atomic_accumulate_kernel(
    input: wp.array[NonAtomicFieldStruct],
    sum_out: wp.array[NonAtomicFieldStruct],
    add_out: wp.array[NonAtomicFieldStruct],
):
    t = wp.tile_load(input, shape=TILE_M)
    # Field-wise reduction and arithmetic accumulate the float field and carry the
    # bool / int8 fields unchanged, consistent with tile_atomic_add.
    wp.tile_store(sum_out, wp.tile_sum(t))
    wp.tile_store(add_out, t + t)


@wp.kernel
def tile_struct_scatter_stack_kernel(
    input: wp.array[TileMapStruct],
    scatter_atomic_out: wp.array[TileMapStruct],
    scatter_non_atomic_out: wp.array[TileMapStruct],
    stack_out: wp.array[TileMapStruct],
):
    _tile, lane = wp.tid()
    value = input[lane]

    atomic_tile = wp.tile_zeros(shape=1, dtype=TileMapStruct, storage="shared")
    wp.tile_scatter_add(atomic_tile, 0, value, lane == 0)
    atomic_total = wp.tile_extract(atomic_tile, 0)
    if lane == 0:
        scatter_atomic_out[0] = atomic_total

    non_atomic_tile = wp.tile_zeros(shape=TILE_M, dtype=TileMapStruct, storage="shared")
    wp.tile_scatter_add(non_atomic_tile, lane, value, True, atomic=False)
    scatter_non_atomic_out[lane] = wp.tile_extract(non_atomic_tile, lane)

    stack = wp.tile_stack(capacity=8, dtype=TileMapStruct)
    wp.tile_stack_push(stack, value, lane == 0)
    popped, slot = wp.tile_stack_pop(stack)
    if slot != -1:
        stack_out[0] = popped


@wp.kernel
def tile_struct_constructor_assign_stack_kernel(
    input: wp.array[TileMapStruct],
    from_thread_out: wp.array[TileMapStruct],
    assign_out: wp.array[TileMapStruct],
    element_assign_out: wp.array[TileMapStruct],
    stack_counts: wp.array[int],
):
    _tile, lane = wp.tid()

    fill = TileMapStruct()
    fill.x = wp.float32(4.0)
    fill.y = wp.vec3(5.0, 6.0, 7.0)

    thread_value = input[lane]
    from_thread = wp.tile_from_thread(shape=TILE_M, value=thread_value, thread_idx=3, storage="shared")
    wp.tile_store(from_thread_out, from_thread)

    src = wp.tile_full(TILE_M, value=fill, dtype=TileMapStruct, storage="register")
    dst = wp.tile_zeros(TILE_M, dtype=TileMapStruct, storage="shared")
    wp.tile_assign(dst, src)
    wp.tile_store(assign_out, dst)

    element_dst = wp.tile_zeros(TILE_M, dtype=TileMapStruct, storage="shared")
    element_dst[0] = input[0]
    wp.tile_store(element_assign_out, element_dst)

    stack = wp.tile_stack(capacity=8, dtype=TileMapStruct)
    wp.tile_stack_push(stack, input[0], lane == 0)
    if lane == 0:
        stack_counts[0] = wp.tile_stack_count(stack)

    wp.tile_stack_clear(stack)
    if lane == 0:
        stack_counts[1] = wp.tile_stack_count(stack)


@wp.kernel
def tile_struct_tile_assign_grad_kernel(input: wp.array[TileMapStruct], loss: wp.array[float]):
    src = wp.tile_load(input, shape=TILE_M, storage="register")
    dst = wp.tile_zeros(TILE_M, dtype=TileMapStruct, storage="shared")
    wp.tile_assign(dst, src)

    values = wp.tile_map(tile_map_struct_sum, dst)
    wp.tile_store(loss, wp.tile_sum(values))


@wp.kernel
def tile_struct_add_grad_kernel(input: wp.array[TileMapStruct], loss: wp.array[float]):
    src = wp.tile_load(input, shape=TILE_M, storage="register")
    added = src + src

    values = wp.tile_map(tile_map_struct_sum, added)
    wp.tile_store(loss, wp.tile_sum(values))


@wp.kernel
def tile_struct_tile_add_inplace_grad_kernel(input: wp.array[TileMapStruct], loss: wp.array[float]):
    src = wp.tile_load(input, shape=TILE_M, storage="register")
    dst = wp.tile_zeros(TILE_M, dtype=TileMapStruct, storage="shared")
    dst += src

    values = wp.tile_map(tile_map_struct_sum, dst)
    wp.tile_store(loss, wp.tile_sum(values))


@wp.kernel
def tile_struct_sub_grad_kernel(input: wp.array[TileMapStruct], loss: wp.array[float]):
    src = wp.tile_load(input, shape=TILE_M, storage="register")
    zero = wp.tile_zeros(TILE_M, dtype=TileMapStruct, storage="register")
    diff = zero - src

    values = wp.tile_map(tile_map_struct_sum, diff)
    wp.tile_store(loss, wp.tile_sum(values))


@wp.kernel
def tile_struct_element_assign_grad_kernel(input: wp.array[TileMapStruct], loss: wp.array[float]):
    _i, j = wp.tid()
    t = wp.tile_zeros(TILE_M, dtype=TileMapStruct, storage="shared")
    t[j] = input[j]

    values = wp.tile_map(tile_map_struct_sum, t)
    wp.tile_store(loss, wp.tile_sum(values))


@wp.kernel
def tile_struct_sum_axis_grad_kernel(input: wp.array[TileMapStruct], loss: wp.array[float]):
    t = wp.tile_load(input, shape=TILE_M, storage="shared")
    reshaped = wp.tile_reshape(t, shape=(TILE_ROWS, TILE_COLS))
    row_sums = wp.tile_sum(reshaped, axis=1)
    values = wp.tile_map(tile_map_struct_sum, row_sums)
    wp.tile_store(loss, wp.tile_sum(values))


@wp.kernel
def tile_struct_atomic_add_grad_kernel(input: wp.array[TileMapStruct], output: wp.array[TileMapStruct]):
    t = wp.tile_load(input, shape=TILE_M, storage="register")
    wp.tile_atomic_add(output, t)


@wp.kernel
def tile_struct_scatter_add_grad_kernel(input: wp.array[TileMapStruct], output: wp.array[TileMapStruct]):
    _tile, j = wp.tid()
    t = wp.tile_zeros(TILE_M, dtype=TileMapStruct, storage="shared")
    wp.tile_scatter_add(t, j, input[j], True)
    output[j] = wp.tile_extract(t, j)


@wp.kernel
def tile_struct_scatter_add_collision_grad_kernel(input: wp.array[TileMapStruct], output: wp.array[TileMapStruct]):
    _tile, j = wp.tid()
    t = wp.tile_zeros(1, dtype=TileMapStruct, storage="shared")
    wp.tile_scatter_add(t, 0, input[j], True)
    wp.tile_store(output, t)


@wp.kernel
def tile_struct_store_indexed_grad_kernel(
    input: wp.array[TileMapStruct],
    indices: wp.array[int],
    output: wp.array[TileMapStruct],
):
    idx = wp.tile_load(indices, shape=TILE_M, storage="shared")
    t = wp.tile_load(input, shape=TILE_M, storage="register")
    wp.tile_store_indexed(output, indices=idx, t=t, axis=0)


@wp.kernel
def tile_struct_atomic_add_indexed_kernel(
    input: wp.array[TileMapStruct],
    indices: wp.array[int],
    output: wp.array[TileMapStruct],
):
    idx = wp.tile_load(indices, shape=TILE_M, storage="shared")
    t = wp.tile_load(input, shape=TILE_M, storage="register")
    wp.tile_atomic_add_indexed(output, indices=idx, t=t, axis=0)


@wp.kernel
def tile_struct_load_indexed_grad_kernel(
    input: wp.array[TileMapStruct],
    indices: wp.array[int],
    loss: wp.array[float],
):
    idx = wp.tile_load(indices, shape=TILE_M, storage="shared")
    indexed = wp.tile_load_indexed(input, indices=idx, shape=(TILE_M,), axis=0)
    indexed_values = wp.tile_map(tile_map_struct_sum, indexed)
    wp.tile_store(loss, wp.tile_sum(indexed_values))


@wp.kernel
def tile_struct_extract_grad_kernel(input: wp.array[TileMapStruct], loss: wp.array[float]):
    _tile, lane = wp.tid()
    shared = wp.tile_load(input, shape=TILE_M, storage="shared")
    extracted = wp.tile_extract(shared, 3)
    if lane == 0:
        loss[0] = tile_map_struct_sum(extracted)


@wp.kernel
def tile_struct_broadcast_grad_kernel(input: wp.array[TileMapStruct], loss: wp.array[float]):
    shared = wp.tile_load(input, shape=TILE_M, storage="shared")
    first = wp.tile_view(shared, offset=(0,), shape=(1,))
    broadcasted = wp.tile_broadcast(first, shape=(TILE_M,))
    broadcast_values = wp.tile_map(tile_map_struct_sum, broadcasted)
    wp.tile_store(loss, wp.tile_sum(broadcast_values))


@wp.kernel
def tile_struct_weighted_sum_grad_kernel(
    input: wp.array[TileMapStruct],
    weights: wp.array[float],
    loss: wp.array[float],
):
    t = wp.tile_load(input, shape=TILE_M, storage="shared")
    w = wp.tile_load(weights, shape=TILE_M, storage="shared")
    values = wp.tile_map(tile_map_struct_sum, t)
    wp.tile_store(loss, wp.tile_sum(values * w))


@wp.kernel
def tile_struct_single_sum_grad_kernel(input: wp.array[TileMapStruct], loss: wp.array[float]):
    loss[0] = tile_map_struct_sum(input[0])


def make_tile_map_struct_data():
    data = []
    for i in range(TILE_M):
        s = TileMapStruct()
        s.x = float(i + 1)
        s.y = wp.vec3(float(i + 1), float(i + 2), float(i + 3))
        data.append(s)
    return data


def make_tile_map_struct_expected_values():
    x = np.arange(1, TILE_M + 1, dtype=np.float32)
    y = np.stack((x, x + 1.0, x + 2.0), axis=1)
    value_sums = x + np.sum(y, axis=1)
    transformed_value_sums = x + 11.0 + 2.0 * np.sum(y, axis=1) + 6.0
    expected_unit_grad = np.ones(TILE_M, dtype=np.float32)
    expected_transformed_y_grad = np.full((TILE_M, 3), 2.0, dtype=np.float32)
    return x, y, value_sums, transformed_value_sums, expected_unit_grad, expected_transformed_y_grad


def make_tile_map_nested_struct_data():
    data = []
    for i in range(TILE_M):
        payload = TileMapStruct()
        payload.x = float(i + 1)
        payload.y = wp.vec3(float(i + 1), float(i + 2), float(i + 3))

        s = TileMapNestedStruct()
        s.payload = payload
        s.weight = float(10 + i)
        data.append(s)

    return data


def assert_tile_map_struct_array(actual, expected_x, expected_y):
    assert_np_equal(actual["x"], expected_x)
    assert_np_equal(actual["y"], expected_y)


def assert_tile_map_nested_struct_array(actual, expected_x, expected_y, expected_weight):
    assert_np_equal(actual["payload"]["x"], expected_x)
    assert_np_equal(actual["payload"]["y"], expected_y)
    assert_np_equal(actual["weight"], expected_weight)


def assert_tile_map_struct_grad_components(actual, expected_x, expected_y):
    assert_np_equal(actual["x"], expected_x)
    assert_np_equal(actual["y"], expected_y)


def assert_tile_map_struct_grad(actual, expected):
    assert_tile_map_struct_grad_components(actual, expected, np.repeat(expected[:, None], 3, axis=1))


def test_tile_map_custom_struct(test, device):
    # tile_map over structs (unary scale + binary with a struct constant) and its adjoint
    data = []
    for i in range(TILE_M):
        s = TileMapStruct()
        s.x = float(i)
        s.y = wp.vec3(float(i), float(i + 1), float(i + 2))
        data.append(s)

    input_wp = wp.array(data, dtype=TileMapStruct, device=device)
    output_wp = wp.empty(TILE_M, dtype=TileMapStruct, device=device)

    wp.launch_tiled(
        tile_map_custom_struct_kernel,
        dim=[1],
        inputs=[input_wp],
        outputs=[output_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    output_np = output_wp.numpy()
    expected_i = np.arange(TILE_M, dtype=np.float32)
    expected_y = np.stack((2.0 * expected_i + 1.0, 2.0 * expected_i + 4.0, 2.0 * expected_i + 7.0), axis=1)
    assert_np_equal(output_np["x"], expected_i + 11.0)
    assert_np_equal(output_np["y"], expected_y)

    grad_input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    loss_wp = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_map_custom_struct_grad_kernel,
            dim=[1],
            inputs=[grad_input_wp],
            outputs=[loss_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    expected_loss = np.sum(expected_i + 11.0 + np.sum(expected_y, axis=1))
    test.assertAlmostEqual(loss_wp.numpy()[0], expected_loss, places=5)

    tape.backward(loss_wp)
    input_grad_np = grad_input_wp.grad.numpy()
    assert_np_equal(input_grad_np["x"], np.ones(TILE_M, dtype=np.float32))
    assert_np_equal(input_grad_np["y"], np.full((TILE_M, 3), 2.0, dtype=np.float32))


def test_tile_nested_struct_ops(test, device):
    # nested-struct recursion through map / sum / atomic_add / sort, forward + adjoint
    data = make_tile_map_nested_struct_data()
    input_wp = wp.array(data, dtype=TileMapNestedStruct, requires_grad=True, device=device)
    sort_keys_np = np.arange(TILE_M - 1, -1, -1, dtype=np.int32)
    sort_keys = wp.array(sort_keys_np, dtype=int, device=device)

    mapped_out = wp.empty(TILE_M, dtype=TileMapNestedStruct, device=device)
    sum_out = wp.empty(1, dtype=TileMapNestedStruct, device=device)
    atomic_out = wp.zeros(TILE_M, dtype=TileMapNestedStruct, device=device)
    sort_out = wp.empty(TILE_M, dtype=TileMapNestedStruct, device=device)

    wp.launch_tiled(
        tile_nested_struct_ops_kernel,
        dim=[1],
        inputs=[input_wp, sort_keys],
        outputs=[mapped_out, sum_out, atomic_out, sort_out],
        block_dim=32,
        device=device,
    )

    x = np.arange(1, TILE_M + 1, dtype=np.float32)
    y = np.stack((x, x + 1.0, x + 2.0), axis=1)
    weight = np.arange(10, 10 + TILE_M, dtype=np.float32)

    assert_tile_map_nested_struct_array(mapped_out.numpy(), x + 1.0, 2.0 * y, weight + 3.0)
    assert_tile_map_nested_struct_array(
        sum_out.numpy(),
        np.array([np.sum(x)], dtype=np.float32),
        np.sum(y, axis=0)[None, :],
        np.array([np.sum(weight)], dtype=np.float32),
    )
    assert_tile_map_nested_struct_array(atomic_out.numpy(), x, y, weight)
    assert_tile_map_nested_struct_array(sort_out.numpy(), x[::-1], y[::-1], weight[::-1])

    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_nested_struct_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    expected_loss = np.sum(x + 1.0 + np.sum(2.0 * y, axis=1) + weight + 3.0)
    assert_np_equal(loss.numpy(), np.array([expected_loss], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()

    input_grad = input_wp.grad.numpy()
    assert_np_equal(input_grad["payload"]["x"], np.ones(TILE_M, dtype=np.float32))
    assert_np_equal(input_grad["payload"]["y"], np.full((TILE_M, 3), 2.0, dtype=np.float32))
    assert_np_equal(input_grad["weight"], np.ones(TILE_M, dtype=np.float32))


def test_tile_nested_field_store_grad(test, device):
    """Verify gradients through direct nested struct field stores (``out.payload.x = ...``).

    This covers the field-store path as opposed to whole-struct assignment, which
    ``test_tile_nested_struct_ops`` covers.
    """
    data = make_tile_map_nested_struct_data()
    input_wp = wp.array(data, dtype=TileMapNestedStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_nested_field_store_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    x = np.arange(1, TILE_M + 1, dtype=np.float32)
    y = np.stack((x, x + 1.0, x + 2.0), axis=1)
    weight = np.arange(10, 10 + TILE_M, dtype=np.float32)

    # loss = sum(2*x + 3*(y0 + y1 + y2) + (weight + 1))
    expected_loss = np.sum(2.0 * x + 3.0 * np.sum(y, axis=1) + weight + 1.0)
    assert_np_equal(loss.numpy(), np.array([expected_loss], dtype=np.float32))

    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()

    input_grad = input_wp.grad.numpy()
    assert_np_equal(input_grad["payload"]["x"], np.full(TILE_M, 2.0, dtype=np.float32))
    assert_np_equal(input_grad["payload"]["y"], np.full((TILE_M, 3), 3.0, dtype=np.float32))
    assert_np_equal(input_grad["weight"], np.ones(TILE_M, dtype=np.float32))


def test_tile_struct_array_payload_sort(test, device):
    # array_t descriptor field carried through tile_sort (exercises the array_t warp-shuffle overload)
    arrays = [wp.array([100.0 + float(i)], dtype=float, device=device) for i in range(TILE_M)]
    payloads = []
    for i, array in enumerate(arrays):
        payload = TileMapArrayPayload()
        payload.data = array
        payload.tag = i
        payloads.append(payload)

    input_wp = wp.array(payloads, dtype=TileMapArrayPayload, device=device)
    sort_keys_np = np.arange(TILE_M - 1, -1, -1, dtype=np.int32)
    sort_keys = wp.array(sort_keys_np, dtype=int, device=device)
    sorted_payloads = wp.empty(TILE_M, dtype=TileMapArrayPayload, device=device)
    values_out = wp.empty(TILE_M, dtype=float, device=device)
    tags_out = wp.empty(TILE_M, dtype=int, device=device)

    wp.launch_tiled(
        tile_struct_array_payload_sort_kernel,
        dim=[1],
        inputs=[input_wp, sort_keys],
        outputs=[sorted_payloads],
        block_dim=32,
        device=device,
    )
    wp.launch(
        tile_struct_array_payload_read_kernel,
        dim=TILE_M,
        inputs=[sorted_payloads],
        outputs=[values_out, tags_out],
        device=device,
    )

    assert_np_equal(values_out.numpy(), np.arange(100.0 + TILE_M - 1, 99.0, -1.0, dtype=np.float32))
    assert_np_equal(tags_out.numpy(), np.arange(TILE_M - 1, -1, -1, dtype=np.int32))


def test_tile_struct_indexed_array_payload_sort(test, device):
    """Sort a tile payload of structs with an indexed-array field.

    Exercises the indexedarray_t warp-shuffle overload when the struct is permuted as a tile
    sort payload.
    """
    arrays = [wp.array([100.0 + float(i)], dtype=float, device=device) for i in range(TILE_M)]
    index_arrays = [wp.array([0], dtype=wp.int32, device=device) for _ in range(TILE_M)]
    payloads = []
    for i in range(TILE_M):
        payload = TileMapIndexedArrayPayload()
        payload.data = wp.indexedarray1d(arrays[i], [index_arrays[i]])
        payload.tag = i
        payloads.append(payload)

    input_wp = wp.array(payloads, dtype=TileMapIndexedArrayPayload, device=device)
    sort_keys_np = np.arange(TILE_M - 1, -1, -1, dtype=np.int32)
    sort_keys = wp.array(sort_keys_np, dtype=int, device=device)
    sorted_payloads = wp.empty(TILE_M, dtype=TileMapIndexedArrayPayload, device=device)
    values_out = wp.empty(TILE_M, dtype=float, device=device)
    tags_out = wp.empty(TILE_M, dtype=int, device=device)

    wp.launch_tiled(
        tile_struct_indexed_array_payload_sort_kernel,
        dim=[1],
        inputs=[input_wp, sort_keys],
        outputs=[sorted_payloads],
        block_dim=32,
        device=device,
    )
    wp.launch(
        tile_struct_indexed_array_payload_read_kernel,
        dim=TILE_M,
        inputs=[sorted_payloads],
        outputs=[values_out, tags_out],
        device=device,
    )

    assert_np_equal(values_out.numpy(), np.arange(100.0 + TILE_M - 1, 99.0, -1.0, dtype=np.float32))
    assert_np_equal(tags_out.numpy(), np.arange(TILE_M - 1, -1, -1, dtype=np.int32))


def test_tile_struct_value_ops(test, device):
    """Exercise the forward struct tile surface.

    Covers indexed load/store, field-wise add/sub/sum, transpose, extract, ``atomic_add``, sort, tile
    and element ``+=`` / ``-=``, plus ``scatter_add`` and stack push/pop.
    """
    data = make_tile_map_struct_data()
    input_wp = wp.array(data, dtype=TileMapStruct, device=device)

    load_indices_np = np.arange(TILE_M - 1, -1, -1, dtype=np.int32)
    store_indices_np = np.arange(TILE_M, dtype=np.int32) * 2
    sort_keys_np = np.arange(TILE_M - 1, -1, -1, dtype=np.int32)

    load_indices = wp.array(load_indices_np, dtype=int, device=device)
    store_indices = wp.array(store_indices_np, dtype=int, device=device)
    sort_keys = wp.array(sort_keys_np, dtype=int, device=device)

    indexed_out = wp.empty(TILE_M, dtype=TileMapStruct, device=device)
    indexed_store_out = wp.zeros(TILE_M * 2, dtype=TileMapStruct, device=device)
    add_out = wp.empty(TILE_M, dtype=TileMapStruct, device=device)
    sub_out = wp.empty(TILE_M, dtype=TileMapStruct, device=device)
    sum_out = wp.empty(1, dtype=TileMapStruct, device=device)
    transpose_out = wp.empty((4, 2), dtype=TileMapStruct, device=device)
    extract_out = wp.empty(1, dtype=TileMapStruct, device=device)
    atomic_out = wp.zeros(TILE_M, dtype=TileMapStruct, device=device)
    sort_out = wp.empty(TILE_M, dtype=TileMapStruct, device=device)
    tile_inplace_out = wp.empty(TILE_M, dtype=TileMapStruct, device=device)
    tile_sub_inplace_out = wp.empty(TILE_M, dtype=TileMapStruct, device=device)
    element_inplace_out = wp.empty(1, dtype=TileMapStruct, device=device)
    element_sub_inplace_out = wp.empty(1, dtype=TileMapStruct, device=device)

    wp.launch_tiled(
        tile_struct_value_ops_kernel,
        dim=[1],
        inputs=[input_wp, load_indices, store_indices, sort_keys],
        outputs=[
            indexed_out,
            indexed_store_out,
            add_out,
            sub_out,
            sum_out,
            transpose_out,
            extract_out,
            atomic_out,
            sort_out,
            tile_inplace_out,
            tile_sub_inplace_out,
            element_inplace_out,
            element_sub_inplace_out,
        ],
        block_dim=32,
        device=device,
    )

    x = np.arange(1, TILE_M + 1, dtype=np.float32)
    y = np.stack((x, x + 1.0, x + 2.0), axis=1)

    assert_tile_map_struct_array(indexed_out.numpy(), x[::-1], y[::-1])

    expected_store_x = np.zeros(TILE_M * 2, dtype=np.float32)
    expected_store_y = np.zeros((TILE_M * 2, 3), dtype=np.float32)
    expected_store_x[store_indices_np] = x
    expected_store_y[store_indices_np] = y
    assert_tile_map_struct_array(indexed_store_out.numpy(), expected_store_x, expected_store_y)

    assert_tile_map_struct_array(add_out.numpy(), 2.0 * x, 2.0 * y)
    assert_tile_map_struct_array(sub_out.numpy(), x, y)
    assert_tile_map_struct_array(sum_out.numpy(), np.array([np.sum(x)], dtype=np.float32), np.sum(y, axis=0)[None, :])

    transpose_np = transpose_out.numpy()
    assert_np_equal(transpose_np["x"], x.reshape(2, 4).T)
    assert_np_equal(transpose_np["y"], y.reshape(2, 4, 3).transpose(1, 0, 2))

    assert_tile_map_struct_array(extract_out.numpy(), np.array([x[3]], dtype=np.float32), y[3:4])
    assert_tile_map_struct_array(atomic_out.numpy(), x, y)
    assert_tile_map_struct_array(sort_out.numpy(), x[::-1], y[::-1])
    assert_tile_map_struct_array(tile_inplace_out.numpy(), 2.0 * x, 2.0 * y)
    assert_tile_map_struct_array(tile_sub_inplace_out.numpy(), x, y)
    assert_tile_map_struct_array(element_inplace_out.numpy(), np.array([x[0]], dtype=np.float32), y[0:1])
    assert_tile_map_struct_array(element_sub_inplace_out.numpy(), np.array([-x[0]], dtype=np.float32), -y[0:1])

    scatter_atomic_out = wp.empty(1, dtype=TileMapStruct, device=device)
    scatter_non_atomic_out = wp.zeros(TILE_M, dtype=TileMapStruct, device=device)
    stack_out = wp.empty(1, dtype=TileMapStruct, device=device)

    wp.launch_tiled(
        tile_struct_scatter_stack_kernel,
        dim=[1],
        inputs=[input_wp],
        outputs=[scatter_atomic_out, scatter_non_atomic_out, stack_out],
        block_dim=8,
        device=device,
    )

    assert_tile_map_struct_array(scatter_atomic_out.numpy(), np.array([x[0]], dtype=np.float32), y[0:1])
    if device.is_cpu:
        expected_scatter_x = np.zeros(TILE_M, dtype=np.float32)
        expected_scatter_y = np.zeros((TILE_M, 3), dtype=np.float32)
        expected_scatter_x[0] = x[0]
        expected_scatter_y[0] = y[0]
    else:
        expected_scatter_x = x
        expected_scatter_y = y
    assert_tile_map_struct_array(scatter_non_atomic_out.numpy(), expected_scatter_x, expected_scatter_y)
    assert_tile_map_struct_array(stack_out.numpy(), np.array([x[0]], dtype=np.float32), y[0:1])


def test_tile_struct_constructor_assign_stack(test, device):
    # struct tile construction (tile_from_thread / tile_full+tile_assign / element assign) and stack ops
    data = make_tile_map_struct_data()
    input_wp = wp.array(data, dtype=TileMapStruct, device=device)

    from_thread_out = wp.empty(TILE_M, dtype=TileMapStruct, device=device)
    assign_out = wp.empty(TILE_M, dtype=TileMapStruct, device=device)
    element_assign_out = wp.empty(TILE_M, dtype=TileMapStruct, device=device)
    stack_counts = wp.empty(2, dtype=int, device=device)

    wp.launch_tiled(
        tile_struct_constructor_assign_stack_kernel,
        dim=[1],
        inputs=[input_wp],
        outputs=[from_thread_out, assign_out, element_assign_out, stack_counts],
        block_dim=8,
        device=device,
    )

    x = np.arange(1, TILE_M + 1, dtype=np.float32)
    y = np.stack((x, x + 1.0, x + 2.0), axis=1)

    from_thread_index = 0 if device.is_cpu else 3
    assert_tile_map_struct_array(
        from_thread_out.numpy(),
        np.full(TILE_M, x[from_thread_index], dtype=np.float32),
        np.tile(y[from_thread_index], (TILE_M, 1)),
    )
    assert_tile_map_struct_array(
        assign_out.numpy(), np.full(TILE_M, 4.0, dtype=np.float32), np.tile([5.0, 6.0, 7.0], (TILE_M, 1))
    )

    expected_element_x = np.zeros(TILE_M, dtype=np.float32)
    expected_element_y = np.zeros((TILE_M, 3), dtype=np.float32)
    expected_element_x[0] = x[0]
    expected_element_y[0] = y[0]
    assert_tile_map_struct_array(element_assign_out.numpy(), expected_element_x, expected_element_y)
    assert_np_equal(stack_counts.numpy(), np.array([1, 0], dtype=np.int32))


def test_tile_struct_ones_rejected(test, device):
    # contract: tile_ones rejects struct dtypes (no canonical "one" value)
    @wp.kernel(module="unique")
    def kernel_fn(output: wp.array[TileMapStruct]):
        t = wp.tile_ones(TILE_M, dtype=TileMapStruct)
        wp.tile_store(output, t)

    output = wp.empty(TILE_M, dtype=TileMapStruct, device=device)

    with test.assertRaisesRegex((RuntimeError, TypeError), "tile_ones.*Warp struct"):
        wp.launch_tiled(
            kernel_fn,
            dim=[1],
            inputs=[],
            outputs=[output],
            block_dim=8,
            device=device,
        )


def test_tile_struct_bitwise_inplace_rejected(test, device):
    # contract: bitwise in-place ops (&=, |=, ^=) reject struct dtypes at tile and element scope
    @wp.kernel(module="unique")
    def tile_bit_and_kernel(input: wp.array[TileMapStruct], output: wp.array[TileMapStruct]):
        t = wp.tile_load(input, shape=TILE_M, storage="shared")
        t &= t
        wp.tile_store(output, t)

    @wp.kernel(module="unique")
    def tile_bit_or_kernel(input: wp.array[TileMapStruct], output: wp.array[TileMapStruct]):
        t = wp.tile_load(input, shape=TILE_M, storage="shared")
        t |= t
        wp.tile_store(output, t)

    @wp.kernel(module="unique")
    def tile_bit_xor_kernel(input: wp.array[TileMapStruct], output: wp.array[TileMapStruct]):
        t = wp.tile_load(input, shape=TILE_M, storage="shared")
        t ^= t
        wp.tile_store(output, t)

    @wp.kernel(module="unique")
    def element_bit_and_kernel(input: wp.array[TileMapStruct], output: wp.array[TileMapStruct]):
        _tile, lane = wp.tid()
        value = TileMapStruct()
        if lane == 0:
            value = input[0]

        t = wp.tile_zeros(shape=1, dtype=TileMapStruct, storage="shared")
        t[0] &= value
        total = wp.tile_extract(t, 0)
        if lane == 0:
            output[0] = total

    input_wp = wp.array(make_tile_map_struct_data(), dtype=TileMapStruct, device=device)
    output = wp.empty(TILE_M, dtype=TileMapStruct, device=device)

    for kernel_fn in (tile_bit_and_kernel, tile_bit_or_kernel, tile_bit_xor_kernel, element_bit_and_kernel):
        with test.assertRaisesRegex((RuntimeError, TypeError), "bitwise inplace.*Warp struct"):
            wp.launch_tiled(
                kernel_fn,
                dim=[1],
                inputs=[input_wp],
                outputs=[output],
                block_dim=8,
                device=device,
            )


@wp.func
def tile_struct_reduce_to_scalar(a: TileMapStruct, b: TileMapStruct) -> wp.float32:
    # A reduction operator whose return type is NOT the struct element type; tile_reduce()
    # must reject this rather than silently reinterpreting the scalar result as a struct.
    return a.x + b.x


def test_tile_struct_reduction_ops_rejected(test, device):
    """Reject ordering, custom-reduction, and range tile ops on struct dtypes cleanly at codegen.

    These ops must fail at codegen rather than reaching the native compiler. ``tile_min()``,
    ``tile_argmin()``, and ``tile_arange()`` are variadic, so the ``Scalar`` dtype constraint is not
    enforced by overload matching and needs an explicit guard. ``tile_reduce()`` must reject an
    operator with no overload accepting the struct element, or one whose return type is not the struct
    element type.
    """

    @wp.kernel(module="unique")
    def tile_min_kernel(input: wp.array[TileMapStruct], output: wp.array[TileMapStruct]):
        t = wp.tile_load(input, shape=TILE_M, storage="shared")
        wp.tile_store(output, wp.tile_min(t))

    @wp.kernel(module="unique")
    def tile_argmin_kernel(input: wp.array[TileMapStruct], output: wp.array[int]):
        t = wp.tile_load(input, shape=TILE_M, storage="shared")
        wp.tile_store(output, wp.tile_argmin(t))

    @wp.kernel(module="unique")
    def tile_reduce_min_kernel(input: wp.array[TileMapStruct], output: wp.array[TileMapStruct]):
        t = wp.tile_load(input, shape=TILE_M, storage="shared")
        wp.tile_store(output, wp.tile_reduce(wp.min, t))

    @wp.kernel(module="unique")
    def tile_reduce_bad_return_kernel(input: wp.array[TileMapStruct], output: wp.array[TileMapStruct]):
        t = wp.tile_load(input, shape=TILE_M, storage="shared")
        wp.tile_store(output, wp.tile_reduce(tile_struct_reduce_to_scalar, t))

    @wp.kernel(module="unique")
    def tile_arange_kernel(output: wp.array[TileMapStruct]):
        t = wp.tile_arange(TILE_M, dtype=TileMapStruct)
        wp.tile_store(output, t)

    input_wp = wp.array(make_tile_map_struct_data(), dtype=TileMapStruct, device=device)
    struct_out = wp.empty(TILE_M, dtype=TileMapStruct, device=device)
    int_out = wp.empty(TILE_M, dtype=int, device=device)

    with test.assertRaisesRegex((RuntimeError, TypeError), "tile_min.*Warp struct"):
        wp.launch_tiled(tile_min_kernel, dim=[1], inputs=[input_wp], outputs=[struct_out], block_dim=8, device=device)

    with test.assertRaisesRegex((RuntimeError, TypeError), "tile_argmin.*Warp struct"):
        wp.launch_tiled(tile_argmin_kernel, dim=[1], inputs=[input_wp], outputs=[int_out], block_dim=8, device=device)

    with test.assertRaisesRegex((RuntimeError, TypeError), "tile_reduce.*overload.*Warp struct"):
        wp.launch_tiled(
            tile_reduce_min_kernel, dim=[1], inputs=[input_wp], outputs=[struct_out], block_dim=8, device=device
        )

    with test.assertRaisesRegex((RuntimeError, TypeError), "tile_reduce.*must return the tile element type"):
        wp.launch_tiled(
            tile_reduce_bad_return_kernel, dim=[1], inputs=[input_wp], outputs=[struct_out], block_dim=8, device=device
        )

    with test.assertRaisesRegex((RuntimeError, TypeError), "tile_arange.*Warp struct"):
        wp.launch_tiled(tile_arange_kernel, dim=[1], inputs=[], outputs=[struct_out], block_dim=8, device=device)


def test_tile_struct_grad_ops(test, device):
    # adjoints for indexed-load, extract, and broadcast of struct tiles
    data = make_tile_map_struct_data()
    x = np.arange(1, TILE_M + 1, dtype=np.float32)
    y = np.stack((x, x + 1.0, x + 2.0), axis=1)
    value_sums = x + np.sum(y, axis=1)

    indices_np = np.arange(TILE_M, dtype=np.int32) // 2
    indices = wp.array(indices_np, dtype=int, device=device)
    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_load_indexed_grad_kernel,
            dim=[1],
            inputs=[input_wp, indices],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    assert_np_equal(loss.numpy(), np.array([np.sum(value_sums[indices_np])], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()

    expected_indexed_grad = np.zeros(TILE_M, dtype=np.float32)
    for idx in indices_np:
        expected_indexed_grad[idx] += 1.0
    input_grad = input_wp.grad.numpy()
    assert_np_equal(input_grad["x"], expected_indexed_grad)
    assert_np_equal(input_grad["y"], np.repeat(expected_indexed_grad[:, None], 3, axis=1))

    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_extract_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    assert_np_equal(loss.numpy(), np.array([value_sums[3]], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()

    expected_extract_grad = np.zeros(TILE_M, dtype=np.float32)
    expected_extract_grad[3] = 1.0
    input_grad = input_wp.grad.numpy()
    assert_np_equal(input_grad["x"], expected_extract_grad)
    assert_np_equal(input_grad["y"], np.repeat(expected_extract_grad[:, None], 3, axis=1))

    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_broadcast_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    assert_np_equal(loss.numpy(), np.array([float(TILE_M) * value_sums[0]], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()

    expected_broadcast_grad = np.zeros(TILE_M, dtype=np.float32)
    expected_broadcast_grad[0] = float(TILE_M)
    input_grad = input_wp.grad.numpy()
    assert_np_equal(input_grad["x"], expected_broadcast_grad)
    assert_np_equal(input_grad["y"], np.repeat(expected_broadcast_grad[:, None], 3, axis=1))


def test_tile_struct_additional_grad_ops(test, device):
    """Verify adjoints for the distinct struct mechanisms.

    Covers ``tile_assign`` (movement), add, tile ``+=``, sub, element assign, axis-sum,
    ``atomic_add``, ``store_indexed``, and ``atomic_add_indexed``.
    """
    data = make_tile_map_struct_data()
    x, y, value_sums, transformed_value_sums, expected_unit_grad, expected_transformed_y_grad = (
        make_tile_map_struct_expected_values()
    )

    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_tile_assign_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    assert_np_equal(loss.numpy(), np.array([np.sum(value_sums)], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()
    assert_tile_map_struct_grad(input_wp.grad.numpy(), expected_unit_grad)

    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_add_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    assert_np_equal(loss.numpy(), np.array([2.0 * np.sum(value_sums)], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()
    assert_tile_map_struct_grad(input_wp.grad.numpy(), 2.0 * expected_unit_grad)

    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_tile_add_inplace_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    assert_np_equal(loss.numpy(), np.array([np.sum(value_sums)], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()
    assert_tile_map_struct_grad(input_wp.grad.numpy(), expected_unit_grad)

    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_sub_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    assert_np_equal(loss.numpy(), np.array([-np.sum(value_sums)], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()
    assert_tile_map_struct_grad(input_wp.grad.numpy(), -expected_unit_grad)

    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_element_assign_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    expected_element_assign_grad = np.ones(TILE_M, dtype=np.float32)
    if device.is_cpu:
        expected_element_assign_grad[1:] = 0.0

    assert_np_equal(loss.numpy(), np.array([np.sum(value_sums * expected_element_assign_grad)], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()
    assert_tile_map_struct_grad(input_wp.grad.numpy(), expected_element_assign_grad)

    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_sum_axis_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    assert_np_equal(loss.numpy(), np.array([np.sum(value_sums)], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()
    assert_tile_map_struct_grad(input_wp.grad.numpy(), expected_unit_grad)

    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    output_wp = wp.zeros(TILE_M, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_atomic_add_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[output_wp],
            block_dim=8,
            device=device,
        )
        wp.launch_tiled(
            tile_map_custom_struct_grad_kernel,
            dim=[1],
            inputs=[output_wp],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    assert_tile_map_struct_array(output_wp.numpy(), x, y)
    assert_np_equal(loss.numpy(), np.array([np.sum(transformed_value_sums)], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()
    assert_tile_map_struct_grad_components(input_wp.grad.numpy(), expected_unit_grad, expected_transformed_y_grad)

    indices_np = np.arange(TILE_M - 1, -1, -1, dtype=np.int32)
    indices = wp.array(indices_np, dtype=int, device=device)
    weights_np = np.arange(1, TILE_M + 1, dtype=np.float32)
    weights = wp.array(weights_np, dtype=float, device=device)
    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    output_wp = wp.zeros(TILE_M, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_store_indexed_grad_kernel,
            dim=[1],
            inputs=[input_wp, indices],
            outputs=[output_wp],
            block_dim=8,
            device=device,
        )
        wp.launch_tiled(
            tile_struct_weighted_sum_grad_kernel,
            dim=[1],
            inputs=[output_wp, weights],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    expected_store_indexed_grad = weights_np[indices_np]
    assert_tile_map_struct_array(output_wp.numpy(), x[::-1], y[::-1])
    assert_np_equal(loss.numpy(), np.array([np.sum(value_sums * expected_store_indexed_grad)], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()
    assert_tile_map_struct_grad(input_wp.grad.numpy(), expected_store_indexed_grad)

    indices_np = np.ones(TILE_M, dtype=np.int32)
    indices = wp.array(indices_np, dtype=int, device=device)
    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    output_wp = wp.zeros(TILE_M, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_atomic_add_indexed_kernel,
            dim=[1],
            inputs=[input_wp, indices],
            outputs=[output_wp],
            block_dim=8,
            device=device,
        )
        wp.launch_tiled(
            tile_map_custom_struct_grad_kernel,
            dim=[1],
            inputs=[output_wp],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    expected_atomic_x = np.zeros(TILE_M, dtype=np.float32)
    expected_atomic_y = np.zeros((TILE_M, 3), dtype=np.float32)
    expected_atomic_x[1] = np.sum(x)
    expected_atomic_y[1] = np.sum(y, axis=0)
    assert_tile_map_struct_array(output_wp.numpy(), expected_atomic_x, expected_atomic_y)
    assert_np_equal(loss.numpy(), np.array([np.sum(transformed_value_sums)], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()
    assert_tile_map_struct_grad_components(input_wp.grad.numpy(), expected_unit_grad, expected_transformed_y_grad)


def test_tile_struct_scatter_add_grad_ops(test, device):
    # scatter_add adjoint, including the collision case (multiple lanes accumulating into one slot)
    data = make_tile_map_struct_data()
    x, y, value_sums, transformed_value_sums, expected_unit_grad, expected_transformed_y_grad = (
        make_tile_map_struct_expected_values()
    )

    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    output_wp = wp.zeros(TILE_M, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_scatter_add_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[output_wp],
            block_dim=8,
            device=device,
        )
        wp.launch_tiled(
            tile_map_custom_struct_grad_kernel,
            dim=[1],
            inputs=[output_wp],
            outputs=[loss],
            block_dim=8,
            device=device,
        )

    assert_tile_map_struct_array(output_wp.numpy(), x, y)
    assert_np_equal(loss.numpy(), np.array([np.sum(transformed_value_sums)], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()
    assert_tile_map_struct_grad_components(input_wp.grad.numpy(), expected_unit_grad, expected_transformed_y_grad)

    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    output_wp = wp.zeros(1, dtype=TileMapStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_struct_scatter_add_collision_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[output_wp],
            block_dim=8,
            device=device,
        )
        wp.launch(
            tile_struct_single_sum_grad_kernel,
            dim=1,
            inputs=[output_wp],
            outputs=[loss],
            device=device,
        )

    assert_tile_map_struct_array(output_wp.numpy(), np.array([np.sum(x)], dtype=np.float32), np.sum(y, axis=0)[None, :])
    assert_np_equal(loss.numpy(), np.array([np.sum(value_sums)], dtype=np.float32))
    loss.grad = wp.ones_like(loss, device=device)
    tape.backward()
    assert_tile_map_struct_grad(input_wp.grad.numpy(), expected_unit_grad)


def test_tile_struct_half_fields(test, device):
    """Verify half-precision struct fields compile through the generated struct shuffle helpers.

    Half-precision struct fields previously tripped these helpers during NVRTC compile:
    ``warp_shuffle_down`` built an anonymous union over the half field (deleted default constructor),
    and ``warp_shuffle_xor`` resolved to an ambiguous ``__shfl_xor_sync`` overload. Both are CUDA-only
    code paths.
    """
    data = []
    for i in range(TILE_M):
        s = HalfFieldStruct()
        s.s = wp.float16(float(i + 1))
        s.v = wp.vec3h(wp.float16(float(i + 1)), wp.float16(float(i + 2)), wp.float16(float(i + 3)))
        s.m = wp.mat22h(
            wp.float16(float(i + 1)),
            wp.float16(float(i + 2)),
            wp.float16(float(i + 3)),
            wp.float16(float(i + 4)),
        )
        data.append(s)
    input_wp = wp.array(data, dtype=HalfFieldStruct, device=device)

    sort_keys_np = np.arange(TILE_M - 1, -1, -1, dtype=np.int32)
    sort_keys = wp.array(sort_keys_np, dtype=int, device=device)

    store_out = wp.empty(TILE_M, dtype=HalfFieldStruct, device=device)
    sum_out = wp.empty(1, dtype=HalfFieldStruct, device=device)
    sort_out = wp.empty(TILE_M, dtype=HalfFieldStruct, device=device)

    wp.launch_tiled(
        tile_struct_half_fields_kernel,
        dim=[1],
        inputs=[input_wp, sort_keys],
        outputs=[store_out, sum_out, sort_out],
        block_dim=32,
        device=device,
    )

    # Values are small integers exact in half precision, so exact comparison is valid.
    s = np.arange(1, TILE_M + 1, dtype=np.float32)
    v = np.stack((s, s + 1.0, s + 2.0), axis=1)
    m = np.stack((s, s + 1.0, s + 2.0, s + 3.0), axis=1).reshape(TILE_M, 2, 2)

    store_np = store_out.numpy()
    assert_np_equal(store_np["s"].astype(np.float32), s)
    assert_np_equal(store_np["v"].astype(np.float32), v)
    assert_np_equal(store_np["m"].astype(np.float32), m)

    # tile_sum exercises warp_shuffle_down on each half field at runtime.
    sum_np = sum_out.numpy()
    assert_np_equal(sum_np["s"].astype(np.float32), np.array([np.sum(s)], dtype=np.float32))
    assert_np_equal(sum_np["v"].astype(np.float32), np.sum(v, axis=0)[None, :])
    assert_np_equal(sum_np["m"].astype(np.float32), np.sum(m, axis=0)[None, :, :])

    # tile_sort permutes the half struct value payload via warp_shuffle_xor; descending
    # keys reverse the row order.
    sort_np = sort_out.numpy()
    assert_np_equal(sort_np["s"].astype(np.float32), s[::-1])
    assert_np_equal(sort_np["v"].astype(np.float32), v[::-1])
    assert_np_equal(sort_np["m"].astype(np.float32), m[::-1])

    # Control: non-struct half tile load/store already compiled and ran before the fix.
    half_in = wp.array(s, dtype=wp.float16, device=device)
    half_out = wp.empty(TILE_M, dtype=wp.float16, device=device)
    wp.launch_tiled(tile_half_scalar_kernel, dim=[1], inputs=[half_in], outputs=[half_out], block_dim=32, device=device)
    assert_np_equal(half_out.numpy().astype(np.float32), s)


def test_tile_struct_quat_transform_fields(test, device):
    """Load, sum, and sort structs with quaternion and transform fields.

    Quaternion and transform struct fields tripped the generated struct shuffle helpers during
    NVRTC compile. Unlike vectors and matrices (which have dedicated overloads), these value types
    fall through to the generic ``warp_shuffle_down`` / ``warp_shuffle_xor`` templates.
    ``warp_shuffle_down`` built an anonymous union over the field (deleted default constructor for
    types with a non-trivial default constructor), and ``warp_shuffle_xor`` resolved to a
    ``__shfl_xor_sync`` overload that does not accept ``quat_t`` / ``transform_t``. Both are
    CUDA-only code paths and fail even for a plain load/store.
    """
    data = []
    for i in range(TILE_M):
        s = QuatTransformFieldStruct()
        s.q = wp.quat(float(i + 1), float(i + 2), float(i + 3), float(i + 4))
        s.xform = wp.transform(
            wp.vec3(float(i + 1), float(i + 2), float(i + 3)),
            wp.quat(float(i + 4), float(i + 5), float(i + 6), float(i + 7)),
        )
        s.v = wp.vec3(float(i + 1), float(i + 2), float(i + 3))
        s.s = float(i + 1)
        data.append(s)
    input_wp = wp.array(data, dtype=QuatTransformFieldStruct, device=device)

    sort_keys_np = np.arange(TILE_M - 1, -1, -1, dtype=np.int32)
    sort_keys = wp.array(sort_keys_np, dtype=int, device=device)

    store_out = wp.empty(TILE_M, dtype=QuatTransformFieldStruct, device=device)
    sum_out = wp.empty(1, dtype=QuatTransformFieldStruct, device=device)
    sort_out = wp.empty(TILE_M, dtype=QuatTransformFieldStruct, device=device)

    wp.launch_tiled(
        tile_struct_quat_transform_fields_kernel,
        dim=[1],
        inputs=[input_wp, sort_keys],
        outputs=[store_out, sum_out, sort_out],
        block_dim=32,
        device=device,
    )

    base = np.arange(1, TILE_M + 1, dtype=np.float32)
    q = np.stack([base + k for k in range(4)], axis=1)
    xform = np.stack([base + k for k in range(7)], axis=1)
    v = np.stack([base + k for k in range(3)], axis=1)
    s = base

    # Plain load/store round-trips every field.
    store_np = store_out.numpy()
    assert_np_equal(store_np["q"], q)
    assert_np_equal(store_np["xform"], xform)
    assert_np_equal(store_np["v"], v)
    assert_np_equal(store_np["s"], s)

    # tile_sum exercises warp_shuffle_down on each value field at runtime.
    sum_np = sum_out.numpy()
    assert_np_equal(sum_np["q"], np.sum(q, axis=0)[None, :])
    assert_np_equal(sum_np["xform"], np.sum(xform, axis=0)[None, :])
    assert_np_equal(sum_np["v"], np.sum(v, axis=0)[None, :])
    assert_np_equal(sum_np["s"], np.array([np.sum(s)], dtype=np.float32))

    # tile_sort permutes the struct value payload via warp_shuffle_xor; descending keys
    # reverse the row order.
    sort_np = sort_out.numpy()
    assert_np_equal(sort_np["q"], q[::-1])
    assert_np_equal(sort_np["xform"], xform[::-1])
    assert_np_equal(sort_np["v"], v[::-1])
    assert_np_equal(sort_np["s"], s[::-1])


def test_tile_struct_float64_fields(test, device):
    """Load, sum, and sort structs with float64 fields.

    float64 struct fields route through the generic ``warp_shuffle_down`` / ``warp_shuffle_xor``
    templates, which shuffle word-by-word over a buffer of ``unsigned int`` (4-byte aligned) and
    then reinterpret it as the field type. float64 requires 8-byte alignment, so reading the value
    back through that under-aligned buffer was undefined behavior. The buffers are now aligned for
    the field type. CUDA-only code path.
    """
    data = []
    for i in range(TILE_M):
        s = Float64FieldStruct()
        s.s = wp.float64(float(i + 1))
        s.v = wp.vec3d(wp.float64(float(i + 1)), wp.float64(float(i + 2)), wp.float64(float(i + 3)))
        s.m = wp.mat22d(
            wp.float64(float(i + 1)),
            wp.float64(float(i + 2)),
            wp.float64(float(i + 3)),
            wp.float64(float(i + 4)),
        )
        data.append(s)
    input_wp = wp.array(data, dtype=Float64FieldStruct, device=device)

    sort_keys_np = np.arange(TILE_M - 1, -1, -1, dtype=np.int32)
    sort_keys = wp.array(sort_keys_np, dtype=int, device=device)

    store_out = wp.empty(TILE_M, dtype=Float64FieldStruct, device=device)
    sum_out = wp.empty(1, dtype=Float64FieldStruct, device=device)
    sort_out = wp.empty(TILE_M, dtype=Float64FieldStruct, device=device)

    wp.launch_tiled(
        tile_struct_float64_fields_kernel,
        dim=[1],
        inputs=[input_wp, sort_keys],
        outputs=[store_out, sum_out, sort_out],
        block_dim=32,
        device=device,
    )

    s = np.arange(1, TILE_M + 1, dtype=np.float64)
    v = np.stack((s, s + 1.0, s + 2.0), axis=1)
    m = np.stack((s, s + 1.0, s + 2.0, s + 3.0), axis=1).reshape(TILE_M, 2, 2)

    # Plain load/store round-trips every field.
    store_np = store_out.numpy()
    assert_np_equal(store_np["s"], s)
    assert_np_equal(store_np["v"], v)
    assert_np_equal(store_np["m"], m)

    # tile_sum exercises warp_shuffle_down on each float64 field at runtime.
    sum_np = sum_out.numpy()
    assert_np_equal(sum_np["s"], np.array([np.sum(s)], dtype=np.float64))
    assert_np_equal(sum_np["v"], np.sum(v, axis=0)[None, :])
    assert_np_equal(sum_np["m"], np.sum(m, axis=0)[None, :, :])

    # tile_sort permutes the float64 struct value payload via warp_shuffle_xor; descending
    # keys reverse the row order.
    sort_np = sort_out.numpy()
    assert_np_equal(sort_np["s"], s[::-1])
    assert_np_equal(sort_np["v"], v[::-1])
    assert_np_equal(sort_np["m"], m[::-1])


def test_tile_struct_non_atomic_fields(test, device):
    """Verify struct fields whose scalar type has no ``atomicAdd`` overload still compile.

    On CUDA, such fields (bool, ``int8``) previously tripped the generated struct ``atomic_add`` helper
    during NVRTC compile, even for a plain load/store that never accumulates.
    """
    data = []
    for i in range(TILE_M):
        s = NonAtomicFieldStruct()
        s.flag = bool(i % 2)
        s.small = wp.int8(i)
        s.value = float(i + 1)
        data.append(s)
    input_wp = wp.array(data, dtype=NonAtomicFieldStruct, device=device)

    store_out = wp.empty(TILE_M, dtype=NonAtomicFieldStruct, device=device)
    atomic_out = wp.zeros(TILE_M, dtype=NonAtomicFieldStruct, device=device)

    wp.launch_tiled(
        tile_struct_non_atomic_fields_kernel,
        dim=[1],
        inputs=[input_wp],
        outputs=[store_out, atomic_out],
        block_dim=32,
        device=device,
    )

    flag = np.array([bool(i % 2) for i in range(TILE_M)])
    small = np.arange(TILE_M, dtype=np.int8)
    value = np.arange(1, TILE_M + 1, dtype=np.float32)

    # Load/store round-trips every field, including the non-atomic ones.
    store_np = store_out.numpy()
    assert_np_equal(store_np["flag"], flag)
    assert_np_equal(store_np["small"], small)
    assert_np_equal(store_np["value"], value)

    # tile_atomic_add accumulates the supported float field; the bool / int8 fields are
    # not accumulated and keep the destination's initial (zero) value.
    atomic_np = atomic_out.numpy()
    assert_np_equal(atomic_np["value"], value)
    assert_np_equal(atomic_np["flag"], np.zeros(TILE_M, dtype=bool))
    assert_np_equal(atomic_np["small"], np.zeros(TILE_M, dtype=np.int8))

    # Field-wise reduction and arithmetic must treat the fields the same way as
    # tile_atomic_add: accumulate the float field, carry the bool / int8 fields. Use
    # uniform non-float fields so the carried value is well-defined regardless of which
    # lane's value survives the reduction.
    uniform = []
    for i in range(TILE_M):
        s = NonAtomicFieldStruct()
        s.flag = True
        s.small = wp.int8(3)
        s.value = float(i + 1)
        uniform.append(s)
    uniform_wp = wp.array(uniform, dtype=NonAtomicFieldStruct, device=device)
    sum_out = wp.empty(1, dtype=NonAtomicFieldStruct, device=device)
    add_out = wp.empty(TILE_M, dtype=NonAtomicFieldStruct, device=device)

    wp.launch_tiled(
        tile_struct_non_atomic_accumulate_kernel,
        dim=[1],
        inputs=[uniform_wp],
        outputs=[sum_out, add_out],
        block_dim=32,
        device=device,
    )

    sum_np = sum_out.numpy()
    assert_np_equal(sum_np["value"], np.array([value.sum()], dtype=np.float32))
    assert_np_equal(sum_np["flag"], np.array([True]))
    assert_np_equal(sum_np["small"], np.array([3], dtype=np.int8))

    add_np = add_out.numpy()
    assert_np_equal(add_np["value"], 2.0 * value)
    assert_np_equal(add_np["flag"], np.ones(TILE_M, dtype=bool))
    assert_np_equal(add_np["small"], np.full(TILE_M, 3, dtype=np.int8))


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


class TestTileStruct(unittest.TestCase):
    pass


add_function_test(TestTileStruct, "test_tile_map_custom_struct", test_tile_map_custom_struct, devices=devices)
add_function_test(TestTileStruct, "test_tile_nested_struct_ops", test_tile_nested_struct_ops, devices=devices)
add_function_test(
    TestTileStruct, "test_tile_nested_field_store_grad", test_tile_nested_field_store_grad, devices=devices
)
add_function_test(
    TestTileStruct, "test_tile_struct_array_payload_sort", test_tile_struct_array_payload_sort, devices=cuda_devices
)
add_function_test(
    TestTileStruct,
    "test_tile_struct_indexed_array_payload_sort",
    test_tile_struct_indexed_array_payload_sort,
    devices=cuda_devices,
)
add_function_test(TestTileStruct, "test_tile_struct_value_ops", test_tile_struct_value_ops, devices=devices)
add_function_test(TestTileStruct, "test_tile_struct_half_fields", test_tile_struct_half_fields, devices=cuda_devices)
add_function_test(
    TestTileStruct,
    "test_tile_struct_quat_transform_fields",
    test_tile_struct_quat_transform_fields,
    devices=cuda_devices,
)
add_function_test(
    TestTileStruct,
    "test_tile_struct_float64_fields",
    test_tile_struct_float64_fields,
    devices=cuda_devices,
)
add_function_test(
    TestTileStruct, "test_tile_struct_non_atomic_fields", test_tile_struct_non_atomic_fields, devices=devices
)
add_function_test(
    TestTileStruct,
    "test_tile_struct_constructor_assign_stack",
    test_tile_struct_constructor_assign_stack,
    devices=devices,
)
add_function_test(TestTileStruct, "test_tile_struct_ones_rejected", test_tile_struct_ones_rejected, devices=devices)
add_function_test(
    TestTileStruct,
    "test_tile_struct_bitwise_inplace_rejected",
    test_tile_struct_bitwise_inplace_rejected,
    devices=devices,
)
add_function_test(
    TestTileStruct,
    "test_tile_struct_reduction_ops_rejected",
    test_tile_struct_reduction_ops_rejected,
    devices=devices,
)
add_function_test(TestTileStruct, "test_tile_struct_grad_ops", test_tile_struct_grad_ops, devices=devices)
add_function_test(
    TestTileStruct, "test_tile_struct_additional_grad_ops", test_tile_struct_additional_grad_ops, devices=devices
)
add_function_test(
    TestTileStruct, "test_tile_struct_scatter_add_grad_ops", test_tile_struct_scatter_add_grad_ops, devices=cuda_devices
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
