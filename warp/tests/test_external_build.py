# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_test_devices


class NativeColor(ctypes.Structure):
    _fields_ = [("r", ctypes.c_float), ("g", ctypes.c_float), ("b", ctypes.c_float)]


class NativePixel(ctypes.Structure):
    _fields_ = [("color", NativeColor), ("index", ctypes.c_int32)]


class NativeImage(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


wp.build_experimental.add_native_type(
    NativeColor,
    native_name="warp_test::Color",
    fields={"r": wp.float32, "g": wp.float32, "b": wp.float32},
    initializer="aggregate",
)
wp.build_experimental.add_native_type(
    NativePixel,
    native_name="warp_test::Pixel",
    fields={"color": NativeColor, "index": wp.int32},
    initializer="aggregate",
)
wp.build_experimental.add_native_type(NativeImage, native_name="warp_test::Image")
wp.build_experimental.add_builtin(
    "_test_native_scale",
    {"value": NativeColor, "scale": wp.float32},
    NativeColor,
    native_name="warp_test::scale",
)
wp.build_experimental.add_builtin(
    "_test_native_image_handle",
    {"image": NativeImage},
    wp.uint64,
    native_name="warp_test::image_handle",
)

external_scale = wp.build_experimental.add_builtin(
    "_test_external_overloaded_scale",
    {"value": wp.float32, "factor": wp.float32},
    wp.float32,
    native_name="warp_test::scale_scalar",
)
wp.build_experimental.add_builtin(
    "_test_external_overloaded_scale",
    {"value": wp.int32, "factor": wp.int32},
    wp.int32,
    native_name="warp_test::scale_scalar",
)
external_write = wp.build_experimental.add_builtin(
    "_test_external_write",
    {"output": wp.ref[wp.float32], "value": wp.float32},
    native_name="warp_test::write",
)
external_alias = wp.build_experimental.add_builtin(
    "_test_external_alias",
    {"left": wp.ref[wp.float32], "right": wp.ref[wp.float32]},
    native_name="warp_test::alias",
)

external_square = wp.build_experimental.add_builtin(
    "_test_external_square_adjoint",
    {"value": float},
    float,
    native_name="warp_test::square_scalar",
    is_differentiable=True,
)


external_root = wp.build_experimental.add_builtin(
    "_test_external_root_adjoint",
    {"value": float},
    float,
    native_name="warp_test::root_scalar",
    is_differentiable=True,
    adjoint_uses_forward_output=True,
)
external_root_product = wp.build_experimental.add_builtin(
    "_test_external_root_product_adjoint",
    {"value": float, "factor": float},
    float,
    native_name="warp_test::root_product",
    is_differentiable=True,
    adjoint_uses_forward_output=True,
)
external_roots = wp.build_experimental.add_builtin(
    "_test_external_roots_adjoint",
    {"value": float},
    wp.vec3,
    native_name="warp_test::root_vector",
    is_differentiable=True,
    adjoint_uses_forward_output=True,
)


NATIVE_COLOR_CONSTANT = NativeColor(0.25, 0.5, 0.75)


@wp.struct
class NativeContainer:
    color: NativeColor


@wp.struct
class NativeGradientContext:
    nested: NativeContainer
    weight: float


@wp.kernel(module="test_external_native_types")
def native_array_kernel(
    pixels: wp.array[NativePixel],
    images: wp.array[NativeImage],
    colors: wp.array[NativeColor],
    handles: wp.array[wp.uint64],
):
    i = wp.tid()
    color = NativeColor(wp.float32(i), NATIVE_COLOR_CONSTANT.g, 3.0)
    pixel = NativePixel(color, i)
    pixels[i] = pixel
    colors[i] = wp._test_native_scale(pixel.color, 2.0)
    handles[i] = wp._test_native_image_handle(images[i])


@wp.kernel(module="test_external_native_types")
def native_struct_kernel(
    container: NativeContainer,
    image: NativeImage,
    color: wp.array[NativeColor],
    handle: wp.array[wp.uint64],
):
    color[0] = container.color
    handle[0] = wp._test_native_image_handle(image)


_NATIVE_TEST_PREAMBLE = """
#if defined(__CUDACC__)
#define WARP_TEST_CALLABLE __host__ __device__ inline
#else
#define WARP_TEST_CALLABLE inline
#endif
namespace warp_test {
struct Color {
    float r;
    float g;
    float b;
};
struct Pixel {
    Color color;
    int index;
};
struct Image {
    unsigned long long handle;
};
WARP_TEST_CALLABLE Color scale(Color value, float scale) {
    return Color{value.r * scale, value.g * scale, value.b * scale};
}
WARP_TEST_CALLABLE unsigned long long image_handle(Image image) {
    return image.handle;
}
WARP_TEST_CALLABLE float square_scalar(float value) { return value * value; }
WARP_TEST_CALLABLE void adj_square_scalar(float value, float& adj_value, float adj_ret) {
    adj_value += 2.0f * value * adj_ret;
}
WARP_TEST_CALLABLE float root_scalar(float value) { return wp::sqrt(value); }
WARP_TEST_CALLABLE void adj_root_scalar(float value, float result, float& adj_value, float adj_ret) {
    adj_value += adj_ret / (2.0f * result);
}
WARP_TEST_CALLABLE float root_product(float value, float factor) { return wp::sqrt(value * factor); }
WARP_TEST_CALLABLE void adj_root_product(float value, float factor, float result, float& adj_value, float& adj_factor, float adj_ret) {
    adj_value += factor * adj_ret / (2.0f * result);
    adj_factor += value * adj_ret / (2.0f * result);
}
WARP_TEST_CALLABLE wp::vec3 root_vector(float value) {
    return wp::vec3(wp::sqrt(value), wp::sqrt(2.0f * value), wp::sqrt(3.0f * value));
}
WARP_TEST_CALLABLE void adj_root_vector(float value, const wp::vec3& result, float& adj_value, const wp::vec3& adj_ret) {
    for (int i = 0; i < 3; ++i) adj_value += float(i + 1) * adj_ret[i] / (2.0f * result[i]);
}
WARP_TEST_CALLABLE float scale_scalar(float value, float factor) { return value * factor; }
WARP_TEST_CALLABLE int scale_scalar(int value, int factor) { return value * factor; }
WARP_TEST_CALLABLE void write(float& output, float value) { output = value; }
WARP_TEST_CALLABLE void alias(float& left, float& right) { left += 1.0f; right *= 2.0f; }
}  // namespace warp_test
#undef WARP_TEST_CALLABLE
"""

wp.set_module_options(
    {
        "extra_build_options": wp.ModuleBuildOptions(
            extra_cpu_preamble=_NATIVE_TEST_PREAMBLE,
            extra_cuda_preamble=_NATIVE_TEST_PREAMBLE,
        ),
    },
    module=native_array_kernel.module,
)


@wp.kernel(module="test_external_native_types", enable_backward=False)
def external_function_kernel(values: wp.array[float], integers: wp.array[int], colors: wp.array[NativeColor]):
    """Exercise overloads, aliased references, and native return values."""
    index = wp.tid()
    value = values[index]
    external_write(value, external_scale(factor=3.0, value=value))
    external_write(output=values[index], value=value)
    external_alias(values[index], values[index])
    integers[index] = external_scale(integers[index], factor=4)
    colors[index] = wp._test_native_scale(scale=2.0, value=colors[index])


def test_external_function_execution(test, device):
    """Verify external functions preserve values and addressable output mutations."""
    values = wp.array([2.0, 3.0], dtype=float, device=device)
    integers = wp.array([5, 6], dtype=int, device=device)
    colors = wp.array([NativeColor(1, 2, 3), NativeColor(4, 5, 6)], dtype=NativeColor, device=device)
    wp.launch(external_function_kernel, dim=2, inputs=[values, integers, colors], device=device)
    np.testing.assert_array_equal(values.numpy(), [14.0, 20.0])
    np.testing.assert_array_equal(integers.numpy(), [20, 24])
    np.testing.assert_array_equal(colors.numpy()["b"], [6.0, 12.0])


def test_external_reference_forward_loops(test, device):
    """Forward-only kernels retain mutation and aliasing through loops and helpers."""

    @wp.func
    def mutate(values: wp.array[float]):
        """Mutate aliased array elements through an external call."""
        external_alias(values[0], values[0])

    @wp.kernel(module="unique", enable_backward=False)
    def forward(values: wp.array[float], count: int):
        """Repeat reference mutations through helpers and dynamic loops."""
        for _i in range(count):
            mutate(values)
        j = int(0)
        while j < count:
            external_alias(values[0], values[0])
            j += 1

    wp.set_module_options(
        {
            "extra_build_options": wp.ModuleBuildOptions(
                extra_cpu_preamble=_NATIVE_TEST_PREAMBLE,
                extra_cuda_preamble=_NATIVE_TEST_PREAMBLE,
            ),
        },
        module=forward.module,
    )
    values = wp.array([2.0], dtype=float, device=device)
    wp.launch(forward, dim=1, inputs=[values, 1], device=device)
    np.testing.assert_array_equal(values.numpy(), [14.0])


def test_external_reference_forward_after_rejected_grad(test, device):
    """A rejected gradient build must not poison a shared helper's forward use."""

    @wp.func
    def mutate(value: float):
        """Overwrite a local before computing its square."""
        external_write(value, 2.0)
        return value * value

    @wp.kernel(module="unique", enable_backward=False)
    def gradient(values: wp.array[float]):
        """Request the gradient of a mutating helper."""
        values[1] = wp.grad(mutate)(values[0])

    with test.assertRaisesRegex(wp._src.codegen.WarpCodegenError, "backward replay could repeat mutations"):
        test._codegen_source(gradient, "cpu")

    @wp.kernel(module="unique", enable_backward=False)
    def forward(values: wp.array[float]):
        """Call the shared helper without requesting its gradient."""
        values[1] = mutate(values[0])

    wp.set_module_options(
        {
            "extra_build_options": wp.ModuleBuildOptions(
                extra_cpu_preamble=_NATIVE_TEST_PREAMBLE,
                extra_cuda_preamble=_NATIVE_TEST_PREAMBLE,
            ),
        },
        module=forward.module,
    )
    values = wp.array([3.0, 0.0], dtype=float, device=device)
    wp.launch(forward, dim=1, inputs=[values], device=device)
    np.testing.assert_array_equal(values.numpy(), [3.0, 4.0])


def test_external_reference_callback_forward_helpers(test, device):
    """Callback validation must not request adjoints for ordinary forward helpers."""

    @wp.func
    def write_ref(output: wp.ref[float], value: float):
        """Write a value through a reference argument."""
        output = value

    @wp.func
    def forward_helper(value: float):
        """Return the input without invoking its replay callback."""
        return value

    @wp.func_replay(forward_helper)
    def unused_replay(value: float):
        """Mutate the input if this unused replay callback is invoked."""
        external_write(value, 2.0)
        return value

    @wp.func
    def square(value: float):
        """Compute the square of the input."""
        return value * value

    @wp.func_grad(square)
    def square_grad(value: float, adj_ret: float):
        """Accumulate a gradient through a reference-writing helper."""
        local = wp.float32(0.0)
        write_ref(local, 2.0 * forward_helper(value) * adj_ret)
        wp.adjoint[value] += local

    @wp.func_replay(square)
    def square_replay(value: float):
        """Recompute the square through a reference-writing helper."""
        local = wp.float32(0.0)
        write_ref(local, forward_helper(value) * value)
        return local

    @wp.kernel(module="unique")
    def kernel(values: wp.array[float], loss: wp.array[float], count: int):
        """Call the custom-gradient function inside a dynamic loop."""
        for _i in range(count):
            loss[0] = square(values[0])

    # forward_helper's custom replay is registered but not called: both
    # callbacks use its ordinary forward body, which has no external mutation.
    wp.set_module_options(
        {
            "extra_build_options": wp.ModuleBuildOptions(
                extra_cpu_preamble=_NATIVE_TEST_PREAMBLE,
                extra_cuda_preamble=_NATIVE_TEST_PREAMBLE,
            ),
        },
        module=kernel.module,
    )
    values = wp.array([3.0], dtype=float, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
    with wp.Tape() as tape:
        wp.launch(kernel, dim=1, inputs=[values, loss, 1], device=device)
    tape.backward(loss)
    np.testing.assert_array_equal(loss.numpy(), [9.0])
    np.testing.assert_array_equal(values.grad.numpy(), [6.0])


def test_external_native_adjoint(test, device):
    """Use native adjoints and existing Python custom gradients on both backends."""

    @wp.func
    def wrapped_square(value: float):
        """Evaluate a square through a non-differentiable external function."""
        return external_scale(value, value)

    @wp.func_grad(wrapped_square)
    def wrapped_square_grad(value: float, adj_ret: float):
        """Supply the derivative explicitly in Warp."""
        wp.adjoint[value] += 2.0 * value * adj_ret

    @wp.kernel(module="unique")
    def kernel(values: wp.array[float], output: wp.array[float]):
        """Exercise the registered operation from a Warp kernel."""
        i = wp.tid()
        output[i] = external_square(values[i]) + wrapped_square(values[i])

    @wp.kernel(module=kernel.module, enable_backward=False)
    def gradient(values: wp.array[float], direct_grad: wp.array[float]):
        """Evaluate an explicit gradient independently of Tape."""
        i = wp.tid()
        direct_grad[i] = wp.grad(external_square)(values[i])

    wp.set_module_options(
        {
            "extra_build_options": wp.ModuleBuildOptions(
                extra_cpu_preamble=_NATIVE_TEST_PREAMBLE,
                extra_cuda_preamble=_NATIVE_TEST_PREAMBLE,
            )
        },
        module=kernel.module,
    )
    samples = np.array([-2.0, 0.5, 3.0], dtype=np.float32)
    values = wp.array(samples, requires_grad=True, device=device)
    output = wp.zeros(3, requires_grad=True, device=device)
    direct_grad = wp.zeros(3, device=device)
    with wp.Tape() as tape:
        wp.launch(kernel, dim=3, inputs=[values, output], device=device)
    tape.backward(grads={output: wp.ones(3, device=device)})
    wp.launch(gradient, dim=3, inputs=[values, direct_grad], device=device)
    epsilon = 1.0e-3
    numerical = (2.0 * (samples + epsilon) ** 2 - 2.0 * (samples - epsilon) ** 2) / (2.0 * epsilon)
    np.testing.assert_allclose(output.numpy(), 2.0 * samples**2)
    np.testing.assert_allclose(values.grad.numpy(), numerical, rtol=3.0e-4)
    np.testing.assert_allclose(direct_grad.numpy(), 2.0 * samples)


def test_external_native_output_adjoint(test, device):
    """Reconstruct native results for Tape, loops, helpers, tile callables, and grad()."""

    @wp.func
    def wrapped_root(value: float):
        """Call the native root through a Warp helper."""
        return external_root(value)

    @wp.func
    def stateful_root(value: float, state: wp.array[float]):
        """Change forward state exactly once, then evaluate the native operation."""
        external_alias(state[0], state[0])
        return external_root(value)

    @wp.func_grad(stateful_root)
    def stateful_root_grad(value: float, state: wp.array[float], adj_ret: float):
        """Accumulate the native root gradient without mutating state."""
        wp.adjoint[value] += wp.grad(external_root)(value) * adj_ret

    @wp.func_replay(stateful_root)
    def stateful_root_replay(value: float, state: wp.array[float]):
        """Recompute the root without repeating the state mutation."""
        return external_root(value)

    @wp.kernel(module="unique")
    def kernel(values: wp.array[float], output: wp.array[float], state: wp.array[float], mode: int, count: int):
        """Exercise native output adjoints through different call paths."""
        if mode == 0:
            output[0] = external_root(values[0])
        elif mode == 1:
            total = float(0.0)
            for i in range(count):
                total += external_root(values[0] + float(i))
            output[0] = total
        elif mode == 2:
            output[0] = wrapped_root(values[0])
        elif mode == 3:
            wp.tile_store(output, wp.tile_map(external_root, wp.tile_load(values, shape=1)))
        elif mode == 5:
            output[0] = stateful_root(values[0], state)
        elif mode == 6:
            wp.tile_store(output, wp.tile_map(external_root_product, wp.tile_load(values, shape=1), 4.0))
        else:
            roots = external_roots(values[0])
            output[0] = roots[0] + roots[1] + roots[2]

    @wp.kernel(module=kernel.module, enable_backward=False)
    def gradient(values: wp.array[float], output: wp.array[float]):
        """Evaluate scalar and vector native adjoints directly."""
        output[0] = wp.grad(external_root)(values[0])
        output[1] = wp.grad(external_roots)(values[0])

    wp.set_module_options(
        {
            "extra_build_options": wp.ModuleBuildOptions(
                extra_cpu_preamble=_NATIVE_TEST_PREAMBLE, extra_cuda_preamble=_NATIVE_TEST_PREAMBLE
            )
        },
        module=kernel.module,
    )

    def run(values, output, state, mode):
        """Reset state and evaluate the forward and backward operations."""
        state.zero_()
        values.grad.zero_()
        output.grad.zero_()
        with wp.Tape() as tape:
            wp.launch(kernel, dim=1, inputs=[values, output, state, mode, 3], device=device)
        tape.backward(loss=output)

    for mode in range(7):
        with test.subTest(mode=mode):
            values = wp.array([4.0], dtype=float, device=device, requires_grad=True)
            output = wp.zeros(1, device=device, requires_grad=True)
            state = wp.zeros(1, device=device)

            run(values, output, state, mode)
            if mode == 1:
                roots = np.sqrt([4.0, 5.0, 6.0])
                expected_value, expected_grad = roots.sum(), (0.5 / roots).sum()
            elif mode == 4:
                roots = np.sqrt([4.0, 8.0, 12.0])
                expected_value, expected_grad = roots.sum(), (np.array([1.0, 2.0, 3.0]) / (2.0 * roots)).sum()
            elif mode == 6:
                expected_value, expected_grad = 4.0, 0.5
            else:
                expected_value, expected_grad = 2.0, 0.25
            np.testing.assert_array_equal(state.numpy(), [2.0 if mode == 5 else 0.0])
            np.testing.assert_allclose(output.numpy(), [expected_value], rtol=1.0e-6)
            np.testing.assert_allclose(values.grad.numpy(), [expected_grad], rtol=1.0e-6)
            if mode == 5:
                # Cover reference mutation and an output-consuming adjoint without expanding every mode.
                with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
                    run(values, output, state, mode)
                values.fill_(9.0)
                state.fill_(20.0)
                wp.capture_launch(capture.graph)
                expected = {"state": 2.0, "output": 3.0, "gradient": 1.0 / 6.0}
                outputs = {"state": state, "output": output, "gradient": values.grad}
                for name, array in outputs.items():
                    np.testing.assert_allclose(array.numpy(), [expected[name]], rtol=1.0e-6)
                with tempfile.TemporaryDirectory() as directory:
                    path = str(Path(directory) / "external_adjoint")
                    wp.capture_save(capture.graph, path, inputs={"values": values}, outputs=outputs)
                    loaded = wp.capture_load(path, device=device)
                    loaded.set_param("values", wp.array([16.0], dtype=float, device=device))
                    wp.capture_launch(loaded)
                    expected = {"state": 2.0, "output": 4.0, "gradient": 0.125}
                    result = wp.empty(1, device=device)
                    for name, value in expected.items():
                        loaded.get_param(name, result)
                        np.testing.assert_allclose(result.numpy(), [value], rtol=1.0e-6)
    direct_grad = wp.zeros(2, device=device)
    wp.launch(gradient, dim=1, inputs=[values, direct_grad], device=device)
    np.testing.assert_allclose(direct_grad.numpy(), [0.25, np.sqrt([1.0, 2.0, 3.0]).sum() / 4.0], rtol=1.0e-6)


def test_external_reference_manual_backward(test, device):
    """Preserve state while reference-based adjoints compute gradients."""

    @wp.func
    def write_local(value: float):
        """Use an external reference to write a local value."""
        local = float(0.0)
        external_write(local, value)
        return local

    @wp.func
    def operation(value: float, state: wp.array[float]):
        """Exercise an external mutation in a wrapped operation."""
        external_alias(state[0], state[0])
        return write_local(value * value)

    @wp.func_grad(operation)
    def operation_grad(value: float, state: wp.array[float], adj_ret: float):
        """Supply the gradient without repeating the forward mutation."""
        # An ordinary helper in a manual callback does not need its own adjoint.
        wp.adjoint[value] += write_local(2.0 * value * adj_ret)

    @wp.func_replay(operation)
    def operation_replay(value: float, state: wp.array[float]):
        """Recompute the result without changing external state."""
        local = float(0.0)
        external_write(local, value * value)
        return local

    @wp.kernel(module="unique")
    def kernel(values: wp.array[float], state: wp.array[float], loss: wp.array[float], count: int):
        """Exercise the registered operation from a Warp kernel."""
        total = float(0.0)
        for _i in range(count):
            total += operation(values[0], state)
        loss[0] = total

    @wp.func
    def manual_square(value: float):
        """Square a value through the reference-writing helper."""
        return write_local(value * value)

    @wp.func_grad(manual_square)
    def manual_square_grad(value: float, adj_ret: float):
        """Accumulate the square derivative through a manual helper."""
        wp.adjoint[value] += write_local(2.0 * value * adj_ret)

    @wp.kernel(module=kernel.module, enable_backward=False)
    def gradient(values: wp.array[float], output: wp.array[float]):
        # wp.grad() calls only the supplied adjoint; no forward replay is needed.
        """Evaluate an explicit gradient independently of Tape."""
        output[0] = wp.grad(manual_square)(values[0])

    wp.set_module_options(
        {
            "extra_build_options": wp.ModuleBuildOptions(
                extra_cpu_preamble=_NATIVE_TEST_PREAMBLE,
                extra_cuda_preamble=_NATIVE_TEST_PREAMBLE,
            )
        },
        module=kernel.module,
    )
    values = wp.array([3.0], dtype=float, requires_grad=True, device=device)
    state = wp.zeros(1, device=device)
    loss = wp.zeros(1, requires_grad=True, device=device)
    with wp.Tape() as tape:
        wp.launch(kernel, dim=1, inputs=[values, state, loss, 3], device=device)
    np.testing.assert_array_equal(state.numpy(), [14.0])
    tape.backward(loss)
    np.testing.assert_allclose(loss.numpy(), [27.0])
    np.testing.assert_allclose(values.grad.numpy(), [18.0])
    np.testing.assert_array_equal(state.numpy(), [14.0])
    direct_grad = wp.zeros(1, device=device)
    wp.launch(gradient, dim=1, inputs=[values, direct_grad], device=device)
    np.testing.assert_allclose(direct_grad.numpy(), [6.0])
    np.testing.assert_array_equal(state.numpy(), [14.0])


@wp.struct
class ReplayItem:
    """Carry a differentiable value alongside mutable external state."""

    value: float
    state: wp.array[float]


def test_external_reference_replay_dispatch(test, device):
    """Preserve state across scalar, tuple, condition, and callable replay paths."""

    @wp.func
    def operation(item: ReplayItem):
        """Update external state and return the squared input."""
        external_alias(item.state[0], item.state[0])
        return item.value * item.value

    @wp.func_grad(operation)
    def operation_grad(item: ReplayItem, adj_ret: float):
        """Accumulate the derivative without modifying external state."""
        wp.adjoint[item].value += 2.0 * item.value * adj_ret

    @wp.func_replay(operation)
    def operation_replay(item: ReplayItem):
        """Recompute the squared input without the state update."""
        return item.value * item.value

    @wp.func
    def pair(item: ReplayItem):
        """Return two outputs from a mutating operation."""
        return operation(item), 3.0 * item.value

    @wp.func_grad(pair)
    def pair_grad(item: ReplayItem, adj_first: float, adj_second: float):
        """Accumulate both output contributions."""
        wp.adjoint[item].value += 2.0 * item.value * adj_first + 3.0 * adj_second

    @wp.func_replay(pair)
    def pair_replay(item: ReplayItem):
        """Recompute both outputs without executing the mutating helper."""
        return item.value * item.value, 3.0 * item.value

    @wp.func
    def automatic_pair(item: ReplayItem):
        """Exercise primal-output evaluation before an automatic tuple adjoint."""
        return operation(item), 3.0 * item.value

    @wp.func_replay(automatic_pair)
    def automatic_pair_replay(item: ReplayItem):
        """Protect the primal-output evaluation from repeating the mutation."""
        return item.value * item.value, 3.0 * item.value

    @wp.func
    def condition(i: int, count: int, item: ReplayItem):
        """Record each forward evaluation of the loop condition."""
        external_alias(item.state[0], item.state[0])
        return i < count

    @wp.func_replay(condition)
    def condition_replay(i: int, count: int, item: ReplayItem):
        """Evaluate the condition without repeating its state update."""
        return i < count

    @wp.kernel(module="unique")
    def kernel(items: wp.array[ReplayItem], output: wp.array[float], mode: int, count: int):
        """Exercise the distinct call sites using the same callback contracts."""
        if mode == 0:
            output[0] = operation(items[0])
        elif mode == 1:
            first, second = pair(items[0])
            output[0] = first + second
        elif mode == 2:
            total = float(0.0)
            i = int(0)
            while condition(i, count, items[0]):
                total += items[0].value * items[0].value
                i += 1
            output[0] = total
        elif mode == 3:
            tile = wp.tile_load(items, shape=1)
            wp.tile_store(output, wp.tile_map(operation, tile))
        else:
            result = wp.grad(automatic_pair)(items[0])
            output[0] = result.value

    @wp.kernel(module=kernel.module, enable_backward=False)
    def gradient(items: wp.array[ReplayItem], output: wp.array[float]):
        """Call a tuple-returning custom adjoint without evaluating its primal."""
        result = wp.grad(pair)(items[0])
        output[0] = result.value

    wp.set_module_options(
        {
            "extra_build_options": wp.ModuleBuildOptions(
                extra_cpu_preamble=_NATIVE_TEST_PREAMBLE, extra_cuda_preamble=_NATIVE_TEST_PREAMBLE
            )
        },
        module=kernel.module,
    )
    for mode, expected_output, expected_grad, expected_state in (
        (0, 9.0, 6.0, 2.0),
        (1, 18.0, 9.0, 2.0),
        (2, 18.0, 12.0, 14.0),
        (3, 9.0, 6.0, 2.0),
        (4, 9.0, 0.0, 2.0),  # wp.grad() is forward-only; replay must still preserve state.
    ):
        with test.subTest(mode=mode):
            state = wp.zeros(1, device=device)
            item = ReplayItem()
            item.value = 3.0
            item.state = state
            items = wp.array([item], dtype=ReplayItem, requires_grad=True, device=device)
            output = wp.zeros(1, requires_grad=True, device=device)
            with wp.Tape() as tape:
                wp.launch_tiled(kernel, dim=1, inputs=[items, output, mode, 2], block_dim=1, device=device)
            np.testing.assert_array_equal(state.numpy(), [expected_state])
            tape.backward(output)
            np.testing.assert_allclose(output.numpy(), [expected_output])
            np.testing.assert_allclose(items.grad.numpy()["value"], [expected_grad])
            np.testing.assert_array_equal(state.numpy(), [expected_state])
    wp.launch(gradient, dim=1, inputs=[items, output], block_dim=1, device=device)
    np.testing.assert_allclose(output.numpy(), [9.0])
    np.testing.assert_array_equal(state.numpy(), [expected_state])


def test_external_native_context_automatic_gradient(test, device):
    """Preserve ordinary gradients through native context without native adjoint writes."""

    @wp.func
    def multiply(color: NativeColor, value: float):
        """Scale an ordinary value using inactive native context."""
        return color.r * value

    @wp.func
    def transitive(pixel: NativePixel, value: float):
        """Pass nested native context through a helper."""
        return multiply(pixel.color, value)

    @wp.func
    def from_array(colors: wp.array[NativeColor], value: float):
        """Read native context from an array before scaling."""
        return multiply(colors[0], value)

    @wp.func
    def mixed(context: NativeGradientContext, value: float):
        """Combine native context with an active struct field."""
        return multiply(context.nested.color, value) * context.weight

    @wp.func
    def repeated(color: NativeColor, value: float, count: int):
        """Accumulate scaled values inside a dynamic loop."""
        total = float(0.0)
        for _i in range(count):
            total += multiply(color, value)
        return total

    @wp.kernel(module="unique")
    def kernel(color: NativeColor, colors: wp.array[NativeColor], x: wp.array[float], out: wp.array[float], mode: int):
        """Exercise automatic gradients with different native contexts."""
        i = wp.tid()
        if mode == 0:
            out[i] = color.r * x[i]
        elif mode == 1:
            out[i] = multiply(color, x[i])
        elif mode == 2:
            out[i] = transitive(NativePixel(color, 0), x[i])
        elif mode == 3:
            out[i] = from_array(colors, x[i])
        elif mode == 4:
            context = NativeGradientContext(NativeContainer(color), x[i])
            out[i] = mixed(context, x[i])
        else:
            out[i] = repeated(color, x[i], mode - 2)

    @wp.kernel(module=kernel.module, enable_backward=False)
    def gradient(color: NativeColor, colors: wp.array[NativeColor], x: wp.array[float], out: wp.array2d[float]):
        """Evaluate native-context helper gradients directly."""
        i = wp.tid()
        adj_color, direct_dx = wp.grad(multiply)(color, x[i])
        adj_pixel, transitive_dx = wp.grad(transitive)(NativePixel(color, 0), x[i])
        _adj_colors, array_dx = wp.grad(from_array)(colors, x[i])
        out[i, 0] = direct_dx
        out[i, 1] = transitive_dx
        out[i, 2] = array_dx
        context = NativeGradientContext(NativeContainer(color), x[i])
        adj_context, dx = wp.grad(mixed)(context, x[i])
        out[i, 3] = dx + adj_context.weight
        _adj_repeated_color, repeated_dx, _adj_count = wp.grad(repeated)(color, x[i], 3)
        out[i, 4] = repeated_dx
        out[i, 5] = adj_color.r + adj_pixel.color.r + adj_context.nested.color.r

    wp.set_module_options(
        {
            "extra_build_options": wp.ModuleBuildOptions(
                extra_cpu_preamble=_NATIVE_TEST_PREAMBLE, extra_cuda_preamble=_NATIVE_TEST_PREAMBLE
            )
        },
        module=kernel.module,
    )
    color = NativeColor(3.0, 0.0, 0.0)
    colors = wp.array([color], dtype=NativeColor, device=device)
    samples = np.array([-2.0, 0.5, 3.0], dtype=np.float32)
    for mode in range(6):
        with test.subTest(mode=mode):
            values = wp.array(samples, device=device, requires_grad=True)
            output = wp.zeros(3, device=device, requires_grad=True)
            with wp.Tape() as tape:
                wp.launch(kernel, dim=3, inputs=[color, colors, values, output, mode], device=device)
            tape.backward(grads={output: wp.ones(3, device=device)})
            expected_values = 3.0 * samples
            expected_grad = np.full_like(samples, 3.0)
            if mode == 4:
                expected_values *= samples
                expected_grad = 6.0 * samples
            elif mode == 5:
                expected_values *= 3.0
                expected_grad *= 3.0
            np.testing.assert_allclose(output.numpy(), expected_values)
            np.testing.assert_allclose(values.grad.numpy(), expected_grad)
    direct = wp.empty((3, 6), device=device)
    wp.launch(gradient, dim=3, inputs=[color, colors, values, direct], device=device)
    expected = np.column_stack((np.full((3, 3), 3.0), 6.0 * samples, np.full(3, 9.0), np.zeros(3)))
    np.testing.assert_allclose(direct.numpy(), expected)
    test.assertIsNone(colors.grad)
    np.testing.assert_array_equal(colors.numpy()["r"], [3.0])


def test_external_native_return_array_gradient(test, device):
    """Keep ordinary array-write gradients when helpers return inactive native context."""

    @wp.func
    def write_void(color: NativeColor, value: float, storage: wp.array[float], index: int):
        """Write a differentiable result without returning a value."""
        storage[index] = color.r * value

    @wp.func
    def write_native(color: NativeColor, value: float, storage: wp.array[float], index: int):
        """Write a differentiable result and return inactive context."""
        storage[index] = color.r * value
        return color

    @wp.func
    def write_nested(color: NativeColor, value: float, storage: wp.array[float], index: int):
        """Write through a nested helper returning native context."""
        write_native(color, value, storage, index)

    @wp.func
    def write_native_array(colors: wp.array[NativeColor], value: float, storage: wp.array[float], index: int):
        """Write a differentiable result and return a native array."""
        storage[index] = colors[0].r * value
        return colors

    @wp.func
    def write_native_pair(
        color: NativeColor, colors: wp.array[NativeColor], value: float, storage: wp.array[float], index: int
    ):
        """Write a differentiable result and return two native values."""
        storage[index] = color.r * value
        return NativePixel(color, 0), colors

    @wp.func
    def write_struct(color: NativeColor, value: float, storage: wp.array[float], index: int):
        """Return a struct with both native context and an active field."""
        storage[index] = color.r * value
        return NativeGradientContext(NativeContainer(color), value)

    @wp.func
    def write_loop(color: NativeColor, value: float, storage: wp.array[float], index: int, count: int):
        """Repeat differentiable writes through a native-returning helper."""
        for _i in range(count):
            write_native(color, value, storage, index)
        return color

    @wp.kernel(module="unique")
    def kernel(
        color: NativeColor,
        colors: wp.array[NativeColor],
        values: wp.array[float],
        storage: wp.array[float],
        output: wp.array[float],
        mode: int,
        count: int,
    ):
        """Differentiate array writes across native return forms."""
        i = wp.tid()
        factor = color.r
        if mode == 0:
            storage[i] = color.r * values[i]
        elif mode == 1:
            write_void(color, values[i], storage, i)
        elif mode == 2:
            returned = write_native(color, values[i], storage, i)
            factor = returned.r
        elif mode == 3:
            write_native(color, values[i], storage, i)
        elif mode == 4:
            write_nested(color, values[i], storage, i)
        elif mode == 5:
            returned_array = write_native_array(colors, values[i], storage, i)
            factor = returned_array[0].r
        elif mode == 6:
            returned_pixel, returned_colors = write_native_pair(color, colors, values[i], storage, i)
            factor = returned_pixel.color.r + returned_colors[0].g
        elif mode == 7:
            returned_struct = write_struct(color, values[i], storage, i)
            factor = returned_struct.weight
        else:
            returned_loop = write_loop(color, values[i], storage, i, count)
            factor = returned_loop.r
        output[i] = storage[i] * factor

    wp.set_module_options(
        {
            "extra_build_options": wp.ModuleBuildOptions(
                extra_cpu_preamble=_NATIVE_TEST_PREAMBLE, extra_cuda_preamble=_NATIVE_TEST_PREAMBLE
            )
        },
        module=kernel.module,
    )
    color = NativeColor(3.0, 0.0, 0.0)
    colors = wp.array([color], dtype=NativeColor, device=device)
    samples = np.array([-2.0, 0.5, 3.0], dtype=np.float32)
    cases = ("inline", "void", "native", "ignored", "nested", "native_array", "native_pair", "struct", "loop")
    for mode, case in enumerate(cases):
        with test.subTest(case=case):
            values = wp.array(samples, device=device, requires_grad=True)
            storage = wp.zeros(3, device=device, requires_grad=True)
            output = wp.zeros(3, device=device, requires_grad=True)
            with wp.Tape() as tape:
                wp.launch(kernel, dim=3, inputs=[color, colors, values, storage, output, mode, 3], device=device)
            tape.backward(grads={output: wp.ones(3, device=device)})
            expected_output = 3.0 * samples**2 if case == "struct" else 9.0 * samples
            expected_grad = 6.0 * samples if case == "struct" else np.full_like(samples, 9.0)
            np.testing.assert_allclose(output.numpy(), expected_output)
            np.testing.assert_allclose(values.grad.numpy(), expected_grad)
            np.testing.assert_allclose(storage.numpy(), 3.0 * samples)
    test.assertIsNone(colors.grad)
    np.testing.assert_array_equal(colors.numpy()["r"], [3.0])


def test_external_native_return_explicit_gradient(test, device):
    """Allow explicit derivatives and callback-only snippet calls with array outputs."""

    @wp.struct
    class Palette:
        colors: wp.array[NativeColor]

    @wp.func_native("return palette.colors[0];")
    def read_context(palette: Palette) -> NativeColor:
        """Read native context from an array inside a struct."""
        ...

    @wp.func_native(
        "output[0] = color.r * value; return color;",
        adj_snippet="adj_value += color.r * adj_output[0];",
    )
    def write_explicit(color: NativeColor, value: float, output: wp.array[float]) -> NativeColor:
        """Write a scaled value using an explicit native adjoint."""
        ...

    @wp.func_native("workspace[0] = context.color.r * value; return context.color;")
    def write_context(context: NativeContainer, value: float, workspace: wp.array[float]) -> NativeColor:
        """Write workspace data and return native context."""
        ...

    @wp.func
    def scale(color: NativeColor, value: float, workspace: wp.array[float]):
        """Scale a value using context returned by a native snippet."""
        context = write_context(NativeContainer(color), value, workspace)
        return context.r * value

    @wp.func_grad(scale)
    def scale_grad(color: NativeColor, value: float, workspace: wp.array[float], adj_ret: float):
        """Accumulate the scalar gradient using a native snippet."""
        context = write_context(NativeContainer(color), value, workspace)
        wp.adjoint[value] += context.r * adj_ret

    @wp.func_replay(scale)
    def scale_replay(color: NativeColor, value: float, workspace: wp.array[float]):
        """Recompute the scalar result using a native snippet."""
        context = write_context(NativeContainer(color), value, workspace)
        return context.r * value

    @wp.kernel(module="unique")
    def kernel(
        color: NativeColor,
        colors: wp.array[NativeColor],
        values: wp.array[float],
        workspace: wp.array[float],
        output: wp.array[float],
        mode: int,
    ):
        """Exercise explicit derivatives of native snippet array writes."""
        if mode == 0:
            write_explicit(color, values[0], output)
        elif mode == 1:
            output[0] = scale(color, values[0], workspace)
        else:
            context = read_context(Palette(colors))
            output[0] = context.r * values[0]

    @wp.kernel(module=kernel.module, enable_backward=False)
    def direct_gradient(color: NativeColor, workspace: wp.array[float], output: wp.array[float]):
        """Evaluate the custom gradient with native context directly."""
        _adj_color, adj_value, _adj_workspace = wp.grad(scale)(color, 2.0, workspace)
        output[0] = adj_value

    wp.set_module_options(
        {
            "extra_build_options": wp.ModuleBuildOptions(
                extra_cpu_preamble=_NATIVE_TEST_PREAMBLE, extra_cuda_preamble=_NATIVE_TEST_PREAMBLE
            )
        },
        module=kernel.module,
    )
    color = NativeColor(3.0, 0.0, 0.0)
    colors = wp.array([color], dtype=NativeColor, device=device)
    workspace = wp.zeros(1, device=device)
    for mode in (0, 1, 2):
        with test.subTest(mode=mode):
            values = wp.array([2.0], dtype=float, device=device, requires_grad=True)
            output = wp.zeros(1, device=device, requires_grad=True)
            with wp.Tape() as tape:
                wp.launch(kernel, dim=1, inputs=[color, colors, values, workspace, output, mode], device=device)
            tape.backward(output)
            np.testing.assert_allclose(output.numpy(), [6.0])
            np.testing.assert_allclose(values.grad.numpy(), [3.0])
    wp.launch(direct_gradient, dim=1, inputs=[color, workspace, output], device=device)
    np.testing.assert_allclose(output.numpy(), [3.0])
    np.testing.assert_allclose(workspace.numpy(), [6.0])


def test_external_native_context_custom_gradient(test, device):
    """Treat native values and arrays as opaque context for an explicit gradient."""

    @wp.func
    def color_result(color: NativeColor, value: float):
        """Return mixed context and arithmetic for forward use inside callbacks."""
        return color, external_scale(value, color.r)

    @wp.func
    def scaled_by_color(color: NativeColor, value: float):
        """Scale a scalar using a native value as context."""
        _color, result = color_result(color, value)
        return result

    @wp.func_grad(scaled_by_color)
    def scaled_by_color_grad(color: NativeColor, value: float, adj_ret: float):
        """Accumulate the scalar derivative from native-value context."""
        _color, coefficient = color_result(color, 1.0)
        wp.adjoint[value] += coefficient * adj_ret

    @wp.func_replay(scaled_by_color)
    def scaled_by_color_replay(color: NativeColor, value: float):
        """Replay only the ordinary result of a mixed-return helper."""
        _color, result = color_result(color, value)
        return result

    @wp.func
    def color_result_custom(color: NativeColor, value: float):
        """Return native context alongside a scaled ordinary value."""
        return color, color.r * value

    @wp.func_grad(color_result_custom)
    def color_result_custom_grad(color: NativeColor, value: float, adj_color: NativeColor, adj_value: float):
        """Accumulate the ordinary gradient of a mixed-return helper."""
        wp.adjoint[value] += color.r * adj_value

    @wp.func
    def scaled_by_palette(colors: wp.array[NativeColor], value: float):
        """Scale a scalar using a native array as context."""
        return external_scale(value, colors[0].g)

    @wp.func_grad(scaled_by_palette)
    def scaled_by_palette_grad(colors: wp.array[NativeColor], value: float, adj_ret: float):
        """Accumulate the scalar derivative without native array gradients."""
        wp.adjoint[value] += colors[0].g * adj_ret

    @wp.func_native("return color.r * value;", adj_snippet="adj_value += color.r * adj_ret;")
    def scaled_by_native_snippet(color: NativeColor, value: float) -> float:
        """Scale a value using an explicit native adjoint and opaque context."""
        ...

    @wp.kernel(module="unique")
    def kernel(color: NativeColor, colors: wp.array[NativeColor], values: wp.array[float], output: wp.array[float]):
        """Combine operations using native value and array context."""
        i = wp.tid()
        _color, custom_result = color_result_custom(color, values[i])
        output[i] = (
            scaled_by_color(color, values[i])
            + scaled_by_palette(colors, values[i])
            + scaled_by_native_snippet(color, values[i])
            + custom_result
        )

    @wp.kernel(module=kernel.module, enable_backward=False)
    def gradient(color: NativeColor, colors: wp.array[NativeColor], values: wp.array[float], output: wp.array[float]):
        """Evaluate both explicit gradients without native gradient writes."""
        i = wp.tid()
        _adj_color, adj_value = wp.grad(scaled_by_color)(color, values[i])
        _adj_colors, adj_array_value = wp.grad(scaled_by_palette)(colors, values[i])
        _adj_native_color, adj_native_value = wp.grad(scaled_by_native_snippet)(color, values[i])
        output[i] = adj_value + adj_array_value + adj_native_value

    wp.set_module_options(
        {
            "extra_build_options": wp.ModuleBuildOptions(
                extra_cpu_preamble=_NATIVE_TEST_PREAMBLE,
                extra_cuda_preamble=_NATIVE_TEST_PREAMBLE,
            )
        },
        module=kernel.module,
    )
    color = NativeColor(2.5, 0.0, 0.0)
    palette_color = NativeColor(0.0, 1.75, 0.0)
    colors = wp.array([palette_color], dtype=NativeColor, device=device)
    expected_scale = 3.0 * color.r + palette_color.g
    samples = np.array([-2.0, 0.5, 3.0], dtype=np.float32)
    values = wp.array(samples, requires_grad=True, device=device)
    output = wp.zeros(3, requires_grad=True, device=device)
    with wp.Tape() as tape:
        wp.launch(kernel, dim=3, inputs=[color, colors, values], outputs=[output], device=device)
    tape.backward(grads={output: wp.ones(3, device=device)})
    np.testing.assert_allclose(output.numpy(), expected_scale * samples)
    np.testing.assert_allclose(values.grad.numpy(), np.full_like(samples, expected_scale))
    direct_grad = wp.zeros(3, device=device)
    wp.launch(gradient, dim=3, inputs=[color, colors, values], outputs=[direct_grad], device=device)
    np.testing.assert_allclose(direct_grad.numpy(), np.full_like(samples, expected_scale - color.r))


def test_native_value_types(test, device):
    pixels = wp.zeros(2, dtype=NativePixel, device=device)
    images = wp.array([NativeImage(11), NativeImage(29)], dtype=NativeImage, device=device)
    colors = wp.zeros(2, dtype=NativeColor, device=device)
    handles = wp.zeros(2, dtype=wp.uint64, device=device)

    wp.launch(native_array_kernel, dim=2, inputs=[pixels, images, colors, handles], device=device)

    np.testing.assert_array_equal(handles.numpy(), [11, 29])
    np.testing.assert_allclose(colors.numpy()["r"], [0.0, 2.0])
    np.testing.assert_allclose(colors.numpy()["g"], [1.0, 1.0])
    np.testing.assert_allclose(colors.numpy()["b"], [6.0, 6.0])
    np.testing.assert_array_equal(pixels.numpy()["index"], [0, 1])
    np.testing.assert_allclose(pixels.numpy()["color"]["g"], [0.5, 0.5])

    container = NativeContainer()
    container.color = NativeColor(4.0, 5.0, 6.0)
    wp.launch(native_struct_kernel, dim=1, inputs=[container, NativeImage(41), colors, handles], device=device)
    container_result = colors.numpy()
    np.testing.assert_allclose(
        (container_result["r"][0], container_result["g"][0], container_result["b"][0]),
        (4.0, 5.0, 6.0),
    )
    test.assertEqual(handles.numpy()[0], 41)

    test.assertEqual(colors.list()[0].r, 4.0)
    if device.is_cpu:
        test.assertEqual(colors.cptr()[0].g, 5.0)
    roundtrip = wp.array(colors.numpy(), dtype=NativeColor, device=device)
    np.testing.assert_allclose(roundtrip.numpy()["b"], [6.0, 6.0])

    opaque_numpy = images.numpy()
    test.assertEqual(opaque_numpy.dtype, np.dtype(f"V{ctypes.sizeof(NativeImage)}"))
    opaque_roundtrip = wp.array(opaque_numpy, dtype=NativeImage, device=device)
    test.assertEqual([value.handle for value in opaque_roundtrip.list()], [11, 29])

    structured_opaque_data = np.zeros(2, dtype=[("handle", np.uint64)])
    with test.assertRaisesRegex(RuntimeError, "Invalid source data type for native array"):
        wp.array(structured_opaque_data, dtype=NativeImage, device=device)

    with test.assertRaisesRegex(ValueError, "automatic differentiation"):
        wp.zeros(1, dtype=NativeColor, device=device, requires_grad=True)


@wp.kernel(module="test_external_native_types")
def native_value_tape_kernel(image: NativeImage, values: wp.array[float], loss: wp.array[float]):
    loss[0] = values[0] * 2.0


def test_native_value_tape_adjoint(test, device):
    values = wp.array([3.0], dtype=wp.float32, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch(native_value_tape_kernel, dim=1, inputs=[NativeImage(7), values], outputs=[loss], device=device)

    tape.backward(loss)
    test.assertEqual(values.grad.numpy()[0], 2.0)


class TestExternalBuild(unittest.TestCase):
    @staticmethod
    def _codegen_source(kernel, device):
        module = kernel.module
        options = module.resolve_options(wp.config)
        hasher = wp._src.context.ModuleHasher(module._get_live_kernels(), options)
        builder = wp._src.context.ModuleBuilder(module, options=options, hasher=hasher)
        return builder.codegen(device)

    def test_add_builtin(self):
        name = "_test_external_square"
        function = wp.build_experimental.add_builtin(
            name,
            {"value": wp.float32},
            wp.float32,
            native_name="external::square",
            doc="Square a value.",
        )

        self.assertIsInstance(function, wp.Function)
        overload = function.overloads[-1]
        self.assertEqual(overload.namespace, "external::")
        self.assertEqual(overload.native_func, "square")
        self.assertFalse(overload.is_differentiable)
        self.assertTrue(overload.hidden)
        self.assertFalse(overload.export)

        duplicate = wp.build_experimental.add_builtin(
            name,
            {"value": wp.float32},
            wp.float32,
            native_name="external::square",
            doc="A duplicate registration is harmless.",
        )
        self.assertIs(duplicate, function)

    def test_add_builtin_rejects_ambiguous_overloads(self):
        """Reject overloads that differ only in labels, returns, or reference qualifiers."""
        name = "_test_external_conflicting_labels"
        function = wp.build_experimental.add_builtin(name, {"value": wp.int32}, wp.int32, native_name="external::first")
        original_overloads = tuple(function.overloads)
        for inputs, result, target in (
            ({"renamed": wp.int32}, wp.int32, "external::first"),
            ({"renamed": wp.int32}, wp.int32, "external::second"),
            ({"value": wp.int32}, wp.float32, "external::first"),
            ({"value": wp.ref[wp.int32]}, wp.int32, "external::first"),
        ):
            with self.subTest(inputs=inputs, result=result, target=target):
                with self.assertRaisesRegex(RuntimeError, "conflicting external builtin"):
                    wp.build_experimental.add_builtin(name, inputs, result, native_name=target)
                self.assertEqual(tuple(function.overloads), original_overloads)

        duplicate = wp.build_experimental.add_builtin(name, {"value": int}, int, native_name="external::first")
        self.assertIs(duplicate, function)
        self.assertEqual(tuple(function.overloads), original_overloads)
        self.assertIs(function.get_overload([], {"value": wp.int32}), original_overloads[0])
        self.assertIsNone(function.get_overload([], {"renamed": wp.int32}))

    def test_external_reference_rejects_temporary(self):
        """Reject a temporary bound to an external reference parameter."""

        @wp.kernel(module="unique", enable_backward=False)
        def invalid_reference():
            """Attempt to bind a literal to a mutable external reference."""
            external_write(1.0, 2.0)

        with self.assertRaisesRegex(wp._src.codegen.WarpCodegenError, "requires an addressable expression"):
            self._codegen_source(invalid_reference, "cpu")

    def test_external_reference_rejects_backward(self):
        """Reject mutable references before straight-line or loop replay can run."""

        @wp.kernel(module="unique")
        def direct(values: wp.array[float], count: int):
            """Mutate aliased references in a backward-enabled kernel."""
            external_alias(values[0], values[0])

        @wp.kernel(module="unique")
        def dynamic_for(values: wp.array[float], count: int):
            """Mutate aliased references inside a dynamic for loop."""
            for _i in range(count):
                external_alias(values[0], values[0])

        @wp.kernel(module="unique")
        def dynamic_while(values: wp.array[float], count: int):
            """Mutate aliased references inside a dynamic while loop."""
            i = int(0)
            while i < count:
                external_alias(values[0], values[0])
                i += 1

        for kernel in (direct, dynamic_for, dynamic_while):
            with self.subTest(kernel=kernel.key):
                with self.assertRaisesRegex(wp._src.codegen.WarpCodegenError, "backward replay could repeat mutations"):
                    self._codegen_source(kernel, "cpu")

    def test_external_reference_rejects_native_value_backward(self):
        """Native-value suppression of derivative calls must not bypass replay validation."""
        mutate_color = wp.build_experimental.add_builtin(
            "_test_external_ref_color",
            {"value": wp.ref[NativeColor]},
            NativeColor,
            native_name="warp_test::mutate_color",
        )

        @wp.kernel(module="unique")
        def native_reference(values: wp.array[NativeColor]):
            """Pass a native array element to a mutating reference call."""
            values[1] = mutate_color(values[0])

        with self.assertRaisesRegex(wp._src.codegen.WarpCodegenError, "backward replay could repeat mutations"):
            self._codegen_source(native_reference, "cpu")

    def test_external_reference_shared_helper_rejects_backward(self):
        """Validate nested helpers even when a forward-only caller builds them first."""

        @wp.func
        def mutate(values: wp.array[float]):
            """Mutate aliased array elements through an external call."""
            external_alias(values[0], values[0])

        @wp.func
        def helper(values: wp.array[float]):
            """Call the mutating helper through another function."""
            mutate(values)

        @wp.kernel(module="unique", enable_backward=False)
        def forward(values: wp.array[float]):
            """Call the shared helper without generating a backward pass."""
            helper(values)

        @wp.kernel(module=forward.module)
        def backward(values: wp.array[float]):
            """Call the shared helper with backward generation enabled."""
            helper(values)

        with self.assertRaisesRegex(wp._src.codegen.WarpCodegenError, "backward replay could repeat mutations"):
            self._codegen_source(forward, "cpu")

    def test_external_reference_rejects_grad(self):
        """wp.grad() needs replay validation even inside a forward-only kernel."""

        @wp.func
        def mutate(value: float):
            """Overwrite a local before computing its square."""
            external_write(value, 2.0)
            return value * value

        @wp.func
        def helper(value: float):
            """Call a mutating function through a nested helper."""
            return mutate(value)

        @wp.kernel(module="unique", enable_backward=False)
        def gradient(values: wp.array[float]):
            # Build both functions through a forward call before requesting an adjoint.
            """Request a nested helper gradient after a forward call."""
            values[1] = helper(values[0])
            values[2] = wp.grad(helper)(values[0])

        with self.assertRaisesRegex(wp._src.codegen.WarpCodegenError, "backward replay could repeat mutations"):
            self._codegen_source(gradient, "cpu")

        @wp.func
        def direct(value: float):
            """Overwrite a local before computing its square."""
            external_write(value, 2.0)
            return value * value

        @wp.kernel(module="unique", enable_backward=False)
        def direct_gradient(values: wp.array[float]):
            """Request a mutating function gradient directly."""
            values[1] = wp.grad(direct)(values[0])

        with self.assertRaisesRegex(wp._src.codegen.WarpCodegenError, "backward replay could repeat mutations"):
            self._codegen_source(direct_gradient, "cpu")

    def test_external_reference_rejects_grad_in_tile_callback(self):
        """Discover custom callbacks before validating their wp.grad() targets."""

        @wp.func
        def mutate(value: float):
            """Overwrite a local before computing its square."""
            external_write(value, 2.0)
            return value * value

        @wp.func
        def scale(value: float):
            """Double the input value."""
            return value * 2.0

        @wp.func_grad(scale)
        def scale_grad(value: float, adj_ret: float):
            """Request a mutating function gradient inside a callback."""
            wp.adjoint[value] += wp.grad(mutate)(value) * adj_ret

        @wp.kernel(module="unique")
        def tiled(values: wp.array[float], output: wp.array[float]):
            """Invoke the custom-gradient function through tile mapping."""
            tile = wp.tile_load(values, shape=32)
            wp.tile_store(output, wp.tile_map(scale, tile))

        with self.assertRaisesRegex(wp._src.codegen.WarpCodegenError, "backward replay could repeat mutations"):
            self._codegen_source(tiled, "cuda")

    def test_external_reference_requires_complete_manual_boundary(self):
        """Reject incomplete callbacks and unprotected forward paths."""

        def check_mode(mode):
            """Check one incomplete or unprotected callback boundary."""

            @wp.func
            def operation(value: float):
                """Exercise an external mutation in a wrapped operation."""
                external_write(value, 2.0)
                return value * value

            if mode != "replay_only":

                @wp.func_grad(operation)
                def operation_grad(value: float, adj_ret: float):
                    """Supply the gradient without repeating the forward mutation."""
                    wp.adjoint[value] += 0.0 * adj_ret

            if mode not in ("gradient_only", "callable_gradient_only"):

                @wp.func_replay(operation)
                def operation_replay(value: float):
                    """Recompute the result without changing external state."""
                    return 4.0

            @wp.func
            def helper(value: float):
                """Call the mutating operation through an ordinary forward helper."""
                return operation(value)

            @wp.func
            def pair(value: float):
                """Require primal outputs when computing an automatic tuple adjoint."""
                return operation(value), value * 3.0

            @wp.kernel(module="unique")
            def kernel(values: wp.array[float], loss: wp.array[float]):
                """Exercise the registered operation from a Warp kernel."""
                if wp.static(mode == "grad_primal"):
                    loss[0] = wp.grad(pair)(values[0])
                elif wp.static(mode == "callable_gradient_only"):
                    tile = wp.tile_load(values, shape=1)
                    wp.tile_store(loss, wp.tile_map(operation, tile))
                elif wp.static(mode == "nested_forward"):
                    # Replaying helper calls operation's original forward body.
                    loss[0] = helper(values[0])
                elif wp.static(mode == "shared_forward"):
                    loss[0] = operation(values[0])
                    # A separate unprotected path must still be rejected.
                    external_write(values[0], 7.0)
                else:
                    loss[0] = operation(values[0])

            with self.assertRaisesRegex(wp.WarpCodegenError, "backward replay could repeat mutations"):
                self._codegen_source(kernel, "cpu")

        for mode in (
            "gradient_only",
            "replay_only",
            "nested_forward",
            "shared_forward",
            "callable_gradient_only",
            "grad_primal",
        ):
            with self.subTest(mode=mode):
                check_mode(mode)

    def test_external_derivative_registration_contract(self):
        """Validate derivative policy, forward-result requirements, and canonical aliases."""
        register = wp.build_experimental.add_builtin
        args = ("_test_external_derivative_contract", {"value": float}, float)
        function = register(*args, native_name="warp_test::square_scalar", is_differentiable=True)
        self.assertIs(register(*args, native_name="warp_test::square_scalar", is_differentiable=True), function)
        with self.assertRaisesRegex(RuntimeError, "conflicting external builtin"):
            register(*args, native_name="warp_test::square_scalar", is_differentiable=False)
        self.assertTrue(function.overloads[0].is_differentiable)
        self.assertIs(function.overloads[0].input_types["value"], wp.float32)
        self.assertIs(function.overloads[0].value_type, wp.float32)
        self.assertIs(
            register(
                args[0],
                {"value": wp.float32},
                wp.float32,
                native_name="warp_test::square_scalar",
                is_differentiable=True,
            ),
            function,
        )
        with self.assertRaisesRegex(TypeError, "is_differentiable must be a bool"):
            register("_test_external_invalid_derivative", is_differentiable=1)
        with self.assertRaisesRegex(ValueError, "wp.ref"):
            register("_test_external_ref_derivative", {"value": wp.ref[float]}, is_differentiable=True)

        def check_native_signature(result_type, generic=False):
            """Apply the same native signature restriction to both call paths."""
            native = register(
                f"_test_external_native_derivative_{result_type.__name__}_{generic}",
                {"color": Any if generic else NativeColor} if result_type is float else {"value": float},
                result_type,
                native_name="warp_test::native_derivative",
                is_differentiable=True,
            )

            @wp.kernel(module="unique")
            def native_kernel(colors: wp.array[NativeColor]):
                """Reject an unsupported native adjoint signature in an ordinary call."""
                if wp.static(result_type is float):
                    native(colors[0])
                else:
                    native(3.0)

            @wp.kernel(module="unique", enable_backward=False)
            def native_gradient(colors: wp.array[NativeColor]):
                """Reject the same unsupported signature in an explicit gradient call."""
                if wp.static(result_type is float):
                    wp.grad(native)(colors[0])
                else:
                    wp.grad(native)(3.0)

            @wp.kernel(module="unique")
            def native_map(colors: wp.array[NativeColor], output: wp.array[float]):
                """Apply the signature restriction when resolving a native callable."""
                tile = wp.tile_load(colors, shape=1)
                wp.tile_store(output, wp.tile_map(native, tile))

            kernels = (
                (native_kernel, native_gradient, native_map)
                if result_type is float
                else (native_kernel, native_gradient)
            )
            for kernel in kernels:
                with self.subTest(result_type=result_type, kernel=kernel.key):
                    with self.assertRaisesRegex(wp.WarpCodegenError, "native value types without adjoint storage"):
                        self._codegen_source(kernel, "cpu")

        for result_type, generic in ((float, False), (NativeColor, False), (float, True)):
            check_native_signature(result_type, generic)

        register = wp.build_experimental.add_builtin
        args = ("_test_external_output_contract", {"value": float}, float)
        options = {
            "native_name": "warp_test::root_scalar",
            "is_differentiable": True,
            "adjoint_uses_forward_output": True,
        }
        function = register(*args, **options)
        self.assertIs(register(*args, **options), function)
        self.assertTrue(function.overloads[0].require_original_output_arg)
        with self.assertRaisesRegex(RuntimeError, "conflicting external builtin"):
            register(*args, native_name="warp_test::root_scalar", is_differentiable=True)
        with self.assertRaisesRegex(TypeError, "adjoint_uses_forward_output must be a bool"):
            register("_test_external_invalid_output_flag", adjoint_uses_forward_output=1)
        with self.assertRaisesRegex(ValueError, "differentiable function with a non-void return"):
            register("_test_external_nondiff_output", {"value": float}, float, adjoint_uses_forward_output=True)
        with self.assertRaisesRegex(ValueError, "differentiable function with a non-void return"):
            register("_test_external_void_output", is_differentiable=True, adjoint_uses_forward_output=True)

    def test_native_context_snippet_requires_explicit_gradient(self):
        """Do not infer derivatives from a native snippet with inactive context."""

        @wp.func_native("return color.r * value;")
        def native_only(color: NativeColor, value: float) -> float:
            """Scale a value in native code without providing an adjoint."""
            ...

        @wp.kernel(module="unique", enable_backward=False)
        def gradient(color: NativeColor, output: wp.array[float]):
            """Request a native snippet gradient directly."""
            _adj_color, value = wp.grad(native_only)(color, 2.0)
            output[0] = value

        with self.assertRaisesRegex(wp.WarpCodegenError, "native context requires a Warp function body"):
            self._codegen_source(gradient, "cpu")

        @wp.kernel(module="unique")
        def kernel(color: NativeColor, values: wp.array[float], output: wp.array[float]):
            """Call a native snippet with automatic differentiation enabled."""
            output[0] = native_only(color, values[0])

        with self.assertRaisesRegex(wp.WarpCodegenError, "require an explicit custom gradient"):
            self._codegen_source(kernel, "cpu")

    def test_native_snippet_array_outputs_require_gradient(self):
        """Reject implicit derivatives of ordinary array writes behind inactive returns."""

        @wp.struct
        class Buffers:
            output: wp.array[float]

        @wp.struct
        class NestedBuffers:
            inner: Buffers

        @wp.struct
        class NestedContext:
            inner: NativeContainer

        @wp.func_native("output[0] = color.r * value;")
        def write_void(color: NativeColor, value: float, output: wp.array[float]):
            """Write a scaled value without providing a native adjoint."""
            ...

        @wp.func_native("output[0] = color.r * value; return color;")
        def write_native(color: NativeColor, value: float, output: wp.array[float]) -> NativeColor:
            """Write a scaled value and return native context."""
            ...

        @wp.func_native("output[0] = color.r * value; return {};")
        def write_native_array(color: NativeColor, value: float, output: wp.array[float]) -> wp.array[NativeColor]:
            """Write a scaled value and return an empty native array."""
            ...

        @wp.func_native("buffers.inner.output[0] = color.r * value; return color;")
        def write_nested(color: NativeColor, value: float, buffers: NestedBuffers) -> NativeColor:
            """Write through nested array storage and return native context."""
            ...

        @wp.func_native("output[0] = context.inner.color.r * value;")
        def write_nested_context(context: NestedContext, value: float, output: wp.array[float]):
            """Write a value scaled by nested native context."""
            ...

        @wp.func_native("return context.inner.color.r * value;")
        def read_nested_context(context: NestedContext, value: float) -> float:
            """Read nested native context to scale an ordinary value."""
            ...

        @wp.kernel(module="unique", enable_backward=False)
        def snippet_gradient(color: NativeColor, output: wp.array[float]):
            """Request the gradient of a nested-context native snippet."""
            _adj_context, adj_value = wp.grad(read_nested_context)(NestedContext(NativeContainer(color)), 2.0)
            output[0] = adj_value

        with self.assertRaisesRegex(wp.WarpCodegenError, "native context requires a Warp function body"):
            self._codegen_source(snippet_gradient, "cpu")

        def check(operation, nested, nested_context):
            """Check implicit differentiation and forward-only snippet calls."""

            @wp.func
            def wrapper(color: NativeColor, value: float, output: wp.array[float]):
                """Invoke a native snippet through its selected context shape."""
                if wp.static(nested_context):
                    operation(NestedContext(NativeContainer(color)), value, output)
                elif wp.static(nested):
                    operation(color, value, NestedBuffers(Buffers(output)))
                else:
                    operation(color, value, output)
                return value

            @wp.kernel(module="unique")
            def backward(color: NativeColor, values: wp.array[float], output: wp.array[float]):
                """Call the snippet wrapper with automatic differentiation enabled."""
                wrapper(color, values[0], output)

            @wp.kernel(module="unique", enable_backward=False)
            def direct_gradient(color: NativeColor, output: wp.array[float]):
                """Request the snippet wrapper gradient directly."""
                wp.grad(wrapper)(color, 2.0, output)

            @wp.kernel(module="unique", enable_backward=False)
            def forward(color: NativeColor, output: wp.array[float]):
                """Call the snippet wrapper without backward generation."""
                wrapper(color, 2.0, output)

            for kernel in (backward, direct_gradient):
                with self.subTest(operation=operation.key, kernel=kernel.key):
                    with self.assertRaisesRegex(wp.WarpCodegenError, "require an explicit custom gradient"):
                        self._codegen_source(kernel, "cpu")
            self._codegen_source(forward, "cpu")

        for operation, nested, nested_context in (
            (write_void, False, False),
            (write_native, False, False),
            (write_native_array, False, False),
            (write_nested, True, False),
            (write_nested_context, False, True),
        ):
            check(operation, nested, nested_context)

    def test_mixed_native_output_requires_custom_gradient(self):
        """Reject unsupported reverse edges without suppressing ordinary result gradients."""

        @wp.func
        def mixed_result(color: NativeColor, value: float):
            """Return native context alongside a scaled ordinary value."""
            return color, color.r * value

        @wp.func
        def scalar_result(color: NativeColor, value: float):
            """Extract the ordinary result from a mixed-return helper."""
            _color, result = mixed_result(color, value)
            return result

        @wp.kernel(module="unique")
        def kernel(color: NativeColor, values: wp.array[float], output: wp.array[float]):
            """Differentiate the ordinary result of a mixed-return helper."""
            output[0] = scalar_result(color, values[0])

        @wp.kernel(module="unique", enable_backward=False)
        def gradient(color: NativeColor, output: wp.array[float]):
            """Request the scalar wrapper gradient directly."""
            _adj_color, result = wp.grad(scalar_result)(color, 2.0)
            output[0] = result

        @wp.kernel(module="unique", enable_backward=False)
        def direct_gradient(color: NativeColor, output: wp.array[float]):
            """Request a gradient with an unsupported native output seed."""
            _adj_color, result = wp.grad(mixed_result)(color, 2.0)
            output[0] = result

        for requested in (kernel, gradient):
            with self.subTest(kernel=requested.key):
                with self.assertRaisesRegex(wp.WarpCodegenError, "require an explicit custom gradient"):
                    self._codegen_source(requested, "cpu")
        with self.assertRaisesRegex(wp.WarpCodegenError, "native value outputs do not have a gradient seed"):
            self._codegen_source(direct_gradient, "cpu")

    def test_grad_native_output_has_no_seed(self):
        """Require a supported output seed even for a custom gradient."""

        @wp.func
        def make_color(value: float):
            """Construct a native result without an implicit gradient seed."""
            return NativeColor(value, 0.0, 0.0)

        @wp.func_grad(make_color)
        def make_color_grad(value: float, adj_ret: NativeColor):
            """Define the scalar derivative for an explicitly supplied native adjoint."""
            wp.adjoint[value] += adj_ret.r

        @wp.kernel(module="unique", enable_backward=False)
        def gradient(output: wp.array[float]):
            """Request a gradient whose native return has no implicit seed."""
            output[0] = wp.grad(make_color)(3.0)

        with self.assertRaisesRegex(wp.WarpCodegenError, "native value outputs do not have a gradient seed"):
            self._codegen_source(gradient, "cpu")

    def test_add_builtin_validation(self):
        with self.assertRaisesRegex(ValueError, "valid Python identifier"):
            wp.build_experimental.add_builtin("not-valid")
        with self.assertRaisesRegex(ValueError, "qualified C\\+\\+ identifier"):
            wp.build_experimental.add_builtin("_test_invalid_native_name", native_name="external::bad-name")
        with self.assertRaisesRegex(TypeError, "mapping"):
            wp.build_experimental.add_builtin("_test_invalid_inputs", [])  # type: ignore[arg-type]

    def test_add_builtin_signature_uses_parameter_order_and_native_schema(self):
        class FirstValue(ctypes.Structure):
            _fields_ = [("value", ctypes.c_float)]

        class ReloadedValue(ctypes.Structure):
            _fields_ = [("value", ctypes.c_float)]

        native_name = "warp_test::BuiltinReloadedValue"
        for value_type in (FirstValue, ReloadedValue):
            wp.build_experimental.add_native_type(
                value_type,
                native_name=native_name,
                fields={"value": wp.float32},
                initializer="aggregate",
            )

        name = "_test_ordered_external_signature"
        function = wp.build_experimental.add_builtin(
            name,
            {"value": FirstValue, "scale": wp.float32},
            FirstValue,
            native_name="external::ordered",
        )
        duplicate = wp.build_experimental.add_builtin(
            name,
            {"value": ReloadedValue, "scale": wp.float32},
            ReloadedValue,
            native_name="external::ordered",
        )
        self.assertIs(duplicate, function)

        wp.build_experimental.add_builtin(
            name,
            {"scale": wp.float32, "value": ReloadedValue},
            ReloadedValue,
            native_name="external::ordered_reversed",
        )
        signatures = [tuple(overload.input_types) for overload in function.overloads]
        self.assertIn(("value", "scale"), signatures)
        self.assertIn(("scale", "value"), signatures)

    def test_external_builtin_contract_affects_module_hash(self):
        function = wp.build_experimental.add_builtin(
            "_test_external_hash_target",
            {"value": wp.float32},
            wp.float32,
            native_name="external::first_target",
        )

        @wp.kernel(module="unique")
        def target_kernel(value: wp.array[float]):
            target = wp._test_external_hash_target
            value[0] = target(value[0])

        references = target_kernel.adj.get_references()[2]
        self.assertIn(function, references)
        overload = function.overloads[0]
        first_hash = target_kernel.module.hash_module()
        old_contract = overload._external_builtin_contract
        try:
            overload._external_builtin_contract = (
                old_contract[0],
                "other_external::",
                "second_target",
                *old_contract[3:],
            )
            second_hash = target_kernel.module.hash_module()
        finally:
            overload._external_builtin_contract = old_contract

        self.assertNotEqual(first_hash, second_hash)

    def test_builtin_callable_contract_hashing(self):
        tile_map = wp._src.context.builtin_functions["tile_map"]
        expected_hash = hashlib.sha256()
        expected_hash.update(b"builtin")
        expected_hash.update(tile_map.key.encode())
        expected_hash.update(tile_map.native_func.encode())
        self.assertEqual(wp._src.context.ModuleHasher.hash_builtin_function(tile_map), expected_hash.digest())

        @wp.kernel(module="unique")
        def callable_builtin_kernel(values: wp.array[float]):
            value_tile = wp.tile_load(values, shape=4)
            wp.tile_store(values, wp.tile_map(wp.neg, value_tile))

        self.assertIsNotNone(callable_builtin_kernel.module.hash_module())

    def test_add_native_type_validation(self):
        self.assertIs(
            wp.build_experimental.add_native_type(
                NativeColor,
                native_name="warp_test::Color",
                fields={"r": wp.float32, "g": wp.float32, "b": wp.float32},
                initializer="aggregate",
            ),
            NativeColor,
        )

        class InvalidNativeType(ctypes.Structure):
            _fields_ = [("value", ctypes.c_int16)]

        with self.assertRaisesRegex(ValueError, "ctypes storage"):
            wp.build_experimental.add_native_type(
                InvalidNativeType,
                native_name="warp_test::InvalidNativeType",
                fields={"value": wp.int32},
            )
        with self.assertRaisesRegex(ValueError, "requires exposed fields"):
            wp.build_experimental.add_native_type(
                InvalidNativeType,
                native_name="warp_test::OpaqueAggregate",
                initializer="aggregate",
            )

        class ReorderedNativeType(ctypes.Structure):
            _fields_ = [("first", ctypes.c_float), ("second", ctypes.c_float)]

        with self.assertRaisesRegex(ValueError, "every ctypes field in declaration order"):
            wp.build_experimental.add_native_type(
                ReorderedNativeType,
                native_name="warp_test::ReorderedNativeType",
                fields={"second": wp.float32, "first": wp.float32},
                initializer="aggregate",
            )

        with self.assertRaisesRegex(ValueError, "every ctypes field in declaration order"):
            wp.build_experimental.add_native_type(
                ReorderedNativeType,
                native_name="warp_test::PartialNativeType",
                fields={"first": wp.float32},
                initializer="aggregate",
            )

    def test_padded_native_type_numpy_roundtrip(self):
        """Verify padded native arrays round-trip through Warp's NumPy output."""

        class PaddedValue(ctypes.Structure):
            _fields_ = [("tag", ctypes.c_uint8), ("value", ctypes.c_uint32)]

        wp.build_experimental.add_native_type(
            PaddedValue,
            native_name="warp_test::PaddedValue",
            fields={"tag": wp.uint8, "value": wp.uint32},
            initializer="aggregate",
        )

        source = wp.array([PaddedValue(7, 13)], dtype=PaddedValue, device="cpu")
        try:
            restored = wp.array(source.numpy(), dtype=PaddedValue, device="cpu")
        except RuntimeError as error:
            self.fail(f"Warp's NumPy output should be accepted as native array input: {error}")
        value = restored.list()[0]

        self.assertEqual((value.tag, value.value), (7, 13))

    def test_add_native_type_equivalent_redefinition(self):
        class FirstDefinition(ctypes.Structure):
            _fields_ = [("value", ctypes.c_float)]

        class ReloadedDefinition(ctypes.Structure):
            _fields_ = [("value", ctypes.c_float)]

        native_name = "warp_test::ReloadableNativeType"
        wp.build_experimental.add_native_type(
            FirstDefinition,
            native_name=native_name,
            fields={"value": wp.float32},
            initializer="aggregate",
        )
        self.assertIs(
            wp.build_experimental.add_native_type(
                ReloadedDefinition,
                native_name=native_name,
                fields={"value": wp.float32},
                initializer="aggregate",
            ),
            ReloadedDefinition,
        )

        class ConflictingDefinition(ctypes.Structure):
            _fields_ = [("value", ctypes.c_double)]

        with self.assertRaisesRegex(RuntimeError, "different definition"):
            wp.build_experimental.add_native_type(
                ConflictingDefinition,
                native_name=native_name,
                fields={"value": wp.float64},
                initializer="aggregate",
            )

    def test_add_nested_native_type_equivalent_redefinition(self):
        class FirstInner(ctypes.Structure):
            _fields_ = [("value", ctypes.c_float)]

        class FirstOuter(ctypes.Structure):
            _fields_ = [("inner", FirstInner)]

        class ReloadedInner(ctypes.Structure):
            _fields_ = [("value", ctypes.c_float)]

        class ReloadedOuter(ctypes.Structure):
            _fields_ = [("inner", ReloadedInner)]

        inner_native_name = "warp_test::ReloadableNestedInner"
        outer_native_name = "warp_test::ReloadableNestedOuter"
        wp.build_experimental.add_native_type(
            FirstInner,
            native_name=inner_native_name,
            fields={"value": wp.float32},
            initializer="aggregate",
        )
        wp.build_experimental.add_native_type(
            FirstOuter,
            native_name=outer_native_name,
            fields={"inner": FirstInner},
            initializer="aggregate",
        )
        wp.build_experimental.add_native_type(
            ReloadedInner,
            native_name=inner_native_name,
            fields={"value": wp.float32},
            initializer="aggregate",
        )
        self.assertIs(
            wp.build_experimental.add_native_type(
                ReloadedOuter,
                native_name=outer_native_name,
                fields={"inner": ReloadedInner},
                initializer="aggregate",
            ),
            ReloadedOuter,
        )

    def test_module_build_options_merged(self):
        base = wp.ModuleBuildOptions(
            extra_cuda_include_dirs=["cuda/base", "shared"],
            extra_cpu_include_dirs=["cpu/base"],
            extra_cuda_preamble="#define BASE_CUDA 1",
            extra_cpu_preamble="#define BASE_CPU 1\n",
            extra_build_dependencies=["base.h"],
        )
        addon = wp.ModuleBuildOptions(
            extra_cuda_include_dirs=["shared", "cuda/addon"],
            extra_cpu_include_dirs=["cpu/addon"],
            extra_cuda_preamble="#define ADDON_CUDA 1\n",
            extra_cpu_preamble="#define ADDON_CPU 1",
            extra_build_dependencies=["base.h", "addon.h"],
        )

        merged = base.merged(addon)

        self.assertEqual(merged.extra_cuda_include_dirs, ["cuda/base", "shared", "cuda/addon"])
        self.assertEqual(merged.extra_cpu_include_dirs, ["cpu/base", "cpu/addon"])
        self.assertEqual(merged.extra_cuda_preamble, "#define BASE_CUDA 1\n#define ADDON_CUDA 1\n")
        self.assertEqual(merged.extra_cpu_preamble, "#define BASE_CPU 1\n#define ADDON_CPU 1")
        self.assertEqual(merged.extra_build_dependencies, ["base.h", "addon.h"])
        self.assertEqual(base.extra_cuda_include_dirs, ["cuda/base", "shared"])

        with self.assertRaisesRegex(TypeError, "ModuleBuildOptions"):
            base.merged(object())

    def test_build_dependency_affects_hash(self):
        @wp.kernel(module="unique")
        def dependency_kernel():
            return

        module = dependency_kernel.module
        old_options = module.options.copy()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                dependency = Path(tmpdir) / "external.h"
                dependency.write_text("#define EXTERNAL_VALUE 1\n", encoding="utf-8")
                wp.set_module_options(
                    {"extra_build_options": wp.ModuleBuildOptions(extra_build_dependencies=[dependency.resolve()])},
                    module=module,
                )
                first_hash = module.hash_module()

                dependency.write_text("#define EXTERNAL_VALUE 2\n", encoding="utf-8")
                second_hash = module.hash_module()
                self.assertNotEqual(first_hash, second_hash)
        finally:
            module.options = old_options
            module.mark_modified()

    def test_module_object_options(self):
        @wp.kernel(module="unique")
        def module_options_kernel():
            return

        module = module_options_kernel.module
        old_options = module.options.copy()
        try:
            wp.set_module_options({"max_unroll": 17}, module=module)
            self.assertEqual(wp.get_module_options(module)["max_unroll"], 17)
        finally:
            module.options = old_options
            module.mark_modified()

    def test_native_field_type_contract_is_emitted(self):
        source = self._codegen_source(native_array_kernel, "cpu")
        self.assertIn(
            "wp_external_type_is_same<decltype(((warp_test::Color*)0)->r), wp::float32>::value",
            source,
        )
        self.assertIn(
            "wp_external_type_is_same<decltype(((warp_test::Pixel*)0)->color), warp_test::Color>::value",
            source,
        )


def test_preamble_follows_warp_headers(test, device):
    """Verify that the preamble must be able to use Warp's macros on every backend.

    A leading preamble lands after Warp's headers on CPU (Clang injects the precompiled
    builtin.h ahead of the translation unit) but before them under NVRTC, so the same
    text compiles on one backend and not the other.
    """
    preamble = "CUDA_CALLABLE inline float warp_test_preamble_double(float x) { return x + x; }\n"

    @wp.kernel(module="unique")
    def preamble_kernel(out: wp.array[wp.float32]):
        out[0] = wp.warp_test_preamble_double(1.5)

    wp.set_module_options(
        {
            "extra_build_options": wp.ModuleBuildOptions(
                extra_cpu_preamble=preamble,
                extra_cuda_preamble=preamble,
            )
        },
        module=preamble_kernel.module,
    )

    out = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(preamble_kernel, dim=1, outputs=[out], device=device)
    test.assertEqual(out.numpy()[0], 3.0)


def test_preamble_allows_function_style_casts(test, device):
    """Verify external preambles can use ordinary C++ function-style casts."""

    # Warp emits float() and int() macros for generated code. The preamble must
    # appear before those macros so these casts remain ordinary C++.
    preamble = """
namespace warp_test {
CUDA_CALLABLE inline float preamble_cast(int value) {
    int rounded = int(0.5f);
    return float(value + rounded);
}
}  // namespace warp_test
"""

    @wp.kernel(module="unique")
    def preamble_cast_kernel(out: wp.array[wp.float32]):
        out[0] = wp.warp_test_preamble_cast(7)

    wp.set_module_options(
        {
            "extra_build_options": wp.ModuleBuildOptions(
                extra_cpu_preamble=preamble,
                extra_cuda_preamble=preamble,
            )
        },
        module=preamble_cast_kernel.module,
    )

    out = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(preamble_cast_kernel, dim=1, outputs=[out], device=device)
    test.assertEqual(out.numpy()[0], 7.0)


wp.build_experimental.add_builtin(
    "warp_test_preamble_double",
    {"x": wp.float32},
    wp.float32,
    native_name="warp_test_preamble_double",
)
wp.build_experimental.add_builtin(
    "warp_test_preamble_cast",
    {"value": wp.int32},
    wp.float32,
    native_name="warp_test::preamble_cast",
)

add_function_test(
    TestExternalBuild, "test_external_function_execution", test_external_function_execution, devices=get_test_devices()
)
add_function_test(TestExternalBuild, "test_native_value_types", test_native_value_types, devices=get_test_devices())
add_function_test(
    TestExternalBuild,
    "test_native_value_tape_adjoint",
    test_native_value_tape_adjoint,
    devices=get_test_devices(),
)
add_function_test(
    TestExternalBuild,
    "test_preamble_follows_warp_headers",
    test_preamble_follows_warp_headers,
    devices=get_test_devices(),
)
add_function_test(
    TestExternalBuild,
    "test_preamble_allows_function_style_casts",
    test_preamble_allows_function_style_casts,
    devices=get_test_devices(),
)


add_function_test(
    TestExternalBuild,
    "test_external_reference_forward_loops",
    test_external_reference_forward_loops,
    devices=get_test_devices(),
)


add_function_test(
    TestExternalBuild,
    "test_external_reference_forward_after_rejected_grad",
    test_external_reference_forward_after_rejected_grad,
    devices=get_test_devices(),
)


add_function_test(
    TestExternalBuild,
    "test_external_reference_callback_forward_helpers",
    test_external_reference_callback_forward_helpers,
    devices=get_test_devices(),
)


add_function_test(
    TestExternalBuild, "test_external_native_adjoint", test_external_native_adjoint, devices=get_test_devices()
)
add_function_test(
    TestExternalBuild,
    "test_external_reference_manual_backward",
    test_external_reference_manual_backward,
    devices=get_test_devices(),
)


add_function_test(
    TestExternalBuild,
    "test_external_reference_replay_dispatch",
    test_external_reference_replay_dispatch,
    devices=get_test_devices(),
)


add_function_test(
    TestExternalBuild,
    "test_external_native_context_custom_gradient",
    test_external_native_context_custom_gradient,
    devices=get_test_devices(),
)


add_function_test(
    TestExternalBuild,
    "test_external_native_output_adjoint",
    test_external_native_output_adjoint,
    devices=get_test_devices(),
)

add_function_test(
    TestExternalBuild,
    "test_external_native_context_automatic_gradient",
    test_external_native_context_automatic_gradient,
    devices=get_test_devices(),
)


add_function_test(
    TestExternalBuild,
    "test_external_native_return_array_gradient",
    test_external_native_return_array_gradient,
    devices=get_test_devices(),
)


add_function_test(
    TestExternalBuild,
    "test_external_native_return_explicit_gradient",
    test_external_native_return_explicit_gradient,
    devices=get_test_devices(),
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
