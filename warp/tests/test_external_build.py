# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import hashlib
import tempfile
import unittest
from pathlib import Path

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

NATIVE_COLOR_CONSTANT = NativeColor(0.25, 0.5, 0.75)


@wp.struct
class NativeContainer:
    color: NativeColor


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

        with self.assertRaisesRegex(RuntimeError, "conflicting external builtin"):
            wp.build_experimental.add_builtin(
                name,
                {"value": wp.float32},
                wp.int32,
                native_name="external::square",
            )

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
            overload._external_builtin_contract = (old_contract[0], "other_external::", "second_target")
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
    """The preamble must be able to use Warp's macros on every backend.

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


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
