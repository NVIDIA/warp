# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import contextvars
import gc
import importlib
import inspect
import pickle
import sys
import threading
import unittest
import weakref
from typing import ClassVar

import numpy as np

import warp as wp
from warp._src import function_dispatch as fd
from warp.tests.unittest_utils import add_function_test, get_test_devices

_HANDLED = object()


if __name__ == "__main__":
    # Like application aliases, these helpers must follow startup activation.
    wp.config.enable_array_function_overrides = True


@fd.warp_function
def _dispatch_callsite():
    return fd.get_warp_function_callsite()


@fd.warp_function
def _dispatch_callback(callback):
    return callback()


@fd.warp_function
def _dispatch_identity(value):
    return value


class _Probe:
    calls: ClassVar[list] = []
    result = _HANDLED

    @classmethod
    def __warp_function__(cls, func, types, args=(), kwargs=None):
        cls.calls.append((cls, func, types, args, kwargs))
        return cls.result


class _ObservedArray(wp.array):
    calls: ClassVar[list] = []

    @classmethod
    def __warp_function__(cls, func, types, args=(), kwargs=None):
        cls.calls.append((func, types, args, kwargs))
        return func(*args, **kwargs)


class _EqualGroupMeta(type):
    def __eq__(cls, other):
        return isinstance(other, _EqualGroupMeta) and cls.group == other.group

    def __hash__(cls):
        return hash(cls.group)


@wp.kernel
def _increment(value: wp.array[float]):
    i = wp.tid()
    value[i] += 1.0


@wp.kernel
def _increment_tiled(value: wp.array[float]):
    i, _lane = wp.tid()
    value[i] += 1.0


class TestFunctionDispatch(unittest.TestCase):
    def setUp(self):
        self.addCleanup(
            setattr, wp.config, "enable_array_function_overrides", wp.config.enable_array_function_overrides
        )
        wp.config.enable_array_function_overrides = True
        _Probe.calls.clear()
        _ObservedArray.calls.clear()

    def test_wrapper_alias_honors_later_toggles(self):
        alias = _dispatch_identity
        value = _Probe()
        wp.config.enable_array_function_overrides = False
        self.assertIs(alias(value), value)
        self.assertEqual(_Probe.calls, [])

        wp.config.enable_array_function_overrides = True
        self.assertIs(alias(value), _HANDLED)
        self.assertEqual(len(_Probe.calls), 1)

        wp.config.enable_array_function_overrides = False
        self.assertIs(alias(value), value)
        self.assertEqual(len(_Probe.calls), 1)

    def test_disabled_overrides_do_not_lookup_hooks(self):
        class HookLookupMeta(type):
            def __getattribute__(cls, name):
                if name == "__warp_function__":
                    raise AssertionError("Disabled overrides must not inspect operand hooks")
                return super().__getattribute__(name)

        class Operand(metaclass=HookLookupMeta):
            pass

        value = Operand()
        wp.config.enable_array_function_overrides = False
        self.assertIs(_dispatch_identity(value), value)
        with self.assertRaisesRegex(AssertionError, "must not inspect operand hooks"):
            wp.config.enable_array_function_overrides = True
            _dispatch_identity(value)

    def test_activation_keeps_early_aliases_native(self):
        calls = []

        class Source(np.ndarray):
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                calls.append(func)
                return _HANDLED

        source = np.arange(3, dtype=np.float32).view(Source)
        wp.config.enable_array_function_overrides = False
        early = wp.from_numpy
        constructor = wp.array.__init__
        self.assertFalse(hasattr(early, "__wrapped__"))
        self.assertFalse(hasattr(constructor, "__wrapped__"))

        wp.config.enable_array_function_overrides = True
        active = wp.from_numpy
        self.assertIs(active.__wrapped__, early)
        self.assertIs(wp.array.__init__.__wrapped__, constructor)
        self.assertIs(active(source), _HANDLED)
        result = early(source, device="cpu")
        np.testing.assert_array_equal(result.numpy(), source)
        self.assertEqual(calls, [active])

        wp.config.enable_array_function_overrides = False
        self.assertIs(wp.from_numpy, early)
        self.assertIs(wp.array.__init__, constructor)
        wp.config.enable_array_function_overrides = True
        self.assertIs(wp.from_numpy, active)

    def test_activation_updates_internal_aliases_and_keeps_external_patches(self):
        oriented_bounding_box = importlib.import_module("warp._src.geometry.oriented_bounding_box")

        wp.config.enable_array_function_overrides = False
        native = wp.utils.array_scan
        self.assertIs(oriented_bounding_box.array_scan, native)
        wp.config.enable_array_function_overrides = True
        self.assertIs(oriented_bounding_box.array_scan, wp.utils.array_scan)
        self.assertIs(wp.utils.array_scan.__wrapped__, native)

        active = wp.clone

        def replacement(*args, **kwargs):
            return None

        wp.clone = replacement
        try:
            wp.config.enable_array_function_overrides = False
            self.assertIs(wp.clone, replacement)
            wp.config.enable_array_function_overrides = True
            self.assertIs(wp.clone, replacement)
        finally:
            wp.clone = active

    def test_disabled_overrides_do_not_capture_callsite_or_suppression(self):
        @fd.suppress_warp_function
        def suppressed():
            return contextvars.copy_context()

        wp.config.enable_array_function_overrides = False
        for copied in (_dispatch_callback(contextvars.copy_context), suppressed()):
            self.assertIsNone(copied.run(fd.get_warp_function_callsite))
            wp.config.enable_array_function_overrides = True
            self.assertIs(copied.run(_dispatch_identity, _Probe()), _HANDLED)
            wp.config.enable_array_function_overrides = False

    def test_disabled_conversion_uses_native_validation(self):
        class ObservedNumPyArray(np.ndarray):
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                return _HANDLED

        source = np.arange(3, dtype=np.float32).view(ObservedNumPyArray)
        alias = wp.from_numpy
        wp.config.enable_array_function_overrides = False
        value = alias(source, device="cpu")
        self.assertIs(type(value), wp.array)
        np.testing.assert_array_equal(value.numpy(), source)
        wp.config.enable_array_function_overrides = True
        self.assertIs(alias(source, device="cpu"), _HANDLED)

    def test_original_call_shape(self):
        value = _Probe()
        arguments = [value, value]
        result = wp.launch(None, 2, arguments, device="cpu", record_cmd=True)
        self.assertIs(result, _HANDLED)
        cls, func, types, args, kwargs = _Probe.calls[0]
        self.assertIs(cls, _Probe)
        self.assertIs(func, wp.launch)
        self.assertEqual(types, (_Probe,))
        self.assertEqual(args[:2], (None, 2))
        self.assertIs(args[2], arguments)
        self.assertEqual(kwargs, {"device": "cpu", "record_cmd": True})
        self.assertNotIn("outputs", kwargs)
        self.assertEqual(len(_Probe.calls), 1)

    def test_keyword_order_follows_signature(self):
        class Destination(_Probe):
            result = NotImplemented

        class Source(_Probe):
            pass

        source, destination = Source(), Destination()
        self.assertIs(wp.copy(src=source, dest=destination), _HANDLED)
        self.assertEqual([call[0] for call in _Probe.calls], [Destination, Source])
        self.assertEqual(_Probe.calls[0][2], (Destination, Source))
        self.assertEqual(list(_Probe.calls[0][4]), ["src", "dest"])

    def test_utils_array_operands_precede_native_processing(self):
        cases = (
            (wp.utils.array_scan, ("in_array", "out_array", "temp_storage"), {}),
            (wp.utils.array_sum, ("values", "out"), {}),
            (wp.utils.array_inner, ("a", "b", "out"), {}),
            (wp.utils.radix_sort_pairs, ("keys", "values"), {"count": 3}),
            (
                wp.utils.segmented_sort_pairs,
                ("keys", "values", "segment_start_indices", "segment_end_indices"),
                {"count": 3},
            ),
            (wp.utils.runlength_encode, ("values", "run_values", "run_lengths", "run_count"), {}),
        )
        for function, names, options in cases:
            for name in names:
                with self.subTest(function=function.__name__, operand=name):
                    kwargs = dict.fromkeys(names, object())
                    kwargs.update(options)
                    kwargs[name] = _Probe()
                    _Probe.calls.clear()
                    self.assertIs(function(**kwargs), _HANDLED)
                    self.assertEqual(len(_Probe.calls), 1)
                    self.assertIs(_Probe.calls[0][1], function)
                    self.assertEqual(_Probe.calls[0][2], (_Probe,))
                    self.assertEqual(_Probe.calls[0][3], ())
                    self.assertEqual(_Probe.calls[0][4], kwargs)

    def test_generator_sequence_metadata_and_omitted_hook_default(self):
        calls = []

        class Value:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func, args, kwargs))
                return _HANDLED

        default = Value()

        @fd.warp_function(sequence_parameters=(name for name in ("values",)))
        def operation(values=default):
            return "native"

        self.assertEqual(operation(), "native")
        self.assertEqual(calls, [])
        self.assertIs(operation(values=[default]), _HANDLED)
        self.assertEqual(calls, [(operation, (), {"values": [default]})])
        calls.clear()
        self.assertIs(operation(default), _HANDLED)
        self.assertEqual(calls, [(operation, (default,), {})])

    def test_launch_argument_groups(self):
        for function in (wp.launch, wp.launch_tiled):
            for parameter in ("inputs", "outputs", "adj_inputs", "adj_outputs"):
                with self.subTest(function=function.__name__, parameter=parameter):
                    _Probe.calls.clear()
                    value = _Probe()
                    self.assertIs(function(None, 1, **{parameter: [value]}), _HANDLED)
                    self.assertIs(_Probe.calls[0][1], function)
                    self.assertIs(_Probe.calls[0][4][parameter][0], value)

    def test_launch_positional_groups(self):
        for offset in range(4):
            with self.subTest(offset=offset):
                _Probe.calls.clear()
                arguments = [None, 1, [], [], [], []]
                arguments[offset + 2] = [_Probe()]
                self.assertIs(wp.launch(*arguments), _HANDLED)
                self.assertEqual(len(_Probe.calls), 1)

    def test_launch_group_order(self):
        class Inputs(_Probe):
            result = NotImplemented

        class Outputs(_Probe):
            result = NotImplemented

        class AdjInputs(_Probe):
            result = NotImplemented

        class AdjOutputs(_Probe):
            pass

        result = wp.launch(
            None,
            1,
            adj_outputs=[AdjOutputs()],
            outputs=[Outputs()],
            adj_inputs=[AdjInputs()],
            inputs=[Inputs()],
        )
        self.assertIs(result, _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (Inputs, Outputs, AdjInputs, AdjOutputs))
        self.assertEqual([call[0] for call in _Probe.calls], [Inputs, Outputs, AdjInputs, AdjOutputs])

    def test_nested_sequences_and_cycles(self):
        value = _Probe()
        sequence = []
        sequence.extend([sequence, ([value], (value,))])
        self.assertIs(wp.launch(None, 1, inputs=sequence), _HANDLED)
        self.assertEqual(len(_Probe.calls), 1)
        self.assertEqual(_Probe.calls[0][2], (_Probe,))
        self.assertIs(_Probe.calls[0][4]["inputs"], sequence)

    def test_discovery_does_not_consume_or_probe(self):
        consumed = []

        def generator():
            consumed.append(True)
            yield _Probe()

        class Untouched:
            @property
            def __warp_function__(self):
                raise AssertionError("Instance hooks must not be read")

            @property
            def __array_interface__(self):
                raise AssertionError("Array interfaces must not be read")

            def __dlpack__(self):
                raise AssertionError("DLPack must not be requested")

        # An instance property is a non-callable class hook, so test array
        # interface probing with a separate class that has no protocol member.
        class Foreign:
            @property
            def __array_interface__(self):
                raise AssertionError("Array interfaces must not be read")

            def __dlpack__(self):
                raise AssertionError("DLPack must not be requested")

        class CustomSequence(list):
            def __iter__(self):
                raise AssertionError("Custom containers must not be traversed")

        hidden = _Probe()
        for sequence in (generator(), {"hidden": hidden}, CustomSequence([hidden]), [Foreign()]):
            with self.subTest(sequence=type(sequence).__name__):
                _Probe.calls.clear()
                self.assertIs(wp.launch(_Probe(), 1, inputs=sequence), _HANDLED)
                self.assertEqual(len(_Probe.calls), 1)
        self.assertEqual(consumed, [])
        with self.assertRaisesRegex(TypeError, "must be callable"):
            wp.clone(Untouched())

    def test_only_designated_parameters_recurse(self):
        class Hidden(_Probe):
            pass

        hidden = Hidden()
        self.assertIs(wp.launch(_Probe(), 1, device=[hidden]), _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (_Probe,))
        self.assertEqual(len(_Probe.calls), 1)
        with self.assertRaises(AttributeError):
            wp.clone([hidden])

    def test_custom_argument_container_participates(self):
        class Container(_Probe, list):
            def __iter__(self):
                raise AssertionError("Participating containers must not be traversed")

        values = Container([_Probe()])
        self.assertIs(wp.launch(None, 1, inputs=values), _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (Container,))

    def test_metaclass_equality_does_not_enable_container_traversal(self):
        class EqualContainerMeta(type):
            def __eq__(cls, other):
                return cls is other or other is cls.container

            def __hash__(cls):
                return hash(cls.container)

        for container_type in (list, tuple):
            with self.subTest(container=container_type.__name__):

                class NotAContainer(metaclass=EqualContainerMeta):
                    container = container_type

                    def __iter__(self):
                        raise AssertionError("Custom containers must not be traversed")

                _Probe.calls.clear()
                self.assertIs(wp.launch(_Probe(), 1, inputs=[NotAContainer()]), _HANDLED)
                self.assertEqual(len(_Probe.calls), 1)
                self.assertEqual(_Probe.calls[0][2], (_Probe,))

    def test_unhashable_array_metaclass(self):
        class UnhashableMeta(type):
            __hash__ = None

        class Array(_ObservedArray, metaclass=UnhashableMeta):
            pass

        value = Array([1.0, 2.0], dtype=wp.float32, device="cpu")

        result = wp.clone(value)
        np.testing.assert_allclose(result.numpy(), [1.0, 2.0])
        self.assertEqual(len(_ObservedArray.calls), 1)
        func, types, args, _kwargs = _ObservedArray.calls[0]
        self.assertIs(func, wp.clone)
        self.assertEqual(len(types), 1)
        self.assertIs(types[0], Array)
        self.assertIs(args[0], value)

    def test_equal_classes_participate_separately(self):
        class First(_Probe, metaclass=_EqualGroupMeta):
            group = "same"
            result = NotImplemented

        class Second(_Probe, metaclass=_EqualGroupMeta):
            group = "same"

        first, second = First(), Second()
        self.assertIs(wp.launch(None, 1, inputs=[first, second, first, Second()]), _HANDLED)
        self.assertEqual(len(_Probe.calls), 2)
        self.assertIs(_Probe.calls[0][0], First)
        self.assertIs(_Probe.calls[1][0], Second)
        for call in _Probe.calls:
            self.assertEqual(len(call[2]), 2)
            self.assertIs(call[2][0], First)
            self.assertIs(call[2][1], Second)

    def test_metaclass_equality_does_not_change_precedence(self):
        class Base(_Probe, metaclass=_EqualGroupMeta):
            group = "base"

        class Unrelated(_Probe, metaclass=_EqualGroupMeta):
            group = "base"
            result = NotImplemented

        class Child(Base):
            group = "child"

        self.assertIs(wp.copy(Unrelated(), Child()), _HANDLED)
        self.assertEqual(len(_Probe.calls), 2)
        self.assertIs(_Probe.calls[0][0], Unrelated)
        self.assertIs(_Probe.calls[1][0], Child)
        for call in _Probe.calls:
            self.assertEqual(len(call[2]), 2)
            self.assertIs(call[2][0], Unrelated)
            self.assertIs(call[2][1], Child)

    def test_subclass_precedence_and_repeated_types(self):
        class Base(_Probe):
            result = NotImplemented

        class Other(_Probe):
            pass

        class Child(Base):
            result = NotImplemented

        self.assertIs(wp.launch(None, 1, inputs=[Base(), Other(), Child(), Child(), Base()]), _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (Child, Base, Other))
        self.assertEqual([call[0] for call in _Probe.calls], [Child, Base, Other])

    def test_multiple_inheritance_precedence(self):
        class Left(_Probe):
            result = NotImplemented

        class Right(_Probe):
            pass

        class Child(Left, Right):
            result = NotImplemented

        self.assertIs(wp.launch(None, 1, inputs=[Left(), Right(), Child()]), _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (Child, Left, Right))

    def test_virtual_abc_does_not_change_precedence(self):
        class VirtualBase(_Probe, metaclass=abc.ABCMeta):
            result = NotImplemented

        class Registered(_Probe):
            pass

        VirtualBase.register(Registered)
        self.assertIs(wp.copy(VirtualBase(), Registered()), _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (VirtualBase, Registered))

    def test_inherited_user_hooks_bind_concrete_types(self):
        class First(_Probe):
            result = NotImplemented

        class Second(_Probe):
            pass

        self.assertIs(wp.copy(First(), Second()), _HANDLED)
        self.assertEqual([call[0] for call in _Probe.calls], [First, Second])

    def test_false_and_empty_results_are_handled(self):
        for result in (None, False, 0, [], {}, ()):
            with self.subTest(result=result):

                class Handles(_Probe):
                    pass

                Handles.result = result
                self.assertIs(wp.clone(Handles()), result)

    def test_hook_exception_propagates(self):
        class Failure:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                raise ValueError("Hook failed")

        with self.assertRaisesRegex(ValueError, "Hook failed"):
            wp.copy(Failure(), _Probe())
        self.assertEqual(_Probe.calls, [])
        self.assertIs(wp.clone(_Probe()), _HANDLED)

    def test_instance_assignment_does_not_participate(self):
        class InstanceOnly:
            pass

        value = InstanceOnly()
        value.__warp_function__ = lambda *args: _HANDLED
        with self.assertRaisesRegex(RuntimeError, "must be arrays"):
            wp.copy(value, value)

    def test_hook_none_and_non_callable(self):
        class Disabled(_Probe):
            __warp_function__ = None

        class Invalid:
            __warp_function__ = 1

        self.assertIs(wp.copy(Disabled(), _Probe()), _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (_Probe,))
        with self.assertRaisesRegex(TypeError, "must be callable or None"):
            wp.clone(Invalid())

    def test_instance_method_hooks_are_rejected_before_invocation(self):
        class InstanceHook:
            def __warp_function__(self, func, types, args=(), kwargs=None):
                raise AssertionError("An instance-method hook must not be invoked")

        class Inherited(InstanceHook):
            pass

        for cls in (InstanceHook, Inherited):
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(TypeError, rf"{cls.__name__}.*@classmethod"):
                    _dispatch_identity(cls())

    def test_discovery_releases_hooks_with_gc_disabled(self):
        class Hook:
            def __call__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        class Operand:
            pass

        class FailingDescriptor:
            def __get__(self, instance, owner=None):
                raise RuntimeError("Intentional hook lookup failure")

        class FailingOperand:
            __warp_function__ = FailingDescriptor()

        extractor = fd._ArgumentExtractor(inspect.signature(wp.copy), ())

        def discover(fails):
            hook = Hook()
            reference = weakref.ref(hook)
            Operand.__warp_function__ = hook
            try:
                if fails:
                    try:
                        extractor.candidates((Operand(), FailingOperand()), {})
                    except RuntimeError:
                        pass
                    else:
                        self.fail("The hook descriptor must raise")
                else:
                    extractor.candidates((Operand(), None), {})
            finally:
                del Operand.__warp_function__
            return reference

        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for fails in (False, True):
                with self.subTest(descriptor_error=fails):
                    references = [discover(fails) for _ in range(16)]
                    self.assertTrue(all(reference() is None for reference in references))
        finally:
            if was_enabled:
                gc.enable()

    def test_call_shape_errors_precede_hooks(self):
        value = _Probe()
        calls = (
            lambda: wp.copy(value, value, src=value),
            lambda: wp.copy(value),
            lambda: wp.copy(value, value, unexpected=value),
            lambda: wp.copy(value, value, 0, 0, 0, None, value),
            lambda: wp.launch(None, 1, inputs=[value], unexpected=True),
            lambda: wp.launch_tiled(None, 1, inputs=[value], unexpected=True),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()
                self.assertEqual(_Probe.calls, [])

    def test_call_shape_errors_precede_hook_lookup(self):
        class RejectLookup(type):
            def __getattribute__(cls, name):
                if name == "__warp_function__":
                    raise AssertionError("Invalid calls must not look up hooks")
                return super().__getattribute__(name)

        class Value(metaclass=RejectLookup):
            pass

        value = Value()
        calls = (
            lambda: wp.copy(value),
            lambda: wp.copy(value, value, src=value),
            lambda: wp.copy(value, value, unexpected=value),
            lambda: wp.launch_tiled(None, 1, inputs=[value], unexpected=True),
            lambda: wp.launch_tiled(None, 1, inputs=[value], dim=2, block_dim=1),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

    def test_stateful_metaclass_hook_is_resolved_once_per_operand(self):
        class Alternating(type):
            reads = 0

            def __getattribute__(cls, name):
                if name == "__warp_function__":
                    cls.reads += 1
                    if cls.reads % 2 == 0:
                        return None
                return super().__getattribute__(name)

        class Value(_Probe, metaclass=Alternating):
            pass

        value = Value()
        self.assertIs(_dispatch_identity(value), _HANDLED)
        self.assertIs(_dispatch_identity(value), value)
        self.assertIs(_dispatch_identity(value), _HANDLED)
        self.assertEqual(Value.reads, 3)
        self.assertEqual(len(_Probe.calls), 2)

    def test_hook_descriptors_are_resolved_once_per_operand(self):
        reads = []

        class HookDescriptor(classmethod):
            def __get__(self, instance, owner=None):
                reads.append(("hook", owner))
                return super().__get__(instance, owner)

        class DescriptorValue:
            @HookDescriptor
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                return _HANDLED

        class CallableHook:
            @property
            def __func__(self):
                reads.append(("func", self))
                return self

            def __call__(self, func, types, args=(), kwargs=None):
                return _HANDLED

        hook = CallableHook()

        class CallableValue:
            __warp_function__ = hook

        self.assertIs(wp.clone(DescriptorValue()), _HANDLED)
        self.assertIs(wp.clone(CallableValue()), _HANDLED)
        self.assertEqual(reads, [("hook", DescriptorValue), ("func", hook)])

    def test_descriptor_error_survives_mutated_boxed_kwargs(self):
        failure = TypeError("Intentional descriptor failure")
        kwargs = {}
        reads = []

        class HookDescriptor(classmethod):
            def __get__(self, instance, owner=None):
                reads.append(owner)
                kwargs["unexpected"] = True
                raise failure

        class Value:
            @HookDescriptor
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                raise AssertionError("The descriptor must raise before invocation")

        def body(value):
            return value

        kwargs["value"] = Value()
        extractor = fd._ArgumentExtractor(inspect.signature(body), ())
        with self.assertRaises(TypeError) as caught:
            extractor.candidates((), kwargs)
        self.assertIs(caught.exception, failure)
        self.assertEqual(reads, [Value])
        self.assertIn("unexpected", kwargs)

    def test_public_descriptor_error_survives_mutated_kwargs(self):
        failure = TypeError("Intentional public descriptor failure")
        reads = []

        class HookDescriptor(classmethod):
            def __get__(self, instance, owner=None):
                reads.append(owner)
                # Locate the public wrapper by code identity rather than a
                # fixed stack depth. Mutating its boxed kwargs makes a second
                # bind fail, which must not replace this descriptor's error.
                frame = sys._getframe(1)
                try:
                    while frame is not None and frame.f_code is not operation.__code__:
                        frame = frame.f_back
                    self_test.assertIsNotNone(frame)
                    keyword_parameter = next(
                        parameter.name
                        for parameter in inspect.signature(operation, follow_wrapped=False).parameters.values()
                        if parameter.kind == inspect.Parameter.VAR_KEYWORD
                    )
                    frame.f_locals[keyword_parameter]["unexpected"] = True
                finally:
                    del frame
                raise failure

        class Value:
            @HookDescriptor
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                raise AssertionError("The descriptor must raise before invocation")

        @fd.warp_function
        def operation(value):
            raise AssertionError("A failing descriptor must prevent default execution")

        self_test = self
        with self.assertRaises(TypeError) as caught:
            operation(value=Value())
        self.assertIs(caught.exception, failure)
        self.assertEqual(reads, [Value])

    def test_shared_launch_containers_are_visited_once(self):
        class Alternating(type):
            reads = 0

            def __getattribute__(cls, name):
                if name == "__warp_function__":
                    cls.reads += 1
                    if cls.reads == 1:
                        return None
                return super().__getattribute__(name)

        class Value(_Probe, metaclass=Alternating):
            pass

        @fd.warp_function(sequence_parameters=("inputs", "outputs"))
        def operation(inputs=(), outputs=()):
            return "native"

        value = [Value()]
        self.assertEqual(operation(inputs=[value], outputs=[value]), "native")
        self.assertEqual(Value.reads, 1)
        self.assertEqual(_Probe.calls, [])

    def test_hook_delegates_same_function_directly(self):
        calls = []

        class Delegates:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                calls.append(func)
                return func(*args, **kwargs)

        value = Delegates()
        self.assertIs(_dispatch_identity(value), value)
        self.assertEqual(calls, [_dispatch_identity])
        self.assertIs(wp.clone(_Probe()), _HANDLED)

    def test_hook_delegation_can_replace_operands(self):
        adapted = object()

        class Adapts:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                return func(adapted)

        self.assertIs(_dispatch_identity(Adapts()), adapted)

    def test_other_functions_inside_hook_use_native_implementation(self):
        value = _Probe()

        class Nested:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                return _dispatch_callback(lambda: _dispatch_identity(value))

        self.assertIs(_dispatch_identity(Nested()), value)
        self.assertEqual(_Probe.calls, [])
        self.assertIs(_dispatch_identity(value), _HANDLED)

    def test_hook_and_native_exceptions_restore_overrides(self):
        class Fails:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                self.assertIs(_dispatch_identity(value), value)
                raise ValueError("Hook failure")

        value = _Probe()
        with self.assertRaisesRegex(ValueError, "Hook failure"):
            _dispatch_identity(Fails())
        self.assertIsNone(fd.get_warp_function_callsite())
        self.assertIs(_dispatch_identity(value), _HANDLED)

        def fail():
            self.assertIs(_dispatch_identity(value), value)
            raise ValueError("Native failure")

        with self.assertRaisesRegex(ValueError, "Native failure"):
            _dispatch_callback(fail)
        self.assertIsNone(fd.get_warp_function_callsite())
        self.assertIs(_dispatch_identity(value), _HANDLED)

    def test_hook_suppression_isolated_between_contexts_and_threads(self):
        values = []
        failures = []

        def independent_call():
            try:
                values.append(wp.clone(_Probe()))
            except Exception as error:
                failures.append(error)

        class ChecksContexts:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                contextvars.Context().run(independent_call)
                # Free-threaded Python can inherit the parent's context; explicitly enter a fresh one.
                worker = threading.Thread(target=contextvars.Context().run, args=(independent_call,))
                worker.start()
                worker.join()
                value = _Probe()
                self.assertIs(_dispatch_identity(value), value)
                return func(*args, **kwargs)

        value = ChecksContexts()
        self.assertIs(_dispatch_identity(value), value)
        self.assertEqual(failures, [])
        self.assertEqual(values, [_HANDLED, _HANDLED])

    def test_hook_calls_do_not_accumulate_context_variables(self):
        class Delegates:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                return func(*args, **kwargs)

        def check_context():
            value = Delegates()
            self.assertIs(_dispatch_identity(value), value)
            variables = len(contextvars.copy_context())
            for _ in range(32):
                self.assertIs(_dispatch_identity(value), value)
            self.assertEqual(len(contextvars.copy_context()), variables)

        contextvars.Context().run(check_context)

    def test_each_candidate_runs_with_overrides_suppressed(self):
        calls = []
        value = _Probe()

        class First:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                calls.append(cls)
                self.assertIs(_dispatch_identity(value), value)
                return NotImplemented

        class Second:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                calls.append(cls)
                self.assertIs(_dispatch_identity(value), value)
                return _HANDLED

        self.assertIs(wp.copy(First(), Second()), _HANDLED)
        self.assertEqual(calls, [First, Second])
        self.assertEqual(_Probe.calls, [])

    def test_inherited_neutral_hook_and_super_deferral(self):
        class Neutral(wp.array):
            pass

        class Defers(wp.array):
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                self.assertIs(super().__warp_function__(func, types, args, kwargs), NotImplemented)
                return NotImplemented

        neutral = Neutral(dtype=wp.float32)
        defers = Defers(dtype=wp.float32)
        self.assertIs(wp.copy(neutral, _Probe()), _HANDLED)
        self.assertEqual(_Probe.calls[-1][2], (_Probe,))
        self.assertIs(wp.copy(defers, _Probe()), _HANDLED)
        self.assertEqual(_Probe.calls[-1][2], (Defers, _Probe))

    def test_native_fallback_after_all_decline(self):
        class Declining(wp.array):
            calls: ClassVar[list] = []

            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append(func)
                return super().__warp_function__(func, types, args, kwargs)

        value = Declining([1.0, 2.0], dtype=wp.float32, device="cpu")
        result = wp.clone(value)
        np.testing.assert_allclose(result.numpy(), [1.0, 2.0])
        self.assertEqual(Declining.calls, [wp.clone])
        self.assertIs(type(result), wp.array)

        class DecliningWrapper(_Probe):
            result = NotImplemented

        with self.assertRaisesRegex(RuntimeError, "must be arrays"):
            wp.copy(DecliningWrapper(), DecliningWrapper())

    def test_native_method_delegation_preserves_callsite(self):
        # Inspect the internal location explicitly for this regression: array
        # methods call covered APIs without establishing an outer dispatch.
        from warp._src.function_dispatch import get_warp_function_callsite  # noqa: PLC0415

        calls = []

        class CapturesCallsite(wp.array):
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func, get_warp_function_callsite()))
                return func(*args, **kwargs)

        value = CapturesCallsite([0.0, 0.0], dtype=wp.float32, device="cpu")
        source = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        call_line = sys._getframe().f_lineno + 1
        value.assign(source)
        self.assertEqual(calls, [(wp.copy, (__file__, call_line, "test_native_method_delegation_preserves_callsite"))])
        np.testing.assert_allclose(value.numpy(), [1.0, 2.0])

    def test_default_callback_preserves_outer_boundary_and_suppresses_hooks(self):
        value = _Probe()

        def callback():
            return _dispatch_identity(value), _dispatch_callsite()

        line = sys._getframe().f_lineno + 1
        result, location = _dispatch_callback(callback)
        self.assertIs(result, value)
        self.assertEqual(location, (__file__, line, self._testMethodName))
        self.assertEqual(_Probe.calls, [])
        self.assertIsNone(fd.get_warp_function_callsite())

    def test_other_operations_inside_hook_retain_outer_boundary(self):
        nested_locations = []

        class Captures:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                before = fd.get_warp_function_callsite()
                nested = _dispatch_callback(_dispatch_callsite)
                nested_locations.append(nested)
                return before, fd.get_warp_function_callsite()

        line = sys._getframe().f_lineno + 1
        before, after = _dispatch_identity(Captures())
        self.assertEqual(before, (__file__, line, self._testMethodName))
        self.assertEqual(after, before)
        self.assertEqual(nested_locations, [before])
        self.assertIsNone(fd.get_warp_function_callsite())

    def test_copied_context_retains_origin_without_retaining_caller_locals(self):
        class Payload:
            pass

        def invoke():
            payload = Payload()
            owner = weakref.ref(payload)
            line = sys._getframe().f_lineno + 1
            copied = _dispatch_callback(contextvars.copy_context)
            return owner, copied, line

        owner, copied, line = invoke()
        gc.collect()
        self.assertIsNone(owner())
        self.assertEqual(copied.run(_dispatch_callsite), (__file__, line, "invoke"))
        self.assertIsNone(fd.get_warp_function_callsite())

    def test_copied_context_inside_hook_retains_origin_and_suppression(self):
        class Captures:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                return _dispatch_callback(contextvars.copy_context)

        line = sys._getframe().f_lineno + 1
        copied = _dispatch_identity(Captures())
        self.assertEqual(copied.run(_dispatch_callsite), (__file__, line, self._testMethodName))
        value = _Probe()
        self.assertIs(copied.run(_dispatch_identity, value), value)
        self.assertIs(_dispatch_identity(value), _HANDLED)
        self.assertIsNone(fd.get_warp_function_callsite())

    def test_dynamic_hook_replacement(self):
        class Dynamic(np.ndarray):
            pass

        source = np.arange(3, dtype=np.float32).view(Dynamic)
        self.assertIsInstance(wp.from_numpy(source), wp.array)
        Dynamic.__warp_function__ = classmethod(lambda cls, func, types, args=(), kwargs=None: _HANDLED)
        self.assertIs(wp.from_numpy(source), _HANDLED)
        Dynamic.__warp_function__ = None
        self.assertIsInstance(wp.from_numpy(source), wp.array)

    def test_public_function_identity_and_aliases(self):
        probe = _Probe()
        calls = (
            (wp.launch, (None, 1), {"inputs": [probe]}),
            (wp.launch_tiled, (None, 1), {"outputs": [probe]}),
            (wp.copy, (probe, probe), {}),
            (wp.clone, (probe,), {}),
            (wp.empty_like, (probe,), {}),
            (wp.zeros_like, (probe,), {}),
            (wp.ones_like, (probe,), {}),
            (wp.full_like, (probe, 2.0), {}),
            (wp.from_numpy, (probe,), {}),
            (wp.from_torch, (probe,), {}),
            (wp.to_torch, (probe,), {}),
            (wp.from_dlpack, (probe,), {}),
            (wp.to_dlpack, (probe,), {}),
            (wp.utils.array_scan, (probe, probe), {}),
            (wp.utils.array_sum, (probe,), {}),
            (wp.utils.array_inner, (probe, probe), {}),
            (wp.utils.radix_sort_pairs, (probe, probe, 3), {}),
            (wp.utils.segmented_sort_pairs, (probe, probe, 3, probe), {}),
            (wp.utils.runlength_encode, (probe, probe, probe), {}),
        )
        for function, args, kwargs in calls:
            with self.subTest(function=function.__name__):
                _Probe.calls.clear()
                alias = function
                self.assertIs(alias(*args, **kwargs), _HANDLED)
                self.assertIs(_Probe.calls[0][1], function)
                self.assertTrue(function.__doc__)
                self.assertTrue(hasattr(function, "__wrapped__"))
                self.assertTrue(inspect.isfunction(function))
                self.assertIs(weakref.ref(function)(), function)
                self.assertIs(pickle.loads(pickle.dumps(function)), function)
                for attribute in ("__name__", "__qualname__", "__module__", "__doc__", "__annotations__"):
                    self.assertEqual(getattr(function, attribute), getattr(function.__wrapped__, attribute))
                if function is not wp.launch_tiled:
                    self.assertEqual(inspect.signature(function), inspect.signature(function.__wrapped__))

    def test_public_function_alias_retains_method_binding(self):
        class Bound(_Probe):
            operation = wp.clone

        value = Bound()
        self.assertIs(Bound.operation, wp.clone)
        self.assertIs(value.operation.__func__, wp.clone)
        self.assertIs(value.operation(), _HANDLED)
        self.assertEqual(len(_Probe.calls), 1)
        self.assertIs(_Probe.calls[0][1], wp.clone)
        self.assertEqual(_Probe.calls[0][3], (value,))

    def test_unregistered_dispatch_wrapper_cycle_is_collected(self):
        class Payload:
            pass

        def create_wrapper():
            payload = Payload()

            def operation(value):
                return payload

            wrapped = fd.warp_function(operation)
            fd.WARP_FUNCTIONS.pop(wrapped)
            return weakref.ref(payload), weakref.ref(operation), weakref.ref(wrapped)

        references = create_wrapper()
        gc.collect()
        self.assertEqual([reference() for reference in references], [None, None, None])

    def test_conversion_hook_precedes_data_protocols(self):
        class Foreign(_Probe):
            @property
            def __array_interface__(self):
                raise AssertionError("Array interface must be queried after dispatch")

            @property
            def __cuda_array_interface__(self):
                raise AssertionError("CUDA array interface must be queried after dispatch")

            def __array__(self, *args, **kwargs):
                raise AssertionError("NumPy conversion must happen after dispatch")

            def __dlpack__(self, *args, **kwargs):
                raise AssertionError("DLPack must be requested after dispatch")

            def __dlpack_device__(self):
                raise AssertionError("DLPack device must be queried after dispatch")

        source = Foreign()
        for function in (wp.from_numpy, wp.from_torch, wp.from_dlpack):
            with self.subTest(function=function.__name__):
                self.assertIs(function(source), _HANDLED)

    def test_conversion_gradient_is_a_direct_operand(self):
        gradient = _Probe()
        source = object()
        self.assertIs(wp.from_torch(source, grad=gradient), _HANDLED)
        self.assertIs(_Probe.calls[0][1], wp.from_torch)
        self.assertEqual(_Probe.calls[0][3], (source,))
        self.assertIs(_Probe.calls[0][4]["grad"], gradient)


def test_native_delegation(test, device):
    _ObservedArray.calls.clear()
    value = _ObservedArray([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    test.assertEqual(_ObservedArray.calls, [])

    result = wp.clone(value)
    test.assertEqual([call[0] for call in _ObservedArray.calls], [wp.clone])
    np.testing.assert_allclose(result.numpy(), [1.0, 2.0, 3.0])
    test.assertIs(type(result), wp.array)

    pointer = value.ptr
    alias = wp.array(ptr=pointer, shape=value.shape, dtype=wp.float32, device=device)
    replacement = wp.array([4.0, 5.0, 6.0], dtype=wp.float32, device=device)
    test.assertIsNone(wp.copy(dest=value, src=replacement))
    test.assertEqual(value.ptr, pointer)
    np.testing.assert_allclose(alias.numpy(), [4.0, 5.0, 6.0])
    test.assertEqual([call[0] for call in _ObservedArray.calls], [wp.clone, wp.copy])

    for function, expected in ((wp.zeros_like, 0.0), (wp.ones_like, 1.0)):
        _ObservedArray.calls.clear()
        result = function(value)
        test.assertEqual([call[0] for call in _ObservedArray.calls], [function])
        np.testing.assert_allclose(result.numpy(), np.full(3, expected))
    _ObservedArray.calls.clear()
    result = wp.full_like(value, 7.0)
    np.testing.assert_allclose(result.numpy(), np.full(3, 7.0))
    test.assertEqual([call[0] for call in _ObservedArray.calls], [wp.full_like])


def test_launch_mutates_inputs_once(test, device):
    value = _ObservedArray([1.0, 2.0], dtype=wp.float32, device=device)
    _ObservedArray.calls.clear()
    test.assertIsNone(wp.launch(_increment, 2, inputs=[value], device=device))
    np.testing.assert_allclose(value.view(value.dtype).numpy(), [2.0, 3.0])
    test.assertEqual([call[0] for call in _ObservedArray.calls], [wp.launch])


def test_launch_tiled_composition(test, device):
    value = _ObservedArray([1.0, 2.0], dtype=wp.float32, device=device)
    _ObservedArray.calls.clear()
    test.assertIsNone(wp.launch_tiled(_increment_tiled, dim=2, inputs=[value], device=device, block_dim=1))
    np.testing.assert_allclose(value.view(value.dtype).numpy(), [2.0, 3.0])
    test.assertEqual([call[0] for call in _ObservedArray.calls], [wp.launch_tiled])


def test_utils_native_delegation(test, device):
    for enabled in (False, True):
        with test.subTest(enabled=enabled):
            wp.config.enable_array_function_overrides = enabled

            def call(function, *args, _enabled=enabled, **kwargs):
                _ObservedArray.calls.clear()
                result = function(*args, **kwargs)
                test.assertEqual([event[0] for event in _ObservedArray.calls], [function] if _enabled else [])
                return result

            values = _ObservedArray([1, 2, 3], dtype=wp.int32, device=device)
            scanned = wp.empty(3, dtype=wp.int32, device=device)
            test.assertIsNone(call(wp.utils.array_scan, values, scanned))
            np.testing.assert_array_equal(scanned.numpy(), [1, 3, 6])

            # Whole-array reduction makes a contiguous copy of this input.
            # Its internal clone must not become a second override event.
            owner = wp.array(np.arange(6, dtype=np.float32), device=device)
            strided = _ObservedArray(ptr=owner.ptr, shape=(3,), strides=(8,), dtype=wp.float32, device=device)
            np.testing.assert_allclose(call(wp.utils.array_sum, strided), 6.0)

            a = _ObservedArray([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
            b = wp.array([4.0, 5.0, 6.0], dtype=wp.float32, device=device)
            np.testing.assert_allclose(call(wp.utils.array_inner, a, b), 32.0)

            keys = _ObservedArray([3, 1, 2, 0, 0, 0], dtype=wp.int32, device=device)
            pairs = wp.array([30, 10, 20, 0, 0, 0], dtype=wp.int32, device=device)
            test.assertIsNone(call(wp.utils.radix_sort_pairs, keys, pairs, 3))
            np.testing.assert_array_equal(keys.view(keys.dtype).numpy()[:3], [1, 2, 3])
            np.testing.assert_array_equal(pairs.numpy()[:3], [10, 20, 30])

            keys = _ObservedArray([4, 1, 3, 2, 0, 0, 0, 0], dtype=wp.int32, device=device)
            pairs = wp.array([40, 10, 30, 20, 0, 0, 0, 0], dtype=wp.int32, device=device)
            offsets = wp.array([0, 2, 4], dtype=wp.int32, device=device)
            test.assertIsNone(call(wp.utils.segmented_sort_pairs, keys, pairs, 4, offsets))
            np.testing.assert_array_equal(keys.view(keys.dtype).numpy()[:4], [1, 4, 2, 3])
            np.testing.assert_array_equal(pairs.numpy()[:4], [10, 40, 20, 30])

            values = _ObservedArray([1, 1, 2, 3, 3], dtype=wp.int32, device=device)
            run_values = wp.empty(5, dtype=wp.int32, device=device)
            run_lengths = wp.empty(5, dtype=wp.int32, device=device)
            test.assertEqual(call(wp.utils.runlength_encode, values, run_values, run_lengths), 3)
            np.testing.assert_array_equal(run_values.numpy()[:3], [1, 2, 3])
            np.testing.assert_array_equal(run_lengths.numpy()[:3], [2, 1, 2])


class TestFunctionDispatchDevices(unittest.TestCase):
    def setUp(self):
        self.addCleanup(
            setattr, wp.config, "enable_array_function_overrides", wp.config.enable_array_function_overrides
        )
        wp.config.enable_array_function_overrides = True


devices = get_test_devices()
add_function_test(TestFunctionDispatchDevices, "test_native_delegation", test_native_delegation, devices=devices)
add_function_test(
    TestFunctionDispatchDevices, "test_launch_mutates_inputs_once", test_launch_mutates_inputs_once, devices=devices
)
add_function_test(
    TestFunctionDispatchDevices, "test_launch_tiled_composition", test_launch_tiled_composition, devices=devices
)
add_function_test(
    TestFunctionDispatchDevices, "test_utils_native_delegation", test_utils_native_delegation, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
