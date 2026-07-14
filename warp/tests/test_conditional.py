# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import re
import sys
import unittest

import numpy as np

import warp as wp
from warp._src.codegen import codegen_func_forward
from warp.tests.unittest_utils import *


@wp.kernel
def test_conditional_if_else():
    a = 0.5
    b = 2.0

    if a > b:
        c = 1.0
    else:
        c = -1.0

    wp.expect_eq(c, -1.0)


@wp.kernel
def test_conditional_if_else_nested():
    a = 1.0
    b = 2.0

    if a > b:
        c = 3.0
        d = 4.0

        if c > d:
            e = 1.0
        else:
            e = -1.0

    else:
        c = 6.0
        d = 7.0

        if c > d:
            e = 2.0
        else:
            e = -2.0

    wp.expect_eq(e, -2.0)


@wp.kernel
def test_conditional_ifexp():
    a = 0.5
    b = 2.0

    c = 1.0 if a > b else -1.0

    wp.expect_eq(c, -1.0)


@wp.kernel
def test_conditional_ifexp_nested():
    a = 1.0
    b = 2.0

    c = 3.0 if a > b else 6.0
    d = 4.0 if a > b else 7.0
    e = 1.0 if (a > b and c > d) else (-1.0 if a > b else (2.0 if c > d else -2.0))

    wp.expect_eq(e, -2.0)


@wp.kernel
def test_conditional_ifexp_constant():
    a = 1.0 if False else -1.0
    b = 2.0 if 123 else -2.0

    wp.expect_eq(a, -1.0)
    wp.expect_eq(b, 2.0)


@wp.kernel
def test_conditional_ifexp_constant_nested():
    a = 1.0 if False else (2.0 if True else 3.0)
    b = 4.0 if 0 else (5.0 if 0 else (6.0 if False else 7.0))
    c = 8.0 if False else (9.0 if False else (10.0 if 321 else 11.0))

    wp.expect_eq(a, 2.0)
    wp.expect_eq(b, 7.0)
    wp.expect_eq(c, 10.0)


@wp.kernel
def test_boolean_and():
    a = 1.0
    b = 2.0
    c = 1.0

    if a > 0.0 and b > 0.0:
        c = -1.0

    wp.expect_eq(c, -1.0)


@wp.kernel
def test_boolean_or():
    a = 1.0
    b = 2.0
    c = 1.0

    if a > 0.0 and b > 0.0:
        c = -1.0

    wp.expect_eq(c, -1.0)


@wp.kernel
def test_boolean_compound():
    a = 1.0
    b = 2.0
    c = 3.0

    d = 1.0

    if (a > 0.0 and b > 0.0) or c > a:
        d = -1.0

    wp.expect_eq(d, -1.0)


@wp.kernel
def test_boolean_literal():
    t = True
    f = False

    r = 1.0

    if t == (not f):
        r = -1.0

    wp.expect_eq(r, -1.0)


@wp.kernel
def test_int_logical_not():
    x = 0
    if not 123:
        x = 123

    wp.expect_eq(x, 0)


@wp.kernel
def test_int_conditional_assign_overload():
    if 123:
        x = 123

    if 234:
        x = 234

    wp.expect_eq(x, 234)


@wp.kernel
def test_bool_param_conditional(foo: bool):
    if foo:
        x = 123

    wp.expect_eq(x, 123)


@wp.kernel
def test_conditional_chain_basic():
    x = -1

    if 0 < x < 1:
        success = False
    else:
        success = True
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_empty_range():
    x = -1
    y = 4

    if -2 <= x <= 10 <= y:
        success = False
    else:
        success = True
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_faker():
    x = -1

    # Not actually a chained inequality
    if (-2 < x) < (1 > 0):
        success = False
    else:
        success = True
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_and():
    x = -1

    if (-2 < x < 0) and (-1 <= x <= -1):
        success = True
    else:
        success = False
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_eqs():
    x = wp.int32(10)
    y = 10
    z = -10

    if x == y != z:
        success = True
    else:
        success = False
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_mixed():
    x = 0

    if x < 10 == 1:
        success = False
    else:
        success = True
    wp.expect_eq(success, True)


def test_conditional_unequal_types(test: unittest.TestCase, device):
    # The bad kernel must be in a separate module, otherwise the current module would fail to load
    # Import the bad fixture only for this test so it can be removed from
    # Warp's user module registry before later force-load checks.
    unequal_types_module = importlib.import_module("warp.tests.aux_test_conditional_unequal_types_kernels")
    unequal_types_kernel = unequal_types_module.unequal_types_kernel

    with test.assertRaises(TypeError):
        wp.launch(unequal_types_kernel, dim=(1,), inputs=[], device=device)

    # remove all references to the bad module so that subsequent calls to wp.force_load()
    # won't try to load it unless we explicitly re-import it again
    del wp._src.context.user_modules["warp.tests.aux_test_conditional_unequal_types_kernels"]
    del sys.modules["warp.tests.aux_test_conditional_unequal_types_kernels"]


@wp.kernel
def test_ifexp_with_array_access_kernel(
    idx: wp.int32,
    transforms: wp.array(dtype=wp.transform),
    result: wp.array(dtype=wp.vec3),
):
    # Conditional expression with array element access in else branch
    # When idx < 0, should use transform_identity() and NOT access transforms[idx]
    # This is the exact pattern that caused the segfault bug.
    t = wp.transform_identity() if idx < 0 else transforms[idx]
    result[0] = wp.transform_get_translation(t)


def test_ifexp_with_array_access(test: unittest.TestCase, device):
    transforms = wp.array((wp.transform(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),), dtype=wp.transform, device=device)
    result = wp.zeros(1, dtype=wp.vec3, device=device)

    wp.launch(
        test_ifexp_with_array_access_kernel,
        dim=1,
        inputs=(-1, transforms),
        outputs=(result,),
        device=device,
    )
    test.assertEqual(result.numpy()[0].tolist(), [0.0, 0.0, 0.0])

    wp.launch(
        test_ifexp_with_array_access_kernel,
        dim=1,
        inputs=(0, transforms),
        outputs=(result,),
        device=device,
    )
    test.assertEqual(result.numpy()[0].tolist(), [1.0, 2.0, 3.0])


@wp.kernel
def test_short_circuit_and_kernel(
    arr: wp.array(dtype=int),
    result: wp.array(dtype=int),
):
    tid = wp.tid()
    # arr[tid] must not be evaluated when arr is None (GH-1329)
    if arr and tid >= 0 and arr[tid] == 0:
        result[tid] = -1
        return
    result[tid] = 1


@wp.kernel
def test_short_circuit_or_kernel(
    arr: wp.array(dtype=int),
    result: wp.array(dtype=int),
):
    tid = wp.tid()
    # Second operand must not be evaluated when first is true
    if not arr or tid < 0 or arr[tid] == 0:
        result[tid] = -1
        return
    result[tid] = 1


def test_short_circuit_and(test: unittest.TestCase, device):
    """Chained `and` must short-circuit so null array is never dereferenced."""
    result = wp.zeros(3, dtype=int, device=device)
    # None array — should short-circuit, never access arr[tid]
    wp.launch(test_short_circuit_and_kernel, dim=3, inputs=[None, result], device=device)
    test.assertEqual(result.numpy().tolist(), [1, 1, 1])

    # Real array — should evaluate fully
    arr = wp.array([0, 1, 0], dtype=int, device=device)
    wp.launch(test_short_circuit_and_kernel, dim=3, inputs=[arr, result], device=device)
    test.assertEqual(result.numpy().tolist(), [-1, 1, -1])


def test_short_circuit_or(test: unittest.TestCase, device):
    """Chained `or` must short-circuit so null array is never dereferenced."""
    result = wp.zeros(3, dtype=int, device=device)
    # None array — `not arr` is true, should short-circuit
    wp.launch(test_short_circuit_or_kernel, dim=3, inputs=[None, result], device=device)
    test.assertEqual(result.numpy().tolist(), [-1, -1, -1])

    # Real array with non-zero values — all conditions false, result = 1
    arr = wp.array([5, 6, 7], dtype=int, device=device)
    wp.launch(test_short_circuit_or_kernel, dim=3, inputs=[arr, result], device=device)
    test.assertEqual(result.numpy().tolist(), [1, 1, 1])


@wp.kernel
def test_short_circuit_and_grad_kernel(
    x: wp.array(dtype=float),
    flag: wp.array(dtype=int),
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    # flag[tid] != 0 and tid < 2: only threads 0,1 with flag set take the branch.
    # The backward pass must replay the same short-circuit guards so that
    # gradients flow only through the operands that were actually evaluated.
    if flag[tid] != 0 and tid < 2:
        out[tid] = x[tid] * 3.0
    else:
        out[tid] = x[tid] * 1.0


@wp.kernel
def test_short_circuit_or_grad_kernel(
    x: wp.array(dtype=float),
    flag: wp.array(dtype=int),
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    # flag[tid] == 0 or tid >= 2: threads where flag is zero OR tid >= 2.
    if flag[tid] == 0 or tid >= 2:
        out[tid] = x[tid] * 1.0
    else:
        out[tid] = x[tid] * 3.0


def test_short_circuit_and_grad(test: unittest.TestCase, device):
    """Backward pass through chained `and` propagates correct gradients."""
    n = 4
    x = wp.array(np.ones(n, dtype=np.float32), device=device, requires_grad=True)
    flag = wp.array([1, 1, 0, 0], dtype=int, device=device)
    out = wp.zeros(n, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(test_short_circuit_and_grad_kernel, dim=n, inputs=[x, flag, out], device=device)

    # flag=1 and tid<2 → *3; else → *1
    np.testing.assert_allclose(out.numpy(), [3.0, 3.0, 1.0, 1.0])

    out.grad = wp.array(np.ones(n, dtype=np.float32), device=device)
    tape.backward()

    np.testing.assert_allclose(tape.gradients[x].numpy(), [3.0, 3.0, 1.0, 1.0])


def test_short_circuit_or_grad(test: unittest.TestCase, device):
    """Backward pass through chained `or` propagates correct gradients."""
    n = 4
    x = wp.array(np.ones(n, dtype=np.float32), device=device, requires_grad=True)
    flag = wp.array([0, 1, 1, 0], dtype=int, device=device)
    out = wp.zeros(n, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(test_short_circuit_or_grad_kernel, dim=n, inputs=[x, flag, out], device=device)

    # flag==0 or tid>=2 → *1; else → *3
    # tid=0: flag=0→true (short-circuit) → *1
    # tid=1: flag=1→false, tid>=2→false → *3
    # tid=2: flag=1→false, tid>=2→true  → *1
    # tid=3: flag=0→true (short-circuit) → *1
    np.testing.assert_allclose(out.numpy(), [1.0, 3.0, 1.0, 1.0])

    out.grad = wp.array(np.ones(n, dtype=np.float32), device=device)
    tape.backward()

    np.testing.assert_allclose(tape.gradients[x].numpy(), [1.0, 3.0, 1.0, 1.0])


@wp.kernel
def branch_local_merge_codegen_kernel(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    # ``r`` is first assigned inside nested ``if``/``else`` branches, so it has no version
    # before the conditional. The ``else`` branch must be lowered against the pre-conditional
    # symbol map; otherwise the ``if`` branch's SSA version of ``r`` leaks into the ``else``
    # branch's generated code (GH-1574) and miscompiles on CUDA under register pressure.
    i = wp.tid()
    v = x[i]
    if v > 0.0:
        if v > 10.0:
            r = 100.0
        else:
            r = 1.0
    else:
        if v < -10.0:
            r = -100.0
        else:
            r = -1.0
    out[i] = r


def test_branch_local_merge_codegen(test: unittest.TestCase, device):
    # GH-1574: a variable reassigned inside nested ``if``/``else`` branches must never be
    # referenced across sibling branches in the generated code. The fault is a register-
    # pressure-dependent CUDA miscompile that produces no incorrect value (CPU and CUDA-debug
    # compute the right answer), so the reliable, deterministic signal is the SSA invariant on
    # the generated source itself. Because each assignment lowers to a fresh ``var_N``, no
    # ``var_N`` first assigned inside the ``if`` block may appear inside the sibling ``else``
    # block (and vice versa). Codegen is device-independent, so this does not depend on ``device``.
    branch_local_merge_codegen_kernel.adj.build(builder=None)
    source = codegen_func_forward(branch_local_merge_codegen_kernel.adj, func_type="kernel", device="cpu")

    def match(code, start, opener, closer):
        depth = 0
        for j in range(start, len(code)):
            depth += (code[j] == opener) - (code[j] == closer)
            if depth == 0:
                return j
        raise ValueError("unbalanced delimiters")

    def top_level_if_blocks(code):
        # Return (condition, body) for each brace-depth-0 ``if (condition) { body }``.
        blocks = []
        i = depth = 0
        while i < len(code):
            ch = code[i]
            if ch == "{" or ch == "}":
                depth += 1 if ch == "{" else -1
                i += 1
            elif (
                depth == 0
                and code.startswith("if", i)
                and (i + 2 >= len(code) or not (code[i + 2].isalnum() or code[i + 2] == "_"))
            ):
                lp = code.index("(", i)
                rp = match(code, lp, "(", ")")
                ob = code.index("{", rp)
                cb = match(code, ob, "{", "}")
                blocks.append((code[lp + 1 : rp].strip(), code[ob + 1 : cb]))
                i = cb + 1
            else:
                i += 1
        return blocks

    forward = source.split("// forward", 1)[1]
    code = "\n".join(line.split("//", 1)[0] for line in forward.splitlines())  # drop comments
    blocks = top_level_if_blocks(code)
    if_body = next(body for cond, body in blocks if not cond.startswith("!"))
    else_body = next(body for cond, body in blocks if cond.startswith("!"))

    def assigned(s):
        return set(re.findall(r"(var_\d+)\s*=(?!=)", s))

    def referenced(s):
        return set(re.findall(r"var_\d+", s))

    test.assertEqual(
        assigned(if_body) & referenced(else_body), set(), "if-branch SSA versions leak into the else branch (GH-1574)"
    )
    test.assertEqual(
        assigned(else_body) & referenced(if_body), set(), "else-branch SSA versions leak into the if branch"
    )


def test_branch_local_merge_runtime(test: unittest.TestCase, device):
    # End-to-end check that the branch-local merge executes correctly on each device. This is
    # not a standalone regression guard for GH-1574 (the buggy lowering computes the same
    # values and only crashes on CUDA under register pressure), but it confirms the generated
    # code runs and that the merge selects the right branch on every path.
    x = wp.array([20.0, 5.0, -5.0, -20.0], dtype=float, device=device)
    out = wp.zeros(4, dtype=float, device=device)
    wp.launch(branch_local_merge_codegen_kernel, dim=4, inputs=[x], outputs=[out], device=device)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(out.numpy(), [100.0, 1.0, -1.0, -100.0])


devices = get_test_devices()


class TestConditional(unittest.TestCase):
    pass


add_kernel_test(TestConditional, kernel=test_conditional_if_else, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_if_else_nested, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_ifexp, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_ifexp_nested, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_ifexp_constant, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_ifexp_constant_nested, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_boolean_and, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_boolean_or, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_boolean_compound, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_boolean_literal, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_int_logical_not, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_int_conditional_assign_overload, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_bool_param_conditional, dim=1, inputs=[True], devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_basic, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_empty_range, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_faker, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_and, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_eqs, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_mixed, dim=1, devices=devices)
add_function_test(TestConditional, "test_conditional_unequal_types", test_conditional_unequal_types, devices=devices)
add_function_test(TestConditional, "test_ifexp_with_array_access", test_ifexp_with_array_access, devices=devices)
add_function_test(TestConditional, "test_short_circuit_and", test_short_circuit_and, devices=devices)
add_function_test(TestConditional, "test_short_circuit_or", test_short_circuit_or, devices=devices)
add_function_test(TestConditional, "test_short_circuit_and_grad", test_short_circuit_and_grad, devices=devices)
add_function_test(TestConditional, "test_short_circuit_or_grad", test_short_circuit_or_grad, devices=devices)
add_function_test(TestConditional, "test_branch_local_merge_codegen", test_branch_local_merge_codegen, devices=devices)
add_function_test(TestConditional, "test_branch_local_merge_runtime", test_branch_local_merge_runtime, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
