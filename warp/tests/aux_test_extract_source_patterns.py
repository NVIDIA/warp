# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the success and fallback paths of
:meth:`warp._src.codegen.Adjoint.extract_function_source`.

Most fixtures pin a branch of ``_try_extract_function_source``'s forward walk
— the accompanying tests in :mod:`warp.tests.test_codegen` mock
``inspect.getsourcelines`` to raise, so each subtest proves the fast path
produced a parseable slice on its own.

``contains_truncating_string`` (last in the file) is the parse-fallback fixture:
its multi-line string body sits at column 0, so the indent walk truncates the
slice inside an unclosed ``\"\"\"``. ``ast.parse`` raises ``SyntaxError`` and
``extract_function_source`` re-extracts via ``inspect.getsourcelines``.
"""


# === plain function: end == max_line, no forward walk needed ===
def plain():
    return 42


# === function whose last statement spans multiple lines via parens ===
# The closing ``)`` sits past max_line at indent > base_indent — the forward
# walk's "indented non-comment" branch absorbs it. ``# fmt: off`` keeps the
# formatter from collapsing this back onto one line.
# fmt: off
def multiline_paren_return():
    return (
        1
        + 2
    )
# fmt: on


# === function with a multi-line docstring ===
# The docstring lives between ``def`` and the first executable line, so it is
# always within ``[co_firstlineno, max_line]`` and needs no forward walk.
def with_multiline_docstring():
    """A docstring that
    spans multiple
    lines.
    """
    return 1


# === function with a multi-line string in the body, indented past base ===
# The body of the string sits at indent > base_indent on every line, so the
# forward walk absorbs it without bailing.
def with_indented_body_multiline_string():
    s = """
    inside the body
    indented past base
    """
    return s


# === function containing a nested ``def`` ===
# The nested function's body is fully inside ``[co_firstlineno, max_line]``
# (the outer code object's ``co_lines`` reports the ``def inner`` line and the
# ``return inner`` line).
def with_nested_def():
    def inner():
        return 1

    return inner


# === trailing body-indent comment past max_line ===
# Forward walk's "body-indent comment" branch absorbs it.
def trailing_body_comment():
    return 1
    # trailing comment lives past max_line at indent > base_indent


# === async function ===
async def async_function():
    return 42


# A trailing base-indent comment between fixtures forces the forward walk's
# "base-indent comment" branch to stop. It also separates the previous fixture
# from the next ``def`` so the next ``def`` line itself is the "base-indent
# non-comment" stop branch.


def the_function_that_follows_the_comment():
    return 0


# Blank lines below ``plain_with_trailing_blanks`` exercise the trailing-blank
# trimming pass after the forward walk.
def plain_with_trailing_blanks():
    return 99


# === parse-fallback fixture ===
# Multi-line string body at column 0: the indent walk bails at the first
# string-content line, the slice ends inside an unclosed ``"""``, and
# ``ast.parse`` raises ``SyntaxError`` — exercising the fallback to
# ``inspect.getsourcelines`` in ``extract_function_source``.
def contains_truncating_string():
    return """
not indented past the def
def stop_here(): pass
"""
