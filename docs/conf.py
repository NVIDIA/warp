# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import ast
import importlib
import inspect
import operator
import os
import pkgutil
import sys

import docutils
import sphinx
from sphinx.ext.autosummary.generate import AutosummaryRenderer
from sphinx.ext.napoleon.docstring import GoogleDocstring

# -- Path setup --------------------------------------------------------------

HERE = os.path.dirname(__file__)
WARP_PATH = os.path.realpath(os.path.join(HERE, ".."))

sys.path.insert(0, WARP_PATH)

try:
    import warp as wp
    import warp._src
except ImportError as e:
    raise ImportError(
        f"Could not import warp module: {e}. "
        "Warp must be importable to build documentation. "
        "Run build_lib.py first, then run build_docs.py with the docs extra installed."
    ) from e

# Determine the Git version/tag from CI environment variables.
# 1. Check for GitHub Actions' variable.
# 2. Check for GitLab CI's variable.
# 3. Fallback to 'main' for local builds.
GITHUB_VERSION = os.environ.get("GITHUB_REF_NAME") or os.environ.get("CI_COMMIT_REF_NAME") or "main"


# -- Project information -----------------------------------------------------

project = "Warp"
version = wp.__version__
release = version


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # Expands `auto*` directives with their corresponding docstrings.
    "sphinx.ext.autosummary",  # Builds stub .rst files and summary tables.
    "sphinx.ext.doctest",  # Tests code snippets in docs.
    "sphinx.ext.extlinks",  # Generates markups to shorten external links.
    "sphinx.ext.githubpages",  # Makes the doc compatible with GitHub Pages.
    "sphinx.ext.intersphinx",  # Allows linking to external projects' documentations.
    "sphinx.ext.linkcode",  # Adds GitHub source code links.
    "sphinx.ext.napoleon",  # Supports Google and NumPy style docstrings..
    # Third-party extensions.
    "myst_parser",  # Parses markdown files.
    "sphinx_copybutton",  # Adds a copy button to code blocks.
]

# Enable nitpicky mode to warn about unresolved references.
nitpicky = False

# Ignore warnings for Warp types in function signatures that Sphinx can't
# resolve without fully qualified names (e.g., "int32" vs "warp.int32").
# This includes:
# - Type aliases (Scalar, Int, Float, Vector, Quaternion, Matrix, Array, Transformation, Tile)
# - Array types (IndexedArray, IndexedFabricArray, FabricArray)
# - Concrete types (int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64)
# - Array type parameters (ndim=3, dtype=float32) from wp.array() annotations
# - Internal _src paths that leak into type annotations
nitpick_ignore_regex = [
    # Warp type aliases and meta-types
    (
        r"py:class",
        r"(Scalar|Int|Float|Vector|Quaternion|Matrix|Array|Transformation|Tile|IndexedArray|IndexedFabricArray|FabricArray|Shape|DType|Any)",
    ),
    # Concrete numeric types
    (r"py:class", r"(int|uint|float)\d+"),
    # Array type parameters from wp.array() annotations
    (r"py:class", r"(ndim|dtype)=\w+"),
    # Internal _src paths
    (r"py:class", r"warp\._src\..*"),
    # Internal C++/Python interop methods on geometric types
    (
        r"py:obj",
        r"warp\.(vec|mat|quat|transform|spatial_vector|spatial_matrix)\w*\.(from_ptr|scalar_export|scalar_import)",
    ),
    # Matrix accessor methods
    (r"py:obj", r"warp\.(mat|spatial_matrix)\w+\.(get_col|get_row|set_col|set_row)"),
    # Built-in numeric methods/properties inherited by enums/int/float subclasses
    (
        r"py:obj",
        r".*\.(conjugate|bit_length|bit_count|to_bytes|from_bytes|as_integer_ratio|is_integer|real|imag|numerator|denominator)",
    ),
]


# -- Options for source files ------------------------------------------------

exclude_patterns = [".DS_Store", "Thumbs.db", "_build", "_src"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


# -- Options for templating --------------------------------------------------

templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": True,
    "copyright_override": {"start": 2022},
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "footer_links": {},
    "github_url": "https://github.com/NVIDIA/warp",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/warp-lang",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "navigation_depth": 1,
}
html_title = f"Warp {version}"
html_context = {
    "github_user": "NVIDIA",
    "github_repo": "warp",
    "github_version": GITHUB_VERSION,
    "doc_path": "docs",
}
html_css_files = ["custom.css"]
html_static_path = ["_static"]
html_show_sphinx = False


# -- sphinx.ext.autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,  # Includes all public members, not just the class' doscstring.
    "member-order": "bysource",  # Keep members in the order they appear in the source code.
    "undoc-members": False,  # Skips documenting members without a docstring.
    "exclude-members": "__weakref__",  # Skips these names even if they have a docstring.
    "autosummary": True,  # Generate summary tables for members.
}

# Mock external dependencies that might not be installed.
autodoc_mock_imports = ["jax", "paddle", "pxr", "torch"]

# Show typehints as content of the function or method insert of in the signature.
autodoc_typehints = "description"

# Show the literal expression for default arguments instead of evaluating them.
autodoc_preserve_defaults = True

# Prevent docstrings from being inehrited from parent classes or methods.
autodoc_inherit_docstrings = False


# -- sphinx.ext.autosummary --------------------------------------------------

# Document imported classes and functions.
autosummary_imported_members = True

# Only document symbols defined in `__all__` if present.
autosummary_ignore_module_all = False

# Map object names to unique filenames to avoid collisions on case-insensitive
# file systems (e.g., Windows NTFS).
autosummary_filename_map = {
    "warp.e": "warp.e_constant",
    "warp.half_pi": "warp.half_pi_constant",
    "warp.inf": "warp.inf_constant",
    "warp.ln2": "warp.ln2_constant",
    "warp.ln10": "warp.ln10_constant",
    "warp.log10e": "warp.log10e_constant",
    "warp.log2e": "warp.log2e_constant",
    "warp.nan": "warp.nan_constant",
    "warp.phi": "warp.phi_constant",
    "warp.pi": "warp.pi_constant",
    "warp.tau": "warp.tau_constant",
    "warp.kernel": "warp.kernel_decorator",
    "warp.launch": "warp.launch_function",
    "warp.fem.cells": "warp.fem.cells_function",
    "warp.fem.integrand": "warp.fem.integrand_decorator",
}


def normalize_docstring(doc: str) -> str:
    """Normalize docstrings for consistent RST indentation and formatting."""
    if not doc:
        return ""
    cleaned = inspect.cleandoc(doc)
    if not cleaned:
        return ""
    return str(GoogleDocstring(cleaned))


class AutosummaryRenderer(AutosummaryRenderer):
    # Module containing Warp's built-ins functions and requiring special handling.
    BUILTINS_TEMPLATE_FILE = "builtins.rst"

    def render(self, template_name, context):
        if template_name == self.BUILTINS_TEMPLATE_FILE:
            fullname = context["fullname"]
            symbol = fullname.split(".")[-1]

            head = wp._src.context.builtin_functions[symbol]

            # Collect all overloads, filtering out hidden ones.
            # Note: head.overloads already includes the head itself (see Function.__init__)
            all_funcs = head.overloads if hasattr(head, "overloads") else [head]
            visible_overloads = [f for f in all_funcs if not f.hidden]

            # Build overload info for each visible overload
            overloads_info = []
            for func in visible_overloads:
                args = {k: wp._src.context.type_str(v) for k, v in func.input_types.items()}

                try:
                    return_type = wp._src.context.type_str(func.value_func(None, None))
                except Exception:
                    return_type = "None"

                if hasattr(func, "overloads"):
                    sig = wp._src.context.resolve_exported_function_sig(func)
                    is_exported = sig is not None
                else:
                    is_exported = False

                overloads_info.append(
                    {
                        "args": ", ".join(f"{k}: {v}" for k, v in args.items()),
                        "return_type": return_type,
                        "is_exported": is_exported,
                        "is_differentiable": func.is_differentiable,
                        "doc": normalize_docstring(func.doc),
                    }
                )

            # Insert metadata that can be accessed from the template.
            context.update(
                {
                    "wp_display_name": f"warp.{symbol}",
                    "wp_overloads": overloads_info,
                }
            )

        return super().render(template_name, context)


sphinx.ext.autosummary.generate.AutosummaryRenderer = AutosummaryRenderer


# -- sphinx.ext.doctest ------------------------------------------------------

# Code to run for every doctest block.
doctest_global_setup = """
from typing import Any
import numpy as np
import warp as wp
wp.config.quiet = True
wp.init()
"""


# -- sphinx.ext.extlinks -----------------------------------------------------

extlinks = {
    # Short-hand for :github:`path/to/file.py#L123`.
    "github": (f"https://github.com/NVIDIA/warp/blob/{GITHUB_VERSION}/%s", "%s"),
}


# -- sphinx.ext.intersphinx --------------------------------------------------

# Mapping to external documentation to enable cross-linking (e.g., :class:`numpy.ndarray`)
intersphinx_mapping = {
    "jax": ("https://docs.jax.dev/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "pytorch": ("https://pytorch.org/docs/stable/", None),
}


# -- sphinx.ext.linkcode -----------------------------------------------------


def linkcode_resolve(domain, info):
    """Tries to generate external links to code hosted on the Warp GitHub

    This is used for the sphinx.ext.linkcode extension.

    References:
        https://github.com/google/jax/blob/main/docs/conf.py
        https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html
    """
    if domain != "py":
        return None
    if not info["module"]:
        return None
    if not info["fullname"]:
        return None

    try:
        mod = sys.modules.get(info["module"])
        obj = operator.attrgetter(info["fullname"])(mod)
        if isinstance(obj, property):
            obj = obj.fget
        filename = inspect.getsourcefile(obj)
        source, linenum = inspect.getsourcelines(obj)
    except Exception:
        return None

    filename = os.path.relpath(filename, start=os.path.dirname(wp.__file__))
    lines = f"#L{linenum}-L{linenum + len(source)}" if linenum else ""

    return f"https://github.com/NVIDIA/warp/blob/{GITHUB_VERSION}/warp/{filename}{lines}"


# -- sphinx_copybutton -------------------------------------------------------

copybutton_prompt_is_regexp = True  # Tell copybutton the text variable below is a regex.
copybutton_prompt_text = r">>> |\.\.\. |\$ "  # Prompts (>>>, ..., $) to strip when copying code.


# -- Hooks -------------------------------------------------------------------


BUILTIN_DOCSTRINGS = (
    "Convert a string or number to a floating-point number",
    "Return the integer represented by the given array of bytes",
    "bool(x) -> bool",
    "int([x]) -> integer",
)


def filter_builtin_docstrings(app, what, name, obj, options, lines):
    """Remove docstrings inherited from Python built-in types."""
    if not lines:
        return

    for prefix in BUILTIN_DOCSTRINGS:
        if lines[0].startswith(prefix):
            lines.clear()
            return


def build_constant_docs_cache():
    """Build a cache of constant docstrings from all warp._src modules."""
    out = {}

    for _, modname, _ in pkgutil.walk_packages(warp._src.__path__, prefix=warp._src.__name__ + "."):
        try:
            module = importlib.import_module(modname)
            source = inspect.getsource(module)
        except Exception:
            continue

        tree = ast.parse(source)

        for idx, node in enumerate(tree.body[:-1]):
            docstring = ""
            next_node = tree.body[idx + 1]
            if isinstance(next_node, ast.Expr):
                value = next_node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    docstring = inspect.cleandoc(value.value)
            if not docstring:
                continue

            # Handle regular assignments (ast.Assign)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        out[target.id] = docstring
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                out[elt.id] = docstring
            # Handle annotated assignments (ast.AnnAssign), e.g. `x: int = 1`
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                out[node.target.id] = docstring

    return out


CONSTANT_DOCS_CACHE = build_constant_docs_cache()


def populate_reexported_docstrings(app, what, name, obj, options, lines):
    """Populate docstrings for re-exported module-level constants.

    When a constant is re-exported via `from module import X as X`, Sphinx's
    autodoc cannot find its docstring. This function looks up the docstring
    from the source module using string literals that follow constant
    assignments.
    """
    if lines:
        # Already has a docstring, nothing to do
        return

    if what != "data":
        return

    # Extract the simple name from the full qualified name
    simple_name = name.split(".")[-1]

    if simple_name in CONSTANT_DOCS_CACHE:
        lines.append(CONSTANT_DOCS_CACHE[simple_name])


def rewrite_internal_module_paths(app, doctree, docname):
    """Replace internal module paths with public paths in the rendered output.

    Replaces all occurrences of '._src.lang.' with '.' in text nodes.
    """
    for node in doctree.traverse(docutils.nodes.Text):
        if "._src.lang." in node.astext():
            new_text = node.astext().replace("._src.lang.", ".")
            node.parent.replace(node, docutils.nodes.Text(new_text))


def setup(app):
    """Sphinx extension setup."""
    app.connect("autodoc-process-docstring", filter_builtin_docstrings)
    app.connect("autodoc-process-docstring", populate_reexported_docstrings)
    app.connect("doctree-resolved", rewrite_internal_module_paths)
